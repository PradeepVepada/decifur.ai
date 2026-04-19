import json
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


API_URL = "http://127.0.0.1:8000/api/chat"
INPUT_FILE = Path("data/eval_questions.json")
OUTPUT_DIR = Path("evaluation/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize(text: str) -> str:
    return " ".join(safe_text(text).lower().split())


def extract_answer(payload: Dict[str, Any]) -> str:
    for key in ["answer", "response", "output"]:
        if key in payload and payload[key]:
            return safe_text(payload[key])
    return ""


def extract_context(payload: Dict[str, Any]) -> str:
    parts: List[str] = []

    for key in ["context", "retrieved_context", "sources", "source_chunks", "chunks"]:
        value = payload.get(key)
        if not value:
            continue

        if isinstance(value, str):
            parts.append(value)

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    for text_key in ["text", "content", "chunk", "page_content"]:
                        if item.get(text_key):
                            parts.append(safe_text(item[text_key]))
                            break

    return "\n".join(parts)


def ask_chatbot(question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"query": question, "stream": False}

    if chat_history is not None:
        payload["chat_history"] = chat_history

    response = requests.post(API_URL, json=payload, timeout=120)
    try:
        data = response.json()
    except Exception:
        data = {}
    if response.ok:
        return data
    # On server errors, return whatever partial JSON the server returned
    if data:
        return data
    return {"answer": "", "error": f"HTTP {response.status_code}"}


def contains_any(text: str, phrases: List[str]) -> bool:
    text_n = normalize(text)
    return any(normalize(p) in text_n for p in phrases)


def token_overlap_score(text_a: str, text_b: str) -> float:
    """
    Simple overlap score between two texts.
    """
    a_tokens = set(normalize(text_a).split())
    b_tokens = set(normalize(text_b).split())

    if not a_tokens or not b_tokens:
        return 0.0

    overlap = a_tokens.intersection(b_tokens)
    return len(overlap) / max(len(b_tokens), 1)


def score_retrieval(context: str, reference_answer: str) -> int:
    """
    1 to 5
    Measures whether retrieved context appears related to the reference answer.
    """
    overlap = token_overlap_score(context, reference_answer)

    if not context.strip():
        return 1
    if overlap >= 0.50:
        return 5
    if overlap >= 0.35:
        return 4
    if overlap >= 0.20:
        return 3
    if overlap >= 0.10:
        return 2
    return 1


def score_correctness(answer: str, reference_answer: str) -> int:
    """
    1 to 5
    Measures similarity of answer to reference answer.
    """
    bad_patterns = [
        "i don't know",
        "not in the papers",
        "no information",
        "cannot answer",
        "not available"
    ]

    if contains_any(answer, bad_patterns):
        return 1

    overlap = token_overlap_score(answer, reference_answer)

    if not answer.strip():
        return 1
    if overlap >= 0.55:
        return 5
    if overlap >= 0.40:
        return 4
    if overlap >= 0.25:
        return 3
    if overlap >= 0.12:
        return 2
    return 1


def score_groundedness(answer: str, context: str) -> int:
    """
    1 to 5
    Measures whether the answer appears supported by retrieved context.
    """
    if not answer.strip():
        return 1
    if not context.strip():
        return 1

    overlap = token_overlap_score(answer, context)

    if overlap >= 0.55:
        return 5
    if overlap >= 0.40:
        return 4
    if overlap >= 0.25:
        return 3
    if overlap >= 0.12:
        return 2
    return 1


def score_memory(question: str, answer: str, previous_question: str, previous_answer: str) -> int:
    """
    Special score for follow-up memory questions.
    Example: 'Explain that in simpler words.'
    """
    q = normalize(question)
    a = normalize(answer)
    prev_q = normalize(previous_question)
    prev_a = normalize(previous_answer)

    if not answer.strip():
        return 1

    memory_failure_patterns = [
        "what do you mean",
        "please clarify",
        "i need more context",
        "not enough context",
        "which topic",
        "can you specify"
    ]

    if contains_any(answer, memory_failure_patterns):
        return 1

    overlap_prev_answer = token_overlap_score(a, prev_a)
    overlap_prev_question = token_overlap_score(a, prev_q)

    if "simpler words" in q or "explain that" in q or "that" in q:
        combined = max(overlap_prev_answer, overlap_prev_question)

        if combined >= 0.40:
            return 5
        if combined >= 0.28:
            return 4
        if combined >= 0.16:
            return 3
        if combined >= 0.08:
            return 2
        return 1

    return 3


def compute_overall_score(
    retrieval_score: int,
    correctness_score: int,
    groundedness_score: int,
    memory_score: Optional[int] = None,
) -> float:
    if memory_score is None:
        return round((0.30 * retrieval_score) + (0.40 * correctness_score) + (0.30 * groundedness_score), 2)

    return round(
        (0.25 * retrieval_score) +
        (0.30 * correctness_score) +
        (0.25 * groundedness_score) +
        (0.20 * memory_score),
        2,
    )


def evaluate_one(
    item: Dict[str, Any],
    chat_history: Optional[List[Dict[str, str]]] = None,
    previous_turn: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    question = item["question"]
    reference_answer = item["reference_answer"]

    start = time.perf_counter()
    raw = ask_chatbot(question, chat_history=chat_history)
    latency = round(time.perf_counter() - start, 3)

    answer = extract_answer(raw)
    context = extract_context(raw)

    retrieval_score = score_retrieval(context, reference_answer)
    correctness_score = score_correctness(answer, reference_answer)
    groundedness_score = score_groundedness(answer, context)

    memory_score = None
    if item["type"] == "memory_followup" and previous_turn is not None:
        memory_score = score_memory(
            question=question,
            answer=answer,
            previous_question=previous_turn["question"],
            previous_answer=previous_turn["answer"],
        )

    overall_score = compute_overall_score(
        retrieval_score=retrieval_score,
        correctness_score=correctness_score,
        groundedness_score=groundedness_score,
        memory_score=memory_score,
    )

    return {
        "id": item["id"],
        "type": item["type"],
        "question": question,
        "reference_answer": reference_answer,
        "answer": answer,
        "context_preview": context[:1500],
        "latency_sec": latency,
        "retrieval_score": retrieval_score,
        "correctness_score": correctness_score,
        "groundedness_score": groundedness_score,
        "memory_score": memory_score,
        "overall_score": overall_score,
        "raw_response": raw,
    }


def save_json(results: List[Dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def save_csv(results: List[Dict[str, Any]], path: Path) -> None:
    rows = []
    for r in results:
        rows.append({
            "id": r["id"],
            "type": r["type"],
            "question": r["question"],
            "latency_sec": r["latency_sec"],
            "retrieval_score": r["retrieval_score"],
            "correctness_score": r["correctness_score"],
            "groundedness_score": r["groundedness_score"],
            "memory_score": r["memory_score"] if r["memory_score"] is not None else "",
            "overall_score": r["overall_score"],
            "answer_preview": r["answer"][:300].replace("\n", " "),
        })

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "type",
                "question",
                "latency_sec",
                "retrieval_score",
                "correctness_score",
                "groundedness_score",
                "memory_score",
                "overall_score",
                "answer_preview",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(results: List[Dict[str, Any]]) -> None:
    total = len(results)
    avg_latency = sum(r["latency_sec"] for r in results) / total if total else 0.0
    avg_retrieval = sum(r["retrieval_score"] for r in results) / total if total else 0.0
    avg_correctness = sum(r["correctness_score"] for r in results) / total if total else 0.0
    avg_groundedness = sum(r["groundedness_score"] for r in results) / total if total else 0.0

    memory_scores = [r["memory_score"] for r in results if r["memory_score"] is not None]
    avg_memory = sum(memory_scores) / len(memory_scores) if memory_scores else 0.0

    avg_overall = sum(r["overall_score"] for r in results) / total if total else 0.0

    print("\n" + "=" * 70)
    print("IMPROVED CHATBOT EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total questions        : {total}")
    print(f"Average latency        : {avg_latency:.3f} sec")
    print(f"Average retrieval      : {avg_retrieval:.2f} / 5")
    print(f"Average correctness    : {avg_correctness:.2f} / 5")
    print(f"Average groundedness   : {avg_groundedness:.2f} / 5")
    if memory_scores:
        print(f"Average memory         : {avg_memory:.2f} / 5")
    print(f"Average overall        : {avg_overall:.2f} / 5")
    print("=" * 70)

    print("\nLOW-SCORING CASES:")
    for r in results:
        if r["overall_score"] < 3.0:
            print("-" * 70)
            print(f"ID                : {r['id']}")
            print(f"Type              : {r['type']}")
            print(f"Question          : {r['question']}")
            print(f"Retrieval         : {r['retrieval_score']}/5")
            print(f"Correctness       : {r['correctness_score']}/5")
            print(f"Groundedness      : {r['groundedness_score']}/5")
            if r["memory_score"] is not None:
                print(f"Memory            : {r['memory_score']}/5")
            print(f"Overall           : {r['overall_score']}/5")
            print(f"Answer preview    : {r['answer'][:250]}")
            print(f"Context preview   : {r['context_preview'][:250]}")


def main() -> None:
    questions = load_questions(INPUT_FILE)
    results: List[Dict[str, Any]] = []

    chat_history: List[Dict[str, str]] = []
    previous_turn: Optional[Dict[str, str]] = None

    for item in questions:
        print(f"Evaluating {item['id']} ({item['type']})...")
        if item["type"] in {"memory", "memory_followup"}:
            result = evaluate_one(item, chat_history=chat_history, previous_turn=previous_turn)

            chat_history.append({"role": "user", "content": item["question"]})
            chat_history.append({"role": "assistant", "content": result["answer"]})

            previous_turn = {
                "question": item["question"],
                "answer": result["answer"],
            }
        else:
            result = evaluate_one(item, chat_history=None, previous_turn=None)

        results.append(result)

    json_path = OUTPUT_DIR / "evaluation_results_improved.json"
    csv_path = OUTPUT_DIR / "evaluation_results_improved.csv"

    save_json(results, json_path)
    save_csv(results, csv_path)
    print_summary(results)

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()
