"""
llm_judge.py
------------
Evaluates the chatbot using LLM-as-a-Judge pattern.
Uses OpenAI gpt-4o-mini as an independent judge (separate model family
from the Mistral production pipeline) to avoid self-preference bias.

Fixes applied:
  - elapsed time saved in results
  - Score range validation — out-of-range or unparseable scores logged, not silently zeroed
  - Magic number weights replaced with named constants
  - Golden set ground truths now grounded in actual paper content
  - memory_info included in results for traceability
"""

import json
import os
import time
import openai
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_engine import RAGEngine, REFUSAL_MESSAGE

# ---------------------------------------------------------------------------
# Scoring weights (named constants — not magic numbers)
# ---------------------------------------------------------------------------
FAITHFULNESS_WEIGHT = 0.4
RELEVANCE_WEIGHT    = 0.6
PASS_THRESHOLD      = 0.6

# ---------------------------------------------------------------------------
# Golden set — replace with questions specific to your corpus
# These examples target Dr. Devreotes' Dictyostelium / chemotaxis research.
# ---------------------------------------------------------------------------
GOLDEN_SET = [
    {
        "question": "What role does PTEN play in chemotaxis in Dictyostelium?",
        "ground_truth": (
            "PTEN is a phosphatase that localises to the back of the cell and degrades "
            "PI(3,4,5)P3, restricting PI3K signalling to the leading edge and thereby "
            "establishing the front-back polarity required for directed cell migration."
        ),
    },
    {
        "question": "How does Ras activation relate to G-protein coupled receptor signalling?",
        "ground_truth": (
            "G-protein coupled receptors activate Ras at the leading edge of the cell, "
            "which in turn stimulates PI3K to produce PIP3, amplifying the directional "
            "signal perceived by the cell during chemotaxis."
        ),
    },
    {
        "question": "What methodology is used to study cell polarity in the Devreotes lab?",
        "ground_truth": (
            "Fluorescence microscopy of GFP-tagged biosensors (e.g. PHAkt-GFP for PIP3) "
            "in Dictyostelium cells exposed to cAMP gradients, combined with genetic "
            "knockouts and rescue experiments."
        ),
    },
]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _parse_score(raw: str, context_label: str) -> float:
    """Parse a float score from LLM output with range validation and logging."""
    try:
        score = float(raw.strip())
        if not 0.0 <= score <= 1.0:
            print(f"  Warning: {context_label} score out of range ({score}), clamping to [0,1]")
            score = max(0.0, min(1.0, score))
        return score
    except (ValueError, AttributeError) as e:
        print(f"  Warning: Could not parse {context_label} score '{raw}' ({e}). Defaulting to 0.0")
        return 0.0


def evaluate_faithfulness(client: openai.OpenAI, answer: str, context: str) -> float:
    prompt = (
        "Does the following answer reflect ONLY what the provided context says "
        "without adding outside knowledge? "
        "Score from 0.0 to 1.0. Output ONLY the float number, nothing else.\n\n"
        f"Context:\n{context}\n\nAnswer:\n{answer}"
    )
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return _parse_score(res.choices[0].message.content or "", "faithfulness")


def evaluate_relevance(client: openai.OpenAI, answer: str, question: str) -> float:
    prompt = (
        "Does the following answer directly and completely address the question asked? "
        "Score from 0.0 to 1.0. Output ONLY the float number, nothing else.\n\n"
        f"Question:\n{question}\n\nAnswer:\n{answer}"
    )
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return _parse_score(res.choices[0].message.content or "", "relevance")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_judge_evaluation(output_file: str = "data/judge_results.json") -> None:
    print("=" * 70)
    print("  GraphRAG — LLM-as-Judge Evaluator (OpenAI gpt-4o-mini)")
    print(f"  Faithfulness weight: {FAITHFULNESS_WEIGHT}  |  Relevance weight: {RELEVANCE_WEIGHT}")
    print(f"  Pass threshold: {PASS_THRESHOLD}")
    print("=" * 70)

    engine = RAGEngine()
    engine.load()
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    results = []

    for i, item in enumerate(GOLDEN_SET, 1):
        print(f"\n[{i}/{len(GOLDEN_SET)}] Q: {item['question']}")
        print("  Generating answer...")

        engine.reset_conversation()
        start = time.time()

        try:
            answer, chunks, _intent, memory_info = engine.ask(
                item["question"], multi_turn=False
            )
            elapsed = round(time.time() - start, 2)

            context_text = "\n\n".join([c["text"] for c in chunks])

            faithfulness = evaluate_faithfulness(client, answer, context_text)
            relevance    = evaluate_relevance(client, answer, item["question"])
            verdict      = (FAITHFULNESS_WEIGHT * faithfulness) + (RELEVANCE_WEIGHT * relevance)
            passed       = verdict >= PASS_THRESHOLD

            print(f"  A: {answer[:120]}...")
            print(f"  Faithfulness : {faithfulness:.3f}")
            print(f"  Relevance    : {relevance:.3f}")
            print(f"  Verdict      : {verdict:.3f}  ({'PASS' if passed else 'FAIL'})")
            print(f"  Latency      : {elapsed}s")
            print("-" * 70)

            results.append({
                "question":      item["question"],
                "ground_truth":  item["ground_truth"],
                "chatbot_answer": answer,
                "scores": {
                    "faithfulness": faithfulness,
                    "relevance":    relevance,
                    "verdict":      round(verdict, 4),
                    "passed":       passed,
                },
                "latency_seconds": elapsed,
                "memory_info":    memory_info,
                "status":         "ok",
            })

        except Exception as e:
            elapsed = round(time.time() - start, 2)
            print(f"  ERROR: {e}")
            results.append({
                "question":        item["question"],
                "status":          "error",
                "error":           str(e),
                "latency_seconds": elapsed,
            })

    os.makedirs("data", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    ok_results  = [r for r in results if r["status"] == "ok"]
    avg_verdict = sum(r["scores"]["verdict"] for r in ok_results) / max(len(ok_results), 1)
    pass_count  = sum(1 for r in ok_results if r["scores"]["passed"])
    print(f"\nSummary: {pass_count}/{len(ok_results)} passed | Avg verdict: {avg_verdict:.3f}")
    print(f"Results saved to {output_file}")

    engine.close()


if __name__ == "__main__":
    run_judge_evaluation()
