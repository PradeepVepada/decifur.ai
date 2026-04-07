"""
evaluate.py
-----------
Runs the chatbot against a set of benchmark questions and outputs a report.

Usage:
    python evaluate.py
"""

import json
import os
import time
from dotenv import load_dotenv
load_dotenv()

from rag_engine import RAGEngine


QUESTIONS = [
    {
        "category": "Overview",
        "question": "What are the main research themes in this corpus?",
    },
    {
        "category": "Methods",
        "question": "What experimental methods recur across these papers?",
    },
    {
        "category": "Authors",
        "question": "Which authors appear most often in this corpus?",
    },
    {
        "category": "Key Findings",
        "question": "What are the most significant findings across these papers?",
    },
    {
        "category": "Connections",
        "question": "How do the papers relate to each other?",
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_evaluation(output_file: str = "data/evaluation_results.json") -> None:
    print("=" * 70)
    print("  GraphRAG Chatbot — Evaluation")
    print("=" * 70)

    engine = RAGEngine()
    engine.load()

    results = []

    for i, item in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/{len(QUESTIONS)}] Category: {item['category']}")
        print(f"  Q: {item['question']}")
        print("  Answering...", end="", flush=True)

        start = time.time()
        # Each question is independent (no conversation history)
        engine.reset_conversation()

        try:
            answer, chunks, _intent, memory_info = engine.ask(
                item["question"], multi_turn=False
            )
            elapsed = time.time() - start

            sources = list({
                c["source"]: f"{c['title']} ({c['year']})"
                for c in chunks
            }.values())

            print(f"\r  A: {answer[:120]}{'...' if len(answer) > 120 else ''}")
            print(f"     Sources: {' | '.join(s[:40] for s in sources)}")
            print(f"     Time: {elapsed:.1f}s")

            results.append({
                "category": item["category"],
                "question": item["question"],
                "answer": answer,
                "sources_cited": sources,
                "top_chunk_scores": [c["score"] for c in chunks],
                "elapsed_seconds": round(elapsed, 2),
                "status": "ok",
            })

        except Exception as e:
            print(f"\r  ERROR: {e}")
            results.append({
                "category": item["category"],
                "question": item["question"],
                "answer": None,
                "error": str(e),
                "status": "error",
            })

    # Save results
    os.makedirs("data", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{'=' * 70}")
    print(f"  Completed: {ok}/{len(QUESTIONS)} questions answered successfully")
    print(f"  Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
