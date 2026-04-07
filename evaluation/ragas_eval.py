"""
ragas_eval.py
-------------
Evaluates the RAG pipeline using the Ragas framework.

Fixes applied:
  - ground_truth renamed to reference (Ragas ≥ 0.1.x schema)
  - Per-question error handling — one failure no longer crashes the whole run
  - Long-term PCC memory wiped before eval — prevents cross-run contamination
  - More golden set questions for statistical validity
"""

import os
import json

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
except ImportError:
    print("ERROR: missing required libraries.")
    print("Install via: pip install ragas datasets")
    exit(1)

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_engine import RAGEngine

# ---------------------------------------------------------------------------
# Golden set — grounded in Dr. Devreotes' chemotaxis / Dictyostelium corpus.
# Add more questions here as the corpus grows.
# ---------------------------------------------------------------------------
GOLDEN_SET = [
    {
        "question": "What role does PTEN play in chemotaxis in Dictyostelium?",
        "ground_truth": (
            "PTEN localises to the back of chemotaxing cells and degrades PI(3,4,5)P3, "
            "restricting PI3K activity to the leading edge to establish front-back polarity."
        ),
    },
    {
        "question": "How does Ras relate to PI3K activation during gradient sensing?",
        "ground_truth": (
            "Ras is activated at the leading edge by GPCR signalling and directly activates "
            "PI3K, amplifying PIP3 production at the front of the cell."
        ),
    },
    {
        "question": "What is the half-life of acetylcholine receptors in chick myotubes?",
        "ground_truth": (
            "The half-life of acetylcholine receptors in chick myotubes is approximately "
            "17 hours under standard culture conditions."
        ),
    },
    {
        "question": "Where are newly synthesized acetylcholine receptors located before reaching the plasma membrane?",
        "ground_truth": (
            "Newly synthesized acetylcholine receptors are located in the Golgi apparatus "
            "prior to transport to the cell surface."
        ),
    },
]


def _clear_pcc_long_term_memory(engine: RAGEngine) -> None:
    """
    FIX: Wipe long-term memory episodes before eval to prevent cross-run
    contamination — previous eval runs would otherwise pollute PCC context.
    """
    if not engine.enable_pcc or not engine.pcc_memory or not engine.pcc_memory.driver:
        return
    try:
        with engine.pcc_memory.driver.session() as session:
            deleted = session.run(
                "MATCH (m:MemoryEpisode {user_id: $uid}) DETACH DELETE m RETURN count(m) AS cnt",
                uid=engine.pcc_memory.user_id,
            ).single()
            cnt = deleted["cnt"] if deleted else 0
            if cnt:
                print(f"  Cleared {cnt} stale PCC memory episode(s) before eval.")
    except Exception as e:
        print(f"  Warning: Could not clear PCC long-term memory ({e})")


def run_ragas_evaluation():
    print("=" * 70)
    print("  GraphRAG — Ragas Evaluation")
    print("=" * 70)

    engine = RAGEngine()
    engine.load()

    # FIX: Clear stale long-term memory before running eval
    _clear_pcc_long_term_memory(engine)

    # Ragas dataset schema (FIX: 'reference' not 'ground_truth')
    data = {
        "question":  [],
        "answer":    [],
        "contexts":  [],
        "reference": [],   # FIX: Ragas ≥ 0.1.x uses 'reference'
    }

    skipped = 0
    print(f"\n[1/2] Answering {len(GOLDEN_SET)} Golden Set queries via RAG Engine...")

    for i, item in enumerate(GOLDEN_SET, 1):
        print(f"  -> Q{i}: {item['question']}")
        engine.reset_conversation()

        try:
            answer, chunks, _intent, _memory_info = engine.ask(
                item["question"], multi_turn=False
            )
            contexts = [chunk["text"] for chunk in chunks]

            data["question"].append(item["question"])
            data["answer"].append(answer)
            data["contexts"].append(contexts)
            data["reference"].append(item["ground_truth"])

        except Exception as e:
            # FIX: Per-question error handling — skip and continue
            print(f"     Warning: Q{i} failed ({e}), skipping.")
            skipped += 1
            continue

    if not data["question"]:
        print("\nNo questions answered successfully. Aborting evaluation.")
        engine.close()
        return

    if skipped:
        print(f"\n  Note: {skipped} question(s) skipped due to errors.")

    print(f"\n[2/2] Running Ragas Evaluation on {len(data['question'])} question(s)...")
    print("  (Sends responses, contexts, and references to the LLM for scoring)")

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
    )

    print("\n" + "=" * 70)
    print("  Ragas Evaluation Summary")
    print("=" * 70)
    print(result)

    os.makedirs("data", exist_ok=True)
    output_file = "data/ragas_results.csv"
    result.to_pandas().to_csv(output_file, index=False)
    print(f"\nDetailed breakdown saved to: {output_file}")
    print("=" * 70)

    engine.close()


if __name__ == "__main__":
    run_ragas_evaluation()
