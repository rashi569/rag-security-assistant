import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_chain import retrieve_context

RETRIEVAL_EVAL_SET = [
    {"question": "How many vacation days do employees get per year?", "expected_source": "vacation_policy.txt"},
    {"question": "What is the carryover limit for unused vacation?", "expected_source": "vacation_policy.txt"},
    {"question": "Is vacation paid out when an employee leaves?", "expected_source": "vacation_policy.txt"},
    {"question": "What is the reimbursement limit for client meals?", "expected_source": "expense_policy_ATTACK_DEMO.txt"},
    {"question": "How long do I have to submit an expense receipt?", "expected_source": "expense_policy_ATTACK_DEMO.txt"},
    {"question": "How many business days does reimbursement take after approval?", "expected_source": "expense_policy_ATTACK_DEMO.txt"},
]

def evaluate_retrieval(top_k: int = 4):
    hits = 0
    total = len(RETRIEVAL_EVAL_SET)
    results = []

    for item in RETRIEVAL_EVAL_SET:
        retrieval = retrieve_context(item["question"], top_k=top_k)
        retrieved_sources = retrieval["sources"]
        hit = item["expected_source"] in retrieved_sources
        hits += int(hit)
        results.append((item["question"], item["expected_source"], retrieved_sources, hit))

    precision_at_k = hits / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"RETRIEVAL EVALUATION (precision@{top_k})")
    print(f"{'='*60}")
    print(f"{hits}/{total} questions retrieved at least one chunk from the correct source document.")
    print(f"Precision@{top_k}: {precision_at_k:.2%}\n")

    for question, expected, retrieved, hit in results:
        status = "✓" if hit else "✗ MISS"
        print(f"  [{status}] \"{question}\"")
        print(f"      expected: {expected} | retrieved from: {set(retrieved)}")

    print(f"\nNote: this is a small (n={total}), hand-labeled eval set covering only the")
    print(f"two sample documents shipped with this project - useful to catch obvious")
    print(f"retrieval regressions, not a statistically rigorous benchmark. A stronger")
    print(f"version would use a larger, more diverse document set and public QA benchmarks.\n")

    return precision_at_k


if __name__ == "__main__":
    evaluate_retrieval()
