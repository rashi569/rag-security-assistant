import sys
import os
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.dataset import get_eval_set
from security.prompt_injection_detector import scan_text_for_injection


def evaluate(use_llm_layer: bool = False):
    dataset = get_eval_set()

    tp = fp = tn = fn = 0
    by_category = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    misses = []

    if use_llm_layer:
        from security.prompt_injection_detector import scan_with_defense_in_depth

    for item in dataset:
        text, label, category = item["text"], item["label"], item["category"]

        if use_llm_layer:
            result = scan_with_defense_in_depth(text, use_llm_layer=True)
            predicted = result["flagged"]
        else:
            result = scan_text_for_injection(text)
            predicted = result["flagged"]

        if label == 1 and predicted:
            tp += 1
            by_category[category]["tp"] += 1
        elif label == 1 and not predicted:
            fn += 1
            by_category[category]["fn"] += 1
            misses.append(("FALSE NEGATIVE (missed attack)", text, category))
        elif label == 0 and predicted:
            fp += 1
            by_category[category]["fp"] += 1
            misses.append(("FALSE POSITIVE (over-flagged)", text, category))
        else:
            tn += 1
            by_category[category]["tn"] += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0

    layer_desc = "Regex + LLM (defense-in-depth)" if use_llm_layer else "Regex only"
    print(f"\n{'='*60}")
    print(f"INJECTION DETECTOR EVALUATION - {layer_desc}")
    print(f"{'='*60}")
    print(f"Dataset size: {total} examples ({sum(1 for i in dataset if i['label']==1)} malicious, {sum(1 for i in dataset if i['label']==0)} benign)\n")

    print(f"Precision: {precision:.2%}  (of what we flagged, how much was truly malicious)")
    print(f"Recall:    {recall:.2%}  (of actual attacks, how many we caught)")
    print(f"F1 score:  {f1:.2%}")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"\nConfusion matrix: TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    print(f"\n--- Breakdown by category ---")
    for cat, counts in sorted(by_category.items()):
        cat_total = sum(counts.values())
        cat_correct = counts["tp"] + counts["tn"]
        print(f"  {cat:28s}  {cat_correct}/{cat_total} correct")

    if misses:
        print(f"\n--- Misses (be specific about what the detector gets wrong) ---")
        for kind, text, category in misses:
            print(f"  [{kind}] ({category})")
            print(f"    \"{text[:80]}\"")

    print()
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true", help="Also test the LLM detection layer (requires GEMINI_API_KEY set)")
    args = parser.parse_args()

    evaluate(use_llm_layer=args.with_llm)
