import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from security.prompt_injection_detector import scan_text_for_injection

DATASET_NAME = "deepset/prompt-injections"

def load_real_dataset(split="test"):
    from datasets import load_dataset
    return load_dataset(DATASET_NAME, split=split)

def normalize_label(raw_label) -> int:
    if isinstance(raw_label, str):
        return 1 if raw_label.strip().upper() in ("INJECTION", "1", "TRUE") else 0
    return int(raw_label)

def evaluate(split: str = "test", use_ml_layer: bool = False, use_transformer_layer: bool = False, use_llm_layer: bool = False, max_examples: int = None, quiet: bool = False):
    ds = load_real_dataset(split=split)

    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    if not quiet:
        print(f"Loaded {len(ds)} examples from {DATASET_NAME} [{split}]")
        print(f"Sample row (verify label mapping matches normalize_label() above): {ds[0]}\n")

    any_layer = use_ml_layer or use_transformer_layer or use_llm_layer
    if any_layer:
        from security.prompt_injection_detector import scan_with_defense_in_depth

    tp = fp = tn = fn = 0
    misses = []

    for row in ds:
        text = row["text"]
        label = normalize_label(row["label"])

        if any_layer:
            predicted = scan_with_defense_in_depth(
                text, use_ml_layer=use_ml_layer, use_transformer_layer=use_transformer_layer, use_llm_layer=use_llm_layer
            )["flagged"]
        else:
            predicted = scan_text_for_injection(text)["flagged"]

        if label == 1 and predicted:
            tp += 1
        elif label == 1 and not predicted:
            fn += 1
            misses.append(("FALSE NEGATIVE (missed attack)", text))
        elif label == 0 and predicted:
            fp += 1
            misses.append(("FALSE POSITIVE (over-flagged)", text))
        else:
            tn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = (tp + tn) / total if total else 0

    layers = ["regex"] + (["ml"] if use_ml_layer else []) + (["transformer"] if use_transformer_layer else []) + (["llm"] if use_llm_layer else [])
    layer_desc = " + ".join(layers)

    if not quiet:
        print(f"{'='*60}")
        print(f"{DATASET_NAME} [{split} split, n={total}] - layers: {layer_desc}")
        print(f"{'='*60}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall:    {recall:.2%}")
        print(f"F1 score:  {f1:.2%}")
        print(f"Accuracy:  {accuracy:.2%}")
        print(f"Confusion matrix: TP={tp} FP={fp} TN={tn} FN={fn}\n")

        if misses:
            print(f"First 10 misses (of {len(misses)} total) - inspect these for real patterns to add to the regex list:")
            for kind, text in misses[:10]:
                print(f"  [{kind}] \"{text[:90]}\"")

    return {"layers": layer_desc, "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "n": total}


def compare_layers(split: str = "test", max_examples: int = None):
   
    configs = [
        ("Regex only", dict(use_ml_layer=False, use_transformer_layer=False)),
        ("Regex + TF-IDF/LogisticRegression", dict(use_ml_layer=True, use_transformer_layer=False)),
        ("Regex + Fine-tuned Transformer", dict(use_ml_layer=False, use_transformer_layer=True)),
    ]

    results = []
    for name, kwargs in configs:
        print(f"\nRunning: {name}...")
        result = evaluate(split=split, max_examples=max_examples, quiet=True, **kwargs)
        results.append((name, result))

    print(f"\n{'='*70}")
    print(f"COMPARISON on {DATASET_NAME} [{split} split]")
    print(f"{'='*70}")
    print(f"{'Approach':<38} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    for name, r in results:
        print(f"{name:<38} {r['precision']:>9.2%} {r['recall']:>9.2%} {r['f1']:>9.2%}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-ml", action="store_true", help="Also test the TF-IDF/LogisticRegression layer (requires train_classifier.py run first)")
    parser.add_argument("--with-transformer", action="store_true", help="Also test the fine-tuned transformer layer (requires fine_tune_transformer.py run first)")
    parser.add_argument("--with-llm", action="store_true", help="Also test the LLM layer (requires GEMINI_API_KEY set - slow, one API call per example)")
    parser.add_argument("--compare", action="store_true", help="Run regex-only, +ml, and +transformer as SEPARATE evals and print them side by side")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--max-examples", type=int, default=None, help="Limit dataset size for a quick run")
    args = parser.parse_args()

    if args.compare:
        compare_layers(split=args.split, max_examples=args.max_examples)
    else:
        evaluate(
            split=args.split,
            use_ml_layer=args.with_ml,
            use_transformer_layer=args.with_transformer,
            use_llm_layer=args.with_llm,
            max_examples=args.max_examples,
        )
