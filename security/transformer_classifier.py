import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "transformer_model")

_pipeline_cache = None


def _load_pipeline():
    global _pipeline_cache
    if _pipeline_cache is None:
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(
                f"No fine-tuned transformer found at {MODEL_DIR}. "
                f"Run fine_tune_transformer.py (ideally on Kaggle GPU) first, "
                f"then copy the output folder here."
            )
        from transformers import pipeline
        _pipeline_cache = pipeline(
            "text-classification",
            model=MODEL_DIR,
            tokenizer=MODEL_DIR,
            top_k=None,  
        )
    return _pipeline_cache


def classify_with_transformer(text: str) -> dict:
    try:
        clf = _load_pipeline()
        results = clf(text[:512])[0] 
        injection_score = next(r["score"] for r in results if r["label"] == "injection")
        return {
            "flagged": injection_score >= 0.5,
            "confidence": float(injection_score),
            "error": None,
        }
    except Exception as e:
        return {"flagged": False, "confidence": 0.0, "error": str(e)}
