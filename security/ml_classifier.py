import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml_model.joblib")

_model_cache = None


def _load_model():
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run `python train_classifier.py` first."
            )
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def classify_with_ml(text: str) -> dict:
    try:
        model = _load_model()
        proba = model.predict_proba([text])[0]
        injection_confidence = float(proba[1])
        return {
            "flagged": injection_confidence >= 0.5,
            "confidence": injection_confidence,
            "error": None,
        }
    except Exception as e:
        return {"flagged": False, "confidence": 0.0, "error": str(e)}
