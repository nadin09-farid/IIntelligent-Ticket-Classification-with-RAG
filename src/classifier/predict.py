from typing import Dict
import joblib
from src.config import CLASSIFIER_PATH, VECTORIZER_PATH
from src.classifier.utils import decode_predictions

_CLASSIFIER_MODEL = None
_VECTORIZER = None

def load_classifier_models():
    """Lazy loader for classifier and vectorizer models."""
    global _CLASSIFIER_MODEL, _VECTORIZER
    if _CLASSIFIER_MODEL is None and CLASSIFIER_PATH.exists():
        _CLASSIFIER_MODEL = joblib.load(CLASSIFIER_PATH)
    if _VECTORIZER is None and VECTORIZER_PATH.exists():
        _VECTORIZER = joblib.load(VECTORIZER_PATH)
    return _CLASSIFIER_MODEL, _VECTORIZER

def predict_ticket(text: str) -> Dict[str, str]:
    """
    Predicts ticket category target attributes (type, priority, queue) for an input ticket text.
    """
    clf, vec = load_classifier_models()
    
    # Fallback/Mock prediction logic if model files are missing
    if clf is None or vec is None:
        return {
            "type": "Incident",
            "priority": "high",
            "queue": "Technical Support"
        }

    features = vec.transform([text])
    preds = clf.predict(features)
    
    raw_results = {
        "type": preds[0][0] if len(preds[0]) > 0 else "Incident",
        "priority": preds[0][1] if len(preds[0]) > 1 else "medium",
        "queue": preds[0][2] if len(preds[0]) > 2 else "Technical Support"
    }
    return decode_predictions(raw_results)