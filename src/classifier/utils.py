from typing import Dict, Any

def format_classifier_input(text: str) -> str:
    """Formats raw text for classification pipeline input."""
    return text.strip()

def decode_predictions(raw_preds: Dict[str, Any]) -> Dict[str, str]:
    """Converts raw model outputs/indices into readable target label strings."""
    return {
        "type": str(raw_preds.get("type", "Unknown")),
        "priority": str(raw_preds.get("priority", "Low")),
        "queue": str(raw_preds.get("queue", "General"))
    }