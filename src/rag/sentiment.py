"""
AI feature: sentiment & urgency analysis.

This is an independent, LLM-based triage signal that complements (not
replaces) the trained ML priority classifier in predict.py. Two signals
that disagree ("priority: low" from the classifier vs. "urgency_score: 5,
sentiment: angry" from the LLM) is itself a useful escalation trigger.
"""
import json
from typing import Dict

from src.rag.generator import get_llm_client
from src.config import GROQ_MODEL_NAME

_ALLOWED_SENTIMENT = {"frustrated", "angry", "neutral", "satisfied", "confused"}

_FALLBACK: Dict[str, str] = {
    "sentiment": "neutral",
    "urgency_score": "3",
    "rationale": "LLM unavailable, default values used.",
}


def analyze_sentiment_urgency(ticket_text: str) -> Dict[str, str]:
    """Scores customer sentiment (categorical) and urgency (1-5) via the LLM."""
    client = get_llm_client()
    if client is None:
        return dict(_FALLBACK)

    prompt = f"""Analyze the customer's emotional tone and urgency in this support ticket.

Ticket:
{ticket_text}

Respond ONLY with valid JSON in this exact shape, no extra text, no markdown fences:
{{"sentiment": "<one of: frustrated, angry, neutral, satisfied, confused>", "urgency_score": <integer 1-5>, "rationale": "<one short sentence>"}}"""

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You output only strict JSON, no markdown, no commentary."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
max_tokens=400,        )
        raw = completion.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)

        sentiment = str(parsed.get("sentiment", "neutral")).lower()
        if sentiment not in _ALLOWED_SENTIMENT:
            sentiment = "neutral"

        urgency = parsed.get("urgency_score", 3)
        try:
            urgency = max(1, min(5, int(urgency)))
        except (TypeError, ValueError):
            urgency = 3

        return {
            "sentiment": sentiment,
            "urgency_score": str(urgency),
            "rationale": str(parsed.get("rationale", ""))[:200],
        }
    except Exception:
        return dict(_FALLBACK)
