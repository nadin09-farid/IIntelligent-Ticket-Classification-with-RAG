"""
AI feature: groundedness guardrail.

Before a generated response reaches a customer, this asks the LLM to check
whether the response is actually supported by the retrieved contexts, or
whether it contains claims the retrieval step never surfaced (a common
RAG failure mode). This is a second, independent LLM call acting as a
lightweight fact-checker over the first.
"""
import json
from typing import Any, Dict, List

from src.rag.generator import get_llm_client
from src.config import GROQ_MODEL_NAME


def check_groundedness(generated_response: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Flags whether the generated response is supported by retrieved context."""
    client = get_llm_client()
    if client is None or not contexts:
        return {"is_grounded": None, "notes": "Skipped: no LLM client or no retrieved context."}

    context_block = "\n".join(
        f"[{i + 1}] {c.get('answer', c.get('text', ''))}" for i, c in enumerate(contexts)
    )

    prompt = f"""You are a fact-checking assistant. Given the reference context and a
draft customer support response, determine whether the response is fully
supported by the context, or whether it contains claims not present in it.

Reference Context:
{context_block}

Draft Response:
{generated_response}

Respond ONLY with valid JSON, no markdown fences:
{{"is_grounded": <true or false>, "notes": "<one short sentence explaining why>"}}"""

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You output only strict JSON, no markdown, no commentary."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        raw = completion.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()

        if not raw:
            return {"is_grounded": None, "notes": "LLM returned an empty response (likely ran out of tokens before answering)."}

        parsed = json.loads(raw)
        return {
            "is_grounded": bool(parsed.get("is_grounded", True)),
            "notes": str(parsed.get("notes", ""))[:200],
        }
    except Exception as e:
        return {"is_grounded": None, "notes": f"Groundedness check failed: {str(e)}"}
