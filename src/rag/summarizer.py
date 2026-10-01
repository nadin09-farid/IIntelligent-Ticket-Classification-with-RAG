"""
AI feature: ticket summarization.

Generates a short, one-line gist of an incoming ticket, meant for agent
dashboards / queue views where reading the full ticket body isn't practical.
"""
from src.rag.generator import get_llm_client
from src.config import GROQ_MODEL_NAME


def summarize_ticket(ticket_text: str, max_words: int = 25) -> str:
    """Generates a short, one-line summary of a support ticket using the LLM."""
    client = get_llm_client()

    if client is None:
        # Fallback: naive truncation so the field is never empty
        return ticket_text[:120] + ("..." if len(ticket_text) > 120 else "")

    prompt = (
        f"Summarize the following customer support ticket in one clear sentence "
        f"(max {max_words} words). Do not add commentary, greetings, or labels — "
        f"output only the summary itself.\n\n"
        f"Ticket:\n{ticket_text}\n\nSummary:"
    )

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You write extremely concise, factual one-line ticket summaries."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
max_tokens=300,        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Summary unavailable: {str(e)}"
