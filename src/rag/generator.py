import os
from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL_NAME

def get_llm_client():
    """Initializes Groq client with environment API key."""
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    if not api_key:
        return None
    return Groq(api_key=api_key)

def generate_response_from_llm(prompt: str) -> str:
    """Generates text response using Groq LLM API."""
    client = get_llm_client()
    if client is None:
        return "Error: GROQ_API_KEY environment variable is not set."
        
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful customer support agent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"