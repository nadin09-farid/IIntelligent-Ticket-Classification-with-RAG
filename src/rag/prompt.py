from typing import List, Dict, Any

def construct_rag_prompt(ticket_text: str, predictions: Dict[str, str], contexts: List[Dict[str, Any]]) -> str:
    """Builds a structured prompt incorporating ticket metadata and retrieved contexts."""
    context_str = ""
    for idx, ctx in enumerate(contexts, 1):
        context_str += f"\n--- Reference Answer {idx} ---\n{ctx.get('answer', ctx.get('text', ''))}\n"
        
    prompt = f"""You are an expert customer support assistant. Answer the following support ticket accurately based on the context provided.

Ticket Metadata:
- Type: {predictions.get('type', 'Unknown')}
- Priority: {predictions.get('priority', 'Medium')}
- Target Queue: {predictions.get('queue', 'General')}

User Ticket:
{ticket_text}

Relevant Context & Past Answers:
{context_str}

Instruction:
Generate a professional, helpful, and concise response to address the user's ticket request.
Response:"""
    return prompt