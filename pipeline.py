from typing import Dict, Any

from src.data.preprocessing import construct_ticket_text
from src.classifier.predict import predict_ticket
from src.retrieval.hybrid_retriever import hybrid_search
from src.rag.prompt import construct_rag_prompt
from src.rag.generator import generate_response_from_llm
from src.rag.summarizer import summarize_ticket
from src.rag.sentiment import analyze_sentiment_urgency
from src.rag.groundedness import check_groundedness


def run_pipeline(subject: str, body: str) -> Dict[str, Any]:
    """
    End-to-end Support Ticket Classification and RAG Generation pipeline.

    New in this version:
      - AI-generated one-line ticket summary (for dashboards/queue views)
      - LLM-based sentiment & urgency scoring, independent of the trained
        priority classifier (disagreement between the two is itself a
        useful escalation signal)
      - A groundedness guardrail that checks the generated response against
        the retrieved contexts before it's returned, to catch hallucinated
        claims
    """
    # 1. Preprocessing
    ticket_text = construct_ticket_text(subject, body)

    # 2. Classification (trained ML model: type / priority / queue)
    predictions = predict_ticket(ticket_text)

    # 3. AI Summary
    summary = summarize_ticket(ticket_text)

    # 4. AI Sentiment & Urgency (independent LLM-based triage signal)
    sentiment_analysis = analyze_sentiment_urgency(ticket_text)

    # 5. Hybrid Retrieval (BM25 + FAISS -> RRF -> Reranker)
    contexts = hybrid_search(ticket_text)

    # 6. Prompt Construction
    prompt = construct_rag_prompt(ticket_text, predictions, contexts)

    # 7. LLM Response Generation
    generated_response = generate_response_from_llm(prompt)

    # 8. Groundedness Guardrail
    groundedness = check_groundedness(generated_response, contexts)

    return {
        "input_ticket": ticket_text,
        "summary": summary,
        "predictions": predictions,
        "sentiment_analysis": sentiment_analysis,
        "retrieved_contexts": contexts,
        "prompt": prompt,
        "generated_response": generated_response,
        "groundedness": groundedness,
    }
