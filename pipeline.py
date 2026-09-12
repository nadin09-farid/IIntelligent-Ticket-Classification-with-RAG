import logging
from typing import Dict, Any

from src.data.preprocessing import construct_ticket_text
from src.classifier.predict import predict_ticket, ClassifierNotAvailableError
from src.retrieval.hybrid_retriever import hybrid_search
from src.rag.prompt import construct_rag_prompt
from src.rag.generator import (
    generate_response_from_llm,
    LLMNotConfiguredError,
    LLMGenerationError,
)

logger = logging.getLogger(__name__)


def run_pipeline(subject: str, body: str) -> Dict[str, Any]:
    """
    Executes the end-to-end Support Ticket Classification and RAG Generation pipeline.

    Unlike the original version, this does not let a missing classifier or a
    failed LLM call silently produce fake-looking output. Failures are
    surfaced in the returned "errors" field so callers (API layer, UI, logs)
    can tell success apart from failure.
    """
    ticket_text = construct_ticket_text(subject, body)

    errors: Dict[str, Any] = {}

    # 1. Classification
    try:
        predictions = predict_ticket(ticket_text)
    except ClassifierNotAvailableError as e:
        logger.error("Classification failed: %s", e)
        predictions = {"type": None, "priority": None, "queue": None}
        errors["classification"] = str(e)

    # 2. Hybrid Retrieval (BM25 + FAISS -> RRF -> Reranker)
    contexts = hybrid_search(ticket_text)

    # 3. Prompt Construction
    prompt = construct_rag_prompt(ticket_text, predictions, contexts)

    # 4. LLM Response Generation
    try:
        generated_response = generate_response_from_llm(prompt)
    except (LLMNotConfiguredError, LLMGenerationError) as e:
        logger.error("Response generation failed: %s", e)
        generated_response = None
        errors["generation"] = str(e)

    return {
        "input_ticket": ticket_text,
        "predictions": predictions,
        "retrieved_contexts": contexts,
        "prompt": prompt,
        "generated_response": generated_response,
        "errors": errors,  # empty dict means everything succeeded
    }
