"""
FastAPI wrapper exposing the ticket classification + RAG pipeline as a
deployable HTTP API.

Run locally:
    uvicorn api:app --reload --port 8000

Then try it at http://127.0.0.1:8000/docs (auto-generated Swagger UI).

Endpoints:
    GET  /health              -> liveness check
    POST /tickets/process     -> runs the full pipeline on a ticket
"""
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline import run_pipeline
app = FastAPI(
    title="Intelligent Ticket Classification & RAG API",
    description=(
        "Classifies support tickets and generates RAG-backed responses, "
        "with AI summarization, sentiment/urgency scoring, and a groundedness guardrail."
    ),
    version="1.0.0",
)


class TicketRequest(BaseModel):
    subject: str = Field(..., min_length=1, description="Ticket subject line")
    body: str = Field(..., min_length=1, description="Ticket body / description")


class TicketResponse(BaseModel):
    input_ticket: str
    summary: str
    predictions: Dict[str, str]
    sentiment_analysis: Dict[str, str]
    retrieved_contexts: List[Dict[str, Any]]
    generated_response: str
    groundedness: Dict[str, Any]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/tickets/process", response_model=TicketResponse)
def process_ticket(payload: TicketRequest) -> Dict[str, Any]:
    try:
        result = run_pipeline(payload.subject, payload.body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    # Internal prompt text isn't part of the public API contract
    result.pop("prompt", None)
    return result
