from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from src.config import RERANKER_MODEL_NAME, RRF_K, TOP_K_FINAL
from src.retrieval.bm25 import search_bm25
from src.retrieval.faiss_retriever import search_faiss

_RERANKER = None

def get_reranker() -> CrossEncoder:
    """Loads and caches CrossEncoder reranking model."""
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder(RERANKER_MODEL_NAME)
    return _RERANKER

def reciprocal_rank_fusion(
    bm25_results: List[Dict[str, Any]], 
    faiss_results: List[Dict[str, Any]], 
    k: int = RRF_K
) -> List[Dict[str, Any]]:
    """Combines BM25 and FAISS search rankings using Reciprocal Rank Fusion."""
    rrf_scores = {}
    doc_map = {}

    for rank, doc in enumerate(bm25_results):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    for rank, doc in enumerate(faiss_results):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
    fused_docs = []
    for doc_id in sorted_doc_ids:
        item = doc_map[doc_id].copy()
        item["rrf_score"] = rrf_scores[doc_id]
        fused_docs.append(item)
    return fused_docs

def rerank_contexts(query: str, candidates: List[Dict[str, Any]], top_k: int = TOP_K_FINAL) -> List[Dict[str, Any]]:
    """Reranks fusion candidate contexts using CrossEncoder."""
    if not candidates:
        return []
        
    reranker = get_reranker()
    pairs = [[query, doc["text"]] for doc in candidates]
    scores = reranker.predict(pairs)
    
    for idx, score in enumerate(scores):
        candidates[idx]["rerank_score"] = float(score)
        
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]

def hybrid_search(query: str, top_k: int = TOP_K_FINAL) -> List[Dict[str, Any]]:
    """Executes full hybrid search pipeline: BM25 + FAISS -> RRF -> Reranker."""
    bm25_hits = search_bm25(query)
    faiss_hits = search_faiss(query)
    fused_hits = reciprocal_rank_fusion(bm25_hits, faiss_hits)
    top_contexts = rerank_contexts(query, fused_hits, top_k=top_k)
    return top_contexts