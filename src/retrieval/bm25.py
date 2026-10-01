from typing import List, Dict, Any
import pickle
from rank_bm25 import BM25Okapi
from src.config import BM25_INDEX_PATH, TOP_K_BM25

_BM25_INDEX = None
_CORPUS = []

def build_bm25_index(corpus: List[Dict[str, Any]]):
    """Builds and persists BM25 index from knowledge base corpus."""
    global _BM25_INDEX, _CORPUS
    _CORPUS = corpus
    tokenized_corpus = [doc["text"].lower().split(" ") for doc in corpus]
    _BM25_INDEX = BM25Okapi(tokenized_corpus)
    
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump((_BM25_INDEX, _CORPUS), f)

def search_bm25(query: str, top_k: int = TOP_K_BM25) -> List[Dict[str, Any]]:
    """Performs keyword BM25 sparse retrieval."""
    global _BM25_INDEX, _CORPUS
    if _BM25_INDEX is None and BM25_INDEX_PATH.exists():
        with open(BM25_INDEX_PATH, "rb") as f:
            _BM25_INDEX, _CORPUS = pickle.load(f)
            
    if _BM25_INDEX is None or not _CORPUS:
        return []

    tokenized_query = query.lower().split(" ")
    scores = _BM25_INDEX.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    results = []
    for idx in top_indices:
        item = _CORPUS[idx].copy()
        item["score"] = float(scores[idx])
        results.append(item)
    return results