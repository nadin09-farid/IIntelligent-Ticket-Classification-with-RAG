from typing import List, Dict, Any
import pickle
import faiss
import numpy as np
from src.config import FAISS_INDEX_PATH, TOP_K_FAISS
from src.retrieval.embeddings import get_embeddings

_FAISS_INDEX = None
_DOCS = []

def build_faiss_index(embeddings: np.ndarray, docs: List[Dict[str, Any]]):
    """Builds and persists a FAISS vector index."""
    global _FAISS_INDEX, _DOCS
    _DOCS = docs
    dimension = embeddings.shape[1]
    _FAISS_INDEX = faiss.IndexFlatIP(dimension)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    _FAISS_INDEX.add(embeddings)
    
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(_FAISS_INDEX, str(FAISS_INDEX_PATH))
    with open(str(FAISS_INDEX_PATH) + ".docs.pkl", "wb") as f:
        pickle.dump(_DOCS, f)

def search_faiss(query: str, top_k: int = TOP_K_FAISS) -> List[Dict[str, Any]]:
    """Performs dense vector retrieval via FAISS."""
    global _FAISS_INDEX, _DOCS
    if _FAISS_INDEX is None and FAISS_INDEX_PATH.exists():
        _FAISS_INDEX = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(str(FAISS_INDEX_PATH) + ".docs.pkl", "rb") as f:
            _DOCS = pickle.load(f)
            
    if _FAISS_INDEX is None or not _DOCS:
        return []

    q_embedding = get_embeddings([query])
    faiss.normalize_L2(q_embedding)
    
    distances, indices = _FAISS_INDEX.search(q_embedding, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(_DOCS) and idx >= 0:
            item = _DOCS[idx].copy()
            item["score"] = float(dist)
            results.append(item)
    return results