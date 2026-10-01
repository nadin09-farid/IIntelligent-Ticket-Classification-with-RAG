from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME

_EMBEDDER = None

def load_embedding_model() -> SentenceTransformer:
    """Loads and caches the SentenceTransformer model."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDER
def get_embeddings(texts: List[str]) -> np.ndarray:
    """Generates dense embedding vectors for a list of input texts."""
    model = load_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings