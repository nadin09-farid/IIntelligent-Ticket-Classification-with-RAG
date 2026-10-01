import os
from pathlib import Path
from dotenv import load_dotenv          # ADD THIS

load_dotenv() 

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "combined_dataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_dataset.csv"
MODELS_DIR = BASE_DIR / "models"

# Model Paths
CLASSIFIER_PATH = MODELS_DIR / "classifier.joblib"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index.bin"
BM25_INDEX_PATH = MODELS_DIR / "bm25_index.pkl"

# Model Configs
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM Configs
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = "openai/gpt-oss-20b"

# Retrieval Configs
TOP_K_BM25 = 10
TOP_K_FAISS = 10
TOP_K_FINAL = 3
RRF_K = 60