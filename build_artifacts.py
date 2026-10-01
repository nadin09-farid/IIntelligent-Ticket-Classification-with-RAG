"""
One-time setup script: trains the ticket classifier and builds the BM25 +
FAISS retrieval indexes from your dataset.

Run this ONCE (and again any time your dataset changes) before starting
the API:

    python build_artifacts.py

Expects a CSV with at least these columns: text, type, priority, queue, answer
(language, version, strata are ignored if present).

Adjust TRAIN_FILE below to point at whichever file you want to train on —
by default it looks for data/processed/train.csv.
"""
import sys

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier

from src.config import CLASSIFIER_PATH, VECTORIZER_PATH, MODELS_DIR, BASE_DIR
from src.retrieval.embeddings import get_embeddings
from src.retrieval.bm25 import build_bm25_index
from src.retrieval.faiss_retriever import build_faiss_index

# Adjust this if your file is named differently or lives elsewhere
TRAIN_FILE = BASE_DIR / "src"/"data" / "processed" / "train.csv"

REQUIRED_COLUMNS = {"text", "type", "priority", "queue"}


def load_dataset(path) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: could not find dataset at {path}")
        print("Edit TRAIN_FILE at the top of this script to point at the right file.")
        sys.exit(1)

    # auto-detects whether the file is comma, tab, or semicolon separated
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"ERROR: dataset is missing required column(s): {missing}")
        print(f"Columns found: {list(df.columns)}")
        sys.exit(1)

    before = len(df)
    df = df.dropna(subset=["text", "type", "priority", "queue"])
    df = df[df["text"].str.strip() != ""]
    after = len(df)
    print(f"Loaded {after} usable rows (dropped {before - after} incomplete rows).")
    return df


def train_classifier(df: pd.DataFrame):
    print("Training classifier...")
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["text"])
    y = df[["type", "priority", "queue"]]

    clf = MultiOutputClassifier(
        LogisticRegression(max_iter=1000, n_jobs=-1)
    )
    clf.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, CLASSIFIER_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Saved classifier -> {CLASSIFIER_PATH}")
    print(f"Saved vectorizer -> {VECTORIZER_PATH}")


def build_retrieval_indexes(df: pd.DataFrame):
    print("Building retrieval indexes...")
    if "answer" in df.columns:
        corpus_df = df.dropna(subset=["answer"])
        corpus_df = corpus_df[corpus_df["answer"].str.strip() != ""]
    else:
        print("WARNING: no 'answer' column found — generated responses will have no reference answers.")
        corpus_df = df

    corpus = corpus_df[["text"] + [c for c in ["answer", "type", "priority", "queue"] if c in corpus_df.columns]].to_dict("records")
    for i, doc in enumerate(corpus):
        doc["id"] = i  # required by hybrid_retriever.py's Reciprocal Rank Fusion

    if not corpus:
        print("ERROR: no rows available to build the retrieval corpus (all missing 'answer').")
        sys.exit(1)

    print(f"Embedding {len(corpus)} documents (this may take a minute)...")
    texts = [doc["text"] for doc in corpus]
    embeddings = get_embeddings(texts)

    build_faiss_index(embeddings, corpus)
    build_bm25_index(corpus)
    print(f"Built retrieval indexes over {len(corpus)} documents.")


if __name__ == "__main__":
    df = load_dataset(TRAIN_FILE)
    train_classifier(df)
    build_retrieval_indexes(df)
    print("\nDone. You can now (re)start the API:")
    print("  uvicorn api:app --reload --port 8000")
