"""
FastAPI application entrypoint.

Responsibilities:
- Create the FastAPI app
- Load static data on startup
- Expose HTTP endpoints
- Delegate ML logic to app.ml.embeddings

This file should stay thin: no ML logic, no heavy computation at import time.
"""

from fastapi import FastAPI
from typing import List

from app.ml.embeddings import embed, cosine_sim


app = FastAPI(title="AI APEX Semantic Search API")


# --- Static data (example corpus) -------------------------------------------
# In production this would usually come from a DB, vector store, or file.
TEXTS: List[str] = [
    "Oracle APEX is a low-code development platform.",
    "FastAPI is a modern Python web framework.",
    "Vector databases store embeddings for similarity search.",
    "Cats like to sleep during the day.",
]

# Cached embeddings (computed once on startup)
TEXT_EMBEDDINGS = []


# --- Application lifecycle --------------------------------------------------
@app.on_event("startup")
def startup_event() -> None:
    """
    Runs once when the application starts.

    We pre-compute embeddings here instead of at import time so that:
    - startup behavior is explicit
    - reloads are predictable
    - this code is safe for testing and multi-worker setups
    """
    global TEXT_EMBEDDINGS
    TEXT_EMBEDDINGS = [embed(text) for text in TEXTS]


# --- Routes -----------------------------------------------------------------
@app.get("/")
def root() -> dict:
    """Health check endpoint."""
    return {"status": "AI APEX API running"}


@app.post("/search")
def search(query: str) -> dict:
    """
    Perform semantic search over the in-memory text corpus.

    Args:
        query: Free-text search query (query parameter)

    Returns:
        Best matching text and similarity score
    """
    # Generate embedding for the query
    query_embedding = embed(query)

    # Compute cosine similarity against cached embeddings
    scores = [
        cosine_sim(query_embedding, emb)
        for emb in TEXT_EMBEDDINGS
    ]

    # Find index of best match
    best_index = int(max(range(len(scores)), key=lambda i: scores[i]))

    return {
        "query": query,
        "best_match": TEXTS[best_index],
        "score": scores[best_index],
    }
