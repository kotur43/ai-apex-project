"""
Embedding utilities.

Responsibilities:
- Load and own the sentence-transformer model
- Provide embedding and similarity helpers

This module is intentionally framework-agnostic so it can be reused in:
- FastAPI
- background jobs
- CLI scripts
- tests
"""

from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer


# --- Model loading -----------------------------------------------------------
# Loaded once per process. SentenceTransformer internally caches weights.
_MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(_MODEL_NAME)


# --- Public API --------------------------------------------------------------
def embed(text: Union[str, list[str]]) -> np.ndarray:
    """
    Convert text (or list of texts) into dense vector embeddings.

    Args:
        text: Single string or list of strings

    Returns:
        Numpy array containing embedding(s)

    Notes:
        - Returned vectors are NOT normalized by default
        - Normalization can be added later if needed
    """
    return _model.encode(text)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First embedding vector
        b: Second embedding vector

    Returns:
        Cosine similarity score in range [-1, 1]
    """
    # Defensive programming: avoid division by zero
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0

    return float(np.dot(a, b) / denom)
