from __future__ import annotations
import numpy as np
from .softmax import softmax


def cosine_similarity_matrix(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    X: (T, d)
    returns S: (T, T) where S[i,j] = cosine(x_i, x_j)
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
    Xn = X / norms
    return Xn @ Xn.T


def simple_self_attention(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Week 1 attention:
    - Use cosine similarity between token embeddings as "scores"
    - Convert scores -> weights with softmax
    - Output is weighted sum of values (values are X itself)

    X: (T, d)
    returns:
      A: (T, d) attention outputs
      W: (T, T) attention weights
    """
    scores = cosine_similarity_matrix(X)          # (T, T)
    W = softmax(scores, axis=-1)                  # (T, T) each row sums to 1
    A = W @ X                                     # (T, d)
    return A, W
