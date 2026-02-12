from __future__ import annotations
import numpy as np


class EmbeddingLookup:
    """
    Week 1: random embedding table + lookup.
    """
    def __init__(self, vocab_size: int, d_model: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Small init helps keep numbers sane
        self.table = rng.normal(loc=0.0, scale=0.02, size=(vocab_size, d_model)).astype(np.float32)

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        """
        token_ids: (T,) int array
        returns: (T, d_model)
        """
        return self.table[token_ids]