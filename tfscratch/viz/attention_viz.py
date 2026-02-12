from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def plot_attention_heatmap(weights: np.ndarray, tokens: list[str], title: str = "Attention Weights"):
    """
    weights: (T, T)
    tokens: length T
    """
    T = len(tokens)
    assert weights.shape == (T, T)

    plt.figure()
    plt.imshow(weights, aspect="auto")
    plt.colorbar()
    plt.xticks(range(T), tokens, rotation=45, ha="right")
    plt.yticks(range(T), tokens)
    plt.title(title)
    plt.tight_layout()
    plt.savefig("outputs/figures/attention_heatmap.png")
    plt.close()
