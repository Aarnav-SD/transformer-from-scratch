from __future__ import annotations
from pathlib import Path
import numpy as np

from tfscratch.tokenization.basic_tokenizer import BasicTokenizer
from tfscratch.embeddings.lookup import EmbeddingLookup
from tfscratch.attention.simple_attention import simple_self_attention
from tfscratch.viz.attention_viz import plot_attention_heatmap


def read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main():
    corpus_path = Path("data/sample/tiny_corpus.txt")
    lines = read_lines(corpus_path)

    tokenizer = BasicTokenizer(lower=True)
    vocab = tokenizer.build_vocab(lines, min_freq=1)

    sentence = "the cat chased the dog"
    token_ids = np.array(tokenizer.encode(sentence), dtype=np.int64)
    tokens = [tokenizer.vocab.id_to_token[i] for i in token_ids.tolist()]

    emb = EmbeddingLookup(vocab_size=vocab.size, d_model=16, seed=42)
    X = emb(token_ids)                 # (T, d)

    A, W = simple_self_attention(X)

    print("Sentence:", sentence)
    print("Tokens:", tokens)
    print("Token IDs:", token_ids.tolist())
    print("X shape:", X.shape, "A shape:", A.shape, "W shape:", W.shape)
    print("Row sums (should be ~1):", W.sum(axis=-1))

    plot_attention_heatmap(W, tokens, title="Week 1: Simple Self-Attention (cosine + softmax)")


if __name__ == "__main__":
    main()
