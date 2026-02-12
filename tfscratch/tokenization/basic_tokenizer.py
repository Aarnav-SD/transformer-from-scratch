from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Iterable

@dataclass
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]

    @property
    def size(self) -> int:
        return len(self.token_to_id)
    
class BasicTokenizer:
    """
    Week 1 tokenizer: simple whitespace split.
    Adds special tokens: <pad>, <unk>.
    """
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, lower: bool = True):
        self.lower = lower
        self.vocab: Vocab | None = None
    
    def _normalize(self, text: str) -> str:
        return text.lower() if self.lower else text
    
    def build_vocab(self, texts: Iterable[str], min_freq: int = 1) -> Vocab:
        freq: Dict[str, int] = {}
        for t in texts:
            t = self._normalize(t)
            for tok in t.split():
                freq[tok] = freq.get(tok, 0) + 1
        
        token_to_id: Dict[str, int] = {self.PAD: 0, self.UNK: 1}

        for tok, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
            if c >= min_freq and tok not in token_to_id:
                token_to_id[tok] = len(token_to_id)

        id_to_token = {i: t for t, i in token_to_id.items()}
        self.vocab = Vocab(token_to_id=token_to_id, id_to_token=id_to_token)
        return self.vocab
    
    def encode(self, text: str) -> List[int]:
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        text = self._normalize(text)
        ids = []
        for tok in text.split():
            ids.append(self.vocab.token_to_id.get(tok, self.vocab.token_to_id[self.UNK]))
        return ids
    
    def decode(self, ids: List[int]) -> str:
        if self.vocab is None:
            raise RuntimeError("Vobab is not built. Call build_vocab() first.")
        return " ".join(self.vocab.id_to_token.get(i, self.UNK) for i in ids)
    