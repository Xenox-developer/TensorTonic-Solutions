import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        unique_words = []
        for text in texts:
            for word in text.lower().split():
                if word not in unique_words:
                    unique_words.append(word)
        
        all_tokens = special_tokens + unique_words
        
        self.word_to_id = {token: idx for idx, token in enumerate(all_tokens)}
        self.id_to_word = {idx: token for token, idx in self.word_to_id.items()}
        self.vocab_size = len(all_tokens)
    
    def encode(self, text: str) -> List[int]:
        unk_id = self.word_to_id[self.unk_token]
        words = text.lower().split()
        return [self.word_to_id.get(word, unk_id) for word in words]
    
    def decode(self, ids: List[int]) -> str:
        words = [self.id_to_word[id] for id in ids if id in self.id_to_word]
        return " ".join(words)