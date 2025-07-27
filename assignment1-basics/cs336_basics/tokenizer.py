from typing import Iterable, Iterator
import json
import pickle
import regex as re
from cs336_basics.utils import merge, str_to_bytes_tuple, PAT

class Tokenizer(object):
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.reversed_vocab = {token: id for id, token in self.vocab.items()}
        self.merges = merges 
        self.special_tokens = special_tokens

        self.vocab_size = len(self.vocab)

        # Add new special tokens that were not included during `train_bpe`
        if special_tokens:
            # Handle overlapping special tokens (e.g., "<|endoftext|>" and "<|endoftext|><|endoftext|>")
            self.special_tokens.sort(key=len, reverse=True)
            for token in self.special_tokens:
                encoded_token = token.encode("utf-8")
                if encoded_token not in self.vocab.values():
                    self.vocab[self.vocab_size] = encoded_token
                    self.reversed_vocab[encoded_token] = self.vocab_size
                    self.vocab_size += 1
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        """Initialize a tokenizer from saved vocabulary and merges pickle files."""
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """Encode a string of text into a list of token IDs."""
        ids = []
        chunk_lst = re.split(
            "(" + "|".join(map(re.escape, self.special_tokens)) + ")", 
            text
        ) if self.special_tokens else [text]
        
        for chunk in chunk_lst:
            if self.special_tokens and chunk in self.special_tokens:
                encoded_chunk = chunk.encode("utf-8")
                ids.append(self.reversed_vocab[encoded_chunk])
                continue
            for match in re.finditer(PAT, chunk):
                token = match.group()
                token_bytes = str_to_bytes_tuple(token)
                for pair in self.merges:
                    if len(token_bytes) == 1:
                        break
                    for i in range(len(token_bytes) - 1):
                        if i + 1 < len(token_bytes) and token_bytes[i:i+2] == pair:
                            new_token = pair[0] + pair[1]
                            token_bytes = merge(token_bytes, pair, new_token)
                for token in token_bytes:
                    ids.append(self.reversed_vocab[token])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Generates token IDs from an iterable of strings, for saving memory."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        bytes_seq = []
        for idx in ids:
            assert idx in self.vocab.keys(), f"Do not exist id {idx}!"
            bytes_seq.append(self.vocab[idx])
        text = b"".join(bytes_seq).decode("utf-8", errors="replace")
        return text


if __name__ == "__main__":
    import os
    print(os.path.abspath(os.path.dirname(__file__)))
    # test `from_files`
    vocab_filepath = "./save_tokenizer/TinyStoriesV2-GPT4-valid-vocab.pickle"
    merges_filepath = "./save_tokenizer/TinyStoriesV2-GPT4-valid-merges.pickle"
    special_tokens = ["<|endoftext|>", "<START>", "<|endoftext|><|endoftext|>"]
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    vocab = tokenizer.vocab
    merges = tokenizer.merges
    #print(vocab)
    #print(merges)
    
    # test `encode`
    text = "hello<|endoftext|>to you<|endoftext|><|endoftext|>"
    encoded_text = tokenizer.encode(text)
    print(encoded_text)
    decoded_text = tokenizer.decode(encoded_text)
    print(decoded_text)
    #print(tokenizer.decode([1000, 100]))