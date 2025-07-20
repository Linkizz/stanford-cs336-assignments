import time
from collections import defaultdict
from cs336_basics.utils import pretokenize_cnt, get_pair_cnt, merge, PAT

def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Initialize vocabulary
    idx = 256
    vocab = {i: bytes([i]) for i in range(256)} # {int: bytes}
    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    for token_bytes in special_token_bytes:
        if token_bytes not in vocab.values():
            vocab[idx] = token_bytes
            idx += 1
    # print(vocab)

    # Pre-tokenization
    pre_cnt = pretokenize_cnt(text, PAT, special_tokens) # {(bytes1, bytes2, ...): cnt}
    
    # Merge
    t0 = time.time()
    num_merges = vocab_size - len(vocab)
    merges = [] # [(bytes1, bytes2), (bytes3, bytes4), ...]
    for i in range(num_merges):
        pair_cnt = {} # {(bytes1, bytes2): cnt}
        for token_bytes, num in pre_cnt.items():
            pair_cnt = get_pair_cnt(token_bytes, num, pair_cnt)

        # Most frequent and lexicographically greater pair
        max_cnt_pair = max(pair_cnt.items(), key=lambda x: (x[1], x[0]))[0]
        new_token = max_cnt_pair[0] + max_cnt_pair[1]

        new_pre_cnt = defaultdict(int)
        for token_bytes, cnt in pre_cnt.items():
            new_token_bytes = merge(token_bytes, max_cnt_pair, new_token)
            new_pre_cnt[new_token_bytes] += cnt
        pre_cnt = new_pre_cnt
        
        # Update merges and vocab
        # print(f"merge [{i + 1}/{num_merges}]: {max_cnt_pair} -> {new_token}, {idx}")
        merges.append(max_cnt_pair)
        vocab[idx] = new_token
        idx += 1
    t1 = time.time()
    print(f"time: {t1 - t0}")
    
    return vocab, merges
    
if __name__ == "__main__":
    input_path = "data/TinyStories/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 270
    special_tokens = ["<|endoftext|>"]
    train_bpe(input_path, vocab_size, special_tokens)