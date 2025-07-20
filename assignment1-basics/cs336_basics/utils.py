# Helper function for tokenizer

import regex as re
import os
from typing import BinaryIO
from multiprocessing import Process, Manager, Pool, cpu_count

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_pair_cnt(
    token_bytes: tuple[bytes, ...], 
    num: int=1, 
    cnt: dict[tuple[bytes, bytes], int]=None
) -> dict[tuple[bytes, bytes], int]:
    """Counts occurrences of consecutive token bytes pairs.

    Args:
        token_bytes: A tuple of UTF-8 bytes (e.g., b'hello', b'a').
        num: The count increment for each observed pair.
        cnt: An optional dictionary to update an existing dictionary of counts.

    Returns:
        A dictionary mapping (bytes1, bytes2) pairs to their occurrences.

    Examples:
        >>> get_pair_cnt((b'a', b'b', b'b', b'c', b'a', b'b'))
        {(b'a', b'b'): 2, (b'b', b'b'): 1, (b'b', b'c'): 1, (b'c', b'a'): 1}

        >>> get_pair_cnt((b'x', b'y', b'x', b'y', b'x', b'y', b'y'), 3)
        {(b'x', b'y'): 9, (b'y', b'x'): 6, (b'y', b'y'): 3}

        >>> get_pair_cnt([b'single_byte'])
        {}

        >>> token_bytes = (b'1', b'2', b'1', b'2', b'2', b'1', b'5')
        >>> cnt = {(b'1', b'2'): 10, (b'5', b'5'): 100, (b'1', b'5'): 5}
        >>> get_pair_cnt(token_bytes, 2, cnt)
        {(b'1', b'2'): 14, (b'5', b'5'): 100, (b'1', b'5'): 7, (b'2', b'1'): 4, (b'2', b'2'): 2}
    """
    pair_cnt = cnt if cnt else {}
    for i in range(len(token_bytes) - 1):
        merged_pair = (token_bytes[i], token_bytes[i + 1])
        pair_cnt[merged_pair] = pair_cnt.get(merged_pair, 0) + num

    return pair_cnt

def merge(
    token_bytes: tuple[bytes, ...], 
    pair: tuple[bytes, bytes], 
    new_token: bytes
) -> tuple[bytes, ...]:
    """Merges a specific pair of token bytes into a new token bytes.
    
    Args:
        token_bytes: A tuple of UTF-8 bytes (e.g., b'hello', b'a').
        pair: The specific pair (bytes1, bytes2) need to be merged.
        new_token: The new single token bytes assigned to the merged pair.

    Returns:
        A tuple of new UTF-8 bytes.

    Notes:
        The `new_token` parameter is included for efficiency. In the `train_bpe` loop,
        `new_token` (which is `pair[0] + pair[1]`) is pre-calculated once per merge iteration
        before `merge` is called multiple times to update `pre_cnt`. This avoids redundant
        byte concatenation operations inside the inner loop.

    Examples:
        >>> merge((b'h', b'e', b'l', b'l', b'o'), (b'h', b'e'), b'he')
        (b'he', b'l', b'l', b'o')

        >>> merge((b'a', b'b', b'c', b'a', b'b', b'd'), (b'a', b'b'), b'ab')
        (b'ab', b'c', b'ab', b'd')
    """
    new_token_bytes = []
    i = 0
    while i < len(token_bytes):
        if i + 1 < len(token_bytes) and token_bytes[i:i+2] == pair:
            new_token_bytes.append(new_token)
            i += 2
        else:
            new_token_bytes.append(token_bytes[i])
            i += 1  
    return tuple(new_token_bytes)

def pretokenize_cnt(
    text: str, 
    pattern: str,
    special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
    """Pre-tokenize input string and count occurrences of each unique pre-token.
    
    Args:
        text: The input string to be pre-tokenized.
        pattern: The regular expression pattern used to identify pre-tokens.
        special_tokens: A list of string tokens to be excluded from pattern matching.

    Returns:
        A dictionary mapping each unique pre-token (as a tuple of bytes) to its count.
    """
    # Remove special tokens
    chunk_lst = re.split("|".join(map(re.escape, special_tokens)), text) if special_tokens else [text]
    # Same as above
    # escaped_special_tokens = [re.escape(token) for token in special_tokens]
    # special_pattern = "|".join(escaped_special_tokens) if escaped_special_tokens else None
    # chunk_lst = re.split(special_pattern, text) if special_pattern else [text]
    
    cnt = {}
    for chunk in chunk_lst:
        # Use `finditer` to avoid storing the pre-tokenized words at once
        for match in re.finditer(pattern, chunk):
            token = match.group()
            token_bytes = str_to_bytes_tuple(token)
            cnt[token_bytes] = cnt.get(token_bytes, 0) + 1
    return cnt

def str_to_bytes_tuple(token: str) -> tuple[bytes]:
    """Transform a string token to a tuple of bytes."""
    return tuple([bytes([x]) for x in tuple(token.encode("utf-8"))])


if __name__ == "__main__":
    # test `get_pair_cnt`
    token_bytes = (b'1', b'2', b'1', b'2', b'2', b'1', b'5')
    cnt = {(b'1', b'2'): 10, (b'5', b'5'): 100, (b'1', b'5'): 5}
    num = 5
    pair_cnt = get_pair_cnt(token_bytes, num, cnt)
    # {(b'1', b'2'): 20, (b'5', b'5'): 100, (b'1', b'5'): 10, (b'2', b'1'): 10, (b'2', b'2'): 5}
    print(f"pair count dict: {pair_cnt}") 

    # test `merge`
    token_bytes = (b'h', b'e', b'l', b'l', b'o', b'he', b'h', b'e')
    pair = (b'h', b'e')
    new_token = b'he'
    new_token_bytes = merge(token_bytes, pair, new_token)
    # (b'he', b'l', b'l', b'o', b'he', b'he')
    print(f"new token bytes list: {new_token_bytes}")
    
    # test `pretokenize_cnt``
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    text = "hello<SEP>world hello<SEP>python."
    special_tokens = ["<SEP>"]
    cnt = pretokenize_cnt(text, PAT, special_tokens)
    print(f"pretokenized count dict: {cnt}")

