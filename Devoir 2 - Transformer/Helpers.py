import re
import urllib.request
from pathlib import Path
from typing import List, Tuple, Dict

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# 1) Download the text
def download_tinyshakespeare(out_path: str = "tinyshakespeare.txt") -> str:
    out_path = Path(out_path)
    if not out_path.exists():
        urllib.request.urlretrieve(URL, out_path)
    return out_path.read_text(encoding="utf-8")


# 2) Split into 70% / 15% / 15% (contiguous splits)
def split_text(text: str) -> Tuple[str, str, str]:
    n = len(text)
    cut1 = int(0.70 * n)
    cut2 = int(0.85 * n)
    return text[:cut1], text[cut1:cut2], text[cut2:]


# 3) Extract words using regex \b\w+\b
_word_re = re.compile(r"\b\w+\b")


def extract_words(text: str, lowercase: bool = False) -> List[str]:
    # normalize curly apostrophes then REMOVE apostrophes so "won't" -> "wont"
    text = text.replace("’", "'").replace("'", "")
    if lowercase:
        text = text.lower()
    return _word_re.findall(text)


# 4) Make fixed-length sequences with overlap
def make_overlapping_sequences(tokens: List[str], seq_len: int = 64, overlap: int = 32) -> List[List[str]]:
    assert 0 <= overlap < seq_len
    stride = seq_len - overlap
    if stride <= 0:
        raise ValueError("overlap must be < seq_len")

    return [tokens[i:i + seq_len] for i in range(0, len(tokens) - seq_len + 1, stride)]


# 5) Build dictionary (word -> id) with <UNKNOWN>
def build_vocab(tokens: List[str], unk_token: str = "<UNKNOWN>") -> Dict[str, int]:
    # unique words from training set
    unique_words = set(tokens)
    vocab = {unk_token: 0}
    for i, w in enumerate(unique_words, start=1):
        vocab[w] = i
    return vocab


def encode_sequence(seq: List[str], vocab: Dict[str, int], unk_token: str = "<UNKNOWN>") -> List[int]:
    unk_id = vocab[unk_token]
    return [vocab.get(w, unk_id) for w in seq]


def encode_sequences(seqs: List[List[str]], vocab: Dict[str, int], unk_token: str = "<UNKNOWN>") -> List[List[int]]:
    return [encode_sequence(seq, vocab, unk_token=unk_token) for seq in seqs]


if __name__ == "__main__":
    text = download_tinyshakespeare()
    train_text, val_text, test_text = split_text(text)

    train_tokens = extract_words(train_text, lowercase=True)
    val_tokens = extract_words(val_text, lowercase=True)
    test_tokens = extract_words(test_text, lowercase=True)

    SEQ_LEN = 64
    OVERLAP = 32  # seq_len=64 -> stride=32 (50% overlap)

    train_seqs = make_overlapping_sequences(train_tokens, seq_len=SEQ_LEN, overlap=OVERLAP)
    val_seqs = make_overlapping_sequences(val_tokens, seq_len=SEQ_LEN, overlap=OVERLAP)
    test_seqs = make_overlapping_sequences(test_tokens, seq_len=SEQ_LEN, overlap=OVERLAP)

    # Build vocab from TRAIN only (includes <UNKNOWN>)
    vocab = build_vocab(train_tokens, unk_token="<UNKNOWN>")
    vocab_size = len(vocab)

    # Encode sequences to ids
    train_ids = encode_sequences(train_seqs, vocab, unk_token="<UNKNOWN>")
    val_ids = encode_sequences(val_seqs, vocab, unk_token="<UNKNOWN>")
    test_ids = encode_sequences(test_seqs, vocab, unk_token="<UNKNOWN>")

    print("Tokens (train/val/test):", len(train_tokens), len(val_tokens), len(test_tokens))
    print("Sequences (train/val/test):", len(train_seqs), len(val_seqs), len(test_seqs))
    print("Vocab size (train + <UNKNOWN>):", vocab_size)

    # Show example (words + ids)
    print("\nExample last train sequence (words):")
    print(train_seqs[-1])
    print("\nExample last train sequence (ids):")
    print(train_ids[-1])

    # Sanity check: <UNKNOWN> id
    print("\n<UNKNOWN> id:", vocab["<UNKNOWN>"])

