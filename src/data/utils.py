from pathlib import Path

def detect_languages(clean_dir):
    """Return list of language codes (subdirectories in clean_dir)."""
    return sorted([p.name for p in Path(clean_dir).iterdir() if p.is_dir()])

def build_char_vocab_from_file(vocab_path):
    """
    Load characters from vocab file.
    Ensures [UNK] is present at index 0.
    Returns: (char2id, id2char)
    """
    vocab = []
    with open(vocab_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln and ln not in vocab:
                vocab.append(ln)

    # Ensure [UNK] is at index 0
    if "[UNK]" not in vocab:
        vocab = ["[UNK]"] + vocab
    else:
        # move [UNK] to front if misplaced
        vocab = ["[UNK]"] + [ch for ch in vocab if ch != "[UNK]"]

    char2id = {ch: i for i, ch in enumerate(vocab)}
    id2char = {i: ch for i, ch in enumerate(vocab)}
    return char2id, id2char
