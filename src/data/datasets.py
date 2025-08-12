from pathlib import Path

def load_sentences(lang, split, base="data/clean"):
    p = Path(base) / lang / f"{split}.txt"
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

def load_eval_sentences(lang, base="data/clean"):
    p = Path(base) / lang / "eval_sentences.txt"
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]