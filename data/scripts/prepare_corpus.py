#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare multilingual corpora for MC/HMM/Neural-HMM language ID:
- Clean + sentence-split ALL 8 languages
- Create train/dev/test splits
- Build char vocabulary (all 8 languages)
- Train BPE tokenizers (1k & 2k) on all 8 languages
- Copy eval sentences for all 8 languages
- Print a compact dataset report

Requires:
  pip install regex tokenizers nltk
"""
from __future__ import annotations
import argparse
import math
import random
import re
import unicodedata
from pathlib import Path
from typing import List, Dict

# Optional: sentence tokenizer fallback
try:
    import nltk
    nltk_available = True
except Exception:
    nltk_available = False

# HuggingFace tokenizers for BPE
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    tokenizers_available = True
except Exception:
    tokenizers_available = False


# ---------------------------
# Cleaning & splitting utils
# ---------------------------

KEEP_NOT = r"[^0-9A-Za-z\u00C0-\u024F\u1E00-\u1EFF\s\.,;:\?\!\'\"\-\(\)]"
KEEP_NOT_RE = re.compile(KEEP_NOT)
SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = KEEP_NOT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sentence_split(text: str, use_nltk: bool = False, lang_hint: str | None = None) -> List[str]:
    if use_nltk and nltk_available:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        from nltk.tokenize import sent_tokenize
        try:
            sents = sent_tokenize(text, language={
                'en':'english','es':'spanish','fr':'french','de':'german',
                'it':'italian','pt':'portuguese','nl':'dutch','sv':'swedish'
            }.get(lang_hint, 'english'))
        except Exception:
            sents = sent_tokenize(text)
        return [s.strip() for s in sents if s.strip()]
    return [s.strip() for s in SPLIT_RE.split(text) if s.strip()]

def filter_by_length(sents: List[str], min_chars: int = 30, max_chars: int = 200) -> List[str]:
    return [s for s in sents if min_chars <= len(s) <= max_chars]


# ---------------------------
# IO helpers
# ---------------------------

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

def ensure_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")


# ---------------------------
# Core preparation
# ---------------------------

def process_language(
    lang: str,
    raw_dir: Path,
    clean_dir: Path,
    seed: int,
    min_chars: int,
    max_chars: int
) -> Dict[str, int]:
    random.seed(seed)
    raw_path = raw_dir / lang / "train_text.txt"
    ensure_exists(raw_path)
    raw = read_text(raw_path)
    cleaned = clean_text(raw)
    sents = sentence_split(cleaned, use_nltk=False, lang_hint=lang)
    sents = filter_by_length(sents, min_chars, max_chars)

    random.shuffle(sents)
    n = len(sents)
    n_train = int(0.8 * n)
    n_dev = int(0.9 * n)
    train, dev, test = sents[:n_train], sents[n_train:n_dev], sents[n_dev:]

    lang_dir = clean_dir / lang
    write_lines(lang_dir / "train.txt", train)
    write_lines(lang_dir / "dev.txt", dev)
    write_lines(lang_dir / "test.txt", test)

    return {
        "n_total": n,
        "n_train": len(train),
        "n_dev": len(dev),
        "n_test": len(test),
        "avg_len_train": int(sum(map(len, train))/max(1,len(train))),
        "avg_len_dev": int(sum(map(len, dev))/max(1,len(dev))),
        "avg_len_test": int(sum(map(len, test))/max(1,len(test))),
    }

def build_char_vocab(langs: List[str], clean_dir: Path, tokenizer_dir: Path) -> int:
    chars = set()
    for lang in langs:
        for split in ("train", "dev", "test"):
            p = clean_dir / lang / f"{split}.txt"
            ensure_exists(p)
            txt = read_text(p)
            chars.update(list(txt))
    chars = sorted(c for c in chars if (c == " ") or (not c.isspace()))
    vocab_path = tokenizer_dir / "vocab_char.txt"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    write_lines(vocab_path, chars)
    return len(chars)

def train_bpe(vocab_size: int, outpath: Path, langs: List[str], clean_dir: Path) -> None:
    if not tokenizers_available:
        print("[WARN] HuggingFace `tokenizers` not installed; skipping BPE training.")
        return
    files = [str(clean_dir / lang / "train.txt") for lang in langs]
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
    tok.train(files, trainer)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(outpath))


# ---------------------------
# Reporting
# ---------------------------

def report_stats(stats: Dict[str, Dict[str,int]]) -> None:
    print(f"\n== Sentence splits (all languages) ==")
    header = f"{'lang':<6} {'total':>7} {'train':>7} {'dev':>7} {'test':>7} {'avg_len_tr':>10} {'avg_len_dev':>12} {'avg_len_te':>12}"
    print(header)
    print("-"*len(header))
    for lang, d in stats.items():
        print(f"{lang:<6} {d['n_total']:>7} {d['n_train']:>7} {d['n_dev']:>7} {d['n_test']:>7} "
              f"{d['avg_len_train']:>10} {d['avg_len_dev']:>12} {d['avg_len_test']:>12}")


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--clean_dir", type=Path, default=Path("data/clean"))
    ap.add_argument("--tokenizer_dir", type=Path, default=Path("tokenizers"))
    ap.add_argument("--min_chars", type=int, default=30)
    ap.add_argument("--max_chars", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--train_bpe", action="store_true")
    args = ap.parse_args()

    # All 8 languages
    all_langs = ["en","es","fr","de","it","pt","nl","sv"]

    # Process all languages
    stats = {}
    for lang in all_langs:
        s = process_language(lang, args.raw_dir, args.clean_dir, args.seed, args.min_chars, args.max_chars)
        stats[lang] = s

    # Report
    report_stats(stats)

    # Build char vocab from all 8 languages
    vocab_size = build_char_vocab(all_langs, args.clean_dir, args.tokenizer_dir)
    print(f"\nChar vocab written to {args.tokenizer_dir/'vocab_char.txt'}  (size={vocab_size})")

    # Train BPE
    if args.train_bpe:
        train_bpe(1000, args.tokenizer_dir / "bpe_1k.json", all_langs, args.clean_dir)
        train_bpe(2000, args.tokenizer_dir / "bpe_2k.json", all_langs, args.clean_dir)

    # Copy eval sentences for all languages
    eval_errors = []
    for lang in all_langs:
        src_eval = args.raw_dir / lang / "test_sentences.txt"
        dst_eval = args.clean_dir / lang / "eval_sentences.txt"
        if src_eval.is_file():
            lines = [ln.strip() for ln in read_text(src_eval).splitlines() if ln.strip()]
            if len(lines) != 4:
                eval_errors.append(f"{lang}: found {len(lines)} sentences (expected 4)")
            write_lines(dst_eval, lines)
        else:
            eval_errors.append(f"{lang}: missing file {src_eval}")

    if eval_errors:
        print("\n[WARN] Issues with evaluation sentences:")
        for e in eval_errors:
            print(f"  - {e}")
    else:
        print(f"\n[OK] Evaluation sentences copied to {args.clean_dir}/{{lang}}/eval_sentences.txt for all languages.")

if __name__ == "__main__":
    main()