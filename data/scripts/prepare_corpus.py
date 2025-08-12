#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare multilingual corpora for MC/HMM/Neural-HMM language ID:
- Clean + sentence-split 4 training languages (EN/ES/FR/DE)
- Create train/dev/test splits
- Build char vocabulary (train languages only)
- Train BPE tokenizers (1k & 2k) on train languages only
- Print a compact dataset report

Requires:
  pip install regex tokenizers nltk
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import random
import re
import unicodedata
from typing import List, Dict, Tuple

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

# Allow Latin letters (including extended ranges), digits, space, and light punctuation.
# Latin ext ranges: \u00C0-\u024F, \u1E00-\u1EFF
KEEP_NOT = r"[^0-9A-Za-z\u00C0-\u024F\u1E00-\u1EFF\s\.,;:\?\!\'\"\-\(\)]"
KEEP_NOT_RE = re.compile(KEEP_NOT)

# Naive sentence split: split on ., !, ? followed by whitespace.
SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")

def clean_text(text: str) -> str:
    # Normalize + lowercase
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # Remove non-allowed chars
    text = KEEP_NOT_RE.sub(" ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sentence_split(text: str, use_nltk: bool = False, lang_hint: str | None = None) -> List[str]:
    if use_nltk and nltk_available:
        # Try punkt per language if available; fallback to english
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        from nltk.tokenize import sent_tokenize
        try:
            sents = sent_tokenize(text, language={
                'en':'english','es':'spanish','fr':'french','de':'german'
            }.get(lang_hint, 'english'))
        except Exception:
            sents = sent_tokenize(text)
        return [s.strip() for s in sents if s.strip()]
    # Fallback regex splitting
    return [s.strip() for s in SPLIT_RE.split(text) if s.strip()]

def filter_by_length(sents: List[str], min_chars: int = 30, max_chars: int = 200) -> List[str]:
    out = []
    for s in sents:
        if min_chars <= len(s) <= max_chars:
            out.append(s)
    return out


# ---------------------------
# IO helpers
# ---------------------------

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

def ensure_exists(path: Path, kind: str = "file") -> None:
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"Missing {kind}: {path}")
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"Missing {kind}: {path}")


# ---------------------------
# Core preparation
# ---------------------------

def process_language(
    lang: str,
    raw_dir: Path,
    clean_dir: Path,
    seed: int = 13,
    min_chars: int = 30,
    max_chars: int = 200,
) -> Dict[str, int]:
    """Clean, split, and save train/dev/test for a training language."""
    random.seed(seed)
    raw_path = raw_dir / lang / "train_text.txt"
    ensure_exists(raw_path, "file")
    raw = read_text(raw_path)
    cleaned = clean_text(raw)
    sents = sentence_split(cleaned, use_nltk=False, lang_hint=lang)
    sents = filter_by_length(sents, min_chars=min_chars, max_chars=max_chars)

    # Shuffle and split 80/10/10
    random.shuffle(sents)
    n = len(sents)
    n_train = int(0.8 * n)
    n_dev   = int(0.9 * n)
    train, dev, test = sents[:n_train], sents[n_train:n_dev], sents[n_dev:]

    # Save
    lang_dir = clean_dir / lang
    write_lines(lang_dir / "train.txt", train)
    write_lines(lang_dir / "dev.txt", dev)
    write_lines(lang_dir / "test.txt", test)

    # Return stats
    return {
        "n_total": n,
        "n_train": len(train),
        "n_dev": len(dev),
        "n_test": len(test),
        "avg_len_train": int(sum(map(len, train))/max(1,len(train))),
        "avg_len_dev": int(sum(map(len, dev))/max(1,len(dev))),
        "avg_len_test": int(sum(map(len, test))/max(1,len(test))),
    }

def build_char_vocab(train_langs: List[str], clean_dir: Path, tokenizer_dir: Path) -> int:
    """Build char vocab from train/dev/test of training languages only."""
    chars = set()
    for lang in train_langs:
        for split in ("train", "dev", "test"):
            p = clean_dir / lang / f"{split}.txt"
            ensure_exists(p, "file")
            txt = read_text(p)
            chars.update(list(txt))
    # Keep space, drop other pure whitespace
    chars = sorted(c for c in chars if (c == " ") or (not c.isspace()))
    vocab_path = tokenizer_dir / "vocab_char.txt"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    write_lines(vocab_path, chars)
    return len(chars)

def train_bpe(vocab_size: int, outpath: Path, train_langs: List[str], clean_dir: Path) -> None:
    if not tokenizers_available:
        print("[WARN] HuggingFace `tokenizers` not installed; skipping BPE training.")
        return
    files = [str(clean_dir / lang / "train.txt") for lang in train_langs]
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
    tok.train(files, trainer)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(outpath))


def bpe_coverage_check(model_path: Path, texts: List[str]) -> float:
    """% of tokens that became [UNK] on given texts (lower is better)."""
    if not tokenizers_available or not model_path.is_file():
        return math.nan
    tok = Tokenizer.from_file(str(model_path))
    total = 0
    unks = 0
    for t in texts:
        enc = tok.encode(t)
        total += len(enc.tokens)
        unks  += sum(1 for tok_ in enc.tokens if tok_ == "[UNK]")
    if total == 0: return 0.0
    return 100.0 * unks / total


# ---------------------------
# Reporting
# ---------------------------

def report_stats(title: str, stats: Dict[str, Dict[str,int]]) -> None:
    print(f"\n== {title} ==")
    header = f"{'lang':<6} {'total':>7} {'train':>7} {'dev':>7} {'test':>7} {'avg_len_tr':>10} {'avg_len_dev':>12} {'avg_len_te':>12}"
    print(header)
    print("-"*len(header))
    for lang, d in stats.items():
        print(f"{lang:<6} {d['n_total']:>7} {d['n_train']:>7} {d['n_dev']:>7} {d['n_test']:>7} {d['avg_len_train']:>10} {d['avg_len_dev']:>12} {d['avg_len_test']:>12}")

def quick_bpe_report(train_langs: List[str], clean_dir: Path, tokenizer_dir: Path) -> None:
    dev_texts = []
    for lang in train_langs:
        p = clean_dir / lang / "dev.txt"
        if p.is_file():
            dev_texts.extend(read_text(p).splitlines())

    bpe1 = tokenizer_dir / "bpe_1k.json"
    bpe2 = tokenizer_dir / "bpe_2k.json"
    c1 = bpe_coverage_check(bpe1, dev_texts)
    c2 = bpe_coverage_check(bpe2, dev_texts)
    if not math.isnan(c1):
        print(f"\nBPE coverage on dev (lower [UNK]% is better):")
        print(f"  bpe_1k.json  UNK% ≈ {c1:.3f}%")
        print(f"  bpe_2k.json  UNK% ≈ {c2:.3f}%")
    else:
        print("\n[INFO] BPE coverage skipped (tokenizers not installed or models missing).")


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--clean_dir", type=Path, default=Path("data/clean"))
    ap.add_argument("--eval_dir", type=Path, default=Path("data/eval_sentences"))
    ap.add_argument("--tokenizer_dir", type=Path, default=Path("tokenizers"))
    ap.add_argument("--train_langs", nargs="+", default=["en","es","fr","de"])
    ap.add_argument("--unseen_langs", nargs="+", default=["it","pt","nl","sv"])
    ap.add_argument("--min_chars", type=int, default=30)
    ap.add_argument("--max_chars", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--train_bpe", action="store_true", help="Train BPE tokenizers (1k & 2k) if tokenizers is installed.")
    args = ap.parse_args()

    # Validate raw files exist
    for lang in args.train_langs + args.unseen_langs:
        p = args.raw_dir / lang / "train_text.txt"
        if not p.is_file():
            print(f"[WARN] Missing raw file for {lang}: {p} (only needed for training langs).")

    # Process training languages
    stats = {}
    for lang in args.train_langs:
        s = process_language(
            lang=lang,
            raw_dir=args.raw_dir,
            clean_dir=args.clean_dir,
            seed=args.seed,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )
        stats[lang] = s

    # Report split stats
    report_stats("Sentence splits (training languages)", stats)

    # Build char vocab
    vocab_size = build_char_vocab(args.train_langs, args.clean_dir, args.tokenizer_dir)
    print(f"\nChar vocab written to {args.tokenizer_dir/'vocab_char.txt'}  (size={vocab_size})")

    # Train BPE (optional)
    if args.train_bpe:
        if not tokenizers_available:
            print("[WARN] Skipping BPE training: install `tokenizers` first (pip install tokenizers).")
        else:
            train_bpe(1000, args.tokenizer_dir / "bpe_1k.json", args.train_langs, args.clean_dir)
            train_bpe(2000, args.tokenizer_dir / "bpe_2k.json", args.train_langs, args.clean_dir)
            quick_bpe_report(args.train_langs, args.clean_dir, args.tokenizer_dir)

    # Copy eval sentences for all languages (train + unseen)
    eval_errors = []
    for lang in args.train_langs + args.unseen_langs:
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