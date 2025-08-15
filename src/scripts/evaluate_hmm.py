#!/usr/bin/env python3
import argparse
from pathlib import Path
from src.data.datasets import load_sentences, load_eval_sentences
from src.models.hmm import HMMClassifier

def detect_languages(clean_dir=Path("data/clean")):
    return sorted([p.name for p in clean_dir.iterdir() if p.is_dir()])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_states", type=int, default=6)
    ap.add_argument("--n_iter", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--clean_dir", type=Path, default=Path("data/clean"))
    args = ap.parse_args()

    ALL_LANGS = detect_languages(args.clean_dir)
    print(f"[INFO] Detected languages: {ALL_LANGS}")

    data_by_lang = {l: load_sentences(l, "train") for l in ALL_LANGS}

    clf = HMMClassifier(n_states=args.n_states, alpha=args.alpha, n_iter=args.n_iter)
    clf.fit_per_language(data_by_lang)

    correct, total = 0, 0
    conf_matrix = {true: {pred: 0 for pred in ALL_LANGS} for true in ALL_LANGS}

    for l in ALL_LANGS:
        for s in load_sentences(l, "dev"):
            pred, _ = clf.predict(s)
            conf_matrix[l][pred] += 1
            if pred == l:
                correct += 1
            total += 1

    acc_dev = correct / max(1, total)
    print(f"Dev accuracy (all langs): {acc_dev:.3f}\n")

    print("Confusion Matrix (Dev set):")
    header = "true\\pred".ljust(8) + "".join(f"{lang:>6}" for lang in ALL_LANGS)
    print(header)
    for true in ALL_LANGS:
        row = true.ljust(8) + "".join(f"{conf_matrix[true][pred]:>6}" for pred in ALL_LANGS)
        print(row)
    print()

    print(f"{'True':<5} {'Pred':<5} " + " ".join(f"{lang:>8}" for lang in ALL_LANGS) + "  Sentence")
    print("-" * 120)
    for lang in ALL_LANGS:
        for s in load_eval_sentences(lang):
            pred, scores = clf.predict(s)
            score_str = " ".join(f"{scores[lg]:8.1f}" for lg in ALL_LANGS)
            print(f"{lang:<5} {pred:<5} {score_str}  {s}")

if __name__ == "__main__":
    main()
