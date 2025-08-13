#!/usr/bin/env python3
import argparse
from collections import defaultdict
from src.data.datasets import load_sentences, load_eval_sentences
from src.models.markov_chain import MarkovClassifier

ALL_LANGS = ["en","es","fr","de","it","pt","nl","sv"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    # 1. Load training data for all langs
    data_by_lang = {l: load_sentences(l, "train") for l in ALL_LANGS}

    # 2. Train classifier
    clf = MarkovClassifier(alpha=args.alpha)
    clf.fit_per_language(data_by_lang)

    # 3. Evaluate on dev set (all langs)
    correct = 0
    total = 0
    conf_matrix = {true: {pred: 0 for pred in ALL_LANGS} for true in ALL_LANGS}

    for l in ALL_LANGS:
        for s in load_sentences(l, "dev"):
            pred, _ = clf.predict(s)
            conf_matrix[l][pred] += 1
            if pred == l:
                correct += 1
            total += 1

    acc_dev = correct / max(1, total)

    # 4. Evaluate on eval_sentences.txt
    rows = []
    for l in ALL_LANGS:
        for s in load_eval_sentences(l):
            pred, scores = clf.predict(s)
            rows.append((l, s, pred, scores))

    # 5. Print accuracy
    print(f"Dev accuracy (all langs): {acc_dev:.3f}\n")

    # 6. Print confusion matrix
    print("Confusion Matrix (Dev set):")
    header = "true\\pred".ljust(8) + "".join(f"{lang:>6}" for lang in ALL_LANGS)
    print(header)
    for true in ALL_LANGS:
        row = true.ljust(8) + "".join(f"{conf_matrix[true][pred]:>6}" for pred in ALL_LANGS)
        print(row)
    print()

    # 7. Print eval sentences
    print(f"{'True':<5} {'Pred':<5} " + " ".join(f"{lang:>8}" for lang in ALL_LANGS) + "  Sentence")
    print("-" * 120)
    for lang, sent, pred, scores in rows:
        score_str = " ".join(f"{scores[lg]:8.1f}" for lg in ALL_LANGS)
        print(f"{lang:<5} {pred:<5} {score_str}  {sent}")

if __name__ == "__main__":
    main()
