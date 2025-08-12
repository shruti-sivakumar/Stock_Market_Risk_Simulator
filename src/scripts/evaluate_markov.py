#!/usr/bin/env python3
import argparse
from collections import defaultdict
from src.data.datasets import load_sentences, load_eval_sentences
from src.models.markov_chain import MarkovClassifier

TRAIN_LANGS = ["en","es","fr","de"]
ALL_LANGS   = ["en","es","fr","de","it","pt","nl","sv"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    # Load training data
    data_by_lang = {l: load_sentences(l, "train") for l in TRAIN_LANGS}

    # Train classifier
    clf = MarkovClassifier(alpha=args.alpha)
    clf.fit_per_language(data_by_lang)

    # Evaluate on dev (trained langs)
    correct = 0; total = 0
    for l in TRAIN_LANGS:
        for s in load_sentences(l, "dev"):
            pred, _ = clf.predict(s)
            correct += (pred == l)
            total += 1
    acc_dev = correct / max(1,total)

    # Evaluate 32 eval sentences (all 8 langs)
    rows = []
    for l in ALL_LANGS:
        for s in load_eval_sentences(l):
            pred, scores = clf.predict(s)
            rows.append((l, s, pred, scores))

    # Print summary
    print(f"Dev accuracy (train langs only): {acc_dev:.3f}\n")

    # Print all eval outputs in a table
    print(f"{'True':<5} {'Pred':<5} {'en':>10} {'es':>10} {'fr':>10} {'de':>10}  Sentence")
    print("-" * 90)
    for lang, sent, pred, scores in rows:
        print(f"{lang:<5} {pred:<5} "
              f"{scores['en']:.1f} {scores['es']:.1f} {scores['fr']:.1f} {scores['de']:.1f}  {sent}")

if __name__ == "__main__":
    main()