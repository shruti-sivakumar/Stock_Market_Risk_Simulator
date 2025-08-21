import argparse
from pathlib import Path
from src.data.datasets import load_sentences, load_eval_sentences
from src.models.hmm_multinomial import train_lang_hmm, score_sentence
from collections import defaultdict

def detect_languages(clean_dir: Path):
    return sorted([p.name for p in clean_dir.iterdir() if p.is_dir()])

def read_lines(p: Path):
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

def build_char_vocab(vocab_path: Path):
    itos = read_lines(vocab_path)
    if "[UNK]" not in itos:
        itos = itos + ["[UNK]"]
    stoi = {c: i for i, c in enumerate(itos)}
    return stoi, itos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", type=Path, default=Path("data/clean"))
    ap.add_argument("--vocab_path", type=Path, default=Path("tokenizers/vocab_char.txt"))
    ap.add_argument("--states", type=int, nargs="+", default=[2, 5, 8])
    ap.add_argument("--iter", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diag", type=float, default=0.85)
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = ap.parse_args()

    langs = detect_languages(args.clean_dir)
    print(f"[INFO] Detected languages: {langs}")
    char2id, _ = build_char_vocab(args.vocab_path)
    args.outdir.mkdir(parents=True, exist_ok=True)

    for K in args.states:
        print(f"\n=== Training MultinomialHMM (K={K}) ===")
        # Train one HMM per language
        models = {}
        for lg in langs:
            train = load_sentences(lg, "train", base=str(args.clean_dir))
            models[lg] = train_lang_hmm(
                train, char2id,
                n_states=K, n_iter=args.iter,
                sticky_diag=args.diag, seed=args.seed
            )

        # Dev evaluation
        conf = {t: {p: 0 for p in langs} for t in langs}
        correct = total = 0
        for t in langs:
            for s in load_sentences(t, "dev", base=str(args.clean_dir)):
                scores = {lg: score_sentence(models[lg], s, char2id) for lg in langs}
                pred = max(scores, key=scores.get)
                conf[t][pred] += 1
                correct += int(pred == t)
                total += 1
        acc = correct / max(1, total)
        print(f"Dev accuracy (K={K}): {acc:.3f}")

        # Eval-sentences table
        rows = []
        for t in langs:
            try:
                eval_sents = load_eval_sentences(t, base=str(args.clean_dir))
            except Exception:
                eval_sents = []
            for s in eval_sents:
                sc = {lg: score_sentence(models[lg], s, char2id) for lg in langs}
                pred = max(sc, key=sc.get)
                rows.append((t, pred, sc, s))

        # Pretty print + save
        header = "true\\pred".ljust(8) + "".join(f"{lg:>6}" for lg in langs)
        lines = [
            f"[INFO] Detected languages: {langs}",
            f"Dev accuracy (K={K}): {acc:.3f}",
            "",
            "Confusion Matrix (Dev set):",
            header,
        ]
        for t in langs:
            lines.append(t.ljust(8) + "".join(f"{conf[t][p]:>6}" for p in langs))
        lines.append("")
        lines.append("True  Pred  " + " ".join(f"{lg:>8}" for lg in langs) + "  Sentence")
        lines.append("-" * 120)
        for t, p, sc, s in rows:
            score_str = " ".join(f"{sc[lg]:8.1f}" for lg in langs)
            lines.append(f"{t:<4} {p:<4} {score_str}  {s}")

        outpath = args.outdir / f"hmm_k{K}.txt"
        outpath.write_text("\n".join(lines), encoding="utf-8")
        print(f"[SAVED] {outpath}")

if __name__ == "__main__":
    main()