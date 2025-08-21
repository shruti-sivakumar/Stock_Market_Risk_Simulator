import argparse, math
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from src.data.datasets import load_sentences, load_eval_sentences
from src.data.utils import detect_languages, build_char_vocab_from_file
from src.models.neural_hmm import NeuralHMM


# ------------------------
# Dataset / Collate
# ------------------------

class CharDataset(Dataset):
    def __init__(self, sentences, char2id):
        self.sents = sentences
        self.stoi = char2id
        # ensure [UNK] exists (caller adds if missing)
        self.unk = char2id["[UNK]"]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, i):
        ids = [self.stoi.get(c, self.unk) for c in self.sents[i]]
        return torch.tensor(ids, dtype=torch.long)


def make_pad_collate(pad_id: int):
    """Factory to create a collate_fn that pads with pad_id."""
    def pad_collate(batch):
        # batch: list of 1D LongTensors with variable length
        lengths = torch.tensor([b.numel() for b in batch], dtype=torch.long)
        T = int(lengths.max().item()) if len(lengths) > 0 else 0
        padded = torch.full((len(batch), T), pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            if b.numel() > 0:
                padded[i, : b.numel()] = b
        return padded, lengths
    return pad_collate


# ------------------------
# Scoring helper
# ------------------------

@torch.no_grad()
def score_sentence(model: NeuralHMM, sent: str, char2id, device):
    unk_id = char2id["[UNK]"]
    ids = [char2id.get(c, unk_id) for c in sent]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
    return float(model.log_forward(x, lengths).item())


# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", type=Path, default=Path("data/clean"))
    ap.add_argument("--vocab_path", type=Path, default=Path("tokenizers/vocab_char.txt"))

    ap.add_argument("--states", type=int, default=5)
    ap.add_argument("--emb_dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--context", type=int, default=1, help="char context window on each side")

    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.0, help="Dropout prob in emission MLP (0 disables)")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/neural_hmm"))
    ap.add_argument("--max_dev_eval", type=int, default=200, help="cap per-language dev scoring per epoch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    langs = detect_languages(args.clean_dir)
    print(f"[INFO] Detected languages: {langs}")

    # ---- Vocab (ensure special tokens exist) ----
    char2id, _ = build_char_vocab_from_file(args.vocab_path)

    # Force [PAD], [UNK] in vocab
    if "[PAD]" not in char2id:
        char2id["[PAD]"] = len(char2id)
    if "[UNK]" not in char2id:
        char2id["[UNK]"] = len(char2id)

    # Build inverse index (list of tokens)
    itos = [None] * len(char2id)
    for ch, i in char2id.items():
        itos[i] = ch
        
    V = len(itos)
    pad_id = char2id["[PAD]"]
    unk_id = char2id["[UNK]"]
    collate_fn = make_pad_collate(pad_id)

    args.outdir.mkdir(parents=True, exist_ok=True)

    models = {}
    for lg in langs:
        print(f"\n=== Training Neural-HMM for [{lg}] (K={args.states}) ===")
        train_sents = load_sentences(lg, "train", base=str(args.clean_dir))
        dev_sents   = load_sentences(lg, "dev",   base=str(args.clean_dir))

        # Dataloaders
        train_ds = CharDataset(train_sents, char2id)
        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        # Model (supports optional dropout if your NeuralHMM accepts it)
        try:
            model = NeuralHMM(
                vocab_size=V,
                n_states=args.states,
                emb_dim=args.emb_dim,
                hidden=args.hidden,
                context=args.context,
                dropout=args.dropout,   # your NeuralHMM should ignore if not used
                pad_id=pad_id           # optional: let model know the pad index if it uses masking internally
            ).to(args.device)
        except TypeError:
            # Backward-compat if your class doesn't accept dropout/pad_id
            model = NeuralHMM(
                vocab_size=V,
                n_states=args.states,
                emb_dim=args.emb_dim,
                hidden=args.hidden,
                context=args.context,
            ).to(args.device)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best = math.inf
        ckpt = args.outdir / f"{lg}_k{args.states}.pt"

        # ---- Train ----
        for ep in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            for xb, lengths in train_dl:
                xb, lengths = xb.to(args.device), lengths.to(args.device)
                loss = model.neg_log_likelihood(xb, lengths)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                running += float(loss.item())
            train_nll = running / max(1, len(train_dl))

            # quick dev check (avg negative log-likelihood)
            model.eval()
            with torch.no_grad():
                dev_loss = 0.0
                limit = min(args.max_dev_eval, len(dev_sents))
                for i in range(limit):
                    dev_loss += -score_sentence(model, dev_sents[i], char2id, args.device)
                dev_loss /= max(1, limit)

            if dev_loss < best:
                best = dev_loss
                torch.save(model.state_dict(), ckpt)

            print(f"[{lg}] epoch {ep:02d} | train_nll={train_nll:.3f} dev_nll={dev_loss:.3f} (best={best:.3f})")

        # load best
        model.load_state_dict(torch.load(ckpt, map_location=args.device))
        model.eval()
        models[lg] = model

    # ---- Full evaluation across languages (dev + eval_sentences) ----
    conf = {t: {p: 0 for p in langs} for t in langs}
    correct = total = 0
    for t in langs:
        dev_sents = load_sentences(t, "dev", base=str(args.clean_dir))
        for s in dev_sents:
            scores = {lg: score_sentence(models[lg], s, char2id, args.device) for lg in langs}
            pred = max(scores, key=scores.get)
            conf[t][pred] += 1
            correct += int(pred == t)
            total += 1
    acc = correct / max(1, total)
    print(f"\nDev accuracy (Neural-HMM, K={args.states}): {acc:.3f}")

    # Eval sentences (if present)
    rows = []
    for t in langs:
        try:
            eval_sents = load_eval_sentences(t, base=str(args.clean_dir))
        except Exception:
            eval_sents = []
        for s in eval_sents:
            sc = {lg: score_sentence(models[lg], s, char2id, args.device) for lg in langs}
            pred = max(sc, key=sc.get)
            rows.append((t, pred, sc, s))

    # Save report
    header = "true\\pred".ljust(8) + "".join(f"{lg:>6}" for lg in langs)
    lines = [
        f"Dev accuracy (Neural-HMM, K={args.states}): {acc:.3f}",
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

    out = args.outdir / f"report_k{args.states}.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
