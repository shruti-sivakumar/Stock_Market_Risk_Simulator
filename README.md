# Neural-Enhanced Hidden Markov Models for Language Identification

## Project Overview
This project implements **Markov Chains (MC)**, **Hidden Markov Models (HMM)**, and a **Neural-Enhanced HMM (Neural-HMM)** from scratch for **language identification** on text written in Latin script.  

The Neural-HMM replaces the classical HMM's fixed emission probability table with a small neural network that outputs learned emission probabilities, integrating **deep learning** with **probabilistic reasoning**.

We evaluate all models on:
- **4 training languages:** English, Spanish, French, German
- **4 unseen test languages:** Italian, Portuguese, Dutch, Swedish

---

## Objectives
1. Implement MC, HMM, and Neural-HMM from scratch.
2. Train on 4 languages, test on both trained and unseen languages.
3. Compare performance in terms of:
   - Classification accuracy
   - Mean log-likelihood
   - Probability calibration
   - Plausibility on unseen languages
4. Interpret hidden states for **K=2** models.
5. Run ablations to understand the contribution of:
   - Model type (MC vs HMM vs Neural-HMM)
   - Tokenization method (char vs BPE)
   - Hidden state count (K)
   - Emission parametrization
   - Calibration methods (temperature scaling)

---

## Planned Ablations
| Ablation Group | Variable Changed | Settings |
|----------------|------------------|----------|
| **A – Model family** | MC / HMM / Neural-HMM | A1: MC (char), A2: HMM (char), A3: Neural-HMM (char) |
| **B – Tokenization** | Representation | Char, BPE-1k, BPE-2k |
| **C – Hidden States** | Number of states K | 2, 4, 6, 8 |
| **D – Emission NN** | Architecture | Table, MLP-128, MLP-256, BiLSTM context |
| **E – Calibration** | Temp scaling | Before vs after temp scaling |

---

## Repository Structure
```
src/
  data/               # Corpus preparation, tokenization
  models/             # MC, HMM, Neural-HMM implementations
  eval/               # Metrics & evaluation scripts
notebooks/            # Data exploration, experiments, result analysis
plots/                # Saved figures
results/              # Experiment outputs
requirements.txt      # Dependencies
DATA.md               # Dataset sources & licenses
README.md
LICENSE               
```

---

## Dataset
- **Training languages:** English, Spanish, French, German
- **Unseen languages:** Italian, Portuguese, Dutch, Swedish
- Public-domain corpora (e.g., Project Gutenberg, Wikipedia extracts)
- See [`DATA.md`](DATA.md) for detailed sources and licenses.

---

## Requirements
- Python 3.9+
- PyTorch
- NumPy, Pandas, Matplotlib
- `tokenizers` (for BPE)
- tqdm

Install with:
```bash
pip install -r requirements.txt
```

---

## Usage
### 1. Prepare Data
```bash
python src/data/prepare_corpus.py --languages en es fr de it pt nl sv --output data/processed
```

### 2. Train Models
Example: Train Neural-HMM on char tokens, K=6
```bash
python src/models/neural_hmm.py --train data/processed --tokens char --states 6
```

### 3. Evaluate Models
```bash
python src/eval/evaluate.py --model neural_hmm --test data/processed
```

---

## License
MIT License — see [`LICENSE`](./LICENSE) for details.
