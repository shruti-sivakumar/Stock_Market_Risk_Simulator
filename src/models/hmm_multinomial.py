import numpy as np

# Prefer CategoricalHMM (single symbol per time step).
# Fall back to MultinomialHMM by one-hot encoding if CategoricalHMM is unavailable.
try:
    from hmmlearn.hmm import CategoricalHMM as _HMM
    _CATEGORICAL = True
except Exception:
    from hmmlearn.hmm import MultinomialHMM as _HMM
    _CATEGORICAL = False


def pack_char_sequences(sentences, char2id, unk_id=None):
    """Convert list[str] -> (X, lengths) for hmmlearn.
    - For CategoricalHMM: X is (N, 1) int-ids.
    - For MultinomialHMM (fallback): we will one-hot later.
    """
    if unk_id is None:
        unk_id = char2id.get("[UNK]", len(char2id) - 1)
    ids = []
    lengths = []
    for s in sentences:
        seq = [char2id.get(c, unk_id) for c in s]
        if not seq:
            continue
        ids.extend(seq)
        lengths.append(len(seq))
    if not lengths:
        X = np.zeros((0, 1), dtype=np.int64)
    else:
        X = np.asarray(ids, dtype=np.int64).reshape(-1, 1)
    return X, np.asarray(lengths, dtype=np.int32)


def one_hot(X_int, vocab_size):
    """X_int: (N,1) ints -> (N,V) one-hot for MultinomialHMM fallback."""
    N = X_int.shape[0]
    Y = np.zeros((N, vocab_size), dtype=float)
    if N:
        Y[np.arange(N), X_int[:, 0]] = 1.0
    return Y


def init_sticky_transitions(n_states: int, diag: float = 0.85) -> np.ndarray:
    """Near-diagonal transition matrix for stability."""
    off = (1.0 - diag) / max(1, n_states - 1)
    A = np.full((n_states, n_states), off, dtype=float)
    np.fill_diagonal(A, diag)
    return A


def train_lang_hmm(
    train_sentences,
    char2id,
    n_states: int = 5,
    n_iter: int = 50,
    tol: float = 1e-3,
    sticky_diag: float = 0.85,
    seed: int = 0,
    smooth_eps: float = 1e-6,
):
    """Train a per-language HMM on characters."""
    V = len(char2id)
    X_int, lengths = pack_char_sequences(train_sentences, char2id)
    if len(lengths) == 0:
        raise ValueError("No training data after packing sequences.")

    hmm = _HMM(
        n_components=n_states,
        n_iter=n_iter,
        tol=tol,
        random_state=seed,
        verbose=False,
    )

    # Number of symbols / features
    # CategoricalHMM uses n_features = V for symbol ids.
    # MultinomialHMM uses n_features = V for multinomial count vectors.
    hmm.n_features = V

    # Sticky init
    hmm.startprob_ = np.full(n_states, 1.0 / n_states, dtype=float)
    hmm.transmat_ = init_sticky_transitions(n_states, diag=sticky_diag)

    if _CATEGORICAL:
        # X_int is (N,1) of ints -> perfect for CategoricalHMM
        hmm.fit(X_int, lengths)
    else:
        # Fallback: convert to one-hot for MultinomialHMM
        X_oh = one_hot(X_int, V)  # (N,V)
        hmm.fit(X_oh, lengths)

    # Smooth emissions to avoid zeros
    # CategoricalHMM: emissionprob_ shape (K,V)
    # MultinomialHMM: emissionprob_ shape (K,V) as well.
    B = hmm.emissionprob_
    B = (B + smooth_eps) / (B + smooth_eps).sum(axis=1, keepdims=True)
    hmm.emissionprob_ = B
    return hmm


def score_sentence(hmm, sentence: str, char2id) -> float:
    """Return log-likelihood under the trained model."""
    V = len(char2id)
    X_int, lengths = pack_char_sequences([sentence], char2id)
    if len(lengths) == 0:
        return float("-inf")

    if _CATEGORICAL:
        return float(hmm.score(X_int, lengths))
    else:
        X_oh = one_hot(X_int, V)
        return float(hmm.score(X_oh, lengths))