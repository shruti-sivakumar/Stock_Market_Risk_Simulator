import math
from collections import defaultdict
import numpy as np

class HMM:
    def __init__(self, n_states=6, alpha=0.01):
        self.n_states = n_states
        self.alpha = alpha
        self.vocab = None
        self.char2idx = None
        self.idx2char = None
        self.start_probs = None
        self.trans_probs = None
        self.emit_probs = None
        self.trained = False

    def set_vocab(self, vocab):
        self.vocab = sorted(vocab)
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}

    def _init_params(self):
        V = len(self.vocab)
        self.start_probs = np.full(self.n_states, 1.0 / self.n_states)
        self.trans_probs = np.full((self.n_states, self.n_states), 1.0 / self.n_states)
        self.emit_probs = np.full((self.n_states, V), 1.0 / V)

    def fit(self, sequences, n_iter=15):
        """Train using Baum-Welch (simplified)."""
        self._init_params()
        for _ in range(n_iter):
            # Expectation
            A_num = np.zeros((self.n_states, self.n_states))
            A_den = np.zeros(self.n_states)
            B_num = np.zeros((self.n_states, len(self.vocab)))
            B_den = np.zeros(self.n_states)
            pi_num = np.zeros(self.n_states)

            for seq in sequences:
                alpha, scale = self._forward(seq)
                beta = self._backward(seq, scale)

                xi_sum = np.zeros((self.n_states, self.n_states))
                gamma = (alpha * beta) / (alpha * beta).sum(axis=1, keepdims=True)

                for t in range(len(seq) - 1):
                    xi = alpha[t][:, None] * self.trans_probs * self.emit_probs[:, seq[t + 1]] * beta[t + 1]
                    xi_sum += xi / xi.sum()
                A_num += xi_sum
                A_den += gamma[:-1].sum(axis=0)
                for t, obs in enumerate(seq):
                    B_num[:, obs] += gamma[t]
                B_den += gamma.sum(axis=0)
                pi_num += gamma[0]

            # Maximization with Laplace smoothing
            self.start_probs = (pi_num + self.alpha) / (pi_num.sum() + self.alpha * self.n_states)
            self.trans_probs = (A_num + self.alpha) / (A_den[:, None] + self.alpha * self.n_states)
            self.emit_probs = (B_num + self.alpha) / (B_den[:, None] + self.alpha * len(self.vocab))
        self.trained = True

    def _forward(self, seq):
        alpha = np.zeros((len(seq), self.n_states))
        scale = np.zeros(len(seq))
        alpha[0] = self.start_probs * self.emit_probs[:, seq[0]]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        for t in range(1, len(seq)):
            alpha[t] = (alpha[t - 1] @ self.trans_probs) * self.emit_probs[:, seq[t]]
            scale[t] = alpha[t].sum()
            alpha[t] /= scale[t]
        return alpha, scale

    def _backward(self, seq, scale):
        beta = np.zeros((len(seq), self.n_states))
        beta[-1] = 1.0 / scale[-1]
        for t in reversed(range(len(seq) - 1)):
            beta[t] = (self.trans_probs @ (self.emit_probs[:, seq[t + 1]] * beta[t + 1])) / scale[t]
        return beta

    def logprob(self, sentence):
        assert self.trained
        seq = [self.char2idx[c] for c in sentence if c in self.char2idx]
        if not seq:
            return -float('inf')
        _, scale = self._forward(seq)
        return sum(np.log(scale))

class HMMClassifier:
    def __init__(self, n_states=6, alpha=0.01, n_iter=15):
        self.models = {}
        self.n_states = n_states
        self.alpha = alpha
        self.n_iter = n_iter

    def fit_per_language(self, data_by_lang):
        vocab = set()
        BOS, EOS = "<s>", "</s>"
        for sents in data_by_lang.values():
            for s in sents:
                vocab.update([BOS, EOS] + list(s))
        for lang, sents in data_by_lang.items():
            sequences = []
            hmm = HMM(n_states=self.n_states, alpha=self.alpha)
            hmm.set_vocab(vocab)
            for s in sents:
                seq = [BOS] + list(s) + [EOS]
                seq_idx = [hmm.char2idx[c] for c in seq if c in hmm.char2idx]
                if seq_idx:
                    sequences.append(seq_idx)
            hmm.fit(sequences, n_iter=self.n_iter)
            self.models[lang] = hmm

    def score(self, sentence):
        return {lang: m.logprob("<s>" + sentence + "</s>") for lang, m in self.models.items()}

    def predict(self, sentence):
        scores = self.score(sentence)
        return max(scores, key=scores.get), scores
