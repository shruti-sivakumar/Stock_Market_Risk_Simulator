import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def logsumexp(tensor, dim=-1, keepdim=False):
    m, _ = tensor.max(dim=dim, keepdim=True)
    out = m + (tensor - m).exp().sum(dim=dim, keepdim=True).log()
    return out if keepdim else out.squeeze(dim)


class EmissionMLP(nn.Module):
    """
    Contextual emission network:
      - Input: token ids [B, T]
      - Build windowed context of size (2C+1) using PAD for out-of-range
      - Embedding -> MLP(hidden, dropout, ReLU) -> Linear to K*V
      - Output logits shaped [B, T, K, V]
    """
    def __init__(self, vocab_size: int, n_states: int, emb_dim: int, hidden: int,
                 context: int, pad_idx: int, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_states = n_states
        self.context = context
        self.pad_idx = pad_idx

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)

        in_dim = emb_dim * (2 * context + 1)
        self.ff1 = nn.Linear(in_dim, hidden)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ff2 = nn.Linear(hidden, n_states * vocab_size)

        # Kaiming init for hidden, small init for output
        nn.init.kaiming_uniform_(self.ff1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.ff1.bias)
        nn.init.xavier_uniform_(self.ff2.weight)
        nn.init.zeros_(self.ff2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T] long
        returns: logits [B, T, K, V]
        """
        B, T = x.size()
        C = self.context

        if C > 0:
            # Build padded sequence for windowed lookup
            pad = x.new_full((B, C), self.pad_idx)
            xp = torch.cat([pad, x, pad], dim=1)  # [B, T+2C]
            ctx = [xp[:, i:(i + T)] for i in range(0, 2 * C + 1)]  # length 2C+1, each [B, T]
            ctx = torch.stack(ctx, dim=2)  # [B, T, 2C+1]
        else:
            ctx = x.unsqueeze(2)  # [B, T, 1]

        E = self.emb(ctx)                 # [B, T, 2C+1, emb]
        E = E.reshape(B, T, -1)           # [B, T, (2C+1)*emb]
        h = self.ff1(E)                   # [B, T, hidden]
        h = F.relu(h)
        h = self.drop(h)
        out = self.ff2(h)                 # [B, T, K*V]
        out = out.view(B, T, self.n_states, self.vocab_size)  # [B, T, K, V]
        return out


class NeuralHMM(nn.Module):
    """
    Neural-HMM with:
      - Learnable start logits (K)
      - Learnable transition logits (K x K)
      - Neural emissions via EmissionMLP producing log p(x_t | z_t=k)
    Training uses negative log-likelihood via log-space forward algorithm.
    """
    def __init__(self, vocab_size: int, n_states: int,
                 emb_dim: int, hidden: int, context: int,
                 pad_idx: int = 0, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.K = n_states
        self.pad_idx = pad_idx

        # HMM parameters (logits)
        self.start_logits = nn.Parameter(torch.zeros(self.K))       # [K]
        self.trans_logits  = nn.Parameter(torch.zeros(self.K, self.K))  # [K, K]

        # Emission network
        self.emitter = EmissionMLP(
            vocab_size=vocab_size,
            n_states=n_states,
            emb_dim=emb_dim,
            hidden=hidden,
            context=context,
            pad_idx=pad_idx,
            dropout=dropout,
        )

        # Mildly bias transitions to be near-diagonal at init
        with torch.no_grad():
            self.trans_logits.copy_(torch.full((self.K, self.K), -2.0))
            diag = torch.arange(self.K)
            self.trans_logits[diag, diag] = 2.0

    def _log_emissions_for_observed(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T] long
        returns log emission probs for observed tokens: [B, T, K]
        """
        logits = self.emitter(x)  # [B, T, K, V]
        log_probs = F.log_softmax(logits, dim=-1)  # over V
        # Gather at the observed token indices per (B,T)
        B, T = x.size()
        x_exp = x.unsqueeze(-1).unsqueeze(-1).expand(B, T, self.K, 1)  # [B,T,K,1]
        log_e = log_probs.gather(dim=-1, index=x_exp).squeeze(-1)      # [B,T,K]
        return log_e

    def log_forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T] long with PAD on right
        lengths: [B] lengths (without PAD)
        returns: [B] log-likelihood for each sequence
        """
        device = x.device
        B, T = x.size()

        # HMM params as log-probs
        log_start = F.log_softmax(self.start_logits, dim=-1)        # [K]
        log_trans = F.log_softmax(self.trans_logits, dim=-1)        # [K, K]
        log_emit  = self._log_emissions_for_observed(x)             # [B, T, K]

        ll = x.new_zeros(B, dtype=torch.float32).to(device)

        for b in range(B):
            L = int(lengths[b].item())
            if L <= 0:
                ll[b] = 0.0
                continue

            # alpha[0,k] = log_start[k] + log_emit[b,0,k]
            alpha = log_start + log_emit[b, 0, :]  # [K]

            # Iterate over time
            for t in range(1, L):
                # For each next state k: logsumexp_i( alpha_i + log_trans[i,k] )
                # alpha: [K], log_trans: [K, K]
                trans_part = alpha.unsqueeze(1) + log_trans  # [K, K]
                alpha = log_emit[b, t, :] + logsumexp(trans_part, dim=0)  # [K]

            ll[b] = logsumexp(alpha, dim=0)

        return ll  # [B]

    def neg_log_likelihood(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        ll = self.log_forward(x, lengths)  # [B]
        # Mean NLL over batch
        return -(ll.mean())