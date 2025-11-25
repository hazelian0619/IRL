"""
Sequence encoders for 60-day IRL temporal features.

This module implements a BiLSTM + attention encoder that operates on
sliding-window statistics, optionally concatenated with a global baseline
vector, as described in the planning doc (`进度/12`).

Input:
    - rolling_stats: (num_windows=54, stats_dim=5)
    - global_baseline: (baseline_dim,)  (e.g. 1 for scalar mood baseline)

Output:
    - sequence_embeddings: (num_windows, hidden_dim*2 + baseline_dim)
    - pooled_embedding: (hidden_dim*2 + baseline_dim)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn, Tensor


@dataclass
class SequenceEncoderConfig:
    stats_dim: int = 5
    baseline_dim: int = 1
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.1


class AttentionPooling(nn.Module):
    """
    Temporal attention pooling.

    Given a sequence of hidden states H (T, B, D), this layer learns a
    scalar score per time step and returns a weighted sum over time:
        pooled = sum_t alpha_t * h_t
    where alpha_t is softmax-normalized along the time dimension.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            h: (T, B, D)

        Returns:
            pooled: (B, D)
            attn: (T, B, 1) attention weights
        """
        scores = self.score(h)  # (T, B, 1)
        attn = torch.softmax(scores, dim=0)
        pooled = (attn * h).sum(dim=0)  # (B, D)
        return pooled, attn


class BiLSTMAttentionEncoder(nn.Module):
    """
    BiLSTM + attention encoder for rolling window statistics.

    For each sample in a batch:
        - Input sequence: (T, stats_dim)
        - Global baseline: (baseline_dim,)

    The encoder:
        - runs BiLSTM over the sequence;
        - applies attention pooling over time;
        - concatenates the global baseline to each time step and pooled vector.
    """

    def __init__(self, cfg: SequenceEncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SequenceEncoderConfig()

        self.lstm = nn.LSTM(
            input_size=self.cfg.stats_dim,
            hidden_size=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout if self.cfg.num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = AttentionPooling(hidden_dim=self.cfg.hidden_dim * 2)

    @property
    def output_dim(self) -> int:
        """Dimension of the encoder output per time step / pooled vector."""
        return self.cfg.hidden_dim * 2 + self.cfg.baseline_dim

    def forward(self, x: Tensor, baseline: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: (T, B, stats_dim) rolling window stats
            baseline: (B, baseline_dim) global baseline vector

        Returns:
            seq_embeddings: (T, B, output_dim)
            pooled: (B, output_dim)
            attn: (T, B, 1)
        """
        T, B, D = x.shape
        assert D == self.cfg.stats_dim, f"Expected stats_dim={self.cfg.stats_dim}, got {D}"

        h, _ = self.lstm(x)  # (T, B, hidden*2)
        pooled, attn = self.attn(h)  # (B, hidden*2), (T, B, 1)

        # Expand baseline across time dimension for concatenation.
        if baseline.dim() != 2 or baseline.shape[0] != B or baseline.shape[1] != self.cfg.baseline_dim:
            raise ValueError(
                f"baseline must be (B, {self.cfg.baseline_dim}), got {tuple(baseline.shape)}"
            )
        baseline_time = baseline.unsqueeze(0).expand(T, B, -1)  # (T, B, baseline_dim)

        seq_emb = torch.cat([h, baseline_time], dim=-1)  # (T, B, hidden*2 + baseline_dim)
        pooled_emb = torch.cat([pooled, baseline], dim=-1)  # (B, hidden*2 + baseline_dim)
        return seq_emb, pooled_emb, attn

