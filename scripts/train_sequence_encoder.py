#!/usr/bin/env python3
"""
Train a sequence encoder on 60-day temporal features.

当前脚本的目标：
- 在已经提取好的 temporal 特征（rolling_stats + global_baseline）之上，
  对 BiLSTM+注意力编码器做一次**自监督预训练**，让模型学会捕捉情绪轨迹模式，
  而不是停留在随机初始化。

说明：
- 当前我们使用的是 `data/isabella_irl_3d_clean` 这条 60 天轨迹，它有完整的多模态 nightly 数据，
  但在本仓库中**没有找到与 Isabella 一一配套的 BFI 报告**。
- BFI-44 前测/验证数据目前是为 Alice Chen 准备的（见 `validation/Alice_Chen_pretest_*.json`），
  且 Alice 的 60 天数据集还未完全就绪，因此此脚本不直接做
  “Isabella 轨迹 -> Big Five” 的监督回归，以避免混用不同 persona 的标签。
- 在此阶段，本脚本采用重构 rolling_stats 的自监督任务：
    - 输入：rolling_stats (54×5)、global_baseline (1×1)
    - 编码：BiLSTMAttentionEncoder
    - 解码：线性层将每个时间步 embedding 投回 5 维统计量，MSE 作为损失。
- 等后续有「多 persona + 各自 60 天轨迹 + BFI 标签」时，可以在此基础上增加人格回归/分类头。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.sequence_encoder import BiLSTMAttentionEncoder, SequenceEncoderConfig


class TemporalSequenceDataset(Dataset):
    """
    Thin dataset wrapper over precomputed temporal features.

    Each item is a single trajectory:
        - rolling_stats: (T, stats_dim)
        - baseline: (baseline_dim,)
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        feat_dir = self.root / "features"
        self.rolling = np.load(feat_dir / "rolling_stats.npy")  # (T, stats_dim)
        self.baseline = np.load(feat_dir / "global_baseline.npy")  # (1,)
        meta = json.loads((feat_dir / "temporal_meta.json").read_text(encoding="utf-8"))
        self.meta = meta

    def __len__(self) -> int:
        # 当前每个 dataset root 对应一条轨迹（一个样本）。
        return 1

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        if idx != 0:
            raise IndexError("This dataset currently exposes only one trajectory (idx=0).")
        x = torch.from_numpy(self.rolling).float()  # (T, stats_dim)
        b = torch.from_numpy(self.baseline).float()  # (1,)
        return x, b


def collate_single(batch):
    """
    将单条 (T, D) 序列打包成 (T, B, D)，B=1。
    """
    assert len(batch) == 1
    x, b = batch[0]
    T, D = x.shape
    x = x.view(T, 1, D)  # (T, 1, D)
    b = b.view(1, -1)    # (1, baseline_dim)
    return x, b


class SequenceAutoencoder(nn.Module):
    """
    使用 BiLSTMAttentionEncoder 的简单自监督 autoencoder：

    - Encoder: BiLSTMAttentionEncoder
    - Decoder: 线性层将每个时间步 embedding 映射回 5 维 stats
    - Loss: MSE(预测的 rolling_stats, 原始 rolling_stats)
    """

    def __init__(self, cfg: SequenceEncoderConfig) -> None:
        super().__init__()
        self.encoder = BiLSTMAttentionEncoder(cfg)
        self.decoder = nn.Linear(self.encoder.output_dim, cfg.stats_dim)

    def forward(self, x: Tensor, baseline: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            x: (T, B, stats_dim)
            baseline: (B, baseline_dim)

        Returns:
            recon: (T, B, stats_dim)
            seq_emb: (T, B, output_dim)
            pooled: (B, output_dim)
            attn: (T, B, 1)
        """
        seq_emb, pooled, attn = self.encoder(x, baseline)
        recon = self.decoder(seq_emb)  # (T, B, stats_dim)
        return recon, seq_emb, pooled, attn


def train_autoencoder(
    dataset_root: str | Path,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
) -> None:
    ds = TemporalSequenceDataset(dataset_root)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_single)

    cfg = SequenceEncoderConfig(stats_dim=5, baseline_dim=1, hidden_dim=128)
    model = SequenceAutoencoder(cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"[INFO] Training autoencoder on {dataset_root}")
    print(f"[INFO] Encoder hidden_dim={cfg.hidden_dim}, layers={cfg.num_layers}")

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, b in loader:
            x = x.to(device)
            b = b.to(device)
            recon, _, _, _ = model(x, b)
            loss = criterion(recon, x)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / len(loader)
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            print(f"[EPOCH {epoch:03d}] recon MSE: {avg_loss:.6f}")

    # 训练结束后，保存模型权重（encoder+decoder），供下游 IRL 或人格回归使用。
    out_dir = Path(dataset_root) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "sequence_autoencoder.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.__dict__,
        },
        ckpt_path,
    )
    print(f"[INFO] Saved sequence autoencoder checkpoint to {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sequence encoder autoencoder.")
    parser.add_argument(
        "--root",
        type=str,
        default="data/isabella_irl_3d_clean",
        help="Dataset root (contains features/rolling_stats.npy, global_baseline.npy).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to train on (e.g. 'cpu', 'cuda').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_autoencoder(args.root, epochs=args.epochs, lr=args.lr, device=args.device)


if __name__ == "__main__":
    main()
