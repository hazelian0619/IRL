#!/usr/bin/env python3
"""
Train a sequence encoder to predict fused emotion labels over time.

设计目标（对齐《12》 Step 6 的 V0 版本）：
- 在已经提取好的 temporal 特征（基于 fusion_valence 的 rolling_stats/global_baseline）上，
  使用 BiLSTM+注意力编码器对 7 日窗口情绪模式进行建模；
- 以 `fusion_daily.npy` 的 7 日滑窗平均概率的 argmax 作为窗口级情绪标签，
  用交叉熵损失训练一个简单的情绪预测头。

说明：
- 当前仅在 `data/isabella_irl_3d_clean` 上训练，样本为单条 60 天轨迹；
- 目标是验证“结构和训练目标”与主线规划对齐，而非追求数值上的泛化性能。
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


def _window_avg_probs(fusion_probs: np.ndarray, window: int = 7, step: int = 1) -> np.ndarray:
    """
    对 60×K 的每日 fusion 概率做 7 日滑窗平均，得到 54×K 的窗口级概率。
    """
    T, K = fusion_probs.shape
    if T < window:
        raise ValueError(f"Sequence too short for window={window}: length={T}")
    windows = []
    for start in range(0, T - window + 1, step):
        windows.append(fusion_probs[start : start + window].mean(axis=0))
    return np.stack(windows, axis=0).astype(np.float32)  # (num_windows, K)


class TemporalEmotionDataset(Dataset):
    """
    单轨迹的窗口级情绪数据集：
        - x: rolling_stats (T, stats_dim)
        - baseline: global_baseline (1,)
        - y: 窗口级情绪标签 index (T,)
    """

    def __init__(self, root: str | Path, window: int = 7, step: int = 1) -> None:
        self.root = Path(root)
        feat_dir = self.root / "features"
        rolling = np.load(feat_dir / "rolling_stats.npy")  # (T, stats_dim)
        baseline = np.load(feat_dir / "global_baseline.npy")  # (1,)
        fusion = np.load(feat_dir / "fusion_daily.npy")  # (60, K)

        win_probs = _window_avg_probs(fusion, window=window, step=step)  # (T, K)
        labels = win_probs.argmax(axis=1).astype(np.int64)  # (T,)

        if rolling.shape[0] != labels.shape[0]:
            raise ValueError(
                f"rolling_stats T={rolling.shape[0]} and window labels T={labels.shape[0]} mismatch"
            )

        self.x = torch.from_numpy(rolling).float()  # (T, stats_dim)
        self.baseline = torch.from_numpy(baseline).float()  # (1,)
        self.y = torch.from_numpy(labels).long()  # (T,)

    def __len__(self) -> int:
        # 单条轨迹，视为一个样本。
        return 1

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        if idx != 0:
            raise IndexError("This dataset currently exposes only one trajectory (idx=0).")
        return self.x, self.baseline, self.y


def collate_single(batch):
    """
    将单条 (T, D) 序列打包成 (T, B, D)，B=1，并保留窗口级标签。
    """
    assert len(batch) == 1
    x, b, y = batch[0]
    T, D = x.shape
    x = x.view(T, 1, D)  # (T, 1, D)
    b = b.view(1, -1)    # (1, baseline_dim)
    y = y.view(T)        # (T,)
    return x, b, y


class EmotionSequenceModel(nn.Module):
    """
    基于 BiLSTMAttentionEncoder 的序列情绪预测模型：

    - Encoder: BiLSTMAttentionEncoder
    - Classifier: Linear(output_dim, num_classes)
    - Loss: 跨时间步的交叉熵（每个 7 日窗口一个标签）
    """

    def __init__(self, cfg: SequenceEncoderConfig, num_classes: int) -> None:
        super().__init__()
        self.encoder = BiLSTMAttentionEncoder(cfg)
        self.classifier = nn.Linear(self.encoder.output_dim, num_classes)

    def forward(self, x: Tensor, baseline: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            x: (T, B, stats_dim)
            baseline: (B, baseline_dim)

        Returns:
            logits: (T, B, num_classes)
            seq_emb: (T, B, output_dim)
            pooled: (B, output_dim)
            attn: (T, B, 1)
        """
        seq_emb, pooled, attn = self.encoder(x, baseline)
        logits = self.classifier(seq_emb)  # (T, B, num_classes)
        return logits, seq_emb, pooled, attn


def train_emotion_model(
    dataset_root: str | Path,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
) -> None:
    root = Path(dataset_root)
    ds = TemporalEmotionDataset(root)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_single)

    # canonical_labels 在 emotion_fusion 中固定为 4。
    num_classes = 4
    cfg = SequenceEncoderConfig(stats_dim=5, baseline_dim=1, hidden_dim=128)
    model = EmotionSequenceModel(cfg, num_classes=num_classes).to(device)

    # 如有自监督 autoencoder 预训练权重，可加载 encoder 部分。
    ckpt_path = root / "models" / "sequence_autoencoder.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        # 仅加载 encoder.* 权重，忽略 decoder。
        encoder_state = {k.replace("encoder.", "encoder."): v for k, v in state_dict.items() if k.startswith("encoder.")}
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        print(f"[INFO] Loaded encoder weights from {ckpt_path}")
        if missing:
            print(f"[INFO] Missing keys (expected): {missing}")
        if unexpected:
            print(f"[INFO] Unexpected keys (ignored): {unexpected}")

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"[INFO] Training emotion sequence model on {dataset_root}")
    print(f"[INFO] Encoder hidden_dim={cfg.hidden_dim}, layers={cfg.num_layers}, num_classes={num_classes}")

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, b, y in loader:
            x = x.to(device)
            b = b.to(device)
            y = y.to(device)  # (T,)

            logits, _, _, _ = model(x, b)  # (T, B, C)
            logits = logits[:, 0, :]       # (T, C)

            loss = criterion(logits, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / len(loader)
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            # 粗略计算训练集上的窗口分类准确率
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = float((preds == y).float().mean().item())
            print(f"[EPOCH {epoch:03d}] loss={avg_loss:.6f}, acc={acc:.3f}")

    out_dir = root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "emotion_sequence_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "num_classes": num_classes,
        },
        ckpt_path,
    )
    print(f"[INFO] Saved emotion sequence model checkpoint to {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sequence encoder for fused emotion prediction.")
    parser.add_argument(
        "--root",
        type=str,
        default="data/isabella_irl_3d_clean",
        help="Dataset root (contains features/rolling_stats.npy, global_baseline.npy, fusion_daily.npy).",
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
    train_emotion_model(args.root, epochs=args.epochs, lr=args.lr, device=args.device)


if __name__ == "__main__":
    main()

