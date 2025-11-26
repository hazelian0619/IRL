"""
IRL MVP: simple reward regression on temporal embeddings.

设计目标（主线 Step 7 的 V0 原型）：
- 使用已训练的时序情绪编码器输出的状态嵌入 z_t；
- 使用从 fusion_valence 导出的窗口级 reward 序列 r_t 作为 reward 代理；
- 训练一个简单的线性/MLP 回归器 R(z_t) ≈ r_t；
- 作为 IRL / 逆回归的最小原型，为后续更复杂的 IRL 方法（MaxEnt 等）预留接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader


class TemporalRewardDataset(Dataset):
    """
    简单的数据集封装：每个样本是一个 (z_t, r_t) 对。
    """

    def __init__(self, embeddings: np.ndarray, rewards: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be (T, D), got {embeddings.shape}")
        if rewards.ndim != 1:
            raise ValueError(f"rewards must be (T,), got {rewards.shape}")
        if embeddings.shape[0] != rewards.shape[0]:
            raise ValueError(
                f"embeddings T={embeddings.shape[0]} and rewards T={rewards.shape[0]} mismatch"
            )
        self.emb = torch.from_numpy(embeddings).float()
        self.r = torch.from_numpy(rewards).float()

    def __len__(self) -> int:
        return self.emb.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.emb[idx], self.r[idx]


@dataclass
class RewardRegressorConfig:
    input_dim: int
    hidden_dim: int = 64
    use_mlp: bool = True


class RewardRegressor(nn.Module):
    """
    简单的 reward 回归器：
        - 线性：R(z) = wᵀ z + b
        - 或两层 MLP：ReLU(hidden) -> scalar
    """

    def __init__(self, cfg: RewardRegressorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.use_mlp:
            self.net = nn.Sequential(
                nn.Linear(cfg.input_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, 1),
            )
        else:
            self.net = nn.Linear(cfg.input_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        return out.squeeze(-1)  # (T,)


def train_reward_regressor(
    dataset_root: str | Path,
    epochs: int = 500,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
) -> Dict[str, float]:
    """
    在 <root>/features/temporal_embeddings.npy 和 window_valence.npy 上训练 reward 回归器。

    返回训练后的 MSE/MAE 等指标，并将模型权重保存到 <root>/models/irl_reward_regressor.pt。
    """
    root = Path(dataset_root)
    feat_dir = root / "features"
    model_dir = root / "models"

    emb = np.load(feat_dir / "temporal_embeddings.npy")  # (T, D)
    rewards = np.load(feat_dir / "window_valence.npy")   # (T,)

    ds = TemporalRewardDataset(embeddings=emb, rewards=rewards)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    cfg = RewardRegressorConfig(input_dim=emb.shape[1], hidden_dim=64, use_mlp=True)
    model = RewardRegressor(cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"[INFO] Training reward regressor on {dataset_root}")
    print(f"[INFO] input_dim={cfg.input_dim}, hidden_dim={cfg.hidden_dim}, use_mlp={cfg.use_mlp}")

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, r in loader:
            x = x.to(device)
            r = r.to(device)
            pred = model(x)
            loss = criterion(pred, r)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / len(loader)
        if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
            print(f"[EPOCH {epoch:03d}] train MSE: {avg_loss:.6f}")

    # 训练结束后，在全数据上计算 MSE/MAE，保存模型。
    model.eval()
    with torch.no_grad():
        x_all = torch.from_numpy(emb).float().to(device)
        r_all = torch.from_numpy(rewards).float().to(device)
        pred_all = model(x_all)
        mse = float(nn.functional.mse_loss(pred_all, r_all).item())
        mae = float(nn.functional.l1_loss(pred_all, r_all).item())

    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / "irl_reward_regressor.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "metrics": {"mse": mse, "mae": mae},
        },
        ckpt_path,
    )
    print(f"[INFO] Saved reward regressor checkpoint to {ckpt_path}")
    print(f"[INFO] Final MSE={mse:.6f}, MAE={mae:.6f}")

    return {"mse": mse, "mae": mae}

