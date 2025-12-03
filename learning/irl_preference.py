"""
Preference-based IRL utilities for weekly reward learning (IRL V2).

本模块对应《执行.md》中的 Phase C / Phase D：

- Phase C:
    - 从周级 reward 序列 r_t（window_valence）构造偏好对 (i ≻ j)；
    - 提供简单的 summary 函数，方便把偏好对映射回 day 范围做 story 检查。
- Phase D:
    - 在周级偏好特征 φ(s_t) 空间上，使用 logistic preference loss 拟合 θ，
      得到线性偏好函数 R_θ(s) = θ^T φ(s)。

注意：
- 这里的 IRL 是「state-level preference IRL」：
    - 仅在周级状态上建模偏好排序；
    - 不显式建 MDP / action / transition；
    - 更适合当前单 persona、单轨迹的设定。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


PreferencePair = Tuple[int, int]  # (i, j) 表示 week i ≻ week j


@dataclass
class PreferencePairConfig:
    """
    构造周级偏好对的配置。

    参数含义：
        top_k   : 选出 r_t 最高的前 top_k 个周索引作为 Top 集合；
        bottom_k: 选出 r_t 最低的后 bottom_k 个周索引作为 Bottom 集合；
        margin  : 若 r_i - r_j > margin，则认为 i 明显优于 j，可额外加入偏好对。
    """

    top_k: int = 5
    bottom_k: int = 5
    margin: float = 0.1


def build_preference_pairs(r: np.ndarray, cfg: PreferencePairConfig) -> List[PreferencePair]:
    """
    根据周级 reward 序列 r_t 构造偏好对 (i ≻ j)。

    设计思路：
        - 先找出「明显高 reward」的 Top weeks 和「明显低 reward」的 Bottom weeks；
        - 对所有 (i in Top, j in Bottom) 生成高置信度偏好对；
        - 可选：对满足 r_i - r_j > margin 的任意 (i, j) 再生成一批「差距足够大」的偏好对，
          捕获更多中间区域的信息。

    Args:
        r: shape (T_week,) 的 1D 周级 reward 序列（例如 window_valence）。
        cfg: PreferencePairConfig 配置。

    Returns:
        一个二维列表，元素为 (i, j) 的元组，表示 i ≻ j。
    """
    if r.ndim != 1:
        raise ValueError(f"Expected 1D reward array, got shape {r.shape}")
    T = r.shape[0]
    if T == 0:
        return []

    # 排序索引：从小到大。
    sort_idx = np.argsort(r)  # shape (T,)
    bottom_idx = sort_idx[: cfg.bottom_k]
    top_idx = sort_idx[-cfg.top_k :] if cfg.top_k > 0 else np.array([], dtype=int)

    pairs: List[PreferencePair] = []

    # 1) 高置信度 Top vs Bottom 偏好对。
    for i in top_idx:
        for j in bottom_idx:
            if i == j:
                continue
            pairs.append((int(i), int(j)))

    # 2) 可选：基于 margin 的额外偏好对。
    if cfg.margin > 0.0:
        for i in range(T):
            for j in range(T):
                if i == j:
                    continue
                if r[i] - r[j] > cfg.margin:
                    pairs.append((int(i), int(j)))

    return pairs


def summarize_pairs(
    r: np.ndarray,
    pairs: Sequence[PreferencePair],
    days_for_week: Sequence[Tuple[int, int]],
) -> str:
    """
    生成一个偏好对 summary 字符串，帮助人眼检查 (i ≻ j) 是否符合 story。

    Args:
        r: shape (T_week,) 的 reward 序列。
        pairs: 若干 (i, j) 偏好对。
        days_for_week: 长度为 T_week 的列表，每个元素为 (start_day, end_day)，
                       例如 (1,7) 表示窗口覆盖 day 1–7。

    Returns:
        summary: 多行文本，可打印或写入日志。
    """
    if r.ndim != 1:
        raise ValueError(f"Expected 1D reward array, got shape {r.shape}")
    T = r.shape[0]
    if len(days_for_week) != T:
        raise ValueError(
            f"days_for_week length {len(days_for_week)} does not match reward length {T}"
        )

    lines: List[str] = []
    lines.append(f"Total weeks: {T}")
    lines.append(f"Total preference pairs: {len(pairs)}")
    lines.append("")

    # 只列出前若干条偏好对，避免输出过长。
    max_show = min(40, len(pairs))
    lines.append(f"First {max_show} preference pairs (i ≻ j):")
    for k in range(max_show):
        i, j = pairs[k]
        r_i, r_j = float(r[i]), float(r[j])
        d_i = days_for_week[i]
        d_j = days_for_week[j]
        lines.append(
            f"  #{k:02d}: week_i={i:2d} (days {d_i[0]}–{d_i[1]}, r_i={r_i:.3f})"
            f"  ≻  week_j={j:2d} (days {d_j[0]}–{d_j[1]}, r_j={r_j:.3f})"
        )

    return "\n".join(lines)


def week_day_windows_from_sequence_length(num_days: int, window: int = 7, step: int = 1):
    """
    小工具：给定 60 日长度，生成和 temporal_features/export_temporal_embeddings
    一致的周级窗口 day 范围（1-based，包含两端）。
    """
    if num_days < window:
        raise ValueError(f"num_days={num_days} smaller than window={window}")
    windows: List[Tuple[int, int]] = []
    for start in range(0, num_days - window + 1, step):
        # 输出为 1-based 且右端包含：例如 start=0 -> day 1–7。
        windows.append((start + 1, start + window))
    return windows


def fit_preference_reward(
    phi: np.ndarray,
    pairs: Sequence[PreferencePair],
    l2_reg: float = 1e-3,
    lr: float = 1e-2,
    num_steps: int = 500,
) -> np.ndarray:
    """
    在周级偏好特征 φ(s_t) 空间上，用 logistic preference loss 拟合 θ。

    目标：
        对每个偏好对 (i ≻ j)，最大化 log σ(θ^T(φ_i - φ_j))，并加入 L2 正则：

            L(θ) = Σ_{(i,j)} log σ(θ^T(φ_i - φ_j)) - λ||θ||^2

    这里使用一个简洁的 batch 梯度上升实现，样本规模较小（T≈54，pairs≈1e3），
    纯 numpy 即可。

    Args:
        phi: shape (T_week, F) 的特征矩阵。
        pairs: 偏好对列表，每个元素为 (i, j)，表示 week i ≻ week j。
        l2_reg: L2 正则系数 λ。
        lr: 学习率。
        num_steps: 梯度上升步骤数。

    Returns:
        theta: shape (F,) 的参数向量。
    """
    if phi.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {phi.shape}")
    T, F = phi.shape
    if not pairs:
        raise ValueError("No preference pairs provided to fit_preference_reward")

    # 将 pairs 转为 numpy 索引向量，便于批量运算。
    idx_i = np.array([p[0] for p in pairs], dtype=np.int64)
    idx_j = np.array([p[1] for p in pairs], dtype=np.int64)
    if idx_i.min() < 0 or idx_i.max() >= T or idx_j.min() < 0 or idx_j.max() >= T:
        raise ValueError("Preference pair indices out of range for phi")

    theta = np.zeros((F,), dtype=np.float32)

    for _step in range(num_steps):
        # φ_i, φ_j: shape (N_pairs, F)
        phi_i = phi[idx_i]  # (N, F)
        phi_j = phi[idx_j]  # (N, F)
        diff = phi_i - phi_j  # (N, F)

        # delta = θ^T(φ_i - φ_j)
        delta = diff @ theta  # (N,)

        # σ(delta)
        sigma = 1.0 / (1.0 + np.exp(-delta))

        # 对 log σ(delta) 的梯度： (1 - σ(delta)) * (φ_i - φ_j)
        # 汇总所有 pairs 的贡献，再减去 L2 正则梯度 2λθ。
        grad = (1.0 - sigma)[:, None] * diff  # (N, F)
        grad = grad.mean(axis=0) - 2.0 * l2_reg * theta  # (F,)

        theta = theta + lr * grad.astype(np.float32)

    return theta.astype(np.float32)


def preference_accuracy(phi: np.ndarray, pairs: Sequence[PreferencePair], theta: np.ndarray) -> float:
    """
    计算在给定 θ 下，偏好对 (i ≻ j) 中有多少比例满足 R_θ(s_i) > R_θ(s_j)。

    返回：
        accuracy: 0-1 之间的浮点数。
    """
    if phi.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {phi.shape}")
    T, F = phi.shape
    if theta.shape != (F,):
        raise ValueError(f"theta shape {theta.shape} incompatible with feature dim {F}")
    if not pairs:
        return float("nan")

    idx_i = np.array([p[0] for p in pairs], dtype=np.int64)
    idx_j = np.array([p[1] for p in pairs], dtype=np.int64)
    if idx_i.min() < 0 or idx_i.max() >= T or idx_j.min() < 0 or idx_j.max() >= T:
        raise ValueError("Preference pair indices out of range for phi")

    r = phi @ theta  # (T,)
    satisfied = (r[idx_i] > r[idx_j]).astype(np.float32)
    return float(satisfied.mean())


__all__ = [
    "PreferencePairConfig",
    "build_preference_pairs",
    "summarize_pairs",
    "week_day_windows_from_sequence_length",
    "fit_preference_reward",
    "preference_accuracy",
]

