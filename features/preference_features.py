"""
Weekly preference feature extraction for IRL V2.

设计目的：
    - 在已有的日级融合情绪 / 行为特征基础上，构建一组「周级、可解释」的特征 φ(s_t)；
    - 为后续的偏好 IRL (R_θ(s) = θ^T φ(s)) 提供一个稳定、低维、带语义的输入空间；
    - 只依赖现有 artifacts：window_valence, rolling_stats, X_daily, feature_names, story phases。

输出：
    - <root>/features/preference_features.npy         # shape = (T_week, F)
    - <root>/features/preference_feature_names.json  # 对应每一列的名字及含义

当前版本的特征维度（按列顺序）：
    1. valence_mean      : 该周 7 日窗口的 valence 均值（= window_valence[t]）
    2. valence_var       : 该周 7 日窗口的 valence 方差
    3. valence_trend     : 该周 valence 的线性斜率（向上/向下）
    4. social_level      : 该周社交强度（beh_social_count 的日均值）
    5. rest_level        : 该周休息/低强度活动强度（beh_rest_count 日均值）
    6. work_level        : 该周工作强度（beh_work_count 日均值）
    7. conflict_level    : 该周「复杂」情绪标签占比（emotion_复杂 的日均值，近似情绪劳动/冲突）
    8. phase_valentine   : 该周是否覆盖情人节相关阶段（valentine_prep / valentine_peak）
    9. phase_election    : 该周是否覆盖选举阶段（election）
   10. phase_recovery    : 该周是否覆盖恢复阶段（recovery）

注意：
    - 这里刻意保持特征数量在 10 维以内，方便后续解释和和 BFI / 故事线对齐；
    - 如果未来需要扩展，可以在不破坏接口的前提下，在末尾追加新特征列。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from utils.dataset_loader import DailyDataset
from story.isabella_story import phase_for_day


@dataclass
class WeeklyWindowSpec:
    """定义周级窗口参数（与 temporal_features 的 SlidingWindowConfig 对齐）。"""

    window: int = 7
    step: int = 1


def _compute_week_day_indices(num_days: int, spec: WeeklyWindowSpec) -> List[Tuple[int, int]]:
    """
    生成每个周窗口对应的日级索引区间（左闭右开，0-based）。

    返回：
        windows: 列表，元素为 (start_idx, end_idx)，表示包含天 [start_idx, end_idx)。
    """
    if num_days < spec.window:
        raise ValueError(f"Sequence too short for window={spec.window}: length={num_days}")
    indices: List[Tuple[int, int]] = []
    for start in range(0, num_days - spec.window + 1, spec.step):
        indices.append((start, start + spec.window))
    return indices


def _safe_get_column(mat: np.ndarray, names: List[str], key: str) -> np.ndarray:
    """
    从日级特征矩阵中安全地取出某一列；如不存在则返回全 0。
    """
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix for daily features, got {mat.shape}")
    try:
        idx = names.index(key)
    except ValueError:
        return np.zeros((mat.shape[0],), dtype=np.float32)
    col = mat[:, idx].astype(np.float32)
    return col


def export_preference_features(dataset_root: str | Path) -> None:
    """
    从 <root> 下已有 artifacts 推导周级偏好特征，并落盘。

    要求存在：
        - <root>/features/window_valence.npy
        - <root>/features/rolling_stats.npy
        - <root>/features/X_daily.npy
        - <root>/features/feature_names.json

    其中：
        - window_valence / rolling_stats 用于构造 valence 向量统计；
        - X_daily / feature_names 用于构造行为 & 情绪标签相关的周级特征；
        - DailyDataset / phase_for_day 用于根据 day→phase 生成阶段 one-hot 特征。
    """
    root = Path(dataset_root)
    feat_dir = root / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    # --- 加载基础 artifacts ---
    ds = DailyDataset(root)
    days = list(map(int, ds.days))
    num_days = len(days)

    window_valence_path = feat_dir / "window_valence.npy"
    rolling_stats_path = feat_dir / "rolling_stats.npy"
    x_daily_path = feat_dir / "X_daily.npy"
    feat_names_path = feat_dir / "feature_names.json"

    if not window_valence_path.exists():
        raise FileNotFoundError(f"Missing window_valence.npy under {feat_dir}")
    if not rolling_stats_path.exists():
        raise FileNotFoundError(f"Missing rolling_stats.npy under {feat_dir}")
    if not x_daily_path.exists():
        raise FileNotFoundError(f"Missing X_daily.npy under {feat_dir}")
    if not feat_names_path.exists():
        raise FileNotFoundError(f"Missing feature_names.json under {feat_dir}")

    window_valence = np.load(window_valence_path).astype(np.float32)  # (T_week,)
    rolling_stats = np.load(rolling_stats_path).astype(np.float32)  # (T_week, 5)
    X_daily = np.load(x_daily_path).astype(np.float32)  # (T_day, F_daily)
    feature_names: List[str] = json.loads(feat_names_path.read_text(encoding="utf-8"))

    if X_daily.shape[0] != num_days:
        raise ValueError(
            f"X_daily first dim {X_daily.shape[0]} does not match num_days {num_days}"
        )

    T_week = window_valence.shape[0]
    if rolling_stats.shape[0] != T_week:
        raise ValueError(
            f"rolling_stats length {rolling_stats.shape[0]} mismatch window_valence {T_week}"
        )

    # 与 temporal_features 使用的窗口参数保持一致。
    win_spec = WeeklyWindowSpec(window=7, step=1)
    week_day_windows = _compute_week_day_indices(num_days, win_spec)  # len = T_week
    if len(week_day_windows) != T_week:
        raise ValueError(
            f"Weekly window count {len(week_day_windows)} mismatch window_valence {T_week}"
        )

    # --- 预先取出我们关心的日级列 ---
    beh_social = _safe_get_column(X_daily, feature_names, "beh_social_count")
    beh_rest = _safe_get_column(X_daily, feature_names, "beh_rest_count")
    beh_work = _safe_get_column(X_daily, feature_names, "beh_work_count")
    # 用 emotion_复杂 作为「复杂情绪/情绪劳动」的近似 proxy（0/1 one-hot）。
    emo_complex = _safe_get_column(X_daily, feature_names, "emotion_复杂")

    # rolling_stats 列含义：mean, var, max, min, slope
    if rolling_stats.shape[1] < 5:
        raise ValueError(f"Expected rolling_stats second dim >=5, got {rolling_stats.shape}")
    val_mean_from_stats = rolling_stats[:, 0]
    val_var = rolling_stats[:, 1]
    val_slope = rolling_stats[:, 4]

    # sanity：window_valence 应与 rolling_stats 的 mean 基本一致（允许轻微差异）。
    if not np.allclose(window_valence, val_mean_from_stats, atol=1e-4):
        # 不直接报错，只是留痕，便于排查数据差异。
        print(
            "[WARN] window_valence and rolling_stats[:,0] differ beyond 1e-4; "
            "using window_valence as valence_mean."
        )

    # --- 为每个周窗口构造特征 ---
    features: List[List[float]] = []
    for w_idx, (start, end) in enumerate(week_day_windows):
        # 安全防御：确保索引合法。
        if not (0 <= start < end <= num_days):
            raise ValueError(f"Invalid window indices ({start}, {end}) for num_days={num_days}")

        # 对应的日级切片。
        beh_social_week = beh_social[start:end]
        beh_rest_week = beh_rest[start:end]
        beh_work_week = beh_work[start:end]
        emo_complex_week = emo_complex[start:end]

        # 行为/情绪相关特征：用 7 日窗口的日均值，便于跨窗口比较。
        social_level = float(beh_social_week.mean()) if beh_social_week.size > 0 else 0.0
        rest_level = float(beh_rest_week.mean()) if beh_rest_week.size > 0 else 0.0
        work_level = float(beh_work_week.mean()) if beh_work_week.size > 0 else 0.0
        conflict_level = float(emo_complex_week.mean()) if emo_complex_week.size > 0 else 0.0

        # valence 统计：均值来自 window_valence，方差/趋势来自 rolling_stats。
        val_mean = float(window_valence[w_idx])
        val_var_w = float(val_var[w_idx])
        val_slope_w = float(val_slope[w_idx])

        # 阶段 one-hot 特征：查看该周覆盖的 day 所属 PHASE。
        days_this_week = [days[d] for d in range(start, end)]
        phase_names = [phase_for_day(int(d))["name"] for d in days_this_week]
        has_valentine = any(
            name in ("valentine_prep", "valentine_peak") for name in phase_names
        )
        has_election = any(name == "election" for name in phase_names)
        has_recovery = any(name == "recovery" for name in phase_names)

        phase_valentine = 1.0 if has_valentine else 0.0
        phase_election = 1.0 if has_election else 0.0
        phase_recovery = 1.0 if has_recovery else 0.0

        features.append(
            [
                val_mean,        # valence_mean
                val_var_w,       # valence_var
                val_slope_w,     # valence_trend
                social_level,    # social_level
                rest_level,      # rest_level
                work_level,      # work_level
                conflict_level,  # conflict_level
                phase_valentine, # phase_valentine
                phase_election,  # phase_election
                phase_recovery,  # phase_recovery
            ]
        )

    feat_mat = np.array(features, dtype=np.float32)

    # --- 落盘 ---
    np.save(feat_dir / "preference_features.npy", feat_mat)

    feature_names_out = [
        "valence_mean",
        "valence_var",
        "valence_trend",
        "social_level",
        "rest_level",
        "work_level",
        "conflict_level",
        "phase_valentine",
        "phase_election",
        "phase_recovery",
    ]
    meta = {
        "dataset_root": str(root),
        "num_weeks": int(T_week),
        "num_days": int(num_days),
        "window": win_spec.window,
        "step": win_spec.step,
        "feature_names": feature_names_out,
        "description": {
            "valence_mean": "7 日窗口 valence 均值（window_valence[t]）",
            "valence_var": "7 日窗口 valence 方差（rolling_stats[:,1]）",
            "valence_trend": "7 日窗口内 valence 的线性趋势斜率（正=上升，负=下降）",
            "social_level": "该周平均社交强度（beh_social_count 日均值）",
            "rest_level": "该周平均休息/低强度活动强度（beh_rest_count 日均值）",
            "work_level": "该周平均工作强度（beh_work_count 日均值）",
            "conflict_level": "该周情绪标签为「复杂」的日占比，近似情绪劳动/冲突频率",
            "phase_valentine": "该周是否覆盖情人节筹备/高峰阶段（valentine_prep/peak）",
            "phase_election": "该周是否覆盖选举阶段（election）",
            "phase_recovery": "该周是否覆盖恢复阶段（recovery）",
        },
    }
    (feat_dir / "preference_feature_names.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[INFO] Saved weekly preference features to {feat_dir/'preference_features.npy'}")
    print(f"[INFO] Saved feature metadata to {feat_dir/'preference_feature_names.json'}")


__all__ = ["export_preference_features"]

