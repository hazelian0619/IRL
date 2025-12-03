#!/usr/bin/env python3
"""
Run preference-based IRL for Isabella 60d (weekly level, IRL V2).

作用：
    - 读取：
        - 周级偏好特征 φ(s_t)：preference_features.npy
        - 周级 reward 序列 r_t：window_valence.npy
        - 周级偏好对 (i ≻ j)：irl_preference_pairs.json
    - 在 φ(s_t) 空间上，用 logistic preference loss 拟合 θ；
    - 计算每周的 R_θ(s_t)；
    - 评估偏好对上的排序准确率。

输出：
    - <root>/features/irl_theta.npy
    - <root>/features/irl_reward_weekly.npy
    - 控制台：θ 每一维的值 + 偏好对准确率。

用法示例：

    cd companion-robot-irl
    python3 scripts/run_irl_preference_isabella.py --root data/isabella_irl_60d_openai_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.irl_preference import (
    PreferencePairConfig,
    fit_preference_reward,
    preference_accuracy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weekly preference-based IRL for Isabella.")
    parser.add_argument(
        "--root",
        type=str,
        default="data/isabella_irl_60d_openai_v2",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=800,
        help="Number of gradient ascent steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate.",
    )
    parser.add_argument(
        "--l2-reg",
        type=float,
        default=1e-3,
        help="L2 regularization strength.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    feat_dir = root / "features"

    # 加载周级特征 φ(s_t)。
    phi_path = feat_dir / "preference_features.npy"
    phi_meta_path = feat_dir / "preference_feature_names.json"
    if not phi_path.exists() or not phi_meta_path.exists():
        raise FileNotFoundError(
            f"Missing preference_features.npy or preference_feature_names.json under {feat_dir}"
        )
    phi = np.load(phi_path).astype(np.float32)  # (T_week, F)
    phi_meta = json.loads(phi_meta_path.read_text(encoding="utf-8"))
    feature_names = phi_meta.get("feature_names", [])

    # 加载偏好对 (i ≻ j)。
    pairs_path = feat_dir / "irl_preference_pairs.json"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing irl_preference_pairs.json under {feat_dir}")
    pairs_meta = json.loads(pairs_path.read_text(encoding="utf-8"))
    pairs_raw = pairs_meta.get("pairs", [])
    pairs = [(int(i), int(j)) for i, j in pairs_raw]
    pair_cfg = PreferencePairConfig(**pairs_meta.get("config", {}))

    T, F = phi.shape
    print(f"[INFO] Dataset root                  : {root}")
    print(f"[INFO] Num weeks (T)                 : {T}")
    print(f"[INFO] Num features (F)              : {F}")
    print(f"[INFO] Num preference pairs          : {len(pairs)}")
    print(
        f"[INFO] Pair config                   : top_k={pair_cfg.top_k}, "
        f"bottom_k={pair_cfg.bottom_k}, margin={pair_cfg.margin}"
    )

    # 拟合 θ。
    theta = fit_preference_reward(
        phi,
        pairs,
        l2_reg=float(args.l2_reg),
        lr=float(args.lr),
        num_steps=int(args.num_steps),
    )

    # 基于 θ 计算每周 reward R_θ(s_t)。
    reward_weekly = phi @ theta  # (T,)

    # 计算偏好对准确率。
    acc = preference_accuracy(phi, pairs, theta)

    # 落盘。
    np.save(feat_dir / "irl_theta.npy", theta.astype(np.float32))
    np.save(feat_dir / "irl_reward_weekly.npy", reward_weekly.astype(np.float32))

    print("\n[INFO] Learned theta (per feature):")
    for idx, val in enumerate(theta):
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        print(f"  θ[{idx:02d}] ({name:16s}) = {val:+.4f}")

    print(f"\n[INFO] Preference accuracy (R(i) > R(j) on pairs): {acc:.3f}")
    print(f"[INFO] Saved irl_theta.npy and irl_reward_weekly.npy under {feat_dir}")


if __name__ == "__main__":
    main()

