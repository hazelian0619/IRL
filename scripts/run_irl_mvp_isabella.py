#!/usr/bin/env python3
"""
Run IRL MVP experiment on Isabella's 60-day dataset.

流程：
    1. 使用 export_temporal_embeddings.py 导出窗口级状态嵌入和 reward 代理；
    2. 使用 learning.irl_mvp.train_reward_regressor 在 (z_t, r_t) 上训练简单的 reward 回归器；
    3. 输出训练误差（MSE/MAE），并将模型权重保存至 <root>/models/irl_reward_regressor.pt。

说明：
    - 这是 IRL / 逆回归的最小原型，侧重结构与接口的对齐；
    - reward 代理目前使用 fusion_valence 的 7 日滑窗平均，后续可以替换为更丰富的 reward 定义。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_temporal_embeddings import export_temporal_embeddings
from learning.irl_mvp import train_reward_regressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IRL MVP on Isabella 60-day dataset.")
    parser.add_argument(
        "--root",
        type=str,
        default="data/isabella_irl_3d_clean",
        help="Dataset root.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (e.g. 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs for reward regressor.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for reward regressor.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 1) 导出 temporal embeddings 和窗口级 reward 序列。
    export_temporal_embeddings(args.root, device=args.device)

    # 2) 在 (z_t, r_t) 上训练 reward 回归器。
    metrics = train_reward_regressor(args.root, epochs=args.epochs, lr=args.lr, device=args.device)

    print("\n===== IRL MVP Summary =====")
    print(f"Dataset root : {args.root}")
    print(f"Final MSE    : {metrics['mse']:.6f}")
    print(f"Final MAE    : {metrics['mae']:.6f}")


if __name__ == "__main__":
    main()

