#!/usr/bin/env python3
"""
Export temporal embeddings and window-level rewards for IRL/analysis.

设计目的：
- 在已经训练好的时序情绪模型基础上，导出：
  - 每个 7 日窗口的状态嵌入 z_t（BiLSTM+注意力编码器的输出）；
  - 每个窗口对应的 reward 代理 r_t（当前使用 fusion_valence 的窗口平均作为 V0）。

输出：
- <root>/features/temporal_embeddings.npy : (T, D)
- <root>/features/window_valence.npy      : (T,)
- <root>/features/irl_meta.json           : 元信息（维度、窗口参数、源文件等）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.sequence_encoder import BiLSTMAttentionEncoder, SequenceEncoderConfig
from features.emotion_fusion import fusion_valence


def _window_avg(seq: np.ndarray, window: int = 7, step: int = 1) -> np.ndarray:
    """
    对 1D 序列做 7 日滑窗平均，得到窗口级 reward 序列。
    """
    if seq.ndim != 1:
        raise ValueError(f"Expected 1D sequence, got {seq.shape}")
    T = seq.shape[0]
    if T < window:
        raise ValueError(f"Sequence too short for window={window}: length={T}")
    windows = []
    for start in range(0, T - window + 1, step):
        windows.append(seq[start : start + window].mean())
    return np.array(windows, dtype=np.float32)


def export_temporal_embeddings(dataset_root: str | Path, device: str = "cpu") -> None:
    root = Path(dataset_root)
    feat_dir = root / "features"
    model_dir = root / "models"

    rolling = np.load(feat_dir / "rolling_stats.npy")  # (T, 5)
    baseline = np.load(feat_dir / "global_baseline.npy")  # (1,)
    fusion_probs = np.load(feat_dir / "fusion_daily.npy")  # (60, K)

    # 1D daily reward proxy: fusion_valence.
    valence = fusion_valence(fusion_probs)  # (60,)
    window_valence = _window_avg(valence, window=7, step=1)  # (T,)

    # Load encoder weights from emotion sequence model if available, otherwise from autoencoder.
    encoder_cfg = SequenceEncoderConfig(stats_dim=5, baseline_dim=1, hidden_dim=128)
    encoder = BiLSTMAttentionEncoder(encoder_cfg).to(device)

    ckpt_candidates = [
        model_dir / "emotion_sequence_model.pt",
        model_dir / "sequence_autoencoder.pt",
    ]
    loaded = False
    for ckpt_path in ckpt_candidates:
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt.get("state_dict", ckpt)
            # 过滤出 encoder.* 或直接匹配名称。
            enc_state = {
                k.replace("encoder.", "encoder."): v
                for k, v in state_dict.items()
                if k.startswith("encoder.")
            } or {
                k: v
                for k, v in state_dict.items()
                if k.startswith("lstm.") or k.startswith("attn.")
            }
            missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
            print(f"[INFO] Loaded encoder weights from {ckpt_path}")
            if missing:
                print(f"[INFO] Missing keys (expected): {missing}")
            if unexpected:
                print(f"[INFO] Unexpected keys (ignored): {unexpected}")
            loaded = True
            break
    if not loaded:
        print("[WARN] No pretrained encoder checkpoint found, using randomly initialised encoder.")

    encoder.eval()
    # Prepare tensors.
    x = torch.from_numpy(rolling).float().to(device)       # (T, 5)
    b = torch.from_numpy(baseline).float().view(1, -1).to(device)  # (1, 1)
    T_len, D = x.shape
    x = x.view(T_len, 1, D)  # (T, 1, 5)

    with torch.no_grad():
        seq_emb, pooled, attn = encoder(x, b)  # (T, 1, output_dim)

    seq_emb_np = seq_emb[:, 0, :].cpu().numpy().astype(np.float32)  # (T, output_dim)

    # Sanity check: window_valence length should match T.
    if window_valence.shape[0] != seq_emb_np.shape[0]:
        raise ValueError(
            f"window_valence length {window_valence.shape[0]} does not match embeddings T={seq_emb_np.shape[0]}"
        )

    np.save(feat_dir / "temporal_embeddings.npy", seq_emb_np)
    np.save(feat_dir / "window_valence.npy", window_valence)

    meta = {
        "dataset_root": str(root),
        "T": int(seq_emb_np.shape[0]),
        "embedding_dim": int(seq_emb_np.shape[1]),
        "reward_source": "fusion_valence_window_avg",
        "inputs": {
            "rolling_stats": str(feat_dir / "rolling_stats.npy"),
            "global_baseline": str(feat_dir / "global_baseline.npy"),
            "fusion_daily": str(feat_dir / "fusion_daily.npy"),
        },
        "outputs": {
            "temporal_embeddings": str(feat_dir / "temporal_embeddings.npy"),
            "window_valence": str(feat_dir / "window_valence.npy"),
        },
    }
    (feat_dir / "irl_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[INFO] Saved temporal embeddings to {feat_dir/'temporal_embeddings.npy'}")
    print(f"[INFO] Saved window-level rewards to {feat_dir/'window_valence.npy'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export temporal embeddings and rewards for IRL.")
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
        help="Device to run encoder on (e.g. 'cpu', 'cuda').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_temporal_embeddings(args.root, device=args.device)


if __name__ == "__main__":
    main()

