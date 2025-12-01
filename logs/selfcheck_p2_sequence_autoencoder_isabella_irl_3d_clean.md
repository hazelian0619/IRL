P2 自查：isabella_irl_3d_clean 序列自编码器预训练
===============================================

检查时间：当前开发会话。  
数据根目录：`data/isabella_irl_3d_clean`  
模型输出目录：`data/isabella_irl_3d_clean/models`

目标
----

在已有的 temporal 特征（rolling_stats + global_baseline）之上，对 BiLSTM+注意力编码器进行一次自监督预训练，让模型学会压缩和重构情绪轨迹模式，为后续 IRL / 人格回归提供非随机的时序表征。

使用脚本
--------

命令：

```bash
cd companion-robot-irl
python3 scripts/export_temporal_features.py --root data/isabella_irl_3d_clean
python3 scripts/train_sequence_encoder.py --root data/isabella_irl_3d_clean --epochs 40 --device cpu
```

主要结果
--------

训练日志（关键行）：

```text
[INFO] Training autoencoder on data/isabella_irl_3d_clean
[INFO] Encoder hidden_dim=128, layers=1
[EPOCH 001] recon MSE: 12.661565
[EPOCH 020] recon MSE: 1.927435
[EPOCH 040] recon MSE: 1.314219
[INFO] Saved sequence autoencoder checkpoint to data/isabella_irl_3d_clean/models/sequence_autoencoder.pt
```

- 初始重构 MSE ≈ 12.66，40 个 epoch 后下降到 ≈ 1.31；
- 训练过程平稳、无数值发散。

模型与数据接口
--------------

- temporal 特征来自：
  - `data/isabella_irl_3d_clean/features/rolling_stats.npy`：形状 `(54, 5)`；
  - `data/isabella_irl_3d_clean/features/global_baseline.npy`：形状 `(1,)`；
  - 配置说明在 `temporal_meta.json` 中。

- 数据集包装：
  - `scripts/train_sequence_encoder.py` 中的 `TemporalSequenceDataset`：
    - 每个 dataset root 当前对应一条轨迹（一个样本）；
    - `__getitem__` 返回：`x ∈ R^{T×5}` 和 `baseline ∈ R^{1}`。

- 模型结构：
  - `models/sequence_encoder.py` 中的 `BiLSTMAttentionEncoder`：
    - 输入：`x ∈ R^{T×B×5}`, `baseline ∈ R^{B×1}`；
    - 输出：`seq_emb ∈ R^{T×B×(2H+1)}`, `pooled_emb ∈ R^{B×(2H+1)}`；
    - 当前配置：`H=128`，`stats_dim=5`，`baseline_dim=1`。
  - 自编码器封装：`SequenceAutoencoder`：
    - 解码器：`Linear(output_dim, 5)`，逐时间步重构 rolling_stats。

- 模型保存：
  - `data/isabella_irl_3d_clean/models/sequence_autoencoder.pt`：
    - `state_dict`：编码器 + 解码器权重；
    - `config`：`{'stats_dim': 5, 'baseline_dim': 1, 'hidden_dim': 128, 'num_layers': 1, 'dropout': 0.1}`。

质量评估与限制
--------------

- 自监督任务性质：
  - 当前任务是重构 `rolling_stats`，属于时序 autoencoder，而非直接的人格回归；
  - 目的在于让 BiLSTM+注意力对 60 天情绪轨迹的短周期统计有“压缩-还原”的能力，为后续 IRL/人格任务提供可用的时序表示；
- 数据与标签现状：
  - 本仓库中，Isabella 有完整的 60 天多模态 nightly 轨迹，但暂未看到与之配套的 BFI 报告文件；
  - Alice Chen 则有 BFI-44 前测/验证报告（`validation/Alice_Chen_pretest_*.json`），但 Alice 的 60 天轨迹尚未完备；
  - 出于不混用不同 persona 标签的原则，本阶段不直接做“Isabella 轨迹 -> Alice 的 BFI” 这类监督学习，避免在标签层面降级/混淆。

- 数值表现：
  - MSE 从 12.66 降到 1.31 表明模型能够捕捉到 rolling_stats 的主要结构；
  - 具体是否足够好（例如是否过拟合）需要在未来有多条轨迹时进一步验证（目前样本数极少）。

- 对《12》的对齐：
  - 模型结构层面已经对齐“BiLSTM+注意力 + 短期统计 + 长期基线”的设计；
  - 训练目标目前为 V0 自监督重构，后续在人格标签到位后，将增加：
    - 人格/偏好回归头；
    - 与 BFI 报告 / 预设人格的相关性指标（r、MAE 等）。

结论
----

- 在 `data/isabella_irl_3d_clean` 上，BiLSTM+注意力 encoder 已经经过一次可重复的自监督预训练，重构误差显著下降；
- 模型权重与配置已保存，可作为后续 IRL / 人格对齐实验的基础；
- 真正的“轨迹 -> Big Five”实验需要更多 persona + 60 天轨迹配合 BFI 标签，目前保留在规划中的下一阶段。
