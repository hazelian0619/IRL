P2 自查：isabella_irl_3d_clean 时序情绪模型（基于融合概率）
======================================================

检查时间：当前开发会话。  
数据根目录：`data/isabella_irl_3d_clean`  
模型输出目录：`data/isabella_irl_3d_clean/models`

目标
----

在已经实现的多模态情绪融合基础上，对 BiLSTM+注意力编码器进行一次
“以融合情绪为目标”的时序建模训练，以对齐《12》中的 Step 6 设计：

- 输入：基于 fusion_valence 的 7 日滑窗统计（rolling_stats）和指数衰减基线；
- 输出：每个 7 日窗口的情绪类别（积极/消极/中性/复杂）预测；
- 训练目标：交叉熵损失，而非仅依靠 MSE 自监督重构。

使用脚本
--------

命令：

```bash
cd companion-robot-irl
python3 scripts/export_fusion_features.py --root data/isabella_irl_3d_clean
python3 scripts/export_temporal_features.py --root data/isabella_irl_3d_clean
python3 scripts/train_emotion_sequence_model.py --root data/isabella_irl_3d_clean --epochs 80 --device cpu
```

训练日志（关键行）：

```text
[INFO] Loaded encoder weights from data/isabella_irl_3d_clean/models/sequence_autoencoder.pt
[INFO] Missing keys (expected): ['classifier.weight', 'classifier.bias']
[INFO] Training emotion sequence model on data/isabella_irl_3d_clean
[INFO] Encoder hidden_dim=128, layers=1, num_classes=4
[EPOCH 001] loss=1.398761, acc=0.093
[EPOCH 020] loss=0.017209, acc=1.000
[EPOCH 040] loss=0.002714, acc=1.000
[EPOCH 060] loss=0.001593, acc=1.000
[EPOCH 080] loss=0.001239, acc=1.000
[INFO] Saved emotion sequence model checkpoint to data/isabella_irl_3d_clean/models/emotion_sequence_model.pt
```

数据与目标
----------

- 输入特征：
  - `features/rolling_stats.npy`：形状 `(54, 5)`，基于 `fusion_valence` 的 7 日滑窗统计；
  - `features/global_baseline.npy`：形状 `(1,)`，基于 `fusion_valence` 的 EWMA 全局基线；
  - `features/fusion_daily.npy`：形状 `(60, 4)`，多模态融合后的每日情绪概率。

- 窗口级情绪标签：
  - 从 `fusion_daily.npy` 通过 7 日滑窗平均得到 `(54, 4)` 的窗口级情绪概率；
  - 对每个窗口取 argmax 作为标签 ∈ {0,1,2,3}，对应 canonical labels：

    ```text
    ["积极", "消极", "中性", "复杂"]
    ```

模型结构
--------

- `models/sequence_encoder.py`：
  - `BiLSTMAttentionEncoder`：
    - 输入：`x ∈ R^{T×B×5}`, `baseline ∈ R^{B×1}`；
    - 输出：`seq_emb ∈ R^{T×B×(2H+1)}`, `pooled_emb ∈ R^{B×(2H+1)}`；
    - 当前配置：`H=128`，`stats_dim=5`，`baseline_dim=1`。

- `scripts/train_emotion_sequence_model.py` 中的 `EmotionSequenceModel`：
  - `encoder`: BiLSTMAttentionEncoder；
  - `classifier`: `Linear(output_dim, num_classes=4)`；
  - Loss: 时间维上的交叉熵。
  - 如存在 `sequence_autoencoder.pt`，先加载 encoder 权重作为初始化。

质量评估与限制
--------------

- 训练表现：
  - 初始 epoch 1：loss≈1.40，acc≈0.093（接近均匀随机猜的水平）；
  - epoch 20 起：loss 快速下降到≈0.017，acc≈1.000；
  - epoch 80：loss≈0.0012，acc≈1.000；
  - 说明在单条 60 天轨迹上，模型几乎可以完美拟合窗口级融合情绪模式。

- 限制：
  - 当前只有 Isabella 一条轨迹（54 个窗口），无法评估泛化能力；
  - 单模态预测器和融合 GT 本身是规则化 V0 实现，因此时序模型的“完美拟合”目前更多说明：
    - 编码器 + 分类头足够表达这些规则注入的情绪模式；
    - 而不代表已经学到跨 persona 可泛化的情绪模型。

- 与《12》的对齐：
  - 结构：已实现“7 日滑窗统计 + 衰减基线 → BiLSTM+注意力 → 窗口级情绪预测”的结构；
  - 目标：训练目标已从纯 MSE 自监督提升为使用情绪标签（由融合概率导出）做分类的监督任务；
  - 依赖：输入使用的是多模态融合的 valence 序列，而非单一路径的 mood_score。

结论
----

- 在 `data/isabella_irl_3d_clean` 上，时序情绪模型已经按主线结构实现：
  - 多模态融合 → fusion_daily；
  - fusion_valence → 滑窗/衰减；
  - BiLSTM+注意力编码器 + 情绪分类头。
- 这一层提供了 IRL 和人格对齐之前的“情绪时序表示”，后续可以在此基础上：
  - 设计 IRL 状态/动作/奖励结构；
  - 使用 BFI 前测/后测向量对 embedding 做 probe 或回归，实现人格对齐实验。

