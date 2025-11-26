P3 自查：Isabella 60 天 IRL MVP（奖励回归原型）
============================================

检查时间：当前开发会话。  
数据根目录：`data/isabella_irl_3d_clean`  
模型输出目录：`data/isabella_irl_3d_clean/models`

目标
----

在已完成的多模态情绪融合与时序情绪模型基础上，构建一个最小可用的 IRL / 逆回归原型：

- 状态：BiLSTM+注意力编码器输出的 7 日窗口情绪嵌入 z_t；
- reward 代理：fusion_valence 的 7 日滑窗平均 r_t；
- 模型：简单的 MLP reward regressor，学习 R(z_t) ≈ r_t；
- 用 MSE/MAE 衡量拟合质量，为后续更复杂的 IRL 方法预留接口。

使用脚本
--------

命令：

```bash
cd companion-robot-irl
python3 scripts/export_fusion_features.py --root data/isabella_irl_3d_clean
python3 scripts/export_temporal_features.py --root data/isabella_irl_3d_clean
python3 scripts/run_irl_mvp_isabella.py --root data/isabella_irl_3d_clean --epochs 300 --device cpu
```

主要结果
--------

终端输出（节选）：

```text
[INFO] Fusion computed for dataset data/isabella_irl_3d_clean
[INFO] Per-modality accuracies:
  - text    : 0.050
  - behavior: 0.867
  - emotion : 1.000
  - score   : 0.567
...
[INFO] Loaded encoder weights from data/isabella_irl_3d_clean/models/emotion_sequence_model.pt
[INFO] Saved temporal embeddings to data/isabella_irl_3d_clean/features/temporal_embeddings.npy
[INFO] Saved window-level rewards to data/isabella_irl_3d_clean/features/window_valence.npy
[INFO] Training reward regressor on data/isabella_irl_3d_clean
[INFO] input_dim=257, hidden_dim=64, use_mlp=True
[EPOCH 001] train MSE: 0.034521
[EPOCH 300] train MSE: 0.000215
[INFO] Saved reward regressor checkpoint to data/isabella_irl_3d_clean/models/irl_reward_regressor.pt
[INFO] Final MSE=0.000241, MAE=0.011613
```

数据与状态表示
--------------

- 输入状态嵌入：
  - 来源：`scripts/export_temporal_embeddings.py`；
  - 使用 `BiLSTMAttentionEncoder`（加载 `emotion_sequence_model.pt` 的 encoder 权重）；
  - 输入特征：`rolling_stats.npy` (54×5) + `global_baseline.npy` (1×1)；
  - 输出：`temporal_embeddings.npy`，形状 `(T=54, D=257)`。
    - 257 = 2×hidden_dim(128) + baseline_dim(1)。

- reward 代理：
  - 来源：`fusion_daily.npy` (60×4)；
  - 首先计算 daily `fusion_valence = P(积极) - P(消极)`；
  - 再用 7 日滑窗平均得到 `window_valence.npy`，形状 `(T=54,)`；
  - 作为每个窗口的 reward 目标 r_t。

IRL MVP 模型
-----------

- 定义见 `learning/irl_mvp.py`：
  - 数据集：`TemporalRewardDataset`（每个样本是 (z_t, r_t) 对）；
  - 模型：`RewardRegressor`（两层 MLP）：

    ```text
    z_t ∈ R^{257}
      -> Linear(257, 64) + ReLU
      -> Linear(64, 1) -> R(z_t)
    ```

  - Loss：MSE，训练使用 16 的 batch size 在 54 个窗口上做多轮扰动拟合。

拟合质量
--------

- 训练过程：
  - epoch 1：MSE ≈ 0.0345；
  - epoch 50–300：MSE 稳定下降到 ≈ 2e-4 量级。

- 全数据误差：
  - 最终 MSE ≈ 0.000241；
  - 最终 MAE ≈ 0.0116；
  - 在 reward 值约在 [-1, 1] 范围内的情况下，这表示回归器几乎可以精确重构每个窗口的 valence reward。

限制与解释
----------

- 此 IRL MVP 目前是“单轨迹 + 单 persona”的极小规模实验：
  - 参数数量远多于样本数（D=257, T=54），因此接近于“可逆映射”，难以评估泛化能力；
  - reward 代理本身是从 fusion 概率导出的 valence，而 fusion 又依赖规则化预测器，因此整体仍属 V0 原型。

- 然而从结构角度看，主线已打通：
  - 60 天多模态 → 多模态融合 → fusion_valence → 滑窗/衰减 → BiLSTM+注意力状态序列 → R(z_t) ≈ r_t；
  - reward 回归器的输出 R(z_t) 可视为 IRL 中“奖励函数参数化”的一个具体实现。

后续演化方向
------------

- 在此基础上，可以进一步接入：
  - 动作空间（例如 nightly 访谈策略 / 干预强度）；
  - 真正的 IRL 算法（如 MaxEnt IRL）在 (s_t, a_t, r_t) 上学习偏好；
  - 将 learned reward 参数与 BFI 前测/后测人格向量做相关性与可解释性分析。

结论
----

- 本次实验给出了一个结构完整的 IRL MVP：
  - 使用情绪时序嵌入作为状态；
  - 使用融合 valence 的窗口平均作为 reward 代理；
  - 成功拟合 R(z_t) ≈ r_t，并以 MSE/MAE 提供量化指标。
- 这为下一步引入动作维度和更完整的 IRL 框架（以及与人格对齐的分析）提供了可复用的架构基础。

