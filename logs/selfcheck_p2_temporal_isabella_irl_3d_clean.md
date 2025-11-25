P2 自查：isabella_irl_3d_clean 滑动窗口与指数衰减特征
===================================================

检查时间：当前开发会话。  
数据根目录：`data/isabella_irl_3d_clean`  
特征输出目录：`data/isabella_irl_3d_clean/features`

使用脚本
--------

命令：

```bash
cd companion-robot-irl
python3 scripts/export_temporal_features.py --root data/isabella_irl_3d_clean
```

主要结果
--------

- 输出文件（追加到 features 目录）：
  - `rolling_stats.npy`：形状 `(54, 5)`；
  - `global_baseline.npy`：形状 `(1,)`；
  - `decay_weights.npy`：形状 `(60,)`；
  - `temporal_meta.json`：配置与 shape 元信息。

- 终端检查（简化版）：

  ```text
  rolling_stats shape: (54, 5)
  baseline: [3.9955297]
  weights shape: (60,) sum= 1.0
  meta.sliding_window: {'window': 7, 'step': 1, 'num_windows': 54, 'num_stats': 5}
  ```

  - 54 窗口：源于 60 天、窗口 7、步长 1：`60 - 7 + 1 = 54`；
  - 5 个统计量：均值、方差、最大、最小、线性趋势（斜率）；
  - `decay_weights` 为 60 维，并已归一化（sum=1.0）。

实现细节与《12》对齐
--------------------

1. 滑动窗口统计（rolling_stats.npy）

- 序列选择：
  - 当前版本使用 `mood_scores` 作为 60 天融合情绪的 1D 序列；
  - 未来可扩展为对多维概率序列做同样处理。

- 配置：
  - 窗口 `W=7`，步长 `S=1`；
  - 每个窗口内计算：
    - `mean`：7 天平均情绪得分；
    - `var`：方差，刻画波动度（高方差 → 更不稳定，接近高 N 的行为特点）；
    - `max`：窗口内的峰值；
    - `min`：窗口内的谷值；
    - `trend`：通过线性回归斜率衡量情绪随时间的上升/下降趋势（上升趋势可与高外向性正反馈相关）。

- 输出：
  - `rolling_stats.npy` 为 `(54, 5)` 矩阵，对应《12》中的“(54×F)统计矩阵（例如F=5）”；
  - 元信息记录在 `temporal_meta.json` 的 `sliding_window` 字段中。

2. 指数衰减基线（global_baseline.npy + decay_weights.npy）

- 配置：
  - `lambda_ = 0.05`，与《12》中“超参数 λ=0.05”一致；
  - 权重定义：

    ```text
    weights_t = λ * (1 - λ)^(T-1-t),  t=0..T-1
    ```

  - 并对所有 t 做归一化（sum=1.0）。

- 直觉：
  - 越靠近 Day 60 的数据权重越高；
  - Day 1–10 等冷启动阶段的噪声影响被自然削弱；
  - 支持“用户价值观可能随时间微调”的假设。

- 输出：
  - `global_baseline.npy`：形状 `(1,)` 的标量基线，代表 60 天情绪的指数加权平均；
  - `decay_weights.npy`：形状 `(60,)` 的权重序列，可作为后续 IRL 或可视化使用；
  - 元信息记录在 `temporal_meta.json` 的 `decay` 部分。

质量评估
--------

- 数值稳定性：
  - `decay_weights` 明确归一化，sum=1.0；
  - 滑窗统计中时间索引的方差 `t_var` 显式检查，避免除零问题；
  - 所有输出均为 `float32`，便于与下游 PyTorch 等框架对接。

- 与《12》的匹配程度：
  - 已经严格实现：
    - 7 天滑动窗口、步长 1；
    - F=5 个窗口内统计量（mean/var/max/min/trend）；
    - 60 天 EWMA/global baseline + 权重序列；
  - 输出路径与命名也与文档对应：
    - `/features/rolling_stats.npy`；
    - `/features/global_baseline.npy`；
    - `/features/decay_weights.npy`。

结论
----

- 在 `data/isabella_irl_3d_clean` 上，P2 所需的“短周期统计 + 长期衰减基线”特征已经按《12》规范生成；
- 这些特征现在可以直接作为后续 BiLSTM/GRU+注意力编码器的输入，以及 IRL 阶段的状态基础；
- 下一步将基于这些 temporal 特征实现时间序列人格回归的 baseline 模型。

