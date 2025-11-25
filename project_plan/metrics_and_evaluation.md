指标与评估规范
==============

目标
----

为后续所有基于 60 天 IRL 数据的实验提供统一的指标与自查口径，确保：

- 各阶段结果可以纵向对比（假数据 → 真数据）；
- 不同人实现的模型可以横向对比（baseline → 改进版）；
- 和《进度/12》中提出的人格/情绪相关性要求对齐。

数据质量自查（P1 层面）
-----------------------

使用脚本：`scripts/eda_60d_dataset.py`。

必查项：

- 目录结构完整：
  - `conversations/`, `behaviors/`, `emotions/`, `scores/mood_scores.csv` 均存在；
- `mood_scores.csv`：
  - header 必须为 `day,score`；
  - day 列可重复（表示多次试跑），系统默认采用**最后一次出现**的得分；
  - 验证时如存在重复，会提示：
    - `Duplicate mood_scores entries for days: [...] (using the last occurrence for each)`；
  - day 范围应为连续整数，例如 1–60。
- 每日文件：
  - `conversations/day_XXX.md`：存在、header 格式正确、turns>0；
  - `behaviors/day_XXX.json`：存在，包含非空 `behaviors` 列表；
  - `emotions/day_XXX.json`：存在，至少包含 `label` 字段；
  - `scores/mood_scores.csv` 中的每个 day 都必须有对应三类文件。

若出现：

- 缺文件 / JSON 解析失败 / day 不匹配 → 视为 **阻塞问题**，需要先修复数据；
- 仅有少量 `warnings`（例如重复 score）→ 可以在记录清楚的前提下继续实验，但需要在报告中注明。

情绪/心情标签指标（P1/P2 共用）
------------------------------

用于衡量模型从 60 天轨迹中提取情绪/心情模式的质量。

基础指标：

- `mood_score` 分布：
  - min / max / mean / std；
  - 必须避免长期被极端值（全 7 分等）主导；
- 时间维度：
  - 滑动平均或中位数曲线是否符合故事阶段预期（参考 `story/isabella_story.py` 中的 PHASES）。

后续可选指标（待 P2 实现）：

- 用模型预测的日级情绪标签 / 心情分数，与 `mood_scores.csv` 的相关性：
  - Pearson r；
  - MAE / RMSE。

人格对齐指标（P2/P3 层面）
-------------------------

来源：

- 预设 Big Five 参数：`data/personas/preset_personality.json`；
- BFI-44 报告：`validation/*.json`（由 `agents/bfi_validator.py` 生成）。

要求：

- 任何「轨迹 → 人格」的模型，都应至少报告：
  - 预测向量 vs 预设人格向量之间的 Pearson r；
  - 平均绝对误差（MAE）；
  - 各维度误差（O/C/E/A/N）；
  - 是否满足《12》中类似 r>0.75 的约束（视具体实验而定）。

参考实现：

- 可直接复用 `BFI44Validator.validate_against_preset` 的输出结构：
  - `pearson_r`, `mae`, `dimension_errors` 等字段；
  - 在新的实验脚本中使用相同字段名，便于自动汇总。

IRL / 偏好建模指标（P3 层面）
----------------------------

当进入 IRL 实验时，需同时关注：

- IRL 参数本身的稳定性与可解释性：
  - 不同随机初始、不同子样本上的方差；
  - 与动作/状态特征的相关性。
- IRL 推导出的偏好向量，与人格/情绪特征的关系：
  - 例如，将 IRL 参数映射到 Big Five 空间后，报告与预设/问卷人格的相关系数；
  - 分析哪些偏好维度与哪类人格特质最强相关。

报告要求
--------

任何阶段性的「结果」在提交前，应至少包含：

- 数据版本：
  - 使用了哪个数据集根目录（例如 `data/isabella_irl_3d_clean`）；
  - 是否存在重复 day、缺失文件等现象。
- 模型与特征版本：
  - 使用了哪些特征（可简单列出 feature_names）；
  - 序列编码器或 IRL 模型的关键超参数。
- 指标：
  - 按上述规范给出必要指标；
  - 若未达到《12》期望的阈值（例如 r>0.75），需附上简短分析与下一步优化方向。

