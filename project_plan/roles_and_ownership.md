角色与职责（3 人分工）
======================

说明：这里默认有 3 位核心成员，可按实际情况映射到具体人名。  
记号：A = Owner-Data，B = Owner-Sequence，C = Owner-IRL。

Owner-Data（A）：数据与特征管线
------------------------------

核心职责：

- 负责「60 天假数据」与未来真数据的**数据入口与清洗**：
  - 设计并实现统一的 `DatasetLoader`；
  - 管理 `data/*` 目录的结构与约定（conversations/behaviors/emotions/scores）。
- 抽取稳定、可解释的**日级基础特征**：
  - 文本特征（TF-IDF / 简单 embedding）；
  - 行为统计特征（社交频次、活动类别等）；
  - 情绪标签编码（one-hot 等）。
- 输出给下游的一致接口：
  - `X_daily`：形如 `(num_days, feature_dim)` 的数组；
  - `y_daily`：如 `mood_scores` 或其他标签；
  - 标注好 persona、实验 id 等元信息。

主要交付物：

- `utils/dataset_loader.py`（或同等模块）；
- `features/basic_features.py`（或等价脚本）；
- 简单的 EDA / 自查脚本（缺失率、分布图等）。

Owner-Sequence（B）：时间序列与情绪模型
--------------------------------------

核心职责：

- 在 A 提供的日级特征基础上：
  - 实现滑动窗口统计（7/14 天）与指数衰减基线；
  - 设计并训练 BiLSTM/GRU + 注意力等**时间序列编码器**。
- 建立**人格回归/分类 baseline**：
  - 输入：60 天序列（或滑窗特征）；
  - 输出：预设 Big Five / BFI 得分的回归预测；
  - 指标：Pearson r、MAE、可视化（预测 vs 真实）。
- 与 C 协作，为 IRL 提供「状态表示」：
  - 明确 `state_t` 包含哪些分量，以及维度/归一化方式；
  - 固化为一个独立模块，供 IRL 重用。

主要交付物：

- `models/sequence_encoder.py`（或等价实现）；
- `scripts/train_personality_regressor.py`；
- 实验记录（日志 / 简单图表），便于后续写论文和汇报。

Owner-IRL（C）：IRL 与评价体系
------------------------------

核心职责：

- 设计适配当前数据的**动作空间与奖励 proxy**：
  - 动作可以是提问风格、干预强度等离散选择；
  - 奖励可用 `mood_scores` 或情绪 label 代理。
- 在 B 的状态表示基础上，实现 IRL / 逆回归 MVP：
  - 可以先参考 `simulated_legacy/learning/online_irl.py` 的结构；
  - 再迭代到更标准的 MaxEnt IRL 等。
- 建立人格 / 偏好对齐的评价体系：
  - 定义从 IRL 参数 → 人格/偏好向量的映射；
  - 与 Big Five / BFI 报告做定量对比；
  - 输出实验报告（表格 + 结论）。

主要交付物：

- `learning/irl_mvp.py`（或等价模块）；
- `scripts/run_irl_experiment.py`（统一入口）；
- 若干 IRL 实验的结果与总结文档。

协作边界与接口
--------------

- A → B：
  - 提供「日级特征 + 标签」的数据接口（含规范文档）；
  - 保证后续真数据接入时，接口保持兼容。

- B → C：
  - 定义并实现 `encode_sequence()` / `get_state_embeddings()` 等接口；
  - 在接口设计上考虑 IRL 的需求（例如是否需要行动历史、是否需要 phase embedding）。

- C → A/B：
  - 反馈哪些特征 / 表达在 IRL 学习中最有用，以反向指导特征与模型设计；
  - 提出对数据采集与标注的改进建议，为未来的真数据阶段做准备。

