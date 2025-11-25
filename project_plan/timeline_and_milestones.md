排期与里程碑（假数据阶段）
===========================

假设 3 人都能稳定投入，每周以「1 个主要工作单元」计。  
以下以 3 周为参考节奏，可根据实际情况拉长/缩短。

Week 1：P1 – 数据接口 & 基础特征
--------------------------------

- Owner-Data（A）
  - 完成 `DatasetLoader`：
    - 支持 `data/isabella_irl_3d_clean` 与 `data/alice_irl_60d`；
    - 输出：日级文本、行为、情绪、得分 + 元信息。
  - 写一个简单 EDA 脚本，绘制 60 天的 mood 曲线、缺失率等。

- Owner-Sequence（B）
  - 与 A 对齐特征 schema；
  - 从 behaviors/emotions 中抽 5–10 个稳定的日级统计特征；
  - 产出一个 `X_daily (60×F)` + `y_daily` 的 Numpy/Pandas 版本。

- Owner-IRL（C）
  - 梳理 persona & BFI 工具（现有 `agents/bfi_validator.py` 和 `validation/*`）；  
  - 起草实验指标规范（r、MAE、可视化要求），固化在一个小文档里。

里程碑：

- 有一个可以在 Notebook / 脚本中直接调用的 `load_60d_dataset()`；
- 有一套 agreed 的日级特征定义与基本统计结果。

Week 2：P2 – 时间序列表示 & 人格回归
------------------------------------

- Owner-Data（A）
  - 实现 7 天滑动窗口特征（均值/方差/趋势）；
  - 实现指数衰减基线（基于 mood_scores 或部分特征）。

- Owner-Sequence（B）
  - 实现 BiLSTM/GRU + 注意力时间序列编码器：
    - 输入：日级序列或滑窗特征；
    - 输出：人格向量预测。
  - 完成一次基准实验（至少 Isabella 60 天），记录 r/MAE。

- Owner-IRL（C）
  - 封装训练/评估脚本：
    - 统一命令行入口，如 `scripts/train_personality_regressor.py`；
    - 输出日志与基本图表。

里程碑：

- 一条「60 天轨迹 → Big Five 预测」的可运行 pipeline；
- 至少知道：在假数据上，baseline 能达到怎样的相关性水平。

Week 3：P3 – IRL / 偏好建模 MVP
------------------------------

- Owner-Data（A）
  - 与 B/C 一起定义动作空间的编码方式；
  - 将序列/状态 + 动作 + reward proxy 打包为「专家轨迹」格式。

- Owner-Sequence（B）
  - 如时间允许，对序列编码器做小规模改进：
    - 加入 phase embedding（来自 `story/isabella_story.py`）；
    - 或者 day index embedding，观察指标变化。

- Owner-IRL（C）
  - 在 `learning/` 下实现 IRL MVP：
    - 可先实现类似 `online_irl` 的简化版；
    - 明确输入/输出接口；
  - 设计并跑通一组 IRL 实验：
    - 产出偏好参数；
    - 与 Big Five / BFI 做对齐分析。

里程碑：

- 有一个可运行的 IRL 原型（哪怕简化），可以讲清楚「状态 → 奖励/偏好 → 人格对齐」的故事；
- 有一份简短实验报告描述结果与下一步改进方向。

后续（真数据阶段）的挂钩点
--------------------------

- A：将真实 Town / 真人数据适配到同样的 `DatasetLoader` 接口；
- B：在不动模型结构的前提下，基于新数据重训/微调，并对比假数据结果；
- C：重新跑 IRL 实验，对比「假数据 vs 真数据」在偏好参数、人格对齐指标上的差异。

