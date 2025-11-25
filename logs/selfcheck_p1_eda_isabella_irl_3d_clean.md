P1 自查：isabella_irl_3d_clean 数据集 EDA
========================================

检查时间：自动生成于当前开发会话。  
数据根目录：`data/isabella_irl_3d_clean`

使用脚本
--------

命令：

```bash
cd companion-robot-irl
python3 scripts/eda_60d_dataset.py --root data/isabella_irl_3d_clean
```

主要结果
--------

- 成功加载数据集：
  - Days: 1–60（连续）
- 验证结果：
  - 出现一条 `warnings`：
    - `Duplicate mood_scores entries for days: [4, 5, 6, 7, 8, 9, 10] (using the last occurrence for each)`
  - 含义：`scores/mood_scores.csv` 中，day 4–10 出现了多条记录（多次试跑或补写），
    `DailyDataset` 当前策略为**保留每个 day 的最后一次记录**，并在 `validate()` 中显式给出告警。
- 基础特征矩阵：
  - shape: `(60, 10)`
  - feature names:
    - `emotion_other`
    - `emotion_中性`
    - `emotion_复杂`
    - `emotion_消极`
    - `emotion_积极`
    - `len_chars`
    - `mood_score`
    - `neg_word_count`
    - `num_behaviors`
    - `pos_word_count`
- mood_score 分布：
  - min: 1.0
  - max: 8.0
  - mean: 4.0
  - std : 1.835

质量评估与后续动作
------------------

- 重复的 day 记录：
  - 从工业标准来看，允许多次试跑是合理的，但必须保证下游使用的是**明确定义的版本**；
  - 当前实现中，`DailyDataset`：
    - 在 `_load_scores()` 阶段保留每个 day 的最后一条记录；
    - 在 `validate()` 时记录所有出现重复的 day，并通过 warnings 提醒。
  - 后续建议：
    - 如需要“只保留某一版实验”的严格数据集，可在清洗阶段输出一个去重后的 `mood_scores_clean.csv`；
    - 在真实数据阶段，尽量通过实验版本号 / 时间戳区分，不在同一文件中混合多次试跑结果。

- 当前基础特征仅涵盖文本长度、行为条数、情绪关键词和情绪标签：
  - 作为 P1 的 baseline 是可接受的；
  - P2 之后需要补充更细的行为模式特征（例如社交/工作/休息比例等），并引入更强的文本表示（embedding）。

结论
----

- `data/isabella_irl_3d_clean` 可以作为当前 60 天假数据阶段的主数据源；
- 已知的数据质量问题只有：day 4–10 的 `mood_scores` 存在重复记录，系统已采用“保留最后值 + 告警”的策略处理；
- 后续在报告中必须注明这一点，保证实验可复现与可解释。

