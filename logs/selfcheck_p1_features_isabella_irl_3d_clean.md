P1 自查：isabella_irl_3d_clean 日级特征导出
==========================================

检查时间：当前开发会话。  
数据根目录：`data/isabella_irl_3d_clean`  
特征输出目录：`data/isabella_irl_3d_clean/features`

使用脚本
--------

命令：

```bash
cd companion-robot-irl
python3 scripts/export_daily_features.py --root data/isabella_irl_3d_clean
```

主要结果
--------

- 数据集加载：
  - 成功通过 `DailyDataset` 加载 day 1–60；
  - 结构/文件检查同 `selfcheck_p1_eda_isabella_irl_3d_clean.md`。

- 特征导出：
  - 输出文件：
    - `X_daily.npy`：形状 `(60, 14)`；
    - `y_mood_scores.npy`：形状 `(60,)`；
    - `X_daily.csv`：用于快速人工检查；
    - `feature_names.json`：14 维特征名称；
    - `days.json`：按顺序列出的 day 索引（1–60）。
  - 脚本终端输出：

    ```text
    [INFO] Exported features for dataset data/isabella_irl_3d_clean
      - days: 1..60 (total 60)
      - X shape: (60, 14)
      - y shape: (60,)
      - out dir: data/isabella_irl_3d_clean/features
    ```

- 当前特征列表（feature_names.json）：

  ```json
  [
    "beh_creative_count",
    "beh_rest_count",
    "beh_social_count",
    "beh_work_count",
    "emotion_other",
    "emotion_中性",
    "emotion_复杂",
    "emotion_消极",
    "emotion_积极",
    "len_chars",
    "mood_score",
    "neg_word_count",
    "num_behaviors",
    "pos_word_count"
  ]
  ```

  含义：

  - `beh_*_count`：
    - 从 `behaviors/day_XXX.json` 的 description 中基于关键词进行粗分类；
    - `beh_social_count`：包含社交相关行为的条目数；
    - `beh_work_count`：包含工作/任务相关行为的条目数；
    - `beh_rest_count`：包含休息/自我照顾相关行为的条目数；
    - `beh_creative_count`：包含创作/艺术/爱好相关行为的条目数。
  - `emotion_*`：
    - 来自 `emotions/day_XXX.json` 的 label；
    - 显式编码：积极 / 消极 / 中性 / 复杂 / other。
  - 文本长度与词频：
    - `len_chars`：每日对话 Markdown 的字符数；
    - `pos_word_count`：对话中出现积极情绪词的次数；
    - `neg_word_count`：对话中出现负向情绪/压力类词的次数。
  - 其他：
    - `num_behaviors`：behaviors 列表长度；
    - `mood_score`：来自 `scores/mood_scores.csv` 的日级心情得分。

质量评估与《12》对齐
--------------------

- 多模态覆盖：
  - 文本：通过 `len_chars`、正负情绪关键词计数；
  - 行为：通过 `num_behaviors` 和四类行为计数（social / work / rest / creative）；
  - 情绪标签：通过 one-hot `emotion_*`；
  - 心情分数：`mood_score`；后续 P2 中会同时作为标签和结合时序的状态属性。
  - 与《12》中“60 天 × 4 维（文本/动作/表情/分数）→ 结构化特征表”的要求一致，当前为规则化 V0 实现。

- 工业化稳健性：
  - 如某天行为描述缺少关键字，则对应行为类别计数为 0，不影响整体 pipeline；
  - 文本缺失/为空时，长度和词频均为 0；
  - 所有特征为实数标量，适合后续滑窗与神经网络输入。

结论
----

- `data/isabella_irl_3d_clean/features` 中的导出结果满足当前 P1 阶段的结构化、多模态特征要求；
- 特征维度和含义清晰，与《12》规划的 P1 数据形态一致；
- 后续 P2 将在此基础上实现滑动窗口统计、指数衰减与时间序列人格回归模型。

