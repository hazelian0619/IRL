P2 自查：isabella_irl_3d_clean 多模态情绪融合（Fusion）
====================================================

检查时间：当前开发会话。  
数据根目录：`data/isabella_irl_3d_clean`  
特征输出目录：`data/isabella_irl_3d_clean/features`

目标
----

按《12》中的 Step 4 要求，补齐“多模态情绪融合”这一层结构：

- 为文本 / 行为 / 表情 / 分数四个模态分别给出日级情绪预测；
- 以 `emotions/day_xxx.json['label']` 作为初始 GT；
- 计算每个模态的预测精度，得到权重向量 w；
- 做 late fusion，生成 60 天的融合情绪概率序列 `fusion_daily.npy`；
- 后续滑窗与 LSTM 不再直接用 `mood_score`，而是用从融合概率导出的 valence。

使用脚本
--------

命令：

```bash
cd companion-robot-irl
python3 scripts/export_fusion_features.py --root data/isabella_irl_3d_clean
python3 scripts/export_temporal_features.py --root data/isabella_irl_3d_clean
```

主要结果
--------

终端输出（fusion）：

```text
[INFO] Fusion computed for dataset data/isabella_irl_3d_clean
[INFO] Days: 1..60 (total 60)
[INFO] Per-modality accuracies:
  - text    : 0.050
  - behavior: 0.867
  - emotion : 1.000
  - score   : 0.567
[INFO] Fusion weights (softmax over accuracies):
  - text    : 0.133
  - behavior: 0.301
  - emotion : 0.344
  - score   : 0.223
```

输出文件（features 目录）：

- `text_probs.npy`      - 文本模态预测的情绪概率 (60×4)；
- `behavior_probs.npy`  - 行为模态预测的情绪概率 (60×4)；
- `emotion_probs.npy`   - 情绪 JSON 模态预测的情绪概率 (60×4)；
- `score_probs.npy`     - 分数模态预测的情绪概率 (60×4)；
- `fusion_daily.npy`    - 融合后情绪概率 (60×4)；
- `fusion_meta.json`    - days / canonical_labels / accuracies / weights；
- `rolling_stats.npy`   - 已重新基于 fusion_valence 导出 (54×5)；
- `global_baseline.npy` - 基于 fusion_valence 的 EWMA 基线；
- `temporal_meta.json`  - 记录 `source_series: "fusion_valence"`。

融合定义与规则（V0）
--------------------

canonical label 空间：

```text
["积极", "消极", "中性", "复杂"]
```

GT：
- 来自 `emotions/day_xxx.json['label']`，经 `canonical_label()` 统一映射。

四个模态的规则化预测：

- 文本模态 (`text_probs`)：
  - 使用 `features/basic_features.py` 中的 `pos_word_count` / `neg_word_count`；
  - pos > neg → “积极”；neg > pos → “消极”；全 0 → “中性”；其他 → “复杂”。

- 行为模态 (`behavior_probs`)：
  - 使用 `beh_social_count`、`beh_work_count`、`beh_rest_count` 等；
  - 高 social + 有 rest + mood_score≥6 → “积极”；
  - work≥2 且 rest=0 且 mood_score≤4 → “消极”；
  - rest≥1 且 social=0 且 work=0 → “中性”；
  - 其他 → “复杂”。

- 情绪 JSON 模态 (`emotion_probs`)：
  - 直接用 `emotions/day_xxx.json['label']` 映射到 canonical label。

- 分数模态 (`score_probs`)：
  - mood_score≤3 → “消极”；≥7 → “积极”；中间 → “复杂”。

精度与权重：

- Per-modality accuracy（与 GT 的 argmax 一致率）：
  - text ≈ 0.05
  - behavior ≈ 0.867
  - emotion = 1.0
  - score ≈ 0.567
- 权重通过 softmax(acc) 得到（见 fusion_meta.json）：
  - text    ≈ 0.133
  - behavior≈ 0.301
  - emotion ≈ 0.344
  - score   ≈ 0.223

融合概率 (`fusion_daily.npy`) 定义为：

```text
P_fusion = w_text * P_text + w_beh * P_beh + w_emotion * P_emotion + w_score * P_score
```

与滑窗/时序模型的对接
----------------------

- 在更新后的 `features/temporal_features.py` 中：
  - 若 `<root>/features/fusion_daily.npy` 存在，则：
    - 先计算 `valence = P(积极) - P(消极)`；
    - 用该 1D valence 序列执行 7 天滑窗统计和 EWMA；
    - `temporal_meta.json` 里的 `source_series` 字段标记为 `"fusion_valence"`。
  - 若不存在 fusion，则回退到 `mood_score`。

当前结果：

- 对于 `data/isabella_irl_3d_clean`：
  - `temporal_meta.json` 中已经显示：

    ```json
    "source_series": "fusion_valence"
    ```

  - 即后续 LSTM 编码使用的是“多模态融合后的情绪倾向(积极-消极)”而非单一 mood_score。

与《12》的对齐情况
------------------

- 结构层面：
  - 单模态预测 → 相关性度量 → softmax 权重 → late fusion 概率；
  - 再从融合概率中提取 1D valence 序列驱动滑窗与指数衰减；
  - 与《12》中“多模态 → 权重回流 → 融合情感序列 → 滑窗统计”的结构一致。

- 实现层面（V0）：
  - 当前单模态预测器使用规则化启发式，重点在于先打通管线；
  - 将来可以在 `features/emotion_fusion.py` 内部替换为更强的文本/行为模型，而不改变对外接口。

结论
----

- `data/isabella_irl_3d_clean` 上，多模态情绪融合模块已经按主线结构落地：
  - 4 个模态的日级预测；
  - 权重学习与 fusion 概率；
  - 融合 valence 序列驱动的滑窗与时序特征。
- 这为后续基于融合情绪序列训练时序情绪模型以及 IRL / 人格对齐提供了结构完备的输入层。

