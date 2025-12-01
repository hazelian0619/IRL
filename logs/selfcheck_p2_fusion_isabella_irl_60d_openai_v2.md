P2 自查：isabella_irl_60d_openai_v2 多模态情绪融合（Fusion）
===========================================================

检查时间：当前开发会话。  
数据根目录：`data/isabella_irl_60d_openai_v2`  
特征输出目录：`data/isabella_irl_60d_openai_v2/features`

一、目标与角色
--------------

本报告严密梳理在 `isabella_irl_60d_openai_v2` 数据集上，多模态情绪融合模块的设计与实现，
用于后续 IRL / 人格对齐阶段的自查与追踪。

目标（对应《12》的 Step 4）：

1. 对每天的多模态信号（文本 / 行为 / 情绪 JSON / mood_score），构造 4 个日级情绪预测器：  
   `P_text(day), P_behavior(day), P_emotion(day), P_score(day)` ∈ R^4（四类：积极/消极/中性/复杂）。
2. 使用 `emotions/day_xxx.json['label']` 的 canonical 版本作为初始 GT，评估每个模态在 60 天上的“预测准确率”。
3. 通过 softmax(acc) 从 4 个准确率中导出融合权重向量 `w = (w_text, w_behavior, w_emotion, w_score)`。
4. 对每一天做 late fusion：

   > `P_fusion(day) = Σ_mod w_mod * P_mod(day)`  
   > 得到统一的日级情绪概率分布 `fusion_daily.npy`（形状 60×4）。

5. 从 `P_fusion(day)` 中抽取一维 valence 序列：

   > `valence(day) = P_fusion(积极) - P_fusion(消极)`  
   > 作为后续时序特征（rolling_stats / global_baseline）与 IRL reward 代理的唯一情绪输入。

简而言之：

> 每天所有模态各投一次票，我们用 60 天历史评估“谁更靠谱”，  
> 再按可靠度给每个模态分配话语权，把 4 票加权平均成一条统一的情绪概率。

二、使用脚本与主要结果
----------------------

运行命令：

```bash
cd companion-robot-irl
python3 scripts/export_fusion_features.py --root data/isabella_irl_60d_openai_v2
```

终端输出（节选）：

```text
[INFO] Fusion computed for dataset data/isabella_irl_60d_openai_v2
[INFO] Days: 1..60 (total 60)
[INFO] Per-modality accuracies:
  - text    : 0.567
  - behavior: 0.383
  - emotion : 1.000
  - score   : 0.767
[INFO] Fusion weights (softmax over accuracies):
  - text    : 0.218
  - behavior: 0.181
  - emotion : 0.336
  - score   : 0.266
```

输出文件（位于 `features` 目录）：

- `text_probs.npy`      – 文本模态预测的情绪概率矩阵 `(60, 4)`；
- `behavior_probs.npy`  – 行为模态预测的情绪概率矩阵 `(60, 4)`；
- `emotion_probs.npy`   – 情绪 JSON 模态预测的情绪概率矩阵 `(60, 4)`；
- `score_probs.npy`     – 分数模态预测的情绪概率矩阵 `(60, 4)`；
- `fusion_daily.npy`    – 融合后情绪概率矩阵 `(60, 4)`；
- `fusion_meta.json`    – 记录 `days`、`canonical_labels`、`accuracies`、`weights`；
- `rolling_stats.npy`   – 后续由 `export_temporal_features.py` 基于 fusion_valence 导出 `(54, 5)`；
- `global_baseline.npy` – 基于 fusion_valence 的 EWMA 基线；
- `temporal_meta.json`  – 标记 `source_series: "fusion_valence"`。

三、数据结构与 canonical label 空间
---------------------------------

1. 日级多模态数据结构（`DailyRecord`，见 `utils/dataset_loader.py`）：

   | 字段           | 来源                                   | 含义                       |
   |----------------|----------------------------------------|----------------------------|
   | `day`          | mood_scores.csv                        | 第几天（1..60）           |
   | `transcript_md`| `conversations/day_xxx.md`             | 夜聊全文                   |
   | `behaviors`    | `behaviors/day_xxx.json["behaviors"]`  | 当天行为片段列表          |
   | `emotion`      | `emotions/day_xxx.json`                | 日级情绪标签/强度/原因    |
   | `mood_score`   | `scores/mood_scores.csv`               | 1–10 日级心情打分         |

2. canonical label 空间（`features/emotion_fusion.py`）：

   ```python
   CANONICAL_LABELS = ("积极", "消极", "中性", "复杂")
   _LABEL_INDEX = {lbl: i for i, lbl in enumerate(CANONICAL_LABELS)}
   # → {"积极":0, "消极":1, "中性":2, "复杂":3}
   ```

3. GT（ground truth）：

   - 对每天从 `emotion["label"]` 通过 `canonical_label(raw)` 映射到四类之一；
   - 构成长度为 60 的序列 `gt_labels`，例如：

     ```text
     gt_labels = ["积极", "积极", "复杂", "消极", ...]   # 共 60 项
     ```

   - 后续计算每个模态的准确率时，都以这个 `gt_labels` 为参照。

四、日级特征抽取：feats(day)
---------------------------

`features/basic_features.py` 中的 `extract_daily_features(record)` 将一个 `DailyRecord` 转为小而可解释的特征字典 `feats(day)`，主要特征如下：

| 特征名              | 来源             | 含义（示意）                                      |
|---------------------|------------------|---------------------------------------------------|
| `len_chars`         | 对话文本         | 对话字符数，粗略代表“今天说了多少话”             |
| `pos_word_count`    | 对话文本         | 正向情绪词计数（开心/满足/兴奋/期待/温暖…）      |
| `neg_word_count`    | 对话文本         | 负向情绪词计数（累/疲惫/压力/焦虑/失落…）        |
| `beh_social_count`  | behaviors 文本   | 今天是否有社交行为（朋友/顾客/聊天/社区…）       |
| `beh_work_count`    | behaviors 文本   | 是否有工作/任务行为（准备/布置/忙碌…）           |
| `beh_rest_count`    | behaviors 文本   | 是否有休息/放松行为（休息/散步/放空…）           |
| `beh_creative_count`| behaviors 文本   | 是否有创作/艺术行为（画画/展览/写作…）           |
| `emotion_积极/消极/中性/复杂` | emotions label | 当日情绪标签的 one-hot                           |
| `emotion_other`     | emotions label   | 其他稀有标签占位                                 |
| `mood_score`        | mood_scores      | 当日 1–10 心情打分                               |
| `num_behaviors`     | behaviors        | 当天记录的行为片段数                             |

这一层的作用：

- 将“文本 + 行为 + 标签 + 分数”统一投影到数值特征空间，供所有单模态预测器使用；
- 保持简单可解释，便于审查和未来替换更复杂模型时对齐接口。

五、四个单模态情绪预测器：P_mod(day)
-----------------------------------

在 `features/emotion_fusion.py` 中，实现了 4 个启发式 V0 预测器，每个给出一天的 4 维情绪概率（当前为 one-hot）。

1. 文本模态：`_text_probs(day_features)`

   - 输入：`feats(day)` 中的 `pos_word_count`, `neg_word_count`；
   - 规则：

     | 条件                      | 预测类别 |
     |---------------------------|----------|
     | pos == 0 且 neg == 0     | 中性     |
     | pos > neg                | 积极     |
     | neg > pos                | 消极     |
     | pos == neg 且 > 0        | 复杂     |

   - 输出：`label_to_onehot(类别)` → `P_text(day) ∈ {0,1}^4`。

2. 行为模态：`_behavior_probs(day_features)`

   - 输入：`beh_social_count`, `beh_work_count`, `beh_rest_count`, `mood_score`；
   - 规则：

     | 条件                                           | 预测类别 |
     |------------------------------------------------|----------|
     | social≥1 且 rest≥1 且 mood_score≥6             | 积极     |
     | work≥2 且 rest==0 且 mood_score≤4             | 消极     |
     | rest≥1 且 social==0 且 work==0                 | 中性     |
     | 其他                                            | 复杂     |

   - 输出：`P_behavior(day) ∈ {0,1}^4`。

3. 情绪 JSON 模态：`_emotion_probs(raw_label)`

   - 输入：原始 `emotions/day_xxx.json["label"]`；
   - 规则：`canonical_label(raw_label)` → one-hot；
   - 输出：`P_emotion(day) ∈ {0,1}^4`，即“LLM 给的日级情绪标签本身”视角。

4. 分数模态：`_score_probs(mood_score)`

   - 输入：`mood_score ∈ [1,10]`；
   - 规则：

     | 条件              | 预测类别 |
     |-------------------|----------|
     | mood_score ≤ 3    | 消极     |
     | mood_score ≥ 7    | 积极     |
     | 其余（4–6）       | 复杂     |

   - 输出：`P_score(day) ∈ {0,1}^4`。

这 4 个预测器分别从“文本用词”“行为结构”“情绪标签”“自评分”四个视角粗略判断当天情绪，为后续“谁更靠谱”的评估提供基础。

六、用 GT 评估每个模态的准确率 acc_mod
--------------------------------------

### 6.1 收集 60 天的 GT 与预测矩阵

在 `compute_fusion` 的循环中：

```python
gt = canonical_label(rec.emotion.get("label", ""))
gt_labels.append(gt)

probs_text.append(_text_probs(feats))
probs_beh.append(_behavior_probs(feats))
probs_em.append(_emotion_probs(rec.emotion.get("label", "")))
probs_score.append(_score_probs(rec.mood_score))
```

循环结束后：

- `gt_labels`：长度为 60 的列表，每一项是 `"积极/消极/中性/复杂"`；
- `probs_text/beh/emotion/score`：分别堆叠成 `(T=60, K=4)` 的矩阵：

  ```python
  probs_text = np.stack(probs_text, axis=0)    # 形状 (60, 4)
  ...
  ```

其中第 `t` 行 `probs_text[t]` 对应 day_t 的 4 类概率分布。

### 6.2 将 GT 转成类别编号 `gt_idx`

```python
gt_idx = np.array([_LABEL_INDEX[g] for g in gt_labels], dtype=np.int64)
```

例如（示意）：

| day 索引 t | gt_labels[t] | `_LABEL_INDEX[...]` | gt_idx[t] |
|------------|--------------|---------------------|-----------|
| 0          | 积极         | 0                   | 0         |
| 1          | 复杂         | 3                   | 3         |
| 2          | 消极         | 1                   | 1         |
| ...        | ...          | ...                 | ...       |

得到长度为 60 的整数数组 `gt_idx`，每个元素 ∈ {0,1,2,3}。

### 6.3 `acc(probs)` 的实现与含义

定义：

```python
def acc(probs: np.ndarray) -> float:
    preds = probs.argmax(axis=1)
    gt_idx = np.array([_LABEL_INDEX[g] for g in gt_labels], dtype=np.int64)
    return float((preds == gt_idx).mean())
```

对给定模态的预测矩阵 `probs ∈ R^{60×4}`，该函数完成三步：

1. `preds = probs.argmax(axis=1)`  
   - 对每一行（对应一天）取 4 维中的最大值索引；  
   - 得到长度为 60 的预测类别编号数组 `preds[t] ∈ {0,1,2,3}`；
2. 将 `gt_labels` 映射为 `gt_idx`（如上）；
3. `(preds == gt_idx)` 得到布尔数组，`True` 代表该日预测正确；  
   `.mean()` 将 `True/False` 当作 1/0 求平均，即：

   > `accuracy = (预测正确的天数) / 60`

这就是“分类器准确率”的标准定义：在所有样本上，预测类别与真实类别相等的比例。

### 6.4 在 v2 数据集上的准确率结果

对 4 个模态分别调用 `acc(...)`：

```python
acc_text = acc(probs_text)
acc_beh = acc(probs_beh)
acc_em = acc(probs_em)
acc_score = acc(probs_score)
```

得到：

| 模态      | 准确率 acc（与 GT 的一致率） |
|-----------|------------------------------|
| text      | 0.567                        |
| behavior  | 0.383                        |
| emotion   | 1.000                        |
| score     | 0.767                        |

解释：

- 在 `isabella_irl_60d_openai_v2` 上：
  - 情绪标签模态（emotions）在定义上与 GT 完全一致，因此 accuracy = 1.0；
  - mood_score 模态有约 77% 的天与情绪标签一致；
  - 文本关键词和行为规则较粗糙，准确率分别约为 57% 和 38%；
- 这四个准确率成为回头评估“谁更可靠”的依据。

七、从准确率到融合权重 w：softmax(acc)
--------------------------------------

我们希望从 4 个 accuracy 得到 4 个权重 w，要求：

- 每个权重非负；
- 权重和为 1；
- accuracy 越高，权重越大；
- 区别更大的模态权重差异更明显。

实现：

```python
acc_vec = np.array(list(accuracies.values()), dtype=np.float32)
# acc_vec = [acc_text, acc_beh, acc_em, acc_score]

# 为了数值稳定和区分度，对 acc 做简单缩放再 softmax。
logits = acc_vec / max(acc_vec.max(), 1e-6)
w = np.exp(logits)
w = w / w.sum()

weights = {name: float(w[i]) for i, name in enumerate(accuracies.keys())}
```

对 v2 数据集，具体数值（acc → w）为：

| 模态      | 准确率 acc | 权重 w（softmax(acc)） |
|-----------|-----------:|------------------------|
| text      | 0.567      | 0.218                  |
| behavior  | 0.383      | 0.181                  |
| emotion   | 1.000      | 0.336                  |
| score     | 0.767      | 0.266                  |

解释：

- 权重满足 `w_text + w_behavior + w_emotion + w_score = 1`；
- 情绪标签模态（emotions）最可靠，权重最大（约 0.336）；
- mood_score 次之（约 0.266）；
- 文本与行为模态权重较小但非零，保留补充信息。

这一步对应的是“从历史表现中学习模态权重”，而非手写常数。

八、日级 late fusion：P_fusion(day)
-----------------------------------

对每一天 day，我们已有 4 个模态的预测向量：

```text
P_text(day), P_behavior(day), P_emotion(day), P_score(day) ∈ R^4
```

以及对应权重：

```text
(w_text, w_behavior, w_emotion, w_score)
```

融合定义：

```text
P_fusion(day) =
    w_text     * P_text(day)
  + w_behavior * P_behavior(day)
  + w_emotion  * P_emotion(day)
  + w_score    * P_score(day)
```

具体而言，若我们用向量表示：

```text
P_mod(day) = [P_mod(积极), P_mod(消极), P_mod(中性), P_mod(复杂)]
```

则融合后的每一类概率为：

```text
P_fusion(积极) = Σ_mod w_mod * P_mod(积极)
P_fusion(消极) = Σ_mod w_mod * P_mod(消极)
P_fusion(中性) = Σ_mod w_mod * P_mod(中性)
P_fusion(复杂) = Σ_mod w_mod * P_mod(复杂)
```

在当前 V0 实现中，4 个模态的 `P_mod(day)` 均为 one-hot 向量，  
因此融合后的 `P_fusion(day)` 可以理解为“四个专家投票，按权重加权”的结果。

所有天拼在一起，得到 `fusion_daily.npy`（形状 60×4）：

```text
fusion_probs[day, k] = P_fusion(day)[k]
```

九、从融合概率到 valence：给时序与 IRL 用的 1D 信号
-------------------------------------------------

在 `features/emotion_fusion.py` 中：

```python
def fusion_valence(fusion_probs: np.ndarray) -> np.ndarray:
    """
    从融合概率中导出 1D valence 序列：
        valence = P(积极) - P(消极)
    """
    idx_pos = _LABEL_INDEX["积极"]   # 0
    idx_neg = _LABEL_INDEX["消极"]   # 1
    return (fusion_probs[:, idx_pos] - fusion_probs[:, idx_neg]).astype(np.float32)
```

定义：

```text
valence(day) = P_fusion(积极) - P_fusion(消极)
```

解释：

- valence(day) > 0：这天整体偏积极；
- valence(day) < 0：偏消极；
- 接近 0：积极 / 消极成分接近，更多是复杂或中性。

`features/temporal_features.py` 使用该一维序列作为输入：

- 计算 7 日滑窗统计（mean / var / max / min / trend），得到 `rolling_stats.npy`（形状 `(54,5)`）；
- 计算 60 日 EWMA 基线，得到 `global_baseline.npy`；
- `temporal_meta.json` 中 `source_series` 显式标为 `"fusion_valence"`。

十、设计 rationale 与后续演化
----------------------------

1. 为什么要用 4 个单模态预测器而不是直接用某一个模态？

   - 文本、行为、标签、自评分各有偏差来源：  
     - 文本可能语言乐观但行为高压；  
     - mood_score 会受当日打分习惯影响；  
     - emotions JSON 受上游模型 / prompt 影响。  
   - 用 4 个简单、可解释的视角先拉出“候选判断”，后续再根据历史表现加权，是更稳健的工程策略。

2. 为什么用 `emotions/day_xxx.json['label']` 做 GT？

   - 当前 60 天数据是 synthetic / LLM 生成，`emotions` 是我们对“这天整体情绪”的最直接标注；  
   - 把它作为初始 GT，可以在不引入额外标注的情况下评估其他模态的可靠度；  
   - 将来如果有更可靠的人类标注，可以替换这一层，而不改动 fusion 结构。

3. 为什么要 `argmax` + accuracy？

   - 我们需要一个统一的“硬标签预测”来对比 GT，这在分类任务中是标准做法（MAP 决策）；  
   - accuracy = “预测正确的样本数 / 总样本数”是最基础也最直观的可靠度指标；  
   - 这一层评估的是“在 60 天任务上，这个模态单独做分类有多靠谱”，为权重学习提供信号。

4. 为什么用 softmax(acc) 作为融合权重？

   - softmax(acc) 保证权重非负且和为 1；  
   - acc 越大，权重越大，同时通过指数将差距适当放大，  
     让明显更可靠的模态有更大话语权；  
   - 实现简单、可解释，同时兼容未来替换为更复杂的权重学习方法。

5. 为什么要从 `P_fusion` 提取 valence，而不是直接用类别？

   - 后续 LSTM 与 IRL 关心的是“这一段时间整体偏好 / 偏糟”，属于连续信号；  
   - valence = P(积极) - P(消极) 将“好坏”信息压缩为一维，便于做滑窗统计与趋势建模；  
   - 保留了概率信息（例如 “70% 积极 + 20% 消极” 和 “50% 积极 + 0% 消极” 会得到不同的数值）。

十一、结论
---------

- 在 `data/isabella_irl_60d_openai_v2` 上，多模态情绪融合模块已经按设计完整落地：
  - `DailyRecord` 统一整合文本/行为/情绪/分数；
  - 4 个单模态预测器在日级做情绪分类；
  - 使用 `emotions` 作为 GT，计算每个模态在 60 天上的预测准确率；
  - 通过 softmax(acc) 得到模态融合权重 w；
  - 对每天做加权融合，得到统一的 `fusion_daily.npy`；
  - 从中导出 `fusion_valence`，驱动时序特征与 IRL reward 代理。
- 该设计在工程上简单、可解释，在理论上符合“弱模态协同 + 数据驱动权重”的思路，  
  为后续 IRL 与人格对齐提供了一个结构清晰、可审计的情绪输入层。

