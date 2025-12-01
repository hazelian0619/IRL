多模态情绪 → 标量 reward：valence 校准实验报告
=========================================

撰写目的
--------

本报告整理我们围绕“如何从融合后的日级情绪分布 `P_fusion(day)` 中构造一个标量 reward”
而尝试的两种方案，以及在 `isabella_irl_3d_clean` 与
`isabella_irl_60d_openai_v2` 两个数据集上的实证结果和取舍结论。

目标是回答：

1. 我们当前使用的 `valence = P(积极) - P(消极)` 是否合理？  \n
2. 是否有更“专业/数据驱动”的映射方式？  \n
3. 在现有 60 日 IRL 主线下，如何“抓大放小”选择 reward 方案。

背景：当前结构
--------------

日级多模态情绪融合层输出：

- 四个模态：text / behavior / emotion / score；  \n
- 每天，各模态输出一条 4 维情绪概率：

  ```text
  P_mod(day) = [P(积极), P(消极), P(中性), P(复杂)]
  ```

- 使用 `emotions.label` 的 canonical 版本作为 GT，计算每个模态在 60 天上的准确率 acc_mod；  \n
- 使用 softmax(acc_mod) 得到模态融合权重 w_mod；  \n
- 对每天做加权平均，得到融合后的情绪分布：

  ```text
  P_fusion(day) = Σ_mod w_mod * P_mod(day)
  ```

当前 pipeline 中，从 `P_fusion` 压成标量 reward 的定义为：

```text
valence_raw(day) = P_fusion(day, "积极") - P_fusion(day, "消极")
```

后续 temporal / IRL 用法：

- `fusion_valence` 产生长度为 60 的 valence 序列；  \n
- `export_temporal_features` 在该 1D 序列上做 7 日滑窗平均，得到：
  - `window_valence[t]`：作为每个窗口（7 日）的 reward 代理 r_t；  \n
  - `rolling_stats[t]`：作为周级状态特征（mean/var/max/min/slope）。  \n
- IRL MVP / Discovery 在 `(z_t, r_t)` 上工作。

研究问题
--------

上述 `valence_raw = P_pos - P_neg` 在直觉上合理，但在“工业化/规范化”的视角下，
存在两个潜在问题：

1. `reward(积极)=+1, reward(消极)=-1, 中性/复杂=0` 这一设定是手工指定的，没有直接数据来源；  \n
2. 我们已有 daily `mood_score` 这条“主观情绪刻度”信息，是否可以用来校准
   `P_fusion`→valence 的 mapping，使得 reward 更贴近真实心情。

因此我们尝试了两条校准思路：

- 方案 A：直接用线性回归学习 `valence = w · P_fusion` 中的权重 w，使其逼近 `mood_score`；  \n
- 方案 B：按 canonical label 分组，统计每类的平均 `mood_score`，映射到 [-1,1] 作为
  `reward(label)`，再对 `P_fusion` 做期望。

方案 A：线性回归校准 w·P_fusion
-------------------------------

实现位置：`scripts/experiment_valence_calibration.py`

核心思路：

> 已知：每天的融合情绪分布 `P_fusion(day) = [p_pos, p_neg, p_neu, p_comp]`，  \n
> 以及对应的 `mood_score(day)`；  \n
> 将 `mood_score` 线性缩放到大致 [-1,1] 得到 `mood_norm(day)`；  \n
> 用最小二乘拟合一个 4 维线性映射：
>
> ```text
> mood_norm(day) ≈ w_pos * p_pos + w_neg * p_neg + w_neu * p_neu + w_comp * p_comp
> ```
>
> 从而得到一组权重 w，使得：
>
> ```text
> valence_cal(day) = w · P_fusion(day)
> ```
>
> 在“拟合 mood_score”的目标下，比原始 `valence_raw = p_pos - p_neg` 更贴合 self-report。

实验步骤：

1. 加载 `<root>/features/fusion_daily.npy`，得到 60×4 的 `fusion_probs`；  \n
2. 使用 `DailyDataset(root)` 加载 60 天的 `mood_score`，得到一维 `scores`；  \n
3. 用 `(scores - 5.5) / 4.5` 将 mood_score 线性映射到约 [-1,1] 的 `mood_norm`；  \n
4. 原始 valence：

   ```text
   valence_raw(day) = fusion_valence(fusion_probs[day])
                    = p_pos(day) - p_neg(day)
   ```

5. 校准 valence：

   ```text
   w = argmin_w || fusion_probs @ w - mood_norm ||^2
   valence_cal(day) = fusion_probs[day] · w
   ```

6. 对比两种 valence 与 `mood_norm` 的关系：
   - corr(valence_raw,  mood_norm)；  \n
   - corr(valence_cal, mood_norm)；  \n
   - MSE(valence_raw  vs mood_norm)；  \n
   - MSE(valence_cal vs mood_norm)。

实验结果（简要）
----------------

### A.1 数据集：`data/isabella_irl_3d_clean`

（早期 3/10 日 sample，用来验证方法；非主线，但有参考意义）

学得权重 w（近似值）：

```text
w[积极] ≈ +0.706
w[消极] ≈ -1.722
w[中性] ≈ -1.172
w[复杂] ≈ -0.102
```

对齐程度：

```text
corr(mood_norm, valence_raw)        ≈ +0.864
corr(mood_norm, valence_calibrated) ≈ +0.887

MSE(valence_raw  vs mood_norm)      ≈ 0.100
MSE(valence_calibb vs mood_norm)    ≈ 0.035
```

解读：

- 相关系数从 ~0.86 提升到 ~0.89，略有提高；  \n
- MSE 从 ~0.10 降到 ~0.035，显著降低；  \n
- 表明在这个 sample 上，用线性回归学到的 w 确实使 valence 更贴近 mood_score；
  说明“用 P_fusion → 线性回归校准到 mood_score”这一思路是有效的。

### A.2 数据集：`data/isabella_irl_60d_openai_v2`

（主线 60 天数据，当前所有 IRL 实验的主体）

学得权重 w（近似）：

```text
w[积极] ≈ +0.980
w[消极] ≈ -1.644
w[中性] ≈ -0.366
w[复杂] ≈ -0.140
```

对齐程度：

```text
corr(mood_norm, valence_raw)        ≈ +0.793
corr(mood_norm, valence_calibrated) ≈ +0.794

MSE(valence_raw  vs mood_norm)      ≈ 0.095
MSE(valence_calibb vs mood_norm)    ≈ 0.079
```

解读：

- 原始 `P_pos - P_neg` 与 `mood_norm` 的相关性已经较高（约 0.79）；  \n
- 线性校准后的相关性略有提升（0.793→0.794，提升甚微）；  \n
- MSE 有一定下降（0.095→0.079），说明 valence_cal 在数值上更贴近 mood_score；
- 学到的 w 也符合直觉：
  - 积极权重为正，消极权重为较大的负值；  \n
  - 中性/复杂在此数据中更多出现在“低于积极但不极端负”的情境下，因此系数略为负。

总体结论（方案 A）：

- 在 clean sample 上，线性校准明显改善了 valence 与 mood_score 的对齐；  \n
- 在主线 v2 上，线性校准保持了原有相关性并略有改善，MSE 有可观下降；  \n
- 结构上，它符合“期望 reward = w · P_fusion”的形式，只是 w 由数据拟合而非手写。

方案 B：按 label 分组求均值的 reward_map
-------------------------------------

方案 B 的出发点是：

> “对每个 canonical label（积极/消极/中性/复杂），统计它对应的 mood_score 平均值，
>  作为该情绪类别的 valence，再用期望的方式映射 P_fusion → 标量 reward。”

具体步骤（在 `data/isabella_irl_60d_openai_v2` 上试验）：

1. 对 60 天数据，按 `canonical_label(emotions.label)` 分组，得到四个桶：  \n
   - 积极；消极；中性；复杂。  \n
2. 在每个桶里求 `mood_score` 的均值 μ(label)：  \n
   - 积极：n=34，mean≈8.29；  \n
   - 消极：n=0（无样本）；  \n
   - 中性：n=0（无样本）；  \n
   - 复杂：n=26，mean≈5.89。  \n
3. 用线性映射 `(score - 5.5)/4.5` 将均值映射到 [-1,1]，得到 reward_map(label)：  \n
   - reward(积极) ≈ +0.62；  \n
   - reward(复杂) ≈ +0.09；  \n
   - reward(消极/中性)：因无样本，无法估计。  \n
4. 定义：

   ```text
   valence_new(day) = Σ reward(label_k) * P_fusion(label_k | day)
                    ≈ 0.62 * P_pos(day) + 0.09 * P_complex(day)
   ```

5. 与 `mood_norm` 对比：

   - corr(valence_raw, mood_norm) ≈ 0.793；  \n
   - corr(valence_new, mood_norm) ≈ 0.781（略低）；  \n
   - 且 valence_new(day) 全为正值（最小值约 0.12），完全没有负区间。

解读（方案 B 的问题）：

- 由于当前 v2 数据中 canonical label 几乎只有“积极”和“复杂”两类，
  且两者的 `mood_score` 均值均为正，方案 B 得到的 reward(label) 全是正值；  \n
- 这样得到的日级 valence_new 无负值，所有天都被压在“略好/比较好”的区间；  \n
- 在 IRL 场景下，这会削弱“真正低 reward 周”的对比，对偏好结构的识别不利；  \n
- 在 `corr` 指标上，valence_new 甚至略逊于 valence_raw。

因此：

- 方案 B 虽然形式上“心理学味道更浓”（按情绪类别分组求均值），但在当前这条
  60 日轨迹的标签分布下（严重偏向积极/复杂），实际效果不如原始 valence，  
  也不如方案 A 的线性校准；  \n
- 我们将其视为一次探索，不作为当前主线的 reward 定义。

取舍与推荐
----------

综合上述实验：

1. “从分类概率 P_fusion 映射到标量 reward”的结构本身是合理且业界常见的：  \n
   - 形式上 `r_day = Σ reward(label_k)*P_fusion(label_k | day)` 是“期望情绪强度”的标准表达；  \n
   - 原始 valence `P_pos - P_neg` 对应于一个简单的 reward_map：  
     `reward(积极)=+1, reward(消极)=-1, 其余≈0`。  \n

2. 方案 A（线性回归校准）在 clean 与 v2 数据上均证明：
   - 可以在不破坏原结构的前提下，让日级 valence 与 mood_score 的对齐略有提升（尤其是 MSE）；  \n
   - 学到的权重 w 也符合直觉（积极正向，消极强负，中性/复杂略负）；  \n
   - 但在主线 v2 上，提升主要体现在 MSE，相关性提升很小，对 IRL 整体 story 影响有限。  \n

3. 方案 B（按 label 分组求均值）在 v2 数据上受限于标签分布（几乎只有积极/复杂），
   反而削弱了负向信息，因此不推荐作为当前 reward 定义。

在当前主线下的“抓大放小”决策：

- 必需的结构优化已经完成：  \n
  - 文本 / 行为模态的情绪预测已从词频规则升级为统一 LLM backend；  \n
  - 多模态 fusion 使用 softmax(acc) 的 late fusion 结构，简单稳定且可解释；  \n
- 在 reward 定义上，当前 `valence_raw = P_pos - P_neg` 已能提供足够的区分度：
  - 能明显区分高/低 reward 周；  \n
  - 与 `mood_score` 有较高相关；  \n
  - IRL discovery 的高/低 reward 周 story 已经可以合理讲通。  \n

因此，本阶段推荐：

1. 保留 `valence_raw = P_fusion(积极) - P_fusion(消极)` 作为 IRL V1 的 reward proxy；  \n
2. 将方案 A 的线性校准视为一个“可选升级路径”：  \n
   - 在需要对外强调“与 mood_score 标尺一致性”的场合，可以将 `w·P_fusion` 引入为
     新的 valence 定义，并在文档中引用本实验结果；  \n
3. 明确记录方案 B 的尝试及其局限，避免未来在相同标签分布下再次走入“label 统计洗白负向”的陷阱。

未来工作建议
------------

- 若后续获得更多 persona 或真实用户数据，尤其是具备更丰富的“消极/中性”标注，  \n
  可以在以下方向进一步迭代：

  1. 在更大数据集上重新拟合线性 w，作为统一的 valence 标尺；  \n
  2. 引入简单的分段/非线性校准（例如对极端高/低 mood_score 作特殊处理）；  \n
  3. 在 IRL 分析中对比 raw valence / calibrated valence 在高/低 reward pattern 上的一致性。

在当前单 persona 60 日 IRL 实验阶段，我们将重点放在：

- LLM 驱动的多模态情绪感知；  \n
- z_t 空间中的高/低 reward 模式发现；  \n
- f(s)+w 偏好轴与人格故事的对齐，

而在 reward 定义上采用 `valence_raw` 这一简洁且已经验证合理的 V1 方案。

