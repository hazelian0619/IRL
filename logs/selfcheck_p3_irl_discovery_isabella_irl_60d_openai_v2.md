P3 自查：Isabella 60 天 IRL Discovery（reward-aware 模式挖掘）
===========================================================

检查时间：当前开发会话。  
数据根目录：`data/isabella_irl_60d_openai_v2`  
模型输出目录：`data/isabella_irl_60d_openai_v2/models`

目标
----

在已经完成的 IRL MVP 基础上，对 `isabella_irl_60d_openai_v2` 做一轮最小可用的
“reward-aware 模式挖掘”（对应 irl_design_report.md 里的阶段 1）：

- 利用 `temporal_embeddings.npy` 与窗口级 reward 代理 `window_valence.npy`；
- 训练简单的 reward regressor `R(z_t) ≈ r_t`，确认状态中确实编码了 valence 型 reward；
- 按 `r_t` 对 54 个 7 日窗口排序，并回看若干高 / 低 reward 窗口的原始生活片段；
- 用自然语言总结候选的“高价值生活模式 / 低价值生活模式”，为后续 f(s)+w 设计提供素材。

使用脚本
--------

本次自查使用的主要命令：

```bash
cd companion-robot-irl

# 1）导出时序嵌入与窗口级 reward（内部会调用 rolling_stats / fusion_valence）
python3 scripts/export_temporal_embeddings.py \
  --root data/isabella_irl_60d_openai_v2 --device cpu

# 2）在 (z_t, r_t) 上训练 IRL MVP 回归器
python3 scripts/run_irl_mvp_isabella.py \
  --root data/isabella_irl_60d_openai_v2 --epochs 300 --device cpu
```

核心中间结果：

- `features/temporal_embeddings.npy`：形状 `(T=54, D=257)` 的周级状态嵌入 `z_t`；
- `features/window_valence.npy`：形状 `(54,)` 的窗口级 reward 代理 `r_t`
  （基于 fusion_valence 的 7 日滑窗平均）；
- `models/irl_reward_regressor.pt`：在 `(z_t, r_t)` 上训练得到的 MLP 回归器。

IRL MVP 拟合质量（v2 数据）
--------------------------

终端输出（节选）：

```text
[INFO] Training reward regressor on data/isabella_irl_60d_openai_v2
[INFO] input_dim=257, hidden_dim=64, use_mlp=True
[EPOCH 001] train MSE: 0.377829
[EPOCH 050] train MSE: 0.004289
[EPOCH 100] train MSE: 0.002741
[EPOCH 150] train MSE: 0.001771
[EPOCH 200] train MSE: 0.001156
[EPOCH 250] train MSE: 0.000906
[EPOCH 300] train MSE: 0.000901
[INFO] Saved reward regressor checkpoint to data/isabella_irl_60d_openai_v2/models/irl_reward_regressor.pt
[INFO] Final MSE=0.000851, MAE=0.023511
```

说明：

- 在 reward 范围约为 `[0, 1]` 的前提下，MSE ≈ 8.5e-4、MAE ≈ 0.024，回归器可以平滑重构
  每个窗口的 `window_valence`；
- 与 `logs/selfcheck_p3_irl_mvp_isabella_irl_3d_clean.md` 一致，说明 v2 数据在结构上
  也满足“z_t → valence 型 reward 可拟合”的前提，可以放心用来做 reward-aware 模式发现。

窗口级 reward 分布与挑选的样本
------------------------------

对 `window_valence.npy` 做排序（T=54）：

- 最高的几个窗口（high reward）：
  - idx 0：days 1–7，`r_0 ≈ 0.819`
  - idx 1：days 2–8，`r_1 ≈ 0.819`
  - idx 5：days 6–12，`r_5 ≈ 0.771`
  - idx 6：days 7–13，`r_6 ≈ 0.771`
- 最低的几个窗口（relative low reward）：
  - idx 24：days 25–31，`r_24 ≈ 0.129`
  - idx 25：days 26–32，`r_25 ≈ 0.129`
  - idx 26：days 27–33，`r_26 ≈ 0.191`
  - idx 28：days 29–35，`r_28 ≈ 0.143`
  - idx 33：days 34–40，`r_33 ≈ 0.191`

可以看到：

- 所有窗口的 `r_t` 都在 `[0.12, 0.82]` 的正区间内；
- 但早期情人节准备期（day 1–13）的窗口明显处在高端；
- 选举 / 艺术展混杂期（大致 day 25–40）的窗口相对较低且更集中在“复杂”标签。

高 reward 窗口：生活模式观察
----------------------------

代表性窗口：idx 0, 1, 5, 6（day 1–7, 2–8, 6–12, 7–13）。

从 `emotions/day_*.json` 与 `conversations/day_*.md` 回看，高 reward 周的大体共性：

- 情绪标签：
  - 绝大多数天是“开心”，少数夹杂“感动、温暖”；
  - reason 高频关键词：被认可、被感谢、被支持、温暖、满足、意义感。
- 行为结构与情境：
  - 处于情人节筹备与派对阶段：
    - 在 Hobbs Cafe 忙着布置、设计菜单、邀请顾客；
    - 有老教授送画、老顾客送感谢卡、年轻情侣计划来派对；
  - 社交质量高：
    - 互动以“表达感谢、分享幸福、温暖交流”为主；
    - 很少出现明显的激烈争论或情绪劳动。
- 负荷感：
  - 文本会提到“忙、稍微疲惫”，但整体 tone 是“忙得有意义、被需要”，
    而不是被压垮；
  - 一周内通常能看到少量恢复性的、相对轻松的片段（旧友重联络、艺术交流等）。

可以抽象成一种“理想忙碌周”的 prototype：

> 有节奏的忙碌 + 被认可 + 温暖社交 + 适度恢复  
> （情绪整体高、波动适中）。

低 reward 窗口：生活模式观察
----------------------------

代表性窗口：idx 24, 25, 26, 28, 33（大致覆盖 day 25–40）。

从情绪与对话文本看，这一段的典型特征：

- 情绪标签：
  - “复杂”出现得非常频繁，也有少量“鼓舞”、“开心”；
  - reason 中常见：压力、争论、冲突、疲惫，与“责任感、意义感”交织。
- 行为结构与情境：
  - 处于选举 + 社区讨论 + 艺术展的高压阶段：
    - 咖啡馆变成政治讨论与辩论的场所；
    - 需要不断调解不同立场之间的冲突；
  - 社交质量下降：
    - 互动中包含大量情绪劳动（解释政治、平息争论、承受他人情绪）；
    - 仍然有温暖时刻，但被持续的紧张氛围稀释。
- 负荷与恢复：
  - 文本中反复出现“压力”“疲惫”“很难完全放松”等描述；
  - 一周内几乎看不到明确的休息日或“放空”的安排；
  - valence 在数值上仍为正，但波动更大、主观体验明显更拉扯。

可以抽象成一种“高压情绪劳动周”的 prototype：

> 高强度工作 + 持续政治争论 + 情绪劳动 + 休息缺失  
> （有意义但累，情绪体验复杂、紧绷、难以恢复）。

候选“偏好模式”原型
-------------------

基于上述高 / 低 reward 周的对比，当前可以先用自然语言列出两类候选偏好原型：

- 模式 A：有节奏的忙碌 + 被认可 + 温暖社交 + 适度恢复
  - 一周内有若干天高工作量，但整体可控；
  - 存在明确的“连接事件”（感谢、礼物、深度对话）；
  - 至少有 1–2 天相对轻松或带来恢复感的活动（艺术、旧友、安静时刻）；
  - valence 长期偏正，波动适中。

- 模式 B：高压工作 + 高情绪劳动 + 休息缺失
  - 选举期与艺术展叠加、工作连续高负荷；
  - 社交从“温暖交流”转为“政治争论 + 调解冲突”；
  - 几乎没有清晰的恢复日，持续疲惫；
  - valence 数值仍然为正，但主观体验明显偏“复杂 / 拉扯 / 累”。

在 IRL 语言下，可以粗略理解为：

- 高 reward 周 ≈ 模式 A 出现频率较高的周；
- 相对低 reward 周 ≈ 模式 B 占主导或显著存在的周。

候选偏好特征 f(s)（草案 V0）
-----------------------------

下一步不是立刻写复杂模型，而是先把这些原型结构化成少量“偏好轴” f_i(s)，
后续在 f(s) 空间上学习 w。结合现有特征管线，当前候选设计如下（只列意图，不写公式）：

- 情绪层特征（可直接由 rolling_stats 导出）：
  - `f_valence_mean(t)`：该窗口 fusion_valence 的均值（高 vs 低）；
  - `f_valence_var(t)`：该窗口 valence 的方差（稳定 vs 波动大）；
  - `f_valence_trend(t)`：7 日 valence 的线性趋势（恢复 vs 走向更糟）。

- 行为结构特征（基于 daily 行为特征聚合，后续在代码中实现）：
  - `f_work_load(t)`：一周内工作相关行为的强度 / 天数；
  - `f_rest_presence(t)`：一周内是否存在明显的休息日或低强度日；
  - `f_social_support(t)`：温暖 / 支持型社交的频率（可用积极情绪 + 社交行为的组合近似）；
  - `f_social_conflict(t)`：冲突 / 情绪劳动型社交的频率（可用“复杂”情绪占比 + 负向词等近似）；
  - `f_creative_activity(t)`：创作相关活动的存在与强度（艺术、策展等）。

- 故事阶段特征（需要简单的 day→phase 映射）：
  - `f_phase_valentine(t)`：窗口是否落在情人节筹备与派对阶段；
  - `f_phase_election(t)`：窗口是否落在选举高峰期；
  - `f_phase_recovery(t)`：窗口是否处于后期恢复 / 整理阶段。

这些 f_i(s) 的目标是：

- 数量控制在 5–10 维左右，保持可解释性；
- 每一维都能在高 / 低 reward 窗口之间看到明显差异；
- 将来可以与 Big Five / persona 文本做一一对照（例如高 E/A ↔ 高 `f_social_support`）。

对应关系（直觉校验，尚未正式拟合）：

- 高 reward 周（模式 A）预期：
  - `f_valence_mean` 高、`f_valence_var` 中等；
  - `f_work_load` 中高但 `f_rest_presence` 为真；
  - `f_social_support` 高、`f_social_conflict` 低；
  - 常落在 `f_phase_valentine` 或带有创作活动的阶段。

- 低 reward 周（模式 B）预期：
  - `f_valence_mean` 略降、`f_valence_var` 偏高；
  - `f_work_load` 高但 `f_rest_presence` 为假；
  - `f_social_conflict` 高，情绪劳动多；
  - 常落在 `f_phase_election` 高峰期。

与 irl_design_report.md 的对齐
-----------------------------

与 `logs/irl_design_report.md` 中的三层计划对应关系：

1. **Discovery（当前这一步）**  
   - 已经在 z_t + r_t 空间做了 reward-aware 的窗口排序；
   - 回看了若干高 / 低 reward 周，对应地整理出模式 A / 模式 B 的文字原型；
   - 这些原型与报告中“有节奏的忙碌 + 有恢复” vs “长期高压 + 无恢复”的描述一致。

2. **偏好轴 f(s) + 线性结构 R(s)=w·f(s)**  
   - 本自查给出了一组候选 f_i(s) 的草案（情绪统计 + 行为结构 + phase）；
   - 下一步将把这些特征在代码层显式实现，并在 `window_valence` 上拟合简单线性 / 浅层 MLP；
   - 目标是得到一个可解释的偏好权重向量 w。

3. **人格对齐（w 与 Big Five / persona story 对齐）**  
   - 该部分尚未在代码中实现，但设计方向清晰：
     - 例如预期高 E/A ↔ 正向的 `w_social_support`；
     - 高 C ↔ 对“有结构的忙碌”正向、对“超载无恢复”负向的 w 结构。

下一步建议
----------

结合本次 discovery，后续 IRL 部分的具体执行建议：

1. 在 `features/` 或 `scripts/` 中实现一个偏好特征导出器：
   - 输入：`fusion_valence`、daily 行为特征（social/work/rest/creative 等）、day→phase 映射；
   - 输出：`preference_features.npy`（形状 `(T=54, F)`）以及对应的 `preference_feature_names.json`。
2. 在 `learning/` 中新增一个简单的 `preference_reward.py`：
   - 定义线性模型 / 浅 MLP：`R_pref(s) = w·f(s)`；
   - 在 `window_valence` 上做拟合，输出 w 以及与候选偏好轴的对齐解释。
3. 在后续自查文档中：
   - 把学到的 w 与 BFI 前测 / 后测做一次 sanity check；
   - 用一两页图表讲清楚“高 reward 周对应的偏好结构”和“与人格 story 的关系”。

本文件作为阶段 1（Discovery）的规范化记录，为后续 f(s)+w 与人格对齐提供可追溯的依据。

