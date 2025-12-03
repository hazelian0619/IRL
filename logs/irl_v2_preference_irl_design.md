IRL V2 设计稿：基于偏好的 IRL（Preference-based IRL）
===============================================

撰写目的
--------

本设计稿整理我们在当前 60 天 IRL 主线下，将 IRL 层从 V0“reward 回归 + pattern discovery”
升级为 V2“真正意义上的基于偏好的 IRL（preference-based IRL）”的方案。

目标不是立即实现一个完整 MDP 下的 MaxEnt IRL，而是：

> 在现有周级状态表示 z_t 和 reward proxy r_t 上，  
> 引入一个具有明确 IRL 形式的 reward 学习过程：  
> 通过“周与周之间的偏好比较”来学习 R_θ(s)，  
> 并将 R_θ(s) 拆解成可解释的偏好轴 f_i(s)+w_i。

这条线要满足两点：

1. 从方法论上，属于 IRL/Reward from Preferences 家族，而不是普通回归；  \n
2. 从工程上，能自然接入当前 pipeline，不需要我们虚构一个完整 MDP 动态。

一、现状回顾：V0 做到了什么，哪里还不是 IRL
----------------------------------------

当前 V0 状态：

1. 日级多模态情绪感知层：  
   - 文本 / 行为模态已经用 LLM backend 做日级情绪预测；  \n
   - emotions JSON / mood_score 作为 teacher 模态，输出 P_emotion / P_score；  \n
   - 利用 60 天上的准确率做 softmax(acc) → w_mod，并做 late fusion，得到 `P_fusion(day)`；  \n
   - 从 fusion 概率中抽出 `valence(day) = P(积极) - P(消极)`，得到一条 60 日 valence 序列。

2. 周级时序状态层：  
   - 对 valence 做 7 日滑窗平均，得到 `window_valence[t]`（T≈54），作为 reward proxy r_t：  \n
     「这一周整体的平均好坏」。  \n
   - 对 valence 做 7 日窗口 mean/var/max/min/slope，得到 `rolling_stats[t]`；  \n
   - 用 BiLSTM+Attention + global_baseline 编码 rolling_stats 序列，得到 `z_t`：周级状态 embedding；  \n
   - 保存为 `temporal_embeddings.npy`（z_t）和 `window_valence.npy`（r_t）。

3. IRL V0：  
   - 用一个小 MLP `R_MLP(z_t) ≈ r_t` 做了一次 supervised 回归实验（`run_irl_mvp_isabella.py`），  \n
   - 证明 z_t 中确实包含足够信息去重构 valence 型 reward。  \n
   - 用 r_t 将 54 周按高/低排序，回看 story，读出了若干高/低 reward 模式，如：  
     - 高 reward 周：情人节准备期，“有节奏、有连接、有被认可、有恢复”的忙碌周；  \n
     - 低 reward 周：选举高压期，“高压、争论多、情绪劳动密集、几乎无休息”的周。

这些工作让我们有：

- state s_t（z_t）和 reward proxy r_t（window_valence）；  \n
- IRL discovery 的“文字原型”：高/低 reward 周的生活模式。

但严格来说：

- `R_MLP(z_t) ≈ r_t` 只是 reward 回归，不是 IRL 的 reward-from-demonstration；  \n
- 我们尚未引入一个明确的 IRL objective，比如：  
  - 从偏好比较学习 reward；  
  - 从轨迹分布匹配 feature expectation 学 reward。

二、为什么不直接上“完整 MDP + MaxEnt IRL”？
----------------------------------------

在教科书 IRL（Ng, Abbeel, Ziebart）里，我们需要：

1. 明确的动作空间 A：专家在每个状态下选择了什么动作 a_t；  \n
2. 环境动态 P(s'|s,a)：不同动作如何影响状态转移；  \n
3. 一批专家 demonstration 轨迹 τ = (s_0,a_0,s_1,a_1,...)。

在当前 60 天 IRL 主线下，我们的情况是：

- 有：  
  - 状态轨迹 s_t（z_t 或其 cluster）；  \n
  - reward proxy r_t（周级平均 valence）；  
- 无：  
  - 干净的“决策动作日志”：机器人/Isabella 在每周有哪些可控行为 choice；  
  - 可重复采样的 MDP 动态（Town/story 的生成逻辑是为故事服务，不是一个优化策略的环境）。

可以做的折衷（在设计稿中曾提出）：

- 从 behaviors 特征抽象出“窗口级 action”（比如 WorkHeavy / SocialHeavy / RestHeavy）；  \n
- 对 z_t 做聚类得到离散 state；  \n
- 用 (s_t, a_t, s_{t+1}) 构建一个小经验 MDP，跑 MaxEnt IRL。

问题在于：

- 数据极少：每 persona 只有一条 60 日轨迹，每条只有 ~54 个窗口；  \n
- 动作 a_t 是事后从行为“抽象”出来的，而不是 agent 的真实决策 log；  \n
- 状态转移基本固定（t→t+1），探索空间很窄，难以体现“在 alternative 路径上 reward 的差别”；  \n
- IRL 得到的 R_θ(s) 容易退化为“某种复杂版 valence 回归”，解释成本高而收益有限。

因此，从“抓大放小”和“当前数据规模”的角度看，在 V2 阶段上完整 MDP+MaxEnt IRL：

- 方法上是正统的 IRL，但工程成本高；  \n
- 在主线 story 和人格对齐上，不一定比轻量的偏好式 IRL 更加分。

三、V2 的选择：偏好式 IRL（preference-based IRL）
----------------------------------------------

为了既对得起“IRL”三字，又不把主线拖进过重的 MDP 假设，我们将 IRL V2 定义为：

> 在周级状态空间上，使用“偏好式 IRL”的框架，从高/低 reward 周之间的偏好关系  
> 学习一个 reward 函数 R_θ(s)，而不是直接用回归拟合 r_t。

思想来源：

- Christiano et al., “Deep Reinforcement Learning from Human Preferences” (2017)：  \n
  从人类对片段 (segment) 的 pairwise preference（A ≻ B）中学习 reward 模型；  \n
- Ng & Russell 系 IRL 的现代变体：从比较而不是从动作序列直接反推 reward。

在我们现有的结构里，自然存在一批“偏好比较”：

- 对任意两个周 t_i, t_j，若 `r_{t_i} > r_{t_j} + ε`，  \n
  则可以认为“周 i 比 周 j 更好”（i ≻ j）。

使用这些偏好对，我们可以在状态空间上定义一个 IRL 目标：

```text
R_θ(s) = θᵀ φ(s)
P_θ(i ≻ j) = σ(R_θ(s_i) - R_θ(s_j))
maximize Σ_{(i,j): r_i > r_j} log P_θ(i ≻ j)
```

这与 MLP 回归的区别在于：

- 回归：逼近绝对值 r_t；  \n
- 偏好 IRL：只关心“哪一周好过哪一周”，  
  reward 函数是通过这些比较约束学习出来的。

四、具体设计：IRL V2（偏好式 IRL）在当前管线上的定义
--------------------------------------------}

### 4.1 状态 s_t：选用可解释的周级特征 φ(s_t)

虽然我们已有 z_t (257 维)，但为了便于解释和与人格对齐，我们在 IRL V2 中建议：

- 定义一组低维、可解释的状态特征 φ(s_t)，例如：  
  - φ_1(s_t)：该周 window_valence 均值（mean valence）；  
  - φ_2(s_t)：valence 方差（波动大小）；  
  - φ_3(s_t)：valence 最低点（这一周是否有极糟的一天）；  
  - φ_4(s_t)：trend（这一周是向上恢复还是向下滑落）；  
  - φ_5(s_t)：这一周行为中“social/work/rest”的平衡指标；  
  - φ_6(s_t)：这一周的“高压情绪劳动”指标（来自 behavior/emotion 的模式）；  
  - φ_7(s_t)：是否处在某个 story phase（如 election、recovery）。

这些 φ_i 本质上就是我们在 IRL discovery 中 verbal 的那些 pattern 的量化版。

在工程上，可以：

- 用 rolling_stats + 日级行为统计 + phase 标签，一次性构建 φ(s_t) ∈ R^F（F≈5–10）；  \n
- 与 z_t 并行存在：z_t 供其他模型使用，φ(s_t) 用于 IRL V2 的 reward 表达。

### 4.2 偏好对 (i, j) 的构造

利用已有的 reward proxy r_t（window_valence[t]）：

1. 对 54 个 r_t 排序；  \n
2. 选择偏好对集合 Pairs：  
   - 简单策略：  
     - top-K vs bottom-K：对所有 top K 周和 bottom K 周，两两构造 i ≻ j；  \n
       例如 K=5，得到 5×5 = 25 个偏好约束；  \n
   - 或者：
     - 对所有 (i,j)，若 r_i - r_j > ε（例如 0.1）则认为 i ≻ j。  \n
3. 得到一组偏好对集合：

   ```text
   Pairs = {(i,j) | 周 i 明显好于 周 j}
   ```

这一步将原本的标量 reward signal 转换为 IRL 需要的“比较约束”。

### 4.3 reward 模型 R_θ(s) 与 IRL 目标

定义线性 reward 模型：

```text
R_θ(s_t) = θᵀ φ(s_t)
```

对每个偏好对 (i,j)∈Pairs，定义偏好概率：

```text
P_θ(i ≻ j) = σ(R_θ(s_i) - R_θ(s_j)) = 1 / (1 + exp(-(R_θ(s_i)-R_θ(s_j))))
```

IRL 目标函数：

```text
L(θ) = Σ_{(i,j)∈Pairs} log P_θ(i ≻ j)
```

或者加入正则项：

```text
L(θ) = Σ log P_θ(i ≻ j) - λ ||θ||²
```

优化 θ 即可得到一个“符合这些偏好比较”的 reward 函数。

这样，我们不再是简单地在 (s_t) 上回归 r_t，而是用 IRL 风格的 pairwise preference likelihood 来学 θ。

### 4.4 输出与解释

训练完成后，我们得到：

- 参数向量 θ ∈ R^F：f_i(s) 的权重；  \n
- 每个周的 IRL-style reward：R_θ(s_t)。

解释层面：

- θ_i > 0 且较大：说明 φ_i 对“高 reward 状态”的贡献大，代表 Isabella 强偏好这一维；  \n
- θ_i < 0 且较大：说明该维高时，状态往往被判为低 reward（她反感这一维）；  \n
- 这些 θ_i 可以直接和 BFI/persona 文本对齐，例如：  
  - 高 E / 高 A 可能对应积极的 social quality 权重；  \n
  - 高 C 对“有结构的高工作量 + 有恢复日”权重为正，对“无恢复高压”权重为负。

五、与主线的关系：为什么这是“真正的 IRL”而不是普通回归？
-----------------------------------------------

1. 从方法论上：

   - 普通回归：给你 (φ(s_t), r_t)，用 MSE 拟合 r_t；  \n
   - 偏好式 IRL：给你 (φ(s_t)), 以及“i ≻ j”的比较约束，最大化 Σ log σ(R(s_i)-R(s_j))。  

   后者更接近 IRL 文献的精神：  
   reward 是通过“偏好约束”反推出的，而不是直接 supervised label。

2. 从数据形态上：

   - 我们只有单 persona / 少量轨迹，动作和环境动态难以可靠建模；  \n
   - 但我们天然拥有“周与周之间的偏好”信息（来自 window_valence）；  
   - 偏好式 IRL 正好利用的是这种信息。

3. 从主线目标上：

   - 我们想要的是偏好结构 f(s)+w 与人格故事对齐；  \n
   - 偏好式 IRL 学出来的 θ 就是一个 **偏好权重向量**，非常适合拿来做这种解释；  \n
   - 它和经典的线性 IRL（手工 f(s)，学 w）在形式上是一脉相承的。

六、与未来 MDP+MaxEnt IRL（V3）的关系
---------------------------------

在更长远的规划中，我们仍然可以：

- 为多 persona、多 run 构建一个离散化 state graph + 高层行为动作；  \n
- 在这个小 MDP 上运行 MaxEnt IRL，学习 R(s) 或 R(s,a)；  \n
- 把轨迹结构纳入 reward 学习过程。

但在当前单 persona、单轨迹、有限行动日志的条件下：

- 强行上 MDP+MaxEnt IRL 会面临状态-动作空间稀疏、动态难以解释的问题；  \n
- 偏好式 IRL V2 已足以展示 IRL 层的优势，并与人格主线高度契合。

因此本设计稿将偏好式 IRL 定位为 IRL V2 的主线方案，  
而 MDP+MaxEnt IRL 留作 IRL V3（多数据、多 persona 阶段）的扩展方向。

七、下一步执行思路（简要）
-----------------------

1. 整理 IRL discovery：  
   - 从当前 r_t（window_valence）中正式选出 top-K / bottom-K 周；  \n
   - 为这些周写清楚 story-level 模式（高/低 reward pattern 列表）。

2. 设计并实现 φ(s_t)：  
   - 基于 rolling_stats、行为统计、phase 标签，构造 5–10 个偏好特征；  \n
   - 在 features/ 或 learning/ 新增一个模块导出 `preference_features.npy`。

3. 构造偏好对 Pairs：  
   - 在 learning/irl_preference.py 中，根据 r_t 构造 (i,j) 对，支持 top-vs-bottom 或基于差值阈值的构造策略。

4. 实现偏好式 IRL 训练：  

   - 定义 R_θ(s)=θᵀφ(s)，θ 初始化为 0 或来自简单 valence 映射；  \n
   - 用 σ(R(s_i)-R(s_j)) 定义偏好概率，最大化 log-likelihood；  \n
   - 保存 θ，以及每周的 R_θ(s_t)。

5. 解释与人格对齐：  

   - 查看 θ_i 的符号和大小，对应到每个 φ_i 的语义；  \n
   - 将这些偏好方向与 Isabella 的 BFI（O/C/E/A/N）和 persona 文本对齐撰写解释；  \n
   - 形式上即为：f(s)+w 的 IRL 偏好结构，与人格 story 一致。

本设计稿可作为后续实现 IRL V2（偏好式 IRL）的工程蓝图。

