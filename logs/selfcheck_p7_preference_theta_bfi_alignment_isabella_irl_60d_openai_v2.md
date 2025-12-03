## selfcheck_p7：θ 的偏好结构 & BFI 对齐 – Isabella 60d v2

本轮对应《执行.md》中 **Phase E：人格对齐 & 报告** 的第一步：  
对已经学到的 `θ` 做结构化解释，并和 Isabella 的 BFI 前后测 / 故事线对齐，同时对比一个简单的监督回归 baseline。

---

### 1. 背景：我们手上有哪些向量？

1）周级偏好特征 φ(s_t)

- 文件：`data/isabella_irl_60d_openai_v2/features/preference_features.npy` (54, 10)
- 每一维含义（见 Phase B）：
  1. `valence_mean`   – 7 日窗口 valence 均值  
  2. `valence_var`    – valence 方差  
  3. `valence_trend`  – valence 线性斜率  
  4. `social_level`   – 社交强度（日均 `beh_social_count`）  
  5. `rest_level`     – 休息/低强度活动强度（日均 `beh_rest_count`）  
  6. `work_level`     – 工作强度（日均 `beh_work_count`）  
  7. `conflict_level` – “复杂”情绪日占比（情绪劳动/冲突 proxy）  
  8. `phase_valentine` – 是否覆盖 Valentine 阶段  
  9. `phase_election`  – 是否覆盖 election 阶段  
  10. `phase_recovery` – 是否覆盖 recovery 阶段

2）偏好 IRL 学到的 θ（Phase D）

- 文件：`data/isabella_irl_60d_openai_v2/features/irl_theta.npy` (10,)
- Reward 函数：`R_θ(s_t) = θ^T φ(s_t)`
- θ 数值：

  ```text
  valence_mean     : +0.6530
  valence_var      : -0.1525
  valence_trend    : +0.0123
  social_level     : +0.2187
  rest_level       : -0.1926
  work_level       : +0.4656
  conflict_level   : -0.7775
  phase_valentine  : +0.5944
  phase_election   : -1.1292
  phase_recovery   : -0.0963
  ```

3）Isabella 的 BFI（前/后测）

- 前测（pretest REAL LLM，0-1）：

  ```json
  {"E": 0.8438, "A": 0.9444, "C": 0.8056, "N": 0.2188, "O": 0.8056}
  ```

- 后测（posttest IRL REAL LLM，0-1）：

  ```json
  {"E": 0.8125, "A": 0.9167, "C": 0.8611, "N": 0.2188, "O": 0.8333}
  ```

- 差值 Δ(post - pre)：

  ```text
  O: +0.0277
  C: +0.0555
  E: -0.0313
  A: -0.0277
  N: +0.0000
  ```

---

### 2. θ 的偏好结构：她到底喜欢什么样的“周模式”？

从 θ 的符号和大小看，IRL 学到的偏好可以总结为：

1）情绪层

- `valence_mean`：+0.65  
  → 更偏好「整体 valence 高」的一周；
- `valence_var`：−0.15  
  → 同样的均值下，更偏好波动较小（更稳定）的周；
- `valence_trend`：+0.01（接近 0）  
  → 对“正在变好”的周有轻微偏好，但影响很小。

2）行为层

- `social_level`：+0.22  
  → 一周内有更多社交（尤其是情人节准备期那种温暖社交）是加分项；
- `work_level`：+0.47  
  → 有一定工作强度是加分的，“有节奏的忙碌周”被偏好；
- `rest_level`：−0.19  
  → 在这条 60d 轨迹里，“高 rest 的周”多出现在 recovery 段，valence 中等略上，  
     IRL 在 “Valentine ≻ election 高压” 的主偏好下，暂时学成了略负权重；  
     这部分需要在后续分析里区分“良性恢复 vs 消极躺平”。

3）冲突 / 情绪劳动

- `conflict_level`：−0.78（绝对值很大）  
  → 情绪标签“复杂”出现频率越高，这一周被 IRL 明确视为“明显扣分”；  
  → 这直接对应选举阶段的高压政治讨论与情绪劳动。

4）故事阶段

- `phase_valentine`：+0.59  
  → 落在 Valentine 准备/高峰期的 weeks 被整体加分；
- `phase_election`：−1.13（绝对值最大的一项）  
  → 落在 election 阶段的 weeks 被强烈扣分；
- `phase_recovery`：−0.10（轻微负）  
  → recovery 段介于高/低之间，Reward 略偏下，但远不如 election 段那么负。

简短人话版：

> IRL θ 学到的是：  
>   - 最被偏好的，是「高 valence + 有节奏的忙碌 + 有社交 + 冲突少 + 在 Valentine 段」的 weeks；  
>   - 最被厌恶的，是「处在 election 高压阶段 + 冲突/复杂情绪很多」的 weeks。

这与我们在 IRL discovery 和故事阅读中的 qualitative 结论高度一致。

---

### 3. 和 BFI / 人格故事的对齐

结合 BFI（前测高 O/C/E/A、低 N，后测 C/O 略升、E/A 略降）：

1）外向性 E & 宜人性 A

预期：

- 高 E / A → 喜欢温暖、支持性社交，不喜欢冲突型社交；

θ 中：

- `social_level` > 0：社交相关周是加分项；
- `conflict_level` < 0（幅度大）：冲突/情绪劳动多的周被强烈扣分；

对应故事：

- 情人节准备阶段：高社交但几乎没有冲突 → reward 高；
- 选举阶段：社交场合里充满争论和情绪劳动 → reward 明显低。

这可以解读为：

> “一个高 E/A 的 Isabella，在 60d 生活 pattern 上体现为：  
>   她明显偏好『温暖社交』的 weeks，同时对『高冲突社交』有强烈负偏好。”

2）尽责性 C

预期：

- 高 C → 喜欢“有结构、有意义的工作 + 有节奏的忙碌”，讨厌“被无意义的高压工作压垮”；

θ 中：

- `work_level` 明显正：工作强度高一些通常是加分，而不是扣分；
- `phase_valentine` 正：情人节准备期忙碌但有意义，被 reward 加分；
- `phase_election` 强负：选举阶段的高压/信息过载/争吵被认为是负向；

和 BFI 后测里 C 略升（+0.0555）结合，可以讲成：

> “她在这 60 天里，强化了自己对『有结构、有意义的忙碌』的偏好，  
>  同时对『高压政治争议 + 情绪劳动』这类工作环境更加明确地说不。”

3）神经质 N 与情绪稳定性

预期：

- N 低且基本不变 → 情绪总体稳定，不倾向 dramatic 波动；

θ 中：

- `valence_var` 略负：variance 越大，reward 略降；
- `conflict_level` 明显负：高冲突 weeks 被明显视为负向；

可以解读为：

> “她偏好『平稳、少冲突』的一周，  
>  即便 valence 不是极高，只要不被情绪劳动拖入过山车，就更接近她的「好周」定义。”

4）开放性 O & 恢复/创作（未来扩展）

当前 φ(s_t) 中还没有显式的 “novelty / creative” 轴，  
但故事里的 art_show 段，valence 和行为模式相对中性/略高，  
可以后续在特征中加入 `creative_level` 一维，把 O 对创作/探索的偏好显式拉出来。  
目前 θ 没有直接反映 O 的变化，只能说：  
Valentine/选举阶段的强偏好/厌恶，与 O 的微小上升关系不大，更多是 C/E/A 的投影。

---

### 4. 和“普通回归” w_reg 的对比：θ 真的有多一层结构

为对比 IRL 与监督回归的差异，做了一个最简单 baseline：

- 用同一组特征 φ(s_t) 对 `r_t = window_valence` 做最小二乘回归：

  ```python
  w_reg, *_ = np.linalg.lstsq(phi, r, rcond=None)
  ```

得到：

```text
theta (IRL):
  valence_mean    : +0.6530
  valence_var     : -0.1525
  valence_trend   : +0.0123
  social_level    : +0.2187
  rest_level      : -0.1926
  work_level      : +0.4656
  conflict_level  : -0.7775
  phase_valentine : +0.5944
  phase_election  : -1.1292
  phase_recovery  : -0.0963

w_reg (regression to r_t):
  valence_mean    : +1.0000
  valence_var     : +0.0000
  valence_trend   : -0.0000
  social_level    : -0.0000
  rest_level      : +0.0000
  work_level      : -0.0000
  conflict_level  : +0.0000
  phase_valentine : +0.0000
  phase_election  : +0.0000
  phase_recovery  : -0.0000
```

可以看到：

- 回归 w_reg 几乎是一个「只看 valence_mean」的向量：  
  其它轴几乎全是 0；
- θ 则把：
  - phase_valentine / phase_election  
  - conflict_level / social_level / work_level  
  都纳入了「好 vs 坏 weeks」的结构中。

两者的相关系数：

```text
cor(theta, w_reg) ≈ 0.42
```

意味着：

- 它们都承认 “valence_mean 是重要的”（相关性不为 0）；  
- 但 θ 在「除了 valence 均值之外」的轴上引入了大量额外结构：  
  特别是 election / conflict / social / work 这些 story 相关维度。

可以说：

> 监督回归只学到：“valence 高的周分数高”；  
> 偏好 IRL 学到的 θ 则在同一个特征空间里，把  
>   – 社交质量、  
>   – 冲突/情绪劳动、  
>   – 不同故事阶段（Valentine vs election）  
> 这些因素显式地刻成了「价值观坐标轴」。

---

### 5. 小结：θ 作为“人格化偏好向量”的地位

综合这一轮自查：

1. θ 在 story 层面：
   - 清楚地把「情人节准备/高峰」与「选举高压段」拉开；
   - 高 reward weeks 的模式 = 高 valence + 高社交 + 有工作成就 + 冲突低 + Valentine 段；
   - 低 reward weeks 的模式 = election 段 + conflict_level 高。

2. θ 在人格层面：
   - 外向性/宜人性的高分 → 映射为 social_level 正、conflict_level 强负；
   - 尽责性的中高+略升 → 映射为 work_level 正、phase_election 强负；
   - 低神经质 → 映射为 valence_var 略负、对高冲突/高波动 weeks 的负偏好。

3. 和回归 baseline 的对比：
   - 回归只在 valence_mean 一维上发力；  
   - IRL θ 在多个语义维度上写出了“她真正偏好哪种 life pattern”的结构。

从这个角度看，θ 已经可以被视为：

> “Isabella 在这条 60d 生活轨迹上的『情绪价值观向量』”，  
> 它既受 BFI 人格前测约束，又通过偏好 IRL 把具体的 weekly pattern 学进来了。

后续如要写论文/设计文档，这一节可以自然组织成：

- 表 1：φ(s_t) 特征定义；  
- 表 2：θ_i（IRL） vs 各特征的定性解释；  
- 表 3：BFI 向量 vs θ 的映射关系；  
- 附图：按 R_θ 排序的 weeks 与 story 时间线对齐图。  

