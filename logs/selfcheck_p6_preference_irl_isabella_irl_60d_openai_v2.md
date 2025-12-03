## selfcheck_p6：偏好 IRL 训练 θ – Isabella 60d v2

本轮执行对应《执行.md》中 **Phase D：偏好 IRL 训练 θ**。

目标是：在已经构造好的周级特征 φ(s_t) 和偏好对 (i ≻ j) 上，拟合一个线性偏好函数 `R_θ(s) = θ^T φ(s)`，并检查：

- θ 在每个特征轴上的符号和大小是否符合 story / 直觉；
- 在偏好对上的排序准确率是否足够高。

---

### 1. 实现位置与调用方式

- 偏好 IRL 核心逻辑：
  - `companion-robot-irl/learning/irl_preference.py`
    - 新增：
      - `fit_preference_reward(phi, pairs, l2_reg=1e-3, lr=1e-2, num_steps=500) -> theta`
      - `preference_accuracy(phi, pairs, theta) -> float`

- 训练脚本：
  - `companion-robot-irl/scripts/run_irl_preference_isabella.py`
    - 用法：

      ```bash
      cd companion-robot-irl
      python3 scripts/run_irl_preference_isabella.py --root data/isabella_irl_60d_openai_v2
      ```

**输入：**

- 周级特征：
  - `data/isabella_irl_60d_openai_v2/features/preference_features.npy` (54, 10)
  - `data/isabella_irl_60d_openai_v2/features/preference_feature_names.json`
- 周级偏好对：
  - `data/isabella_irl_60d_openai_v2/features/irl_preference_pairs.json`
    - pairs≈1091，对应 Phase C 的输出。

**输出：**

- 参数：
  - `data/isabella_irl_60d_openai_v2/features/irl_theta.npy` (10,)
- 每周 reward：
  - `data/isabella_irl_60d_openai_v2/features/irl_reward_weekly.npy` (54,)
- 控制台 summary（θ 和偏好准确率）。

---

### 2. 训练配置与核心公式

训练配置（脚本默认）：

- 步数：`num_steps = 800`
- 学习率：`lr = 1e-2`
- L2 正则：`l2_reg = 1e-3`

优化目标：

- 对每个偏好对 (i ≻ j)，定义：

  - `diff_ij = φ(s_i) - φ(s_j)`
  - `delta = θ^T diff_ij`
  - `Pθ(i ≻ j) = σ(delta) = 1 / (1 + exp(-delta))`

- 最大化：

  ```text
  L(θ) = Σ_{(i,j)} log σ(θ^T(φ_i - φ_j)) - λ||θ||^2
  ```

- 使用 batch 梯度上升，梯度为：

  ```text
  ∂/∂θ log σ(delta) = (1 - σ(delta)) * (φ_i - φ_j)
  ```

  所以总体更新为：

  ```text
  grad = mean_{(i,j)}[(1 - σ(delta_ij)) * (φ_i - φ_j)] - 2λθ
  θ ← θ + lr * grad
  ```

---

### 3. 学到的 θ：每个特征轴的偏好权重

训练输出（关键部分）：

```text
[INFO] Learned theta (per feature):
  θ[00] (valence_mean    ) = +0.6530
  θ[01] (valence_var     ) = -0.1525
  θ[02] (valence_trend   ) = +0.0123
  θ[03] (social_level    ) = +0.2187
  θ[04] (rest_level      ) = -0.1926
  θ[05] (work_level      ) = +0.4656
  θ[06] (conflict_level  ) = -0.7775
  θ[07] (phase_valentine ) = +0.5944
  θ[08] (phase_election  ) = -1.1292
  θ[09] (phase_recovery  ) = -0.0963
```

结合 φ(s_t) 定义，可读作：

- `valence_mean`：+0.65
  - 平均 valence 越高，这周越被偏好；
  - 与我们用 window_valence 定义 high/low reward 的设定一致。
- `valence_var`：−0.15
  - 情绪波动越大的周略微扣分；
  - 说明在同样的均值下，「更平稳」的周略偏好。
- `valence_trend`：+0.01（接近 0）
  - 轻微倾向于“正在变好”的周，但权重很小，可忽略。
- `social_level`：+0.22
  - 社交强度较高的周整体加分；
  - 与情人节筹备期（高社交+高 valence）的 pattern 一致。
- `rest_level`：−0.19
  - 这是一个稍微「反直觉」但与数据有关的现象：
    - 在这条 60d 轨迹里，高 reward 周大多是“高社交+高工作+少休息”的 valentine 段；
    - 真正“躺平恢复”的 days 多出现在 recovery phase，valence 中等略低；
  - 所以 IRL 在「Valentine ≻ 选举高压」这个主约束下，默认对 rest 的权重略负。
  - 这部分需要在 Phase E 和 BFI/故事做更细致解释（比如区分“恢复性休息”和“消极躺平”）。
- `work_level`：+0.47
  - 适度工作量强的周被偏好（忙碌但有意义的 weeks 被推上去了）；
  - 很符合情人节、艺术展阶段“有节奏的忙”的故事。
- `conflict_level`：−0.78（绝对值较大）
  - 情绪标签“复杂”的日占比越高，这周越明显扣分；
  - 这恰好对应选举高压阶段：大量「复杂」标签和情绪劳动；
  - IRL 很强地学会了“高 conflict weeks 是负向偏好”。
- `phase_valentine`：+0.59
  - 落在情人节 prep/peak 阶段的周整体加分；
  - 在表示“Valentine 模式 ≻ 其他模式”这块起重要作用。
- `phase_election`：−1.13（绝对值最大）
  - 落在选举阶段的周被强烈扣分；
  - 这相当于是一个“高压 election 周”的强负偏好轴。
- `phase_recovery`：−0.10（轻微负）
  - 整体偏轻微负，说明在当前偏好对构造下，recovery 段的 weeks 略落在中下游；
  - 这与 60d 轨迹中 recovery phase 的 valence 中等偏上、但不如情人节高的设定一致。

总体结构是：

> IRL 学到的偏好：  
>   - 高 valence、低 conflict、处在 Valentine 段、适度高工作/社交的 weeks → 高 reward；  
>   - 处在 election 段、conflict_level 高的 weeks → 强负 reward。

---

### 4. 在偏好对上的排序准确率

脚本输出：

```text
[INFO] Preference accuracy (R(i) > R(j) on pairs): 0.967
```

即：

- 在 1091 个偏好对 (i ≻ j) 中，有约 **96.7%** 满足 `R_θ(s_i) > R_θ(s_j)`；
- 远高于随机 50%；
- 说明这组 θ 确实把「Valentine 模式 ≻ election 模式」以及其他符合 margin 的偏好关系很好地拉开了，而不是只拟合噪声。

另外检查：

- 取 IRL reward 序列 `irl_reward_weekly.npy`，排序 Top5 / Bottom5：

  ```text
  Top-5 weeks by IRL reward:
    idx 0,1,2,3,4 对应 days 1–7, 2–8, 3–9, 4–10, 5–11
    R_θ ≈ 2.07–2.29，对应 r_val ≈ 0.77–0.82（情人节筹备阶段）

  Bottom-5 weeks by IRL reward:
    idx 28, 32–35 大致覆盖 days 29–40
    R_θ ≈ -1.44–-1.19，对应 r_val ≈ 0.14–0.26（选举高压阶段）
  ```

- 这说明 IRL 的最终排序结果与最初 IRL discovery 文档中“Valentine 高 reward 段 vs 选举低 reward 段”的故事是一致的。

---

### 5. 与 BFI / 人格主线的初步对齐（直觉层）

更系统的 BFI 对齐会在 Phase E 单独展开，这里先给一个直觉预览：

- 高 A / 高 E（宜人、外向）：
  - 期望：偏好温暖社交、讨厌冲突型社交；
  - θ 中：
    - `social_level` 为正；
    - `conflict_level` 权重大幅为负；
    - 与预期一致。
- 高 C（尽责）：
  - 期望：偏好“有结构的高工作量 + 有意义的付出”，讨厌“无休息的无意义透支”；
  - θ 中：
    - `work_level` 较大正值（忙碌是加分项）；
    - `phase_election` 强负（高压政治争论+情绪劳动被视为负向）；  
    - 对 `rest_level` 的权重略负，需要结合 recovery phase 的 valence 曲线去解释：目前数据里“高 rest 的周”未必代表“良性恢复”，而更多出现在略低 valence 的恢复/反思段。
- N（神经质）：
  - 期望：偏好稳定 vs 高波动；
  - θ 中：
    - `valence_var` 略为负：波动越大，reward 越低；
    - 搭配 `conflict_level` 负，表示“高冲突 + 高波动”的 weeks 被 IRL 判为明显不偏好。

后续可以在 Phase E 中把 θ 的这些分量与 BFI 前测/后测的具体数值做更精确对齐，这里只是确认 **θ 的整体结构方向是和我们对 Isabella 的人格故事兼容的**。

---

### 6. 小结：Phase D 完成度与下一步

当前 Phase D 的验收：

- 技术指标：
  - θ 已成功训练并落盘；
  - 偏好对上的排序准确率 ≈ 0.967，远高于随机；
  - IRL reward 排序的 Top/Bottom weeks 与原始 high/low reward 周（Valentine vs election）一致。

- 解释性：
  - θ 在各特征轴上的符号与大小，与 story 中“情人节忙碌被喜欢、选举高压被讨厌、冲突被强烈扣分”的直觉相吻合；
  - 已经可以开始写「她偏好哪种周模式」的故事性描述。

下一步（Phase E）将：

- 以 `irl_theta.npy` + `preference_feature_names.json` 为基础，  
  写一份专门的偏好向量解释文档（包括人格对齐、与监督回归 baseline 的对比）；
- 把这一部分整理成可以直接放入 IRL 设计报告 / 论文 skeleton 的段落。  

