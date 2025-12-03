## selfcheck_p8：周级 IRL reward + RL value 曲线 – Isabella 60d v2

这一轮是把前面学到的 IRL reward R_θ(s_t) 放到一个最小的 RL 框架里看：

- 把 54 个周窗口视为一条 Markov Reward Process（MRP）轨迹；
- 用折扣因子 γ 计算从每一周出发的长期回报 V_θ(t)；
- 观察 R_θ(s_t) / V_θ(t) 在 Valentine / art_show / election / recovery 四个阶段的形状。

这一步不改变任何参数，只是给现有 IRL 结果加上一层 “value function” 的 RL 解释。

---

### 1. 实现位置与调用方式

- 新增脚本：
  - `companion-robot-irl/scripts/inspect_irl_value_isabella.py`
    - 用法：

      ```bash
      cd companion-robot-irl
      python3 scripts/inspect_irl_value_isabella.py --root data/isabella_irl_60d_openai_v2
      ```

    - 默认使用：
      - `features/irl_reward_weekly_v2.npy`（full-ranking weighted IRL 的 R_θ(s_t)）；
      - 若不存在，则退回 `features/irl_reward_weekly.npy`。
    - 折扣因子：
      - 默认 γ=0.95，可通过 `--gamma` 调整。

**内部逻辑：**

1. 加载周级 reward 序列：

   - `r[t] = R_θ(s_t)`，`t=0..T-1`，T=54。

2. 通过 `DailyDataset` 和 `week_day_windows_from_sequence_length` 确定：

   - 每个周 t 对应的 day 范围 (d_start, d_end)；
   - 用 `phase_for_day((d_start+d_end)/2)` 标注其 story 阶段：
     - `valentine_prep` / `valentine_peak` / `art_show` / `election` / `recovery`。

3. 计算 discounted value：

   定义单轨迹 MRP 上的 V_θ(t)：

   ```text
   V_θ(t) = Σ_{k=t..T-1} γ^{k-t} * r[k]
   ```

   用从后往前 DP 计算：

   ```python
   running = 0
   for t in reversed(range(T)):
       running = r[t] + gamma * running
       V[t] = running
   ```

4. 打印表格（每周一行）：

   - `week_idx | days | phase | R_θ(s_t) | V_θ(t)`。

5. 按阶段汇总：

   - 对每个 phase 计算 mean R、mean V。

---

### 2. 关键输出（局部）与直观解读

运行（γ=0.95）后的前几行：

```text
[INFO] Dataset root       : data/isabella_irl_60d_openai_v2
[INFO] Reward source      : irl_reward_weekly_v2.npy
[INFO] Num weeks (T)      : 54
[INFO] Discount factor γ  : 0.95

week_idx | days      | phase            | R_theta(s_t)   | V_theta(t)
---------+-----------+------------------+---------------+-----------
       0 |  1- 7   | valentine_prep   |     +2.2163   |  +15.5970
       1 |  2- 8   | valentine_prep   |     +2.1869   |  +14.0850
       2 |  3- 9   | valentine_prep   |     +2.1227   |  +12.5243
       3 |  4-10   | valentine_prep   |     +2.0935   |  +10.9490
       4 |  5-11   | valentine_peak   |     +2.0019   |   +9.3216
       5 |  6-12   | valentine_peak   |     +1.9985   |   +7.7049
       6 |  7-13   | valentine_peak   |     +1.9986   |   +6.0068
       7 |  8-14   | art_show         |     +1.8311   |   +4.2192
       8 |  9-15   | art_show         |     +1.7398   |   +2.5138
       9 | 10-16   | art_show         |     +1.6810   |   +0.8148
      10 | 11-17   | art_show         |     +0.9300   |   -0.9119
      11 | 12-18   | art_show         |     +0.8387   |   -1.9388
      12 | 13-19   | art_show         |     +0.7803   |   -2.9237
      ...
      19 | 20-26   | art_show         |     -0.2767   |  -11.4371
      20 | 21-27   | art_show         |     -0.5088   |  -11.7478
      21 | 22-28   | art_show         |     -0.5995   |  -11.8305
      22 | 23-29   | election         |     -0.8484   |  -11.8221
      23 | 24-30   | election         |     -0.9579   |  -11.5512
      24 | 25-31   | election         |     -1.1281   |  -11.1509
      ...
```

末尾部分（recovery）示意：

```text
      41 | 42-48   | recovery         |     +0.5500   |   +4.0304
      42 | 43-49   | recovery         |     +0.6384   |   +3.6636
      43 | 44-50   | recovery         |     +0.6095   |   +3.1844
      44 | 45-51   | recovery         |     +0.5770   |   +2.7105
      45 | 46-52   | recovery         |     +0.4895   |   +2.2458
      46 | 47-53   | recovery         |     +0.4278   |   +1.8487
      47 | 48-54   | recovery         |     +0.5992   |   +1.4958
      48 | 49-55   | recovery         |     +0.3126   |   +0.9438
      49 | 50-56   | recovery         |     +0.2742   |   +0.6644
      50 | 51-57   | recovery         |     +0.2833   |   +0.4107
      51 | 52-58   | recovery         |     +0.1098   |   +0.1340
      52 | 53-59   | recovery         |     +0.0375   |   +0.0255
      53 | 54-60   | recovery         |     -0.0126   |   -0.0126
```

从 RL 视角的直观：

- 在 valentine_prep / peak 段：
  - 即时 reward R_θ(s_t) ≈ 2.0+；
  - 从最开始的周看，长期回报 V_θ(t) ≈ 15 → 6，逐渐衰减；
  - 表达的是：“从这几周任一周往后看，整体是一个很高价值的轨迹开端”。

- 在 art_show 段：
  - 早期 art_show 周 reward 仍为正（≈0.8–1.8），但 V_θ(t) 已经跌到负；
  - 表明：“虽然这一周本身还不错，但考虑到后面即将到来的 election 高压周，整段未来的折扣回报开始变差”。

- 在 election 段：
  - 即时 reward 为负（≈-1.0 左右），V_θ(t) 长时间保持 ≈-11 ~ -5 的低谷；
  - 这是整个 60d 中“长期价值最低”的一段。

- 在 recovery 段：
  - 即时 reward 又回到轻微正值（≈0.1–0.6），V_θ(t) 从负值缓慢爬升到接近 0；
  - 说明：“从恢复期任何一点往后看，未来整体已经不再严重负值，但也没有回到 valentine 那样的高长期收益”，更像是一个“修复、归零”的尾巴。

简要的 phase-level summary：

脚本会给出每个阶段的 mean R / mean V，例如：

```text
[INFO] Phase-level summary (mean R, mean V):
  - valentine_prep  : mean_R=+2.1548, mean_V=+13.2888
  - valentine_peak  : mean_R=+1.9996, mean_V=+7.6778
  - art_show        : mean_R=+0.7708, mean_V=-4.9039
  - election        : mean_R=-1.1676, mean_V=-7.2810
  - recovery        : mean_R=+0.1426, mean_V=+1.8836
```

解释：

- **Valentine 段**：即时 + 长期都是最高的——“高 hedonic + 高长期价值”的阶段；
- **Art_show 段**：即时 reward 仍偏正，但长期 V 为负——“当下不错，但因未来选举高压在前方，整段 trajector 的期望就开始被拉低”；
- **Election 段**：即时/长期都是最负——“高压情绪劳动周”的长期价值谷底；
- **Recovery 段**：即时略正，长期略正——“在负谷之后缓慢爬回 zero 的修复段”，但没有回到 valentine 的高度。

这给了你一个 RL 语言的描述：

> 从任何一周 t 往后看，V_θ(t) 描述的是“如果照现在这条生活轨迹走下去，你这一周开始的人生 prefix 的价值高不高”。

在这个意义上：  
- Valentine 段不仅是“当下快乐”，也是“从这里开始的人生 prefix 最漂亮”的段落；  
- Election 段则是“当前痛苦 + prefix 压得很低”的段落；  
- Recovery 段是在一个“从负值逐渐向零修复”的轨道上。

---

### 3. 这一步的 RL 味道在哪？

技术上，我们做了两件事：

1. 明确地把周级 IRL reward 放进了一个 Markov Reward Process 视角：
   - 状态序列：s₀..s₅₃（周）；  
   - reward R_θ(s_t) 来自 IRL；  
   - 策略固定（真实人生轨迹）；  
   - 值函数 V_θ(t) 是这个固定策略下的折扣回报。

2. 用 V_θ(t) 的形状来解释“不同阶段的长期情绪价值”：
   - 不只是看每个周的好坏（R_θ），  
   - 也看从这周起，往后所有周叠加起来的长程效应（V_θ）。

这和“一步 MDP 的 IRL”相比，多了一层“return / value”的 RL 结构；  
而不需要虚构一个复杂的 P(s'|s,a) 或 action 集合。

---

### 4. 下一步的可能扩展（可选）

如果需要进一步强化 RL 味道，可以考虑：

- 把 V_θ(t) 和原来的 valence / mood_score 曲线叠加比较：  
  看 “短期情绪起伏 vs 长期情绪价值” 的差异；
- 定义一些简单的“策略变体”（例如调换 election 和 recovery 的顺序），  
  在这个 MRP 上用同一个 R_θ 计算 counterfactual 的 V_θ(t)，  
  以序列偏好形式说明 “当前这条 60d 轨迹是否优于一些直观的反例”。

这些都可以在不改变已有 IRL 训练结果的情况下，用 RL 语言来加一层解释。

当前这一轮（p8）的目标已经达到：  
- 在不造假的前提下，引入了一个最小的、形式上自洽的 RL 视角；  
- 并且可以用 R_θ + V_θ 的两条曲线讲清楚：  
  “这 60 天里，每个阶段在 Isabella 的长期情绪价值模型里处于什么位置”。  

