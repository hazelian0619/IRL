多模态情绪融合：LLM Backend 升级设计（Text & Behavior）
===============================================

背景与目标
----------

现有的多模态情绪融合实现中，文本与行为两个模态的日级情绪预测器采用的是极简规则：

- 文本模态 `_text_probs`：仅基于 `pos_word_count` / `neg_word_count` 的 if/else；
- 行为模态 `_behavior_probs`：仅基于 `beh_social_count` / `beh_work_count` / `beh_rest_count` / `mood_score` 的 if/else；

问题：

- 完全无法利用句子结构与语义（否定、对比、讽刺等），在复杂场景（如选举期）表达力不足；
- 行为模态只看到行为计数，看不到行为描述本身（争论 vs 温暖交流）；
- 这会在 IRL 高压阶段放大噪声，削弱我们在 z_t 空间做模式发现的质量。

目标：

在不改动多模态融合主线（4 模态 → 精度 → softmax 权重 → P_fusion → valence）的前提下：

1. 将文本模态 `P_text(day)` 从“词数规则”升级为“统一 LLM 情绪分类 backend 的输出”；  
2. 将行为模态 `P_behavior(day)` 从“计数规则”升级为“行为描述文本经同一 LLM backend 推理后的输出”；  
3. 保持 `_emotion_probs`（emotions JSON）与 `_score_probs`（mood_score）不变，作为 teacher 模态；
4. 在未配置 LLM 或 LLM 失败时，自动退回到原有规则化实现，保证 pipeline 稳定。

总体结构
--------

升级后的 compute_fusion 逻辑可以抽象为：

```text
for each day:
    rec = DailyDataset.get_day(day)
    feats = extract_daily_features(rec)

    # 文本模态
    P_text_LLM  = llm_emotion_probs_from_text(rec.transcript_md)
    P_text      = P_text_LLM if not None else _text_probs(feats)

    # 行为模态
    P_beh_LLM   = llm_emotion_probs_from_behaviors(rec.behaviors)
    P_behavior  = P_beh_LLM if not None else _behavior_probs(feats)

    # 其余两模态
    P_emotion   = _emotion_probs(rec.emotion["label"])
    P_score     = _score_probs(rec.mood_score)

    # 融合与后续流程保持不变：
    #  - 计算各模态 accuracy；
    #  - softmax(acc) → 权重 w；
    #  - P_fusion(day) = Σ w_mod * P_mod(day)；
    #  - fusion_valence(P_fusion) → valence；
    #  - temporal_features / IRL。
```

技术方案：统一的 LLM 情绪 backend
-------------------------------

设计约束：

- 不在当前工程中重新训练大模型或复杂分类器；
- 复用已有的 LLM 调用封装（external_town 中的 `call_llama`），方便在 OpenRouter / 本地 Llama / 兼容 OpenAI API 环境间切换；
- 使用环境变量控制是否启用 LLM backend，避免在无 key 环境意外触发网络调用。

实现要点：

1. 新增模块：`features/emotion_llm_backend.py`

   职责：

   - 提供两个函数：

     ```python
     llm_emotion_probs_from_text(text: str) -> Optional[np.ndarray]
     llm_emotion_probs_from_behaviors(behaviors: List[Dict]) -> Optional[np.ndarray]
     ```

   - 返回：
     - 若启用 LLM backend 且调用成功：长度为 4 的概率向量 `[P(积极), P(消极), P(中性), P(复杂)]`；
     - 若未启用或解析失败：返回 None，由上层回退到规则版。

   - 启用开关：

     ```text
     IRL_FUSION_USE_LLM_BACKEND ∈ {"1","true","on","yes"}  → 启用
     其他 / 未设置                                           → 不启用
     ```

2. 复用 existing LLM 封装：`call_llama`

   - 从 `external_town.reverie.backend_server.persona.prompt_template.gpt_structure`
     import `call_llama`；
   - 该函数已经封装了：
     - OpenRouter 模式（兼容 OpenAI API）；
     - 本地 Llama(Ollama) 模式；
     - fallback 行为（缺少 key / 请求失败时不抛死）。

3. 统一的 JSON 输出格式

   两个函数都通过 prompt 要求 LLM 输出严格 JSON：

   ```json
   {
     "积极": 0.7,
     "消极": 0.1,
     "中性": 0.1,
     "复杂": 0.1
   }
   ```

   - 四个键固定为 `"积极"、"消极"、"中性"、"复杂"`；
   - 值为 0–1 浮点数，和约等于 1；
   - 不允许输出额外文本。

   在 `emotion_llm_backend` 中：

   - 用字符串切片提取第一个 `{` 到最后一个 `}`，以防前后有冗余；
   - `json.loads` 解析；
   - 按 canonical label 顺序填入向量；
   - 若所有值非正或解析失败 → 返回 None。

4. 行为文本的构造

   将一天的 `behaviors` 列表转换为文本：

   ```text
   "上午: ...描述...\n下午: ...描述...\n晚上: ...描述..."
   ```

   - 只基于 `time` 与 `description` 字段；
   - 若缺少 `time` 则仅用描述；
   - 用换行分隔，方便 LLM 理解是一天内的若干行为片段。

compute_fusion 中的改动
------------------------

原本（简化）：

```python
feats = extract_daily_features(rec)
probs_text.append(_text_probs(feats))
probs_beh.append(_behavior_probs(feats))
probs_em.append(_emotion_probs(rec.emotion["label"]))
probs_score.append(_score_probs(rec.mood_score))
```

升级后：

```python
feats = extract_daily_features(rec)

# 文本模态：LLM 优先，失败退回规则版
llm_text = llm_emotion_probs_from_text(rec.transcript_md)
if llm_text is not None:
    probs_text.append(llm_text)
else:
    probs_text.append(_text_probs(feats))

# 行为模态：LLM 优先，失败退回规则版
llm_beh = llm_emotion_probs_from_behaviors(rec.behaviors)
if llm_beh is not None:
    probs_beh.append(llm_beh)
else:
    probs_beh.append(_behavior_probs(feats))

probs_em.append(_emotion_probs(rec.emotion["label"]))
probs_score.append(_score_probs(rec.mood_score))
```

后续步骤（accuracy → softmax 权重 → P_fusion → valence）不做任何修改。

使用方式与注意事项
------------------

1. 环境配置：

   - 保持现有 OpenRouter / OpenAI / Ollama 配置不变（`openai.env.local`、TOWN_ 前缀等）；
   - 仅在需要启用 LLM backend 时增加：

     ```bash
     export IRL_FUSION_USE_LLM_BACKEND=true
     ```

   - 若未配置或 LLM backend 不可用，则自动退回原有规则版。  

2. 性能与成本：

   - 每天最多调用 2 次 LLM（一次对话文本，一次行为文本）；  
   - 对 60 天数据最多 120 次调用，可通过：
     - 缩短文本（截断对话、行为描述长度）；
     - 调整 LLM 模型（例如使用较小模型或本地 Llama），平衡成本。

3. 兼容性：

   - 如果 external_town 未正确安装或 `call_llama` 不可用，`emotion_llm_backend` 会自动关闭 LLM backend；
   - 所有已有脚本（`export_fusion_features.py` 等）依旧可运行，只是默认仍使用规则实现。

预期收益
--------

- 文本模态：
  - 从关键词计数升级为语义级情绪判断，可以处理：
    - 复杂句式（“虽然累，但很值得”）；
    - 长期情绪总结（晚间对话中的反思）；
    - 更细腻的正负/复杂情绪。

- 行为模态：
  - 不再只看 social/work/rest 计数，而是：
    - 区分“温暖社交 vs 激烈争论”；  
    - 感知高压工作 vs 创造性活动；  
    - 更合理地为“高压长日子”输出偏负/复杂。

- 多模态融合：
  - 在不更改融合结构的前提下，提升 `P_text` 与 `P_behavior` 的质量；
  - 减少 IRL 后续在 z_t 空间看到的“假高/假低 reward 模式”，让模式发现更贴近 story。

与后续工作的关系
----------------

- 本次升级仅作用于“第一层：单模态日级情绪预测”的实现方式；
- 不改变：
  - emotions / mood_score 模态；
  - accuracy 计算与 softmax 权重；
  - fusion_daily → valence → temporal_features → IRL 的主线；
- 后续若需要进一步工业化：
  - 可以在 LLM backend 基础上，替换为专门训练的文本/行为情绪模型（使用当前 LLM 输出或 emotions label 作为 teacher）；
  - 也可以将融合层从 static softmax(acc) 升级为轻量可训练的 fusion 网络。

总结
----

通过引入 `features/emotion_llm_backend.py` 并在 `emotion_fusion.compute_fusion` 中优先调用
LLM backend，我们实现了：

- 文本与行为两个模态共享同一套工业级情绪分析引擎；  
- 在未配置 LLM 时保持原有规则版行为不变；  
- 为 IRL 上层提供更可信的 `P_text(day)` 和 `P_behavior(day)`，而无需重写后续 pipeline。

