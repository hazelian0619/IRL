# Isabella IRL 60‑Day Dataset (OpenAI v2)

This folder contains the **60‑day nightly IRL dataset** for  
**Isabella Rodriguez** (Hobbs Cafe owner in *the Ville*), generated in
November 2025 using the updated pipeline.

It supersedes the early 3‑/10‑day sample under `data/isabella_irl_3d_clean/`,
which is now kept only as a small illustrative subset for prompt / pipeline
ablation.

## How This Dataset Was Generated

- **World**: Stanford generative agents environment, map `the_ville`.
- **Agents**: 3 personas
  - `Isabella Rodriguez` (Hobbs Cafe owner)
  - `Maria Lopez`
  - `Klaus Mueller`
- **Base simulation**:  
  `external_town/environment/frontend_server/storage/July1_the_ville_isabella_maria_klaus-step-3-20`
- **Simulation driver**: `scripts/run_town_irl_days.py`
  - Example invocation (OpenAI backend):

    ```bash
    cd companion-robot-irl
    source ../openai.env.local
    python scripts/run_town_irl_days.py \
      --fork-sim July1_the_ville_isabella_maria_klaus-step-3-20 \
      --sim-code isabella_n3_live_irl_openai_60d_v2 \
      --days 60 \
      --sec-per-step 86400 \
      --output-root data/isabella_irl_60d_openai_v2
    ```

- **Nightly interview**: `agents/robot_interviewer.py`
  - L1–L6 structure (20:00 each day, local time)
    1. 今天过得怎么样？
    2. 今天发生了什么特别的事吗？
    3. 这些事情让你有什么感受？
    4. 今天最让你印象深刻的时刻是什么？
    5. 明天有什么计划或者期待吗？
    6. 给今天整体心情打分 (`Score: <1–10>`)
- **Emotion / mood model**: `story/isabella_story.py`
  - Encodes 60‑day phase schedule (Valentine prep / peak / art show / election / recovery)
  - Maintains latent `fatigue`, `stress`, `hope`
  - Computes a **corrected daily mood_score** combining:
    - phase base_score
    - positive / negative keywords in nightly transcript
    - accumulated fatigue / stress
    - Isabella’s self‑reported `Score: n`

### Nightly Interview Implementation (LLM API)

The nightly interview itself is implemented in `agents/robot_interviewer.py`:

- `load_agent_profile(agent_name)`:
  - Reads `data/personas/<agent>_biography_prompt.txt` and
    `data/personas/preset_personality.json`.
  - Provides a short biography + Big‑Five scores to the LLM as persona context.
- `RobotInterviewer`:
  - Uses the official `openai.OpenAI` client with an OpenAI‑compatible API
    (typically OpenRouter):

    ```python
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENROUTER_API_BASE") or os.getenv("OPENAI_BASE_URL"),
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", ""),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Companion Robot IRL"),
        },
    )
    ```

  - For each L1–L6 turn:
    - Builds a prompt that includes:
      - persona biography (truncated to ~800 chars),
      - current simulated date/time and story phase (`phase_for_day(day)`),
      - recent dialogue history,
      - the current L1–L6 question.
    - Calls `client.responses.create(model=..., input=prompt, ...)` to obtain
      Isabella’s reply, using an OpenAI‑compatible `/responses` API.
  - After the conversation:
    - Builds a second prompt that asks the LLM to summarise the day into a JSON:

      ```json
      {
        "behaviors": [{"time": "...", "description": "..."}],
        "dominant_emotion": "...",
        "emotion_reason": "...",
        "tone_intensity": 0.0-1.0,
        "mood_score": 1-10
      }
      ```

    - Parses the JSON, falling back to a regex on the transcript for
      `Score: <n>` if `mood_score` is missing.

For non‑Town experiments, the same interviewer can be used via
`scripts/run_daily_batch.py`, which simply loops over dates and writes
Phase‑2 data through `utils.data_collector.DataCollector`:

```bash
cd companion-robot-irl
source ../openai.env.local
python scripts/run_daily_batch.py \
  --agent isabella \
  --start-day 1 --days 60 \
  --start-date 2025-11-01 \
  --output-root data/isabella_irl_60d_openai_v2
```

## Folder Structure

- `conversations/day_XXX.md`  
  Full nightly dialogue (robot + Isabella, 12 turns) with header:
  - `agent`: `"Isabella"`
  - `date`: ISO datetime (20:00 on that simulated day)
  - `turns`: number of turns (should be 12 for L1–L6)

- `behaviors/day_XXX.json`  
  JSON with key `"behaviors"`:
  - each element: `{"time": "...", "description": "..."}`  
    summarising key events across morning / afternoon / evening.

- `emotions/day_XXX.json`  
  JSON with fields:
  - `"day"`: day index (1–60)
  - `"label"`: one of `["开心","放松","累","焦虑","失落","生气","平静","复杂"]`
  - `"intensity"`: float in `[0,1]`
  - `"reason"`: natural language justification

- `scores/mood_scores.csv`  
  Daily **latent mood score** (not equal to the raw L6 `Score: n`):
  - `day,score`
  - 1 row per day (1–60), integer in `[1,10]`

## Downstream Features & Multi‑Modal Fusion (Pre‑IRL)

All downstream feature extraction and fusion scripts operate directly on this
folder and write into `<root>/features/`. For Isabella v2 the typical pipeline is:

1. **Daily feature matrix (per‑day multi‑modal features)**  
   Script: `scripts/export_daily_features.py`  
   Uses: `features/basic_features.py`  
   Example:

   ```bash
   cd companion-robot-irl
   python3 scripts/export_daily_features.py --root data/isabella_irl_60d_openai_v2
   ```

   Outputs (under `data/isabella_irl_60d_openai_v2/features/`):
   - `X_daily.npy`          – (num_days, num_features) daily feature matrix  
   - `feature_names.json`   – names/semantics of each column in `X_daily.npy`

2. **Multi‑modal emotion fusion (text/behavior/emotion/score)**  
   Script: `scripts/export_fusion_features.py`  
   Uses: `features/emotion_fusion.py` +
   `features/emotion_llm_backend.py` (LLM‑based text/behavior sentiment),
   and `utils.dataset_loader.DailyDataset`.

   ```bash
   python3 scripts/export_fusion_features.py --root data/isabella_irl_60d_openai_v2
   ```

   For each day, we compute:
   - `P_text(day)`     – via LLM over transcript_md (or keyword fallback)  
   - `P_behavior(day)` – via LLM over behavior descriptions  
   - `P_emotion(day)`  – one‑hot from `emotions/day_XXX.json["label"]`  
   - `P_score(day)`    – from `mood_score` intervals  
   and fuse them into `P_fusion(day)` using accuracy‑based softmax weights.

   Outputs:
   - `text_probs.npy`       – (T,4) text‑view emotion probs  
   - `behavior_probs.npy`   – (T,4) behavior‑view probs  
   - `emotion_probs.npy`    – (T,4) label‑view probs  
   - `score_probs.npy`      – (T,4) score‑view probs  
   - `fusion_daily.npy`     – (T,4) fused probs aligned with `"积极","消极","中性","复杂"`  
   - `fusion_meta.json`     – accuracies, fusion weights, days

3. **Temporal features (7‑day sliding windows + EWMA baseline)**  
   Script: `scripts/export_temporal_features.py`  
   Uses: `features/temporal_features.py`.

   ```bash
   python3 scripts/export_temporal_features.py --root data/isabella_irl_60d_openai_v2
   ```

   If `fusion_daily.npy` exists, we use:
   - `valence(day) = P_fusion(积极) - P_fusion(消极)`  
   Otherwise, we fall back to `mood_score`. From this 1D sequence we export:

   - `rolling_stats.npy`   – (num_windows, 5)  
     mean / variance / max / min / trend over 7‑day windows  
   - `global_baseline.npy` – (1,) EWMA baseline over the full 60‑day sequence  
   - `decay_weights.npy`   – (T,) EWMA weights per day  
   - `temporal_meta.json`  – config and shapes

4. **Temporal embeddings & window‑level reward proxy**  
   Script: `scripts/export_temporal_embeddings.py`  
   Uses: `models/sequence_encoder.BiLSTMAttentionEncoder`.

   ```bash
   python3 scripts/export_temporal_embeddings.py --root data/isabella_irl_60d_openai_v2
   ```

   This script:
   - feeds `rolling_stats.npy` (T,5) and `global_baseline.npy` into a BiLSTM+attention encoder;  
   - computes per‑window embeddings `z_t` (seq embeddings);  
   - recomputes a 7‑day sliding‑window average of `valence` as a window‑level reward proxy.

   Outputs:
   - `temporal_embeddings.npy` – (T, D) weekly state embeddings `z_t`  
   - `window_valence.npy`      – (T,) weekly reward proxies (valence window average)  
   - `irl_meta.json`           – meta info (T, embedding_dim, reward_source, input/output paths)

These three steps (daily features → fusion → temporal features/embeddings) define the
**pre‑IRL feature pipeline**. All IRL‑related modules (`learning/irl_mvp.py`,
`learning/irl_preference.py` etc.) operate on the outputs in `features/`,
and can be considered “downstream consumers” of this dataset.

## BFI‑44 Pre / Post Reports

Isabella’s Big Five personality was measured with BFI‑44 before and after
the 60‑day IRL run:

- **Pretest (baseline)** – LLM‑assisted answers based on static persona:
  - `validation/Isabella_Rodriguez_pretest_REAL_LLM_*.json`
  - Generated via `agents/bfi_interviewer.py`
- **Posttest (IRL‑conditioned)** – answers after reading a 60‑day emotional
  diary summary:
  - `validation/Isabella_Rodriguez_posttest_IRL_REAL_LLM_*.json`
  - Generated via:

    ```bash
    python scripts/run_isabella_bfi_post_from_irl.py \
      --irl-root data/isabella_irl_60d_openai_v2 \
      --days 60 \
      --method llm
    ```

These reports contain:
- per‑item 1–5 Likert scores (`responses_scores`)
- detailed natural‑language rationales (`responses_detailed`)
- metadata (`test_type` = `"pretest_REAL"` / `"posttest_IRL_REAL"`)

## Relationship to Older Sample (`isabella_irl_3d_clean`)

- `data/isabella_irl_3d_clean/` was an early 3‑/10‑day sample used for prompt
  and pipeline tuning (sometimes described as a “placeholder 60‑day dataset”).
- `data/isabella_irl_60d_openai_v2/` is the **canonical 60‑day Isabella IRL
  dataset** used for Phase 2 / P1 experiments; new analyses and reviewers
  should treat this folder as the primary source.
