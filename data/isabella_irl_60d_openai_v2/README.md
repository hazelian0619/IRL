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

