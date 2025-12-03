# Companion Robot IRL Project

This repository now distinguishes between two execution modes:

- **real** (default): connects to the Stanford Town environment through
  `TownBridge`, expects data to be gathered via shell scripts and ingestion
  tools, and forbids importing any legacy fake-data modules.
- **simulated**: explicitly opt-in mode (`python ../main.py --mode simulated` or
  `SIMULATED_LEGACY_MODE=1 ...`) that unlocks the historical IRL toy pipeline
  under `simulated_legacy/`.

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/start_town.sh` | Launch the Django frontend on port 8000. |
| `scripts/start_town_backend.sh` | Launch `reverie.py` (backend) after validating `utils.py`. |
| `scripts/ingest_real_data.py` | Copy REAL artifacts from `external_town/.../storage` into `data/real_ingest/<simulation>/<timestamp>/` with metadata. |
| `tools/audit_data_sources.py` | Scan metadata files and fail when non-REAL labels slip into REAL folders. |

After collecting data, run

```bash
python tools/audit_data_sources.py --roots data/real_ingest --require-agent --require-simulation
```

to prove that every snapshot includes metadata and carries the right
`data_type` label.

## Legacy simulator

Legacy IRL modules (`agents/alice.py`, `agents/interactions.py`, `learning/*`)
are now parked under `simulated_legacy/` and guarded by the
`SIMULATED_LEGACY_MODE` flag. Attempting to import them without explicitly
exporting `SIMULATED_LEGACY_MODE=1` will raise an error. To run the old
experiments:

```bash
cd /Users/pluviophile/IRLv2
python main.py --mode simulated --exp 1 --user alice --days 30
```

Any production workflow must continue to operate in `--mode real`, orchestrated
through the Town scripts and ingestion utilities listed above.

## 60‑day synthetic dataset (Phase 2 / P1)

For the IRL prototype we also maintain a **60‑day synthetic nightly dataset**
driven by LLM interviews instead of live Town runs. This follows the design
described in the planning doc `进度/12`:

- Persona & Big Five preset: `data/personas/preset_personality.json` and
  `data/personas/*_biography_prompt.txt`.
- Nightly interview driver (L1–L6 questions, 20:00 every day):
  `agents/robot_interviewer.py`.
- Multi‑modal writer (text + behaviors + emotions + mood score):
  `utils/data_collector.py`.
- Story‑driven mood curve for Isabella (phase schedule + fatigue/stress/hope):
  `story/isabella_story.py`.
- One‑day entry point (debug / manual run):
  `scripts/schedule_daily_talk.py`.
- Batch generator for N days (e.g. 3/7/60 days):
  `scripts/run_daily_batch.py`.

Example datasets under `data/`:

- `isabella_irl_60d_openai_v2/` – **current 60‑day P1 dataset** for Isabella
  (nightly talks + behaviors + emotions + `scores/mood_scores.csv`), generated
  via `run_town_irl_days.py` + `RobotInterviewer` using an OpenAI‑compatible
  backend.
- `isabella_irl_3d_clean/` – earlier 3‑/10‑day sample used for prompt and
  pipeline tuning; now kept as a small illustrative subset / ablation dataset.
- `alice_irl_60d/` – work‑in‑progress Alice variant, generated with the same
  tools but currently only a few days populated.

These artifacts are intended as **P1 synthetic trajectories** for downstream
emotion / IRL modelling; any production‑grade IRL work should still treat the
Stanford Town pipeline as the source of REAL data.

## Weekly Preference IRL on Isabella 60d

On top of the 60‑day synthetic dataset for Isabella, we implement a
weekly preference‑based IRL layer:

- `features/preference_features.py` and
  `scripts/export_preference_features.py` build 10‑D weekly state vectors
  φ(s_t) from valence, behaviors, and story phase.
- `learning/irl_preference.py` and
  `scripts/run_irl_preference_isabella.py` learn a linear reward
  R_θ(s) = θᵀφ(s) from week‑to‑week preferences derived from
  `window_valence.npy`.
- `scripts/inspect_irl_preference_pairs.py` and
  `scripts/inspect_irl_value_isabella.py` help inspect preference pairs
  and the discounted value curve V_θ(t).
- IRL artifacts are saved under
  `data/isabella_irl_60d_openai_v2/features/` (preference_features,
  irl_theta_v2, irl_reward_weekly_v2, state_embeddings, etc.) and
  figures for the paper/live demo under `figures/`.

This layer turns the 60‑day multimodal trace into an interpretable weekly
preference model that can be aligned with Isabella's Big‑5 profile.
