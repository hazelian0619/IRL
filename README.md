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

- `isabella_irl_3d_clean/` – current 60‑day P1 dataset (nightly talks +
  behaviors + emotions + `scores/mood_scores.csv`).
- `alice_irl_60d/` – work‑in‑progress Alice variant, generated with the same
  tools but currently only a few days populated.

These artifacts are intended as **P1 synthetic trajectories** for downstream
emotion / IRL modelling; any production‑grade IRL work should still treat the
Stanford Town pipeline as the source of REAL data.
