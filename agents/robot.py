"""Legacy robot policy placeholder.

The IRL-based policy that produces synthetic suggestions lives in
``simulated_legacy.agents.robot`` now. Keeping this stub prevents accidental
imports inside new Town-backed pipelines.
"""

raise RuntimeError(
    "agents.robot has been removed. Import simulated_legacy.agents.robot "
    "after enabling SIMULATED_LEGACY_MODE instead."
)
