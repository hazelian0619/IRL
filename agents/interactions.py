"""Legacy interaction loop placeholder.

All fake feedback loops now live under ``simulated_legacy.agents``. Importing
this module without explicitly opting into simulation mode is considered an
error because it would mix fabricated data with the TownBridge workflow.
"""

raise RuntimeError(
    "agents.interactions is disabled. Use simulated_legacy.agents.interactions "
    "after setting SIMULATED_LEGACY_MODE=1."
)
