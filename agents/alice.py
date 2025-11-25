"""Deprecated entry point for the old Alice simulator.

The implementation now lives in ``simulated_legacy.agents.alice`` so that the
fake-feedback pipeline never sneaks into production by accident. Import this
module only after setting ``SIMULATED_LEGACY_MODE=1`` **and** after explicitly
acknowledging that you are running in simulation mode (for example via
``python main.py --mode simulated``).
"""

raise RuntimeError(
    "agents.alice has been removed from the production surface. Import "
    "simulated_legacy.agents.alice after setting SIMULATED_LEGACY_MODE=1 or "
    "use `python main.py --mode simulated` instead."
)
