"""Guard rail that keeps the legacy simulator behind an explicit opt-in.

Set ``SIMULATED_LEGACY_MODE`` to ``1``, ``true``, ``yes`` or ``sim`` before you
import any of these modules. Without that flag, importing this package raises a
``RuntimeError`` so legacy code that generates synthetic data cannot leak into
the production entry points by accident.

Example::

    SIMULATED_LEGACY_MODE=1 python main.py --mode simulated

"""

from __future__ import annotations

import os


_FLAG = "SIMULATED_LEGACY_MODE"
_ALLOWED_VALUES = {"1", "true", "yes", "sim"}


def legacy_mode_enabled() -> bool:
    """Return True when the simulation-only modules are allowed to load."""

    raw_value = os.environ.get(_FLAG, "")
    return raw_value.strip().lower() in _ALLOWED_VALUES


if not legacy_mode_enabled():
    raise RuntimeError(
        "simulated_legacy modules are disabled. Set SIMULATED_LEGACY_MODE=1 or "
        "run `python main.py --mode simulated` to access the legacy simulator."
    )


__all__ = ["legacy_mode_enabled"]
