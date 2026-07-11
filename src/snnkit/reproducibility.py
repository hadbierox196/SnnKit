"""Reproducibility conventions used throughout snnkit.

Every notebook / experiment script should call `set_seed` at the top and
call `get_package_versions` when reporting results, so runs are
reconstructable later. This module is intentionally tiny: reproducibility
should be a five-second habit, not a framework.
"""

from __future__ import annotations

import importlib.metadata as _metadata
import platform
import random
import sys
from typing import Any

import jax
import numpy as np

#: The default seed used across snnkit examples, notebooks, and tests
#: unless explicitly overridden. Fixed here once so every deliverable
#: in the roadmap uses the same convention instead of ad hoc seeding.
DEFAULT_SEED = 0


def set_seed(seed: int = DEFAULT_SEED) -> jax.Array:
    """Seed Python's `random`, NumPy, and return a fresh JAX PRNGKey.

    JAX does not use global random state, so "seeding JAX" really means:
    return a `PRNGKey` derived from `seed` that the caller threads through
    every stochastic operation (`jax.random.split`, `jax.random.normal`, ...).

    Returns:
        A `jax.random.PRNGKey(seed)` for the caller to split/consume.
    """
    random.seed(seed)
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)


def get_package_versions() -> dict[str, Any]:
    """Return a dict of versions/environment info worth logging with results.

    Intended to be printed or JSON-dumped alongside any reported number
    (benchmark, accuracy, fit quality) so results are reconstructable.
    """
    packages = ["jax", "jaxlib", "optax", "sympy", "numpy"]
    versions: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in packages:
        try:
            versions[pkg] = _metadata.version(pkg)
        except _metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    versions["jax_backend"] = jax.default_backend()
    versions["jax_devices"] = [str(d) for d in jax.devices()]
    return versions
