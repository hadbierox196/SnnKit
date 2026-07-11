"""snnkit: a JAX-native spiking neural network simulator.

Combines biologically faithful, equation-based modeling with GPU-scale
performance via JAX, and supports training via both surrogate-gradient
BPTT and local learning rules (STDP, SuperSpike).
"""

from snnkit._version import __version__
from snnkit.reproducibility import get_package_versions, set_seed

__all__ = ["__version__", "set_seed", "get_package_versions"]
