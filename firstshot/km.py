"""Two-constant Kubelka-Munk theory with Saunderson surface correction.

For an opaque paint film with mixture absorption K and scattering S:
    R_inf = 1 + K/S - sqrt((K/S)^2 + 2 K/S)
computed here in the numerically stable form 1 / (1 + q + sqrt(q^2 + 2q)).
Saunderson converts the internal reflectance to what a spectrophotometer reads.
"""
from __future__ import annotations

import numpy as np

K1 = 0.03  # external surface reflection
K2 = 0.6   # internal surface reflection


def r_infinity(k_mix: np.ndarray, s_mix: np.ndarray) -> np.ndarray:
    q = k_mix / np.maximum(s_mix, 1e-9)
    return 1.0 / (1.0 + q + np.sqrt(q * q + 2.0 * q))


def saunderson(r_inf: np.ndarray) -> np.ndarray:
    return K1 + (1.0 - K1) * (1.0 - K2) * r_inf / (1.0 - K2 * r_inf)


def reflectance(conc: np.ndarray, K: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Reflectance (n, 31) for recipes `conc` (n, P) whose rows sum to 1."""
    conc = np.atleast_2d(conc)
    return saunderson(r_infinity(conc @ K, conc @ S))
