"""Pigment sets and synthetic lab history."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .pigments import COLORANTS, N_PIGMENTS, WHITE

MAX_COLORANTS = 3


def pigment_sets(max_colorants: int = MAX_COLORANTS) -> list[tuple[int, ...]]:
    """Every combination of 1..max_colorants colorants (white is always available)."""
    sets: list[tuple[int, ...]] = []
    for k in range(1, max_colorants + 1):
        sets.extend(combinations(COLORANTS, k))
    return sets


def sample_recipes(n: int, rng: np.random.Generator,
                   size_weights: tuple[float, float, float] = (1.0, 0.35, 0.12)) -> tuple[np.ndarray, np.ndarray]:
    """Random recipes like a lab's history: 1-3 colorants let down with white.

    `size_weights` is the relative weight of each 1-, 2- and 3-colorant pigment set.
    Returns concentrations (n, P) summing to 1 and a 0/1 matrix of colorants used.
    """
    sets = pigment_sets()
    sizes = np.array([len(s) for s in sets])
    weights = np.asarray(size_weights, dtype=float)[sizes - 1]
    choice = rng.choice(len(sets), size=n, p=weights / weights.sum())
    conc = np.zeros((n, N_PIGMENTS))
    used = np.zeros((n, N_PIGMENTS), dtype=np.int8)
    for i, idx in enumerate(choice):
        s = list(sets[idx])
        load = 10 ** rng.uniform(-2.2, -0.1)                 # 0.6% .. 80% colorant
        conc[i, s] = load * rng.dirichlet(np.full(len(s), 0.9))
        conc[i, WHITE] = 1.0 - load
        used[i, s] = 1
    return conc, used


def calibration_panels() -> np.ndarray:
    """Standard tints a lab sprays when a new pigment lot arrives: one per colorant."""
    conc = np.zeros((len(COLORANTS), N_PIGMENTS))
    for row, i in enumerate(COLORANTS):
        share = 0.02 if i == 1 else 0.08          # carbon black is far stronger
        conc[row, i] = share
        conc[row, WHITE] = 1.0 - share
    return conc
