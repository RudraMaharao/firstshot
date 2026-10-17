"""A virtual paint lab that plays the role of physical reality.

It adds effects that plain Kubelka-Munk ignores: pigment saturation at high
loading (flocculation), concentration-dependent absorption band shifts,
titanium-white scattering loss when organic pigments are present, lot-to-lot
pigment strength differences and spectrophotometer noise.

The models never see these parameters. They learn only from "measured" panels,
exactly as they would from a real lab's history.
"""
from __future__ import annotations

import numpy as np

from . import km
from .pigments import K_matrix, N_PIGMENTS, S_matrix, WHITE, palette

HISTORICAL_BATCH = 11   # pigment lots used for the historical training data
CURRENT_BATCH = 23      # pigment lots on the shelf today


class VirtualLab:
    def __init__(self, batch: int = HISTORICAL_BATCH, noise: float = 0.002):
        hidden = np.random.default_rng(1234)              # fixed physics, same for every batch
        self.flocculation = hidden.uniform(1.0, 3.0, N_PIGMENTS)
        self.band_shift = hidden.uniform(-12.0, 12.0, N_PIGMENTS)  # nm per unit concentration
        self.flocculation[WHITE] = 0.0
        self.band_shift[WHITE] = 0.0
        self.white_loss = 1.2

        lot = np.random.default_rng(batch)                # pigment lot strength
        self.strength = 1.0 + lot.normal(0.0, 0.08, N_PIGMENTS)
        self.strength[WHITE] = 1.0

        self.organic = np.array([p.organic for p in palette()])
        self.K = K_matrix() * self.strength[:, None]
        self.S = S_matrix()
        self.dK = np.gradient(self.K, 10.0, axis=1)       # per nm
        self.noise = noise

    def reflectance(self, conc: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Spectrophotometer reading of panels sprayed with `conc` (n, P)."""
        conc = np.atleast_2d(np.asarray(conc, dtype=float))
        eff = conc / (1.0 + self.flocculation * conc)
        k_mix = eff @ self.K - (eff * self.band_shift * conc) @ self.dK
        loss = self.white_loss * conc[:, WHITE] * conc[:, self.organic].sum(axis=1)
        s_mix = eff @ self.S - loss[:, None] * self.S[WHITE]
        r = km.saunderson(km.r_infinity(np.maximum(k_mix, 1e-6), np.maximum(s_mix, 1e-4)))
        if rng is not None and self.noise > 0:
            r = r + rng.normal(0.0, self.noise, r.shape) + rng.normal(0.0, self.noise / 2, (len(r), 1))
        return np.clip(r, 0.001, 1.0)
