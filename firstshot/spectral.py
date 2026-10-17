"""Spectral grid, CIE 1964 10-degree colour-matching functions and illuminants.

The tables are precomputed by scripts/build_tables.py with colour-science and
shipped as a small .npz, so the running service only needs NumPy.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
TABLES_PATH = DATA_DIR / "tables.npz"

ILLUMINANTS: tuple[str, ...] = ("D65", "A", "FL11", "LED-B4")
SECONDARY: tuple[str, ...] = ILLUMINANTS[1:]
ILLUMINANT_LABELS = {
    "D65": "Daylight",
    "A": "Incandescent",
    "FL11": "Store fluorescent",
    "LED-B4": "White LED",
}


@dataclass(frozen=True)
class SpectralTables:
    wavelengths: np.ndarray              # (31,) 400-700 nm in 10 nm steps
    weights: dict[str, np.ndarray]       # illuminant -> (31, 3), perfect white has Y = 100
    white: dict[str, np.ndarray]         # illuminant -> XYZ of the perfect diffuser
    patch_names: tuple[str, ...]         # ColorChecker patch names
    patch_reflectance: np.ndarray        # (24, 31) measured reflectance


@lru_cache(maxsize=1)
def tables() -> SpectralTables:
    raw = np.load(TABLES_PATH, allow_pickle=False)
    cmf = raw["cmf"]
    weights: dict[str, np.ndarray] = {}
    white: dict[str, np.ndarray] = {}
    for name, spd in zip((str(n) for n in raw["illuminants"]), raw["spd"]):
        w = cmf * spd[:, None]
        w = w * (100.0 / w[:, 1].sum())
        weights[name] = w
        white[name] = w.sum(axis=0)
    return SpectralTables(
        wavelengths=raw["wavelengths"],
        weights=weights,
        white=white,
        patch_names=tuple(str(n) for n in raw["patch_names"]),
        patch_reflectance=raw["patch_reflectance"],
    )
