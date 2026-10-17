"""Build the spectral lookup tables shipped with FirstShot.

Uses colour-science (a dev dependency) once, then stores everything the
service needs in firstshot/data/tables.npz so production needs only NumPy.

    python scripts/build_tables.py
"""
from __future__ import annotations

from pathlib import Path

import colour
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "firstshot" / "data" / "tables.npz"
SHAPE = colour.SpectralShape(400, 700, 10)
ILLUMINANTS = ["D65", "A", "FL11", "LED-B4"]


def main() -> None:
    cmfs = colour.MSDS_CMFS["CIE 1964 10 Degree Standard Observer"].copy().align(SHAPE)
    spd = np.array([colour.SDS_ILLUMINANTS[n].copy().align(SHAPE).values for n in ILLUMINANTS])
    patches = colour.SDS_COLOURCHECKERS["BabelColor Average"]
    names = list(patches.keys())
    refl = np.array([sd.copy().align(SHAPE).values for sd in patches.values()])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        wavelengths=np.asarray(SHAPE.wavelengths, dtype=float),
        cmf=np.asarray(cmfs.values, dtype=float),
        illuminants=np.array(ILLUMINANTS),
        spd=spd.astype(float),
        patch_names=np.array(names),
        patch_reflectance=refl.astype(float),
    )
    print(f"wrote {OUT} ({len(names)} ColorChecker patches, {len(ILLUMINANTS)} illuminants)")


if __name__ == "__main__":
    main()
