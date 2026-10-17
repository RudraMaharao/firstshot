"""Colorimetry from first principles: reflectance -> XYZ -> CIELAB -> CIEDE2000.

Everything is vectorised over leading dimensions so the optimiser can score
thousands of candidate recipes in one call. Validated against colour-science
in tests/test_colorimetry.py.
"""
from __future__ import annotations

import numpy as np

from .spectral import ILLUMINANTS, tables

_EPS = 216 / 24389
_KAPPA = 24389 / 27

SRGB_WHITE = np.array([95.047, 100.0, 108.883])
_BRADFORD = np.array(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]]
)
_BRADFORD_INV = np.linalg.inv(_BRADFORD)
_XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)
_SRGB_TO_XYZ = np.linalg.inv(_XYZ_TO_SRGB)


def reflectance_to_xyz(reflectance: np.ndarray, illuminant: str = "D65") -> np.ndarray:
    return np.asarray(reflectance) @ tables().weights[illuminant]


def xyz_to_lab(xyz: np.ndarray, white: np.ndarray) -> np.ndarray:
    r = np.asarray(xyz, dtype=float) / white
    f = np.where(r > _EPS, np.cbrt(r), (_KAPPA * r + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab: np.ndarray, white: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=float)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0

    def finv(f: np.ndarray) -> np.ndarray:
        f3 = f**3
        return np.where(f3 > _EPS, f3, (116.0 * f - 16.0) / _KAPPA)

    yr = np.where(lab[..., 0] > _KAPPA * _EPS, fy**3, lab[..., 0] / _KAPPA)
    return np.stack([finv(fx), yr, finv(fz)], axis=-1) * white


def reflectance_to_lab(reflectance: np.ndarray, illuminant: str = "D65") -> np.ndarray:
    return xyz_to_lab(reflectance_to_xyz(reflectance, illuminant), tables().white[illuminant])


def lab_under_all(reflectance: np.ndarray) -> dict[str, np.ndarray]:
    return {name: reflectance_to_lab(reflectance, name) for name in ILLUMINANTS}


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference (Sharma et al. 2005), broadcasting over inputs."""
    lab1 = np.asarray(lab1, dtype=float)
    lab2 = np.asarray(lab2, dtype=float)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    c_bar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2.0
    g = 0.5 * (1.0 - np.sqrt(c_bar**7 / (c_bar**7 + 25.0**7)))
    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    d_lp = L2 - L1
    d_cp = c2p - c1p
    d_hp = h2p - h1p
    d_hp = np.where(d_hp > 180.0, d_hp - 360.0, d_hp)
    d_hp = np.where(d_hp < -180.0, d_hp + 360.0, d_hp)
    zero_chroma = (c1p * c2p) == 0
    d_hp = np.where(zero_chroma, 0.0, d_hp)
    d_hp_big = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(d_hp / 2.0))

    l_bar = (L1 + L2) / 2.0
    c_barp = (c1p + c2p) / 2.0
    h_sum = h1p + h2p
    h_barp = np.where(
        zero_chroma,
        h_sum,
        np.where(
            np.abs(h1p - h2p) <= 180.0,
            h_sum / 2.0,
            np.where(h_sum < 360.0, (h_sum + 360.0) / 2.0, (h_sum - 360.0) / 2.0),
        ),
    )
    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_barp - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_barp))
        + 0.32 * np.cos(np.radians(3.0 * h_barp + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_barp - 63.0))
    )
    d_theta = 30.0 * np.exp(-(((h_barp - 275.0) / 25.0) ** 2))
    r_c = 2.0 * np.sqrt(c_barp**7 / (c_barp**7 + 25.0**7))
    s_l = 1.0 + 0.015 * (l_bar - 50.0) ** 2 / np.sqrt(20.0 + (l_bar - 50.0) ** 2)
    s_c = 1.0 + 0.045 * c_barp
    s_h = 1.0 + 0.015 * c_barp * t
    r_t = -np.sin(np.radians(2.0 * d_theta)) * r_c
    tl, tc, th = d_lp / s_l, d_cp / s_c, d_hp_big / s_h
    return np.sqrt(np.maximum(tl**2 + tc**2 + th**2 + r_t * tc * th, 0.0))


def chromatic_adapt(xyz: np.ndarray, src_white: np.ndarray, dst_white: np.ndarray) -> np.ndarray:
    """Bradford chromatic adaptation between two white points."""
    rho_src = _BRADFORD @ src_white
    rho_dst = _BRADFORD @ dst_white
    m = _BRADFORD_INV @ np.diag(rho_dst / rho_src) @ _BRADFORD
    return np.asarray(xyz) @ m.T


def _xyz_to_hex(xyz: np.ndarray) -> str:
    rgb = (np.asarray(xyz) / 100.0) @ _XYZ_TO_SRGB.T
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)
    r, g, b = np.round(np.clip(rgb, 0.0, 1.0) * 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def lab_to_hex(lab: np.ndarray, illuminant: str = "D65") -> str:
    """Render a colour as seen under an illuminant by an observer adapted to it."""
    white = tables().white[illuminant]
    xyz = chromatic_adapt(lab_to_xyz(lab, white), white, SRGB_WHITE)
    return _xyz_to_hex(xyz)


def hex_to_lab(hex_value: str) -> np.ndarray:
    h = hex_value.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError("expected a hex colour like #3a6ea5")
    rgb = np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ _SRGB_TO_XYZ.T) * 100.0
    white = tables().white["D65"]
    return xyz_to_lab(chromatic_adapt(xyz, SRGB_WHITE, white), white)
