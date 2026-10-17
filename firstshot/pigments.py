"""Pigment library: Kubelka-Munk spectra plus cost, carbon and compliance data.

Spectra are parametric models of real pigment classes (Colour Index names),
built from absorption bands and edges at their characteristic wavelengths.
Absorption was fitted so each pigment's full-strength color and its 10% tint
in white land near typical published values for that pigment, with scattering
fixed by pigment class: organic pigments are strong, transparent tinters and
inorganic pigments are weaker and opaque. They stand in for a company's
colorant database; swap in measured K/S data when available.

Material data
-------------
cost     Relative price per kg (titanium white = 1), from typical bulk prices.
carbon   Relative embodied-carbon index per kg by pigment chemistry. These are
         illustrative placeholders: in production they are replaced by supplier
         product carbon footprints (TfS PCF guideline), the data AkzoNobel
         already collects from its suppliers.
status   "standard" pigments may be used. "restricted" pigments are never
         proposed: lead chromates (India caps lead in household and decorative
         paint at 90 ppm under its 2016 rules) and cadmium pigments (restricted
         in paints under EU REACH). They exist here only so legacy formulas
         that contain them can be reproduced and replaced.
exterior True when the pigment is weather-fast enough for exterior use.

Index 0 is always the titanium white base.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

WL = np.arange(400.0, 701.0, 10.0)


def _band(mu: float, sigma: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((WL - mu) / sigma) ** 2)


def _edge(at: float, width: float, height: float) -> np.ndarray:
    """Strong absorption below `at` nm, falling off over `width` nm."""
    return height / (1.0 + np.exp((WL - at) / width))


def _scatter(level: float, power: float = 1.0) -> np.ndarray:
    return level * (WL / 550.0) ** (-power)


@dataclass(frozen=True)
class Pigment:
    key: str
    name: str
    ci: str
    K: np.ndarray
    S: np.ndarray
    cost: float
    carbon: float
    organic: bool
    exterior: bool = True
    status: str = "standard"
    note: str = ""


@lru_cache(maxsize=1)
def palette() -> tuple[Pigment, ...]:
    return (
        Pigment("white", "Titanium white", "PW6", 0.0003 + _edge(400, 6, 0.08), _scatter(1.0, 0.5),
                1.0, 3.0, False, note="The hiding pigment; often the largest share of a paint's pigment footprint."),
        Pigment("black", "Carbon black", "PBk7", 25.0 + 0.01 * (WL - 550), _scatter(0.08),
                1.6, 2.5, False),
        Pigment("phthalo_blue", "Phthalo blue", "PB15:3",
                0.005 + _band(605, 35, 67) + _band(680, 30, 120) + _band(395, 25, 3), _scatter(0.03),
                4.2, 6.0, True),
        Pigment("ultramarine", "Ultramarine blue", "PB29", 0.019 + _band(615, 60, 4.1) + _band(545, 30, 2.7),
                _scatter(0.15), 2.5, 2.0, False, exterior=False, note="Acid-sensitive; interior use."),
        Pigment("phthalo_green", "Phthalo green", "PG7", 0.005 + _band(647, 45, 44) + _band(427, 35, 6.5),
                _scatter(0.03), 4.8, 6.5, True),
        Pigment("oxide_green", "Chromium oxide green", "PG17",
                0.010 + _band(640, 40, 2.2) + _edge(502, 15, 2.0), _scatter(0.6), 2.2, 2.5, False),
        Pigment("quin_magenta", "Quinacridone magenta", "PR122", 0.005 + _band(527, 35, 7.5) + _band(402, 30, 1.4),
                _scatter(0.1), 9.5, 9.0, True),
        Pigment("pyrrole_red", "Pyrrole red", "PR254", 0.0025 + _edge(586.5, 3.0, 6.4), _scatter(0.25),
                10.5, 8.0, True),
        Pigment("pyrrole_orange", "Pyrrole orange", "PO73", 0.0054 + _edge(561, 3.2, 4.4), _scatter(0.25),
                9.0, 8.0, True),
        Pigment("arylide_yellow", "Arylide yellow", "PY74", 0.0011 + _edge(499, 5.0, 4.0), _scatter(0.15),
                5.2, 5.0, True, exterior=False, note="Fades in pale exterior tints."),
        Pigment("bismuth_yellow", "Bismuth vanadate", "PY184", 0.0022 + _edge(487, 4.5, 3.0),
                _scatter(0.6), 7.0, 4.0, False, note="Opaque lead-free yellow; the usual base for replacing chrome yellows."),
        Pigment("isoindoline_yellow", "Isoindoline yellow", "PY139", 0.0025 + _edge(512.3, 5.1, 8.0),
                _scatter(0.2), 8.5, 7.0, True, note="Reddish organic yellow that pairs with bismuth vanadate."),
        Pigment("oxide_red", "Red iron oxide", "PR101", 0.001 + _edge(550.5, 25.6, 3.8), _scatter(0.5),
                0.5, 1.0, False, note="Cheapest chromatic pigment; low-carbon grades are on the market."),
        Pigment("oxide_yellow", "Yellow iron oxide", "PY42", 0.065 + _edge(491.5, 15.5, 3.0), _scatter(0.45),
                0.55, 1.0, False),
        Pigment("dioxazine_violet", "Dioxazine violet", "PV23", 0.005 + _band(559, 45, 13.4), _scatter(0.04),
                12.0, 10.0, True),
        Pigment("cobalt_blue", "Cobalt blue", "PB28",
                0.086 + _band(560, 22, 7.7) + _band(607, 25, 1.7) + _band(653, 28, 1.7), _scatter(0.35),
                15.0, 7.0, False, note="Costly; cobalt supply is concentrated."),
        Pigment("lead_chrome", "Lead chromate", "PY34", 0.002 + _edge(498.5, 5.5, 4.8), _scatter(0.7),
                2.0, 2.0, False, status="restricted",
                note="Contains lead; India limits lead in decorative paint to 90 ppm."),
        Pigment("cadmium_red", "Cadmium red", "PR108", 0.029 + _edge(575.4, 3.0, 6.5), _scatter(0.6),
                14.0, 3.0, False, status="restricted", note="Cadmium is restricted in paints under EU REACH."),
    )


def K_matrix() -> np.ndarray:
    return np.stack([p.K for p in palette()])


def S_matrix() -> np.ndarray:
    return np.stack([p.S for p in palette()])


def costs() -> np.ndarray:
    return np.array([p.cost for p in palette()])


def carbon() -> np.ndarray:
    return np.array([p.carbon for p in palette()])


WHITE = 0
BLACK = 1
N_PIGMENTS = len(palette())
COLORANTS = tuple(i for i, p in enumerate(palette()) if i != WHITE and p.status == "standard")
RESTRICTED = tuple(i for i, p in enumerate(palette()) if p.status == "restricted")
EXTERIOR = tuple(i for i in COLORANTS if palette()[i].exterior)
