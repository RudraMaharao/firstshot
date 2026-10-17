"""Lab-in-the-loop: how many sprayed trials until the color passes?

A colorist sprays the recipe, measures it, and if it fails, corrects it. The
correction uses the measured offset: the model's error at the last recipe is
assumed to hold nearby, so the next recipe is searched close to the last one
with the offset applied. Fewer trials means less paint, fewer panels and less
time.
"""
from __future__ import annotations

import numpy as np

from .colorimetry import delta_e2000, reflectance_to_lab
from .matcher import SECONDARY_TOLERANCE, TOLERANCE, Candidate, Matcher, Target
from .spectral import SECONDARY
from .virtual_lab import VirtualLab


def measure(target: Target, measured: np.ndarray) -> tuple[float, float | None, bool]:
    """Color difference of a measured panel under daylight and the worst other light."""
    de = float(delta_e2000(reflectance_to_lab(measured, "D65")[0], target.lab["D65"]))
    worst_other = None
    passed = de <= TOLERANCE
    if target.spectral:
        worst_other = float(max(delta_e2000(reflectance_to_lab(measured, k)[0], target.lab[k]) for k in SECONDARY))
        passed = passed and worst_other <= SECONDARY_TOLERANCE
    return de, worst_other, passed


def simulate_trials(target: Target, matcher: Matcher, model, lab: VirtualLab, max_trials: int = 6,
                    mode: str = "ai", max_colorants: int = 3, priority: str = "fewest",
                    exterior_only: bool = False, exclude: tuple[int, ...] = (), seed: int = 7,
                    first: Candidate | None = None) -> dict:
    """Spray, measure, correct, repeat. `first` reuses a recipe the model already proposed."""
    rng = np.random.default_rng(seed)
    if first is None:
        first = matcher.match(target, model, mode=mode, max_colorants=max_colorants, priority=priority,
                              exterior_only=exterior_only, exclude=exclude).chosen
    pigments = first.pigments
    conc = first.conc
    history = []
    for trial in range(1, max_trials + 1):
        measured = lab.reflectance(conc[None], rng)
        de, worst_other, passed = measure(target, measured)
        history.append({"trial": trial, "de_daylight": round(de, 2),
                        "de_other_lights": None if worst_other is None else round(worst_other, 2),
                        "passed": passed})
        if passed:
            break
        offset = measured[0] - model.predict(conc[None])[0]
        conc = matcher.correct(conc, target, model, offset, priority=priority, rng=rng)
    return {
        "model": model.key,
        "label": model.label,
        "pigments": len(pigments),
        "trials": len(history),
        "passed": history[-1]["passed"],
        "history": history,
    }
