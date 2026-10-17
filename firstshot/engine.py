"""Application layer: loads the trained models once and turns requests into
JSON-ready results for the API. Nothing here imports scikit-learn."""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import __version__, km
from .colorimetry import hex_to_lab, lab_to_hex, reflectance_to_lab
from .data import MAX_COLORANTS
from .matcher import (MIN_PASS, PRIORITIES, SECONDARY_TOLERANCE, TOLERANCE, Candidate, Matcher,
                      MatchResult, Target)
from .models import HybridModel, PhysicsModel, PigmentShortlist
from .pigments import (COLORANTS, EXTERIOR, K_matrix, N_PIGMENTS, RESTRICTED, S_matrix, WHITE, carbon,
                       costs, palette)
from .spectral import ILLUMINANT_LABELS, ILLUMINANTS, tables
from .trials import simulate_trials
from .virtual_lab import CURRENT_BATCH, VirtualLab

MODEL_DIR = Path(__file__).resolve().parent / "models"
KEY_TO_INDEX = {p.key: i for i, p in enumerate(palette())}
MODEL_ORDER = ("physics", "hybrid", "tuned")


@dataclass(frozen=True)
class Legacy:
    """A formula still in use that relies on a pigment worth replacing."""

    key: str
    name: str
    use: str
    exterior: bool
    recipe: tuple[tuple[str, float], ...]   # colorant key, share; white makes up the rest
    replace: tuple[str, ...]                # pigments the new formula must not use
    reason: str

    def conc(self) -> np.ndarray:
        c = np.zeros(N_PIGMENTS)
        for key, share in self.recipe:
            c[KEY_TO_INDEX[key]] = share
        c[WHITE] = 1.0 - c.sum()
        return c


LEGACY: tuple[Legacy, ...] = (
    Legacy("safety_yellow", "Safety yellow", "Machinery and road-marking yellow", True,
           (("lead_chrome", 0.40),), ("lead_chrome",),
           "Lead chromate. India caps lead in decorative paint at 90 ppm."),
    Legacy("signal_red", "Signal red", "Equipment and signage red", True,
           (("cadmium_red", 0.85),), ("cadmium_red",),
           "Cadmium pigment. Restricted in paints under EU REACH."),
    Legacy("heritage_green", "Heritage green", "Railings and front doors", True,
           (("lead_chrome", 0.30), ("phthalo_blue", 0.03)), ("lead_chrome",),
           "Classic chrome green built on lead chromate."),
    Legacy("cobalt_sky", "Cobalt sky", "Interior feature wall", False,
           (("cobalt_blue", 0.10),), ("cobalt_blue",),
           "Cobalt blue is the most expensive pigment on the shelf and cobalt supply is concentrated."),
)


def _r(x: float, n: int = 2) -> float:
    return float(round(float(x), n))


def _pct(new: float, old: float) -> float:
    return _r(100.0 * (new - old) / old, 1) if old else 0.0


@lru_cache(maxsize=1)
def _chips() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """10% tint in white and full-strength color for every pigment, for UI chips."""
    K, S = K_matrix(), S_matrix()
    tint, mass = [], []
    for i in range(N_PIGMENTS):
        for share, out in ((0.10, tint), (1.0, mass)):
            c = np.zeros(N_PIGMENTS)
            c[i] += share
            c[WHITE] += 1.0 - share
            out.append(lab_to_hex(reflectance_to_lab(km.reflectance(c, K, S), "D65")[0]))
    return tuple(tint), tuple(mass)


def pigment_json(i: int) -> dict:
    p = palette()[i]
    tint, mass = _chips()
    return {
        "key": p.key, "name": p.name, "ci": p.ci, "status": p.status, "exterior": p.exterior,
        "organic": p.organic, "cost": p.cost, "carbon": p.carbon, "note": p.note,
        "chip": tint[i], "masstone": mass[i],
    }


def _recipe_json(conc: np.ndarray) -> list[dict]:
    used = [i for i in range(N_PIGMENTS) if conc[i] > 0 and i != WHITE]
    used.sort(key=lambda i: -conc[i])
    rows = [WHITE, *used] if conc[WHITE] > 0 else used
    tint, _ = _chips()
    return [
        {"key": palette()[i].key, "name": palette()[i].name, "ci": palette()[i].ci,
         "share": _r(100.0 * conc[i], 2), "chip": tint[i], "status": palette()[i].status}
        for i in rows
    ]


class Engine:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.models = {
            "physics": PhysicsModel.load(model_dir / "physics.npz"),
            "hybrid": HybridModel.load(model_dir / "hybrid.npz"),
            "tuned": HybridModel.load(model_dir / "tuned.npz"),
        }
        self.matcher = Matcher({
            "lab": PigmentShortlist.load(model_dir / "shortlist.npz"),
            "spectral": PigmentShortlist.load(model_dir / "shortlist_spectral.npz"),
        })
        self.lab = VirtualLab(CURRENT_BATCH)
        path = model_dir / "metrics.json"
        self.metrics = json.loads(path.read_text()) if path.exists() else {}
        bench = model_dir.parent.parent / "results" / "benchmark.json"
        self.benchmark = json.loads(bench.read_text()) if bench.exists() else None
        self._cache: OrderedDict[tuple, MatchResult] = OrderedDict()
        self._lock = threading.Lock()

    # ---------- targets ----------
    def target(self, kind: str, value) -> Target:
        if kind == "patch":
            names = tables().patch_names
            if value not in names:
                raise ValueError(f"unknown ColorChecker patch: {value}")
            return Target.from_reflectance(tables().patch_reflectance[names.index(value)], str(value))
        if kind == "hex":
            return Target.from_lab(hex_to_lab(str(value)), "#" + str(value).strip().lstrip("#").lower())
        if kind == "lab":
            lab = np.asarray(value, dtype=float).reshape(-1)
            if lab.shape != (3,) or not 0.0 <= lab[0] <= 100.0 or np.abs(lab[1:]).max() > 128:
                raise ValueError("Lab needs L in 0..100 and a, b in -128..128")
            return Target.from_lab(lab, f"L {lab[0]:.1f}  a {lab[1]:.1f}  b {lab[2]:.1f}")
        if kind == "legacy":
            preset = self.legacy_preset(str(value))
            return Target.from_reflectance(self.lab.reflectance(preset.conc()[None])[0], preset.name)
        raise ValueError("target kind must be patch, hex, lab or legacy")

    @staticmethod
    def target_json(t: Target) -> dict:
        return {
            "name": t.name,
            "spectral": t.spectral,
            "lab": [_r(v) for v in t.lab["D65"]],
            "appearance": {k: lab_to_hex(t.lab[k], k) for k in t.lab},
        }

    def legacy_preset(self, key: str) -> Legacy:
        for preset in LEGACY:
            if preset.key == key:
                return preset
        raise ValueError(f"unknown legacy formula: {key}")

    # ---------- matching ----------
    def _match(self, cache_key: tuple, target: Target, model_key: str, mode: str, max_colorants: int,
               priority: str, exterior_only: bool, exclude: tuple[int, ...] = ()) -> MatchResult:
        key = (*cache_key, model_key, mode, max_colorants, priority, exterior_only, exclude)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        result = self.matcher.match(target, self.models[model_key], mode=mode, max_colorants=max_colorants,
                                    priority=priority, exterior_only=exterior_only, exclude=exclude)
        with self._lock:
            self._cache[key] = result
            while len(self._cache) > 128:
                self._cache.popitem(last=False)
        return result

    def candidate_json(self, c: Candidate, model_key: str = "tuned") -> dict:
        refl = self.models[model_key].predict(c.conc[None])
        return {
            "pigments": _recipe_json(c.conc),
            "colorants": c.n_colorants,
            "delta_e": {k: _r(v) for k, v in c.de.items()},
            "pass_probability": _r(c.pass_probability),
            "passes": bool(c.pass_probability >= MIN_PASS),
            "cost_index": _r(c.cost, 3),
            "carbon_index": _r(c.carbon, 3),
            "lab": [_r(v) for v in reflectance_to_lab(refl, "D65")[0]],
            "appearance": {k: lab_to_hex(reflectance_to_lab(refl, k)[0], k) for k in ILLUMINANTS},
        }

    def match(self, kind: str, value, priority: str = "fewest", exterior_only: bool = False,
              max_colorants: int = MAX_COLORANTS, model_key: str = "tuned", mode: str = "ai") -> dict:
        target = self.target(kind, value)
        result = self._match((kind, str(value)), target, model_key, mode, max_colorants, priority, exterior_only)
        chosen, closest = result.chosen, result.closest()
        total = len(self.matcher.candidate_sets(target, "exhaustive", max_colorants, exterior_only))
        return {
            "target": self.target_json(target),
            "viable": result.viable,
            "recipe": self.candidate_json(chosen, model_key),
            "closest": self.candidate_json(closest, model_key),
            "by_count": [
                {**self.candidate_json(c, model_key), "chosen": c is chosen}
                for c in result.by_pigment_count().values()
            ],
            "versus_closest": {
                "same_recipe": closest is chosen,
                "pigments_saved": closest.n_colorants - chosen.n_colorants,
                "cost_pct": _pct(chosen.cost, closest.cost),
                "carbon_pct": _pct(chosen.carbon, closest.carbon),
            },
            "search": {
                "mode": mode, "sets_checked": result.sets_checked, "sets_total": total,
                "recipes_scored": result.recipes_scored, "ms": int(round(result.seconds * 1000)),
                "widened": result.widened,
            },
            "settings": {"priority": priority, "exterior_only": exterior_only, "max_colorants": max_colorants,
                         "model": self.models[model_key].label},
        }

    def trials(self, kind: str, value, priority: str = "fewest", exterior_only: bool = False,
               max_colorants: int = MAX_COLORANTS) -> dict:
        target = self.target(kind, value)
        runs = []
        for model_key in MODEL_ORDER:
            first = self._match((kind, str(value)), target, model_key, "ai", max_colorants, priority,
                                exterior_only).chosen
            runs.append(simulate_trials(target, self.matcher, self.models[model_key], self.lab,
                                        priority=priority, first=first))
        return {"target": self.target_json(target), "runs": runs}

    # ---------- safer, cheaper, lower-carbon materials ----------
    def reformulate(self, key: str, priority: str = "cost", exterior_only: bool | None = None) -> dict:
        preset = self.legacy_preset(key)
        exterior = preset.exterior if exterior_only is None else exterior_only
        target = self.target("legacy", key)
        exclude = tuple(sorted(KEY_TO_INDEX[k] for k in preset.replace))
        result = self._match(("legacy", key), target, "tuned", "ai", MAX_COLORANTS, priority, exterior, exclude)
        chosen = result.chosen
        old = preset.conc()
        old_cost, old_carbon = float(old @ costs()), float(old @ carbon())

        trial = simulate_trials(target, self.matcher, self.models["tuned"], self.lab, priority=priority,
                                first=chosen, seed=5)
        options = []
        for p in PRIORITIES:
            c, ok = Matcher.choose(result.candidates, p)
            options.append({"priority": p, "viable": ok, **self.candidate_json(c),
                            "cost_pct": _pct(c.cost, old_cost), "carbon_pct": _pct(c.carbon, old_carbon)})
        return {
            "legacy": {
                "key": preset.key, "name": preset.name, "use": preset.use, "exterior": preset.exterior,
                "reason": preset.reason,
                "pigments": _recipe_json(old),
                "replaced": [palette()[KEY_TO_INDEX[k]].name for k in preset.replace],
                "cost_index": _r(old_cost, 3),
                "carbon_index": _r(old_carbon, 3),
            },
            "target": self.target_json(target),
            "viable": result.viable,
            "replacement": self.candidate_json(chosen),
            "change": {"cost_pct": _pct(chosen.cost, old_cost), "carbon_pct": _pct(chosen.carbon, old_carbon)},
            "lab_check": trial,
            "options": options,
            "settings": {"priority": priority, "exterior_only": exterior},
        }

    # ---------- reference data ----------
    @staticmethod
    def palette() -> list[dict]:
        return [pigment_json(i) for i in range(N_PIGMENTS)]

    @staticmethod
    def targets() -> list[dict]:
        t = tables()
        return [
            {"name": n, "hex": lab_to_hex(reflectance_to_lab(r, "D65"))}
            for n, r in zip(t.patch_names, t.patch_reflectance)
        ]

    def legacy(self) -> list[dict]:
        out = []
        for p in LEGACY:
            refl = self.lab.reflectance(p.conc()[None])[0]
            out.append({"key": p.key, "name": p.name, "use": p.use, "exterior": p.exterior, "reason": p.reason,
                        "hex": lab_to_hex(reflectance_to_lab(refl, "D65")), "pigments": _recipe_json(p.conc())})
        return out

    def model_card(self) -> dict:
        return {
            "version": __version__,
            "illuminants": ILLUMINANT_LABELS,
            "tolerance": {"daylight": TOLERANCE, "other_lights": SECONDARY_TOLERANCE, "min_pass_probability": MIN_PASS},
            "palette": {"colorants": len(COLORANTS), "exterior": len(EXTERIOR), "restricted": len(RESTRICTED)},
            "models": {k: m.label for k, m in self.models.items()},
            "training": self.metrics,
            "benchmark": self.benchmark,
        }


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return Engine()
