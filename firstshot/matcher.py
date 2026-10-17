"""Recipe search that prefers fewer pigments, cheaper and lower-carbon materials,
and matches that hold up under different lights and pigment lots.

For each candidate pigment set, concentrations are found with the cross-entropy
method (gradient-free), vectorised across all sets at once. Every candidate is
then scored under four light sources and across model and pigment-lot
uncertainty. Only recipes that pass robustly are eligible; among those the
priority decides:

fewest   fewest pigments, then lowest cost and carbon, then closest (default)
cost     lowest pigment cost, preferring fewer pigments when within 3%
carbon   lowest embodied carbon, preferring fewer pigments when within 3%
closest  smallest color difference
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .colorimetry import delta_e2000, reflectance_to_lab
from .data import MAX_COLORANTS, pigment_sets
from .pigments import COLORANTS, EXTERIOR, N_PIGMENTS, WHITE, carbon, costs
from .spectral import ILLUMINANTS, SECONDARY

TOLERANCE = 1.0            # CIEDE2000 limit under daylight
SECONDARY_TOLERANCE = 2.0  # limit under the other three light sources
MIN_SHARE = 0.0002         # a pigment below 0.02% of the recipe is dropped
MIN_PASS = 0.6             # pass probability needed to call a recipe viable
NEAR = 0.03                # cost or carbon within 3% counts as a tie
METAMERISM_WEIGHT = 0.35   # weight of the other lights in the search objective

# How strongly each priority trades color accuracy (in dE units) for materials
# while concentrations are optimised.
PRIORITIES: dict[str, dict[str, float]] = {
    "fewest": {"cost": 0.02, "carbon": 0.02},
    "cost": {"cost": 0.06, "carbon": 0.0},
    "carbon": {"cost": 0.0, "carbon": 0.06},
    "closest": {"cost": 0.0, "carbon": 0.0},
}


@dataclass
class Target:
    name: str
    lab: dict[str, np.ndarray]
    spectral: bool
    reflectance: np.ndarray | None = None

    @classmethod
    def from_reflectance(cls, refl: np.ndarray, name: str) -> "Target":
        refl = np.asarray(refl, dtype=float)
        return cls(name, {k: reflectance_to_lab(refl, k) for k in ILLUMINANTS}, True, refl)

    @classmethod
    def from_lab(cls, lab: np.ndarray, name: str) -> "Target":
        return cls(name, {"D65": np.asarray(lab, dtype=float)}, False)


@dataclass
class Candidate:
    pigments: tuple[int, ...]
    conc: np.ndarray
    de: dict[str, float]
    pass_probability: float
    de_spread: float
    cost: float
    carbon: float

    @property
    def n_colorants(self) -> int:
        return len(self.pigments)


@dataclass
class MatchResult:
    chosen: Candidate
    viable: bool
    candidates: list[Candidate] = field(repr=False)
    sets_checked: int = 0
    recipes_scored: int = 0
    seconds: float = 0.0
    priority: str = "fewest"
    widened: bool = False      # the shortlist found nothing that passes, so the search was widened

    def closest(self) -> Candidate:
        return min(self.candidates, key=lambda c: c.de["D65"])

    def by_pigment_count(self) -> dict[int, Candidate]:
        """The best recipe for each number of colorants: the chosen one for its own count,
        otherwise the closest passing one, otherwise the closest. White-only recipes are left out."""
        best: dict[int, Candidate] = {}
        for c in self.candidates:
            k = c.n_colorants
            rank = (c.pass_probability < MIN_PASS, c.de["D65"])
            if k and (k not in best or rank < (best[k].pass_probability < MIN_PASS, best[k].de["D65"])):
                best[k] = c
        if self.chosen.n_colorants:
            best[self.chosen.n_colorants] = self.chosen
        return dict(sorted(best.items()))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class Matcher:
    def __init__(self, shortlists: dict | None = None, seed: int = 0, pop: int = 64, elite: int = 10,
                 iters: int = 24, polish: int = 8, n_draws: int = 12, lot_sd: float = 0.04,
                 shortlist_size: int = 5, widen_size: int = 6, chunk: int = 48):
        # {"lab": PigmentShortlist, "spectral": PigmentShortlist}; either may be missing.
        self.shortlists = shortlists or {}
        self.sets = pigment_sets()
        self.costs = costs()
        self.carbon = carbon()
        self.seed, self.pop, self.elite, self.iters, self.polish = seed, pop, elite, iters, polish
        self.shortlist_size = shortlist_size
        self.widen_size = widen_size
        self.chunk = chunk
        draws = np.random.default_rng(seed + 1).normal(0.0, lot_sd, (n_draws, N_PIGMENTS))
        self.strength_draws = 1.0 + draws
        self.strength_draws[:, WHITE] = 1.0

    # ---------- material scores (per kg of pigment blend, relative) ----------
    def cost_index(self, conc: np.ndarray) -> np.ndarray:
        return np.atleast_2d(conc) @ self.costs

    def carbon_index(self, conc: np.ndarray) -> np.ndarray:
        return np.atleast_2d(conc) @ self.carbon

    def materials_penalty(self, conc: np.ndarray, priority: str) -> np.ndarray:
        w = PRIORITIES[priority]
        return w["cost"] * self.cost_index(conc) + w["carbon"] * self.carbon_index(conc)

    @staticmethod
    def objective(refl: np.ndarray, target: Target, mode: str = "sum") -> np.ndarray:
        """sum: daylight dE plus a share of the other lights; smooth, used for the search.
        ratio: the worst dE as a fraction of its tolerance; exactly the pass test, used for corrections."""
        de = delta_e2000(reflectance_to_lab(refl, "D65"), target.lab["D65"])
        if not target.spectral:
            return de / TOLERANCE if mode == "ratio" else de
        others = np.stack([delta_e2000(reflectance_to_lab(refl, k), target.lab[k]) for k in SECONDARY])
        if mode == "ratio":
            worst = np.maximum(de / TOLERANCE, others.max(axis=0) / SECONDARY_TOLERANCE)
            return worst + 0.1 * (de + others.mean(axis=0))
        return de + METAMERISM_WEIGHT * others.mean(axis=0)

    # ---------- which pigment sets to search ----------
    @staticmethod
    def allowed(exterior_only: bool = False, exclude: tuple[int, ...] = ()) -> tuple[int, ...]:
        """Colorants the search may use. Restricted pigments are never in this list."""
        base = EXTERIOR if exterior_only else COLORANTS
        return tuple(i for i in base if i not in exclude)

    def shortlist_pigments(self, target: Target, allowed: set[int], size: int, widen: bool = False) -> set[int]:
        """Pigments the networks expect in the recipe. A measured spectrum is read by the spectral
        network; a widened search also asks the Lab network, which knows which pigments look alike
        when the original pigment is excluded."""
        picks: list[int] = []
        if target.spectral and "spectral" in self.shortlists:
            picks += self.shortlists["spectral"].top(target.reflectance, size, allowed=allowed)
        if "lab" in self.shortlists and (widen or not picks):
            picks += self.shortlists["lab"].top(target.lab["D65"], size, allowed=allowed)
        return set(picks)

    def candidate_sets(self, target: Target, mode: str, max_colorants: int, exterior_only: bool = False,
                       exclude: tuple[int, ...] = (), widen: bool = False) -> list[tuple[int, ...]]:
        allowed = set(self.allowed(exterior_only, exclude))
        sets = [s for s in self.sets if len(s) <= max_colorants and set(s) <= allowed]
        if mode == "exhaustive" or not self.shortlists:
            return sets
        size = self.widen_size if widen else self.shortlist_size
        top = self.shortlist_pigments(target, allowed, size, widen)
        return [s for s in sets if set(s) <= top]

    # ---------- concentrations: cross-entropy method, all sets in one batch ----------
    @staticmethod
    def _layout(sets: list[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray]:
        """Pigment index and validity mask per set; white is always the first component."""
        width = 1 + max(len(s) for s in sets)
        idx = np.zeros((len(sets), width), dtype=int)
        mask = np.zeros((len(sets), width), dtype=bool)
        for j, s in enumerate(sets):
            comps = (WHITE, *s)
            idx[j, : len(comps)] = comps
            mask[j, : len(comps)] = True
        return idx, mask

    def _cem(self, idx, mask, target, model, offset, priority, rng, mu, sd, iters, sd_floor,
             mode: str = "sum", bounds: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        n, width = mu.shape
        rows = np.arange(n)
        best = np.zeros((n, N_PIGMENTS))
        best_j = np.full(n, np.inf)
        for _ in range(iters):
            z = mu[:, None, :] + sd[:, None, :] * rng.standard_normal((n, self.pop, width))
            z[:, 0, :] = mu                                   # keep the current mean in the population
            if bounds is not None:
                z = np.clip(z, bounds[0][:, None, :], bounds[1][:, None, :])
            w = _softmax(np.where(mask[:, None, :], z, -np.inf))
            conc = np.zeros((n, self.pop, N_PIGMENTS))
            for c in range(width):
                conc[rows, :, idx[:, c]] += w[:, :, c] * mask[:, c][:, None]
            flat = conc.reshape(n * self.pop, N_PIGMENTS)
            refl = model.predict(flat)
            if offset is not None:
                refl = np.clip(refl + offset, 0.001, 1.0)
            j = (self.objective(refl, target, mode) + self.materials_penalty(flat, priority)).reshape(n, self.pop)
            order = np.argsort(j, axis=1)
            top = j[rows, order[:, 0]]
            improved = top < best_j
            best_j = np.where(improved, top, best_j)
            best[improved] = conc[rows[improved], order[improved, 0]]
            elite = np.take_along_axis(z, order[:, : self.elite, None], axis=1)
            mu = np.where(mask, 0.7 * elite.mean(axis=1) + 0.3 * mu, 0.0)
            sd = np.where(mask, 0.7 * elite.std(axis=1) + 0.3 * sd + sd_floor, 0.0)
        return best

    def optimise_sets(self, sets: list[tuple[int, ...]], target: Target, model, offset=None,
                      priority: str = "fewest", rng: np.random.Generator | None = None) -> np.ndarray:
        """Best concentrations for every pigment set, shape (len(sets), P)."""
        rng = rng or np.random.default_rng(self.seed)
        out = np.zeros((len(sets), N_PIGMENTS))
        for start in range(0, len(sets), self.chunk):
            part = sets[start : start + self.chunk]
            idx, mask = self._layout(part)
            mu = np.zeros(mask.shape)
            mu[:, 0] = 1.5
            sd = np.where(mask, 2.0, 0.0)
            best = self._cem(idx, mask, target, model, offset, priority, rng, mu, sd, self.iters, 0.02)
            if self.polish:
                # restart narrowly around each winner to settle the last few tenths of a dE
                shares = np.take_along_axis(best, idx, axis=1)
                mu = np.where(mask, np.log(np.maximum(shares, 1e-6)), 0.0)
                sd = np.where(mask, 0.15, 0.0)
                best = self._cem(idx, mask, target, model, offset, priority, rng, mu, sd, self.polish, 0.005)
            out[start : start + len(part)] = best
        out[out < MIN_SHARE] = 0.0
        return out / out.sum(axis=1, keepdims=True)

    def optimise_set(self, pigments: tuple[int, ...], target: Target, model, offset=None,
                     priority: str = "fewest", rng: np.random.Generator | None = None) -> np.ndarray:
        return self.optimise_sets([pigments], target, model, offset, priority, rng)[0]

    def correct(self, conc: np.ndarray, target: Target, model, offset: np.ndarray, priority: str = "fewest",
                rng: np.random.Generator | None = None, step: float = 0.25, iters: int = 16) -> np.ndarray:
        """A colorist's correction: adjust the current recipe, do not start over.

        The measured offset (lab minus model) is only valid near the recipe that was sprayed,
        so the search stays in a trust region around it, and it minimises exactly the pass
        test: the worst color difference as a fraction of its tolerance.
        """
        rng = rng or np.random.default_rng(self.seed)
        pigments = tuple(int(i) for i in np.flatnonzero(conc) if i != WHITE)
        idx, mask = self._layout([pigments])
        shares = np.take_along_axis(conc[None], idx, axis=1)
        mu = np.where(mask, np.log(np.maximum(shares, 1e-6)), 0.0)
        sd = np.where(mask, step, 0.0)
        bounds = (mu - 3.0 * step, mu + 3.0 * step)       # hard trust region: shares at most ~2x either way
        best = self._cem(idx, mask, target, model, offset, priority, rng, mu, sd, iters, 0.01, mode="ratio",
                         bounds=bounds)[0]
        best[best < MIN_SHARE] = 0.0
        return best / best.sum()

    # ---------- robust scoring ----------
    def evaluate_many(self, concs: np.ndarray, target: Target, model, offset=None) -> list[Candidate]:
        concs = np.atleast_2d(concs)
        mean = model.predict(concs)
        samples = model.predict_samples(concs, self.strength_draws)      # (draws, n, bands)
        if offset is not None:
            mean = np.clip(mean + offset, 0.001, 1.0)
            samples = np.clip(samples + offset, 0.001, 1.0)
        de = {k: delta_e2000(reflectance_to_lab(mean, k), target.lab[k]) for k in target.lab}
        de_samples = delta_e2000(reflectance_to_lab(samples, "D65"), target.lab["D65"])
        ok = de_samples <= TOLERANCE
        if target.spectral:
            for k in SECONDARY:
                ok &= delta_e2000(reflectance_to_lab(samples, k), target.lab[k]) <= SECONDARY_TOLERANCE
        cost, carb = self.cost_index(concs), self.carbon_index(concs)
        out = []
        for j, conc in enumerate(concs):
            used = tuple(int(i) for i in np.flatnonzero(conc) if i != WHITE)
            out.append(Candidate(used, conc, {k: float(v[j]) for k, v in de.items()}, float(ok[:, j].mean()),
                                 float(de_samples[:, j].std()), float(cost[j]), float(carb[j])))
        return out

    def evaluate(self, conc: np.ndarray, target: Target, model, offset=None) -> Candidate:
        return self.evaluate_many(conc[None], target, model, offset)[0]

    # ---------- the whole search ----------
    def match(self, target: Target, model, mode: str = "ai", max_colorants: int = MAX_COLORANTS,
              priority: str = "fewest", exterior_only: bool = False, exclude: tuple[int, ...] = (),
              offset=None) -> MatchResult:
        if priority not in PRIORITIES:
            raise ValueError(f"priority must be one of {sorted(PRIORITIES)}")
        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        sets = self.candidate_sets(target, mode, max_colorants, exterior_only, exclude)
        candidates = self.evaluate_many(self.optimise_sets(sets, target, model, offset, priority, rng),
                                        target, model, offset)
        chosen, viable = self.choose(candidates, priority)
        widened = False
        if not viable and mode == "ai":
            # Fast path failed: widen the shortlist once before giving up.
            seen = set(sets)
            extra = [s for s in self.candidate_sets(target, mode, max_colorants, exterior_only, exclude,
                                                    widen=True) if s not in seen]
            if extra:
                candidates += self.evaluate_many(self.optimise_sets(extra, target, model, offset, priority, rng),
                                                 target, model, offset)
                sets = sets + extra
                chosen, viable = self.choose(candidates, priority)
                widened = True
        scored = len(sets) * self.pop * (self.iters + self.polish)
        return MatchResult(chosen, viable, candidates, len(sets), scored, time.perf_counter() - start, priority,
                           widened)

    @staticmethod
    def choose(candidates: list[Candidate], priority: str = "fewest") -> tuple[Candidate, bool]:
        viable = [c for c in candidates if c.pass_probability >= MIN_PASS]
        if not viable:
            return min(candidates, key=lambda c: c.de["D65"]), False
        if priority == "closest":
            return min(viable, key=lambda c: (c.de["D65"], c.n_colorants)), True
        if priority in ("cost", "carbon"):
            def score(c: Candidate) -> float:
                return c.cost if priority == "cost" else c.carbon
            best = min(score(c) for c in viable)
            near = [c for c in viable if score(c) <= best * (1.0 + NEAR)]
            return min(near, key=lambda c: (c.n_colorants, score(c), c.de["D65"])), True
        return min(viable, key=lambda c: (c.n_colorants, round(c.cost + c.carbon, 2), c.de["D65"])), True
