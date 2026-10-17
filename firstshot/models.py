"""The three models behind FirstShot.

PhysicsModel      Kubelka-Munk with the nominal pigment database. What most
                  color-formulation software does.
HybridModel       Kubelka-Munk plus a neural-network ensemble that learns the
                  physics' systematic error from lab history. The spread of the
                  ensemble is the model's uncertainty. It can be fine-tuned to a
                  new pigment lot from a handful of calibration panels.
PigmentShortlist  A multi-label network that reads a target color (or its
                  reflectance curve) and returns the pigments most likely to be
                  needed, so the search only checks a few pigment sets instead
                  of all of them.

Training uses scikit-learn; inference is plain NumPy so the service image stays
small and starts fast on AWS Lambda.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from . import km
from .pigments import COLORANTS, K_matrix, S_matrix

Layers = list[tuple[np.ndarray, np.ndarray]]


def _forward(x: np.ndarray, layers: Layers, activation) -> np.ndarray:
    h = x
    for w, b in layers[:-1]:
        h = activation(h @ w + b)
    w, b = layers[-1]
    return h @ w + b


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _pack_layers(prefix: str, members: list[Layers]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {f"{prefix}_shape": np.array([len(members), len(members[0])])}
    for m, layers in enumerate(members):
        for l, (w, b) in enumerate(layers):
            out[f"{prefix}_{m}_{l}_w"] = w
            out[f"{prefix}_{m}_{l}_b"] = b
    return out


def _unpack_layers(prefix: str, raw) -> list[Layers]:
    n_members, n_layers = raw[f"{prefix}_shape"]
    return [
        [(raw[f"{prefix}_{m}_{l}_w"], raw[f"{prefix}_{m}_{l}_b"]) for l in range(n_layers)]
        for m in range(n_members)
    ]


def fit_strengths(model, conc: np.ndarray, refl: np.ndarray, sweeps: int = 2) -> None:
    """Re-estimate each pigment's tinting strength from calibration panels (in place)."""
    grid = np.linspace(0.7, 1.3, 61)
    for _ in range(sweeps):
        for i in COLORANTS:
            rows = conc[:, i] > 0
            if not rows.any():
                continue
            errors = []
            for g in grid:
                model.strength[i] = g
                errors.append(np.mean((model.predict(conc[rows]) - refl[rows]) ** 2))
            model.strength[i] = grid[int(np.argmin(errors))]


class PhysicsModel:
    key = "physics"
    label = "Physics only"

    def __init__(self) -> None:
        self.K = K_matrix()
        self.S = S_matrix()
        self.strength = np.ones(len(self.K))

    def predict(self, conc: np.ndarray) -> np.ndarray:
        return km.reflectance(conc, self.K * self.strength[:, None], self.S)

    def predict_samples(self, conc: np.ndarray, strengths: np.ndarray) -> np.ndarray:
        """Nominal prediction plus one prediction per pigment-lot strength draw."""
        conc = np.atleast_2d(conc)
        k = self.K * self.strength[:, None]
        draws = [km.reflectance(conc, k * s[:, None], self.S) for s in strengths]
        return np.stack([self.predict(conc), *draws])

    def calibrate(self, conc: np.ndarray, refl: np.ndarray) -> "PhysicsModel":
        """Classic colorant re-characterisation: fit pigment strengths, nothing else."""
        m = copy.deepcopy(self)
        m.key, m.label = "physics", "Physics, calibrated"
        fit_strengths(m, np.atleast_2d(conc), refl)
        return m

    def save(self, path: Path) -> None:
        np.savez_compressed(path, strength=self.strength, meta=np.array([self.key, self.label]))

    @classmethod
    def load(cls, path: Path) -> "PhysicsModel":
        raw = np.load(path, allow_pickle=False)
        m = cls()
        m.strength = raw["strength"]
        m.key, m.label = (str(v) for v in raw["meta"])
        return m


class HybridModel:
    key = "hybrid"
    label = "Physics + ML"

    def __init__(self) -> None:
        self.K = K_matrix()
        self.S = S_matrix()
        self.strength = np.ones(len(self.K))   # per-pigment lot strength, set by fine-tuning
        self.members: list[Layers] = []
        self._stack = None

    # ---------- inference ----------
    def _k(self) -> np.ndarray:
        return self.K * self.strength[:, None]

    def _features(self, conc: np.ndarray, r_km: np.ndarray) -> np.ndarray:
        z = (r_km - self.in_mean) @ self.in_comps.T
        return (np.hstack([conc, z]) - self.x_mu) / self.x_sd

    def _stacked(self):
        """All members' layers packed as float32: one pass for the whole ensemble, SIMD tanh.

        The first layer of every member is concatenated into one wide matrix (one
        BLAS call); the later layers are stacked for a batched matmul.
        """
        if getattr(self, "_stack", None) is None:
            first_w = np.concatenate([m[0][0] for m in self.members], axis=1).astype(np.float32)
            first_b = np.concatenate([m[0][1] for m in self.members]).astype(np.float32)
            rest = [
                (np.stack([m[l][0] for m in self.members]).astype(np.float32),
                 np.stack([m[l][1] for m in self.members])[:, None, :].astype(np.float32))
                for l in range(1, len(self.members[0]))
            ]
            self._stack = (first_w, first_b, rest)
        return self._stack

    def _residuals(self, x: np.ndarray) -> np.ndarray:
        """Residual spectra from every member, shape (members, n, bands). In-place where possible:
        fresh large arrays cost page faults, which dominate on small VMs such as Lambda."""
        first_w, first_b, rest = self._stacked()
        n, m = len(x), len(self.members)
        h = x.astype(np.float32) @ first_w
        h += first_b
        np.tanh(h, out=h)
        h = np.ascontiguousarray(h.reshape(n, m, -1).transpose(1, 0, 2))
        for w, b in rest[:-1]:
            h = h @ w
            h += b
            np.tanh(h, out=h)
        w, b = rest[-1]
        y = h @ w
        y += b
        y = y.reshape(m * n, -1).astype(np.float64)
        y *= self.y_sd
        y += self.y_mu
        out = y @ self.out_comps
        out += self.out_mean
        return out.reshape(m, n, -1)

    def predict_members(self, conc: np.ndarray) -> np.ndarray:
        conc = np.atleast_2d(conc)
        r_km = km.reflectance(conc, self._k(), self.S)
        out = self._residuals(self._features(conc, r_km))
        out += r_km[None]
        return np.clip(out, 0.001, 1.0, out=out)

    def predict(self, conc: np.ndarray) -> np.ndarray:
        return self.predict_members(conc).mean(axis=0)

    def predict_samples(self, conc: np.ndarray, strengths: np.ndarray) -> np.ndarray:
        """Ensemble members (model uncertainty) plus pigment-lot draws (process uncertainty)."""
        conc = np.atleast_2d(conc)
        members = self.predict_members(conc)
        k = self._k()
        residual = members.mean(axis=0) - km.reflectance(conc, k, self.S)
        draws = [np.clip(km.reflectance(conc, k * s[:, None], self.S) + residual, 0.001, 1.0) for s in strengths]
        return np.concatenate([members, np.stack(draws)])

    # ---------- training ----------
    @classmethod
    def train(
        cls,
        conc: np.ndarray,
        refl: np.ndarray,
        n_members: int = 5,
        hidden: tuple[int, ...] = (64, 64),
        max_iter: int = 300,
        seed: int = 0,
        n_in: int = 6,
        n_out: int = 8,
    ) -> "HybridModel":
        from sklearn.neural_network import MLPRegressor

        m = cls()
        r_km = km.reflectance(conc, m.K, m.S)
        m.in_mean = r_km.mean(axis=0)
        m.in_comps = np.linalg.svd(r_km - m.in_mean, full_matrices=False)[2][:n_in]
        x = np.hstack([conc, (r_km - m.in_mean) @ m.in_comps.T])
        m.x_mu, m.x_sd = x.mean(axis=0), x.std(axis=0) + 1e-8
        xn = (x - m.x_mu) / m.x_sd

        resid = refl - r_km
        m.out_mean = resid.mean(axis=0)
        m.out_comps = np.linalg.svd(resid - m.out_mean, full_matrices=False)[2][:n_out]
        y = (resid - m.out_mean) @ m.out_comps.T
        m.y_mu, m.y_sd = y.mean(axis=0), y.std(axis=0) + 1e-8
        yn = (y - m.y_mu) / m.y_sd

        rng = np.random.default_rng(seed)
        for k in range(n_members):
            idx = rng.integers(0, len(xn), len(xn))           # bootstrap -> diverse members
            net = MLPRegressor(
                hidden_layer_sizes=hidden,
                activation="tanh",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=2e-3,
                max_iter=max_iter,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=seed + k,
            )
            net.fit(xn[idx], yn[idx])
            m.members.append([(w.copy(), b.copy()) for w, b in zip(net.coefs_, net.intercepts_)])
        return m

    def fine_tune(self, conc: np.ndarray, refl: np.ndarray, ridge: float = 10.0,
                  refit_head: bool = True) -> "HybridModel":
        """Adapt to a new pigment lot from a handful of calibration panels.

        Parameter-efficient fine-tuning in two steps:
        1. Re-estimate each pigment's tinting strength (one physical parameter
           per pigment) by coordinate search against the panels.
        2. Re-fit only each member's output layer with ridge regularisation
           toward the original weights; hidden layers stay frozen, so a dozen
           panels cannot overfit the network.
        """
        tuned = copy.deepcopy(self)
        tuned.key, tuned.label = "tuned", "Physics + ML, tuned"
        conc = np.atleast_2d(conc)
        fit_strengths(tuned, conc, refl)
        if not refit_head:
            return tuned

        r_km = km.reflectance(conc, tuned._k(), self.S)
        x = tuned._features(conc, r_km)
        y = ((refl - r_km) - self.out_mean) @ self.out_comps.T
        yn = (y - self.y_mu) / self.y_sd
        members = []
        for layers in tuned.members:
            h = x
            for w, b in layers[:-1]:
                h = np.tanh(h @ w + b)
            H = np.hstack([h, np.ones((len(h), 1))])
            w0 = np.vstack([layers[-1][0], layers[-1][1][None, :]])
            w_new = np.linalg.solve(H.T @ H + ridge * np.eye(H.shape[1]), H.T @ yn + ridge * w0)
            members.append(list(layers[:-1]) + [(w_new[:-1], w_new[-1])])
        tuned.members = members
        tuned._stack = None
        return tuned

    # ---------- persistence ----------
    _ARRAYS = ("strength", "in_mean", "in_comps", "x_mu", "x_sd", "out_mean", "out_comps", "y_mu", "y_sd")

    def save(self, path: Path) -> None:
        arrays = {name: getattr(self, name) for name in self._ARRAYS}
        arrays.update(_pack_layers("net", self.members))
        arrays["meta"] = np.array([self.key, self.label])
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "HybridModel":
        raw = np.load(path, allow_pickle=False)
        m = cls()
        for name in cls._ARRAYS:
            setattr(m, name, raw[name])
        m.members = _unpack_layers("net", raw)
        m._stack = None
        m.key, m.label = (str(v) for v in raw["meta"])
        return m


def _lab_features(lab: np.ndarray) -> np.ndarray:
    lab = np.atleast_2d(lab)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    c = np.hypot(a, b)
    h = np.arctan2(b, a)
    return np.stack([L / 100, a / 100, b / 100, c / 100, np.cos(h), np.sin(h)], axis=1)


def _spectral_features(refl: np.ndarray) -> np.ndarray:
    """Kubelka-Munk absorption-to-scattering ratio, log scale: pigment fingerprints add up here."""
    r = np.clip(np.atleast_2d(refl), 0.002, 0.999)
    return np.log((1.0 - r) ** 2 / (2.0 * r))


class PigmentShortlist:
    """Target color -> probability that each colorant belongs in its recipe.

    kind="lab"       reads a CIELAB color, all you get from a hex code or a Lab value
    kind="spectral"  reads a measured reflectance curve, which also says which
                     pigments will match under other lights
    """

    def __init__(self, kind: str = "lab") -> None:
        self.kind = kind
        self.layers: Layers = []
        self.pigments = np.array(COLORANTS)

    def _features(self, x: np.ndarray) -> np.ndarray:
        return _lab_features(x) if self.kind == "lab" else _spectral_features(x)

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        z = (self._features(x) - self.x_mu) / self.x_sd
        logits = _forward(z, self.layers, _relu)
        return 1.0 / (1.0 + np.exp(-logits))

    def top(self, x: np.ndarray, k: int = 5, allowed=None) -> list[int]:
        """The k most likely colorants, optionally limited to an allowed set."""
        p = self.probabilities(x)[0]
        ranked = [int(self.pigments[i]) for i in np.argsort(p)[::-1]]
        if allowed is not None:
            ranked = [i for i in ranked if i in allowed]
        return ranked[:k]

    @classmethod
    def train(cls, x: np.ndarray, used: np.ndarray, kind: str = "lab", seed: int = 0) -> "PigmentShortlist":
        from sklearn.neural_network import MLPClassifier

        s = cls(kind)
        f = s._features(x)
        s.x_mu, s.x_sd = f.mean(axis=0), f.std(axis=0) + 1e-8
        y = used[:, s.pigments]
        net = MLPClassifier(
            hidden_layer_sizes=(128, 128),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=2e-3,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=seed,
        )
        net.fit((f - s.x_mu) / s.x_sd, y)
        s.layers = [(w.copy(), b.copy()) for w, b in zip(net.coefs_, net.intercepts_)]
        return s

    def save(self, path: Path) -> None:
        arrays = {"x_mu": self.x_mu, "x_sd": self.x_sd, "pigments": self.pigments, "kind": np.array(self.kind)}
        arrays.update(_pack_layers("net", [self.layers]))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "PigmentShortlist":
        raw = np.load(path, allow_pickle=False)
        s = cls(str(raw["kind"]) if "kind" in raw else "lab")
        s.x_mu, s.x_sd, s.pigments = raw["x_mu"], raw["x_sd"], raw["pigments"]
        s.layers = _unpack_layers("net", raw)[0]
        return s
