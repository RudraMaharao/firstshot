"""Train FirstShot's models on lab history and write a model card.

    python scripts/train.py           # full run, a few minutes on an M2
    python scripts/train.py --quick   # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from firstshot.colorimetry import delta_e2000, reflectance_to_lab  # noqa: E402
from firstshot.data import calibration_panels, sample_recipes  # noqa: E402
from firstshot.models import HybridModel, PhysicsModel, PigmentShortlist  # noqa: E402
from firstshot.virtual_lab import CURRENT_BATCH, HISTORICAL_BATCH, VirtualLab  # noqa: E402

MODEL_DIR = ROOT / "firstshot" / "models"


def color_error(pred: np.ndarray, measured: np.ndarray) -> np.ndarray:
    return delta_e2000(reflectance_to_lab(pred), reflectance_to_lab(measured))


def summary(errors: np.ndarray) -> dict:
    return {
        "median": round(float(np.median(errors)), 3),
        "p90": round(float(np.percentile(errors, 90)), 3),
        "within_1": round(float(np.mean(errors <= 1.0)), 3),
    }


def main(quick: bool = False, seed: int = 0) -> None:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n_history = 3000 if quick else 14000
    members = 2 if quick else 5
    max_iter = 60 if quick else 300

    print(f"simulating {n_history} historical lab panels")
    history = VirtualLab(HISTORICAL_BATCH)
    conc, used = sample_recipes(n_history, rng)
    refl = history.reflectance(conc, rng)
    set_ids = used @ (2 ** np.arange(used.shape[1]))
    physics = PhysicsModel()

    print("cross-validating on unseen pigment sets (GroupKFold)")
    folds = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=4).split(conc, groups=set_ids)):
        if fold == 2:
            break
        m = HybridModel.train(conc[tr], refl[tr], n_members=2, max_iter=max_iter, seed=100 + fold)
        folds.append({
            "physics": summary(color_error(physics.predict(conc[te]), refl[te])),
            "hybrid": summary(color_error(m.predict(conc[te]), refl[te])),
        })

    print(f"training the hybrid ensemble ({members} members)")
    hybrid = HybridModel.train(conc, refl, n_members=members, max_iter=max_iter, seed=seed)

    print("training the pigment shortlists (from Lab and from reflectance curves)")
    shortlist = PigmentShortlist.train(reflectance_to_lab(refl), used, kind="lab", seed=seed)
    spectral = PigmentShortlist.train(refl, used, kind="spectral", seed=seed)

    current = VirtualLab(CURRENT_BATCH)
    cal_conc = calibration_panels()
    print(f"fine-tuning to the current pigment lot from {len(cal_conc)} calibration panels")
    cal_refl = current.reflectance(cal_conc, rng)
    tuned = hybrid.fine_tune(cal_conc, cal_refl, refit_head=False)
    physics_cal = physics.calibrate(cal_conc, cal_refl)       # the classic baseline

    print("evaluating on 1500 fresh panels from the current lot")
    test_conc, test_used = sample_recipes(1500, np.random.default_rng(seed + 9))
    test_refl = current.reflectance(test_conc, rng)
    forward = {
        "physics_uncalibrated": summary(color_error(physics.predict(test_conc), test_refl)),
        "physics_calibrated": summary(color_error(physics_cal.predict(test_conc), test_refl)),
        "hybrid_untuned": summary(color_error(hybrid.predict(test_conc), test_refl)),
        "hybrid_tuned": summary(color_error(tuned.predict(test_conc), test_refl)),
    }
    test_lab = reflectance_to_lab(test_refl)

    def recall(model, inputs) -> float:
        hits = [set(np.flatnonzero(u)) <= set(model.top(x, 5)) for u, x in zip(test_used, inputs)]
        return round(float(np.mean(hits)), 3)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    physics_cal.save(MODEL_DIR / "physics.npz")
    hybrid.save(MODEL_DIR / "hybrid.npz")
    tuned.save(MODEL_DIR / "tuned.npz")
    shortlist.save(MODEL_DIR / "shortlist.npz")
    spectral.save(MODEL_DIR / "shortlist_spectral.npz")

    card = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "history_panels": n_history,
        "ensemble_members": members,
        "calibration_panels": len(cal_conc),
        "unseen_pigment_sets_cv": folds,
        "current_lot_test": forward,
        "shortlist_top5_recall": {"from_lab": recall(shortlist, test_lab), "from_spectrum": recall(spectral, test_refl)},
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    main(parser.parse_args().quick)
