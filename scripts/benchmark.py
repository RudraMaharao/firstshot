"""Benchmark FirstShot against the usual way of doing it.

    python scripts/benchmark.py            # about five minutes on an M2
    python scripts/benchmark.py --quick    # a handful of targets

Four questions, answered on targets the models never saw:

1. Search   Does the AI shortlist find recipes as good as searching every
            pigment set, and how much faster is it?
2. Pigments How many pigments does the chosen recipe need compared with the
            closest-possible recipe, and what does that do to cost and carbon?
3. Trials   How many spray-measure-correct rounds does each model need before
            the color passes under all four lights?
4. Materials Can legacy formulas with lead, cadmium or cobalt be rebuilt with
            standard pigments, and at what cost and carbon?

Writes results/benchmark.json and results/benchmark.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from firstshot.data import sample_recipes  # noqa: E402
from firstshot.engine import LEGACY, Engine  # noqa: E402
from firstshot.matcher import Target  # noqa: E402
from firstshot.spectral import tables  # noqa: E402
from firstshot.trials import simulate_trials  # noqa: E402
from firstshot.virtual_lab import CURRENT_BATCH, VirtualLab  # noqa: E402

RESULTS = ROOT / "results"


def customer_targets(n: int, seed: int = 2024) -> list[Target]:
    """Customer samples: two- and three-colorant panels sprayed from the current pigment lot."""
    conc, used = sample_recipes(n, np.random.default_rng(seed), size_weights=(0.0, 1.0, 1.0))
    refl = VirtualLab(CURRENT_BATCH).reflectance(conc, np.random.default_rng(seed + 1))
    targets = [Target.from_reflectance(r, f"sample {i + 1}") for i, r in enumerate(refl)]
    for t, u in zip(targets, used):
        t.original_colorants = int(u.sum())      # how many colorants the sample was made with
    return targets


def checker_targets() -> list[Target]:
    t = tables()
    return [Target.from_reflectance(r, n) for n, r in zip(t.patch_names, t.patch_reflectance)]


def mean(values) -> float:
    values = list(values)
    return round(float(np.mean(values)), 2) if values else float("nan")


def run(quick: bool) -> dict:
    engine = Engine()
    matcher, tuned = engine.matcher, engine.models["tuned"]
    groups = {
        "customer samples": customer_targets(8 if quick else 40),
        "ColorChecker": checker_targets()[: 6 if quick else 24],
    }
    out: dict = {"generated": time.strftime("%Y-%m-%d %H:%M"), "groups": {}}

    for group, targets in groups.items():
        print(f"\n== {group}: {len(targets)} targets")
        rows = []
        for t in targets:
            ai = matcher.match(t, tuned, mode="ai")
            full = matcher.match(t, tuned, mode="exhaustive")
            closest = full.closest()
            row = {
                "target": t.name,
                "ai": {"viable": ai.viable, "colorants": ai.chosen.n_colorants, "de": round(ai.chosen.de["D65"], 2),
                       "sets": ai.sets_checked, "seconds": round(ai.seconds, 3),
                       "cost": round(ai.chosen.cost, 3), "carbon": round(ai.chosen.carbon, 3)},
                "exhaustive": {"viable": full.viable, "colorants": full.chosen.n_colorants,
                               "de": round(full.chosen.de["D65"], 2), "sets": full.sets_checked,
                               "seconds": round(full.seconds, 3)},
                "closest": {"colorants": closest.n_colorants, "cost": round(closest.cost, 3),
                            "carbon": round(closest.carbon, 3)},
                "original_colorants": getattr(t, "original_colorants", None),
                "trials": {},
            }
            for key in ("physics", "hybrid", "tuned"):
                first = ai.chosen if key == "tuned" else None
                r = simulate_trials(t, matcher, engine.models[key], engine.lab, first=first)
                row["trials"][key] = {"trials": r["trials"], "passed": r["passed"],
                                      "first_de": r["history"][0]["de_daylight"]}
            rows.append(row)
            print(f"{t.name:22s} ai {'pass' if ai.viable else 'fail'} {ai.chosen.n_colorants}p "
                  f"{ai.seconds:5.2f}s | full {'pass' if full.viable else 'fail'} {full.chosen.n_colorants}p "
                  f"{full.seconds:5.2f}s | trials " +
                  " ".join(f"{k}={v['trials']}{'' if v['passed'] else 'x'}" for k, v in row["trials"].items()))

        reachable = [r for r in rows if r["exhaustive"]["viable"]]
        both = [r for r in reachable if r["ai"]["viable"]]
        summary = {
            "targets": len(rows),
            "reachable": len(reachable),
            "search": {
                "ai_found_when_reachable": round(len(both) / max(len(reachable), 1), 3),
                "ai_sets": mean(r["ai"]["sets"] for r in rows),
                "all_sets": mean(r["exhaustive"]["sets"] for r in rows),
                "ai_seconds": mean(r["ai"]["seconds"] for r in rows),
                "all_seconds": mean(r["exhaustive"]["seconds"] for r in rows),
                "same_or_fewer_pigments": round(
                    float(np.mean([r["ai"]["colorants"] <= r["exhaustive"]["colorants"] for r in both])), 3)
                if both else None,
            },
            "pigments": {
                "chosen": mean(r["ai"]["colorants"] for r in both),
                "closest": mean(r["closest"]["colorants"] for r in both),
                "original": mean(r["original_colorants"] for r in both if r["original_colorants"]),
                "fewer_than_original": round(float(np.mean(
                    [r["ai"]["colorants"] < r["original_colorants"] for r in both if r["original_colorants"]])), 3)
                if any(r["original_colorants"] for r in both) else None,
                "cost_change_pct": mean(100 * (r["ai"]["cost"] - r["closest"]["cost"]) / r["closest"]["cost"]
                                        for r in both),
                "carbon_change_pct": mean(100 * (r["ai"]["carbon"] - r["closest"]["carbon"]) / r["closest"]["carbon"]
                                          for r in both),
            },
            "trials": {},
        }
        for key in ("physics", "hybrid", "tuned"):
            runs = [r["trials"][key] for r in reachable]
            summary["trials"][key] = {
                "label": engine.models[key].label,
                "mean_trials": mean(min(x["trials"], 6) if x["passed"] else 7 for x in runs),
                "first_shot_pass": round(float(np.mean([x["passed"] and x["trials"] == 1 for x in runs])), 3)
                if runs else None,
                "pass_within_3": round(float(np.mean([x["passed"] and x["trials"] <= 3 for x in runs])), 3)
                if runs else None,
            }
        out["groups"][group] = {"summary": summary, "rows": rows}
        print(json.dumps(summary, indent=2))

    print("\n== legacy formulas")
    legacy = []
    for preset in LEGACY:
        rows = {}
        for priority in ("cost", "carbon", "closest"):
            r = engine.reformulate(preset.key, priority=priority)
            rows[priority] = {
                "recipe": [(p["ci"], p["share"]) for p in r["replacement"]["pigments"]],
                "viable": r["viable"],
                "de": r["replacement"]["delta_e"],
                "pass_probability": r["replacement"]["pass_probability"],
                "cost_pct": r["change"]["cost_pct"],
                "carbon_pct": r["change"]["carbon_pct"],
                "trials": r["lab_check"]["trials"],
                "passed": r["lab_check"]["passed"],
            }
        legacy.append({"key": preset.key, "name": preset.name, "replaced": list(preset.replace), "by_priority": rows})
        print(preset.name, json.dumps(rows["cost"]))
    out["legacy"] = legacy
    return out


def markdown(res: dict) -> str:
    lines = ["# Benchmark", "", f"Generated {res['generated']}. All numbers come from `scripts/benchmark.py`.", ""]
    for group, data in res["groups"].items():
        s = data["summary"]
        lines += [f"## {group[0].upper() + group[1:]} ({s['targets']} targets, {s['reachable']} reachable within tolerance)",
                  ""]
        q = s["search"]
        lines += [
            "| Search | Pigment sets checked | Seconds per target | Passing recipe found |",
            "|---|---:|---:|---:|",
            f"| AI shortlist | {q['ai_sets']} | {q['ai_seconds']} | {round(100 * q['ai_found_when_reachable'])}% of reachable |",
            f"| Every set | {q['all_sets']} | {q['all_seconds']} | 100% of reachable |",
            "",
        ]
        p = s["pigments"]
        if p.get("fewer_than_original") is not None:
            lines += [f"The samples were made with {p['original']} colorants on average. FirstShot's passing "
                      f"recipes use {p['chosen']}, and use fewer than the original in "
                      f"{round(100 * p['fewer_than_original'])}% of cases.", ""]
        lines += [
            f"Chosen recipes use {p['chosen']} colorants on average against {p['closest']} for the closest "
            f"possible recipe, changing pigment cost by {p['cost_change_pct']}% and carbon by "
            f"{p['carbon_change_pct']}%.",
            "",
            "| Model | Mean lab trials | Pass on first shot | Pass within 3 trials |",
            "|---|---:|---:|---:|",
        ]
        for t in s["trials"].values():
            fs = "n/a" if t["first_shot_pass"] is None else f"{round(100 * t['first_shot_pass'])}%"
            w3 = "n/a" if t["pass_within_3"] is None else f"{round(100 * t['pass_within_3'])}%"
            lines.append(f"| {t['label']} | {t['mean_trials']} | {fs} | {w3} |")
        lines += ["", "Trials are capped at six; a color that never passes counts as seven.", ""]
    lines += ["## Legacy formulas", "",
              "| Formula | Priority | New recipe | dE daylight | Cost change | Carbon change | Lab trials |",
              "|---|---|---|---:|---:|---:|---:|"]
    for item in res["legacy"]:
        for priority, r in item["by_priority"].items():
            recipe = " + ".join(f"{ci} {share}%" for ci, share in r["recipe"])
            trials = f"{r['trials']}" if r["passed"] else f"not passed in {r['trials']}"
            lines.append(f"| {item['name']} | {priority} | {recipe} | {r['de']['D65']} | {r['cost_pct']}% | "
                         f"{r['carbon_pct']}% | {trials} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    res = run(parser.parse_args().quick)
    RESULTS.mkdir(exist_ok=True)
    summary = {k: v for k, v in res.items() if k != "groups"}
    summary["groups"] = {g: {"summary": d["summary"]} for g, d in res["groups"].items()}
    (RESULTS / "benchmark.json").write_text(json.dumps(summary, indent=2))
    (RESULTS / "benchmark_rows.json").write_text(json.dumps(res, indent=2))
    (RESULTS / "benchmark.md").write_text(markdown(res))
    print(f"\nwrote {RESULTS / 'benchmark.md'}")


if __name__ == "__main__":
    main()
