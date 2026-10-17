# Benchmark

Generated 2026-10-17 20:01. All numbers come from `scripts/benchmark.py`.

## Customer samples (40 targets, 40 reachable within tolerance)

| Search | Pigment sets checked | Seconds per target | Passing recipe found |
|---|---:|---:|---:|
| AI shortlist | 25.0 | 0.33 | 100% of reachable |
| Every set | 575.0 | 7.76 | 100% of reachable |

The samples were made with 2.8 colorants on average. FirstShot's passing recipes use 2.35, and use fewer than the original in 45% of cases.

Chosen recipes use 2.35 colorants on average against 2.9 for the closest possible recipe, changing pigment cost by 1.74% and carbon by -1.18%.

| Model | Mean lab trials | Pass on first shot | Pass within 3 trials |
|---|---:|---:|---:|
| Physics, calibrated | 2.27 | 50% | 85% |
| Physics + ML | 1.73 | 60% | 92% |
| Physics + ML, tuned | 1.38 | 85% | 92% |

Trials are capped at six; a color that never passes counts as seven.

## ColorChecker (24 targets, 23 reachable within tolerance)

| Search | Pigment sets checked | Seconds per target | Passing recipe found |
|---|---:|---:|---:|
| AI shortlist | 39.29 | 0.48 | 96% of reachable |
| Every set | 575.0 | 6.75 | 100% of reachable |

Chosen recipes use 2.45 colorants on average against 2.95 for the closest possible recipe, changing pigment cost by -21.82% and carbon by -8.89%.

| Model | Mean lab trials | Pass on first shot | Pass within 3 trials |
|---|---:|---:|---:|
| Physics, calibrated | 2.22 | 17% | 96% |
| Physics + ML | 1.83 | 26% | 100% |
| Physics + ML, tuned | 1.57 | 44% | 100% |

Trials are capped at six; a color that never passes counts as seven.

## Legacy formulas

| Formula | Priority | New recipe | dE daylight | Cost change | Carbon change | Lab trials |
|---|---|---|---:|---:|---:|---:|
| Safety yellow | cost | PW6 52.68% + PY184 45.72% + PY139 1.6% | 0.6 | 175.9% | 35.4% | 2 |
| Safety yellow | carbon | PW6 52.26% + PY184 46.13% + PY139 1.61% | 0.6 | 177.7% | 35.6% | 2 |
| Safety yellow | closest | PW6 52.16% + PY184 46.23% + PY139 1.61% | 0.6 | 178.2% | 35.6% | 2 |
| Signal red | cost | PW6 50.77% + PR254 27.46% + PO73 15.69% + PY42 6.09% | 0.02 | -59.9% | 67.9% | 1 |
| Signal red | carbon | PW6 47.7% + PR254 40.12% + PY184 8.74% + PY42 3.44% | 0.01 | -55.8% | 67.5% | 2 |
| Signal red | closest | PW6 47.72% + PR254 40.11% + PY184 8.71% + PY42 3.46% | 0.01 | -55.9% | 67.4% | 2 |
| Heritage green | cost | PW6 57.65% + PY184 27.83% + PY42 11.78% + PB15:3 2.74% | 0.01 | 93.7% | 12.0% | 1 |
| Heritage green | carbon | PW6 57.86% + PY184 27.63% + PY42 11.77% + PB15:3 2.74% | 0.03 | 92.9% | 11.9% | 1 |
| Heritage green | closest | PW6 54.96% + PY184 40.53% + PB15:3 2.28% + PO73 2.22% | 0.01 | 163.8% | 28.5% | 2 |
| Cobalt sky | cost | PW6 93.99% + PB29 5.73% + PB15:3 0.28% | 0.37 | -54.4% | -13.2% | not passed in 6 |
| Cobalt sky | carbon | PW6 93.98% + PB29 5.73% + PB15:3 0.28% | 0.37 | -54.4% | -13.2% | not passed in 6 |
| Cobalt sky | closest | PW6 93.98% + PB29 5.73% + PB15:3 0.28% | 0.37 | -54.4% | -13.2% | not passed in 6 |
