# FirstShot

**Match a paint color with fewer pigments, in fewer lab trials, with safer materials.**

Give FirstShot a color: a spectrophotometer reading, a screen color or a Lab value. It returns a
paint recipe that uses as few pigments as possible, still matches under daylight, incandescent,
store fluorescent and LED light, and is likely to pass on the first sprayed panel. It can also
rebuild legacy colors that rely on lead, cadmium or cobalt from standard pigments, and it says so
plainly when no standard-pigment recipe is good enough.

Under the hood: Kubelka-Munk physics, a neural-network ensemble that learns what the physics
gets wrong, a second network that shortlists pigments so the search is many times faster, and a
FastAPI service packaged as one container that runs locally, in Docker and on AWS Lambda.

## Results

All numbers come from `scripts/benchmark.py` against the simulated lab, on colors the models never
trained on: 40 customer samples (two- and three-colorant panels sprayed from today's pigment lot)
and the 24 patches of a ColorChecker chart, which were not made from paint at all.

**Right first time.** Lab trials until the color passes under all four lights:

| Model | Customer samples: right first time | Mean trials | ColorChecker: right first time | Mean trials |
|---|---:|---:|---:|---:|
| Physics, calibrated to today's lot | 50% | 2.27 | 17% | 2.22 |
| Physics + ML, trained on older lots | 60% | 1.73 | 26% | 1.83 |
| Physics + ML, fine-tuned from 15 panels | **85%** | **1.38** | **44%** | **1.57** |

Trials are capped at six; a color that never passes counts as seven. On 1,500 fresh panels the
fine-tuned model predicts color with a median error of 0.25 CIEDE2000 units (92% within 1.0),
against 0.71 (67% within 1.0) for calibrated physics.

**Faster search.** The shortlist network searches 25 pigment sets instead of 575, 24 times faster
(0.33 s against 7.8 s per color). It found a passing recipe for 100% of the customer samples and
96% of the ColorChecker patches that a full search could match.

**Fewer pigments.** The customer samples were made with 2.8 colorants on average. FirstShot's
passing recipes use 2.35, fewer than the original in 45% of cases. The closest-possible recipe
would use 2.9.

**Safer materials.** Legacy formulas rebuilt with the lowest-cost passing recipe:

| Legacy color | Removes | Replacement | dE daylight / worst other light | Cost | Carbon |
|---|---|---|---:|---:|---:|
| Safety yellow | Lead chromate (PY34) | Bismuth vanadate 45.7% + isoindoline yellow 1.6% | 0.60 / 1.19 | +176% | +35% |
| Signal red | Cadmium red (PR108) | Pyrrole red 27.5% + pyrrole orange 15.7% + yellow iron oxide 6.1% | 0.02 / 0.67 | −60% | +68% |
| Heritage green | Lead chromate (PY34) | Bismuth vanadate 27.8% + yellow iron oxide 11.8% + phthalo blue 2.7% | 0.01 / 0.82 | +94% | +12% |
| Cobalt sky | Cobalt blue (PB28) | None passes: the closest daylight match shifts by 5.6 under store fluorescent light | 0.37 / 5.59 | | |

The lead-free yellow pairs an inorganic yellow with an organic one, the approach the industry
describes for replacing lead chromate, and it costs more, as reported. Replacing cadmium cuts cost but raises the carbon index. For cobalt sky the honest answer
is to keep cobalt or accept a visible shift.

Full benchmark tables are in [results/benchmark.md](results/benchmark.md); every target, row by row, is in `results/benchmark_rows.json`.

## Why this problem

- **Color matching is slow and expensive.** Conventional effect-paint matching often takes ten or
  more trial rounds, and automotive refinish R&D can spend up to 80% of its budget on color
  matching. Smart tools cut formula iterations from 8 to 12 down to 2 to 4 (X-Rite in PCI, 2025).
  Every trial is a sprayed panel, a measurement and someone's time.
- **Some classic pigments are now off limits.** India's 2016 rules cap lead in household and
  decorative paint at 90 ppm. EU REACH bars paints with 0.01% cadmium or more. Lead chromate
  replacements usually combine an inorganic yellow with organic pigments and cost more (PCI, 2015).
- **Materials now carry a carbon number.** AkzoNobel collects product carbon footprints from its
  suppliers through the Together for Sustainability PCF guideline. Suppliers are responding:
  LANXESS reports about 35% lower footprint for a Scopeblue iron oxide yellow, and AkzoNobel and
  BASF report at least 5% lower footprint from biomass-balanced raw materials.

A formulation tool that picks pigments, not just concentrations, can act on all three at once.

## What it does

**Fewest pigments that pass.** The search tries every combination of one to three colorants from
a shortlist, plus titanium white. A recipe is eligible only if it is likely to pass under all four
lights (CIEDE2000 1.0 in daylight, 2.0 under the others) across model uncertainty and pigment-lot
variation. Among eligible recipes, the fewest pigments wins, then cost and carbon. Other
priorities are one click away: lowest cost, lowest carbon or closest match.

**Right first time.** The model is fine-tuned to the pigment lot on the shelf today from one
calibration panel per pigment. A simulated lab sprays each proposed recipe, measures it and
corrects it until it passes, so models can be compared on what matters: trials to pass.

**Safer, cheaper or lower-carbon materials.** Restricted pigments are never proposed. Legacy
formulas that contain them can be rebuilt from standard pigments, compared on cost and carbon,
and checked under the same four lights. Exterior jobs can be limited to weather-fast pigments.

**Honest about limits.** A screen color has no spectrum, so it is matched in daylight only and the
interface says so. If nothing passes, the closest recipe is shown as failing, with the light it
fails under. The cobalt blue example below cannot be replaced without a visible shift under warm
light, and FirstShot reports that instead of hiding it.

## How it works

```
 target: reflectance curve | hex | Lab
        |
        v
 pigment shortlist network ---- 5 likely pigments -> 25 of 575 pigment sets
        |                       (widens once if nothing passes)
        v
 cross-entropy search, all sets batched  <-- hybrid model: Kubelka-Munk + neural-network ensemble,
        |                                    fine-tuned to today's pigment lot from 15 panels
        v
 robust scoring: 4 lights x (5 ensemble members + 12 pigment-lot draws) = pass probability
        |
        v
 choose: must pass -> fewest pigments -> cost and carbon -> closest
        |
        v
 lab loop (simulated): spray -> measure -> correct -> repeat
```

### Color science

- Reflectance from 400 to 700 nm in 10 nm steps; CIE 1964 10 degree observer; illuminants D65
  (daylight), A (incandescent), FL11 (store fluorescent) and LED-B4 (white LED). Tables are built
  once with colour-science and shipped as a 6 KB file. CIELAB matches colour-science to 1e-8.
- CIEDE2000 written from Sharma, Wu and Dalal (2005), vectorised, and tested against their
  published test pairs.
- Two-constant Kubelka-Munk theory for the paint film, with the Saunderson correction for surface
  reflection (K1 = 0.03, K2 = 0.6).
- A pigment library of 18 real pigments by Colour Index name (PW6, PB15:3, PY184, PR254,
  PY34 and others). Absorption spectra are parametric bands and edges fitted so each pigment's
  full-strength color and 10% tint land near typical published values; scattering is set by
  class, transparent for organic pigments and opaque for inorganic ones.
- Metamerism is checked explicitly: a match that only holds in daylight is not a match.

### Machine learning

- **The lab is simulated, and the models never see how.** The virtual lab adds what plain
  Kubelka-Munk ignores: pigment flocculation at high loading, concentration-dependent absorption
  shifts, titanium-white scattering loss when organic pigments are present, 8% lot-to-lot pigment
  strength variation and instrument noise. Models learn only from "measured" panels.
- **Hybrid model.** The network predicts the physics' error, not the color. Inputs are the recipe
  and six principal components of the Kubelka-Munk spectrum; outputs are eight principal
  components of the residual. Five bootstrap-trained MLPs (64 x 64, tanh) form an ensemble whose
  spread is the model's uncertainty. Training uses scikit-learn; inference is NumPy.
- **Fine-tuning to a new pigment lot.** Each pigment's tinting strength is re-fitted from one
  calibration panel per pigment, the panels a lab already sprays when a lot arrives. One
  parameter per pigment cannot overfit 15 panels. Re-fitting the output layer as well was tried
  and dropped because it made results worse.
- **Pigment shortlist.** Two multi-label networks predict which pigments a recipe needs: one reads
  the log K/S fingerprint of a measured spectrum, one reads a Lab value. The spectral one finds
  every pigment of a held-out recipe in its top five 82% of the time, against 48% from Lab alone,
  because a spectrum shows which pigments will also match under other lights.
- **Validation.** Cross-validation is grouped by pigment set, so test recipes use combinations the
  model never trained on. Final numbers come from 1,500 fresh panels from the current lot.

### Search

- Concentrations are a softmax over each set's pigments, optimised with the cross-entropy method:
  64 candidates, 24 iterations and an 8-iteration polish, with every pigment set in one NumPy batch.
- The objective is CIEDE2000 in daylight plus 0.35 times the mean under the other lights, plus a
  small cost or carbon term depending on the priority. Pigments under 0.1% are dropped.
- If no shortlisted set passes, the search widens once to the union of both networks' top six.
  This is what finds the lead-free safety yellow, whose lead chromate fingerprint the spectral
  network has never seen.

### Engineering

- FastAPI with typed request and response schemas; OpenAPI docs at `/docs`.
- The service image needs only NumPy, FastAPI and uvicorn. Models ship as 0.8 MB of `.npz`.
- The ensemble runs as one float32 pass with in-place operations. Profiling showed NumPy spending
  much of its time in page faults from fresh allocations on small VMs. In-place operations plus
  keeping freed memory in the heap (`MALLOC_*` settings in the Dockerfile) cut a match from about
  480 ms to 310 ms on a 2-vCPU Firecracker VM, the same kind of VM Lambda uses.
- One Docker image for local runs and AWS Lambda (arm64, Graviton), using the AWS Lambda Web
  Adapter so the FastAPI app runs unchanged behind a public HTTPS Function URL.
- 25 pytest tests, including CIEDE2000 against published data and the API end to end. GitHub
  Actions runs them and smoke-tests the container.

## Run it on a MacBook (Apple silicon)

Needs Python 3.12 (`brew install python@3.12`) and, for Docker and AWS, Docker Desktop.

```bash
cd firstshot
python3.12 -m venv .venv && source .venv/bin/activate
make setup          # installs requirements-dev.txt
make test           # 25 tests, under a minute
make run            # open http://localhost:8080
```

The trained models are included. To rebuild everything from scratch:

```bash
make tables         # CIE tables and ColorChecker spectra, needs colour-science
make train          # about 3 minutes on an M2
make benchmark      # about 10 minutes; writes results/benchmark.md
```

With Docker:

```bash
make docker-build && make docker-run    # http://localhost:8080
```

## Deploy to AWS

You need an AWS account, AWS CLI v2 (a 2025 or later release: `brew install awscli`), Docker Desktop
running, and credentials set up with `aws configure` or `aws configure sso`.

```bash
make deploy         # prints the public URL when it is live
make teardown       # removes the function, URL, image repository, logs and role
```

`deploy/deploy.sh` creates an ECR repository, builds the image for arm64 and pushes it, creates a
minimal execution role (CloudWatch Logs only, 14-day retention), creates or updates the Lambda
function (3008 MB, 60 s) and a public Function URL with both permissions AWS has required since
October 2025. The region defaults to `ap-south-1` (Mumbai); override with `AWS_REGION=...`.

Demo traffic fits in the Lambda free tier. The image is about 200 MB in ECR, a few cents a month.
Run `make teardown` when you are done.

## API

```bash
curl -s http://localhost:8080/api/match -H 'content-type: application/json' \
  -d '{"target": {"kind": "patch", "value": "orange"}, "priority": "fewest"}'

curl -s http://localhost:8080/api/match -H 'content-type: application/json' \
  -d '{"target": {"kind": "hex", "value": "#3a6ea5"}, "priority": "carbon", "exterior_only": true}'

curl -s http://localhost:8080/api/reformulate -H 'content-type: application/json' \
  -d '{"legacy": "safety_yellow", "priority": "cost"}'
```

| Endpoint | What it does |
|---|---|
| `POST /api/match` | Recipe with the fewest pigments that passes, the best recipe for each pigment count, search statistics |
| `POST /api/trials` | Spray, measure and correct until it passes, for the physics, hybrid and fine-tuned models |
| `POST /api/reformulate` | Rebuild a legacy formula without its restricted or costly pigment |
| `GET /api/palette` | Pigments with Colour Index names, cost, carbon, status and exterior rating |
| `GET /api/targets`, `/api/legacy` | ColorChecker patches and legacy formulas |
| `GET /api/model-card` | Training metrics and benchmark results |

## Project layout

```
firstshot/        the library
  spectral.py     CIE tables and illuminants
  colorimetry.py  reflectance -> XYZ -> CIELAB, CIEDE2000, sRGB for display
  km.py           Kubelka-Munk and Saunderson
  pigments.py     18 pigments: spectra, cost, carbon, compliance, exterior rating
  virtual_lab.py  the simulated lab that stands in for reality
  models.py       physics model, hybrid ensemble, pigment shortlist networks
  matcher.py      batched cross-entropy search and robust recipe choice
  trials.py       spray-measure-correct loop
  engine.py       loads models, legacy formulas, JSON for the API
service/          FastAPI app and the web interface (plain HTML, CSS and JavaScript)
scripts/          build_tables.py, train.py, benchmark.py
deploy/           deploy.sh and teardown.sh for AWS Lambda
tests/            pytest suite
results/          benchmark results
```

## Limitations

- The lab is simulated. A real deployment needs measured K/S data for each pigment and a lab's
  own history of recipes and measurements; the code is built to take both.
- Pigment spectra are parametric fits to typical colors, not measurements.
- Carbon figures are illustrative relative indices and cost figures are rough relative prices.
  In production they would come from supplier PCF data and purchasing.
- The model covers opaque, solid colors with up to three colorants plus white: no metallic or
  pearl effects, gloss or substrate show-through.
- Pass probability is calibrated against the simulator, not against real tolerance outcomes.

## Talking points

- **Why a hybrid model?** Physics extrapolates and explains; the network fixes its systematic
  errors; the ensemble says how sure it is, which is what "right first time" needs.
- **Why fine-tune only pigment strengths?** Lot-to-lot drift is the shift a lab sees every week.
  One parameter per pigment adapts from the 15 panels a lab already sprays, and cannot overfit.
- **Why spectra beat hex codes.** Two paints can match on screen and differ under store lights.
  A measured spectrum lets the search avoid that, and lets the shortlist network work better.
- **Guardrails.** Restricted pigments are never proposed, and a replacement that shifts under
  another light is reported as failing.
- **Engineering.** NumPy-only inference, batched search, a page-fault fix that sped up the hot
  path by a third, one image for laptop and Lambda.

## Sources

- [Optimizing effect paint formulation with smart technology](https://www.pcimag.com/articles/113655-optimizing-effect-paint-formulation-with-smart-technology), X-Rite in PCI Magazine, 2025
- [Computer-aided color formulation](https://www.xrite.com/blog/computer-aided-color-formulation), X-Rite
- [Lead content in household and decorative paints rules, 2016](https://indiaenvironmentportal.org.in/reports-and-documents/notification-on-lead-content-in-hosuehold-and-decorative-paints-rules-2016), India Environment Portal
- [Commission Regulation (EU) 2016/217, cadmium entry in REACH Annex XVII](https://www.legislation.gov.uk/eur/2016/217/data.htm?view=plain)
- [Lead chromate replacement](https://pcimag.com/articles/101063-lead-chromate-replacement), PCI Magazine
- [Pigments for low carbon footprint coating formulations](https://www.pcimag.com/articles/113304-pigments-for-low-carbon-footprint-coating-formulations), LANXESS in PCI Magazine, 2025
- [AkzoNobel and BASF announce strategic partnership](https://www.pcimag.com/articles/113563-akzonobel-and-basf-announce-strategic-partnership), PCI Magazine, 2025
- [AkzoNobel responsible sourcing, TfS PCF and supplier decarbonization](https://netzerocompare.com/policies/akzonobel-responsible-sourcing-policy-tfs-pcf-system-and-supplier-decarbonization-framework), Net Zero Compare
- [EU Ecolabel criteria for paints and varnishes](https://ec.europa.eu/environment/ecolabel/documents/paints.pdf), European Commission
- [Function URL access control](https://docs.aws.amazon.com/lambda/latest/dg/urls-auth.html), AWS Lambda documentation
- [AWS Lambda Web Adapter](https://github.com/aws/aws-lambda-web-adapter)
- G. Sharma, W. Wu and E. N. Dalal, "The CIEDE2000 color-difference formula: implementation notes, supplementary test data, and mathematical observations", Color Research and Application 30(1), 2005
