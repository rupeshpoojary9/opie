# OPIE — Open Product-Intelligence Engine

From a packaged-food image, OPIE runs **extraction → evidence-tracing → validation → taxonomy mapping → personalized health scoring**, served as an **API with batch + real-time job status**, and backed by an **eval harness that scores itself against public Open Food Facts ground truth**.

It is not a one-shot label parser. It is the full *image → trustworthy structured data → personalized decision* pipeline: multi-agent extraction where every field carries evidence and confidence, a rules/compliance validator over the LLM output, ingredient-taxonomy normalization, explainable per-profile scoring, and attribute-level evaluation you can reproduce.

> **Independent personal project.** Built entirely on the public [Open Food Facts](https://openfoodfacts.org) dataset (ODbL) and the public Nutri-Score algorithm. No employer or client proprietary code, data, taxonomy, or rating logic.

---

## Architecture

```
image (Open Food Facts)
   │
   ▼
[1] Ingest & preprocess
   │
   ▼
[2] OCR + layout ──────── Claude vision (transcribe the label)     [offline: Tesseract]
   │
   ▼
[3] Multi-agent extraction (LangGraph, fan-out)
     ├─ NutritionPanelAgent → per-nutrient value + unit
     ├─ IngredientListAgent → ordered ingredient tokens
     └─ ClaimsAgent         → front-of-pack claims
        every field emits: value + evidence span + confidence
   │
   ▼
[4] Validation agent ──── configurable rules engine + cross-field consistency
     ├─ regulatory / plausibility thresholds (sugars, sat-fat, salt, energy)
     ├─ arithmetic sanity (4·carb + 4·protein + 9·fat ≈ energy; salt ≈ 2.5·sodium)
     └─ low-confidence / conflicting fields → routed to review (exception routing)
   │
   ▼
[5] Taxonomy mapping ──── ingredients → class, E-numbers, allergens, ultra-processing
   │
   ▼
[6] Scoring
     ├─ base Nutri-Score grade (A–E) from the extracted attributes
     └─ personalization matrix → red/yellow/green per profile, with reasons
   │
   ▼
[7] Serve ─────────────── FastAPI: sync + batch + real-time job status + CSV export
   │
   ▼
[8] Eval harness ──────── attribute precision/recall/F1, validation catch-rate,
                          score agreement vs Nutri-Score, p50/p95 latency, cost/product
```

## Stack

Python · **LangGraph** (multi-agent extraction/validation graph) · **Claude vision** for OCR + attribute extraction with structured outputs · **Pydantic** typed schema · **FastAPI** + SQLite job store · **Open Food Facts** as data and ground truth · **Nutri-Score** as the scoring benchmark.

> **Note on OCR:** the original design specified Azure Document Intelligence. This build uses a **vision-capable Claude model** to do OCR + layout + attribute extraction in one, which removes the Azure dependency and gives a stronger per-field evidence story. The seam is abstracted (`opie/llm/`), so an Azure DI backend could be added without touching the graph.

---

## Two backends

| Backend | OCR + extraction | When it runs | Metrics |
|---|---|---|---|
| `anthropic` | Claude vision, structured output, per-field evidence + confidence | when an Anthropic credential is present | **real, resume-grade** |
| `offline` | local Tesseract + deterministic heuristic parser | default with no key | plumbing only — **not** representative |

The offline backend exists so the entire system — graph, rules, taxonomy, scoring, API, and eval harness — is testable end-to-end with **no API key**. It is a genuine (weak) extractor and never reads ground truth, but offline eval numbers reflect the heuristic parser, not the Claude-vision pipeline, and are **not** the numbers that go on a resume.

---

## Quickstart

```bash
make install          # venv + editable install
make test             # 63 tests, all offline (no key, no network)

# Truth & deception — catch misleading front-of-pack claims:
opie deception-eval                # confusion matrix + MISLEADING recall + worked examples

# The data flywheel — proves F1 rises + review queue shrinks across rounds:
opie feedback-eval --rounds 5      # deterministic, offline, no key needed
make feedback                      # same, persists capture to SQLite

# Real numbers (needs an Anthropic credential):
export ANTHROPIC_API_KEY=sk-ant-...   # or: ant auth login  +  export OPIE_BACKEND=anthropic
make download         # pull a held-out OFF subset (images + ground truth)
make eval             # run the harness -> results/RESULTS.md + results/eval.json

make serve            # FastAPI on :8000
make demo CODE=<off_product_code>
```

### API

```
GET  /health
POST /v1/analyze              multipart image + optional profiles -> full record (sync)
POST /v1/batch               multiple images -> { job_id }
GET  /v1/jobs/{job_id}       real-time status + results
GET  /v1/jobs/{job_id}/export.csv
```

### Typed output (shape)

```json
{
  "product_id": "...",
  "backend": "anthropic",
  "nutrition": {
    "sugars_100g": {"value": 34, "unit": "g", "confidence": 0.91,
                    "evidence": {"text": "of which sugars 34g"}}
  },
  "ingredients": [{"name": "soy lecithin", "rank": 4, "taxonomy": "additive",
                   "enumber": "E322", "is_allergen": true, "confidence": 0.88}],
  "validation": {"passed": false, "review_required": true,
                 "flags": [{"rule": "sugars_high", "severity": "high"}]},
  "score": {"base_grade": "D", "base_points": 14,
            "personalized": [{"profile": "diabetic", "flag": "red",
                              "why": ["34 g sugars is high (>= 15)."]}]}
}
```

---

## Metrics the harness measures (all reproducible)

- **Attribute extraction precision / recall / F1** — per nutrition field (numeric, tolerance-based) and per ingredient (set-based), vs OFF ground truth.
- **Validation catch-rate** — inject known inconsistencies (sugars > carbs, sat-fat > fat, energy ≠ macros, salt ≠ 2.5·sodium, >100g/100g) into ground-truth panels and measure the share the rules engine flags, per error type and overall.
- **Score agreement vs Nutri-Score** — exact-grade accuracy, within-one-grade accuracy, and Cohen's κ against OFF's published `nutriscore_grade`.
- **Performance** — p50 / p95 latency and mean cost per product.

## Results

> **Pending a live run.** These are filled in by `make eval` on the Anthropic backend against a held-out OFF set, and are the only numbers permitted on a resume. Offline numbers are diagnostic only. Nothing is claimed until measured here.

| Metric | Value |
|---|---|
| Nutrition F1 | _pending_ |
| Ingredient F1 | _pending_ |
| Validation catch-rate | _pending_ |
| Score agreement (accuracy / κ) | _pending_ |
| p50 / p95 latency | _pending_ |
| Cost / product | _pending_ |

---

## Truth & deception layer — catch misleading claims

The headline feature. For every front-of-pack marketing claim, OPIE decides whether the product's actual nutrition and ingredients **support** it, and flags deceptive claims with an explainable, evidence-backed verdict grounded in **public regulatory definitions** — every threshold is cited in code.

```
claim text ──▶ normalize to a typed claim id (sugar_free, high_protein, gluten_free, ...)
                     │
                     ▼
        regulatory definition (EU 1924/2006 · Codex CAC/GL 23-1997 · FSSAI 2020 · ...)
                     │  cross-reference against extracted nutrition + ingredients
                     ▼
        verdict  SUPPORTED | MISLEADING | UNVERIFIABLE
                 + basis (which rule + citation), offending_fact, evidence, severity
                     │
                     ▼
        health-halo check: a technically-true claim on a grade D/E or very-high-sugar product
```

Every verdict cites the rule and the fact that triggered it. Example (a chocolate bar, base grade E):

| Claim | Verdict | Why |
|---|---|---|
| `sugar free` | **MISLEADING** (high) | sugars 52g exceeds the 0.5g/100g `sugar_free` limit |
| `high in protein` | **MISLEADING** (high) | protein supplies only 5% of energy (need ≥20%) |
| `gluten free` | SUPPORTED · **HEALTH-HALO** | true, but distracts from a grade-E profile |
| `organic` | UNVERIFIABLE | requires certification, not derivable from facts |

Some claims are honestly **UNVERIFIABLE** from extracted facts (`organic`, `fortified`, `source_of_vitamins`) rather than guessed — the detector says so.

### Deception eval (reproducible: `opie deception-eval`)

Labeled set built from the corpus: a fixed claim roster is attached to every product, producing honest cases **and** adversarial cases (a claim attached to a product whose facts violate its definition). Ground truth = the public definition adjudicated on the product's **true** facts; predictions run on **extracted** facts, so the recall gap from 1.0 is the extraction-induced miss rate.

Confusion matrix on a 400-product corpus (5,200 claim cases), rows = ground truth:

| GT \ Pred | SUPPORTED | MISLEADING | UNVERIFIABLE |
|---|---|---|---|
| **SUPPORTED** | 1654 | 8 | 0 |
| **MISLEADING** | 226 | 3312 | 0 |
| **UNVERIFIABLE** | 0 | 0 | 800 |

**MISLEADING recall (deception catch-rate): 0.936** on extracted facts (precision 0.998, F1 0.966) · **1.000 on oracle facts** (the rule set is self-consistent, so the 6.4% gap is entirely extraction noise — exactly what the flywheel below reduces). Macro-F1 0.967.

> Optimized for **recall on MISLEADING** — how many deceptive labels we catch — because a missed deceptive claim is the costly error.

## Data flywheel — the system gets more accurate over time

OPIE closes a real-world validation feedback loop: corrections flow back in, accuracy rises, and the low-confidence review queue shrinks. This is the part that makes the project a *system*, not a parser.

```
extraction run ──▶ capture every field (value, confidence, evidence)  [SQLite]
       │
       ▼
route low-confidence OR validator-flagged fields ──▶ review queue
       │
       ▼
reviewer oracle (Open Food Facts record) ──▶ correction records (field, extracted, correct)
       │
       ├─▶ correction memory: embed each case; at extraction time retrieve k-NN past
       │   corrections and apply the learned transform (numeric ratio / token-normalization
       │   map) or inject them as few-shot exemplars into the vision agent's prompt
       │
       └─▶ confidence recalibration: refit per-field reliability -> the review-queue
           threshold moves as calibration sharpens
       │
       ▼
next round measured COLD (disjoint products, no leakage) ──▶ F1 up, queue down, ECE down
```

**Learning without fine-tune (v1):** corrections are learned by *retrieval* (k-NN over embedded correction cases → median correct/extracted ratio for numeric fields; a learned corrupted→correct token map for ingredients) and by *confidence recalibration* (reliability binning). No model weights change. The corrections learn transferable transforms, not product identities, so measuring on disjoint rounds is leakage-free.

### Round-over-round result (reproducible: `opie feedback-eval --rounds 5`)

Deterministic run on a seeded 400-product corpus, 5 disjoint rounds, simulated-error extractor. Each round is measured cold with state learned only from earlier rounds; the frozen baseline runs the same rounds with the loop disabled.

| Round | Learned F1 | Baseline F1 | Review queue | ECE | Corrections learned |
|---|---|---|---|---|---|
| 1 | 0.822 | 0.822 | 709 | 0.204 | 450 |
| 2 | 0.945 | 0.839 | 106 | 0.103 | 556 |
| 3 | 0.954 | 0.844 | 102 | 0.062 | 658 |
| 4 | 0.942 | 0.830 | 99  | 0.046 | 757 |
| 5 | **0.956** | 0.835 | **96** | **0.050** | 850 |

**Attribute F1 0.822 → 0.956** (+0.13 across rounds, +0.12 vs the flat no-feedback baseline) · **review queue 709 → 96** (−86%) · **calibration ECE 0.204 → 0.050**. The flywheel lifts accuracy; it isn't just plumbing.

> **On the numbers:** the reproducible loop above runs on a *simulated-error extractor* — a seeded, deterministic stand-in that makes the error modes a real OCR/LLM makes (decimal-comma scale errors, salt/sodium unit swaps, OCR token corruptions). The measurements of the loop's effect are real and reproducible; the same feedback loop wraps the Claude-vision backend via `opie feedback-eval --backend anthropic` when a key + downloaded OFF images are present. As with the rest of the project, real-extractor numbers replace these before anything goes on a resume.

## Honesty guardrails

- Public data, own code. Open Food Facts / Nutri-Score only. No employer or client proprietary code, data, taxonomy, or rating logic.
- Build first, resume second. No resume bullet until it exists, is pushed, and every metric is a real number measured on a held-out set.
- Every extraction layer is defensible unaided — that is the point of building it.

## License

MIT (code). Data and Nutri-Score are public (ODbL / open algorithm). See `LICENSE`.
