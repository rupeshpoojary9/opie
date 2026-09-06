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
make test             # 37 tests, all offline (no key, no network)

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

## Honesty guardrails

- Public data, own code. Open Food Facts / Nutri-Score only. No employer or client proprietary code, data, taxonomy, or rating logic.
- Build first, resume second. No resume bullet until it exists, is pushed, and every metric is a real number measured on a held-out set.
- Every extraction layer is defensible unaided — that is the point of building it.

## License

MIT (code). Data and Nutri-Score are public (ODbL / open algorithm). See `LICENSE`.
