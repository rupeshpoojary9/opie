"""FastAPI delivery: sync analyze, batch submit, real-time job status, CSV export."""
from __future__ import annotations

import csv
import io
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from opie import __version__
from opie.api.jobs import JobItem, JobStore
from opie.config import SETTINGS
from opie.pipeline.graph import Pipeline
from opie.scoring.personalization import PROFILES

app = FastAPI(title="OPIE — Open Product-Intelligence Engine", version=__version__)

_pipeline: Optional[Pipeline] = None
_jobs: Optional[JobStore] = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


def get_jobs() -> JobStore:
    global _jobs
    if _jobs is None:
        _jobs = JobStore()
    return _jobs


def _parse_profiles(profiles: Optional[str]) -> list[str]:
    if not profiles:
        return ["general"]
    requested = [p.strip() for p in profiles.split(",") if p.strip()]
    unknown = [p for p in requested if p not in PROFILES]
    if unknown:
        raise HTTPException(400, f"Unknown profile(s): {unknown}. Known: {list(PROFILES)}")
    return requested or ["general"]


@app.get("/health")
def health() -> dict:
    p = get_pipeline()
    return {
        "status": "ok",
        "version": __version__,
        "backend": p.extractor.backend_name,
        "engine": p.engine,
        "model": SETTINGS.model if p.extractor.backend_name == "anthropic" else None,
        "profiles": list(PROFILES),
    }


@app.post("/v1/analyze")
async def analyze(
    image: UploadFile = File(...),
    product_id: str = Form("uploaded"),
    profiles: Optional[str] = Form(None),
) -> dict:
    """Synchronous single-image analysis."""
    profile_list = _parse_profiles(profiles)
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "Empty image upload.")
    product = get_pipeline().run(product_id, image_bytes, profiles=profile_list)
    return product.model_dump()


@app.post("/v1/batch")
async def batch(
    images: list[UploadFile] = File(...),
    profiles: Optional[str] = Form(None),
) -> dict:
    """Submit a batch; returns a job_id to poll."""
    profile_list = _parse_profiles(profiles)
    items = [JobItem(product_id=img.filename or f"item-{i}", image_bytes=await img.read())
             for i, img in enumerate(images)]
    if not items:
        raise HTTPException(400, "No images in batch.")
    jobs = get_jobs()
    pipeline = get_pipeline()
    job_id = jobs.create(total=len(items))
    jobs.run_async(job_id, items, lambda pid, data: pipeline.run(pid, data, profiles=profile_list))
    return {"job_id": job_id, "status": "queued", "total": len(items)}


@app.get("/v1/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    status = get_jobs().status(job_id)
    if status is None:
        raise HTTPException(404, f"No such job: {job_id}")
    return status


@app.get("/v1/jobs/{job_id}/export.csv")
def job_csv(job_id: str) -> StreamingResponse:
    status = get_jobs().status(job_id)
    if status is None:
        raise HTTPException(404, f"No such job: {job_id}")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "product_id", "backend", "base_grade", "base_points",
        "energy_kcal_100g", "sugars_100g", "saturated_fat_100g", "salt_100g",
        "n_ingredients", "validation_passed", "review_required", "personalized_flags",
    ])
    for r in status["results"]:
        nut = r.get("nutrition", {})
        writer.writerow([
            r.get("product_id"),
            r.get("backend"),
            (r.get("score") or {}).get("base_grade"),
            (r.get("score") or {}).get("base_points"),
            (nut.get("energy_kcal_100g") or {}).get("value"),
            (nut.get("sugars_100g") or {}).get("value"),
            (nut.get("saturated_fat_100g") or {}).get("value"),
            (nut.get("salt_100g") or {}).get("value"),
            len(r.get("ingredients", [])),
            (r.get("validation") or {}).get("passed"),
            (r.get("validation") or {}).get("review_required"),
            ";".join(f"{p['profile']}={p['flag']}" for p in (r.get("score") or {}).get("personalized", [])),
        ])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="opie_{job_id}.csv"'},
    )
