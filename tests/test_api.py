import time

import pytest
from conftest import FakeExtractor

from opie.api import app as app_module
from opie.pipeline.graph import Pipeline


@pytest.fixture
def client(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    ex = FakeExtractor(
        nutrition={"sugars_100g": 34, "saturated_fat_100g": 12, "salt_100g": 1.2,
                   "energy_kcal_100g": 512},
        ingredients=["wheat flour", "sugar"],
        claims=["high in protein"],
    )
    app_module._pipeline = Pipeline(extractor=ex)
    app_module._jobs = app_module.JobStore(db_path=tmp_path / "jobs.sqlite")
    return fastapi_testclient.TestClient(app_module.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "diabetic" in body["profiles"]


def test_analyze_sync(client):
    r = client.post(
        "/v1/analyze",
        files={"image": ("label.jpg", b"fake-bytes", "image/jpeg")},
        data={"product_id": "p1", "profiles": "general,diabetic"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["product_id"] == "p1"
    assert body["nutrition"]["sugars_100g"]["value"] == 34
    assert len(body["score"]["personalized"]) == 2


def test_analyze_rejects_unknown_profile(client):
    r = client.post(
        "/v1/analyze",
        files={"image": ("label.jpg", b"x", "image/jpeg")},
        data={"profiles": "nonsense"},
    )
    assert r.status_code == 400


def test_batch_and_job_status_and_csv(client):
    r = client.post(
        "/v1/batch",
        files=[
            ("images", ("a.jpg", b"aaa", "image/jpeg")),
            ("images", ("b.jpg", b"bbb", "image/jpeg")),
        ],
        data={"profiles": "general"},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # poll until the background worker finishes
    for _ in range(50):
        status = client.get(f"/v1/jobs/{job_id}").json()
        if status["status"] == "succeeded":
            break
        time.sleep(0.05)
    assert status["status"] == "succeeded"
    assert status["done"] == 2
    assert len(status["results"]) == 2

    csv = client.get(f"/v1/jobs/{job_id}/export.csv")
    assert csv.status_code == 200
    assert "product_id" in csv.text
    assert "base_grade" in csv.text


def test_job_not_found(client):
    assert client.get("/v1/jobs/nope").status_code == 404
