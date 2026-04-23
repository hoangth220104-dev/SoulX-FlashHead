"""
Tests for the FlashHead server generation endpoints and job lifecycle.

These tests use a stub provider — no GPU or real model weights required.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from flashhead_server.contracts import GenerationRequest, JobStatus


# ── Stub provider ──────────────────────────────────────────────────────────

class _StubProvider:
    """Always healthy, writes a tiny placeholder MP4 on generate()."""

    provider_name = "stub"

    def health_check(self) -> bool:
        return True

    def supported_modes(self) -> list[str]:
        return ["lite", "pro"]

    def generate(self, request: GenerationRequest, output_path: Path) -> None:
        # Write a minimal non-empty file to satisfy the download endpoint.
        output_path.write_bytes(b"FAKE_MP4_DATA")


class _FailingProvider(_StubProvider):
    """Always raises on generate()."""

    provider_name = "stub_failing"

    def generate(self, request: GenerationRequest, output_path: Path) -> None:
        raise RuntimeError("Simulated GPU OOM")


class _NotReadyProvider(_StubProvider):
    provider_name = "stub_not_ready"

    def health_check(self) -> bool:
        return False


# ── App fixture ────────────────────────────────────────────────────────────

def _make_client(provider=None, tmp_path=None) -> TestClient:
    import os

    import flashhead_server.app as app_module
    import flashhead_server.routes.generation as gen_module
    from flashhead_server.app import create_app
    from flashhead_server.config import Settings

    settings = Settings(
        host="127.0.0.1",
        port=8002,
        debug=False,
        environment="test",
        ckpt_dir="/fake/ckpt",
        wav2vec_dir="/fake/wav2vec",
        model_mode="lite",
        use_face_crop=False,
        output_dir=tmp_path or Path("/tmp/flashhead_test"),
        flashhead_root=Path("/fake/root"),
    )

    original_build = app_module._build_provider
    original_chdir = os.chdir

    def _stub_build(s):
        return provider or _StubProvider()

    app_module._build_provider = _stub_build
    os.chdir = lambda _: None

    try:
        app = create_app(settings)
    finally:
        app_module._build_provider = original_build
        os.chdir = original_chdir

    # Reset in-process job store so tests don't bleed into each other
    with gen_module._jobs_lock:
        gen_module._jobs.clear()

    return TestClient(app)


@pytest.fixture
def client(tmp_path):
    return _make_client(provider=_StubProvider(), tmp_path=tmp_path)


@pytest.fixture
def failing_client(tmp_path):
    return _make_client(provider=_FailingProvider(), tmp_path=tmp_path)


@pytest.fixture
def not_ready_client(tmp_path):
    return _make_client(provider=_NotReadyProvider(), tmp_path=tmp_path)


# ── System endpoint tests ──────────────────────────────────────────────────

def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["provider_ready"] is True


def test_info(client):
    resp = client.get("/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "flashhead-server"
    assert "lite" in body["supported_modes"]


def test_health_not_ready(not_ready_client):
    resp = not_ready_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["provider_ready"] is False


# ── Generation endpoint tests ──────────────────────────────────────────────

def _make_files():
    return {
        "face_image": ("portrait.jpg", b"\xff\xd8\xff_fake_jpg", "image/jpeg"),
        "audio": ("speech.wav", b"RIFF_fake_wav", "audio/wav"),
    }


def test_create_job_returns_202(client):
    resp = client.post("/api/generate", files=_make_files())
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["status"] == "pending"


def test_create_job_invalid_mode(client):
    files = _make_files()
    resp = client.post("/api/generate", files=files, data={"mode": "turbo"})
    assert resp.status_code == 422


def test_create_job_invalid_encode_mode(client):
    files = _make_files()
    resp = client.post("/api/generate", files=files, data={"audio_encode_mode": "batch"})
    assert resp.status_code == 422


def test_create_job_provider_not_ready(not_ready_client):
    resp = not_ready_client.post("/api/generate", files=_make_files())
    assert resp.status_code == 503


# ── Job status & download tests ────────────────────────────────────────────

def test_get_job_not_found(client):
    resp = client.get("/api/jobs/nonexistent-id")
    assert resp.status_code == 404


def test_job_lifecycle_success(client):
    # Create job
    resp = client.post("/api/generate", files=_make_files())
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    # Poll until terminal (stub is synchronous via TestClient background tasks)
    for _ in range(20):
        status_resp = client.get(f"/api/jobs/{job_id}")
        assert status_resp.status_code == 200
        body = status_resp.json()
        if body["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert body["status"] == "completed"
    assert body["video_ready"] is True
    assert body["duration_seconds"] is not None


def test_job_lifecycle_failure(failing_client):
    resp = failing_client.post("/api/generate", files=_make_files())
    job_id = resp.json()["job_id"]

    for _ in range(20):
        status_resp = failing_client.get(f"/api/jobs/{job_id}")
        body = status_resp.json()
        if body["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert body["status"] == "failed"
    assert "Simulated GPU OOM" in body["error_message"]
    assert body["video_ready"] is False


def test_download_video_before_completion(client):
    resp = client.post("/api/generate", files=_make_files())
    job_id = resp.json()["job_id"]

    # Immediately try to download — may still be pending/processing in race,
    # but we just verify the endpoint exists and returns 404 when not ready.
    # (TestClient background tasks run synchronously so job may already be done.)
    status = client.get(f"/api/jobs/{job_id}").json()["status"]
    if status != "completed":
        dl = client.get(f"/api/jobs/{job_id}/video")
        assert dl.status_code == 404


def test_download_video_success(client, tmp_path):
    resp = client.post("/api/generate", files=_make_files())
    job_id = resp.json()["job_id"]

    # Wait for completion
    for _ in range(20):
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] == "completed":
            break
        time.sleep(0.05)

    dl = client.get(f"/api/jobs/{job_id}/video")
    assert dl.status_code == 200
    assert dl.headers["content-type"] == "video/mp4"
    assert dl.content == b"FAKE_MP4_DATA"


def test_download_video_failed_job(failing_client):
    resp = failing_client.post("/api/generate", files=_make_files())
    job_id = resp.json()["job_id"]

    for _ in range(20):
        body = failing_client.get(f"/api/jobs/{job_id}").json()
        if body["status"] == "failed":
            break
        time.sleep(0.05)

    dl = failing_client.get(f"/api/jobs/{job_id}/video")
    assert dl.status_code == 404
