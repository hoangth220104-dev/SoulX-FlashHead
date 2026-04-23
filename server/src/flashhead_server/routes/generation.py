from __future__ import annotations

import time
import threading
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse

from flashhead_server.contracts import (
    AudioEncodeMode,
    GenerationRequest,
    JobRecord,
    JobStatus,
    ModelMode,
)

router = APIRouter(prefix="/api", tags=["generation"])

# ── In-process job store ───────────────────────────────────────────────────

_jobs: dict[str, JobRecord] = {}
_jobs_lock = threading.Lock()


def _get_job(job_id: str) -> JobRecord:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


def _update_job(job: JobRecord, **kwargs) -> None:
    for k, v in kwargs.items():
        setattr(job, k, v)
    job.updated_at = time.time()


# ── Background worker ──────────────────────────────────────────────────────

def _run_generation(job: JobRecord, request: GenerationRequest, output_path: Path, provider) -> None:
    with _jobs_lock:
        _update_job(job, status=JobStatus.processing)

    t0 = time.time()
    try:
        provider.generate(request, output_path)
        elapsed = round(time.time() - t0, 2)
        with _jobs_lock:
            _update_job(
                job,
                status=JobStatus.completed,
                video_path=str(output_path),
                duration_seconds=elapsed,
            )
    except Exception as exc:
        with _jobs_lock:
            _update_job(job, status=JobStatus.failed, error_message=str(exc))


# ── Routes ─────────────────────────────────────────────────────────────────

@router.post("/generate", status_code=202)
async def create_generation_job(
    request: Request,
    background_tasks: BackgroundTasks,
    face_image: Annotated[UploadFile, File(description="Portrait image (JPEG/PNG)")],
    audio: Annotated[UploadFile, File(description="Audio track (WAV, 16kHz mono)")],
    mode: Annotated[str, Form()] = "lite",
    use_face_crop: Annotated[bool, Form()] = True,
    audio_encode_mode: Annotated[str, Form()] = "stream",
    base_seed: Annotated[int, Form()] = 42,
) -> dict[str, object]:
    try:
        model_mode = ModelMode(mode)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid mode '{mode}'. Use 'lite' or 'pro'.")

    try:
        encode_mode = AudioEncodeMode(audio_encode_mode)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid audio_encode_mode '{audio_encode_mode}'. Use 'stream' or 'once'.")

    settings = request.app.state.settings
    provider = request.app.state.provider

    if not provider.health_check():
        raise HTTPException(status_code=503, detail="FlashHead model not ready")

    job_id = str(uuid.uuid4())
    upload_dir: Path = settings.output_dir / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Persist uploaded files
    image_suffix = Path(face_image.filename or "image.jpg").suffix or ".jpg"
    image_path = upload_dir / f"face_image{image_suffix}"
    image_path.write_bytes(await face_image.read())

    audio_suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    audio_path = upload_dir / f"audio{audio_suffix}"
    audio_path.write_bytes(await audio.read())

    output_path = upload_dir / "output.mp4"

    gen_request = GenerationRequest(
        job_id=job_id,
        face_image_path=str(image_path),
        audio_path=str(audio_path),
        mode=model_mode,
        use_face_crop=use_face_crop,
        audio_encode_mode=encode_mode,
        base_seed=base_seed,
    )

    job = JobRecord(
        job_id=job_id,
        status=JobStatus.pending,
        face_image_path=str(image_path),
        audio_path=str(audio_path),
        mode=mode,
    )

    with _jobs_lock:
        _jobs[job_id] = job

    background_tasks.add_task(_run_generation, job, gen_request, output_path, provider)

    return {"job_id": job_id, "status": "pending"}


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> dict[str, object]:
    return _get_job(job_id).to_dict()


@router.get("/jobs/{job_id}/video")
def download_video(job_id: str) -> FileResponse:
    job = _get_job(job_id)

    if job.status != JobStatus.completed or job.video_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Video not ready (current status: {job.status.value})",
        )

    video_file = Path(job.video_path)
    if not video_file.exists():
        raise HTTPException(status_code=404, detail="Video file missing from disk")

    return FileResponse(
        path=str(video_file),
        media_type="video/mp4",
        filename=f"talking_head_{job_id[:8]}.mp4",
    )
