from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["system"])


@router.get("/health")
def healthcheck(request: Request) -> dict[str, object]:
    provider = request.app.state.provider
    settings = request.app.state.settings
    return {
        "status": "ok",
        "provider": provider.provider_name,
        "provider_ready": provider.health_check(),
        "model_mode": settings.model_mode,
    }


@router.get("/info")
def server_info(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    provider = request.app.state.provider
    return {
        "service": "flashhead-server",
        "environment": settings.environment,
        "model_mode": settings.model_mode,
        "supported_modes": provider.supported_modes(),
        "use_face_crop": settings.use_face_crop,
        "output_dir": str(settings.output_dir),
    }
