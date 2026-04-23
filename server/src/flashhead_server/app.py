from __future__ import annotations

import logging
import os

from fastapi import FastAPI

from flashhead_server.config import Settings
from flashhead_server.routes.generation import router as generation_router
from flashhead_server.routes.system import router as system_router

logger = logging.getLogger(__name__)


def _build_provider(settings: Settings):
    from flashhead_server.providers.flashhead_provider import FlashHeadProvider
    return FlashHeadProvider(
        ckpt_dir=settings.ckpt_dir,
        wav2vec_dir=settings.wav2vec_dir,
        model_mode=settings.model_mode,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings.from_env()

    # flash_head/inference.py opens the YAML at import time using a relative
    # path, so we must chdir to the SoulX-FlashHead root before importing it.
    os.chdir(resolved.flashhead_root)

    resolved.output_dir.mkdir(parents=True, exist_ok=True)

    provider = _build_provider(resolved)
    logger.info(
        "FlashHead server starting (mode=%s, env=%s)",
        resolved.model_mode,
        resolved.environment,
    )

    app = FastAPI(
        title="FlashHead Server",
        description="Talking-head video generation service for AutoRepr",
        version="1.0.0",
    )

    app.state.settings = resolved
    app.state.provider = provider

    app.include_router(system_router)
    app.include_router(generation_router)

    return app
