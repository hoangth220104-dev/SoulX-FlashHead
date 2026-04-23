from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _read_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    debug: bool
    environment: str
    # FlashHead model directories — must be set in production
    ckpt_dir: str
    wav2vec_dir: str
    model_mode: str          # "lite" or "pro"
    use_face_crop: bool
    output_dir: Path
    # The server must be started from the SoulX-FlashHead/ root so that
    # flash_head/configs/infer_params.yaml is reachable by relative path.
    flashhead_root: Path

    @classmethod
    def from_env(cls) -> "Settings":
        # Default root: parent of the server/ directory
        default_root = Path(__file__).resolve().parents[4]
        flashhead_root = Path(os.getenv("FLASHHEAD_ROOT", str(default_root)))

        default_output = flashhead_root / "server" / "outputs"
        output_dir_raw = os.getenv("FLASHHEAD_OUTPUT_DIR", str(default_output))
        output_dir = Path(output_dir_raw)
        if not output_dir.is_absolute():
            output_dir = flashhead_root / output_dir

        environment = os.getenv("FLASHHEAD_ENV", "development")
        debug = _read_bool("FLASHHEAD_DEBUG", default=environment != "production")

        return cls(
            host=os.getenv("FLASHHEAD_HOST", "127.0.0.1"),
            port=int(os.getenv("FLASHHEAD_PORT", "8002")),
            debug=debug,
            environment=environment,
            ckpt_dir=os.getenv("FLASHHEAD_CKPT_DIR", ""),
            wav2vec_dir=os.getenv("FLASHHEAD_WAV2VEC_DIR", ""),
            model_mode=os.getenv("FLASHHEAD_MODEL_MODE", "lite"),
            use_face_crop=_read_bool("FLASHHEAD_USE_FACE_CROP", default=True),
            output_dir=output_dir,
            flashhead_root=flashhead_root,
        )
