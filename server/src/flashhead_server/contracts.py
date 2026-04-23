from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModelMode(str, Enum):
    lite = "lite"
    pro = "pro"


class AudioEncodeMode(str, Enum):
    stream = "stream"
    once = "once"


class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


@dataclass(frozen=True)
class GenerationRequest:
    job_id: str
    face_image_path: str
    audio_path: str
    mode: ModelMode = ModelMode.lite
    use_face_crop: bool = True
    audio_encode_mode: AudioEncodeMode = AudioEncodeMode.stream
    base_seed: int = 42


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    face_image_path: str
    audio_path: str
    mode: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    video_path: Optional[str] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "mode": self.mode,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "video_ready": self.video_path is not None,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
        }
