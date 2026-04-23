from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from flashhead_server.contracts import GenerationRequest


@runtime_checkable
class TalkingHeadProvider(Protocol):
    provider_name: str

    def generate(self, request: GenerationRequest, output_path: Path) -> None:
        """Run inference and write MP4 to output_path."""
        ...

    def health_check(self) -> bool:
        """Return True if the model is loaded and ready."""
        ...

    def supported_modes(self) -> list[str]:
        """Return list of supported model mode strings."""
        ...
