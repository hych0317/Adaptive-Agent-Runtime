"""In-memory inference backend registration with identity enforcement."""

from __future__ import annotations

from adaptive_agent_runtime.llm.contracts import InferenceBackend
from adaptive_agent_runtime.llm.errors import (
    InferenceTargetAlreadyRegisteredError,
    InferenceTargetNotFoundError,
)
from adaptive_agent_runtime.llm.models import InferenceTargetProfile


class InMemoryInferenceBackendRegistry:
    module_id = "llm.inference_registry.in_memory"

    def __init__(self) -> None:
        self._backends: dict[str, InferenceBackend] = {}

    def register(self, backend: InferenceBackend) -> None:
        if not isinstance(backend.profile, InferenceTargetProfile):
            raise TypeError("inference backend requires an inference profile")
        if backend.target_id != backend.profile.target_id:
            raise ValueError("backend identity does not match its profile")
        if backend.target_id in self._backends:
            raise InferenceTargetAlreadyRegisteredError(backend.target_id)
        self._backends[backend.target_id] = backend

    def backend_for(self, target_id: str) -> InferenceBackend:
        try:
            return self._backends[target_id]
        except KeyError as exc:
            raise InferenceTargetNotFoundError(target_id) from exc

    def list_profiles(self) -> tuple[InferenceTargetProfile, ...]:
        return tuple(
            self._backends[target_id].profile
            for target_id in sorted(self._backends)
        )
