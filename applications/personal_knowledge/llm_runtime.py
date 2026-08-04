"""Runtime LLM target selection for the personal knowledge application."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from adaptive_agent_runtime.llm import BackendAvailability
from adaptive_agent_runtime.llm.capabilities import (
    ArtifactGenerationCapability,
    MemoryExtractionCapability,
)

from applications.research_agent.llm_deployment import (
    ResearchLLMDeployment,
    ResearchLLMDeploymentConfig,
    build_research_llm_deployment,
)
from applications.research_agent.web_llm import WebLLMSettings


@dataclass(frozen=True)
class PersonalKnowledgeLLMDeployment:
    generator: ArtifactGenerationCapability
    memory_extractor: MemoryExtractionCapability | None
    target_name: str
    target_id: str
    model_id: str
    owned_deployment: ResearchLLMDeployment


class PersonalKnowledgeLLMManager:
    """Reuse the workspace target catalog while keeping app behavior separate."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        active_target: str | None = None,
    ) -> None:
        self._settings = WebLLMSettings(
            config_path,
            active_target=active_target,
            activate_config_default=False,
        )

    def describe(self) -> dict[str, object]:
        return self._settings.describe()

    def activate(
        self,
        target_name: str,
        *,
        api_key: str | None = None,
        model_id: str | None = None,
    ) -> PersonalKnowledgeLLMDeployment:
        config = self._settings.activate(
            target_name,
            api_key=api_key,
            model_id=model_id,
        )
        return self._build(target_name, config)

    def load_active(self) -> PersonalKnowledgeLLMDeployment | None:
        config = self._settings.load_active()
        if config is None:
            return None
        active = self._settings.active_target
        if active is None:
            return None
        return self._build(active, config)

    async def discover_models(
        self,
        target_name: str,
        *,
        api_key: str | None = None,
    ) -> dict[str, object]:
        config = self._settings.load_target(target_name, api_key=api_key)
        deployment = build_research_llm_deployment(config)
        probe = await deployment.probe()
        availability = probe.availability
        if (
            availability is BackendAvailability.AVAILABLE
            and "authentication_missing" in probe.diagnostics
        ):
            availability = BackendAvailability.AUTH_REQUIRED
        return {
            "availability": availability.value,
            "target": target_name,
            "currentModel": deployment.model_id,
            "models": list(probe.available_model_ids),
            "runtimeVersion": probe.runtime_version,
            "authMethod": probe.active_auth_method,
            "diagnostics": list(probe.diagnostics),
        }

    async def probe_active(self) -> dict[str, object]:
        deployment = self.load_active()
        if deployment is None:
            raise ValueError("select an LLM target before probing")
        return await self._probe_deployment(deployment)

    async def probe_selection(
        self,
        target_name: str,
        *,
        api_key: str | None = None,
        model_id: str | None = None,
    ) -> tuple[dict[str, object], PersonalKnowledgeLLMDeployment]:
        """Probe the form selection without making it active first."""

        config = self._settings.load_target(target_name, api_key=api_key)
        if model_id is not None:
            selected_model = model_id.strip()
            if not selected_model:
                raise ValueError("model must be a non-empty string")
            if len(selected_model) > 256:
                raise ValueError("model is too long")
            config = replace(
                config,
                target=config.target.model_copy(
                    update={"model_id": selected_model},
                ),
            )
        deployment = self._build(target_name, config)
        return await self._probe_deployment(deployment), deployment

    @staticmethod
    async def _probe_deployment(
        deployment: PersonalKnowledgeLLMDeployment,
    ) -> dict[str, object]:
        probe = await deployment.owned_deployment.probe()
        availability = probe.availability
        if (
            availability is BackendAvailability.AVAILABLE
            and "authentication_missing" in probe.diagnostics
        ):
            availability = BackendAvailability.AUTH_REQUIRED
        return {
            "availability": availability.value,
            "target": deployment.target_name,
            "targetId": deployment.target_id,
            "model": deployment.model_id,
            "availableModels": list(probe.available_model_ids),
            "runtimeVersion": probe.runtime_version,
            "authMethod": probe.active_auth_method,
            "diagnostics": list(probe.diagnostics),
        }

    @staticmethod
    def _build(
        target_name: str,
        config: ResearchLLMDeploymentConfig,
    ) -> PersonalKnowledgeLLMDeployment:
        # Research's deployment config is the workspace-level parser/composer for
        # the same Runtime target catalog. No Research Agent workflow is reused.
        deployment = build_research_llm_deployment(config)
        generator = deployment.cognitive_capabilities.report_generator
        if generator is None:
            raise ValueError("selected target does not enable generation")
        return PersonalKnowledgeLLMDeployment(
            generator=generator,
            memory_extractor=deployment.cognitive_capabilities.memory_extractor,
            target_name=target_name,
            target_id=deployment.target_id,
            model_id=deployment.model_id,
            owned_deployment=deployment,
        )
