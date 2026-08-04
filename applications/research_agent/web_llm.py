"""Local-only LLM settings for the Research Agent web application."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from threading import Lock
import tomllib
from typing import Any, cast

from adaptive_agent_runtime.llm import (
    LLMTargetSelection,
    TOMLProviderConfigRepository,
)

from applications.research_agent.cli import load_llm_config_file
from applications.research_agent.llm_deployment import (
    ResearchLLMDeploymentConfig,
)


_CLI_SERVICES = frozenset({"codex_cli", "claude_code"})
_NO_KEY_SERVICES = frozenset({*_CLI_SERVICES, "local"})


@dataclass(frozen=True)
class WebLLMTarget:
    """Sanitized target metadata safe to return to the browser."""

    name: str
    service: str
    model: str
    structured_output: str
    requires_api_key: bool
    credential_configured: bool
    credential_source: str
    active: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "service": self.service,
            "model": self.model,
            "structuredOutput": self.structured_output,
            "requiresApiKey": self.requires_api_key,
            "credentialConfigured": self.credential_configured,
            "credentialSource": self.credential_source,
            "active": self.active,
        }


class WebLLMSettings:
    """Manage one fixed public config and its sibling private key file.

    The browser may select only target names already declared in ``llm.toml``.
    It cannot choose paths or mutate non-secret Runtime target definitions.
    """

    def __init__(
        self,
        config_path: str | Path,
        *,
        active_target: str | None = None,
        activate_config_default: bool = False,
    ) -> None:
        self._config_path = Path(config_path).resolve()
        self._private_path = self._config_path.with_name(
            f"{self._config_path.stem}.local{self._config_path.suffix}"
        )
        self._lock = Lock()
        self._write_lock = Lock()
        default_target, targets = self._read_public_config()
        selected = default_target if activate_config_default else active_target
        if selected is not None and selected not in targets:
            raise ValueError(f"LLM target '{selected}' is not configured")
        self._active_target = selected

    @property
    def active_target(self) -> str | None:
        with self._lock:
            return self._active_target

    def describe(self) -> dict[str, object]:
        default_target, targets = self._read_public_config()
        private_keys = self._load_private_keys()
        selections = self._load_target_selections()
        with self._lock:
            active_target = self._active_target
        summaries = [
            self._summarize_target(
                name,
                table,
                private_keys=private_keys,
                selections=selections,
                active_target=active_target,
            ).as_dict()
            for name, table in targets.items()
        ]
        return {
            "activeTarget": active_target,
            "defaultTarget": default_target,
            "targets": summaries,
            "privateConfig": self._private_path.name,
        }

    def activate(
        self,
        target_name: str,
        *,
        api_key: str | None = None,
        model_id: str | None = None,
    ) -> ResearchLLMDeploymentConfig:
        target = target_name.strip()
        config = self.load_target(target, api_key=api_key)
        if model_id is not None:
            selected_model = model_id.strip()
            if not selected_model:
                raise ValueError("model must be a non-empty string")
            if len(selected_model) > 256:
                raise ValueError("model is too long")
            with self._write_lock:
                repository = TOMLProviderConfigRepository(self._private_path)
                current = (
                    repository.load_target_selections().get(target)
                    if self._private_path.is_file()
                    else None
                )
                repository.save_target_selection(
                    target,
                    LLMTargetSelection(
                        model_id=selected_model,
                        reasoning_effort=(
                            current.reasoning_effort
                            if current is not None
                            else None
                        ),
                    ),
                )
            config = self.load_target(target)
        with self._lock:
            self._active_target = target
        return config

    def load_target(
        self,
        target_name: str,
        *,
        api_key: str | None = None,
    ) -> ResearchLLMDeploymentConfig:
        """Load one configured target without changing the active selection."""

        target = target_name.strip()
        if not target:
            raise ValueError("target must be a non-empty string")
        _, targets = self._read_public_config()
        table = targets.get(target)
        if table is None:
            raise ValueError(f"LLM target '{target}' is not configured")
        service = self._target_service(table, target)
        if api_key is not None:
            secret = api_key.strip()
            if not secret:
                raise ValueError("API key must be non-empty when supplied")
            if len(secret) > 8192:
                raise ValueError("API key is too long")
            if service in _NO_KEY_SERVICES:
                raise ValueError(f"target service '{service}' does not accept an API key")
            self._save_private_key(service, secret)
        config = load_llm_config_file(
            self._config_path,
            target_name=target,
            private_path=(self._private_path if self._private_path.is_file() else None),
        )
        return config

    def load_active(self) -> ResearchLLMDeploymentConfig | None:
        with self._lock:
            target = self._active_target
        if target is None:
            return None
        return load_llm_config_file(
            self._config_path,
            target_name=target,
            private_path=(self._private_path if self._private_path.is_file() else None),
        )

    def _read_public_config(
        self,
    ) -> tuple[str, dict[str, dict[str, object]]]:
        try:
            with self._config_path.open("rb") as stream:
                parsed: object = tomllib.load(stream)
        except OSError as exc:
            raise ValueError(
                f"cannot read LLM config '{self._config_path}': {exc}"
            ) from exc
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(
                f"invalid TOML in LLM config '{self._config_path}': {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError("LLM config root must be a TOML table")
        llm = parsed.get("llm")
        if not isinstance(llm, dict):
            raise ValueError("LLM config must contain an [llm] table")
        default_target = llm.get("active_target")
        raw_targets = llm.get("targets")
        if not isinstance(default_target, str) or not default_target.strip():
            raise ValueError("[llm].active_target must be a non-empty string")
        if not isinstance(raw_targets, dict) or not raw_targets:
            raise ValueError("LLM config must contain configured targets")
        targets: dict[str, dict[str, object]] = {}
        for name, value in raw_targets.items():
            if isinstance(name, str) and isinstance(value, dict):
                targets[name] = cast(dict[str, object], value)
        if default_target not in targets:
            raise ValueError("[llm].active_target is not configured")
        return default_target, targets

    def _load_private_keys(self) -> dict[str, str]:
        if not self._private_path.is_file():
            return {}
        loaded = TOMLProviderConfigRepository(self._private_path).load_api_keys()
        return {
            provider: secret.get_secret_value()
            for provider, secret in loaded.items()
        }

    def _load_target_selections(self) -> dict[str, LLMTargetSelection]:
        if not self._private_path.is_file():
            return {}
        return dict(
            TOMLProviderConfigRepository(
                self._private_path
            ).load_target_selections()
        )

    def _save_private_key(self, service: str, api_key: str) -> None:
        with self._write_lock:
            TOMLProviderConfigRepository(self._private_path).save_api_key(
                service,
                api_key,
            )

    def _summarize_target(
        self,
        name: str,
        table: dict[str, object],
        *,
        private_keys: dict[str, str],
        selections: dict[str, LLMTargetSelection],
        active_target: str | None,
    ) -> WebLLMTarget:
        service = self._target_service(table, name)
        requires_key = service not in _NO_KEY_SERVICES
        environment_name = table.get("api_key_env")
        environment_configured = (
            isinstance(environment_name, str)
            and bool(os.environ.get(environment_name))
        )
        private_configured = bool(private_keys.get(service))
        if service in _CLI_SERVICES:
            credential_source = "oauth_cli"
            configured = True
        elif not requires_key:
            credential_source = "not_required"
            configured = True
        elif private_configured:
            credential_source = "llm.local.toml"
            configured = True
        elif environment_configured:
            credential_source = "environment"
            configured = True
        else:
            credential_source = "missing"
            configured = False
        configured_model = table.get("model")
        selection = selections.get(name)
        model = selection.model_id if selection is not None else configured_model
        structured_output = table.get("structured_output")
        return WebLLMTarget(
            name=name,
            service=service,
            model=str(model) if model is not None else "unknown",
            structured_output=(
                str(structured_output)
                if structured_output is not None
                else "unknown"
            ),
            requires_api_key=requires_key,
            credential_configured=configured,
            credential_source=credential_source,
            active=name == active_target,
        )

    @staticmethod
    def _target_service(table: dict[str, object], name: str) -> str:
        service = table.get("service")
        if not isinstance(service, str) or not service.strip():
            raise ValueError(f"LLM target '{name}' has no service")
        return service.strip()
