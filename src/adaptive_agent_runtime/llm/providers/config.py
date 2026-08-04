"""Runtime-owned persistent provider credentials and target selections."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from threading import Lock
import tomllib
from types import MappingProxyType
from typing import Mapping, cast

from pydantic import Field, SecretStr

from adaptive_agent_runtime.core.contracts import RuntimeModule
from adaptive_agent_runtime.llm.json_types import LLMModel
from adaptive_agent_runtime.llm.models import ReasoningEffort


_TOML_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


class LLMTargetSelection(LLMModel):
    """User-selected model settings for a stable Runtime target name."""

    model_id: str = Field(min_length=1)
    reasoning_effort: ReasoningEffort | None = None


class TOMLProviderConfigRepository(RuntimeModule):
    """Load and atomically persist one Runtime-owned private TOML file."""

    module_id = "llm.provider_config.toml"

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._write_lock = Lock()

    @property
    def path(self) -> Path:
        return self._path

    def load_api_keys(self) -> Mapping[str, SecretStr]:
        api_keys, _ = self._load()
        return api_keys

    def load_target_selections(self) -> Mapping[str, LLMTargetSelection]:
        _, selections = self._load()
        return selections

    def save_api_key(self, provider_id: str, api_key: str) -> None:
        provider_id = provider_id.strip()
        api_key = api_key.strip()
        if not provider_id:
            raise ValueError("private provider identifier must be non-empty")
        if not api_key:
            raise ValueError("provider API key must be non-empty")
        with self._write_lock:
            api_keys, selections = self._load_or_empty()
            mutable = dict(api_keys)
            mutable[provider_id] = SecretStr(api_key)
            self._write(mutable, selections)

    def save_target_selection(
        self,
        target_name: str,
        selection: LLMTargetSelection,
    ) -> None:
        target_name = target_name.strip()
        if not target_name:
            raise ValueError("LLM target name must be non-empty")
        with self._write_lock:
            api_keys, selections = self._load_or_empty()
            mutable = dict(selections)
            mutable[target_name] = selection
            self._write(api_keys, mutable)

    def _load(
        self,
    ) -> tuple[
        Mapping[str, SecretStr],
        Mapping[str, LLMTargetSelection],
    ]:
        try:
            with self._path.open("rb") as stream:
                parsed: object = tomllib.load(stream)
        except OSError as exc:
            raise ValueError(
                f"cannot read private LLM config '{self._path}': {exc}"
            ) from exc
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(
                f"invalid TOML in private LLM config '{self._path}': {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError("private LLM config root must be a TOML table")
        root = cast(dict[str, object], parsed)
        unknown_sections = set(root) - {
            "schema_version",
            "providers",
            "selections",
        }
        if unknown_sections:
            names = ", ".join(sorted(unknown_sections))
            raise ValueError(
                f"unknown private LLM config section(s): {names}"
            )
        schema_version = root.get("schema_version")
        if schema_version != 1 or isinstance(schema_version, bool):
            raise ValueError("private LLM config schema_version must be 1")
        providers_value = root.get("providers", {})
        if not isinstance(providers_value, dict):
            raise ValueError("private LLM config [providers] must be a table")

        api_keys: dict[str, SecretStr] = {}
        for provider_id, provider_value in providers_value.items():
            if not provider_id.strip():
                raise ValueError("private provider identifiers must be non-empty")
            if not isinstance(provider_value, dict):
                raise ValueError(f"[providers.{provider_id}] must be a table")
            provider = cast(dict[str, object], provider_value)
            unknown_keys = set(provider) - {"api_key"}
            if unknown_keys:
                names = ", ".join(sorted(unknown_keys))
                raise ValueError(
                    f"unknown [providers.{provider_id}] option(s): {names}"
                )
            value = provider.get("api_key")
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"[providers.{provider_id}].api_key must be a non-empty string"
                )
            api_keys[provider_id] = SecretStr(value.strip())
        selections_value = root.get("selections", {})
        if not isinstance(selections_value, dict):
            raise ValueError("private LLM config [selections] must be a table")
        selections: dict[str, LLMTargetSelection] = {}
        for target_name, selection_value in selections_value.items():
            if not target_name.strip():
                raise ValueError("private LLM target names must be non-empty")
            if not isinstance(selection_value, dict):
                raise ValueError(f"[selections.{target_name}] must be a table")
            selection = cast(dict[str, object], selection_value)
            unknown_keys = set(selection) - {"model", "reasoning_effort"}
            if unknown_keys:
                names = ", ".join(sorted(unknown_keys))
                raise ValueError(
                    f"unknown [selections.{target_name}] option(s): {names}"
                )
            model_id = selection.get("model")
            if not isinstance(model_id, str) or not model_id.strip():
                raise ValueError(
                    f"[selections.{target_name}].model must be a non-empty string"
                )
            effort = selection.get("reasoning_effort")
            if effort is not None and not isinstance(effort, str):
                raise ValueError(
                    f"[selections.{target_name}].reasoning_effort must be a string"
                )
            selections[target_name] = LLMTargetSelection(
                model_id=model_id.strip(),
                reasoning_effort=(
                    ReasoningEffort(effort) if effort is not None else None
                ),
            )
        return MappingProxyType(api_keys), MappingProxyType(selections)

    def _load_or_empty(
        self,
    ) -> tuple[
        Mapping[str, SecretStr],
        Mapping[str, LLMTargetSelection],
    ]:
        if not self._path.is_file():
            return MappingProxyType({}), MappingProxyType({})
        return self._load()

    def _write(
        self,
        api_keys: Mapping[str, SecretStr],
        selections: Mapping[str, LLMTargetSelection],
    ) -> None:
        lines = [
            "# Managed by Adaptive Agent Runtime.",
            "# This file contains secrets and must remain ignored by Git.",
            "",
            "schema_version = 1",
            "",
        ]
        for provider_id in sorted(api_keys):
            secret = api_keys[provider_id].get_secret_value()
            lines.extend(
                (
                    f"[providers.{_toml_key(provider_id)}]",
                    f"api_key = {json.dumps(secret, ensure_ascii=False)}",
                    "",
                )
            )
        for target_name in sorted(selections):
            selection = selections[target_name]
            lines.extend(
                (
                    f"[selections.{_toml_key(target_name)}]",
                    f"model = {json.dumps(selection.model_id, ensure_ascii=False)}",
                )
            )
            if selection.reasoning_effort is not None:
                lines.append(
                    "reasoning_effort = "
                    + json.dumps(selection.reasoning_effort.value)
                )
            lines.append("")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(
            f".{self._path.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.write_text("\n".join(lines), encoding="utf-8")
            temporary.replace(self._path)
        finally:
            if temporary.exists():
                temporary.unlink()


def _toml_key(value: str) -> str:
    return value if _TOML_BARE_KEY.fullmatch(value) else json.dumps(value)
