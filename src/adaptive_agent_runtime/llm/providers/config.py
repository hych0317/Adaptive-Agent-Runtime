"""Application-owned persistent configuration for model-provider API keys."""

from __future__ import annotations

from pathlib import Path
import tomllib
from types import MappingProxyType
from typing import Mapping, cast

from pydantic import SecretStr

from adaptive_agent_runtime.core.contracts import RuntimeModule


class TOMLProviderConfigRepository(RuntimeModule):
    """Load provider API keys from one Runtime-owned private TOML file."""

    module_id = "llm.provider_config.toml"

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load_api_keys(self) -> Mapping[str, SecretStr]:
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
        unknown_sections = set(root) - {"schema_version", "providers"}
        if unknown_sections:
            names = ", ".join(sorted(unknown_sections))
            raise ValueError(
                f"unknown private LLM config section(s): {names}"
            )
        schema_version = root.get("schema_version")
        if schema_version != 1 or isinstance(schema_version, bool):
            raise ValueError("private LLM config schema_version must be 1")
        providers_value = root.get("providers")
        if not isinstance(providers_value, dict):
            raise ValueError("private LLM config must contain a [providers] table")

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
        return MappingProxyType(api_keys)
