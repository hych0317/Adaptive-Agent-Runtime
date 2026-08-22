"""Strict YAML loading with stable scenario identity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError
import yaml  # type: ignore[import-untyped]

from adaptive_agent_runtime.decisioning import decision_fingerprint
from applications.governance_scenario_suite.contracts import (
    ScenarioCatalogSpec,
    ScenarioSpec,
)


class ScenarioLoadError(ValueError):
    pass


def load_scenario(path: str | Path) -> ScenarioSpec:
    resolved = Path(path).resolve()
    payload = _load_single_document(resolved)
    try:
        return ScenarioSpec.model_validate(payload)
    except ValidationError as exc:
        raise ScenarioLoadError(
            f"Scenario '{resolved}' failed strict validation: {exc}"
        ) from exc


def load_catalog(path: str | Path) -> ScenarioCatalogSpec:
    resolved = Path(path).resolve()
    payload = _load_single_document(resolved)
    try:
        return ScenarioCatalogSpec.model_validate(payload)
    except ValidationError as exc:
        raise ScenarioLoadError(
            f"Scenario catalog '{resolved}' failed strict validation: {exc}"
        ) from exc


def load_scenario_directory(directory: str | Path) -> tuple[ScenarioSpec, ...]:
    root = Path(directory).resolve()
    scenarios = tuple(
        load_scenario(path)
        for path in sorted(root.glob("*.yaml"))
        if path.name != "catalog.yaml"
    )
    if not scenarios:
        raise ScenarioLoadError(f"Scenario directory '{root}' contains no scenarios")
    identities: dict[str, tuple[str, Path | None]] = {}
    for scenario in scenarios:
        fingerprint = decision_fingerprint(scenario)
        existing = identities.get(scenario.id)
        if existing is not None:
            previous_fingerprint, _ = existing
            if previous_fingerprint != fingerprint:
                raise ScenarioLoadError(
                    f"Scenario id '{scenario.id}' was reused with different content"
                )
            raise ScenarioLoadError(f"Scenario id '{scenario.id}' is duplicated")
        identities[scenario.id] = (fingerprint, None)
    return scenarios


def _load_single_document(path: Path) -> Any:
    if not path.is_file():
        raise ScenarioLoadError(f"Scenario file '{path}' does not exist")
    try:
        with path.open("r", encoding="utf-8") as stream:
            documents = tuple(yaml.safe_load_all(stream))
    except yaml.YAMLError as exc:
        raise ScenarioLoadError(f"Scenario file '{path}' is invalid YAML: {exc}") from exc
    if len(documents) != 1:
        raise ScenarioLoadError(
            f"Scenario file '{path}' must contain exactly one YAML document"
        )
    if not isinstance(documents[0], dict):
        raise ScenarioLoadError(f"Scenario file '{path}' must contain a mapping")
    return documents[0]
