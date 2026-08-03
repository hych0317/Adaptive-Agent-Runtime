"""Deeply immutable JSON types local to Context-Memory Runtime."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from math import ceil, isfinite
from types import MappingProxyType
from typing import Annotated, Any, Mapping, Self, TypeAlias

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    JsonValue,
    PlainSerializer,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("JSON numbers must be finite")
    return value


ImmutableJsonObject: TypeAlias = Annotated[
    Mapping[str, JsonValue],
    BeforeValidator(_thaw_json),
    AfterValidator(_freeze_json),
    PlainSerializer(_thaw_json),
]
ImmutableJsonValue: TypeAlias = Annotated[
    JsonValue,
    BeforeValidator(_thaw_json),
    AfterValidator(_freeze_json),
    PlainSerializer(_thaw_json),
]


class ContextMemoryModel(BaseModel):
    """Base model for deeply immutable Context-Memory snapshots."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Return a validated copy so updates cannot bypass deep immutability."""

        del deep
        values = self.model_dump(mode="python")
        if update:
            values.update(update)
        return self.__class__.model_validate(values)


def estimate_tokens(value: Any) -> int:
    """Return a deterministic budget estimate without invoking a tokenizer."""

    serialized = json.dumps(
        _thaw_json(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return max(1, ceil(len(serialized) / 4))
