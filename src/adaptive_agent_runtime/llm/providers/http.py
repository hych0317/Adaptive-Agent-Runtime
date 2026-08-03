"""Minimal async JSON transport abstraction and httpx implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import Field

from adaptive_agent_runtime.core.contracts import RuntimeModule
from adaptive_agent_runtime.llm.json_types import (
    ImmutableJsonObject,
    ImmutableJsonValue,
    LLMModel,
)


class HTTPTransportError(Exception):
    """Base transport failure before an HTTP response is available."""


class HTTPTransportTimeoutError(HTTPTransportError):
    pass


class HTTPTransportUnavailableError(HTTPTransportError):
    pass


class HTTPJSONResponse(LLMModel):
    status_code: int = Field(ge=100, le=599)
    headers: ImmutableJsonObject = Field(default_factory=dict)
    body: ImmutableJsonValue


@runtime_checkable
class AsyncJSONTransport(RuntimeModule, Protocol):
    async def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HTTPJSONResponse: ...

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HTTPJSONResponse: ...


class HttpxJSONTransport:
    module_id = "llm.http_transport.httpx"

    async def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=dict(headers),
                    timeout=timeout_seconds,
                )
        except httpx.TimeoutException as exc:
            raise HTTPTransportTimeoutError from exc
        except httpx.RequestError as exc:
            raise HTTPTransportUnavailableError from exc
        return _normalize_httpx_response(response)

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=dict(headers),
                    json=dict(body),
                    timeout=timeout_seconds,
                )
        except httpx.TimeoutException as exc:
            raise HTTPTransportTimeoutError from exc
        except httpx.RequestError as exc:
            raise HTTPTransportUnavailableError from exc
        return _normalize_httpx_response(response)


def _normalize_httpx_response(response: httpx.Response) -> HTTPJSONResponse:
    try:
        payload: Any = response.json()
    except ValueError:
        payload = response.text
    return HTTPJSONResponse(
        status_code=response.status_code,
        headers=dict(response.headers),
        body=payload,
    )
