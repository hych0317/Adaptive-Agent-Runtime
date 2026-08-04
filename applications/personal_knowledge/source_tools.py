"""Runtime Tool registrations for manual text and web source acquisition."""

from __future__ import annotations

from collections.abc import Mapping
import asyncio
from dataclasses import dataclass
from html.parser import HTMLParser
import json
import ipaddress
import os
from pathlib import Path
import sys
import socket
import tempfile
from typing import Protocol, cast
from urllib.parse import parse_qs, urlparse

import httpx
from pydantic import Field

from adaptive_agent_runtime.governance import (
    BoundGovernedOperation,
    DecisionOutcome,
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceAuthorizationIssuer,
    GovernedOperationExecutor,
    InMemoryAuthorizationConsumptionStore,
    InMemoryHumanReviewService,
    RuntimeGovernanceEvaluator,
    StrictAuthorizationVerifier,
    ToolGovernanceAdapter,
    default_governance_policy,
)
from adaptive_agent_runtime.tool_ecosystem import (
    Capability,
    CapabilityRequirement,
    CapabilityResolver,
    DeterministicToolSelector,
    ExactCapabilityMatcher,
    InMemoryCapabilityCatalog,
    InMemoryToolRegistry,
    InMemoryToolTraceSink,
    ManagedToolExecutor,
    RetryPolicy,
    ToolCorrelation,
    ToolExecutionPolicy,
    ToolInvocation,
    ToolObservation,
    ToolProviderMetadata,
    ToolProviderResult,
    ToolSelectionContext,
)
from adaptive_agent_runtime.tool_ecosystem.contracts import ToolExecutor
from adaptive_agent_runtime.tool_ecosystem.errors import ToolIntegrationError
from adaptive_agent_runtime.tool_ecosystem.models import ImmutableJsonObject

from applications.personal_knowledge.models import (
    KnowledgeModel,
    SourceKind,
    TranscriptSegment,
)


SOURCE_FETCH_TEXT = "source.fetch_text"
VIDEO_EXTRACT_TRANSCRIPT = "video.extract_transcript"
MANUAL_TEXT_PROVIDER = "personal_knowledge.source.manual_text"
WEB_TEXT_PROVIDER = "personal_knowledge.source.web_text"
VIDEO_TRANSCRIPT_PROVIDER = "personal_knowledge.video.transcript"
MAX_FETCH_BYTES = 5 * 1024 * 1024


class FetchedSourceText(KnowledgeModel):
    kind: SourceKind
    text: str = Field(min_length=1)
    canonical_url: str | None = Field(default=None, min_length=1)
    title: str | None = Field(default=None, min_length=1)
    creator: str | None = Field(default=None, min_length=1)
    segments: tuple[TranscriptSegment, ...] = ()


@dataclass(frozen=True)
class HTTPTextResponse:
    final_url: str
    content_type: str
    text: str


class HTTPTextFetcher(Protocol):
    async def fetch(self, url: str) -> HTTPTextResponse: ...


class ExtractedVideoTranscript(KnowledgeModel):
    canonical_url: str = Field(min_length=1)
    text: str = Field(min_length=1)
    segments: tuple[TranscriptSegment, ...] = Field(min_length=1)
    title: str | None = Field(default=None, min_length=1)
    creator: str | None = Field(default=None, min_length=1)
    duration_seconds: float = Field(gt=0)


class VideoTranscriptExtractor(Protocol):
    async def extract(self, url: str) -> ExtractedVideoTranscript: ...


def _validated_http_url(url: str) -> str:
    value = url.strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("source URL must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("source URL cannot contain credentials")
    return value


def _validate_public_host(url: str) -> None:
    parsed = urlparse(_validated_http_url(url))
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("source URL has no hostname")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, None)}
    except socket.gaierror as exc:
        raise ValueError("source hostname could not be resolved") from exc
    if not addresses:
        raise ValueError("source hostname resolved to no address")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("source URL cannot target a private or local address")


def normalize_video_url(url: str) -> str:
    value = url.strip()
    if value.startswith("BV") and value.replace("_", "").isalnum():
        return f"https://www.bilibili.com/video/{value}/"
    validated = _validated_http_url(value)
    parsed = urlparse(validated)
    if "bilibili.com" in parsed.netloc.casefold() and "watchlater" in parsed.path:
        bvid = parse_qs(parsed.query).get("bvid", [None])[0]
        if bvid:
            return f"https://www.bilibili.com/video/{bvid}/"
    return validated


class UnavailableVideoTranscriptExtractor:
    async def extract(self, url: str) -> ExtractedVideoTranscript:
        del url
        raise RuntimeError(
            "video transcript helper is not configured; set "
            "PERSONAL_KNOWLEDGE_VIDEO_SCRIPT"
        )


class SubprocessVideoTranscriptExtractor:
    """Run the transcript-first Hermes-compatible helper in an isolated temp dir."""

    def __init__(
        self,
        script_path: str | Path,
        *,
        python_executable: str = sys.executable,
    ) -> None:
        self._script_path = Path(script_path).expanduser().resolve()
        self._python_executable = python_executable

    async def extract(self, url: str) -> ExtractedVideoTranscript:
        target = normalize_video_url(url)
        if not self._script_path.is_file():
            raise RuntimeError(f"video transcript helper not found: {self._script_path}")
        with tempfile.TemporaryDirectory(prefix="personal-knowledge-video-") as raw_dir:
            output_dir = Path(raw_dir).resolve()
            probe = await self._run_helper(
                target,
                output_dir,
                probe_only=True,
                timeout_seconds=60,
            )
            timeout_value = probe.get("recommended_timeout_seconds")
            if not isinstance(timeout_value, (int, float)):
                raise RuntimeError("video probe omitted the extraction timeout")
            timeout = float(timeout_value)
            if not 1 <= timeout <= 600:
                raise RuntimeError("video probe returned an invalid extraction timeout")
            result = await self._run_helper(
                target,
                output_dir,
                probe_only=False,
                timeout_seconds=timeout,
            )
            self._validate_result(result)
            transcript_path = self._safe_output_path(
                output_dir,
                result.get("transcript_path"),
            )
            segments_path = self._safe_output_path(
                output_dir,
                result.get("segments_path") or result.get("segments_file"),
            )
            text = transcript_path.read_text(encoding="utf-8").strip()
            segments = self._read_segments(segments_path)
            if not text or not segments:
                raise RuntimeError("video extraction produced no timestamped transcript")
            return ExtractedVideoTranscript(
                canonical_url=target,
                text=text,
                segments=segments,
                title=self._optional_text(result.get("title")),
                creator=self._optional_text(
                    result.get("creator") or result.get("uploader")
                ),
                duration_seconds=self._required_number(
                    result.get("duration_seconds"),
                    "duration_seconds",
                ),
            )

    async def _run_helper(
        self,
        url: str,
        output_dir: Path,
        *,
        probe_only: bool,
        timeout_seconds: float,
    ) -> dict[str, object]:
        arguments = [
            self._python_executable,
            str(self._script_path),
            "--url",
            url,
            "--output-dir",
            str(output_dir),
        ]
        if probe_only:
            arguments.append("--probe-only")
        arguments.append("--json")
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        environment.setdefault(
            "HUGGINGFACE_HUB_CACHE",
            str(Path(environment["HF_HOME"]) / "hub"),
        )
        process = await asyncio.create_subprocess_exec(
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            process.kill()
            await process.communicate()
            raise RuntimeError("video transcript extraction timed out") from None
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(detail or "video transcript helper failed")
        try:
            payload = json.loads(stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("video transcript helper returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("video transcript helper returned a non-object result")
        return payload

    @staticmethod
    def _validate_result(result: Mapping[str, object]) -> None:
        transcript_char_count = result.get("transcript_char_count")
        required = (
            result.get("success") is True,
            result.get("asr_status") == "transcribed",
            result.get("transcript_source") == "asr",
            isinstance(transcript_char_count, int) and transcript_char_count > 0,
            bool(result.get("transcript_path")),
            bool(result.get("duration_seconds")),
        )
        if not all(required):
            raise RuntimeError("video helper did not return a successful ASR transcript")

    @staticmethod
    def _required_number(value: object, field: str) -> float:
        if not isinstance(value, (int, float)):
            raise RuntimeError(f"video helper omitted {field}")
        return float(value)

    @staticmethod
    def _safe_output_path(output_dir: Path, value: object) -> Path:
        if not isinstance(value, str) or not value:
            raise RuntimeError("video helper omitted an output path")
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = output_dir / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(output_dir) or not resolved.is_file():
            raise RuntimeError("video helper returned an unsafe or missing output path")
        return resolved

    @staticmethod
    def _read_segments(path: Path) -> tuple[TranscriptSegment, ...]:
        segments: list[TranscriptSegment] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            segments.append(
                TranscriptSegment(
                    start_seconds=item["start"],
                    end_seconds=item["end"],
                    text=item["text"],
                )
            )
        return tuple(segments)

    @staticmethod
    def _optional_text(value: object) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None


class HttpxTextFetcher:
    """Bounded public HTTP text fetcher used by the web Tool Provider."""

    def __init__(self, *, timeout_seconds: float = 20.0) -> None:
        self._timeout_seconds = timeout_seconds

    async def fetch(self, url: str) -> HTTPTextResponse:
        target = _validated_http_url(url)

        async def validate_request(request: httpx.Request) -> None:
            await asyncio.to_thread(_validate_public_host, str(request.url))

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=self._timeout_seconds,
            headers={"User-Agent": "AdaptiveAgentRuntime-Knowledge/0.1"},
            event_hooks={"request": [validate_request]},
        ) as client:
            response = await client.get(target)
            response.raise_for_status()
        final_url = _validated_http_url(str(response.url))
        if len(response.content) > MAX_FETCH_BYTES:
            raise ValueError("source response exceeds the 5 MiB text limit")
        content_type = response.headers.get("content-type", "").lower()
        if not (
            content_type.startswith("text/")
            or "application/xhtml+xml" in content_type
        ):
            raise ValueError("source response is not textual content")
        return HTTPTextResponse(
            final_url=final_url,
            content_type=content_type,
            text=response.text,
        )


class _ReadableHTMLParser(HTMLParser):
    _IGNORED = {"script", "style", "noscript", "svg", "template"}
    _BREAKS = {
        "article",
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "p",
        "section",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        normalized = tag.casefold()
        if normalized in self._IGNORED:
            self._ignored_depth += 1
        if normalized == "title":
            self._in_title = True
        if normalized in self._BREAKS and not self._ignored_depth:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized == "title":
            self._in_title = False
        if normalized in self._IGNORED and self._ignored_depth:
            self._ignored_depth -= 1
        if normalized in self._BREAKS and not self._ignored_depth:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        self.parts.append(data)

    def result(self) -> tuple[str, str | None]:
        lines = [" ".join(part.split()) for part in "".join(self.parts).splitlines()]
        text = "\n".join(line for line in lines if line).strip()
        title = " ".join("".join(self.title_parts).split()).strip() or None
        return text, title


def extract_readable_text(payload: HTTPTextResponse) -> tuple[str, str | None]:
    if "html" not in payload.content_type:
        normalized = "\n".join(
            line.strip() for line in payload.text.splitlines() if line.strip()
        )
        return normalized, None
    parser = _ReadableHTMLParser()
    parser.feed(payload.text)
    return parser.result()


class ManualTextProvider:
    module_id = "personal_knowledge.provider.manual_text"
    provider_id = MANUAL_TEXT_PROVIDER

    async def invoke(self, invocation: ToolInvocation) -> ToolProviderResult:
        text = invocation.arguments.get("text")
        title = invocation.arguments.get("title")
        if not isinstance(text, str) or not text.strip():
            return ToolProviderResult.failed(
                error="manual text input requires non-empty text",
                retryable=False,
            )
        if title is not None and not isinstance(title, str):
            return ToolProviderResult.failed(
                error="manual text title must be a string",
                retryable=False,
            )
        result = FetchedSourceText(
            kind=SourceKind.PLAIN_TEXT,
            text=text.strip(),
            title=title.strip() if isinstance(title, str) and title.strip() else None,
        )
        return ToolProviderResult.ok(output=result.model_dump(mode="json"))


class WebTextProvider:
    module_id = "personal_knowledge.provider.web_text"
    provider_id = WEB_TEXT_PROVIDER

    def __init__(self, fetcher: HTTPTextFetcher) -> None:
        self._fetcher = fetcher

    async def invoke(self, invocation: ToolInvocation) -> ToolProviderResult:
        raw_url = invocation.arguments.get("url")
        if not isinstance(raw_url, str):
            return ToolProviderResult.failed(
                error="web source input requires a URL",
                retryable=False,
            )
        try:
            response = await self._fetcher.fetch(raw_url)
            text, title = extract_readable_text(response)
        except (httpx.HTTPError, ValueError) as exc:
            return ToolProviderResult.failed(error=str(exc), retryable=True)
        if not text:
            return ToolProviderResult.failed(
                error="web source contained no readable text",
                retryable=False,
            )
        result = FetchedSourceText(
            kind=SourceKind.WEB_ARTICLE,
            text=text,
            canonical_url=response.final_url,
            title=title,
        )
        return ToolProviderResult.ok(output=result.model_dump(mode="json"))


class VideoTranscriptProvider:
    module_id = "personal_knowledge.provider.video_transcript"
    provider_id = VIDEO_TRANSCRIPT_PROVIDER

    def __init__(self, extractor: VideoTranscriptExtractor) -> None:
        self._extractor = extractor

    async def invoke(self, invocation: ToolInvocation) -> ToolProviderResult:
        raw_url = invocation.arguments.get("url")
        if not isinstance(raw_url, str):
            return ToolProviderResult.failed(
                error="video source input requires a URL or Bilibili BV id",
                retryable=False,
            )
        try:
            extracted = await self._extractor.extract(raw_url)
        except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
            return ToolProviderResult.failed(error=str(exc), retryable=False)
        result = FetchedSourceText(
            kind=SourceKind.VIDEO,
            text=extracted.text,
            canonical_url=extracted.canonical_url,
            title=extracted.title,
            creator=extracted.creator,
            segments=extracted.segments,
        )
        return ToolProviderResult.ok(output=result.model_dump(mode="json"))


class LowRiskGovernedToolExecutor:
    """Authorize each read-only Tool call before managed execution."""

    module_id = "personal_knowledge.tool_executor.governed"

    def __init__(self, delegate: ToolExecutor) -> None:
        reviews = InMemoryHumanReviewService()
        self._delegate = delegate
        self._governance = RuntimeGovernanceEvaluator(
            policy=default_governance_policy(),
            rule_evaluator=DeterministicRuleEvaluator(),
            confidence_evaluator=DeterministicConfidenceEvaluator(),
            review_service=reviews,
        )
        self._issuer = GovernanceAuthorizationIssuer()
        self._operation_executor = GovernedOperationExecutor(
            verifier=StrictAuthorizationVerifier(),
            consumption_store=InMemoryAuthorizationConsumptionStore(),
        )
        self._adapter = ToolGovernanceAdapter()

    async def execute(
        self,
        invocation: ToolInvocation,
        policy: ToolExecutionPolicy,
    ) -> ToolObservation:
        request = self._adapter.to_request(invocation)
        decision = self._governance.evaluate(request)
        if decision.outcome is not DecisionOutcome.ALLOW:
            raise ToolIntegrationError(
                f"source Tool invocation was not authorized: {decision.reason}"
            )
        authorization = self._issuer.issue(request, decision)

        async def apply() -> ToolObservation:
            return await self._delegate.execute(invocation, policy)

        return await self._operation_executor.execute(
            request=request,
            decision=decision,
            authorization=authorization,
            target=BoundGovernedOperation(
                module_id="personal_knowledge.source_tool_operation",
                operation=request.operation,
                target=request.target,
                subject=invocation,
                apply=apply,
            ),
        )


@dataclass(frozen=True)
class KnowledgeSourceToolStack:
    catalog: InMemoryCapabilityCatalog
    registry: InMemoryToolRegistry
    resolver: CapabilityResolver
    selector: DeterministicToolSelector
    trace_sink: InMemoryToolTraceSink
    executor: LowRiskGovernedToolExecutor


def build_source_tool_stack(
    *,
    fetcher: HTTPTextFetcher | None = None,
    video_extractor: VideoTranscriptExtractor | None = None,
) -> KnowledgeSourceToolStack:
    catalog = InMemoryCapabilityCatalog()
    catalog.register(
        Capability(
            capability_id=SOURCE_FETCH_TEXT,
            name="Source Text Acquisition",
            description="Acquire normalized text from manual or web sources.",
            tags=("source", "text"),
        )
    )
    catalog.register(
        Capability(
            capability_id=VIDEO_EXTRACT_TRANSCRIPT,
            name="Video Transcript Extraction",
            description="Extract a timestamped ASR transcript from a video URL.",
            tags=("source", "video", "transcript"),
        )
    )
    registry = InMemoryToolRegistry(catalog)
    manual = ManualTextProvider()
    web = WebTextProvider(fetcher or HttpxTextFetcher())
    configured_video_extractor = video_extractor
    if configured_video_extractor is None:
        script = os.environ.get("PERSONAL_KNOWLEDGE_VIDEO_SCRIPT")
        configured_video_extractor = (
            SubprocessVideoTranscriptExtractor(script)
            if script
            else UnavailableVideoTranscriptExtractor()
        )
    video = VideoTranscriptProvider(configured_video_extractor)
    for provider, capability_id, tags, schema in (
        (
            manual,
            SOURCE_FETCH_TEXT,
            ("manual", "plain_text"),
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "title": {"type": "string"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        ),
        (
            web,
            SOURCE_FETCH_TEXT,
            ("web", "article"),
            {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "minLength": 1},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        ),
        (
            video,
            VIDEO_EXTRACT_TRANSCRIPT,
            ("video", "transcript", "asr"),
            {
                "type": "object",
                "properties": {"url": {"type": "string", "minLength": 1}},
                "required": ["url"],
                "additionalProperties": False,
            },
        ),
    ):
        registry.register(
            ToolProviderMetadata(
                provider_id=provider.provider_id,
                name=provider.provider_id,
                capability_id=capability_id,
                description="Personal knowledge source acquisition provider.",
                input_schema=cast(ImmutableJsonObject, schema),
                tags=tags,
                selection_priority=100,
            ),
            provider,
        )
    resolver = CapabilityResolver(
        catalog=catalog,
        registry=registry,
        matcher=ExactCapabilityMatcher(),
    )
    trace_sink = InMemoryToolTraceSink()
    managed = ManagedToolExecutor(registry=registry, trace_sink=trace_sink)
    return KnowledgeSourceToolStack(
        catalog=catalog,
        registry=registry,
        resolver=resolver,
        selector=DeterministicToolSelector(),
        trace_sink=trace_sink,
        executor=LowRiskGovernedToolExecutor(managed),
    )


class SourceToolRunner:
    def __init__(self, stack: KnowledgeSourceToolStack) -> None:
        self._stack = stack

    async def acquire_manual_text(
        self,
        text: str,
        *,
        title: str | None = None,
        correlation: ToolCorrelation | None = None,
    ) -> tuple[FetchedSourceText, ToolObservation]:
        arguments: dict[str, str] = {"text": text}
        if title is not None:
            arguments["title"] = title
        return await self._acquire(
            required_provider_tag="plain_text",
            arguments=arguments,
            correlation=correlation,
        )

    async def acquire_web_text(
        self,
        url: str,
        *,
        correlation: ToolCorrelation | None = None,
    ) -> tuple[FetchedSourceText, ToolObservation]:
        return await self._acquire(
            capability_id=SOURCE_FETCH_TEXT,
            required_provider_tag="web",
            arguments={"url": url},
            correlation=correlation,
        )

    async def acquire_video_transcript(
        self,
        url: str,
        *,
        correlation: ToolCorrelation | None = None,
    ) -> tuple[FetchedSourceText, ToolObservation]:
        return await self._acquire(
            capability_id=VIDEO_EXTRACT_TRANSCRIPT,
            required_provider_tag="video",
            arguments={"url": url},
            correlation=correlation,
        )

    async def _acquire(
        self,
        *,
        capability_id: str = SOURCE_FETCH_TEXT,
        required_provider_tag: str,
        arguments: Mapping[str, str],
        correlation: ToolCorrelation | None,
    ) -> tuple[FetchedSourceText, ToolObservation]:
        requirement = CapabilityRequirement(
            capability_id=capability_id,
            required_provider_tags=(required_provider_tag,),
        )
        candidates = self._stack.resolver.candidates(requirement)
        selection = self._stack.selector.select(
            requirement,
            candidates,
            ToolSelectionContext(tags=("personal_knowledge",)),
        )
        invocation = ToolInvocation(
            requirement_id=requirement.requirement_id,
            capability_id=requirement.capability_id,
            provider_id=selection.provider_id,
            arguments=dict(arguments),
            correlation=correlation or ToolCorrelation(),
        )
        observation = await self._stack.executor.execute(
            invocation,
            ToolExecutionPolicy(
                timeout_seconds=(
                    620 if capability_id == VIDEO_EXTRACT_TRANSCRIPT else 30
                ),
                retry=RetryPolicy(max_retries=1, delay_seconds=0),
            ),
        )
        if not observation.succeeded or not isinstance(observation.output, Mapping):
            raise ToolIntegrationError(
                observation.error or "source Tool returned no structured text"
            )
        return FetchedSourceText.model_validate(observation.output), observation
