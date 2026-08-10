"""Composition root for one isolated AAR Terminal Sequential trial."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Literal, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import JsonValue, SecretStr

from adaptive_agent_runtime.core import (
    AgentRuntime,
    AgentTask,
    RunResult,
    RunStopPolicy,
)
from adaptive_agent_runtime.governance import (
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    RuntimeGovernanceEvaluator,
    default_governance_policy,
)
from adaptive_agent_runtime.llm import (
    AnthropicAPITargetDefinition,
    AsyncJSONTransport,
    AsyncProcessTransport,
    AnthropicMessagesConfig,
    BackendLimits,
    BackendMetering,
    BackendTransportFeatures,
    CodexCLIAuthProbeMode,
    CodexCLIInferenceConfig,
    CodexCLIInferenceTargetDefinition,
    HTTPJSONResponse,
    InferenceExecutionBudget,
    InferenceGatewayPolicy,
    InferenceRoutingPolicy,
    HttpxJSONTransport,
    ManagedInferenceComposition,
    OpenAICompatibleService,
    OpenAICompatibleTargetDefinition,
    ReasoningEffort,
    StructuredOutputLevel,
    SubprocessTransport,
    compose_managed_inference,
)
from adaptive_agent_runtime.persistence import SQLitePersistence
from adaptive_agent_runtime.tool_ecosystem import (
    Capability,
    InMemoryCapabilityCatalog,
    InMemoryToolRegistry,
    InMemoryToolTraceSink,
    ToolInvocationDecisionPayload,
    ToolInvocationEffect,
    ToolInvocationProposalDraft,
    build_permit_bound_tool_executor,
)
from adaptive_agent_runtime.decisioning import DecisionCheckpoint

from applications.terminal_bench.contracts import (
    TerminalEnvironment,
    TerminalTurnProposalCapability,
)
from applications.terminal_bench.executor import TerminalActionExecutor
from applications.terminal_bench.models import (
    AAR_TERMINAL_SEQUENTIAL_PROFILE,
    TERMINAL_COMMAND_CAPABILITY,
    TerminalExecutionPolicy,
    TerminalTrialSummary,
    utc_now,
)
from applications.terminal_bench.planner import (
    GatewayTerminalTurnProposalCapability,
    JsonlTerminalTrialJournal,
    TerminalSequentialPlanner,
)
from applications.terminal_bench.tool_decision import (
    TerminalToolInvocationDecisionHandler,
)
from applications.terminal_bench.tools import (
    TerminalCommandProvider,
    terminal_provider_metadata,
)


_TERMINAL_TRIAL_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "adaptive-agent-runtime/terminal-sequential/trial",
)


class _RequestBodyOverrideJSONTransport:
    """Apply explicit provider request extensions before transport."""

    module_id = "terminal_bench.transport.request_body_override"

    def __init__(
        self,
        delegate: AsyncJSONTransport,
        overrides: Mapping[str, object],
    ) -> None:
        self._delegate = delegate
        self._overrides = dict(overrides)

    async def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        return await self._delegate.get_json(
            url,
            headers=headers,
            timeout_seconds=timeout_seconds,
        )

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, object],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        extended_body = dict(body)
        extended_body.update(self._overrides)
        return await self._delegate.post_json(
            url,
            headers=headers,
            body=extended_body,
            timeout_seconds=timeout_seconds,
        )


class _TrailingJSONDelimiterRepairTransport:
    """Remove only redundant JSON closing delimiters after one valid object."""

    module_id = "terminal_bench.transport.trailing_json_delimiter_repair"
    _allowed_trailing_characters = frozenset(' \t\r\n"}]')

    def __init__(self, delegate: AsyncJSONTransport) -> None:
        self._delegate = delegate

    async def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        return await self._delegate.get_json(
            url,
            headers=headers,
            timeout_seconds=timeout_seconds,
        )

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, object],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        response = await self._delegate.post_json(
            url,
            headers=headers,
            body=body,
            timeout_seconds=timeout_seconds,
        )
        return self._repair(response)

    def _repair(self, response: HTTPJSONResponse) -> HTTPJSONResponse:
        response_body = response.body
        if not isinstance(response_body, Mapping):
            return response
        choices = response_body.get("choices")
        if not isinstance(choices, (list, tuple)) or not choices:
            return response
        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            return response
        message = first_choice.get("message")
        if not isinstance(message, Mapping):
            return response
        content = message.get("content")
        if not isinstance(content, str):
            return response
        try:
            json.loads(content)
            return response
        except json.JSONDecodeError:
            pass
        try:
            parsed, end = json.JSONDecoder().raw_decode(content)
        except json.JSONDecodeError:
            return response
        trailing = content[end:]
        if (
            not isinstance(parsed, dict)
            or not trailing
            or any(
                character not in self._allowed_trailing_characters
                for character in trailing
            )
        ):
            return response
        normalized_body = response.model_dump(mode="json")["body"]
        assert isinstance(normalized_body, dict)
        normalized_choices = list(normalized_body["choices"])
        normalized_choice = dict(normalized_choices[0])
        normalized_message = dict(normalized_choice["message"])
        normalized_message["content"] = json.dumps(
            parsed,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        normalized_choice["message"] = normalized_message
        normalized_choices[0] = normalized_choice
        normalized_body["choices"] = normalized_choices
        return HTTPJSONResponse(
            status_code=response.status_code,
            headers=response.headers,
            body=normalized_body,
        )


class _CapturingJSONTransport:
    """Persist provider responses without recording request-side secrets."""

    module_id = "terminal_bench.transport.response_capture"

    def __init__(
        self,
        delegate: AsyncJSONTransport,
        capture_path: str | Path,
    ) -> None:
        self._delegate = delegate
        self._capture_path = Path(capture_path)

    async def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        response = await self._delegate.get_json(
            url,
            headers=headers,
            timeout_seconds=timeout_seconds,
        )
        self._append_response(response)
        return response

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, object],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        response = await self._delegate.post_json(
            url,
            headers=headers,
            body=body,
            timeout_seconds=timeout_seconds,
        )
        self._append_response(response)
        return response

    def _append_response(self, response: HTTPJSONResponse) -> None:
        record = {
            "status_code": response.status_code,
            "body": response.model_dump(mode="json")["body"],
        }
        encoded = (
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        fd = os.open(
            self._capture_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, encoded)
        finally:
            os.close(fd)


@dataclass(frozen=True)
class TerminalModelConfig:
    model_name: str
    api_key: str | None = None
    base_url: str | None = None
    max_output_tokens: int | None = 32768
    inference_timeout_sec: float = 300.0
    delivery_inference_timeout_sec: float = 180.0
    minimum_inference_timeout_sec: float = 120.0
    minimum_delivery_inference_timeout_sec: float = 60.0
    deepseek_reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH
    deepseek_thinking: Literal["enabled", "disabled"] = "enabled"
    codex_executable: str = "codex"
    codex_reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH


@dataclass(frozen=True)
class TerminalRunArtifacts:
    runtime_result: RunResult
    summary: TerminalTrialSummary


class TerminalSequentialApplication:
    """Own exactly one AgentRuntime and its trial-scoped dependencies."""

    def __init__(
        self,
        *,
        trial_id: str,
        runtime: AgentRuntime,
        journal: JsonlTerminalTrialJournal,
        persistence: SQLitePersistence,
    ) -> None:
        self.trial_id = trial_id
        self.runtime = runtime
        self.journal = journal
        self.persistence = persistence
        self._closed = False

    async def run(self, instruction: str) -> TerminalRunArtifacts:
        started_at = utc_now()
        started_clock = monotonic()
        task_id = uuid5(_TERMINAL_TRIAL_NAMESPACE, f"task|{self.trial_id}")
        run_id = uuid5(_TERMINAL_TRIAL_NAMESPACE, f"run|{self.trial_id}")
        result = await self.runtime.run(
            AgentTask(
                task_id=task_id,
                description=instruction,
                input={
                    "profile": AAR_TERMINAL_SEQUENTIAL_PROFILE,
                    "trial_id": self.trial_id,
                },
            ),
            run_id=run_id,
        )
        completed_at = utc_now()
        session = self.journal.snapshot()
        output = result.final_state.output
        agent_complete = bool(
            result.succeeded
            and isinstance(output, Mapping)
            and output.get("agent_complete") is True
        )
        summary = TerminalTrialSummary(
            trial_id=self.trial_id,
            run_id=run_id,
            task_id=task_id,
            agent_complete=agent_complete,
            runtime_status=result.final_state.status.value,
            final_output=_plain_json(output),
            command_count=session.committed_commands,
            denial_count=session.denied_commands,
            timeout_count=session.timed_out_commands,
            in_doubt_count=session.in_doubt_commands,
            input_tokens=session.input_tokens,
            output_tokens=session.output_tokens,
            total_tokens=session.total_tokens,
            cost_usd=session.cost_usd,
            latency_ms=max(0, int((monotonic() - started_clock) * 1000)),
            trace_consistent=self.journal.trace_consistent,
            started_at=started_at,
            completed_at=completed_at,
        )
        self.journal.write_summary(summary)
        return TerminalRunArtifacts(runtime_result=result, summary=summary)

    def close(self) -> None:
        if not self._closed:
            self.persistence.close()
            self._closed = True


def build_terminal_application(
    *,
    trial_id: str,
    logs_dir: str | Path,
    environment: TerminalEnvironment,
    proposal_capability: TerminalTurnProposalCapability | None = None,
    model_config: TerminalModelConfig | None = None,
    policy: TerminalExecutionPolicy | None = None,
) -> TerminalSequentialApplication:
    """Compose one profile instance without Docker or Harbor dependencies."""

    execution_policy = policy or TerminalExecutionPolicy()
    root = Path(logs_dir)
    root.mkdir(parents=True, exist_ok=True)
    persistence = SQLitePersistence(root / "aar-runtime.sqlite3")
    journal = JsonlTerminalTrialJournal(
        trial_id,
        root / "aar-transcript.jsonl",
    )
    capability = proposal_capability
    if capability is None:
        if model_config is None:
            persistence.close()
            raise ValueError("model_config or proposal_capability is required")
        capability, _ = build_terminal_model_capability(
            model_config,
            transport=_CapturingJSONTransport(
                HttpxJSONTransport(),
                root / "aar-model-responses.jsonl",
            ),
        )

    catalog = InMemoryCapabilityCatalog()
    catalog.register(
        Capability(
            capability_id=TERMINAL_COMMAND_CAPABILITY,
            name="Trial terminal command",
            description="Execute one bounded command in the current trial.",
            tags=("terminal", "trial_scoped"),
        )
    )
    registry = InMemoryToolRegistry(catalog)
    metadata = terminal_provider_metadata(execution_policy)
    registry.register(
        metadata,
        TerminalCommandProvider(
            environment=environment,
            journal=journal,
            policy=execution_policy,
        ),
    )
    tool_trace = InMemoryToolTraceSink()
    permit_bound = build_permit_bound_tool_executor(
        registry=registry,
        trace_sink=tool_trace,
        permit_verifier=persistence.commit_permit_verifier,
    )
    governance = RuntimeGovernanceEvaluator(
        policy=default_governance_policy(),
        rule_evaluator=DeterministicRuleEvaluator(),
        confidence_evaluator=DeterministicConfidenceEvaluator(),
        review_service=persistence.human_review_service,
    )
    decisions = TerminalToolInvocationDecisionHandler(
        provider_metadata=metadata,
        executor=permit_bound,
        operation_executor=persistence.operation_executor,
        authorization_issuer=persistence.authorization_issuer,
        trace_sink=persistence.trace_sink,
        journal=journal,
        policy=execution_policy,
        governance=governance,
        reviews=persistence.human_review_service,
        checkpoint_store=persistence.create_decision_checkpoint_store(
            DecisionCheckpoint[
                ToolInvocationDecisionPayload,
                ToolInvocationProposalDraft,
                ToolInvocationEffect,
            ]
        ),
    )
    planner = TerminalSequentialPlanner(
        capability=capability,
        journal=journal,
        policy=execution_policy,
    )
    executor = TerminalActionExecutor(decisions=decisions, journal=journal)
    runtime = AgentRuntime(
        planner=planner,
        executor=executor,
        state_store=persistence.state_store,
        trace_sink=persistence.trace_sink,
        stop_policy=RunStopPolicy(
            max_action_steps=(
                execution_policy.max_commands
                + execution_policy.max_completion_rejections
                + execution_policy.max_proposal_rejections
            ),
            max_wall_clock_seconds=execution_policy.max_wall_clock_seconds,
            max_active_execution_seconds=(
                execution_policy.max_active_execution_seconds
            ),
            default_tool_timeout_seconds=float(
                execution_policy.default_timeout_sec
            ),
            max_tool_timeout_seconds=float(execution_policy.max_timeout_sec),
            external_job_deadline_seconds=(
                execution_policy.external_job_deadline_seconds
            ),
            cleanup_grace_seconds=execution_policy.cleanup_grace_seconds,
            # The Terminal Planner and inference Gateway own model budgets. A
            # duplicate Core limit could stop after a successful verification
            # Action, before the Planner can finalize its zero-token checkpoint.
            max_total_tokens=None,
            max_monetary_cost=None,
            currency=None,
            repeated_invocation_limit=(
                execution_policy.repeated_invocation_limit
            ),
            max_no_progress_steps=execution_policy.max_no_progress_steps,
            max_no_progress_seconds=execution_policy.max_no_progress_seconds,
        ),
    )
    return TerminalSequentialApplication(
        trial_id=trial_id,
        runtime=runtime,
        journal=journal,
        persistence=persistence,
    )


def build_terminal_model_capability(
    config: TerminalModelConfig,
    *,
    transport: AsyncJSONTransport | None = None,
    process_transport: AsyncProcessTransport | None = None,
) -> tuple[GatewayTerminalTurnProposalCapability, ManagedInferenceComposition]:
    provider, model_id = _split_model_name(config.model_name)
    structured_output = (
        StructuredOutputLevel.JSON_OBJECT
        if provider == "deepseek"
        else StructuredOutputLevel.JSON_SCHEMA
    )
    features = BackendTransportFeatures(
        structured_output=structured_output,
        tool_intent=False,
        multimodal=False,
    )
    metering = BackendMetering(
        reports_token_usage=True,
        reports_monetary_cost=False,
    )
    secret = SecretStr(config.api_key) if config.api_key else None
    target_id = f"terminal-bench:{provider}:{model_id}"
    if provider == "codex-cli":
        codex_definition = CodexCLIInferenceTargetDefinition(
            target_id=target_id,
            model_id=model_id,
            features=features,
            supported_cognitive_capability_ids=("terminal_turn_proposal",),
            config=CodexCLIInferenceConfig(
                executable=config.codex_executable,
                auth_probe_mode=CodexCLIAuthProbeMode.REQUIRED,
                reasoning_effort=config.codex_reasoning_effort,
            ),
            metering=metering,
            limits=BackendLimits(
                default_timeout_seconds=config.inference_timeout_sec,
            ),
        )
        backend = codex_definition.build_backend(
            process_transport or SubprocessTransport()
        )
    elif provider == "anthropic":
        definition: AnthropicAPITargetDefinition | OpenAICompatibleTargetDefinition
        definition = AnthropicAPITargetDefinition(
            target_id=target_id,
            model_id=model_id,
            features=features,
            supported_cognitive_capability_ids=("terminal_turn_proposal",),
            config=AnthropicMessagesConfig(
                base_url=config.base_url or "https://api.anthropic.com/v1",
                api_key=secret,
            ),
            metering=metering,
        )
        backend = definition.build_backend(transport or HttpxJSONTransport())
    else:
        service = {
            "openai": OpenAICompatibleService.OPENAI,
            "deepseek": OpenAICompatibleService.DEEPSEEK,
            "qwen": OpenAICompatibleService.QWEN_CHINA,
            "qwen-international": OpenAICompatibleService.QWEN_INTERNATIONAL,
            "local": OpenAICompatibleService.LOCAL,
        }.get(provider)
        if service is None:
            raise ValueError(
                "unsupported Harbor model provider; expected openai, anthropic, "
                "deepseek, qwen, qwen-international, local, or codex-cli"
            )
        definition = OpenAICompatibleTargetDefinition(
            service=service,
            target_id=target_id,
            model_id=model_id,
            features=features,
            supported_cognitive_capability_ids=("terminal_turn_proposal",),
            base_url=config.base_url,
            api_key=secret,
            metering=metering,
            reasoning_effort=(
                config.deepseek_reasoning_effort
                if provider == "deepseek"
                else None
            ),
        )
        selected_transport = transport or HttpxJSONTransport()
        if provider == "deepseek":
            selected_transport = _RequestBodyOverrideJSONTransport(
                selected_transport,
                {
                    "thinking": {
                        "type": config.deepseek_thinking,
                    }
                },
            )
            selected_transport = _TrailingJSONDelimiterRepairTransport(
                selected_transport
            )
        backend = definition.build_backend(selected_transport)
    inference = compose_managed_inference((backend,))
    gateway_policy = InferenceGatewayPolicy(
        routing=InferenceRoutingPolicy(
            allowed_target_ids=(target_id,),
            preferred_target_ids=(target_id,),
        ),
        budget=InferenceExecutionBudget(
            max_attempts=1,
            max_elapsed_seconds=config.inference_timeout_sec,
            max_total_tokens=None,
        ),
    )
    capability = GatewayTerminalTurnProposalCapability(
        gateway=inference.gateway,
        gateway_policy=gateway_policy,
        target_id=target_id,
        required_structured_output=structured_output,
        max_output_tokens=(
            None
            if provider == "codex-cli"
            else config.max_output_tokens
        ),
        strict_json_schema=(provider == "codex-cli"),
        delivery_timeout_seconds=config.delivery_inference_timeout_sec,
        minimum_timeout_seconds=config.minimum_inference_timeout_sec,
        minimum_delivery_timeout_seconds=(
            config.minimum_delivery_inference_timeout_sec
        ),
    )
    return capability, inference


def _split_model_name(model_name: str) -> tuple[str, str]:
    provider, separator, model_id = model_name.partition("/")
    if not separator or not provider or not model_id:
        raise ValueError("Harbor model must use '<provider>/<model>'")
    return provider.lower(), model_id


def _plain_json(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return cast(JsonValue, value)
