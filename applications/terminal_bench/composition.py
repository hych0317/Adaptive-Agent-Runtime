"""Composition root for one isolated AAR Terminal Sequential trial."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import cast
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import JsonValue, SecretStr

from adaptive_agent_runtime.core import AgentRuntime, AgentTask, RunResult
from adaptive_agent_runtime.governance import (
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    RuntimeGovernanceEvaluator,
    default_governance_policy,
)
from adaptive_agent_runtime.llm import (
    AnthropicAPITargetDefinition,
    AnthropicMessagesConfig,
    BackendMetering,
    BackendTransportFeatures,
    InferenceExecutionBudget,
    InferenceGatewayPolicy,
    InferenceRoutingPolicy,
    HttpxJSONTransport,
    ManagedInferenceComposition,
    OpenAICompatibleService,
    OpenAICompatibleTargetDefinition,
    StructuredOutputLevel,
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


@dataclass(frozen=True)
class TerminalModelConfig:
    model_name: str
    api_key: str | None = None
    base_url: str | None = None
    max_output_tokens: int | None = 2048
    inference_timeout_sec: float = 180.0


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
        capability, _ = build_terminal_model_capability(model_config)

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
        max_steps=execution_policy.max_commands,
    )
    return TerminalSequentialApplication(
        trial_id=trial_id,
        runtime=runtime,
        journal=journal,
        persistence=persistence,
    )


def build_terminal_model_capability(
    config: TerminalModelConfig,
) -> tuple[GatewayTerminalTurnProposalCapability, ManagedInferenceComposition]:
    provider, model_id = _split_model_name(config.model_name)
    features = BackendTransportFeatures(
        structured_output=StructuredOutputLevel.JSON_SCHEMA,
        tool_intent=False,
        multimodal=False,
    )
    metering = BackendMetering(
        reports_token_usage=True,
        reports_monetary_cost=False,
    )
    secret = SecretStr(config.api_key) if config.api_key else None
    target_id = f"terminal-bench:{provider}:{model_id}"
    definition: AnthropicAPITargetDefinition | OpenAICompatibleTargetDefinition
    if provider == "anthropic":
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
                "deepseek, qwen, qwen-international, or local"
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
        )
    backend = definition.build_backend(HttpxJSONTransport())
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
        max_output_tokens=config.max_output_tokens,
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
