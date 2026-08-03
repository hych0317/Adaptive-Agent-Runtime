"""Authority-bounded contracts for optional autonomous Agent execution."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID, uuid4

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from adaptive_agent_runtime.llm.json_types import (
    ImmutableJsonObject,
    ImmutableJsonValue,
    LLMModel,
)
from adaptive_agent_runtime.llm.models import (
    BackendDelegatedAccess,
    InferenceCorrelation,
    InferenceUsage,
)


class AgentActionKind(StrEnum):
    COMMAND_EXECUTION = "command_execution"
    FILE_CHANGE = "file_change"
    MCP_TOOL_CALL = "mcp_tool_call"
    WEB_SEARCH = "web_search"


class AgentActionTraceEntry(LLMModel):
    sequence: int = Field(ge=1)
    kind: AgentActionKind
    item_id: str | None = Field(default=None, min_length=1)
    status: str | None = Field(default=None, min_length=1)


class AgentExecutionPolicy(LLMModel):
    delegated_access: BackendDelegatedAccess
    allowed_tool_names: tuple[str, ...] = ()
    max_action_events: int = Field(default=64, ge=0)
    max_capture_characters: int = Field(default=2_000_000, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=0)
    max_agent_turns: int | None = Field(default=None, ge=1)
    max_monetary_cost: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_policy(self) -> AgentExecutionPolicy:
        if len(set(self.allowed_tool_names)) != len(self.allowed_tool_names):
            raise ValueError("Agent tool allowlist must be unique")
        if self.allowed_tool_names and not self.delegated_access.mcp_execution:
            raise ValueError("Agent tool allowlist requires MCP execution access")
        return self


class AutonomousAgentRequest(LLMModel):
    request_id: UUID = Field(default_factory=uuid4)
    goal: str = Field(min_length=1)
    input: ImmutableJsonValue = None
    workspace_root: str = Field(min_length=1)
    response_schema: ImmutableJsonObject | None = None
    policy: AgentExecutionPolicy
    correlation: InferenceCorrelation = Field(
        default_factory=InferenceCorrelation
    )
    timeout_seconds: float = Field(default=300.0, gt=0.0)
    trace_attributes: ImmutableJsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_request(self) -> AutonomousAgentRequest:
        if self.response_schema is not None:
            schema = self.model_dump(mode="json")["response_schema"]
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as exc:
                raise ValueError(
                    f"invalid Agent response schema: {exc.message}"
                ) from exc
        return self


class AutonomousAgentResult(LLMModel):
    request_id: UUID
    target_id: str = Field(min_length=1)
    model_id: str | None = Field(default=None, min_length=1)
    output: ImmutableJsonValue
    usage: InferenceUsage = Field(default_factory=InferenceUsage)
    actions: tuple[AgentActionTraceEntry, ...] = ()
    remote_request_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_result(self) -> AutonomousAgentResult:
        sequences = tuple(item.sequence for item in self.actions)
        if sequences != tuple(range(1, len(self.actions) + 1)):
            raise ValueError("Agent action trace must have contiguous sequence")
        return self
