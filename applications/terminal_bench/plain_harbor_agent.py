"""Harbor boundary for the strict Plain Sequential architecture ablation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from adaptive_agent_runtime.llm import ReasoningEffort

from applications.terminal_bench.composition import TerminalModelConfig
from applications.terminal_bench.harbor_agent import (
    AgentContext,
    BaseAgent,
    BaseEnvironment,
    HarborTerminalEnvironmentAdapter,
    _nonnegative_float,
    _optional_positive_float,
)
from applications.terminal_bench.models import TerminalExecutionPolicy
from applications.terminal_bench.plain_sequential import (
    PlainSequentialApplication,
    build_plain_sequential_application,
)


PlainApplicationFactory = Callable[..., PlainSequentialApplication]


class PlainSequentialHarborAgent(BaseAgent):  # type: ignore[misc]
    """Direct sequential Agent used only as an AAR architecture baseline."""

    SUPPORTS_ATIF = False
    SUPPORTS_RESUME = False
    SUPPORTS_WINDOWS = False

    def __init__(
        self,
        *args: object,
        application_factory: PlainApplicationFactory | None = None,
        execution_policy: TerminalExecutionPolicy | None = None,
        agent_timeout_sec: float | str | None = None,
        deadline_reserve_seconds: float | str | None = None,
        reasoning_effort: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        requested_effort = (reasoning_effort or "high").strip().lower()
        if requested_effort != "high":
            raise ValueError(
                "Plain Sequential strict ablation fixes reasoning_effort=high"
            )
        self._reasoning_effort = ReasoningEffort.HIGH
        self._application_factory = (
            application_factory or build_plain_sequential_application
        )
        policy = execution_policy or TerminalExecutionPolicy()
        timeout = _optional_positive_float(agent_timeout_sec, "agent_timeout_sec")
        reserve = (
            policy.deadline_reserve_seconds
            if deadline_reserve_seconds is None
            else _nonnegative_float(
                deadline_reserve_seconds,
                "deadline_reserve_seconds",
            )
        )
        if reserve != policy.deadline_reserve_seconds:
            policy = policy.model_copy(
                update={"deadline_reserve_seconds": reserve}
            )
        if timeout is not None and policy.max_wall_clock_seconds is None:
            if timeout <= reserve:
                raise ValueError(
                    "agent_timeout_sec must exceed deadline_reserve_seconds"
                )
            policy = policy.model_copy(
                update={"max_wall_clock_seconds": timeout - reserve}
            )
        self._execution_policy = policy

    def _get_env(self, key: str, *alternatives: str) -> str | None:
        import os

        for name in (key, *alternatives):
            if name in self._extra_env:
                return self._extra_env[name]
            if name in os.environ:
                return os.environ[name]
        return None

    @staticmethod
    def name() -> str:
        return "plain-sequential-terminal-ablation"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        del environment

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        model_name = self.model_name
        if not model_name:
            raise ValueError("Plain Sequential Agent requires --model")
        trial_id = str(
            self.context_id
            or getattr(environment, "context_id", None)
            or self.session_id
            or getattr(environment, "session_id", None)
            or uuid4()
        )
        provider = model_name.partition("/")[0].lower()
        api_key_name = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
            "qwen-international": "DASHSCOPE_API_KEY",
        }.get(provider)
        api_key = self._get_env(api_key_name) if api_key_name else None
        base_url = self._get_env(
            "AAR_MODEL_BASE_URL",
            f"{provider.upper().replace('-', '_')}_BASE_URL",
        )
        adapter = HarborTerminalEnvironmentAdapter(
            environment,
            max_output_characters=self._execution_policy.max_output_characters,
        )
        application = self._application_factory(
            trial_id=trial_id,
            logs_dir=self.logs_dir,
            environment=adapter,
            model_config=TerminalModelConfig(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                deepseek_reasoning_effort=self._reasoning_effort,
                codex_reasoning_effort=self._reasoning_effort,
            ),
            policy=self._execution_policy,
        )
        try:
            artifacts = await application.run(instruction)
        finally:
            application.close()
        summary = artifacts.summary
        context.n_input_tokens = summary.input_tokens
        context.n_output_tokens = summary.output_tokens
        context.cost_usd = summary.cost_usd
        metadata = dict(context.metadata or {})
        metadata["plain_sequential"] = summary.model_dump(mode="json")
        metadata["plain_sequential"]["verifier_reward_available"] = False
        context.metadata = metadata


__all__ = ["PlainSequentialHarborAgent"]
