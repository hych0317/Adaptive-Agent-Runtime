"""Thin Harbor boundary for the AAR Terminal Sequential Profile.

This is intentionally the only application module that imports Harbor types.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from applications.terminal_bench.composition import (
    TerminalModelConfig,
    TerminalSequentialApplication,
    build_terminal_application,
)
from applications.terminal_bench.contracts import (
    TerminalExecutionError,
    terminal_exception_indicates_timeout,
    terminal_exception_outcome,
)
from applications.terminal_bench.models import (
    AAR_TERMINAL_SEQUENTIAL_PROFILE,
    TerminalExecResult,
    TerminalExecutionPolicy,
    TerminalExecutionState,
    utc_now,
)


try:  # Harbor is an optional dependency for local, Docker-free tests.
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except ModuleNotFoundError as exc:  # pragma: no cover - exercised through fallback tests
    if exc.name is None or not exc.name.startswith("harbor"):
        raise

    class BaseEnvironment:  # type: ignore[no-redef]
        pass

    class AgentContext:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.n_input_tokens: int | None = None
            self.n_cache_tokens: int | None = None
            self.n_output_tokens: int | None = None
            self.cost_usd: float | None = None
            self.rollout_details: object | None = None
            self.metadata: dict[str, Any] | None = None

    class BaseAgent:  # type: ignore[no-redef]
        SUPPORTS_ATIF = False
        SUPPORTS_RESUME = False
        SUPPORTS_WINDOWS = False

        def __init__(
            self,
            logs_dir: Path,
            model_name: str | None = None,
            *args: object,
            extra_env: dict[str, str] | None = None,
            **kwargs: object,
        ) -> None:
            del args, kwargs
            self.logs_dir = Path(logs_dir)
            self.model_name = model_name
            self.context_id = None
            self.session_id = None
            self._extra_env = dict(extra_env or {})

        def _get_env(self, key: str, *alternatives: str) -> str | None:
            import os

            for name in (key, *alternatives):
                if name in self._extra_env:
                    return self._extra_env[name]
                if name in os.environ:
                    return os.environ[name]
            return None


class HarborTerminalEnvironmentAdapter:
    """Normalize Harbor ExecResult/exception semantics without a user override."""

    def __init__(
        self,
        environment: BaseEnvironment,
        *,
        max_output_characters: int,
    ) -> None:
        self._environment = environment
        self._max_output_characters = max_output_characters

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> TerminalExecResult:
        started_at = utc_now()
        if not command:
            return self._failed_to_start("command is empty", started_at)
        if timeout_sec is not None and timeout_sec < 1:
            return self._failed_to_start("timeout must be positive", started_at)
        try:
            # Deliberately omit `user`: Harbor's orchestrator-owned default Agent
            # user remains authoritative. No service_exec/sidecar API is exposed.
            raw = await self._environment.exec(
                command,
                cwd=cwd,
                env=(dict(env) if env is not None else None),
                timeout_sec=timeout_sec,
            )
        except TerminalExecutionError as exc:
            state, timed_out = terminal_exception_outcome(exc)
            return self._exception_result(
                exc,
                started_at=started_at,
                state=state,
                timed_out=timed_out,
            )
        except TimeoutError as exc:
            state, timed_out = terminal_exception_outcome(exc)
            return self._exception_result(
                exc,
                started_at=started_at,
                state=state,
                timed_out=timed_out,
            )
        except Exception as exc:
            state, timed_out = terminal_exception_outcome(exc)
            return self._exception_result(
                exc,
                started_at=started_at,
                state=state,
                timed_out=timed_out,
            )
        completed_at = utc_now()
        return_code = getattr(raw, "return_code", None)
        if not isinstance(return_code, int):
            return self._exception_result(
                RuntimeError("Harbor ExecResult has no integer return_code"),
                started_at=started_at,
                state=TerminalExecutionState.IN_DOUBT,
                timed_out=False,
            )
        stdout, stdout_truncated = self._truncate(getattr(raw, "stdout", None) or "")
        stderr, stderr_truncated = self._truncate(getattr(raw, "stderr", None) or "")
        return TerminalExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=self._duration_ms(started_at, completed_at),
            execution_state=TerminalExecutionState.COMPLETED,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    def _failed_to_start(
        self,
        reason: str,
        started_at: datetime,
    ) -> TerminalExecResult:
        completed_at = utc_now()
        return TerminalExecResult(
            stderr=reason,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=self._duration_ms(started_at, completed_at),
            execution_state=TerminalExecutionState.FAILED_TO_START,
            transport_failed=False,
        )

    def _exception_result(
        self,
        exc: BaseException,
        *,
        started_at: datetime,
        state: TerminalExecutionState,
        timed_out: bool,
    ) -> TerminalExecResult:
        completed_at = utc_now()
        detail = str(exc) or exc.__class__.__name__
        stderr, truncated = self._truncate(f"{exc.__class__.__name__}: {detail}")
        return TerminalExecResult(
            stderr=stderr,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=self._duration_ms(started_at, completed_at),
            execution_state=state,
            timed_out=timed_out,
            transport_failed=True,
            stderr_truncated=truncated,
        )

    def _truncate(self, value: str) -> tuple[str, bool]:
        limit = self._max_output_characters
        if len(value) <= limit:
            return value, False
        half = max(1, limit // 2)
        return value[:half] + "\n...[output truncated]...\n" + value[-half:], True

    @staticmethod
    def _duration_ms(started_at: datetime, completed_at: datetime) -> int:
        return max(0, int((completed_at - started_at).total_seconds() * 1000))


def _exception_indicates_timeout(exc: BaseException) -> bool:
    """Backward-compatible wrapper for the Harbor-independent classifier."""

    return terminal_exception_indicates_timeout(exc)

ApplicationFactory = Callable[..., TerminalSequentialApplication]


class AdaptiveRuntimeHarborAgent(BaseAgent):  # type: ignore[misc]
    """Harbor BaseAgent implementation for one sequential AAR Runtime."""

    SUPPORTS_ATIF = False
    SUPPORTS_RESUME = False
    SUPPORTS_WINDOWS = False

    def __init__(
        self,
        *args: object,
        application_factory: ApplicationFactory | None = None,
        execution_policy: TerminalExecutionPolicy | None = None,
        agent_timeout_sec: float | str | None = None,
        deadline_reserve_seconds: float | str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._application_factory = application_factory or build_terminal_application
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
        return "adaptive-agent-runtime-terminal-sequential"

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
            raise ValueError("AAR Terminal Sequential Profile requires --model")
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
        metadata["aar"] = summary.model_dump(mode="json")
        metadata["aar"]["verifier_reward_available"] = False
        context.metadata = metadata


def _optional_positive_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    result = _finite_float(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _finite_float(value: object, name: str) -> float:
    import math

    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


__all__ = [
    "AdaptiveRuntimeHarborAgent",
    "HarborTerminalEnvironmentAdapter",
]
