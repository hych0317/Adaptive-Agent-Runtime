"""Post-verifier analysis; never called from BaseAgent.run()."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from applications.terminal_bench.models import (
    TerminalBenchmarkAnalysis,
    TerminalTrialSummary,
)


class TerminalResultAnalyzer:
    module_id = "terminal_bench.result_analyzer"

    def analyze(
        self,
        *,
        harbor_trial_result: Mapping[str, Any] | object,
        aar_summary: TerminalTrialSummary | Mapping[str, Any] | str | Path,
    ) -> TerminalBenchmarkAnalysis:
        trial = _as_mapping(harbor_trial_result)
        summary = _load_summary(aar_summary)
        verifier = trial.get("verifier_result")
        if not isinstance(verifier, Mapping):
            raise ValueError(
                "Harbor verifier_result is required; analyze only after verification"
            )
        raw_rewards = verifier.get("rewards")
        if not isinstance(raw_rewards, Mapping):
            rewards: dict[str, float] = {}
        else:
            rewards = {
                str(key): float(value)
                for key, value in raw_rewards.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        verifier_reward = _primary_reward(rewards)
        benchmark_pass = verifier_reward is not None and verifier_reward > 0.0
        agent_context = trial.get("agent_result")
        context = agent_context if isinstance(agent_context, Mapping) else {}
        input_tokens = _nonnegative_int(
            context.get("n_input_tokens"), summary.input_tokens
        )
        output_tokens = _nonnegative_int(
            context.get("n_output_tokens"), summary.output_tokens
        )
        total_tokens = input_tokens + output_tokens
        cost_usd = _nonnegative_float(context.get("cost_usd"), summary.cost_usd)
        latency_ms = _agent_latency_ms(trial) or summary.latency_ms
        return TerminalBenchmarkAnalysis(
            trial_id=summary.trial_id,
            verifier_rewards=rewards,
            verifier_reward=verifier_reward,
            benchmark_pass=benchmark_pass,
            agent_complete=summary.agent_complete,
            completion_matches_verifier=(summary.agent_complete == benchmark_pass),
            command_count=summary.command_count,
            denial_count=summary.denial_count,
            timeout_count=summary.timeout_count,
            in_doubt_count=summary.in_doubt_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            trace_consistent=summary.trace_consistent,
        )


def _as_mapping(value: Mapping[str, Any] | object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python")
        if isinstance(dumped, Mapping):
            return dumped
    raise TypeError("Harbor TrialResult must be a mapping or Pydantic model")


def _load_summary(
    value: TerminalTrialSummary | Mapping[str, Any] | str | Path,
) -> TerminalTrialSummary:
    if isinstance(value, TerminalTrialSummary):
        return value
    if isinstance(value, Mapping):
        return TerminalTrialSummary.model_validate(value)
    return TerminalTrialSummary.model_validate_json(
        Path(value).read_text(encoding="utf-8")
    )


def _primary_reward(rewards: Mapping[str, float]) -> float | None:
    if "reward" in rewards:
        return rewards["reward"]
    if not rewards:
        return None
    return sum(rewards.values()) / len(rewards)


def _nonnegative_int(value: object, fallback: int) -> int:
    return int(value) if isinstance(value, int) and value >= 0 else fallback


def _nonnegative_float(value: object, fallback: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return fallback


def _agent_latency_ms(trial: Mapping[str, Any]) -> int | None:
    timing = trial.get("agent_execution")
    if not isinstance(timing, Mapping):
        return None
    started = timing.get("started_at")
    finished = timing.get("finished_at")
    try:
        start_time = _datetime(started)
        finish_time = _datetime(finished)
    except (TypeError, ValueError):
        return None
    if start_time is None or finish_time is None or finish_time < start_time:
        return None
    return int((finish_time - start_time).total_seconds() * 1000)


def _datetime(value: object) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise TypeError("unsupported datetime")
