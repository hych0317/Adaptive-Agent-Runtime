"""Post-verifier analysis; never called from BaseAgent.run()."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from applications.terminal_bench.models import (
    TerminalBenchmarkAnalysis,
    TerminalBenchmarkOutcome,
    TerminalTrialSummary,
    TerminalVerifierTimeoutAttribution,
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
        verifier_text = _load_verifier_text(aar_summary)
        summary = _load_summary(aar_summary)
        verifier = trial.get("verifier_result")
        verifier_timed_out = _verifier_timed_out(trial)
        if not isinstance(verifier, Mapping) and not verifier_timed_out:
            raise ValueError(
                "Harbor verifier_result is required; analyze only after verification"
            )
        raw_rewards = (
            verifier.get("rewards")
            if isinstance(verifier, Mapping)
            else None
        )
        if not isinstance(raw_rewards, Mapping):
            rewards: dict[str, float] = {}
        else:
            rewards = {
                str(key): float(value)
                for key, value in raw_rewards.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        verifier_reward = _primary_reward(rewards)
        timeout_attribution = (
            _verifier_timeout_attribution(verifier_text)
            if verifier_timed_out
            and (verifier_reward is None or verifier_reward <= 0.0)
            else None
        )
        infrastructure_reason = _infrastructure_error_reason(
            trial,
            verifier_reward,
            verifier_text,
            timeout_attribution,
        )
        infrastructure_error = infrastructure_reason is not None
        benchmark_pass = (
            not infrastructure_error
            and verifier_reward is not None
            and verifier_reward > 0.0
        )
        outcome = (
            TerminalBenchmarkOutcome.INFRASTRUCTURE_ERROR
            if infrastructure_error
            else TerminalBenchmarkOutcome.INCONCLUSIVE
            if timeout_attribution
            is TerminalVerifierTimeoutAttribution.INCONCLUSIVE
            else TerminalBenchmarkOutcome.BENCHMARK_PASS
            if benchmark_pass
            else TerminalBenchmarkOutcome.BENCHMARK_FAIL
        )
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
            verifier_timeout_attribution=timeout_attribution,
            benchmark_pass=benchmark_pass,
            agent_complete=summary.agent_complete,
            completion_matches_verifier=(
                timeout_attribution
                is not TerminalVerifierTimeoutAttribution.INCONCLUSIVE
                and summary.agent_complete == benchmark_pass
            ),
            outcome=outcome,
            infrastructure_error=infrastructure_error,
            infrastructure_error_reason=infrastructure_reason,
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


def _load_verifier_text(
    summary: TerminalTrialSummary | Mapping[str, Any] | str | Path,
) -> str:
    if not isinstance(summary, (str, Path)):
        return ""
    summary_path = Path(summary)
    trial_dir = (
        summary_path.parent.parent
        if summary_path.parent.name == "agent"
        else summary_path.parent
    )
    verifier_dir = trial_dir / "verifier"
    parts: list[str] = []
    for name in ("test-stdout.txt", "test-stderr.txt", "exception.txt"):
        path = verifier_dir / name
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8", errors="replace")[:500_000])
    return "\n".join(parts)


def _infrastructure_error_reason(
    trial: Mapping[str, Any],
    verifier_reward: float | None,
    verifier_text: str,
    timeout_attribution: TerminalVerifierTimeoutAttribution | None,
) -> str | None:
    if verifier_reward is not None and verifier_reward > 0.0:
        return None
    if timeout_attribution is TerminalVerifierTimeoutAttribution.INFRASTRUCTURE:
        return (
            "verifier timeout occurred during dependency setup or an "
            "unavailable network path"
        )
    exception = trial.get("exception_info")
    if isinstance(exception, Mapping):
        name = str(exception.get("exception_type") or exception.get("type") or "")
        if any(
            marker in name.lower()
            for marker in ("environment", "docker")
        ):
            return f"Harbor infrastructure exception: {name or 'unknown'}"
    lowered = verifier_text.lower()
    network_markers = (
        "unable to connect",
        "connection refused",
        "could not resolve",
        "temporary failure resolving",
        "proxy error",
        "connection timed out",
    )
    setup_markers = (
        "curl: command not found",
        "uvx: command not found",
        "failed to fetch",
        "unable to fetch some archives",
    )
    if any(marker in lowered for marker in network_markers) and any(
        marker in lowered for marker in setup_markers
    ):
        return "verifier dependency setup failed because its network path was unavailable"
    return None


def _verifier_timed_out(trial: Mapping[str, Any]) -> bool:
    exception = trial.get("exception_info")
    if not isinstance(exception, Mapping):
        return False
    name = str(exception.get("exception_type") or exception.get("type") or "")
    return "verifiertimeout" in name.lower()


def _verifier_timeout_attribution(
    verifier_text: str,
) -> TerminalVerifierTimeoutAttribution:
    lowered = verifier_text.lower()
    network_markers = (
        "unable to connect",
        "connection refused",
        "could not resolve",
        "temporary failure resolving",
        "proxy error",
        "connection timed out",
    )
    dependency_setup_markers = (
        "reading package lists",
        "apt-get install",
        "installing collected packages",
        "failed to fetch",
        "unable to fetch some archives",
        "curl:",
    )
    test_execution_markers = (
        "test session starts",
        "collected ",
        "unittest",
        "passed",
        "failed",
    )
    package_download_in_progress = (
        "get:" in lowered
        and "http" in lowered
        and any(marker in lowered for marker in ("ubuntu", "pypi"))
    )
    dependency_setup_seen = package_download_in_progress or any(
        marker in lowered for marker in dependency_setup_markers
    )
    if (
        any(marker in lowered for marker in network_markers)
        and dependency_setup_seen
    ) or (
        dependency_setup_seen
        and not any(marker in lowered for marker in test_execution_markers)
    ):
        return TerminalVerifierTimeoutAttribution.INFRASTRUCTURE
    explicit_task_timeout_markers = (
        "candidate process timed out",
        "submission process timed out",
        "solution process timed out",
        "timed out waiting for candidate",
        "timed out waiting for submission",
        "timed out waiting for solution",
    )
    timeout_expired_for_task_path = (
        "subprocess.timeoutexpired" in lowered
        and any(marker in lowered for marker in ("/app/", "candidate", "submission"))
    )
    if timeout_expired_for_task_path or any(
        marker in lowered for marker in explicit_task_timeout_markers
    ):
        return TerminalVerifierTimeoutAttribution.TASK_ATTRIBUTABLE
    return TerminalVerifierTimeoutAttribution.INCONCLUSIVE


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
