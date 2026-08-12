from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from applications.terminal_bench.composition import TerminalRunArtifacts
from applications.terminal_bench.harbor_agent import (
    AdaptiveRuntimeHarborAgent,
    AgentContext,
    HarborTerminalEnvironmentAdapter,
)
from applications.terminal_bench.models import (
    TerminalBenchmarkOutcome,
    TerminalExecutionPolicy,
    TerminalExecutionState,
    TerminalTrialSummary,
    TerminalVerifierTimeoutAttribution,
    utc_now,
)
from applications.terminal_bench.result_analyzer import TerminalResultAnalyzer


def summary(*, agent_complete: bool = True) -> TerminalTrialSummary:
    now = utc_now()
    return TerminalTrialSummary(
        trial_id="trial-result",
        run_id=uuid4(),
        task_id=uuid4(),
        agent_complete=agent_complete,
        runtime_status="completed",
        final_output={"agent_complete": agent_complete},
        command_count=3,
        denial_count=1,
        timeout_count=1,
        in_doubt_count=1,
        inference_attempt_count=3,
        inference_succeeded_count=2,
        inference_budget_rejection_count=0,
        inference_timeout_count=1,
        inference_transport_failure_count=0,
        inference_backend_failure_count=0,
        inference_cancelled_count=0,
        inference_attempt_latency_ms=987,
        input_tokens=10,
        output_tokens=4,
        total_tokens=14,
        cost_usd=0.25,
        latency_ms=1200,
        trace_consistent=True,
        started_at=now,
        completed_at=now,
    )


def analyze_verifier_timeout(verifier_text: str):
    with tempfile.TemporaryDirectory() as directory:
        trial_dir = Path(directory) / "trial"
        agent_dir = trial_dir / "agent"
        verifier_dir = trial_dir / "verifier"
        agent_dir.mkdir(parents=True)
        verifier_dir.mkdir(parents=True)
        summary_path = agent_dir / "aar-summary.json"
        summary_path.write_text(summary().model_dump_json(), encoding="utf-8")
        (verifier_dir / "test-stdout.txt").write_text(
            verifier_text,
            encoding="utf-8",
        )
        return TerminalResultAnalyzer().analyze(
            harbor_trial_result={
                "verifier_result": None,
                "exception_info": {
                    "exception_type": "VerifierTimeoutError",
                    "exception_message": (
                        "Verifier execution timed out after 900.0 seconds"
                    ),
                },
                "agent_result": {},
            },
            aar_summary=summary_path,
        )


class _FakeApplication:
    def __init__(self, artifacts: TerminalRunArtifacts) -> None:
        self.artifacts = artifacts
        self.closed = False

    async def run(self, instruction: str) -> TerminalRunArtifacts:
        del instruction
        return self.artifacts

    def close(self) -> None:
        self.closed = True


class _RuntimeTimeoutEnvironment:
    async def exec(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        raise RuntimeError("Command timed out after 60 seconds")


class HarborAndResultTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_timeout_wrapper_sets_timeout_metric(self) -> None:
        adapter = HarborTerminalEnvironmentAdapter(
            _RuntimeTimeoutEnvironment(),  # type: ignore[arg-type]
            max_output_characters=1_000,
        )

        result = await adapter.exec("long-running-command", timeout_sec=60)

        self.assertTrue(result.timed_out)
        self.assertTrue(result.transport_failed)
        self.assertIs(result.execution_state, TerminalExecutionState.IN_DOUBT)

    def test_regression12_manifest_is_fixed_and_unique(self) -> None:
        manifest = (
            Path(__file__).parents[2]
            / "applications"
            / "terminal_bench"
            / "evaluation_sets"
            / "regression12.txt"
        )
        tasks = tuple(
            line.strip()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        self.assertEqual(len(tasks), 12)
        self.assertEqual(len(set(tasks)), 12)
        self.assertEqual(
            tasks[-4:],
            (
                "terminal-bench/regex-log",
                "terminal-bench/db-wal-recovery",
                "terminal-bench/nginx-request-logging",
                "terminal-bench/prove-plus-comm",
            ),
        )
        self.assertEqual(tasks[0], "terminal-bench/build-pmars")

    async def test_agent_context_uses_metadata_for_aar_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_result = SimpleNamespace()
            fake_app = _FakeApplication(
                TerminalRunArtifacts(
                    runtime_result=fake_result,  # type: ignore[arg-type]
                    summary=summary(),
                )
            )

            def factory(**kwargs):  # type: ignore[no-untyped-def]
                self.assertIn("model_config", kwargs)
                return fake_app

            agent = AdaptiveRuntimeHarborAgent(
                logs_dir=Path(directory),
                model_name="openai/test-model",
                application_factory=factory,
                extra_env={"OPENAI_API_KEY": "host-only-secret"},
            )
            context = AgentContext()
            environment = SimpleNamespace(
                context_id=uuid4(),
                session_id="trial-session",
            )
            await agent.run("solve", environment, context)  # type: ignore[arg-type]

            self.assertEqual(context.n_input_tokens, 10)
            self.assertEqual(context.n_output_tokens, 4)
            self.assertEqual(context.cost_usd, 0.25)
            self.assertIn("aar", context.metadata)
            self.assertFalse(hasattr(context, "aar_summary"))
            self.assertTrue(fake_app.closed)

    def test_agent_timeout_reserves_time_for_delivery_and_verifier_handoff(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = AdaptiveRuntimeHarborAgent(
                logs_dir=Path(directory),
                model_name="codex-cli/luna-high",
                agent_timeout_sec="900",
                deadline_reserve_seconds="120",
            )

            self.assertEqual(
                agent._execution_policy.max_wall_clock_seconds,
                780.0,
            )
            self.assertEqual(
                agent._execution_policy.deadline_reserve_seconds,
                120.0,
            )

    def test_result_analyzer_runs_after_harbor_verifier(self) -> None:
        analyzer = TerminalResultAnalyzer()
        with self.assertRaisesRegex(ValueError, "after verification"):
            analyzer.analyze(
                harbor_trial_result={"agent_result": {}},
                aar_summary=summary(),
            )
        now = utc_now()
        analysis = analyzer.analyze(
            harbor_trial_result={
                "agent_result": {
                    "n_input_tokens": 12,
                    "n_output_tokens": 5,
                    "cost_usd": 0.3,
                },
                "verifier_result": {"rewards": {"reward": 1}},
                "agent_execution": {
                    "started_at": now.isoformat(),
                    "finished_at": (now + timedelta(seconds=2)).isoformat(),
                },
            },
            aar_summary=summary(),
        )
        self.assertTrue(analysis.benchmark_pass)
        self.assertEqual(analysis.total_tokens, 17)
        self.assertEqual(analysis.latency_ms, 2000)
        self.assertEqual(analysis.inference_attempt_count, 3)
        self.assertEqual(analysis.inference_timeout_count, 1)
        self.assertEqual(analysis.inference_attempt_latency_ms, 987)

    def test_agent_complete_does_not_count_as_benchmark_pass(self) -> None:
        analysis = TerminalResultAnalyzer().analyze(
            harbor_trial_result={
                "verifier_result": {"rewards": {"reward": 0}},
                "agent_result": {},
            },
            aar_summary=summary(agent_complete=True),
        )
        self.assertTrue(analysis.agent_complete)
        self.assertFalse(analysis.benchmark_pass)
        self.assertFalse(analysis.completion_matches_verifier)

    def test_inference_transport_failure_is_infrastructure(self) -> None:
        transport_summary = summary().model_copy(
            update={
                "runtime_status": "failed",
                "inference_attempt_count": 1,
                "inference_succeeded_count": 0,
                "inference_timeout_count": 0,
                "inference_transport_failure_count": 1,
            }
        )
        analysis = TerminalResultAnalyzer().analyze(
            harbor_trial_result={
                "verifier_result": {"rewards": {"reward": 0}},
                "agent_result": {},
            },
            aar_summary=transport_summary,
        )

        self.assertTrue(analysis.infrastructure_error)
        self.assertIs(
            analysis.outcome,
            TerminalBenchmarkOutcome.INFRASTRUCTURE_ERROR,
        )
        self.assertIn(
            "inference transport",
            analysis.infrastructure_error_reason or "",
        )

    def test_verifier_dependency_setup_timeout_is_infrastructure(self) -> None:
        analysis = analyze_verifier_timeout(
            "Reading package lists...\n"
            "curl: (7) unable to connect to package host\n"
        )

        self.assertIs(
            analysis.verifier_timeout_attribution,
            TerminalVerifierTimeoutAttribution.INFRASTRUCTURE,
        )
        self.assertIs(
            analysis.outcome,
            TerminalBenchmarkOutcome.INFRASTRUCTURE_ERROR,
        )
        self.assertTrue(analysis.infrastructure_error)

    def test_explicit_submission_timeout_is_task_attributable(self) -> None:
        analysis = analyze_verifier_timeout(
            "pytest test session starts\n"
            "subprocess.TimeoutExpired: Command ['/app/server'] timed out\n"
        )

        self.assertIs(
            analysis.verifier_timeout_attribution,
            TerminalVerifierTimeoutAttribution.TASK_ATTRIBUTABLE,
        )
        self.assertIs(
            analysis.outcome,
            TerminalBenchmarkOutcome.BENCHMARK_FAIL,
        )
        self.assertFalse(analysis.infrastructure_error)

    def test_ambiguous_verifier_timeout_is_inconclusive(self) -> None:
        analysis = analyze_verifier_timeout("verifier stopped producing output\n")

        self.assertIs(
            analysis.verifier_timeout_attribution,
            TerminalVerifierTimeoutAttribution.INCONCLUSIVE,
        )
        self.assertIs(
            analysis.outcome,
            TerminalBenchmarkOutcome.INCONCLUSIVE,
        )
        self.assertFalse(analysis.infrastructure_error)
        self.assertFalse(analysis.completion_matches_verifier)

    def test_setup_preamble_does_not_override_started_tests(self) -> None:
        analysis = analyze_verifier_timeout(
            "Reading package lists...\n"
            "pytest test session starts\n"
            "collected 3 items\n"
        )

        self.assertIs(
            analysis.verifier_timeout_attribution,
            TerminalVerifierTimeoutAttribution.INCONCLUSIVE,
        )
        self.assertIs(
            analysis.outcome,
            TerminalBenchmarkOutcome.INCONCLUSIVE,
        )

    def test_verifier_network_setup_failure_is_infrastructure_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trial_dir = Path(directory) / "trial"
            agent_dir = trial_dir / "agent"
            verifier_dir = trial_dir / "verifier"
            agent_dir.mkdir(parents=True)
            verifier_dir.mkdir(parents=True)
            summary_path = agent_dir / "aar-summary.json"
            summary_path.write_text(summary().model_dump_json(), encoding="utf-8")
            (verifier_dir / "test-stderr.txt").write_text(
                "curl: (7) unable to connect to package host\n"
                "failed to fetch verifier dependencies\n",
                encoding="utf-8",
            )

            analysis = TerminalResultAnalyzer().analyze(
                harbor_trial_result={
                    "verifier_result": {"rewards": {"reward": 0}},
                    "agent_result": {},
                },
                aar_summary=summary_path,
            )

        self.assertEqual(
            analysis.outcome,
            TerminalBenchmarkOutcome.INFRASTRUCTURE_ERROR,
        )
        self.assertTrue(analysis.infrastructure_error)
        self.assertFalse(analysis.benchmark_pass)
        self.assertIn(
            "network path",
            analysis.infrastructure_error_reason or "",
        )


if __name__ == "__main__":
    unittest.main()
