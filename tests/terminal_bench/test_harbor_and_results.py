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
)
from applications.terminal_bench.models import TerminalTrialSummary, utc_now
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
        input_tokens=10,
        output_tokens=4,
        total_tokens=14,
        cost_usd=0.25,
        latency_ms=1200,
        trace_consistent=True,
        started_at=now,
        completed_at=now,
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


class HarborAndResultTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()
