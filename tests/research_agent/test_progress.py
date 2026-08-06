from __future__ import annotations

import unittest

from applications.research_agent.agent import ResearchAgent as RuntimeResearchAgent
from tests.runtime_database import register_test_resource, runtime_database_path


def ResearchAgent(**kwargs: object) -> RuntimeResearchAgent:
    kwargs.setdefault("persistence_path", runtime_database_path())
    kwargs.setdefault("run_kind", "test")
    kwargs.setdefault("disposable", True)
    return register_test_resource(
        RuntimeResearchAgent(**kwargs)  # type: ignore[arg-type]
    )
from applications.research_agent.progress import (
    ResearchProgressEvent,
    ResearchProgressKind,
)


class RecordingProgressSink:
    def __init__(self) -> None:
        self.events: list[ResearchProgressEvent] = []

    async def publish(self, event: ResearchProgressEvent) -> None:
        self.events.append(event)


class FailingProgressSink:
    async def publish(self, event: ResearchProgressEvent) -> None:
        raise RuntimeError("presentation is unavailable")


class ResearchProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_trace_drives_live_graph_events(self) -> None:
        sink = RecordingProgressSink()

        result = await ResearchAgent().run(
            "分析 Tesla 投资价值",
            progress_sink=sink,
        )

        kinds = [event.kind for event in sink.events]
        self.assertEqual(kinds[0], ResearchProgressKind.RUN_PREPARING)
        self.assertIn(ResearchProgressKind.GRAPH_INITIALIZED, kinds)
        self.assertEqual(kinds.count(ResearchProgressKind.NODE_STARTED), 8)
        self.assertEqual(kinds.count(ResearchProgressKind.NODE_COMPLETED), 8)
        self.assertEqual(kinds.count(ResearchProgressKind.GRAPH_NODE_ADDED), 1)
        self.assertEqual(
            kinds.count(ResearchProgressKind.GRAPH_DEPENDENCY_ADDED),
            1,
        )
        self.assertEqual(
            kinds.count(ResearchProgressKind.TRACE_RECORDED),
            len(result.runtime_trace),
        )
        self.assertEqual(kinds.count(ResearchProgressKind.RUNTIME_COMPLETED), 1)
        self.assertEqual(
            {event.run_id for event in sink.events},
            {result.runtime_result.final_state.run_id},
        )

    async def test_progress_failure_does_not_fail_runtime(self) -> None:
        result = await ResearchAgent().run(
            "分析 Tesla 投资价值",
            progress_sink=FailingProgressSink(),
        )

        self.assertTrue(result.runtime_result.succeeded)


if __name__ == "__main__":
    unittest.main()
