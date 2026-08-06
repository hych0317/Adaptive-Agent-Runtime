from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
from uuid import UUID

from adaptive_agent_runtime.decision_feedback import DECISION_FEEDBACK_DECISION_TYPE
from adaptive_agent_runtime.decisioning import DecisionFaultPoint
from applications.research_agent.agent import ResearchAgent


def crash(database: Path, marker: Path) -> None:
    def fault(point, checkpoint):  # type: ignore[no-untyped-def]
        if (
            point is DecisionFaultPoint.EFFECT_COMMITTED
            and checkpoint.request.decision_type == DECISION_FEEDBACK_DECISION_TYPE
        ):
            marker.write_text(str(checkpoint.run_id), encoding="utf-8")
            os._exit(73)

    agent = ResearchAgent(
        persistence_path=database,
        decision_fault_injector=fault,
    )
    asyncio.run(agent.run("分析 Acme 的投资价值"))
    agent.close()
    raise SystemExit("fault point was not reached")


def resume(database: Path, marker: Path) -> None:
    run_id = UUID(marker.read_text(encoding="utf-8").strip())
    agent = ResearchAgent(persistence_path=database)
    try:
        result = asyncio.run(agent.resume(run_id))
    finally:
        agent.close()
    print(
        json.dumps(
            {
                "run_id": str(run_id),
                "feedback_count": len(result.decision_feedback),
                "effect_fingerprints": [
                    item.effect_fingerprint for item in result.decision_feedback
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    mode = sys.argv[1]
    database = Path(sys.argv[2])
    marker = Path(sys.argv[3])
    if mode == "crash":
        crash(database, marker)
    elif mode == "resume":
        resume(database, marker)
    else:
        raise SystemExit(f"unknown mode: {mode}")
