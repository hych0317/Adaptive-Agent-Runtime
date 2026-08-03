from __future__ import annotations

import unittest
from uuid import UUID, uuid4

from adaptive_agent_runtime.context_memory import (
    ContextAssembler,
    ContextAssembly,
    ContextBudgetExceededError,
    ContextLayer,
    ContextLifecycleState,
    ContextMetadata,
    ContextRequirement,
    ContextScheduler,
    ContextSource,
    ContextUnit,
    ResidencyPolicy,
)


def unit(
    run_id: UUID,
    layer: ContextLayer,
    *,
    tokens: int,
    residency: ResidencyPolicy = ResidencyPolicy.SESSION,
    lifecycle: ContextLifecycleState = ContextLifecycleState.ACTIVE,
) -> ContextUnit:
    return ContextUnit(
        content=f"{layer.value}-{tokens}",
        metadata=ContextMetadata(
            source=ContextSource.WORKING_STATE,
            layer=layer,
            run_id=run_id,
            estimated_tokens=tokens,
        ),
        residency_policy=residency,
        lifecycle_state=lifecycle,
    )


class ContextSchedulerTests(unittest.TestCase):
    def test_scheduler_respects_layer_order_and_budget(self) -> None:
        run_id = uuid4()
        working = unit(
            run_id,
            ContextLayer.WORKING,
            tokens=2,
            residency=ResidencyPolicy.PINNED,
        )
        task = unit(run_id, ContextLayer.TASK, tokens=3)
        semantic = unit(run_id, ContextLayer.SEMANTIC, tokens=3)
        requirement = ContextRequirement(
            run_id=run_id,
            goal="next task",
            max_units=2,
            max_tokens=5,
        )

        schedule = ContextScheduler().schedule(
            requirement,
            (semantic, task, working),
        )
        assembly = ContextAssembler().assemble(schedule)

        self.assertEqual(schedule.selected, (working, task))
        self.assertEqual(schedule.used_tokens, 5)
        self.assertIn(semantic.context_id, schedule.omitted_context_ids)
        self.assertEqual(assembly.units, (working, task))
        self.assertEqual(assembly.working_context, (working,))
        self.assertEqual(assembly.task_context, (task,))

    def test_assembly_contains_all_three_layers_in_order(self) -> None:
        run_id = uuid4()
        working = unit(run_id, ContextLayer.WORKING, tokens=1)
        task = unit(run_id, ContextLayer.TASK, tokens=1)
        semantic = unit(run_id, ContextLayer.SEMANTIC, tokens=1)
        requirement = ContextRequirement(run_id=run_id, goal="all layers")

        schedule = ContextScheduler().schedule(
            requirement,
            (semantic, working, task),
        )
        assembly = ContextAssembler().assemble(schedule)

        self.assertEqual(assembly.units, (working, task, semantic))
        self.assertEqual(assembly.working_context, (working,))
        self.assertEqual(assembly.task_context, (task,))
        self.assertEqual(assembly.semantic_context, (semantic,))

    def test_assembly_rejects_a_unit_in_the_wrong_layer_bucket(self) -> None:
        run_id = uuid4()
        task = unit(run_id, ContextLayer.TASK, tokens=1)
        requirement = ContextRequirement(run_id=run_id, goal="invalid bucket")

        with self.assertRaises(ValueError):
            ContextAssembly(
                requirement=requirement,
                units=(task,),
                working_context=(task,),
                task_context=(),
                semantic_context=(),
                used_tokens=1,
            )

    def test_mandatory_context_over_budget_is_rejected(self) -> None:
        run_id = uuid4()
        pinned = unit(
            run_id,
            ContextLayer.WORKING,
            tokens=6,
            residency=ResidencyPolicy.PINNED,
        )
        requirement = ContextRequirement(
            run_id=run_id,
            goal="small budget",
            max_tokens=5,
        )

        with self.assertRaises(ContextBudgetExceededError):
            ContextScheduler().schedule(requirement, (pinned,))

    def test_archived_context_is_not_assembled(self) -> None:
        run_id = uuid4()
        active = unit(run_id, ContextLayer.TASK, tokens=2)
        archived = unit(
            run_id,
            ContextLayer.TASK,
            tokens=2,
            lifecycle=ContextLifecycleState.ARCHIVED,
        )
        requirement = ContextRequirement(run_id=run_id, goal="active only")

        schedule = ContextScheduler().schedule(
            requirement,
            (archived, active),
        )

        self.assertEqual(schedule.selected, (active,))
        self.assertIn(archived.context_id, schedule.omitted_context_ids)

    def test_latest_archived_snapshot_suppresses_stale_active_snapshot(self) -> None:
        run_id = uuid4()
        active = unit(run_id, ContextLayer.TASK, tokens=2)
        values = active.model_dump(mode="python")
        values.update(
            revision=active.revision + 1,
            lifecycle_state=ContextLifecycleState.ARCHIVED,
        )
        archived = ContextUnit.model_validate(values)
        requirement = ContextRequirement(run_id=run_id, goal="latest only")

        schedule = ContextScheduler().schedule(
            requirement,
            (archived, active),
        )

        self.assertEqual(schedule.selected, ())
        self.assertEqual(schedule.omitted_context_ids, (active.context_id,))


if __name__ == "__main__":
    unittest.main()
