from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from uuid import uuid4

from adaptive_agent_runtime.core import AgentState, AgentTask, RunStatus
from adaptive_agent_runtime.llm import (
    InferenceExecutionBudgetError,
    InferenceGatewayPolicy,
)

from applications.terminal_bench.composition import build_terminal_application
from applications.terminal_bench.deadline import (
    TerminalDeadlineSequence,
    TerminalInferenceTiming,
    allocate_deadline_slots,
    allocate_profiled_terminal_deadline_sequence,
    allocate_terminal_deadline_sequence,
    terra_high_deadline_budget_profile,
)
from applications.terminal_bench.models import (
    TerminalCommandIntent,
    TerminalCommandRecord,
    TerminalCommandRole,
    TerminalExecutionLimits,
    TerminalExecutionPolicy,
    TerminalReconciliationState,
    TerminalRequirement,
    TerminalSessionSnapshot,
    TerminalTurnRequest,
    utc_now,
)
from applications.terminal_bench.planner import (
    GatewayTerminalTurnProposalCapability,
)
from tests.terminal_bench.fakes import (
    FakeTerminalEnvironment,
    ScriptedTerminalTurnCapability,
    completed_result,
)


_TIMING = TerminalInferenceTiming(
    normal_preferred_seconds=300.0,
    compact_preferred_seconds=180.0,
    emergency_preferred_seconds=120.0,
    normal_minimum_seconds=120.0,
    compact_minimum_seconds=60.0,
)
_PROFILE = terra_high_deadline_budget_profile()



class _DeadlineScriptedCapability(ScriptedTerminalTurnCapability):
    deadline_timing = _TIMING

class _ProfiledDeadlineScriptedCapability(ScriptedTerminalTurnCapability):
    deadline_budget_profile = _PROFILE



class _NeverCalledGateway:
    module_id = "test.deadline.never_called"

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, request: object, policy: object) -> object:
        del request, policy
        self.calls += 1
        raise AssertionError("deadline admission must precede backend execution")


class TerminalDeadlineSlotTests(unittest.TestCase):
    def test_named_sequences_reserve_every_followup_phase(self) -> None:
        cases = (
            (
                TerminalDeadlineSequence.RECONCILE_THEN_WORK_VERIFY,
                378.0,
                60,
                240.0,
            ),
            (
                TerminalDeadlineSequence.RECONCILE_THEN_VERIFY,
                258.0,
                60,
                120.0,
            ),
            (
                TerminalDeadlineSequence.WORK_THEN_VERIFY,
                258.0,
                60,
                120.0,
            ),
            (
                TerminalDeadlineSequence.DIRECT_VERIFY,
                198.0,
                120,
                0.0,
            ),
        )

        for sequence, remaining, action_cap, future_reserve in cases:
            with self.subTest(sequence=sequence):
                slots = allocate_terminal_deadline_sequence(
                    sequence=sequence,
                    remaining_seconds=remaining,
                    minimum_inference_seconds=60.0,
                    preferred_inference_seconds=180.0,
                    followup_inference_seconds=60.0,
                    verification_timeout_seconds=(
                        120
                        if sequence is TerminalDeadlineSequence.DIRECT_VERIFY
                        else 60
                    ),
                    cleanup_seconds=18.0,
                )

                self.assertTrue(slots.feasible)
                self.assertIs(slots.sequence, sequence)
                self.assertEqual(slots.action_limit_seconds, action_cap)
                self.assertEqual(slots.future_reserve_seconds, future_reserve + 18.0)

    def test_work_then_verify_fits_exact_258_second_boundary(self) -> None:
        slots = allocate_terminal_deadline_sequence(
            sequence=TerminalDeadlineSequence.WORK_THEN_VERIFY,
            remaining_seconds=258.0,
            minimum_inference_seconds=60.0,
            preferred_inference_seconds=180.0,
            followup_inference_seconds=60.0,
            verification_timeout_seconds=60,
            cleanup_seconds=18.0,
        )

        self.assertTrue(slots.feasible)
        self.assertTrue(slots.complete_sequence_feasible)
        self.assertEqual(slots.inference_limit_seconds, 60.0)
        self.assertEqual(slots.action_limit_seconds, 60)
        self.assertEqual(slots.future_reserve_seconds, 138.0)

        direct = allocate_terminal_deadline_sequence(
            sequence=TerminalDeadlineSequence.DIRECT_VERIFY,
            remaining_seconds=198.0,
            minimum_inference_seconds=60.0,
            preferred_inference_seconds=180.0,
            followup_inference_seconds=60.0,
            verification_timeout_seconds=120,
            cleanup_seconds=18.0,
        )
        self.assertTrue(direct.feasible)
        self.assertEqual(direct.action_limit_seconds, 120)

    def test_repair_then_verify_fits_exact_318_second_boundary(self) -> None:
        slots = allocate_deadline_slots(
            remaining_seconds=318.0,
            minimum_inference_seconds=60.0,
            preferred_inference_seconds=180.0,
            preferred_action_seconds=60,
            maximum_action_seconds=60,
            followup_inference_seconds=60.0,
            verification_seconds=120.0,
            cleanup_seconds=18.0,
        )

        self.assertTrue(slots.feasible)
        self.assertEqual(slots.inference_limit_seconds, 60.0)
        self.assertEqual(slots.action_limit_seconds, 60)
        self.assertEqual(slots.future_reserve_seconds, 198.0)

    def test_repair_cap_shrinks_without_spending_verification_reserve(self) -> None:
        slots = allocate_deadline_slots(
            remaining_seconds=300.0,
            minimum_inference_seconds=60.0,
            preferred_inference_seconds=180.0,
            preferred_action_seconds=60,
            maximum_action_seconds=60,
            followup_inference_seconds=60.0,
            verification_seconds=120.0,
            cleanup_seconds=18.0,
        )

        self.assertTrue(slots.feasible)
        self.assertEqual(slots.inference_limit_seconds, 60.0)
        self.assertEqual(slots.action_limit_seconds, 42)

    def test_named_sequence_degrades_without_a_199_second_cliff(self) -> None:
        exact = allocate_terminal_deadline_sequence(
            sequence=TerminalDeadlineSequence.WORK_THEN_VERIFY,
            remaining_seconds=199.0,
            minimum_inference_seconds=60.0,
            preferred_inference_seconds=180.0,
            followup_inference_seconds=60.0,
            verification_timeout_seconds=60,
            cleanup_seconds=18.0,
        )
        degraded = allocate_terminal_deadline_sequence(
            sequence=TerminalDeadlineSequence.WORK_THEN_VERIFY,
            remaining_seconds=198.0,
            minimum_inference_seconds=60.0,
            preferred_inference_seconds=180.0,
            followup_inference_seconds=60.0,
            verification_timeout_seconds=60,
            cleanup_seconds=18.0,
        )

        self.assertTrue(exact.feasible)
        self.assertTrue(exact.complete_sequence_feasible)
        self.assertEqual(exact.action_limit_seconds, 1)
        self.assertTrue(degraded.feasible)
        self.assertFalse(degraded.complete_sequence_feasible)
        self.assertEqual(degraded.action_limit_seconds, 60)
        self.assertGreaterEqual(
            degraded.inference_limit_seconds or 0.0,
            60.0,
        )
        self.assertEqual(degraded.future_reserve_seconds, 18.0)

    def test_sequence_is_infeasible_without_one_action_second(self) -> None:
        slots = allocate_deadline_slots(
            remaining_seconds=258.0,
            minimum_inference_seconds=60.0,
            preferred_inference_seconds=180.0,
            preferred_action_seconds=60,
            maximum_action_seconds=60,
            followup_inference_seconds=60.0,
            verification_seconds=120.0,
            cleanup_seconds=18.0,
        )

        self.assertFalse(slots.feasible)
        self.assertFalse(slots.complete_sequence_feasible)
        self.assertEqual(slots.action_limit_seconds, 0)

    def test_direct_verification_uses_remaining_action_capacity(self) -> None:
        slots = allocate_deadline_slots(
            remaining_seconds=150.0,
            minimum_inference_seconds=60.0,
            preferred_inference_seconds=180.0,
            preferred_action_seconds=120,
            maximum_action_seconds=120,
            cleanup_seconds=18.0,
        )

        self.assertTrue(slots.feasible)
        self.assertEqual(slots.inference_limit_seconds, 60.0)
        self.assertEqual(slots.action_limit_seconds, 72)

    def test_delivery_balances_inference_action_and_future_verify(self) -> None:
        slots = allocate_deadline_slots(
            remaining_seconds=480.0,
            minimum_inference_seconds=60.0,
            preferred_inference_seconds=180.0,
            preferred_action_seconds=120,
            maximum_action_seconds=300,
            followup_inference_seconds=60.0,
            verification_seconds=120.0,
            cleanup_seconds=18.0,
        )

        self.assertTrue(slots.feasible)
        self.assertEqual(slots.inference_limit_seconds, 162.0)
        self.assertEqual(slots.action_limit_seconds, 120)

    def test_ample_budget_expands_action_after_inference_preference(self) -> None:
        slots = allocate_deadline_slots(
            remaining_seconds=840.0,
            minimum_inference_seconds=120.0,
            preferred_inference_seconds=300.0,
            preferred_action_seconds=120,
            maximum_action_seconds=300,
            followup_inference_seconds=60.0,
            verification_seconds=120.0,
            cleanup_seconds=18.0,
        )

        self.assertTrue(slots.feasible)
        self.assertEqual(slots.inference_limit_seconds, 300.0)
        self.assertEqual(slots.action_limit_seconds, 300)

    def test_no_deadline_leaves_caps_unbounded_by_allocator(self) -> None:
        slots = allocate_deadline_slots(
            remaining_seconds=None,
            minimum_inference_seconds=120.0,
            preferred_inference_seconds=300.0,
            preferred_action_seconds=120,
            maximum_action_seconds=300,
        )

        self.assertTrue(slots.feasible)
        self.assertIsNone(slots.inference_limit_seconds)
        self.assertIsNone(slots.action_limit_seconds)


class TerminalProfileDeadlineSlotTests(unittest.TestCase):
    def test_profile_sequence_interpolates_continuously(self) -> None:
        remaining_values = (160.0, 200.0, 240.0)
        slots = tuple(
            allocate_profiled_terminal_deadline_sequence(
                sequence=TerminalDeadlineSequence.WORK_THEN_VERIFY,
                remaining_seconds=remaining,
                profile=_PROFILE,
                cleanup_seconds=18.0,
                provider_grace_seconds=10.0,
            )
            for remaining in remaining_values
        )

        for item in slots:
            self.assertTrue(item.feasible)
            self.assertTrue(item.complete_sequence_feasible)
        action_caps = tuple(item.action_limit_seconds for item in slots)
        inference_caps = tuple(item.inference_limit_seconds for item in slots)
        self.assertEqual(action_caps, tuple(sorted(action_caps)))
        self.assertEqual(inference_caps, tuple(sorted(inference_caps)))

    def test_emergency_current_turn_reaches_preferred_before_future_growth(
        self,
    ) -> None:
        slots = allocate_profiled_terminal_deadline_sequence(
            sequence=(
                TerminalDeadlineSequence.RECONCILE_THEN_WORK_VERIFY
            ),
            remaining_seconds=231.0,
            profile=_PROFILE,
            cleanup_seconds=18.0,
            provider_grace_seconds=10.0,
            emergency_mode=True,
        )

        self.assertTrue(slots.feasible)
        self.assertTrue(slots.complete_sequence_feasible)
        self.assertEqual(
            slots.inference_limit_seconds,
            _PROFILE.emergency_inference.preferred_seconds,
        )
        self.assertEqual(
            slots.action_limit_seconds,
            int(_PROFILE.reconciliation_inspection.preferred_seconds),
        )

    def test_profile_reserves_future_minimum_and_stage_overheads(self) -> None:
        remaining = 200.0
        slots = allocate_profiled_terminal_deadline_sequence(
            sequence=TerminalDeadlineSequence.WORK_THEN_VERIFY,
            remaining_seconds=remaining,
            profile=_PROFILE,
            cleanup_seconds=18.0,
            provider_grace_seconds=10.0,
        )

        allocated = sum(
            item.allocated_seconds + item.overhead_seconds
            for item in slots.stage_allocations
        )
        self.assertAlmostEqual(allocated + slots.cleanup_seconds, remaining)
        for item in slots.stage_allocations:
            self.assertGreaterEqual(
                item.allocated_seconds,
                item.minimum_seconds,
            )

    def test_profile_degrades_to_current_phase_below_sequence_minimum(
        self,
    ) -> None:
        slots = allocate_profiled_terminal_deadline_sequence(
            sequence=TerminalDeadlineSequence.WORK_THEN_VERIFY,
            remaining_seconds=127.0,
            profile=_PROFILE,
            cleanup_seconds=18.0,
            provider_grace_seconds=10.0,
        )

        self.assertTrue(slots.feasible)
        self.assertFalse(slots.complete_sequence_feasible)
        self.assertEqual(
            tuple(item.current for item in slots.stage_allocations),
            (True, True),
        )
        self.assertEqual(slots.future_reserve_seconds, 18.0)

    def test_profile_ample_budget_restores_long_work_capacity(self) -> None:
        slots = allocate_profiled_terminal_deadline_sequence(
            sequence=None,
            remaining_seconds=840.0,
            profile=_PROFILE,
            cleanup_seconds=18.0,
            provider_grace_seconds=10.0,
        )
        inspection = allocate_profiled_terminal_deadline_sequence(
            sequence=None,
            remaining_seconds=840.0,
            profile=_PROFILE,
            cleanup_seconds=18.0,
            provider_grace_seconds=10.0,
            command_role="inspect",
        )

        self.assertEqual(slots.action_limit_seconds, 300)
        self.assertEqual(slots.inference_limit_seconds, 300.0)
        self.assertEqual(inspection.action_limit_seconds, 60)

    def test_profile_no_deadline_preserves_unbounded_compatibility(self) -> None:
        slots = allocate_profiled_terminal_deadline_sequence(
            sequence=None,
            remaining_seconds=None,
            profile=_PROFILE,
            cleanup_seconds=18.0,
            provider_grace_seconds=10.0,
        )

        self.assertTrue(slots.feasible)
        self.assertIsNone(slots.inference_limit_seconds)
        self.assertIsNone(slots.action_limit_seconds)


class TerminalDeadlineIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_zero_inference_cap_never_calls_backend(self) -> None:
        gateway = _NeverCalledGateway()
        capability = GatewayTerminalTurnProposalCapability(
            gateway=gateway,  # type: ignore[arg-type]
            gateway_policy=InferenceGatewayPolicy(),
            target_id="terminal-bench:test:model",
        )
        request = TerminalTurnRequest(
            run_id=uuid4(),
            task_id=uuid4(),
            instruction="create an artifact",
            requirements=(
                TerminalRequirement(
                    requirement_id="req-001",
                    description="create an artifact",
                ),
            ),
            session=TerminalSessionSnapshot(trial_id="deadline-no-backend"),
            execution_limits=TerminalExecutionLimits(
                max_inference_timeout_sec=0.0,
            ),
            remaining_commands=1,
            remaining_wall_clock_seconds=100.0,
            execution_semantics=("independent commands",),
        )

        with self.assertRaises(InferenceExecutionBudgetError):
            await capability.propose(request)
        self.assertEqual(gateway.calls, 0)

    async def test_known_initial_state_prefers_direct_verify_in_finalization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = _DeadlineScriptedCapability()
            app = build_terminal_application(
                trial_id="deadline-finalization-without-work",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(),
                proposal_capability=capability,
                policy=self._deadline_policy(),
            )
            try:
                app.journal._session = app.journal.snapshot().model_copy(
                    update={
                        "started_at": utc_now() - timedelta(seconds=581),
                    }
                )
                request = app.runtime._planner._turn_request(
                    self._state("create and verify an artifact"),
                    app.journal.snapshot(),
                )

                self.assertTrue(request.finalization_mode)
                self.assertTrue(request.verification_due)
                self.assertIs(
                    request.execution_limits.deadline_sequence,
                    TerminalDeadlineSequence.DIRECT_VERIFY,
                )
                inference_cap = (
                    request.execution_limits.max_inference_timeout_sec or 0.0
                )
                self.assertGreater(inference_cap, 0.0)
                self.assertLessEqual(
                    inference_cap,
                    _TIMING.compact_preferred_seconds,
                )
            finally:
                app.close()

    async def test_finalization_falls_back_to_direct_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = _DeadlineScriptedCapability()
            app = build_terminal_application(
                trial_id="deadline-direct-verify-fallback",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(),
                proposal_capability=capability,
                policy=self._deadline_policy(),
            )
            try:
                trial_id = app.journal.trial_id
                app.journal._records = [
                    self._record(
                        trial_id,
                        "work-1",
                        TerminalCommandRole.WORK,
                    ),
                    self._record(
                        trial_id,
                        "inspect-1",
                        TerminalCommandRole.INSPECT,
                    ),
                ]
                app.journal._session = app.journal.snapshot().model_copy(
                    update={
                        "started_at": utc_now() - timedelta(seconds=590),
                        "committed_commands": 2,
                        "task_generation": 1,
                        "known_state_generation": 1,
                        "successful_work_generation": 1,
                    }
                )
                request = app.runtime._planner._turn_request(
                    self._state("create and verify an artifact"),
                    app.journal.snapshot(),
                )

                self.assertTrue(request.finalization_mode)
                self.assertTrue(request.verification_due)
                self.assertIs(
                    request.execution_limits.deadline_sequence,
                    TerminalDeadlineSequence.DIRECT_VERIFY,
                )
                self.assertEqual(
                    request.execution_limits.max_timeout_sec,
                    120,
                )
                self.assertGreater(
                    request.execution_limits.max_inference_timeout_sec or 0.0,
                    0.0,
                )
                self.assertEqual(
                    request.execution_limits.followup_inference_reserve_seconds,
                    0.0,
                )

                app.journal._session = app.journal.snapshot().model_copy(
                    update={
                        "in_doubt_reconciliation_required": True,
                        "known_state_generation": None,
                        "reconciliation_state": (
                            TerminalReconciliationState.REQUIRED
                        ),
                    }
                )
                reconciliation_request = app.runtime._planner._turn_request(
                    self._state("create and verify an artifact"),
                    app.journal.snapshot(),
                )
                self.assertTrue(reconciliation_request.reconciliation_mode)
                self.assertFalse(reconciliation_request.verification_due)
                self.assertIs(
                    reconciliation_request.execution_limits.deadline_sequence,
                    TerminalDeadlineSequence.RECONCILE_THEN_VERIFY,
                )
            finally:
                app.close()

    async def test_fresh_finalization_keeps_inspection_advisory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="deadline-stale-inspection",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(),
                proposal_capability=_DeadlineScriptedCapability(),
                policy=self._deadline_policy(),
            )
            try:
                session = app.journal.snapshot().model_copy(
                    update={
                        "started_at": utc_now() - timedelta(seconds=581),
                        "successful_work_generation": 0,
                    }
                )
                intent = TerminalCommandIntent(
                    trial_id=app.journal.trial_id,
                    call_key="stale-inspect",
                    command="pwd",
                    command_role=TerminalCommandRole.INSPECT,
                    timeout_sec=1,
                )
                requirements = (
                    TerminalRequirement(
                        requirement_id="req-001",
                        description="create and verify an artifact",
                    ),
                )

                app.runtime._planner._validate_intent(
                    intent,
                    session,
                    finalization_mode=False,
                    reconciliation_mode=False,
                    required_requirements=requirements,
                )

                app.runtime._planner._validate_intent(
                    intent,
                    session,
                    finalization_mode=False,
                    reconciliation_mode=True,
                    required_requirements=requirements,
                )
            finally:
                app.close()

    async def test_historical_work_cannot_enable_direct_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="deadline-stale-work-generation",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(),
                proposal_capability=_DeadlineScriptedCapability(),
                policy=self._deadline_policy(),
            )
            try:
                app.journal._records = [
                    self._record(
                        app.journal.trial_id,
                        "historical-work",
                        TerminalCommandRole.WORK,
                    )
                ]
                app.journal._session = app.journal.snapshot().model_copy(
                    update={
                        "started_at": utc_now() - timedelta(seconds=581),
                        "committed_commands": 1,
                        "task_generation": 2,
                        "known_state_generation": None,
                        "successful_work_generation": None,
                    }
                )

                request = app.runtime._planner._turn_request(
                    self._state("repair and verify an artifact"),
                    app.journal.snapshot(),
                )

                self.assertFalse(request.verification_due)
                self.assertIs(
                    request.execution_limits.deadline_sequence,
                    TerminalDeadlineSequence.WORK_THEN_VERIFY,
                )
            finally:
                app.close()

    async def test_profiled_request_advertises_full_normal_work_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="deadline-profile-normal-work",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(),
                proposal_capability=_ProfiledDeadlineScriptedCapability(),
                policy=self._deadline_policy(),
            )
            try:
                request = app.runtime._planner._turn_request(
                    self._state("build a long-running artifact"),
                    app.journal.snapshot(),
                )

                self.assertFalse(request.finalization_mode)
                self.assertEqual(request.execution_limits.max_timeout_sec, 300)
                self.assertEqual(
                    request.execution_limits.max_work_timeout_sec,
                    300,
                )
                self.assertEqual(
                    request.execution_limits.max_inspection_timeout_sec,
                    60,
                )
                self.assertEqual(
                    request.execution_limits.max_inference_timeout_sec,
                    300.0,
                )
            finally:
                app.close()

    async def test_successful_work_prefers_direct_verify_before_finalization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="deadline-successful-work-direct-verify",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(),
                proposal_capability=_ProfiledDeadlineScriptedCapability(),
                policy=self._deadline_policy(),
            )
            try:
                app.journal._records = [
                    self._record(
                        app.journal.trial_id,
                        "successful-work",
                        TerminalCommandRole.WORK,
                    )
                ]
                app.journal._session = app.journal.snapshot().model_copy(
                    update={
                        "committed_commands": 1,
                        "task_generation": 1,
                        "known_state_generation": 1,
                        "successful_work_generation": 1,
                    }
                )

                request = app.runtime._planner._turn_request(
                    self._state("create and verify an artifact"),
                    app.journal.snapshot(),
                )

                self.assertFalse(request.finalization_mode)
                self.assertTrue(request.verification_due)
                self.assertIs(
                    request.execution_limits.deadline_sequence,
                    TerminalDeadlineSequence.DIRECT_VERIFY,
                )
                self.assertLessEqual(
                    request.execution_limits.max_inference_timeout_sec or 0.0,
                    _PROFILE.compact_inference.maximum_seconds,
                )
            finally:
                app.close()

    async def test_observed_failure_compacts_inference_but_preserves_work_cap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="deadline-failure-compact-recovery",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(),
                proposal_capability=_ProfiledDeadlineScriptedCapability(),
                policy=self._deadline_policy(),
            )
            try:
                app.journal._session = app.journal.snapshot().model_copy(
                    update={
                        "committed_commands": 1,
                        "task_generation": 1,
                        "known_state_generation": 1,
                        "latest_failure_signatures": (
                            "command returned non-zero status 1",
                        ),
                    }
                )

                request = app.runtime._planner._turn_request(
                    self._state("repair a failed artifact build"),
                    app.journal.snapshot(),
                )

                self.assertFalse(request.finalization_mode)
                self.assertTrue(request.recovery_mode)
                self.assertIsNone(request.execution_limits.deadline_sequence)
                self.assertLessEqual(
                    request.execution_limits.max_inference_timeout_sec or 0.0,
                    _PROFILE.compact_inference.maximum_seconds,
                )
                self.assertEqual(
                    request.execution_limits.max_work_timeout_sec,
                    300,
                )
            finally:
                app.close()

    @staticmethod
    def _deadline_policy() -> TerminalExecutionPolicy:
        return TerminalExecutionPolicy(
            max_wall_clock_seconds=840.0,
            max_no_progress_seconds=None,
        )

    @staticmethod
    def _state(description: str) -> AgentState:
        return AgentState(
            run_id=uuid4(),
            task=AgentTask(description=description),
            status=RunStatus.RUNNING,
        )

    @staticmethod
    def _record(
        trial_id: str,
        call_key: str,
        role: TerminalCommandRole,
    ) -> TerminalCommandRecord:
        return TerminalCommandRecord(
            action_id=uuid4(),
            invocation_id=uuid4(),
            decision_request_id=uuid4(),
            effect_fingerprint="0" * 64,
            intent=TerminalCommandIntent(
                trial_id=trial_id,
                call_key=call_key,
                command="true",
                command_role=role,
                timeout_sec=30,
            ),
            result=completed_result(),
            governance_status="applied",
        )


if __name__ == "__main__":
    unittest.main()
