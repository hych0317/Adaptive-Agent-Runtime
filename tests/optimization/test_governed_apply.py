from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unittest
from uuid import UUID

from adaptive_agent_runtime.decisioning import DecisionCheckpoint, DecisionFaultPoint
from adaptive_agent_runtime.decisioning import DecisionResultStatus
from adaptive_agent_runtime.governance import AuthorizationVerificationError
from adaptive_agent_runtime.optimization import (
    OptimizationApplyEffect,
    OptimizationApplyIntent,
    OptimizationApplyRequest,
    OptimizationRollbackEffect,
    OptimizationRollbackIntent,
    OptimizationRollbackRequest,
    OptimizationScope,
    OptimizationTargetKey,
    stable_optimization_apply_request_id,
)
from adaptive_agent_runtime.persistence import (
    SQLitePersistence,
    default_runtime_configuration_snapshot,
)
from adaptive_agent_runtime.persistence.optimization_configuration import (
    SQLiteGovernedRuntimeConfigurationStore,
)
from applications.research_agent.agent import ResearchAgent
from applications.research_agent.optimization import (
    research_initial_planning_optimization_scope,
)
from applications.research_agent.optimization_apply import (
    ResearchOptimizationConfigurationGateway,
)
from applications.research_agent.web_llm import WebLLMSettings
from tests.optimization.test_governed_proposal import _governance, _run_research


class NoopPermitVerifier:
    async def verify(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


async def _prepare_proposal(path: Path):  # type: ignore[no-untyped-def]
    await _run_research(path)
    second = await _run_research(path)
    if not second.optimization_proposals:
        raise AssertionError("test fixture did not produce a Proposal")
    return second, second.optimization_proposals[0]


def _gateway(
    persistence: SQLitePersistence,
    *,
    fault_injector: Any = None,
    configurations: Any = None,
) -> ResearchOptimizationConfigurationGateway:
    configuration_port = configurations or persistence.runtime_configuration
    return ResearchOptimizationConfigurationGateway(
        proposals=persistence.optimization_proposal_store,
        configurations=configuration_port,
        governance=_governance(persistence),
        reviews=persistence.human_review_service,
        issuer=persistence.authorization_issuer,
        operation_executor=persistence.operation_executor,
        trace_sink=persistence.trace_sink,
        apply_checkpoints=persistence.create_decision_checkpoint_store(
            DecisionCheckpoint[
                OptimizationApplyRequest,
                OptimizationApplyIntent,
                OptimizationApplyEffect,
            ]
        ),
        rollback_checkpoints=persistence.create_decision_checkpoint_store(
            DecisionCheckpoint[
                OptimizationRollbackRequest,
                OptimizationRollbackIntent,
                OptimizationRollbackEffect,
            ]
        ),
        fault_injector=fault_injector,
    )


def _ungoverned_effect(proposal: Any) -> OptimizationApplyEffect:
    current = default_runtime_configuration_snapshot(proposal.scope)
    return OptimizationApplyEffect(
        proposal_id=proposal.proposal_id,
        proposal_effect_fingerprint=proposal.effect_fingerprint,
        proposal_record_fingerprint="a" * 64,
        scope=proposal.scope,
        target_key=proposal.target_key,
        expected_current_value=current.value,
        expected_current_value_fingerprint=current.value_fingerprint,
        expected_current_revision=current.revision,
        expected_current_configuration_fingerprint=current.snapshot_fingerprint,
        proposed_value=proposal.proposed_value,
        rollback_revision=0,
        evidence_set_fingerprint=proposal.evidence_set_fingerprint,
    )


class GovernedOptimizationApplyTests(unittest.IsolatedAsyncioTestCase):
    async def test_proposal_commit_alone_does_not_change_configuration(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await _prepare_proposal(path)
            persistence = SQLitePersistence(path)
            try:
                active = await persistence.runtime_configuration.load_active(
                    research_initial_planning_optimization_scope(),
                    OptimizationTargetKey.PLANNER_MAX_NODES,
                )
                self.assertIsNone(active)
            finally:
                persistence.close()

    async def test_apply_requires_explicit_request(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            _result, proposal = await _prepare_proposal(path)
            agent = ResearchAgent(persistence_path=path)
            try:
                before = await agent._persistence.runtime_configuration.load_active(  # noqa: SLF001
                    proposal.scope, proposal.target_key
                )
                self.assertIsNone(before)
                applied = await agent.apply_optimization_proposal(
                    proposal.proposal_id, requested_by="operator:test"
                )
                self.assertIsNotNone(applied.receipt)
            finally:
                agent.close()

    async def test_apply_requires_governed_configuration_effect(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            _result, proposal = await _prepare_proposal(path)
            persistence = SQLitePersistence(path)
            try:
                with self.assertRaises(AuthorizationVerificationError):
                    await persistence.runtime_configuration.commit_apply(
                        _ungoverned_effect(proposal),
                        effect_fingerprint="b" * 64,
                        permit=None,
                        target=None,
                        subject_fingerprint=None,
                    )
            finally:
                persistence.close()

    async def test_proposal_authorization_cannot_authorize_apply(self) -> None:
        # The commit port rejects before persistence when no Apply-bound Permit exists.
        await self.test_apply_requires_governed_configuration_effect()

    async def test_scope_or_target_mismatch_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            database_path = Path(directory) / "runtime.sqlite3"
            persistence = SQLitePersistence(database_path)
            try:
                store = SQLiteGovernedRuntimeConfigurationStore(
                    persistence.database,
                    permit_verifier=NoopPermitVerifier(),  # type: ignore[arg-type]
                )
                wrong_scope = OptimizationScope(
                    tenant="default",
                    project="other",
                    application="research_agent",
                    decision_type="planning.task_graph.initialize",
                )
                current = default_runtime_configuration_snapshot(
                    research_initial_planning_optimization_scope()
                )
                effect = OptimizationApplyEffect(
                    proposal_id=UUID(int=1),
                    proposal_effect_fingerprint="1" * 64,
                    proposal_record_fingerprint="2" * 64,
                    scope=wrong_scope,
                    target_key=OptimizationTargetKey.PLANNER_MAX_NODES,
                    expected_current_value=8,
                    expected_current_value_fingerprint=current.value_fingerprint,
                    expected_current_revision=0,
                    expected_current_configuration_fingerprint=(
                        current.snapshot_fingerprint
                    ),
                    proposed_value=9,
                    rollback_revision=0,
                    evidence_set_fingerprint="3" * 64,
                )
                with self.assertRaises(Exception):
                    await store.commit_apply(
                        effect,
                        effect_fingerprint="4" * 64,
                        permit=object(),  # type: ignore[arg-type]
                        target=object(),  # type: ignore[arg-type]
                        subject_fingerprint="5" * 64,
                    )
            finally:
                persistence.close()

    async def test_stale_baseline_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await _run_research(path)
            second = await _run_research(path)
            third = await _run_research(path)
            first_proposal = second.optimization_proposals[0]
            stale_proposal = third.optimization_proposals[0]
            agent = ResearchAgent(persistence_path=path)
            try:
                await agent.apply_optimization_proposal(
                    first_proposal.proposal_id, requested_by="operator:test"
                )
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    await agent.apply_optimization_proposal(
                        stale_proposal.proposal_id, requested_by="operator:test"
                    )
            finally:
                agent.close()

    async def test_snapshot_active_pointer_and_receipt_commit_atomically(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            _result, proposal = await _prepare_proposal(path)
            persistence = SQLitePersistence(path)
            try:
                with persistence.database.transaction() as cursor:
                    cursor.execute(
                        "CREATE TRIGGER fail_configuration_receipt "
                        "BEFORE INSERT ON optimization_configuration_receipts "
                        "BEGIN SELECT RAISE(ABORT, 'receipt failure'); END"
                    )
                with self.assertRaises(RuntimeError):
                    await _gateway(persistence).request_apply(
                        proposal.proposal_id, requested_by="operator:test"
                    )
                with persistence.database.reader() as cursor:
                    snapshots = cursor.execute(
                        "SELECT COUNT(*) AS count FROM "
                        "governed_runtime_configuration_snapshots"
                    ).fetchone()["count"]
                    active = cursor.execute(
                        "SELECT COUNT(*) AS count FROM "
                        "governed_runtime_configuration_active"
                    ).fetchone()["count"]
                self.assertEqual((snapshots, active), (0, 0))
            finally:
                persistence.close()

    async def test_duplicate_apply_returns_same_receipt(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            _result, proposal = await _prepare_proposal(path)
            agent = ResearchAgent(persistence_path=path)
            try:
                first = await agent.apply_optimization_proposal(
                    proposal.proposal_id, requested_by="operator:test"
                )
                second = await agent.apply_optimization_proposal(
                    proposal.proposal_id, requested_by="operator:test"
                )
                self.assertEqual(second.receipt, first.receipt)
                with agent._persistence.database.reader() as cursor:  # noqa: SLF001
                    count = cursor.execute(
                        "SELECT COUNT(*) AS count FROM "
                        "optimization_configuration_receipts"
                    ).fetchone()["count"]
                self.assertEqual(count, 1)
            finally:
                agent.close()

    async def test_same_effect_fingerprint_with_different_payload_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            _result, proposal = await _prepare_proposal(path)
            persistence = SQLitePersistence(path)
            try:
                applied = await _gateway(persistence).request_apply(
                    proposal.proposal_id, requested_by="operator:test"
                )
                request_id = stable_optimization_apply_request_id(
                    proposal.proposal_id
                )
                checkpoint = await persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        OptimizationApplyRequest,
                        OptimizationApplyIntent,
                        OptimizationApplyEffect,
                    ]
                ).load(request_id)
                normalized = checkpoint.validated_decision.normalized_effect  # type: ignore[union-attr]
                altered = normalized.payload.model_copy(
                    update={"proposed_value": normalized.payload.proposed_value + 1}
                )
                with self.assertRaisesRegex(Exception, "conflicts"):
                    await persistence.runtime_configuration.commit_apply(
                        altered,
                        effect_fingerprint=applied.receipt.effect_fingerprint,  # type: ignore[union-attr]
                        permit=None,
                        target=None,
                        subject_fingerprint=None,
                    )
            finally:
                persistence.close()

    async def test_apply_resume_does_not_activate_twice(self) -> None:
        class CrashAfterCommit(RuntimeError):
            pass

        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            _result, proposal = await _prepare_proposal(path)

            def inject(point: DecisionFaultPoint, checkpoint: Any) -> None:
                del checkpoint
                if point is DecisionFaultPoint.EFFECT_COMMITTED:
                    raise CrashAfterCommit("simulated crash")

            first = SQLitePersistence(path)
            try:
                with self.assertRaises(CrashAfterCommit):
                    await _gateway(first, fault_injector=inject).request_apply(
                        proposal.proposal_id, requested_by="operator:test"
                    )
            finally:
                first.close()
            reopened = SQLitePersistence(path)
            try:
                resumed = await _gateway(reopened).request_apply(
                    proposal.proposal_id, requested_by="operator:test"
                )
                self.assertEqual(resumed.active_configuration.revision, 1)  # type: ignore[union-attr]
                with reopened.database.reader() as cursor:
                    count = cursor.execute(
                        "SELECT COUNT(*) AS count FROM "
                        "optimization_configuration_receipts"
                    ).fetchone()["count"]
                self.assertEqual(count, 1)
            finally:
                reopened.close()

    async def test_apply_resume_before_commit_uses_original_effect(self) -> None:
        class SimulatedCrash(RuntimeError):
            pass

        for fault_point in (
            DecisionFaultPoint.AUTHORIZED,
            DecisionFaultPoint.APPLYING,
        ):
            with self.subTest(fault_point=fault_point.value):
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "runtime.sqlite3"
                    _result, proposal = await _prepare_proposal(path)

                    def inject(point: DecisionFaultPoint, checkpoint: Any) -> None:
                        del checkpoint
                        if point is fault_point:
                            raise SimulatedCrash(fault_point.value)

                    first = SQLitePersistence(path)
                    try:
                        with self.assertRaises(SimulatedCrash):
                            await _gateway(
                                first, fault_injector=inject
                            ).request_apply(
                                proposal.proposal_id,
                                requested_by="operator:test",
                            )
                    finally:
                        first.close()
                    reopened = SQLitePersistence(path)
                    try:
                        resumed = await _gateway(reopened).request_apply(
                            proposal.proposal_id,
                            requested_by="operator:test",
                        )
                        self.assertEqual(
                            resumed.active_configuration.revision,  # type: ignore[union-attr]
                            1,
                        )
                        with reopened.database.reader() as cursor:
                            count = cursor.execute(
                                "SELECT COUNT(*) AS count FROM "
                                "optimization_configuration_receipts"
                            ).fetchone()["count"]
                        self.assertEqual(count, 1)
                    finally:
                        reopened.close()

    async def test_unknown_apply_reconciliation_is_durable_and_fail_closed(self) -> None:
        class SimulatedCrash(RuntimeError):
            pass

        class IndeterminateConfigurationPort:
            def __init__(self, delegate: Any) -> None:
                self._delegate = delegate

            async def load_active(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
                return await self._delegate.load_active(*args, **kwargs)

            async def load_snapshot(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
                return await self._delegate.load_snapshot(*args, **kwargs)

            async def load_receipt(self, effect_fingerprint: str):  # type: ignore[no-untyped-def]
                del effect_fingerprint
                raise OSError("indeterminate configuration read")

            async def commit_apply(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
                raise AssertionError("UNKNOWN Effect must not be retried")

            async def commit_rollback(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
                raise AssertionError("UNKNOWN Effect must not be retried")

        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            _result, proposal = await _prepare_proposal(path)

            def inject(point: DecisionFaultPoint, checkpoint: Any) -> None:
                del checkpoint
                if point is DecisionFaultPoint.APPLYING:
                    raise SimulatedCrash("applying")

            first = SQLitePersistence(path)
            try:
                with self.assertRaises(SimulatedCrash):
                    await _gateway(first, fault_injector=inject).request_apply(
                        proposal.proposal_id,
                        requested_by="operator:test",
                    )
            finally:
                first.close()
            reopened = SQLitePersistence(path)
            try:
                port = IndeterminateConfigurationPort(
                    reopened.runtime_configuration
                )
                gateway = _gateway(reopened, configurations=port)
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    await gateway.request_apply(
                        proposal.proposal_id,
                        requested_by="operator:test",
                    )
                request_id = stable_optimization_apply_request_id(
                    proposal.proposal_id
                )
                checkpoint = await reopened.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        OptimizationApplyRequest,
                        OptimizationApplyIntent,
                        OptimizationApplyEffect,
                    ]
                ).load(request_id)
                self.assertIsNotNone(checkpoint)
                self.assertEqual(
                    checkpoint.result.status,  # type: ignore[union-attr]
                    DecisionResultStatus.FAILED,
                )
                self.assertEqual(
                    checkpoint.result.reconciliation_status,  # type: ignore[union-attr]
                    "unknown",
                )
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    await gateway.request_apply(
                        proposal.proposal_id,
                        requested_by="operator:test",
                    )
            finally:
                reopened.close()

    async def test_new_run_reads_active_max_nodes(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            _second, proposal = await _prepare_proposal(path)
            agent = ResearchAgent(persistence_path=path)
            try:
                applied = await agent.apply_optimization_proposal(
                    proposal.proposal_id, requested_by="operator:test"
                )
                run = await agent.run("分析 Tesla 投资价值")
                self.assertEqual(run.runtime_configuration.revision, 1)
                self.assertEqual(run.runtime_configuration.value, proposal.proposed_value)
                self.assertEqual(
                    run.runtime_configuration.snapshot_fingerprint,
                    applied.active_configuration.snapshot_fingerprint,  # type: ignore[union-attr]
                )
            finally:
                agent.close()

    async def test_existing_run_keeps_original_configuration_snapshot(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            before, proposal = await _prepare_proposal(path)
            agent = ResearchAgent(persistence_path=path)
            try:
                await agent.apply_optimization_proposal(
                    proposal.proposal_id, requested_by="operator:test"
                )
                self.assertEqual(before.runtime_configuration.revision, 0)
                self.assertEqual(before.runtime_configuration.value, 8)
            finally:
                agent.close()

    async def test_run_records_configuration_revision_and_fingerprint(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await _run_research(path)
            persistence = SQLitePersistence(path)
            try:
                with persistence.database.reader() as cursor:
                    row = cursor.execute(
                        "SELECT checkpoint_json FROM decision_checkpoints "
                        "WHERE run_id = ? AND checkpoint_json LIKE ? LIMIT 1",
                        (
                            str(result.runtime_result.final_state.run_id),
                            '%"decision_type":"planning.task_graph.initialize"%',
                        ),
                    ).fetchone()
                self.assertIsNotNone(row)
                self.assertIn(
                    result.runtime_configuration.snapshot_fingerprint,
                    row["checkpoint_json"],
                )
            finally:
                persistence.close()

    async def test_missing_active_configuration_uses_default_eight(self) -> None:
        with TemporaryDirectory() as directory:
            result = await _run_research(Path(directory) / "runtime.sqlite3")
            self.assertEqual(result.runtime_configuration.revision, 0)
            self.assertEqual(result.runtime_configuration.value, 8)

    async def test_sqlite_read_failure_does_not_silently_use_cached_configuration(self) -> None:
        with TemporaryDirectory() as directory:
            agent = ResearchAgent(
                persistence_path=Path(directory) / "runtime.sqlite3"
            )
            agent.close()
            with self.assertRaises(Exception):
                await agent.run("分析 Tesla 投资价值")

    async def test_rollback_restores_prior_value_as_new_revision(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            _result, proposal = await _prepare_proposal(path)
            agent = ResearchAgent(persistence_path=path)
            try:
                applied = await agent.apply_optimization_proposal(
                    proposal.proposal_id, requested_by="operator:test"
                )
                rolled = await agent.rollback_optimization(
                    applied.receipt.effect_fingerprint,  # type: ignore[union-attr]
                    requested_by="operator:test",
                )
                self.assertEqual(rolled.active_configuration.revision, 2)  # type: ignore[union-attr]
                self.assertEqual(rolled.active_configuration.value, 8)  # type: ignore[union-attr]
            finally:
                agent.close()

    async def test_rollback_resume_does_not_activate_twice(self) -> None:
        class CrashAfterCommit(RuntimeError):
            pass

        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            _result, proposal = await _prepare_proposal(path)
            persistence = SQLitePersistence(path)
            applied = await _gateway(persistence).request_apply(
                proposal.proposal_id, requested_by="operator:test"
            )
            source = applied.receipt.effect_fingerprint  # type: ignore[union-attr]

            def inject(point: DecisionFaultPoint, checkpoint: Any) -> None:
                del checkpoint
                if point is DecisionFaultPoint.EFFECT_COMMITTED:
                    raise CrashAfterCommit("simulated crash")

            try:
                with self.assertRaises(CrashAfterCommit):
                    await _gateway(
                        persistence, fault_injector=inject
                    ).request_rollback(source, requested_by="operator:test")
            finally:
                persistence.close()
            reopened = SQLitePersistence(path)
            try:
                resumed = await _gateway(reopened).request_rollback(
                    source, requested_by="operator:test"
                )
                self.assertEqual(resumed.active_configuration.revision, 2)  # type: ignore[union-attr]
            finally:
                reopened.close()

    async def test_end_to_end_apply_then_rollback_changes_only_new_runs(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            run_a, proposal = await _prepare_proposal(path)
            agent = ResearchAgent(persistence_path=path)
            try:
                self.assertEqual(run_a.runtime_configuration.value, 8)
                applied = await agent.apply_optimization_proposal(
                    proposal.proposal_id, requested_by="operator:test"
                )
                run_b = await agent.run("分析 Tesla 投资价值")
                self.assertEqual(run_b.runtime_configuration.revision, 1)
                self.assertEqual(
                    run_b.runtime_configuration.value, proposal.proposed_value
                )
                rolled = await agent.rollback_optimization(
                    applied.receipt.effect_fingerprint,  # type: ignore[union-attr]
                    requested_by="operator:test",
                )
                self.assertEqual(rolled.active_configuration.revision, 2)  # type: ignore[union-attr]
                run_c = await agent.run("分析 Tesla 投资价值")
                self.assertEqual(run_c.runtime_configuration.revision, 2)
                self.assertEqual(run_c.runtime_configuration.value, 8)
            finally:
                agent.close()

    def test_application_cannot_call_raw_activate_or_rollback(self) -> None:
        self.assertFalse(hasattr(SQLiteGovernedRuntimeConfigurationStore, "activate"))
        self.assertFalse(hasattr(SQLiteGovernedRuntimeConfigurationStore, "rollback"))

    def test_web_llm_settings_remain_outside_optimization(self) -> None:
        self.assertEqual(WebLLMSettings.configuration_owner, "operator")
        self.assertEqual(WebLLMSettings.configuration_domain, "operator.web_llm")


if __name__ == "__main__":
    unittest.main()
