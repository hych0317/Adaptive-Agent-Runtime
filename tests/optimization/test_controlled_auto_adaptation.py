from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unittest
from uuid import UUID, uuid4

from pydantic import ValidationError

from adaptive_agent_runtime import (
    AgentState,
    AgentTask,
    InMemoryStateStore,
    InMemoryTraceSink,
    RunStatus,
)
from adaptive_agent_runtime.decisioning import (
    DecisionCheckpoint,
    DecisionCheckpointConflictError,
    DecisionFaultPoint,
    DecisionReconciliationStatus,
    DecisionResultStatus,
    decision_fingerprint,
)
from adaptive_agent_runtime.governance import (
    AuthorizationUseStatus,
    ConfidenceAssessment,
    ConfidencePolicy,
    DecisionOutcome,
    DeterministicRuleEvaluator,
    GovernanceRequest,
    HumanReviewDecision,
    ReviewOutcome,
    RuntimeGovernanceEvaluator,
    default_governance_policy,
)
from adaptive_agent_runtime.optimization import (
    OPTIMIZATION_APPLY_COMMIT_OPERATION,
    OPTIMIZATION_APPLY_DECISION_TYPE,
    AutoAdaptationPolicy,
    AutoAdaptationSkipReason,
    AutoAdaptationStatus,
    AutoAdaptationTriggerRecord,
    OptimizationApplyCommitReceipt,
    OptimizationApplyEffect,
    OptimizationApplyIntent,
    OptimizationApplyRequest,
    OptimizationProposal,
    OptimizationProposalDraft,
    OptimizationProposalStatus,
    OptimizationRiskClassification,
    OptimizationScope,
    OptimizationTargetKey,
    OptimizationTargetType,
    RuntimeConfigurationActivationMode,
    RuntimeConfigurationSnapshot,
)
from adaptive_agent_runtime.persistence import (
    SQLitePersistence,
    default_runtime_configuration_snapshot,
)
from applications.research_agent.agent import ResearchAgent
from applications.research_agent.auto_adaptation import (
    ResearchAutoAdaptationCoordinator,
)
from applications.research_agent.optimization import (
    research_initial_planning_optimization_scope,
)
from applications.research_agent.optimization_apply import (
    ResearchOptimizationConfigurationGateway,
    ResearchOptimizationConfigurationResult,
)
from tests.optimization.test_governed_apply import _gateway, _prepare_proposal
from tests.optimization.test_governed_proposal import _run_research


ROOT = Path(__file__).resolve().parents[2]


class _ProposalQuery:
    def __init__(
        self,
        proposals: tuple[OptimizationProposal, ...],
        *,
        provenance_valid: bool = True,
    ) -> None:
        self.proposals = proposals
        self.provenance_valid = provenance_valid

    async def list_for_scope(
        self,
        scope: OptimizationScope,
    ) -> tuple[OptimizationProposal, ...]:
        del scope
        return self.proposals

    async def load_by_id(
        self,
        proposal_id: UUID,
    ) -> OptimizationProposal | None:
        return next(
            (item for item in self.proposals if item.proposal_id == proposal_id),
            None,
        )

    async def verify_phase3_provenance(self, proposal_id: UUID) -> bool:
        return self.provenance_valid and any(
            item.proposal_id == proposal_id for item in self.proposals
        )


class _ConfigurationQuery:
    def __init__(
        self,
        active: RuntimeConfigurationSnapshot | None = None,
        *,
        applied_proposals: frozenset[UUID] = frozenset(),
    ) -> None:
        self.active = active
        self.applied_proposals = applied_proposals

    async def load_active(
        self,
        scope: OptimizationScope,
        target_key: OptimizationTargetKey,
    ) -> RuntimeConfigurationSnapshot | None:
        del scope, target_key
        return self.active

    async def was_proposal_applied(self, proposal_id: UUID) -> bool:
        return proposal_id in self.applied_proposals


class _TriggerStore:
    def __init__(self) -> None:
        self.records: dict[UUID, AutoAdaptationTriggerRecord] = {}

    async def load_for_run(
        self,
        trigger_run_id: UUID,
    ) -> AutoAdaptationTriggerRecord | None:
        return self.records.get(trigger_run_id)

    async def claim(
        self,
        record: AutoAdaptationTriggerRecord,
    ) -> AutoAdaptationTriggerRecord:
        return self.records.setdefault(record.trigger_run_id, record)

    async def transition(
        self,
        record: AutoAdaptationTriggerRecord,
    ) -> AutoAdaptationTriggerRecord:
        current = self.records[record.trigger_run_id]
        if current.status is AutoAdaptationStatus.SELECTED:
            self.records[record.trigger_run_id] = record
        return self.records[record.trigger_run_id]


class _Gateway:
    def __init__(
        self,
        result: ResearchOptimizationConfigurationResult | None = None,
    ) -> None:
        self.result = result
        self.calls = 0

    async def request_policy_auto_apply(
        self,
        proposal_id: UUID,
        **kwargs: Any,
    ) -> ResearchOptimizationConfigurationResult:
        del proposal_id, kwargs
        self.calls += 1
        if self.result is None:
            raise AssertionError("ineligible Proposal reached the Apply gateway")
        return self.result


class _RaisingGateway:
    def __init__(self, message: str) -> None:
        self.message = message
        self.calls = 0

    async def request_policy_auto_apply(
        self,
        proposal_id: UUID,
        **kwargs: Any,
    ) -> ResearchOptimizationConfigurationResult:
        del proposal_id, kwargs
        self.calls += 1
        raise RuntimeError(self.message)


class _FailingTriggerStore:
    async def load_for_run(
        self,
        trigger_run_id: UUID,
    ) -> AutoAdaptationTriggerRecord | None:
        del trigger_run_id
        raise RuntimeError("trigger ledger unavailable")

    async def claim(
        self,
        record: AutoAdaptationTriggerRecord,
    ) -> AutoAdaptationTriggerRecord:
        del record
        raise RuntimeError("trigger ledger unavailable")

    async def transition(
        self,
        record: AutoAdaptationTriggerRecord,
    ) -> AutoAdaptationTriggerRecord:
        del record
        raise RuntimeError("trigger ledger unavailable")


class _CheckpointConflictGateway(_Gateway):
    async def request_policy_auto_apply(
        self,
        proposal_id: UUID,
        **kwargs: Any,
    ) -> ResearchOptimizationConfigurationResult:
        del proposal_id, kwargs
        self.calls += 1
        raise DecisionCheckpointConflictError(
            "another coordinator owns the stable Apply Decision"
        )


class _FixedConfidenceEvaluator:
    module_id = "test.auto_adaptation.fixed_confidence"

    def __init__(self, outcome: DecisionOutcome) -> None:
        self._outcome = outcome

    def evaluate(
        self,
        request: GovernanceRequest,
        policy: ConfidencePolicy,
    ) -> ConfidenceAssessment:
        del request, policy
        score = 0.1 if self._outcome is DecisionOutcome.DENY else 0.5
        return ConfidenceAssessment(
            score=score,
            outcome=self._outcome,
            evidence_score=1.0,
            history_score=0.5,
            safety_score=0.55,
            reason=f"fixed test outcome: {self._outcome.value}",
        )


def _scope() -> OptimizationScope:
    return research_initial_planning_optimization_scope()


def _default_configuration() -> RuntimeConfigurationSnapshot:
    return default_runtime_configuration_snapshot(_scope())


def _proposal(
    *,
    proposal_id: UUID | None = None,
) -> OptimizationProposal:
    current = _default_configuration()
    identity = proposal_id or uuid4()
    return OptimizationProposal(
        proposal_id=identity,
        source_decision_request_id=uuid4(),
        scope=_scope(),
        target_type=OptimizationTargetType.INITIAL_PLANNING_POLICY,
        target_key=OptimizationTargetKey.PLANNER_MAX_NODES,
        current_value=current.value,
        current_value_fingerprint=current.value_fingerprint,
        current_configuration_revision=current.revision,
        current_configuration_fingerprint=current.snapshot_fingerprint,
        configuration_source=current.configuration_source,
        proposed_value=9,
        supporting_learning_insight_refs=(uuid4(),),
        supporting_feedback_refs=(uuid4(),),
        supporting_experience_refs=(uuid4(),),
        supporting_evaluation_refs=(uuid4(),),
        applicable_conditions=("Same Initial Planning scope.",),
        expected_impact="Bound Initial Planning graph complexity.",
        limitations=("Association only; no causal claim.",),
        risk_classification=OptimizationRiskClassification.LOW,
        rollback_requirements=("Retain the exact prior revision.",),
        evidence_set_fingerprint=decision_fingerprint({"evidence": str(identity)}),
        effect_fingerprint=decision_fingerprint({"effect": str(identity)}),
    )


async def _evaluate_policy(
    proposals: tuple[OptimizationProposal, ...],
    *,
    enabled: bool = True,
    provenance_valid: bool = True,
    active: RuntimeConfigurationSnapshot | None = None,
    applied_proposals: frozenset[UUID] = frozenset(),
    gateway_result: ResearchOptimizationConfigurationResult | None = None,
) -> tuple[
    AutoAdaptationTriggerRecord,
    _Gateway,
    _ConfigurationQuery,
    InMemoryTraceSink,
]:
    run_id = uuid4()
    state_store = InMemoryStateStore()
    await state_store.save(
        AgentState(
            run_id=run_id,
            task=AgentTask(description="completed policy test run"),
            status=RunStatus.COMPLETED,
            revision=1,
        )
    )
    configurations = _ConfigurationQuery(
        active,
        applied_proposals=applied_proposals,
    )
    gateway = _Gateway(gateway_result)
    trace = InMemoryTraceSink()
    coordinator = ResearchAutoAdaptationCoordinator(
        policy=AutoAdaptationPolicy(enabled=enabled, scope=_scope()),
        proposals=_ProposalQuery(
            proposals,
            provenance_valid=provenance_valid,
        ),
        configurations=configurations,
        state_store=state_store,
        triggers=_TriggerStore(),
        gateway=gateway,  # type: ignore[arg-type]
        trace_sink=trace,
    )
    record = await coordinator.evaluate_after_run(
        trigger_run_id=run_id,
        run_configuration=active or _default_configuration(),
    )
    return record, gateway, configurations, trace


async def _run_auto_apply(path: Path):  # type: ignore[no-untyped-def]
    await _run_research(path)
    agent = ResearchAgent(
        persistence_path=path,
        auto_adaptation_enabled=True,
    )
    try:
        return await agent.run("分析 Tesla 投资价值")
    finally:
        agent.close()


async def _run_with_auto_gateway(
    path: Path,
    gateway: object,
):  # type: ignore[no-untyped-def]
    await _run_research(path)
    agent = ResearchAgent(
        persistence_path=path,
        auto_adaptation_enabled=True,
    )
    agent._build_optimization_configuration_gateway = (  # type: ignore[method-assign]
        lambda trace_sink: gateway
    )
    try:
        return await agent.run("分析 Tesla 投资价值")
    finally:
        agent.close()


async def _run_with_trigger_ledger_failure(
    path: Path,
):  # type: ignore[no-untyped-def]
    agent = ResearchAgent(
        persistence_path=path,
        auto_adaptation_enabled=True,
    )
    agent._persistence.auto_adaptation_trigger_store = (  # type: ignore[assignment]
        _FailingTriggerStore()
    )
    try:
        return await agent.run("分析 Tesla 投资价值")
    finally:
        agent.close()


def _configuration_counts(persistence: SQLitePersistence) -> tuple[int, int, int]:
    with persistence.database.reader() as cursor:
        snapshots = int(
            cursor.execute(
                "SELECT COUNT(*) AS count FROM "
                "governed_runtime_configuration_snapshots"
            ).fetchone()["count"]
        )
        receipts = int(
            cursor.execute(
                "SELECT COUNT(*) AS count FROM "
                "optimization_configuration_receipts"
            ).fetchone()["count"]
        )
        triggers = int(
            cursor.execute(
                "SELECT COUNT(*) AS count FROM auto_adaptation_triggers"
            ).fetchone()["count"]
        )
    return snapshots, receipts, triggers


def _governed_gateway(
    persistence: SQLitePersistence,
    outcome: DecisionOutcome,
) -> ResearchOptimizationConfigurationGateway:
    governance = RuntimeGovernanceEvaluator(
        policy=default_governance_policy(),
        rule_evaluator=DeterministicRuleEvaluator(),
        confidence_evaluator=_FixedConfidenceEvaluator(outcome),
        review_service=persistence.human_review_service,
    )
    return ResearchOptimizationConfigurationGateway(
        proposals=persistence.optimization_proposal_store,
        configurations=persistence.runtime_configuration,
        governance=governance,
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
    )


async def _save_completed_run(
    persistence: SQLitePersistence,
    run_id: UUID,
) -> None:
    task = AgentTask(description="completed governed auto-adaptation run")
    await persistence.state_store.save(
        AgentState(
            run_id=run_id,
            task=task,
            revision=0,
        )
    )
    await persistence.state_store.save(
        AgentState(
            run_id=run_id,
            task=task,
            status=RunStatus.COMPLETED,
            revision=1,
        )
    )


def _real_coordinator(
    persistence: SQLitePersistence,
    *,
    gateway: ResearchOptimizationConfigurationGateway | None = None,
) -> ResearchAutoAdaptationCoordinator:
    return ResearchAutoAdaptationCoordinator(
        policy=AutoAdaptationPolicy(enabled=True, scope=_scope()),
        proposals=persistence.optimization_proposal_store,
        configurations=persistence.runtime_configuration,
        state_store=persistence.state_store,
        triggers=persistence.auto_adaptation_trigger_store,
        gateway=gateway or _gateway(persistence),
        trace_sink=persistence.trace_sink,
    )


class ControlledAutoAdaptationTests(unittest.IsolatedAsyncioTestCase):
    async def test_auto_adaptation_is_disabled_by_default(self) -> None:
        parameter = inspect.signature(ResearchAgent).parameters[
            "auto_adaptation_enabled"
        ]
        self.assertIs(parameter.default, False)
        with TemporaryDirectory() as directory:
            result = await _run_research(Path(directory) / "runtime.sqlite3")
        self.assertEqual(result.auto_adaptation.status, AutoAdaptationStatus.SKIPPED)
        self.assertEqual(
            result.auto_adaptation.skip_reason,
            AutoAdaptationSkipReason.DISABLED,
        )

    async def test_disabled_auto_adaptation_never_applies_proposal(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result, proposal = await _prepare_proposal(path)
            self.assertEqual(result.optimization_proposals, (proposal,))
            self.assertEqual(
                result.auto_adaptation.skip_reason,
                AutoAdaptationSkipReason.DISABLED,
            )
            persistence = SQLitePersistence(path)
            try:
                active = await persistence.runtime_configuration.load_active(
                    proposal.scope,
                    proposal.target_key,
                )
                self.assertIsNone(active)
                self.assertEqual(_configuration_counts(persistence)[1], 0)
            finally:
                persistence.close()

    async def test_single_eligible_low_risk_proposal_is_applied_once(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await _run_auto_apply(path)
            self.assertEqual(result.auto_adaptation.status, AutoAdaptationStatus.APPLIED)
            self.assertEqual(result.auto_adaptation.active_revision, 1)
            self.assertEqual(len(result.optimization_proposals), 1)
            persistence = SQLitePersistence(path)
            try:
                active = await persistence.runtime_configuration.load_active(
                    _scope(),
                    OptimizationTargetKey.PLANNER_MAX_NODES,
                )
                self.assertIsNotNone(active)
                self.assertEqual(active.revision, 1)  # type: ignore[union-attr]
                self.assertEqual(active.value, 9)  # type: ignore[union-attr]
                self.assertEqual(_configuration_counts(persistence)[1], 1)
            finally:
                persistence.close()

    async def test_auto_apply_uses_existing_governed_apply_lifecycle(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await _run_auto_apply(path)
            request_id = result.auto_adaptation.apply_request_id
            self.assertIsNotNone(request_id)
            persistence = SQLitePersistence(path)
            try:
                checkpoint = await persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        OptimizationApplyRequest,
                        OptimizationApplyIntent,
                        OptimizationApplyEffect,
                    ]
                ).load(request_id)  # type: ignore[arg-type]
                self.assertIsNotNone(checkpoint)
                assert checkpoint is not None
                self.assertEqual(
                    checkpoint.request.decision_type,
                    OPTIMIZATION_APPLY_DECISION_TYPE,
                )
                self.assertEqual(
                    checkpoint.proposal.producer.capability,  # type: ignore[union-attr]
                    "deterministic_runtime_policy",
                )
                self.assertEqual(
                    checkpoint.proposal.producer.producer_id,  # type: ignore[union-attr]
                    "runtime.policy.auto_adaptation",
                )
                normalized = checkpoint.validated_decision.normalized_effect  # type: ignore[union-attr]
                self.assertEqual(normalized.operation, OPTIMIZATION_APPLY_COMMIT_OPERATION)
                self.assertEqual(
                    normalized.payload.trigger_mode,
                    RuntimeConfigurationActivationMode.POLICY_AUTO,
                )
                self.assertEqual(
                    checkpoint.result.status,  # type: ignore[union-attr]
                    DecisionResultStatus.APPLIED,
                )
                authorization_id = checkpoint.governance_receipt.authorization_id  # type: ignore[union-attr]
                authorization_use = await persistence.authorization_store.load(
                    authorization_id  # type: ignore[arg-type]
                )
                self.assertIsNotNone(authorization_use)
                self.assertEqual(
                    authorization_use.status,  # type: ignore[union-attr]
                    AuthorizationUseStatus.APPLIED,
                )
                receipt = await persistence.runtime_configuration.load_receipt(
                    normalized.effect_fingerprint
                )
                self.assertIsNotNone(receipt)
                self.assertEqual(
                    receipt.effect_fingerprint,  # type: ignore[union-attr]
                    result.auto_adaptation.apply_effect_fingerprint,
                )
                entries = await persistence.trace_sink.entries_for(
                    result.runtime_result.final_state.run_id
                )
                decision_kinds = tuple(
                    entry.event.kind
                    for entry in entries
                    if (
                        entry.event.payload.get("correlation", {}).get("request_id")
                        == str(request_id)
                    )
                )
                self.assertEqual(
                    decision_kinds,
                    (
                        "decision.requested",
                        "decision.context_projected",
                        "decision.proposed",
                        "decision.validation_passed",
                        "decision.governance_requested",
                        "decision.authorized",
                        "decision.apply_started",
                        "decision.applied",
                    ),
                )
            finally:
                persistence.close()

    async def test_auto_selection_has_no_agent_or_ranking_dependency(self) -> None:
        constructor_inputs = set(
            inspect.signature(ResearchAutoAdaptationCoordinator).parameters
        )
        self.assertFalse(
            constructor_inputs
            & {"agent", "llm", "model", "capability", "ranker", "reward_model"}
        )
        source_path = (
            ROOT / "applications" / "research_agent" / "auto_adaptation.py"
        )
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertFalse(
            any(module.startswith("adaptive_agent_runtime.llm") for module in imported_modules)
        )
        self.assertNotIn("assess_optimization(", source)
        self.assertNotIn("rank_proposal", source)

        denied = ResearchOptimizationConfigurationResult(
            request_id=uuid4(),
            receipt=None,
            active_configuration=None,
            governance_record=None,
            decision_status=DecisionResultStatus.REJECTED,
            reason="stop after deterministic selection",
        )
        record, gateway, _configurations, _trace = await _evaluate_policy(
            (_proposal(),),
            gateway_result=denied,
        )
        self.assertEqual(record.status, AutoAdaptationStatus.DENIED)
        self.assertEqual(gateway.calls, 1)

    async def test_auto_apply_requires_complete_phase3_provenance(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            _result, proposal = await _prepare_proposal(path)
            persistence = SQLitePersistence(path)
            try:
                with persistence.database.transaction() as cursor:
                    cursor.execute(
                        "DELETE FROM learning_insights WHERE learning_insight_id = ?",
                        (str(proposal.supporting_learning_insight_refs[0]),),
                    )
                trigger_run_id = uuid4()
                await _save_completed_run(persistence, trigger_run_id)
                record = await _real_coordinator(
                    persistence
                ).evaluate_after_run(
                    trigger_run_id=trigger_run_id,
                    run_configuration=_default_configuration(),
                )
                self.assertEqual(record.status, AutoAdaptationStatus.SKIPPED)
                self.assertEqual(
                    record.skip_reason,
                    AutoAdaptationSkipReason.INCOMPLETE_PHASE3_PROVENANCE,
                )
                self.assertIsNone(record.apply_request_id)
                self.assertIsNone(
                    await persistence.runtime_configuration.load_active(
                        _scope(),
                        OptimizationTargetKey.PLANNER_MAX_NODES,
                    )
                )
                self.assertEqual(_configuration_counts(persistence)[1], 0)
                entries = await persistence.trace_sink.entries_for(trigger_run_id)
                self.assertEqual(
                    entries[-1].event.kind,
                    "optimization.auto_adaptation.skipped",
                )
            finally:
                persistence.close()

    async def test_counterevidence_prevents_auto_apply(self) -> None:
        proposal = _proposal().model_copy(
            update={"counterevidence_refs": ("feedback:unresolved",)}
        )
        record, gateway, _configurations, _trace = await _evaluate_policy(
            (proposal,)
        )
        self.assertEqual(
            record.skip_reason,
            AutoAdaptationSkipReason.COUNTEREVIDENCE_PRESENT,
        )
        self.assertEqual(gateway.calls, 0)

    async def test_multiple_eligible_proposals_result_in_noop(self) -> None:
        record, gateway, _configurations, _trace = await _evaluate_policy(
            (_proposal(), _proposal())
        )
        self.assertEqual(record.status, AutoAdaptationStatus.SKIPPED)
        self.assertEqual(
            record.skip_reason,
            AutoAdaptationSkipReason.MULTIPLE_ELIGIBLE_PROPOSALS,
        )
        self.assertEqual(gateway.calls, 0)

    async def test_non_low_risk_proposal_is_not_auto_applied(self) -> None:
        proposal = _proposal().model_copy(
            update={
                "risk_classification": OptimizationRiskClassification.MEDIUM
            }
        )
        record, gateway, _configurations, _trace = await _evaluate_policy(
            (proposal,)
        )
        self.assertEqual(record.skip_reason, AutoAdaptationSkipReason.RISK_NOT_LOW)
        self.assertEqual(gateway.calls, 0)

    async def test_change_larger_than_one_is_not_auto_applied(self) -> None:
        proposal = _proposal().model_copy(update={"proposed_value": 10})
        record, gateway, _configurations, _trace = await _evaluate_policy(
            (proposal,)
        )
        self.assertEqual(record.skip_reason, AutoAdaptationSkipReason.CHANGE_NOT_ONE)
        self.assertEqual(gateway.calls, 0)

    async def test_stale_baseline_is_not_auto_applied(self) -> None:
        proposal = _proposal().model_copy(
            update={"current_configuration_fingerprint": "f" * 64}
        )
        record, gateway, _configurations, _trace = await _evaluate_policy(
            (proposal,)
        )
        self.assertEqual(record.skip_reason, AutoAdaptationSkipReason.BASELINE_STALE)
        self.assertEqual(gateway.calls, 0)

    async def test_governance_denial_leaves_configuration_unchanged(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await _prepare_proposal(path)
            persistence = SQLitePersistence(path)
            try:
                trigger_run_id = uuid4()
                await _save_completed_run(persistence, trigger_run_id)
                record = await _real_coordinator(
                    persistence,
                    gateway=_governed_gateway(
                        persistence,
                        DecisionOutcome.DENY,
                    ),
                ).evaluate_after_run(
                    trigger_run_id=trigger_run_id,
                    run_configuration=_default_configuration(),
                )
                self.assertEqual(record.status, AutoAdaptationStatus.DENIED)
                self.assertEqual(
                    record.skip_reason,
                    AutoAdaptationSkipReason.GOVERNANCE_DENIED,
                )
                checkpoint = await persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        OptimizationApplyRequest,
                        OptimizationApplyIntent,
                        OptimizationApplyEffect,
                    ]
                ).load(record.apply_request_id)  # type: ignore[arg-type]
                self.assertIsNotNone(checkpoint)
                self.assertEqual(
                    checkpoint.result.status,  # type: ignore[union-attr]
                    DecisionResultStatus.REJECTED,
                )
                self.assertEqual(
                    checkpoint.governance_receipt.outcome.value,  # type: ignore[union-attr]
                    "deny",
                )
                self.assertIsNone(
                    checkpoint.governance_receipt.authorization_id  # type: ignore[union-attr]
                )
                self.assertIsNone(
                    await persistence.runtime_configuration.load_active(
                        _scope(),
                        OptimizationTargetKey.PLANNER_MAX_NODES,
                    )
                )
                self.assertEqual(_configuration_counts(persistence)[1], 0)
            finally:
                persistence.close()

    async def test_review_pending_leaves_configuration_unchanged(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await _prepare_proposal(path)
            persistence = SQLitePersistence(path)
            try:
                trigger_run_id = uuid4()
                await _save_completed_run(persistence, trigger_run_id)
                record = await _real_coordinator(
                    persistence,
                    gateway=_governed_gateway(
                        persistence,
                        DecisionOutcome.REVIEW_REQUIRED,
                    ),
                ).evaluate_after_run(
                    trigger_run_id=trigger_run_id,
                    run_configuration=_default_configuration(),
                )
                self.assertEqual(
                    record.status,
                    AutoAdaptationStatus.REVIEW_PENDING,
                )
                self.assertEqual(
                    record.skip_reason,
                    AutoAdaptationSkipReason.REVIEW_PENDING,
                )
                checkpoint = await persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        OptimizationApplyRequest,
                        OptimizationApplyIntent,
                        OptimizationApplyEffect,
                    ]
                ).load(record.apply_request_id)  # type: ignore[arg-type]
                self.assertIsNotNone(checkpoint)
                self.assertEqual(checkpoint.stage.value, "review_pending")  # type: ignore[union-attr]
                self.assertIsNone(checkpoint.result)  # type: ignore[union-attr]
                review_id = checkpoint.governance_receipt.review_request_id  # type: ignore[union-attr]
                review = persistence.human_review_service.get(review_id)  # type: ignore[arg-type]
                self.assertIsNotNone(review)
                self.assertEqual(review.status.value, "pending")  # type: ignore[union-attr]
                self.assertIsNone(
                    await persistence.runtime_configuration.load_active(
                        _scope(),
                        OptimizationTargetKey.PLANNER_MAX_NODES,
                    )
                )
                self.assertEqual(_configuration_counts(persistence)[1], 0)
            finally:
                persistence.close()

    async def test_review_pending_resumes_original_decision_after_approval(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await _prepare_proposal(path)
            persistence = SQLitePersistence(path)
            try:
                trigger_run_id = uuid4()
                await _save_completed_run(persistence, trigger_run_id)
                coordinator = _real_coordinator(
                    persistence,
                    gateway=_governed_gateway(
                        persistence,
                        DecisionOutcome.REVIEW_REQUIRED,
                    ),
                )
                pending = await coordinator.evaluate_after_run(
                    trigger_run_id=trigger_run_id,
                    run_configuration=_default_configuration(),
                )
                checkpoint_store = persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        OptimizationApplyRequest,
                        OptimizationApplyIntent,
                        OptimizationApplyEffect,
                    ]
                )
                checkpoint = await checkpoint_store.load(
                    pending.apply_request_id  # type: ignore[arg-type]
                )
                self.assertIsNotNone(checkpoint)
                review_id = checkpoint.governance_receipt.review_request_id  # type: ignore[union-attr]
                self.assertIsNotNone(review_id)
                persistence.human_review_service.resolve(
                    review_id,  # type: ignore[arg-type]
                    HumanReviewDecision(
                        outcome=ReviewOutcome.APPROVE,
                        reviewer_id="operator:phase4c-test",
                        rationale="The bounded one-node change is approved.",
                        decided_at=datetime.now(timezone.utc),
                    ),
                )

                resumed = await coordinator.evaluate_after_run(
                    trigger_run_id=trigger_run_id,
                    run_configuration=_default_configuration(),
                )
                self.assertEqual(resumed.status, AutoAdaptationStatus.APPLIED)
                self.assertEqual(resumed.active_revision, 1)
                self.assertEqual(_configuration_counts(persistence)[1], 1)
                active = await persistence.runtime_configuration.load_active(
                    _scope(),
                    OptimizationTargetKey.PLANNER_MAX_NODES,
                )
                self.assertIsNotNone(active)
                self.assertEqual(active.revision, 1)  # type: ignore[union-attr]
                self.assertEqual(active.value, 9)  # type: ignore[union-attr]
            finally:
                persistence.close()

    async def test_current_run_keeps_original_revision(self) -> None:
        with TemporaryDirectory() as directory:
            result = await _run_auto_apply(Path(directory) / "runtime.sqlite3")
        self.assertEqual(result.runtime_configuration.revision, 0)
        self.assertEqual(result.runtime_configuration.value, 8)
        self.assertEqual(result.auto_adaptation.active_revision, 1)

    async def test_next_run_reads_auto_applied_revision(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            applied_run = await _run_auto_apply(path)
            agent = ResearchAgent(
                persistence_path=path,
                auto_adaptation_enabled=False,
            )
            try:
                next_run = await agent.run("分析 Tesla 投资价值")
            finally:
                agent.close()
        snapshot = next_run.runtime_configuration
        self.assertEqual(snapshot.revision, 1)
        self.assertEqual(snapshot.value, 9)
        self.assertEqual(
            snapshot.effective_activation_mode,
            RuntimeConfigurationActivationMode.POLICY_AUTO,
        )
        self.assertEqual(
            snapshot.source_proposal_id,
            applied_run.optimization_proposals[0].proposal_id,
        )
        self.assertEqual(
            snapshot.trigger_run_id,
            applied_run.runtime_result.final_state.run_id,
        )
        self.assertIsNotNone(snapshot.auto_adaptation_policy_fingerprint)

    async def test_auto_apply_resume_does_not_activate_twice(self) -> None:
        class SimulatedCrash(BaseException):
            pass

        for fault_point in (
            DecisionFaultPoint.AUTHORIZED,
            DecisionFaultPoint.APPLYING,
            DecisionFaultPoint.EFFECT_COMMITTED,
        ):
            with self.subTest(fault_point=fault_point.value):
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "runtime.sqlite3"
                    await _run_research(path)
                    trigger_run_ids: list[UUID] = []

                    def inject(point: DecisionFaultPoint, checkpoint: Any) -> None:
                        if (
                            point is fault_point
                            and checkpoint.request.decision_type
                            == OPTIMIZATION_APPLY_DECISION_TYPE
                        ):
                            trigger_run_ids.append(
                                checkpoint.request.correlation.run_id
                            )
                            raise SimulatedCrash(
                                f"simulated process loss at {fault_point.value}"
                            )

                    crashing = ResearchAgent(
                        persistence_path=path,
                        auto_adaptation_enabled=True,
                        decision_fault_injector=inject,
                    )
                    try:
                        with self.assertRaises(SimulatedCrash):
                            await crashing.run("分析 Tesla 投资价值")
                    finally:
                        crashing.close()
                    self.assertEqual(len(trigger_run_ids), 1)

                    reopened = SQLitePersistence(path)
                    try:
                        trigger = await (
                            reopened.auto_adaptation_trigger_store.load_for_run(
                                trigger_run_ids[0]
                            )
                        )
                        self.assertIsNotNone(trigger)
                        self.assertEqual(
                            trigger.status,  # type: ignore[union-attr]
                            AutoAdaptationStatus.SELECTED,
                        )
                        checkpoint_store = (
                            reopened.create_decision_checkpoint_store(
                                DecisionCheckpoint[
                                    OptimizationApplyRequest,
                                    OptimizationApplyIntent,
                                    OptimizationApplyEffect,
                                ]
                            )
                        )
                        before = await checkpoint_store.load(
                            trigger.apply_request_id  # type: ignore[union-attr,arg-type]
                        )
                        self.assertIsNotNone(before)
                        assert before is not None
                        proposal_fingerprint = decision_fingerprint(before.proposal)
                        effect_fingerprint = (
                            before.validated_decision.normalized_effect.effect_fingerprint  # type: ignore[union-attr]
                        )
                        authorization_id = (
                            before.governance_receipt.authorization_id  # type: ignore[union-attr]
                        )

                        resumed = await _real_coordinator(
                            reopened
                        ).evaluate_after_run(
                            trigger_run_id=trigger_run_ids[0],
                            run_configuration=_default_configuration(),
                        )
                        after = await checkpoint_store.load(
                            trigger.apply_request_id  # type: ignore[union-attr,arg-type]
                        )
                        self.assertEqual(
                            resumed.status,
                            AutoAdaptationStatus.APPLIED,
                        )
                        self.assertEqual(resumed.active_revision, 1)
                        self.assertEqual(_configuration_counts(reopened)[1], 1)
                        self.assertEqual(
                            decision_fingerprint(after.proposal),  # type: ignore[union-attr]
                            proposal_fingerprint,
                        )
                        self.assertEqual(
                            after.validated_decision.normalized_effect.effect_fingerprint,  # type: ignore[union-attr]
                            effect_fingerprint,
                        )
                        self.assertEqual(
                            after.governance_receipt.authorization_id,  # type: ignore[union-attr]
                            authorization_id,
                        )
                        self.assertEqual(
                            after.result.status,  # type: ignore[union-attr]
                            DecisionResultStatus.APPLIED,
                        )
                    finally:
                        reopened.close()

    async def test_completed_run_does_not_trigger_auto_apply_twice(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await _run_auto_apply(path)
            trigger_run_id = result.runtime_result.final_state.run_id
            persistence = SQLitePersistence(path)
            try:
                coordinator = ResearchAutoAdaptationCoordinator(
                    policy=AutoAdaptationPolicy(enabled=True, scope=_scope()),
                    proposals=persistence.optimization_proposal_store,
                    configurations=persistence.runtime_configuration,
                    state_store=persistence.state_store,
                    triggers=persistence.auto_adaptation_trigger_store,
                    gateway=_gateway(persistence),
                    trace_sink=persistence.trace_sink,
                )
                repeated = await coordinator.evaluate_after_run(
                    trigger_run_id=trigger_run_id,
                    run_configuration=result.runtime_configuration,
                )
                self.assertEqual(repeated, result.auto_adaptation)
                self.assertEqual(_configuration_counts(persistence)[1], 1)
                active = await persistence.runtime_configuration.load_active(
                    _scope(),
                    OptimizationTargetKey.PLANNER_MAX_NODES,
                )
                self.assertEqual(active.revision, 1)  # type: ignore[union-attr]
            finally:
                persistence.close()

    async def test_unknown_auto_apply_is_not_retried(self) -> None:
        unknown = ResearchOptimizationConfigurationResult(
            request_id=uuid4(),
            receipt=None,
            active_configuration=None,
            governance_record=None,
            decision_status=DecisionResultStatus.FAILED,
            reconciliation_status=DecisionReconciliationStatus.UNKNOWN,
            reason="Commit state is indeterminate.",
        )
        proposal = _proposal()
        run_id = uuid4()
        state_store = InMemoryStateStore()
        await state_store.save(
            AgentState(
                run_id=run_id,
                task=AgentTask(description="unknown reconciliation run"),
                status=RunStatus.COMPLETED,
                revision=1,
            )
        )
        triggers = _TriggerStore()
        gateway = _Gateway(unknown)
        coordinator = ResearchAutoAdaptationCoordinator(
            policy=AutoAdaptationPolicy(enabled=True, scope=_scope()),
            proposals=_ProposalQuery((proposal,)),
            configurations=_ConfigurationQuery(),
            state_store=state_store,
            triggers=triggers,
            gateway=gateway,  # type: ignore[arg-type]
            trace_sink=InMemoryTraceSink(),
        )
        first = await coordinator.evaluate_after_run(
            trigger_run_id=run_id,
            run_configuration=_default_configuration(),
        )
        second = await coordinator.evaluate_after_run(
            trigger_run_id=run_id,
            run_configuration=_default_configuration(),
        )
        self.assertEqual(first.status, AutoAdaptationStatus.UNKNOWN)
        self.assertEqual(
            first.skip_reason,
            AutoAdaptationSkipReason.RECONCILIATION_UNKNOWN,
        )
        self.assertEqual(second, first)
        self.assertEqual(gateway.calls, 1)

    async def test_checkpoint_conflict_keeps_same_run_trigger_resumable(self) -> None:
        proposal = _proposal()
        run_id = uuid4()
        state_store = InMemoryStateStore()
        await state_store.save(
            AgentState(
                run_id=run_id,
                task=AgentTask(description="concurrent coordinator test"),
                status=RunStatus.COMPLETED,
                revision=1,
            )
        )
        triggers = _TriggerStore()
        conflict_gateway = _CheckpointConflictGateway()
        first = ResearchAutoAdaptationCoordinator(
            policy=AutoAdaptationPolicy(enabled=True, scope=_scope()),
            proposals=_ProposalQuery((proposal,)),
            configurations=_ConfigurationQuery(),
            state_store=state_store,
            triggers=triggers,
            gateway=conflict_gateway,  # type: ignore[arg-type]
            trace_sink=InMemoryTraceSink(),
        )
        conflicted = await first.evaluate_after_run(
            trigger_run_id=run_id,
            run_configuration=_default_configuration(),
        )
        self.assertEqual(conflicted.status, AutoAdaptationStatus.SELECTED)
        self.assertIsNone(conflicted.skip_reason)

        receipt = OptimizationApplyCommitReceipt(
            effect_fingerprint="a" * 64,
            operation=OPTIMIZATION_APPLY_COMMIT_OPERATION,
            scope=_scope(),
            target_key=OptimizationTargetKey.PLANNER_MAX_NODES,
            active_revision=1,
            active_snapshot_id=uuid4(),
            active_snapshot_fingerprint="b" * 64,
            previous_revision=0,
            payload_fingerprint="c" * 64,
        )
        winner_gateway = _Gateway(
            ResearchOptimizationConfigurationResult(
                request_id=conflicted.apply_request_id,  # type: ignore[arg-type]
                receipt=receipt,
                active_configuration=None,
                governance_record=None,
                decision_status=DecisionResultStatus.APPLIED,
            )
        )
        winner = ResearchAutoAdaptationCoordinator(
            policy=AutoAdaptationPolicy(enabled=True, scope=_scope()),
            proposals=_ProposalQuery((proposal,)),
            configurations=_ConfigurationQuery(),
            state_store=state_store,
            triggers=triggers,
            gateway=winner_gateway,  # type: ignore[arg-type]
            trace_sink=InMemoryTraceSink(),
        )
        applied = await winner.evaluate_after_run(
            trigger_run_id=run_id,
            run_configuration=_default_configuration(),
        )
        self.assertEqual(applied.status, AutoAdaptationStatus.APPLIED)
        self.assertEqual(applied.apply_effect_fingerprint, receipt.effect_fingerprint)
        self.assertEqual(conflict_gateway.calls, 1)
        self.assertEqual(winner_gateway.calls, 1)

    async def test_unknown_trigger_is_durable_and_not_retried_after_restart(
        self,
    ) -> None:
        unknown = ResearchOptimizationConfigurationResult(
            request_id=uuid4(),
            receipt=None,
            active_configuration=None,
            governance_record=None,
            decision_status=DecisionResultStatus.FAILED,
            reconciliation_status=DecisionReconciliationStatus.UNKNOWN,
            reason="Commit state is indeterminate.",
        )
        proposal = _proposal()
        run_id = uuid4()
        state_store = InMemoryStateStore()
        await state_store.save(
            AgentState(
                run_id=run_id,
                task=AgentTask(description="durable unknown trigger"),
                status=RunStatus.COMPLETED,
                revision=1,
            )
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            first_persistence = SQLitePersistence(path)
            try:
                first_gateway = _Gateway(unknown)
                first = ResearchAutoAdaptationCoordinator(
                    policy=AutoAdaptationPolicy(enabled=True, scope=_scope()),
                    proposals=_ProposalQuery((proposal,)),
                    configurations=_ConfigurationQuery(),
                    state_store=state_store,
                    triggers=first_persistence.auto_adaptation_trigger_store,
                    gateway=first_gateway,  # type: ignore[arg-type]
                    trace_sink=InMemoryTraceSink(),
                )
                recorded = await first.evaluate_after_run(
                    trigger_run_id=run_id,
                    run_configuration=_default_configuration(),
                )
                self.assertEqual(recorded.status, AutoAdaptationStatus.UNKNOWN)
                self.assertEqual(first_gateway.calls, 1)
            finally:
                first_persistence.close()

            reopened = SQLitePersistence(path)
            try:
                forbidden_gateway = _Gateway()
                resumed = await ResearchAutoAdaptationCoordinator(
                    policy=AutoAdaptationPolicy(enabled=True, scope=_scope()),
                    proposals=_ProposalQuery((proposal,)),
                    configurations=_ConfigurationQuery(),
                    state_store=InMemoryStateStore(),
                    triggers=reopened.auto_adaptation_trigger_store,
                    gateway=forbidden_gateway,  # type: ignore[arg-type]
                    trace_sink=InMemoryTraceSink(),
                ).evaluate_after_run(
                    trigger_run_id=run_id,
                    run_configuration=_default_configuration(),
                )
                self.assertEqual(resumed, recorded)
                self.assertEqual(forbidden_gateway.calls, 0)
            finally:
                reopened.close()

    async def test_completed_run_result_survives_auto_adaptation_evidence_failure(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            agent = ResearchAgent(
                persistence_path=path,
                auto_adaptation_enabled=True,
            )

            async def fail_evidence_resolution(
                scope: OptimizationScope,
            ) -> tuple[OptimizationProposal, ...]:
                del scope
                raise RuntimeError("optimization evidence store unavailable")

            agent._persistence.optimization_proposal_store.list_for_scope = (  # type: ignore[method-assign]
                fail_evidence_resolution
            )
            try:
                result = await agent.run("分析 Tesla 投资价值")
            finally:
                agent.close()

        self.assertEqual(result.runtime_result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(result.auto_adaptation.status, AutoAdaptationStatus.SKIPPED)
        self.assertEqual(
            result.auto_adaptation.skip_reason,
            AutoAdaptationSkipReason.INCOMPLETE_PHASE3_PROVENANCE,
        )
        self.assertIn("proposal query failed closed", result.auto_adaptation.error_summary)

    async def test_completed_run_result_survives_trigger_ledger_failure(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            result = await _run_with_trigger_ledger_failure(
                Path(directory) / "runtime.sqlite3"
            )

        self.assertEqual(result.runtime_result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(result.auto_adaptation.status, AutoAdaptationStatus.FAILED)
        self.assertEqual(
            result.auto_adaptation.skip_reason,
            AutoAdaptationSkipReason.INFRASTRUCTURE_FAILURE,
        )
        self.assertFalse(result.auto_adaptation.outcome_persisted)
        self.assertIn("trigger ledger unavailable", result.auto_adaptation.error_summary)

    async def test_completed_run_result_survives_checkpoint_initialization_failure(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            agent = ResearchAgent(
                persistence_path=Path(directory) / "runtime.sqlite3",
                auto_adaptation_enabled=True,
            )

            def fail_gateway_construction(trace_sink: object) -> object:
                del trace_sink
                raise RuntimeError("decision checkpoint store unavailable")

            agent._build_optimization_configuration_gateway = (  # type: ignore[method-assign]
                fail_gateway_construction
            )
            try:
                result = await agent.run("分析 Tesla 投资价值")
            finally:
                agent.close()

        self.assertEqual(result.runtime_result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(result.auto_adaptation.status, AutoAdaptationStatus.FAILED)
        self.assertFalse(result.auto_adaptation.outcome_persisted)
        self.assertIn("checkpoint store unavailable", result.auto_adaptation.error_summary)

    async def test_completed_run_result_survives_governance_failure(self) -> None:
        with TemporaryDirectory() as directory:
            gateway = _RaisingGateway("governance service unavailable")
            result = await _run_with_auto_gateway(
                Path(directory) / "runtime.sqlite3",
                gateway,
            )

        self.assertEqual(result.runtime_result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(result.auto_adaptation.status, AutoAdaptationStatus.UNKNOWN)
        self.assertEqual(
            result.auto_adaptation.skip_reason,
            AutoAdaptationSkipReason.RECONCILIATION_UNKNOWN,
        )
        self.assertIn("governance service unavailable", result.auto_adaptation.error_summary)
        self.assertEqual(gateway.calls, 1)

    async def test_completed_run_result_survives_review_pending(self) -> None:
        pending = ResearchOptimizationConfigurationResult(
            request_id=uuid4(),
            receipt=None,
            active_configuration=None,
            governance_record=None,
            review_pending=True,
            reason="Human Review pending.",
        )
        with TemporaryDirectory() as directory:
            result = await _run_with_auto_gateway(
                Path(directory) / "runtime.sqlite3",
                _Gateway(pending),
            )

        self.assertEqual(result.runtime_result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(
            result.auto_adaptation.status,
            AutoAdaptationStatus.REVIEW_PENDING,
        )
        self.assertIn("Human Review pending", result.auto_adaptation.error_summary)

    async def test_completed_run_result_survives_unknown_reconciliation(self) -> None:
        unknown = ResearchOptimizationConfigurationResult(
            request_id=uuid4(),
            receipt=None,
            active_configuration=None,
            governance_record=None,
            decision_status=DecisionResultStatus.FAILED,
            reconciliation_status=DecisionReconciliationStatus.UNKNOWN,
            reason="Configuration commit state is indeterminate.",
        )
        with TemporaryDirectory() as directory:
            result = await _run_with_auto_gateway(
                Path(directory) / "runtime.sqlite3",
                _Gateway(unknown),
            )

        self.assertEqual(result.runtime_result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(result.auto_adaptation.status, AutoAdaptationStatus.UNKNOWN)
        self.assertTrue(result.auto_adaptation.outcome_persisted)
        self.assertIn("indeterminate", result.auto_adaptation.error_summary)

    async def test_auto_adaptation_failure_does_not_change_run_completion(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await _run_with_trigger_ledger_failure(
                path
            )
            persistence = SQLitePersistence(path)
            try:
                stored_state = await persistence.state_store.load(
                    result.runtime_result.final_state.run_id
                )
            finally:
                persistence.close()

        final_state = result.runtime_result.final_state
        self.assertEqual(final_state.status, RunStatus.COMPLETED)
        self.assertEqual(result.runtime_result.final_state, final_state)
        self.assertIsNotNone(stored_state)
        self.assertEqual(stored_state.status, RunStatus.COMPLETED)  # type: ignore[union-attr]
        self.assertNotEqual(result.auto_adaptation.status, AutoAdaptationStatus.APPLIED)

    async def test_auto_adaptation_failure_does_not_remove_report_or_evaluation(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await _run_research(path)
            result = await _run_with_trigger_ledger_failure(path)

        self.assertEqual(result.runtime_result.final_state.status, RunStatus.COMPLETED)
        self.assertTrue(result.report.markdown)
        self.assertTrue(result.report_commit_receipt.artifact_fingerprint)
        self.assertIsNotNone(result.evaluation.report_id)
        self.assertIsNotNone(result.experience_metadata)
        self.assertTrue(result.decision_feedback)
        self.assertTrue(result.learning_insights)

    async def test_manual_rollback_still_restores_prior_value(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            applied_run = await _run_auto_apply(path)
            effect_fingerprint = applied_run.auto_adaptation.apply_effect_fingerprint
            self.assertIsNotNone(effect_fingerprint)
            agent = ResearchAgent(persistence_path=path)
            try:
                rolled = await agent.rollback_optimization(
                    effect_fingerprint,  # type: ignore[arg-type]
                    requested_by="operator:test",
                )
            finally:
                agent.close()
        self.assertEqual(rolled.active_configuration.revision, 2)  # type: ignore[union-attr]
        self.assertEqual(rolled.active_configuration.value, 8)  # type: ignore[union-attr]
        self.assertEqual(
            rolled.active_configuration.effective_activation_mode,  # type: ignore[union-attr]
            RuntimeConfigurationActivationMode.ROLLBACK,
        )

    async def test_end_to_end_auto_apply_then_manual_rollback_affects_future_runs(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            applied_run = await _run_auto_apply(path)
            effect_fingerprint = applied_run.auto_adaptation.apply_effect_fingerprint
            self.assertIsNotNone(effect_fingerprint)

            agent = ResearchAgent(
                persistence_path=path,
                auto_adaptation_enabled=False,
            )
            try:
                next_run = await agent.run("分析 Tesla 投资价值")
                rolled = await agent.rollback_optimization(
                    effect_fingerprint,  # type: ignore[arg-type]
                    requested_by="operator:phase4c-e2e",
                )
                post_rollback_run = await agent.run("分析 Tesla 投资价值")
            finally:
                agent.close()

        self.assertEqual(applied_run.runtime_configuration.revision, 0)
        self.assertEqual(applied_run.auto_adaptation.active_revision, 1)
        self.assertEqual(next_run.runtime_configuration.revision, 1)
        self.assertEqual(next_run.runtime_configuration.value, 9)
        self.assertEqual(
            next_run.runtime_configuration.effective_activation_mode,
            RuntimeConfigurationActivationMode.POLICY_AUTO,
        )
        self.assertEqual(rolled.active_configuration.revision, 2)  # type: ignore[union-attr]
        self.assertEqual(rolled.active_configuration.value, 8)  # type: ignore[union-attr]
        self.assertEqual(post_rollback_run.runtime_configuration.revision, 2)
        self.assertEqual(post_rollback_run.runtime_configuration.value, 8)
        self.assertEqual(
            post_rollback_run.runtime_configuration.effective_activation_mode,
            RuntimeConfigurationActivationMode.ROLLBACK,
        )

    async def test_agent_cannot_enable_or_modify_auto_adaptation(self) -> None:
        draft = OptimizationProposalDraft(
            target_ref="target-" + ("a" * 32),
            proposed_value=9,
            expected_impact="Bound graph size.",
            applicable_conditions=("Same scope.",),
            limitations=("Proposal only.",),
            supporting_candidate_refs=("optimization-evidence-" + ("b" * 32),),
        ).model_dump(mode="json")
        draft["auto_adaptation_enabled"] = True
        with self.assertRaises(ValidationError):
            OptimizationProposalDraft.model_validate(draft)
        with self.assertRaises(ValueError):
            OptimizationTargetKey("auto_adaptation.enabled")

    async def test_core_event_loop_remains_unchanged(self) -> None:
        source = (
            ROOT / "src" / "adaptive_agent_runtime" / "core" / "runtime.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("auto_adaptation", source.lower())
        self.assertNotIn("optimization", source.lower())

    async def test_expired_revoked_and_already_applied_are_fail_closed(self) -> None:
        now = datetime.now(timezone.utc)
        base = _proposal().model_copy(
            update={
                "created_at": now - timedelta(days=2),
                "expires_at": now - timedelta(days=1),
            }
        )
        expired, _, _, _ = await _evaluate_policy((base,))
        self.assertEqual(
            expired.skip_reason,
            AutoAdaptationSkipReason.PROPOSAL_EXPIRED,
        )

        revoked_proposal = _proposal().model_copy(
            update={"status": OptimizationProposalStatus.REVOKED}
        )
        revoked, _, _, _ = await _evaluate_policy((revoked_proposal,))
        self.assertEqual(
            revoked.skip_reason,
            AutoAdaptationSkipReason.PROPOSAL_INACTIVE,
        )

        applied_proposal = _proposal()
        already_applied, gateway, _, _ = await _evaluate_policy(
            (applied_proposal,),
            applied_proposals=frozenset({applied_proposal.proposal_id}),
        )
        self.assertEqual(
            already_applied.skip_reason,
            AutoAdaptationSkipReason.PROPOSAL_ALREADY_APPLIED,
        )
        self.assertEqual(gateway.calls, 0)


if __name__ == "__main__":
    unittest.main()
