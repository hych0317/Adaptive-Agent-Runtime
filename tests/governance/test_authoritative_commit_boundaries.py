from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import UUID, uuid4

from adaptive_agent_runtime.context_memory import (
    ContextCompressionCommitter,
    ContextCompressionEffect,
    MemoryBatchWrite,
)
from adaptive_agent_runtime.decisioning import decision_fingerprint
from adaptive_agent_runtime.governance import (
    AuthorizationReplayError,
    AuthorizationVerificationError,
    BoundGovernedOperation,
    ConfidenceSignals,
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceRequest,
    GovernanceRule,
    GovernanceScope,
    GovernanceTarget,
    ImpactAssessment,
    RiskLevel,
    RuleEffect,
    RuntimeCommitPermit,
    RuntimeGovernanceEvaluator,
    SUBJECT_FINGERPRINT_ATTRIBUTE,
    governance_fingerprint,
)
from adaptive_agent_runtime.governance.models import GovernancePolicy
from adaptive_agent_runtime.llm import PlanningGraphEffectNormalizer
from adaptive_agent_runtime.orchestration import (
    GraphInitializationApplier,
    TaskGraphCheckpoint,
)
from adaptive_agent_runtime.persistence import SQLitePersistence
from adaptive_agent_runtime.tool_ecosystem import (
    ToolCorrelation,
    ToolExecutionPolicy,
    ToolInvocation,
)
from applications.research_agent.capabilities import (
    COMPANY_PROVIDER,
    INFORMATION_RETRIEVAL,
    build_research_tool_stack,
)
from tests.context_memory.test_context_compression_decision import context_unit
from tests.orchestration.test_adaptive_planning import (
    linear_draft,
    make_proposal,
    make_request,
)
from tests.persistence.test_sqlite_memory_atomicity import memory_unit


NOW = datetime(2026, 8, 5, 8, 0, tzinfo=timezone.utc)


class SimulatedCrash(BaseException):
    pass


def _authorization_binding(
    persistence: SQLitePersistence,
    *,
    request_id: UUID,
    operation: str,
    target: GovernanceTarget,
    subject: object,
):
    request = GovernanceRequest(
        request_id=request_id,
        scope=GovernanceScope.STATE,
        operation=operation,
        target=target,
        risk=RiskLevel.LOW,
        signals=ConfidenceSignals(
            stated_confidence=1.0,
            impact=ImpactAssessment(
                score=0.1,
                reversible=True,
                description="bounded commit-boundary test",
            ),
        ),
        attributes={SUBJECT_FINGERPRINT_ATTRIBUTE: governance_fingerprint(subject)},
        requested_at=NOW,
    )
    evaluator = RuntimeGovernanceEvaluator(
        policy=GovernancePolicy(
            policy_id="test.commit.boundary",
            version="1",
            rules=(
                GovernanceRule(
                    rule_id="allow.boundary",
                    description="allow the bounded test operation",
                    effect=RuleEffect.ALLOW,
                    scopes=(GovernanceScope.STATE,),
                    operations=(operation,),
                    risk_levels=(RiskLevel.LOW,),
                ),
            ),
        ),
        rule_evaluator=DeterministicRuleEvaluator(),
        confidence_evaluator=DeterministicConfidenceEvaluator(),
        review_service=persistence.human_review_service,
        clock=lambda: NOW,
    )
    decision = evaluator.evaluate(request)
    authorization = persistence.authorization_issuer.issue(request, decision)
    return request, decision, authorization


class AuthoritativeCommitBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def _reserved_permit(
        self,
        persistence: SQLitePersistence,
        *,
        operation: str,
        target: GovernanceTarget,
        subject: object,
        request_id: UUID,
    ) -> RuntimeCommitPermit:
        request, decision, authorization = _authorization_binding(
            persistence,
            request_id=request_id,
            operation=operation,
            target=target,
            subject=subject,
        )
        captured: RuntimeCommitPermit | None = None

        async def rejected_raw():
            raise AssertionError("Permit callback was bypassed")

        async def capture(permit: RuntimeCommitPermit):
            nonlocal captured
            captured = permit
            raise SimulatedCrash()

        with self.assertRaises(SimulatedCrash):
            await persistence.operation_executor.execute(
                request=request,
                decision=decision,
                authorization=authorization,
                target=BoundGovernedOperation(
                    module_id="test.capture_permit",
                    operation=operation,
                    target=target,
                    subject=subject,
                    apply=rejected_raw,
                    apply_with_permit=capture,
                ),
            )
        assert captured is not None
        return captured

    async def test_graph_context_memory_workspace_and_tool_reject_no_permit(self) -> None:
        with TemporaryDirectory() as directory:
            persistence = SQLitePersistence(Path(directory) / "runtime.sqlite3")
            planning_request = make_request(UUID(int=610))
            normalized = PlanningGraphEffectNormalizer().normalize(
                planning_request,
                make_proposal(planning_request, linear_draft()),
            )
            with self.assertRaises(AuthorizationVerificationError):
                await persistence.task_graph_store.save(
                    TaskGraphCheckpoint(
                        run_id=normalized.payload.run_id,
                        graph=normalized.payload.graph,
                        state_revision=0,
                        last_effect_fingerprint=normalized.effect_fingerprint,
                    )
                )
            with self.assertRaises(AuthorizationVerificationError):
                await GraphInitializationApplier(
                    persistence.task_graph_store,
                    permit_verifier=persistence.commit_permit_verifier,
                ).commit(
                    normalized.payload,
                    effect_fingerprint=normalized.effect_fingerprint,
                )

            source = context_unit(uuid4())
            await persistence.context_store.save(source, expected_revision=None)
            compression = ContextCompressionEffect(
                context_id=source.context_id,
                source_revision=source.revision,
                source_snapshot_fingerprint=decision_fingerprint(source),
                basis_fingerprint=decision_fingerprint({"source": source}),
                content={"summary": "compressed"},
                core_conclusions=("retained",),
                original_estimated_tokens=source.metadata.estimated_tokens,
                target_max_tokens=40,
                estimated_tokens=40,
            )
            with self.assertRaisesRegex(Exception, "requires a Runtime Permit"):
                await ContextCompressionCommitter(
                    store=persistence.context_store,
                    archive=persistence.context_archive,
                ).commit(
                    source,
                    compression,
                    effect_fingerprint=decision_fingerprint(compression),
                )

            with self.assertRaises(AuthorizationVerificationError):
                await persistence.memory_store.save_batch(
                    (MemoryBatchWrite(memory=memory_unit()),),
                    effect_fingerprint=decision_fingerprint({"memory": "direct"}),
                )

            with self.assertRaises(AuthorizationVerificationError):
                await persistence.memory_recall_bundle_store.commit(
                    None,  # type: ignore[arg-type]
                    effect_fingerprint=decision_fingerprint({"recall": "direct"}),
                    permit=None,
                    target=None,
                    subject_fingerprint=None,
                )

            with self.assertRaises(AuthorizationVerificationError):
                await persistence.experience_metadata_store.commit(
                    None,  # type: ignore[arg-type]
                    effect_fingerprint=decision_fingerprint(
                        {"experience": "direct"}
                    ),
                    permit=None,
                    target=None,
                    subject_fingerprint=None,
                )

            with self.assertRaises(AuthorizationVerificationError):
                await persistence.decision_feedback_store.commit(
                    None,  # type: ignore[arg-type]
                    effect_fingerprint=decision_fingerprint(
                        {"decision_feedback": "direct"}
                    ),
                    permit=None,
                    target=None,
                    subject_fingerprint=None,
                )

            with self.assertRaises(AuthorizationVerificationError):
                await persistence.learning_insight_store.commit(
                    None,  # type: ignore[arg-type]
                    effect_fingerprint=decision_fingerprint(
                        {"learning_insight": "direct"}
                    ),
                    permit=None,
                    target=None,
                    subject_fingerprint=None,
                )

            with self.assertRaises(AuthorizationVerificationError):
                await persistence.workspace_artifact_store.commit(
                    run_id=uuid4(),
                    node_id=uuid4(),
                    artifact_type="research_report",
                    effect_fingerprint=decision_fingerprint({"report": "direct"}),
                    artifact={"content": "bypass"},
                    provenance=("test",),
                    source_decision_request_id=uuid4(),
                    source_proposal_id=uuid4(),
                    permit=None,  # type: ignore[arg-type]
                    subject_fingerprint=decision_fingerprint({"subject": "direct"}),
                )

            tool_stack = build_research_tool_stack(
                permit_verifier=persistence.commit_permit_verifier
            )
            invocation = ToolInvocation(
                requirement_id=uuid4(),
                capability_id=INFORMATION_RETRIEVAL,
                provider_id=COMPANY_PROVIDER,
                arguments={"company": "Acme"},
                correlation=ToolCorrelation(run_id=uuid4()),
            )
            with self.assertRaises(AuthorizationVerificationError):
                await tool_stack.executor.execute(
                    invocation,
                    ToolExecutionPolicy(),
                    permit=None,  # type: ignore[arg-type]
                    target=GovernanceTarget(
                        target_type="tool_provider",
                        target_id=COMPANY_PROVIDER,
                    ),
                    subject_fingerprint=decision_fingerprint(invocation),
                )
            provider = tool_stack.providers[COMPANY_PROVIDER]
            self.assertEqual(provider.invocations, [])  # type: ignore[attr-defined]
            persistence.close()

    async def test_forged_mismatched_and_consumed_permits_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            persistence = SQLitePersistence(Path(directory) / "runtime.sqlite3")
            target = GovernanceTarget(target_type="workspace_report", target_id="r:n")
            subject = {"effect": "approved"}
            permit = await self._reserved_permit(
                persistence,
                operation="workspace.report.commit",
                target=target,
                subject=subject,
                request_id=UUID(int=620),
            )
            forged = permit.model_copy(update={"integrity_seal": "0" * 64})
            with self.assertRaisesRegex(AuthorizationVerificationError, "seal"):
                await persistence.commit_permit_verifier.verify(
                    forged,
                    operation=permit.operation,
                    target=permit.target,
                    subject_fingerprint=permit.subject_fingerprint,
                )
            with self.assertRaisesRegex(AuthorizationVerificationError, "Effect"):
                await persistence.commit_permit_verifier.verify(
                    permit,
                    operation=permit.operation,
                    target=permit.target,
                    subject_fingerprint=decision_fingerprint({"other": True}),
                )
            with self.assertRaisesRegex(AuthorizationVerificationError, "target"):
                await persistence.commit_permit_verifier.verify(
                    permit,
                    operation=permit.operation,
                    target=GovernanceTarget(
                        target_type="workspace_report", target_id="other"
                    ),
                    subject_fingerprint=permit.subject_fingerprint,
                )
            with self.assertRaisesRegex(AuthorizationVerificationError, "operation"):
                await persistence.commit_permit_verifier.verify(
                    permit,
                    operation="memory.write",
                    target=permit.target,
                    subject_fingerprint=permit.subject_fingerprint,
                )

            consumed_subject = {"effect": "consumed"}
            consumed_target = GovernanceTarget(target_type="test", target_id="one")
            request, decision, authorization = _authorization_binding(
                persistence,
                request_id=UUID(int=621),
                operation="test.consume",
                target=consumed_target,
                subject=consumed_subject,
            )
            captured: RuntimeCommitPermit | None = None

            async def raw():
                raise AssertionError("raw callback used")

            async def apply(issued: RuntimeCommitPermit):
                nonlocal captured
                captured = issued
                return {"committed": True}

            await persistence.operation_executor.execute(
                request=request,
                decision=decision,
                authorization=authorization,
                target=BoundGovernedOperation(
                    module_id="test.consume",
                    operation="test.consume",
                    target=consumed_target,
                    subject=consumed_subject,
                    apply=raw,
                    apply_with_permit=apply,
                ),
            )
            assert captured is not None
            with self.assertRaises(AuthorizationReplayError):
                await persistence.commit_permit_verifier.verify(
                    captured,
                    operation="test.consume",
                    target=consumed_target,
                    subject_fingerprint=governance_fingerprint(consumed_subject),
                )
            persistence.close()

    async def test_report_effect_is_idempotent_and_rejects_changed_content(self) -> None:
        with TemporaryDirectory() as directory:
            persistence = SQLitePersistence(Path(directory) / "runtime.sqlite3")
            run_id = uuid4()
            node_id = uuid4()
            target = GovernanceTarget(
                target_type="workspace_report",
                target_id=f"{run_id}:{node_id}",
            )
            effect_fingerprint = decision_fingerprint({"report-effect": "stable"})
            subject = {"effect_fingerprint": effect_fingerprint}
            subject_fingerprint = governance_fingerprint(subject)

            async def commit(request_number: int, artifact: object):
                request, decision, authorization = _authorization_binding(
                    persistence,
                    request_id=UUID(int=request_number),
                    operation="workspace.report.commit",
                    target=target,
                    subject=subject,
                )

                async def raw():
                    raise AssertionError("raw report callback used")

                async def apply(permit: RuntimeCommitPermit):
                    return await persistence.workspace_artifact_store.commit(
                        run_id=run_id,
                        node_id=node_id,
                        artifact_type="research_report",
                        effect_fingerprint=effect_fingerprint,
                        artifact=artifact,  # type: ignore[arg-type]
                        provenance=("analysis:test",),
                        source_decision_request_id=UUID(int=900),
                        source_proposal_id=UUID(int=901),
                        permit=permit,
                        subject_fingerprint=subject_fingerprint,
                        committed_at=NOW,
                    )

                return await persistence.operation_executor.execute(
                    request=request,
                    decision=decision,
                    authorization=authorization,
                    target=BoundGovernedOperation(
                        module_id="test.report.commit",
                        operation="workspace.report.commit",
                        target=target,
                        subject=subject,
                        apply=raw,
                        apply_with_permit=apply,
                    ),
                )

            first = await commit(630, {"content": "stable"})
            second = await commit(631, {"content": "stable"})
            self.assertEqual(first, second)
            with self.assertRaisesRegex(Exception, "fingerprint was reused"):
                await commit(632, {"content": "changed"})
            found = persistence.workspace_artifact_store.load_by_effect(
                effect_fingerprint
            )
            assert found is not None
            self.assertEqual(found[0], {"content": "stable"})
            self.assertEqual(found[1], first)
            persistence.close()


if __name__ == "__main__":
    unittest.main()
