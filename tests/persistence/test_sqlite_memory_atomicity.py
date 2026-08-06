from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import UUID

from adaptive_agent_runtime.context_memory import (
    MemoryBatchWrite,
    MemoryCondition,
    MemoryEvidence,
    MemoryUnit,
)
from adaptive_agent_runtime.decisioning import decision_fingerprint
from adaptive_agent_runtime.governance import (
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
    RuntimeGovernanceEvaluator,
    SUBJECT_FINGERPRINT_ATTRIBUTE,
    governance_fingerprint,
    GovernedOperationError,
)
from adaptive_agent_runtime.governance.models import GovernancePolicy
from adaptive_agent_runtime.persistence import SQLitePersistence


NOW = datetime(2026, 8, 5, 8, 0, tzinfo=timezone.utc)
TARGET = GovernanceTarget(target_type="memory_batch", target_id="run:memory")


def memory_unit(*, suffix: int = 1, content: str = "fact") -> MemoryUnit:
    return MemoryUnit(
        memory_id=UUID(int=1000 + suffix),
        memory_key=f"research.fact.{suffix}",
        content={"fact": content},
        condition=MemoryCondition(facts={"domain": "finance"}),
        evidence=(
            MemoryEvidence(
                evidence_id=UUID(int=2000 + suffix),
                source_reference=f"run:{suffix}",
                note="authoritative test evidence",
                observed_at=NOW,
            ),
        ),
        confidence=0.9,
        last_candidate_id=UUID(int=3000 + suffix),
        last_candidate_fingerprint=decision_fingerprint({"candidate": suffix}),
        created_at=NOW,
        updated_at=NOW,
    )


def _run_memory_apply(
    path: Path,
    *,
    request_number: int,
    writes: tuple[MemoryBatchWrite, ...],
    effect_fingerprint: str,
):
    async def execute():  # type: ignore[no-untyped-def]
        persistence = SQLitePersistence(path)
        try:
            subject = {
                "effect_fingerprint": effect_fingerprint,
                "writes": [write.model_dump(mode="json") for write in writes],
            }
            subject_fingerprint = governance_fingerprint(subject)
            request = GovernanceRequest(
                request_id=UUID(int=request_number),
                scope=GovernanceScope.STATE,
                operation="memory.write",
                target=TARGET,
                risk=RiskLevel.LOW,
                signals=ConfidenceSignals(
                    stated_confidence=1.0,
                    impact=ImpactAssessment(
                        score=0.1,
                        reversible=True,
                        description="atomic Memory batch test",
                    ),
                ),
                attributes={SUBJECT_FINGERPRINT_ATTRIBUTE: subject_fingerprint},
                requested_at=NOW,
            )
            evaluator = RuntimeGovernanceEvaluator(
                policy=GovernancePolicy(
                    policy_id="test.memory.atomicity",
                    version="1",
                    rules=(
                        GovernanceRule(
                            rule_id="allow.memory",
                            description="allow bounded test Memory batch",
                            effect=RuleEffect.ALLOW,
                            scopes=(GovernanceScope.STATE,),
                            operations=("memory.write",),
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

            async def rejected_raw():
                raise AssertionError("permit-bound Memory callback was not used")

            async def apply_with_permit(permit):  # type: ignore[no-untyped-def]
                committed = await persistence.memory_store.save_batch(
                    writes,
                    effect_fingerprint=effect_fingerprint,
                    permit=permit,
                    target=TARGET,
                    subject_fingerprint=subject_fingerprint,
                )
                receipt = await persistence.memory_store.load_batch_receipt(
                    effect_fingerprint
                )
                assert receipt is not None
                return {
                    "memory_ids": [str(item.memory_id) for item in committed],
                    "receipt": receipt.model_dump(mode="json"),
                }

            return await persistence.operation_executor.execute(
                request=request,
                decision=decision,
                authorization=authorization,
                target=BoundGovernedOperation(
                    module_id="test.memory.commit",
                    operation="memory.write",
                    target=TARGET,
                    subject=subject,
                    apply=rejected_raw,
                    apply_with_permit=apply_with_permit,
                ),
            )
        finally:
            persistence.close()

    return asyncio.run(execute())


class SQLiteMemoryAtomicityTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_connections_apply_same_effect_once_and_share_receipt(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            seed = SQLitePersistence(path)
            seed.close()
            writes = (MemoryBatchWrite(memory=memory_unit()),)
            effect_fingerprint = decision_fingerprint({"effect": "memory-batch"})
            first, second = await asyncio.gather(
                asyncio.to_thread(
                    _run_memory_apply,
                    path,
                    request_number=501,
                    writes=writes,
                    effect_fingerprint=effect_fingerprint,
                ),
                asyncio.to_thread(
                    _run_memory_apply,
                    path,
                    request_number=502,
                    writes=writes,
                    effect_fingerprint=effect_fingerprint,
                ),
            )
            self.assertEqual(first["receipt"], second["receipt"])
            reopened = SQLitePersistence(path)
            self.assertEqual(len(await reopened.memory_store.list_all()), 1)
            with reopened.database.reader() as cursor:
                counts = {
                    table: int(
                        cursor.execute(
                            f"SELECT COUNT(*) AS count FROM {table}"
                        ).fetchone()["count"]
                    )
                    for table in (
                        "memory_snapshots",
                        "memory_current",
                        "memory_applied_candidates",
                        "memory_applied_effects",
                        "memory_batch_receipts",
                    )
                }
            self.assertEqual(set(counts.values()), {1})
            reopened.close()

    async def test_receipt_failure_rolls_back_records_markers_and_receipt(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            persistence = SQLitePersistence(path)
            with persistence.database.transaction() as cursor:
                cursor.execute(
                    "CREATE TRIGGER fail_memory_receipt BEFORE INSERT ON "
                    "memory_batch_receipts BEGIN SELECT RAISE(ABORT, "
                    "'receipt failure'); END"
                )
            persistence.close()
            writes = (MemoryBatchWrite(memory=memory_unit()),)
            effect_fingerprint = decision_fingerprint({"effect": "receipt-fail"})
            with self.assertRaises(Exception):
                await asyncio.to_thread(
                    _run_memory_apply,
                    path,
                    request_number=503,
                    writes=writes,
                    effect_fingerprint=effect_fingerprint,
                )
            reopened = SQLitePersistence(path)
            with reopened.database.reader() as cursor:
                for table in (
                    "memory_snapshots",
                    "memory_current",
                    "memory_applied_candidates",
                    "memory_applied_effects",
                    "memory_batch_receipts",
                ):
                    row = cursor.execute(
                        f"SELECT COUNT(*) AS count FROM {table}"
                    ).fetchone()
                    assert row is not None
                    self.assertEqual(int(row["count"]), 0, table)
            reopened.close()

    async def test_same_effect_with_different_payload_is_conflict(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            SQLitePersistence(path).close()
            effect_fingerprint = decision_fingerprint({"effect": "conflict"})
            first = (MemoryBatchWrite(memory=memory_unit(content="first")),)
            second = (MemoryBatchWrite(memory=memory_unit(content="changed")),)
            await asyncio.to_thread(
                _run_memory_apply,
                path,
                request_number=504,
                writes=first,
                effect_fingerprint=effect_fingerprint,
            )
            with self.assertRaisesRegex(
                GovernedOperationError,
                "fingerprint was reused with another batch",
            ):
                await asyncio.to_thread(
                    _run_memory_apply,
                    path,
                    request_number=505,
                    writes=second,
                    effect_fingerprint=effect_fingerprint,
                )


if __name__ == "__main__":
    unittest.main()
