"""Runtime Memory integration for user preferences, never knowledge content."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast
from uuid import UUID, uuid4

from pydantic import Field

from adaptive_agent_runtime.context_memory import (
    ConditionalMemoryRecall,
    EvidenceDrivenMemoryConsolidator,
    MemoryCandidate,
    MemoryCondition,
    MemoryEvidence,
    MemoryEvolutionType,
    MemoryRecallQuery,
    MemoryStore,
    MemoryUnit,
)
from adaptive_agent_runtime.context_memory.json_types import ImmutableJsonValue
from adaptive_agent_runtime.governance import (
    BoundGovernedOperation,
    DecisionOutcome,
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceAuthorizationIssuer,
    GovernancePolicy,
    GovernanceRule,
    GovernanceScope,
    GovernedOperationExecutor,
    InMemoryAuthorizationConsumptionStore,
    InMemoryHumanReviewService,
    MemoryGovernanceAdapter,
    RiskLevel,
    RuleEffect,
    RuntimeGovernanceEvaluator,
    StrictAuthorizationVerifier,
)
from adaptive_agent_runtime.llm.capabilities import (
    CapabilityTurnKind,
    EvidenceReference,
    MemoryExtractionCapability,
    MemoryExtractionRequest,
)
from adaptive_agent_runtime.llm.json_types import ImmutableJsonValue as LLMJsonValue

from applications.personal_knowledge.models import KnowledgeModel


_PREFERENCE_PREFIX = "personal_knowledge.preference."


class PreferenceSuggestion(KnowledgeModel):
    suggestion_id: UUID = Field(default_factory=uuid4)
    key: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,63}$")
    value: str = Field(min_length=1, max_length=500)
    rationale: str = Field(min_length=1, max_length=500)


class PreferenceMemoryService:
    """Explicit preferences write now; inferred preferences wait for confirmation."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._consolidator = EvidenceDrivenMemoryConsolidator(store)
        self._recall = ConditionalMemoryRecall(store)
        policy = GovernancePolicy(
            policy_id="personal-knowledge-preference-memory",
            version="1",
            rules=(
                GovernanceRule(
                    rule_id="allow-confirmed-preference-memory",
                    description="Allow bounded preference candidates after user consent.",
                    effect=RuleEffect.ALLOW,
                    scopes=(GovernanceScope.STATE,),
                    operations=("memory.write",),
                    risk_levels=(RiskLevel.LOW,),
                    priority=100,
                ),
            ),
        )
        self._governance = RuntimeGovernanceEvaluator(
            policy=policy,
            rule_evaluator=DeterministicRuleEvaluator(),
            confidence_evaluator=DeterministicConfidenceEvaluator(),
            review_service=InMemoryHumanReviewService(),
        )
        self._issuer = GovernanceAuthorizationIssuer()
        self._executor = GovernedOperationExecutor(
            verifier=StrictAuthorizationVerifier(),
            consumption_store=InMemoryAuthorizationConsumptionStore(),
        )
        self._adapter = MemoryGovernanceAdapter()

    def suggest_inferred(
        self,
        *,
        key: str,
        value: str,
        rationale: str,
    ) -> PreferenceSuggestion:
        """Return a UI suggestion only; this method performs no Memory write."""
        return PreferenceSuggestion(key=key, value=value, rationale=rationale)

    async def confirm_suggestion(
        self,
        suggestion: PreferenceSuggestion,
    ) -> MemoryUnit:
        return await self.record_explicit(
            key=suggestion.key,
            value=suggestion.value,
            source_reference=f"confirmed-inference:{suggestion.suggestion_id}",
        )

    async def record_explicit(
        self,
        *,
        key: str,
        value: str,
        source_reference: str = "user-explicit",
    ) -> MemoryUnit:
        bounded = PreferenceSuggestion(
            key=key,
            value=value,
            rationale="Explicit user preference.",
        )
        memory_key = _PREFERENCE_PREFIX + bounded.key
        existing = next(
            (
                item
                for item in await self._store.list_all()
                if item.memory_key == memory_key
            ),
            None,
        )
        content = cast(ImmutableJsonValue, {"preference": bounded.value})
        if existing is None:
            evolution = MemoryEvolutionType.EXTEND
            target_memory_id = None
        else:
            evolution = (
                MemoryEvolutionType.SUPPORT
                if existing.content == content
                else MemoryEvolutionType.MODIFY
            )
            target_memory_id = existing.memory_id
        candidate = MemoryCandidate(
            memory_key=memory_key,
            content=content,
            condition=MemoryCondition(
                facts={"application": "personal_knowledge"},
                required_tags=("personal_knowledge",),
                description="Applies only to personal knowledge app behavior.",
            ),
            evidence=(
                MemoryEvidence(
                    source_reference=source_reference,
                    note="User explicitly approved this behavioral preference.",
                    weight=1.0,
                ),
            ),
            confidence=1.0,
            evolution=evolution,
            target_memory_id=target_memory_id,
        )
        request = self._adapter.to_request(candidate, risk=RiskLevel.LOW)
        decision = self._governance.evaluate(request)
        if decision.outcome is not DecisionOutcome.ALLOW:
            raise PermissionError(f"preference Memory write denied: {decision.reason}")
        authorization = self._issuer.issue(request, decision)

        async def consolidate() -> MemoryUnit:
            result = await self._consolidator.consolidate(candidate)
            return result.memory

        return await self._executor.execute(
            request=request,
            decision=decision,
            authorization=authorization,
            target=BoundGovernedOperation(
                module_id="personal_knowledge.preference_memory",
                operation=request.operation,
                target=request.target,
                subject=candidate,
                apply=consolidate,
            ),
        )

    async def recall(self) -> tuple[MemoryUnit, ...]:
        return await self._recall.recall(
            MemoryRecallQuery(
                facts={"application": "personal_knowledge"},
                tags=("personal_knowledge",),
            )
        )


class RuntimePreferenceInferenceService:
    """Use Runtime memory extraction to propose, but never apply, preferences."""

    def __init__(self, capability: MemoryExtractionCapability) -> None:
        self._capability = capability

    async def infer(
        self,
        *,
        observations: Sequence[Mapping[str, object]],
        existing: Sequence[MemoryUnit],
        evidence_reference: str,
    ) -> tuple[PreferenceSuggestion, ...]:
        existing_catalog = [
            {
                "reference_id": f"memory:{item.memory_id}",
                "memory_key": item.memory_key,
                "content": item.content,
            }
            for item in existing
            if item.memory_key.startswith(_PREFERENCE_PREFIX)
        ]
        turn = await self._capability.extract(
            MemoryExtractionRequest(
                observations=cast(LLMJsonValue, list(observations)),
                evidence_catalog=(
                    EvidenceReference(
                        reference_id=evidence_reference,
                        kind="personal_knowledge.review_behavior",
                        summary="User edits and review instructions.",
                        reliability=0.8,
                    ),
                ),
                existing_memories=cast(LLMJsonValue, existing_catalog),
                existing_memory_reference_ids=tuple(
                    f"memory:{item.memory_id}"
                    for item in existing
                    if item.memory_key.startswith(_PREFERENCE_PREFIX)
                ),
            )
        )
        if turn.kind is not CapabilityTurnKind.COMPLETED or turn.result is None:
            return ()
        suggestions: list[PreferenceSuggestion] = []
        for candidate in turn.result:
            if not candidate.memory_key.startswith(_PREFERENCE_PREFIX):
                continue
            content = candidate.content
            if not isinstance(content, Mapping):
                continue
            value = content.get("preference")
            if not isinstance(value, str) or not value.strip():
                continue
            key = candidate.memory_key.removeprefix(_PREFERENCE_PREFIX)
            try:
                suggestions.append(
                    PreferenceSuggestion(
                        key=key,
                        value=value.strip(),
                        rationale=(
                            "Runtime Memory inference candidate "
                            f"(confidence {candidate.confidence:.2f}); user confirmation required."
                        ),
                    )
                )
            except ValueError:
                continue
        return tuple(suggestions)
