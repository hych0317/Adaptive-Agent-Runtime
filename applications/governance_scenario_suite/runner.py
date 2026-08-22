"""Isolated deterministic runner for governance scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from uuid import UUID, uuid5

from adaptive_agent_runtime.decisioning import decision_fingerprint
from applications.ecommerce_support.composition import (
    EcommerceComposition,
    compose_ecommerce_support,
)
from applications.ecommerce_support.models import AuthenticatedPrincipal
from applications.ecommerce_support.policies import DomainPolicyError
from applications.governance_scenario_suite.contracts import (
    ScenarioSpec,
    ScenarioVerdict,
)
from applications.governance_scenario_suite.evidence import (
    EvidenceBundle,
    EvidenceSource,
    ScenarioExecution,
    ScenarioRunResult,
)
from applications.governance_scenario_suite.oracles import (
    DeterministicScenarioOracle,
)
from applications.governance_scenario_suite.variants import (
    ScenarioExecutorRegistry,
    ScriptedModelStub,
)


_RUN_NAMESPACE = UUID("1263be59-10f7-425b-8702-bcd20a00bc77")


@dataclass(frozen=True)
class ScenarioRuntimeContext:
    scenario: ScenarioSpec
    run_id: UUID
    root: Path
    composition: EcommerceComposition
    principal: AuthenticatedPrincipal
    approval_id: str | None
    model: ScriptedModelStub
    random: random.Random


class GovernanceScenarioRunner:
    def __init__(
        self,
        *,
        executors: ScenarioExecutorRegistry,
        oracle: DeterministicScenarioOracle | None = None,
    ) -> None:
        self._executors = executors
        self._oracle = oracle or DeterministicScenarioOracle()

    def run(
        self,
        scenario: ScenarioSpec,
        *,
        workspace: str | Path | None = None,
    ) -> ScenarioRunResult:
        if workspace is None:
            with TemporaryDirectory(prefix="aar-governance-scenario-") as directory:
                return self._run_in_directory(scenario, Path(directory))
        root = Path(workspace).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return self._run_in_directory(scenario, root)

    def _run_in_directory(
        self,
        scenario: ScenarioSpec,
        root: Path,
    ) -> ScenarioRunResult:
        spec_fingerprint = decision_fingerprint(scenario)
        run_id = uuid5(
            _RUN_NAMESPACE,
            f"{scenario.id}|{scenario.profile.value}|{scenario.seed}|{spec_fingerprint}",
        )
        scenario_root = root / str(run_id)
        if scenario_root.exists() and any(scenario_root.iterdir()):
            raise ValueError(
                f"Scenario workspace '{scenario_root}' is not empty; isolation failed"
            )
        scenario_root.mkdir(parents=True, exist_ok=True)
        composition = compose_ecommerce_support(
            scenario_root,
            clock=lambda: scenario.clock,
        )
        try:
            composition.store.seed(scenario.initial_authoritative_state)
            principal = AuthenticatedPrincipal(
                tenant_id=scenario.principal.tenant_id,
                user_id=scenario.principal.user_id,
                roles=scenario.principal.roles,
            )
            approval_id: str | None = None
            if scenario.approved_effect is not None:
                approval_id = f"approval-{scenario.id.lower()}"
                composition.store.create_approval(
                    approval_id=approval_id,
                    principal=principal,
                    effect=scenario.approved_effect,
                    policy_version="ecommerce-policy-v1",
                    created_at=scenario.clock,
                    expires_at=(
                        scenario.approval_expires_at
                        or scenario.clock + timedelta(minutes=10)
                    ),
                )
            pre_state = composition.store.snapshot()
            model = ScriptedModelStub(scenario.model_script.proposals)
            context = ScenarioRuntimeContext(
                scenario=scenario,
                run_id=run_id,
                root=scenario_root,
                composition=composition,
                principal=principal,
                approval_id=approval_id,
                model=model,
                random=random.Random(scenario.seed),
            )
            executor = self._executors.resolve(scenario.profile)
            try:
                execution = executor.execute(context)
            except DomainPolicyError as exc:
                execution = ScenarioExecution(
                    decision=ScenarioVerdict.REJECT,
                    reason_code=exc.reason_code,
                    model_contexts=model.contexts,
                    available_sources=frozenset({EvidenceSource.AUDIT}),
                    model_call_count=model.calls,
                    metadata={"handled_error": exc.__class__.__name__},
                )
            except Exception as exc:
                execution = ScenarioExecution(
                    decision=ScenarioVerdict.FAIL_CLOSED,
                    model_contexts=model.contexts,
                    available_sources=frozenset({EvidenceSource.AUDIT}),
                    model_call_count=model.calls,
                    metadata={"handled_error": exc.__class__.__name__},
                )
            post_state = composition.store.snapshot()
            external_ledger = composition.gateway.ledger()
            available = frozenset(
                {
                    EvidenceSource.AUTHORITATIVE_STATE,
                    EvidenceSource.EXTERNAL_LEDGER,
                    *execution.available_sources,
                }
            )
            evidence = EvidenceBundle(
                scenario=scenario,
                execution=execution,
                pre_state=pre_state,
                post_state=post_state,
                external_ledger=external_ledger,
                available_sources=available,
            )
            verdict, findings, summary = self._oracle.evaluate(evidence)
            return ScenarioRunResult(
                run_id=run_id,
                scenario_id=scenario.id,
                family_id=scenario.family_id,
                profile=scenario.profile,
                spec_fingerprint=spec_fingerprint,
                evaluation_verdict=verdict,
                actual_decision=execution.decision,
                actual_reason_code=execution.reason_code,
                findings=findings,
                evidence=summary,
                metrics={
                    "model_call_count": execution.model_call_count,
                    "external_effect_count": summary.external_effect_count,
                    "external_attempt_count": summary.external_attempt_count,
                },
                started_at=scenario.clock,
                completed_at=scenario.clock,
            )
        finally:
            composition.close()
