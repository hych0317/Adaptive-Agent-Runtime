"""Versioned proposal staging, replay validation, activation, and rollback."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from uuid import UUID

from adaptive_agent_runtime.context_memory.json_types import utc_now
from adaptive_agent_runtime.evaluation import OptimizationProposal
from adaptive_agent_runtime.evolution.contracts import (
    OptimizationChangePlanner,
    ReplayCaseStore,
    ReplayValidationPolicy,
    RuntimeConfigurationStore,
)
from adaptive_agent_runtime.evolution.errors import (
    EvolutionConflictError,
    ReplayValidationError,
    UnsupportedOptimizationError,
)
from adaptive_agent_runtime.evolution.models import (
    OptimizationApplication,
    OptimizationApplicationStatus,
    OptimizationDeployment,
    ReplayCase,
    RuntimeConfigurationSnapshot,
    stable_evolution_id,
)
from adaptive_agent_runtime.evolution.replay import RuntimeReplayRunner


class ConfigurationPatchPlanner:
    """Translate a reviewed config_patch into one immutable candidate version."""

    module_id = "evolution.change_planner.configuration_patch"

    _DEFAULT_ALLOWED_KEYS = {
        "tool": frozenset({"capability_prevalidation", "max_retries"}),
        "orchestration": frozenset({"failure_replanning_enabled"}),
        "context": frozenset({"compression_trigger_ratio"}),
        "memory": frozenset({"memory_conflict_threshold"}),
        "context_memory": frozenset(
            {"compression_trigger_ratio", "memory_conflict_threshold"}
        ),
        "runtime": frozenset({"strict_runtime_validation"}),
    }

    def __init__(
        self,
        *,
        allowed_keys: Mapping[str, frozenset[str]] | None = None,
    ) -> None:
        self._allowed_keys = dict(allowed_keys or self._DEFAULT_ALLOWED_KEYS)

    def plan(
        self,
        proposal: OptimizationProposal,
        baseline: RuntimeConfigurationSnapshot,
    ) -> RuntimeConfigurationSnapshot:
        if proposal.target_component.value != baseline.component:
            raise UnsupportedOptimizationError(
                "proposal target does not match the active configuration"
            )
        patch = proposal.change_spec.get("config_patch")
        if not isinstance(patch, Mapping) or not patch:
            raise UnsupportedOptimizationError(
                "proposal does not contain a concrete config_patch"
            )
        allowed = self._allowed_keys.get(baseline.component)
        if allowed is None:
            raise UnsupportedOptimizationError(
                "target component has no registered configuration schema"
            )
        unknown = set(patch).difference(allowed)
        if unknown:
            raise UnsupportedOptimizationError(
                "proposal contains unsupported configuration keys: "
                + ", ".join(sorted(unknown))
            )
        config = dict(baseline.config)
        config.update(patch)
        if config == dict(baseline.config):
            raise UnsupportedOptimizationError(
                "proposal config_patch does not change the active configuration"
            )
        return RuntimeConfigurationSnapshot(
            component=baseline.component,
            version=baseline.version + 1,
            config=config,
            source_proposal_id=proposal.proposal_id,
        )


class InMemoryEvolutionStore:
    module_id = "evolution.store.in_memory"

    def __init__(self) -> None:
        self._configurations: dict[
            tuple[str, int], RuntimeConfigurationSnapshot
        ] = {}
        self._active: dict[str, int] = {}
        self._applications: dict[UUID, OptimizationApplication] = {}
        self._application_history: defaultdict[
            UUID,
            list[OptimizationApplication],
        ] = defaultdict(list)
        self._cases: dict[UUID, ReplayCase] = {}

    async def initialize(self, snapshot: RuntimeConfigurationSnapshot) -> None:
        existing_version = self._active.get(snapshot.component)
        if existing_version is not None:
            existing = self._configurations[(snapshot.component, existing_version)]
            if existing == snapshot:
                return
            raise EvolutionConflictError("component configuration is initialized")
        self._configurations[(snapshot.component, snapshot.version)] = snapshot
        self._active[snapshot.component] = snapshot.version

    async def load_active(
        self,
        component: str,
    ) -> RuntimeConfigurationSnapshot | None:
        version = self._active.get(component)
        if version is None:
            return None
        return self._configurations[(component, version)]

    async def activate(
        self,
        deployment: OptimizationDeployment,
    ) -> OptimizationApplication:
        if not deployment.validation.passed:
            raise ReplayValidationError("candidate failed Replay validation")
        current = await self.load_active(deployment.baseline.component)
        if current != deployment.baseline:
            raise EvolutionConflictError("active configuration changed after Replay")
        application_id = stable_evolution_id(
            "optimization-application",
            deployment.deployment_id,
        )
        existing = self._applications.get(application_id)
        if existing is not None:
            return existing
        self._configurations[
            (deployment.candidate.component, deployment.candidate.version)
        ] = deployment.candidate
        self._active[deployment.candidate.component] = deployment.candidate.version
        now = utc_now()
        application = OptimizationApplication(
            application_id=application_id,
            deployment=deployment,
            status=OptimizationApplicationStatus.APPLIED,
            applied_at=now,
            updated_at=now,
        )
        self._applications[application_id] = application
        self._application_history[application_id].append(application)
        return application

    async def rollback(
        self,
        application_id: UUID,
    ) -> OptimizationApplication:
        current = self._applications.get(application_id)
        if current is None:
            raise EvolutionConflictError("Optimization application was not found")
        if current.status is OptimizationApplicationStatus.ROLLED_BACK:
            return current
        active = await self.load_active(current.deployment.candidate.component)
        if active != current.deployment.candidate:
            raise EvolutionConflictError("candidate is no longer the active version")
        baseline = current.deployment.baseline
        self._active[baseline.component] = baseline.version
        rolled_back = current.model_copy(
            update={
                "status": OptimizationApplicationStatus.ROLLED_BACK,
                "revision": current.revision + 1,
                "updated_at": utc_now(),
            }
        )
        self._applications[application_id] = rolled_back
        self._application_history[application_id].append(rolled_back)
        return rolled_back

    async def load_application(
        self,
        application_id: UUID,
    ) -> OptimizationApplication | None:
        return self._applications.get(application_id)

    async def save(self, case: ReplayCase) -> None:
        current = self._cases.get(case.case_id)
        if current is not None and current != case:
            raise EvolutionConflictError("Replay case id was reused")
        self._cases[case.case_id] = case

    async def load(self, case_id: UUID) -> ReplayCase | None:
        return self._cases.get(case_id)

    async def list_all(self) -> tuple[ReplayCase, ...]:
        return tuple(self._cases[key] for key in sorted(self._cases, key=str))

    def history_for(
        self,
        application_id: UUID,
    ) -> tuple[OptimizationApplication, ...]:
        return tuple(self._application_history.get(application_id, ()))


class OptimizationDeploymentService:
    """Prepare with Replay; mutate configuration only through apply/rollback."""

    module_id = "evolution.optimization_deployment"

    def __init__(
        self,
        *,
        store: RuntimeConfigurationStore,
        change_planner: OptimizationChangePlanner,
        replay_runner: RuntimeReplayRunner,
        validator: ReplayValidationPolicy,
    ) -> None:
        self._store = store
        self._change_planner = change_planner
        self._replay_runner = replay_runner
        self._validator = validator

    async def prepare(
        self,
        proposal: OptimizationProposal,
        cases: tuple[ReplayCase, ...],
    ) -> OptimizationDeployment:
        if not cases:
            raise ReplayValidationError("Optimization Apply requires Replay cases")
        baseline = await self._store.load_active(
            proposal.target_component.value
        )
        if baseline is None:
            raise EvolutionConflictError(
                "target component has no active configuration baseline"
            )
        candidate = self._change_planner.plan(proposal, baseline)
        observations = await self._replay_runner.replay(cases, candidate)
        validation = self._validator.validate(cases, observations, candidate)
        return OptimizationDeployment(
            deployment_id=stable_evolution_id(
                "optimization-deployment",
                proposal.proposal_id,
                baseline.component,
                baseline.version,
                candidate.version,
            ),
            proposal=proposal,
            baseline=baseline,
            candidate=candidate,
            validation=validation,
        )

    async def apply(
        self,
        deployment: OptimizationDeployment,
    ) -> OptimizationApplication:
        if not deployment.validation.passed:
            raise ReplayValidationError(
                "Optimization candidate cannot apply after failed Replay"
            )
        return await self._store.activate(deployment)

    async def rollback(
        self,
        application_id: UUID,
    ) -> OptimizationApplication:
        return await self._store.rollback(application_id)
