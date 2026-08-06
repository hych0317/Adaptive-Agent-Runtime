"""Planner implementation backed by an execution-driven task graph."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence
from uuid import UUID

from adaptive_agent_runtime import (
    ActionRequest,
    AgentState,
    Observation,
    PlanDecision,
    RuntimeResumeBlockedError,
)
from adaptive_agent_runtime.orchestration.errors import OrchestrationStateError
from adaptive_agent_runtime.orchestration.execution import (
    EXECUTE_NODE_ACTION,
    ORCHESTRATION_METADATA_KEY,
)
from adaptive_agent_runtime.orchestration.contracts import (
    ReadyTaskNodeSelector,
    GraphMutationApplier,
    TaskGraphStore,
)
from adaptive_agent_runtime.orchestration.checkpoint import (
    InFlightTaskAction,
    TaskGraphCheckpoint,
)
from adaptive_agent_runtime.orchestration.graph import DynamicTaskGraph
from adaptive_agent_runtime.orchestration.models import (
    GraphMutation,
    TaskNodeStatus,
)
from adaptive_agent_runtime.orchestration.scheduler import GraphScheduler
from adaptive_agent_runtime.orchestration.recovery import (
    FailureDrivenReplanner,
    RecoveryContext,
    RecoveryPlanApplier,
    RecoveryRecord,
)
from adaptive_agent_runtime.orchestration.recovery_decision import (
    RECOVERY_APPLY_OPERATION,
    RecoveryDecisionHandler,
)
from adaptive_agent_runtime.orchestration.mutation_decision import (
    GRAPH_MUTATION_APPLY_OPERATION,
    GraphMutationDecisionHandler,
)
from adaptive_agent_runtime.orchestration.selection import (
    FirstReadyTaskNodeSelector,
)
from adaptive_agent_runtime.governance.errors import AuthorizationVerificationError
from adaptive_agent_runtime.governance.contracts import CommitPermitValidation
from adaptive_agent_runtime.governance.models import (
    GovernanceTarget,
    RuntimeCommitPermit,
)


class DynamicTaskGraphPlanner:
    """Implement Core Planner using an immutable DynamicTaskGraph per run."""

    module_id = "orchestration.dynamic_task_graph_planner"

    def __init__(
        self,
        initial_graph: DynamicTaskGraph,
        *,
        scheduler: GraphScheduler | None = None,
        ready_node_selector: ReadyTaskNodeSelector | None = None,
        graph_store: TaskGraphStore | None = None,
        mutation_applier: GraphMutationApplier | None = None,
        recovery_planner: FailureDrivenReplanner | None = None,
        recovery_applier: RecoveryPlanApplier | None = None,
        recovery_decision_handler: RecoveryDecisionHandler | None = None,
        mutation_decision_handler: GraphMutationDecisionHandler | None = None,
        commit_permit_verifier: CommitPermitValidation | None = None,
    ) -> None:
        if any(
            node.status is not TaskNodeStatus.PENDING
            for node in initial_graph.nodes
        ):
            raise ValueError("an initial task graph must contain only pending nodes")
        if recovery_decision_handler is not None and (
            recovery_planner is not None or recovery_applier is not None
        ):
            raise ValueError(
                "RecoveryDecisionHandler cannot be combined with legacy recovery ports"
            )
        self._initial_graph = initial_graph
        self._scheduler = scheduler or GraphScheduler()
        self._ready_node_selector = (
            ready_node_selector or FirstReadyTaskNodeSelector()
        )
        self._graph_store = graph_store
        self._mutation_applier = mutation_applier
        self._recovery_planner = recovery_planner
        self._recovery_applier = recovery_applier
        self._recovery_decision_handler = recovery_decision_handler
        self._mutation_decision_handler = mutation_decision_handler
        self._commit_permit_verifier = commit_permit_verifier
        self._graphs: dict[UUID, DynamicTaskGraph] = {}
        self._in_flight: defaultdict[UUID, dict[UUID, UUID]] = defaultdict(dict)
        self._processed_actions: defaultdict[UUID, set[UUID]] = defaultdict(set)
        self._recovery_attempts: defaultdict[UUID, dict[UUID, int]] = defaultdict(dict)
        self._recovery_records: defaultdict[UUID, list[RecoveryRecord]] = defaultdict(list)
        self._checkpoint_revisions: defaultdict[UUID, int] = defaultdict(int)
        self._last_effect_fingerprints: dict[UUID, str | None] = {}

    async def plan(self, state: AgentState) -> PlanDecision:
        graph = await self._load_graph(state)
        if graph is None:
            graph = DynamicTaskGraph(nodes=self._initial_graph.nodes)
            self._graphs[state.run_id] = graph
        graph, failed_node_id = await self._consume_observation(state, graph)
        if failed_node_id is None:
            failed_node_id = self._unhandled_failure_node(state, graph)
        if failed_node_id is not None and (
            self._recovery_planner is not None
            or self._recovery_decision_handler is not None
        ):
            self._graphs[state.run_id] = graph
            await self._save_checkpoint(state, graph)
            graph = await self._recover_failure(state, graph, failed_node_id)
        graph = self._scheduler.propagate_failed_dependencies(graph)
        self._graphs[state.run_id] = graph

        ready_nodes = self._scheduler.ready_nodes(graph)
        if ready_nodes:
            selected_id = await self._ready_node_selector.select_node_id(
                ready_nodes,
                state,
            )
            ready_by_id = {node.node_id: node for node in ready_nodes}
            try:
                selected = ready_by_id[selected_id]
            except KeyError as exc:
                raise OrchestrationStateError(
                    "ready-node selector proposed a node outside the ready set"
                ) from exc
            graph = graph.mark_running(selected.node_id)
            running_node = graph.get_node(selected.node_id)
            action = ActionRequest(
                name=EXECUTE_NODE_ACTION,
                arguments={
                    "graph_id": str(graph.graph_id),
                    "graph_version": graph.version,
                    "node": running_node.model_dump(mode="json"),
                },
            )
            self._graphs[state.run_id] = graph
            self._in_flight[state.run_id][action.action_id] = selected.node_id
            await self._save_checkpoint(state, graph)
            return PlanDecision.execute(
                action,
                reason=f"execute ready task node: {selected.goal}",
            )

        if all(
            node.status in {
                TaskNodeStatus.COMPLETED,
                TaskNodeStatus.RECOVERED,
            }
            for node in graph.nodes
        ):
            await self._save_checkpoint(state, graph)
            return PlanDecision.complete(output=self._graph_output(graph))

        failed_nodes = tuple(
            node
            for node in graph.nodes
            if node.status in {TaskNodeStatus.FAILED, TaskNodeStatus.BLOCKED}
        )
        if failed_nodes:
            failure_text = "; ".join(
                f"{node.goal}: {node.failure_reason}"
                for node in failed_nodes
            )
            await self._save_checkpoint(state, graph)
            return PlanDecision.fail(error=f"task graph failed: {failure_text}")

        running_nodes = tuple(
            node
            for node in graph.nodes
            if node.status is TaskNodeStatus.RUNNING
        )
        if running_nodes:
            action_ids = ", ".join(
                sorted(str(item) for item in self._in_flight[state.run_id])
            )
            raise RuntimeResumeBlockedError(
                "cannot safely resume while action outcome is in doubt: "
                f"{action_ids or 'unknown action'}"
            )
        raise OrchestrationStateError("task graph has no schedulable nodes")

    def graph_for(self, run_id: UUID) -> DynamicTaskGraph:
        """Return an immutable graph snapshot for inspection."""

        try:
            return self._graphs[run_id]
        except KeyError as exc:
            raise OrchestrationStateError(
                f"run '{run_id}' has no initialized task graph"
            ) from exc

    def recovery_records_for(self, run_id: UUID) -> tuple[RecoveryRecord, ...]:
        """Return immutable Failure-driven Replanning records for inspection."""

        return tuple(self._recovery_records.get(run_id, ()))

    async def _load_graph(self, state: AgentState) -> DynamicTaskGraph | None:
        graph = self._graphs.get(state.run_id)
        if graph is not None:
            return graph
        if self._graph_store is None:
            return None
        checkpoint = await self._graph_store.load(state.run_id)
        if checkpoint is None:
            return None
        if checkpoint.state_revision > state.revision:
            raise OrchestrationStateError(
                "task graph checkpoint is ahead of the Core state snapshot"
            )
        self._graphs[state.run_id] = checkpoint.graph
        self._in_flight[state.run_id] = {
            item.action_id: item.node_id for item in checkpoint.in_flight
        }
        self._processed_actions[state.run_id] = set(
            checkpoint.processed_action_ids
        )
        self._recovery_attempts[state.run_id] = dict(
            checkpoint.recovery_attempts
        )
        self._recovery_records[state.run_id] = list(
            checkpoint.recovery_records
        )
        self._checkpoint_revisions[state.run_id] = checkpoint.checkpoint_revision
        self._last_effect_fingerprints[state.run_id] = checkpoint.last_effect_fingerprint
        return checkpoint.graph

    async def _save_checkpoint(
        self,
        state: AgentState,
        graph: DynamicTaskGraph,
        *,
        permit: RuntimeCommitPermit | None = None,
        target: GovernanceTarget | None = None,
        subject_fingerprint: str | None = None,
    ) -> None:
        if self._graph_store is None:
            return
        next_revision = self._checkpoint_revisions[state.run_id] + 1
        checkpoint = TaskGraphCheckpoint(
            run_id=state.run_id,
            graph=graph,
            in_flight=tuple(
                InFlightTaskAction(action_id=action_id, node_id=node_id)
                for action_id, node_id in sorted(
                    self._in_flight[state.run_id].items(),
                    key=lambda item: str(item[0]),
                )
            ),
            processed_action_ids=tuple(
                sorted(
                    self._processed_actions[state.run_id],
                    key=str,
                )
            ),
            recovery_attempts=tuple(
                sorted(
                    self._recovery_attempts[state.run_id].items(),
                    key=lambda item: str(item[0]),
                )
            ),
            recovery_records=tuple(self._recovery_records[state.run_id]),
            state_revision=state.revision,
            checkpoint_revision=next_revision,
            last_effect_fingerprint=self._last_effect_fingerprints.get(state.run_id),
        )
        await self._graph_store.save(
            checkpoint,
            permit=permit,
            target=target,
            subject_fingerprint=subject_fingerprint,
        )
        self._checkpoint_revisions[state.run_id] = next_revision

    async def commit_graph_effect(
        self,
        *,
        state: AgentState,
        graph: DynamicTaskGraph,
        effect_fingerprint: str,
        recovery_record: RecoveryRecord | None = None,
        permit: RuntimeCommitPermit | None = None,
        target: GovernanceTarget | None = None,
        subject_fingerprint: str | None = None,
    ) -> DynamicTaskGraph:
        """Commit and read back the exact governed Graph effect before APPLIED."""

        if self._commit_permit_verifier is not None:
            if permit is None or target is None or subject_fingerprint is None:
                raise AuthorizationVerificationError(
                    "Graph Effect commit requires a Runtime Permit"
                )
            await self._commit_permit_verifier.verify(
                permit,
                operation=(
                    permit.operation
                    if permit.operation
                    in {GRAPH_MUTATION_APPLY_OPERATION, RECOVERY_APPLY_OPERATION}
                    else "graph.invalid_operation"
                ),
                target=target,
                subject_fingerprint=subject_fingerprint,
            )

        if recovery_record is not None:
            node_id = recovery_record.plan.analysis.node_id
            self._recovery_attempts[state.run_id][node_id] = (
                recovery_record.plan.attempt_number
            )
            if recovery_record not in self._recovery_records[state.run_id]:
                self._recovery_records[state.run_id].append(recovery_record)
        self._graphs[state.run_id] = graph
        self._last_effect_fingerprints[state.run_id] = effect_fingerprint
        await self._save_checkpoint(
            state,
            graph,
            permit=permit,
            target=target,
            subject_fingerprint=subject_fingerprint,
        )
        if self._graph_store is None:
            return graph
        committed = await self._graph_store.load(state.run_id)
        if (
            committed is None
            or committed.graph != graph
            or committed.last_effect_fingerprint != effect_fingerprint
        ):
            raise OrchestrationStateError("governed Graph commit failed read-back")
        return committed.graph

    async def load_graph_effect(
        self,
        *,
        run_id: UUID,
        effect_fingerprint: str,
    ) -> DynamicTaskGraph | None:
        if self._graph_store is None:
            if self._last_effect_fingerprints.get(run_id) == effect_fingerprint:
                return self._graphs.get(run_id)
            return None
        checkpoint = await self._graph_store.load(run_id)
        if (
            checkpoint is None
            or checkpoint.last_effect_fingerprint != effect_fingerprint
        ):
            return None
        return checkpoint.graph

    async def _consume_observation(
        self,
        state: AgentState,
        graph: DynamicTaskGraph,
    ) -> tuple[DynamicTaskGraph, UUID | None]:
        observation = state.last_observation
        if observation is None:
            return graph, None
        if observation.action_id in self._processed_actions[state.run_id]:
            return graph, None

        node_id = self._in_flight[state.run_id].get(observation.action_id)
        if node_id is None:
            raise OrchestrationStateError(
                "runtime observation does not match an in-flight task node"
            )

        updated = graph.resolve_node(node_id, observation)
        del self._in_flight[state.run_id][observation.action_id]
        self._processed_actions[state.run_id].add(observation.action_id)
        if observation.succeeded:
            try:
                for mutation in self._mutations_from(observation, node_id):
                    if self._mutation_applier is None:
                        updated = updated.apply_mutation(mutation)
                    else:
                        updated = await self._mutation_applier.apply(
                            updated,
                            mutation,
                            state=state,
                            source_node_id=node_id,
                        )
                if self._mutation_decision_handler is not None:
                    outcome = await self._mutation_decision_handler.handle(
                        graph=updated,
                        state=state,
                        source_node_id=node_id,
                        observation=observation,
                        committer=self,
                    )
                    if outcome is not None:
                        if outcome.graph.graph_id != updated.graph_id:
                            raise OrchestrationStateError(
                                "Graph Mutation Decision replaced graph identity"
                            )
                        if outcome.graph.version <= updated.version:
                            raise OrchestrationStateError(
                                "Graph Mutation Decision did not advance graph version"
                            )
                        updated = outcome.graph
            except Exception as exc:
                detail = str(exc) or exc.__class__.__name__
                mutation_failure = Observation.failed(
                    observation.action_id,
                    error=(
                        "task graph mutation rejected: "
                        f"{exc.__class__.__name__}: {detail}"
                    ),
                )
                updated = graph.resolve_node(node_id, mutation_failure)

        self._graphs[state.run_id] = updated
        return updated, node_id if not observation.succeeded else None

    def _unhandled_failure_node(
        self,
        state: AgentState,
        graph: DynamicTaskGraph,
    ) -> UUID | None:
        observation = state.last_observation
        if observation is None or observation.succeeded:
            return None
        handled_actions = {
            record.plan.analysis.action_id
            for record in self._recovery_records[state.run_id]
        }
        if observation.action_id in handled_actions:
            return None
        for node in graph.nodes:
            if (
                node.status is TaskNodeStatus.FAILED
                and node.observation is not None
                and node.observation.action_id == observation.action_id
            ):
                return node.node_id
        return None

    async def _recover_failure(
        self,
        state: AgentState,
        graph: DynamicTaskGraph,
        node_id: UUID,
    ) -> DynamicTaskGraph:
        planner = self._recovery_planner
        handler = self._recovery_decision_handler
        if planner is None and handler is None:
            return graph
        failed_node = graph.get_node(node_id)
        observation = failed_node.observation
        if observation is None or observation.succeeded:
            raise OrchestrationStateError(
                "failure recovery requires a failed node Observation"
            )
        context = RecoveryContext(
            graph=graph,
            state=state,
            failed_node=failed_node,
            observation=observation,
            prior_attempts=self._recovery_attempts[state.run_id].get(node_id, 0),
        )
        recovered = graph
        if handler is not None:
            outcome = await handler.handle(context, committer=self)
            recovery_plan = outcome.effect.plan
            recovered = outcome.graph
        else:
            assert planner is not None
            recovery_plan = await planner.replan(context)
        if recovery_plan.analysis.node_id != node_id:
            raise OrchestrationStateError(
                "Recovery Planner analyzed a different task node"
            )
        if recovery_plan.analysis.action_id != observation.action_id:
            raise OrchestrationStateError(
                "Recovery Planner analyzed a different action"
            )
        before_version = graph.version
        if handler is None and not recovery_plan.aborts:
            if self._recovery_applier is None:
                raise OrchestrationStateError(
                    "adaptive recovery requires a RecoveryPlanApplier"
                )
            recovered = await self._recovery_applier.apply(
                graph,
                recovery_plan,
                state=state,
            )
            if recovered.graph_id != graph.graph_id:
                raise OrchestrationStateError(
                    "Recovery Plan cannot replace the active graph identity"
                )
            if recovered.version <= graph.version:
                raise OrchestrationStateError(
                    "applied Recovery Plan did not advance graph version"
                )
        recovery_record = RecoveryRecord(
                plan=recovery_plan,
                graph_version_before=before_version,
                graph_version_after=recovered.version,
            )
        if handler is None:
            self._recovery_attempts[state.run_id][node_id] = recovery_plan.attempt_number
            self._recovery_records[state.run_id].append(recovery_record)
        self._graphs[state.run_id] = recovered
        if handler is None:
            await self._save_checkpoint(state, recovered)
        return recovered

    @staticmethod
    def _mutations_from(
        observation: Observation,
        expected_node_id: UUID,
    ) -> tuple[GraphMutation, ...]:
        metadata = observation.model_dump(mode="python")["metadata"]
        raw_envelope = metadata.get(ORCHESTRATION_METADATA_KEY)
        if raw_envelope is None:
            return ()
        if not isinstance(raw_envelope, Mapping):
            raise OrchestrationStateError(
                "orchestration observation metadata must be an object"
            )

        raw_node_id = raw_envelope.get("node_id")
        if raw_node_id != str(expected_node_id):
            raise OrchestrationStateError(
                "observation metadata references a different task node"
            )
        raw_mutations = raw_envelope.get("mutations", [])
        if not isinstance(raw_mutations, Sequence) or isinstance(
            raw_mutations,
            (str, bytes),
        ):
            raise OrchestrationStateError(
                "orchestration mutations must be an array"
            )
        return tuple(
            GraphMutation.model_validate(raw_mutation)
            for raw_mutation in raw_mutations
        )

    @staticmethod
    def _graph_output(graph: DynamicTaskGraph) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        for node in graph.nodes:
            observation_output = None
            if node.observation is not None:
                observation_output = node.observation.model_dump(mode="json")[
                    "output"
                ]
            nodes.append(
                {
                    "node_id": str(node.node_id),
                    "goal": node.goal,
                    "expected_output": node.expected_output,
                    "status": node.status.value,
                    "output": observation_output,
                }
            )
        return {
            "graph_id": str(graph.graph_id),
            "graph_version": graph.version,
            "nodes": nodes,
        }
