from __future__ import annotations

import unittest

from adaptive_agent_runtime.decisioning import (
    ContextOmissionReason,
    ContextProjectionPolicy,
    ContextSensitivity,
    PolicyAgentContextBuilder,
    ProjectionSource,
    ProjectionSources,
)
from tests.decisioning.fakes import make_request


class AgentContextIsolationTests(unittest.TestCase):
    def test_scope_memory_trace_sensitivity_and_tags_are_fail_closed(self) -> None:
        sources = ProjectionSources(
            items=(
                ProjectionSource(
                    source_id="planner-memory",
                    source_type="memory",
                    agent_scope="planner",
                    memory_scope="planner",
                    sensitivity=ContextSensitivity.INTERNAL,
                    content={"fact": "allowed", "secret": "redact"},
                    estimated_tokens=10,
                ),
                ProjectionSource(
                    source_id="recovery-memory",
                    source_type="memory",
                    agent_scope="recovery",
                    memory_scope="recovery",
                    sensitivity=ContextSensitivity.INTERNAL,
                    content={"fact": "other Agent"},
                    estimated_tokens=10,
                ),
                ProjectionSource(
                    source_id="global-memory",
                    source_type="memory",
                    agent_scope="runtime_shared",
                    memory_scope="global",
                    sensitivity=ContextSensitivity.INTERNAL,
                    content={"fact": "global"},
                    estimated_tokens=10,
                ),
                ProjectionSource(
                    source_id="governance-internal",
                    source_type="governance_policy",
                    agent_scope="runtime_shared",
                    sensitivity=ContextSensitivity.INTERNAL,
                    content={"threshold": 0.8},
                    estimated_tokens=10,
                ),
                ProjectionSource(
                    source_id="restricted",
                    source_type="memory",
                    agent_scope="planner",
                    memory_scope="planner",
                    sensitivity=ContextSensitivity.RESTRICTED,
                    content={"fact": "restricted"},
                    estimated_tokens=10,
                ),
                ProjectionSource(
                    source_id="blocked",
                    source_type="memory",
                    agent_scope="planner",
                    memory_scope="planner",
                    sensitivity=ContextSensitivity.INTERNAL,
                    tags=frozenset({"other_agent_context"}),
                    content={"fact": "private reasoning"},
                    estimated_tokens=10,
                ),
            )
        )
        policy = ContextProjectionPolicy(
            policy_id="planner",
            version="1",
            agent_scope="planner",
            allowed_decision_types=frozenset({"fake.change"}),
            allowed_source_types=frozenset({"memory"}),
            allowed_memory_scopes=frozenset({"planner"}),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
            blocked_tags=frozenset({"other_agent_context"}),
            redact_keys=frozenset({"secret"}),
            max_items=10,
            max_context_tokens=100,
        )

        context = PolicyAgentContextBuilder().build(
            make_request(), sources, policy
        )

        self.assertEqual(
            tuple(item.source_id for item in context.blocks),
            ("planner-memory",),
        )
        self.assertEqual(context.blocks[0].content["secret"], "[REDACTED]")
        reasons = {item.source_id: item.reason for item in context.omissions}
        self.assertEqual(reasons["recovery-memory"], ContextOmissionReason.AGENT_SCOPE)
        self.assertEqual(reasons["global-memory"], ContextOmissionReason.MEMORY_SCOPE)
        self.assertEqual(
            reasons["governance-internal"], ContextOmissionReason.SOURCE_TYPE
        )
        self.assertEqual(reasons["restricted"], ContextOmissionReason.SENSITIVITY)
        self.assertEqual(reasons["blocked"], ContextOmissionReason.BLOCKED_TAG)
        for forbidden in (
            "state_store",
            "memory_store",
            "trace_sink",
            "governance",
            "runtime_state",
        ):
            self.assertFalse(hasattr(context, forbidden))

    def test_priority_then_item_and_token_budget_is_deterministic(self) -> None:
        sources = ProjectionSources(
            items=tuple(
                ProjectionSource(
                    source_id=source_id,
                    source_type="summary",
                    agent_scope="planner",
                    sensitivity=ContextSensitivity.PUBLIC,
                    content={"id": source_id},
                    priority=priority,
                    estimated_tokens=tokens,
                )
                for source_id, priority, tokens in (
                    ("low", 1, 5),
                    ("highest", 10, 8),
                    ("middle", 5, 4),
                )
            )
        )
        policy = ContextProjectionPolicy(
            policy_id="planner-budget",
            version="1",
            agent_scope="planner",
            allowed_decision_types=frozenset({"fake.change"}),
            allowed_source_types=frozenset({"summary"}),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.PUBLIC}),
            max_items=2,
            max_context_tokens=10,
        )

        context = PolicyAgentContextBuilder().build(
            make_request(), sources, policy
        )

        self.assertEqual(
            tuple(item.source_id for item in context.blocks),
            ("highest",),
        )
        reasons = {item.source_id: item.reason for item in context.omissions}
        self.assertEqual(reasons["middle"], ContextOmissionReason.TOKEN_BUDGET)
        self.assertEqual(reasons["low"], ContextOmissionReason.TOKEN_BUDGET)
        manifest = context.manifest()
        self.assertEqual(manifest.included_source_ids, ("highest",))
        self.assertNotIn("content", manifest.model_dump(mode="json"))

    def test_evidence_is_visible_only_when_its_source_is_projected(self) -> None:
        sources = ProjectionSources(
            items=(
                ProjectionSource(
                    source_id="hidden-evidence",
                    source_type="evidence",
                    agent_scope="recovery",
                    evidence_id="evidence-1",
                    sensitivity=ContextSensitivity.INTERNAL,
                    content={"fact": "hidden"},
                ),
            )
        )
        policy = ContextProjectionPolicy(
            policy_id="planner",
            version="1",
            agent_scope="planner",
            allowed_decision_types=frozenset({"fake.change"}),
            allowed_source_types=frozenset({"evidence"}),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
        )

        context = PolicyAgentContextBuilder().build(
            make_request(), sources, policy
        )

        self.assertEqual(context.evidence, ())


if __name__ == "__main__":
    unittest.main()
