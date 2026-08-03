from __future__ import annotations

import unittest
from collections.abc import Mapping
from typing import Any, cast
from uuid import UUID, uuid4

from pydantic import ValidationError

from adaptive_agent_runtime.context_memory import (
    ContextAssembly,
    ContextLayer,
    ContextMetadata,
    ContextRequirement,
    ContextSource,
    ContextUnit,
)
from adaptive_agent_runtime.llm import (
    BackendKind,
    BackendLimits,
    CapabilityContextPolicy,
    ContextEgressDeniedError,
    ContextEgressPolicy,
    ContextOmissionReason,
    ContextProjectionBudgetError,
    ContextProjectionRequest,
    ContextSensitivity,
    ContextUnitClassification,
    InferenceTargetProfile,
    LLMContextAdapter,
    LLMContextRole,
    PolicyEnforcedContextAdapter,
)


def context_unit(
    run_id: UUID,
    *,
    source: ContextSource,
    layer: ContextLayer,
    content: Any,
    tokens: int,
    tags: tuple[str, ...] = (),
) -> ContextUnit:
    return ContextUnit(
        content=content,
        metadata=ContextMetadata(
            source=source,
            layer=layer,
            run_id=run_id,
            tags=tags,
            estimated_tokens=tokens,
        ),
    )


def assembly(
    run_id: UUID,
    units: tuple[ContextUnit, ...],
    *,
    required: tuple[UUID, ...] = (),
    max_tokens: int = 4096,
) -> ContextAssembly:
    requirement = ContextRequirement(
        run_id=run_id,
        goal="Analyze the task",
        required_context_ids=required,
        max_tokens=max_tokens,
    )
    working = tuple(
        unit for unit in units if unit.metadata.layer is ContextLayer.WORKING
    )
    task = tuple(
        unit for unit in units if unit.metadata.layer is ContextLayer.TASK
    )
    semantic = tuple(
        unit for unit in units if unit.metadata.layer is ContextLayer.SEMANTIC
    )
    ordered = (*working, *task, *semantic)
    return ContextAssembly(
        requirement=requirement,
        units=ordered,
        working_context=working,
        task_context=task,
        semantic_context=semantic,
        used_tokens=sum(unit.metadata.estimated_tokens for unit in ordered),
    )


def target(*, max_context_tokens: int | None = None) -> InferenceTargetProfile:
    return InferenceTargetProfile(
        target_id="fake/default",
        backend_id="fake",
        backend_kind=BackendKind.LOCAL,
        adapter_version="1",
        model_id="fake-model",
        limits=BackendLimits(max_context_tokens=max_context_tokens),
    )


def projection_request(
    context_assembly: ContextAssembly,
    classifications: tuple[ContextUnitClassification, ...],
    *,
    capability_policy: CapabilityContextPolicy | None = None,
    egress_policy: ContextEgressPolicy | None = None,
    inference_target: InferenceTargetProfile | None = None,
) -> ContextProjectionRequest:
    return ContextProjectionRequest(
        assembly=context_assembly,
        target=inference_target or target(),
        capability_policy=capability_policy
        or CapabilityContextPolicy(cognitive_capability_id="reasoning"),
        egress_policy=egress_policy
        or ContextEgressPolicy(allowed_target_ids=("fake/default",)),
        classifications=classifications,
    )


class ContextAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = PolicyEnforcedContextAdapter()
        self.assertIsInstance(self.adapter, LLMContextAdapter)

    def test_classification_must_cover_every_assembled_unit(self) -> None:
        run_id = uuid4()
        unit = context_unit(
            run_id,
            source=ContextSource.CONVERSATION,
            layer=ContextLayer.TASK,
            content="Question",
            tokens=10,
        )

        with self.assertRaisesRegex(
            ValidationError,
            "exactly cover assembled units",
        ):
            projection_request(assembly(run_id, (unit,)), ())

    def test_target_permission_is_enforced_before_projection(self) -> None:
        run_id = uuid4()
        unit = context_unit(
            run_id,
            source=ContextSource.DOCUMENT,
            layer=ContextLayer.SEMANTIC,
            content="Evidence",
            tokens=10,
        )
        request = projection_request(
            assembly(run_id, (unit,)),
            (
                ContextUnitClassification(
                    context_id=unit.context_id,
                    sensitivity=ContextSensitivity.PUBLIC,
                ),
            ),
            egress_policy=ContextEgressPolicy(
                allowed_target_ids=("another/target",),
            ),
        )

        with self.assertRaisesRegex(
            ContextEgressDeniedError,
            "target 'fake/default' is not allowed",
        ):
            self.adapter.project(request)

    def test_projection_filters_redacts_and_preserves_trust_boundary(self) -> None:
        run_id = uuid4()
        conversation = context_unit(
            run_id,
            source=ContextSource.CONVERSATION,
            layer=ContextLayer.TASK,
            content={
                "question": "Analyze",
                "api_key": "secret",
                "Authorization": "Bearer secret",
            },
            tokens=12,
        )
        document = context_unit(
            run_id,
            source=ContextSource.DOCUMENT,
            layer=ContextLayer.SEMANTIC,
            content="Confidential evidence",
            tokens=20,
        )
        tool = context_unit(
            run_id,
            source=ContextSource.TOOL_RESULT,
            layer=ContextLayer.SEMANTIC,
            content="Untrusted tool output",
            tokens=15,
        )
        request = projection_request(
            assembly(run_id, (conversation, document, tool)),
            (
                ContextUnitClassification(
                    context_id=conversation.context_id,
                    sensitivity=ContextSensitivity.INTERNAL,
                ),
                ContextUnitClassification(
                    context_id=document.context_id,
                    sensitivity=ContextSensitivity.CONFIDENTIAL,
                ),
                ContextUnitClassification(
                    context_id=tool.context_id,
                    sensitivity=ContextSensitivity.PUBLIC,
                ),
            ),
            capability_policy=CapabilityContextPolicy(
                cognitive_capability_id="judge",
                allowed_sources=(
                    ContextSource.CONVERSATION,
                    ContextSource.DOCUMENT,
                ),
            ),
            egress_policy=ContextEgressPolicy(
                allowed_target_ids=("fake/default",),
                allowed_sensitivities=(
                    ContextSensitivity.PUBLIC,
                    ContextSensitivity.INTERNAL,
                ),
                redact_keys=("api_key", "authorization"),
            ),
        )

        package = self.adapter.project(request)

        self.assertEqual(package.cognitive_capability_id, "judge")
        self.assertEqual(len(package.blocks), 1)
        block = package.blocks[0]
        self.assertEqual(block.role, LLMContextRole.USER)
        redacted = cast(Mapping[str, Any], block.content)
        self.assertEqual(redacted["api_key"], "[REDACTED]")
        self.assertEqual(redacted["Authorization"], "[REDACTED]")
        self.assertEqual(
            block.redacted_keys,
            ("Authorization", "api_key"),
        )
        omissions = {item.context_id: item.reason for item in package.omissions}
        self.assertEqual(
            omissions[document.context_id],
            ContextOmissionReason.EGRESS_POLICY,
        )
        self.assertEqual(
            omissions[tool.context_id],
            ContextOmissionReason.CAPABILITY_POLICY,
        )

    def test_required_context_cannot_be_silently_filtered(self) -> None:
        run_id = uuid4()
        required = context_unit(
            run_id,
            source=ContextSource.DOCUMENT,
            layer=ContextLayer.SEMANTIC,
            content="Restricted",
            tokens=20,
        )
        request = projection_request(
            assembly(
                run_id,
                (required,),
                required=(required.context_id,),
            ),
            (
                ContextUnitClassification(
                    context_id=required.context_id,
                    sensitivity=ContextSensitivity.RESTRICTED,
                ),
            ),
        )

        with self.assertRaisesRegex(
            ContextEgressDeniedError,
            "required context",
        ):
            self.adapter.project(request)

    def test_required_context_is_reserved_before_optional_context(self) -> None:
        run_id = uuid4()
        optional = context_unit(
            run_id,
            source=ContextSource.WORKING_STATE,
            layer=ContextLayer.WORKING,
            content="Large optional state",
            tokens=40,
        )
        required = context_unit(
            run_id,
            source=ContextSource.CONVERSATION,
            layer=ContextLayer.TASK,
            content="Required task",
            tokens=20,
        )
        context_assembly = assembly(
            run_id,
            (optional, required),
            required=(required.context_id,),
            max_tokens=60,
        )
        classifications = tuple(
            ContextUnitClassification(
                context_id=unit.context_id,
                sensitivity=ContextSensitivity.PUBLIC,
            )
            for unit in context_assembly.units
        )
        request = projection_request(
            context_assembly,
            classifications,
            inference_target=target(max_context_tokens=30),
        )

        package = self.adapter.project(request)

        self.assertEqual(
            tuple(block.context_id for block in package.blocks),
            (required.context_id,),
        )
        self.assertEqual(package.max_tokens, 30)
        self.assertEqual(
            package.omissions[0].reason,
            ContextOmissionReason.TOKEN_BUDGET,
        )

    def test_required_context_over_effective_budget_is_rejected(self) -> None:
        run_id = uuid4()
        required = context_unit(
            run_id,
            source=ContextSource.CONVERSATION,
            layer=ContextLayer.TASK,
            content="Required task",
            tokens=20,
        )
        request = projection_request(
            assembly(
                run_id,
                (required,),
                required=(required.context_id,),
            ),
            (
                ContextUnitClassification(
                    context_id=required.context_id,
                    sensitivity=ContextSensitivity.PUBLIC,
                ),
            ),
            inference_target=target(max_context_tokens=10),
        )

        with self.assertRaises(ContextProjectionBudgetError) as captured:
            self.adapter.project(request)
        self.assertEqual(captured.exception.required_tokens, 20)
        self.assertEqual(captured.exception.available_tokens, 10)


if __name__ == "__main__":
    unittest.main()
