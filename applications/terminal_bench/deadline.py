"""Pure wall-clock slot allocation for terminal planning.

The allocator keeps inference, the current action, a follow-up verification
turn, and final cleanup in one calculation.  It deliberately knows nothing
about Planner state so the same arithmetic can be reused before inference and
again by the post-inference Action Gate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from math import floor, isfinite


class TerminalDeadlineSequence(StrEnum):
    """Explicit completion sequences admitted by the deadline controller."""

    RECONCILE_THEN_WORK_VERIFY = "reconcile_then_work_verify"
    RECONCILE_THEN_VERIFY = "reconcile_then_verify"
    WORK_THEN_VERIFY = "work_then_verify"
    DIRECT_VERIFY = "direct_verify"


@dataclass(frozen=True, slots=True)
class TerminalInferenceTiming:
    """Provider-backed inference windows used by deadline admission."""

    normal_preferred_seconds: float
    compact_preferred_seconds: float
    emergency_preferred_seconds: float
    normal_minimum_seconds: float
    compact_minimum_seconds: float

    def __post_init__(self) -> None:
        for name, value in (
            ("normal_preferred_seconds", self.normal_preferred_seconds),
            ("compact_preferred_seconds", self.compact_preferred_seconds),
            ("emergency_preferred_seconds", self.emergency_preferred_seconds),
            ("normal_minimum_seconds", self.normal_minimum_seconds),
            ("compact_minimum_seconds", self.compact_minimum_seconds),
        ):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.normal_preferred_seconds < self.normal_minimum_seconds:
            raise ValueError(
                "normal preferred inference seconds cannot be below its minimum"
            )
        if (
            self.compact_preferred_seconds < self.compact_minimum_seconds
            or self.emergency_preferred_seconds
            < self.compact_minimum_seconds
        ):
            raise ValueError(
                "compact preferred inference seconds cannot be below its minimum"
            )


@dataclass(frozen=True, slots=True)
class TerminalBudgetBand:
    """One profile-local minimum, preferred, and maximum stage budget."""

    minimum_seconds: float
    preferred_seconds: float
    maximum_seconds: float

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_seconds", self.minimum_seconds),
            ("preferred_seconds", self.preferred_seconds),
            ("maximum_seconds", self.maximum_seconds),
        ):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.preferred_seconds < self.minimum_seconds:
            raise ValueError("preferred stage budget cannot be below minimum")
        if self.maximum_seconds < self.preferred_seconds:
            raise ValueError("maximum stage budget cannot be below preferred")


@dataclass(frozen=True, slots=True)
class TerminalDeadlineBudgetProfile:
    """Terminal-Bench profile budgets; these are not Core Runtime limits."""

    normal_inference: TerminalBudgetBand
    compact_inference: TerminalBudgetBand
    emergency_inference: TerminalBudgetBand
    normal_inspection: TerminalBudgetBand
    reconciliation_inspection: TerminalBudgetBand
    normal_work: TerminalBudgetBand
    final_work: TerminalBudgetBand
    convergence_inference: TerminalBudgetBand
    verification: TerminalBudgetBand
    model_cancellation_cleanup_seconds: float = 5.0

    def __post_init__(self) -> None:
        _require_nonnegative(
            "model_cancellation_cleanup_seconds",
            self.model_cancellation_cleanup_seconds,
        )


def terra_high_deadline_budget_profile(
    *,
    normal_inference_maximum_seconds: float = 300.0,
) -> TerminalDeadlineBudgetProfile:
    """Return the calibrated Terminal-Bench profile used by Terra-high evals."""

    return TerminalDeadlineBudgetProfile(
        normal_inference=TerminalBudgetBand(
            minimum_seconds=45.0,
            preferred_seconds=90.0,
            maximum_seconds=normal_inference_maximum_seconds,
        ),
        compact_inference=TerminalBudgetBand(30.0, 60.0, 120.0),
        emergency_inference=TerminalBudgetBand(20.0, 45.0, 60.0),
        normal_inspection=TerminalBudgetBand(5.0, 20.0, 60.0),
        reconciliation_inspection=TerminalBudgetBand(5.0, 15.0, 30.0),
        normal_work=TerminalBudgetBand(5.0, 120.0, 300.0),
        final_work=TerminalBudgetBand(5.0, 45.0, 60.0),
        convergence_inference=TerminalBudgetBand(30.0, 60.0, 120.0),
        verification=TerminalBudgetBand(15.0, 60.0, 120.0),
    )


class TerminalDeadlineStage(StrEnum):
    INFERENCE = "inference"
    INSPECT = "inspect"
    WORK = "work"
    FOLLOWUP_INFERENCE = "followup_inference"
    VERIFY = "verify"


@dataclass(frozen=True, slots=True)
class TerminalDeadlineStageAllocation:
    stage: TerminalDeadlineStage
    minimum_seconds: float
    preferred_seconds: float
    maximum_seconds: float
    overhead_seconds: float
    allocated_seconds: float
    current: bool


@dataclass(frozen=True, slots=True)
class TerminalDeadlineSlots:
    sequence: TerminalDeadlineSequence | None
    remaining_seconds: float | None
    minimum_inference_seconds: float
    preferred_inference_seconds: float
    minimum_action_seconds: int
    preferred_action_seconds: int
    maximum_action_seconds: int
    followup_inference_seconds: float
    verification_seconds: float
    cleanup_seconds: float
    inference_limit_seconds: float | None
    action_limit_seconds: int | None
    feasible: bool
    complete_sequence_feasible: bool
    stage_allocations: tuple[TerminalDeadlineStageAllocation, ...] = ()

    @property
    def future_reserve_seconds(self) -> float:
        return (
            self.followup_inference_seconds
            + self.verification_seconds
            + self.cleanup_seconds
        )

    @property
    def followup_action_seconds(self) -> float:
        """Reserved post-current actions, including the final verification."""

        return self.verification_seconds


def allocate_deadline_slots(
    *,
    remaining_seconds: float | None,
    minimum_inference_seconds: float,
    preferred_inference_seconds: float,
    minimum_action_seconds: int = 1,
    preferred_action_seconds: int,
    maximum_action_seconds: int,
    followup_inference_seconds: float = 0.0,
    verification_seconds: float = 0.0,
    cleanup_seconds: float = 0.0,
    sequence: TerminalDeadlineSequence | None = None,
) -> TerminalDeadlineSlots:
    """Allocate compatible inference and action caps within one deadline.

    Preferred action capacity is protected before inference receives time above
    its minimum.  Once the preferred inference window is available, remaining
    capacity is returned to the action up to its maximum.  Consequently the two
    advertised caps can be consumed together without stealing the reserved
    follow-up verification or cleanup slots.
    """

    _require_nonnegative(
        "minimum_inference_seconds",
        minimum_inference_seconds,
    )
    _require_nonnegative(
        "preferred_inference_seconds",
        preferred_inference_seconds,
    )
    _require_nonnegative(
        "followup_inference_seconds",
        followup_inference_seconds,
    )
    _require_nonnegative("verification_seconds", verification_seconds)
    _require_nonnegative("cleanup_seconds", cleanup_seconds)
    if preferred_inference_seconds < minimum_inference_seconds:
        raise ValueError(
            "preferred inference seconds cannot be below the minimum"
        )
    if minimum_action_seconds < 1:
        raise ValueError("minimum action seconds must be positive")
    if preferred_action_seconds < minimum_action_seconds:
        raise ValueError(
            "preferred action seconds cannot be below the minimum"
        )
    if maximum_action_seconds < preferred_action_seconds:
        raise ValueError(
            "maximum action seconds cannot be below the preferred value"
        )

    if remaining_seconds is None:
        return TerminalDeadlineSlots(
            sequence=sequence,
            remaining_seconds=None,
            minimum_inference_seconds=minimum_inference_seconds,
            preferred_inference_seconds=preferred_inference_seconds,
            minimum_action_seconds=minimum_action_seconds,
            preferred_action_seconds=preferred_action_seconds,
            maximum_action_seconds=maximum_action_seconds,
            followup_inference_seconds=followup_inference_seconds,
            verification_seconds=verification_seconds,
            cleanup_seconds=cleanup_seconds,
            inference_limit_seconds=None,
            action_limit_seconds=None,
            feasible=True,
            complete_sequence_feasible=True,
        )
    _require_nonnegative("remaining_seconds", remaining_seconds)

    turn_capacity = max(
        0.0,
        remaining_seconds
        - followup_inference_seconds
        - verification_seconds
        - cleanup_seconds,
    )
    maximum_action_capacity = max(
        0,
        floor(turn_capacity - minimum_inference_seconds),
    )
    action_limit = min(
        preferred_action_seconds,
        maximum_action_seconds,
        maximum_action_capacity,
    )
    inference_limit = min(
        preferred_inference_seconds,
        max(0.0, turn_capacity - action_limit),
    )
    if inference_limit >= preferred_inference_seconds:
        spare = max(0.0, turn_capacity - inference_limit - action_limit)
        action_limit += min(
            maximum_action_seconds - action_limit,
            floor(spare),
        )
    feasible = bool(
        inference_limit >= minimum_inference_seconds
        and action_limit >= minimum_action_seconds
    )
    return TerminalDeadlineSlots(
        sequence=sequence,
        remaining_seconds=remaining_seconds,
        minimum_inference_seconds=minimum_inference_seconds,
        preferred_inference_seconds=preferred_inference_seconds,
        minimum_action_seconds=minimum_action_seconds,
        preferred_action_seconds=preferred_action_seconds,
        maximum_action_seconds=maximum_action_seconds,
        followup_inference_seconds=followup_inference_seconds,
        verification_seconds=verification_seconds,
        cleanup_seconds=cleanup_seconds,
        inference_limit_seconds=inference_limit,
        action_limit_seconds=action_limit,
        feasible=feasible,
        complete_sequence_feasible=feasible,
    )


def allocate_terminal_deadline_sequence(
    *,
    sequence: TerminalDeadlineSequence,
    remaining_seconds: float | None,
    minimum_inference_seconds: float,
    preferred_inference_seconds: float,
    followup_inference_seconds: float,
    work_timeout_seconds: int = 60,
    verification_timeout_seconds: int = 120,
    reconciliation_timeout_seconds: int = 60,
    cleanup_seconds: float = 0.0,
    include_current_inference: bool = True,
) -> TerminalDeadlineSlots:
    """Allocate one named completion sequence from a single source of truth.

    The current action receives a role-specific cap. Every later inference and
    action is reserved before current inference is allowed to grow, so a
    reconciliation cannot consume the work or verification slots that follow.
    """

    for name, value in (
        ("work_timeout_seconds", work_timeout_seconds),
        ("verification_timeout_seconds", verification_timeout_seconds),
        ("reconciliation_timeout_seconds", reconciliation_timeout_seconds),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    _require_nonnegative(
        "followup_inference_seconds",
        followup_inference_seconds,
    )

    if sequence is TerminalDeadlineSequence.RECONCILE_THEN_WORK_VERIFY:
        current_action = reconciliation_timeout_seconds
        reserved_inference = followup_inference_seconds * 2.0
        reserved_actions = float(
            work_timeout_seconds + verification_timeout_seconds
        )
    elif sequence is TerminalDeadlineSequence.RECONCILE_THEN_VERIFY:
        current_action = reconciliation_timeout_seconds
        reserved_inference = followup_inference_seconds
        reserved_actions = float(verification_timeout_seconds)
    elif sequence is TerminalDeadlineSequence.WORK_THEN_VERIFY:
        current_action = work_timeout_seconds
        reserved_inference = followup_inference_seconds
        reserved_actions = float(verification_timeout_seconds)
    else:
        current_action = verification_timeout_seconds
        reserved_inference = 0.0
        reserved_actions = 0.0

    if not include_current_inference:
        minimum_inference_seconds = 0.0
        preferred_inference_seconds = 0.0
    full_sequence = allocate_deadline_slots(
        remaining_seconds=remaining_seconds,
        minimum_inference_seconds=minimum_inference_seconds,
        preferred_inference_seconds=preferred_inference_seconds,
        preferred_action_seconds=current_action,
        maximum_action_seconds=current_action,
        followup_inference_seconds=reserved_inference,
        verification_seconds=reserved_actions,
        cleanup_seconds=cleanup_seconds,
        sequence=sequence,
    )
    if full_sequence.feasible or remaining_seconds is None:
        return full_sequence

    # A future repair/verification reserve is a scheduling preference, not a
    # reason to discard a current action that still fits safely.  Admit the
    # shortest current phase while preserving cleanup; the Planner will
    # refresh remaining time and select the next legal sequence afterwards.
    degraded = allocate_deadline_slots(
        remaining_seconds=remaining_seconds,
        minimum_inference_seconds=minimum_inference_seconds,
        preferred_inference_seconds=preferred_inference_seconds,
        preferred_action_seconds=current_action,
        maximum_action_seconds=current_action,
        cleanup_seconds=cleanup_seconds,
        sequence=sequence,
    )
    return replace(
        degraded,
        complete_sequence_feasible=False,
    )


@dataclass(frozen=True, slots=True)
class _TerminalDeadlineStageSpec:
    stage: TerminalDeadlineStage
    band: TerminalBudgetBand
    overhead_seconds: float
    current: bool


def allocate_profiled_terminal_deadline_sequence(
    *,
    sequence: TerminalDeadlineSequence | None,
    remaining_seconds: float | None,
    profile: TerminalDeadlineBudgetProfile,
    cleanup_seconds: float,
    provider_grace_seconds: float,
    include_current_inference: bool = True,
    emergency_mode: bool = False,
    compact_mode: bool = False,
    command_role: str | None = None,
) -> TerminalDeadlineSlots:
    """Continuously allocate one profile sequence and degrade if necessary.

    Future phases receive at least their minimum before the current turn may
    grow. Once every minimum fits, the current inference and action grow
    continuously to preferred before future phases receive preferred-time
    growth. This keeps future minima intact without starving the only turn
    that can make progress now. Above preferred, surplus grows the current
    action first and then current inference, without exceeding either maximum.
    If the complete sequence does not fit, only the current legal phase is
    considered and the Planner will select the next path after it.
    """

    _require_nonnegative("cleanup_seconds", cleanup_seconds)
    _require_nonnegative("provider_grace_seconds", provider_grace_seconds)
    inference_band = (
        profile.emergency_inference
        if emergency_mode
        else (
            profile.compact_inference
            if compact_mode or sequence is not None
            else profile.normal_inference
        )
    )
    action_stage, action_band = _current_action_band(
        sequence=sequence,
        command_role=command_role,
        profile=profile,
    )
    stages: list[_TerminalDeadlineStageSpec] = []
    if include_current_inference:
        stages.append(
            _TerminalDeadlineStageSpec(
                stage=TerminalDeadlineStage.INFERENCE,
                band=inference_band,
                overhead_seconds=(
                    profile.model_cancellation_cleanup_seconds
                ),
                current=True,
            )
        )
    stages.append(
        _TerminalDeadlineStageSpec(
            stage=action_stage,
            band=action_band,
            overhead_seconds=provider_grace_seconds,
            current=True,
        )
    )
    stages.extend(
        _future_stage_specs(
            sequence=sequence,
            profile=profile,
            provider_grace_seconds=provider_grace_seconds,
        )
    )
    full = _allocate_profiled_stages(
        sequence=sequence,
        remaining_seconds=remaining_seconds,
        cleanup_seconds=cleanup_seconds,
        stages=tuple(stages),
        complete_sequence_feasible=True,
    )
    if full.feasible or remaining_seconds is None:
        return full

    current_only = tuple(item for item in stages if item.current)
    degraded = _allocate_profiled_stages(
        sequence=sequence,
        remaining_seconds=remaining_seconds,
        cleanup_seconds=cleanup_seconds,
        stages=current_only,
        complete_sequence_feasible=False,
    )
    return degraded


def _current_action_band(
    *,
    sequence: TerminalDeadlineSequence | None,
    command_role: str | None,
    profile: TerminalDeadlineBudgetProfile,
) -> tuple[TerminalDeadlineStage, TerminalBudgetBand]:
    if sequence in {
        TerminalDeadlineSequence.RECONCILE_THEN_WORK_VERIFY,
        TerminalDeadlineSequence.RECONCILE_THEN_VERIFY,
    }:
        return (
            TerminalDeadlineStage.INSPECT,
            profile.reconciliation_inspection,
        )
    if sequence is TerminalDeadlineSequence.DIRECT_VERIFY:
        return TerminalDeadlineStage.VERIFY, profile.verification
    if sequence is TerminalDeadlineSequence.WORK_THEN_VERIFY:
        return TerminalDeadlineStage.WORK, profile.final_work
    if command_role == "inspect":
        return TerminalDeadlineStage.INSPECT, profile.normal_inspection
    if command_role == "verify":
        return TerminalDeadlineStage.VERIFY, profile.verification
    return TerminalDeadlineStage.WORK, profile.normal_work


def _future_stage_specs(
    *,
    sequence: TerminalDeadlineSequence | None,
    profile: TerminalDeadlineBudgetProfile,
    provider_grace_seconds: float,
) -> tuple[_TerminalDeadlineStageSpec, ...]:
    inference = _TerminalDeadlineStageSpec(
        stage=TerminalDeadlineStage.FOLLOWUP_INFERENCE,
        band=profile.convergence_inference,
        overhead_seconds=profile.model_cancellation_cleanup_seconds,
        current=False,
    )
    work = _TerminalDeadlineStageSpec(
        stage=TerminalDeadlineStage.WORK,
        band=profile.final_work,
        overhead_seconds=provider_grace_seconds,
        current=False,
    )
    verify = _TerminalDeadlineStageSpec(
        stage=TerminalDeadlineStage.VERIFY,
        band=profile.verification,
        overhead_seconds=provider_grace_seconds,
        current=False,
    )
    if sequence is TerminalDeadlineSequence.RECONCILE_THEN_WORK_VERIFY:
        return inference, work, inference, verify
    if sequence in {
        TerminalDeadlineSequence.RECONCILE_THEN_VERIFY,
        TerminalDeadlineSequence.WORK_THEN_VERIFY,
    }:
        return inference, verify
    if sequence is TerminalDeadlineSequence.DIRECT_VERIFY:
        return ()
    return inference, verify


def _allocate_profiled_stages(
    *,
    sequence: TerminalDeadlineSequence | None,
    remaining_seconds: float | None,
    cleanup_seconds: float,
    stages: tuple[_TerminalDeadlineStageSpec, ...],
    complete_sequence_feasible: bool,
) -> TerminalDeadlineSlots:
    current_inference = next(
        (
            item
            for item in stages
            if item.current and item.stage is TerminalDeadlineStage.INFERENCE
        ),
        None,
    )
    current_action = next(
        (
            item
            for item in stages
            if item.current and item.stage is not TerminalDeadlineStage.INFERENCE
        ),
        None,
    )
    assert current_action is not None
    if remaining_seconds is None:
        return TerminalDeadlineSlots(
            sequence=sequence,
            remaining_seconds=None,
            minimum_inference_seconds=(
                0.0 if current_inference is None else current_inference.band.minimum_seconds
            ),
            preferred_inference_seconds=(
                0.0 if current_inference is None else current_inference.band.preferred_seconds
            ),
            minimum_action_seconds=max(1, floor(current_action.band.minimum_seconds)),
            preferred_action_seconds=max(1, floor(current_action.band.preferred_seconds)),
            maximum_action_seconds=max(1, floor(current_action.band.maximum_seconds)),
            followup_inference_seconds=0.0,
            verification_seconds=0.0,
            cleanup_seconds=cleanup_seconds,
            inference_limit_seconds=None,
            action_limit_seconds=None,
            feasible=True,
            complete_sequence_feasible=True,
        )
    _require_nonnegative("remaining_seconds", remaining_seconds)
    usable = max(0.0, remaining_seconds - cleanup_seconds)
    minimum_total = sum(
        item.band.minimum_seconds + item.overhead_seconds for item in stages
    )
    feasible = usable >= minimum_total
    allocated = [item.band.minimum_seconds for item in stages]
    if feasible:
        surplus = usable - minimum_total
        current_indexes = tuple(
            index for index, item in enumerate(stages) if item.current
        )
        future_indexes = tuple(
            index for index, item in enumerate(stages) if not item.current
        )

        def grow_together(indexes: tuple[int, ...]) -> None:
            nonlocal surplus
            gaps = tuple(
                max(
                    0.0,
                    stages[index].band.preferred_seconds - allocated[index],
                )
                for index in indexes
            )
            total_gap = sum(gaps)
            if surplus <= 0.0 or total_gap <= 0.0:
                return
            growth = min(surplus, total_gap)
            ratio = growth / total_gap
            for index, gap in zip(indexes, gaps, strict=True):
                allocated[index] += ratio * gap
            surplus -= growth

        grow_together(current_indexes)
        grow_together(future_indexes)

        growth_order = [
            index
            for index, item in enumerate(stages)
            if item.current and item.stage is not TerminalDeadlineStage.INFERENCE
        ] + [
            index
            for index, item in enumerate(stages)
            if item.current and item.stage is TerminalDeadlineStage.INFERENCE
        ]
        for index in growth_order:
            growth = min(
                surplus,
                stages[index].band.maximum_seconds - allocated[index],
            )
            allocated[index] += growth
            surplus -= growth
            if surplus <= 0.0:
                break

    stage_allocations = tuple(
        TerminalDeadlineStageAllocation(
            stage=item.stage,
            minimum_seconds=item.band.minimum_seconds,
            preferred_seconds=item.band.preferred_seconds,
            maximum_seconds=item.band.maximum_seconds,
            overhead_seconds=item.overhead_seconds,
            allocated_seconds=value,
            current=item.current,
        )
        for item, value in zip(stages, allocated, strict=True)
    )
    inference_limit = (
        0.0
        if current_inference is None
        else next(
            item.allocated_seconds
            for item in stage_allocations
            if item.current and item.stage is TerminalDeadlineStage.INFERENCE
        )
    )
    action_limit = floor(
        next(
            item.allocated_seconds
            for item in stage_allocations
            if item.current and item.stage is not TerminalDeadlineStage.INFERENCE
        )
    )
    future_inference = sum(
        item.allocated_seconds + item.overhead_seconds
        for item in stage_allocations
        if not item.current
        and item.stage is TerminalDeadlineStage.FOLLOWUP_INFERENCE
    )
    future_actions = sum(
        item.allocated_seconds + item.overhead_seconds
        for item in stage_allocations
        if not item.current
        and item.stage is not TerminalDeadlineStage.FOLLOWUP_INFERENCE
    )
    return TerminalDeadlineSlots(
        sequence=sequence,
        remaining_seconds=remaining_seconds,
        minimum_inference_seconds=(
            0.0 if current_inference is None else current_inference.band.minimum_seconds
        ),
        preferred_inference_seconds=(
            0.0 if current_inference is None else current_inference.band.preferred_seconds
        ),
        minimum_action_seconds=max(1, floor(current_action.band.minimum_seconds)),
        preferred_action_seconds=max(1, floor(current_action.band.preferred_seconds)),
        maximum_action_seconds=max(1, floor(current_action.band.maximum_seconds)),
        followup_inference_seconds=future_inference,
        verification_seconds=future_actions,
        cleanup_seconds=cleanup_seconds,
        inference_limit_seconds=inference_limit,
        action_limit_seconds=action_limit,
        feasible=bool(feasible and action_limit >= 1),
        complete_sequence_feasible=bool(
            feasible and complete_sequence_feasible
        ),
        stage_allocations=stage_allocations,
    )


def _require_nonnegative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
