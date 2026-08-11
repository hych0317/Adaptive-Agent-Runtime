"""Pure wall-clock slot allocation for terminal planning.

The allocator keeps inference, the current action, a follow-up verification
turn, and final cleanup in one calculation.  It deliberately knows nothing
about Planner state so the same arithmetic can be reused before inference and
again by the post-inference Action Gate.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    return allocate_deadline_slots(
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


def _require_nonnegative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
