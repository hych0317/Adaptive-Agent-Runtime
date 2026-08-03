"""Runtime Governance errors."""


class GovernanceError(RuntimeError):
    """Base error for Governance operations."""


class GovernanceInvariantError(GovernanceError):
    """Raised when request, decision, or authorization identities conflict."""


class ReviewNotFoundError(GovernanceError):
    """Raised when a Human Review request does not exist."""


class InvalidReviewTransitionError(GovernanceError):
    """Raised when a resolved Human Review is changed again."""


class AuthorizationVerificationError(GovernanceError):
    """Raised when authorization is not bound to the requested operation."""


class AuthorizationReplayError(GovernanceError):
    """Raised when an authorization has already been reserved or consumed."""


class AuthorizationUseConflictError(GovernanceError):
    """Raised when an authorization-use snapshot is stale."""


class GovernedOperationError(GovernanceError):
    """Raised when an authorized operation fails at its apply point."""
