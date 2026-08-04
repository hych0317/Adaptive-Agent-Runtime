"""Application-owned personal knowledge failures."""


class PersonalKnowledgeError(Exception):
    """Base error for the personal knowledge application."""


class KnowledgeNotFoundError(PersonalKnowledgeError):
    """Raised when a requested domain object does not exist."""


class KnowledgeConflictError(PersonalKnowledgeError):
    """Raised for stale, duplicate, or identity-conflicting writes."""


class KnowledgeInvariantError(PersonalKnowledgeError):
    """Raised when a confirmed change would violate domain rules."""

