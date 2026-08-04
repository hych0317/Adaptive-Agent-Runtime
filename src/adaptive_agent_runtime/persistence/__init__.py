"""Opt-in durable adapters for independent Runtime module contracts."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from adaptive_agent_runtime.decisioning.models import DecisionCheckpoint

from adaptive_agent_runtime.persistence.context_memory import (
    SQLiteContextArchive,
    SQLiteContextStore,
    SQLiteMemoryStore,
)
from adaptive_agent_runtime.persistence.core import (
    SQLiteStateStore,
    SQLiteTraceSink,
)
from adaptive_agent_runtime.persistence.errors import (
    PersistenceConflictError,
    PersistenceError,
    PersistenceSchemaError,
)
from adaptive_agent_runtime.persistence.governance import (
    SQLiteAuthorizationConsumptionStore,
    SQLiteHumanReviewService,
)
from adaptive_agent_runtime.persistence.evolution import SQLiteEvolutionStore
from adaptive_agent_runtime.persistence.decisioning import (
    SQLiteDecisionCheckpointStore,
)
from adaptive_agent_runtime.persistence.orchestration import SQLiteTaskGraphStore
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase


RequestCheckpointT = TypeVar("RequestCheckpointT", bound=BaseModel)
ProposalCheckpointT = TypeVar("ProposalCheckpointT", bound=BaseModel)
EffectCheckpointT = TypeVar("EffectCheckpointT", bound=BaseModel)


class SQLitePersistence:
    """Composition root exposing one database through narrow module stores."""

    def __init__(self, path: str | Path) -> None:
        self.database = SQLiteDatabase(path)
        self.state_store = SQLiteStateStore(self.database)
        self.trace_sink = SQLiteTraceSink(self.database)
        self.task_graph_store = SQLiteTaskGraphStore(self.database)
        self.context_store = SQLiteContextStore(self.database)
        self.context_archive = SQLiteContextArchive(self.database)
        self.memory_store = SQLiteMemoryStore(self.database)
        self.human_review_service = SQLiteHumanReviewService(self.database)
        self.authorization_store = SQLiteAuthorizationConsumptionStore(
            self.database
        )
        self.evolution_store = SQLiteEvolutionStore(self.database)
        self.runtime_configuration_store = self.evolution_store
        self.replay_case_store = self.evolution_store

    def create_decision_checkpoint_store(
        self,
        checkpoint_type: type[
            DecisionCheckpoint[
                RequestCheckpointT,
                ProposalCheckpointT,
                EffectCheckpointT,
            ]
        ],
    ) -> SQLiteDecisionCheckpointStore[
        RequestCheckpointT,
        ProposalCheckpointT,
        EffectCheckpointT,
    ]:
        """Create a typed store without introducing a global Draft registry."""

        return SQLiteDecisionCheckpointStore(self.database, checkpoint_type)

    def close(self) -> None:
        self.database.close()

    def __enter__(self) -> SQLitePersistence:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


__all__ = [
    "PersistenceConflictError",
    "PersistenceError",
    "PersistenceSchemaError",
    "SQLiteContextArchive",
    "SQLiteContextStore",
    "SQLiteDatabase",
    "SQLiteDecisionCheckpointStore",
    "SQLiteEvolutionStore",
    "SQLiteMemoryStore",
    "SQLiteAuthorizationConsumptionStore",
    "SQLiteHumanReviewService",
    "SQLitePersistence",
    "SQLiteStateStore",
    "SQLiteTaskGraphStore",
    "SQLiteTraceSink",
]
