"""Application composition for isolated ecommerce scenario state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from applications.ecommerce_support.payment_gateway import FakePaymentGateway
from applications.ecommerce_support.persistence import EcommerceSQLiteStore
from applications.ecommerce_support.policies import EcommercePolicy
from applications.ecommerce_support.tools import EcommerceTools


@dataclass(frozen=True)
class EcommerceComposition:
    store: EcommerceSQLiteStore
    gateway: FakePaymentGateway
    tools: EcommerceTools

    def close(self) -> None:
        self.gateway.close()
        self.store.close()

    def __enter__(self) -> EcommerceComposition:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def compose_ecommerce_support(
    directory: str | Path,
    *,
    clock: Callable[[], datetime],
    policy: EcommercePolicy | None = None,
) -> EcommerceComposition:
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    store = EcommerceSQLiteStore(root / "authoritative.sqlite3")
    gateway = FakePaymentGateway(root / "gateway.sqlite3")
    tools = EcommerceTools(
        store=store,
        gateway=gateway,
        policy=policy or EcommercePolicy(),
        clock=clock,
    )
    return EcommerceComposition(store=store, gateway=gateway, tools=tools)
