"""App-native subscription polling built above Runtime source Tools."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from hashlib import sha256
from threading import Event, Thread
from typing import Callable

from applications.personal_knowledge.contracts import SubscriptionStore
from applications.personal_knowledge.models import (
    SourceSubscription,
    SubscriptionKind,
    utc_now,
)
from applications.personal_knowledge.service import PersonalKnowledgeService
from applications.personal_knowledge.source_tools import SourceToolRunner


@dataclass(frozen=True)
class SubscriptionPollResult:
    subscription_id: str
    changed: bool
    review_id: str | None = None
    error: str | None = None


class SubscriptionPoller:
    """One polling pass; scheduling stays an application concern."""

    def __init__(
        self,
        *,
        store: SubscriptionStore,
        source_tools: SourceToolRunner,
        workflow: PersonalKnowledgeService,
    ) -> None:
        self._store = store
        self._source_tools = source_tools
        self._workflow = workflow

    async def poll_once(
        self,
        *,
        force: bool = False,
    ) -> tuple[SubscriptionPollResult, ...]:
        now = utc_now()
        subscriptions = await self._store.list_subscriptions(enabled_only=True)
        due = tuple(
            item for item in subscriptions if force or item.next_poll_at <= now
        )
        results: list[SubscriptionPollResult] = []
        for subscription in due:
            results.append(await self._poll(subscription))
        return tuple(results)

    async def _poll(
        self,
        subscription: SourceSubscription,
    ) -> SubscriptionPollResult:
        try:
            if subscription.kind is SubscriptionKind.VIDEO:
                fetched, _ = await self._source_tools.acquire_video_transcript(
                    subscription.url
                )
            else:
                fetched, _ = await self._source_tools.acquire_web_text(
                    subscription.url
                )
            digest = sha256(fetched.text.encode("utf-8")).hexdigest()
            changed = digest != subscription.last_content_hash
            review_id: str | None = None
            if changed:
                review = await self._workflow.start_review_from_fetched(
                    fetched,
                    user_note=f"自动订阅：{subscription.name}",
                )
                review_id = str(review.review.review_id)
            updated = subscription.after_poll(
                content_hash=digest,
                error=None,
            )
            await self._store.save_subscription(
                updated,
                expected_revision=subscription.revision,
            )
            return SubscriptionPollResult(
                subscription_id=str(subscription.subscription_id),
                changed=changed,
                review_id=review_id,
            )
        except Exception as exc:
            detail = str(exc) or exc.__class__.__name__
            updated = subscription.after_poll(content_hash=None, error=detail)
            await self._store.save_subscription(
                updated,
                expected_revision=subscription.revision,
            )
            return SubscriptionPollResult(
                subscription_id=str(subscription.subscription_id),
                changed=False,
                error=detail,
            )


class InProcessSubscriptionScheduler:
    """Small independent scheduler suitable for the local single-user app."""

    def __init__(
        self,
        poller: SubscriptionPoller,
        *,
        tick_seconds: float = 30.0,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._poller = poller
        self._tick_seconds = tick_seconds
        self._on_error = on_error
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(
            target=lambda: asyncio.run(self._run()),
            name="personal-knowledge-subscriptions",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self._tick_seconds + 1.0))

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self._poller.poll_once()
            except Exception as exc:
                if self._on_error is not None:
                    self._on_error(exc)
            await asyncio.to_thread(self._stop.wait, self._tick_seconds)

