"""Dependency-light localhost HTTP application for personal knowledge workflows."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import fields, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import traceback
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import UUID

from pydantic import ValidationError
from adaptive_agent_runtime.persistence import SQLitePersistence
from adaptive_agent_runtime.llm.capabilities import ArtifactGenerationCapability
from adaptive_agent_runtime.llm.capabilities import MemoryExtractionCapability
from adaptive_agent_runtime.tool_ecosystem.errors import ToolIntegrationError

from applications.personal_knowledge.cognition import (
    ChunkedKnowledgeSynthesizer,
    DeterministicKnowledgeSynthesizer,
    RuntimeKnowledgeSynthesizer,
    KnowledgeSynthesizer,
    SwitchableKnowledgeSynthesizer,
)
from applications.personal_knowledge.errors import (
    KnowledgeConflictError,
    KnowledgeNotFoundError,
    PersonalKnowledgeError,
)
from applications.personal_knowledge.models import (
    CategoryChangeType,
    ConfirmedCategoryChange,
    SourceSubscription,
    SubscriptionKind,
)
from applications.personal_knowledge.llm_runtime import (
    PersonalKnowledgeLLMDeployment,
    PersonalKnowledgeLLMManager,
)
from applications.personal_knowledge.persistence import SQLitePersonalKnowledgeStore
from applications.personal_knowledge.preference_memory import (
    PreferenceMemoryService,
    PreferenceSuggestion,
    RuntimePreferenceInferenceService,
)
from applications.personal_knowledge.retrieval import (
    DeterministicKnowledgeQuestionAnswerer,
    KnowledgeRetrievalService,
    KnowledgeQuestionAnswerer,
    RuntimeKnowledgeQuestionAnswerer,
    SwitchableKnowledgeQuestionAnswerer,
)
from applications.personal_knowledge.service import (
    EditableKnowledgeConfirmation,
    PersonalKnowledgeService,
)
from applications.personal_knowledge.source_tools import (
    SourceToolRunner,
    build_source_tool_stack,
)
from applications.personal_knowledge.subscriptions import (
    InProcessSubscriptionScheduler,
    SubscriptionPoller,
)


_STATIC_ROOT = Path(__file__).with_name("static")
_DEFAULT_LLM_CONFIG = Path(__file__).resolve().parents[2] / "config" / "llm.toml"


def _jsonable(value: object) -> object:
    dump = getattr(value, "model_dump", None)
    if dump is not None:
        return dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _jsonable(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


class PersonalKnowledgeApplication:
    """Local application composition root; Runtime Core remains unchanged."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        runtime_database_path: str | Path | None = None,
        generation_capability: ArtifactGenerationCapability | None = None,
        memory_extraction_capability: MemoryExtractionCapability | None = None,
        llm_manager: PersonalKnowledgeLLMManager | None = None,
        video_script_path: str | Path | None = None,
    ) -> None:
        self.store = SQLitePersonalKnowledgeStore(database_path)
        knowledge_path = Path(database_path)
        resolved_runtime_path = runtime_database_path or (
            ":memory:"
            if str(database_path) == ":memory:"
            else knowledge_path.with_name("runtime.sqlite3")
        )
        self.runtime_persistence = SQLitePersistence(resolved_runtime_path)
        self.preferences = PreferenceMemoryService(
            self.runtime_persistence.memory_store
        )
        self.llm_manager = llm_manager
        self._llm_deployment: PersonalKnowledgeLLMDeployment | None = None
        manager_deployment = llm_manager.load_active() if llm_manager else None
        if manager_deployment is not None:
            generation_capability = manager_deployment.generator
            memory_extraction_capability = manager_deployment.memory_extractor
            self._llm_deployment = manager_deployment
        self.preference_inference = (
            RuntimePreferenceInferenceService(memory_extraction_capability)
            if memory_extraction_capability is not None
            else None
        )
        synthesizer: KnowledgeSynthesizer
        answerer: KnowledgeQuestionAnswerer
        if generation_capability is not None:
            synthesizer = RuntimeKnowledgeSynthesizer(generation_capability)
            answerer = RuntimeKnowledgeQuestionAnswerer(generation_capability)
        else:
            synthesizer = DeterministicKnowledgeSynthesizer()
            answerer = DeterministicKnowledgeQuestionAnswerer()
        self.synthesizer = SwitchableKnowledgeSynthesizer(synthesizer)
        self.answerer = SwitchableKnowledgeQuestionAnswerer(answerer)
        self.agent_mode = (
            self._deployment_label(manager_deployment)
            if manager_deployment is not None
            else ("runtime" if generation_capability is not None else "offline")
        )
        self.tool_stack = build_source_tool_stack(
            video_script_path=video_script_path,
        )
        self.source_tools = SourceToolRunner(self.tool_stack)
        self.workflow = PersonalKnowledgeService(
            store=self.store,
            source_tools=self.source_tools,
            synthesizer=ChunkedKnowledgeSynthesizer(self.synthesizer),
        )
        self.retrieval = KnowledgeRetrievalService(
            store=self.store,
            answerer=self.answerer,
        )
        self.poller = SubscriptionPoller(
            store=self.store,
            source_tools=self.source_tools,
            workflow=self.workflow,
        )
        self.scheduler = InProcessSubscriptionScheduler(self.poller)

    async def state(self) -> dict[str, object]:
        all_entries = await self.store.list_entries(include_trashed=True)
        return {
            "categories": _jsonable(await self.store.list_categories()),
            "entries": _jsonable(
                tuple(item for item in all_entries if item.status.value == "active")
            ),
            "trash": _jsonable(
                tuple(item for item in all_entries if item.status.value == "trashed")
            ),
            "reviews": _jsonable(await self.store.list_reviews()),
            "subscriptions": _jsonable(await self.store.list_subscriptions()),
            "preferences": _jsonable(await self.preferences.recall()),
            "agent_mode": self.agent_mode,
        }

    async def ingest(self, payload: dict[str, Any]) -> object:
        kind = payload.get("kind")
        value = payload.get("value")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("value is required")
        if kind == "url":
            return await self.workflow.ingest_url(
                value,
                user_note=self._optional_text(payload.get("note")),
            )
        if kind == "video":
            return await self.workflow.ingest_video(
                value,
                user_note=self._optional_text(payload.get("note")),
            )
        if kind == "idea":
            return await self.workflow.capture_idea(
                value,
                title=self._optional_text(payload.get("title")),
            )
        if kind == "text":
            return await self.workflow.ingest_text(
                value,
                title=self._optional_text(payload.get("title")),
                user_note=self._optional_text(payload.get("note")),
            )
        raise ValueError("kind must be text, idea, url, or video")

    async def confirm_category(self, payload: dict[str, Any]) -> object:
        if payload.get("confirmed") is not True:
            raise ValueError("category changes require explicit confirmation")
        change_type = CategoryChangeType(str(payload.get("change_type")))
        category_id = self._optional_uuid(payload.get("category_id"))
        change = ConfirmedCategoryChange(
            change_type=change_type,
            category_id=category_id or UUID(int=0),
            name=self._optional_text(payload.get("name")),
            target_category_id=self._optional_uuid(payload.get("target_category_id")),
            expected_revision=payload.get("expected_revision"),
        )
        if change_type is CategoryChangeType.CREATE and category_id is None:
            change = ConfirmedCategoryChange(
                change_type=change_type,
                name=self._optional_text(payload.get("name")),
            )
        return await self.store.apply_category_change(change)

    def llm_settings_view(self) -> dict[str, object]:
        if self.llm_manager is None:
            return {
                "enabled": False,
                "activeTarget": None,
                "defaultTarget": None,
                "targets": [],
                "inference": self.agent_mode,
            }
        return {
            "enabled": True,
            **self.llm_manager.describe(),
            "inference": self.agent_mode,
        }

    def configure_llm(
        self,
        target_name: str,
        *,
        api_key: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, object]:
        if self.llm_manager is None:
            raise RuntimeError("LLM target settings are not configured")
        deployment = self.llm_manager.activate(
            target_name,
            api_key=api_key,
            model_id=model_id,
        )
        self._use_llm_deployment(deployment)
        return self.llm_settings_view()

    async def discover_llm_models(
        self,
        target_name: str,
        *,
        api_key: str | None = None,
    ) -> dict[str, object]:
        if self.llm_manager is None:
            raise RuntimeError("LLM target settings are not configured")
        return await self.llm_manager.discover_models(
            target_name,
            api_key=api_key,
        )

    async def probe_llm(
        self,
        *,
        target_name: str | None = None,
        api_key: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, object]:
        if self.llm_manager is None:
            raise RuntimeError("LLM target settings are not configured")
        if target_name is None:
            result = await self.llm_manager.probe_active()
            result["activated"] = False
            result["inference"] = self.agent_mode
            return result
        result, _ = await self.llm_manager.probe_selection(
            target_name,
            api_key=api_key,
            model_id=model_id,
        )
        activated = result["availability"] == "available"
        if activated:
            deployment = self.llm_manager.activate(
                target_name,
                api_key=api_key,
                model_id=model_id,
            )
            self._use_llm_deployment(deployment)
        result["activated"] = activated
        result["inference"] = self.agent_mode
        result["settings"] = self.llm_settings_view()
        return result

    def _use_llm_deployment(
        self,
        deployment: PersonalKnowledgeLLMDeployment,
    ) -> None:
        self._llm_deployment = deployment
        self.synthesizer.use(RuntimeKnowledgeSynthesizer(deployment.generator))
        self.answerer.use(RuntimeKnowledgeQuestionAnswerer(deployment.generator))
        self.preference_inference = (
            RuntimePreferenceInferenceService(deployment.memory_extractor)
            if deployment.memory_extractor is not None
            else None
        )
        self.agent_mode = self._deployment_label(deployment)

    @staticmethod
    def _deployment_label(
        deployment: PersonalKnowledgeLLMDeployment | None,
    ) -> str:
        if deployment is None:
            return "offline"
        return f"{deployment.model_id} · {deployment.target_name}"

    async def create_subscription(self, payload: dict[str, Any]) -> object:
        subscription = SourceSubscription(
            name=str(payload.get("name", "")).strip(),
            kind=SubscriptionKind(str(payload.get("kind"))),
            url=str(payload.get("url", "")).strip(),
            interval_minutes=int(payload.get("interval_minutes", 60)),
        )
        await self.store.save_subscription(subscription, expected_revision=None)
        return subscription

    async def confirm_knowledge(
        self,
        confirmation: EditableKnowledgeConfirmation,
    ) -> dict[str, object]:
        review = await self.store.load_review(confirmation.review_id)
        if review is None:
            raise KnowledgeNotFoundError("review was not found")
        proposal = next(
            (
                item
                for item in review.proposals
                if item.proposal_id == confirmation.proposal_id
            ),
            None,
        )
        if proposal is None:
            raise KnowledgeNotFoundError("proposal does not belong to this review")
        entry = await self.workflow.confirm_publication(confirmation)
        suggestions: tuple[PreferenceSuggestion, ...] = ()
        if self.preference_inference is not None:
            observations: list[dict[str, object]] = [
                {
                    "kind": "review_instruction",
                    "text": message.content,
                }
                for message in review.conversation
                if message.role.value == "user"
            ]
            observations.append(
                {
                    "kind": "edit_pattern",
                    "title_changed": proposal.title != confirmation.title,
                    "body_changed": proposal.body != confirmation.body,
                    "tags_changed": proposal.tags != confirmation.tags,
                    "citation_count": len(confirmation.citations),
                }
            )
            suggestions = await self.preference_inference.infer(
                observations=observations,
                existing=await self.preferences.recall(),
                evidence_reference=f"review:{review.review_id}",
            )
        return {"entry": entry, "preference_suggestions": suggestions}

    @staticmethod
    def _optional_text(value: object) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _optional_uuid(value: object) -> UUID | None:
        return UUID(value) if isinstance(value, str) and value else None

    def close(self) -> None:
        self.store.close()
        self.runtime_persistence.close()


class _KnowledgeRequestHandler(BaseHTTPRequestHandler):
    server_version = "PersonalKnowledge/0.1"
    app: PersonalKnowledgeApplication

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(_STATIC_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            self._run(self.app.state())
            return
        if parsed.path == "/api/llm/settings":
            self._run(self._completed(self.app.llm_settings_view()))
            return
        if parsed.path == "/api/search":
            query = parse_qs(parsed.query)
            term = query.get("q", [""])[0]
            include_archive = query.get("archive", ["false"])[0] == "true"
            self._run(
                self.app.retrieval.search(term, include_archive=include_archive)
            )
            return
        self._error(HTTPStatus.NOT_FOUND, "route not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            self._validate_local_request()
            payload = self._read_json()
            if parsed.path == "/api/ingest":
                self._run(self.app.ingest(payload), status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/categories/confirm":
                self._run(
                    self.app.confirm_category(payload),
                    status=HTTPStatus.CREATED,
                )
                return
            if parsed.path == "/api/subscriptions":
                self._run(
                    self.app.create_subscription(payload),
                    status=HTTPStatus.CREATED,
                )
                return
            if parsed.path == "/api/subscriptions/poll-once":
                self._run(self.app.poller.poll_once(force=True))
                return
            if parsed.path == "/api/preferences/explicit":
                key = payload.get("key")
                value = payload.get("value")
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError("preference key and value are required")
                self._run(
                    self.app.preferences.record_explicit(key=key, value=value)
                )
                return
            if parsed.path == "/api/preferences/suggest":
                suggestion = self.app.preferences.suggest_inferred(
                    key=str(payload.get("key", "")),
                    value=str(payload.get("value", "")),
                    rationale=str(payload.get("rationale", "")),
                )
                self._run(self._completed(suggestion))
                return
            if parsed.path == "/api/preferences/confirm":
                suggestion = PreferenceSuggestion.model_validate(payload)
                self._run(self.app.preferences.confirm_suggestion(suggestion))
                return
            if parsed.path == "/api/answer":
                question = payload.get("question")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("question is required")
                self._run(
                    self.app.retrieval.answer(
                        question,
                        include_archive=payload.get("include_archive") is True,
                    )
                )
                return
            if parsed.path == "/api/llm/settings":
                target = payload.get("target")
                api_key = payload.get("apiKey")
                model = payload.get("model")
                if not isinstance(target, str) or not target.strip():
                    raise ValueError("target is required")
                if api_key is not None and not isinstance(api_key, str):
                    raise ValueError("apiKey must be a string")
                if model is not None and not isinstance(model, str):
                    raise ValueError("model must be a string")
                result = self.app.configure_llm(
                    target,
                    api_key=(api_key or None),
                    model_id=(model or None),
                )
                self._run(self._completed(result))
                return
            if parsed.path == "/api/llm/models":
                target = payload.get("target")
                api_key = payload.get("apiKey")
                if not isinstance(target, str) or not target.strip():
                    raise ValueError("target is required")
                if api_key is not None and not isinstance(api_key, str):
                    raise ValueError("apiKey must be a string")
                self._run(
                    self.app.discover_llm_models(
                        target,
                        api_key=(api_key or None),
                    )
                )
                return
            if parsed.path == "/api/llm/probe":
                target = payload.get("target")
                api_key = payload.get("apiKey")
                model = payload.get("model")
                if target is not None and not isinstance(target, str):
                    raise ValueError("target must be a string")
                if api_key is not None and not isinstance(api_key, str):
                    raise ValueError("apiKey must be a string")
                if model is not None and not isinstance(model, str):
                    raise ValueError("model must be a string")
                self._run(
                    self.app.probe_llm(
                        target_name=(target or None),
                        api_key=(api_key or None),
                        model_id=(model or None),
                    )
                )
                return
            parts = tuple(item for item in parsed.path.split("/") if item)
            if len(parts) == 4 and parts[:2] == ("api", "reviews"):
                review_id = UUID(parts[2])
                if parts[3] == "messages":
                    content = payload.get("content")
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("content is required")
                    self._run(
                        self.app.workflow.append_review_message(
                            review_id,
                            content=content,
                        )
                    )
                    return
                if parts[3] == "confirm":
                    payload["review_id"] = str(review_id)
                    confirmation = EditableKnowledgeConfirmation.model_validate(payload)
                    self._run(
                        self.app.confirm_knowledge(confirmation),
                        status=HTTPStatus.CREATED,
                    )
                    return
            if len(parts) == 4 and parts[:2] == ("api", "entries"):
                entry_id = UUID(parts[2])
                if parts[3] == "review":
                    self._run(
                        self.app.workflow.start_entry_revision(entry_id),
                        status=HTTPStatus.CREATED,
                    )
                    return
                if parts[3] == "status":
                    if payload.get("confirmed") is not True:
                        raise ValueError("status changes require explicit confirmation")
                    action = payload.get("action")
                    if action not in {"delete", "restore"}:
                        raise ValueError("action must be delete or restore")
                    revision = payload.get("expected_revision")
                    if not isinstance(revision, int):
                        raise ValueError("expected_revision is required")
                    self._run(
                        self.app.workflow.confirm_status_change(
                            entry_id=entry_id,
                            expected_revision=revision,
                            restore=action == "restore",
                        )
                    )
                    return
            self._error(HTTPStatus.NOT_FOUND, "route not found")
        except (ValueError, ValidationError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except KnowledgeNotFoundError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except (KnowledgeConflictError, PersonalKnowledgeError) as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except ToolIntegrationError as exc:
            traceback.print_exc()
            self._error(HTTPStatus.BAD_GATEWAY, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"request failed: {exc.__class__.__name__}",
            )

    def _read_json(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0]
        if content_type != "application/json":
            raise ValueError("API requests require application/json")
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > 2 * 1024 * 1024:
            raise ValueError("request body must be between 1 byte and 2 MiB")
        decoded = json.loads(self.rfile.read(content_length).decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("request body must be a JSON object")
        return decoded

    def _validate_local_request(self) -> None:
        if self.headers.get("Sec-Fetch-Site", "").casefold() == "cross-site":
            raise ValueError("cross-site requests are not accepted")
        origin = self.headers.get("Origin")
        if origin is None:
            return
        parsed = urlparse(origin)
        if parsed.scheme not in {"http", "https"} or parsed.netloc != self.headers.get(
            "Host"
        ):
            raise ValueError("request Origin does not match this local application")

    @staticmethod
    async def _completed(value: object) -> object:
        return value

    def _run(
        self,
        awaitable: Any,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        try:
            result = asyncio.run(awaitable)
        except (ValueError, ValidationError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except KnowledgeNotFoundError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
            return
        except (KnowledgeConflictError, PersonalKnowledgeError) as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
            return
        except ToolIntegrationError as exc:
            traceback.print_exc()
            self._error(HTTPStatus.BAD_GATEWAY, str(exc))
            return
        except Exception as exc:
            traceback.print_exc()
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"request failed: {exc.__class__.__name__}",
            )
            return
        encoded = json.dumps(
            _jsonable(result),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            self._error(HTTPStatus.NOT_FOUND, "asset not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status: HTTPStatus, message: str) -> None:
        encoded = json.dumps(
            {"error": message}, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class PersonalKnowledgeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def build_http_server(
    app: PersonalKnowledgeApplication,
    *,
    host: str = "127.0.0.1",
    port: int = 8775,
) -> PersonalKnowledgeHTTPServer:
    handler = type(
        "PersonalKnowledgeRequestHandler",
        (_KnowledgeRequestHandler,),
        {"app": app},
    )
    return PersonalKnowledgeHTTPServer((host, port), handler)


def serve(
    *,
    database_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8775,
    llm_config: str | Path = _DEFAULT_LLM_CONFIG,
    llm_target: str | None = None,
    video_script: str | Path | None = None,
) -> None:
    app = PersonalKnowledgeApplication(
        database_path,
        llm_manager=PersonalKnowledgeLLMManager(
            llm_config,
            active_target=llm_target,
        ),
        video_script_path=video_script,
    )
    server = build_http_server(app, host=host, port=port)
    try:
        app.scheduler.start()
        print(f"Personal Knowledge is available at http://{host}:{port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.scheduler.stop()
        server.server_close()
        app.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the personal knowledge app")
    parser.add_argument("--database", default="data/knowledge.sqlite3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--llm-config", type=Path, default=_DEFAULT_LLM_CONFIG)
    parser.add_argument("--llm-target")
    parser.add_argument(
        "--video-script",
        type=Path,
        help=(
            "Hermes-compatible fetch_video_transcript.py; defaults to "
            "PERSONAL_KNOWLEDGE_VIDEO_SCRIPT or a sibling Hermes repository"
        ),
    )
    parser.add_argument(
        "--poll-once",
        action="store_true",
        help="poll every enabled subscription once and exit",
    )
    args = parser.parse_args()
    if args.poll_once:
        app = PersonalKnowledgeApplication(args.database)
        try:
            result = asyncio.run(app.poller.poll_once(force=True))
            print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
        finally:
            app.close()
        return
    serve(
        database_path=args.database,
        host=args.host,
        port=args.port,
        llm_config=args.llm_config,
        llm_target=args.llm_target,
        video_script=args.video_script,
    )


if __name__ == "__main__":
    main()
