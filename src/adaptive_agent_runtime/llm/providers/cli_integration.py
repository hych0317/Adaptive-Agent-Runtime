"""Local CLI discovery and bounded readiness probes.

This module is a Python adaptation of the executable resolution, launch
resolution, version probing, and authentication probing implemented by Open
Design's ``apps/daemon/src/runtimes`` layer.  It deliberately stops at the
backend boundary: cognitive capability contracts and Runtime policy remain
owned by Adaptive Agent Runtime.

Source: nexu-io/open-design (Apache-2.0), files adapted are recorded in
``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import asyncio
import os
import platform
import re
import shutil
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from adaptive_agent_runtime.llm.providers.process import (
    AsyncProcessTransport,
    ProcessResult,
    ProcessTransportTimeoutError,
    ProcessTransportUnavailableError,
)


_AUTH_FAILURE = re.compile(
    r"(?:\b(?:unauthori[sz]ed|not authenticated|not logged[ _-]?in|"
    r"authentication required|please (?:sign|log)[ _-]?in|"
    r"oauth token (?:has )?expired|session expired|"
    r"credentials? (?:are )?(?:missing|invalid|required))\b|"
    r"(?:\b(?:http|status|error|response)(?:[ _-]?code)?\b[\s:=#-]*)401\b|"
    r"/login\b)",
    re.IGNORECASE,
)
_VERSION = re.compile(r"\b(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)\b")


class CLIAuthStatus(StrEnum):
    OK = "ok"
    MISSING = "missing"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CLIAdapterDefinition:
    adapter_id: str
    name: str
    executable: str
    version_args: tuple[str, ...] = ("--version",)
    version_timeout_seconds: float = 3.0
    auth_args: tuple[str, ...] | None = None
    auth_timeout_seconds: float = 5.0
    model_args: tuple[str, ...] | None = None
    model_timeout_seconds: float = 5.0
    model_parser: Callable[[str], tuple[str, ...] | None] | None = None
    fallback_models: tuple[str, ...] = ()
    executable_override_env: str | None = None
    auth_satisfying_env: tuple[str, ...] = ()
    protocol_version: str | None = None


@dataclass(frozen=True)
class CLIResolvedLaunch:
    configured_override_path: str | None
    path_resolved_path: str | None
    selected_path: str | None
    launch_path: str | None
    launch_kind: str
    child_path_prepend: tuple[str, ...]
    diagnostic: str | None = None


@dataclass(frozen=True)
class CLIDetectionOutcome:
    invocable: bool
    selected_path: str | None
    launch_path: str | None
    runtime_version: str | None
    auth_status: CLIAuthStatus | None
    active_auth_method: str | None
    available_model_ids: tuple[str, ...]
    diagnostics: tuple[str, ...]


def resolve_cli_launch(
    definition: CLIAdapterDefinition,
    transport: AsyncProcessTransport,
    environment: Mapping[str, str],
) -> CLIResolvedLaunch:
    """Resolve exactly the executable that probes and real calls will share."""

    override = None
    if definition.executable_override_env is not None:
        raw_override = environment.get(definition.executable_override_env)
        if raw_override is not None and raw_override.strip():
            override = raw_override.strip()
    candidate = override or definition.executable
    resolved = transport.resolve(candidate)
    if resolved is None:
        return CLIResolvedLaunch(
            configured_override_path=override,
            path_resolved_path=None if override else None,
            selected_path=None,
            launch_path=None,
            launch_kind="unresolved",
            child_path_prepend=(),
        )

    path_resolved = None if override else resolved
    launch_path = resolved
    launch_kind = "selected"
    prepended = [str(Path(resolved).parent)]
    diagnostic = None
    if definition.adapter_id == "codex":
        native = resolve_codex_native_binary(resolved)
        if native is not None:
            launch_path = native
            launch_kind = "codex-native"
            prepended.extend(_codex_native_path_dirs(native))
        elif _looks_like_codex_wrapper(resolved):
            diagnostic = (
                "Codex native binary was not found; falling back to the "
                f"selected wrapper {resolved}"
            )
    return CLIResolvedLaunch(
        configured_override_path=override,
        path_resolved_path=path_resolved,
        selected_path=resolved,
        launch_path=launch_path,
        launch_kind=launch_kind,
        child_path_prepend=_unique_paths(prepended),
        diagnostic=diagnostic,
    )


def apply_cli_launch_environment(
    environment: Mapping[str, str],
    launch: CLIResolvedLaunch,
) -> dict[str, str]:
    """Keep discovery and child PATH resolution symmetric."""

    result = dict(environment)
    path_key = next(
        (key for key in result if key.lower() == "path"),
        "PATH",
    )
    existing = result.get(path_key, "")
    candidates = [
        *launch.child_path_prepend,
        *(part for part in existing.split(os.pathsep) if part),
        *well_known_user_toolchain_bins(),
    ]
    result[path_key] = os.pathsep.join(_unique_paths(candidates))
    return result


async def detect_cli_adapter(
    definition: CLIAdapterDefinition,
    transport: AsyncProcessTransport,
    environment: Mapping[str, str],
    *,
    auth_failure_classifier: Callable[[str], bool] = lambda text: bool(
        _AUTH_FAILURE.search(text)
    ),
    auth_method_classifier: Callable[[ProcessResult], str] | None = None,
) -> CLIDetectionOutcome:
    """Port Open Design's availability/auth separation to one CLI adapter."""

    launch = resolve_cli_launch(definition, transport, environment)
    if launch.launch_path is None:
        return CLIDetectionOutcome(
            invocable=False,
            selected_path=None,
            launch_path=None,
            runtime_version=None,
            auth_status=None,
            active_auth_method=None,
            available_model_ids=definition.fallback_models,
            diagnostics=("executable_not_found",),
        )
    probe_environment = apply_cli_launch_environment(environment, launch)
    diagnostics: list[str] = []
    if launch.diagnostic is not None:
        diagnostics.append("native_binary_unavailable")

    version_result: ProcessResult | None = None
    try:
        version_result = await transport.run(
            (launch.launch_path, *definition.version_args),
            stdin=None,
            cwd=_neutral_probe_directory(),
            environment=probe_environment,
            timeout_seconds=definition.version_timeout_seconds,
        )
    except ProcessTransportTimeoutError:
        diagnostics.append("version_unverified")
    except ProcessTransportUnavailableError:
        return CLIDetectionOutcome(
            invocable=False,
            selected_path=launch.selected_path,
            launch_path=launch.launch_path,
            runtime_version=None,
            auth_status=None,
            active_auth_method=None,
            available_model_ids=definition.fallback_models,
            diagnostics=("executable_not_runnable",),
        )
    if version_result is not None and version_result.exit_code in {126, 127}:
        return CLIDetectionOutcome(
            invocable=False,
            selected_path=launch.selected_path,
            launch_path=launch.launch_path,
            runtime_version=None,
            auth_status=None,
            active_auth_method=None,
            available_model_ids=definition.fallback_models,
            diagnostics=("executable_not_runnable",),
        )
    if version_result is not None and version_result.exit_code != 0:
        diagnostics.append("version_unverified")
    runtime_version = (
        _runtime_version(version_result) if version_result is not None else None
    )

    # Open Design runs independent metadata probes concurrently after the
    # version gate. Preserve that behavior for auth and model discovery.
    (auth_status, active_auth_method), (
        available_model_ids,
        model_catalog_live,
    ) = await asyncio.gather(
        _probe_cli_auth(
            definition,
            transport,
            launch.launch_path,
            probe_environment,
            auth_failure_classifier,
            auth_method_classifier,
        ),
        _probe_cli_models(
            definition,
            transport,
            launch.launch_path,
            probe_environment,
        ),
    )

    if auth_status is CLIAuthStatus.MISSING:
        diagnostics.append("authentication_missing")
    elif auth_status is CLIAuthStatus.UNKNOWN:
        diagnostics.append("authentication_unverified")
    if definition.model_args is not None and not model_catalog_live:
        diagnostics.append("model_catalog_fallback")
    return CLIDetectionOutcome(
        invocable=True,
        selected_path=launch.selected_path,
        launch_path=launch.launch_path,
        runtime_version=runtime_version,
        auth_status=auth_status,
        active_auth_method=active_auth_method,
        available_model_ids=available_model_ids,
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


async def _probe_cli_auth(
    definition: CLIAdapterDefinition,
    transport: AsyncProcessTransport,
    launch_path: str,
    environment: Mapping[str, str],
    failure_classifier: Callable[[str], bool],
    method_classifier: Callable[[ProcessResult], str] | None,
) -> tuple[CLIAuthStatus | None, str | None]:
    satisfying_key = _first_nonempty_env(
        environment,
        definition.auth_satisfying_env,
    )
    if satisfying_key is not None:
        return CLIAuthStatus.OK, "api_key"
    if definition.auth_args is None:
        return None, None
    try:
        result = await transport.run(
            (launch_path, *definition.auth_args),
            stdin=None,
            cwd=_neutral_probe_directory(),
            environment=environment,
            timeout_seconds=definition.auth_timeout_seconds,
        )
    except (ProcessTransportTimeoutError, ProcessTransportUnavailableError):
        return CLIAuthStatus.UNKNOWN, None
    text = result.stdout + "\n" + result.stderr
    if failure_classifier(text):
        return CLIAuthStatus.MISSING, None
    if result.exit_code != 0:
        return CLIAuthStatus.UNKNOWN, None
    method = (
        method_classifier(result)
        if method_classifier is not None
        else "cli_session"
    )
    return CLIAuthStatus.OK, method


async def _probe_cli_models(
    definition: CLIAdapterDefinition,
    transport: AsyncProcessTransport,
    launch_path: str,
    environment: Mapping[str, str],
) -> tuple[tuple[str, ...], bool]:
    if definition.model_args is None or definition.model_parser is None:
        return definition.fallback_models, False
    try:
        result = await transport.run(
            (launch_path, *definition.model_args),
            stdin=None,
            cwd=_neutral_probe_directory(),
            environment=environment,
            timeout_seconds=definition.model_timeout_seconds,
        )
    except (ProcessTransportTimeoutError, ProcessTransportUnavailableError):
        return definition.fallback_models, False
    if result.exit_code != 0:
        return definition.fallback_models, False
    parsed = definition.model_parser(result.stdout)
    if not parsed:
        return definition.fallback_models, False
    return tuple(dict.fromkeys(parsed)), True


def parse_codex_debug_models(stdout: str) -> tuple[str, ...] | None:
    """Parse the visible model identifiers exposed by ``codex debug models``."""

    import json

    try:
        payload = json.loads(stdout)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return None
    result = ["default"]
    seen = {"default"}
    for raw in payload["models"]:
        if not isinstance(raw, dict) or raw.get("visibility") == "hidden":
            continue
        value = raw.get("slug") or raw.get("id")
        if not isinstance(value, str):
            continue
        model_id = value.strip()
        if model_id and model_id not in seen:
            seen.add(model_id)
            result.append(model_id)
    return tuple(result) if len(result) > 1 else None


def resolve_codex_native_binary(
    wrapper_path: str,
    *,
    platform_name: str | None = None,
    machine: str | None = None,
) -> str | None:
    """Upgrade an npm Codex wrapper to its packaged native binary."""

    selected_platform = platform_name or sys.platform
    selected_arch = _normalized_arch(machine or platform.machine())
    if _is_native_codex(wrapper_path, selected_platform):
        return wrapper_path
    package_suffix = _codex_package_suffix(selected_platform, selected_arch)
    target_triple = _codex_target_triple(selected_platform, selected_arch)
    if package_suffix is None or target_triple is None:
        return None
    roots: list[Path] = []
    for seed in _path_seeds(wrapper_path):
        current = seed.parent
        roots.extend([current, *current.parents])
    for root in _unique_path_objects(roots):
        scoped = root / "node_modules" / "@openai"
        # Open Design probes the hoisted ``@openai/codex-<platform>`` scope.
        # npm 11 can keep the optional native package nested below the main
        # ``@openai/codex`` package, so probe that equivalent scope as well.
        scopes = (scoped, scoped / "codex" / "node_modules" / "@openai")
        package_dirs: list[Path] = []
        for package_scope in scopes:
            package_dirs.append(package_scope / f"codex-{package_suffix}")
            try:
                package_dirs.extend(
                    item
                    for item in package_scope.iterdir()
                    if item.is_dir() and item.name.startswith("codex-")
                )
            except OSError:
                pass
        for package_dir in _unique_path_objects(package_dirs):
            for candidate in _codex_native_candidates(
                package_dir,
                target_triple,
                selected_platform,
            ):
                if _is_executable_file(candidate, selected_platform):
                    return str(candidate)
    return None


def well_known_user_toolchain_bins(
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Search the user toolchain locations used by Open Design detection."""

    env = os.environ if environment is None else environment
    if home is not None:
        user_home = home
    else:
        configured_home = env.get("USERPROFILE") or env.get("HOME")
        if configured_home:
            user_home = Path(configured_home)
        else:
            try:
                user_home = Path.home()
            except RuntimeError:
                return ()
    directories: list[Path] = []
    npm_prefix = env.get("NPM_CONFIG_PREFIX") or env.get("npm_config_prefix")
    if npm_prefix and npm_prefix.strip():
        prefix = Path(npm_prefix.strip())
        directories.append(prefix / "bin")
        if os.name == "nt":
            directories.append(prefix)
    if os.name == "nt":
        directories.append(user_home / "AppData" / "Roaming" / "npm")
    directories.extend(
        (
            user_home / ".local" / "bin",
            user_home / ".vite-plus" / "bin",
            user_home / ".opencode" / "bin",
            user_home / ".bun" / "bin",
            user_home / ".volta" / "bin",
            user_home / ".asdf" / "shims",
            user_home / ".cargo" / "bin",
            user_home / ".npm-global" / "bin",
            user_home / ".npm-packages" / "bin",
            user_home / ".deno" / "bin",
            user_home / "go" / "bin",
            user_home / ".pyenv" / "shims",
        )
    )
    if os.name == "nt":
        directories.append(user_home / "scoop" / "shims")
        for key in ("APPDATA",):
            if value := env.get(key):
                directories.append(Path(value) / "npm")
    mise_data = Path(
        env.get("MISE_DATA_DIR", str(user_home / ".local" / "share" / "mise"))
    )
    directories.extend((mise_data / "shims", user_home / ".mise" / "shims"))
    directories.extend(_versioned_node_bins(user_home, env, mise_data))
    return _unique_paths(str(item) for item in directories)


def _versioned_node_bins(
    home: Path,
    environment: Mapping[str, str],
    mise_data: Path,
) -> list[Path]:
    specs = [
        (mise_data / "installs" / "node", ("bin",)),
        (home / ".nvm" / "versions" / "node", ("bin",)),
        (
            home / ".local" / "share" / "fnm" / "node-versions",
            ("installation", "bin"),
        ),
        (home / ".fnm" / "node-versions", ("installation", "bin")),
    ]
    if os.name == "nt":
        fnm_roots: list[Path] = []
        if value := environment.get("FNM_DIR"):
            fnm_roots.append(Path(value))
        else:
            for key in ("LOCALAPPDATA", "APPDATA"):
                if value := environment.get(key):
                    fnm_roots.append(Path(value) / "fnm")
        specs.extend((root / "node-versions", ("installation",)) for root in fnm_roots)
    result: list[Path] = []
    for root, segments in specs:
        try:
            children = sorted(root.iterdir(), key=_version_sort_key, reverse=True)
        except OSError:
            continue
        for child in children:
            candidate = child.joinpath(*segments)
            if child.is_dir() and candidate.exists():
                result.append(candidate)
    return result


def _version_sort_key(path: Path) -> tuple[int, int, int, str]:
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", path.name)
    if match is None:
        return (-1, -1, -1, path.name)
    return (int(match[1]), int(match[2]), int(match[3]), path.name)


def _codex_native_candidates(
    package_dir: Path,
    target_triple: str,
    platform_name: str,
) -> Sequence[Path]:
    executable = "codex.exe" if platform_name == "win32" else "codex"
    return (
        package_dir / "vendor" / target_triple / "codex" / executable,
        # Newer npm Codex packages use ``vendor/<triple>/bin/codex.exe``.
        package_dir / "vendor" / target_triple / "bin" / executable,
        package_dir / executable,
        package_dir / "bin" / executable,
        package_dir / "vendor" / executable,
    )


def _codex_native_path_dirs(native_path: str) -> list[str]:
    path = Path(native_path)
    parents = list(path.parents)
    for parent in parents:
        if parent.name in {"bin", "codex"} and parent.parent.name:
            sibling = parent.parent / "path"
            if sibling.is_dir():
                return [str(sibling)]
    return []


def _path_seeds(path: str) -> tuple[Path, ...]:
    selected = Path(path)
    try:
        resolved = selected.resolve(strict=True)
    except OSError:
        return (selected,)
    return (selected, resolved) if selected != resolved else (selected,)


def _looks_like_codex_wrapper(path: str) -> bool:
    selected = Path(path)
    if selected.suffix.lower() in {".cmd", ".bat", ".ps1", ".js"}:
        return True
    try:
        with selected.open("rb") as stream:
            header = stream.read(64_000)
    except OSError:
        return False
    return bool(re.search(rb"node|@openai/codex|codex-", header, re.IGNORECASE))


def _is_native_codex(path: str, platform_name: str) -> bool:
    selected = Path(path)
    expected = "codex.exe" if platform_name == "win32" else "codex"
    return selected.name.lower() == expected and _is_executable_file(
        selected,
        platform_name,
    )


def _is_executable_file(path: Path, platform_name: str) -> bool:
    try:
        if not path.is_file():
            return False
        return platform_name == "win32" or os.access(path, os.X_OK)
    except OSError:
        return False


def _normalized_arch(machine: str) -> str:
    normalized = machine.lower()
    if normalized in {"amd64", "x86_64", "x64"}:
        return "x64"
    if normalized in {"arm64", "aarch64"}:
        return "arm64"
    return normalized


def _codex_package_suffix(platform_name: str, arch: str) -> str | None:
    platform_label = {
        "win32": "win32",
        "linux": "linux",
        "darwin": "darwin",
    }.get(platform_name)
    return f"{platform_label}-{arch}" if platform_label is not None else None


def _codex_target_triple(platform_name: str, arch: str) -> str | None:
    return {
        ("darwin", "arm64"): "aarch64-apple-darwin",
        ("darwin", "x64"): "x86_64-apple-darwin",
        ("linux", "arm64"): "aarch64-unknown-linux-musl",
        ("linux", "x64"): "x86_64-unknown-linux-musl",
        ("win32", "arm64"): "aarch64-pc-windows-msvc",
        ("win32", "x64"): "x86_64-pc-windows-msvc",
    }.get((platform_name, arch))


def _first_nonempty_env(
    environment: Mapping[str, str],
    names: Sequence[str],
) -> str | None:
    for name in names:
        value = environment.get(name)
        if value is not None and value.strip():
            return name
    return None


def _runtime_version(result: ProcessResult) -> str | None:
    match = _VERSION.search(result.stdout + "\n" + result.stderr)
    return match.group(1) if match else None


def _neutral_probe_directory() -> str:
    return os.getenv("TEMP") or os.getenv("TMP") or str(Path.cwd())


def _unique_paths(paths: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in paths:
        if not raw:
            continue
        normalized = os.path.normcase(os.path.normpath(raw))
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(raw)
    return tuple(result)


def _unique_path_objects(paths: Sequence[Path]) -> tuple[Path, ...]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = os.path.normcase(os.path.normpath(str(path)))
        if key not in seen:
            seen.add(key)
            result.append(path)
    return tuple(result)
