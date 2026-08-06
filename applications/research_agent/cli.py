"""Command-line presentation for the Research Agent Application."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
from typing import cast

from adaptive_agent_runtime.llm import (
    AnthropicAPITargetDefinition,
    AnthropicMessagesConfig,
    BackendAvailability,
    BackendTransportFeatures,
    ClaudeCodeInferenceConfig,
    ClaudeCodeInferenceTargetDefinition,
    CodexCLIInferenceConfig,
    CodexCLIInferenceTargetDefinition,
    ContextSensitivity,
    LLMTargetSelection,
    OpenAICompatibleService,
    OpenAICompatibleProbeMode,
    OpenAICompatibleTargetDefinition,
    ReasoningEffort,
    StructuredOutputLevel,
    TOMLProviderConfigRepository,
)

from applications.research_agent.agent import (
    ResearchAgent,
    ResearchInformationMode,
)
from applications.research_agent.llm_deployment import (
    ResearchInferenceTargetDefinition,
    ResearchLLMCapability,
    ResearchLLMDeploymentConfig,
    build_research_llm_deployment,
    managed_capability_ids,
)
from applications.research_agent.report import ResearchRunResult


CODEX_CLI_SERVICE = "codex_cli"
CLAUDE_CODE_SERVICE = "claude_code"
ANTHROPIC_API_SERVICE = "anthropic"

_LLM_CONFIG_SCHEMA_VERSION = 1

_LLM_TARGET_KEYS = frozenset(
    {
        "service",
        "executable",
        "model",
        "reasoning_effort",
        "structured_output",
        "target_id",
        "base_url",
        "api_key_env",
        "api_probe",
        "capabilities",
        "allow_confidential_context",
        "context_tokens",
        "max_output_tokens",
        "max_attempts",
        "max_elapsed_seconds",
        "reasoner_tool_intents",
        "strict_json_schema",
        "allow_insecure_http",
    }
)

_LLM_FILE_DEFAULTS: dict[str, object] = {
    "llm_config": None,
    "llm_target": None,
    "llm_service": None,
    "llm_executable": None,
    "llm_model": None,
    "llm_reasoning_effort": None,
    "llm_structured_output": None,
    "llm_target_id": None,
    "llm_base_url": None,
    "llm_api_key_env": None,
    "llm_api_key": None,
    "llm_api_probe": None,
    "llm_capability": None,
    "llm_allow_confidential_context": False,
    "llm_context_tokens": 4096,
    "llm_max_output_tokens": None,
    "llm_max_attempts": 1,
    "llm_max_elapsed_seconds": None,
    "llm_reasoner_tool_intents": 0,
    "llm_strict_json_schema": False,
    "llm_allow_insecure_http": False,
}


def format_result(result: ResearchRunResult) -> str:
    lines = ["1. Task Graph"]
    for node in result.task_graph.nodes:
        dependencies = ", ".join(str(item)[:8] for item in node.dependencies) or "-"
        lines.append(
            f"- {str(node.node_id)[:8]} [{node.status.value}] {node.goal} "
            f"(depends: {dependencies})"
        )

    lines.extend(["", "2. Execution Trace"])
    for entry in result.runtime_trace:
        lines.append(f"- {entry.sequence:02d} {entry.event.kind}")
    lines.extend(
        [
            f"- Tool trace entries: {len(result.tool_trace)}",
            f"- Context assemblies: {len(result.context_assemblies)}",
            f"- Autonomous Agent executions: {len(result.agent_executions)}",
            f"- LLM ToolIntent executions: {len(result.llm_tool_intents)}",
            f"- LLM Action proposals: {len(result.llm_action_proposals)}",
            (
                "- LLM Graph mutation proposals: "
                f"{len(result.llm_graph_mutation_proposals)}"
            ),
            "",
            "3. Research Report",
            result.report.markdown,
            "",
            "4. Evaluation Report",
            (
                f"- Task Success: {result.evaluation.outcome.verdict.value} "
                f"(score={result.evaluation.outcome.score})"
            ),
            (
                f"- Trajectory Quality: {result.evaluation.trajectory.verdict.value} "
                f"(score={result.evaluation.trajectory.score})"
            ),
        ]
    )
    for component in result.evaluation.components:
        component_name = component.component.value if component.component else "unknown"
        lines.append(
            f"- {component_name}: {component.verdict.value} "
            f"(score={component.score})"
        )
    if result.llm_judgement is not None:
        lines.append(
            f"- LLM Judge: {result.llm_judgement.summary} "
            f"(score={result.llm_judgement.score})"
        )
    lines.append(
        f"- Failure Patterns: {len(result.failure_analysis.patterns)}"
    )
    for pattern in result.failure_analysis.patterns:
        lines.append(
            f"  - {pattern.pattern_key}: {pattern.root_cause.description}"
        )
    lines.append(f"- Stored Optimization Proposals: {len(result.optimization_proposals)}")
    for proposal in result.optimization_proposals:
        lines.append(
            f"  - {proposal.target_key.value}: {proposal.current_value} -> "
            f"{proposal.proposed_value} (proposal only)"
        )

    lines.extend(["", "5. Governance Decisions"])
    for record in result.governance_records:
        review_text = "reviewed" if record.review is not None else "automatic"
        authorization = (
            str(record.authorization.authorization_id)[:8]
            if record.authorization is not None
            else "none"
        )
        lines.append(
            f"- {record.scenario}: {record.preliminary.outcome.value} -> "
            f"{record.final.outcome.value} ({review_text}, auth={authorization})"
        )
    return "\n".join(lines)


async def async_main(
    task: str,
    llm_config: ResearchLLMDeploymentConfig | None = None,
) -> ResearchRunResult:
    if llm_config is None:
        with TemporaryDirectory(prefix="research-agent-demo-") as directory:
            agent = ResearchAgent(
                persistence_path=Path(directory) / "runtime.sqlite3",
                run_kind="demo",
                disposable=True,
            )
            try:
                result = await agent.run_demo(task)
            finally:
                agent.close()
    else:
        deployment = build_research_llm_deployment(llm_config)
        probe = await deployment.probe()
        if probe.availability is not BackendAvailability.AVAILABLE:
            diagnostics = ",".join(probe.diagnostics) or "no diagnostics"
            raise RuntimeError(
                f"LLM target '{deployment.target_id}' preflight failed: "
                f"{probe.availability.value} ({diagnostics})"
            )
        print(
            "LLM preflight: "
            f"target={deployment.target_id}, model={deployment.model_id}, "
            f"runtime={probe.runtime_version or 'remote'}, "
            f"auth={probe.active_auth_method or 'none'}"
        )
        if probe.diagnostics:
            print("LLM preflight diagnostics: " + ",".join(probe.diagnostics))
        agent = ResearchAgent.for_runtime(
            cognitive_capabilities=deployment.cognitive_capabilities,
            information_mode=ResearchInformationMode.LLM_RESEARCH,
        )
        try:
            result = await agent.run(task)
        finally:
            agent.close()
    print(format_result(result))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Adaptive Runtime Research Agent. LLM use is explicit and "
            "defaults to one run to avoid duplicate provider charges."
        )
    )
    parser.add_argument(
        "task",
        nargs="?",
        default="分析 Tesla 投资价值",
        help="Research task, for example: 分析 Tesla 投资价值",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        help=(
            "Load named LLM targets from a unified TOML file. A sibling "
            "llm.local.toml supplies persistent provider API keys when present."
        ),
    )
    parser.add_argument(
        "--llm-target",
        help=(
            "Select a named target from --llm-config; defaults to the file's "
            "active_target."
        ),
    )
    parser.add_argument(
        "--llm-service",
        choices=(
            *(item.value for item in OpenAICompatibleService),
            ANTHROPIC_API_SERVICE,
            CODEX_CLI_SERVICE,
            CLAUDE_CODE_SERVICE,
        ),
        help=(
            "Opt in to Anthropic API, an OpenAI-compatible service, Codex "
            "CLI, or Claude Code authenticated session."
        ),
    )
    parser.add_argument(
        "--llm-executable",
        help="CLI executable name or path (defaults: codex or claude).",
    )
    parser.add_argument(
        "--llm-model",
        help="Explicit provider model identifier.",
    )
    parser.add_argument(
        "--llm-reasoning-effort",
        choices=tuple(item.value for item in ReasoningEffort),
        help=(
            "Select provider reasoning effort; default omits the provider "
            "override. Support is validated for the selected backend."
        ),
    )
    parser.add_argument(
        "--llm-structured-output",
        choices=(
            StructuredOutputLevel.JSON_OBJECT.value,
            StructuredOutputLevel.JSON_SCHEMA.value,
        ),
        help="Explicitly declare the selected model's structured-output support.",
    )
    parser.add_argument(
        "--llm-target-id",
        help="Stable Runtime target identity; derived when omitted.",
    )
    parser.add_argument(
        "--llm-base-url",
        help="Override the service endpoint; required for local targets.",
    )
    parser.add_argument(
        "--llm-api-key-env",
        help="Environment variable name containing the API key; never the key.",
    )
    parser.add_argument(
        "--llm-api-probe",
        choices=tuple(item.value for item in OpenAICompatibleProbeMode),
        help=(
            "API preflight mode. models_endpoint verifies connectivity, "
            "credentials, and target model without an inference charge."
        ),
    )
    parser.add_argument(
        "--llm-capability",
        action="append",
        choices=tuple(item.value for item in ResearchLLMCapability),
        help="Enable a cognitive capability; repeat as needed (default: generation).",
    )
    parser.add_argument(
        "--llm-allow-confidential-context",
        action="store_true",
        help="Permit confidential Memory recall to cross the selected target boundary.",
    )
    parser.add_argument(
        "--llm-context-tokens",
        type=int,
        default=4096,
        help="Maximum projected Runtime context tokens per request.",
    )
    parser.add_argument(
        "--llm-max-output-tokens",
        type=int,
        help="Optional provider output-token limit.",
    )
    parser.add_argument(
        "--llm-max-attempts",
        type=int,
        default=1,
        help="Bounded Gateway attempt count (default: 1).",
    )
    parser.add_argument(
        "--llm-max-elapsed-seconds",
        type=float,
        help="Optional wall-clock limit for the complete Gateway retry chain.",
    )
    parser.add_argument(
        "--llm-reasoner-tool-intents",
        type=int,
        default=0,
        help=(
            "Allow at most this many Runtime-governed retrieval proposals per "
            "reasoning node (default: disabled)."
        ),
    )
    parser.add_argument(
        "--llm-strict-json-schema",
        action="store_true",
        help="Request provider-side strict JSON Schema enforcement.",
    )
    parser.add_argument(
        "--llm-allow-insecure-http",
        action="store_true",
        help="Allow non-loopback HTTP for an explicitly configured endpoint.",
    )
    return parser


def llm_config_from_args(
    args: argparse.Namespace,
) -> ResearchLLMDeploymentConfig | None:
    config_path = args.llm_config
    if config_path is not None:
        if _has_explicit_llm_options(args):
            raise ValueError(
                "--llm-config cannot be combined with other --llm-* options"
            )
        return load_llm_config_file(config_path, target_name=args.llm_target)

    if args.llm_target is not None:
        raise ValueError("--llm-target requires --llm-config")

    service_value = args.llm_service
    if service_value is None:
        explicit_llm_option = any(
            (
                args.llm_model is not None,
                args.llm_reasoning_effort is not None,
                args.llm_structured_output is not None,
                args.llm_executable is not None,
                args.llm_base_url is not None,
                args.llm_api_key_env is not None,
                args.llm_api_probe is not None,
                args.llm_capability is not None,
                args.llm_reasoner_tool_intents != 0,
                args.llm_allow_confidential_context,
                args.llm_strict_json_schema,
                args.llm_allow_insecure_http,
            )
        )
        if explicit_llm_option:
            raise ValueError("--llm-service is required for LLM configuration")
        return None
    if args.llm_model is None:
        raise ValueError("--llm-model is required with --llm-service")
    if args.llm_structured_output is None:
        raise ValueError(
            "--llm-structured-output is required; model features are not inferred"
        )
    model_id = str(args.llm_model)
    reasoning_effort = (
        ReasoningEffort(args.llm_reasoning_effort)
        if args.llm_reasoning_effort is not None
        else None
    )
    target_id = args.llm_target_id or f"research/{service_value}/{model_id}"
    capabilities = tuple(
        ResearchLLMCapability(item)
        for item in (args.llm_capability or [ResearchLLMCapability.GENERATION])
    )
    sensitivities = [ContextSensitivity.INTERNAL]
    if args.llm_allow_confidential_context:
        sensitivities.append(ContextSensitivity.CONFIDENTIAL)
    structured_output = StructuredOutputLevel(args.llm_structured_output)
    target: ResearchInferenceTargetDefinition
    if service_value in (CODEX_CLI_SERVICE, CLAUDE_CODE_SERVICE):
        service_label = (
            "Codex CLI"
            if service_value == CODEX_CLI_SERVICE
            else "Claude Code"
        )
        if args.llm_base_url is not None or args.llm_api_key_env is not None:
            raise ValueError(
                f"{service_label} inference does not accept API endpoint options"
            )
        if args.llm_api_probe is not None:
            raise ValueError(
                f"{service_label} inference does not use API probe policy"
            )
        if args.llm_allow_insecure_http:
            raise ValueError(
                f"{service_label} inference does not use HTTP endpoint policy"
            )
        if args.llm_strict_json_schema:
            raise ValueError(
                f"{service_label} inference enforces its local output schema"
            )
        if structured_output is not StructuredOutputLevel.JSON_SCHEMA:
            raise ValueError(
                f"{service_label} cognitive capabilities require JSON Schema"
            )
        if args.llm_reasoner_tool_intents:
            raise ValueError(
                f"{service_label} inference cannot expose ToolIntent"
            )
        if service_value == CODEX_CLI_SERVICE:
            target = CodexCLIInferenceTargetDefinition(
                target_id=target_id,
                model_id=model_id,
                features=BackendTransportFeatures(
                    structured_output=structured_output,
                ),
                supported_cognitive_capability_ids=managed_capability_ids(
                    capabilities
                ),
                config=CodexCLIInferenceConfig(
                    executable=args.llm_executable or "codex",
                    reasoning_effort=reasoning_effort,
                ),
                tags=("research", "oauth-cli"),
            )
        else:
            if reasoning_effort not in {None, ReasoningEffort.DEFAULT}:
                raise ValueError(
                    "Claude Code inference does not expose a Runtime-controlled "
                    "reasoning effort option"
                )
            target = ClaudeCodeInferenceTargetDefinition(
                target_id=target_id,
                model_id=model_id,
                features=BackendTransportFeatures(
                    structured_output=structured_output,
                ),
                supported_cognitive_capability_ids=managed_capability_ids(
                    capabilities
                ),
                config=ClaudeCodeInferenceConfig(
                    executable=args.llm_executable or "claude",
                ),
                tags=("research", "oauth-cli"),
            )
    elif service_value == ANTHROPIC_API_SERVICE:
        if args.llm_executable is not None:
            raise ValueError(
                "--llm-executable is only valid for a CLI inference service"
            )
        if args.llm_api_probe is not None:
            raise ValueError(
                "Anthropic API always verifies the exact model during preflight"
            )
        if args.llm_strict_json_schema:
            raise ValueError(
                "Anthropic API uses native strict structured outputs"
            )
        if structured_output is not StructuredOutputLevel.JSON_SCHEMA:
            raise ValueError(
                "Anthropic Research capabilities require JSON Schema"
            )
        target = AnthropicAPITargetDefinition(
            target_id=target_id,
            model_id=model_id,
            features=BackendTransportFeatures(
                structured_output=structured_output,
                tool_intent=bool(args.llm_reasoner_tool_intents),
            ),
            supported_cognitive_capability_ids=managed_capability_ids(
                capabilities
            ),
            config=AnthropicMessagesConfig(
                base_url=(
                    args.llm_base_url or "https://api.anthropic.com/v1"
                ),
                api_key_env=args.llm_api_key_env or "ANTHROPIC_API_KEY",
                api_key=getattr(args, "llm_api_key", None),
                allow_insecure_http=args.llm_allow_insecure_http,
                reasoning_effort=reasoning_effort,
            ),
            tags=("research", "anthropic-api"),
        )
    else:
        if args.llm_executable is not None:
            raise ValueError(
                "--llm-executable is only valid for a CLI inference service"
            )
        service = OpenAICompatibleService(service_value)
        target = OpenAICompatibleTargetDefinition(
            service=service,
            target_id=target_id,
            model_id=model_id,
            features=BackendTransportFeatures(
                structured_output=structured_output,
                tool_intent=bool(args.llm_reasoner_tool_intents),
            ),
            supported_cognitive_capability_ids=managed_capability_ids(
                capabilities
            ),
            base_url=args.llm_base_url,
            api_key_env=args.llm_api_key_env,
            api_key=getattr(args, "llm_api_key", None),
            requires_api_key=(
                False if service is OpenAICompatibleService.LOCAL else None
            ),
            strict_json_schema=args.llm_strict_json_schema,
            allow_insecure_http=args.llm_allow_insecure_http,
            probe_mode=(
                OpenAICompatibleProbeMode(args.llm_api_probe)
                if args.llm_api_probe is not None
                else None
            ),
            reasoning_effort=reasoning_effort,
        )
    return ResearchLLMDeploymentConfig(
        target=target,
        enabled_capabilities=capabilities,
        allowed_context_sensitivities=tuple(sensitivities),
        max_context_tokens=args.llm_context_tokens,
        max_output_tokens=args.llm_max_output_tokens,
        max_attempts=args.llm_max_attempts,
        max_elapsed_seconds=args.llm_max_elapsed_seconds,
        reasoner_tool_intent_limit=args.llm_reasoner_tool_intents,
    )


def load_llm_config_file(
    path: str | Path,
    *,
    target_name: str | None = None,
    private_path: str | Path | None = None,
) -> ResearchLLMDeploymentConfig:
    """Load a target and merge private credentials and target selection."""

    config_path = Path(path)
    try:
        with config_path.open("rb") as stream:
            parsed: object = tomllib.load(stream)
    except OSError as exc:
        raise ValueError(f"cannot read LLM config '{config_path}': {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML in LLM config '{config_path}': {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("LLM config root must be a TOML table")
    root = cast(dict[str, object], parsed)
    unknown_sections = set(root) - {"schema_version", "llm"}
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ValueError(f"unknown LLM config section(s): {names}")
    schema_version = root.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("LLM config schema_version must be an integer")
    if schema_version != _LLM_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "unsupported LLM config schema_version "
            f"{schema_version}; expected {_LLM_CONFIG_SCHEMA_VERSION}"
        )
    table_value = root.get("llm")
    if not isinstance(table_value, dict):
        raise ValueError("LLM config must contain an [llm] table")
    llm_table = cast(dict[str, object], table_value)
    unknown_llm_keys = set(llm_table) - {"active_target", "targets"}
    if unknown_llm_keys:
        if "service" in unknown_llm_keys:
            raise ValueError(
                "legacy single-target [llm] config is unsupported; move provider "
                "options under [llm.targets.<name>]"
            )
        names = ", ".join(sorted(unknown_llm_keys))
        raise ValueError(f"unknown [llm] option(s): {names}")

    active_target = llm_table.get("active_target")
    if not isinstance(active_target, str) or not active_target.strip():
        raise ValueError("[llm].active_target must be a non-empty string")
    targets_value = llm_table.get("targets")
    if not isinstance(targets_value, dict) or not targets_value:
        raise ValueError("LLM config must contain at least one [llm.targets.<name>] table")
    targets = cast(dict[str, object], targets_value)
    validated_targets: dict[str, dict[str, object]] = {}
    for configured_name, configured_value in targets.items():
        if not configured_name.strip():
            raise ValueError("LLM target names must be non-empty strings")
        if not isinstance(configured_value, dict):
            raise ValueError(f"[llm.targets.{configured_name}] must be a table")
        configured_table = cast(dict[str, object], configured_value)
        if "api_key" in configured_table:
            raise ValueError(
                f"[llm.targets.{configured_name}] must not contain api_key; "
                "use api_key_env instead"
            )
        unknown_keys = set(configured_table) - _LLM_TARGET_KEYS
        if unknown_keys:
            names = ", ".join(sorted(unknown_keys))
            raise ValueError(
                f"unknown [llm.targets.{configured_name}] option(s): {names}"
            )
        _validate_llm_file_values(
            configured_table,
            table_name=f"llm.targets.{configured_name}",
        )
        validated_targets[configured_name] = configured_table

    selected_name = target_name or active_target
    if not selected_name.strip():
        raise ValueError("--llm-target must be a non-empty string")
    table = validated_targets.get(selected_name)
    if table is None:
        available = ", ".join(sorted(validated_targets))
        raise ValueError(
            f"LLM target '{selected_name}' is not configured; available: {available}"
        )

    service = table.get("service")
    selection: LLMTargetSelection | None = None
    if isinstance(service, str):
        selected_private_path = (
            Path(private_path)
            if private_path is not None
            else config_path.with_name(
                f"{config_path.stem}.local{config_path.suffix}"
            )
        )
        if private_path is not None or selected_private_path.is_file():
            private_repository = TOMLProviderConfigRepository(
                selected_private_path
            )
            api_keys = private_repository.load_api_keys()
            selections = private_repository.load_target_selections()
            provider_key = api_keys.get(service)
            selection = selections.get(selected_name)
        else:
            provider_key = None
    else:
        provider_key = None

    values = dict(_LLM_FILE_DEFAULTS)
    for key, value in table.items():
        argument_name = (
            "llm_capability" if key == "capabilities" else f"llm_{key}"
        )
        values[argument_name] = value
    if selection is not None:
        values["llm_model"] = selection.model_id
        if selection.reasoning_effort is not None:
            values["llm_reasoning_effort"] = selection.reasoning_effort.value
    values["llm_api_key"] = provider_key
    config = llm_config_from_args(argparse.Namespace(**values))
    if config is None:
        raise ValueError(f"[llm.targets.{selected_name}].service is required")
    return config


def _has_explicit_llm_options(args: argparse.Namespace) -> bool:
    return any(
        (
            args.llm_service is not None,
            args.llm_executable is not None,
            args.llm_model is not None,
            args.llm_reasoning_effort is not None,
            args.llm_structured_output is not None,
            args.llm_target_id is not None,
            args.llm_base_url is not None,
            args.llm_api_key_env is not None,
            args.llm_api_probe is not None,
            args.llm_capability is not None,
            args.llm_allow_confidential_context,
            args.llm_context_tokens != 4096,
            args.llm_max_output_tokens is not None,
            args.llm_max_attempts != 1,
            args.llm_max_elapsed_seconds is not None,
            args.llm_reasoner_tool_intents != 0,
            args.llm_strict_json_schema,
            args.llm_allow_insecure_http,
        )
    )


def _validate_llm_file_values(
    table: dict[str, object],
    *,
    table_name: str,
) -> None:
    string_keys = {
        "service",
        "executable",
        "model",
        "reasoning_effort",
        "structured_output",
        "target_id",
        "base_url",
        "api_key_env",
        "api_probe",
    }
    boolean_keys = {
        "allow_confidential_context",
        "strict_json_schema",
        "allow_insecure_http",
    }
    integer_keys = {
        "context_tokens",
        "max_output_tokens",
        "max_attempts",
        "reasoner_tool_intents",
    }
    for key in string_keys & table.keys():
        if not isinstance(table[key], str):
            raise ValueError(f"[{table_name}].{key} must be a string")
    for key in boolean_keys & table.keys():
        if not isinstance(table[key], bool):
            raise ValueError(f"[{table_name}].{key} must be a boolean")
    for key in integer_keys & table.keys():
        if isinstance(table[key], bool) or not isinstance(table[key], int):
            raise ValueError(f"[{table_name}].{key} must be an integer")
    capabilities = table.get("capabilities")
    if capabilities is not None and (
        not isinstance(capabilities, list)
        or not all(isinstance(item, str) for item in capabilities)
    ):
        raise ValueError(
            f"[{table_name}].capabilities must be an array of strings"
        )
    elapsed = table.get("max_elapsed_seconds")
    if elapsed is not None and (
        isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
    ):
        raise ValueError(f"[{table_name}].max_elapsed_seconds must be a number")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        llm_config = llm_config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    asyncio.run(async_main(args.task, llm_config))


if __name__ == "__main__":
    main()
