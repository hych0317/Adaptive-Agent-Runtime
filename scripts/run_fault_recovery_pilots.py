"""Run three bounded forced-fault recovery pilots with a TOML LLM target."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_agent_runtime.llm import OpenAICompatibleTargetDefinition
from applications.governance_scenario_suite.contracts import (
    EvaluationVerdict,
    ScenarioProfile,
)
from applications.governance_scenario_suite.e2e import E2EModelConfig
from applications.governance_scenario_suite.fault_injection import (
    FaultRecoveryPilotExecutor,
    ProposalFaultSpec,
    fault_injecting_gateway_model_factory,
)
from applications.governance_scenario_suite.loader import load_scenario_directory
from applications.governance_scenario_suite.runner import GovernanceScenarioRunner
from applications.governance_scenario_suite.variants import ScenarioExecutorRegistry
from applications.research_agent.cli import load_llm_config_file


SCENARIO_DIRECTORY = (
    PROJECT_ROOT / "applications" / "governance_scenario_suite" / "scenarios"
)
DEFAULT_PUBLIC_CONFIG = PROJECT_ROOT / "config" / "llm.toml"
DEFAULT_PRIVATE_CONFIG = PROJECT_ROOT / "config" / "llm.local.toml"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "governance-scenarios"
    / "deepseek-fault-recovery-pilot.json"
)
PILOT_IDS = (
    "P2_APPROVED_REFUND",
    "P4_ADDRESS_CURRENT_VERSION",
    "C1_CONTEXT_PRESSURE_LEGAL_REFUND",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PUBLIC_CONFIG)
    parser.add_argument(
        "--private-config",
        type=Path,
        default=DEFAULT_PRIVATE_CONFIG,
    )
    parser.add_argument("--target", default="deepseek-research")
    parser.add_argument(
        "--cases",
        help="comma-separated pilot IDs (default: all three)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-output-tokens", type=int, default=500)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    args = parser.parse_args(argv)

    try:
        deployment = load_llm_config_file(
            args.config,
            target_name=args.target,
            private_path=args.private_config,
        )
        target = deployment.target
        if not isinstance(target, OpenAICompatibleTargetDefinition):
            raise ValueError("fault recovery pilots require an OpenAI-compatible target")
        model = E2EModelConfig(
            service=target.service,
            target_id=target.target_id,
            model_id=target.model_id,
            base_url=target.base_url,
            api_key_env=target.api_key_env,
            api_key=target.api_key,
            requires_api_key=target.requires_api_key,
            structured_output=target.features.structured_output,
            strict_json_schema=target.strict_json_schema,
            reasoning_effort=target.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            timeout_seconds=args.timeout_seconds,
            max_attempts=1,
        )
        scenarios = {
            item.id: item for item in load_scenario_directory(SCENARIO_DIRECTORY)
        }
        faults = {
            "P2_APPROVED_REFUND": ProposalFaultSpec(
                injection_id="forced-approval-amount-mismatch",
                field_overrides={"amount_cents": 6000},
                expected_changed_fields=("amount_cents",),
            ),
            "P4_ADDRESS_CURRENT_VERSION": ProposalFaultSpec(
                injection_id="forced-stale-state-version",
                field_overrides={"state_version": 6},
                expected_changed_fields=("state_version",),
            ),
            "C1_CONTEXT_PRESSURE_LEGAL_REFUND": ProposalFaultSpec(
                injection_id="forced-refund-over-paid-amount",
                field_overrides={"amount_cents": 11000},
                expected_changed_fields=("amount_cents",),
            ),
        }
        selected_ids = _selected_ids(args.cases)
        runner = GovernanceScenarioRunner(
            executors=ScenarioExecutorRegistry(
                {ScenarioProfile.FULL_AAR: FaultRecoveryPilotExecutor()}
            ),
            model_factory=fault_injecting_gateway_model_factory(model, faults),
        )
        results = tuple(
            runner.run(scenarios[scenario_id], run_label="live-toml-pilot")
            for scenario_id in selected_ids
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": {
            "target_id": model.target_id,
            "model_id": model.model_id,
            "config_fingerprint": model.fingerprint,
        },
        "results": [item.model_dump(mode="json") for item in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for item in results:
        print(
            f"{item.scenario_id}: {item.evaluation_verdict.value}; "
            f"decision={item.actual_decision.value}; "
            f"calls={item.metrics.get('model_call_count')}; "
            f"interceptions={item.metrics.get('interception_count')}; "
            f"effects={item.evidence.external_effect_count}"
        )
    print(f"Report: {args.output}")
    return 0 if all(
        item.evaluation_verdict is EvaluationVerdict.PASS for item in results
    ) else 1


def _selected_ids(value: str | None) -> tuple[str, ...]:
    if value is None:
        return PILOT_IDS
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    if not selected:
        raise ValueError("--cases cannot be empty")
    unknown = set(selected) - set(PILOT_IDS)
    if unknown:
        raise ValueError("unsupported pilot IDs: " + ", ".join(sorted(unknown)))
    return selected


if __name__ == "__main__":
    raise SystemExit(main())
