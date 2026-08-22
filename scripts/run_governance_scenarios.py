"""Run deterministic, comparison, demo, release-gate, or live-model scenarios."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_agent_runtime.decisioning import decision_fingerprint
from adaptive_agent_runtime.llm import OpenAICompatibleService
from applications.governance_scenario_suite.campaign import (
    E2ECampaignConfig,
    run_e2e_campaign,
)
from applications.governance_scenario_suite.contracts import (
    EvaluationVerdict,
    ScenarioProfile,
    ScenarioSpec,
)
from applications.governance_scenario_suite.e2e import E2EModelConfig
from applications.governance_scenario_suite.e2e_metrics import E2EAggregate
from applications.governance_scenario_suite.evidence import ScenarioRunResult
from applications.governance_scenario_suite.execution import (
    run_deterministic_suite,
)
from applications.governance_scenario_suite.final_reporting import (
    EvaluationRunManifest,
    GovernanceEvaluationReport,
    ModelTargetSummary,
    ReleaseGateResult,
    evaluate_release_gate,
    write_governance_report,
)
from applications.governance_scenario_suite.loader import (
    load_scenario_directory,
)
from applications.governance_scenario_suite.metrics import (
    EvaluationMatrix,
    build_evaluation_matrix,
)
from applications.governance_scenario_suite.profiles import (
    build_profile_manifest,
)
from applications.governance_scenario_suite.reporting import build_batch_result


SCENARIO_DIRECTORY = (
    PROJECT_ROOT / "applications" / "governance_scenario_suite" / "scenarios"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "governance-scenarios"
DEMO_CASES = (
    "P2B_APPROVAL_AMOUNT_TAMPERING",
    "R1_COMMITTED_RESPONSE_LOST",
    "M1B_CROSS_USER",
)
DEFAULT_E2E_CASES = (
    "P1_FOREIGN_ORDER",
    "P1_OWN_ORDER",
    "P2B_APPROVAL_AMOUNT_TAMPERING",
    "P2_APPROVED_REFUND",
    "P5_MALICIOUS_PRODUCT_DESCRIPTION",
    "P5_BENIGN_PRODUCT_DESCRIPTION",
)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    scenarios = load_scenario_directory(SCENARIO_DIRECTORY)
    generated_at = datetime.now(timezone.utc)
    try:
        if args.command == "e2e":
            return _run_e2e(args, scenarios, generated_at, parser)
        profiles = _profiles_for(args)
        cases = DEMO_CASES if args.command == "demo" else _csv(args.cases)
        before = _database_snapshot() if args.command == "gate" else None
        results = tuple(
            result
            for profile in profiles
            for result in run_deterministic_suite(
                scenarios,
                profile=profile,
                scenario_ids=cases,
            )
        )
        after = _database_snapshot() if args.command == "gate" else None
        matrix = build_evaluation_matrix(results, {item.id: item for item in scenarios})
        gate = (
            evaluate_release_gate(
                matrix,
                production_database_unchanged=before == after,
            )
            if args.command == "gate"
            else None
        )
        report = _report(
            mode=args.command,
            scenarios=scenarios,
            profiles=profiles,
            results=results,
            matrix=matrix,
            generated_at=generated_at,
            gate=gate,
        )
        paths = write_governance_report(report, args.output_dir)
        _print_outcome(report, paths)
        if gate is not None:
            return 0 if gate.passed else 1
        if args.command in {"deterministic", "demo"}:
            return 0 if report.batch.release_gate_passed else 1
        return 0
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
        return 2


def _run_e2e(
    args: argparse.Namespace,
    scenarios: tuple[ScenarioSpec, ...],
    generated_at: datetime,
    parser: argparse.ArgumentParser,
) -> int:
    required = {
        "--service": args.service,
        "--target-id": args.target_id,
        "--model-id": args.model_id,
    }
    missing = tuple(name for name, value in required.items() if not value)
    if missing:
        parser.error("E2E mode requires " + ", ".join(missing))
    model = E2EModelConfig(
        service=OpenAICompatibleService(args.service),
        target_id=args.target_id,
        model_id=args.model_id,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        requires_api_key=False if args.no_api_key else None,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
    )
    profile = ScenarioProfile(args.profile)
    campaign = run_e2e_campaign(
        scenarios,
        E2ECampaignConfig(
            scenario_ids=_csv(args.cases) or DEFAULT_E2E_CASES,
            profile=profile,
            repeat=args.repeat,
            model=model,
        ),
        generated_at=generated_at,
    )
    matrix = build_evaluation_matrix(
        campaign.results,
        {item.id: item for item in scenarios},
    )
    report = _report(
        mode="e2e",
        scenarios=scenarios,
        profiles=(profile,),
        results=campaign.results,
        matrix=matrix,
        generated_at=generated_at,
        model=model,
        e2e_aggregate=campaign.aggregate,
    )
    paths = write_governance_report(report, args.output_dir)
    _print_outcome(report, paths)
    return 0 if campaign.aggregate.mechanism_pass.count == campaign.aggregate.conclusive_runs and campaign.aggregate.provider_failure_count == 0 else 1


def _report(
    *,
    mode: str,
    scenarios: tuple[ScenarioSpec, ...],
    profiles: tuple[ScenarioProfile, ...],
    results: tuple[ScenarioRunResult, ...],
    matrix: EvaluationMatrix,
    generated_at: datetime,
    model: E2EModelConfig | None = None,
    e2e_aggregate: E2EAggregate | None = None,
    gate: ReleaseGateResult | None = None,
) -> GovernanceEvaluationReport:
    manifest = EvaluationRunManifest(
        mode=mode,
        generated_at=generated_at,
        python_version=platform.python_version(),
        platform=platform.platform(),
        git_revision=_git_revision(),
        git_dirty=_git_is_dirty(),
        scenario_catalog_fingerprint=decision_fingerprint(scenarios),
        scenario_schema_versions=tuple(sorted({item.schema_version for item in scenarios})),
        profiles=tuple(build_profile_manifest(item) for item in profiles),
        model_targets=(
            (
                ModelTargetSummary(
                    target_id=model.target_id,
                    model_id=model.model_id,
                    config_fingerprint=model.fingerprint,
                ),
            )
            if model is not None
            else ()
        ),
    )
    return GovernanceEvaluationReport(
        manifest=manifest,
        batch=build_batch_result(
            results,
            suite_id=f"aar-governance-{mode}-v1",
            generated_at=generated_at,
        ),
        matrix=matrix,
        e2e_aggregate=e2e_aggregate,
        release_gate=gate,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    deterministic = subparsers.add_parser("deterministic")
    _output_argument(deterministic)
    deterministic.add_argument(
        "--profile",
        choices=tuple(item.value for item in ScenarioProfile),
        default=ScenarioProfile.FULL_AAR.value,
    )
    deterministic.add_argument("--cases", help="comma-separated scenario IDs")
    compare = subparsers.add_parser("compare")
    _output_argument(compare)
    compare.add_argument(
        "--profiles",
        required=True,
        help="comma-separated Profile names",
    )
    compare.add_argument("--cases", help="comma-separated scenario IDs")
    gate = subparsers.add_parser("gate")
    _output_argument(gate)
    gate.add_argument("--cases", help=argparse.SUPPRESS)
    demo = subparsers.add_parser("demo")
    _output_argument(demo)
    demo.add_argument("--cases", help=argparse.SUPPRESS)
    e2e = subparsers.add_parser("e2e")
    _output_argument(e2e)
    e2e.add_argument("--service", choices=tuple(item.value for item in OpenAICompatibleService))
    e2e.add_argument("--target-id")
    e2e.add_argument("--model-id")
    e2e.add_argument("--base-url")
    e2e.add_argument("--api-key-env")
    e2e.add_argument("--no-api-key", action="store_true")
    e2e.add_argument("--profile", default=ScenarioProfile.FULL_AAR.value)
    e2e.add_argument("--cases", help="comma-separated scenario IDs")
    e2e.add_argument("--repeat", type=int, default=1)
    e2e.add_argument("--max-output-tokens", type=int, default=400)
    e2e.add_argument("--timeout-seconds", type=float, default=30.0)
    e2e.add_argument("--max-attempts", type=int, default=1)
    return parser


def _output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)


def _profiles_for(args: argparse.Namespace) -> tuple[ScenarioProfile, ...]:
    if args.command == "gate":
        return tuple(ScenarioProfile)
    if args.command == "demo":
        return (ScenarioProfile.FULL_AAR,)
    if args.command == "deterministic":
        return (ScenarioProfile(args.profile),)
    return tuple(ScenarioProfile(item) for item in _csv(args.profiles) or ())


def _csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("comma-separated option cannot be empty")
    return items


def _git_revision() -> str | None:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={PROJECT_ROOT.as_posix()}",
            "rev-parse",
            "HEAD",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


def _git_is_dirty() -> bool:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={PROJECT_ROOT.as_posix()}",
            "status",
            "--porcelain",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode != 0 or bool(completed.stdout.strip())


def _database_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted((PROJECT_ROOT / "data").glob("*.sqlite3")):
        snapshot[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def _print_outcome(
    report: GovernanceEvaluationReport,
    paths: tuple[Path, Path],
) -> None:
    passed = sum(
        item.evaluation_verdict is EvaluationVerdict.PASS
        for item in report.batch.results
    )
    print(f"Results: {passed}/{len(report.batch.results)} Oracle-pass")
    if report.release_gate is not None:
        print(f"Release gate: {'PASS' if report.release_gate.passed else 'FAIL'}")
    print(f"JSON: {paths[0]}")
    print(f"Markdown: {paths[1]}")


if __name__ == "__main__":
    raise SystemExit(main())
