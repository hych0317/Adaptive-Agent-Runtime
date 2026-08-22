"""Safe JSON and Markdown reports derived only from ScenarioRunResult."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from applications.governance_scenario_suite.contracts import EvaluationVerdict
from applications.governance_scenario_suite.evidence import (
    ScenarioBatchResult,
    ScenarioRunResult,
)


def build_batch_result(
    results: tuple[ScenarioRunResult, ...],
    *,
    suite_id: str,
    generated_at: datetime,
) -> ScenarioBatchResult:
    return ScenarioBatchResult(
        suite_id=suite_id,
        results=results,
        generated_at=generated_at,
    )


def write_reports(
    batch: ScenarioBatchResult,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    root = Path(output_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "governance-scenarios.json"
    markdown_path = root / "governance-scenarios.md"
    json_path.write_text(batch.model_dump_json(indent=2), encoding="utf-8")
    run_directory = root / "runs"
    run_directory.mkdir(exist_ok=True)
    for result in batch.results:
        run_name = (
            f"{result.scenario_id}__{result.profile.value}__{result.run_id}.json"
        )
        (run_directory / run_name).write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
    persisted_batch = ScenarioBatchResult.model_validate_json(
        json_path.read_text(encoding="utf-8")
    )
    markdown_path.write_text(render_markdown(persisted_batch), encoding="utf-8")
    return json_path, markdown_path


def render_markdown(batch: ScenarioBatchResult) -> str:
    passed = sum(
        item.evaluation_verdict is EvaluationVerdict.PASS for item in batch.results
    )
    failed = sum(
        item.evaluation_verdict is EvaluationVerdict.FAIL for item in batch.results
    )
    inconclusive = sum(
        item.evaluation_verdict is EvaluationVerdict.INCONCLUSIVE
        for item in batch.results
    )
    lines = [
        "# AAR Governance Scenario Suite Report",
        "",
        f"- Suite: `{batch.suite_id}`",
        f"- Generated: `{batch.generated_at.isoformat()}`",
        f"- Release gate: `{'PASS' if batch.release_gate_passed else 'FAIL'}`",
        f"- Results: `{passed} passed / {failed} failed / {inconclusive} inconclusive`",
        "",
        "| Scenario | Profile | Evaluation | Decision | Reason | External effects |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for result in batch.results:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{result.scenario_id}`",
                    f"`{result.profile.value}`",
                    f"`{result.evaluation_verdict.value}`",
                    f"`{result.actual_decision.value}`",
                    (
                        f"`{result.actual_reason_code.value}`"
                        if result.actual_reason_code is not None
                        else "—"
                    ),
                    str(result.evidence.external_effect_count),
                )
            )
            + " |"
        )
    lines.extend(("", "## Findings", ""))
    for result in batch.results:
        non_passing = tuple(
            item
            for item in result.findings
            if item.verdict is not EvaluationVerdict.PASS
        )
        if not non_passing:
            lines.append(f"- `{result.scenario_id}`: all deterministic checks passed.")
            continue
        for finding in non_passing:
            lines.append(
                f"- `{result.scenario_id}` / `{finding.code}` / "
                f"`{finding.verdict.value}`: {finding.message}"
            )
    lines.append("")
    return "\n".join(lines)
