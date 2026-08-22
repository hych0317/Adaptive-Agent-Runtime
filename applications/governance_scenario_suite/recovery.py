"""Cross-process runner for named refund crash and reconciliation scenarios."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import UUID, uuid5

from adaptive_agent_runtime.decisioning import decision_fingerprint
from applications.ecommerce_support.composition import compose_ecommerce_support
from applications.ecommerce_support.models import DomainSnapshot
from applications.governance_scenario_suite.contracts import (
    FaultPoint,
    ReasonCode,
    ScenarioSpec,
    ScenarioVerdict,
)
from applications.governance_scenario_suite.evidence import (
    EvidenceBundle,
    EvidenceSource,
    ScenarioExecution,
    ScenarioRunResult,
)
from applications.governance_scenario_suite.oracles import DeterministicScenarioOracle


_RECOVERY_NAMESPACE = UUID("b2392360-43f2-42b4-ae78-b98393605ee3")


class CrossProcessRecoveryRunner:
    def __init__(self, oracle: DeterministicScenarioOracle | None = None) -> None:
        self._oracle = oracle or DeterministicScenarioOracle()

    def run(self, scenario: ScenarioSpec, *, workspace: str | Path | None = None) -> ScenarioRunResult:
        if workspace is None:
            with TemporaryDirectory(prefix="aar-recovery-") as directory:
                return self._run(scenario, Path(directory))
        return self._run(scenario, Path(workspace).resolve())

    def _run(self, scenario: ScenarioSpec, root: Path) -> ScenarioRunResult:
        root.mkdir(parents=True, exist_ok=True)
        fingerprint = decision_fingerprint(scenario)
        run_id = uuid5(_RECOVERY_NAMESPACE, f"{scenario.id}|{scenario.profile.value}|{fingerprint}")
        run_root = root / str(run_id)
        run_root.mkdir(parents=True, exist_ok=False)
        spec_path = run_root / "scenario.json"
        spec_path.write_text(scenario.model_dump_json(indent=2), encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "applications.governance_scenario_suite.recovery_worker",
            "--root",
            str(run_root),
            "--scenario",
            str(spec_path),
        ]
        is_normal = not scenario.fault_schedule
        started = subprocess.run(
            [*command, "--action", "normal" if is_normal else "crash"],
            check=False,
            capture_output=True,
            text=True,
        )
        if is_normal:
            if started.returncode != 0:
                raise RuntimeError(started.stderr)
        else:
            fault = scenario.fault_schedule[0].point
            expected_code = 91 if fault is FaultPoint.AFTER_EXTERNAL_COMMIT_BEFORE_LOCAL_RECEIPT else 92
            if started.returncode != expected_code:
                raise RuntimeError(
                    f"crash worker exited {started.returncode}, expected {expected_code}: {started.stderr}"
                )
            resumed = subprocess.run(
                [*command, "--action", "recover"],
                check=False,
                capture_output=True,
                text=True,
            )
            if resumed.returncode != 0:
                raise RuntimeError(resumed.stderr)
        outcome = json.loads((run_root / "outcome.json").read_text(encoding="utf-8"))
        pre = DomainSnapshot.model_validate_json(
            (run_root / "pre_state.json").read_text(encoding="utf-8")
        )
        composition = compose_ecommerce_support(run_root, clock=lambda: scenario.clock)
        try:
            post = composition.store.snapshot()
            ledger = composition.gateway.ledger()
            execution = ScenarioExecution(
                decision=ScenarioVerdict(outcome["decision"]),
                reason_code=(ReasonCode(outcome["reason_code"]) if outcome["reason_code"] else None),
                available_sources=frozenset({EvidenceSource.AUDIT}),
                model_call_count=outcome["model_call_count"],
            )
            evidence = EvidenceBundle(
                scenario=scenario,
                execution=execution,
                pre_state=pre,
                post_state=post,
                external_ledger=ledger,
                available_sources=frozenset(
                    {EvidenceSource.AUTHORITATIVE_STATE, EvidenceSource.EXTERNAL_LEDGER, EvidenceSource.AUDIT}
                ),
            )
            verdict, findings, summary = self._oracle.evaluate(evidence)
        finally:
            composition.close()
        return ScenarioRunResult(
            run_id=run_id,
            scenario_id=scenario.id,
            family_id=scenario.family_id,
            profile=scenario.profile,
            spec_fingerprint=fingerprint,
            evaluation_verdict=verdict,
            actual_decision=execution.decision,
            actual_reason_code=execution.reason_code,
            findings=findings,
            evidence=summary,
            metrics={"model_call_count": execution.model_call_count, "start_exit_code": started.returncode},
            started_at=scenario.clock,
            completed_at=scenario.clock,
        )
