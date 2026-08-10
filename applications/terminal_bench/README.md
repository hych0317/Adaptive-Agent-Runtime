# AAR Terminal Sequential Profile

This adapter runs one isolated `AgentRuntime` per Terminal-Bench trial and one
terminal command per Core Action. It deliberately does not add a dynamic DAG,
cross-task memory, automatic learning, auto-adaptation, or a second Runtime.

## Execution and authority path

```text
TerminalTurnDraft
→ ToolInvocationProposalDraft
→ DecisionRequest
→ Runtime Validation
→ ToolInvocationEffect
→ Governance
→ fingerprint-bound Apply
→ PermitBoundToolExecutor
→ Harbor BaseEnvironment.exec()
→ Core Observation commit
```

`command`, `cwd`, the complete explicit `env` map, `timeout_sec`, process
reference metadata, and `trial_id` are all Tool invocation arguments and are
therefore part of the normalized Effect fingerprint. The generic Decision
Lifecycle persists the original Proposal, Effect, and authorization. A resumed
apply either reconciles the original invocation from the transcript or fails
closed; it never automatically repeats an unknown command.

Every `exec()` is an independent, non-interactive shell. A previous `cd`,
`export`, alias, or shell variable is not assumed to persist. Background
services must be detached explicitly and record a PID file, log path, and
status-check command. This profile does not expose tmux or an interactive
terminal capability.

Commands are classified as `inspect`, `work`, or `verify`. A new, distinct
`inspect` result is recovery progress; repeating the same inspection result is
no progress. Once the Runtime enters recovery mode, another inspection-only
proposal is rejected in favor of a targeted repair or verification command.
Host-side Codex inference workspaces such as `/tmp/aar-codex-inference-*` do
not exist in the task container and are rejected before execution.

## Completion gate

The command role is part of the governed Effect fingerprint. A command that
creates or changes a task artifact is `work`. Before declaring `complete`, the
Agent must run an independent `verify` command that encodes its checks in the
process exit status. Independent verification must name the requirements,
artifact paths, validation methods, and artifact/format/semantic/end-to-end
coverage. Official-test verification must identify the actual official test
path or command. Known state-polluting checks, including creation of Git refs,
are rejected before execution.

The Runtime accepts completion only when the last committed command is
`verify`, has a known `COMPLETED` result, returns zero, and has no timeout or
transport failure. The verification must follow at least one `work` command.
A failed verification may be followed by further evidence-driven repair and
verification cycles while the same trial remains within its command, token,
elapsed-time, inspection, and no-progress budgets. A successful verification
creates a locked checkpoint and the Runtime finalizes it without permitting
later mutations.

Any evaluation rule that permits at most one correction per task applies to an
external post-trial correction and rerun. It is enforced by the evaluation
workflow, not by limiting the Agent's repair iterations inside one trial.

A premature `complete` draft is not immediately treated as a successful stop or
as a terminal Runtime failure. Its blocker is persisted in the trial-local
session and returned to the model for bounded structural proposal feedback.
These attempts remain subject to the same inference token, cost, and
elapsed-time budgets; exhausting the configured proposal-rejection limit fails
closed.

This is self-attested Agent-side validation evidence, not a semantic proof or
the benchmark oracle. The Runtime can bind the declared role to the exact
executed Effect and require a successful exit status, but it cannot infer from
an arbitrary shell command whether the check is sufficient for the task. AAR
does not invoke or inspect the Harbor verifier, and `agent_complete` still does
not imply a benchmark pass.

## Harbor boundary

The adapter can reserve cleanup and delivery time inside Harbor's outer Agent
timeout. Pass the effective task Agent timeout explicitly; for a 900-second
task, the recommended settings are:

```text
--ak agent_timeout_sec=900 --ak deadline_reserve_seconds=60
```

This gives the Runtime an 840-second wall-clock budget and switches its prompt to
delivery mode after 70% of that budget. The adapter never extends Harbor's
outer timeout.

AAR provides no verifier or oracle API. It restricts the Agent to the current
trial's `BaseEnvironment`; actual verifier isolation is Harbor task/environment
configuration's responsibility. The adapter does not expose the host file
system, Docker socket, or sidecar/service execution, does not override Harbor's
default Agent user, and does not put model API keys in the task container.
Command size, timeout, token, cost, and action-count budgets are Runtime-owned.

Harbor is an optional dependency and appears only in `harbor_agent.py`. From the
repository root, run the current Harbor import form:

```bash
harbor run \
  -d terminal-bench/terminal-bench-2-1 \
  --agent "applications.terminal_bench.harbor_agent:AdaptiveRuntimeHarborAgent" \
  --model "<provider/model>" \
  --ak agent_timeout_sec=900 \
  --ak deadline_reserve_seconds=60
```

The local adapter writes these host-side trial logs:

- `aar-runtime.sqlite3`: Core, Decision, Governance, and authorization records.
- `aar-transcript.jsonl`: append-only terminal proposal/execution/commit facts.
- `aar-summary.json`: AAR-only run summary. `agent_complete` means only that the
  Agent voluntarily stopped; it is not a benchmark pass.

Harbor runs the verifier after `AdaptiveRuntimeHarborAgent.run()`. Only then may
`TerminalResultAnalyzer` combine the Harbor `TrialResult.verifier_result` with
`aar-summary.json`. The analyzer is intentionally not called by the Agent.
Verifier dependency setup failures with corroborating network evidence are
reported as `infrastructure_error`, separately from a genuine benchmark
failure; a network-looking string alone is not enough to reclassify a result.

## Luna through Codex CLI

`codex-cli/luna-*` models use the installed Codex CLI only as a bounded,
read-only inference backend. Runtime remains the sole command authority. The
default Codex CLI reasoning effort is `high`; it is passed as
`model_reasoning_effort="high"`. A side-effect-free CLI `turn.failed` event is
retried once before any Runtime action is proposed. Timeouts, process failures,
protocol errors, and action-bearing CLI traces are not automatically retried.

## Fixed Core48 evaluation set

The version-controlled task manifest is
[`evaluation_sets/core48.txt`](evaluation_sets/core48.txt). It contains exactly
48 unique, fully qualified Terminal-Bench 2.1 task names. The visual-input tasks
`chess-best-move` and `code-from-image` are intentionally excluded and replaced
by `pypi-server` and `fix-code-vulnerability`.

Build repeated Harbor include arguments without relying on dataset ordering:

```bash
CORE48_ARGS=()
while IFS= read -r task; do
  CORE48_ARGS+=(--include-task-name "$task")
done < applications/terminal_bench/evaluation_sets/core48.txt

harbor run \
  -d terminal-bench/terminal-bench-2-1 \
  "${CORE48_ARGS[@]}" \
  --agent "applications.terminal_bench.harbor_agent:AdaptiveRuntimeHarborAgent" \
  --model "codex-cli/luna-high" \
  --env docker \
  --n-concurrent 2 \
  --n-attempts 1 \
  --max-retries 0 \
  --ak agent_timeout_sec=900 \
  --ak deadline_reserve_seconds=60 \
  --jobs-dir ./jobs \
  --job-name "aar-tb21-core48-<DATE>" \
  --debug
```

The command is documentation only; importing or testing this adapter never
starts a Harbor evaluation.
