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

## Completion gate

Each execute draft labels its command as `work` or `verify`. This role is part
of the governed Effect fingerprint. A command that creates or changes a task
artifact is `work`. Before declaring `complete`, the Agent must run an
independent `verify` command that encodes its checks in the process exit status.
The Runtime accepts completion only when the last committed command is `verify`,
has a known `COMPLETED` result, returns zero, and has no timeout or transport
failure. The verification must follow at least one `work` command, and any later
`work` command invalidates that evidence.

A premature `complete` draft is not immediately treated as a successful stop or
as a terminal Runtime failure. Its blocker is persisted in the trial-local
session and returned to the model for a bounded correction attempt. These
attempts remain subject to the same inference token, cost, and elapsed-time
budgets; exhausting the configured correction limit fails closed.

This is self-attested Agent-side validation evidence, not a semantic proof or
the benchmark oracle. The Runtime can bind the declared role to the exact
executed Effect and require a successful exit status, but it cannot infer from
an arbitrary shell command whether the check is sufficient for the task. AAR
does not invoke or inspect the Harbor verifier, and `agent_complete` still does
not imply a benchmark pass.

## Harbor boundary

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
  --model "<provider/model>"
```

The local adapter writes these host-side trial logs:

- `aar-runtime.sqlite3`: Core, Decision, Governance, and authorization records.
- `aar-transcript.jsonl`: append-only terminal proposal/execution/commit facts.
- `aar-summary.json`: AAR-only run summary. `agent_complete` means only that the
  Agent voluntarily stopped; it is not a benchmark pass.

Harbor runs the verifier after `AdaptiveRuntimeHarborAgent.run()`. Only then may
`TerminalResultAnalyzer` combine the Harbor `TrialResult.verifier_result` with
`aar-summary.json`. The analyzer is intentionally not called by the Agent.
