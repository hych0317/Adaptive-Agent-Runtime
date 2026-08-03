# Open Design backend detection port

## Scope

This port is intentionally limited to local CLI discovery, readiness probing,
and process connection. It does not replace Adaptive Agent Runtime's cognitive
capabilities or authority boundaries.

```text
Adaptive Runtime Capability
        ↓
Managed Gateway + Context Adapter
        ↓
Adaptive Runtime response contract
        ↓
ported CLI discovery / connection layer
        ↓
Codex CLI
```

## Source mapping

| Open Design source | Adaptive Agent Runtime target | Ported behavior |
|---|---|---|
| `runtimes/executables.ts` and `platform/toolchain.ts` | `providers/process.py`, `providers/cli_integration.py` | PATH plus user-toolchain discovery and explicit binary override |
| `runtimes/launch.ts` | `providers/cli_integration.py` | detection and invocation share one launch path; Codex npm wrapper upgrades to the packaged native binary |
| `runtimes/detection.ts` | `providers/cli_integration.py` | only spawn failures and exit 126/127 mark a CLI unavailable; generic version failures and timeouts retain availability |
| `runtimes/auth.ts` | `providers/cli_integration.py` | authentication is `ok`, `missing`, or `unknown` independently from executable availability |
| `runtimes/defs/codex.ts` | `providers/codex_cli.py` | declarative version/auth probe commands and stdin prompt delivery |
| `runtimes/invocation.ts` and `platform/command.ts` | `providers/process.py`, native Codex launch resolution | bounded subprocess execution; Windows npm shims are avoided when their native Codex binary is available |

The local source snapshot used for the port is
`2e377a5152bf6d3d1c25b0e85b0e00fa41b07644`.

## Project-owned behavior

The following remains implemented exclusively by Adaptive Agent Runtime:

- Reasoning, planning, generation, judging, compression, and extraction
- Capability request/result schemas
- Context packaging, sensitivity policy, and token budgets
- Managed Gateway routing, retry, trace, and response validation
- ToolIntent governance and Runtime-owned execution
- Task Graph, Memory, Evaluation, and Governance
- Inference-only and autonomous-agent permission boundaries

## Intentional adaptations

- The source TypeScript implementation is translated to typed asynchronous
  Python interfaces.
- Codex inference keeps this project's read-only, no-tool, ephemeral policy.
- Current npm Codex packages may nest the native platform package under
  `@openai/codex/node_modules/@openai`; this layout is handled in addition to
  Open Design's hoisted-package search.
- Prompt contents remain on stdin and are never interpolated into a Windows
  command line.
- Raw auth output, API keys, and OAuth material are not included in Runtime
  diagnostics.

