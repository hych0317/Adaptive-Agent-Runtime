# Third-party notices

## Open Design

Parts of the local CLI discovery and connection implementation are adapted
from [nexu-io/open-design](https://github.com/nexu-io/open-design), local
source snapshot `2e377a5152bf6d3d1c25b0e85b0e00fa41b07644`.

Copyright 2026 Open Design contributors.

Open Design is licensed under the Apache License, Version 2.0. The applicable
license text is included in `LICENSES/Apache-2.0.txt`.

Adapted source areas:

- `apps/daemon/src/runtimes/defs/codex.ts`
- `apps/daemon/src/runtimes/detection.ts`
- `apps/daemon/src/runtimes/auth.ts`
- `apps/daemon/src/runtimes/executables.ts`
- `apps/daemon/src/runtimes/launch.ts`
- `apps/daemon/src/runtimes/invocation.ts`
- `packages/platform/src/toolchain.ts`
- `packages/platform/src/command.ts`

The code was translated from TypeScript to Python and integrated only at the
backend discovery/process boundary. Adaptive Agent Runtime's Capability,
Context Adapter, Gateway, response validation, execution, memory, and
governance contracts are not derived from Open Design.

