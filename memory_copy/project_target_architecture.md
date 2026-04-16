---
name: Target architecture — ML compiler philosophy
description: North star for jolt-zkvm: compiler lowers all protocol knowledge to primitive ops, runtime is a flat mechanical dispatcher
type: project
originSessionId: f0719e20-9c07-4478-bf0f-7ad6f4e1a5cf
---
## ML Compiler Philosophy

jolt-zkvm follows an ML compiler model. The **compiler** (jolt-compiler) lowers protocol descriptions into flat sequences of primitive ops. The **runtime** (jolt-zkvm) executes them mechanically — every handler ≤ 30 LOC, zero interpretation of protocol data.

### Current State (2026-04-15)
- Standard sumcheck path: DONE — fully compiled, handlers are 15-20 LOC
- Address-decomposition path: VIOLATION — 5 macro ops with complex handlers (220, 114, 73, 37, 20 LOC) + checkpoint_eval.rs (1059 LOC mini-interpreter)
- BytecodeVal InputBinding: VIOLATION — protocol-specific materialization
- LookupTraceData: VIOLATION — protocol-specific struct in RuntimeState

### Tracking
- `TASKS.md` (repo root): ordered violation list with status
- `crates/jolt-zkvm/ARCHITECTURE.md`: north star design doc
- `CLAUDE.md`: loop protocol for autonomous execution

**Why:** Long agentic sessions lose context and compromise on design. The task loop + architecture doc keep the agent on track.

**How to apply:** On session start, read TASKS.md, follow the loop protocol in CLAUDE.md. Never compromise — if a handler is > 30 LOC, the abstraction is wrong.
