---
name: compiler-runtime split
description: The jolt-zkvm compiler expresses the full protocol; the runtime is a dumb executor. No escape hatches or parallel prover systems.
type: feedback
originSessionId: 646687f6-8840-4f23-8f2f-cd018fb3170f
---
The jolt-zkvm architecture has a strict compiler/runtime split:
- The **compiler** (jolt-compiler module builder) expresses the complete protocol as a sequence of Ops, KernelDefs, InputBindings, and Iterations
- The **runtime** (jolt-zkvm runtime.rs) is a generic executor that walks Ops, manages state, and gets data from providers

**Why:** The goal is semantic equivalence with jolt-core within the new modular crate design. If the kernel system can't express something, extend the kernel system (new InputBinding variants, new Iteration modes) rather than adding parallel prover dispatch or escape hatches.

**How to apply:** When encountering complex sumcheck instances (prefix-suffix decomposition, multi-phase reductions, eq-gathered inputs), extend KernelSpec/InputBinding/Iteration to express them. Never add "InstanceProverKind" or "NativeProver" escape hatches. The runtime stays dumb.
