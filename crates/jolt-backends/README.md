# jolt-backends

Backend compute rails for modular proving.

This crate owns backend traits plus request/result types. Protocol crates own
stage order, claims, transcript labels, and verifier-visible proof shape.

North-star references for CPU prover replacement work:

- [`specs/jolt-prover-cpu-backend-port.md`](../../specs/jolt-prover-cpu-backend-port.md)
- [`specs/jolt-prover-frontier-harness.md`](../../specs/jolt-prover-frontier-harness.md)
- [`specs/jolt-core-prover-optimization-inventory.md`](../../specs/jolt-core-prover-optimization-inventory.md)
- [`crates/jolt-prover-harness/src/optimization.rs`](../jolt-prover-harness/src/optimization.rs)

Structure rules:

- request/result contracts live by family, for example
  `commitments/request.rs`, `sumcheck/request.rs`, and
  `openings/request.rs`;
- concrete CPU code lives under `cpu/` and keeps request orchestration separate
  from hot compute helpers;
- for the first CPU backend, performance beats modularity. Prefer coarse,
  relation-specific kernels that look like the optimized `jolt-core` routines
  over tidy generic abstractions when those goals conflict;
- CPU hot paths may be Jolt-specific and aggressively optimized, but they
  consume explicit requests and return slot-keyed results;
- root request/result contracts should stay hardware-agnostic and reasonably
  protocol-agnostic; protocol meaning enters through backend-local relation
  IDs, witness oracle refs, value slots, and caller-supplied request labels;
- CPU modules may split by protocol primitive when optimization demands it,
  but only behind the generic request family that scheduled the work;
- hot CPU requests should carry relation and optimization metadata when that
  helps the harness account for core-parity work without changing protocol
  semantics;
- CPU kernel ports should add or extend focused microbenchmarks before being
  marked parity-ready:
  `cargo bench -p jolt-backends --bench sumcheck_kernels`;
- run the focused microbenchmark and inspect time plus analytical memory before
  integrating the kernel through `jolt-prover`; long prover-path iteration is
  reserved for kernels that already look plausible in isolation;
- if `jolt-core` has a specialized algorithm for a relation, the CPU backend
  port should preserve that algorithm directly before `jolt-prover` wires the
  stage. Generic sparse or materialized fallbacks are reference paths, not
  performance-parity implementations;
- every optimized CPU kernel must be accounted for by the harness backend
  kernel ledger with optimization IDs, source locations, entrypoints,
  microbenchmarks, and parity status;
- the global CPU-backend inventory coverage check must pass before goal-mode
  proceeds to prover-frontier integration;
- promotion to `ParityCertified` requires `KernelBenchmarkEvidence` that passes
  the 15% timing and peak-memory threshold plus analytical memory accounting;
- certified kernel ledger entries must name JSON evidence files that the
  harness can load and validate;
- benchmark evidence should be written with the harness canonical artifact
  helper so later validation does not require rerunning the benchmark;
- representation-specific commitment behavior, such as dense streaming versus
  one-hot sparse commitment, is hidden behind one CPU commitment accumulator
  path;
- `field-inline` and `zk` features are backend capability gates, not protocol
  schedulers.

The canonical CPU backend must preserve the current `jolt-core` prover's
streaming, memory, Dory hint, advice, BlindFold, and one-hot/RA fast paths.

Fast iteration order:

1. Backend unit/reference tests for the touched request family.
2. Benchmark compile check with `cargo bench -p jolt-backends --bench sumcheck_kernels --no-run`.
3. Focused backend microbench with analytical memory accounting.
4. Harness frontier replay only after the kernel is plausible in isolation.
5. Canonical prover perf only after the narrow replay passes.
