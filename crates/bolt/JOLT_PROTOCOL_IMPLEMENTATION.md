# Jolt Protocol Implementation Plan

This is the stage-addition playbook for the Bolt implementation of Jolt. The
goal is to make every new protocol stage boring to add: first specify the stage
in MLIR, then lower it through the role/compute/CPU pipeline, then emit uniform
Rust artifacts, then wire arithmetic, real-data equivalence, tamper tests, and
performance gates.

## Current State

Commitment, Stage 1, and Stage 2 have the active compiler rails: protocol MLIR,
per-party lowering, compute/CPU lowering, generated Rust fixtures, and real-data
equivalence tests against `jolt-core`. After the upstream sync, the generated
crate and verifier rails are green, but the real-data equivalence gates expose
Stage 1/2 proof-coefficient divergence that must be repaired before either
stage is treated as complete. Stage 3 has protocol/codegen/arithmetic work in
progress and is in the optimization/equivalence-hardening phase; it is not
considered complete until the real-data internal/core parity gates, tamper
checks, and sub-20% stage perf gate are green.

## Stage Algorithm

For a new stage `N`, do these steps in order. Do not treat synthetic tests as an
acceptance gate; they are only unit scaffolding before real trace data is wired.

1. **Scope the stage contract**
   - Identify the paper-level stage purpose, claims, sumchecks, transcript
     labels, opening obligations, and proof slots.
   - Identify the matching `jolt-core` verifier/prover fields and the exact
     prefix boundary where Bolt artifacts can be spliced into a core proof.
   - Write down any new data dependencies from earlier stages as typed opening
     inputs, not ad hoc runtime lookups.

2. **Model protocol MLIR first**
   - Encode protocol facts under `protocol`, `piop`, `poly`, `field`,
     `transcript`, `commit`, and `pcs`.
   - Keep Jolt-specific facts inside `protocols/jolt`; generic dialect schemas
     must not learn Jolt-only names.
   - Use SSA operands/results for claims, points, transcript state, and opening
     flow. Attributes are for stable metadata such as names, domains, counts,
     labels, policies, and proof slots.
   - Add validation before lowering: ordered claims, compatible opening
     equalities, expected point arities, round counts, degree, proof-slot
     uniqueness, transcript threading, and role visibility.

3. **Project roles and lower through MLIR**
   - Lower `protocol -> concrete` with explicit Fiat-Shamir state.
   - Project `concrete -> party` for prover and verifier.
   - Lower each party to `compute`, preserving SSA dataflow.
   - Resolve prover kernels as a compute-to-compute pass. Verifier IR must not
     receive kernels.
   - Lower `compute -> cpu`. CPU IR is still MLIR; Rust is only emitted after
     this point.

4. **Emit uniform Rust artifacts**
   - The compiler output is a pair of role crates:
     `jolt-prover` and `jolt-verifier`.
   - Stage modules live at `src/stages/<stage>.rs` in both crates.
   - `assemble_jolt_generated_crates` is the canonical crate assembly rail; it
     must emit manifests, `src/lib.rs`, `src/stages/mod.rs`, and every stage
     module in a shape that `cargo check` accepts.
   - `jolt-prover` may import coarse, unstable CPU kernels from
     `jolt-kernels`.
   - `jolt-verifier` must import only audit-scope modular crates and local
     generated modules; it must not import `jolt-kernels`, `jolt-core`,
     `jolt-equivalence`, `jolt-bench`, or tracer internals.
   - Generated verifier code should be explicit, human-readable glue around
     modular verifier APIs. If verifier logic is hidden in `jolt-kernels`, the
     stage is not audit-stable.

5. **Wire arithmetic**
   - Start with the coarsest prover kernels that can plausibly reach performance
     parity with `jolt-core`.
   - Implement kernels below the dialect boundary. Kernel internals can be
     optimized aggressively, but the protocol order, claims, transcript labels,
     and opening obligations must remain owned by MLIR.
   - Reuse or improve modular primitive crates for verifier-side proof
     structure and checking. Treat `jolt-kernels` as prover-side implementation
     detail, not soundness-critical verifier infrastructure.

6. **Add real-data equivalence**
   - Use real traced program data for the acceptance gate. Current standard
     workload is `sha2-chain` for perf and real program fixtures such as
     `muldiv` for focused semantic debugging.
   - Run Bolt prover and Bolt verifier on the same trace and assert proof
     acceptance plus transcript-state equality through the new stage boundary.
   - Splice Bolt stage artifacts into the matching `jolt-core` proof prefix and
     assert the core verifier accepts through the same boundary.
   - Compare Bolt vs core transcript states and observable proof components
     step-for-step through the stage: sumcheck round polynomials, opening
     claims, normalized points, commitments, and proof slots as applicable.

7. **Harden with negative tests**
   - Add tamper tests for each new verifier obligation: sumcheck coefficients,
     claimed output, opening evaluations, opening points, equality constraints,
     batching order, transcript labels, and proof-slot shape.
   - Each tamper should fail in the generated/Bolt verifier before any core
     fallback is consulted.
   - Add schema-level negative tests for malformed IR that could otherwise
     erase or reorder a soundness obligation.

8. **Optimize after correctness**
   - Use `jolt-bench`, `jolt-profiling`, and targeted instrumentation in
     `jolt-core`/Bolt kernels to isolate gaps.
   - Add the stage to the `bolt-stage` bench selector before claiming perf
     parity.
   - Perf gates use at most three timing iterations for iteration speed.
   - A stage is performance-complete only when Bolt prover time for the newly
     wired stage is within 20% of the corresponding `jolt-core` stage on the
     agreed workload. Re-run correctness after any accepted optimization.

9. **Mark the stage done**
   - Check in MLIR fixtures for protocol/concrete/prover party/verifier
     party/prover compute/verifier compute/prover CPU/verifier CPU.
   - Check in generated Rust fixtures for both role artifacts.
   - Keep `jolt-equivalence` gates named by stage and using real data.
   - Update the stage status matrix and only then start scoping the next stage.

## Acceptance Criteria

A stage is done only when all conditions below are true:

- **MLIR correctness**: protocol, concrete, party, compute, kernelized compute,
  and CPU schemas pass for both roles; verifier IR is kernel-free.
- **Artifact correctness**: generated prover and verifier modules compile under
  the canonical `jolt-prover`/`jolt-verifier` layout, including whole-crate
  `cargo check` of the generated manifests and stage module graph.
- **Import policy**: generated verifier imports only modular audit-scope crates;
  generated prover does not import `jolt-core` or equivalence-only crates.
- **Internal real-data parity**: Bolt prover output is accepted by the Bolt
  verifier on real trace data, and prover/verifier transcript states match
  step-for-step through the stage boundary.
- **Core acceptance**: `jolt-core` accepts a proof prefix whose stage artifacts
  were produced by Bolt.
- **Core transcript/artifact parity**: Bolt and `jolt-core` match transcript
  states and observable stage artifacts through the implemented boundary.
- **Tamper coverage**: generated verifier rejects representative mutations for
  every new soundness obligation introduced by the stage.
- **Performance**: Bolt prover time for the new stage is within 20% of
  `jolt-core` on the agreed `sha2-chain` workload, with perf gates capped at
  three iterations.

## Required Gates

Use focused gates during development, then run the full stage gate before moving
on:

```bash
cargo nextest run -p bolt --cargo-quiet
cargo check -p jolt-equivalence --tests --quiet
cargo nextest run -p jolt-equivalence --cargo-quiet
cargo nextest run -p jolt-kernels --cargo-quiet
```

For stage perf:

```bash
cargo run --release -p jolt-bench --bin bolt-stage -- \
  --program sha2-chain --stage <stage> --log-t 16 \
  --num-iters 16 --iters 3 --warmup 1 \
  --json perf/bolt-<stage>-last.json
```

Run `jolt-core` first when comparing wall time. If Bolt exceeds a 10x timeout
relative to core, stop timing and profile rather than waiting for a full run.
