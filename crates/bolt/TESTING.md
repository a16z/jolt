# Bolt Testing Pattern

The Bolt compiler earns correctness one protocol stage at a time. The full
stage-addition algorithm lives in `JOLT_PROTOCOL_IMPLEMENTATION.md`; this file
defines the concrete gates that make a stage acceptable.

## Stage Done Means

A stage is complete only when all of these are true on real trace data:

- **Bolt acceptance**: generated/Bolt prover artifacts are accepted by the
  generated/Bolt verifier.
- **Bolt transcript parity**: prover and verifier transcript states match
  step-for-step through the stage boundary.
- **Core acceptance**: `jolt-core` accepts the proof prefix after Bolt-produced
  artifacts are spliced into the matching core proof fields.
- **Core transcript/artifact parity**: Bolt matches `jolt-core` transcript
  states and observable proof components through the stage boundary.
- **Tamper rejection**: generated verifier rejects representative mutations for
  every new soundness obligation introduced by the stage.
- **Perf parity**: Bolt prover time for the newly added stage is within 20% of
  `jolt-core` on the agreed `sha2-chain` workload, with perf gates capped at
  three iterations.

Synthetic fixtures are allowed for early unit tests, but they do not count as
stage acceptance.

## Local Compiler Gates

Run:

```bash
cargo nextest run -p bolt --cargo-quiet
```

This verifies:

- IRDL dialect registration and parsing.
- Jolt protocol schema validation.
- Concrete transcript threading.
- Prover/verifier role projection.
- `compute` and `cpu` schema validation.
- Kernel resolution only on prover IR.
- Golden MLIR fixtures for every implemented stage.
- Generated Rust compilation.
- Canonical generated artifact layout:
  `crates/jolt-prover/src/stages/<stage>.rs` and
  `crates/jolt-verifier/src/stages/<stage>.rs`.
- Whole generated role-crate assembly and `cargo check` for `jolt-prover` and
  `jolt-verifier`.
- Checked-in generated crate source stays synchronized with
  `assemble_jolt_workspace_generated_crates` and the artifact writer can
  materialize the same layout under a `crates/` root.
- Top-level generated `prover.rs`/`verifier.rs` APIs are emitted by the same
  artifact rail. `jolt-verifier` owns proof types and must not import
  `jolt-prover`; `jolt-prover` may import verifier-owned proof types but not
  verifier stage internals.
- Generated stage registries match between prover and verifier so
  `jolt-equivalence` and `jolt-bench` can discover the implemented prefix.
- Generated verifier import policy: no `jolt-kernels`, `jolt-core`,
  `jolt-equivalence`, `jolt-bench`, or tracer internals.

## Equivalence Gates

Run:

```bash
cargo check -p jolt-equivalence --tests --quiet
cargo nextest run -p jolt-equivalence --cargo-quiet
```

`jolt-equivalence` is the real-data oracle. Stage tests should be named by stage
and should check both internal Bolt parity and Bolt-vs-core parity. Each newly
wired stage should add or extend tests that:

- Generate the same real trace data for Bolt and core.
- Run Bolt prover and Bolt verifier through the complete implemented prefix.
- Assert Bolt proof acceptance and transcript-state parity.
- Splice Bolt artifacts into the corresponding `jolt-core` proof prefix.
- Assert `jolt-core` accepts through the same prefix.
- Assert transcript states and proof components match core through the stage.
- Mutate representative proof components and assert the generated verifier
  rejects them.

## Kernel Gates

Run:

```bash
cargo nextest run -p jolt-kernels --cargo-quiet
```

Kernel tests can use synthetic data for arithmetic-local coverage, but stage
completion still requires real-data `jolt-equivalence` coverage.

## Perf Gate

Run the stage perf oracle after correctness is green:

```bash
cargo run --release -p jolt-bench --bin bolt-stage -- \
  --program sha2-chain --stage <stage> --log-t 16 \
  --num-iters 16 --iters 3 --warmup 1 \
  --json perf/bolt-<stage>-last.json
```

Run core first for a fair timeout baseline. If Bolt exceeds 10x the core time,
stop and profile rather than waiting for a full timing run. The selector runs
core first, checks correctness before timing Bolt, and exits nonzero when Bolt
exceeds `--max-ratio` (default `1.2`). Adding the stage to the `bolt-stage`
bench selector is part of the stage implementation if the selector does not
support it yet.

## Current Stage Status

- **Commitment**: active compiler/equivalence rails are green.
- **Stage 1**: compiler/generated-artifact rails are green; the focused
  real-data equivalence gate passes on `muldiv` after the full-field challenge
  repair, including generated verifier tamper coverage. The `bolt-stage`
  selector supports Stage 1, and the `sha2-chain` release stage perf gate is
  green (`ratio_vs_core ~= 0.79` in `perf/bolt-stage1-last.json`).
- **Stage 2**: compiler/generated-artifact rails are green; the focused
  real-data product-uniskip and batched-sumcheck parity gates pass on `muldiv`
  after the full-field challenge and RAM round-polynomial repairs, including
  generated verifier tamper coverage. The `bolt-stage` selector supports Stage
  2, and the `sha2-chain` release stage perf gate is green
  (`ratio_vs_core ~= 1.19` in `perf/bolt-stage2-last.json`).
- **Stage 3**: compiler/generated-artifact rails are green; the focused
  real-data Bolt prover/verifier parity, core acceptance/transcript/artifact
  parity, kernel/generated Stage 3 verifier tamper gates, and monolithic
  generated verifier acceptance/tamper gates pass on `muldiv`. The default
  checked-in verifier programs are still fixture-shaped, but
  `verify_jolt_with_programs` is covered with real `muldiv` plans that exercise
  44 commitment slots and 9 Stage 3 rounds. The `bolt-stage` selector supports
  Stage 3, and the `sha2-chain` release stage perf gate is green
  (`ratio_vs_core ~= 0.89` in `perf/bolt-stage3-last.json` after split-eq and
  Spartan shift prefix/suffix optimizations).
- **Stage 4**: protocol/lowering/generated Rust rails exist for register
  read/write checking plus RAM value checking. `commitment_ir` covers protocol,
  concrete transcript threading, party, compute, kernelized compute, CPU IR,
  generated Rust fixtures, and checked-in generated artifact crates, including
  the `ram_val_check_gamma` absorb-bytes separator. `jolt-equivalence` and the
  `bolt-stage` correctness gate cover real Bolt-produced Stage 4 prover output
  on `muldiv`, Bolt verifier acceptance, core acceptance of the spliced Stage 4
  proof, core/Bolt transcript and artifact parity, generated Stage 4 verifier
  tamper rejection, and monolithic generated verifier acceptance/tamper
  rejection. The `bolt-stage` selector supports Stage 4, and the `sha2-chain`
  release stage perf gate is green after RAM LT materialization, sparse
  register parallelization, split-eq/Gruen register cycle rounds, reusable
  sparse bind buffers, and sparse read-selector optimizations
  (`ratio_vs_core ~= 1.06` in `perf/bolt-stage4-last.json`).
- **Stage 5**: protocol/lowering/generated Rust rails exist for instruction
  read RAF, RAM RA claim reduction, and register value evaluation. The focused
  `jolt-equivalence` gate covers real Bolt-produced Stage 5 prover output,
  kernel/generated verifier acceptance, core opening/artifact/transcript
  parity, monolithic generated verifier acceptance, and representative Stage 5
  tamper rejection. The `bolt-stage` selector supports Stage 5 and its
  correctness gate passes, and the `sha2-chain` release perf gate is green:
  after grouping duplicate instruction lookup keys, moving instruction-read
  table evaluation onto the same prefix/suffix address-phase path used by core,
  adding full-field small-scalar accumulation for suffix buckets, flattening
  RAF Q bucket materialization, hoisting RAF shift scaling, and avoiding Rayon
  overhead on late address binds and intermediate read-table message
  allocation, the latest permissive-timeout three-iteration run in
  `perf/bolt-stage5-last.json` records core at `100.891ms`, Bolt at `98.196ms`,
  and `ratio_vs_core ~= 0.97`.
- **Stage 6**: protocol/lowering IR rails have started for bytecode read RAF,
  booleanity, hamming booleanity, RAM RA virtualization, instruction RA
  virtualization, and increment claim reduction. `commitment_ir` covers the
  Stage 6 protocol schema, concrete transcript threading, party projection,
  compute lowering, prover-only kernel resolution, CPU lowering, and generated
  Stage 6 prover/verifier Rust extraction plus standalone source compilation.
  The checked-in generated `jolt-prover`/`jolt-verifier` crates now include the
  Stage 6 modules, generated stage registries, and top-level proof/API wiring.
  The generated verifier has output-claim arithmetic for booleanity, hamming
  booleanity, RAM RA virtualization, instruction RA virtualization, and
  increment claim reduction, and bytecode read RAF when `Stage6VerifierData`
  supplies the preprocessing-derived bytecode table and entry bytecode index.
  The verifier evaluates the bytecode Val polynomials at the revealed sumcheck
  address point. The `muldiv` equivalence gate now derives and passes that
  verifier data, produces a Bolt Stage 6 proof from real trace witnesses,
  compares its proof artifacts and transcript state against `jolt-core`,
  verifies it with the standalone generated verifier, verifies it through the
  monolithic generated verifier after the generated Stage 5 prefix, replays the
  `jolt-core` proof through the generated Stage 6 prover CPU plan's
  proof-carrying kernel bridge, runs the top-level generated prover
  `prove_jolt_with_programs` with real commitment and Stage 1-6 CPU plans, and
  verifies that monolithic prover proof with the monolithic generated verifier.
  It also rejects representative standalone and monolithic Stage 6 sumcheck
  tampering. The kernel crate now has real Stage 6
  prover executor slices for bytecode read RAF, booleanity, hamming booleanity,
  RAM RA virtualization, instruction RA virtualization, and increment claim
  reduction, with final-claim tamper coverage for those single-relation batches.
  The bytecode read RAF prover uses a two-phase address/cycle path rather than
  expanding the full `K*T` domain, and booleanity uses a specialized sparse
  cubic evaluator. The `bolt-stage` selector supports Stage 6 as a
  correctness-gated timing run. A release `muldiv` smoke run with `--log-t 10
  --iters 1 --warmup 0` passes the 1.2x gate (`ratio_vs_core ~= 1.09`), and the
  documented `sha2-chain` three-iteration release gate is green:
  `perf/bolt-stage6-last.json` records core at `2011.115ms`, Bolt at
  `1266.910ms`, and `ratio_vs_core ~= 0.63`.
- **Stage 7**: generated prefix rails, standalone verifier coverage, and
  monolithic generated prover/verifier coverage are green on real `muldiv`
  traces. The equivalence gate compares Bolt-produced Stage 7 artifacts and
  transcript state against `jolt-core`, verifies the spliced core proof prefix,
  replays the core proof through the proof-carrying Stage 7 kernel bridge, and
  rejects representative Stage 7 tampering. The generated crates and monolithic
  APIs use the full-field transcript path (`Transcript<Challenge = Fr>`), which
  matches the patched `jolt-core` full-field `JoltField::Challenge` aliases for
  `ark_bn254::Fr` and `TrackedFr`. The `bolt-stage` selector supports Stage 7,
  and the `sha2-chain` smoke perf gate is green:
  `perf/bolt-stage7-smoke.json` records core at `2.551ms`, Bolt at `2.701ms`,
  and `ratio_vs_core ~= 1.06`.
- **Stage 8 / evaluation proof**: the generated monolithic prover now emits the
  Dory joint opening proof when Stage 7 opening inputs are supplied, and the
  generated monolithic verifier checks it when `evaluation_setup` is present.
  The monolithic `JoltProverPrograms`/`JoltVerifierPrograms` include the Stage
  8 evaluation plan, which keeps the Dory RLC claim count and order aligned
  with real compiler-owned trace plans rather than the default fixture plan.
  The `muldiv` equivalence gate asserts generated Stage 8 acceptance, core
  acceptance after replacing core's `joint_opening_proof` with Bolt's proof,
  full Stage 8 transcript parity, missing-proof/setup rejection, and unrelated
  Dory proof rejection. The `bolt-stage` selector has a correctness-gated
  `stage8` path. Stage 8 is correctness-green and perf-green after skipping
  zero one-hot cells during joint polynomial materialization and moving modular
  Dory hint/opening routines onto the same GLV vector operations used by core.
  `perf/bolt-stage8-smoke.json` records the current `muldiv` `log_t = 10`
  smoke with core at `152.451ms`, Bolt at `162.147ms`, and
  `ratio_vs_core ~= 1.06`. The documented `sha2-chain` three-iteration release
  gate is also green: `perf/bolt-stage8-last.json` records core at
  `796.183ms`, Bolt at `922.799ms`, and `ratio_vs_core ~= 1.16`.
