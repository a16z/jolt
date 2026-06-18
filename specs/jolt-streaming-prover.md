# Spec: Streaming Prover

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @sashafrolov,Claude            |
| Created     | 2026-06-17                     |
| Status      | proposed                       |
| PR          | #1629                        |

## Summary

The Jolt prover currently materializes the full execution trace (`Vec<Cycle>`)
and trace-derived witness data in memory, so peak prover RAM grows linearly with
the cycle count `T` — under 2 GB per million cycles, but unboundedly as programs
grow. This restricts proving to high-memory servers and caps the size of
programs a given machine can prove. This spec finishes the "streaming" Jolt
prover described in [Proving CPU Executions in Small
Space](https://eprint.iacr.org/2025/611): switch every access to the trace
vector to a lazily generated, streaming view so peak prover RAM scales with
`√T` instead of `T`, bounding memory to a few GB for arbitrarily long
executions and making Jolt usable on laptops and other memory-constrained
machines. The feature is gated behind a new compile-time `streaming` Cargo
feature (modeled on the existing `zk` flag) and is a backend-only change: the
proof it produces is byte-identical to the non-streaming proof, so the verifier,
proof format, and all downstream consumers are untouched. Per the paper's
estimates the streaming prover runs at most ~2× slower than the linear-space
prover. This spec targets the modular `jolt-prover` stack (`jolt-prover` plus
`jolt-witness` and `jolt-backends`, with sumcheck/PCS compute in
`jolt-backends::cpu`, `jolt-openings`, and `jolt-dory`), which is slated to
become the production prover; the legacy `jolt-core` prover is unaffected.
Partial scaffolding already exists across these crates (streaming witness
providers, streaming Dory commitment, CPU streaming sumcheck kernels, lazy trace
iteration) but is not yet wired through the whole prover.

## Intent

### Goal

Route every prover access to the execution trace through a lazily generated,
parallel-consumable streaming view — gated behind a compile-time `streaming`
Cargo feature — so that peak prover RAM scales with `√T` rather than `T` while
the prover emits byte-identical proofs.

Key abstractions and boundaries:

- A **streaming trace view** that lazily regenerates trace chunks on demand
  (recomputation, never disk), built on the existing `LazyTraceIterator`
  ([tracer/src/lib.rs](tracer/src/lib.rs)). Chunk access must be 
  parallelizable.
- Two streaming-aware core algorithms underpin the feature: (1) a **streaming
  Dory commitment** that computes the commitment's vector-matrix products over
  trace chunks without materializing the polynomial. The streaming
  commitment surface exists today
  ([crates/jolt-dory/src/streaming.rs](crates/jolt-dory/src/streaming.rs)), but
  the opening path is **not yet fully streaming** — part of it still
  materializes a witness that should be consumed chunk-wise, which this spec
  must eliminate;
  and (2) a **streaming/windowed sumcheck** that proves the early sumcheck rounds
  in bounded memory with a few passes over the recomputed trace before switching
  to the linear-space algorithm — **implemented in the legacy `jolt-core`**
  ([subprotocols/streaming_sumcheck.rs](jolt-core/src/subprotocols/streaming_sumcheck.rs))
  but **not yet ported** to the modular stack (`jolt-backends::cpu`).
- A `streaming` Cargo feature on `jolt-prover` that forwards to the backend
  capability crates (`jolt-backends`, `jolt-witness`, `jolt-dory`), selecting
  the streaming code path at compile time via `#[cfg(feature = "streaming")]`
  exactly as `zk`/`field-inline` forward today
  ([jolt-prover/Cargo.toml](crates/jolt-prover/Cargo.toml)). With the feature
  off, the prover's computation is unchanged — it produces the same proofs —
  though internal code paths may be refactored.
- Per-stage wiring so each of the 9 prover stages drives the algorithms above
  from the streaming trace view instead of a fully materialized polynomial. The
  crate-level placement of this work is detailed under Design → Architecture.
- A hard boundary: streaming changes are confined to the trace
  access/generation and polynomial commitment/sumcheck layers. They must not
  alter claim formulas, transcript behavior, stage order, or any
  verifier-visible logic.

### Invariants

This feature is a backend change whose defining correctness property is
**proof byte-identity** between the modular streaming prover and the `jolt-core`
reference prover. Using the [`jolt-eval`](../jolt-eval/README.md) framework:

Existing invariants — all must continue to hold **unchanged**; none need
modification:

- `soundness` — only one `(output, panic)` pair is accepted by the verifier for
  a deterministic guest+input. Streaming changes access patterns, not the
  verifier or the proof, so this is preserved (and is implied by byte-identity).
- `split_eq_bind_low_high` / `split_eq_bind_high_low` — `GruenSplitEqPolynomial::bind`
  matches `DensePolynomial::bound_poly_var_*`. Streaming must not change binding
  results.
- `field_mul_scalar` — unaffected.

New invariant to add during implementation — via the `/new-invariant` Claude
Code skill ([.claude/skills/new-invariant](.claude/skills/new-invariant)) 
following the [`jolt-eval`](../jolt-eval/README.md) framework:

- **`streaming_proof_byte_identical`**: for a fixed guest program, input, and
  RNG seed, the proof produced by the modular prover stack with `streaming`
  enabled is byte-identical to the proof `jolt-core` produces for the same
  inputs (same Fiat-Shamir state stream, same proof bytes, same verifier
  outcome). Both proofs are generated on the fly within the test — `jolt-core`
  and the modular stack are separate crates linkable in one binary, so no
  committed baseline blob is needed. **Byte-identity is required only in non-ZK
  mode.** Under the `zk` feature the proof embeds Pedersen-committed round
  polynomials and BlindFold blinding whose RNG draw order can differ between the
  two independent implementations, so byte equality may be impossible; the ZK
  combination is therefore checked **shape-only** — identical proof layout and
  field/segment sizes, the same Fiat-Shamir-derived public values, and an
  accepting verifier outcome — rather than byte-for-byte.

Prover/verifier consistency: the verifier is not modified. A proof produced
with `streaming` enabled must verify under the unmodified verifier exactly as
its non-streaming counterpart does.

### Non-Goals

- **Disk-backed streaming** (as in [Scribe](https://eprint.iacr.org/2024/1970))
  is out of scope: the witness is recomputed on demand rather than spilled to
  storage. Initial calculations show a disk-backed approach would rapidly wear
  out SSDs for any reasonable frequency of proof generation.
- **Reducing prover time is not a goal.** Streaming trades time for memory and
  may run up to ~2× slower than the non-streaming path; this is a memory
  project, not a speed optimization.
- **Sub-`√T` proving** (e.g. polylog-space) is out of scope; `√T` is the target
  memory regime.
- **Reducing the prover-memory term that is linear in the zkVM's addressable
  RAM or I/O** is out of scope; streaming addresses only the trace-derived term.
  Total prover RAM is therefore not strictly `√T` — the `√T` target describes
  the dominant trace-derived portion.
- **No change to the third-party/public API** beyond the `streaming` build flag;
  internal tuning knobs (chunk size, pass count) are not user-facing.
- **Deprecating or removing the non-streaming path is not a goal.** Like `zk`,
  the `streaming` flag is permanent and the non-streaming path stays as the
  default; the parallel `jolt-core` prover is the byte-identity oracle.
- **Streaming non-CPU backends is out of scope**: this targets the CPU prover
  only. A future GPU backend will handle streaming separately.

## Evaluation

### Acceptance Criteria

- [ ] A `streaming` Cargo feature on `jolt-prover` (forwarding to
  `jolt-backends`/`jolt-witness`/`jolt-dory`) exists; with it disabled the
  modular prover's computation is unchanged — it produces the same proofs —
  though internal code paths may be refactored.
- [ ] Each of the 9 prover stages (`jolt-prover` `stages/stage0`–`stage8`) reads
  the trace through the streaming view when `streaming` is enabled; no stage
  retains a full trace, full one-hot/RA, or full dense-witness materialization
  on the streaming path.
- [ ] Every application of the sumcheck protocol over polynomials whose size is
  proportional to `T` runs on the streaming/windowed sumcheck
  (`HalfSplitSchedule`) when `streaming` is enabled; none falls back to the
  linear (fully materialized) sumcheck path.
- [ ] The `streaming_proof_byte_identical` invariant passes in non-ZK mode
  (`--features streaming`): the modular prover stack's proof bytes match a
  `jolt-core` proof generated on the fly for the same inputs, for the canonical
  guests (`muldiv`, `fibonacci`, `sha2-chain`). In ZK mode
  (`--features zk,streaming`) the same guests pass a **shape-only** check
  (identical proof layout/segment sizes, matching Fiat-Shamir public values, and
  an accepting verifier outcome), since byte equality across the two
  implementations is not guaranteed under ZK randomness.
- [ ] For each stage, the within-stage peak RAM usage strictly decreases versus
  the non-streaming baseline for each converted stage.
- [ ] Total peak prover RAM grows sub-linearly (≈`√T`) across at least two trace
  sizes (`sha2-chain-2^16` and `sha2-chain-2^20`).
- [ ] End-to-end prover time with `streaming` enabled is within ~2× of the
  non-streaming prover on `sha2-chain-2^16`, after the whole-prover
  optimization pass.
- [ ] Streaming works in both non-ZK and ZK modes — the `streaming` and
  `streaming,zk` feature combinations both pass (the `streaming`-off rows are
  covered by the computation-unchanged criterion above).

### Testing Strategy

Existing tests that must keep passing:

- The modular prover's `muldiv` e2e correctness check, accepted by
  `jolt-verifier` (e.g. via `cargo nextest run -p jolt-prover
  -p jolt-prover-harness`), with `streaming` both off and on, in transparent
  and ZK (`zk`) modes.
- The full `cargo nextest run` suite passes with `streaming` enabled.

New tests:

- A byte-identity test backing the `streaming_proof_byte_identical` invariant:
  for a fixed guest+input+seed, generate a proof from both `jolt-core` and the
  modular prover stack (with `streaming` enabled) on the fly and assert the
  bytes are equal in non-ZK mode, for `muldiv`, `fibonacci`, and `sha2-chain`.
  In ZK mode the same test instead asserts shape-only equality (proof
  layout/segment sizes, Fiat-Shamir public values, accepting verifier outcome).
- A peak-RAM check confirming each stage's within-stage peak and the total
  `√T` scaling (see Performance).
- Feature-matrix compilation/clippy + `muldiv` e2e with `streaming` enabled in
  both transparent (`--features streaming`) and ZK (`--features zk,streaming`)
  modes.

### Performance

Tracked **bench-only**; no new `jolt-eval` objective is registered (per design
decision). Using the [`jolt-eval`](../jolt-eval/README.md) framework for
context:

- The existing `jolt-eval` performance objectives
  (`prover_time_sha2_chain_100`, `prover_time_fibonacci_100`) benchmark the
  `jolt-core` prover, which this spec does not touch, so they are **not expected
  to move**. The modular streaming prover's time (which may rise up to ~2×) is
  tracked separately in the `jolt-prover-harness` perf benches.
- Streaming's success metric — **peak prover RSS** — is not currently a
  `jolt-eval` objective and is **not** being added as one. It is measured in
  this spec's benchmark harness instead.

Benchmark policy:

- Workload `sha2-chain`, sizes `2^16` (primary) and `2^20` (confirmation); add
  `2^24` only if the `√T` scaling is not yet convincing at the smaller sizes,
  since it takes a while to run.
- Measure peak prover RSS via the `jolt-profiling` crate and the
  `jolt-prover-harness` perf benches (plus the `monitor` memory instrumentation)
  and wall-clock time, for streaming-on vs streaming-off.
- Gates: total peak RSS grows ≈`√T` across the two sizes; end-to-end time ≤ ~2×
  the non-streaming baseline. A missing peak-RAM measurement is a failure, not
  a pass.

## Design

### Architecture

The work targets the modular `jolt-prover` stack and spans several crates;
`jolt-core` is untouched (it remains the parity oracle). Affected crates:

- **`tracer`** — owns the lazy-trace mechanism (`LazyTraceIterator`,
  [tracer/src/lib.rs](tracer/src/lib.rs)). The streaming trace view is built
  here; trace ownership stays in `tracer`. The first CPU run is serial, but
  subsequent recompute passes run in parallel (rayon), so chunk access must be
  parallelizable and chunked streaming must not serialize the prover's hot
  loops.
- **`jolt-witness`** — exposes the streaming witness view the prover reads
  instead of a materialized polynomial; builds on the existing
  [crates/jolt-witness/src/streaming.rs](crates/jolt-witness/src/streaming.rs)
  and witness providers. The prover↔witness interface must offer a streaming
  oracle, not a fully materialized one, and must allow parallel access
  to its witness chunks.
- **`jolt-backends` (`cpu`)** — the CPU streaming compute and the bulk of the
  new work. `cpu/schedule/` defines the `StreamingSchedule` trait
  (`HalfSplitSchedule`/`LinearOnlySchedule`) — the streaming-vs-linear and
  window-size policy. `cpu/sumcheck/` holds the per-relation kernels but **no
  schedule consumer yet**: the streaming-sumcheck driver, ported from
  jolt-core's `subprotocols/streaming_sumcheck.rs` (the
  `StreamingSumcheckWindow` / `LinearSumcheckStage` window state machine and the
  `StreamingSchedule`-driven per-round loop), is added here and adapted to the
  modular kernel structure — which has no `SumcheckInstanceProver` /
  `SumcheckInstanceParams` trait layer. `cpu/commitments/stream.rs` does the
  streaming commitment, and the backend request/result contracts
  (`sumcheck/request.rs`, `commitments/result.rs`) gain streaming-aware inputs.
  `streaming` is a `jolt-backends` capability feature.
- **`jolt-dory` / `jolt-openings` / `jolt-poly`** — the polynomial commitment
  scheme: streaming Dory commitment
  ([crates/jolt-dory/src/streaming.rs](crates/jolt-dory/src/streaming.rs))
  behind the `jolt-openings` commitment-scheme contracts, over `jolt-poly`
  polynomial types. The commitment surface is in place; the remaining work is
  the opening path, which still materializes a witness that should instead be
  consumed chunk-wise.
- **`jolt-sumcheck`** — the shared sumcheck protocol/verifier-contract crate
  (proof, round-proof, claim, and domain types + verifier). Its proof and
  round-proof types must stay byte-identical, so it is not expected to change.
- **`jolt-prover`** — orchestrates the 9 stages
  ([crates/jolt-prover/src/stages/stage0](crates/jolt-prover/src/stages/)`–stage8`),
  drives Fiat-Shamir, and dispatches backend requests. It owns the `streaming`
  Cargo feature, which forwards to `jolt-backends`/`jolt-witness`/`jolt-dory`
  exactly as `zk`/`field-inline` do today. Each stage is converted
  independently.

Placement rules:

- Streaming changes stay localized to the trace access/generation and
  polynomial commitment/sumcheck layers; they must not leak into unrelated
  prover code or any verifier-visible path.
- Streaming algorithms expose internal tuning knobs (chunk size, pass count)
  for trading memory against time; these are not user-facing configuration.

### Alternatives Considered

- **Continuations / SNARK recursion sharding** (the approach other zkVMs take)
  breaks execution into ~1M-cycle shards, proves each shard independently, and
  recursively aggregates the shard proofs. It was rejected because it adds
  significant complexity and performance overhead that Jolt's streaming
  sum-check approach avoids entirely (see
  [book/src/roadmap/streaming.md](book/src/roadmap/streaming.md)).
- **Disk-backed streaming** (Scribe) spills the witness to storage instead of
  recomputing it. It was rejected because the repeated writes would rapidly wear
  out SSDs for any frequently running prover (see Non-Goals).
- **The status quo (linear-space prover)** leaves the current design in place.
  It was rejected because prover memory grows linearly with `T`, which caps
  program size and excludes low-memory machines.
- **A full rewrite** of the prover into a streaming form was rejected in favor
  of an incremental, stage-by-stage migration gated by per-stage byte-identity
  and memory checks, so that regressions are caught one stage at a time. The
  non-streaming prover is also retained, since there are still use cases where
  it is preferable.

## Documentation

Update [book/src/roadmap/streaming.md](book/src/roadmap/streaming.md): it
currently states "the Jolt prover is not streaming … work to stream-ify the
prover is underway." Rewrite it to describe streaming as an implemented feature
— how to enable the `streaming` flag, the `√T`-memory / ≤2×-time tradeoff, and
when to use it (e.g. laptops / low-RAM machines) — and move it out of
the `roadmap/` section. Expand it if implementation surfaces interesting
optimizations or issues worth documenting.

## Execution

Implement these steps in an incremental step-by-step migration. Use these steps
in goal-mode:

1. **Set up shared streaming infrastructure once, before the loop.** Take stock
   of the existing streaming/lazy-trace scaffolding and add only the remaining
   helper functions the stages need to read it — chunked, replayable forward
   access whose chunks can be consumed in parallel — gated behind `streaming`.
   Stand up the per-stage gate harness (the `jolt-prover-harness`): a
   byte-identical proof comparison against a `jolt-core` proof generated on the
   fly for the same input, and per-stage peak-RAM logging.
2. **Convert one stage at a time, stages 0 → 8 in order.** Do not start a stage
   until the previous one is accepted. For each stage `S`: replace its
   materialized-trace accesses with the streaming view (no proof-visible logic
   changes); assert byte-identity to the `jolt-core` proof; confirm within-stage
   peak RAM
   strictly decreases; confirm per-stage time regression stays under 2× (an
   early guard, not the final bar).
3. **Run a whole-prover optimization pass after all 9 stages are converted**,
   since per-stage 2× guards can compound — optimize until end-to-end prover
   time is within ~2× of the non-streaming baseline.

Promotion rule: a stage is "accepted" only when its trace access is fully
streamed, its proof is byte-identical, its within-stage peak RAM strictly
decreases, and its per-stage time regression is under 2×. The migration is done
when all 9 stages are accepted, total peak RAM scales with `√T`, and end-to-end
prover time is within ~2× of the non-streaming baseline.

## References

- [Proving CPU Executions in Small Space (NTZ25)](https://eprint.iacr.org/2025/611)
  — Source for algorithms for the streaming Jolt prover.
- [Scribe](https://eprint.iacr.org/2024/1970) — disk-backed
  approach considered and rejected.
- [Improved streaming sum-check (BCFFMMZ25)](https://eprint.iacr.org/2025/1473)
  — Source for the more efficient streaming "windowed" sumcheck algorithm.
- [CTY11](https://arxiv.org/abs/1109.6882),
  [Clover 2014](https://eprint.iacr.org/2014/846) — origins of streaming
  sum-check.
- [book/src/roadmap/streaming.md](book/src/roadmap/streaming.md) — current
  streaming roadmap page (to be updated).
- Key code references (modular stack):
  - [jolt-prover stages](crates/jolt-prover/src/stages/) — the 9 prover stages.
  - [jolt-backends cpu](crates/jolt-backends/src/cpu/) — has the
    `StreamingSchedule` (`cpu/schedule/`) and streaming commitment
    (`cpu/commitments/stream.rs`); the streaming sumcheck driver is still to be
    ported here (`cpu/sumcheck/`).
  - [jolt-witness streaming](crates/jolt-witness/src/streaming.rs) — streaming
    witness providers.
  - [jolt-dory streaming](crates/jolt-dory/src/streaming.rs) — streaming Dory
    commitment.
  - [tracer `LazyTraceIterator`](tracer/src/lib.rs) — lazy trace generation.
- Oracle/reference (the `jolt-core` prover, a read-only parity oracle not
  modified by this spec):
  - [jolt-core/src/zkvm/prover.rs](jolt-core/src/zkvm/prover.rs) — the
    linear-space prover and its 9 stages.
  - [jolt-core/src/subprotocols/streaming_sumcheck.rs](jolt-core/src/subprotocols/streaming_sumcheck.rs)
    — the streaming/windowed sumcheck driver to port.
  - [jolt-core/src/subprotocols/streaming_schedule.rs](jolt-core/src/subprotocols/streaming_schedule.rs)
    — the `StreamingSchedule` reference implementation.
  - [jolt-core/src/poly/rlc_polynomial.rs](jolt-core/src/poly/rlc_polynomial.rs)
    — the streaming RLC / Dory vector-matrix-product reference.
