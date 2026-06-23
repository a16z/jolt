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
machines. Concretely, "streaming" means that neither the trace itself nor any
data derived from it whose size is `O(T)` may be fully materialized in memory;
such data must instead be regenerated chunk-wise from the streaming trace view.
Only sub-`O(T)` data (e.g. address-, RAM/I/O-, or advice-layout-sized
structures) may remain materialized. This spec scopes a **first implementation** whose primary goal is
lowering peak prover RAM; squeezing the streaming prover to maximal performance
is deferred to follow-up work, but reasonable optimization effort is still
expected in this pass. In particular, the trace must be re-streamed only a
logarithmic number of times (or less) so the time regression stays bounded
rather than open-ended. We accept a larger time regression than the eventual
target and will tighten it through later optimization once the memory-reduction
path is correct and wired end to end. The feature is gated behind a new
compile-time `streaming` Cargo feature (modeled on the existing `zk` flag) and
is a backend-only change: the proof it produces is byte-identical to the
non-streaming proof, so the verifier, proof format, and all downstream consumers
are untouched. Per the paper's estimates a fully optimized streaming prover can
run at most ~2× slower than the linear-space prover; this initial implementation
does not yet target that bound. This spec targets the modular `jolt-prover`
stack (`jolt-prover` plus `jolt-witness` and `jolt-backends`, with sumcheck/PCS
compute in `jolt-backends::cpu`, `jolt-openings`, and `jolt-dory`), which is
slated to become the production prover; the legacy `jolt-core` prover is
unaffected. Partial scaffolding already exists across these crates (streaming
witness providers, streaming Dory commitment, CPU streaming sumcheck kernels,
lazy trace iteration) but is not yet wired through the whole prover.

## Intent

### Goal

Deliver a first streaming implementation that routes every prover access to the
execution trace through a lazily generated, parallel-consumable streaming view —
gated behind a compile-time `streaming` Cargo feature — so that peak prover RAM
scales with `√T` rather than `T` while the prover emits byte-identical proofs.
The defining property is that no `O(T)`-sized data is fully materialized: not
the trace, and not any trace-derived structure (witness polynomials, one-hot/RA
matrices, dense per-cycle vectors) whose length grows with `T`. Each such
structure must be consumed chunk-wise from the streaming view; only sub-`O(T)`
data may stay resident in memory. The primary goal of this pass is the memory
reduction and its correctness;
squeezing the time regression toward the paper's ~2× bound is left to later
optimization work. Reasonable optimization effort is still expected, though: the
trace should be re-streamed only a logarithmic number of times (or less) so
recompute cost stays logarithmically bounded.

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
  from the streaming trace view instead of a fully materialized polynomial. No
  stage may hold the trace or any `O(T)` trace-derived structure fully
  materialized on the streaming path. The crate-level placement of this work is
  detailed under Design → Architecture.
- A hard boundary: streaming changes are confined to the trace
  access/generation and polynomial commitment/sumcheck layers. They must not
  alter claim formulas, transcript behavior, stage order, or any
  verifier-visible logic.

### Invariants

This feature is a backend change whose defining correctness property is
**proof byte-identity** between the streaming and non-streaming builds. Using
the [`jolt-eval`](../jolt-eval/README.md) framework:

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
  RNG seed, the proof emitted by the `streaming` build is byte-identical to a
  committed baseline blob captured from the non-streaming build (same
  Fiat-Shamir state stream, same proof bytes, same verifier outcome). Because
  `streaming` is a compile-time feature, this is a golden-baseline comparison
  (similar to the wire-format test in
  [unify-field-hierarchy.md](./unify-field-hierarchy.md)), not a two-build
  runtime check. **Byte-identity is required in both non-ZK and ZK modes.**
  Under the `zk` feature the proof embeds Pedersen-committed round polynomials
  and BlindFold blinding, but streaming changes only how the trace is accessed —
  not the RNG draw order, blinding values, or any proof-visible logic — so the
  streaming `zk` build must produce byte-identical proofs to the non-streaming
  `zk` build against its own committed baseline. There are therefore two
  separate golden blobs per guest — one captured from the non-streaming non-ZK
  build and one from the non-streaming `zk` build — and each mode is compared
  only against the baseline captured in the matching mode. The streaming code path must
  preserve the exact Fiat-Shamir and randomness draw order so this holds.

Prover/verifier consistency: the verifier is not modified. A proof produced
with `streaming` enabled must verify under the unmodified verifier exactly as
its non-streaming counterpart does.

### Non-Goals

- **Disk-backed streaming** (as in [Scribe](https://eprint.iacr.org/2024/1970))
  is out of scope: the witness is recomputed on demand rather than spilled to
  storage. Initial calculations show a disk-backed approach would rapidly wear
  out SSDs for any reasonable frequency of proof generation.
- **Minimizing prover time is not the primary goal of this pass.**
  Streaming trades time for memory, and this first pass may run
  noticeably slower than the eventual ~2× bound the paper targets; squeezing the
  time regression to that bound is deferred follow-up work. Reasonable
  optimization effort is still expected — in particular the trace must be
  re-streamed only logarithmically many times, keeping recompute cost logarithmically
  bounded rather than open-ended. This is a memory project first, but not a
  license to ignore time.
- **Sub-`√T` proving** (e.g. polylog-space) is out of scope; `√T` is the target
  memory regime.
- **Reducing the prover-memory term that is linear in the zkVM's addressable
  RAM or I/O** is out of scope; streaming addresses only the trace-derived term.
  Total prover RAM is therefore not strictly `√T` — the `√T` target describes
  the dominant trace-derived portion. We assume the non-trace term does not
  dominate the trace-derived term — a reasonable assumption for the long
  executions that streaming targets — so streaming the trace is expected to reduce
  each materializing stage's within-stage peak.
- **No change to the third-party/public API** beyond the `streaming` build flag;
  internal tuning knobs (chunk size, pass count) are not user-facing.
- **Deprecating or removing the non-streaming path is not a goal.** Like `zk`,
  the `streaming` flag is permanent and the non-streaming path stays as the
  default and the byte-identity parity oracle.
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
  retains a full trace, full one-hot/RA, or full trace-derived dense-witness
  materialization on the streaming path. Address-sized, RAM/I/O-sized, or
  advice-layout-sized materializations that already exist outside the
  trace-derived term are allowed when they are not the source of linear-in-`T`
  memory growth.
- [ ] Interface-level trace-materialization ban: for every one of the 9
  stages, the materialized trace must not appear in the stage's public interface:
  no stage `prove`/`prove_committed_boundary` entry point may take a fully
  materialized trace (e.g. `Vec<Cycle>`, `&[Cycle]`) or a trace-sized dense
  witness vector as an argument. Stages receive trace data only through the
  streaming witness view — the `WitnessProvider` / streaming-input type
  parameter (e.g. `Stage1ProverInput<'_, W>` with `W: WitnessProvider<..>`) —
  never the concrete trace. This is a structural, type-level requirement
  verifiable by inspecting the stage signatures in
  `crates/jolt-prover/src/stages/stage0`–`stage8`, independent of runtime
  behavior; it makes "stages do not handle materialized trace vectors" a hard
  interface guarantee rather than only an implementation property.
- [ ] Every application of the sumcheck protocol over polynomials whose size is
  proportional to `T` either runs on the streaming/windowed sumcheck
  (`HalfSplitSchedule`) or preserves an existing specialized representation
  that avoids materializing the logical trace-sized vector while consuming trace
  chunks from the streaming view. None may fall back to a generic linear
  fully-materialized trace-vector path. Conversely, sumcheck applications whose
  vectors are **not** proportional to `T` stay on the linear sumcheck and do not
  have to be migrated to a streaming instance.
- [ ] The `streaming_proof_byte_identical` invariant passes in both non-ZK mode
  (`--features streaming`) and ZK mode (`--features zk,streaming`): the
  `streaming` build's proof bytes match the committed non-streaming baseline for
  the canonical guests (`muldiv`, `fibonacci`, `sha2-chain`) under each feature
  combination.
- [ ] Total peak RSS with `streaming` enabled does not increase by more than
  50% versus the non-streaming baseline (and is generally expected to decrease,
  or increase by much less). This is the automated gate. Confirming that each
  converted stage's within-stage peak RAM actually decreases is done manually
  for now — there is no good way to automate the per-stage check yet. A stage
  that never materializes the trace has nothing to stream and is exempt from the
  manual per-stage check.
- [ ] Total peak prover RAM grows sub-linearly (≈`√T`) across at least two trace
  sizes (`sha2-chain-2^16` and `sha2-chain-2^20`).
- [ ] End-to-end prover time with `streaming` enabled is measured and reported
  against the non-streaming prover on `sha2-chain-2^16`. The ~2× bound is a
  target for later optimization work, not an acceptance gate for this initial
  implementation; the time figure is recorded as a baseline for that follow-up.
- [ ] Streaming works in both non-ZK and ZK modes — the `streaming` and
  `streaming,zk` feature combinations both pass (the `streaming`-off rows are
  covered by the computation-unchanged criterion above).
- [ ] The existing static rail checks still pass on the streaming changes: the
  `jolt-prover-harness` Rust rail tests and the Semgrep rails
  (`scripts/semgrep-rails.sh`) both succeed. In particular the
  `jolt-backends-no-transcript` rail must pass, confirming the ported
  streaming-sumcheck driver respects the transcript split described in Design.
- [ ] Gate — crate-boundary check: the change-set boundary check
  (`scripts/check-streaming-crate-boundary.sh <base>`) must pass: the diff vs
  the streaming feature's base ref touches only the affected crates listed in
  Design → Architecture and does not modify `jolt-verifier`, `jolt-core`, or
  other verifier-visible/protocol crates. This check must pass. If a change
  outside these crates is genuinely needed, it should be justified in the commit
  message(s).

### Testing Strategy

Existing tests that must keep passing:

- The modular prover's `muldiv` e2e correctness check, accepted by
  `jolt-verifier` (e.g. via `cargo nextest run -p jolt-prover
  -p jolt-prover-harness`), with `streaming` both off and on, in transparent
  and ZK (`zk`) modes.
- The full `cargo nextest run` suite passes with `streaming` enabled.
- The static rail checks pass on the streaming changes — the
  `jolt-prover-harness` rail tests (`cargo nextest run -p jolt-prover-harness`)
  and the Semgrep rails (`scripts/semgrep-rails.sh`), which cover the
  `jolt-prover`/`jolt-backends`/`jolt-witness` boundaries the port touches.
- The change-set boundary check (`scripts/check-streaming-crate-boundary.sh`)
  passes: a git-diff guard asserting the modified files stay within the affected
  crates and never touch `jolt-verifier` or `jolt-core`. Pass the streaming
  feature's base ref as the first argument (defaults to the prior commit,
  `HEAD~1`).

New tests:

- A golden byte-identity test backing the `streaming_proof_byte_identical`
  invariant: for a fixed guest+input+seed, serialize the `streaming` build's
  proof and assert it equals the committed non-streaming baseline blob captured
  in the matching mode, for `muldiv`, `fibonacci`, and `sha2-chain`, in both
  non-ZK and ZK (`--features zk,streaming`) modes — i.e. one golden blob per
  guest per mode (six blobs total).
- A peak-RAM check confirming total peak RSS stays within +50% of the
  non-streaming baseline and the total `√T` scaling (see Performance). The
  per-stage within-stage peak is confirmed manually, not by an automated check.
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
  to move**. The modular streaming prover's time (which may rise substantially
  in this initial implementation) is tracked separately in the
  `jolt-prover-harness` perf benches.
- Streaming's success metric — **peak prover RSS** — is not currently a
  `jolt-eval` objective and is **not** being added as one. It is measured in
  this spec's benchmark harness instead.

Benchmark policy:

- Workload `sha2-chain`, sizes `2^16` (primary) and `2^20` (confirmation); add
  `2^24` only if the `√T` scaling is not yet convincing at the smaller sizes,
  since it takes a while to run.
- Measure peak prover RSS with a standard external command — e.g.
  `/usr/bin/time -v` (the "Maximum resident set size" line) or `/usr/bin/time -l`
  on macOS — running the prover binary for streaming-on vs streaming-off. The
  `jolt-profiling` crate, `jolt-prover-harness` perf benches, and `monitor`
  memory instrumentation are not fully implemented yet, so they are not relied on
  for this gate. Also record wall-clock time for both configurations.
- Gates: total peak RSS does not increase by more than 50% versus the
  non-streaming baseline (see Acceptance Criteria). The ≈`√T` scaling is a gate
  for the full initial pass — once all 9 stages are converted, total peak RSS
  must grow ≈`√T` across the two sizes. It is not a gate for individual stage
  conversions along the way: those are tracked via the +50% peak-RSS bound and
  the manual per-stage check, with `√T` measured and reported to show progress
  toward the end-state gate. End-to-end time is measured and reported against the
  non-streaming baseline but is not a gate for this initial implementation — the
  ~2× bound is a target for later optimization work. A missing peak-RAM
  measurement is a failure, not a pass.

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
  to its witness chunks. To maintain separation of concerns, the `jolt-witness`
  public interface must be independent of the `streaming` Cargo feature: the
  same API is exposed regardless of whether `streaming` is enabled, so callers
  do not branch on the flag. The only permitted feature-dependent surface is an
  optional parallel/chunked access interface offered alongside the
  flag-independent one; `streaming` must not gate, rename, or change the
  signatures of the core witness API.
- **`jolt-backends` (`cpu`)** — the CPU streaming compute and the bulk of the
  new work. `cpu/schedule/` defines the `StreamingSchedule` trait
  (`HalfSplitSchedule`/`LinearOnlySchedule`) — the streaming-vs-linear and
  window-size policy. `cpu/sumcheck/` holds the per-relation kernels but **no
  schedule consumer yet**: the streaming-sumcheck driver, ported from
  jolt-core's `subprotocols/streaming_sumcheck.rs` (the
  `StreamingSumcheckWindow` / `LinearSumcheckStage` window state machine and the
  `StreamingSchedule`-driven per-round loop), is added here and adapted to the
  modular kernel structure — which has no `SumcheckInstanceProver` /
  `SumcheckInstanceParams` trait layer. Note: the jolt-core source driver
  references `Transcript`, but `jolt-backends` must not own transcript
  operations (the `jolt-backends-no-transcript` rail), and jolt-prover drives
  Fiat-Shamir. The port must therefore be compatible with that transcript
  split: strip transcript ownership out of the driver rather than copying it
  verbatim. `cpu/commitments/stream.rs` does the
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
- Streaming algorithms should expose internal tuning knobs (chunk size, pass count)
  for trading memory against time; though these are not user-facing configuration.
- The streaming migration must preserve the existing memory-saving
  representations used across the prover's sumcheck stages.
  These are part of the prover's performance contract, not incidental
  implementation details. Examples include:
  - Stage 2 RAM read-write checking, which uses sparse cycle-major/address-major
    read-write matrices and materializes dense `ra`/`val` vectors only after the
    configured phase split has reduced the live `K*T` domain.
  - Stage 4 RAM value checking, which fixes the address side at `r_address` and
    stores per-cycle `wa = eq(r_address, row_address)`, `inc`, and `Lt(r_cycle)`
    tables instead of a dense RAM address-by-cycle matrix.
  - Stage 6 RA booleanity and RA virtualization, which use compact per-cycle
    RA chunk indices, `Kc` equality/pushforward tables, `RaPolynomial`, and
    `SharedRaPolynomials` rather than full `Kc*T` committed-RA matrices.
  - Stage 7 hamming-weight RA-family claim reduction, which starts from
    pushed-forward `G_i(k) = sum_j eq(r_cycle, j) * ra_i(k, j)` tables and
    proves only over the `Kc` address-chunk domain.
  - Stage 1 Spartan outer remainder and Stage 3 Spartan shift, which use
    prefix/suffix equality or `eq+1` decompositions instead of keeping all
    helper tables in dense form.
  A streaming implementation may replace the data source for these
  representations with replayable trace chunks, but it must not replace them
  with generic dense materialization that increases peak memory.

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
prover is underway." Rewrite it to describe streaming as an initial implemented
feature — how to enable the `streaming` flag, the `√T`-memory tradeoff (noting
that the time regression is larger in this first implementation, with a ~2×
bound as an eventual optimization goal), and when to use it (e.g. laptops /
low-RAM machines) — and move it out of the `roadmap/` section. Expand it if implementation surfaces interesting
optimizations or issues worth documenting.

## Execution

Implement these steps in an incremental step-by-step migration. Use these steps
in goal-mode:

1. **Set up shared streaming infrastructure once, before the loop.** Take stock
   of the existing streaming/lazy-trace scaffolding and add only the remaining
   helper functions the stages need to read it — chunked, replayable forward
   access whose chunks can be consumed in parallel — gated behind `streaming`.
   Extend the existing per-stage gate harness (the `jolt-prover-harness`, which
   already provides `parity.rs`, `core_fixture.rs`, `frontier_stage0..8`, and
   `metrics.rs`) with the byte-identical proof comparison against the committed
   non-streaming baseline and per-stage peak-RAM logging.
2. **Convert one stage at a time, stages 0 → 8 in order.** Do not start a stage
   until the previous one is accepted. For each stage `S`: replace its
   materialized-trace accesses with the streaming view (no proof-visible logic
   changes); assert byte-identity to the non-streaming baseline; manually
   confirm within-stage peak RAM decreases (no automated check yet) while total
   peak RSS stays within +50% of baseline; and record the per-stage time
   regression as a baseline for later optimization work (not a gate at this
   stage).

Time optimization — tightening the end-to-end regression toward the paper's ~2×
bound — is intentionally out of scope for this initial implementation and is
tracked as follow-up work once all stages are streaming and correct.

Promotion rule: a stage is "accepted" only when its trace access is fully
streamed, its proof is byte-identical, total peak RSS stays within +50% of the
non-streaming baseline, and its within-stage peak RAM is manually confirmed to
decrease (if the stage materialized the trace in the first place). The
migration is done when all 9 stages are accepted and total peak RAM scales with
`√T`; the end-to-end time regression is measured and recorded but is not an
acceptance bar for this initial implementation.

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
