# Jolt Equivalence Goal

`jolt-equivalence` is the bootstrap oracle and gate suite for Jolt-on-Bolt.
It exists to prove that the Bolt-generated prover/verifier is equivalent to
`jolt-core` while Bolt is maturing. It must stay a thin logical comparison
crate: run both implementations, translate public artifacts into common shapes,
compare them, mutate proofs, and enforce perf gates.

It must not become a shadow implementation of Jolt.

## Objective

Shrink `jolt-equivalence` into a strict oracle/gating crate while moving all
Jolt semantic construction required for equivalence into Jolt-on-Bolt itself.

The target state is:

```text
core temporary oracle -> canonical public artifacts
Bolt generated prover -> canonical public artifacts
Bolt generated verifier -> accepts valid Bolt proof and rejects mutated proofs
jolt-equivalence -> compares artifacts, drives fuzz/tamper/perf gates
```

The harness should answer:

```text
Did Bolt produce the same public protocol artifacts as core?
Did Bolt verifier accept the honest Bolt proof?
Did Bolt verifier reject malformed or tampered proofs?
Did Bolt stay within configured perf bounds versus core?
```

The harness should not answer:

```text
How are Stage 6/7 opening points normalized?
How is hamming-weight claim reduction constructed?
How are witness polynomials evaluated?
How do private core verifier stages compute intermediate state?
```

Those are Bolt/codegen/kernel responsibilities.

## Non-Negotiables

- `jolt-equivalence` must not encode major Jolt stage semantics.
- New semantic mismatches must be fixed in Bolt IR/codegen/kernels/generated
  prover/verifier, not by adding reconstruction logic to the harness.
- Core access must use public APIs or explicit public oracle/export APIs added
  for equivalence. Do not depend on private verifier stage methods.
- Generated Bolt artifacts must expose enough canonical data for the harness to
  compare without rebuilding hidden state.
- Checked-in `jolt-prover` and `jolt-verifier` are generated artifacts, not
  hand-maintained source. Any behavioral change visible there must be made in
  Bolt IR/codegen/kernels or shared runtime/witness crates, then kept in sync
  through the generated-artifact rail.
- The full-field non-zk path is the active equivalence target:
  `Transcript<Challenge = Fr>` / `challenge-254-bit`.
- Later-stage prover-side witness data, claim inputs, and opening inputs must
  come from generated Bolt prover artifacts or shared `jolt-witness`
  primitives, not from harness reconstruction.
- The eventual replacement path is Bolt over core. Core is only the temporary
  oracle until Bolt is mature enough to deprecate it.

## Allowed Responsibilities

`jolt-equivalence` may contain:

- test fixtures that run core and Bolt on the same guest/input
- adapters from public core artifacts into canonical comparison structs
- adapters from generated Bolt artifacts into the same structs
- transcript checkpoint comparison
- commitment and opening-claim comparison
- proof acceptance checks
- proof mutation and fuzz/tamper helpers
- perf metric collection and threshold checks
- small, generic conversion helpers between core/Bolt field/proof types

Adapters may normalize representation, but only when the normalization is a
thin public-format translation. If the code needs to know a stage-specific
formula, construct a witness polynomial, rebuild a claim reduction, or infer
hidden verifier state, it belongs outside `jolt-equivalence`.

## Forbidden Responsibilities

Do not add or preserve harness code that:

- reconstructs Stage 6/7 opening points from sumcheck internals
- rebuilds hamming-weight claim reduction inputs
- evaluates committed witness polynomials to compensate for missing artifacts
- maintains a local value store mirroring generated verifier internals
- uses private `jolt-core` stage verifier methods as the main parity path
- hard-codes stage-specific point-order semantics in tests
- fixes equivalence failures by changing expected values only in the harness

Existing code in this category is migration debt and should shrink each slice.

## Current Baseline

The important current baseline is approximately:

```text
jolt-core full-field challenge patch: 4 files, 60 insertions, 19 deletions
jolt-equivalence Rust LOC:          6,064 total
tests/bolt_commitment.rs:           317 LOC
tests/bolt_perf.rs:                 20 LOC
tests/generated_role_crates.rs:     142 LOC
adapters.rs:                        62 LOC
artifacts.rs:                      144 LOC
bolt_programs.rs:                  170 LOC
checkpoint.rs:                     273 LOC
plan_adapters.rs:                   755 LOC
plan_adapters/generated_*.rs:       271 LOC
plan_adapters/stage{1..7}.rs:       187 LOC
bolt_oracle.rs:                     842 LOC
commitment_oracle.rs:               419 LOC
core_conversion.rs:                 374 LOC
tamper.rs:                          897 LOC
checks.rs:                          566 LOC
core_oracle.rs:                     520 LOC
perf.rs:                             77 LOC
jolt-kernels/src/trace.rs:          391 LOC
jolt-witness/src/lib.rs:            1,133 LOC
```

The core diff is acceptable in principle. The equivalence harness is too large
and currently contains semantic reconstruction that should move into Bolt.

Known active blocker:

```text
None for the focused muldiv equivalence oracle. Stage 6 Booleanity, Stage 7
hamming-weight claim reduction, generated verification, Dory opening proof
bytes, and core acceptance all pass the focused real-trace parity gate.
```

Current progress on this branch:

- Stage 6 execution artifacts now carry public opening-claim values, so Stage 7
  inputs can be adapted from Stage 6 artifacts instead of replaying a local
  Stage 6 verifier store inside `jolt-equivalence`.
- Stage 6 core acceptance now includes a strict generated-eval to
  `jolt_core::poly::opening_proof::OpeningId` comparison for public opening
  claims. The check maps generated eval names to core opening IDs and compares
  values only; it does not synthesize opening points or rebuild claims.
- Stage 1 kernel/generated plan adapters now use the shared adapter macro path
  instead of bespoke per-role copies, reducing duplicated adapter code while
  preserving the generated/kernel plan contract.
- Generated commitment prover/verifier plan adapters now share one local
  lowering macro, preserving the same generated plan contract while cutting the
  generated adapter subtree by another 60 LOC.
- Core/Bolt uni-skip proof conversion and Stage 1/2 uni-skip coefficient
  checks now share generic helpers; Stage 4/5/6/7 artifact match wrappers now
  share one macro. These are representation-only harness cleanups and do not
  alter generated prover/verifier semantics.
- Stage 6 Booleanity now has an indexed kernel path that ports core's sparse
  Booleanity semantics into `jolt-kernels`, with sparse RA index data supplied
  by the caller. The dense fallback remains internally verifier-consistent, but
  the core-equivalent path is sparse/indexed.
- Generated Stage 6 verifier output-claim reconstruction now uses core's
  Booleanity segment-wise point convention via the Bolt verifier-common
  template; `jolt-verifier` is regenerated from that source.
- Generic one-hot index/materialization helpers moved to `jolt-witness`:
  sparse chunk indices, address-major one-hot materialization, point evals from
  sparse indices, and MSB-first chunk/point splitting.
- Stage 6 witness materialization moved out of `jolt-equivalence` into
  `jolt-witness`: sparse RA indices, Booleanity oracles, read-RAF chunks,
  virtual RA evals, hamming-weight inputs, and `rd_inc`/`ram_inc` columns are
  now produced by shared `Stage6WitnessInputs`/`Stage6WitnessParams` helpers.
  The harness still invokes this shared primitive until generated prover
  artifacts expose the same data directly.
- Stage 6/7 ordered witness-slice views moved into `jolt-witness` via
  `Stage6WitnessSlices`; `bolt_oracle.rs` no longer assembles Booleanity,
  read-RAF, virtual RA, or hamming-weight index chunk ordering by hand.
- Stage 6 witness construction no longer materializes dense RA Booleanity
  one-hot polynomials for the normal sparse/indexed Booleanity path. The
  generated prover now supplies sparse index chunks through `jolt-witness`, and
  `jolt-kernels` derives the Booleanity domain from the sparse path when
  indices are present. On SHA2-chain `2^20`, this moved
  `bolt.prove.inputs.stage6_witness` from roughly `5.46s` to roughly `0.24s`
  and reduced Bolt prove time from roughly `28.9s` to `22.6-22.8s`.
- Stage 6 bytecode read-RAF now consumes the sparse bytecode RA index chunks
  already produced by `jolt-witness` when initializing its cycle phase, while
  retaining the dense one-hot fallback for synthetic callers. On SHA2-chain
  `2^20`, this reduced the bytecode read-RAF Stage 6 bucket from roughly
  `4.3s` to roughly `1.6s` per Stage 6 execution and moved Bolt prove time to
  roughly `19.4-19.9s` (`~1.97-2.00x` core). The SHA2-chain `2^16` prove ratio
  is now `1.174x`, inside the 20% target.
- Stage 6 bytecode read-RAF witness construction no longer materializes dense
  address-major bytecode RA chunks in the generated prover path. `jolt-witness`
  now carries bytecode RA chunk lengths alongside sparse bytecode RA indices,
  and `jolt-kernels` can run bytecode read-RAF from sparse indices plus chunk
  lengths with no dense chunk input. The dense chunk fallback remains covered
  by synthetic kernel tests.
- Stage 6/7 witness-slice-to-kernel input wiring moved onto `jolt-kernels`
  public builder methods: `Stage6ProverInputs::with_stage6_witness` and
  `Stage7ProverInputs::with_stage6_witness_indices`. The equivalence harness
  now invokes kernel-owned routing instead of spelling out each witness field.
- Stage 6 witness construction from kernel opening inputs moved onto
  `jolt-kernels::stage6_witness_from_opening_inputs`; `bolt_oracle.rs` no
  longer builds `Stage6OpeningInputRef` lists or `Stage6WitnessInputs` locally.
- Generated `jolt-prover` now exposes `stage6_witness_from_opening_inputs`
  from the Bolt artifact generator, so the full oracle calls the generated
  prover API for Stage 6 witness materialization instead of invoking
  `jolt-kernels` directly.
- Generated `jolt-prover` now also exposes `stage6_prover_inputs` and
  `stage7_prover_inputs` from the Bolt artifact generator. The full oracle no
  longer constructs Stage 6/7 prover input builders directly from
  `jolt-kernels`; it asks the generated prover to route shared `jolt-witness`
  data into the kernel executor shape.
- Generated `jolt-prover` now exposes Stage 1/2/3/4/5 prover-input helpers
  from the Bolt artifact generator. The full oracle no longer spells out the
  Stage 1 outer evaluator builder, Stage 2 product/instruction/RAM builder
  chain, Stage 3 cycle builder, or Stage 4/5 sparse-trace builder chains in
  either staged or monolithic prover setup.
- Generated `jolt-prover` now exposes Stage 5/6 opening-input derivation from
  prior public artifacts. The full oracle no longer derives later-stage prover
  opening inputs in `jolt-equivalence`; it only adapts the generated prover's
  kernel-shaped opening inputs into generated-verifier-shaped inputs for
  verifier calls.
- Generated `jolt-prover` now also exposes Stage 2/3/4 opening-input
  derivation, including the Stage 4 initial-RAM precomputed opening through
  `jolt-witness`. The generic source-claim opening-input derivation loop and
  Stage 1/2/3/4 claim lookup helpers were deleted from
  `jolt-equivalence::adapters`; remaining call sites delegate to the generated
  prover API.
- Generated `jolt-prover` now exposes its stage artifact-to-proof conversion
  helpers. `jolt-equivalence::adapters` no longer re-implements the generated
  proof construction for Stage 2/3/4/5/6/7 or the monolithic Jolt proof stage
  wrappers; it delegates to generated prover APIs and only keeps the narrow
  Stage 1 tamper proof bridge and kernel replay bridges.
- Repetitive Bolt program lowering helpers in `bolt_programs.rs` were collapsed
  into one local lowering macro. This is not a semantic move, but it removes
  duplicated harness scaffolding around the Bolt compiler pipeline.
- Stage 2/3 full plan adapters now share the same static-plan conversion
  macros as the later-stage adapter layer.
- The bespoke Stage 2 product-uniskip prefix adapter was deleted. The focused
  product gate now uses the full Stage 2 generated/kernel plans while still
  comparing and tampering the product sumcheck directly, so the harness no
  longer owns a special partial-plan construction path.
- Stage 2/3 core opening-claim parity checks now share one comparison helper.
- Transcript state-history assertions now accept checkpoint logs directly,
  removing repeated `transcript_states(...)` wrapping from the oracle and
  tamper call sites without changing the gate semantics.
- Generated commitment prover/verifier artifact-to-trace conversion now shares
  one adapter macro, keeping the commitment oracle focused on public
  representation conversion.
- Commitment-oracle raw prover/verifier helpers are private and the one-use
  prover-with-cycles wrapper is gone; tests use the focused pair runners.
- The harness-side Stage 1 core R1CS-row to RV64-row replay was deleted.
  Stage 1 now gates typed RV64 and generic R1CS evaluators directly against
  public core uni-skip proof coefficients, instead of maintaining a second
  local core-row translation path.
- Monolithic Jolt tamper rejection assertions now share one local error-shape
  assertion macro, preserving each negative case while removing repeated
  `matches!(Err(...))` scaffolding.
- Removed a stale `core_conversion.rs` comment from the incremental staging
  period that no longer described the proof-conversion adapter.
- Core proof-conversion helpers that are only used inside the crate are now
  `fn`/`pub(crate)` instead of public API; the public surface keeps the
  commitment transcript adapters used by tests.
- Commitment-oracle helpers that only wire generated artifacts internally are
  now `fn`/`pub(crate)`, leaving the public surface focused on test-facing
  commitment traces and transcript construction.
- Adapter canonicalizers, later-stage tamper inputs, and internal core/check
  helpers are crate-private; an unused Stage 4 artifact match wrapper was
  deleted after clippy exposed it as stale public surface.
- Core-oracle entry points now expose only the fixture builders and focused
  Stage 1/2 acceptance checks used by integration tests; deeper full-oracle
  acceptance helpers and generic guest fixture construction are crate-private.
- Unused `CheckpointTranscript` convenience methods and their stale usage
  example were removed; live call sites use the borrowed checkpoint log.
- Obvious artifact/core-conversion comments that restated item names were
  deleted while retaining representation and invariant notes.
- Redundant `bolt_oracle.rs` generated verifier-program rebundling, a
  borrowed Stage 5 opening clone, and a duplicate transcript-prefix assertion
  were deleted; the existing generated verifier bundle now drives all Stage
  5/6/7/full verifier gates.
- Plan-adapter re-exports now keep only the Stage 1/2 adapters used by the
  focused integration tests public; commitment, later-stage, and Stage 8 plan
  adapters are crate-private full-oracle plumbing.
- Bolt program builders now use the same public boundary: commitment and Stage
  1/2 builders remain public for focused tests, while Stage 3-8 builders are
  crate-private full-oracle plumbing.
- `adapters` and `artifacts` are no longer public modules; canonical artifact
  types remain re-exported from the crate root, while representation adapter
  internals stay crate-private.
- Monolithic wrong-stage-slot tamper rejection now uses the shared local
  verifier-error assertion macro instead of hand-rolled `matches!` scaffolding.
- `CoreMuldivCommitmentFixture` now exposes only the fields required by the
  focused tests as public; later-stage witness/core-verifier data and Stage 6
  witness params are crate-private full-oracle inputs.
- Stage 2/3 core opening-claim parity now fails if Bolt omits any eval for a
  mapped opening that `jolt-core` exposes publicly, and it fails if a stage's
  mapping matches no public core claims. Candidate mappings that are not public
  core opening claims remain ignored by this specific public-oracle gate.
- The full Bolt oracle now uses generated `jolt-prover` Stage 6/7 proof
  conversion helpers instead of hand-building generated proof structs.
- The full Bolt oracle now calls generated prover stage execution wrappers for
  Stage 1/2/3/4/5/6/7 prover paths and Stage 5/6/7 proof-carrying replay paths
  instead of invoking `jolt-kernels` execution functions directly. Verifier-mode
  kernel prefix checks still use direct kernel execution as public parity gates.
- Generated `jolt-prover` now also exposes Stage 5/6/7 proof-carrying replay
  helpers. The full oracle no longer constructs later-stage proof-carrying
  kernel executors directly; it asks the generated prover API to replay the
  public proof artifacts with the generated prover plans.
- Generated `jolt-prover` also exposes Stage 1/2/3/4 verifier-mode replay
  helpers for the public prefix parity gates. The full oracle no longer builds
  early-stage verifier kernel executors or selects verifier execution mode
  locally.
- Real-trace commitment proving now uses generated `jolt-prover` commitment
  phase APIs with `jolt-witness::CommitmentTraceSources` /
  `SparseCommitmentInputs`. The harness no longer interprets
  `OracleGeneration` or materializes dense/one-hot witness oracles for the
  muldiv commitment path.
- The synthetic commitment transcript-ordering gate also runs through the
  generated commitment prover/verifier with a tiny synthetic input provider.
  The old harness-local commitment commit/replay loop was deleted.
- Commitment transcript reconstruction now replays the generated commitment
  phase's recorded append bytes after the Jolt preamble. Later-stage tests,
  tamper gates, and the full Bolt oracle no longer pass Bolt commitment CPU
  programs around only to recover transcript step ordering.
- Commitment oracle helpers now own generated commitment plan adaptation for
  test-facing prover/verifier trace construction. The focused tests no longer
  leak generated commitment plans at each call site.
- Core-vs-Bolt perf thresholds are now a named constant in
  `jolt-equivalence::perf`, so the perf oracle configuration lives with the
  perf helpers instead of inside the full Bolt oracle driver.
- Generated commitment trace-source storage now lives behind
  `GeneratedCommitmentInputStorage` in `jolt-equivalence::commitment_oracle`.
  The full Bolt oracle no longer imports generated commitment input types or
  selects trace-source columns at the monolithic prover call site.
- The SHA2-chain `2^20` perf oracle job now runs on pull requests as a standard
  CI gate. Manual workflow dispatch can still skip it with `include_large =
  false`.
- Bolt equivalence and perf CI gates no longer use path filters, so required
  PR gates cannot be skipped by changes in shared crates, guests, inlines, or
  root `jolt-core`.
- The Bolt equivalence workflow now also has an optional full
  `jolt-equivalence` sweep: it runs on the nightly schedule and can be triggered
  manually with `include_full_sweep=true`, while the smaller generated-role and
  real-data parity/tamper jobs remain the standard PR gates.
- `crates/bolt/TESTING.md` documents both SHA2-chain perf oracle PR gates,
  the required nested commitment/evaluation/verification spans, and exact
  local `cargo nextest` commands for the ignored perf tests.
- Perf span gating now uses span names observed by `jolt-profiling` rather
  than a harness-populated allowlist, and `bolt.prove` is part of the required
  top-level Bolt span contract.
- Stage 1 univariate-skip target ordering and proof-polynomial recovery now
  live in `jolt-kernels`. `jolt-equivalence` still compares the resulting
  values, but no longer owns that Stage 1 formula.
- The canonical artifact model now uses `jolt_profiling::PerfMetrics`
  directly. The unused local `PerfSnapshot`/`PerfSpan` vocabulary was deleted
  so perf gates have one metric shape.
- Non-generated `jolt-equivalence`/`jolt-kernels` code no longer uses local
  `#[allow(...)]` lint suppressions. Intentional fail-fast oracle/test helper
  panics are documented with `#[expect(...)]`, and
  `cargo clippy -p jolt-equivalence --tests --offline -- -D warnings`
  passes through the full local dependency stack.
- Bolt emitter/template sources now carry the lint expectations and mechanical
  cleanups needed by generated `jolt-prover`/`jolt-verifier`, so the checked-in
  generated artifacts stay aligned with their generator rail instead of being
  hand-maintained.
- Generated commitment and evaluation proof code now emits nested perf spans
  for Dory commitment work, Stage 8 joint-opening construction, and generated
  verifier evaluation-proof checking. The core-vs-Bolt perf gate requires
  those observed spans, giving CI visibility into commit/opening costs instead
  of only coarse stage totals.
- Generated verifier common runtime now validates opening-input count, expected
  symbols, and point arity before seeding verifier state. This is implemented
  in the Bolt verifier-common template and reflected in checked-in
  `jolt-verifier`, so extra/missing malformed opening inputs are rejected by
  generated code rather than tolerated by the harness.
- Generated `jolt-prover` now owns construction of kernel prover executors from
  generated stage-input structs, including a `JoltProverStageInputs` wrapper for
  monolithic proving. The full oracle still supplies witness/input data, but no
  longer instantiates prover executors or wires `JoltProverInputs` directly.
- Generated `jolt-prover` now exposes Stage 5 kernel-proof and Stage 6 bytecode
  read-RAF data storage helpers. The full oracle no longer imports Stage 5/6
  kernel modules just to build those representation values.
- Generated `jolt-prover` now owns generated-to-kernel Stage 6/7 proof-shape
  conversion for proof-carrying replay. The local
  `define_kernel_proof_adapter!` macro was deleted from `adapters.rs`.
- Generated `jolt-verifier` now exposes public stage challenge/execution
  artifact aliases, and generated `jolt-prover` now exposes Stage 6/7 generated
  execution-artifact wrapping plus partial `JoltProof` assembly helpers. The
  oracle and tamper gates call those generated APIs directly instead of
  maintaining local Stage 6/7 artifact wrappers or partial proof constructors in
  `jolt-equivalence`.
- Early-stage tamper gates now replay Stage 1/2/3 proof prefixes through
  generated `jolt-prover` replay helpers instead of instantiating verifier
  kernel executors in `jolt-equivalence`.
- Generated `jolt-prover` now exposes Stage 1 kernel-proof to generated-proof
  conversion, and `jolt-equivalence::adapters` no longer maps Stage 1
  sumcheck/eval records by hand.
- The remaining pass-through generated proof adapters for Stages 2/3/4/5 were
  deleted; oracle and tamper checks now call the generated `jolt-prover`
  `stage*_proof` helpers directly.
- Generated `jolt-verifier` now exposes top-level Stage 2 RAM data aliases and
  generated `jolt-prover` owns kernel-to-verifier Stage 2 RAM data storage.
  The local `GeneratedStage2RamData` adapter was deleted from
  `jolt-equivalence`.
- Generated `jolt-verifier` now exposes a top-level verifier opening-input
  alias, and generated `jolt-prover` owns generic kernel-to-verifier
  opening-input conversion. The six local generated opening-input adapter
  functions were deleted from `jolt-equivalence`.
- Generated `jolt-verifier` now reexports standalone stage verifier entry
  points and verifier program plan aliases at the crate root. The oracle and
  tamper gates no longer invoke generated verifier stage functions through
  `jolt_verifier::stages::*`; stage modules are only used where typed generated
  proof/input aliases are still the public data shape.
- Generated `jolt-verifier` also reexports Stage 6 verifier-data aliases at
  the crate root, and generated `jolt-prover` constructs them through that
  boundary. The tamper module no longer imports generated verifier stage
  modules for proof/opening/verifier-data type aliases.
- Common generated proof, opening-input, and execution-artifact shapes are now
  named through `jolt_verifier` crate-root aliases in the oracle, conversion,
  adapter, check, and tamper layers. Remaining `jolt_verifier::stages::*`
  imports in `jolt-equivalence` are limited to real stage constants/plans such
  as Stage 6 opening-claim plans and generated program adapters.
- Stale equivalence documentation was brought back in sync with the current
  full standard non-zk prefix: `crates/jolt-equivalence/README.md` no longer
  describes Stage 1/2 failures, and `crates/bolt/TESTING.md` now points at the
  `jolt-equivalence` goal and nextest-based gates.
- Generated `jolt-prover` now owns Stage 6 witness-entry to generated verifier
  bytecode data conversion, using public aliases emitted in generated
  `jolt-verifier`. The local Stage 6 bytecode verifier-data adapter was deleted
  from `jolt-equivalence`.
- The generated/kernel static-plan adapter macros are now mode-aware instead
  of maintaining separate Stage 2/3 legacy and Stage 4/5/6/7 kernel/generated
  mapping bodies. This keeps the remaining plan bridge as thin format
  translation; the per-stage adapter submodules now carry much less repeated
  field-mapping code.
- Stage 2/3 generated/kernel sumcheck plans now use the shared optional
  `kernel`/`relation` shape. Generated Stage 2/3 verifier code aliases the
  same shared plan structs as the later stages, and the equivalence adapter
  uses one no-absorb generic lowering path instead of the stale Stage 2/3
  legacy adapter macro.
- Stage 1 generated/kernel sumcheck plans now also use the shared optional
  `kernel`/`relation` shape. Generated Stage 1 verifier code aliases the same
  shared plan structs, and the last `legacy_kernel`/`legacy_generated`
  conversion arms were deleted from the equivalence plan adapter layer.
- The stale verifier-only sumcheck claim/driver plan structs were deleted from
  the Bolt verifier-common template and checked-in `jolt-verifier` runtime.
  All generated verifier stages now use the same shared sumcheck plan structs.
- Generated stage plus monolithic-prefix proof tamper rejection now shares one
  local helper in `tamper.rs`, and stale one-use prefix replay wrappers were
  deleted. The covered coefficient, eval, point, and opening-input mutations
  are unchanged.
- The Stage 2 product-uniskip oracle now runs through the full Stage 2
  generated/kernel plans and passes Stage 2 RAM data through the existing
  verifier input path. This deleted the focused product-only plan adapter and
  reduced `plan_adapters.rs` to 767 LOC and the crate's Rust total to 6,572
  LOC without dropping the product proof parity or tamper gate.
- The full Bolt oracle now reuses the already-verified Stage 2 boundary
  transcript and opening inputs when seeding the standalone generated Stage 3
  verifier path. This deletes a redundant Stage 1/2 prefix replay from
  `bolt_oracle.rs`; the oracle still compares the same staged, generated, and
  monolithic transcript histories.
- Stage 2/3 public core opening-claim expectation tables in `checks.rs` now
  use a compact row macro with explicit one-row-per-claim mappings. This keeps
  the public-oracle map visible while removing repeated wrapper boilerplate,
  reducing `checks.rs` to 585 LOC and the crate's Rust total to 6,424 LOC.
- The shared Stage 1/2/3/4/5/6/7 plan adapter macros no longer carry stale
  kernel/generated dispatch arms for already-unified claim and driver shapes.
  The generated and kernel plan contracts are unchanged, but
  `plan_adapters.rs` is down to 755 LOC and the crate's Rust total is 6,412
  LOC.
- Full core-proof conversion now reuses the staged Stage 1-7 core-proof patch
  helper chain before patching commitments and the evaluation proof. This
  removes a second hand-written list of Stage 1-7 sumcheck-field assignments
  from `core_conversion.rs`.
- The one-use core opening-claims clone wrapper was deleted from
  `core_conversion.rs`; the proof clone now spells the public `Claims` clone
  directly at the field assignment site.
- Generated-role crate tests now share one small driver-presence assertion
  macro, preserving the same Stage 1-7 prover/verifier surface checks while
  trimming repeated test scaffolding.
- Stale commitment oracle wrappers were deleted: the unused direct
  trace-source prover runner and the one-use transcript-trace method. The live
  commitment path still goes through `GeneratedCommitmentInputStorage` and the
  generated prover/verifier commitment APIs.
- Generated Stage 8 prover/verifier plan adapters now share the same typed
  conversion macro, cutting another 67 LOC from `plan_adapters.rs` without
  changing the public adapter functions.
- Stage 1/2/3/4 trace-row materialization moved into
  `jolt-kernels::trace`, next to the kernel ABI structs that consume those
  rows. The core oracle now calls shared kernel-owned helpers for RV64 rows,
  product-virtual rows, instruction-lookup rows, RAM accesses, Stage 3 cycles,
  Stage 4 register accesses, and Stage 5 lookup trace columns instead of
  constructing them locally.
- Stage 6 bytecode-entry materialization also moved into
  `jolt-kernels::trace`, using public `jolt-trace` instruction flag helpers.
  The core oracle now supplies only the core lookup-table index callback,
  keeping bytecode witness row construction out of the harness.
- Generated `jolt-prover` now exposes `JoltProverWitnessInputs` plus
  `prove_jolt_with_witness_inputs`, emitted from the Bolt artifact generator.
  The full Bolt oracle no longer constructs each monolithic kernel stage input
  struct itself; it passes the public witness/opening views into the generated
  prover API and lets that crate assemble the stage inputs.
- Generated `jolt-prover` now also exposes standalone Stage 1/2/3/4/5/6/7
  witness-input prove wrappers emitted from the Bolt artifact generator. The
  staged full oracle now passes public witness/opening views to those generated
  APIs instead of constructing each stage input struct locally.
- Generated `jolt-prover` crate-root exports now include those standalone
  witness prove wrappers plus Stage 6 verifier-data conversion. The harness no
  longer reaches into `jolt_prover::prover` for equivalence paths; it consumes
  the generated crate's public API boundary directly.
- Focused Stage 1/2 tests and the full Bolt oracle no longer instantiate kernel
  executors directly. They call generated prover/replay APIs and keep direct
  kernel execution plumbing out of the harness path.
- The Stage 1 plan adapter no longer synthesizes missing verifier kernels in
  the harness. Kernel replay uses the prover plan, while generated-verifier
  checks use the generated verifier plan.
- Deleted the unused `CoreScaffold` bridge from `core_conversion.rs`; core-proof
  conversion now keeps only live proof translation helpers.
- Stage 3/4/5 tamper checks now consume generated-verifier start transcripts
  and opening inputs produced by the full oracle instead of replaying Stage 1/2
  prefixes and reconstructing Stage 3 openings internally for each mutation.
- Stage 2 batched tamper checks now consume the focused test's Stage 2 start
  transcript and opening inputs instead of replaying Stage 1 inside every
  mutation check.
- Stage 2 product-uniskip tamper checks reuse the verified Stage 2 start
  transcript and generated opening inputs from their positive chain check,
  avoiding repeated Stage 1 prefix replay per mutation.
- Stage 1 tamper checks now receive the already-created Stage 1 start
  transcript from the focused test instead of rebuilding the commitment
  transcript per mutation.
- Generated `jolt-prover` now owns Stage 6 witness construction inside
  monolithic proving and exposes Stage 6/7 trace-witness prove wrappers. The
  full oracle passes `Stage6WitnessParams` plus trace inputs instead of creating
  `Stage6WitnessPolynomials` or witness slices locally.
- Generated `jolt-prover` also owns Stage 4/5 sparse trace witness
  construction inside monolithic proving and exposes Stage 4/5 trace-witness
  prove wrappers. The full oracle passes register/RAM accesses instead of
  creating `Stage45SparseTraceWitness` locally, and the core fixture no longer
  exposes that witness builder.
- Fixture-derived Stage 2 RAM data and Stage 6 witness parameters are now
  exposed as named public views on `CoreMuldivCommitmentFixture`; `bolt_oracle.rs`
  no longer spells out the fixture field mapping for those kernel inputs.
- Fixture-derived Stage 1 R1CS key/data views are now exposed on
  `CoreMuldivCommitmentFixture`; the focused tests and Bolt oracle no longer
  construct `Stage1OuterRv64Data`/`Stage1OuterR1csData` directly from fixture
  fields.
- Stage 4/5 sparse trace witness routing moved onto `jolt-kernels` public
  builder methods: `Stage4ProverInputs::with_sparse_trace_witness` and
  `Stage5ProverInputs::with_sparse_trace_witness`. The equivalence harness no
  longer constructs those kernel witness structs by hand in both standalone and
  monolithic prover paths.
- Stage 4/5 sparse trace witness bundle routing moved further into
  `jolt-kernels` via `with_stage45_sparse_trace_witness`; `bolt_oracle.rs`
  now passes the shared `Stage45SparseTraceWitness` bundle instead of selecting
  the per-stage sparse columns itself.
- Stage 4/5 sparse trace witness columns moved into
  `jolt-witness::stage4_5_sparse_trace_witness`; `bolt_oracle.rs` no longer
  computes `rd_inc`, `ram_inc`, RAM address, or RD write-address columns by
  hand.
- Additional primitive prover-side helpers moved into `jolt-witness`: generated
  dense/one-hot `CycleInput` source selection, one-hot padding policy mapping,
  optional oracle fixture data, `u64` increment columns, optional address
  columns, and generic `u64` MLE evaluation. The harness now calls shared
  helpers instead of defining local `stage4_rd_inc`, `stage2_ram_inc`,
  `dense_source`, `one_hot_source`, `padding_value`, `optional_oracle_data`, or
  `mle_eval_u64` functions.
- Generated commitment trace-source grouping moved into
  `jolt-witness::CommitmentTraceSources`, and the Bolt commitment emitter now
  generates `CommitmentOracleInputs::from_trace_sources`. The equivalence
  oracle no longer selects generated commitment columns by source-string at the
  monolithic prover call site.
- The commitment oracle now exposes canonical `CommitmentTrace` and
  `TranscriptTrace` snapshots. The commitment tests compare these
  representation-only artifacts through `EquivalenceRun` for Bolt
  prover/verifier parity and core commitment parity, starting to wire the
  canonical model into live gates.
- Monolithic commitment parity now also compares canonical `CommitmentTrace`
  snapshots, and the old raw optional-commitment checker was deleted from
  `checks.rs`.
- Stage 4/5/6/7 artifact parity now compares canonical `StageArtifacts`
  snapshots instead of comparing generated/kernel proof structs directly.
- The full Bolt oracle now compares staged-vs-monolithic generated prover
  output as canonical `EquivalenceRun` commitments plus Stage 4/5/6/7 snapshots.
- Generated `jolt-verifier` now exposes explicit
  `verify_jolt_through_stage{5,6,7}_with_programs` entrypoints emitted from
  Bolt's artifact generator. The equivalence oracle no longer fabricates empty
  Stage 6/7 verifier programs to scope prefix gates.
- Bolt perf metrics now include generated proof byte counts, and the perf
  oracle enables the proof-size ratio gate instead of leaving proof size
  unreported.
- Generated proof-size accounting now lives in `jolt-equivalence::perf`
  instead of the full Bolt oracle, keeping measurement details with the perf
  gate helpers.
- The full generated prover/verifier parity check now compares canonical
  `EquivalenceRun` commitment and Stage 4/5/6/7 snapshots for both sides
  instead of only checking late-stage sumcheck counts.
- Generated `jolt-prover` now exposes
  `stage7_opening_inputs_from_stage6_artifacts{,_with_program}` from the Bolt
  artifact generator. The full oracle no longer looks up Stage 6 opening
  claims by Stage 7 source strings locally; `jolt-equivalence` only converts
  the generated prover's kernel input shape into the generated verifier input
  shape.
- The generated standalone verifier crate now has an explicit `serde`
  dependency override, so the generated-artifact crate-layout gate compiles it
  without relying on workspace path dependency resolution.
- Stage 6 bytecode read-RAF source data now uses a neutral
  `jolt-witness::Stage6BytecodeEntry` fixture shape. Generated verifier data is
  rendered from that shared witness shape, while kernel prover bytecode input is
  rendered by `jolt-kernels::stage6::Stage6BytecodeReadRafDataStorage`. This
  deletes the generated-verifier-to-kernel bytecode conversion from
  `jolt-equivalence::adapters` and preserves the generated prover/verifier
  import boundary enforced by the artifact gate.
- Stage 2 product-virtual witness construction moved into
  `Stage2ProverInputs::with_product_virtual_witness`; the kernel now derives
  the product uniskip extended evaluations from the Product opening point
  instead of requiring `jolt-equivalence` to find `tau_low` and compute them.
  The focused Stage 2 tests now use this builder too, so
  `tests/bolt_commitment.rs` no longer imports or calls
  `product_virtual_uniskip_extended_evals` directly.
- Stage 4/5 sparse trace witness construction over kernel access types moved
  into `jolt-kernels::stage4::stage4_5_sparse_trace_witness_from_accesses`;
  the equivalence oracle no longer maps `Stage4RegisterAccess` and
  `Stage2RamAccess` into generic tuple streams itself.
- The core fixture now exposes the Stage 4/5 sparse trace witness as a named
  public view; `bolt_oracle.rs` no longer pairs fixture register/RAM access
  fields to build that witness at the call site.
- Stage 4 initial-RAM opening construction moved into
  `jolt-witness::stage4_ram_val_init_opening`; `jolt-equivalence` no longer
  performs the `RamValInit` MLE evaluation directly.
- Point-order helpers moved into `jolt-witness`: `reverse_point`,
  `reversed_suffix`, `normalized_stage4_registers_rw_point`,
  `stage5_instruction_cycle_point`, `stage5_instruction_ra_point`,
  `stage5_ram_ra_point`, and `stage5_registers_val_point`. The harness still
  wires opening inputs from generated/core artifacts, but no longer owns those
  point-normalization primitives.
- Transcript state-history checker helpers moved from `tests/bolt_commitment.rs`
  into `jolt-equivalence::checkpoint`, starting the Slice 4 split of checker
  utilities out of the monolithic integration test.
- Commitment/proof checker helpers moved into `jolt-equivalence::checks`, and
  the malformed unrelated Dory proof fixture moved into
  `jolt-equivalence::tamper`.
- Stage 1 tamper rejection moved into `jolt-equivalence::tamper`; the
  integration test invokes the tamper gate instead of owning the mutation and
  verifier-rejection loop.
- Stage 2 product uni-skip chain verifier/tamper rejection moved into
  `jolt-equivalence::tamper`; `tests/bolt_commitment.rs` no longer owns the
  Stage 2 product mutation loop.
- Stage 2 batched verifier tamper rejection moved into
  `jolt-equivalence::tamper`; the integration test now only supplies the
  public Stage 1/2 artifacts, RAM data, and verifier plans.
- Repeated tamper mutation construction in `jolt-equivalence::tamper` now uses
  small shared helpers for coefficient/eval/point edits, preserving the same
  rejection coverage while shrinking stage-local boilerplate.
- Repeated tamper transcript replay, Stage 1/2 prefix replay, and generated
  monolithic verifier rejection setup now use small shared helpers in
  `jolt-equivalence::tamper`; the mutation set and verifier coverage are
  unchanged.
- Bolt transcript construction with the Jolt preamble and commitment transcript
  steps is now centralized in `jolt-equivalence::commitment_oracle`; the test
  driver, `bolt_oracle.rs`, and `tamper.rs` no longer each own local
  `CheckpointTranscript` bootstrap helpers.
- Repeated batched sumcheck tamper triplets now use one local macro in
  `jolt-equivalence::tamper`; the same coefficient/eval/point mutations are
  still applied to every covered stage.
- Stage 6/7 generated verifier and monolithic-prefix tamper rejection moved
  into `jolt-equivalence::tamper`; `bolt_oracle.rs` now passes public
  artifacts into those gates instead of owning the mutation loops.
- Stage 6/7 tamper gates now also mutate public opening-claim input
  evaluations and require both the standalone generated verifier and generated
  monolithic prefix verifier to reject them. The generated proof shape has no
  separate opening-claim field, so this gates the public claim contract the
  verifier actually consumes.
- Stage 6/7 tamper gates now also remove one opening-claim input and append a
  duplicate extra opening-claim input, and shorten an opening point, requiring
  both the standalone generated verifier and generated monolithic prefix
  verifier to reject malformed public input lists and invalid point arity.
- Generated verifier opening-input validation now lives in the Bolt verifier
  common template and checked-in generated verifier: malformed input count,
  duplicate/missing symbols, and invalid point arity are rejected before a
  stage accepts public opening inputs.
- The generated verifier cleanup gate is back under its strict targets after a
  generator-level top-level API compaction and regeneration:
  generated surface 5,966 LOC, shared runtime 1,789 LOC, total verifier 7,755
  LOC, and `verifier.rs` 487 LOC.
- Monolithic full-proof tamper gates now mutate the Stage 8 opening-point
  suffix consumed from the declared Stage 7 opening input, and require the
  generated verifier/Dory check to reject the proof. Prefix point coordinates
  are intentionally not gated here because Stage 8 reconstructs them from the
  Stage 7 sumcheck point rather than trusting the opening input.
- Stage 3/4/5 generated verifier and monolithic-prefix tamper rejection moved
  into `jolt-equivalence::tamper`; the full-trace Bolt oracle no longer owns
  per-stage proof mutation loops.
- Monolithic full-proof verifier rejection gates moved into
  `jolt-equivalence::tamper`: missing verifier setup, missing evaluation proof,
  tampered evaluation proof, missing required commitment, missing Stage 1-7
  proofs, and swapped Stage 6/7 proof slots.
  `bolt_oracle.rs` now passes the honest monolithic proof plus public verifier
  inputs into one tamper gate.
- Repeated generated monolithic `JoltVerifierInputs` literals in
  `bolt_oracle.rs` now use one borrowed verifier-input view with explicit
  stage-prefix constructors, reducing verifier wiring duplication without
  changing any stage semantics.
- Generated `jolt-verifier` now emits target-scoping methods on
  `JoltVerifierInputs` (`through_stage5`, `through_stage6`, `through_stage7`,
  and `full`). The equivalence harness consumes that generated API directly,
  so `adapters.rs` no longer owns monolithic Jolt verifier input construction.
- Stage 2/3 core opening-claim parity tables moved into
  `jolt-equivalence::checks`; `tests/bolt_commitment.rs` no longer imports core
  opening IDs or witness polynomial enums for those comparisons.
- Stage 1/2/3/6 core proof coefficient parity helpers moved into
  `jolt-equivalence::checks`, further narrowing `tests/bolt_commitment.rs` to
  orchestration plus core acceptance wiring.
- Stage 1 uniskip extended-eval parity moved into
  `jolt-equivalence::checks`, removing another checker helper from the
  monolithic integration driver.
- Stage 4/5/6/7 artifact equality assertions moved into
  `jolt-equivalence::checks`, continuing the split of checker code out of the
  monolithic `bolt_commitment.rs` test.
- Repeated Stage 4/5/6/7 artifact equality loops and Stage 2/3/6 compressed
  sumcheck coefficient checks now use local checker macros; the assertions are
  still thin public-artifact comparisons rather than semantic reconstruction.
- Repeated canonical Stage 4/5/6/7 artifact wrapper functions in
  `jolt-equivalence::adapters` now share one typed adapter macro, preserving
  public function names while cutting another 25 LOC of representation-only
  boilerplate.
- The perf tracing setup helper moved into `jolt-equivalence::perf`; the perf
  oracle tests now live in the dedicated `tests/bolt_perf.rs` target.
- Thin generated/kernel representation adapters for opening inputs, Stage 2
  RAM data, and Stage 6 bytecode entries moved into
  `jolt-equivalence::adapters`.
- Repeated generated/kernel opening-input representation adapters now share one
  typed adapter macro in `jolt-equivalence::adapters`.
- Stage 2/3/4/5/6 opening-input derivation now shares one generic
  `source_stage`/`source_claim` adapter loop; the stage-specific source map is
  explicit, but the harness no longer repeats the same map/assert/panic
  scaffolding in every stage adapter.
- Generated Stage 2 RAM conversion now uses
  `jolt-equivalence::adapters::GeneratedStage2RamData`; `bolt_oracle.rs` and
  Stage 2 tamper checks no longer rebuild generated RAM access/layout structs
  locally.
- Generated-to-kernel Stage 6 bytecode read-RAF conversion now uses
  `jolt-equivalence::adapters::KernelStage6BytecodeData`; the Bolt oracle no
  longer unwraps generated verifier data and rebuilds kernel bytecode data
  locally.
- Bolt oracle transcript initialization now uses small preamble/commitment
  helpers, keeping transcript setup explicit while reducing repeated harness
  scaffolding.
- Generated Stage 6 verifier-data construction from bytecode preprocessing
  moved into `jolt-equivalence::adapters`; `tests/bolt_commitment.rs` now
  treats it as a public-format adapter instead of owning the conversion.
- Kernel/generated/Jolt proof-shape converters moved into
  `jolt-equivalence::adapters`; `bolt_commitment.rs` now imports
  `to_generated_*`, `to_kernel_*`, `to_jolt_*`, and generated execution
  artifact adapters instead of defining them locally.
- Core proof-shape converters moved into `jolt-equivalence::core_conversion`;
  `bolt_commitment.rs` now imports `clone_core_proof`,
  `to_core_uniskip_proof`, and `to_core_stage2_uniskip_proof` instead of
  defining local core proof constructors.
- Core proof patch builders for staged Bolt acceptance and full/evaluation
  proofs moved into `jolt-equivalence::core_conversion`; the integration test
  now only calls public core verifier acceptance on those converted proofs.
- Core oracle fixture construction, trace-to-kernel public oracle data, RV64
  row parity checks, and public core acceptance gates moved into
  `jolt-equivalence::core_oracle`; the integration test no longer owns the
  jolt-core runner or staged core verifier wrappers.
- Core commitment transcript replay moved into
  `jolt-equivalence::core_conversion`; the monolithic test now supplies only
  public commitment records and transcript steps to the oracle helper.
- Bolt commitment-phase replay moved into
  `jolt-equivalence::commitment_oracle`; the test no longer owns commitment
  materialization, optional-commit skip policy handling, Dory layout commits,
  or commitment transcript-log construction.
- Bolt preamble transcript append logic moved into
  `jolt-equivalence::commitment_oracle` behind a small source trait, removing
  local label/u64/bytes transcript encoding helpers from the integration test.
- Commitment oracle data materialization for real traces moved into
  `jolt-equivalence::commitment_oracle`; the integration test now asks the
  Bolt adapter for oracle inputs instead of owning `OracleGeneration` handling.
- Bolt protocol-program construction moved into
  `jolt-equivalence::bolt_programs`; `tests/bolt_commitment.rs` no longer owns
  MLIR lowering/projecting/extraction glue for each stage.
- Generated/kernel static plan adapters moved into
  `jolt-equivalence::plan_adapters`; the integration test now asks for
  translated plans instead of owning the `leak_*` conversion block.
- Stage 1/2/3/4/5/6/7 kernel plan adapters were split into focused
  `plan_adapters/stage*.rs` submodules, reducing the monolithic
  `plan_adapters.rs` while preserving the same public adapter functions.
- Generated commitment and Stage 1/2/3/4/5/6/7 verifier plan adapters were
  split into focused `plan_adapters/generated_*.rs` submodules.
- Stage 4/5/6/7 generated-verifier and kernel plan adapters now use typed
  macro-generated field mapping instead of repeated stage-local boilerplate,
  reducing the adapter subtree by about 1.3k LOC without changing the public
  adapter functions.
- Kernel/generated/Jolt proof-shape converters now use typed adapter macros
  instead of repeated stage-local conversion bodies, preserving the same public
  conversion functions while cutting more boilerplate from `adapters.rs`.
- Full real-trace Bolt prover/verifier orchestration moved into
  `jolt-equivalence::bolt_oracle`; `tests/bolt_commitment.rs` is now focused
  on commitment/Stage 1/Stage 2 orchestration and public gate calls.
- Stage 7 opening-input adaptation from public Stage 6 opening claims moved
  into `jolt-equivalence::adapters`.
- Stage 2/3/4/5/6/7 kernel execution artifacts now record public opening-claim
  values produced by the stage plans, including logical aliases that are
  transcript-deduped. Stage 3/4/5/6 opening inputs are now derived by thin
  `source_claim` lookups over generated/kernel `opening_inputs` instead of
  local stage point/eval formulas in `bolt_commitment.rs`.
- Stage 2 opening inputs are also derived by thin `source_claim` lookups over
  the generated/kernel `opening_inputs` plan. The local
  `stage2_product_opening_inputs` and `stage2_opening_inputs` builders were
  deleted from `tests/bolt_commitment.rs`.
- Stage 2 RAM read-write instance point order is fixed in Bolt codegen and the
  checked-in generated Stage 2 prover/verifier constants so committed `RamInc`
  opening points match core-facing trace-domain semantics. This removed the
  harness-only Stage 6 `RamInc` normalization workaround.
- Dead `#[cfg(any())]` private-core replay blocks were deleted from
  `tests/bolt_commitment.rs`.
- The remaining live private-core preamble replay path was deleted:
  `run_core_preamble`, `core_challenge_to_fr`, and
  `assert_core_stage1_tau_matches_bolt` no longer exist. Stage 1 parity now
  relies on public verifier acceptance plus direct proof-coefficient parity.
- Checked-in generated `jolt-prover` changes are regenerated from Bolt emitter
  changes; `jolt-prover`/`jolt-verifier` remain generated artifacts.
- The focused `bolt_stage3_batched_real_muldiv_self_parity` oracle now hard-gates
  Stage 5/6/7 core acceptance, generated verifier acceptance, full core
  acceptance, and Dory joint-opening proof byte equality.
- Added a dedicated `Bolt equivalence` CI workflow with explicit macOS/LLVM
  nextest jobs for generated role full-field parity and real-data
  parity/tamper gates.
- Moved the ignored SHA2-chain perf oracle tests into dedicated
  `tests/bolt_perf.rs` and pointed the perf-oracle workflow at that nextest
  target.
- `scripts/setup-bolt-dev.sh` now installs `cargo-nextest` along with the
  local LLVM/MLIR environment, matching the branch's documented gates.
- Deleted local generated-verifier replay debt: `Stage6LocalValueStore`,
  `stage6_value_store`, `stage6_instance_point`, and
  `stage6_opening_as_stage7_input`.
- Deleted stale `tests/hash_debug.rs`, a one-off Blake2b op-112 divergence
  scanner with no assertions and no role in the equivalence or CI gates.
- Deleted the unused legacy `StageTrace`/`ProtocolTrace` API from `lib.rs`;
  the crate now exposes the canonical `artifacts.rs` model instead of two
  parallel comparison vocabularies.

## Initial Helper Classification

This classification is the Slice 1 migration ledger for
`tests/bolt_commitment.rs`. It is intentionally conservative: anything that
constructs stage-specific protocol meaning is semantic debt until it is moved
into Bolt, generated crates, or `jolt-witness`.

Adapter/oracle helpers allowed to remain, but should move out of the monolithic
test file:

- fixture runners: `core_muldiv_commitment_fixture`,
  `core_sha2_chain_commitment_fixture`, `core_guest_commitment_fixture`
- Bolt protocol builders: `bolt_commitment_programs_with_params`,
  `bolt_stage{1,2,3,4,5,6,7,8}_programs_with_params`
- generated-plan adapters: `leak_*`, `role_name`, `synthetic_stage1_*`,
  `leak_stage*_program`, `leak_generated_*_program`
- proof/artifact representation adapters: `to_generated_*`, `to_jolt_*`,
  `to_kernel_*`, `generated_stage*_execution_artifacts`,
  `kernel_stage*_opening_inputs`, `generated_stage*_opening_inputs`
- commitment/transcript adapters: generated commitment phase runners,
  `append_bolt_preamble`, `append_bolt_commitment_trace`,
  `core_commitment_log`, `core_commitments_transcript_log`
- thin field/serialization conversions: `core_challenge_to_fr`,
  `core_append_serializable_bytes`, `commit_with_layout`,
  `into_padded_oracle`

Checker helpers allowed to remain, but should move to focused check modules:

- transcript/checkpoint checks: `transcript_states`,
  `assert_state_history_match`, `assert_state_history_prefix_match`
- artifact equality checks: `assert_stage*_artifacts_match`,
  `assert_commitments_match`, `assert_dory_proofs_match`
- core/Bolt acceptance gates that use public verifier APIs:
  `assert_core_verifies_proof`, `assert_core_accepts_bolt_stage*`,
  `assert_core_accepts_bolt_evaluation_proof`,
  `assert_core_accepts_full_bolt_proof`
- early-stage parity checks that do not reconstruct hidden later-stage
  semantics: `assert_core_stage*_sumcheck_proof_matches_bolt`,
  `assert_stage1_uniskip_extended_evals_match_core`

Tamper/fuzz helpers are allowed, but should move to `tamper.rs`:

- `assert_bolt_stage1_tamper_rejected`
- stage-specific proof mutation helpers now live in `tamper.rs`
- malformed proof helpers such as `unrelated_dory_proof`

Perf helpers are allowed, but should move to `perf.rs`:

- `maybe_setup_perf_trace`
- dedicated ignored perf gates in `tests/bolt_perf.rs`
- core/Bolt timing, RSS, proof-size, and tracing metric collection

Semantic debt that must leave `jolt-equivalence`:

- private-core or accumulator-derived stage reconstruction:
  `ProofOpeningClaims`, `core_stage{4,5,6,7}_artifacts`,
  `core_stage8_transcript_states`,
  `core_stage{4,5,6,7}_round_polynomials`,
  `core_stage{4,5,6,7}_opening_inputs`,
  `core_stage{4,5,6,7}_evals`,
  `core_stage*_virtual_*`, `core_stage*_committed_*`
- core transcript state replay that depends on private verifier internals:
  deleted from the live harness; keep it out.

The expected trend is that adapter/checker/tamper/perf helpers move into small
modules, while semantic-debt helpers disappear because Bolt/generated artifacts
or `jolt-witness` expose the required public data directly.

## Target Architecture

Use a small canonical model inside `jolt-equivalence`:

```rust
struct EquivalenceRun {
    commitments: CommitmentTrace,
    transcript: TranscriptTrace,
    stages: Vec<StageArtifacts>,
    opening_claims: OpeningClaims,
    verifier_result: VerifierResult,
    perf: PerfMetrics,
}
```

Core and Bolt should each have narrow adapters:

```text
core_oracle.rs
  run_core(...)
  public artifacts -> EquivalenceRun

bolt_oracle.rs
  run_generated_bolt(...)
  generated artifacts -> EquivalenceRun

checks.rs
  compare_commitments
  compare_transcripts
  compare_opening_claims
  compare_stage_outputs
  compare_perf

tamper.rs
  mutate proof artifacts
  assert generated verifier rejects
```

The heavy semantic logic must live in:

```text
crates/bolt                IR, typed plans, lowering, codegen
crates/jolt-witness        prover-side witness/data materialization primitives
crates/jolt-kernels        prover/runtime relation execution
crates/jolt-prover         generated prover artifact production
crates/jolt-verifier       generated verifier artifact checking
```

For later-stage prover inputs, Bolt should use or adapt `jolt-witness` rather
than rebuilding witness and claim materialization inside `jolt-equivalence`.
If Stage 6/7 needs reusable primitives for RA chunks, hamming-weight data,
opening-input materialization, claim inputs, or committed oracle layouts, add
those primitives to `jolt-witness` and have generated prover code consume them.
The equivalence crate may inspect the resulting public artifacts, but it must
not be the place where those prover-side semantics are constructed.

`jolt-witness` is the canonical home for reusable prover-side construction
that is not inherently codegen-specific. That includes witness columns,
committed-oracle layouts, RA chunk/index materialization, hamming-weight inputs,
and claim/opening input data needed by later stages. The generated prover may
wrap or specialize those primitives, but `jolt-equivalence` must only compare
the public artifacts emitted by core and Bolt.

## Focused Goal-Mode Slices

### Slice 1: Define The Boundary

Objective:

```text
Introduce a canonical equivalence artifact model and classify existing
bolt_commitment.rs helpers as adapter, checker, tamper, perf, or semantic debt.
```

Acceptance gates:

- No new semantic reconstruction is added.
- Documented list of debt helpers exists in code comments or this file.
- New canonical structs are representation-only.
- Existing tests are not weakened or deleted unless replaced by equivalent
  stricter gates.

### Slice 2: Move Stage 6 Opening Semantics Into Bolt

Objective:

```text
Make generated Bolt Stage 6 expose canonical opening claims and normalized
points/evals directly, matching core public semantics.
```

Required work:

- Represent point order/normalization as typed protocol facts, not loose
  stage-local string recovery in the harness.
- Use or extend `jolt-witness` for Stage 6 prover-side witness/data
  materialization, including committed RA oracle layouts and claim inputs.
- Generated Stage 6 prover artifacts must carry the exact opening claims
  Stage 7 consumes.
- Generated Stage 6 verifier must check those claims through the same public
  artifact contract.
- Remove matching Stage 6 reconstruction from `bolt_commitment.rs`.

Acceptance gates:

- Stage 6 proof produced by Bolt self-verifies.
- Stage 6 canonical opening claims compare against core oracle.
- `jolt-equivalence` no longer synthesizes Stage 6 output opening points.
- Harness LOC decreases or any increase is limited to generic adapters/checks.

### Slice 3: Move Stage 7 Hamming-Weight Semantics Into Bolt

Objective:

```text
Make generated Bolt Stage 7 consume Stage 6 canonical openings and construct
the hamming-weight claim reduction with core-equivalent semantics.
```

Required work:

- Stage 7 input claim is derived from generated Stage 6 canonical openings.
- Use or extend `jolt-witness` for Stage 7 prover-side RA chunk/index and
  hamming-weight input materialization.
- Hamming-weight reduction point construction matches core semantics in Bolt
  kernels/codegen.
- Generated Stage 7 prover and verifier agree without harness-side witness MLE
  evaluation.
- Stage 7 output opening claims are public generated artifacts.

Acceptance gates:

- Current failing Stage 7 relation output mismatch is resolved in Bolt code.
- `bolt_stage3_batched_real_muldiv_self_parity` passes without semantic
  reconstruction in `jolt-equivalence`.
- Stage 7 tamper checks reject coefficient, eval, point, and opening-claim
  mutations.
- Harness LOC decreases.

### Slice 4: Thin The Harness

Objective:

```text
Refactor bolt_commitment.rs into focused oracle/check/tamper/perf modules and
delete semantic debt.
```

Target shape:

```text
tests/bolt_commitment.rs       small integration driver
src/core_oracle.rs             core public oracle adapter
src/bolt_oracle.rs             generated Bolt adapter
src/checks.rs                  parity assertions
src/tamper.rs                  proof mutations
src/perf.rs                    perf gates
```

Acceptance gates:

- No local value-store replay in tests.
- No witness polynomial MLE evaluation in tests except generic checker helpers
  operating on already-public artifacts.
- No private core verifier stage methods.
- `bolt_commitment.rs` is reduced to orchestration and assertions.

### Slice 5: CI Gates

Objective:

```text
Make equivalence checks standard PR gates with small always-on tests and
explicit perf/fuzz lanes.
```

Required gates:

- small real guest parity gate
- generated role crate full-field challenge parity gate
- Bolt proof tamper rejection gate
- SHA2-chain `2^16` core-vs-Bolt perf oracle
- SHA2-chain `2^20` core-vs-Bolt perf oracle
- optional longer fuzz/tamper lane

Acceptance gates:

- CI names are explicit and stable.
- Perf thresholds are configured in one place.
- Perf failures report stage-level timing, heavy computation spans, proof
  bytes, and peak RSS.
- Ignored/local perf tests can be run manually with one documented command.

### Slice 6: SHA2-Chain Perf Closure

Objective:

```text
Bring Jolt-on-Bolt SHA2-chain proving overhead within 20% of jolt-core while
keeping the equivalence crate a thin oracle/gate and keeping all semantic
changes in Bolt codegen, generated artifacts, jolt-kernels, or jolt-witness.
```

Current measured baseline:

```text
sha2-chain 2^16:
  prove_ms:     core=1987.685, bolt=2523.469, ratio=1.270x
  verify_ms:    core=89.220,   bolt=118.322,  ratio=1.326x
  proof_bytes:  core=80209,    bolt=111198,   ratio=1.386x
  peak_rss_mb:  core=209,      bolt=1248,     ratio=5.971x

sha2-chain 2^20:
  prove_ms:     core ~= 9.9s,  bolt ~= 22.6-22.8s, ratio ~= 2.29-2.30x
  verify_ms:    core ~= 0.11s, bolt ~= 0.13s,      ratio ~= 1.18-1.26x
  proof_bytes:  core=89041,    bolt=121398,        ratio=1.363x
  peak_rss_mb:  core ~= 1.84GB, bolt ~= 6.1-6.3GB, ratio ~= 3.3-3.4x
```

Perfetto/tracing overhead is not the cause of the 2^20 regression. Traced and
untraced runs landed in the same ratio band.

Current post-witness-fix 2^20 hotspot shape:

```text
bolt.stage6:       ~10.6s
  bytecode_read_raf:        ~4.3s
  instruction_ra_virtual:   ~2.9s
  booleanity:               ~2.6s
bolt.commitment:   ~5.3s
bolt.stage8:       ~3.8s
  joint_opening_hint:       ~1.4s
  dory_open:                ~1.9s
```

Current post-bytecode-sparse 2^20 shape:

```text
sha2-chain 2^16:
  prove_ms:     core=2020.866, bolt=2373.067, ratio=1.174x
  verify_ms:    core=90.403,   bolt=118.888,  ratio=1.315x
  proof_bytes:  core=80209,    bolt=111198,   ratio=1.386x
  peak_rss_mb:  core=207,      bolt=1109,     ratio=5.357x

sha2-chain 2^20:
  prove_ms:     core=9825.685, bolt=19381.338, ratio=1.973x
  verify_ms:    core=104.379,  bolt=132.872,   ratio=1.273x
  proof_bytes:  core=89041,    bolt=121398,    ratio=1.363x
  peak_rss_mb:  core=1839,     bolt=6094,      ratio=3.314x

Stage 6 per execution:
  bytecode_read_raf:        ~1.58s
  instruction_ra_virtual:   ~2.75-3.32s
  booleanity:               ~2.31-2.44s
```

The final dense bytecode read-RAF witness allocation removal was verified by
`jolt-witness`, focused Stage 6 kernel tests, kernel/witness clippy, and full
`jolt-equivalence`. Its 2^20 perf rerun was intentionally stopped during
laptop handoff, so the latest measured perf numbers above are from the sparse
kernel path before that allocation-only cleanup. Re-run the 2^20 perf gate
before judging any memory/setup movement from the allocation cleanup.

#### Handoff Plan Of Attack

Retained optimization:

- Bytecode read-RAF now uses sparse bytecode RA index chunks plus explicit
  chunk lengths. This is a core-equivalent direction and belongs in
  `jolt-witness`/`jolt-kernels`, not `jolt-equivalence`.
- Dense bytecode RA read-RAF materialization is removed from normal
  `stage6_witness_polynomials` output; callers that still provide dense chunks
  continue through the dense fallback.

Rejected experiment:

- Parallelizing indexed Booleanity cycle-state construction and dense binds
  made the 2^20 run slower (`prove_ms` about `25.5s`, ratio `2.13x`) and was
  reverted. Do not reapply that shape without a more targeted profile.

Next targets:

1. Port `instruction_ra_virtual` from generic dense product terms toward the
   core `RaPolynomial`/split-eq sum-of-products algorithm. This is now the
   largest Stage 6 bucket, roughly `2.7-3.3s` per Stage 6 execution.
2. Improve indexed Booleanity with an algorithmic change rather than broad
   parallel binding. Current bucket is roughly `2.3-2.4s` per Stage 6
   execution.
3. Re-run the post-allocation-cleanup 2^20 perf gate and record setup/prove/RSS
   movement. The interrupted command was stopped deliberately before commit.
4. After Stage 6 is no longer dominant, attack `bolt.commitment` (`~5.3s`) and
   Stage 8/Dory opening (`~3.8s`) with generated spans intact.
5. Tighten the perf thresholds from disabled diagnostic ratios to the actual
   target gates once 2^20 prove is within `1.20x`.

Primary target:

```text
sha2-chain 2^16 prove_ms ratio <= 1.20x
sha2-chain 2^20 prove_ms ratio <= 1.20x
```

Secondary targets:

```text
sha2-chain 2^16 verify_ms ratio <= 1.35x
sha2-chain 2^20 verify_ms ratio <= 1.35x
sha2-chain 2^16 proof_bytes ratio <= 1.40x
sha2-chain 2^20 proof_bytes ratio <= 1.40x
sha2-chain 2^16 peak_rss_mb <= 1.25GB
sha2-chain 2^20 peak_rss_mb <= 6.30GB, and should continue decreasing unless
a documented protocol change explains the tradeoff.
```

Required work:

- Port the remaining Stage 6 RA-heavy prover paths toward core-equivalent
  sparse/specialized algorithms instead of dense generic `DenseStage6State`
  products. The first targets are bytecode read-RAF, instruction RA virtual,
  and Booleanity.
- Keep semantic parity in `jolt-kernels`, `jolt-witness`, Bolt IR/codegen, or
  checked-in generated artifacts. Do not add perf-specific reconstruction or
  shortcuts to `jolt-equivalence`.
- Keep instrumentation code-generated when it concerns generated
  `jolt-prover`/`jolt-verifier` code. Direct runtime/kernel spans belong in
  the owning shared crate.
- Preserve the existing `bolt.prove`, `bolt.commitment`, `bolt.stage*`,
  `bolt.evaluate.*`, and verifier perf span contract so regressions remain
  attributable in CI.
- Treat `setup_ms`, commitment, and Stage 8/Dory opening work as follow-on
  targets after Stage 6 is no longer the dominant gap.

Acceptance gates:

- Both ignored SHA2-chain perf oracle tests pass with `prove_ms <= 1.20x`.
- The 2^20 perf report identifies no single uninstrumented Bolt prover bucket
  larger than `500ms`.
- `jolt-equivalence` contains no new stage-semantic reconstruction for the perf
  fix.
- Focused Stage 6 kernel tests pass.
- The full `jolt-equivalence` suite passes.
- Generated artifacts are regenerated from Bolt when generated code changes.

#### Active Goal Contract

Goal:

```text
Close the SHA2-chain Jolt-on-Bolt prover regression to <= 20% overhead versus
jolt-core at both 2^16 and 2^20, without moving Jolt semantics into the
equivalence harness.
```

Why this matters:

```text
The equivalence crate is supposed to be an oracle/gate. If Bolt is slower or
semantically different, the fix must land in Bolt-generated code,
jolt-kernels, jolt-witness, or shared runtime code. The harness may measure and
compare; it must not compensate for missing prover semantics.
```

Non-negotiable scope:

- Do not hand-edit generated `jolt-prover` or `jolt-verifier` behavior. Change
  Bolt/codegen/templates/shared crates and regenerate.
- Do not add perf-specific witness reconstruction, opening reconstruction, or
  stage-specific semantic shortcuts to `jolt-equivalence`.
- Do not weaken correctness, tamper, proof-size, verifier, or perf gates to
  make the goal pass.
- Keep generated prover/verifier spans code-generated when the work being
  measured is generated code.
- Keep runtime/kernel spans in the crate that owns the runtime/kernel work.
- Treat Perfetto overhead as negligible unless a repeat measurement proves
  otherwise; the current traced and untraced runs agree on the 2^20 gap.

Current blocking regression:

```text
2^16 prove: core 2.02s, Bolt 2.37s, 1.174x
2^20 prove: core about 9.8s, Bolt about 19.4s, 1.97x
```

The current evidence points at Stage 6 first, then commitment and Stage 8:

```text
bolt.stage6:     still dominant, with instruction_ra_virtual and booleanity largest
bolt.commitment: about 5.3s
bolt.stage8:     about 3.8s
```

Stage 6 subtargets, in priority order:

```text
instruction_ra_virtual about 2.7-3.3s per Stage 6 execution
booleanity             about 2.3-2.4s per Stage 6 execution
bytecode_read_raf      about 1.6s per Stage 6 execution after sparse-index port
```

Implementation direction:

- Prefer core-equivalent sparse/indexed paths over dense generic bind loops for
  RA-heavy Stage 6 work.
- Use `jolt-witness` for reusable prover-side witness/index/chunk material
  that is not inherently Bolt-codegen-specific.
- Use `jolt-kernels` for core-equivalent relation execution and specialized
  prover algorithms.
- Use Bolt artifact generation for generated prover/verifier APIs and spans.
- Keep `jolt-equivalence` changes limited to metric collection, reporting,
  thresholds, and thin public-artifact comparison.

Completion evidence required:

- Run the 2^16 and 2^20 SHA2-chain perf gates and report:
  `setup_ms`, `prove_ms`, `verify_ms`, `proof_bytes`, peak RSS, and top spans.
- Re-run any surprising perf result at least once before treating it as real.
- Show no uninstrumented Bolt prover bucket larger than `500ms` at 2^20.
- Run focused Stage 6 kernel tests.
- Run the full `jolt-equivalence` suite.
- Run `cargo fmt --check` and `git diff --check`.
- If generated artifacts changed, show the generator command used to regenerate
  them.

## Strict Per-Slice Gates

Every goal-mode slice must satisfy:

```text
cargo fmt --check
git diff --check
cargo check -p jolt-core --no-default-features --features minimal --offline
cargo check -p jolt-core --no-default-features --features minimal,challenge-254-bit --offline
```

Use the local Bolt MLIR/LLVM environment for equivalence tests:

```sh
env MLIR_SYS_220_PREFIX=/opt/homebrew/opt/llvm \
  PATH=/opt/homebrew/opt/llvm/bin:/Users/mgeorghiades/.cargo/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  BINDGEN_EXTRA_CLANG_ARGS=-isysroot/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  cargo nextest run -p jolt-equivalence --test generated_role_crates \
  --cargo-quiet --offline

env MLIR_SYS_220_PREFIX=/opt/homebrew/opt/llvm \
  PATH=/opt/homebrew/opt/llvm/bin:/Users/mgeorghiades/.cargo/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  BINDGEN_EXTRA_CLANG_ARGS=-isysroot/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  cargo nextest run -p jolt-equivalence --test bolt_commitment \
  --cargo-quiet --offline <focused test>
```

Before a slice is considered complete, also run the broadest feasible
equivalence gate:

```sh
env MLIR_SYS_220_PREFIX=/opt/homebrew/opt/llvm \
  PATH=/opt/homebrew/opt/llvm/bin:/Users/mgeorghiades/.cargo/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  BINDGEN_EXTRA_CLANG_ARGS=-isysroot/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  cargo nextest run -p jolt-equivalence --cargo-quiet --offline
```

Perf oracle gates are required before landing perf-sensitive slices:

```sh
env MLIR_SYS_220_PREFIX=/opt/homebrew/opt/llvm \
  PATH=/opt/homebrew/opt/llvm/bin:/Users/mgeorghiades/.cargo/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  BINDGEN_EXTRA_CLANG_ARGS=-isysroot/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  cargo nextest run -p jolt-equivalence --test bolt_perf --release \
  --cargo-quiet --offline --run-ignored only --no-capture \
  bolt_sha2_chain_2_16_core_vs_bolt_perf_oracle

env MLIR_SYS_220_PREFIX=/opt/homebrew/opt/llvm \
  PATH=/opt/homebrew/opt/llvm/bin:/Users/mgeorghiades/.cargo/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  BINDGEN_EXTRA_CLANG_ARGS=-isysroot/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  cargo nextest run -p jolt-equivalence --test bolt_perf --release \
  --cargo-quiet --offline --run-ignored only --no-capture \
  bolt_sha2_chain_2_20_core_vs_bolt_perf_oracle
```

## Regression Rules

A change fails review if it:

- increases `jolt-equivalence` semantic reconstruction
- hides a Bolt/core mismatch behind harness normalization
- adds prover witness or claim materialization to `jolt-equivalence` instead of
  `jolt-witness`/generated prover code
- weakens tamper rejection coverage
- removes a parity gate without replacing it with a stricter one
- adds new private-core dependencies
- makes generated verifier depend on prover-only crates or core internals
- makes perf gates less observable

## Definition Of Done

This goal is complete when:

- Jolt-on-Bolt passes real-data completeness parity against core.
- Generated Bolt verifier rejects representative mutated proofs.
- Stage 6/7 semantics live in Bolt/codegen/kernels, not in the harness.
- `jolt-equivalence` is mostly adapters, checks, tamper/fuzz, and perf gates.
- Small correctness gates are practical for PR CI.
- Larger SHA2-chain perf gates are documented, stable, and actionable.
- The harness is small enough that future compiler work uses it as a gate
  instead of treating it as a place to repair semantics.
