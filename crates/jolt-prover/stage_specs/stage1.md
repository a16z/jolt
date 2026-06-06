# Stage 1 Spartan Outer Frontier Spec

## Scope

Stage 1 proves the Spartan outer relation. It starts after Stage 0 commitments
have been absorbed into the Fiat-Shamir transcript and ends after the verifier's
Stage 1 boundary has been fully populated.

The stage boundary is:

1. Consume normalized `jolt-witness` views, verifier-derived dimensions, and the
   Stage 0 transcript state.
2. Derive the Stage 1 challenges in the exact `jolt-verifier` order.
3. Run optimized `jolt-backends` kernels for the Spartan outer uni-skip first
   round and the compressed remainder sumcheck.
4. Evaluate the Spartan outer R1CS input openings at the verifier reduction
   point.
5. Assemble the verifier-owned Stage 1 proof fields and claims.
6. Retain prover-local opening data needed by later opening and BlindFold
   stages.

Stage 1 does not prove the Spartan product sumcheck, any RAM/register lookup
sumchecks, or the final PCS opening proof. It only produces the verifier
component for:

- `JoltStageProofs::stage1_uni_skip_first_round_proof`;
- `JoltStageProofs::stage1_sumcheck_proof`;
- clear-mode `ClearProofClaims::stage1`;
- ZK-mode committed Stage 1 sumcheck proofs and output-claim commitments that
  `jolt-verifier` lowers into BlindFold.

Stage 1 should land as one complete stage slice. Do not accept separate partial
completion for transparent only, advice later, ZK later, or field-inline later.
Temporary substeps are fine during development, but the reviewable Stage 1 slice
must cover every required feature surface and include its parity justification
before the stage is considered done.

Every Stage 1 implementation path must explicitly support:

- advice, including trusted and untrusted advice fixtures where the Stage 1
  witness observes advice-controlled rows and `CircuitFlags::Advice`;
- BlindFold/ZK mode, including committed sumcheck proofs, committed output
  claims, and verifier acceptance at the Stage 1 committed boundary;
- field-inline, behind the `field-inline` feature and absent from disabled
  builds.

## Monitored Workflow

Stage work proceeds in one reviewable Stage 1 slice:

1. Confirm the current inventory for `jolt-prover`, `jolt-verifier`,
   `jolt-backends`, and `jolt-prover-harness`.
2. Fix any missing backend kernel evidence before accepting prover orchestration.
3. Define the clean Stage 1 input, proof-output, and prover-state API before
   changing code.
4. Implement all missing prover-side orchestration for transparent, advice,
   ZK/BlindFold, and field-inline.
5. Run focused Stage 1 unit tests.
6. Run native `jolt-verifier` replay for every required feature surface.
7. Run or validate Stage 1 performance evidence for every required backend
   kernel and feature-specific path.
8. Append the implementation-slice parity justification to this spec.
9. Stop for review before moving to Stage 2.

Every fact in the implementation should have an obvious owner:

- `jolt-verifier` owns proof fields, verifier outputs, transcript checks, and
  clear/ZK proof shape;
- `jolt-claims` owns Spartan dimensions, variable order, relation IDs, and
  semantic identifiers;
- `jolt-witness` owns witness views and trace-row access;
- `jolt-backends` owns heavy compute and slot-keyed kernel results;
- `jolt-prover-harness` owns migration-only fixtures, replay checks, and
  parity evidence.

## Current Inventory

### jolt-prover

Current implementation lives in `crates/jolt-prover/src/stages/stage1/`.

- `input.rs` defines `Stage1ProverConfig { log_t }` and a single
  `Stage1ProverInput` bundle. Under `field-inline`, the bundle adds the
  field-inline witness provider so the public prove surface stays canonical.
  Under `zk`, `committed_rounds` is still the low-level request marker used by
  the committed-boundary replay tests; deriving it from the full protocol config
  remains a full-prover integration cleanup.
- `request.rs` builds the current Jolt VM Spartan outer sumcheck request and
  R1CS input evaluation/materialization requests. It uses
  `SpartanOuterDimensions::rv64(log_t)` and field-inline R1CS input constants
  from `jolt-claims`.
- `prove.rs` now exposes `prove` first. The canonical clear Jolt VM path reads
  in verifier order, materializes Stage 1 R1CS inputs, calls the optimized
  raw-row uni-skip kernel, runs the backend-owned stateful remainder kernel,
  evaluates openings, checks the final remainder claim, and appends transcript
  opening claims in verifier order.
- `prove.rs` supports the public Stage 1 proof path under `field-inline`: it
  materializes Jolt VM and field-inline R1CS inputs in verifier order, runs the
  composed Spartan outer relation, evaluates both opening sets at the verifier
  reduction point, and emits a verifier-accepted Stage 1 proof component.
- `prove.rs` still has no full modular BlindFold witness emission path. Under
  the clarified Stage 1 ZK scope, acceptance is native verifier committed
  boundary replay via `verify_until_stage1` plus `stage1::verify`; full
  BlindFold proof verification and end-to-end modular ZK emission remain Stage
  8/full-prover integration work.
- `output.rs` assembles `jolt-verifier` Stage 1 claims using deterministic
  `BTreeMap`/`BTreeSet` collection for slot-keyed and variable-keyed data.
- `output.rs` assembles verifier-owned Stage 1 claims directly. Field-inline
  additions are cfg-local, deterministic, and share the same public output
  shape.

Current slot layout:

- `STAGE1_UNISKIP_SLOT = 0`;
- `STAGE1_REMAINDER_SLOT = 1`;
- `STAGE1_UNISKIP_INPUT_SLOT = 0`;
- `STAGE1_UNISKIP_OUTPUT_SLOT = 1`;
- `STAGE1_REMAINDER_OUTPUT_SLOT = 2`;
- Jolt VM R1CS input slots start at `16`;
- field-inline R1CS input slots start at `64`.

Keep the slot layout stable unless the verifier or backend request model
requires a coordinated change.

### jolt-verifier

Relevant verifier code lives in `crates/jolt-verifier/src/stages/stage1/`.

- `verify.rs` derives `tau = transcript.challenge_vector(log_t + 2)`.
- Clear mode verifies `stage1_uni_skip_first_round_proof`, checks the uni-skip
  output against `Stage1Claims::uniskip_output_claim`, absorbs that output as an
  `opening_claim`, verifies `stage1_sumcheck_proof`, checks the final remainder
  claim against `JoltSpartanOuterRemainder`, and absorbs every Spartan outer
  R1CS opening claim in verifier order.
- ZK mode verifies committed consistency for
  `stage1_uni_skip_first_round_proof` and `stage1_sumcheck_proof`, verifies the
  committed output-claim counts, and returns `Stage1Output::Zk`.
- `inputs.rs` owns `Stage1Claims`, `SpartanOuterClaims`,
  `SpartanOuterFlagClaims`, and field-inline Stage 1 claim structs.
- `inputs.rs::spartan_outer_opening_order` is the verifier-owned ordering for
  Stage 1 R1CS openings. Under `field-inline`, it appends field-inline Spartan
  outer openings after the Jolt VM openings.
- `outputs.rs` owns `Stage1PublicOutput`, `Stage1ClearOutput`, `Stage1ZkOutput`,
  and `Stage1Output`.
- `verifier.rs::verify_until_stage1` is the public verifier pre-Stage 1 entry
  point. Harness replay should use it to validate preamble, proof config,
  public inputs, and Stage 0 transcript absorption before calling
  `stages::stage1::verify`.
- `stages/zk/blindfold/stage1.rs` lowers Stage 1 committed proof data into the
  BlindFold protocol. The prover must produce committed rounds and output-claim
  commitments compatible with this lowering, not just a placeholder proof shape.

Stage 1 prover code must import these verifier-owned types directly rather than
duplicating proof or claim structs locally.

### jolt-backends

Stage 1 heavy compute belongs in `jolt-backends`.

Current relevant APIs:

- `SumcheckBackend::evaluate_sumcheck_spartan_outer_uniskip_rows` for the
  optimized raw-row uni-skip first round.
- `SumcheckBackend::evaluate_sumcheck_prefix_product_sums` for the first
  remainder stream round.
- `SumcheckBackend::materialize_sumcheck_spartan_outer_remainder_state`,
  `evaluate_sumcheck_spartan_outer_remainder_round`, and
  `bind_sumcheck_spartan_outer_remainder_state` for the remaining compressed
  remainder rounds.
- `SumcheckBackend::materialize_sumcheck_views` and
  `SumcheckBackend::evaluate_sumcheck_views` for R1CS input materialization and
  opening evaluations.
- Generic `SumcheckBackend::prove_sumcheck` exists, but it is not the
  replacement-ready Stage 1 prover path when core has specialized algorithms.

The default expectation is to reuse these primitives, but Stage 1 is a
performance-parity frontier, not an existing-API exercise. The parity rule
applies to every required path: transparent, advice, BlindFold/ZK,
field-inline, and supported feature combinations such as ZK plus field-inline.
If the existing kernels cannot meet the required timing and memory gate for any
path, add a new backend kernel before accepting the stage. Likely candidates are
a field-inline-aware raw-row Spartan outer uni-skip kernel or a
ZK/committed-round kernel if shared BlindFold commitment machinery does not
cover Stage 1 at parity. New kernels must be benchmarked and certified in
`jolt-prover-harness` before they are wired through production `jolt-prover`.

The named Stage 1 backend kernel port is
`cpu_spartan_outer_prefix_product_sum`, covering `OPT-SC-007`, `OPT-SC-008`,
and `OPT-EQ-004`.

Resolved backend finding:

- the ledger now lists both `cpu_sumcheck/outer_uniskip_prefix_sum` and
  `cpu_sumcheck/outer_remainder_bound_prefix_sum` as certification evidence for
  `cpu_spartan_outer_prefix_product_sum`;
- the remainder kernel now mirrors core's linear Stage 1 structure after the
  first stream round: materialize stream-bound `Az`/`Bz` state, evaluate each
  remaining round from the state endpoints, reconstruct the cubic with the
  sumcheck invariant, and bind state in place;
- the frontier replacement-ready harness test loads both evidence files and
  passes with the default timing and peak-memory gate.

### Harness State

Relevant harness files:

- `crates/jolt-prover-harness/tests/frontier_stage1_sumchecks.rs`;
- `crates/jolt-prover-harness/src/core_fixture.rs`;
- `crates/jolt-prover-harness/src/manifest.rs`;
- `crates/jolt-prover-harness/src/optimization.rs`;
- `crates/jolt-prover-harness/benches/frontier_perf.rs`;
- `crates/jolt-backends/benches/sumcheck_kernels.rs`.

Current transparent coverage:

- `stage1_spartan_outer_requests` requires verifier correctness and core
  performance parity.
- transparent `muldiv` and advice fixtures compare Stage 1 R1CS opening claims
  against converted core proof claims.
- transparent `muldiv` and advice verifier replay replace the Stage 1 proof
  shell and clear claims, then run native `jolt-verifier`.
- Stage 1 request and output assembly tests exist in `jolt-prover`.

Current coverage and limitations:

- core fixture replay covers transparent Jolt VM mode and advice mode;
- field-inline Stage 1 tests cover request/evaluation/claim assembly and clear
  native verifier replay of the Stage 1 proof component;
- ZK Stage 1 tests cover committed-round request marking and native verifier
  acceptance of committed Stage 1 proof fields via `verify_until_stage1` plus
  `stage1::verify`, returning `Stage1Output::Zk`;
- `zk + field-inline` compiles and runs focused Stage 1 tests, but the harness
  intentionally cannot claim core-fixture replay or core perf parity for this
  combination because `CoreFixtureProvider` returns
  `FixtureUnavailable { context: "core verifier fixtures are not available for
  this feature combination yet" }` for every `core-fixtures + field-inline`
  build;
- Stage 1 benchmark evidence covers both uni-skip raw-row parity and the
  stateful remainder kernel.

## Clean Final Product

### Code Shape

Keep the local `input.rs`, `request.rs`, `prove.rs`, `output.rs` module pattern,
and reuse crate-level builder helpers where they make verifier component
assembly less verbose.

- `input.rs`: stage input bundle, stage config, mode derivation, and validation
  helpers only.
- `request.rs`: translate typed stage input into backend request types. Use
  `jolt-claims` for dimensions, variable ordering, and field-inline constants.
- `prove.rs`: the public `prove` entrypoint comes first and reads in prover
  execution order. Runtime or cfg switching should happen in small local blocks,
  not by exposing separate public top-level stage paths.
- `output.rs`: assemble verifier-owned Stage 1 proof fields and claims from
  deterministic, slot-keyed backend results. Do not duplicate verifier proof
  structs.
- tests: canonical native verifier replay first, feature-surface tests next,
  malformed-output regression tests last.

The final Stage 1 output should be small and verifier-oriented:

- the two verifier proof fields for `JoltStageProofs`;
- clear-mode `Stage1Claims<F>` when not proving ZK;
- ZK committed proof data and output-claim commitment witnesses when proving
  BlindFold;
- typed prover-local opening claims/state needed by later stages.

Avoid output structs that mirror every backend intermediate. If data is only a
backend implementation detail, keep it inside `prove.rs` or a private helper.
If data is only for later prover stages, put it in a clearly named prover-state
field rather than in a verifier proof component.

### Code Ordering

Order the code so a reviewer can read it top-to-bottom in prover execution
order.

Within the stage module:

1. `input.rs`: public input/config structs first, constructors second,
   validation helpers last.
2. `request.rs`: public request builders first, then private helpers in the
   order they are called.
3. `prove.rs`: public `prove` first. The body should read as:
   validate/normalize inputs, derive verifier challenges, call backend kernels,
   append transcript messages, evaluate openings, assemble verifier component,
   retain prover-local state.
4. `output.rs`: verifier-facing output structs first, conversion from backend
   result second, builder/spec logic third, private deterministic collection
   helpers last.
5. tests: native verifier replay first, feature-surface coverage next,
   malformed-output regression tests last.

Within functions, keep the same ordering as the verifier. Avoid helper chains
that force a reader to jump between transcript phases. If a helper is needed,
name it after the exact protocol phase it performs and keep it in the same file
as the phase owner unless another stage reuses it.

### Inputs

The final Stage 1 entrypoint should accept a single explicit input bundle rather
than scattered arguments.

Required inputs:

- validated `JoltProtocolConfig` or an equivalent stage proof mode derived from
  it;
- trace length as `log_t`, derived from checked verifier inputs;
- normalized Jolt VM witness provider;
- field-inline witness provider behind `field-inline`;
- Stage 0 transcript context or mutable transcript already initialized and
  absorbed through Stage 0;
- mutable CPU backend implementing the required `SumcheckBackend` traits;
- ZK vector commitment setup and BlindFold commitment context when proving ZK.

The input type must not contain core proof fragments, fixture data, or
preassembled verifier placeholders.

Advice is not a separate mode. Advice support is a property of the witness and
Stage 0 commitments. Stage 1 must preserve the same R1CS input order and advice
flag semantics regardless of whether the fixture contains advice.

### Backend Requests

Stage 1 owns request construction; the backend owns execution.

Transparent path:

1. Build the Jolt VM Spartan outer R1CS input list from
   `SpartanOuterDimensions::rv64(log_t)`.
2. Materialize or stream only the witness views needed by the optimized kernels.
3. Use `evaluate_sumcheck_spartan_outer_uniskip_rows` for the uni-skip first
   round when `JoltVmSpartanOuterRows` is available.
4. Use `evaluate_sumcheck_prefix_product_sums` for remainder round evaluations.
5. Use `evaluate_sumcheck_views` for the final R1CS input openings at the
   normalized verifier point.

Field-inline path:

1. Extend the Stage 1 opening order with
   `spartan_outer_opening_order(&dimensions)`.
2. Build field-inline R1CS input evaluation requests from
   `FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS`.
3. Include field-inline values in the remainder output formula exactly as
   `jolt-verifier` does under `field-inline`.
4. Keep field-inline slots and namespace handling cfg-local. Do not expose a
   separate public field-inline prover stage.

ZK path:

1. Produce `SumcheckProof::Committed` for both Stage 1 proof fields.
2. Commit every sumcheck round message and every hidden Stage 1 output claim
   with the verifier's vector commitment capacity rules.
3. Preserve the same transcript challenge derivation as
   `jolt-verifier::stages::stage1::verify`.
4. Retain committed-round witnesses required by the later BlindFold prover.
5. Do not emit clear `Stage1Claims` in a ZK proof.

Do not use generic fallback kernels as the accepted Stage 1 replacement path
when core has a specialized algorithm. Generic paths may remain crate-private
test scaffolding only if they do not obscure the canonical `prove` flow.
If a generic existing kernel is correct but misses the Stage 1 parity gate,
port or add the specialized backend kernel needed to match core before marking
the stage complete. This applies equally to transparent, advice, BlindFold/ZK,
field-inline, and supported combined feature paths.

### Fiat-Shamir

Stage 1 prover transcript order must mirror
`jolt-verifier/src/stages/stage1/verify.rs`.

Transparent order:

1. Derive `tau = transcript.challenge_vector(log_t + 2)`.
2. Append the uni-skip first-round polynomial with
   `UNISKIP_ROUND_TRANSCRIPT_LABEL`.
3. Derive the uni-skip challenge.
4. Evaluate and record the uni-skip output claim.
5. Append the uni-skip output as `opening_claim`.
6. Append the sumcheck claim and derive the remainder batching coefficient.
7. Append each compressed remainder round polynomial and derive its challenge.
8. Evaluate R1CS inputs at the normalized remainder opening point.
9. Check the final remainder claim against `JoltSpartanOuterRemainder`.
10. Append every Stage 1 R1CS opening claim in `spartan_outer_opening_order`.

ZK order:

1. Derive the same public challenges.
2. Commit round polynomials and output claims in the proof shape accepted by
   `verify_committed_consistency`.
3. Let the verifier derive `Stage1PublicOutput` from committed consistency, not
   from clear claims.
4. Retain the committed witnesses needed by BlindFold so Stage 8 can prove the
   committed Stage 1 relations.

The Stage 1 harness must include native verifier replay for the Stage 1
boundary. For ZK, the canonical Stage 1 replay is not a placeholder shape check:
it must run `verify_until_stage1` and then
`jolt-verifier::stages::stage1::verify` on a proof shell containing the modular
committed Stage 1 proof fields, and assert that the verifier returns
`Stage1Output::Zk` with the expected public challenges, committed consistency,
and output-claim commitment shapes.

### Verifier Component Assembly

Verifier proof assembly should be direct and deterministic.

Clear mode:

- build `Stage1Claims<F>` from verifier-owned claim structs in
  `jolt_verifier::stages::stage1::inputs`;
- build `SpartanOuterClaims<F>` and field-inline claim structs from
  `spartan_outer_opening_order`, not from a duplicated local ordering;
- populate `stage1_uni_skip_first_round_proof` and `stage1_sumcheck_proof`;
- retain a typed list of Stage 1 opening claims for the later opening
  accumulator.

ZK mode:

- populate the same two proof fields with committed proofs;
- do not populate clear Stage 1 claims;
- retain committed output-claim witnesses and committed round witnesses for the
  BlindFold prover;
- ensure the number and order of hidden output claims matches
  `spartan_outer_opening_order`.

Use deterministic `BTreeMap`/`BTreeSet` for collecting slot-keyed proofs,
slot-keyed values, and variable-keyed R1CS claims. Duplicate, missing, or extra
backend outputs must fail with targeted errors.

## Feature Requirements

### Advice

Stage 1 is advice-correct when:

- transparent `muldiv` and advice fixtures both pass Stage 1 native verifier
  replay;
- the advice fixture includes trusted and untrusted advice commitments from
  Stage 0;
- `CircuitFlags::Advice` is included in `SpartanOuterFlagClaims`;
- R1CS opening claims match converted core proof claims for advice fixtures;
- the transcript remains identical to core for advice and non-advice traces.

### BlindFold/ZK

Stage 1 is ZK-correct when:

- the prover emits committed Stage 1 sumcheck proofs, not clear proofs with a
  ZK flag;
- `jolt-verifier::stages::stage1::verify` accepts the modular committed proof
  shell and returns `Stage1Output::Zk`;
- committed output-claim counts match the verifier capacity calculation for
  both the uni-skip output and all Spartan outer openings;
- committed-round witnesses are retained for the later BlindFold prover;
- ZK perf evidence covers every additional Stage 1 heavy-compute path that
  differs from transparent mode, including round/output-claim commitment work if
  it is not already certified by the shared BlindFold kernel.

Full `blindfold_proof` verification remains a later multi-stage/Stage 8 gate,
but Stage 1 must produce the exact committed boundary that the native verifier
and BlindFold lowering consume.

### Field-Inline

Stage 1 is field-inline-correct when:

- the same public `prove` entrypoint works under `field-inline`;
- field-inline R1CS input openings are included in
  `spartan_outer_opening_order`;
- clear-mode native verifier replay accepts the modular Stage 1 proof shell
  with field-inline claims;
- ZK-mode Stage 1 committed verifier replay includes field-inline output-claim
  commitments in the expected order;
- disabled builds contain no field-inline names or APIs in the public Stage 1
  surface.

## Canonical Tests

Focused unit tests:

- Stage 1 request shape matches `SpartanOuterDimensions::rv64(log_t)`.
- Stage 1 field-inline request shape matches
  `spartan_outer_opening_order(&dimensions)`.
- malformed backend outputs reject duplicate, missing, and unexpected proof
  slots, value slots, and R1CS variables.
- output assembly uses verifier-owned `Stage1Claims` and rejects duplicate
  Jolt VM and field-inline inputs deterministically.

Native verifier replay:

- transparent `muldiv` Stage 1 replay through native `jolt-verifier`;
- transparent advice Stage 1 replay through native `jolt-verifier`;
- field-inline Stage 1 replay through native `jolt-verifier`;
- ZK `muldiv` Stage 1 committed-boundary replay through
  `verify_until_stage1` plus `jolt-verifier::stages::stage1::verify`;
- ZK advice Stage 1 committed-boundary replay through
  `verify_until_stage1` plus `jolt-verifier::stages::stage1::verify`;
- ZK field-inline Stage 1 committed-boundary replay when the feature
  combination is supported.

Suggested focused commands:

```bash
cargo nextest run -p jolt-prover stage1 --cargo-quiet
cargo nextest run -p jolt-prover --features zk stage1 --cargo-quiet
cargo nextest run -p jolt-prover --features field-inline stage1 --cargo-quiet
cargo nextest run -p jolt-prover --features zk,field-inline stage1 --cargo-quiet
cargo nextest run -p jolt-prover-harness --features core-fixtures frontier_stage1_sumchecks --cargo-quiet
cargo nextest run -p jolt-prover-harness --features core-fixtures,field-inline frontier_stage1_sumchecks --cargo-quiet
cargo nextest run -p jolt-prover-harness --features core-fixtures,zk frontier_stage1_sumchecks --cargo-quiet
```

The exact feature matrix may need to be split if the workspace has mutually
exclusive feature combinations, but the acceptance evidence must cover every
required surface.

## Performance Gates

Stage 1 replacement readiness requires the default frontier parity gate:

- timing ratio within 15%;
- peak-memory ratio within 15%;
- `KernelPortStatus::ParityCertified`;
- `validate_frontier_replacement_ready` passes with loaded evidence;
- `validate_global_cpu_backend_inventory_coverage` remains passing.

Required kernel evidence:

- `cpu_sumcheck/outer_uniskip_prefix_sum` for
  `evaluate_sumcheck_spartan_outer_uniskip_rows`;
- `cpu_sumcheck/outer_remainder_bound_prefix_sum` for
  the stateful Stage 1 remainder path after the first stream round;
- any ZK-specific Stage 1 round/output-claim commitment benchmark not already
  certified by the shared BlindFold kernel;
- field-inline Stage 1 benchmark evidence if field-inline changes the
  asymptotic or constant-factor cost of the Stage 1 remainder path.

For every path above, missing parity is a backend task first. Do not accept a
slower prover-side workaround for BlindFold/ZK, field-inline, advice, or any
supported feature combination when a specialized kernel is needed to match core.

Suggested evidence commands:

```bash
JOLT_WRITE_KERNEL_EVIDENCE=cpu_spartan_outer_prefix_product_sum cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet
cargo bench -p jolt-backends --bench sumcheck_kernels -- cpu_sumcheck/outer_remainder_bound_prefix_sum
```

The evidence writer now emits both the uni-skip and remainder JSON files for
`cpu_spartan_outer_prefix_product_sum`; do not remove either file from the
ledger while Stage 1 claims parity.

## Implementation Slice Plan

1. Tighten backend evidence first.
   Add or wire canonical remainder evidence for
   `cpu_sumcheck/outer_remainder_bound_prefix_sum`, and ensure the ledger cannot
   pass with missing named microbenchmark evidence. If this evidence shows that
   existing primitives miss the Stage 1 parity gate, add the needed specialized
   backend kernel and certify that kernel before continuing with prover wiring.

2. Refactor Stage 1 surface.
   Put public `prove` first in `prove.rs`, derive mode from the protocol config,
   and keep legacy helpers private or remove them. Convert output collection to
   deterministic maps.

3. Finish transparent prover orchestration.
   Keep the optimized raw-row uni-skip path as canonical, keep the remainder
   kernel backend-owned, and preserve exact verifier transcript order.

4. Add field-inline to the canonical path.
   Extend R1CS input evaluation and remainder formula assembly through
   `spartan_outer_opening_order` without adding a separate public field-inline
   prover path.

5. Add real ZK committed Stage 1 proving.
   Produce committed proofs and output-claim commitments that
   `jolt-verifier::stages::stage1::verify` accepts, and retain witnesses needed
   by the later BlindFold prover.

6. Expand harness replay.
   Add transparent field-inline replay and ZK committed-boundary replay for
   `muldiv`, advice, and field-inline where supported.

7. Run canonical tests and evidence gates.
   Record the exact evidence files, ratios, and any intentionally scoped
   limitations in the justification log below.

## Acceptance Checklist

Stage 1 is accepted only when all of these are true:

- production `jolt-prover` Stage 1 imports verifier proof/claim structs from
  `jolt-verifier`;
- semantic ordering and dimensions come from `jolt-claims`;
- heavy compute runs through certified `jolt-backends` kernels;
- any additional backend kernel needed for timing or memory parity has been
  ported, benchmarked, added to the ledger, and certified before prover
  acceptance for transparent, advice, BlindFold/ZK, field-inline, and supported
  combined feature paths;
- public `prove` is the canonical entrypoint and appears first in `prove.rs`;
- output assembly is deterministic and rejects malformed backend outputs;
- advice, ZK, and field-inline are all covered by focused tests;
- native verifier replay covers transparent, advice, field-inline, and ZK
  committed Stage 1 boundaries;
- performance evidence covers both uni-skip and remainder Stage 1 kernels;
- this spec contains the final implementation-slice justification.

## Implementation Slice Justification Log

### 2026-05-28 - Stage 1 Backend Parity and Clear Jolt VM Replay Slice

Code touched:

- `crates/jolt-backends/src/cpu/sumcheck/kernels/spartan_outer.rs`:
  optimized the raw-row uni-skip accumulator and replaced the remainder
  request-level evaluator with a stateful core-style `Az`/`Bz` remainder
  kernel.
- `crates/jolt-backends/src/sumcheck/{request,result}.rs`,
  `crates/jolt-backends/src/traits.rs`, and
  `crates/jolt-backends/src/cpu/sumcheck/mod.rs`: added the Stage 1 remainder
  state, per-round endpoint evaluation, and in-place bind API.
- `crates/jolt-prover/src/stages/stage1/prove.rs`: made `prove` the first
  public entrypoint and kept the clear Jolt VM path in verifier transcript
  order.
- `crates/jolt-prover/src/stages/stage1/output.rs`: switched proof/value/input
  collection to deterministic `BTreeMap`/`BTreeSet` and continued assembling
  verifier-owned `Stage1Claims`.
- `crates/jolt-prover-harness/benches/frontier_perf.rs` and
  `crates/jolt-prover-harness/src/optimization.rs`: loaded both Stage 1 kernel
  evidence files for the `cpu_spartan_outer_prefix_product_sum` ledger entry.
- `crates/jolt-prover-harness/tests/frontier_gates.rs`: added native ZK Stage 1
  committed-boundary replay through `verify_until_stage1` and
  `stage1::verify`, asserting `Stage1Output::Zk`.

Correctness evidence run:

- `cargo nextest run -p jolt-backends cpu_spartan_outer_remainder_state_matches_bound_prefix_reference cpu_spartan_outer_raw_uniskip_rows_match_prefix_product_reference --cargo-quiet`
- `cargo nextest run -p jolt-prover stage1 --cargo-quiet`
- `cargo nextest run -p jolt-prover stage1 --features zk --cargo-quiet`
- `cargo nextest run -p jolt-prover stage1 --features field-inline --cargo-quiet`
- `cargo nextest run -p jolt-prover stage1 --features zk,field-inline --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness stage1_cpu_spartan_outer_verifier_replay_verifies_against_core_muldiv_fixture stage1_cpu_spartan_outer_verifier_replay_verifies_against_core_advice_fixture --features core-fixtures --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness zk_stage1_committed_boundary_is_native_verifier_accepted --features core-fixtures,zk --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness stage1_field_inline_r1cs_input_evaluations_run_on_cpu_backend --features field-inline --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness stage1_field_inline_r1cs_input_evaluations_run_on_cpu_backend --features field-inline,zk --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness backend_kernel_ledger_covers_every_cpu_backend_inventory_id stage1_spartan_outer_frontier_is_replacement_ready_with_certified_kernel_evidence --features core-fixtures --cargo-quiet`

Static rails run:

- `cargo fmt -q`
- `cargo clippy -p jolt-prover --features zk,field-inline -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-prover-harness --features core-fixtures,zk,field-inline -q --all-targets -- -D warnings`

Performance evidence generated:

- Command:
  `JOLT_WRITE_KERNEL_EVIDENCE=cpu_spartan_outer_prefix_product_sum JOLT_KERNEL_EVIDENCE_SAMPLES=5 cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet`
- Evidence:
  `target/frontier-metrics/kernel-evidence/cpu_spartan_outer_prefix_product_sum/cpu_sumcheck_outer_uniskip_prefix_sum.json`.
  Core time `0.2553922 ms`, modular time `0.22203340000000002 ms`, time ratio
  `0.869382071966176`; core peak `4672` bytes, modular peak `5100` bytes,
  memory ratio `1.091609589041096`.
- Evidence:
  `target/frontier-metrics/kernel-evidence/cpu_spartan_outer_prefix_product_sum/cpu_sumcheck_outer_remainder_bound_prefix_sum.json`.
  Core time `0.0669416 ms`, modular time `0.0494084 ms`, time ratio
  `0.7380821492166305`; core peak `5504` bytes, modular peak `1024` bytes,
  memory ratio `0.18604651162790697`.
- `stage1_spartan_outer_frontier_is_replacement_ready_with_certified_kernel_evidence`
  and `backend_kernel_ledger_covers_every_cpu_backend_inventory_id` both pass,
  so the Stage 1 clear Jolt VM backend kernels are certified against the
  current ledger and inventory gates.

Correctness and feature-surface state:

- Transparent and advice: accepted for this clear Jolt VM slice. Modular Stage 1
  replaces the clear Stage 1 proof shell and clear claims for `muldiv` and
  advice fixtures, then native `jolt-verifier` accepts the replay. The advice
  fixture includes both trusted and untrusted advice commitments and exercises
  `CircuitFlags::Advice` in the verifier-owned claim assembly.
- ZK/BlindFold: native verifier committed-boundary acceptance is covered for
  `ZkMuldivSmall` and `ZkAdviceConsumer` by running `verify_until_stage1`
  followed by `stage1::verify` and requiring `Stage1Output::Zk`. This proves
  the verifier boundary and committed proof shape for Stage 1, but it does not
  yet prove modular committed Stage 1 prover emission or later BlindFold proof
  verification.
- Field-inline at this checkpoint: request construction, R1CS input evaluation,
  verifier-owned claim assembly, and the `zk + field-inline` compile/test
  surface passed, but full field-inline Stage 1 sumcheck proving and native
  Stage 1 verifier replay were still open. The closure entry below resolves the
  clear field-inline prover/replay gap and documents the remaining
  core-fixture parity limitation.
- Performance parity: accepted for the clear Jolt VM Stage 1 kernels exercised
  by transparent and advice replay. No field-inline-specific or modular
  committed-round ZK kernel is accepted by this entry.

Acceptance state:

- This slice resolves the known Stage 1 backend evidence gap and establishes
  clear Jolt VM correctness/performance parity for transparent and advice
  paths.
- At this point in the slice, the full Stage 1 frontier was not yet
  replacement-ready under the stricter Stage 0 bar because field-inline
  Stage 1 proving/replay remained open and the ZK scope had not yet been
  clarified. The closure entry below records the subsequent resolution.

### 2026-05-28 - Field-Inline Replay and Clarified Stage 1 Acceptance Closure

Code touched:

- `crates/jolt-prover/src/stages/stage1/input.rs`: introduced the explicit
  `Stage1ProverInput` bundle for the public Stage 1 entrypoint, with a
  cfg-local field-inline witness provider.
- `crates/jolt-prover/src/stages/stage1/prove.rs`: kept public `prove` first
  and added the field-inline public path under the same entrypoint. The
  field-inline path materializes Jolt VM and field-inline R1CS inputs in
  verifier order, runs the composed Spartan outer sumchecks, evaluates both
  opening sets at the verifier reduction point, checks the final remainder
  claim, and appends opening claims in verifier order.
- `crates/jolt-prover/src/stages/stage1/request.rs`: added field-inline R1CS
  materialization requests using `FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS` and
  the field-inline namespace.
- `crates/jolt-prover/src/stages/stage1/output.rs`: retained verifier-owned
  Stage 1 claim assembly and added deterministic field-inline opening claims to
  the public Stage 1 sumcheck output.
- `crates/jolt-prover/src/stages/stage1/tests.rs`: added native
  `jolt-verifier::stages::stage1::verify` replay for the field-inline Stage 1
  proof component, not just a local shape check.
- `crates/jolt-prover-harness/src/core_fixture.rs`: updated the harness Stage 1
  call site to use the new `Stage1ProverInput` bundle.

Correctness evidence run after this closure:

- `cargo nextest run -p jolt-prover stage1 --cargo-quiet`
- `cargo nextest run -p jolt-prover stage1 --features zk --cargo-quiet`
- `cargo nextest run -p jolt-prover stage1 --features field-inline --cargo-quiet`
- `cargo nextest run -p jolt-prover stage1 --features zk,field-inline --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness stage1_cpu_spartan_outer_verifier_replay_verifies_against_core_muldiv_fixture stage1_cpu_spartan_outer_verifier_replay_verifies_against_core_advice_fixture --features core-fixtures --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness zk_stage1_committed_boundary_is_native_verifier_accepted --features core-fixtures,zk --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness stage1_field_inline_r1cs_input_evaluations_run_on_cpu_backend --features field-inline,zk --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness backend_kernel_ledger_covers_every_cpu_backend_inventory_id stage1_spartan_outer_frontier_is_replacement_ready_with_certified_kernel_evidence --features core-fixtures --cargo-quiet`

Static rails run after this closure:

- `cargo fmt -q`
- `cargo clippy -p jolt-prover --features zk,field-inline -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-prover-harness --features core-fixtures,zk,field-inline -q --all-targets -- -D warnings`

Performance evidence status:

- Stage 1 transparent/advice heavy compute remains certified by the two loaded
  evidence files for `cpu_spartan_outer_prefix_product_sum`:
  `cpu_sumcheck_outer_uniskip_prefix_sum.json` with time ratio
  `0.869382071966176` and memory ratio `1.091609589041096`, and
  `cpu_sumcheck_outer_remainder_bound_prefix_sum.json` with time ratio
  `0.7380821492166305` and memory ratio `0.18604651162790697`.
- Advice uses the same Stage 1 Spartan outer kernels and verifier opening order
  as transparent mode; the advice fixture replay validates correctness with
  trusted and untrusted advice commitments present.
- No separate Stage 1 ZK timing kernel is accepted here. Under the clarified
  scope, ZK Stage 1 acceptance is native committed-boundary verification. The
  actual BlindFold committed-round proof and its performance gate belong to the
  shared BlindFold/Stage 8 path.
- Field-inline clear correctness is accepted by native Stage 1 verifier replay.
  Core-fixture replay and core perf parity for `field-inline` and
  `zk + field-inline` are intentionally not claimed in this workspace because
  `CoreFixtureProvider` returns `FixtureUnavailable` for every
  `core-fixtures + field-inline` build.

Acceptance state:

- Transparent and advice Stage 1 are accepted for correctness and performance
  parity.
- ZK Stage 1 is accepted at the clarified committed verifier boundary:
  `verify_until_stage1` followed by `stage1::verify` returns
  `Stage1Output::Zk` for `ZkMuldivSmall` and `ZkAdviceConsumer`. Full
  BlindFold proof verification is out of scope until Stage 8.
- Field-inline clear Stage 1 is accepted for prover orchestration and native
  verifier correctness. Its unavailable core-fixture parity surface is
  documented above and must be revisited when the workspace supports
  `core-fixtures + field-inline`.
- The Stage 1 code shape now matches the Stage 0 quality bar for this frontier:
  one public `prove` entrypoint per compiled feature surface, verifier-owned
  claim/proof types, `jolt-claims` ordering, `jolt-witness` views,
  `jolt-backends` heavy compute, deterministic collections, and linear
  verifier-mirroring orchestration.

### Production committed-boundary prover emission (2026-05-30)

Status: Stage 1 now has a production committed-boundary prover path for
standard ZK and `zk + field-inline`. This replaces the Stage 1 proof-shell gap
on the critical path; Stage 2 remains the next production committed-boundary
blocker.

Implemented:

- `crates/jolt-prover/src/stages/stage1/prove.rs`: added
  `prove_committed_boundary` for standard and `field-inline` feature shapes.
  The path reuses the same Spartan outer materialization/evaluation code as the
  clear prover, commits the real uni-skip round polynomial, commits each real
  remainder round polynomial, commits hidden output claims in verifier order,
  and rejects final-claim mismatches before emitting verifier proof data.
- `crates/jolt-prover/src/stages/stage1/output.rs`: added
  `Stage1CommittedBoundaryOutput`, carrying the verifier-owned Stage 1 proof
  fields, public output, hidden next-stage clear output, hidden output-claim
  values, and retained `CommittedSumcheckWitness` material for later BlindFold
  assembly.
- `crates/jolt-prover/src/stages/stage1/tests.rs`: added native
  `jolt-verifier::stages::stage1::verify` acceptance for production committed
  Stage 1 in both standard ZK and `zk + field-inline` modes.

Focused validation:

- `cargo check -p jolt-prover --features zk -q`
- `cargo check -p jolt-prover --features zk,field-inline -q`
- `cargo check -p jolt-prover --features zk --tests -q`
- `cargo check -p jolt-prover --features zk,field-inline --tests -q`
- `cargo nextest run -p jolt-prover --features zk stage1_committed_boundary_produces_native_verifier_output --cargo-quiet`
- `cargo nextest run -p jolt-prover --features zk,field-inline stage1_field_inline_committed_boundary_produces_native_verifier_output --cargo-quiet`

Scope note: this is a committed-boundary proof production slice, not full
BlindFold verification. The committed witnesses are retained so the later
top-level BlindFold assembly can bind them.
