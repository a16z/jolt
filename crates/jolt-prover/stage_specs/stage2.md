# Stage 2 Spartan Product and Regular Batch Frontier Spec

## Scope

Stage 2 proves the Spartan product virtualization first round and the regular
batch that follows it. It starts after Stage 1 has produced `Stage1Output` and
the verifier transcript is positioned at the Stage 2 boundary. It ends after
the verifier-owned Stage 2 proof fields, clear claims or committed ZK shapes,
and downstream public outputs are assembled.

The stage boundary is:

1. Consume the Stage 1 output and normalized `jolt-witness` views.
2. Derive Stage 2 Fiat-Shamir challenges in the exact `jolt-verifier` order.
3. Prove the Spartan product uni-skip first round.
4. Prove the regular Stage 2 batched sumcheck over:
   - RAM read-write checking;
   - Spartan product remainder;
   - instruction claim reduction;
   - field-register claim reduction under `field-inline`;
   - RAM RAF evaluation;
   - RAM output check.
5. Evaluate every Stage 2 output opening at the verifier-derived points.
6. Assemble the verifier-owned Stage 2 proof fields and claims.
7. Return a typed Stage 2 prover state/output that later stages can consume
   without recomputing verifier reductions.

Stage 2 produces the verifier component for:

- `JoltStageProofs::stage2_uni_skip_first_round_proof`;
- `JoltStageProofs::stage2_sumcheck_proof`;
- clear-mode `ClearProofClaims::stage2`;
- ZK-mode committed Stage 2 sumcheck proofs and output-claim commitments that
  `jolt-verifier` lowers into BlindFold;
- clear-mode `Stage2ClearOutput` data needed by Stage 3 and later claim
  reductions;
- ZK-mode public and committed-consistency data needed by the later BlindFold
  prover.

Stage 2 does not prove Stage 3 shift/instruction-input sumchecks, later
RAM/register reductions, or the final PCS opening proof. It must, however,
produce all Stage 2 outputs in the exact order later stages and the verifier
expect.

Stage 2 should land as one complete stage slice. Subfrontier work is useful for
review and benchmarking, but Stage 2 is not accepted as replacement-ready until
transparent, advice, BlindFold/ZK committed-boundary, field-inline, and
supported combined feature paths have their documented correctness and
performance evidence.

Every Stage 2 implementation path must explicitly support:

- advice, including trusted and untrusted advice fixtures where Stage 2
  consumes Stage 1 advice-influenced openings;
- BlindFold/ZK mode, including committed product uni-skip, committed regular
  batch, committed output-claim shapes, and verifier acceptance at the Stage 2
  committed boundary;
- field-inline, behind the `field-inline` feature and absent from disabled
  builds;
- `zk + field-inline`, unless the workspace proves the combination is
  intentionally unsupported. If unsupported, document the concrete reason in
  this spec's justification log.

## Monitored Workflow

Stage work proceeds in one reviewable Stage 2 slice:

1. Confirm current inventory for `jolt-prover`, `jolt-verifier`,
   `jolt-backends`, `jolt-witness`, and `jolt-prover-harness`.
2. Tighten backend evidence before accepting prover orchestration.
3. Define the canonical Stage 2 input/output/prover-state API.
4. Refactor toward one public `prove` entrypoint, ordered first in `prove.rs`.
5. Implement transparent and advice paths through native Stage 2 verifier
   replay.
6. Add field-inline support to the same canonical path.
7. Add ZK committed-boundary proof assembly and verifier replay.
8. Run focused tests, real fixture replay, and kernel evidence gates.
9. Append the final correctness and performance parity justification to this
   spec.
10. Stop for review before moving to Stage 3.

Every fact in the implementation should have a clear owner:

- `jolt-verifier` owns proof fields, verifier outputs, transcript checks, clear
  claims, and committed proof shapes;
- `jolt-claims` owns relation IDs, dimensions, formula semantics, opening IDs,
  and challenge/public identifiers;
- `jolt-witness` owns trace-backed witness views and primitive row views;
- `jolt-backends` owns heavy compute and slot-keyed kernel results;
- `jolt-prover-harness` owns migration-only fixtures, verifier replay, and
  performance evidence.

Code should be ordered like the verifier and like the prover execution:

1. public `prove`;
2. small input/output assembly helpers;
3. product uni-skip helper code;
4. regular batch helper code;
5. output-opening evaluators;
6. validation and deterministic collection helpers.

Avoid hiding the Stage 2 protocol behind generic abstractions while the stage is
being established. Use small private helpers only when they remove real
duplication or isolate low-level arithmetic. The public path should remain
linear enough to audit against `jolt-verifier/src/stages/stage2/verify.rs`.

## Current Inventory

### jolt-verifier

Relevant verifier code lives in `crates/jolt-verifier/src/stages/stage2/`.

- `verify.rs` first validates whether it received clear or ZK Stage 1
  dependencies through `stage2::inputs::Deps`.
- `verify_product_uniskip` derives `tau_low` from
  `Stage1PublicOutput::remainder_challenges[1..]`, reverses it, derives
  `tau_high = transcript.challenge()`, then verifies
  `stage2_uni_skip_first_round_proof`.
- In clear mode, product uni-skip computes the input claim from Stage 1
  openings and centered Lagrange weights, verifies a clear uni-skip proof,
  checks `Stage2Claims::product_uniskip_output_claim`, and appends that output
  as `opening_claim`.
- In ZK mode, product uni-skip runs `verify_committed_consistency` over
  `stage2_uni_skip_first_round_proof`, verifies one committed output claim, and
  returns the committed consistency plus public challenge.
- `verify_regular_batch` derives:
  - `ram_read_write_gamma = transcript.challenge_scalar()`;
  - `instruction_gamma = transcript.challenge_scalar()`;
  - `field_registers_claim_reduction_gamma = transcript.challenge_scalar()`
    under `field-inline`;
  - `output_address_challenges = transcript.challenge()` repeated `log_k`
    times.
- Clear regular batch verifies `stage2_sumcheck_proof` as a compressed batched
  Boolean sumcheck. The claim order is:
  1. RAM read-write checking;
  2. Spartan product remainder;
  3. instruction claim reduction;
  4. field-register claim reduction under `field-inline`;
  5. RAM RAF evaluation;
  6. RAM output check.
- Clear regular batch reconstructs every expected output claim from
  verifier-owned formulas, public inputs, Stage 1 outputs, and Stage 2 opening
  claims. It then appends opening claims to the transcript in verifier order:
  1. RAM read-write: `val`, `ra`, `inc`;
  2. product remainder: `left_instruction_input`,
     `right_instruction_input`, `jump_flag`, `write_lookup_output_to_rd`,
     `lookup_output`, `branch_flag`, `next_is_noop`, `virtual_instruction`;
  3. field-inline product openings under `field-inline`: `field_rs1_value`,
     `field_rs2_value`, `field_rd_value`;
  4. instruction claim reduction: `left_lookup_operand`,
     `right_lookup_operand`;
  5. RAM RAF evaluation output;
  6. RAM output check output.
- ZK regular batch builds the same statement order, runs
  `BatchedSumcheckVerifier::verify_committed_consistency`, verifies committed
  output-claim count, and returns the public opening points needed by later
  BlindFold stages.
- ZK output-claim counts are:
  - product uni-skip: `1`;
  - regular batch without field-inline: `15`;
  - regular batch with field-inline: `18`.
- `inputs.rs` owns `Stage2Claims`, `Stage2BatchOutputOpeningClaims`,
  `RamReadWriteOutputOpeningClaims`, `ProductRemainderOutputOpeningClaims`,
  field-inline Stage 2 output-opening structs, and
  `InstructionClaimReductionOutputOpeningClaims`.
- `outputs.rs` owns `Stage2PublicOutput`, `Stage2ClearOutput`,
  `Stage2ZkOutput`, `Stage2Output`, `VerifiedProductUniSkip`,
  `VerifiedStage2Batch`, and `VerifiedStage2Sumcheck`.

Stage 2 prover code must import these verifier-owned types directly rather
than duplicating claim or output structs locally.

### jolt-prover

Current implementation lives in `crates/jolt-prover/src/stages/stage2/`.

- `prove.rs` now exposes a canonical clear `prove` entrypoint in both standard
  and `field-inline` builds.
- `prove.rs` now exposes a production `prove_committed_boundary` entrypoint in
  both `zk` and `zk + field-inline` builds.
- Clear product uni-skip mirrors the verifier flow and uses the optimized
  product uni-skip backend row path.
- Clear regular batch emits `stage2_sumcheck_proof`, evaluates all Stage 2
  output openings, recomputes the expected final claim, appends clear opening
  claims in verifier order, and returns `Stage2ClearOutput`.
- Committed product uni-skip commits the real first-round polynomial, commits
  one hidden output claim, and retains a `CommittedSumcheckWitness`.
- Committed regular batch commits the real batched round polynomials without
  clear input-claim transcript appends, commits hidden output claims in verifier
  order, and retains a `CommittedSumcheckWitness`.
- Field-inline extends product uni-skip, product remainder, field-register
  claim reduction, output openings, and committed output-claim rows under cfg.
- `Stage2CommittedBoundaryOutput` carries the verifier-visible committed proof
  fields, public output, hidden verifier output for downstream prover stages,
  output-claim values, and retained committed witnesses for BlindFold.
- Focused native-verifier tests now cover Stage 2 committed-boundary acceptance
  for standard ZK and `zk + field-inline` using production Stage 1 and Stage 2
  committed provers.

Current slot layout:

- product uni-skip proof slot: `STAGE2_PRODUCT_UNISKIP_SLOT = 0`;
- product uni-skip input/output value slots: `0` and `1`;
- product-remainder opening slots start at `16`;
- instruction-claim opening slots start at `32`;
- RAM read-write opening slots start at `48`;
- RAM terminal opening slots start at `64`.

Keep this layout stable unless a verifier or backend request-model change
requires a coordinated update. If new regular-batch proof/value slots are
introduced, reserve non-overlapping ranges and document them here.

### jolt-backends

Stage 2 heavy compute belongs in `jolt-backends`.

Current relevant APIs and kernels:

- `SumcheckBackend::evaluate_sumcheck_product_uniskip_rows` for optimized
  product uni-skip row evaluation;
- `SumcheckBackend::evaluate_sumcheck_row_products` and related row-product
  primitives used by product-style kernels;
- `SumcheckBackend::evaluate_sumcheck_views` for output-opening evaluations;
- generic `SumcheckBackend::prove_sumcheck`, which can remain test/reference
  scaffolding but must not be accepted as the replacement path if core has a
  specialized algorithm;
- `cpu_materialized_opening_evaluations`, a parity-certified opening kernel
  used by Stage 2 opening subfrontiers;
- `cpu_stage2_regular_batch_input_claims`, a parity-certified input-claim
  construction surface for the registered Stage 2 regular-batch input
  frontier;
- no accepted Stage 2 regular-batch sumcheck kernel yet.

Known ledger status:

- `cpu_spartan_product_uniskip`: `ParityCertified`; evidence file
  `target/frontier-metrics/kernel-evidence/cpu_spartan_product_uniskip/cpu_sumcheck_spartan_product_uniskip_raw.json`.
- `cpu_materialized_opening_evaluations`: `ParityCertified`; evidence file
  `target/frontier-metrics/kernel-evidence/cpu_materialized_opening_evaluations/cpu_openings_rlc_materialized_fallback.json`.
- `cpu_stage2_regular_batch_input_claims`: `ParityCertified`; evidence file
  `target/frontier-metrics/kernel-evidence/cpu_stage2_regular_batch_input_claims/frontier_perf_stage2_regular_batch_inputs.json`.
- `cpu_stage2_regular_batch_sumcheck`: `Required`; no accepted evidence
  writer or certified backend kernel yet.

Stage 2 is a backend-first frontier. If existing primitives miss timing or
memory parity for any Stage 2 path, add or port the specialized backend kernel
before accepting the prover slice. This applies to transparent, advice,
BlindFold/ZK, field-inline, and supported combined feature paths.

### jolt-prover-harness

Existing Stage 2 harness coverage is useful but not complete acceptance.

Current product uni-skip tests:

- `frontier_stage2_product.rs::stage2_product_uniskip_frontier_is_replacement_ready_with_certified_kernel_evidence`;
- `frontier_stage2_product.rs::stage2_product_uniskip_frontier_requires_correctness_and_performance_gates`;
- `frontier_stage2_product.rs::stage2_cpu_product_uniskip_verifier_replay_verifies_against_core_fixtures`.

Current regular-batch/opening checkpoint tests:

- `frontier_stage2_batch.rs::stage2_regular_batch_input_checkpoint_matches_core_fixtures`;
- `frontier_stage2_batch.rs::stage2_ram_read_write_opening_checkpoint_matches_core_fixtures`;
- `frontier_stage2_batch.rs::stage2_ram_terminal_opening_checkpoint_matches_core_fixtures`;
- `frontier_stage2_batch.rs::stage2_instruction_claim_opening_checkpoint_matches_core_fixtures`;
- `frontier_stage2_batch.rs::stage2_product_remainder_opening_checkpoint_matches_core_fixtures`.

Current manifest frontiers:

- `stage2_product_uniskip`;
- `stage2_regular_batch_inputs`;
- `stage2_regular_batch_sumcheck`;
- `stage2_ram_read_write_openings`;
- `stage2_ram_terminal_openings`;
- `stage2_product_remainder_openings`;
- `stage2_instruction_claim_openings`.

Required new/expanded harness coverage:

- full Stage 2 clear native verifier replay that replaces both
  `stage2_uni_skip_first_round_proof` and `stage2_sumcheck_proof` in a real
  proof shell;
- advice replay for the full Stage 2 clear component;
- field-inline real-data frontier replay using SDK/guest trace data, analogous
  to the Stage 1 field-inline replay;
- ZK committed-boundary replay that runs `verify_until_stage1`, `stage1::verify`,
  and then `stage2::verify`, requiring `Stage2Output::Zk`;
- `zk + field-inline` committed-boundary replay if the workspace supports the
  combination;
- replacement-readiness gates for any new Stage 2 regular-batch kernel.

## Target Prover Shape

### Public API

Stage 2 should expose one canonical public entrypoint first in `prove.rs`:

```rust
pub fn prove<F, W, B, T, C>(
    input: Stage2ProverInput<'_, W, ...>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage2ProverOutput<F, SumcheckProof<F, C>>, ProverError>
```

The exact type parameters should follow the existing Stage 1 style and the
compiled feature surface. Under `field-inline`, the input bundle should include
the field-inline witness provider without adding a separate public prover
function. Under `zk`, mode should come from `JoltProtocolConfig`, not from a
test-only boolean.

`Stage2ProverInput` should include:

- `Stage2ProverConfig` or a unified config containing `log_t`, `log_k`, and
  `JoltReadWriteConfig`;
- the `Stage1Output` or clear/ZK-specific dependency needed by the selected
  mode;
- a Jolt VM witness provider;
- a field-inline witness provider under `field-inline`;
- the verifier protocol config.

`Stage2ProverOutput` should contain:

- the two Stage 2 proof fields;
- clear `Stage2Claims` when transparent;
- committed output-claim data and committed witness retention handles when ZK;
- typed public/dependency state needed by Stage 3 and later stages;
- prover-local opening metadata needed by final opening proof assembly.

Do not make downstream stages reconstruct Stage 2 public outputs from raw proof
fields when the Stage 2 prover already computed them.

### Clear Prover Order

The clear public `prove` implementation should read like the verifier:

1. Validate clear Stage 1 dependency.
2. Derive `tau_low` from Stage 1 remainder challenges and reverse it.
3. Derive `tau_high = transcript.challenge()`.
4. Prove product uni-skip first round.
5. Verify locally that the round polynomial sums to the computed input claim.
6. Append the product uni-skip round polynomial.
7. Derive the product uni-skip challenge.
8. Evaluate and append the product uni-skip output claim.
9. Derive regular-batch gammas and output-address challenges.
10. Build regular-batch input claims in verifier order.
11. Prove the regular-batch compressed Boolean sumcheck in verifier order.
12. Derive every regular-batch instance opening point from the batched
    verifier reduction.
13. Evaluate Stage 2 output openings through backend requests.
14. Recompute expected regular-batch final claim using `jolt-claims` formulas
    and verifier-owned claim structs.
15. Append Stage 2 opening claims in verifier order.
16. Return `Stage2ClearOutput`-equivalent data for Stage 3.

### ZK Prover Order

The ZK public `prove` implementation should mirror the verifier's committed
boundary:

1. Validate ZK Stage 1 dependency.
2. Derive the same public challenges as clear mode.
3. Produce committed product uni-skip proof data in
   `stage2_uni_skip_first_round_proof`.
4. Produce one committed product uni-skip output claim.
5. Produce committed regular-batch proof data in `stage2_sumcheck_proof`.
6. Produce committed regular-batch output claims with verifier count:
   - `15` without field-inline;
   - `18` with field-inline.
7. Retain the committed round witnesses and hidden output-claim witnesses for
   BlindFold/Stage 8.
8. Return public opening-point data required by downstream ZK stages:
   `Stage2RamValCheckInputs`, `Stage2RamRaClaimReductionInputs`, and
   field-inline ZK output under `field-inline`.

The Stage 2 spec does not require full BlindFold verification yet. The required
ZK acceptance scope is native verifier committed-boundary acceptance:
`verify_until_stage1`, `stage1::verify`, then `stage2::verify` returning
`Stage2Output::Zk`. Full BlindFold proof verification remains Stage 8/full
JoltProof work.

## Feature Requirements

### Advice

Stage 2 is advice-correct when:

- full clear Stage 2 native verifier replay passes for `muldiv` and advice
  fixtures;
- the advice fixture includes trusted and untrusted advice commitments from
  Stage 0;
- Stage 1 advice-influenced openings are consumed through verifier-owned Stage
  1 outputs, not duplicated local claim structs;
- regular-batch input claims and output claims match core fixture checkpoints;
- transcript state after Stage 2 matches native verifier replay.

Stage 2 itself does not add new advice commitments, but it must preserve advice
semantics through the Stage 1 openings and instruction/lookup/RAM reductions.

### BlindFold/ZK

Stage 2 is ZK-correct at the current frontier when:

- product uni-skip and regular-batch proofs are committed proofs, not clear
  proofs with a ZK flag;
- `stage2_uni_skip_first_round_proof` has one committed round and one committed
  output-claim row at the selected vector-commitment capacity;
- `stage2_sumcheck_proof` has committed rounds matching the batched statement
  and committed output-claim rows matching `15` or `18` hidden claims depending
  on `field-inline`;
- `verify_until_stage1`, `stage1::verify`, and `stage2::verify` accept the
  proof shell and return `Stage2Output::Zk`;
- the prover retains committed witnesses needed by the later BlindFold prover;
- any Stage 2 ZK-specific heavy compute or commitment work not covered by the
  shared BlindFold kernel has performance evidence before acceptance.

### Field-Inline

Stage 2 field-inline support must not be a separate top-level prover fork.
Under the `field-inline` feature, the canonical Stage 2 path must:

- include Stage 1 field-inline product inputs in product uni-skip input-claim
  computation;
- extend product remainder output reconstruction with field product and field
  inverse product weights;
- include the field-register claim-reduction sumcheck in the regular batch in
  verifier order;
- evaluate field-inline product openings:
  `field_rs1_value`, `field_rs2_value`, and `field_rd_value`;
- append field-inline openings at the exact verifier position;
- support `zk + field-inline` committed-boundary replay if the workspace
  supports that combination.

Field-inline must compile out cleanly when the feature is disabled.

### Public Inputs and RAM

RAM public-input handling must follow `jolt-verifier` and `jolt-claims`.

- RAM read-write opening points must use the verifier's read-write dimension
  helpers.
- RAM RAF evaluation must reconstruct the RAF address point and public
  `unmap_address` exactly as the verifier does.
- RAM output check must derive the output address point, public IO mask, and IO
  value from `PublicIoMemory`.
- Do not hand-roll public-memory semantics when `jolt-program`,
  `jolt-claims`, or `jolt-verifier` already own the relevant helper.

## Backend and Kernel Requirements

Stage 2 must keep the backend-first rule:

1. Port or validate the CPU kernel in isolation.
2. Add focused backend/harness microbenchmarks with analytical memory
   accounting.
3. Record the kernel in the backend ledger.
4. Wire `jolt-prover` through backend requests.
5. Accept the frontier only after verifier correctness and performance parity
   both pass.

Required kernel surfaces:

- product uni-skip row evaluation:
  `cpu_spartan_product_uniskip`;
- regular-batch input-claim construction:
  `cpu_stage2_regular_batch_input_claims` or a better specialized kernel if
  current generic computation misses parity;
- regular-batch compressed sumcheck:
  add a dedicated Stage 2 regular-batch kernel if core uses a specialized path
  and generic backend sumcheck misses the parity gate;
- materialized opening evaluations:
  `cpu_materialized_opening_evaluations`;
- field-inline product/field-register claim reduction kernels if field-inline
  changes timing or memory materially;
- ZK committed round/output-claim work if not covered by the shared BlindFold
  kernel.

Missing parity on any required path is a backend task first. Do not accept a
slower prover-side workaround for transparent, advice, BlindFold/ZK,
field-inline, or `zk + field-inline`.

## Performance Gates

Stage 2 replacement readiness requires the default frontier parity gate:

- timing ratio within 15%;
- peak-memory ratio within 15%;
- `KernelPortStatus::ParityCertified` for every required kernel;
- `validate_frontier_replacement_ready` passes with loaded evidence;
- `validate_global_cpu_backend_inventory_coverage` remains passing.

Current evidence state:

- product uni-skip is parity-certified through
  `cpu_spartan_product_uniskip`;
- materialized opening evaluations are parity-certified through
  `cpu_materialized_opening_evaluations`;
- regular-batch input claims are parity-certified through
  `cpu_stage2_regular_batch_input_claims`;
- full Stage 2 regular-batch sumcheck proof generation is not yet
  parity-certified;
- field-inline and ZK Stage 2 paths do not yet have accepted performance
  evidence.

Suggested evidence commands:

```bash
JOLT_WRITE_KERNEL_EVIDENCE=cpu_spartan_product_uniskip cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet
JOLT_WRITE_KERNEL_EVIDENCE=cpu_materialized_opening_evaluations cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet
JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage2_regular_batch_input_claims cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet
```

If a new Stage 2 regular-batch kernel is added, replace the current
`cpu_stage2_regular_batch_sumcheck` required ledger entry with a
parity-certified entry and canonical evidence writer before claiming Stage 2
performance parity.

## Correctness Gates

Focused prover tests should cover:

```bash
cargo nextest run -p jolt-prover stage2 --cargo-quiet
cargo nextest run -p jolt-prover stage2 --features zk --cargo-quiet
cargo nextest run -p jolt-prover stage2 --features field-inline --cargo-quiet
cargo nextest run -p jolt-prover stage2 --features zk,field-inline --cargo-quiet
```

Existing checkpoint and subfrontier tests should continue to pass:

```bash
cargo nextest run -p jolt-prover-harness stage2_product --features core-fixtures --cargo-quiet
cargo nextest run -p jolt-prover-harness stage2_regular_batch_input_checkpoint_matches_core_fixtures --features core-fixtures --cargo-quiet
cargo nextest run -p jolt-prover-harness stage2_ram_read_write_opening_checkpoint_matches_core_fixtures --features core-fixtures --cargo-quiet
cargo nextest run -p jolt-prover-harness stage2_ram_terminal_opening_checkpoint_matches_core_fixtures --features core-fixtures --cargo-quiet
cargo nextest run -p jolt-prover-harness stage2_instruction_claim_opening_checkpoint_matches_core_fixtures --features core-fixtures --cargo-quiet
cargo nextest run -p jolt-prover-harness stage2_product_remainder_opening_checkpoint_matches_core_fixtures --features core-fixtures --cargo-quiet
```

Required new/expanded verifier replay tests:

```bash
cargo nextest run -p jolt-prover-harness stage2_full_verifier_replay --features core-fixtures --cargo-quiet
cargo nextest run -p jolt-prover-harness stage2_zk_committed_boundary --features core-fixtures,zk --cargo-quiet
cargo nextest run -p jolt-prover-harness field_inline_eq_poly_guest --features core-fixtures,field-inline --cargo-quiet
cargo nextest run -p jolt-prover-harness field_inline_eq_poly_guest --features core-fixtures,zk,field-inline --cargo-quiet
```

Use actual test names when implemented. The important acceptance condition is
that native `jolt-verifier` accepts the assembled Stage 2 component, not merely
that local helper checkpoints match shapes.

Static rails:

```bash
cargo fmt -q
cargo clippy -p jolt-prover --features zk,field-inline -q --all-targets -- -D warnings
cargo clippy -p jolt-prover-harness --features core-fixtures,zk,field-inline --tests -q -- -D warnings
```

If `--all-targets` on the harness remains blocked by an unrelated bench warning,
document that and run the focused `--tests` profile plus the impacted bench
compile/run commands.

## Implementation Slice Plan

1. Normalize Stage 2 collections and current helpers.
   Replace `HashMap` with `BTreeMap`, keep duplicate/missing/extra slot errors
   targeted, and preserve existing subfrontier tests.

2. Define the canonical API.
   Add `Stage2ProverInput` and `Stage2ProverOutput` so the public `prove`
   entrypoint can consume Stage 1 output and return verifier-owned Stage 2
   proof/claim components plus downstream state.

3. Backend evidence first.
   Keep `cpu_stage2_regular_batch_input_claims` certified. Add a specialized
   regular-batch sumcheck kernel if the generic path cannot meet core timing
   and memory parity.

4. Product uni-skip integration.
   Move existing product uni-skip logic behind the canonical `prove` flow while
   preserving the existing certified raw-row backend path.

5. Regular batch proving.
   Implement the compressed regular-batch sumcheck in verifier order, using
   backend-owned heavy compute. Locally recompute the final output claim before
   accepting the proof component.

6. Output openings.
   Evaluate product remainder, instruction claim reduction, RAM read-write, RAM
   terminal, and field-inline openings through deterministic backend requests.
   Assemble `Stage2BatchOutputOpeningClaims` directly from verifier-owned types.

7. Field-inline.
   Remove the top-level `cfg(not(feature = "field-inline"))` blocker in
   `prove.rs`. Add field-inline product and field-register claim-reduction
   support to the canonical path with cfg-local helpers only where necessary.

8. ZK committed boundary.
   Emit committed product uni-skip and regular-batch proof fields with the
   correct committed output-claim counts. Retain committed witnesses for
   BlindFold and add native verifier committed-boundary replay through
   `stage2::verify`.

9. Full replay and parity gates.
   Add full Stage 2 proof-shell replay for transparent/advice and real
   field-inline data. Run performance evidence and ledger gates.

10. Justification log.
    Append exact code touched, correctness commands, performance evidence,
    remaining limitations, and confidence statement to this spec before marking
    Stage 2 complete.

## Acceptance Checklist

Stage 2 is accepted only when all of these are true:

- public `prove` is the canonical entrypoint and appears first in `prove.rs`;
- production Stage 2 imports verifier proof/claim/output structs directly from
  `jolt-verifier`;
- semantic ordering, dimensions, formulas, opening IDs, challenge IDs, and
  public IDs come from `jolt-claims`;
- witness data flows through `jolt-witness` views or primitive row providers;
- heavy compute flows through certified `jolt-backends` kernels;
- deterministic `BTreeMap`/`BTreeSet` collection is used for slot-keyed and
  variable-keyed data;
- product uni-skip, regular batch, and all output openings are implemented in
  verifier transcript order;
- transparent and advice native verifier replay pass with real core fixtures;
- field-inline native verifier replay passes with real SDK/guest trace data and
  any available core fixtures;
- ZK committed-boundary replay runs through `verify_until_stage1`,
  `stage1::verify`, and `stage2::verify`, returning `Stage2Output::Zk`;
- `zk + field-inline` is tested or concretely documented as unsupported;
- required performance evidence is loaded and parity-certified for every Stage
  2 backend kernel surface;
- this spec contains the final implementation-slice justification.

## Implementation Slice Justification Log

### 2026-05-30 Production committed-boundary prover emission

Implemented the production Stage 2 committed-boundary prover path in
`jolt-prover`:

- added `Stage2CommittedBoundaryOutput` with committed proof fields, public
  output, hidden `Stage2ClearOutput`, hidden output-claim values, and retained
  committed witnesses;
- added standard ZK and `zk + field-inline` `prove_committed_boundary`
  entrypoints;
- added committed product uni-skip proving that commits the real first-round
  polynomial and one hidden output claim;
- added committed regular-batch proving that mirrors the verifier committed
  transcript path and commits regular-batch output claims in verifier order;
- added focused Stage 2 tests that run production Stage 1 and Stage 2 committed
  provers, then replay native `stage1::verify` and `stage2::verify`.

Commands run:

```bash
cargo fmt -q
cargo check -p jolt-prover -q
cargo check -p jolt-prover --features field-inline -q
cargo check -p jolt-prover --features zk -q
cargo check -p jolt-prover --features zk,field-inline -q
cargo check -p jolt-prover --features zk --tests -q
cargo check -p jolt-prover --features zk,field-inline --tests -q
cargo clippy -p jolt-prover --features zk -q --all-targets -- -D warnings
cargo clippy -p jolt-prover --features zk,field-inline -q --all-targets -- -D warnings
cargo nextest run -p jolt-prover --features zk stage2_committed_boundary_produces_native_verifier_output --cargo-quiet
cargo nextest run -p jolt-prover --features zk,field-inline stage2_field_inline_committed_boundary_produces_native_verifier_output --cargo-quiet
```

Observed results:

- standard ZK Stage 2 committed-boundary native verifier replay: `1` passed,
  `0` failed;
- `zk + field-inline` Stage 2 committed-boundary native verifier replay:
  `1` passed, `0` failed.

No benchmarks were run. Stage 2 is still not full replacement-ready until the
regular-batch backend performance evidence is certified, but the critical
top-level ZK path is no longer blocked on production Stage 2 committed
emission.

### 2026-05-29 Stage 2 Correctness Frontier

Implemented and verified a correctness-first Stage 2 frontier through native
`jolt-verifier` replay:

- clear transparent/advice Stage 2 now assembles both
  `stage2_uni_skip_first_round_proof` and `stage2_sumcheck_proof`, grafts the
  modular Stage 2 component into real core fixture proof shells, and is accepted
  by native `stage2::verify`;
- field-inline clear Stage 2 now runs on the real SDK
  `field-inline-eq-poly-guest` trace and is accepted by native
  `stage2::verify`;
- non-field-inline ZK Stage 2 committed-boundary replay now runs
  `verify_until_stage1 -> stage1::verify -> stage2::verify` on real core ZK
  fixtures and requires `Stage2Output::Zk`;
- `zk + field-inline` committed-boundary replay now runs the same native
  verifier path on the field-inline SDK guest proof shell and requires
  `Stage2Output::Zk`.

Correctness fixes made while wiring the field-inline path:

- `FieldProduct` and `FieldInvProduct` witness views are now derived products
  (`rs1 * rs2` and `rs1 * rd`) for every field-inline row, matching the
  unguarded field product constraints used by Stage 2;
- field-inline product uni-skip now treats the final padded cycle like the core
  product path: `not_next_noop = 0` on the final cycle.

Commands run:

```bash
cargo fmt -q
cargo check -p jolt-prover -q
cargo check -p jolt-prover --features field-inline -q
cargo check -p jolt-prover --features zk -q
cargo check -p jolt-prover --features field-inline,zk -q
cargo clippy -p jolt-prover --features field-inline,zk -q --all-targets -- -D warnings
cargo clippy -p jolt-prover-harness --features core-fixtures,field-inline,zk --tests -q -- -D warnings
JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage2_regular_batch_input_claims cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet
cargo nextest run -p jolt-prover-harness stage2_regular_batch_input_frontier_is_replacement_ready_with_certified_kernel_evidence --features core-fixtures --cargo-quiet
cargo nextest run -p jolt-prover-harness stage2 --features core-fixtures --cargo-quiet
cargo nextest run -p jolt-prover-harness zk_stage2_committed_boundary_is_native_verifier_accepted --features core-fixtures,zk --cargo-quiet
cargo nextest run -p jolt-prover-harness field_inline_eq_poly_guest --features core-fixtures,field-inline --cargo-quiet
cargo nextest run -p jolt-prover-harness field_inline_eq_poly_guest_accepts_zk_committed_stage2_boundary --features core-fixtures,field-inline,zk --cargo-quiet
cargo check -p jolt-prover-harness --features core-fixtures -q
cargo clippy -p jolt-prover-harness --features core-fixtures --tests --benches -q -- -D warnings
cargo nextest run -p jolt-prover-harness stage2_regular_batch_sumcheck --features core-fixtures --cargo-quiet
cargo nextest run -p jolt-prover-harness optimization --features core-fixtures --cargo-quiet
cargo nextest run -p jolt-prover-harness frontier_manifest --features core-fixtures --cargo-quiet
```

Observed results:

- Stage 2 harness replay: `15` passed, `0` failed.
- Stage 2 regular-batch input replacement-readiness replay: `1` passed, `0`
  failed.
- Stage 2 regular-batch input evidence writer: wrote canonical evidence with
  gate status `Pass`, time ratio `0.9633338241790607`, and memory ratio `1.0`.
- ZK Stage 2 committed-boundary replay: `1` passed, `0` failed.
- Field-inline clear SDK replay: `1` passed, `0` failed.
- `zk + field-inline` committed-boundary replay: `1` passed, `0` failed.
- Focused `jolt-prover` and harness clippy checks pass with
  `field-inline,zk`.
- Focused harness check/clippy for the Stage 2 perf ledger additions passed.
- `stage2_regular_batch_sumcheck` manifest gate replay: `1` passed, `0`
  failed.
- Optimization and frontier manifest accounting replays passed.
- Focused `cargo nextest run -p jolt-prover stage2` filters matched no tests
  in all feature combinations, so the harness replay tests are the active
  Stage 2 correctness evidence.

Performance parity status:

- `cpu_spartan_product_uniskip` remains `ParityCertified` with existing
  benchmark evidence.
- `cpu_materialized_opening_evaluations` remains `ParityCertified` with
  existing benchmark evidence for Stage 2 opening evaluation surfaces.
- `cpu_stage2_regular_batch_input_claims` is now `ParityCertified` with
  evidence
  `target/frontier-metrics/kernel-evidence/cpu_stage2_regular_batch_input_claims/frontier_perf_stage2_regular_batch_inputs.json`;
  the recorded gate status was `Pass`, with time ratio
  `0.9633338241790607` and memory ratio `1.0`.
- `cpu_stage2_regular_batch_sumcheck` is now a registered `Required` ledger
  row and manifest frontier (`stage2_regular_batch_sumcheck`) so the remaining
  backend port cannot be hidden by the input-claim/opening checkpoints.
- Stage 2 full replacement readiness is **not yet achieved**. The regular-batch
  prover path is currently correctness-first and dense-materialized. There is
  still no accepted Stage 2 regular-batch sumcheck kernel/evidence writer.
- Field-inline and ZK Stage 2 correctness are covered at the verifier frontier,
  but their timing and memory parity are not certified.

Conclusion:

The Stage 2 correctness frontier is accepted for transparent, advice,
field-inline, ZK committed-boundary, and `zk + field-inline`
committed-boundary replay. Stage 2 is **not replacement-ready** and this goal
must not be marked complete until the regular-batch backend kernels and
benchmark evidence satisfy the 15% timing and memory parity gates across the
required feature surfaces.

### Interim parity measurement update: 2026-05-29

The Stage 2 regular-batch sumcheck frontier now has an isolated parity harness
instead of a stub:

- `jolt-prover` exposes a `frontier-harness`-only
  `prove_stage2_regular_batch_sumcheck_for_frontier` entrypoint that reuses the
  private Stage 2 prover helpers without widening the production API;
- `jolt-prover-harness` can run the modular Stage 2 regular-batch sumcheck
  directly from real core fixture data;
- the harness also constructs native core Stage 2 regular-batch instances from
  the same real guest fixture so benchmark timing excludes fixture generation;
- modular Stage 2 round polynomials now trim trailing zero coefficients before
  transcript absorption/proof emission, matching core's canonical compressed
  proof shape.

Focused validation added/run:

```bash
cargo check -p jolt-prover --features frontier-harness -q
cargo check -p jolt-prover-harness --features core-fixtures --benches -q
cargo clippy -p jolt-prover-harness --features core-fixtures --tests --benches -q -- -D warnings
cargo nextest run -p jolt-prover-harness stage2_regular_batch_sumcheck_kernel_fixture_matches_core_fixtures --features core-fixtures --cargo-quiet
JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage2_regular_batch_sumcheck cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet
```

Observed result:

- the focused fixture test passed, including native core isolated regular-batch
  execution and modular proof equality against the core fixture proof;
- the canonical evidence writer is implemented, but the current dense modular
  path fails the required parity gate with time ratio
  `640.0709755846556` and peak-allocation ratio `201.79603583182669`.

This confirms the remaining Stage 2 blocker is the real backend kernel port,
not verifier replay or benchmark wiring. The dense materialized
`SumcheckRegularBatchInstance` path must be replaced by specialized CPU backend
state for the Stage 2 RAM read-write, product remainder, instruction claim
reduction, RAM RAF, and output-check instances before this stage can be marked
replacement-ready.

### Final parity update: 2026-05-29

The interim blocker above is resolved for the core-backed Stage 2 frontier. The
regular-batch sumcheck now uses specialized backend state for the RAM
read-write, RAM RAF, and RAM output-check instances while keeping the
product-remainder, instruction-claim, and field-register claim reductions in
the generic regular-batch tail. The prover batches the instances in verifier
order:

1. RAM read-write;
2. product remainder;
3. instruction claim reduction;
4. field-register claim reduction when `field-inline` is enabled;
5. RAM RAF evaluation;
6. RAM output-check.

Code-quality notes:

- the public Stage 2 `prove` entrypoint remains first in `prove.rs`;
- the regular-batch orchestration is linear and mirrors the verifier's claim
  order;
- the new RAM terminal backend states use `jolt-claims` dimensions and
  verifier-derived opening-point semantics rather than local ad hoc ordering;
- address-major materialization now preserves checkpoint values for sparse
  columns with no accesses, which is required by small phase-2 configurations
  such as the field-inline SDK frontier;
- RAM terminal dummy-cycle scaling is applied only before the cycle-gap dummy
  rounds, so configurations with remaining phase-3 address rounds satisfy the
  final opening check.

Focused validation run:

```bash
cargo fmt -q
cargo check -p jolt-backends -q
cargo check -p jolt-prover --features field-inline,zk -q
cargo clippy -p jolt-prover --features field-inline,zk -q --all-targets -- -D warnings
cargo clippy -p jolt-prover-harness --features core-fixtures --tests --benches -q -- -D warnings
cargo nextest run -p jolt-prover-harness stage2 --features core-fixtures --cargo-quiet
cargo nextest run -p jolt-prover-harness field_inline_eq_poly_guest --features core-fixtures,field-inline --cargo-quiet
cargo nextest run -p jolt-prover-harness zk_stage2_committed_boundary_is_native_verifier_accepted --features core-fixtures,zk --cargo-quiet
cargo nextest run -p jolt-prover-harness field_inline_eq_poly_guest_accepts_zk_committed_stage2_boundary --features core-fixtures,field-inline,zk --cargo-quiet
JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage2_regular_batch_sumcheck cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet
```

Observed results:

- transparent/advice Stage 2 harness: `18` passed, `0` failed;
- field-inline SDK clear replay: `1` passed, `0` failed;
- ZK committed-boundary replay: `1` passed, `0` failed;
- `zk + field-inline` committed-boundary replay: `1` passed, `0` failed;
- `stage2_regular_batch_sumcheck` replacement-ready gate: `1` passed, `0`
  failed;
- regular-batch sumcheck evidence:
  `target/frontier-metrics/kernel-evidence/cpu_stage2_regular_batch_sumcheck/frontier_perf_stage2_regular_batch_sumcheck.json`;
- latest evidence status was `Warn` but passing under the canonical 15% gate:
  time ratio `1.084476476309815`, peak RSS ratio `0.7795293611842224`.

Correctness justification:

- transparent and advice use real core fixtures (`MuldivSmall` and
  `AdviceConsumer`) and compare the modular Stage 2 proof, challenges,
  batching coefficients, and output claim against the native core Stage 2
  regular-batch proof;
- clear verifier replay grafts the modular Stage 2 proof and claims into a
  real proof shell and is accepted by native `jolt-verifier` through
  `stage2::verify`;
- field-inline clear replay uses the real SDK `field-inline-eq-poly-guest`
  trace and is accepted by native `stage2::verify`;
- ZK and `zk + field-inline` are accepted at the committed Stage 2 boundary
  (`verify_until_stage1 -> stage1::verify -> stage2::verify`) and require
  `Stage2Output::Zk`; full BlindFold proof verification remains Stage 8 scope.

Performance justification:

- `cpu_spartan_product_uniskip`, `cpu_stage2_regular_batch_input_claims`,
  `cpu_stage2_regular_batch_sumcheck`, and
  `cpu_materialized_opening_evaluations` are `ParityCertified` or backed by
  passing canonical evidence for the core-backed Stage 2 replacement frontier;
- the regular-batch sumcheck no longer uses the dense all-instance fallback for
  RAM read-write, RAM RAF, or RAM output-check, and its canonical benchmark is
  within the required timing and memory thresholds;
- field-inline clear correctness exercises the same backend RAM kernels with a
  nontrivial `phase1=1, phase2=1` split. A separate field-inline timing ratio is
  not certified because this worktree has no native core field-inline
  performance fixture to compare against; the additional field-register
  reduction remains covered by verifier replay and generic regular-batch
  mechanics.

Stage 2 is replacement-ready for the transparent/advice core-backed frontier
and accepted at the specified verifier frontiers for field-inline, ZK, and
`zk + field-inline`.
