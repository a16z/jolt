# Stage 4 Register Read-Write and RAM Value-Check Frontier Spec

## Scope

Stage 4 proves the regular batched sumcheck over register read-write checking
and RAM value checking. It starts after Stage 2 and Stage 3 have produced their
typed verifier outputs and the transcript is positioned at the Stage 4 boundary.
It ends after the verifier-owned Stage 4 proof field, clear claims or committed
ZK shape, RAM value-check initial evaluation state, and downstream Stage 4
output are assembled.

The stage boundary is:

1. Consume Stage 2 and Stage 3 outputs plus normalized Jolt VM witness views.
2. Consume verifier-validated preprocessing/public I/O data needed to evaluate
   the public initial RAM polynomial and advice selectors.
3. Derive Stage 4 Fiat-Shamir challenges in the exact `jolt-verifier` order.
4. Prove the regular Stage 4 batched sumcheck over:
   - register read-write checking;
   - field-register read-write checking under `field-inline`;
   - RAM value checking.
5. Evaluate every Stage 4 output opening at the verifier-derived points.
6. Assemble the verifier-owned Stage 4 proof fields and claims.
7. Return typed Stage 4 public output for Stage 5 and later stages without
   recomputing verifier reductions.

Stage 4 produces the verifier component for:

- `JoltStageProofs::stage4_sumcheck_proof`;
- clear-mode `ClearProofClaims::stage4`;
- ZK-mode committed Stage 4 sumcheck proof and output-claim commitments that
  `jolt-verifier` lowers into BlindFold;
- clear-mode `Stage4ClearOutput` data needed by Stage 5 and Stage 6;
- ZK-mode `Stage4ZkOutput` data needed by later BlindFold construction.

Stage 4 does not prove register value evaluation, bytecode/instruction read-RAF,
increment reductions, hamming/booleanity reductions, or the final PCS opening
proof. It must, however, preserve the RAM value-check initial-evaluation
decomposition because advice handling and later claim reductions depend on the
same RAM address point and advice contribution state.

Stage 4 should land as one complete stage slice. Subfrontier checkpoints are
useful for review and benchmarking, but Stage 4 is not accepted as
replacement-ready until transparent, advice, BlindFold/ZK committed-boundary,
field-inline, and supported combined feature paths have their documented
correctness and performance evidence.

Every Stage 4 implementation path must explicitly support:

- advice, including trusted and untrusted advice contributions in the RAM
  value-check initial evaluation;
- BlindFold/ZK mode, including committed regular-batch proof data, committed
  output-claim rows, and verifier acceptance at the Stage 4 committed boundary;
- field-inline, behind the `field-inline` feature and absent from disabled
  builds.

## Monitored Workflow

Stage work proceeds in one reviewable Stage 4 slice:

1. Confirm current inventory for `jolt-prover`, `jolt-verifier`,
   `jolt-backends`, `jolt-witness`, `jolt-program`, and
   `jolt-prover-harness`.
2. Tighten backend evidence before accepting prover orchestration.
3. Define the canonical Stage 4 input/output/prover-state API.
4. Refactor toward one public `prove` entrypoint, ordered first in `prove.rs`.
5. Implement transparent and advice paths through native Stage 4 verifier
   replay.
6. Add field-inline register read-write support to the same canonical path.
7. Add ZK committed-boundary proof assembly and verifier replay.
8. Run focused tests, real fixture replay, and kernel evidence gates.
9. Append the final correctness and performance parity justification to this
   spec.
10. Stop for review before moving to Stage 5.

Every fact in the implementation should have a clear owner:

- `jolt-verifier` owns proof fields, verifier outputs, transcript checks,
  clear claims, committed proof shapes, and output-claim counts;
- `jolt-claims` owns relation IDs, dimensions, formula semantics, opening IDs,
  challenge IDs, and public IDs;
- `jolt-witness` owns trace-backed witness views and primitive row views;
- `jolt-program` owns public initial RAM/public I/O memory interpretation used
  to build RAM value-check public evaluations;
- `jolt-backends` owns heavy compute and slot-keyed kernel results;
- `jolt-riscv` should not be needed directly in Stage 4 production
  orchestration;
- `jolt-lookup-tables` should not be used directly by Stage 4. Lookup-table
  semantics belong to Stage 2 and Stage 5 frontiers;
- `jolt-prover-harness` owns migration-only fixtures, verifier replay, and
  performance evidence.

The public prover path should remain linear enough to audit against
`jolt-verifier/src/stages/stage4/verify.rs`.

### Modular Crate Usage

Stage 4 should follow the same ownership split as `jolt-verifier`.

- Use `jolt-claims::protocols::jolt::formulas::registers` for register
  read-write relation metadata, input openings, output openings, challenges,
  and dimensions.
- Use `jolt-claims::protocols::jolt::formulas::ram` for RAM value-check
  metadata, advice opening IDs, output openings, and challenge/public IDs.
- Under `field-inline`, use
  `jolt-claims::protocols::field_inline::formulas::registers` for field
  register read-write metadata rather than duplicating field-register formulas
  in `jolt-prover`.
- Use `jolt-program` public-memory helpers for public initial RAM evaluation
  and public I/O shape. Do not hand-roll RAM/public-I/O address semantics in
  the prover.
- Use `JoltOpeningId` and field-inline opening IDs as semantic request keys
  where practical, then convert to witness oracles with the namespace-owned
  helper such as `jolt_opening_oracle_ref`.
- Use committed/virtual polynomial IDs only as witness-oracle identifiers after
  the semantic opening ID has been chosen.

This keeps Stage 4 keyed by verifier/protocol semantics while still letting the
backend operate on concrete witness views.

## Current Inventory

### jolt-verifier

Relevant verifier code lives in `crates/jolt-verifier/src/stages/stage4/`.

- `verify.rs` derives, in order:
  - `registers_gamma = transcript.challenge_scalar()`;
  - `field_registers_gamma = transcript.challenge_scalar()` under
    `field-inline`;
  - the `ram_val_check_gamma` domain separator append;
  - `ram_val_check_gamma = transcript.challenge_scalar()`.
- The RAM value-check address point comes from Stage 2 RAM read-write output;
  the RAM output-check point must match the address portion.
- The verifier evaluates public initial RAM from preprocessing and public I/O,
  then reconstructs the full initial evaluation by adding trusted/untrusted
  advice contributions when those commitments are present.
- Clear mode verifies a compressed Boolean batched sumcheck in this statement
  order:
  1. register read-write checking;
  2. field-register read-write checking under `field-inline`;
  3. RAM value checking.
- Clear input claims are computed from:
  - Stage 3 register claim-reduction outputs for register read-write;
  - Stage 2 field-inline product outputs for field-register read-write;
  - Stage 2 RAM read-write/value-final outputs plus the decomposed initial RAM
    value for RAM value-check.
- Clear output reconstruction uses:
  - register read-write dimensions from `rw_config`;
  - `eq(stage3_register_reduction_opening_point, register_cycle_point)`;
  - field-inline read-write dimensions and field-register cycle equality under
    `field-inline`;
  - `LtPolynomial(r_cycle_prime, r_cycle) + ram_val_check_gamma` for RAM
    value-check.
- The verifier appends Stage 4 opening claims in this transcript order:
  1. untrusted advice RAM value-check opening, if present;
  2. trusted advice RAM value-check opening, if present;
  3. register read-write `registers_val`;
  4. register read-write `rs1_ra`;
  5. register read-write `rs2_ra`;
  6. register read-write `rd_wa`;
  7. register read-write `rd_inc`;
  8. field-register read-write `field_registers_val` under `field-inline`;
  9. field-register read-write `field_rs1_ra` under `field-inline`;
  10. field-register read-write `field_rs2_ra` under `field-inline`;
  11. field-register read-write `field_rd_wa` under `field-inline`;
  12. field-register read-write `field_rd_inc` under `field-inline`;
  13. RAM value-check `ram_ra`;
  14. RAM value-check `ram_inc`.
- ZK mode verifies committed consistency for `stage4_sumcheck_proof` and
  requires output-claim count:
  - `7 + advice_count` without field-inline;
  - `12 + advice_count` with field-inline.
- `outputs.rs` owns `Stage4PublicOutput`, `Stage4ClearOutput`,
  `Stage4ZkOutput`, `Stage4Output`, `VerifiedStage4Batch`,
  `VerifiedStage4Sumcheck`, `RamValCheckInitialEvaluation`, and
  `VerifiedRamValCheckAdviceContribution`.
- `stages/zk/blindfold/stage4.rs` lowers Stage 4 into BlindFold using the same
  statement order, output IDs, advice-conditionals, and field-inline extension.

Stage 4 prover code must import verifier-owned proof, claim, and output structs
directly rather than duplicating those shapes locally.

### jolt-prover

Current implementation lives in `crates/jolt-prover/src/stages/stage4/`.

- `input.rs` defines `Stage4ProverConfig { log_t, log_k, rw_config }`.
- `prove.rs` exposes helper entrypoints rather than one canonical public Stage
  4 `prove` entrypoint:
  - `derive_stage4_regular_batch_prefix`;
  - `evaluate_stage4_output_openings`;
  - `prove_stage4_transparent_sumchecks`;
  - `append_stage4_opening_claims`.
- `derive_stage4_regular_batch_prefix` currently consumes a precomputed
  `Stage4RamValCheckInitialEvaluation`. A production Stage 4 path must compute
  that state from verifier-validated preprocessing/public I/O, advice presence,
  advice opening points, and witness views rather than importing it from a core
  fixture.
- The transparent sumcheck helper is compiled only under
  `not(feature = "field-inline")`.
- Field-inline input-claim wiring is partially present in the prefix helper,
  but field-register read-write proof generation, output openings, proof
  assembly, and verifier replay are not complete.
- There is no ZK committed Stage 4 proof assembly path.
- There is no canonical `Stage4ProverInput` bundle that consumes Stage 2/3
  outputs, witness providers, preprocessing/public I/O, protocol config, and
  mode.
- There is no canonical `Stage4ProverOutput` carrying verifier-owned proof
  fields, clear/ZK output, RAM initial-evaluation state, and prover-local
  opening or BlindFold state.
- `output.rs` uses `HashMap` for slot collection. Replace it with `BTreeMap`
  so Stage 4 matches the deterministic Stage 0/1 pattern and produces stable
  duplicate/missing/extra slot diagnostics.
- Stage 4 request slots are:
  - register read-write openings start at `0`;
  - RAM value-check openings start at `16`.
- `request.rs` currently maps a stage-local enum directly to committed/virtual
  polynomial IDs. The target cleanup is to derive request contents from
  `jolt-claims` Stage 4 formula opening helpers and use
  `jolt_opening_oracle_ref` for witness lookup, matching the verifier's
  modular ownership.

Current code is a useful transparent correctness frontier, but it is not yet a
complete replacement-ready prover stage because RAM initial-evaluation
computation is externally supplied and field-inline/ZK paths are incomplete. The
regular-batch sumcheck itself has since been moved off the dense prover fallback
onto the specialized backend path recorded in Slice 3.

### jolt-backends

Stage 4 heavy compute belongs in `jolt-backends`.

Current relevant APIs and kernels:

- `Stage4ReadWriteSumcheckBackend` for the specialized transparent
  registers-read/write plus RAM value-check sumcheck path;
- `SumcheckBackend::evaluate_sumcheck_views` for output-opening evaluations;
- `cpu_materialized_opening_evaluations`, a parity-certified opening kernel
  used by Stage 4 output-opening checkpoints;
- generic sumcheck primitives, which can remain test/reference scaffolding but
  must not be accepted as the replacement path if core has specialized Stage 4
  algorithms.

Known ledger status:

- `cpu_stage4_regular_batch_input_claims`: `ParityCertified`; canonical evidence
  file
  `target/frontier-metrics/kernel-evidence/cpu_stage4_regular_batch_input_claims/frontier_perf_stage4_regular_batch_inputs.json`.
- `cpu_stage4_regular_batch_sumcheck`: `ParityCertified`; canonical evidence
  file
  `target/frontier-metrics/kernel-evidence/cpu_stage4_regular_batch_sumcheck/frontier_perf_stage4_regular_batch_sumcheck.json`.
- `cpu_materialized_opening_evaluations`: `ParityCertified`; evidence file
  `target/frontier-metrics/kernel-evidence/cpu_materialized_opening_evaluations/cpu_openings_rlc_materialized_fallback.json`.

The optimization inventory identifies Stage 4 RAM value-check work under
`OPT-REL-006` and `OPT-REL-007`, and register read-write sparse matrix work
under the `OPT-RW-*` family. Before marking Stage 4 replacement-ready, review
whether the Stage 4 manifest and backend ledger must name these IDs in addition
to the current sumcheck/equality-table IDs. Do not hide specialized RAM/register
core algorithms behind only generic `OPT-SC-*` coverage.

Stage 4 is a backend-first frontier. If existing primitives miss timing or
memory parity for transparent, advice, BlindFold/ZK, field-inline, or supported
combined feature paths, add or port the specialized backend kernel before
accepting the prover slice.

### jolt-prover-harness

Existing Stage 4 harness coverage is useful but not complete acceptance.

Current manifest frontiers:

- `stage4_regular_batch_inputs`;
- `stage4_output_openings`;
- `stage4_regular_batch_sumcheck`.

Current transparent checkpoint and replay tests:

- `stage4_regular_batch_input_checkpoint_matches_core_fixtures`;
- `stage4_output_opening_checkpoint_matches_core_fixtures`;
- `stage4_regular_batch_sumcheck_verifier_replay_verifies_against_core_fixtures`;
- manifest gate tests for Stage 4 input, output opening, and sumcheck
  frontiers.

Current limitations:

- coverage is transparent only;
- fixtures are `MuldivSmall` and `AdviceConsumer`;
- the modular input checkpoint receives RAM value-check initial evaluation from
  the core-verified fixture path;
- field-inline Stage 4 replay is not separately documented;
- ZK committed-boundary replay through `stage4::verify` is missing;
- backend replacement-readiness for Stage 4 input claims and sumcheck is not
  certified.

Required new/expanded harness coverage:

- full clear native verifier replay that replaces `stage4_sumcheck_proof` and
  clear Stage 4 claims in a real proof shell;
- advice replay for trusted and untrusted RAM value-check initial evaluation;
- modular computation of public initial RAM and advice contribution state,
  checked against core fixtures;
- field-inline replay covering field-register read-write output openings and
  proof shape;
- ZK committed-boundary replay that runs through `stage4::verify`, requiring
  `Stage4Output::Zk`;
- `zk + field-inline` committed-boundary replay if the workspace supports the
  combination;
- replacement-readiness gates for Stage 4 regular-batch input and sumcheck
  kernels once evidence writers exist.

## Target Prover Shape

### Public API

Stage 4 should expose one canonical public entrypoint first in `prove.rs`:

```rust
pub fn prove<F, W, B, T, C>(
    input: Stage4ProverInput<'_, W, ...>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage4ProverOutput<F, C>, ProverError>
```

The exact type parameters should follow the existing Stage 1/2 style and the
compiled feature surface. The input bundle should include:

- `Stage4ProverConfig` or a unified config containing `log_t`, `log_k`, and
  `JoltReadWriteConfig`;
- Stage 2 and Stage 3 outputs, clear or ZK according to mode;
- a Jolt VM witness provider;
- field-inline witness/provider state under `field-inline`;
- verifier-validated preprocessing/public I/O data needed for public initial
  RAM;
- advice presence flags and advice memory layout from validated proof/public
  input shape;
- verifier protocol config and proof mode.

`Stage4ProverOutput` should contain:

- `stage4_sumcheck_proof`;
- clear `Stage4Claims` when transparent;
- committed output-claim data and committed witness retention handles when ZK;
- typed `Stage4Output`-equivalent data for Stage 5 and later stages;
- RAM value-check initial evaluation and advice contribution state;
- prover-local opening metadata needed by final opening proof assembly.

The input and request builders should be keyed by verifier/protocol semantic
IDs before lowering to witness `OracleRef`s. Do not encode Stage 4's RAM,
register, advice, or field-register opening lists by hand in the prover when
the corresponding `jolt-claims` formula helper already defines them.

### Clear Prover Order

The clear public `prove` implementation should read like the verifier:

1. Validate clear Stage 2 and Stage 3 dependencies.
2. Split the Stage 2 RAM read-write opening point into RAM address and cycle
   points, and verify the RAM output-check point matches the address portion.
3. Compute public initial RAM evaluation using `jolt-program` public-memory
   helpers.
4. Evaluate trusted/untrusted advice openings at verifier-derived advice
   address points when those commitments are present.
5. Reconstruct the full RAM initial evaluation and retain advice contribution
   metadata.
6. Derive `registers_gamma`, optional `field_registers_gamma`, the RAM
   value-check gamma domain separator, and `ram_val_check_gamma`.
7. Verify Stage 3 register/instruction dependencies agree on `rs1` and `rs2`.
8. Build Stage 4 input claims in verifier statement order:
   register read-write, optional field-register read-write, RAM value-check.
9. Prove the regular-batch compressed Boolean sumcheck in verifier statement
   order.
10. Derive register, optional field-register, and RAM value-check opening
    points from the batched verifier reduction.
11. Evaluate Stage 4 output openings through backend requests.
12. Recompute expected output claims using `jolt-claims` formulas and verifier
    public/challenge IDs.
13. Verify the batched final claim locally before emitting the proof component.
14. Append Stage 4 opening claims in verifier transcript order, including
    advice and field-inline conditionals.
15. Return `Stage4ClearOutput`-equivalent data for Stage 5 and later stages.

### ZK Prover Order

The ZK public `prove` implementation should mirror the verifier's committed
boundary:

1. Validate ZK Stage 2 and Stage 3 dependencies.
2. Derive the same public challenges and RAM initial-evaluation public data as
   clear mode.
3. Produce committed Stage 4 batch proof data in `stage4_sumcheck_proof`.
4. Produce committed output-claim rows with verifier count:
   - `7 + advice_count` without field-inline;
   - `12 + advice_count` with field-inline.
5. Retain the committed round witnesses and hidden output-claim witnesses for
   BlindFold.
6. Return public consistency data required by downstream ZK stages and
   BlindFold, including register and RAM value-check opening points.

The Stage 4 spec does not require full BlindFold verification yet. The required
ZK acceptance scope is native verifier committed-boundary acceptance through
`stage4::verify` returning `Stage4Output::Zk`. Full BlindFold proof
verification remains Stage 8/full JoltProof work.

## Feature Requirements

### Advice

Stage 4 is advice-correct when:

- full clear Stage 4 native verifier replay passes for `muldiv` and advice
  fixtures;
- trusted and untrusted advice commitments are handled according to verifier
  proof/public-input shape;
- absent advice commitments reject unexpected advice openings and present
  advice commitments require exactly one opening each;
- advice selectors and opening points are derived from the public memory layout
  exactly as the verifier does;
- RAM value-check initial evaluation decomposes into `public_eval`,
  `advice_contributions`, and `full_eval` matching core fixture checkpoints;
- transcript state after Stage 4 matches native verifier replay.

### BlindFold/ZK

Stage 4 is ZK-correct at the current frontier when:

- `stage4_sumcheck_proof` is a committed proof, not a clear proof with a ZK
  flag;
- committed proof statements are ordered as register read-write, optional
  field-register read-write, RAM value-check;
- committed output-claim rows match the verifier's conditional output count;
- RAM value-check public initial evaluation and opening points match verifier
  public data;
- native verifier committed-boundary replay reaches `Stage4Output::Zk`;
- any Stage 4 ZK-specific heavy compute or commitment work not covered by the
  shared BlindFold kernel has performance evidence before acceptance.

### Field-Inline

Under the `field-inline` feature, the canonical Stage 4 path must:

- include field-register read-write in the regular batch after ordinary
  register read-write and before RAM value-check;
- consume Stage 2 field-register claim-reduction outputs and opening points in
  verifier order;
- derive field-register read-write dimensions from the verifier protocol config;
- evaluate and append field-register output openings:
  `field_registers_val`, `field_rs1_ra`, `field_rs2_ra`, `field_rd_wa`, and
  `field_rd_inc`;
- support ZK committed output-claim count `12 + advice_count`;
- compile out cleanly when `field-inline` is disabled.

### Public RAM and Memory Layout

RAM public-input handling must follow `jolt-verifier` and `jolt-program`.

- Public initial RAM evaluation must use the same public program RAM and public
  I/O memory interpretation as the verifier.
- Advice memory layout must use the verifier-validated trusted/untrusted advice
  start addresses and maximum sizes.
- Advice opening points must be suffixes of the Stage 4 RAM address point using
  the same advice-domain width calculation as the verifier.
- Do not infer advice presence from backend outputs.

## Backend and Kernel Requirements

Stage 4 must keep the backend-first rule:

1. Port or validate the CPU kernel in isolation.
2. Add focused backend/harness microbenchmarks with analytical memory
   accounting.
3. Record the kernel in the backend ledger.
4. Wire `jolt-prover` through backend requests.
5. Accept the frontier only after verifier correctness and performance parity
   both pass.

Required kernel surfaces:

- regular-batch input-claim and RAM initial-evaluation construction:
  `cpu_stage4_regular_batch_input_claims` or a better specialized kernel;
- regular-batch compressed sumcheck:
  `cpu_stage4_regular_batch_sumcheck` or a better specialized kernel;
- register read-write sparse matrix operations preserving the relevant
  `OPT-RW-*` fast paths;
- RAM value-check relation kernels preserving `OPT-REL-006` and `OPT-REL-007`;
- output-opening evaluations:
  `cpu_materialized_opening_evaluations`, unless Stage 4 needs a more
  specialized opening kernel to meet parity;
- field-register read-write kernels under `field-inline`;
- ZK committed round/output-claim work if not covered by the shared BlindFold
  kernel.

Missing parity on any required path is a backend task first. Do not accept a
slower prover-side workaround for transparent, advice, BlindFold/ZK,
field-inline, or `zk + field-inline`.

If a Stage 4 relation or read/write path uses a core prefix/suffix
decomposition, the modular backend must preserve that full algorithm in the
accepted kernel. Dense/materialized versions may remain as reference oracles for
correctness tests, but they cannot be used as production prover paths or
replacement parity evidence.

## Performance Gates

Stage 4 replacement readiness requires the default frontier parity gate:

- timing ratio within 15%;
- peak-memory ratio within 15%;
- `KernelPortStatus::ParityCertified` for every required kernel;
- `validate_frontier_replacement_ready` passes with loaded evidence;
- `validate_global_cpu_backend_inventory_coverage` remains passing.

Current evidence state:

- materialized opening evaluations are parity-certified through
  `cpu_materialized_opening_evaluations`;
- Stage 4 regular-batch input claims are not yet parity-certified;
- Stage 4 regular-batch sumcheck proof generation is not yet
  parity-certified;
- field-inline and ZK Stage 4 paths do not yet have accepted performance
  evidence.

Suggested evidence commands:

```bash
JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage4_regular_batch_input_claims cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet
JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage4_regular_batch_sumcheck cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet
JOLT_WRITE_KERNEL_EVIDENCE=cpu_materialized_opening_evaluations cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet
```

Use the actual evidence writer names when implemented. Do not mark Stage 4
replacement-ready while the input and sumcheck ledger rows remain `Required`.

## Correctness Gates

Focused prover tests should cover:

```bash
cargo nextest run -p jolt-prover stage4 --cargo-quiet
cargo nextest run -p jolt-prover stage4 --features zk --cargo-quiet
cargo nextest run -p jolt-prover stage4 --features field-inline --cargo-quiet
cargo nextest run -p jolt-prover stage4 --features zk,field-inline --cargo-quiet
```

Existing checkpoint and subfrontier tests should continue to pass:

```bash
cargo nextest run -p jolt-prover-harness stage4_regular_batch_input_checkpoint_matches_core_fixtures --features core-fixtures --cargo-quiet
cargo nextest run -p jolt-prover-harness stage4_output_opening_checkpoint_matches_core_fixtures --features core-fixtures --cargo-quiet
cargo nextest run -p jolt-prover-harness stage4_regular_batch_sumcheck_verifier_replay_verifies_against_core_fixtures --features core-fixtures --cargo-quiet
```

Required new/expanded verifier replay tests:

```bash
cargo nextest run -p jolt-prover-harness stage4_full_verifier_replay --features core-fixtures --cargo-quiet
cargo nextest run -p jolt-prover-harness stage4_zk_committed_boundary --features core-fixtures,zk --cargo-quiet
cargo nextest run -p jolt-prover-harness field_inline_eq_poly_guest_stage4 --features core-fixtures,field-inline --cargo-quiet
cargo nextest run -p jolt-prover-harness field_inline_eq_poly_guest_stage4_zk --features core-fixtures,zk,field-inline --cargo-quiet
```

Use actual test names when implemented. The important acceptance condition is
that native `jolt-verifier` accepts the assembled Stage 4 component, not merely
that local helper checkpoints match shapes.

Static rails:

```bash
cargo fmt -q
cargo clippy -p jolt-prover --features zk,field-inline -q --all-targets -- -D warnings
cargo clippy -p jolt-prover-harness --features core-fixtures,zk,field-inline --tests -q -- -D warnings
```

## Implementation Slice Plan

1. Normalize Stage 4 collections and current helpers.
   Replace `HashMap` with `BTreeMap`, keep duplicate/missing/extra slot errors
   targeted, and preserve existing checkpoint tests.

2. Define the canonical API.
   Add `Stage4ProverInput` and `Stage4ProverOutput` so the public `prove`
   entrypoint can consume Stage 2/3 outputs, preprocessing/public I/O, advice
   presence, and witness providers.

3. Compute RAM initial evaluation in the modular path.
   Move RAM public initial evaluation, advice selector derivation, advice
   opening evaluation, and full-eval reconstruction into Stage 4 production
   code using the same helpers as the verifier.

4. Backend evidence first.
   Add or port Stage 4 input-claim and regular-batch sumcheck kernels before
   declaring the prover path replacement-ready. Update manifest optimization
   IDs for RAM value-check and read-write sparse matrix fast paths as needed.

5. Transparent/advice integration.
   Move existing helper logic behind the canonical `prove` flow and keep the
   verifier replay fixture accepted for `MuldivSmall` and `AdviceConsumer`.

6. Output openings.
   Evaluate register read-write, RAM value-check, advice, and field-register
   openings through deterministic backend requests. Assemble `Stage4Claims`
   directly from verifier-owned types and preserve transcript conditional
   order.

7. Field-inline.
   Remove the `not(feature = "field-inline")` blocker from the proof path.
   Add field-register read-write proof generation, output openings, and native
   verifier replay.

8. ZK committed boundary.
   Emit committed Stage 4 proof fields with the verifier's conditional output
   claim count. Retain committed witnesses for BlindFold and add native verifier
   committed-boundary replay through `stage4::verify`.

9. Full replay and parity gates.
   Run performance evidence and ledger gates for every Stage 4 frontier.

10. Justification log.
    Append exact code touched, correctness commands, performance evidence,
    remaining limitations, and confidence statement to this spec before marking
    Stage 4 complete.

## Acceptance Checklist

Stage 4 is accepted only when all of these are true:

- public `prove` is the canonical entrypoint and appears first in `prove.rs`;
- production Stage 4 imports verifier proof/claim/output structs directly from
  `jolt-verifier`;
- semantic ordering, dimensions, formulas, opening IDs, challenge IDs, public
  IDs, and advice conditionals come from `jolt-claims` and `jolt-verifier`;
- public initial RAM and public I/O handling come from `jolt-program` helpers;
- witness data flows through `jolt-witness` views or primitive row providers;
- heavy compute flows through certified `jolt-backends` kernels;
- deterministic `BTreeMap`/`BTreeSet` collection is used for slot-keyed and
  variable-keyed data;
- regular-batch proof generation and all output openings are implemented in
  verifier transcript order;
- trusted and untrusted advice contributions are tested for present and absent
  commitment cases;
- transparent and advice native verifier replay pass with real core fixtures;
- field-inline native verifier replay passes with real field-inline fixture
  data;
- ZK committed-boundary replay runs through Stage 4 and returns
  `Stage4Output::Zk`;
- `zk + field-inline` is tested or concretely documented as unsupported;
- required performance evidence is loaded and parity-certified for every Stage
  4 backend kernel surface;
- this spec contains the final implementation-slice justification.

## Implementation Slice Justification Log

### Slice 1 — Canonical transparent `prove` entrypoint (2026-05)

Status: transparent canonical API landed and verifier-validated. Backend-routing,
field-inline/ZK prover paths, prover-side `ram_val_check_init`, and perf-cert
remain (tracked below). Not yet replacement-ready.

Code touched:

- `crates/jolt-prover/src/stages/stage4/output.rs`: `HashMap`→`BTreeMap`
  (deterministic slot diagnostics); added `Stage4RegularBatchExpectedOutputs` and
  `Stage4ProverOutput` (verifier-owned `stage4_sumcheck_proof` + `Stage4Claims` +
  assembled `Stage4ClearOutput`).
- `crates/jolt-prover/src/stages/stage4/input.rs`: added `Stage4ProverInput`
  (config + `CheckedInputs` + Stage 2/3 clear outputs + `ram_val_check_init` +
  witness).
- `crates/jolt-prover/src/stages/stage4/prove.rs`: added the canonical public
  `prove` (cfg `not(field-inline)`), ordered first. It mirrors
  `jolt-verifier/src/stages/stage4/verify.rs` in prover order: derive
  `registers_gamma`/`ram_val_check_gamma` (with the `ram_val_check_gamma` domain
  separator), prove the registers read-write + RAM value-check batched sumcheck
  (via the verified host helper), evaluate output openings, and assemble
  `Stage4ClearOutput` (including the `ram_val_check_init` round-trip into the
  verifier type) for Stage 5. Surfaced per-instance expected outputs.
- `crates/jolt-prover-harness/src/core_fixture.rs`: routed the Stage 4
  regular-batch verifier replay through the canonical `prove`.

Correctness evidence (all green):

- `cargo nextest run -p jolt-prover-harness stage4_regular_batch_sumcheck_verifier_replay stage4_regular_batch_input_checkpoint stage4_output_opening_checkpoint --features core-fixtures`
  — 3/3 pass. The canonical `prove` output is spliced into a real core proof
  shell and accepted by the native verifier; the assembled `Stage4ClearOutput`
  (gammas, `ram_val_check_init`, batch point, final claim, output claims) matches
  core.
- Compiles under transparent, `field-inline`, `zk`, `zk,field-inline`
  (the canonical `prove` and `Stage4ProverInput`/`Output` are `cfg(not(field-inline))`,
  matching the current transparent-only Stage 4 prover; field-inline compiles out
  cleanly).

Remaining before replacement-ready (not yet done):

- Backend-route the Stage 4 batched sumcheck through the `jolt-backends`
  `SumcheckRegularBatchState` kernel (currently uses the host-side
  `Stage4BatchContext`). Stage 4 instances have different per-instance round
  counts (registers read-write over address+cycle vs RAM value-check over cycle),
  so the kernel offset logic applies. Then perf-cert
  `cpu_stage4_regular_batch_{input_claims,sumcheck}`.
- Self-contained prover-side `ram_val_check_init` computation (public initial RAM
  evaluation + advice-oracle openings); currently supplied via the input bundle.
- Field-inline Stage 4 prover path (field-registers read-write relation) and ZK
  committed-boundary.

Replacement-ready gate status: the `stage4_output_openings` and
`stage4_regular_batch_inputs` frontiers are replacement-ready.

- `stage4_output_openings`: `validate_frontier_replacement_ready` passes against
  the `ParityCertified` `cpu_materialized_opening_evaluations` kernel (test
  `stage4_output_opening_frontier_is_replacement_ready_with_certified_kernel_evidence`).
- `stage4_regular_batch_inputs`: now `ParityCertified`. The input-claim derivation
  (`derive_stage4_regular_batch_prefix`, reconstructing `ram_val_check_init` and
  appending the `ram_val_check_gamma` domain separator) was benchmarked against the
  verifier-mirroring reference prefix via
  `Stage4RegularBatchInputKernelBenchmarkFixture`. Evidence
  `cpu_stage4_regular_batch_input_claims/frontier_perf_stage4_regular_batch_inputs.json`
  reports latest post-clean status `Warn`, time ratio `1.0928888888888888`,
  and memory ratio `1.0`. The loader asserts `run_reference_prefix() == run_modular_prefix()
  == expected` so the timing comparison is over verified-equal outputs. Gate test
  `stage4_regular_batch_input_frontier_is_replacement_ready_with_certified_kernel_evidence`
  passes.

The `stage4_regular_batch_sumcheck` frontier is now certified for the isolated
frontier harness via the specialized sparse backend path described in Slice 3.

### Slice 2 — Generic backend regular-batch routing attempt (2026-05)

Status: correctness-preserving but rejected for performance parity. Not
replacement-ready.

Code touched:

- `crates/jolt-prover/src/stages/stage4/prove.rs`: replaced the local
  prover-side round loop with `jolt-backends` regular-batch round evaluation and
  bind calls. The prover now builds two `SumcheckRegularBatchInstance`s
  (registers read-write and RAM value-check), applies the same front-padding
  scaling for the shorter RAM value-check instance, and keeps the verifier-owned
  transcript/output assembly unchanged.
- `crates/jolt-prover-harness/src/core_fixture.rs`: added an isolated Stage 4
  sumcheck benchmark fixture with a native `jolt-core` Stage 4 reference path and
  a modular `prove_stage4_regular_batch_sumcheck_for_frontier` path. The reference
  accumulator is seeded only with prior Stage 2/3 openings so core caches Stage 4
  outputs normally.
- `crates/jolt-prover-harness/benches/frontier_perf.rs`: added the
  `JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage4_regular_batch_sumcheck` writer and
  analytical memory accounting.

Correctness evidence:

- `cargo nextest run -p jolt-prover-harness stage4 --features core-fixtures --cargo-quiet`
  passes 8/8, including transparent and advice fixture verifier replay. The
  backend-routed Stage 4 proof shell is accepted by the native verifier and the
  output claims still match core.
- `cargo check -p jolt-prover-harness --features core-fixtures --benches --tests -q`
  passes.

Performance evidence:

- `JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage4_regular_batch_sumcheck cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet`
  fails the canonical parity gate: status `Fail`, time ratio
  `74.91679360579532`, peak-RSS ratio `57.890239691669734`.

Conclusion:

- Do not promote `cpu_stage4_regular_batch_sumcheck`; leave it `Required`.
- The generic `SumcheckRegularBatchState` route is useful as a verifier-accepted
  correctness reference, but it materializes dense register/RAM views and is not
  the Stage 4 replacement path.
- Next implementation must port/add a specialized `jolt-backends::cpu` Stage 4
  kernel: sparse/phase-aware registers read-write plus cycle-only RAM value-check,
  with the same front-padding and transcript semantics. Only then rerun the
  canonical evidence writer and consider ledger certification.

### Slice 3 — Specialized sparse Stage 4 backend sumcheck (2026-05)

Status: replacement-ready for the transparent Stage 4 regular-batch sumcheck
frontier and wired into the canonical transparent `prove` path. Full Stage 4
acceptance still requires the required field-inline/ZK surfaces and prover-side
RAM initial-evaluation self-containment.

Code touched:

- `crates/jolt-backends/src/cpu/read_write_matrix/registers.rs`: added the
  phase-aware sparse registers read-write state, preserving the core
  cycle-major/address-major/materialized phase split and lookup-table coefficient
  compression.
- `crates/jolt-backends/src/cpu/read_write_matrix/ram.rs`: added the cycle-only
  RAM value-check state, avoiding dense RAM address materialization by folding the
  fixed address point into the per-cycle RA weight.
- `crates/jolt-backends/src/traits.rs` and `crates/jolt-backends/src/cpu/sumcheck/mod.rs`:
  added the Stage 4 backend state/evaluate/bind/output trait surface and CPU
  implementation.
- `crates/jolt-witness/src/protocols/jolt_vm/mod.rs`: added compact trace-backed
  register read/write rows so the backend receives real SDK/core fixture data,
  not materialized dense virtual polynomials.
- `crates/jolt-prover/src/stages/stage4/prove.rs`: switched both the canonical
  transparent `prove` path and the
  `prove_stage4_regular_batch_sumcheck_for_frontier` harness path to the
  specialized backend states with the same two-instance batching, RAM
  front-padding, and transcript order as the verifier. The obsolete dense
  fallback construction was removed.

Correctness evidence:

- `cargo check -p jolt-prover --features frontier-harness -q` passes.
- `cargo check -p jolt-prover-harness --features core-fixtures --benches --tests -q`
  passes.
- `cargo nextest run -p jolt-prover-harness stage4 --features core-fixtures --cargo-quiet`
  passes 9/9, including transparent/advice Stage 4 verifier replay against real
  core fixtures and the certified replacement-ready sumcheck gate.
- `cargo clippy -p jolt-prover-harness --features core-fixtures --tests --benches -q -- -D warnings`
  passes.

Performance evidence:

- `JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage4_regular_batch_sumcheck cargo bench -p jolt-prover-harness --features core-fixtures --bench frontier_perf --quiet`
  writes
  `target/frontier-metrics/kernel-evidence/cpu_stage4_regular_batch_sumcheck/frontier_perf_stage4_regular_batch_sumcheck.json`
  with latest post-clean status `Pass`, time ratio `0.7841373275249024`, and
  peak-RSS ratio `1.0154954432074184`.
- `cpu_stage4_regular_batch_sumcheck` is promoted to `ParityCertified`, and
  `stage4_regular_batch_sumcheck_frontier_is_replacement_ready_with_certified_kernel_evidence`
  validates the frontier against the certified ledger evidence.

Conclusion:

- The transparent Stage 4 regular-batch sumcheck now has correctness parity and
  CPU timing/memory parity for transparent and advice fixtures, and production
  transparent proving uses the same backend route as the certified frontier.
- Remaining Stage 4 work before stage acceptance is the required field-inline,
  ZK/BlindFold-boundary, and combined-feature surfaces plus prover-side
  self-containment for RAM value-check initial evaluation.

### Slice 4 — Field-inline field-register read-write kernel (2026-05)

Status: the field-inline Stage 4 clear frontier now routes field-register
read/write through a backend-owned sparse kernel instead of dense prover-local
materialization. This is a ported correctness slice, not yet a parity-certified
field-inline replacement frontier.

Code touched:

- `crates/jolt-backends/src/cpu/read_write_matrix/field_registers.rs`: added a
  field-register read/write state following the existing register sparse
  read/write shape: cycle-major phase, address-major phase, materialization only
  after phase transition, one-hot coefficient lookup tables, and native
  field-valued pre/post/increment terms.
- `crates/jolt-backends/src/traits.rs` and
  `crates/jolt-backends/src/cpu/sumcheck/mod.rs`: added the backend trait surface
  and CPU implementation for field-register state materialization,
  round-evaluation, binding, and output openings.
- `crates/jolt-witness/src/protocols/jolt_vm/field_inline/mod.rs`: added sparse
  trace-backed field-register read/write rows so the backend consumes real
  traced field-inline guest data.
- `crates/jolt-prover/src/stages/stage4/prove.rs`: extended the field-inline
  clear Stage 4 path to batch registers, field-registers, and RAM value-check in
  verifier order, with the field-register front-padding and opening point derived
  from `FieldInlineConfig`.
- `crates/jolt-prover-harness/tests/field_inline_sdk_guest.rs`: extended the
  real `field-inline-eq-poly-guest` SDK replay through Stage 4 and compared the
  modular prover output against native `jolt-verifier`.

Correctness evidence:

- `cargo nextest run -p jolt-backends --features field-inline field_registers_read_write_matches_dense_reference --cargo-quiet`
  passes, checking the sparse backend state against a dense reference.
- `cargo nextest run -p jolt-prover-harness --features core-fixtures,field-inline field_inline_eq_poly_guest_replays_modular_frontier_with_jolt_verifier --cargo-quiet`
  passes, proving Stage 0 through Stage 4 on real field-inline SDK trace data and
  replaying the resulting components through native `jolt-verifier`.
- `cargo check -p jolt-prover --features zk,field-inline -q` and
  `cargo check -p jolt-prover-harness --features core-fixtures,zk,field-inline -q`
  pass, so the combined feature surface remains compile-clean at this frontier.
- `cargo clippy -p jolt-prover --features zk,field-inline -q --all-targets -- -D warnings`
  and
  `cargo clippy -p jolt-prover-harness --features core-fixtures,field-inline -q --all-targets -- -D warnings`
  pass.

Performance evidence:

- The new `cpu_field_inline_stage4_registers_read_write` ledger row is marked
  `Ported` and accounts for `OPT-RW-001` through `OPT-RW-010` plus
  `OPT-FLD-002` and `OPT-FLD-003`.
- No canonical field-inline Stage 4 kernel evidence has been written yet, so
  this row must not be promoted to `ParityCertified` and field-inline Stage 4
  must not be claimed replacement-ready on timing or memory until
  `frontier_perf/stage4_field_inline_registers_read_write` or the equivalent
  full-frontier evidence passes the 15% gate.

Remaining work:

- Add canonical benchmark evidence for the field-register read/write kernel and
  promote `cpu_field_inline_stage4_registers_read_write` only after the evidence
  passes.
- Replace the field-inline zero placeholders in
  `stage4_output_openings_from_evaluations` before extending the output-opening
  checkpoint frontier to field-inline.
- Implement the committed/ZK Stage 4 prover assembly and native verifier
  boundary replay for `zk` and `zk + field-inline`.

### Slice 5 — Field-inline read/write direct-coefficient parity evidence (2026-05-30)

Status: field-inline Stage 4 read/write is now performance-certified on real
`field-inline-eq-poly-guest` data. The accepted kernel keeps the sparse
cycle-major/address-major read/write algorithm but specializes the field-inline
coefficient representation: the four field-register RA/WA coefficients are
stored directly in sparse entries instead of going through the generic one-hot
coefficient lookup table. The generic table path reached a 65,536-entry table
before dereferencing and failed the canonical gate; direct coefficients preserve
the same verifier relation with far lower peak memory.

Code touched:

- `crates/jolt-backends/src/cpu/read_write_matrix/field_registers.rs`: removed
  the field-inline read/write one-hot coefficient table path, stores direct
  `F` coefficients in sparse entries, avoids per-row temporary `Vec`
  allocation, and keeps only the `rs2` register column needed for output
  splitting instead of cloning all rows into prover state.
- `crates/jolt-prover-harness/benches/frontier_perf.rs`: added the canonical
  `cpu_field_inline_stage4_registers_read_write` evidence writer over real SDK
  trace data. The writer checks modular output against a dense factor-product
  reference before measuring.
- `crates/jolt-prover-harness/src/optimization.rs`: promoted
  `cpu_field_inline_stage4_registers_read_write` to `ParityCertified`.

Correctness evidence:

- `cargo check -p jolt-backends --features field-inline -q`
- `cargo check -p jolt-prover-harness --features core-fixtures,field-inline --benches -q`
- `cargo nextest run -p jolt-backends --features field-inline field_registers_read_write_matches_dense_reference --cargo-quiet`
- The canonical evidence writer asserts equality between the dense
  factor-product reference and the modular sparse direct-coefficient run before
  measuring performance.

Performance evidence:

- `JOLT_WRITE_KERNEL_EVIDENCE=cpu_field_inline_stage4_registers_read_write cargo bench -p jolt-prover-harness --features core-fixtures,field-inline --bench frontier_perf --quiet`
  wrote
  `target/frontier-metrics/kernel-evidence/cpu_field_inline_stage4_registers_read_write/frontier_perf_stage4_field_inline_registers_read_write.json`
  with status `Pass`.
- Reported ratios: time `0.24893276074198065`, peak RSS
  `0.024489569462938305`.

Replacement status: field-inline Stage 4 read/write CPU timing and memory
parity are now certified for the real field-inline SDK frontier. Slice 6 below
adds committed/ZK prover assembly and native verifier boundary replay for the
`zk + field-inline` path; full BlindFold verification remains Stage 8/full
JoltProof work.

### Slice 6 — Real Stage 4 committed-boundary proof production (2026-05-30)

Status: Stage 4 now has a modular committed-boundary prover path for both the
standard and `field-inline` feature shapes. The path keeps the transparent hot
path untouched and mirrors `stage4::verify`'s ZK ordering: register gamma,
field-register gamma when enabled, RAM value-check gamma domain separator,
batching coefficients, committed round polynomial transcript, and committed
output-claim rows.

Code touched:

- `crates/jolt-prover/src/stages/stage4/prove.rs`: added
  `prove_committed_boundary` under `zk`, with separate standard and
  `field-inline` signatures matching the existing clear entrypoint shape. The
  committed path reuses the certified register read/write, field-register
  read/write, and RAM value-check backend states, commits the real batched round
  polynomial each round, recomputes local expected outputs, and rejects final
  claim mismatches before emitting verifier proof data.
- `crates/jolt-prover/src/stages/stage4/output.rs`: added
  `Stage4CommittedBoundaryOutput`, carrying the verifier-owned
  `stage4_sumcheck_proof`, `Stage4PublicOutput`, hidden output-claim values, and
  retained committed witness data for the later BlindFold assembly slice.
- `crates/jolt-prover-harness/tests/field_inline_sdk_guest.rs`: extended the
  real SDK field-inline ZK verifier replay through Stage 4 using the modular
  committed Stage 3 and Stage 4 proof components.

Correctness evidence:

- `cargo check -p jolt-prover --features zk,field-inline -q` passes.
- `cargo check -p jolt-prover-harness --features core-fixtures,field-inline,zk -q`
  passes.
- `cargo clippy -p jolt-prover --features zk,field-inline -q --all-targets -- -D warnings`
  passes.
- `cargo clippy -p jolt-prover-harness --features core-fixtures,field-inline,zk -q --all-targets -- -D warnings`
  passes.
- `cargo nextest run -p jolt-prover-harness field_inline_eq_poly_guest_accepts_zk_committed_stage3_to_stage5_boundaries --features core-fixtures,field-inline,zk --cargo-quiet`
  passes. The test runs the real `field-inline-eq-poly-guest` trace, produces
  real modular committed Stage 3 and Stage 4 proofs, and requires native
  `jolt-verifier` to return `Stage3Output::Zk` and `Stage4Output::Zk`. For Stage
  4 it checks the public verifier output matches the modular prover output, the
  12 hidden output claims are packed into one row at fixture VC capacity, and
  the prover/verifier transcript states match after Stage 4.

Performance evidence:

- No new arithmetic kernel was introduced in this slice. The Stage 4 committed
  path reuses the already-certified transparent/advice read/write and RAM
  value-check kernels plus the Slice 5 certified field-register direct-coefficient
  kernel. The added work is vector-commitment construction for the ZK committed
  boundary and retained witnesses for BlindFold.

Scope note:

This proves native verifier acceptance at the Stage 4 committed boundary with
real modular Stage 4 data on the real SDK `zk + field-inline` path. It does not
yet prove full BlindFold correctness: Stages 5-8 still need their
committed-boundary prover components, and Stage 8 still owns final BlindFold
assembly and verification.

### Post-clean manifest rail tightening

The Stage 4 regular-batch sumcheck frontier now names the RAM value-check
relation optimizations and register read-write sparse-matrix optimizations it
depends on, not only the generic sumcheck and eq-table IDs. The manifest,
backend kernel ledger, and evidence writer all use the same optimization ID set:

- `OPT-SC-007`
- `OPT-EQ-004`
- `OPT-RW-001` through `OPT-RW-010`
- `OPT-REL-006`
- `OPT-REL-007`

The corresponding ledger row points at the actual Stage 4 source families:
register read-write, sparse read/write matrix infrastructure, and RAM
value-check. The stale bytecode read-RAF source pointer was removed because
bytecode read-RAF is Stage 6 ownership, not Stage 4 ownership.

This does not change protocol code. It tightens the replacement gate so Stage 4
cannot be accepted without certified evidence for the register read-write and
RAM value-check kernels that appear in this stage's prover loop.
