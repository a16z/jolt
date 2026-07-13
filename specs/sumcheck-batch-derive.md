# Spec: SumcheckBatch derive macro

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @moodlezoup                    |
| Created     | 2026-06-29                     |
| Status      | proposed                       |
| PR          |                                |

> **NOTE (2026-07-03):** the implementation has evolved past this spec: the
> aggregates are single-generic (`StageNOutputClaims<F>`, with the `*Points`
> aggregates carrying the `Vec<F>` cell form), the batched verification drivers
> described below as deferred ARE generated (`verify_clear` / `verify_zk` /
> `derive_opening_points` / `expected_final_claim`), and the shared per-member
> logic lives as generic functions in `jolt-verifier`'s `stages::relations`.
> The crate-level docs of `jolt-verifier-derive` are the current reference.

## Summary

Each verifier stage is a batch of one or more sumcheck instances, and the data
flowing between stages is conceptually uniform: the **output claims** of an
upstream sumcheck are consumed as the **input claims** of a downstream sumcheck,
and the **Fiat-Shamir challenges** each stage draws are passed downstream.
Beyond that, a stage consumes only a small handful of auxiliary data. Today the
verifier hand-writes, per stage, a family of aggregate types
(`StageNOutputClaims`, `StageNChallenges`, ad-hoc input locals) and the
delegating canonical-order / transcript plumbing over them. These aggregates are
mechanical compositions of the per-relation `SymbolicSumcheck` / `ConcreteSumcheck`
claim structs the prior refactors (#1640, #1653, and the symbolic /
instance-data-model specs) introduced, but writing them by hand lets the four
load-bearing orderings (challenge draw, claim array, point slicing, output
recombination) drift, and it is the boilerplate the instance-first data model was
built to eliminate. This feature introduces a `#[derive(SumcheckBatch)]` proc
macro (new `jolt-verifier-derive` crate) that generates those aggregate types and
their plumbing from a single per-stage source-of-truth struct listing the stage's
`ConcreteSumcheck` instances, reflecting the conceptual simplicity of the
inter-stage dataflow directly in the type system while leaving the irreducible 5%
to small, explicit escape hatches.

## Intent

### Goal

Add a `#[derive(SumcheckBatch)]` proc macro that, from a per-stage struct whose
fields are the stage's `ConcreteSumcheck` instances (e.g.
`Stage5Sumchecks<F> { instruction_read_raf: InstructionReadRaf<F>,
ram_ra_claim_reduction: RamRaClaimReduction<F>, registers_val_evaluation:
RegistersValEvaluation<F> }`), generates the per-stage aggregate claim types
`StageNInputClaims<F, C>`, `StageNOutputClaims<F, C>`, and `StageNChallenges<F>`
— each with one field per instance, projected through the existing
`ConcreteSumcheckInputs` / `ConcreteSumcheckOutputs` / `ConcreteSumcheckChallenges`
aliases. The macro always emits the struct definitions; the set of *delegated
implementations* it emits (Fiat-Shamir opening plumbing, id resolution, etc.)
starts minimal and grows iteratively to whatever the migration actually consumes.
Migrate stages 1–7 to consume the generated aggregates — including refactoring the
per-relation `*_from_upstream` wiring helpers to construct the generated
`StageNInputClaims` aggregates — and handle each stage's residual auxiliary data
via inherent impls on the generated struct or in the stage's `verify` body.

Key abstractions introduced / modified:

- **`jolt-verifier-derive`** — a new `proc-macro = true` crate (sibling to
  `jolt-claims-derive`) owning `#[derive(SumcheckBatch)]`. Its generated code
  lands in `jolt-verifier`, so the aggregate types remain jolt-verifier types
  (batching is a concrete concern); the derive references `ConcreteSumcheck` /
  `SymbolicSumcheck` through `crate::`-rooted paths.
- **`StageNSumchecks<F>`** — the per-stage source-of-truth struct holding the
  stage's `ConcreteSumcheck` wrapper instances. It is the single input that drives
  this macro now and the deferred verification driver later (the driver needs the
  concrete `input_claim` / `expected_output` / `draw_challenges` methods, and
  `concrete → symbolic` is one-way via `::Symbolic`).
- **`StageNInputClaims<F, C>` / `StageNOutputClaims<F, C>` / `StageNChallenges<F>`**
  — the generated cell-generic input/output aggregates and the field-generic
  challenge aggregate, one field per instance.

### Invariants

This is a behavior-preserving refactor: the generated aggregates must produce the
**byte-identical verifier behavior** as the hand-written code they replace.

Existing `jolt-eval` invariants:

- **`soundness`** (RedTeam) — must continue to hold unchanged. It is the binding
  correctness invariant: for any deterministic guest + input, only one
  (output, panic) pair is accepted. The refactor must not change which proofs
  verify. No modification to the invariant itself is required.

Properties the implementation must preserve (enforced by tests + `soundness`, not
new `jolt-eval` invariants — they are "no behavior change" properties, already
covered):

- **Fiat-Shamir order is byte-identical.** The generated `opening_values` /
  `append_to_transcript` must emit exactly the same scalars, under the same
  labels, in the same order as today. The macro single-sources this from member
  field-declaration order; the order must match the prover's commitment order.
- **Canonical order ↔ values consistency.** `canonical_order()[k]` is the id of
  `opening_values()[k]` for every generated aggregate (the existing
  `OutputClaims` / `InputClaims` contract, lifted to aggregates).
- **BlindFold claim/constraint sync is untouched.** The macro only *aggregates*
  per-instance claim structs; it never authors `input_expression` /
  `output_expression` or any constraint logic. The `StageNChallenges` shape
  change from flat fields (`instruction_gamma`) to nested per-instance members
  (`instruction_read_raf.gamma`) is mirrored in each migrated stage's
  `blindfold/stageN.rs` reader.

No new `jolt-eval` invariant is required; the change is verified by the existing
`soundness` red-team invariant plus the e2e suite (see Testing Strategy). A
codegen-level property — "generated `opening_values` follows field-declaration
order" — is captured by unit tests in `jolt-verifier-derive`, not by `jolt-eval`.

### Non-Goals

- **The batched sumcheck verification driver.** Generating the
  draw → `input_claim` → `SumcheckClaim` → `try_instance_point` →
  `derive_opening_points` → `expected_output` → fold → check → append control
  flow is explicitly deferred to a follow-up. The `StageNSumchecks` struct is
  designed to drive it later.
- **Stage 8.** It is a PCS RLC opening proof, not a sumcheck batch — no
  rounds/degree/input_claim/output_claim, no per-instance claim aggregates. Out
  of scope entirely.
- **Removing `OpeningClaim<F>` / collapsing the opening cell to two forms, and
  making the aggregates single-generic (`<F>`).** A separate future refactor.
  This spec accepts the two-generic `StageNOutputClaims<F, C>` shape that concrete
  wrapper fields force.
- **Any prover-side logic change and any change to the serialized proof's value
  sequence.** The wire form's scalar order/semantics must be identical; only the
  Rust type that holds it gains an `F` parameter (`<F>` → `<F, F>`).
- **Performance changes.** Verifier glue is not a hot path.

## Evaluation

### Acceptance Criteria

- [ ] A new `jolt-verifier-derive` crate (`proc-macro = true`) exports
  `#[derive(SumcheckBatch)]`; `jolt-verifier` depends on it.
- [ ] From a `StageNSumchecks<F>` struct of `ConcreteSumcheck` fields, the derive
  generates `StageNInputClaims<F, C>`, `StageNOutputClaims<F, C>`, and
  `StageNChallenges<F>` with one field per instance projected through
  `ConcreteSumcheckInputs` / `ConcreteSumcheckOutputs` / `ConcreteSumcheckChallenges`.
- [ ] The macro always emits the three struct definitions and supports
  `Option<member>` fields (conditional instances/openings). The set of generated
  delegated impls starts **minimal** — initially the Fiat-Shamir opening plumbing
  consumed today (`Output` `opening_values` / `append_to_transcript`) — and grows
  iteratively as consumers require; impls only a deferred driver would use (e.g.
  aggregate `resolve_output`) are not emitted up front.
- [ ] The per-relation `*_from_upstream` wiring helpers are refactored to
  construct the generated `StageNInputClaims` aggregates (the canonical place the
  Outputs→Inputs dataflow is expressed) rather than ad-hoc per-relation locals.
- [ ] Stage 5 is migrated to the generated aggregates as the reference case, with
  byte-identical Fiat-Shamir behavior (`muldiv` e2e passes in both modes).
- [ ] Stages 1–7 are migrated to derive their applicable aggregate families
  (`InputClaims` for 2–7; `OutputClaims` and `Challenges` for 1–7) via
  `#[derive(SumcheckBatch)]`. Per-stage residuals (alias-curated opening order +
  `validate`, the bare program-image opening, multi-phase / enum / array
  openings, non-instance challenge `Vec`s, the Spartan-outer R1CS assembly) are
  handled by inherent impls on the generated struct or in the `verify` body, per
  the coverage map in Architecture.
- [ ] Each migrated stage's `blindfold/stageN.rs` reads the new nested challenge
  fields (e.g. `challenges.instruction_read_raf.gamma`).
- [ ] `muldiv` e2e passes in both `--features host` and `--features host,zk`; the
  `advice` e2e (non-ZK + advice) passes.
- [ ] `cargo clippy --all --features host` and `--features host,zk` are clean
  (`-D warnings`); `cargo fmt` clean.
- [ ] `jolt-verifier-derive` has unit tests asserting the generated aggregates'
  `opening_values` follows field-declaration order, handles `Option<member>`
  presence/absence, and threads `F` / `C` correctly.
- [ ] Net reduction in hand-written aggregate/plumbing lines across
  `crates/jolt-verifier/src/stages/`.

### Testing Strategy

Existing tests that must keep passing:

- The full `jolt-verifier` test suite, including the stage `outputs.rs` ordering
  lock tests (`opening_values_follow_canonical_order`) and `validate` tests
  (stage 2 / stage 3), and the `relations.rs` `draw_challenges` /
  `append_openings` recorder tests.
- `crates/jolt-verifier/tests/support/{proof_claims,tamper_manifest}.rs` — these
  construct `StageNOutputClaims` and must be updated for the `<F>` → `<F, F>`
  signature change.
- The prover's `crates/jolt-prover-legacy/src/zkvm/clear_claims.rs`, which
  constructs `StageNOutputClaims<F>` (same signature update).
- `crates/jolt-verifier/src/proof.rs` (`stageN: StageNOutputClaims<F>` serialized
  field) — signature update; serialized bytes unchanged.

New tests:

- `jolt-verifier-derive` codegen unit tests (declaration-order, `Option<member>`,
  generic threading).
- Per migrated stage: the existing ordering-lock test is retargeted to the
  generated aggregate (or kept as-is if the struct is drop-in).

Mode coverage: `muldiv` in **both** `--features host` and `--features host,zk` is
the primary correctness gate (catches Fiat-Shamir/transcript desync); the
`advice` e2e exercises the non-ZK advice path. Both are mandatory after every
stage migration.

### Performance

This is verifier glue, not a prover hot path.

- Existing `jolt-eval` performance objectives: none expected to move. The
  `bind_parallel_*` and `prover_time_*` benchmarks are prover-side; the `lloc`
  code-quality objective targets `crates/jolt-prover-legacy/src/`, while this
  refactor touches `crates/jolt-verifier/`, so `lloc` is unaffected.
- No new `jolt-eval` objective. "No regression" is sufficient and needs no new
  benchmark — the change is type-level and the generated plumbing is the same
  delegating chain as today. (The verifier-side LLOC reduction is real but is not
  a tracked objective; the `lloc` objective's target directory would need to
  include `crates/jolt-verifier/src/` to capture it, which is out of scope here.)

## Design

### Architecture

**Crate layout.** A new `jolt-verifier-derive` crate (`proc-macro = true`,
depending on `syn` / `quote` / `proc-macro2`) owns `#[derive(SumcheckBatch)]`.
`jolt-verifier` adds it as a dependency. A proc-macro derive cannot live in a
normal library crate, so the macro *source* lives here; its generated *output*
lands wherever the derive is invoked (always `jolt-verifier`), so the aggregate
types remain jolt-verifier types. The derive emits `crate::stages::relations::…`
(or `::jolt_verifier::…` via `extern crate self`) for `ConcreteSumcheck` /
`ConcreteSumcheckInputs` / `…Outputs` / `…Challenges`, and `::jolt_claims::…` /
`::jolt_field::…` for the claim-data traits, mirroring the absolute-path
convention `jolt-claims-derive` already uses.

**Source of truth.** Each stage declares one struct of its `ConcreteSumcheck`
instances:

```rust
#[derive(SumcheckBatch)]
struct Stage5Sumchecks<F: Field> {
    instruction_read_raf:     InstructionReadRaf<F>,
    ram_ra_claim_reduction:   RamRaClaimReduction<F>,
    registers_val_evaluation: RegistersValEvaluation<F>,
}
```

The `verify` body instantiates it (`InstructionReadRaf::new(dims)`, …) and reads
inputs/challenges through it, so the struct is live now and is the same object the
future driver will iterate.

**Generated types.** Concrete wrapper fields carry `F`, so the projection the
derive must emit (`<InstructionReadRaf<F> as ConcreteSumcheck<F>>::Symbolic …`)
mentions `F`; therefore the input/output aggregates are two-generic and the
challenge aggregate is one-generic:

```rust
struct Stage5InputClaims<F: Field, C>  { instruction_read_raf: ConcreteSumcheckInputs<F, InstructionReadRaf<F>, C>,  … }
struct Stage5OutputClaims<F: Field, C> { instruction_read_raf: ConcreteSumcheckOutputs<F, InstructionReadRaf<F>, C>, … }
struct Stage5Challenges<F: Field>      { instruction_read_raf: ConcreteSumcheckChallenges<F, InstructionReadRaf<F>>, … }
```

Existing `StageNOutputClaims<C>` therefore becomes `<F, C>`, rippling `<F>` →
`<F, F>` (wire), `<OpeningClaim<F>>` → `<F, OpeningClaim<F>>` (clear), and
`<Vec<F>>` → `<F, Vec<F>>` (points) at the ~handful of use sites in `proof.rs`,
`verifier.rs`, the prover's `clear_claims.rs`, and the test support files.
(`PhantomData<F>` is added only if E0392 fires on a projection-only `F`.)

**Generated impls (minimal, then iterate).** The macro always emits the three
struct definitions. The *delegated impls* start as a minimal set and grow only as
the migration actually consumes them — over-generating risks emitting unused or
subtly-wrong plumbing and forcing needless opt-outs. The guiding tension is
single-sourced logic vs. escape hatches: more generated impls means fewer
hand-written escape hatches but more macro surface; we converge on the balance
empirically rather than specifying the full set up front.

- **Initial set**: the Fiat-Shamir opening plumbing consumed today —
  `Output` `opening_values` / `append_to_transcript`, delegating to each member in
  declaration order, with the per-method opt-out for alias-curated stages.
- **Add as needed**: aggregate `canonical_order`, `zip_openings` /
  `*_output_claims_with_points`, the `Challenges` `from_transcript_values`
  builder, and `resolve_input` / `resolve_output` / `resolve_challenge` — these
  last three are consumed by the deferred driver's `input_claim` /
  `expected_output`, so they are **not** emitted up front.

`Option<member>` fields are chained when `Some` in whichever impls are emitted;
instances with no challenges use `NoChallenges<F>`.

**Input wiring.** The `*_from_upstream` helpers (e.g.
`ram_ra_claim_reduction_inputs_from_upstream`) are the cross-stage wiring that
says which upstream opening feeds which downstream input — the explicit form of
the Outputs→Inputs dataflow. They are refactored to construct the generated
`StageNInputClaims` aggregates directly (one helper assembling the stage's
aggregate, or per-relation helpers feeding it), so the generated type is the
single place that dataflow is expressed rather than ad-hoc per-relation locals.

**Field grammar.** A field is one of:

- a per-instance member — `<Instance>::{Inputs/Outputs/Challenges}` — delegated to;
- `Option<member>` — a conditional instance/opening group, chained when `Some`.

Anything else is an **escape hatch**, handled outside the generated plumbing:

- An inherent impl on the generated struct (e.g. `validate()`, an alias-curated
  `opening_values`/`append` when a stage opts out of the generated one, extra
  `*_point()` accessors, auxiliary derived getters).
- The stage's `verify` body (the `*_inputs_from_upstream` wiring that says which
  upstream opening feeds which input, injected transcript domain separators,
  multi-phase point offsets, R1CS-input assembly).

To make alias-curation composable, the derive supports opting a stage out of the
generated `opening_values` / `append_to_transcript` / `canonical_order` (so the
stage supplies its own consistent trio) while still generating the struct and the
`resolve_*` half.

**Per-stage coverage map** (the implementation plan; validated by `muldiv`):

| Stage | Inputs | Outputs | Challenges | Principal escape hatches |
|---|---|---|---|---|
| 5 | ✅ members | ✅ members | ✅ members + `NoChallenges` | none |
| 7 | ✅ members + `Option` | ✅ members + `Option` | ✅ member + `NoChallenges` | none |
| 3 | ✅ members | ✅ members; opt-out opening order | ✅ members | inherent alias-curated `opening_values` + `validate` |
| 4 | ✅ members | ✅ members + bare `Option<C>` | ✅ members | bare program-image opening; `advice` sub-group; `ram_val_check_gamma` separator in `verify` |
| 2 | ✅ members | ✅ inner-batch members; opt-out opening order | ⚠️ partial: member gammas + non-instance scalars/`Vec` | inherent alias `opening_values`+`validate`; uniskip scalar + `tau_high` + `output_address_challenges` in `verify` |
| 6 | ✅ members + `Option` | ⚠️ members + `Option` + phase-group/enum/array | ⚠️ partial: member gammas + cross-stage `Vec`s + powers | inherent for address-phase group, `BytecodeCyclePhase` enum, `[C; N]`; gamma `Vec`s in `verify` |
| 1 | — (no inputs) | ⚠️ uniskip scalar + Spartan-outer R1CS struct | ⚠️ partial: `tau` `Vec` + uniskip scalar | R1CS-input assembly + `tau` draw in `verify` |
| 8 | excluded | excluded | excluded | — |

Stages 5, 7 are clean (members + `Option`); 3, 4 add small inherent/verify
escape hatches; 2, 6, 1 concentrate the residual but still derive their
per-instance members and resolve their non-instance data in inherent impls or
`verify`.

### Alternatives Considered

- **Symbolic relation types in the source struct** (no `F` param on the
  aggregates, no `<F>`→`<F,F>` ripple). Rejected: the source struct must also
  drive the deferred verification driver, which calls `ConcreteSumcheck` methods
  (`input_claim`, `expected_output`); `concrete → symbolic` is one-way
  (`::Symbolic`), so symbolic tokens would strand the driver. The aggregate types
  still live in jolt-verifier either way, so symbolic tokens buy nothing here.
- **Moving `F` from the `ConcreteSumcheck` trait into its methods** (so concrete
  wrappers drop `F` and aggregates stay single-generic). Rejected: 6 of 32
  wrappers store real field data (`RamValCheck`'s `public_eval`/`selector`, the
  Spartan uniskip/remainder `tau_high`/`uniskip_challenge`), so a method-generic
  `F` cannot unify with the struct's stored `F`; the de-genericization is not
  uniformly implementable and stage 4 keeps `F` regardless.
- **Concretizing `C` into per-cell named structs** (`StageNOutputValues<F>` +
  `StageNOutputPoints<F>`) to avoid the two-generic. Deferred: best done together
  with removing the redundant `OpeningClaim<F>` cell (a separate, pervasive
  refactor touching every `ConcreteSumcheck` method signature).
- **`macro_rules!` instead of a proc macro.** Rejected: the derive must generate
  struct definitions with GAT-projection field types, multiple output types, and
  field-level opt-outs, with good error spans — a proc macro fits; the existing
  `jolt-claims-derive` machinery (`named_fields`, generic-param threading,
  where-clause synthesis) is directly reusable.
- **Attribute-encoded alias-skip** vs **inherent-impl escape hatch.** Chose
  inherent impls on the generated struct (and the `verify` body) for the residual
  5%, keeping the macro's grammar small and the escape hatches explicit and
  reviewable, rather than growing per-field attributes (`#[alias]`,
  `#[domain_separator]`, …) that would re-encode each stage's idiosyncrasies in
  the macro.

## Documentation

No Jolt book (`book/`) changes — this is an internal verifier refactor with no
user-facing surface. Module-level docs on `jolt-verifier-derive` (mirroring
`jolt-claims-derive`'s doc) describe the field grammar and escape-hatch
mechanism. The `symbolic-sumcheck.md` and `sumcheck-instance-data-model.md` specs
gain a cross-reference to this spec as the aggregate-generation follow-on.

## Execution

Suggested order (each step gated on `muldiv` in both modes + `advice` e2e +
clippy):

1. Scaffold `jolt-verifier-derive`; wire it into `jolt-verifier`.
2. Implement `#[derive(SumcheckBatch)]`: struct generation (the three families
   via the `ConcreteSumcheck*` projections) plus the **minimal** initial impl set
   (`Output` `opening_values` / `append`), `Option<member>` support, and the
   per-method opt-out for alias-curated stages. Reuse `jolt-claims-derive`
   helpers (`named_fields`, generic-param threading, where-clause synthesis).
   Grow the generated-impl set in later steps only as migrations demand it.
3. Migrate **stage 5** (reference): replace hand-written `Stage5OutputClaims` /
   `Stage5Challenges`, introduce `Stage5InputClaims` and refactor stage 5's
   `*_inputs_from_upstream` helpers to build it, update the `verify` body and
   `blindfold/stage5.rs`. Validate.
4. Migrate **stages 7, 3, 4** (clean → light escape hatches), each incl. its
   `*_from_upstream` helpers.
5. Migrate **stages 2, 6, 1** (heaviest residual via inherent impls / `verify`),
   each incl. its `*_from_upstream` helpers.
6. Update `proof.rs`, `verifier.rs`, the prover `clear_claims.rs`, and the test
   support files for the `<F>` → `<F, F>` signature change.

## References

- PRs [#1640](https://github.com/a16z/jolt/pull/1640),
  [#1653](https://github.com/a16z/jolt/pull/1653) — the sumcheck-instance data
  model this builds on.
- `specs/symbolic-sumcheck.md`, `specs/sumcheck-instance-data-model.md`,
  `specs/self-contained-sumcheck-relations.md`.
- `crates/jolt-claims-derive/src/lib.rs` — the existing leaf-claim derives whose
  machinery this reuses.
- `CLAUDE.md` — the BlindFold claim/constraint synchronization invariant.
