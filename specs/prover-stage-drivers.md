# Spec: Generated Prover Stage Drivers

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Andrew Tretyakov               |
| Created     | 2026-07-17                     |
| Revised     | 2026-07-21 (v2: prover-owned driver, universal kernels) |
| Status      | proposed                       |
| PR          |                                |

## Summary

The clean-slate prover (#1669, `specs/clean-slate-prover.md`) shares the batch *head* with the
verifier — the generated `begin_batch` makes Fiat-Shamir head drift structurally impossible —
but everything after the head is hand-written per stage: `prove_stageX` constructs the
relations a second time, hard-codes the member vector in declaration order, plumbs each slot's
`prepare` with arguments that restate the relation's constructor, hand-assembles the
`StageNOutputClaims` literal, and copy-pastes the final-claim epilogue eight times. Stage 6b
hand-mirrors ~170 lines of `Stage6bSumchecks::build`. Each of these is a
"shared-by-convention" seam: a verifier-side change to one batch leg compiles cleanly and
fails only at proof time, in whichever configs the byte-diff harness happens to exercise.

This PR single-sources the prover's protocol structure from the batch struct declaration
while keeping **jolt-verifier prover-free**: `#[derive(SumcheckBatch)]` emits, besides the
verify drivers, only an inert member-list callback macro; `jolt-prover` owns the
`StageProver` trait and expands one recorder-generic `prove` per stage from that token list.
Every batch member's kernel is minted through ONE universal `PrepareKernel<F, R>` trait — no
bespoke sumcheck slot traits remain — and the two-call round API (`compute_message` +
`ingest_challenge`) is fused into a bind-then-message contract that accelerator backends can
implement as a single pass. The prover's protocol structure then has exactly one definition —
the batch struct declaration — consumed by both sides, and `prove_stageX` shrinks to what is
genuinely stage-specific.

A v1 of this design (driver emitted into jolt-verifier as `prove_clear`, dependency-inverted
`PrepareSumcheck` bound, `#[sumcheck(external)]` escape hatch, ~8 bespoke slot traits) is
implemented on `feat/prover-stage-drivers`; §Execution describes the revision from that
state. The v1 mechanisms this revision deletes are recorded under §Alternatives.

## Intent

### Goal

Generate the prove-side stage driver from the same `#[derive(SumcheckBatch)]` declaration
that generates the verifier's — member order, presence, typed I/O, and the
prepare→prove→extract→check→finish sequence single-sourced — with the generated prover code
living in `jolt-prover`, kernels reached through one universal `PrepareKernel<F, R>` trait
whose request object is the relation instance itself, and **zero prover-facing items in
jolt-verifier**.

Key abstractions introduced or modified:

- **`ProverInputs<'a, F, R: ConcreteSumcheck<F>>`** (jolt-kernels): ONE generic bundle over
  the per-relation projections the claims data model already generates —
  `{ relation: &R, claims: &SumcheckInputClaims<F, R>, points: &SumcheckInputPoints<F, R>,
  challenges: &ConcreteSumcheckChallenges<F, R> }`. Deliberately NOT macro-generated per
  relation: per-relation shape is generated once (the cell-generic `Inputs`/`Outputs`
  structs), so `ProverInputs<'_, F, SpartanShift<F>>` already is the nominal per-relation
  type. All four fields are protocol data — pure functions of the relation and upstream
  carriers — which is what lets the generated driver construct the bundle mechanically per
  member. Backend context (`session`, `witness`) stays outside the bundle, as positional
  arguments: it is compute plumbing, not protocol input.
- **`PrepareKernel<F, R: ConcreteSumcheck<F>>`** (jolt-kernels): the universal backend
  trait — `prepare(&self, session: &mut ProofSession, witness: &dyn WitnessProvider<F,
  JoltVmNamespace>, inputs: ProverInputs<'_, F, R>)
  → Result<Box<dyn SumcheckKernel<F, Relation = R>>, KernelError<F>>` — serving **every**
  batch member of every stage. Naming follows std's `BuildHasher` shape (the stored
  verb-phrase trait mints the worker that does the compute): `JoltBackend` holds one
  `PrepareKernel` per relation, which mints one `SumcheckKernel` per proof run. The relation
  IS the typed request: the verifier constructs each relation with full geometry, and kernels
  read dimensions/points/carried vectors off the relation's public accessors (established in
  v1; kept — see §Alternatives on pub fields) instead of restated constructor arguments.
  Members whose kernels cannot be minted from oracle data alone conform via the two
  non-`ProverInputs` channels `prepare` already receives:
  - *Typed witness rows* (stage-5 instruction read-RAF, stage-6 bytecode indices): fetched
    inside `prepare` through the witness plane's typed-row accessors.
  - *`ProofSession` residency*: prover-retained program data (bytecode rows for the stage-6
    table folds) is parked in the session at proof start; uni-skip pre-phase instances and
    the precommitted 6b→7 phase-spanning kernel state are parked by the front / the previous
    stage's kernel and reclaimed by the next `prepare` — the session's documented purpose
    ("cross-stage carries"). A missing or stale carry is a `KernelError` at proof time; the
    byte-diff harness and the toy-stage twins gate it.
  Only the non-sumcheck slots (`commit`, `joint_opening`) keep hand-shaped traits — they
  have no relation `R` to be universal over.
- **`SumcheckKernel<F>`** (jolt-kernels; né `ProveSumcheck`): the execution object — pairs
  the fused `ProveRounds` round interface with typed extraction (`output_claims()`,
  `validate_derived_tables()`). Returns to jolt-kernels (its v0 home) together with
  `SumcheckKernelError`: with the driver generated into jolt-prover, nothing in
  jolt-verifier needs to name it. Kernels own no relation copy — the stage's relation is the
  single source of geometry, threaded back through `validate_derived_tables`.
- **`HasKernel<F, R>`** (jolt-kernels): type-indexed slot resolution on the backend —
  `fn kernel(&self) -> &dyn PrepareKernel<F, R>`. Implemented once per slot, adjacent to the
  `JoltBackend` field declarations (a small `macro_rules!` emits field + impl together, so
  registry and resolution cannot diverge). This replaces v1's `BackendPreparer` as "the one
  place backend field names are spelled" — and unlike `BackendPreparer`, any registry
  (e.g. `ReferenceBackend` directly, or a test double) can implement it.
- **`StageProver<F>`** (jolt-prover): the driver trait, implemented for each stage batch
  struct (local trait, foreign type — orphan-rule clean):
  ```rust
  pub trait StageProver<F: Field>: Sized {
      type InputClaims; type InputPoints; type Challenges;
      type OutputClaims; type OutputPoints;
      fn prove<B, Rec, T>(
          &self, kernels: &B, session: &mut ProofSession,
          witness: &dyn WitnessProvider<F, JoltVmNamespace>,
          inputs: &Self::InputClaims, input_points: &Self::InputPoints,
          challenges: &Self::Challenges, recorder: Rec, transcript: &mut T,
      ) -> Result<Proved<F, Self, Rec::Commitment>, ProverError<F>>
      where B: KernelSource<F, Self>, Rec: SumcheckRecorder<F>,
            T: Transcript<Challenge = F>;
  }
  ```
  ONE recorder-generic `prove` — no `prove_clear`/`prove_zk` split. The driver is
  mode-agnostic by construction (the recorder is the clear/committed seam, exactly like
  `begin_batch`); the genuinely zk-divergent code (uni-skip clear vs committed arms, wire
  assembly, BlindFold witness carry) lives in the stage fronts and is PR C's scope.
  `KernelSource<F, S>` is the per-stage bound collector: the consumer macro emits one
  blanket impl per stage — `impl<B> KernelSource<F, Stage3Sumchecks<F>> for B where
  B: HasKernel<F, SpartanShift<F>> + HasKernel<F, InstructionInput<F>> + ...` — so the
  trait method's `B` bound is uniform while each stage demands exactly its members' slots.
  `Proved<F, S, C>` is one generic carrier in jolt-prover
  `{ recorded, output_claims: S::OutputClaims, output_points: S::OutputPoints, final_claim }`
  (replaces v1's per-stage generated `ProvedStageN`).
  Output curation (stage-6b's dedup'd absorb order) is a per-impl
  `curate_opening_values(&self, claims: &mut Self::OutputClaims, points:
  &Self::OutputPoints) -> Result<Vec<F>, _>` hook: the macro emits the default body
  (`self.opening_values(claims)`, the derive-generated canonical order) and accepts an
  override block at the invocation site for the curated stages (6b passes the promoted
  `stage6b_opening_values`).
- **Member-list callback macros** (emitted by `#[derive(SumcheckBatch)]`): for each batch
  struct, an inert `#[macro_export] macro_rules! <snake_case_struct>_members` that forwards
  a structured token list to a caller-chosen macro:
  `{ batch = Stage3Sumchecks, flags = [..per-batch opt-outs..], members = [ { name: shift,
  relation: ::jolt_verifier::stages::stage3::outputs::SpartanShift, presence: required },
  { name: ..., presence: optional }, ... ] }` (fully-qualified relation paths; exact token
  grammar settled in implementation). This is the single-sourcing handoff: jolt-prover's
  consumer `macro_rules! impl_stage_prover` expands the `StageProver` + `KernelSource`
  impls from it, so no stage's member list, order, or presence is ever restated. The derive
  emits **no other prover-facing code**: v1's `prove_clear`, `ProvedStageN`, and
  `<Stage>ExternalMembers` emissions are removed. Escalation path: if the consumer outgrows
  `macro_rules!` (it should not — the aggregate-level calls remain ordinary generated
  methods on the batch struct), a function-like proc macro consuming the same token list is
  the fallback, in its own crate.
- **Fused round API** (jolt-sumcheck; done in v1, unchanged):
  `ProveRounds::{compute_message, ingest_challenge}` became
  `prove_round(&mut self, bind: Option<F>, round: usize, previous_claim: F)
  → Result<UnivariatePoly<F>, SumcheckError<F>>` plus terminal `finish_rounds(&mut self,
  bind: F)`. `bind` is the member-local previous active round's challenge (`None` on the
  member's first active round); the engine threads it through the activation-window
  bookkeeping. A backend binds and evaluates in one pass over its tables — the contract the
  GPU discussion wanted.
- **Constructor promotions** (jolt-verifier; done in v1, unchanged):
  `Stage6bSumchecks::build_from_parts` consumes the clear-output carriers both sides hold;
  `Stage1Output`'s remainder-point tail / cycle-binding derivations promoted so stage2's
  `τ_low` and stage6a/6b's point derivations stop hand-copying them. Relation accessor
  blocks (the kernel read path) likewise land in v1 and stay.

What remains per-stage in `prove_stageX` (deliberately): upstream carrier unpacking, the
mid-head hand choreography the protocol currently requires (stage2's `τ_high` + uni-skip +
post-gamma output-address draws, stage6a's reference-address pad/truncate draws, stage6b's
carried gammas and conditional `eta`), the uni-skip pre-phase fronts (stages 1–2, which park
their bound instance in the session for the remainder member's `prepare`), and stage-0/8 (no
batch sumcheck). These fronts are protocol content frozen at the current wire format; they
shrink in the later idiosyncrasy-purge PR, not here.

### Invariants

All eight invariants of `specs/clean-slate-prover.md` §Invariants carry over verbatim. This
PR strengthens #1 and adds three:

1. **Fiat-Shamir byte-identity, now body-wide.** Previously structural for the head
   (`begin_batch`) and by-convention for the body; after this PR the round loop ordering,
   output extraction order, final-claim check, and finish absorb are the *same generated
   code path* on both sides (verify: derive-emitted; prove: consumer-macro-emitted from the
   derive's member list). The per-stage hand choreography that remains (fronts listed above)
   is the exhaustive list of by-convention transcript code left in the prover.
2. **Declaration order is the only order.** No stage recipe contains a member vector, a
   hand-assembled `StageNOutputClaims` literal, or a per-slot `prepare` argument that
   duplicates relation-constructor data. Member order, presence, and typed I/O derive from
   the batch struct declaration alone, on both sides — the prove side through the emitted
   member-list macro, never through a hand-maintained copy.
3. **jolt-verifier is prover-free.** No prover-only trait, type, method, or generated code in
   `crates/jolt-verifier/src`: no `SumcheckKernel`, `ProverInputs`, `PrepareKernel`,
   `PrepareSumcheck`, no `prove_*` driver emission. The derive's only prover-facing output is
   the inert member-list token macro. Grep-enforced (see Acceptance Criteria).
4. **Wire freeze.** Proof bytes, Fiat-Shamir conventions, fixtures, and the serialized
   `JoltProof` are byte-identical to pre-PR state. This is a pure restructuring; the
   byte-diff harness passes with zero regenerated fixtures.

`jolt-eval` plan: no invariant definitions change. `legacy_proof_byte_equality` (the
byte-diff harness) is the primary gate and must pass unmodified; `kernel_naive_equivalence`
and `prover_verifier_stage_consistency` (planned in the clean-slate spec) become *easier* to
add after this PR since every sumcheck kernel sits behind one trait — implementing them here
is welcome but not required.

### Non-Goals

- **No protocol changes.** The idiosyncrasy purge (Michael's list: absorb rescheduling,
  domain-separator cleanup, stage-6b dedup removal, front-draw simplification) is the later
  wire-breaking PR (`PLAN_OF_PRS.md` PR D). This PR must not need a single regenerated
  fixture.
- **No generated round loop.** `prove_batch` stays ordinary engine code in jolt-sumcheck.
  What is generated is the stage *driver* — the code between the head and the engine —
  where declaration order is load-bearing and duplication is the drift risk.
- **No new backends, no ZK wiring, no tracing.** Those are PRs B/C of `PLAN_OF_PRS.md`. The
  single `prove` is recorder-generic so PR C is a recorder swap plus front work, but
  committed-path wiring is out of scope.
- **No bespoke sumcheck slots.** (Reverses v1's non-goal.) Every batch member — including
  the uni-skip remainders, the stage-6a address phases, instruction read-RAF, and the
  precommitted reduction family — is served through `PrepareKernel<F, R>`. The only
  hand-shaped slot traits left are the non-sumcheck ones: `commit` and `joint_opening`.
- **No naive-prover promotion.** It remains a test oracle behind the same `SumcheckKernel`
  object; optimized backends implementing `PrepareKernel` are expected not to touch the
  symbolic `Expr` at all (the symbolic representation stays verifier-adjacent +
  naive-prover-only).
- **No `jolt-prover-legacy` changes** beyond whatever `ProveRounds` rename fallout the
  compiler forces (expected: none — legacy has its own sumcheck engine).

## Evaluation

### Acceptance Criteria

- [ ] Byte-diff harness (`crates/jolt-prover/tests/byte_diff.rs`, `--features
      prover-fixtures`): all ten tests pass against unmodified `jolt-prover-legacy`, with no
      fixture regeneration anywhere in the workspace.
- [ ] jolt-verifier is prover-free:
      `grep -rE "SumcheckKernel|ProverInputs|PrepareKernel|PrepareSumcheck|SumcheckPreparer|ProveRounds" crates/jolt-verifier/src` → no hits
      (test modules exercising the emitted member-list macros excepted, and only if they
      need none of these names). The derive emits no `prove_clear`, `Proved*`, or
      `*ExternalMembers`.
- [ ] Every stage batch (1, 2, 3, 4, 5, 6a, 6b, 7) has a `StageProver` impl expanded by the
      consumer macro from its emitted member-list macro, and every `prove_stageX` calls
      `prove`; no stage recipe contains a `Vec<&mut dyn ProveRounds>` literal, a
      `StageNOutputClaims` literal, or an `expected_final_claim`/`FinalClaimMismatch` block
      (single macro-expanded instance).
- [ ] `#[sumcheck(external)]`, `<Stage>ExternalMembers`, `PrepareSumcheck`,
      `SumcheckPreparer`, and `BackendPreparer` do not exist in the workspace.
- [ ] jolt-kernels crate-root slot-trait modules reduced to the non-sumcheck set
      (`commitment`, `opening`, plus `backend`/`error`); every batch member's kernel is
      reachable via `HasKernel<F, R>` → `PrepareKernel<F, R>`; relation double-construction
      is gone — kernels receive `&R`.
- [ ] The stage-6b recipe contains no hand-mirrored batch legs (built by the promoted
      `build_from_parts`), and the stage-1/2 fronts hand no kernel objects to the driver —
      the remainder members mint through `PrepareKernel` from session state.
- [ ] `ProveRounds` has the fused `prove_round`/`finish_rounds` shape; jolt-sumcheck's
      twin-transcript engine tests (clear, committed, head-aligned, uni-skip) pass against
      the unchanged generated verify drivers.
- [ ] `StageProver::prove` is generic over `SumcheckRecorder` (compiles against
      `CommittedSumcheckRecorder` in a type-check test even though nothing wires it).
- [ ] Toy-stage twin tests for the macro-expanded driver (hand-rolled batch with a plain
      member, an `Option` member absent and present, and a session-carried member) live in
      jolt-prover and byte-match the generated `verify_clear` on a twin transcript.
- [ ] `cargo clippy` clean under `host`, `host,zk`, and the field-inline crate set;
      `cargo fmt`; full workspace `cargo nextest` green.
- [ ] Net line count of `crates/jolt-prover/src/stages/` decreases by ≥ 30% vs the #1669
      baseline (guards against the refactor smearing complexity around).

### Testing Strategy

- **Primary gate:** the byte-diff harness — stage-granular `muldiv` ratchet plus the
  whole-proof ratchets across advice × committed-program × trace-order. Run after each
  migration step, not only at the end (the harness is stage-granular precisely to localize
  drift).
- **Engine:** jolt-sumcheck twin tests on the fused API (done in v1), including the test
  that a member implementing fused bind+eval byte-matches a reference member that binds and
  evaluates separately.
- **Driver:** v1's `prove_clear` twin fixtures move to jolt-prover and are re-expressed
  against the consumer macro (including the session-carry member replacing v1's `external`
  fixture member).
- **Verifier suites:** all standard + zk + BlindFold + tampering fixture suites pass against
  cached fixtures (verifier behavior untouched except promotions, same bar as #1669).
- Both `--features host` and `--features host,zk` clippy/test matrices, per repo policy.

### Performance

No prover-performance expectations: the reference backend is a test oracle, and the driver
is orchestration (per-stage constant work). The fused round API must not regress the naive
prover's round loop measurably. Session carries are per-stage constant-count map operations.
No new benchmarks; `jolt-eval` objectives untouched. The real performance payoff is
deferred: the fused contract is what lets a future GPU backend halve per-round table
traffic, and PR B's tracing will measure it.

## Design

### Architecture

Dependency direction is the one real constraint, and this revision aligns the code with it
instead of inverting around it: prover code lives above the verifier, full stop.

```
jolt-sumcheck        ProveRounds (fused), prove_batch, recorders            [engine]
jolt-verifier        ConcreteSumcheck + relations (with accessor blocks),
                     generated begin_batch + verify_clear + aggregates +
                     member-list callback macros                            [protocol; prover-free]
jolt-kernels         SumcheckKernel, ProverInputs, PrepareKernel,
                     HasKernel, JoltBackend/ProofSession,
                     commit/joint_opening slots, reference/ impls           [compute]
jolt-prover          StageProver + KernelSource + Proved + consumer macro
                     (impls expanded per stage), stage fronts, stage 0/8    [orchestration]
```

Driver expansion shape (schematic; exact token grammar settled in implementation):

```rust
// jolt-verifier (emitted by the derive; inert):
#[macro_export]
macro_rules! stage3_sumchecks_members { ($cb:path) => { $cb! {
    batch = ::jolt_verifier::stages::stage3::outputs::Stage3Sumchecks,
    members = [
        { name: shift, relation: ::jolt_verifier::stages::stage3::outputs::SpartanShift, presence: required },
        { name: instruction_input, relation: ..., presence: required },
        { name: registers_claim_reduction, relation: ..., presence: required },
    ]
} } }

// jolt-prover (hand-written, one line per stage; override block only for curated stages):
stage3_sumchecks_members!(impl_stage_prover);
```

`impl_stage_prover` expands: the `KernelSource<F, Stage3Sumchecks<F>>` blanket impl
(collecting `HasKernel` bounds), and `impl StageProver<F> for Stage3Sumchecks<F>` whose
`prove` runs: `begin_batch` (existing generated head) → per-member
`kernels.kernel::<R>().prepare(session, witness, ProverInputs { .. })` in declaration order
(`Option` members gated on presence, mismatched presence attributed to the member's relation
id) → `prove_batch` → `derive_opening_points` → per-member `validate_derived_tables` → typed
`output_claims()` into the aggregate → `validate_output_claims` → `expected_final_claim`
hard check → `curate_opening_values` (default: generated canonical order) →
`recorder.finish`.

**Edge classes and their mechanisms** (the exhaustive list; v1's `external` row replaced):

| Edge | Mechanism |
|---|---|
| Mid-head hand draws (stage 2/6a/6b fronts) | Stay in `prove_stageX`, between `draw_challenges` and `prove` — same seam `verify` uses today. The derive's existing draw-suppression opt-outs are unchanged. |
| Uni-skip pre-phases (stages 1–2) | The front runs the uni-skip round (`SpartanOuterInstance` etc.) and parks the bound instance in `ProofSession`; the remainder member is a regular batch member whose `PrepareKernel<F, OuterRemainder>::prepare` reclaims the instance and calls `into_remainder(&relation)`. |
| Cross-batch 6b→7 precommitted carry | The 6b kernel parks its post-cycle bound state in `ProofSession` at extraction; stage 7's `PrepareKernel` for the address-phase relation reclaims it. Intermediate-vs-final wire claims (`has_address_phase()`) resolve inside the kernel's `output_claims()` — the layout lives on the relation. |
| Output curation (6b opening-value dedup) | `curate_opening_values` override at the macro invocation site, calling the promoted `stage6b_opening_values`. The derive's `no_opening_values` opt-out continues to suppress the verify-side generated absorb for the same stages. |
| Non-oracle witness channels (read-RAF rows, stage-6 bytecode indices, prover-retained program data) | Typed rows: fetched inside `prepare` via the witness plane's typed accessors. Program data: `ProofSession` residency established at proof start. Never a driver concern. |

### Alternatives Considered

- **Status quo (hand-written recipes).** Rejected: the #1669 review identified the drift
  class concretely — stage6b's mirrored legs, three inline point-derivation copies, 8×
  epilogues, 27 trait files restating constructors.
- **v1: emit `prove_clear` into jolt-verifier (implemented, now rejected).** The derive
  expands at the struct definition site, so the driver landed in jolt-verifier — dragging
  `SumcheckKernel`, `ProverInputs`, and a dependency-inverted `PrepareSumcheck`/
  `SumcheckPreparer`/`BackendPreparer` apparatus into/around the verifier crate. Rejected:
  the verifier crate must stay prover-free (embedded verification is its deployment shape),
  and the inversion apparatus existed *only* to let verifier-resident code reach
  jolt-kernels without naming it. Moving the expansion into jolt-prover (member-list
  callback macro) deletes all three traits and `BackendPreparer` outright.
- **Feature-gate the prover emission in jolt-verifier instead of relocating.** 10× cheaper,
  compiles out for verifier-only builds — rejected: the code would still textually live in
  the verifier crate and its trait paths would still be `jolt_verifier::`; the crate
  boundary is the architecture statement.
- **`prove_clear`/`prove_zk` split mirroring the verify side.** Rejected: the driver is
  mode-agnostic — the recorder is already the clear/committed seam (`begin_batch`
  precedent), and the genuinely zk-divergent code is in the stage fronts. One
  recorder-generic `prove`; the verify side's split reflects wire-format dispatch the
  prover doesn't have.
- **`#[sumcheck(external)]` members (implemented in v1, now rejected).** The escape hatch
  for kernels not mintable at prepare time (uni-skip remainders, 6b→7 spans). Replaced by
  `ProofSession` carries so every member is uniform. Acknowledged tradeoff: externals were
  compile-time-visible object flow; session carries are runtime type-keyed state, so a
  wiring bug surfaces at proof time as a `KernelError` instead of at compile time — accepted
  for the uniformity (no second member kind in the macro grammar, no `ExternalMembers`
  structs, no presence cross-checks) and gated by the byte-diff harness and twin tests.
- **Per-slot request structs instead of `&relation`.** Rejected: a request struct is a
  second, hand-maintained copy of the relation's constructor data — the drift this PR exists
  to kill. The verifier already builds the relation with full geometry; passing it is free
  and makes batch/kernel geometry divergence unrepresentable.
- **Public relation fields instead of accessor methods.** Rejected: all-pub fields enable
  struct-literal construction bypassing the validated (sometimes fallible,
  Fiat-Shamir-adjacent) constructors — reopening the double-construction hole; and several
  accessors are not field reads at all (`BytecodeReadRafCycle` dispatches over a private
  variant enum; `stage_values_at_r_address` computes/validates). A mixed convention is worse
  than uniform methods, and methods keep representation freedom ahead of the wire-breaking
  purge PR.
- **`SymbolicSumcheck` as the kernel data source instead of accessors.** Rejected: the
  symbolic layer is field-independent by design (holds only `Shape`); it cannot carry the
  F-valued instance data kernels need (bound points, carried challenge vectors, public
  inputs), and extending it would break the one-symbolic-object-any-field property.
- **Slot resolution via a generated `StageNSlots` struct.** Rejected in v0 and still: the
  struct would be filled field-by-field per recipe. `HasKernel` puts the field spelling
  adjacent to the field declaration itself, macro-paired so they cannot diverge.
- **Generate the round loop too (full generated prover).** Rejected in
  `specs/jolt-prover-model-crate.md` and reaffirmed: the loop is stage-invariant engine code
  with real algebra that benefits from being ordinary, testable Rust in one place.
- **Keep the two-call round API; fuse via the stash idiom.** Rejected on inverted cost
  grounds: every `ProveRounds` impl was rewritten by v1 anyway, so the fused contract was
  nearly free and removes the idiom every fusing backend would otherwise need.
- **Do the protocol idiosyncrasy purge first.** Rejected: the purge is wire-breaking, which
  retires the legacy byte-diff pin — leaving this refactor (the riskier restructuring)
  without its strongest gate. Refactor under the pin; break the wire after (PR D ordering).

## Documentation

Update `book/` prover-architecture material added by #1669 to describe the `StageProver`
driver, the member-list callback handoff, and `PrepareKernel`.
`specs/clean-slate-prover.md` §Design gains a one-paragraph pointer here (its "begin_batch
generates the head" statement becomes "head and driver"). Crate-level rustdoc: `jolt-kernels`
(`PrepareKernel`, `HasKernel`, session-carry conventions), `jolt-verifier-derive` (member-list
emission), `jolt-prover` (`StageProver`, consumer macro), `jolt-sumcheck` (`ProveRounds`
contract, done in v1).

## Execution

Starting point: `feat/prover-stage-drivers` (v1 complete — fused rounds, v1 driver, universal
`PrepareKernel` for the naive-served set, all stages migrated, byte-diff green). Each step
below is gated by the stage-granular byte-diff run.

1. **Bespoke conformance under the v1 driver** (keeps the working driver as the harness
   while the kernel surface changes):
   a. Program-data session residency at proof start; migrate the stage-6a address slots
      (`bytecode_read_raf_address`, `booleanity_address`) to `PrepareKernel` — carried 6a
      draws reach them via relation/challenge accessors, typed bytecode indices via the
      witness plane.
   b. `instruction_read_raf` → `PrepareKernel` (typed rows fetched inside `prepare`);
      delete `Stage5PrepareContext`.
   c. Uni-skip session carry: stage-1/2 fronts park the bound instance;
      `PrepareKernel<F, {Outer,Product}Remainder>` reclaims it; flip the members from
      `external` to regular.
   d. Precommitted 6b→7 session carry; intermediate-vs-final claims move into the kernels'
      `output_claims()`; flip the stage-7 members from `external` to regular; delete
      `Stage6aPrepareContext`/remaining context plumbing.
2. **Derive swap**: emit member-list callback macros; delete `prove_clear`/`ProvedStageN`/
   `ExternalMembers`/`#[sumcheck(external)]` emission (no member is external anymore).
3. **jolt-prover driver**: `StageProver`, `KernelSource`, `Proved`, `HasKernel` registry
   macro in jolt-kernels, consumer `impl_stage_prover` macro; migrate stages 3 → 5 → 4 → 1 →
   2 → 6a → 7 → 6b (curation override), byte-diff after each; move
   `SumcheckKernel`/`SumcheckKernelError`/`ProverInputs` to jolt-kernels and delete
   `PrepareSumcheck`/`SumcheckPreparer`/`BackendPreparer` as their last consumers go.
4. **Twin-test relocation**: v1's `prove_clear` twins move to jolt-prover against the
   consumer macro (session-carry fixture member replacing the `external` one).
5. **Sweep**: acceptance-criteria greps (prover-free verifier, no externals, slot-file
   count), line-count check, clippy matrices, full suite.

## References

- `specs/clean-slate-prover.md` — parent architecture; §Invariants and §Non-Goals carried
  over.
- `specs/sumcheck-batch-derive.md` — the derive this PR extends.
- `specs/jolt-prover-model-crate.md` — superseded predecessor; source of the "no generated
  round loop" position this spec preserves.
- PR #1669 — the clean-slate prover; its review threads (stage6b mirrored legs, slot-trait
  double-construction, epilogue duplication) are the concrete motivation.
- `feat/prover-stage-drivers` v1 commits (`910414376..458b6fd8c`) — the implemented v1 this
  revision builds on and partially deletes.
- `PLAN_OF_PRS.md` (repo root, untracked) — the PR sequencing this spec is "PR A" of.
- Andrew ↔ Michael design discussion, 2026-07-15 (universal `ConcreteProverSumcheck`,
  `#[SumcheckBatch]` driving the prover, fused round API, idiosyncrasy list deferral);
  Andrew design session, 2026-07-21 (prover-owned `StageProver`, member-list callback
  handoff, bespoke-set removal, single recorder-generic `prove`).
