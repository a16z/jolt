# Spec: Generated Prover Stage Drivers

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Andrew Tretyakov               |
| Created     | 2026-07-17                     |
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
This PR extends `#[derive(SumcheckBatch)]` to also emit the prove-side stage driver
(`prove_clear`), introduces one universal `PrepareKernel` trait in place of ~20 single-method
per-relation trait files, and fuses the two-call round API (`compute_message` +
`ingest_challenge`) into a bind-then-message contract that accelerator backends can implement
as a single pass. The prover's protocol structure then has exactly one definition — the batch
struct declaration — consumed by both sides, and `prove_stageX` shrinks to what is genuinely
stage-specific.

## Intent

### Goal

Generate the prove-side stage driver from the same `#[derive(SumcheckBatch)]` declaration that
generates the verifier's, so that member order, presence, typed I/O, and the
prepare→prove→extract→check→finish sequence are single-sourced — with kernels reached through
one universal `PrepareKernel<F, R>` trait whose request object is the relation instance itself.

Key abstractions introduced or modified:

- **`ProverInputs<'a, F, R: ConcreteSumcheck<F>>`** (jolt-verifier, `stages::relations`): ONE
  generic bundle over the per-relation projections the claims data model already generates —
  `{ relation: &R, claims: &SumcheckInputClaims<F, R>, points: &SumcheckInputPoints<F, R>,
  challenges: &ConcreteSumcheckChallenges<F, R> }`. This is the prover-side mirror of the
  verifier's per-sumcheck check inputs, deliberately NOT macro-generated per relation:
  per-relation shape is generated once (the cell-generic `Inputs`/`Outputs` structs), so
  `ProverInputs<'_, F, SpartanShift<F>>` already is the nominal per-relation type. All four
  fields are protocol data — pure functions of the relation and upstream carriers — which is
  what lets the generated driver construct the bundle mechanically per member (all four off
  the same batch field name, unmixable by construction). Backend context (`session`,
  `witness`) stays outside the bundle, as positional arguments: it is compute plumbing, not
  protocol input. Extending the seam later means adding a field, not re-signaturing every
  slot impl. The outputs side needs no new type: per member, `SumcheckKernel::output_claims()`
  returns the verifier's own `SumcheckOutputClaims<F, R>`; per stage, `ProvedStageN` bundles
  claims + points + proof + final claim (output points are derived batch-wide by
  `derive_opening_points`, so a per-member outputs pair would just re-split them).
- **`PrepareKernel<F, R: ConcreteSumcheck<F>>`** (jolt-kernels; the trait discussed as
  `ConcreteProverSumcheck`): the universal backend trait —
  `prepare(&self, session: &mut ProofSession, witness: &dyn WitnessProvider<F,
  JoltVmNamespace>, inputs: ProverInputs<'_, F, R>)
  → Result<Box<dyn SumcheckKernel<F, Relation = R>>, KernelError<F>>`.
  Naming follows std's `BuildHasher` shape (the stored verb-phrase trait mints the worker
  that does the compute), giving the ladder platform → operation → execution: `JoltBackend`
  holds one `PrepareKernel` per relation, which mints one `SumcheckKernel` per proof run.
  The relation IS the typed request: the verifier constructs each relation with its full
  geometry, so kernels read dimensions/points off relation accessors instead of receiving them
  as restated constructor arguments. Replaces the ~20 naive-served single-method trait files;
  `JoltBackend` fields become `Box<dyn PrepareKernel<F, SpartanShift<F>>>` etc.
- **`PrepareSumcheck<F, R>`** (jolt-verifier, `stages::relations`): the dependency-inverted
  preparer bound the *generated* driver names (jolt-verifier cannot depend on jolt-kernels) —
  `prepare(&mut self, inputs: ProverInputs<'_, F, R>) → Result<Box<dyn SumcheckKernel<F,
  Relation = R>>, Self::Error>`, one associated `Error` type with the `From` bounds the
  driver needs. `jolt-prover` implements it for every relation on a small context struct
  (`BackendPreparer<'a, F, PCS> { backend, session, witness }`) that forwards the bundle to
  the backend slots — the only place backend field names are spelled.
- **Generated `prove_clear`** (emitted by `#[derive(SumcheckBatch)]` on `StageNSumchecks`,
  beside `verify_clear`): begin_batch (existing head) → per-member `prepare` in declaration
  order (`Option` members gated on presence, exactly like the head's coefficient draws) →
  `prove_batch` → `derive_opening_points` → per-member `validate_derived_tables` →
  typed `output_claims()` into `StageNOutputClaims` → `validate_output_claims` →
  `expected_final_claim` hard check → `recorder.finish(self.opening_values(&outputs))`.
  Recorder-generic (clear/committed) from day one, like `begin_batch`.
- **`SumcheckKernel` relocation + rename**: `ProveSumcheck` moves from jolt-kernels to
  jolt-verifier (`stages::relations`) so the generated driver can name it, and is renamed
  `SumcheckKernel` — it is the execution object (bound tables + round loop), and every impl
  is rewritten by the fused round API anyway, so the rename rides for free. Its current home
  was forced by "jolt-sumcheck may not name `ConcreteSumcheck`"; jolt-verifier names both.
  jolt-kernels re-exports for downstream stability. (`ProveRounds`, the protocol-neutral
  engine supertrait in jolt-sumcheck, keeps its name.)
- **Fused round API** (jolt-sumcheck): `ProveRounds::{compute_message, ingest_challenge}`
  becomes `prove_round(&mut self, bind: Option<F>, round: usize, previous_claim: F)
  → Result<UnivariatePoly<F>, SumcheckError<F>>` plus terminal
  `finish_rounds(&mut self, bind: F)`. `bind` is the member-local previous active round's
  challenge (`None` on the member's first active round); the engine threads it through the
  activation-window bookkeeping. A backend binds and evaluates in one pass over its tables —
  the contract the GPU discussion wanted — instead of a bind-write pass followed by an
  eval-read pass (the current API permits fusion only via a stash-the-challenge idiom).
- **Constructor promotions** (jolt-verifier): the batch-construction logic the prover still
  inline-copies becomes shared code — `Stage6bSumchecks::build` refactored so its leg-assembly
  core (`build_from_parts`) consumes the clear-output carriers both sides hold (named as an
  intended promotion by `specs/clean-slate-prover.md` §Non-Goals); `Stage1Output`'s
  remainder-point tail / cycle-binding derivations promoted so stage2's `τ_low` and
  stage6a/6b's `bytecode_stage_points` stop hand-copying them.

What remains per-stage in `prove_stageX` (deliberately): upstream carrier unpacking, the
mid-head hand choreography the protocol currently requires (stage2's `τ_high` + uni-skip +
post-gamma output-address draws, stage6a's reference-address pad/truncate draws, stage6b's
carried gammas and conditional `eta`), uni-skip pre-phases (stages 1–2), the 6b→7 precommitted
kernel carry, and stage-0/8 (no batch sumcheck). These fronts are protocol content frozen at
the current wire format; they shrink in the later idiosyncrasy-purge PR, not here.

### Invariants

All eight invariants of `specs/clean-slate-prover.md` §Invariants carry over verbatim. This PR
strengthens #1 and adds two:

1. **Fiat-Shamir byte-identity, now body-wide.** Previously structural for the head
   (`begin_batch`) and by-convention for the body; after this PR the round loop ordering,
   output extraction order, final-claim check, and finish absorb are the *same generated code
   path* on both sides. The per-stage hand choreography that remains (fronts listed above) is
   the exhaustive list of by-convention transcript code left in the prover.
2. **Declaration order is the only order.** No stage recipe contains a member vector, a
   hand-assembled `StageNOutputClaims` literal for generated members, or a per-slot `prepare`
   argument that duplicates relation-constructor data. Member order, presence, and typed I/O
   derive from the batch struct declaration alone.
3. **Wire freeze.** Proof bytes, Fiat-Shamir conventions, fixtures, and the serialized
   `JoltProof` are byte-identical to pre-PR state. This is a pure restructuring; the byte-diff
   harness passes with zero regenerated fixtures.

`jolt-eval` plan: no invariant definitions change. `legacy_proof_byte_equality` (the byte-diff
harness) is the primary gate and must pass unmodified; `kernel_naive_equivalence` and
`prover_verifier_stage_consistency` (planned in the clean-slate spec) become *easier* to add
after this PR since every naive-served kernel sits behind one trait — implementing them here
is welcome but not required.

### Non-Goals

- **No protocol changes.** The idiosyncrasy purge (Michael's list: absorb rescheduling,
  domain-separator cleanup, stage-6b dedup removal, front-draw simplification) is the later
  wire-breaking PR (`PLAN_OF_PRS.md` PR D). This PR must not need a single regenerated fixture.
- **No generated round loop.** `prove_batch` stays ordinary engine code in jolt-sumcheck (the
  position taken by `specs/clean-slate-prover.md` and `specs/jolt-prover-model-crate.md`
  stands). What is generated is the stage *driver* — the code between the head and the
  engine — where declaration order is load-bearing and duplication is the drift risk.
- **No new backends, no ZK wiring, no tracing.** Those are PRs B/C of `PLAN_OF_PRS.md`. The
  generated driver must be recorder-generic so PR C is a recorder swap, but committed-path
  wiring is out of scope.
- **Bespoke slots stay bespoke.** `spartan_outer` / `spartan_product` (uni-skip instance →
  remainder handoff), `instruction_read_raf` (typed per-cycle rows), the precommitted
  reduction family (6b→7 phase carry, `PrecommittedReductionProver`), `commit`, and
  `joint_opening` keep hand-shaped traits. The universal trait covers the naive-served
  remainder (~20 slots). `bytecode_read_raf_{address,cycle}` and `booleanity_{address,cycle}`
  are audited during implementation: universal if their relation instances carry (or can
  cheaply carry) the data their kernels read — the stage-6b recipe-level stage-value fold
  moves into the cycle kernel's `prepare` as part of that audit — else they stay bespoke and
  are marked `external` in the batch (see Design).
- **No naive-prover promotion.** It remains a test oracle behind the same `SumcheckKernel`
  object; optimized backends implementing `PrepareKernel` are expected not to touch the symbolic
  `Expr` at all (the symbolic representation stays verifier-adjacent + naive-prover-only).
- **No `jolt-prover-legacy` changes** beyond whatever `ProveRounds` rename fallout the
  compiler forces (expected: none — legacy has its own sumcheck engine).

## Evaluation

### Acceptance Criteria

- [ ] Byte-diff harness (`crates/jolt-prover/tests/byte_diff.rs`, `--features
      prover-fixtures`): all ten tests pass against unmodified `jolt-prover-legacy`, with no
      fixture regeneration anywhere in the workspace.
- [ ] `#[derive(SumcheckBatch)]` emits `prove_clear` for every stage batch (1, 2, 3, 4, 5, 6a,
      6b, 7), and every `prove_stageX` calls it; no stage recipe contains a
      `Vec<&mut dyn ProveRounds>` literal, a `StageNOutputClaims` literal for generated
      members, or an `expected_final_claim`/`FinalClaimMismatch` block (single generated
      instance).
- [ ] The stage-6b recipe contains no hand-mirrored batch legs: `Stage6bSumchecks` is built by
      the promoted verifier constructor from the clear carriers; `bytecode_stage_points` and
      the stage-2 `τ_low` inline copies are deleted in favor of promoted helpers.
- [ ] Crate-root slot-trait files in jolt-kernels reduced from 27 to the bespoke set
      (≤ 8); every removed trait's kernel is reachable via `PrepareKernel<F, R>`; relation
      double-construction (recipe builds `R::new(...)`, kernel rebuilds the same `R`) is gone —
      kernels receive `&R`.
- [ ] `ProveRounds` has the fused `prove_round`/`finish_rounds` shape; jolt-sumcheck's
      twin-transcript engine tests (clear, committed, head-aligned, uni-skip) pass against the
      unchanged generated verify drivers.
- [ ] Generated `prove_clear` is generic over `SumcheckRecorder` (compiles against
      `CommittedSumcheckRecorder` in a type-check test even though nothing wires it).
- [ ] `cargo clippy` clean under `host`, `host,zk`, and the field-inline crate set;
      `cargo fmt`; full workspace `cargo nextest` green.
- [ ] Net line count of `crates/jolt-prover/src/stages/` decreases by ≥ 30% (mechanical
      consequence of the above; guards against the refactor smearing complexity around).

### Testing Strategy

- **Primary gate:** the byte-diff harness — stage-granular `muldiv` ratchet plus the whole-proof
  ratchets across advice × committed-program × trace-order. Run after each stage's migration,
  not only at the end (the harness is stage-granular precisely to localize drift).
- **Engine:** jolt-sumcheck twin tests updated to the fused API; add one test that a member
  implementing fused bind+eval (single table pass) byte-matches a reference member that binds
  and evaluates separately.
- **Derive:** `begin_batch_tests`-style fixtures in `jolt-verifier/src/stages/relations.rs`
  extended with `prove_clear` twins — the generated driver against a hand-rolled toy stage,
  including an `Option` member (absent and present) and an `external` member.
- **Verifier suites:** all standard + zk + BlindFold + tampering fixture suites pass against
  cached fixtures (verifier behavior untouched except promotions, same bar as #1669).
- Both `--features host` and `--features host,zk` clippy/test matrices, per repo policy.

### Performance

No prover-performance expectations: the reference backend is a test oracle, and the driver is
orchestration (per-stage constant work). The fused round API must not regress the naive
prover's round loop measurably (bind-then-eval in one virtual call vs two; if anything,
marginally fewer dyn dispatches). No new benchmarks; `jolt-eval` objectives untouched. The
real performance payoff is deferred: the fused contract is what lets a future GPU backend
halve per-round table traffic, and PR B's tracing will measure it.

## Design

### Architecture

Dependency direction is the one real constraint. The derive emits code into jolt-verifier,
which must not depend on jolt-kernels (jolt-kernels depends on jolt-verifier). Hence the
split:

```
jolt-sumcheck        ProveRounds (fused), prove_batch, recorders            [engine]
jolt-verifier        ConcreteSumcheck, SumcheckKernel (moved here, né
                     ProveSumcheck), PrepareSumcheck<F,R> bound, generated
                     begin_batch + prove_clear + verify_clear               [protocol]
jolt-kernels         PrepareKernel<F,R>, ProofSession, bespoke slot traits,
                     reference/ implementations                             [compute]
jolt-prover          BackendPreparer: impl PrepareSumcheck<F,R> for every R
                     (forwards to backend slots), stage fronts, stage 0/8   [orchestration]
```

Generated driver shape (schematic; exact naming settled in implementation):

```rust
impl<F: Field> Stage3Sumchecks<F> {
    pub fn prove_clear<P, Rec, T>(
        &self,
        preparer: &mut P,
        inputs: &Stage3InputClaims<F>,
        input_points: &Stage3InputPoints<F>,
        challenges: &Stage3Challenges<F>,
        mut recorder: Rec,
        transcript: &mut T,
    ) -> Result<ProvedStage3<F, Rec::Commitment>, P::Error>
    where
        P: PrepareSumcheck<F, SpartanShift<F>>
         + PrepareSumcheck<F, InstructionInput<F>>
         + PrepareSumcheck<F, RegistersClaimReduction<F>>,
        Rec: SumcheckRecorder<F>, T: Transcript<Challenge = F>,
    { /* head → prepare (declaration order) → prove_batch → points →
         validate_derived_tables → output_claims → validate → expected_final_claim
         (mismatch surfaces through P::Error via a From bound) → finish */ }
}
```

`ProvedStageN` carries the recorded proof, `StageNOutputClaims`, `StageNOutputPoints`, and the
final claim — the same data the current recipes assemble by hand.

**Edge classes and their mechanisms** (the exhaustive list, from the #1669 review):

| Edge | Mechanism |
|---|---|
| Mid-head hand draws (stage 2/6a/6b fronts) | Stay in `prove_stageX`, between `draw_challenges` and `prove_clear` — same seam `verify` uses today. The derive's existing draw-suppression opt-outs are unchanged. |
| Uni-skip pre-phases (stages 1–2) | Stay bespoke: `SpartanOuterInstance`/product instance produce the uni-skip poly, then `into_remainder` yields the remainder member, which the recipe hands to `prove_clear` as an `external` member. |
| Cross-batch 6b→7 precommitted carry | Members declared `#[sumcheck(external)]` in the batch: the driver takes them as caller-supplied `&mut dyn ProveRounds` (appended in declaration order) and returns their slots in the output struct unfilled; the recipe curates their wire claims exactly as today (intermediate-vs-final via `has_address_phase()`). |
| Output curation (6b opening-value dedup, stage-4 advice attach) | Unchanged: the driver's finish consumes `opening_values()` where the derive already generates it, and keeps the existing `no_opening_values` opt-out for the curated stages, whose recipes call `recorder.finish` with the promoted curated helpers (`stage6b_opening_values`) as today. |
| Non-oracle witness channels (read-RAF rows, stage-6 bytecode indices, prover-retained program data) | Owned by the bespoke slots / `BackendPreparer` context, never by the generated driver. |

**`external` members** are the single escape hatch: a per-field attribute making the driver
accept the member object and skip its typed extraction. Everything expressible without it
should be; the acceptance criteria cap its use to the uni-skip remainders and the
precommitted family.

**Fused rounds, engine side.** `prove_batch` currently: for each round, `compute_message` on
active members → batch/trim/absorb → squeeze → `ingest_challenge` on active members. After:
for each round, `prove_round(pending[i].take(), …)` on members *becoming or staying* active →
absorb → squeeze → store the challenge into `pending[i]` for each active member; after the
loop, `finish_rounds(pending[i].take().…)` for every member that was ever active. Inactive
padding, `claim/2` constant rounds, head-alignment offsets, and the running-claim self-check
are unchanged. The naive prover's `prove_round` = bind pending tables, then the existing
pointwise evaluation; hand kernels update mechanically.

### Alternatives Considered

- **Status quo (hand-written recipes).** Rejected: the #1669 review identified the drift class
  concretely — stage6b's mirrored legs, three inline point-derivation copies, 8× epilogues,
  27 trait files restating constructors. Each hardens with every stage, backend, and protocol
  change layered on top.
- **Generate the round loop too (full generated prover).** Rejected in
  `specs/jolt-prover-model-crate.md` and reaffirmed in `specs/clean-slate-prover.md`: the loop
  is stage-invariant engine code with real algebra (padding scales, activation windows,
  trimming) that benefits from being ordinary, testable Rust in one place. Declaration order
  is load-bearing only in the driver — that is exactly what gets generated.
- **Per-slot request structs instead of `&relation`.** Rejected: a request struct is a second,
  hand-maintained copy of the relation's constructor data — the drift this PR exists to kill.
  The verifier already builds the relation with full geometry; passing it is free and makes
  batch/kernel geometry divergence unrepresentable (the #1669 fix wiring
  `validate_derived_tables` against the stage's relation instance is the precedent). The
  distinction from `ProverInputs`: bundling *references to already-derived protocol data* is
  fine (one generic struct, no per-relation duplication); generating nominal per-relation
  request structs carrying copies of relation data is what's rejected.
- **Macro-generated `SumcheckXProverInputs` / `SumcheckXProverOutputs` per relation.**
  Rejected as redundant: the cell-generic claims data model already generates the
  per-relation shape once; `ProverInputs<'_, F, R>` instantiates to the per-relation type for
  free, and outputs are the verifier's own `SumcheckOutputClaims<F, R>` + the stage-level
  `ProvedStageN`. A second nominal struct per relation would be a second name for the same
  data with its own drift surface.
- **Other names for `PrepareKernel`.** `ConcreteProverSumcheck` (the discussion's working
  name) — not a sumcheck but the thing that prepares one, and confusable with both
  `ConcreteSumcheck` and the kernel trait. `ProverSlot` — "slot" names the backend field, not
  the behavior, and overclaims against the permanently-bespoke `commit`/`joint_opening`
  slots. `SumcheckBackend` — "backend" is platform-granular in this codebase
  (`JoltBackend`, `ReferenceBackend`); this trait is operation-granular.
  `SumcheckProverPreparer`/`*Factory`/`*Builder` — agent-noun forms of the same idea; std's
  precedent for the stored-minter shape is the verb phrase (`BuildHasher`, not
  `HasherFactory`).
- **Slot resolution via a generated `StageNSlots` struct instead of `PrepareSumcheck` bounds.**
  Rejected: the struct would have to be filled field-by-field in each recipe from backend
  fields — re-introducing per-stage plumbing — and jolt-verifier still couldn't name the field
  types. The trait-bound form puts the backend-field spelling in exactly one file
  (`BackendPreparer`).
- **Keep the two-call round API; fuse via the stash idiom.** Viable — `ingest_challenge`
  stashes, `compute_message` binds-then-evals — and was the earlier GPU-discussion resolution.
  Rejected here on cost grounds inverted: every `ProveRounds` impl is being rewritten by this
  PR anyway, so the fused contract is nearly free now and removes the idiom every fusing
  backend would otherwise have to know. Doing it standalone later would be all churn.
- **Do the protocol idiosyncrasy purge first, then generate drivers against the cleaner
  protocol.** Rejected: the purge is wire-breaking, which retires the legacy byte-diff pin —
  leaving this refactor (the riskier restructuring) without its strongest gate. Refactor under
  the pin; break the wire after (PR D ordering in `PLAN_OF_PRS.md`).

## Documentation

Update `book/` prover-architecture material added by #1669 (if merged by then) to describe the
generated `prove_clear` driver and `PrepareKernel`; otherwise fold into that PR's pending
docs task. `specs/clean-slate-prover.md` §Design gains a one-paragraph pointer here (its
"begin_batch generates the head" statement becomes "head and driver"). Crate-level rustdoc:
`jolt-kernels` (`PrepareKernel`), `jolt-verifier-derive` (new emission), `jolt-sumcheck`
(`ProveRounds` contract change).

## Execution

Suggested order, each step gated by the stage-granular byte-diff run:

1. **Fused `ProveRounds`** in jolt-sumcheck + engine + all existing members (naive, hand
   kernels, precommitted, twin tests). Smallest blast radius first; everything after builds on
   the final trait shape.
2. **Move + rename `ProveSumcheck` → `SumcheckKernel`** into jolt-verifier (re-export from
   jolt-kernels); add `PrepareSumcheck<F, R>` and the driver's error plumbing.
3. **`PrepareKernel<F, R>`** in jolt-kernels; migrate the naive-served slots (delete their
   trait files); audit `bytecode_read_raf_*` / `booleanity_*` (moving the stage-6b
   stage-value fold into the cycle kernel's `prepare` in the process); `BackendPreparer` in
   jolt-prover.
4. **Derive emission of `prove_clear`** + toy-stage twin tests, including `external` and
   `Option` members.
5. **Stage migrations**, simplest first: 3 → 5 → 4 → 1 → 2 → 6a → 7 → 6b, running the byte-diff
   harness after each. Stage 6b last, together with the `build_from_parts` promotion and the
   remainder-point/cycle-binding helper promotions (deleting the stage2/6a copies).
6. Sweep: acceptance-criteria greps, line-count check, clippy matrices, full suite.

## References

- `specs/clean-slate-prover.md` — parent architecture; §Invariants and §Non-Goals carried over.
- `specs/sumcheck-batch-derive.md` — the derive this PR extends.
- `specs/jolt-prover-model-crate.md` — superseded predecessor; source of the "no generated
  round loop" position this spec preserves.
- PR #1669 — the clean-slate prover; its review threads (stage6b mirrored legs, slot-trait
  double-construction, epilogue duplication) are the concrete motivation.
- `PLAN_OF_PRS.md` (repo root, untracked) — the PR sequencing this spec is "PR A" of.
- Andrew ↔ Michael design discussion, 2026-07-15 (universal `ConcreteProverSumcheck`,
  `#[SumcheckBatch]` driving the prover, fused round API, idiosyncrasy list deferral).
