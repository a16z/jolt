# Spec: Clean-Slate Prover

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @moodlezoup, Claude            |
| Created     | 2026-07-07                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

The modular prover in PR #1637 (`crates/jolt-prover` + `crates/jolt-backends`, ~14.7k + ~34k LOC)
was designed before the sumcheck abstraction stack (#1640 `SymbolicSumcheck`/`ConcreteSumcheck`,
#1653 claim data model, #1656 `#[derive(SumcheckBatch)]`) existed. It hand-carries protocol
structure the stack now owns — seven copies of the batched-sumcheck round loop, hand-ordered
challenge draws and claim absorbs, claim aggregates borrowed from the pre-refactor verifier — and
its backend seam mirrors the protocol rather than abstracting compute: 96 trait methods across
per-stage capability traits, 19 per-relation state-request structs, and magic `optimization_id`
strings, all implemented by a single `CpuBackend`. It also carries two parallel witness data paths
and a 1.29× end-to-end slowdown against `jolt-prover-legacy`. Rather than rebasing and porting that
code onto the post-#1656 world (paying a large mechanical port and then dismantling the ported
interfaces), this spec builds the prover from a clean slate as a **pure consumer of the abstraction
stack**: jolt-claims defines the algebra, jolt-verifier's relations and generated stage drivers
define the protocol structure, and the prover adds exactly two things — polynomial data and a round
loop.

Two structural commitments make the design self-checking end-to-end. First, the batched-sumcheck
*head* (per-member input claims → sum absorbs → coefficient draws → padded RLC) is **generated**:
a new `begin_batch` method on each `StageNSumchecks`, generic over a `SumcheckRecorder`, with
`verify_clear` refactored onto it — so the prover and verifier cannot drift on any of the four
load-bearing orderings even in principle. Second, every relation gets a **naive reference prover
for free**: since a relation's output `Expr` *is* its sumcheck summand, interpreting that `Expr`
with polynomial-valued leaves yields a slow but correct prover for any relation, making the
semantic ground truth a derived artifact of jolt-claims rather than hand-written kernel code.
Optimized kernels then live behind a runtime-swappable **backend registry** — implemented
internally via a small **form vocabulary** — and are equivalence-tested against the naive tier.

PR #1637 is superseded and mined for parts (recorder design, e2e harness, microbench, kernel
numerics); `jolt-prover-legacy` remains the byte-level parity oracle throughout bring-up.

## Intent

### Goal

Build a new backend-agnostic `jolt-prover` that produces `jolt-verifier` proofs by directly
consuming the `SymbolicSumcheck` / `ConcreteSumcheck` / `SumcheckBatch` abstractions, mirroring
jolt-verifier's stage structure file-for-file, with a prove-side sumcheck engine in `jolt-sumcheck`
and all heavy compute behind a runtime-swappable backend seam in `jolt-kernels` (a slot registry
over a small form vocabulary).

Key abstractions introduced:

- **`SumcheckRecorder`** (in `jolt-sumcheck`): the clear/ZK seam, salvaged from #1637 —
  `absorb_input_claims` / `absorb_round → challenge` / `finish(output_claim_values)`. The clear
  implementation appends (this is also what `verify_clear` uses, via `begin_batch`); the committed
  implementation Pedersen-commits rounds, no-ops claim absorbs, and captures the BlindFold witness.
  Homed in `jolt-sumcheck` deliberately: it sits next to the proof/committed wire types and FS
  labels it uses (`jolt-crypto`/`VectorCommitment` are already dependencies there), and it must be
  visible to jolt-verifier's generated code (below).

- **Generated `begin_batch`** (the one `jolt-verifier-derive` addition): the factored head of
  `verify_clear`, emitted per stage on `StageNSumchecks`, generic over `SumcheckRecorder`:

  ```rust
  // generated; illustrative
  pub fn begin_batch<R: SumcheckRecorder<F>, T: Transcript<Challenge = F>>(
      &self,
      inputs: &StageNInputClaims<F>,
      challenges: &StageNChallenges<F>,
      recorder: &mut R,
      transcript: &mut T,
  ) -> Result<(BatchPrelude<F>, StageNBatchingCoefficients<F>), VerifierError>
  // per-member input_claim (declaration order) → recorder.absorb_input_claims
  // → one coefficient draw per present member → 2^(max − rounds)-padded RLC;
  // BatchPrelude is the engine-form head in jolt-sumcheck (no per-stage wrapper)
  ```

  `verify_clear` becomes `begin_batch(clear recorder)` + the `verify_compressed_boolean` tail —
  a behavior-preserving refactor gated by the verifier fixture suites. Absorb-or-not is decided by
  the recorder **type**; there is no runtime boolean deciding whether transcript bytes are written.

- **`ProveSumcheck` trait**: the prove-side counterpart of a batch member. Split in two because of
  the dependency direction (`jolt-verifier` depends on `jolt-sumcheck`, so nothing in
  `jolt-sumcheck` may name `ConcreteSumcheck`):

  ```rust
  // jolt-sumcheck — object-safe; what the engine's round loop consumes
  trait ProveRounds<F: Field> {
      /// Returns the number of rounds/variables in this sumcheck instance.
      fn num_rounds(&self) -> usize;
      /// Computes the prover's message for a specific round of the sumcheck protocol.
      fn compute_message(&mut self, round: usize, previous_claim: F) -> Result<UnivariatePoly<F>, SumcheckError<F>>;
      /// Ingest the verifier's challenge for a sumcheck round.
      fn ingest_challenge(&mut self, r_j: F, round: usize) -> Result<(), SumcheckError<F>>;
  }

  // jolt-kernels — typed; pairs a ConcreteSumcheck relation with kernel state
  trait ProveSumcheck<F: Field>: ProveRounds<F> {
      type Relation: ConcreteSumcheck<F>;
      fn relation(&self) -> &Self::Relation;   // rounds / degree / instance_point_offset
      fn output_claims(&mut self) -> Result<SumcheckOutputClaims<F, Self::Relation>, KernelError<F>>;
  }
  ```

  The engine round loop takes `&mut [&mut dyn ProveRounds<F>]`; typed output extraction happens
  through per-relation instance traits whose `Relation` is a concrete binding (object-safe — see
  "The backend seam"), so heterogeneous batches need no type erasure of the claim structs.

- **`prove_batch` driver** (in `jolt-sumcheck`): the single batched-sumcheck prover. It consumes
  the `begin_batch` prelude (sums, coefficients, padded claim, max rounds/degree) plus the member
  slice and the recorder, and runs the loop mirroring the generated verifier tail: activation
  windows derived from rounds (inactive members contribute the `previous_claim / 2` constant
  poly), per-round `round_sum == running_claim` check, recorder absorb → challenge → bind, finish
  in generated `opening_values` order. Plus a prove-side **uniskip helper** mirroring
  `jolt-verifier/src/stages/uniskip.rs`, including the `b"opening_claim"` absorb position before
  the remainder coefficient draw.

- **Naive reference prover** (in `jolt-kernels`): a generic `ProveSumcheck` implementation that
  works for *any* relation, derived from the observation that a relation's output `Expr` is its
  sumcheck summand. Its state is a table per `Expr` leaf: `Opening` leaves resolve to dense
  polynomials materialized from the witness, `Challenge` leaves to scalars, and `Derived` leaves
  to their polynomial forms (eq/LT/selector tables — the same objects `derive_output_term`
  evaluates at a point, materialized over the domain). `round_poly` reuses
  `Expr::try_evaluate` (jolt-claims `claims.rs:101`) pointwise:

  ```text
  msg(t) = Σ_{y ∈ {0,1}^remaining} Expr.try_evaluate(
      |opening| table[opening].eval_partial(t, y),
      |challenge| scalar[challenge],
      |derived| table[derived].eval_partial(t, y))     for t in 0..=degree
  ```

  `bind` binds every table; `output_claims` reads each `Opening` table's final value. Cost is
  O((degree+1) · 2^rounds · |Expr|) per round — a **test oracle at harness scale (trace lengths
  ~2^4–2^10), never a performance path**. Two self-checks pin it: the engine's running-claim
  check, and (for each `Derived` leaf) the bound table's final value must equal
  `derive_output_term` at the bound point — tying the hand-written table resolvers to the
  verifier's scalar path.

- **Form-vocabulary kernel crate** (working name `jolt-kernels`): a small set of generic
  computational forms with an opaque-state, four-verb lifecycle
  (`materialize → round_poly → bind → final_evals`) — seeded from the generic shapes that already
  exist in #1637's `jolt-backends` (regular batch of products-of-linear-factors, prefix–suffix,
  one-hot/RA pushforward, Spartan outer row form, booleanity, uniskip first round), plus small
  trait families for commitment streaming, opening/RLC materialization, and BlindFold row
  operations. Within a backend, relations are expressed as **form descriptors (data)**: the
  registry's per-relation slots (see "The backend seam") are how orchestration *selects*
  implementations, while forms are how a backend *implements* many slots with few kernels.
  Fused fast paths are kernel-internal dispatch on the descriptor, never API surface. The three
  tiers form an interpreter/compiler/optimizer chain: the naive prover *interprets* the `Expr`;
  a form descriptor is the `Expr` *compiled* to a generic shape; a fused kernel is the
  *optimized* dispatch — each hop equivalence-tested against the tier above.

- **Per-relation kernel interfaces** (in `jolt-kernels`, one module per relation mirroring the
  verifier's per-relation files, e.g. `spartan_outer`): each crate-root module defines the
  slot's object-safe factory and instance traits — build compute state from the relation's
  typed dimensions and `&dyn WitnessProvider`; instances implement `ProveRounds` and own the
  `final_evals → SumcheckOutputClaims` mapping. The reference implementation lives in the
  module's sibling under `reference/` (the reference backend's directory — the seam never
  depends on it; a future optimized backend is a sibling directory implementing the same
  crate-root traits).
  `jolt-prover` stage recipes hold no compute and name no concrete kernel type: implementations
  are selected by the `JoltBackend` value, so swapping naive-backed → streaming → device never
  touches orchestration.

- **`JoltBackend` + `ProofSession`** (in `jolt-kernels`): the runtime seam. `JoltBackend<F, PCS>`
  is a struct of boxed per-kernel-entry slots — the value jolt-prover proves against; swapping,
  per-slot mixing, side-by-side comparison, and hardware-based selection are value construction,
  not compilation. `ProofSession` is backend-owned opaque state with proof lifetime
  (witness-upload residency, cross-member/cross-stage shared tables, device pools). Full design
  and the seam contract in "The backend seam".

- **Single id space end-to-end**: witness oracle lookup, kernel descriptors, and opening
  bookkeeping are all keyed by the jolt-claims ids (`JoltCommittedPolynomial` /
  `JoltVirtualPolynomial` / `JoltOpeningId`). No parallel naming layer, no per-stage row-type
  vocabulary (one carve-out: the instruction read-RAF rows — see "The backend seam", slot
  granularity), no optimization-id strings. One small derive addition supports this: an id-keyed
  constructor on `#[derive(OutputClaims)]` (`from_opening_values`, `canonical_order`-aligned) so
  generic code — the naive prover foremost — can assemble typed claim structs without naming
  fields.

### Invariants

Prover/verifier consistency (the load-bearing properties):

1. **Fiat-Shamir byte-identity.** For every stage, the prover's transcript operations (challenge
   draws, sum absorbs, coefficient draws, round-poly appends, opening-claim absorbs) are
   byte-identical to the verifier's — enforced structurally by calling the *same* generated code
   (`draw_challenges`, `begin_batch`, `opening_values`/`append_output_claims` order,
   `instance_point_offset`) rather than by convention.
2. **Clear-mode proof determinism and legacy parity.** For a fixed guest, input, and config, the
   new prover's clear-mode `JoltProof` bytes equal `jolt-prover-legacy`'s.
3. **Self-checking claim contract.** Every stage's produced `StageNOutputClaims` passes the
   verifier's generated `validate_output_claims`, and the generated `expected_final_claim` equals
   the driver's running claim — checked as hard errors during proving, not only in tests.
4. **Derived, never chosen, opening points.** The prover obtains opening points exclusively through
   the generated `derive_opening_points`; no prover-side point arithmetic.
5. **Kernels are transcript-free.** Only the engine (`prove_batch`, recorder, uniskip helper)
   touches the transcript; kernel crate APIs consume witness oracles, jolt-claims ids, and field
   elements, and return canonical field/group values — never a transcript, never Fiat-Shamir,
   never RNG state (ZK blinds are sampled host-side and passed in).
6. **Mode symmetry.** Clear and ZK proving share preparation, descriptors, and the round loop;
   only the recorder differs. The prover carries clear values internally in ZK mode; the proof
   alone hides them.
7. **Kernel/reference equivalence.** Every form and every fused fast path is equivalent to the
   naive `Expr` interpreter on the same relation — the semantic anchor is derived from the same
   `Expr` the verifier folds and BlindFold lowers, so a kernel cannot be "correct" against
   anything other than the protocol's own algebra.
8. **Backend-invariant proof bytes.** Proof bytes are a function of (guest, inputs,
   `ProverConfig`) alone: swapping any `JoltBackend` slot implementation or changing backend
   configuration (pools, chunk sizes, thread/stream counts, devices) never changes them.
   Field/group operations are exact, so this is achievable by contract — canonical value
   encodings at the seam — and enforced by running the byte-equality gates across backends.

`jolt-eval` plan:

- Existing `soundness` invariant (RedTeam): unchanged during bring-up (it exercises
  `jolt-prover-legacy`); once the new prover is e2e-complete, extend its harness to also target the
  new prover.
- Existing `transcript_symmetry`, `split_eq_bind_*`, `field_mul_scalar`,
  `source_to_jolt_expansion_equivalence`: untouched; must keep passing.
- New invariants to add via `/new-invariant` during implementation:
  - `legacy_proof_byte_equality` — clear-mode proof bytes from the new prover equal
    `jolt-prover-legacy`'s for the same (guest, input, config); stage-granular variant used as the
    bring-up harness (transcript-state equality at each stage boundary).
  - `kernel_naive_equivalence` — for every form and fused escape hatch, round polynomials and
    final evaluations equal the naive `Expr` interpreter's on the same descriptor and randomized
    small traces (Test + Fuzz). Subsumes fused-vs-generic-form equivalence: both tiers anchor to
    the interpreter. Generalizes per slot to backend-vs-reference equivalence as further
    backends arrive.
  - `prover_verifier_stage_consistency` — each stage's proof component is accepted by the
    corresponding `jolt-verifier` stage verify, on randomized traces (Test).

### Non-Goals

- **No protocol changes.** Wire format and Fiat-Shamir conventions are frozen at current `main`
  (post-#1656). The prover-idiosyncrasy transcript cleanups surveyed earlier (domain-separator
  removal, absorb rescheduling, etc.) are a separate, coordinated proof-version bump.
- **Verifier changes are bounded and behavior-preserving:** the generated `begin_batch` (+
  `verify_clear` refactored onto it), small export promotions (stage-level
  `stageN_input_{values,points}_from_upstream` aggregators, `UniskipParams` choreography,
  `stage6b::batch::build`; stable `verify_until_stage1` / stage-verify signatures), and factoring
  stage 8's transcript-mutating batch-assembly choreography (RLC gamma draws, joint claim/point)
  out of `stage8/verify.rs` into pub helpers both sides call. All gated on byte-identical verifier
  fixtures. Nothing else.
- **No GPU/accelerator backend implementation.** Only the seam that makes one implementable: the
  `JoltBackend` registry and `ProofSession` with opaque device-side state, value-semantics
  contracts, and O(degree) per-round traffic.
- **No Bolt dependency and no generated prover round loop** (unchanged from the superseded
  `jolt-prover-model-crate.md`; `begin_batch` generates the *head*, where declaration order is
  load-bearing — the loop is stage-invariant and stays ordinary Rust).
- **The naive prover is a test oracle only.** It is never benchmarked, never an e2e path, and its
  existence is not a license to defer kernel work for the e2e milestones.
- **No deletion of `jolt-prover-legacy`.** It remains the parity oracle and the production prover
  until the new prover reaches parity plus soak time; retirement is a separate decision.
- **No performance optimization campaign.** Performance is tracked (see below), not gated;
  closing any gap against legacy is follow-up optimization work.
- **No recursion/wrapper/Dory-assist integration** beyond preserving the typed stage outputs those
  tracks consume.

## Evaluation

### Acceptance Criteria

- [ ] `muldiv` and `sha2-chain` e2e (prove with the new prover, verify with `jolt-verifier`) pass
      in clear mode and ZK mode.
- [ ] Clear-mode proofs are **byte-identical** to `jolt-prover-legacy`'s on the e2e corpus
      (`muldiv`, `sha2-chain`, and an advice-exercising guest), verified by the
      `legacy_proof_byte_equality` invariant — and byte-invariant across backends
      (`JoltBackend::reference()` vs the optimized CPU backend, invariant 8).
- [ ] Per-stage prover↔verifier consistency tests exist and pass for stages 1–8 (each stage's
      component accepted by the corresponding verifier stage; transcript states equal at every
      stage boundary).
- [ ] Twin-transcript engine tests: toy members driving `begin_batch` + `prove_batch` against the
      generated `verify_clear`/`verify_zk` produce identical transcript event logs (reusing the
      `draw_recording`/`append_recording` doubles) — landed before any real stage depends on the
      engine.
- [ ] Every sumcheck relation in stages 1–7 is provable at harness scale by the naive
      reference prover: derived ids correspond one-to-one with multilinears (a relation whose
      derived term would be a product of multilinears is SPLIT into constituent derived ids,
      never hand-rolled), so the `Expr` interpreter covers every sumcheck round; uni-skip
      first-round polynomials remain the uni-skip prover's job. Every shipped form/fused kernel
      has a passing `kernel_naive_equivalence` test + fuzz target; the set of fused escape
      hatches is enumerated in the kernel crate's docs.
- [ ] `jolt-prover` invokes kernels exclusively through `JoltBackend` slots — no concrete kernel
      type is named in any stage recipe; `JoltBackend::reference()` serves every slot; the
      optimized CPU backend expresses every stage-1–7 relation as a form descriptor over the
      kernel vocabulary, with fused escape hatches kernel-internal and enumerated in crate docs
      (no optimization-id strings).
- [ ] ZK mode builds the BlindFold witness from state recorded during proving; the full-verifier
      replay runs only under `debug_assertions` as a cross-check.
- [ ] The three new `jolt-eval` invariants are added via `/new-invariant` and green.
- [ ] `cargo fmt` + `cargo clippy` (all feature combinations, `-D warnings`) + a no-default-features
      `cargo check` pass for the new crates.
- [ ] PR #1637 is closed with a supersession note; `CLEANUP_SPEC.md` and
      `jolt-prover-model-crate.md` are marked superseded by this spec.

### Testing Strategy

- **Existing tests that must keep passing:** the full workspace suite (`cargo nextest run`) in both
  standard and ZK modes, including `jolt-verifier`'s fixture suites (which are generated from
  `jolt-prover-legacy`, are unaffected by prover work, and gate the `begin_batch`/export/stage-8
  verifier refactors as byte-identical), and all current `jolt-eval` invariants.
- **New tests:**
  - Stage-granular byte-diff harness against legacy: prove the same trace with both provers,
    assert transcript-state equality at each stage boundary and byte equality of each proof
    component. This is the primary bring-up gate, landed with stage 1 and extended through stage 8.
    The harness is **backend-parameterized**: running it per backend is what gates invariant 8.
  - Recording mock backend: a `JoltBackend` whose slots record invocations — stage-recipe tests
    assert exactly the expected kernel calls, with no compute.
  - Per-stage prover↔verifier consistency tests (randomized traces), stages 1–8 — the coverage the
    #1637 crate only had for stages 0–2. These can run against **naive members before a stage's
    forms exist**, decoupling stage bring-up from kernel work.
  - Twin-transcript engine tests (toy members, see acceptance criteria) — the engine is validated
    against the generated verifier drivers before stage 1 exercises it.
  - `kernel_naive_equivalence` fuzz targets (via `jolt-eval` `fuzz_invariant!`), including the
    `Derived`-table cross-check (bound table's final value vs `derive_output_term` at the bound
    point).
  - E2e tests in both feature modes of the new prover crate (default and `zk`), mirroring #1637's
    `tests/e2e.rs` harness (salvaged).
- ZK-mode coverage is required from the first stage (the recorder makes it cheap), not deferred to
  the end.

### Performance

Per decision: **no acceptance gate — track only.** Correctness (byte parity + e2e) gates
acceptance; performance is measured and recorded so regressions and the gap to legacy are visible,
with optimization as follow-up work.

- Salvage #1637's `e2e_micro` harness (sha2-chain 2^16-class, prove+verify measured loop,
  calibrated iteration count) to report the new-prover / legacy ratio in both modes. For
  reference, the superseded #1637 prover sat at 1.29× transparent.
- `jolt-eval` plan: existing `prover_time_*` objectives (fibonacci, sha2-chain, ecdsa) currently
  measure `jolt-prover-legacy` and are unaffected. Add, via `/new-objective`, new-prover variants
  (e.g. `prover_time_sha2_chain_100_modular`) so the optimizer can track the ratio once the e2e
  path exists. Long-term objective is parity or better; no date or gate attached here.
- Memory: no budget gated; the witness plane must not *require* materializing polynomial copies
  the legacy prover streams (zero-copy views are a design requirement, not a benchmark gate).
- The naive tier is excluded from all measurement.

## Design

### Architecture

> **Update (`specs/prover-stage-drivers.md`):** the derive now generates the
> prove-side stage *driver* too — `begin_batch` generates the head, and the
> generated `prove_clear` owns everything between the head and the engine
> (prepare in declaration order, typed extraction, shape validation, the
> `expected_final_claim` hard check, and the finish absorb), reached through
> the universal `PrepareKernel<F, R>` backend trait. "begin_batch generates
> the head" below should be read as "head and driver".

Layering (each crate one role; the verifier is the executable protocol spec):

```text
jolt-claims      symbolic algebra: SymbolicSumcheck, Expr, ids, claim structs, geometry
jolt-verifier    protocol structure: ConcreteSumcheck relations, #[derive(SumcheckBatch)]
                 StageNSumchecks + generated drivers incl. begin_batch, stage outputs,
                 uniskip params, stage-8 batch-assembly helpers
jolt-sumcheck    sumcheck engine, both sides: existing verify drivers + NEW SumcheckRecorder
                 (clear/committed), ProveRounds, prove_batch, uniskip prover
jolt-kernels     NEW compute crate: the backend seam (JoltBackend slot registry, ProofSession,
                 per-relation slot traits) + ALL prover compute — ProveSumcheck + naive
                 reference backend, form-vocabulary sumcheck kernels (opaque state, 4-verb
                 lifecycle), commitment streaming, opening/RLC materialization, BlindFold
                 row ops; CPU implementation; fused fast paths internal
jolt-witness     one oracle-view interface keyed by jolt-claims ids; typed-row extraction
                 becomes a materialization strategy behind it, not a parallel API (one
                 exception: the instruction read-RAF rows accessor — bit-level relation
                 data no oracle view carries; see "The backend seam")
jolt-prover      NEW orchestration crate: per-stage recipes mirroring jolt-verifier's
                 stages/, transcript sequencing, kernel invocation through JoltBackend
                 slots, proof assembly, BlindFold witness recording — no field-element
                 compute, no concrete kernel types
```

**Dependency direction is load-bearing:** `jolt-verifier` depends on `jolt-sumcheck`, so the
engine pieces homed in `jolt-sumcheck` (`SumcheckRecorder`, `ProveRounds`, `prove_batch`) must not
name `ConcreteSumcheck` or any generated type — they are generic over plain data (sums,
coefficients, member slices) and the recorder. That same direction is what lets the *generated*
`begin_batch` be recorder-generic: jolt-verifier's generated code can name
`jolt_sumcheck::SumcheckRecorder`, not vice versa. The typed `ProveSumcheck` (bounded on
`ConcreteSumcheck`) therefore lives in `jolt-kernels`, which sits above `jolt-verifier`;
`jolt-prover` sits above all of them and names only traits.

What the prover calls from the generated verifier surface (all transcript-pure or
value-independent, except `begin_batch`/`draw_challenges` whose transcript effects are the point):
`draw_challenges`, `begin_batch`, `empty_input_points`, `derive_opening_points`,
`expected_final_claim` (as a hard self-check), `opening_values` / `append_output_claims`,
`output_claim_count`, `instance_point_offset` / `instance_point`, the per-relation
`*_input_{values,points}_from_upstream` wiring, and `validate_inputs_from_parts` /
`verify_until_stage1`. Only the `verify_clear` tail and `verify_zk` have no prover use;
`prove_batch` is their mirror, validated against them by the twin-transcript engine tests.

Module layout mirrors the verifier exactly, twice over: `jolt-prover/src/stages/stage{1..8}/`
holds the stage recipes, and `jolt-kernels`' per-relation modules match the verifier's
per-relation files, plus `stage0/` (witness commitment, which the verifier handles in
preprocessing/proof validation). Stage recipes have the same shape as `verify.rs` with the
proof-read swapped for proof-write:
`draw_challenges → assemble StageNInputClaims from upstream outputs → prepare kernel instances
through the backend's slots → begin_batch(recorder) → prove_batch(recorder) →
derive_opening_points → assemble StageNOutputClaims → expected_final_claim check →
append_output_claims → build StageNClearOutput`.
Cross-stage state is the verifier's own `StageNClearOutput` types in both modes (the prover knows
all values in ZK; the proof hides them). Stages 1–2 open with the shared uniskip prover; stage 6's
carried challenges reuse `Stage6aCarriedChallenges`/`Stage6bCarriedChallenges`; stages 4/6b
hand-order their absorbs exactly where the verifier's `no_opening_values`/`no_draw_challenges`
opt-outs mark stage-curated behavior.

**Cross-member and cross-stage kernel state is a first-class concern**, not an afterthought:
batch members genuinely share compute state (eq tables shared across members — the
`SharedRaPolynomials` precedent in legacy — and the stage-6a→6b kernel-state carry, where
address-phase bound state becomes cycle-phase input). The `ProofSession` owns this sharing (see
"The backend seam"): shared resources are materialized once, backend-side, and referenced by the
slot instances; state that spans 6a→6b lives in the session, while a member spanning two
*batches* (the 6b→7 claim reductions) is a boxed instance the recipe holds across the boundary.
Getting this wrong silently is impossible (byte-parity + consistency gates) but getting it wrong
*expensively* is easy — it is the main reason stage 6 is budgeted as the hardest slice after
stage 1.

**The naive tier in practice.** The naive prover turns the verifier's relation catalog into a
complete (slow) prover: any relation whose output-`Expr` leaves can be resolved to tables is
provable with zero kernel code. Leaf resolvers are the only hand-written content — per
`Derived` id, the polynomial form of what `derive_output_term` computes at a point (eq/LT/init
selector tables), pinned by the final-value cross-check; `Opening` leaves come from the witness
plane by id (including virtual polynomials); `Challenge` leaves from the drawn `Challenges`
struct. Uniskip first rounds (extended univariate domain) remain the uniskip prover's job; the
naive tier covers standard rounds. Its three roles: (1) bring-up — stage recipes land and pass
lockstep/consistency gates with naive members before their forms exist; (2) semantic anchor —
`kernel_naive_equivalence` gives every form and fused kernel a ground truth derived from the same
`Expr` the verifier folds and the BlindFold R1CS lowers (one algebra, three consumers); (3)
future-proofing — a new relation added to jolt-claims is provable before anyone writes its kernel.

ZK path: identical stage recipes with the committed recorder; round-poly Pedersen commitments and
output-claim commitments captured as `CommittedSumcheckWitness` per stage; BlindFold publics
(challenges, points, folded claims) recorded during proving via the same generated drivers, so the
BlindFold witness assembly consumes recorded state directly. The #1637 pattern of re-running the
full verifier to recover those values is demoted to a `debug_assertions` cross-check. The
per-stage committed witnesses are carried in a named-field struct type-linked to the BlindFold
domain order (no positional `Vec` invariants). The symbolic `Expr` feeding `try_evaluate`, the
naive prover, and the BlindFold R1CS lowering means no prover-side claim/constraint
synchronization exists at all — the property the legacy prover maintains by hand.

Salvage inventory from #1637 (parts, not interfaces): recorder design (`stages/recorder.rs`,
`committed.rs`), e2e test harness (`tests/e2e.rs`), microbench (`benches/e2e_micro.rs`), CPU kernel
numerics (`jolt-backends/src/cpu/**` — regular-batch, prefix-product, pushforward, Spartan
remainder, booleanity round/bind kernels), the generic form descriptions
(`SumcheckRegularBatch{LinearTerm,LinearFactor,Product,Instance}`), streaming commitment code, and
the BlindFold row-committer kernels. From `jolt-prover-legacy`: proven polynomial data structures
already in `jolt-poly` (compact scalars, split-eq, prefix–suffix machinery).

### The backend seam (`jolt-prover` ↔ `jolt-kernels`)

**End state.** Orchestration never names a concrete kernel. `jolt-prover` is written against a
**backend value** — a struct of independently swappable kernel slots plus an opaque proof-scoped
session — so *which implementation computes a thing* is decided by constructing a value at
runtime, while *what is computed and absorbed* stays fixed by the generated drivers. This buys,
without recompilation: per-kernel swapping (naive → optimized CPU → device), running two provers
with different kernel sets side by side in one binary, and choosing the kernel configuration at
startup from the hardware.

Three layers, each answering a different question:

- **The registry is the swap point.** `JoltBackend<F, PCS>` holds one boxed object-safe slot per
  kernel entry; stage recipes take `&JoltBackend` plus `&mut` session.
- **The form vocabulary is the second-backend leverage.** An optimized backend implements the ~6
  generic form kernels and serves most slots through thin shared adapters (relation → descriptor
  compilation is shared code derived from the `Expr`); only fused fast paths are bespoke.
- **The naive tier makes slots ~free.** `JoltBackend::reference()` implements every slot through
  the one generic `NaiveSumcheckProver` plus per-relation leaf resolvers — the always-present
  fallback and the equivalence anchor. Derived ids correspond one-to-one with multilinears —
  a relation whose derived term would be a product of multilinears is split into constituent
  derived ids instead (the RAM output check's `EqIoMask` and the Spartan outer remainder's
  quadratic coefficients were both dissolved this way), so every batch member is naive-provable
  and the only hand-written reference compute is uni-skip first-round polynomials.

```rust
// jolt-kernels — illustrative. PCS carries only CommitmentScheme: streaming
// is the reference implementation's strategy, not a seam requirement.
pub struct JoltBackend<F: Field, PCS: CommitmentScheme<Field = F>> {
    pub commit: Box<dyn CommitWitness<F, PCS>>,
    pub spartan_outer: Box<dyn SpartanOuterProver<F>>,
    // one slot per kernel entry, added as stages land
}

// one module per relation: object-safe factory + instance traits
pub trait SpartanOuterProver<F: Field> {
    fn prepare(&self, session: &mut ProofSession, log_t: usize, tau: &[F],
               witness: &dyn WitnessProvider<F, JoltVmNamespace>)
        -> Result<Box<dyn SpartanOuterInstance<F>>, KernelError<F>>;
}
pub trait SpartanOuterInstance<F: Field> {
    fn uniskip_first_round_poly(&self) -> Result<UnivariatePoly<F>, KernelError<F>>;
    fn into_remainder(self: Box<Self>, r0: F)
        -> Result<Box<dyn OuterRemainderInstance<F>>, KernelError<F>>;
}
pub trait OuterRemainderInstance<F: Field>: ProveRounds<F> {
    fn output_claims(&mut self)
        -> Result<SumcheckOutputClaims<F, OuterRemainder<F>>, KernelError<F>>;
}
```

**Slot granularity** is per kernel entry, matching the per-relation kernel modules: entries that
share prepared state stay one slot (uniskip + remainder above share the materialized input
tables), and a slot's inputs are the relation's own typed dimensions and challenges plus
`&dyn WitnessProvider` — jolt-claims ids everywhere, no request structs, no magic strings.

*The one witness-channel exception:* the instruction read-RAF slot takes its witness as typed
per-cycle rows (`Stage5InstructionReadRafRow`: packed 128-bit lookup index, table selection,
operand-interleave flag) instead of `&dyn WitnessProvider`. That relation's witness is not a
multilinear oracle — the kernel consumes the index *bits* (chunking, uninterleaving, suffix
MLEs), which no field-element polynomial view carries losslessly — so the recipe fetches the
rows through jolt-witness's typed-rows accessor (`JoltVmStage5InstructionReadRafRows`, the reason
`prove_stage5`'s witness parameter is generic rather than the plain trait object) and passes
them to `prepare` as typed relation data. The rows are proportional to `T`; the seam's
O(rounds·degree)-traffic property (invariant 2 below) is about kernel *outputs* and
kernel-internal state, which remain unexposed.

**Partial backends compose at construction.** Device backends arrive kernel-by-kernel, so mixed
execution is the steady state, not an edge case: a backend is built over a fallback per slot,
selection is explicit and logged, and capability misses surface at construction/plan time as
recoverable `KernelError` variants (`Unsupported`, `ResourceExhausted`) — never #1637's mid-proof
`UnsupportedTask`. Side-by-side comparison is two `JoltBackend` values; hardware-based choice is
a constructor decision. Cargo features never select among compiled-in backends (they may gate
*linkage*, e.g. a CUDA toolchain dependency).

**Why this is not #1637's relation-major seam** (alternative 2): the failure there was not
per-relation width but that the per-relation surface was the *only* abstraction — 96 bespoke
methods, 19 request structs, magic ids — so a second backend meant reimplementing the protocol
surface. Here a second backend implements the form vocabulary (or even a single slot, composing
over a fallback for the rest); the reference backend derives every slot from one interpreter; and
the slot inputs are the same typed relation data the verifier defines. Shipping incrementally is
the designed path, not a failure mode.

**`ProofSession`** — backend-owned state with proof lifetime, created by
`backend.begin_proof(…)`, threaded `&mut` through every slot call, contents opaque to
orchestration (a boxed backend-private type the backend downcasts internally). It is where
residency and reuse live: witness uploads cached across stages (upload-once-reuse-many keyed by
committed-polynomial id — the commit-time stream must not be re-uploaded for read-write-checking
stages), cross-member shared tables (the `SharedRaPolynomials`/eq-table precedent), the 6a→6b
`ra_indices` carry (~56·T bytes in legacy), device memory pools and per-stage plans.
Recompute-vs-share becomes a backend decision with a home (legacy regenerates the Inc witnesses
from the trace four times and the G pushforward at two different points — reasonable choices
that a backend may revisit per device). Cross-stage state the *orchestrator* owns is just boxed
instances: the 6b→7 spanning claim reductions are members whose cycle rounds run in the 6b batch
and address rounds in the 7 batch — the recipe holds the `Box<dyn …>` across the boundary and
re-enters it, with the phase transition a method on the instance trait.

**Dyn-feasibility constraints** (compile-probed against the current code):

- `WitnessProvider` is dyn-compatible as-is (no associated types, no generic type-parameter
  methods; streams already return boxed) — slots take `&dyn WitnessProvider<F, JoltVmNamespace>`
  directly, no erasure shim.
- `CommitmentScheme`/`StreamingCommitment` are **not** object-safe (`Clone`/`Eq` supertraits,
  no-receiver generic methods), and `PCS::{Output, OpeningHint, ProverSetup}` are wire/plumbing
  types — so `PCS` stays a type parameter of the backend, like `F`. Both are deployment
  constants, not swap targets.
- `ProveRounds` is object-safe (already consumed as `&mut dyn`); consuming handoffs use
  `self: Box<Self>`; per-relation instance traits returning the concrete
  `SumcheckOutputClaims<F, R>` are object-safe — which is what replaces inherent `output_claims`
  methods (unreachable behind `Box<dyn ProveRounds>`).

**Seam contract** (trait-doc law, device-readiness by construction):

1. *Value semantics.* Kernel outputs are canonical field/group **values** — fully reduced
   encodings, affine points. Internal representation (Montgomery form, lazy reduction,
   projective accumulation, sharding, parallel reduction order) is free: field/group operations
   are exact, so reordering is safe for values, and canonical boundaries make it safe for bytes.
   The real hazards are representational — incomplete EC addition formulas reached by different
   schedules, integer-domain accumulators without stated width obligations — and are contract
   items, not folklore.
2. *O(rounds·degree) traffic.* Only round polynomials, claims, and commitments cross the seam;
   nothing T-sized does in the output direction. No accessor exposes kernel-internal tables.
   (Prepare-time *inputs* may be T-sized typed relation data — the read-RAF rows. One
   output-direction carve-out: the joint-opening slot returns `MultilinearPoly` *handles* over
   the grid-embedded committed polynomials, because the PCS opening must drive their
   `fold_rows` — the handles may be lazy or device-backed; only the PCS's O(√N)-sized fold
   results cross back.)
3. *Async-compatible wording.* Contracts promise "returns the values," never "the work has
   completed"; `ingest_challenge` may enqueue. The per-round Fiat-Shamir challenge is the only
   protocol-forced synchronization point; everything else (members within a round, commitment
   MSMs vs sumcheck compute, next-stage prefetch) may overlap.
4. *Host-owned randomness.* ZK blinds are sampled by jolt-prover and passed in; kernels hold no
   RNG state. For a fixed seed, ZK proofs are backend-identical.
5. *Observability.* Every slot call is a tracing span (stage, relation id, sizes); the backend
   records which implementation served each slot — silent fallback is a bug, not a feature —
   and timing is device-event-aware rather than host wall-clock around async calls.

**Backend config is not protocol config.** `ProverConfig` (trace length, `ram_K`, one-hot/rw
shapes) determines proof bytes; backend config (pools, chunk sizes, stream counts, spill
thresholds, device ids) must be proof-invariant — invariant 8 makes the distinction testable.

**Known engine extension, deferred.** `prove_batch` iterates members host-side — one readback
per member per round on a device backend. When one lands, the engine grows a *member group*: one
`ProveRounds` serving N co-located members and returning the pre-folded round polynomial given
the batch coefficients. Nothing in the seam blocks this; recorded here so per-member iteration
is not contracted as seam law.

### Alternatives Considered

1. **Port #1637 onto the new stack** (rebase 10a/10b/10c, mechanical repoint, then refactor in
   place — the previously sketched plan). Rejected: the port's first phase is spent making stale
   interfaces compile against the new world, and the following phases dismantle exactly those
   interfaces; the durable value of #1637 (kernels, recorder, harness) is salvageable as parts
   without paying that toll. The port also inherits the relation-major backend seam and dual
   witness paths as debt.
2. **Relation-major backend API** (#1637's `jolt-backends` design: per-relation
   materialize/evaluate/bind/output quadruples behind per-stage capability traits). Rejected: 96
   methods and 19 request structs make a second backend a rewrite of the protocol surface;
   "backend-agnostic" requires the seam to be a small computational vocabulary. The four-verb
   opaque-state lifecycle itself is kept — it is the right shape for device-resident state. The
   `JoltBackend` slot registry ("The backend seam") is per-relation *at the selection layer* but
   escapes this failure mode: slots are independently implementable with fallback composition,
   reference implementations derive from the one naive interpreter, and optimized backends serve
   most slots through the form layer — a second backend implements the form vocabulary, not the
   protocol surface.
3. **Generic forms only, no fused escape hatches.** Rejected on performance grounds: legacy's
   specialized kernels exist because generic evaluation leaves large constants on the table
   (shared eq tables, one-hot phase switching, small-scalar arithmetic). Fusion stays — but as
   kernel-internal dispatch on the descriptor, verified equivalent by fuzzing.
4. **Forms from day one, no naive tier.** Rejected: it serializes stage bring-up on kernel work
   and leaves fused-kernel equivalence anchored to the generic form — itself new code — rather
   than to the protocol's own `Expr`. The naive interpreter costs one generic implementation and
   buys a semantic oracle plus per-relation incremental kernel replacement.
5. **Engine in `jolt-prover` instead of `jolt-sumcheck`.** Rejected per decision: `jolt-sumcheck`
   already owns the proof/committed types and the verify drivers; housing `prove_batch` there
   makes it the complete sumcheck crate and keeps the engine reusable (BlindFold's own sumchecks,
   future provers). The dependency direction bounds what can live there (see Architecture); the
   typed `ProveSumcheck` stays in `jolt-prover`.
6. **Generated prover drivers** (the `SumcheckBatch` derive emitting `prove_clear`/`prove_zk`
   inside jolt-verifier, e.g. behind a feature). Rejected: Cargo feature unification would make
   the verifier's API vary with the build graph; prover logic would live in the verifier crate;
   and the round loop is stage-invariant given `(rounds, coefficient)` — it gains nothing from
   per-member token-stream expansion. The declaration-order *head* is the only part worth
   generating, and `begin_batch` is exactly that at a fraction of the macro surface.
7. **`begin_batch(…, absorb_sums: bool, …)`** instead of the recorder generic. Rejected: a runtime
   boolean deciding whether transcript bytes are written, at every call site on both sides,
   recreates the silent-drift class this design exists to kill. The recorder type decides; homing
   the trait in `jolt-sumcheck` is what makes that clean.
8. **A shared `jolt-protocol` crate** (moving `ConcreteSumcheck` + aggregates below both sides).
   Deferred: the one-way `jolt-prover → jolt-verifier` dependency already achieves sharing. Flip
   conditions, recorded: a consumer that must not link jolt-verifier; a crates.io semver surface
   for the verifier; the derive needing to generate into jolt-prover.
9. **Defer the kernel crate until a second backend exists (rule of two).** Rejected per decision:
   a separate crate from day one enforces the transcript-free, id-keyed API discipline and lets
   accelerator work proceed in parallel; the risk of the seam ossifying wrong is bounded by the
   deliberately small form vocabulary — and by the naive tier, which keeps the vocabulary honest
   (a form that cannot be checked against the interpreter is suspect).
10. **Generated prover (Bolt track).** Out of scope, unchanged from `jolt-prover-model-crate.md`;
    this crate exists precisely to unblock modular proving without waiting on it.
11. **Generic `B: Backend` type parameter instead of dyn slots.** Rejected: monomorphized
    backends make per-slot mixing combinatorial and side-by-side comparison a recompilation;
    kernel invocations are macro-scale (whole sumchecks, whole commitment passes), so dyn
    dispatch costs nothing. `F` and `PCS` stay as type parameters only because they are
    deployment constants and `CommitmentScheme` is structurally non-object-safe (verified), with
    `PCS::Output`/`OpeningHint` embedded in the wire types.
12. **A single descriptor-executor entry point as the whole seam** (`run(form_descriptor)` as
    the only backend method). Rejected: typed claim extraction, bespoke lifecycles (uniskip →
    remainder handoff, 6a→6b carry, 6b→7 phase transitions), and witness-access strategy do not
    fit a pure-data call. The descriptor layer lives *below* the registry, as the optimized
    backend's implementation strategy — the registry stays the swap point.
13. **Cargo-feature backend selection.** Rejected: features are anti-runtime (no side-by-side,
    no hardware probing) and unify across the build graph. Features may gate linkage (a CUDA
    toolchain dependency), never choice among compiled-in backends. Contrast the `zk` seam,
    where the recorder *type* deciding is the point — there the two modes must not coexist
    within one proof; here coexistence is the requirement.

## Documentation

- New Jolt book page under the architecture section: the modular prover — crate layering diagram,
  the backend registry/session seam, the form vocabulary and the interpreter/compiled/fused
  tiering, the prover-as-consumer-of-the-verifier model, and the parity-oracle relationship to
  `jolt-prover-legacy`.
- Update the batched-sumcheck page for `begin_batch` (the generated head shared by both sides).
- Update any book references to the prover stack that name `jolt-backends`.
- Mark `specs/jolt-prover-model-crate.md` superseded by this spec (its ownership table and
  non-goals largely carry over; the kernel seam and engine placement change).
- `CLAUDE.md` prover-workflow notes (test commands for the new crates) once the e2e path lands.

## Execution

Bring-up is **pipeline order** (each stage needs upstream outputs; the byte-diff harness supplies
real upstream data, so no synthetic-witness scaffolding is needed), every slice gated by the
stage-granular legacy byte-diff harness plus clippy/nextest:

0. **Scaffolding:**
   - Verifier-delta PR: generated `begin_batch` + `verify_clear` refactor, export promotions,
     stage-8 assembly factoring — all fixture-gated. *Open decision: fold into #1656 before it
     merges (it is the macro PR) or land as the first follow-up.*
   - `jolt-kernels` + `jolt-prover` crates; `SumcheckRecorder` + `ProveRounds` + `prove_batch` +
     uniskip prover in `jolt-sumcheck`; form traits + the regular-batch form (lifted from #1637)
     in `jolt-kernels`.
   - The naive reference prover + the `from_opening_values` derive addition.
   - Twin-transcript engine tests (toy members vs `verify_clear`/`verify_zk`) — the engine is
     proven against the generated drivers before any stage uses it.
   - The byte-diff harness against legacy (transcript-state assertions + per-component byte
     equality).
1. **Stage 0** (witness commitment + transcript preamble) — needed before any stage can byte-match.
2. **Stage 1** (Spartan outer uniskip + remainder; exercises the uniskip prover and the Spartan
   row form; the one-member batch is the smallest `prove_batch` use). The streaming remainder is
   the single hardest kernel in the plan — first for gate reasons, budgeted accordingly.
3. **Backend-seam retrofit** (immediately after stage 1, while the surface is two kernel
   entries): slot/instance traits + `JoltBackend` + skeletal `ProofSession`;
   `JoltBackend::reference()` wraps the existing naive-backed kernels; stage 0/1 recipes
   re-pointed at the registry; the byte-diff harness takes the backend as a parameter (bytes
   must not move). Stages 2+ then land *on* the seam instead of adding migration debt.
4. **Stages 2 → 7** in order, one PR-able slice each; each slice adds the stage's per-relation
   slots, any new form (prefix–suffix at stage 5, pushforward/booleanity at stage 6, the
   6a→6b state-carry), and its consistency tests. A slice may land **naive-first** (recipes +
   gates green with naive members at harness scale), with forms replacing naive per relation,
   each replacement gated by `kernel_naive_equivalence`; the e2e milestones require forms
   (the naive tier is orders of magnitude too slow at real trace lengths). Indicative
   relation→form mapping (escape hatches documented as they ship): claim reductions and Spartan
   shift/instruction-input → regular batch of products; read-write checking and val-evaluation
   relations → eq-weighted product forms; instruction read-RAF → prefix–suffix; RA
   virtualizations → one-hot pushforward; booleanity relations → booleanity form; advice phases →
   dense eq-product.
5. **Stage 8** (joint opening) — via the stage-8 helpers factored in step 0, plus prover-only
   hint combination and PCS opening over the streaming RLC source.
6. **ZK e2e:** committed recorder throughout, BlindFold witness from recorded state, replay as
   debug cross-check; `muldiv` ZK green.
7. **Wrap-up:** `jolt-eval` invariants/objectives registered, microbench revived, book page,
   supersession notes, close #1637.

Feature flags on the new crates follow the existing convention: `zk` mirrors
`jolt-verifier/zk`; no `host`-style gating is needed (these crates are host-only by construction).

## References

- PRs: [#1637](https://github.com/a16z/jolt/pull/1637) (superseded jolt-prover/jolt-backends),
  [#1640](https://github.com/a16z/jolt/pull/1640) (SymbolicSumcheck/ConcreteSumcheck split),
  [#1653](https://github.com/a16z/jolt/pull/1653) (sumcheck instance data model),
  [#1656](https://github.com/a16z/jolt/pull/1656) (`#[derive(SumcheckBatch)]` stage drivers).
- Specs: `symbolic-sumcheck.md`, `sumcheck-instance-data-model.md`, `sumcheck-batch-derive.md`,
  `self-contained-sumcheck-relations.md`, and the superseded `jolt-prover-model-crate.md`.
- Code anchors: `crates/jolt-verifier/src/stages/relations.rs` (`ConcreteSumcheck`, the recording
  transcript doubles), `crates/jolt-claims/src/claims.rs:101` (`Expr::try_evaluate` — the naive
  prover's evaluator), `crates/jolt-prover-legacy/src/subprotocols/sumcheck_prover.rs`
  (`SumcheckInstanceProver` — the owned-instance precedent for `ProveRounds`).
- In-crate prior art on the superseded branch: `crates/jolt-prover/CLEANUP_SPEC.md`
  (readability contract; its stage-recipe and ownership principles carry over),
  `crates/jolt-backends/src/sumcheck/request.rs` (the generic regular-batch form this spec
  generalizes).
- [`jolt-eval` framework](../jolt-eval/README.md) for the invariants/objectives named above.
