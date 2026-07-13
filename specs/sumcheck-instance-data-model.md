# Spec: Sumcheck Instance Data Model

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Michael Zhu                    |
| Created     | 2026-06-26                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

The [symbolic/concrete sumcheck split](symbolic-sumcheck.md) gave each relation a
pure-algebra `SymbolicSumcheck` (in `jolt-claims`) and a resolution/wiring
`ConcreteSumcheck` (in `jolt-verifier`). But the *non-opening* data each sumcheck
consumes and produces is still threaded ad-hoc: Fiat-Shamir challenges are drawn
inline in each stage's `verify` fn and forwarded to later stages through
hand-written `StageNPublicOutput` structs, and the `Public`/`Challenge` resolvers
are entangled with per-stage code. There is no single place that says "here is
everything this sumcheck needs, and here is everything it hands downstream."

This spec makes a sumcheck instance's full data interface first-class. Each
instance declares, as derive-linked structs, the openings it **consumes**
(`InputClaims`), the openings it **produces** (`OutputClaims`), and the challenges
it **draws** (`Challenges`); its verifier-evaluable (`Derived`) values are resolved
through two point-aware methods split along the input/output boundary. The claim
data-model (the cells, the `*Claims` traits, the derive macros) moves into
`jolt-claims`, single-sourcing the canonical transcript order and id↔value
resolution for both prover and verifier, while transcript I/O stays in
`jolt-verifier`. `Source::Public` is renamed `Source::Derived`. This is a
**mostly-lateral** refactor whose payoff is a follow-up that abstracts sumcheck
*batching*; it deliberately stops short of that.

## Intent

### Goal

Make each sumcheck instance fully declare its input and output data — consumed and
produced openings, and drawn challenges — as derive-linked structs, with
verifier-evaluable (`Derived`) values resolved through point-aware methods, and
relocate the claim data-model into `jolt-claims` so the canonical transcript order
and id↔value resolution are single-sourced for both prover and verifier.

The model underneath the design (established during design review): for any
sumcheck, distinguish the **data** it exchanges from the **symbols** in its
expressions. The non-opening data a sumcheck passes downstream — call it `(a)` — is
its sumcheck binding challenges plus the Fiat-Shamir draws made before each bind
(batching coefficients, structural references like Spartan's `tau`); it excludes
the post-bind opening-proof randomness. The non-opening input data it receives —
`(1)` — is upstream `(a)` plus public/preprocessing data (e.g. RAM initialization).
The non-opening **symbols** in its input expression `(2)` resolve from `(1)` plus
its own pre-bind draws; the symbols in its output expression `(b)` resolve from
`(1)` and `(a)`. There is no identity map between data and symbols — they are
related by *resolution*, not by being the same objects. This spec gives `(a)`,
`(1)`, `(2)`, `(b)` explicit homes.

Key abstractions introduced or modified:

- **Three `Source` kinds, with `Public` renamed `Derived`.** The principled split is
  `Opening` vs everything-else: an `Opening` is a value the verifier *cannot* obtain
  cheaply, so it is committed (and hidden in ZK); everything else the verifier
  obtains without an opening. That "everything else" subdivides by *how* the verifier
  obtains it: `Challenge` is **sampled** from the transcript (point-free), `Derived`
  is **computed** by the verifier from points / challenges / statement data
  (`eq`/`lt` evaluations, Lagrange weights, R1CS coefficients, `val_check`'s
  `InitEval`). `Challenge` and `Derived` stay separate because they have distinct
  resolution mechanisms (a drawn `Challenges` struct vs. computed on demand) — see
  Alternatives for why they are not merged into one "Aux" kind.
- **`SymbolicSumcheck` gains data-shape associated types and stays field-independent.**
  In addition to the four id types and `Shape`, it declares `type Inputs<C>`,
  `type Outputs<C>` (generic over an opening *cell*), and `type Challenges<F>`. The
  expression methods stay method-generic (`input_expression<F>`/`output_expression<F>`),
  so a symbolic object remains F-free and emits `Expr<F, …>` for any field. Lives in
  the framework half of `jolt-claims`.
- **`ConcreteSumcheck<F>` reworks resolution.** `resolve_public` splits into
  `derive_input_term` (resolves a `Derived` in the *input* expression; no produced
  openings, no bound point) and `derive_output_term` (resolves a `Derived` in the
  *output* expression; sees the produced openings' points). Adds `draw_challenges`
  (populate the `Challenges` struct from the transcript). The `InputClaims`/
  `OutputClaims`/`SumcheckChallenges` bounds are applied at this use site, not on the
  field-independent symbolic trait.
- **The claim data-model moves to `jolt-claims`, split by responsibility.** The
  *canonical-order + id-resolution* half — the opening cells (`OpeningClaim`,
  `GetPoint`, `GetValue`), the `InputClaims`/`OutputClaims`/`SumcheckChallenges`
  traits, and their derive macros — moves to `jolt-claims`, generic over the id
  types. The *transcript-I/O* half (`append_openings`, `draw_challenges`) stays in
  `jolt-verifier` as a thin consumer of the canonical order. `opening_values()`
  (field-declaration order) remains the single source of opening order; `jolt-claims`
  stays transcript-free.
- **Uni-skip pairs become two sequential sumchecks** running the standard instance
  lifecycle, removing the mid-sumcheck interleaved draw special case.
- **(Optional) `jolt-claims` splits into framework + protocols crates** — a clean
  layering boundary that the generalized claim traits enable.

### Invariants

- **Prover/verifier consistency (primary, inherited).** A relation's point
  derivation, challenge/derived resolution, and expected-output algebra MUST be
  identical on both sides; they are shared through `ConcreteSumcheck`.
- **No proof-format, transcript, protocol, or sumcheck-set change.** Proofs produced
  before and after MUST verify identically. The transcript byte stream and the
  opening claim order MUST be byte-identical. (Verified prerequisite: `JoltRelationId`
  is neither serialized into the proof nor absorbed into the transcript, so relation
  ids are free to restructure; the data-model relocation and `Derived` rename are
  verifier-internal.)
- **Canonical opening order single-sourced.** `OutputClaims::opening_values()`
  (field-declaration order) is the one definition of opening order; the
  jolt-verifier `append_openings` MUST iterate exactly that order (the existing
  "append matches values" test stays as the guard).
- **Challenge draw order preserved exactly.** `draw_challenges`, sequenced by the
  verifier, MUST reproduce the current per-stage Fiat-Shamir draw sequence
  byte-for-byte. This is the soundness-sensitive part of the refactor.
- **Input/output `Derived` split is value-preserving.** `derive_input_term` +
  `derive_output_term` together MUST resolve every `Derived` exactly as today's
  `resolve_public(id, inputs, outputs)` did — input-side with no produced openings
  (the input claim precedes binding), output-side with the produced openings' points
  (the output claim follows binding). The bound point is available to output
  resolution and unavailable to input resolution; that asymmetry is the rule, not an
  incidental signature difference.
- **`jolt-claims` stays transcript-free.** No `Transcript` dependency enters
  `jolt-claims`; `append_openings`/`draw_challenges` live in `jolt-verifier`.
- **Resolution failure is a runtime rejection, not UB.** A symbol with no resolver
  yields `Err(VerifierError::Missing…)`.

`jolt-eval` framing: keep `verifier_no_panic` and the
[symbolic-sumcheck](symbolic-sumcheck.md) invariants green throughout. Add, via
`/new-invariant` during implementation: (i) a `Challenges`-coverage invariant —
every `ChallengeId` in `required_challenges()` is resolvable by the relation's
`SumcheckChallenges` impl (catches field↔symbol drift); (ii) a draw-order
equivalence invariant — `draw_challenges` over a fixture reproduces the recorded
transcript-draw sequence; (iii) a `Derived`-split equivalence invariant —
`derive_input_term`/`derive_output_term` match the prior `resolve_public` on a
fixture.

### Non-Goals

- **Not** the batching abstraction. Macros/traits that abstract how instances are
  grouped into a batched sumcheck — including drawing batch-level coefficients and
  the shared binding vector, and *sequencing* per-instance `draw_challenges` in the
  correct global order — are a **follow-up spec**. This spec only makes per-instance
  data first-class so that work has something clean to orchestrate.
- **Not** the "instance = relation id" rename. It is feasible and lateral (relation
  ids are not in the proof or transcript), but it only pays off in the batching
  orchestrator (id-as-key); it is deferred to that follow-up. This spec tolerates
  multiple instances per `RelationId` exactly as today.
- **Not** a `field_inline` verifier / concrete side. `field_inline` gets the renamed
  `Derived` kind and the relocated traits, but remains symbolic-only.
- **Not** re-introducing a lowered runtime claim struct or a `Map<id, _>` store.
  Dispatch stays static; the trait is the representation.

## Evaluation

### Acceptance Criteria

- [ ] `Source::Public` is renamed `Source::Derived`; the `public(..)` builder becomes
      `derived(..)`; `PublicId`/`required_publics` become `DerivedId`/`required_deriveds`
      across `jolt-claims` and `jolt-verifier`.
- [ ] `SymbolicSumcheck` declares `type Inputs<C>`, `type Outputs<C>`,
      `type Challenges<F>`, stays field-independent (no `F` on the trait;
      `input_expression<F>`/`output_expression<F>` method-generic).
- [ ] The opening cells (`OpeningClaim`/`GetPoint`/`GetValue`), the
      `InputClaims`/`OutputClaims`/`SumcheckChallenges` traits, and their derive
      macros live in `jolt-claims`, generic over the id types; `jolt-claims` has no
      `Transcript` dependency.
- [ ] `append_openings` and `draw_challenges` live in `jolt-verifier` and consume the
      `jolt-claims` canonical order; the "append matches `opening_values()`" test passes.
- [ ] `ConcreteSumcheck` replaces `resolve_public` with `derive_input_term`
      (`inputs`, `challenges`) and `derive_output_term` (`inputs`, `outputs`,
      `challenges`); `expected_output` pins `Outputs<OpeningClaim<F>>` (reads values),
      `input_claim` takes `(inputs, challenges)`.
- [ ] Each relation has a `Challenges` struct via `#[derive(SumcheckChallenges)]`;
      `draw_challenges` populates it; `resolve_challenge` resolves by id; per-stage
      challenge draws move out of `verify` fns into instance `draw_challenges`.
- [ ] The Spartan uni-skip/remainder pairs run as two sequential sumchecks through
      the standard lifecycle; no value is drawn mid-bind (the remainder's batching
      coefficient is drawn in the inter-sumcheck gap).
- [ ] `cargo clippy --all --features host` and `--features host,zk` pass `-D warnings`.
- [ ] `cargo nextest run -p jolt-prover-legacy muldiv --features host` and
      `--features host,zk` pass; the `advice` e2e tests pass.
- [ ] `jolt-verifier` fixture suites pass in both modes (completeness +
      soundness/tampering).
- [ ] (If the crate split is taken) the framework crate compiles standalone; the
      protocols crate depends on it; `field_inline` and `jolt` both build on it.

### Testing Strategy

Must continue passing:

- `muldiv` e2e (`jolt-prover-legacy`) in **both** `--features host` and
  `--features host,zk` — the primary correctness gate; catches transcript-order or
  BlindFold-constraint drift.
- The `advice` e2e tests — exercise the non-ZK path and `val_check`'s `Derived` input
  reconstruction; the guard for the input-side `derive_input_term`.
- `jolt-verifier` fixtures in both modes (guest toolchain; sandbox off — guest builds
  write `/tmp/jolt-guest-targets`).

New tests:

- `#[derive(SumcheckChallenges)]` unit tests (field↔`ChallengeId` mapping;
  `resolve_challenge` by id; conditional/`Option` and indexed/`Vec` fields).
- A `draw_challenges` order test per migrated stage: the chunk stream produced by the
  per-instance `draw_challenges` calls equals the recorded stream from the current
  inline draws (the lateral guarantee).
- The three `jolt-eval` invariants above, via `/new-invariant`.

### Performance

The verifier is off the hot path (O(log) per stage, once per proof). The
`ConcreteSumcheck` methods are shared with the performance-sensitive prover:
`input_claim`/`derive_opening_points`/`derive_{input,output}_term`/`draw_challenges`
must avoid per-call heap churn, and the symbolic object stays F-free so it stores no
per-proof field data. The `Challenges` struct is small (a handful of scalars/vectors
per instance). No regression on prover e2e micro-benchmarks; verify before/after.

`jolt-eval` framing: no `jolt-eval/src/objective/` entry is expected to move
materially. If per-instance `Challenges` allocation shows up in prover allocation
counts, add an objective via `/new-objective` and pool/borrow the struct.

## Design

### Architecture

**Three `Source` kinds (`Public` → `Derived`).** `Source<O, D, C>` is
`Opening(O) | Challenge(C) | Derived(D)`. The line that matters is `Opening` vs the
rest: an `Opening` is not cheaply verifier-evaluable, so it is committed and (in ZK)
hidden — it is the only kind BlindFold hides. `Challenge` and `Derived` are both
verifier-obtainable and both bake to BlindFold matrix coefficients, but they are
obtained differently — `Challenge` is read from the transcript, `Derived` is
computed — and that difference is exactly the two resolution paths below, so they
remain distinct kinds. "`Derived`" names the computed-by-the-verifier nature better
than "`Public`".

**`SymbolicSumcheck` (framework half of `jolt-claims`) — field-independent.**

```rust
pub trait SymbolicSumcheck {
    type RelationId;
    type OpeningId;
    type DerivedId;     // was PublicId
    type ChallengeId;
    type Shape;

    // Data shapes. Inputs/Outputs are generic over an opening *cell*
    // (OpeningClaim<F> = point+value | Vec<F> = point-only | F = value-only).
    // A challenge carries no point, so Challenges is parameterized by the field.
    type Inputs<C>;
    type Outputs<C>;
    type Challenges<F>;

    fn new(shape: Self::Shape) -> Self;
    fn id() -> Self::RelationId;          // type-level constant; NOT unique (phase/mode splits share one)
    fn spec(&self) -> SumcheckSpec;

    fn input_expression<F: RingCore>(&self)
        -> Expr<F, Self::OpeningId, Self::DerivedId, Self::ChallengeId>;
    fn output_expression<F: RingCore>(&self)
        -> Expr<F, Self::OpeningId, Self::DerivedId, Self::ChallengeId>;

    // provided: required_openings / required_deriveds / required_challenges <F>
}
```

Keeping `F` off the trait preserves the field-independence the predecessor spec
established: a symbolic object holds only its `Shape` and emits `Expr<F, …>` for any
`F`. The `Inputs<C>`/`Outputs<C>` GATs are pure type constructors; the
field-specific resolution behavior (`InputClaims<F>` etc.) is required where `F` is
known — at the concrete trait.

**`ConcreteSumcheck<F>` (`jolt-verifier`) — resolution and I/O.** The
`InputClaims`/`OutputClaims`/`SumcheckChallenges` bounds are use-site `where`
clauses (verbose fully-qualified projections elided here):

```rust
pub trait ConcreteSumcheck<F: Field>
where
    Sym<Self>::Inputs<OpeningClaim<F>>:  InputClaims<F, JoltOpeningId>,
    Sym<Self>::Outputs<OpeningClaim<F>>: OutputClaims<F, JoltOpeningId>,
    Sym<Self>::Challenges<F>:            SumcheckChallenges<F, JoltChallengeId>,
{
    type Symbolic: SymbolicSumcheck<
        RelationId = JoltRelationId, OpeningId = JoltOpeningId,
        DerivedId = JoltDerivedId,   ChallengeId = JoltChallengeId,
    >;

    fn symbolic(&self) -> &Self::Symbolic;
    fn id(&self) -> JoltRelationId { Self::Symbolic::id() }
    fn spec(&self) -> JoltSumcheckSpec { self.symbolic().spec() }

    /// Draw this instance's own (instance-private) challenges from the transcript.
    /// Batch-level coefficients and the shared binding vector are NOT drawn here —
    /// they belong to the batching layer (follow-up).
    fn draw_challenges<T: Transcript<Challenge = F>>(&self, transcript: &mut T)
        -> Sym<Self>::Challenges<F>;

    /// Map this instance's sumcheck point + upstream input points into the produced
    /// openings' points. Value-independent (runs clear + ZK); cross-input
    /// consistency for a well-defined point (e.g. address agreement) is checked here.
    fn derive_opening_points<C: GetPoint<F>>(&self, sumcheck_point: &[F], inputs: &Sym<Self>::Inputs<C>)
        -> Result<Sym<Self>::Outputs<Vec<F>>, VerifierError>;

    /// Resolve a Derived in the INPUT expression: from input points + challenges.
    /// No produced openings, no bound point (the input claim precedes binding).
    fn derive_input_term<C: GetPoint<F>>(&self, id: &JoltDerivedId,
        inputs: &Sym<Self>::Inputs<C>, challenges: &Sym<Self>::Challenges<F>)
        -> Result<F, VerifierError>;

    /// Resolve a Derived in the OUTPUT expression: from input points, the produced
    /// openings' points (the bound point, post-binding), and challenges.
    fn derive_output_term<C: GetPoint<F>>(&self, id: &JoltDerivedId,
        inputs: &Sym<Self>::Inputs<C>, outputs: &Sym<Self>::Outputs<OpeningClaim<F>>,
        challenges: &Sym<Self>::Challenges<F>) -> Result<F, VerifierError>;

    fn input_claim(&self, inputs: &Sym<Self>::Inputs<OpeningClaim<F>>,
        challenges: &Sym<Self>::Challenges<F>) -> Result<F, VerifierError> {
        self.symbolic().input_expression::<F>().try_evaluate(
            |id| inputs.resolve_input(id).ok_or(VerifierError::MissingOpeningClaim { id: *id }),
            |id| challenges.resolve_challenge(id).ok_or(VerifierError::MissingChallenge { id: *id }),
            |id| self.derive_input_term(id, inputs, challenges),
        )
    }

    fn expected_output<C: GetPoint<F>>(&self, inputs: &Sym<Self>::Inputs<C>,
        outputs: &Sym<Self>::Outputs<OpeningClaim<F>>, challenges: &Sym<Self>::Challenges<F>)
        -> Result<F, VerifierError> {
        self.symbolic().output_expression::<F>().try_evaluate(
            |id| outputs.resolve_output(id).ok_or(VerifierError::MissingOpeningClaim { id: *id }),
            |id| challenges.resolve_challenge(id).ok_or(VerifierError::MissingChallenge { id: *id }),
            |id| self.derive_output_term(id, inputs, outputs, challenges),
        )
    }
}
```

Note the cell pinning: `expected_output` reads output *values* (`resolve_output`),
so `outputs` is pinned to `Outputs<OpeningClaim<F>>`, not a point-only cell; its
`inputs` stay point-generic (the output claim needs input *points*, not values).
`input_claim` reads input *values*, so its `inputs` is `Inputs<OpeningClaim<F>>`.
`derive_*_term` need only points and challenges.

**Resolution split = the input/output asymmetry, in types.** The single
`resolve_public(id, inputs, outputs: Option<…>)` becomes two methods because the
availability of the bound point is the actual distinction: the input claim is the
claimed sum *before* binding, so input-side `Derived` cannot reference the bound
point; the output claim is checked *after* binding, so output-side `Derived` can
(reaching it through the produced openings' points in `outputs`). This is also why
all `eq`/`lt` Deriveds are output-side: they are evaluated at this sumcheck's bound
point, which exists only after the sumcheck runs.

**The data-model split between crates.** Each `*Claims` concern divides into a
transcript-free *order + id-resolution* half (→ `jolt-claims`) and a transcript-I/O
half (→ `jolt-verifier`):

| Concern | `jolt-claims` (pure) | `jolt-verifier` (I/O) |
|---|---|---|
| Produced openings | `opening_values()`, `opening_count()`, `resolve_output(id)` | `append_openings<T: Transcript>` (iterates `opening_values()`) |
| Consumed openings | `resolve_input(id)`, `from = …` wiring metadata | cross-stage population |
| Challenges | `resolve_challenge(id)` (`SumcheckChallenges` derive) | `draw_challenges<T: Transcript>` |
| Cells | `OpeningClaim`, `GetPoint`, `GetValue` | — |

`opening_values()` (field order) stays the one source of canonical opening order;
`append_openings` is a thin consumer that cannot disagree with it. The traits become
generic over the id types (`resolve_output(&O)`, not `&JoltOpeningId`) so they can
live in the framework crate — which also lets `field_inline` reuse them.

**`Challenges` and the challenge taxonomy.** Each relation gets a `Challenges`
struct; `#[derive(SumcheckChallenges)]` links each field to its `ChallengeId` and
generates `resolve_challenge`. The struct holds the instance's drawn Fiat-Shamir
material — both symbol-linked challenges (`gamma`, `eta`; appear in expressions,
served by `resolve_challenge`) and *unlinked* derivation inputs (`tau`,
booleanity/instruction reference points; appear in no expression but
`derive_*_term` reads them by field). Challenges fall into three tiers:

1. **Instance-private** (a relation's own `gamma`): drawn by its `draw_challenges`,
   live in its `Challenges`.
2. **Relation-scoped, shared across sibling phases** (Spartan `tau` across
   uniskip/remainder; `booleanity_gamma` across the booleanity phases): drawn by the
   **first sibling** and read by later siblings downstream — i.e. challenges acquire
   the same `from = upstream` wiring openings have. This is the one place aux data
   flows between instances, by deliberate choice.
3. **Batch-level** (the cross-instance RLC coefficient; the shared binding vector):
   owned by the batching layer (follow-up), not by any instance's `Challenges`.

This replaces the ad-hoc `StageNPublicOutput` carriers. Those structs today hold
only Fiat-Shamir material — challenge scalars, challenge *points* (`tau`, reference
points), and trivial derivations (gamma powers) — never the `eq`/`lt` values
(those are recomputed). In the new model the scalar tiers become instance
`Challenges` (1) or wired relation-scoped draws (2); the remaining non-opening
*points* a downstream stage needs are read off the producing instance's `Outputs`
cells where they are opening-derived, leaving a much smaller renamed
transcript-context carrier for the irreducible free points (`tau`, freshly-padded
references) plus whatever the batching layer owns.

**Uni-skip as two sumchecks.** The symbolic layer already models each uni-skip pair
as two relations sharing a `RelationId`, with the remainder consuming the uni-skip
opening as input (`OuterRemainder::input_expression = opening(outer_uniskip_opening())`,
`spartan.rs:130`; same for product). Making the *concrete/execution* side treat them
as two sequential instances removes the only mid-sumcheck draw: today the remainder
batching coefficient is drawn after the uni-skip binds and its output is absorbed,
before the remainder binds (`stage1/verify.rs:115-117`), which does not fit "drawn
before the sumcheck." As two sumchecks the sequence becomes the standard
`[uniskip bind] → absorb → [draw remainder pre-draws] → [remainder bind]`. This is
the same "one logical relation realized as a monolith *or* a phase-split" pattern
that `Booleanity` (`Booleanity` vs `BooleanityAddressPhase` + `BooleanityCyclePhase`)
and `BytecodeReadRaf` (monolith vs address + cycle(+committed)) already use, so the
trait/lifecycle must handle it generally regardless. Residual specialness, all
expressible: the uni-skip round's centered-integer domain rides in `spec()`; the
instance is a standalone (non-batched) singleton; its input claim is zero; and there
are ≥2 instances under one `RelationId` (tolerated now; cleaned up only if the
follow-up adopts instance = relation id).

**Relation ids are not load-bearing for the proof.** Verified: `JoltRelationId` is
not serialized into the proof and never absorbed into the transcript; it is not used
as a collection key. In-memory it is the disambiguator tag inside `JoltOpeningId`
(~149 construction sites), BlindFold constraint assembly (~94 refs), and error
context (~181, cosmetic). So restructuring ids — including a future "instance =
relation id" — is a lateral, in-memory change; it is deferred to the batching
follow-up where id-as-key pays off, and is a non-goal here.

### Alternatives Considered

- **Merge `Challenge` + `Public` into one "Aux" kind.** Considered (an "Aux" with
  `AuxInputs`/`AuxOutputs`). Rejected: although both bake identically in BlindFold,
  they have genuinely different resolution mechanisms (sampled from the transcript
  into a `Challenges` struct vs. computed on demand by `derive_*_term`) and different
  provenance. The clean axis for *resolution* is sampled-vs-computed, which is
  exactly the `Challenge`/`Derived` line; merging them would erase the distinction
  the two resolution paths depend on. Renaming `Public`→`Derived` captures the intent
  without collapsing the kinds.
- **Put `F` on `SymbolicSumcheck` (`SymbolicSumcheck<F>`).** Simpler bounds (no
  use-site `where` clauses), but it makes relations field-specific, contradicting the
  established field-independence. Rejected in favor of applying the `*Claims` bounds
  at the `ConcreteSumcheck<F>` use site; the cost is verbose, propagating
  `where`-clauses on generic code, localized to glue (concrete impls don't carry
  them).
- **Move `draw_challenges`/`append_openings`/cells into `jolt-claims` too.** Rejected:
  it would pull a `Transcript` dependency into `jolt-claims`. Only the transcript-free
  order + id-resolution half moves; transcript I/O stays in `jolt-verifier` as a
  consumer of `opening_values()`. (The `field_inline`-reuse benefit comes from making
  the traits generic over id types, which is independent of where I/O lives.)
- **Draw a stage's challenges centrally and distribute.** Rejected for per-instance
  `draw_challenges` (instance-private tier) plus a batching layer for batch-level
  draws. Per-instance keeps each relation's draws with the relation; the orchestrator
  (follow-up) only sequences the calls. (Sequencing — the cross-instance call order —
  remains load-bearing and is the batching layer's job.)
- **Draw relation-scoped `tau` in the orchestrator and hand it to both siblings.**
  Rejected for "first sibling draws, others read downstream," which keeps the draw
  inside an instance's `draw_challenges` and makes challenge-wiring uniform with
  opening-wiring. (Consequence: a deliberate uniskip→remainder challenge-flow edge.)
- **Do "instance = relation id" now.** Deferred (non-goal). It is lateral but only
  useful to the batching orchestrator; doing it here would inflate scope and the
  rename touches the ~149 opening-id sites + ~94 BlindFold refs.

## Documentation

- Update [`symbolic-sumcheck.md`](symbolic-sumcheck.md) and `jolt-verifier-model-crate.md`
  to describe the data-shape associated types, the `derive_input_term`/`derive_output_term`
  split, `Challenges`/`draw_challenges`, the `Public`→`Derived` rename, and the
  data-model relocation to `jolt-claims`.
- Update [`field-inline-protocol.md`](field-inline-protocol.md) for the `Derived`
  rename and the relocated/generic claim traits it can now reuse.
- Update `CLAUDE.md` where it references `Source::Public`, `resolve_public`, the
  `input_claim`/`input_claim_constraint` synchronization, and the BlindFold source
  kinds (now `Opening`/`Challenge`/`Derived`).
- No Jolt book (`book/`) changes — internal refactor; the public `verify` signature
  and the proof format are unchanged.

## Execution

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:writing-plans` to expand
> this outline into a task-by-task plan, then `superpowers:subagent-driven-development`
> to execute it. Each phase below is independently gated and committable; prefer a
> strangler migration so `muldiv` (both modes) stays green after every phase.

**Global constraints (every phase):** lint clean in BOTH modes
(`cargo clippy --all --features host,[zk] -q --all-targets -- -D warnings`); `muldiv`
in BOTH modes for any behavior-bearing change; `advice` e2e + fixtures in both modes
before deleting any bridge; `#[expect(...)]` not `#[allow(...)]`; `cargo nextest`,
never `cargo test`; heavy e2e/fixtures need the guest toolchain and sandbox OFF.

Phased outline:

0. **Rename `Public` → `Derived`** (`Source::Derived`, `derived(..)` builder,
   `DerivedId`, `required_deriveds`, `JoltDerivedId`, `*Derived` enums). Pure rename;
   lint + `muldiv` both modes.
1. **Relocate the data-model to `jolt-claims`, generic over id types.** Move the cells
   (`OpeningClaim`/`GetPoint`/`GetValue`) and the `InputClaims`/`OutputClaims` traits +
   derive macros into `jolt-claims`, parameterizing `resolve_*` over the id types;
   leave `append_openings` in `jolt-verifier` as a free fn / extension that iterates
   `opening_values()`. Gate: the append-matches-values test + `muldiv` both modes.
2. **Add the `Challenges` data shape.** Add `type Challenges<F>` to `SymbolicSumcheck`;
   add the `SumcheckChallenges` trait + derive (in `jolt-claims`); add `draw_challenges`
   to `ConcreteSumcheck` (in `jolt-verifier`). Pilot one relation: define its
   `Challenges`, move its inline draws into `draw_challenges`, assert the recorded
   draw stream is unchanged.
3. **Split `resolve_public` → `derive_input_term` / `derive_output_term`** and rewrite
   `input_claim`/`expected_output` accordingly (pinning `Outputs<OpeningClaim<F>>` on
   the output side). Mechanical per impl; gate both modes.
4. **Roll `Challenges` out across relations; retire `StageNPublicOutput`.** Migrate
   each stage's inline challenge draws into the instances' `draw_challenges`; thread
   relation-scoped draws (`tau`, booleanity gammas) via first-drawer wiring; read
   opening-derived points off `Outputs` cells; shrink the renamed transcript-context
   carrier to the irreducible residue. Gate per stage; the draw-order test is the
   guard.
5. **Model the uni-skip pairs as two sequential sumchecks** through the standard
   lifecycle (centered-integer domain via `spec()`; standalone/non-batched; zero
   input claim). Removes the interleaved-draw special case.
6. **(Optional) Split `jolt-claims` into framework + protocols crates** along the
   generic/Jolt-specific boundary the relocated traits enable.

Deferred to a follow-up spec: the batching abstraction (per-instance `draw_challenges`
sequencing, batch-level coefficients, the shared binding vector) and the
"instance = relation id" rename.

## References

- [symbolic-sumcheck.md](symbolic-sumcheck.md) — predecessor; the
  `SymbolicSumcheck`/`ConcreteSumcheck` split this spec builds on.
- [self-contained-sumcheck-relations.md](self-contained-sumcheck-relations.md) — the
  original per-relation `validate`/resolution pattern.
- [field-inline-protocol.md](field-inline-protocol.md) — the second protocol family
  that reuses the framework traits.
- `CLAUDE.md` — BlindFold section (source kinds, `input_claim`/`input_claim_constraint`
  synchronization, ZK feature gate).
