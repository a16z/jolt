# Spec: Self-Contained Sumcheck Relations and Generated Opening-Claim Plumbing

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Michael Zhu                    |
| Created     | 2026-06-18                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

The modular verifier (`crates/jolt-verifier`) is, structurally, a pipeline of
representation conversions: a proof's claims are decoded into typed per-stage
claim structs, resolved by id against an algebraic formula layer, combined into
expected output claims, and threaded forward to later stages. Today every one of
those boundaries is hand-written, and most of it is written **twice** — once
inline in the verifier's `verify()` functions and again in the prover-facing
`stage*_*` helper API that `jolt-prover` consumes. The result is a large surface
of bespoke claim structs and "convert struct A into struct B" boilerplate
(including ~211 hand-written `id == X` match arms and per-stage
`*_output_claim_values` / `append_*_opening_claims` flatteners), plus a recurring
class of bugs where the canonical opening **order** and **count** drift between
the two copies.

This spec reorganizes the verifier around **self-contained sumcheck relations**:
one object per `JoltRelationId` that owns its point derivation and public-value
computation (its algebra and sumcheck spec already live in `jolt-claims` as
`JoltRelationClaims`), and that is driven by *both* the prover and the verifier.
From each relation's claim structs — whose fields correspond to its
`JoltOpeningId = (relation, polynomial)` openings — two `derive` macros
(`InputClaims`, `OutputClaims`) generate the per-struct encode/resolve impls. The
primary win is not raw line count (the generated layer is roughly LoC-neutral on
its own); it is (a) collapsing the prover/verifier parallel re-derivation into one
shared definition, and (b) making opening order/count drift *structurally
impossible*.

## Intent

### Goal

Make each sumcheck relation a single source of truth for its point derivation,
public-value computation, and consistency checks — shared by `jolt-prover` and
`jolt-verifier` — and generate each relation's claim-struct encode/resolve impls
from the structs themselves.

Key abstractions introduced or modified:

- **`JoltRelationClaims` (reused, not duplicated).** `jolt-claims` already
  provides, per relation, a `JoltRelationClaims<F>` carrying the input/output
  algebraic relations (`Expr`), the sumcheck spec (rounds/degree/domain), and the
  relation id. This spec does **not** introduce a new type that re-bundles that.
- **`SumcheckInstance` (new trait, in `jolt-verifier`).** Adds *only* the
  operational behavior currently duplicated between `verify()` and the prover
  helpers — `derive_output_points` and `resolve_public` — on top of a
  `JoltRelationClaims`. Its consumed/produced claim structs are generic over an
  opening *cell* (`OpeningClaim<F>`, `Vec<F>`, or `F`): point-only methods run in
  both clear and ZK, value-reading methods only on the clear path. It lives in
  `jolt-verifier` and is shared with the prover (`jolt-prover` already depends on
  `jolt-verifier`).
- **`#[derive(InputClaims)]` and `#[derive(OutputClaims)]`** (new, conventional
  `syn`/`quote` proc-macros in `crates/jolt-verifier-derive`). `OutputClaims`
  generates a produced-claim struct's canonical encoders plus its output-formula
  resolver; `InputClaims` generates a consumed-claim struct's input-formula
  resolver. Two macros because the two sides have different responsibilities
  (produce + append vs. consume).

Cross-stage dataflow stays **explicit**: there is no runtime claim map in the
modular crates (see Non-Goals / Alternatives).

### Invariants

- **Prover/verifier consistency (primary).** For each relation, the point
  derivation, public-value computation, and expected-output algebra used by the
  prover and the verifier MUST be identical. This is achieved structurally — both
  sides call the same `SumcheckInstance` impl — rather than by hand-synced copies.
  This subsumes the existing CLAUDE.md invariant that a sumcheck's `input_claim()`
  and its BlindFold `input_claim_constraint()` stay in sync.
- **Canonical opening order is single-sourced.** A produced-claim struct's field
  declaration order is the *only* definition of canonical order. The generated
  `opening_values`, `append_openings`, and `opening_count` derive from it, so they
  cannot drift from each other (today they are three hand-written copies).
- **Fiat-Shamir order is preserved.** The generated `append_openings` MUST append
  opening claims to the transcript in exactly the order the prover commits them;
  changing declared order changes both sides together. No transcript operation may
  be added, removed, or reordered relative to current behavior.
- **Clear/ZK count agreement.** The ZK committed-output-claim count and the clear
  appended-claim count MUST both equal `opening_count()` (today they are computed
  independently — the `+ 1 + 1 + 2` style literals — and can disagree).
- **No soundness check is weakened.** Every consistency assertion currently inline
  in `verify()` is preserved: cross-input agreement needed for a well-defined
  opening point (e.g. the RAM address prefix) is checked in
  `SumcheckInstance::derive_output_points` so it runs in both modes, while
  value-equality consistency (e.g. `shift.unexpanded_pc == instruction_input
  .unexpanded_pc`) stays a clear-path check. Both still run before claims are
  consumed.
- **Resolution failure is a runtime rejection, not UB.** A generated resolver
  returns `None` for an unknown id, which callers turn into
  `Err(VerifierError::MissingOpeningClaim { .. })` — never a panic or silent
  default.

`jolt-eval` framing: keep `verifier_no_panic` green throughout. Add, via
`/new-invariant` during implementation, an invariant asserting a migrated
relation's `append_openings` order matches the prover's commitment order, and one
asserting `opening_count() == <relation>.output openings count` for every
relation.

### Non-Goals

- **Not** generating point derivation, public-value computation, or the algebraic
  relations. These are genuine per-relation mathematics; they are *encapsulated*
  and *de-duplicated* (written once in `SumcheckInstance`, shared), not generated.
  A declarative DSL for them is explicitly rejected (see Alternatives).
- **Not** generating the cross-stage wiring. Populating a relation's input struct
  from prior stages' typed outputs (including fallbacks) stays explicit,
  hand-written code in `jolt-verifier`.
- **Not** introducing a runtime `Map<JoltOpeningId, _>` claim store in the modular
  crates. (The `Verifier/ProverOpeningAccumulator` types live in `jolt-core`, not
  here, and are out of scope.)
- **Not** introducing `crabtime` or any build-time-code-execution macro; the
  generation is mechanical token emission (see Alternatives).
- **Not** changing the proof format, protocol, transcript bytes, or set of
  sumchecks. Proofs produced before and after MUST verify identically.
- **Not** a LoC-minimization exercise per se; the generated layer is roughly
  LoC-neutral. The LoC reduction is a side effect of removing the prover/verifier
  duplication.

## Evaluation

### Acceptance Criteria

- [ ] A `SumcheckInstance` trait exists in `jolt-verifier`, composing
      `JoltRelationClaims`, and at least one relation (pilot: stage 5
      `ram_ra_claim_reduction`) is fully migrated to it.
- [ ] For every migrated relation, `derive_output_points` and `resolve_public`
      are invoked by **both** a `jolt-prover` step and `jolt-verifier::verify`,
      with no remaining parallel copy of that logic.
- [ ] `#[derive(OutputClaims)]` generates `opening_values` / `opening_count` /
      `append_openings` / `resolve_output`, and `#[derive(InputClaims)]` generates
      `resolve_input`, for every migrated claim struct; the hand-written
      `*_output_claim_values` / `append_*` bodies and the inline `try_evaluate`
      opening closures delegate to them (or are removed).
- [ ] The ZK `committed_output_claims` count and the clear append count for each
      migrated relation both derive from `opening_count()`.
- [ ] The cross-stage wiring remains explicit and readable in `jolt-verifier`
      (reviewers can see where each input opening comes from).
- [ ] `cargo clippy --all --features host` and `--features host,zk` pass with
      `-D warnings`.
- [ ] `cargo nextest run -p jolt-verifier --features core-fixtures` and
      `--features core-fixtures,zk` pass (completeness + soundness, both modes).
- [ ] `cargo nextest run -p jolt-prover modular_muldiv_proof_verifies` passes in
      default and `zk` mode; `modular_sha2_chain_proof_verifies` passes.
- [ ] `cargo expand` on a migrated claim struct shows the generated impls.
- [ ] Net LoC for fully-migrated stages decreases (the prover/verifier
      duplication is removed); report the delta per stage.

### Testing Strategy

Must continue passing (per migrated relation, before moving to the next):

- `jolt-verifier` unit suite (no extra features).
- `jolt-verifier --features core-fixtures` and `core-fixtures,zk`: the full
  completeness + soundness/tampering suites. The `tampered_stage*_*_reject` and
  `tampered_zk_stage*_*` tests are the soundness guard — they confirm no check was
  weakened. (These require a guest toolchain; run with the command sandbox
  disabled — guest builds write `/tmp/jolt-guest-targets`.)
- `jolt-prover` `e2e.rs` (`modular_muldiv_proof_verifies`,
  `modular_sha2_chain_proof_verifies`) in both modes — the end-to-end
  prover→verifier check, which proves the *shared* `SumcheckInstance` behaves
  identically on both sides.

New tests:

- Per-relation unit tests for `derive_output_points` and `resolve_public`.
- A `derive` expansion/trybuild test covering scalar, `Vec`, `Option`, and
  nested-struct fields for both `InputClaims` and `OutputClaims`.
- `jolt-eval` invariants (append-order-matches-prover; count agreement) via
  `/new-invariant`.

### Performance

The verifier is not on a hot path (O(log) work per stage, once per proof), so the
generated encoders/resolvers are performance-neutral there. The
`SumcheckInstance` methods are **shared with the prover**, which *is*
performance-sensitive: `derive_output_points` / `resolve_public` may run in
hotter prover contexts, so they must avoid per-call heap churn. No regression on
`crates/jolt-prover/benches/e2e_micro`; verify before/after.

`jolt-eval` framing: no existing `jolt-eval/src/objective/` entries are expected
to move materially; if the shared-relation refactor changes prover allocation
counts, add an objective via `/new-objective` to track it.

## Design

### Architecture

A verifier stage is a batch of sumcheck instances; each instance corresponds 1:1
to a `JoltRelationId`. An opening claim is `(point, value)`, identified by
`JoltOpeningId { polynomial: JoltPolynomialId, relation: JoltRelationId }` (plus
`TrustedAdvice`/`UntrustedAdvice`) — i.e. the `(relation, polynomial)` pair, with
indexed families encoded in the polynomial variant (`LookupTableFlag(usize)`,
`InstructionRa(usize)`, …).

**Algebra + spec already exist (`JoltRelationClaims`).** `jolt-claims` already
constructs, per relation (e.g. `instruction::read_raf(dims)`), a
`JoltRelationClaims<F>` with `.input`/`.output` (the `Expr` relations), `.sumcheck`
(rounds/degree/domain), and `.id`. We reuse it as-is.

**Claim structs are generic over an opening *cell*.** An opening's point and value
travel together, but which half is known depends on the path, so each relation's
consumed/produced claim struct is generic over a cell `C` with three
instantiations:

```rust
pub struct OpeningClaim<F> { pub point: Vec<F>, pub value: F }   // clear: both
pub trait GetPoint<F> { fn point(&self) -> &[F]; }               // impl'd by Vec<F>, OpeningClaim<F>
pub trait GetValue<F> { fn value(&self) -> F; }                  // impl'd by F, OpeningClaim<F>
```

| cell | knows | used by |
|---|---|---|
| `F` | value | the serialized proof (wire form) |
| `Vec<F>` | point | the ZK path (values are committed, not cleartext) |
| `OpeningClaim<F>` | both | the clear path (assembled from the wire value + the derived point) |

Because the wire cell is the field element itself, the serialized struct is
byte-identical to a values-only struct — no `#[serde(skip)]`/`transparent` tricks.

**`SumcheckInstance` (new trait, in `jolt-verifier`, shared with the prover).** It
composes `JoltRelationClaims` and adds the operational logic duplicated between
`verify()` and the prover `stage*_*` helpers. Methods that need only points are
generic over any `GetPoint` cell (both modes); methods that read values pin the
`OpeningClaim<F>` cell (clear only). This makes "a ZK opening carries no value" a
compile-time fact: the value-reading methods are unreachable on the ZK path
(which can only build `Vec<F>` cells).

```rust
// crates/jolt-verifier — jolt-prover already depends on jolt-verifier, so it shares these
pub trait SumcheckInstance<F: Field>
where
    Self::Inputs<OpeningClaim<F>>: InputClaims<F>,
    Self::Outputs<OpeningClaim<F>>: OutputClaims<F>,
{
    type Inputs<C>;    // consumed-claim struct, cell-generic
    type Outputs<C>;   // produced-claim struct, cell-generic

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F>;   // algebra + spec (jolt-claims)

    // value-independent → runs in both clear and ZK; also checks any cross-input
    // consistency a well-defined output point needs (e.g. address agreement):
    fn derive_output_points<C: GetPoint<F>>(&self, sumcheck_point: &[F], inputs: &Self::Inputs<C>)
        -> Result<Self::Outputs<Vec<F>>, VerifierError>;

    // value-bearing → clear path. The two source kinds split on origin:
    // `Challenge` = a raw Fiat-Shamir transcript scalar (point-free, e.g. gamma/eta);
    // `Public` = a value the verifier *computes* from the statement/points (e.g. eq
    // evals, table MLEs, Lt) and may therefore be point-derived:
    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError>;
    fn resolve_public<C: GetPoint<F>>(&self, id: &JoltPublicId,
        inputs: &Self::Inputs<C>, outputs: &Self::Outputs<OpeningClaim<F>>) -> Result<F, VerifierError>;
    fn input_claim(&self, inputs: &Self::Inputs<OpeningClaim<F>>) -> Result<F, VerifierError> { /* input Expr */ }
    fn expected_output<C: GetPoint<F>>(&self,
        inputs: &Self::Inputs<C>, outputs: &Self::Outputs<OpeningClaim<F>>) -> Result<F, VerifierError> { /* output Expr */ }
}
```

There is no `Context`/`Points` associated type. A relation's public values are
functions of its input points ∩ its output points, so once both carry their points
(via the cells), `resolve_public`/`expected_output` are pure functions of
`(inputs, outputs)`. `expected_output` takes the inputs over *any* `GetPoint` cell
because the output expectation never needs input *values* — only their points —
which lets the prover pass point-only `Vec<F>` inputs.

Both sides drive the same impl: the verifier after running the sumcheck, the prover
while producing it. Locating it in `jolt-verifier` (with the prover depending on it)
mirrors the current arrangement — the prover already reuses `verify_until_stage1`
and the `stage*_*` helpers.

**No shared runtime claim store.** The `Verifier/ProverOpeningAccumulator` types
are `jolt-core`, not modular, and the modular dataflow deliberately avoids a
runtime `Map<JoltOpeningId, _>`. **Cross-stage wiring stays explicit**: a stage's
`verify()` populates a relation's input struct by hand from prior stages' typed
outputs (fallbacks inline). The derives below operate only on a single struct's
own fields; they do not generate wiring.

**Cell-aware `#[derive(OutputClaims)]`** — on a relation's *produced*-claim struct.
From the cell-generic field list it generates the canonical encoders
(single-sourced from declaration order) and the output-formula resolver, for every
cell `C: GetValue<F>` — so one impl serves the `F` wire form and the
`OpeningClaim<F>` clear form:

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[relation(RamRaClaimReduction)]
pub struct RamRaClaimReductionOutputOpeningClaims<C> {
    #[opening(RamRa)] pub ram_ra: C,               // Vec<C> ↔ indexed family; C ↔ scalar (arity from type)
}
// generated  impl<F: Field, C: GetValue<F>> OutputClaims<F> for ..<C>:
//   opening_values -> Vec<F>   opening_count -> usize
//   append_openings (FS order)   resolve_output(id) -> Option<F>
```

Nested aggregates recurse over their (cell-generic) sub-structs; the serialized
`Stage5Claims` instantiates the wire cell (`..OutputOpeningClaims<F>`).

**Cell-aware `#[derive(InputClaims)]`** — on a relation's *consumed*-claim struct,
generating the input-formula resolver; each field carries the upstream
`(polynomial, from = producing-relation)` it maps to:

```rust
#[derive(Clone, Debug, InputClaims)]
pub struct RamRaClaimReductionInputs<C> {
    #[opening(RamRa, from = RamRafEvaluation)]      pub raf: C,
    #[opening(RamRa, from = RamReadWriteChecking)]  pub read_write: C,
    #[opening(RamRa, from = RamValCheck)]           pub val_check: C,
}
// generated:  resolve_input(id) -> Option<F>
// hand-written cross-stage wiring (shared by verify() and the prover):
impl<F: Field> RamRaClaimReductionInputs<OpeningClaim<F>> {
    fn from_clear(stage2: &Stage2ClearOutput<F>, stage4: &Stage4ClearOutput<F>) -> Self { /* point + value per opening */ }
}
```

`verify()` (clear path) composes the explicit wiring with the shared relation:

```rust
let inputs = RamRaClaimReductionInputs::from_clear(stage2, stage4);     // explicit wiring
let input_claim = relation.input_claim(&inputs)?;                       // input Expr
// ... run the batched sumcheck, obtaining `ram_point` ...
let points  = relation.derive_output_points(ram_point, &inputs)?;       // Outputs<Vec<F>>
let outputs = RamRaClaimReductionOutputOpeningClaims {                  // wire value + derived point
    ram_ra: OpeningClaim { point: points.ram_ra, value: claims.ram_ra_claim_reduction.ram_ra },
};
let expected = relation.expected_output(&inputs, &outputs)?;            // output Expr
if reduction.value != expected { return Err(/* output mismatch */); }
outputs.append_openings(transcript);
```

The ZK path builds `Inputs<Vec<F>>` from the committed-path points and calls only
`derive_output_points` (no values, no `expected_output`).

**Generated vs hand-written:**

| Concern | Treatment |
|---|---|
| Cell-generic claim-struct field lists + `#[opening(..)]` | hand-written (the declaration) |
| `opening_values` / `append_openings` / `opening_count` / `resolve_output` | **generated**, cell-aware (`OutputClaims`) |
| `resolve_input` | **generated**, cell-aware (`InputClaims`) |
| Cross-stage wiring (`from_*` constructors) | **hand-written**, **shared** prover/verifier (no runtime map) |
| Point derivation + cross-input agreement (`derive_output_points`) | hand-written, **shared** prover/verifier |
| Public values (`resolve_public`) | hand-written, **shared** prover/verifier |
| Algebra + sumcheck spec (`JoltRelationClaims`) | **already shared** (jolt-claims) |

The hand-written parts are the irreducible mathematics; they live in one place and
are consumed by both prover and verifier, which is where the duplication is
removed. Compile-time safety comes from two places: the cell GAT makes a ZK
opening's absent value *unreachable* (only the `OpeningClaim<F>` cell exposes
`GetValue`, and the wire/ZK cells cannot construct the value-reading methods), and
claim structs are fully type-checked at construction. The only id-keyed/runtime
surface is each generated resolver's `match`, which returns `None`→`Err` for an
absent id. There is no global runtime claim map.

### Alternatives Considered

- **Status quo** (hand-written typed structs; verifier inline + parallel prover
  helpers). Baseline; high boilerplate and recurring order/count drift. The thing
  being fixed.
- **Runtime claim map / accumulator as the cross-stage store.** Considered, and
  initially in this spec. Rejected: the `Verifier/ProverOpeningAccumulator` types
  are `jolt-core`-only, and a modular runtime `Map<JoltOpeningId, _>` would trade
  compile-time wiring for runtime lookups with no clear way to keep the wiring
  explicit and checked. Cross-stage wiring stays explicit; only the per-struct
  resolvers are generated.
- **One combined `OpeningClaims` derive.** The encoder-only prototype
  (`OpeningClaimValues`) proved the mechanism (behavior-identical: clippy +
  muldiv completeness green) but was ≈flat net LoC alone. Split into
  `InputClaims` (resolver) and `OutputClaims` (encoders + resolver) because the
  consume and produce sides differ; the prototype becomes `OutputClaims`.
- **`crabtime` / compile-time-Rust codegen.** Supports IDE expansion + type hints
  (not opaque), but executes code at build time, its derive/attribute macros are
  WIP, and for mechanical token emission it offers nothing over conventional
  `syn`/`quote` (which is equally IDE-expandable and `cargo expand`-able) for
  *decorating* a struct. Rejected as the default tool — with one exception, the
  generate-from-id-list case immediately below.
- **Generate claim structs from an id list (function-like macro / `crabtime`).**
  The inverse of the chosen design: instead of decorating a hand-written struct
  with `#[opening(..)]`, feed a list of openings and synthesize the whole struct —
  snake_case field names from the polynomial names, `Vec` vs scalar from arity,
  plus the encoders/resolvers. This is the one scenario where `crabtime`'s
  real-Rust-at-build-time genuinely beats the alternatives: it needs actual
  generation logic (case conversion, conditional field types), which `macro_rules!`
  cannot do at all and which is more ergonomic in plain Rust than in `syn`/`quote`
  token-building. Not adopted, for three reasons. (1) `crabtime` cannot introspect
  the id enums, so the input is still `(id, arity, from-relation)`, not bare ids.
  (2) Field names do not mechanically follow from id names — `LookupTableFlag` →
  `lookup_table_flags` (pluralized) but `InstructionRa` → `instruction_ra` (not) —
  so you would need per-field name overrides (back to annotating) or accept
  regenerated names that churn call sites and change a human-facing API. (3) These
  claim structs are serialized into the proof (`Serialize`/`Deserialize` under
  `JoltProofClaims`); generating their field set and order from a macro input puts
  the clear-mode wire format in a macro spec rather than visible source — a
  higher-stakes thing to hide than the impls are. Decision: **decorate, don't
  generate** for the serialized claim structs (explicit fields keep the wire
  format and the field-name API in source; the derives generate only the impls).
  Reserve generate-from-list for internal, non-serialized helper structs if any
  arise, preferring a conventional `syn`/`quote` function-like macro; `crabtime`
  is defensible there specifically, not the default.
- **Full declarative relation DSL** (also generate points/publics). Rejected:
  points/publics are real algebra; a DSL would have to absorb all the per-stage /
  Clear-ZK / optional / committed-program irregularity, eroding auditability in
  soundness-critical code, and a single macro bug would be systemic.

## Documentation

- Update `specs/jolt-verifier-model-crate.md` ("Sumcheck Model", "Typed Opening
  Dataflow") to describe `SumcheckInstance` + the `Input/OutputClaims` derives,
  and that prover and verifier now share per-relation point/public logic.
- Update `specs/jolt-prover-model-crate.md` to reference the shared relation
  behavior (the prover stops carrying its own `stage*_*` re-derivation once a
  relation is migrated).
- Add a `.semgrep/jolt-verifier-boundaries.yml` rule once established: claim
  structs in `stages/**` must use the derives rather than hand-write
  `*_output_claim_values` / `append_*` / inline opening match arms.
- No Jolt book (`book/`) changes — internal refactor; the public `verify`
  signature and proof format are unchanged.

## Execution

Phased, incremental, gated after every relation. The public `stage*_*` helper API
stays stable during migration (bodies delegate to the relation objects) so
`jolt-prover` is never broken mid-flight.

- **Phase 0 — scaffolding.** Define `SumcheckInstance` in `jolt-verifier`.
  Build the `jolt-verifier-derive` crate: an `OutputClaims` derive (the encoder
  half — `opening_values` / `opening_count` / `append_openings` — was prototyped
  and gate-verified during design, then reverted, so re-create it and add
  `resolve_output`), plus an `InputClaims` derive (`resolve_input`) with
  `#[opening(..)]` id annotations.
- **Phase 1 — pilot.** Migrate stage 5 `ram_ra_claim_reduction` (small, but it
  exercises input-point splicing in `derive_output_points`). Define its
  `SumcheckInstance` impl; derive its claim structs; route both `verify()` and the
  prover step through it; keep the wiring explicit. Measure LoC; gate fully.
- **Phase 2 — roll out by stage**, in protocol order. Per relation: declare claim
  structs with the derives, move point/public logic into the `SumcheckInstance`
  impl, write the explicit wiring, delete the verifier-inline + prover-helper
  copies, gate (clippy ×2, completeness/soundness ×2, e2e ×2). Land each stage as
  its own reviewable, behavior-preserving commit.
- **Phase 3 — cleanup.** Remove now-dead parallel prover helpers and `Verify*`
  private mirror structs; add the semgrep rule; reconcile Clear/ZK count
  single-sourcing.

Implementation notes:
- Derives are conventional `syn`/`quote`; emit fully-qualified trait paths
  (`::jolt_transcript::Transcript`); respect `allow_attributes = "deny"` and
  `unused_results` in generated code (`let _ = map.insert(..)`).
- `#[derive(OutputClaims)]` needs `#[relation(..)]` (the owning relation) plus
  per-field polynomial annotations; `#[derive(InputClaims)]` needs per-field
  `(polynomial, from = relation)` annotations. Opening arity is read from the
  field type, not the annotation: `Vec<F>` fields map to indexed id families,
  `F`/`Option<F>` fields to single ids; nested structs recurse.

## References

- `specs/jolt-verifier-model-crate.md` — current verifier architecture.
- `specs/jolt-prover-model-crate.md` — prover side that shares the helper API.
- `crates/jolt-claims/src/protocols/jolt/ids.rs` — `JoltOpeningId`,
  `JoltPolynomialId`, `JoltVirtualPolynomial` (the opening key).
- `crates/jolt-claims` formula layer — `JoltRelationClaims` (`.input`/`.output`
  `Expr`, `.sumcheck`, `.id`) and `Expr::try_evaluate`.
- CLAUDE.md — "CRITICAL INVARIANT: Sumcheck claim/constraint synchronization" (the
  prover/verifier consistency property this design enforces structurally).
- Prior art (prototyped and gate-verified during design, not committed): an
  `OpeningClaimValues` encoder derive in a `jolt-verifier-derive` crate, to become
  `OutputClaims`; the stage 3/4/5/6/7 "consolidate verify() onto shared helpers"
  commits (`80a55a68f`, `adc53f8f7`, `841b7f71e`) that partially de-duplicated
  stage 6.
- Note: `Verifier/ProverOpeningAccumulator` live in `jolt-core` and are **not**
  used by the modular dataflow (cross-stage wiring is explicit).
