# Spec: Symbolic and Concrete Sumcheck Traits

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Michael Zhu                    |
| Created     | 2026-06-24                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

The [self-contained sumcheck relations](self-contained-sumcheck-relations.md) work
left one type, `JoltRelationClaims<F>` (`crates/jolt-claims`), carrying three
different concerns at once: the **symbolic algebra** of a relation (which
opening/public/challenge *ids* it references and the sum-of-products formulas
relating them), plus **cross-instance** concerns (consistency claims) and
**Fiat-Shamir** concerns (transcript-sync challenge bookkeeping). It is built by
~27 free "formula builder" functions and, in `field_inline`, mirrored by an
isomorphic `FieldInlineRelationClaims<F>` + 4 builders.

This spec splits that type along the concern boundary into two traits. **`SymbolicSumcheck`**
(in `jolt-claims`) is solely symbols and algebra: per-relation `input`/`output`
`Expr`s over id types, plus the sumcheck spec, with every runtime field value
expressed as a resolvable `Source` rather than a baked constant. **`ConcreteSumcheck`**
(in `jolt-verifier`, a rename of `SumcheckInstance`) owns the data-model concerns —
symbol *resolution*, cross-stage wiring, batching, and transcript order — and is
tied to its symbolic counterpart by an associated `type Symbolic: SymbolicSumcheck`
and to its derive-generated claim structs as today. `JoltRelationClaims`,
`JoltProtocolClaims`, and the `field_inline` equivalents are **removed**: the trait
*is* the representation. The symbolic layer is the natural input to the BlindFold
R1CS builder, the wrapping pipeline, and formal-verification efforts; the concrete
layer is what the Jolt prover and verifier drive.

## Intent

### Goal

Separate each sumcheck relation's pure symbolic algebra (a `SymbolicSumcheck` in
`jolt-claims`) from its concrete resolution/wiring/transcript concerns (a
`ConcreteSumcheck` in `jolt-verifier`), tying the two by an associated type and
deleting the `JoltRelationClaims`/`JoltProtocolClaims` structs that previously
fused them.

Key abstractions introduced or modified:

- **`SymbolicSumcheck` (new trait, `jolt-claims` crate root).** Generic over a
  relation's four id types (`RelationId`/`OpeningId`/`PublicId`/`ChallengeId`), a
  `Shape` (the per-relation construction input — replaces each builder's dimensions
  argument and subsumes the old `sumcheck: JoltSumcheckSpec` field). Provides
  `new(shape)`, `id()`, `spec()` (returning the shared crate-root `SumcheckSpec`), and
  field-method-generic `input_expression<F>()` / `output_expression<F>()` returning
  `Expr<F, OpeningId, PublicId, ChallengeId>`, plus provided `required_{openings,
  publics,challenges}<F>()` computed from those expressions. Protocol-agnostic:
  implemented for both `jolt` and `field_inline` relation types.
- **Per-relation symbolic types.** Each former free builder
  (`spartan::shift(dims)`, `ram::read_write_checking(dims)`, …) becomes a small
  F-free struct implementing `SymbolicSumcheck`, under a new `relations` module per
  protocol. ~27 jolt + 4 field_inline.
- **`ConcreteSumcheck` (rename of `SumcheckInstance`, `jolt-verifier`).** Gains
  `type Symbolic: SymbolicSumcheck<…>` and an accessor `symbolic(&self)`; holds its
  (F-independent) `Self::Symbolic` instead of a monomorphized
  `claims: JoltRelationClaims<F>`. `input_claim`/`expected_output`/`spec()`
  delegate to the symbolic object. Its `Inputs<C>`/`Outputs<C>` derive macros are
  unchanged.
- **Removed:** `JoltRelationClaims`, `JoltProtocolClaims`,
  `FieldInlineRelationClaims`, `FieldInlineProtocolClaims`, the `ClaimExpression`
  wrapper (+ `InputClaimExpression`/`OutputClaimExpression` aliases),
  `ConsistencyClaim`, `SameEvaluation`/`SameEvaluationAs`, and the
  `with_consistency`/`with_input_challenges`/`pull_challenges_for_transcript_sync`
  machinery.

### Invariants

- **Prover/verifier consistency (primary, inherited).** A relation's point
  derivation, public-value computation, and expected-output algebra MUST be
  identical on both sides. This is preserved: the algebra now lives in one
  `SymbolicSumcheck` impl consumed by the shared `ConcreteSumcheck`, replacing the
  `JoltRelationClaims` that played that role.
- **No proof-format, protocol, transcript, or sumcheck-set change.** Proofs
  produced before and after MUST verify identically. This is a representation
  refactor of verifier-internal types; `JoltRelationClaims` is never serialized.
- **Challenge set is single-sourced as the union of input/output expressions.**
  `required_challenges()` = `input_expr.required_challenges() ∪ output_expr.
  required_challenges()`. This reproduces today's per-relation challenge set
  exactly, because every challenge that `with_input_challenges` declared is already
  present in an expression (verified for all sites: bytecode read-raf
  `Gamma`+`Stage1..5Gamma`, booleanity `Gamma`, instruction-ra-virtualization
  `Gamma`, bytecode-claim-reduction `Eta`). Challenge *values* are resolved by id,
  so order is not load-bearing for resolution.
- **`val_check` R1CS is unchanged.** Modeling `val_check`'s `init_eval`
  decomposition as `Public` symbols instead of baked `Expr` coefficients MUST
  produce an identical verifier R1CS: BlindFold bakes `Source::Public`/`Challenge`
  factors as matrix coefficients exactly as it does literal `Term` coefficients,
  and the advice openings remain hidden `Source::Opening` witnesses. (Soundness
  guard: `muldiv` in ZK mode + the `advice` e2e tests.)
- **No consistency check is weakened.** Every cross-relation equality formerly
  carried as a `with_consistency` claim MUST be enforced by a live `validate`
  method on the relevant per-stage output-claim struct. The audit treats any
  uncovered claim as a bug to fix, never a silent drop.
- **Resolution failure is a runtime rejection, not UB.** A symbol with no resolver
  yields `Err(VerifierError::Missing…)`, never a panic or silent default — as today.

`jolt-eval` framing: keep `verifier_no_panic` and the existing
self-contained-relations invariants green throughout. Add, via `/new-invariant`
during implementation, an invariant asserting `ConcreteSumcheck::Symbolic`'s
`required_openings()`/`required_challenges()` equal the relation's derive-declared
opening set / drawn challenge set (catching symbolic↔concrete drift), and one
asserting the `val_check` Public-symbol input claim equals the prior
baked-coefficient input claim on a fixture.

### Non-Goals

- **Not** porting the special-cased Spartan uni-skip verification (stage 1
  `SpartanOuter`, stage 2 `SpartanProductVirtualization`) to `ConcreteSumcheck`.
  Their *symbolic* sides are ported (their builders return `JoltRelationClaims`
  today, so they must), but they stay Symbolic-only — no `ConcreteSumcheck` link —
  like `field_inline`.
- **Not** building a `field_inline` verifier / `ConcreteSumcheck` side.
  `field_inline` is claims-only today; it gets `SymbolicSumcheck` only.
- **Not** generalizing the `jolt-verifier-derive` macros to other protocols (they
  are hardwired to `::jolt_claims::protocols::jolt::*`; only the jolt
  `ConcreteSumcheck` side uses them).
- **Not** re-introducing a lowered/runtime `JoltRelationClaims` struct or a
  `Map<JoltOpeningId, _>` claim store. Dispatch stays static; the trait is the
  representation.

## Evaluation

### Acceptance Criteria

- [ ] `SymbolicSumcheck` exists at the `jolt-claims` crate root, generic over the
      four id types + `Shape`, with method-generic
      `input_expression<F>`/`output_expression<F>` and provided `required_*<F>`.
- [ ] All ~27 jolt and 4 field_inline former builders are per-relation types
      implementing `SymbolicSumcheck`; the free builder fns are removed.
- [ ] `SumcheckInstance` is renamed `ConcreteSumcheck` with `type Symbolic` +
      `symbolic()`; every impl holds `Self::Symbolic` (not `JoltRelationClaims<F>`);
      `input_claim`/`expected_output`/`spec()` delegate to the symbolic object.
- [ ] `resolve_public` takes `outputs: Option<&Self::Outputs<OpeningClaim<F>>>`;
      `input_claim` calls it with `None`, `expected_output` with `Some`.
- [ ] `val_check` is remodeled: `RamValCheckPublic::{InitEval, InitSelector}` added;
      its expression uses those `Public`s with an F-independent `Shape`; its
      `resolve_public` supplies them (clear + ZK); BlindFold supplies the values via
      `SourceValues` rather than building a `RamValCheckInit`.
- [ ] `with_input_challenges`/`pull_challenges_for_transcript_sync` are removed;
      `required_challenges()` is the input/output union.
- [ ] Every former `with_consistency` claim is enforced by a `validate` method;
      `ConsistencyClaim`/`SameEvaluation(As)`/`with_consistency` are removed.
- [ ] `JoltRelationClaims`, `JoltProtocolClaims`, `FieldInlineRelationClaims`,
      `FieldInlineProtocolClaims`, `ClaimExpression` (+ aliases) are removed.
- [ ] `cargo clippy --all --features host` and `--features host,zk` pass with
      `-D warnings`.
- [ ] `cargo nextest run -p jolt-prover-legacy muldiv --features host` and
      `--features host,zk` pass; the `advice` e2e tests pass.
- [ ] `cargo nextest run` for `jolt-verifier --features core-fixtures` and
      `core-fixtures,zk` pass (completeness + soundness/tampering, both modes).
- [ ] `cargo expand` on a migrated claim struct still shows the derive output
      (the derives are unchanged).

### Testing Strategy

Must continue passing:

- `muldiv` e2e (`jolt-prover-legacy`) in **both** `--features host` and
  `--features host,zk` — the primary correctness gate (per `CLAUDE.md`).
- The `advice` e2e tests — exercise the non-ZK advice path and the `val_check`
  init reconstruction; the soundness guard for the `val_check` remodel.
- `jolt-verifier --features core-fixtures` and `core-fixtures,zk`: completeness +
  soundness/tampering suites. (Guest toolchain required; run with the command
  sandbox disabled — guest builds write `/tmp/jolt-guest-targets`.)

New / moved tests:

- The order-locking formula tests (`required_challenges()`, `num_challenges()`,
  required-openings ordering) move onto the symbolic relation types; values are
  unchanged.
- A `validate` test for every cross-relation consistency claim (covering the audit).
- The `jolt-blindfold` `jolt_claims_pipeline` test switches to the symbolic API and
  asserts the `val_check` Public-symbol expression yields the same R1CS / input
  claim as the prior baked-coefficient form.
- `jolt-eval` invariants (symbolic↔concrete id-set agreement; `val_check` input
  claim equivalence) via `/new-invariant`.

### Performance

The verifier is off the hot path (O(log) work per stage, once per proof), and the
symbolic expressions are rebuilt per call rather than cached — negligible at
verifier scale. The `ConcreteSumcheck` methods are shared with the prover, which
*is* performance-sensitive: `input_claim`/`derive_opening_points`/`resolve_public`
must avoid per-call heap churn, and the symbolic object is F-independent so it
stores no per-proof field data. No regression on `jolt-prover` e2e micro
benchmarks; verify before/after.

`jolt-eval` framing: no `jolt-eval/src/objective/` entry is expected to move
materially. If rebuilding expressions per call shows up in prover allocation
counts, add an objective via `/new-objective` to track it (and cache the
`Expr` on the concrete instance if needed).

## Design

### Architecture

**`SymbolicSumcheck` (`jolt-claims`).** Pure symbols + algebra. `Shape` is the
per-relation construction input (usually a dimensions struct; F-independent for
every relation). Expression methods are generic over `F` at the method level, so a
symbolic object holds only its `Shape` and emits `Expr<F, …>` for any field on
demand — every runtime field value a relation needs is a `Source` (an `Opening`,
`Public`, or `Challenge` id resolved by the concrete side), never a literal
coefficient. (Associated-type *defaults* are unstable on the pinned toolchain — Rust
1.95, edition 2021 — so the original sketch's expression "aliases" are plain return
types; `type Shape;` GATs and generic methods are stable.)

```rust
pub trait SymbolicSumcheck {
    type RelationId;
    type OpeningId;
    type PublicId;
    type ChallengeId;
    type Shape;

    fn new(shape: Self::Shape) -> Self;
    fn id() -> Self::RelationId;             // type-level constant; NOT unique (phase/mode splits share one)
    fn spec(&self) -> SumcheckSpec;      // shared crate-root type, derived from Shape

    fn input_expression<F: RingCore>(&self)
        -> Expr<F, Self::OpeningId, Self::PublicId, Self::ChallengeId>;
    fn output_expression<F: RingCore>(&self)
        -> Expr<F, Self::OpeningId, Self::PublicId, Self::ChallengeId>;

    fn required_openings<F: RingCore>(&self) -> Vec<Self::OpeningId>
    where Self::OpeningId: Clone + Eq { /* input ∪ output */ }
    fn required_publics<F: RingCore>(&self) -> Vec<Self::PublicId>
    where Self::PublicId: Clone + Eq { /* input ∪ output */ }
    fn required_challenges<F: RingCore>(&self) -> Vec<Self::ChallengeId>
    where Self::ChallengeId: Clone + Eq { /* input ∪ output */ }
}
```

`id()` is a type-level constant but not a unique key: several sumchecks share one
`RelationId` (address/cycle-phase splits, the full/committed bytecode modes, and
the Spartan uni-skip/remainder pairs). Inputs carry `PublicId` (not `()`): both
`product_uniskip` (public Lagrange weights) and `val_check` (below) reference
publics in their input.

**`ConcreteSumcheck` (`jolt-verifier`, renamed from `SumcheckInstance`).** Adds the
resolution/operational layer on top of `Self::Symbolic`, holding the F-independent
symbolic object plus the relation's runtime challenges/derived points. `resolve_public`
takes `Option<outputs>` so one resolver serves both claims — `None` from
`input_claim` (the produced openings aren't derived yet) and `Some` from
`expected_output`.

```rust
pub trait ConcreteSumcheck<F: Field>
where
    Self::Inputs<OpeningClaim<F>>: InputClaims<F>,
    Self::Outputs<OpeningClaim<F>>: OutputClaims<F>,
{
    type Symbolic: SymbolicSumcheck<
        RelationId = JoltRelationId, OpeningId = JoltOpeningId,
        PublicId = JoltPublicId, ChallengeId = JoltChallengeId,
    >;
    type Inputs<C>;    // #[derive(InputClaims)]  — unchanged
    type Outputs<C>;   // #[derive(OutputClaims)] — unchanged

    fn symbolic(&self) -> &Self::Symbolic;
    fn id(&self) -> JoltRelationId { Self::Symbolic::id() }
    fn spec(&self) -> JoltSumcheckSpec { self.symbolic().spec() }

    fn derive_opening_points<C: GetPoint<F>>(&self, sumcheck_point: &[F], inputs: &Self::Inputs<C>)
        -> Result<Self::Outputs<Vec<F>>, VerifierError>;
    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> { /* default: err */ }
    fn resolve_public<C: GetPoint<F>>(&self, id: &JoltPublicId,
        inputs: &Self::Inputs<C>, outputs: Option<&Self::Outputs<OpeningClaim<F>>>)
        -> Result<F, VerifierError> { /* default: err */ }

    fn input_claim(&self, inputs: &Self::Inputs<OpeningClaim<F>>) -> Result<F, VerifierError> {
        self.symbolic().input_expression::<F>().try_evaluate(
            |id| inputs.resolve_input(id).ok_or(VerifierError::MissingOpeningClaim { id: *id }),
            |id| self.resolve_challenge(id),
            |id| self.resolve_public(id, inputs, None),
        )
    }
    fn expected_output<C: GetPoint<F>>(&self,
        inputs: &Self::Inputs<C>, outputs: &Self::Outputs<OpeningClaim<F>>) -> Result<F, VerifierError> {
        self.symbolic().output_expression::<F>().try_evaluate(
            |id| outputs.resolve_output(id).ok_or(VerifierError::MissingOpeningClaim { id: *id }),
            |id| self.resolve_challenge(id),
            |id| self.resolve_public(id, inputs, Some(outputs)),
        )
    }
}
```

**No lowered struct; static dispatch.** The trait surface is the full content of
the old `JoltRelationClaims`. There is no `dyn ConcreteSumcheck` anywhere — the
verifier drives each relation statically by name, reducing scalars into a
homogeneous `[SumcheckClaim; N]` per stage. The one place that folds a runtime-length
batch of relations, the BlindFold builder (`batched_input_expr`/`batched_output_expr`),
takes the per-stage instances' `input_expression::<F>()`/`output_expression::<F>()`/
`spec()` directly (gathered where the instances are already named) — no shared
element type.

**`val_check` init as `Public` symbols.** `ram::val_check` is the only builder that
today bakes runtime field values into its expression — the ZK `init_eval`
decomposition `constant(public_eval) - Σ constant(neg_selectorₖ)·opening(adviceₖ)`.
It is remodeled so those scalars become `Source::Public`:

```text
input = opening(ram_val) + challenge(gamma)·opening(ram_val_final)
        - (1 + challenge(gamma)) · ( public(InitEval)
                                     - Σ_k public(InitSelector_k)·opening(advice_k) )
```

This is value-preserving at the R1CS level: BlindFold maps `Source::Public`/`Challenge`
to baked matrix *coefficients* (provably, since `val_check`'s own output
`public(LtCyclePlusGamma)·opening·opening` only fits degree-2 R1CS if `Public` is a
baked constant), so `public(InitSelector)·opening(advice)` is the identical
`constant·witness` term as before, with the advice opening still a hidden witness.
`public·opening` is the universal term shape across all relations. The decomposition
*structure* (which contributions are present; absent in standard/full mode) is
carried by `val_check`'s F-independent `Shape`; the scalar *values*
(`reconstruct_full_eval`/`public_eval`, `advice_selector(r_address)`) move into its
`resolve_public` — centralizing the lockstep that today is split between BlindFold's
`ram_val_check_init` (ZK) and `stage4/verify.rs` (clear). BlindFold supplies the
values through `SourceValues` instead of building a `RamValCheckInit`.

**Consistency → `validate`.** Consistency claims are downstream of batching (two
claims are consistent iff their sumchecks share a batch — a concrete concern), so
they move fully to `jolt-verifier`, enforced on the typed per-stage output-claim
structs — the pattern already used by `Stage2BatchOutputClaims::validate` and
`Stage3OutputClaims::validate`, which compare values and report errors via id-pairs
from `*_consistency_openings()` helpers (those helpers stay in `jolt-claims` as pure
symbol data). The known claims (bytecode `unexpanded_pc` shift↔instruction;
instruction `left`/`right`/`lookup_output` reduced↔product) are already covered;
the migration audits every former `with_consistency` for a live `validate`.

**Where things live.** Symbolic per-relation types go in a new `relations` module
per protocol (`jolt_claims::protocols::jolt::relations`, mirrored for
`field_inline`), replacing `formulas/*` builder fns; the dimensions/layout types
they consume stay put and become the relations' `Shape`s. Concrete types keep their
names in `jolt-verifier`; the crate path disambiguates and `type Symbolic = …`
makes the pairing explicit.

### Alternatives Considered

- **Keep `JoltRelationClaims` as a lowered runtime form the trait produces (via
  `lower()`).** Rejected. It was motivated by an assumed `dyn`-safety constraint,
  but there is no dynamic dispatch — dispatch is static and `JoltRelationClaims` is
  never serialized, so nothing requires a lowered struct. The trait is the
  representation.
- **Put `F` on the trait (`SymbolicSumcheck<F>`) and keep `val_check`'s baked
  coefficients.** Simpler (only `val_check` carries `F`), but leaves runtime field
  values inside the symbolic layer and forces `RamValCheck<F>`. Rejected in favor of
  the `Public`-symbol remodel, which keeps every symbolic object F-free and the
  expression methods method-generic, and is provably R1CS-identical.
- **Type input expressions with `()` for publics.** The original sketch's
  invariant. Rejected: `product_uniskip` and (post-remodel) `val_check` reference
  publics in their inputs; `PublicId` keeps inputs/outputs symmetric.
- **Two resolvers (`resolve_input_public` + `resolve_public`).** Rejected for a
  single `resolve_public(id, inputs, outputs: Option<…>)` — one resolution surface,
  with `None` modeling "outputs not derived yet."
- **Relocate `with_input_challenges` to a `ConcreteSumcheck` challenge-order
  method.** Rejected: every challenge it declared is already in an expression, so
  the input/output union reproduces the set with no method and no overrides; it is
  deleted as dead code.
- **Unify `JoltSumcheckSpec`/`FieldInlineSumcheckSpec`.** Adopted: one crate-root
  `SumcheckSpec` (with a `domain`) replaces both; `field_inline`'s spec gains a
  `domain` (always `BooleanHypercube`), and the `Jolt*`/`FieldInline*` names become
  aliases. `SymbolicSumcheck::spec()` returns it directly, so the trait needs no
  associated spec type. (Reverses the earlier "keep it associated" decision.)

## Documentation

- Update [`self-contained-sumcheck-relations.md`](self-contained-sumcheck-relations.md)
  and `jolt-verifier-model-crate.md` to describe the `SymbolicSumcheck`/`ConcreteSumcheck`
  split (the former `JoltRelationClaims` role is now the `SymbolicSumcheck` trait;
  `SumcheckInstance` is `ConcreteSumcheck` with `type Symbolic`).
- Update [`field-inline-protocol.md`](field-inline-protocol.md) to note its
  relations now implement `SymbolicSumcheck` (Symbolic-only; no verifier side) and
  that `FieldInlineRelationClaims`/`FieldInlineProtocolClaims` are removed.
- Update `CLAUDE.md` where it references `JoltRelationClaims`/`SumcheckInstance` and
  the `input_claim`/`input_claim_constraint` synchronization (now expressed via the
  symbolic expressions).
- No Jolt book (`book/`) changes — internal refactor; the public `verify` signature
  and the proof format are unchanged.

## Execution

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Steps use `- [ ]` checkboxes for tracking.

Task-by-task implementation plan. **Strangler migration**: `JoltRelationClaims` and the free builders are kept as thin bridges during migration so `muldiv` (both modes) stays green after every task; they are deleted only in the final phases.

### Global Constraints

- Lint clean in BOTH modes, every commit: `cargo clippy --all --features host -q --all-targets -- -D warnings` and `cargo clippy --all --features host,zk -q --all-targets -- -D warnings`.
- Workspace lints: `allow_attributes = "deny"` (use `#[expect(...)]`, not `#[allow(...)]`); `unused_results` denied (`let _ = …`). Generated/edited code must comply.
- Primary correctness gate, every task that changes behavior-bearing code: `cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host` AND `--features host,zk`.
- No nightly features. Associated-type *defaults* are unstable — do NOT use them. GATs without defaults (`type Shape;`) and generic methods are fine.
- No change to proof format, transcript bytes, protocol, or the set of sumchecks. `JoltRelationClaims` is verifier-internal and never serialized — confirm nothing serializes it before deleting.
- Math-variable `non_snake_case` is allowed where the codebase already uses it (`log_T`, `ram_K`); match surrounding style.
- Use `cargo nextest`, never `cargo test`. Heavy e2e / `core-fixtures` need the guest toolchain and the command sandbox OFF (guest builds write `/tmp/jolt-guest-targets`).

### File Structure

**jolt-claims (symbolic side):**
- Create `crates/jolt-claims/src/symbolic.rs` — the `SymbolicSumcheck` trait + provided methods. Re-exported from `lib.rs`.
- Create `crates/jolt-claims/src/protocols/jolt/relations/` (`mod.rs` + one file per current `formulas/*` group: `spartan.rs`, `ram.rs`, `registers.rs`, `bytecode.rs`, `instruction.rs`, `booleanity.rs`, `claim_reductions/*`) — the 34 per-relation symbolic types. The relations' expression *bodies* are moved verbatim from the matching `formulas/*` builder.
- Modify `crates/jolt-claims/src/protocols/jolt/formulas/*` — builders become thin bridges (transition), then are deleted.
- Modify `crates/jolt-claims/src/protocols/jolt/ids.rs` — add `RamValCheckPublic::{InitEval, InitSelector}` (val_check remodel).
- Delete (final phase) `crates/jolt-claims/src/protocols/jolt/relation.rs`, `protocols/field_inline/relation.rs`, and the `ClaimExpression`/`ConsistencyClaim`/`SameEvaluation(As)` items in `claims.rs`.
- Mirror the relations module for `field_inline` (5 types).

**jolt-verifier (concrete side):**
- Modify `crates/jolt-verifier/src/stages/relations.rs` — rename `SumcheckInstance`→`ConcreteSumcheck`; add `type Symbolic` + `symbolic()`; `resolve_public` gains `Option<outputs>`; `sumcheck_relation()` becomes a transition-only provided method, then removed.
- Modify the 23 files holding the 29 impls (`stages/stage{2..7}/*.rs`) — store `Self::Symbolic`, impl `ConcreteSumcheck`.
- Modify `crates/jolt-verifier/src/stages/zk/blindfold/*` — consume symbolic expressions; supply `val_check` publics.
- Modify the per-stage `stages/stage*/outputs.rs` — `validate` methods (consistency audit).

---

### Phase 0 — `SymbolicSumcheck` trait

#### Task 0.1: Define the `SymbolicSumcheck` trait

**Files:**
- Create: `crates/jolt-claims/src/symbolic.rs`
- Modify: `crates/jolt-claims/src/lib.rs` (add `mod symbolic; pub use symbolic::SymbolicSumcheck;`)
- Test: in `crates/jolt-claims/src/symbolic.rs` `#[cfg(test)]`

**Interfaces:**
- Produces: `pub trait SymbolicSumcheck` with associated `RelationId/OpeningId/PublicId/ChallengeId/Shape`; `fn new(Shape)->Self`, `fn id()->RelationId`, `fn spec(&self)->SumcheckSpec` (the shared crate-root spec), `fn input_expression<F:RingCore>(&self)->Expr<F,OpeningId,PublicId,ChallengeId>`, `fn output_expression<F:RingCore>(&self)->…`, and provided `fn required_openings<F:RingCore>(&self)->Vec<OpeningId> where OpeningId: Clone+Eq` (+ `required_publics`, `required_challenges`).

- [ ] **Step 1: Write the failing test** (a tiny in-module dummy relation exercising the union + provided methods)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{challenge, opening, public, Expr};
    use jolt_field::Fr;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] enum O { A, B }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] enum P { X }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] enum Ch { G }

    struct Dummy;
    impl SymbolicSumcheck for Dummy {
        type RelationId = u8;
        type OpeningId = O;
        type PublicId = P;
        type ChallengeId = Ch;
        type Shape = ();
        fn new((): ()) -> Self { Self }
        fn id() -> u8 { 7 }
        fn spec(&self) -> SumcheckSpec { SumcheckSpec::boolean(3, 1) }
        fn input_expression<F: jolt_field::RingCore>(&self) -> Expr<F, O, P, Ch> {
            opening(O::A) + challenge(Ch::G) * opening(O::B)
        }
        fn output_expression<F: jolt_field::RingCore>(&self) -> Expr<F, O, P, Ch> {
            public(P::X) * opening(O::B)
        }
    }

    #[test]
    fn required_sets_are_input_then_output_deduped() {
        let d = Dummy;
        assert_eq!(d.required_openings::<Fr>(), vec![O::A, O::B]); // A from input, B from input
        assert_eq!(d.required_publics::<Fr>(), vec![P::X]);
        assert_eq!(d.required_challenges::<Fr>(), vec![Ch::G]);
        assert_eq!(Dummy::id(), 7);
    }
}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cargo nextest run -p jolt-claims symbolic --cargo-quiet`
Expected: FAIL — `SymbolicSumcheck` not found.

- [ ] **Step 3: Write the trait** (`crates/jolt-claims/src/symbolic.rs`)

```rust
use jolt_field::RingCore;

use crate::util::extend_unique;
use crate::{Expr, SumcheckSpec};

/// Pure symbolic description of one sumcheck relation: its id, sumcheck spec, and
/// input/output algebra over the relation's id types. Field-method-generic, so the
/// implementing object is field-independent (holds only `Shape`). See
/// `specs/symbolic-sumcheck.md`.
pub trait SymbolicSumcheck {
    type RelationId;
    type OpeningId;
    type PublicId;
    type ChallengeId;
    type Shape;

    fn new(shape: Self::Shape) -> Self;
    fn id() -> Self::RelationId;
    fn spec(&self) -> SumcheckSpec;

    fn input_expression<F: RingCore>(
        &self,
    ) -> Expr<F, Self::OpeningId, Self::PublicId, Self::ChallengeId>;
    fn output_expression<F: RingCore>(
        &self,
    ) -> Expr<F, Self::OpeningId, Self::PublicId, Self::ChallengeId>;

    fn required_openings<F: RingCore>(&self) -> Vec<Self::OpeningId>
    where
        Self::OpeningId: Clone + Eq,
    {
        let mut ids = self.input_expression::<F>().required_openings();
        extend_unique(&mut ids, &self.output_expression::<F>().required_openings());
        ids
    }

    fn required_publics<F: RingCore>(&self) -> Vec<Self::PublicId>
    where
        Self::PublicId: Clone + Eq,
    {
        let mut ids = self.input_expression::<F>().required_publics();
        extend_unique(&mut ids, &self.output_expression::<F>().required_publics());
        ids
    }

    fn required_challenges<F: RingCore>(&self) -> Vec<Self::ChallengeId>
    where
        Self::ChallengeId: Clone + Eq,
    {
        let mut ids = self.input_expression::<F>().required_challenges();
        extend_unique(&mut ids, &self.output_expression::<F>().required_challenges());
        ids
    }
}
```

Add to `crates/jolt-claims/src/lib.rs`: `mod symbolic;` and `pub use symbolic::SymbolicSumcheck;`. (`extend_unique` is already in `crate::util`; confirm it is `pub(crate)`.)

- [ ] **Step 4: Run the test, confirm it passes**

Run: `cargo nextest run -p jolt-claims symbolic --cargo-quiet`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
cargo clippy -p jolt-claims --features host -q --all-targets -- -D warnings
git add crates/jolt-claims/src/symbolic.rs crates/jolt-claims/src/lib.rs
git commit -m "feat(claims): add SymbolicSumcheck trait"
```

---

### Phase 1 — Rename `SumcheckInstance` → `ConcreteSumcheck` (mechanical)

#### Task 1.1: Pure rename

Pure identifier rename, no behavior change. Do it first so all later work uses the new name.

**Files:** `crates/jolt-verifier/src/stages/relations.rs` and every referencing file (the 23 impl files + stage `verify.rs`/`batch.rs` + `zk/blindfold/*`).

- [ ] **Step 1: Rename the trait and all references**

```bash
cd /Users/mzhu/code/jolt
grep -rl 'SumcheckInstance' crates/jolt-verifier/src --include='*.rs' \
  | xargs sed -i '' 's/\bSumcheckInstance\b/ConcreteSumcheck/g'
```

(Update the doc-comment prose in `relations.rs` too: "A single sumcheck instance…" stays accurate.)

- [ ] **Step 2: Build + lint both modes**

Run: `cargo clippy --all --features host -q --all-targets -- -D warnings` then `--features host,zk`.
Expected: clean (rename only).

- [ ] **Step 3: muldiv both modes**

Run: `cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host` and `--features host,zk`.
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add -u && git commit -m "refactor(verifier): rename SumcheckInstance to ConcreteSumcheck"
```

---

### Phase 2 — Pilot: one symbolic type + bridge (the template)

This establishes the per-relation recipe used by Phase 3. Pilot relation: **`spartan::shift`** (has input openings, a `Challenge` (gamma), `Public`s in the output, and a non-trivial output — exercises every `Source` kind).

#### Task 2.1: Create the `Shift` symbolic type

**Files:**
- Create: `crates/jolt-claims/src/protocols/jolt/relations/mod.rs` (`pub mod spartan;` …) and `relations/spartan.rs`
- Modify: `crates/jolt-claims/src/protocols/jolt/mod.rs` (`pub mod relations;`)
- Modify: `crates/jolt-claims/src/protocols/jolt/formulas/spartan.rs` (bridge `shift`)

**Interfaces:**
- Produces: `pub struct Shift { shape: TraceDimensions }` implementing `SymbolicSumcheck<RelationId=JoltRelationId, OpeningId=JoltOpeningId, PublicId=JoltPublicId, ChallengeId=JoltChallengeId, Shape=TraceDimensions>`.

- [ ] **Step 1: Add the symbolic type** (`relations/spartan.rs`)

Move the body of `formulas/spartan::shift` into `Shift::input_expression`/`output_expression`. The expression-building helpers (`shift_challenge`, `shift_public`, the `*_outer()`/`*_shift()` opening fns) stay in `formulas/spartan.rs` and are `pub(crate)`-imported, OR move with the type — keep them where they are for this task and `use super::super::formulas::spartan::{…}`.

```rust
use jolt_field::RingCore;
use crate::protocols::jolt::{JoltExpr, JoltOpeningId, JoltPublicId, JoltChallengeId, JoltRelationId};
use crate::protocols::jolt::formulas::dimensions::{JoltSumcheckSpec, TraceDimensions};
use crate::SymbolicSumcheck;

pub struct Shift { shape: TraceDimensions }

impl SymbolicSumcheck for Shift {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type PublicId = JoltPublicId;
    type ChallengeId = JoltChallengeId;
    type Shape = TraceDimensions;

    fn new(shape: TraceDimensions) -> Self { Self { shape } }
    fn id() -> JoltRelationId { JoltRelationId::SpartanShift }
    fn spec(&self) -> JoltSumcheckSpec { self.shape.sumcheck(/* SHIFT_DEGREE */ 2) }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = shift_challenge(SpartanShiftChallenge::Gamma);
        opening(next_unexpanded_pc_outer())
            + gamma.clone() * opening(next_pc_outer())
            + gamma.clone().pow(2) * opening(next_is_virtual_outer())
            + gamma.clone().pow(3) * opening(next_is_first_in_sequence_outer())
            + gamma.pow(4) * (JoltExpr::one() - opening(next_is_noop_product()))
    }
    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = shift_challenge(SpartanShiftChallenge::Gamma);
        shift_public(SpartanShiftPublic::EqPlusOneOuter)
            * (opening(unexpanded_pc_shift())
                + gamma.clone() * opening(pc_shift())
                + gamma.clone().pow(2) * opening(is_virtual_shift())
                + gamma.clone().pow(3) * opening(is_first_in_sequence_shift()))
            + shift_public(SpartanShiftPublic::EqPlusOneProduct)
                * gamma.pow(4)
                * (JoltExpr::one() - opening(is_noop_shift()))
    }
}
```

These bodies are moved verbatim from `formulas/spartan.rs::shift` (the current lines ~448-462), reusing its helper fns (`shift_challenge`, `shift_public`, `next_*_outer()`, `*_shift()`, `next_is_noop_product()`) via `use super::super::formulas::spartan::*;`. Make `SHIFT_DEGREE` `pub(crate)`. Each later relation (Phase 3) moves its builder's bodies the same way.

- [ ] **Step 2: Bridge the free builder** — `formulas/spartan.rs::shift` becomes:

```rust
pub fn shift<F: RingCore>(dimensions: TraceDimensions) -> JoltRelationClaims<F> {
    let relation = crate::protocols::jolt::relations::spartan::Shift::new(dimensions);
    JoltRelationClaims::new(
        <relations::spartan::Shift as SymbolicSumcheck>::id(),
        relation.spec(),
        relation.input_expression::<F>(),
        relation.output_expression::<F>(),
    )
}
```

(`shift` has no `with_consistency`/`with_input_challenges`, so the bridge is exact. Builders that DO call those keep the `.with_consistency([...])` / `.with_input_challenges([...])` suffix on the bridge — see Phase 3 recipe.)

- [ ] **Step 3: Port `shift`'s unit tests** onto `Shift` (move `formulas/spartan.rs`'s `shift_*` tests that assert `required_*`/`sumcheck` to assert against `Shift`; keep the evaluate-equivalence tests, now building the Expr via `Shift::new(d).input_expression::<Fr>()`).

- [ ] **Step 4: Lint + test + muldiv**

Run: `cargo nextest run -p jolt-claims spartan --cargo-quiet`, then clippy (host + host,zk), then `cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host` and `host,zk`.
Expected: PASS (the bridge keeps `shift()` behavior identical).

- [ ] **Step 5: Commit**

```bash
git add crates/jolt-claims/src/protocols/jolt/relations crates/jolt-claims/src/protocols/jolt/{mod.rs,formulas/spartan.rs}
git commit -m "refactor(claims): port spartan::shift to Shift symbolic type (bridged)"
```

---

### Phase 3 — Roll out the remaining symbolic types (recipe + table)

Apply the **Task 2.1 recipe** to every remaining jolt builder. Each is independently testable and committable; batch the commit per `formulas/*` module and gate that module.

**Recipe (per builder `formulas/<mod>::<fn>`):**
1. Add `relations::<mod>::<TypeName>` implementing `SymbolicSumcheck`, with `input_expression`/`output_expression` bodies moved verbatim from the builder, `id()` = the builder's `JoltRelationId`, `spec()` from the builder's `dimensions.<…>_sumcheck()`, `Shape` = the builder's argument type (a tuple/struct if it takes >1 arg, e.g. claim_reductions `cycle_phase(dims, chunk_count)` → `Shape = (PrecommittedReductionDimensions, usize)`).
2. Rewrite the builder as a bridge: construct the symbolic type and `JoltRelationClaims::new(id, sumcheck, input_expr, output_expr)`, **preserving any `.with_consistency([...])` and `.with_input_challenges([...])` suffix verbatim** (those are removed later, in Phases 7–8).
3. Move the builder's `required_*`/`sumcheck` unit tests onto the symbolic type.
4. Gate: `cargo nextest run -p jolt-claims <mod> --cargo-quiet`; clippy ×2; `muldiv` ×2.
5. Commit per module.

**`val_check` is special — SKIP it here; it is done in Phase 7.** For Phase 3, leave `formulas/ram::val_check` as-is (it still returns `JoltRelationClaims`); it just doesn't get a symbolic type yet.

**Table — the 33 builders to port here** (authoritative list: `grep -rl '\-> JoltRelationClaims<F>' crates/jolt-claims/src/protocols/jolt/formulas`):

| Module | Builders → symbolic types |
|---|---|
| `spartan.rs` | `outer_uniskip`, `outer_remainder`, `product_uniskip`, `product_remainder` (✅ `shift` done in Phase 2) |
| `ram.rs` | `read_write_checking`, `raf_evaluation`, `output_check`, `ra_claim_reduction`, `ra_virtualization`, `hamming_booleanity` (⏭ `val_check` → Phase 7) |
| `registers.rs` | `read_write_checking`, `val_evaluation` |
| `booleanity.rs` | `booleanity`, `booleanity_address_phase`, `booleanity_cycle_phase` |
| `bytecode.rs` | `read_raf`, `read_raf_address_phase`, `read_raf_cycle_phase` |
| `instruction.rs` | `input_virtualization`, `read_raf` |
| `claim_reductions/advice.rs` | (each builder returning `JoltRelationClaims`) |
| `claim_reductions/bytecode.rs` | `cycle_phase`, `address_phase` |
| `claim_reductions/hamming_weight.rs` | (each builder) |
| `claim_reductions/increments.rs` | `claim_reduction` |
| `claim_reductions/instruction.rs` | `claim_reduction` |
| `claim_reductions/program_image.rs` | `cycle_phase`, `address_phase` |
| `claim_reductions/registers.rs` | `claim_reduction` |

> The `outer_uniskip`/`outer_remainder`/`product_uniskip` symbolic types have no concrete side (Spartan stage-1 uni-skip is special — see spec Non-Goals). Port them anyway (their builders return `JoltRelationClaims`); they are Symbolic-only. `product_uniskip`'s input references `Public`s — fine, `PublicId` is the input public type.

#### Task 3.1 … 3.N

One task per module row above, each following the Recipe. Deliverable per task: that module's builders bridged through symbolic types, its unit tests moved, all gates green. (Reviewer can accept/reject each module independently.)

---

### Phase 4 — `ConcreteSumcheck` sources algebra from `symbolic()`

#### Task 4.1: Add `type Symbolic` + `symbolic()`; make `sumcheck_relation()` a transition provided-method

**Files:** `crates/jolt-verifier/src/stages/relations.rs`

**Interfaces:**
- Produces: `ConcreteSumcheck` gains `type Symbolic: SymbolicSumcheck<RelationId=JoltRelationId, OpeningId=JoltOpeningId, PublicId=JoltPublicId, ChallengeId=JoltChallengeId>;` and `fn symbolic(&self) -> &Self::Symbolic;`. `sumcheck_relation()` becomes provided (built from `symbolic()`), so existing callers keep working while we migrate them.

- [ ] **Step 1: Edit the trait** — add the associated type + `symbolic()`, and rewrite `sumcheck_relation` as provided + add `id`/`sumcheck` helpers + `resolve_public` Option param:

```rust
pub trait ConcreteSumcheck<F: Field>
where
    Self::Inputs<OpeningClaim<F>>: InputClaims<F>,
    Self::Outputs<OpeningClaim<F>>: OutputClaims<F>,
{
    type Symbolic: SymbolicSumcheck<
        RelationId = JoltRelationId, OpeningId = JoltOpeningId,
        PublicId = JoltPublicId, ChallengeId = JoltChallengeId,
    >;
    type Inputs<C>;
    type Outputs<C>;

    fn symbolic(&self) -> &Self::Symbolic;

    fn id(&self) -> JoltRelationId { Self::Symbolic::id() }
    fn spec(&self) -> JoltSumcheckSpec { self.symbolic().spec() }

    /// TRANSITION ONLY — built from `symbolic()` (empty consistency; consistency is
    /// enforced by `validate`, Phase 8). Removed in Phase 9 once all callers use the
    /// expression methods directly.
    fn sumcheck_relation(&self) -> JoltRelationClaims<F> {
        JoltRelationClaims::new(
            Self::Symbolic::id(),
            self.symbolic().spec(),
            self.symbolic().input_expression::<F>(),
            self.symbolic().output_expression::<F>(),
        )
    }

    fn derive_opening_points<C: GetPoint<F>>(/* unchanged */) -> Result<Self::Outputs<Vec<F>>, VerifierError>;

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        Err(VerifierError::MissingStageClaimChallenge { id: *id })
    }
    fn resolve_public<C: GetPoint<F>>(
        &self, id: &JoltPublicId, _inputs: &Self::Inputs<C>,
        _outputs: Option<&Self::Outputs<OpeningClaim<F>>>,
    ) -> Result<F, VerifierError> {
        Err(VerifierError::MissingStageClaimPublic { id: *id })
    }

    fn input_claim(&self, inputs: &Self::Inputs<OpeningClaim<F>>) -> Result<F, VerifierError> {
        self.symbolic().input_expression::<F>().try_evaluate(
            |id| inputs.resolve_input(id).ok_or(VerifierError::MissingOpeningClaim { id: *id }),
            |id| self.resolve_challenge(id),
            |id| self.resolve_public(id, inputs, None),
        )
    }
    fn expected_output<C: GetPoint<F>>(
        &self, inputs: &Self::Inputs<C>, outputs: &Self::Outputs<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        self.symbolic().output_expression::<F>().try_evaluate(
            |id| outputs.resolve_output(id).ok_or(VerifierError::MissingOpeningClaim { id: *id }),
            |id| self.resolve_challenge(id),
            |id| self.resolve_public(id, inputs, Some(outputs)),
        )
    }
}
```

Note `sumcheck_relation` now returns **owned** `JoltRelationClaims<F>` (was `&`). Callers `self.sumcheck_relation().input.expression()` still compile (operate on the temporary). `check_relation_boolean_hypercube(&JoltRelationClaims)` callers pass `&instance.sumcheck_relation()`.

Because `symbolic()` and the `resolve_public` signature changed, **every impl must update in the same commit** (Steps 2–3).

- [ ] **Step 2: Update every impl** (the 29) — add `type Symbolic = <the symbolic type>;` + `fn symbolic(&self) -> &Self::Symbolic { &self.symbolic }`, add a `symbolic: <Type>` field built in `new` (alongside the existing `claims` field, kept for now), drop the old `fn sumcheck_relation`, and change `resolve_public` signatures to `outputs: Option<&…>` (unwrap with `.ok_or(VerifierError::MissingStageClaimPublic { id: *id })?` where the body needs points). Map each impl file → symbolic type via the spec/Phase-3 names (e.g. `stage3/spartan_shift.rs` → `relations::spartan::Shift`).

- [ ] **Step 3: Fix call sites** — `check_relation_boolean_hypercube` takes `&JoltRelationClaims` from a temporary; `domain_spec(first.sumcheck)` etc. unchanged. Build/clippy until clean (both modes).

- [ ] **Step 4: muldiv both modes; commit**

```bash
git add -u && git commit -m "refactor(verifier): ConcreteSumcheck sources algebra from symbolic()"
```

#### Task 4.2: Migrate callers off `sumcheck_relation()` to the expression methods

**Files:** `stages/*/verify.rs`, `stages/stage6/batch.rs`, `stages/relations.rs` (`check_relation_boolean_hypercube`).

- [ ] **Step 1:** Replace `instance.sumcheck_relation().sumcheck` → `instance.spec()`; `…().input.expression()` → `…symbolic().input_expression::<F>()` (and output); `check_relation_boolean_hypercube` takes `(id, sumcheck_spec)` instead of `&JoltRelationClaims`. Leave BlindFold for Phase 6.
- [ ] **Step 2:** clippy ×2, muldiv ×2, commit.

---

### Phase 5 — Remove the temporary `claims` field from concrete instances

#### Task 5.1: Drop double-storage

Now that callers use `symbolic()`/`spec()`/expression methods, the per-instance `claims: JoltRelationClaims<F>` field and the bridge builder call in `new` are dead (except where BlindFold still consumes — see Phase 6; if BlindFold not yet migrated, defer this task until after Phase 6).

- [ ] **Step 1:** Remove the `claims` field + its construction from each of the 29 impls; `new` builds only the `symbolic` object (and the runtime challenge/point state).
- [ ] **Step 2:** clippy ×2, muldiv ×2, commit per stage.

---

### Phase 6 — Switch BlindFold to symbolic expressions

#### Task 6.1: `batched_input_expr`/`batched_output_expr` consume expressions, not `&[JoltRelationClaims]`

**Files:** `crates/jolt-verifier/src/stages/zk/blindfold/mod.rs`, `blindfold/stage{1..7}.rs`.

- [ ] **Step 1:** Change `add_batched_stage`/`batched_input_expr`/`batched_output_expr` to take a slice of `(rounds: usize, input: JoltExpr<F>, output: JoltExpr<F>)` (or a small local struct) instead of `&[JoltRelationClaims<F>]`. Per-stage builders gather these from the named instances via `instance.spec().rounds`, `instance.symbolic().input_expression::<F>()`, `…output_expression::<F>()`.
- [ ] **Step 2:** `domain_spec(first.sumcheck)` → use the gathered spec. Keep `map_jolt_expr`/`scale_expr` unchanged.
- [ ] **Step 3:** clippy ×2; `muldiv --features host,zk`; `core-fixtures,zk` (sandbox off). Commit.

---

### Phase 7 — `val_check` remodel (init → `Public` symbols)

The delicate step. Isolated, with its own gate. See spec §4.1.

#### Task 7.1: Add the public ids

**Files:** `crates/jolt-claims/src/protocols/jolt/ids.rs`

- [ ] **Step 1:** Extend `RamValCheckPublic`:

```rust
pub enum RamValCheckPublic {
    LtCyclePlusGamma,
    InitEval,
    InitSelector(JoltAdviceKind), // or an indexed family if program-image needs a distinct id
}
```

Confirm the contribution kinds: `program_image`, `Untrusted`, `Trusted` (from `ram_val_check_init`). If `JoltAdviceKind` (`Trusted`/`Untrusted`) doesn't cover `program_image`, add a dedicated `RamValCheckPublic::InitSelectorProgramImage` variant. clippy ×2; commit.

#### Task 7.2: `RamValCheck` symbolic type with the public-symbol expression

**Files:** `crates/jolt-claims/src/protocols/jolt/relations/ram.rs`, bridge `formulas/ram::val_check`.

**Interfaces:**
- Produces: `pub struct RamValCheck { shape: RamValCheckShape }` (F-free) where `RamValCheckShape = { dimensions: TraceDimensions, contributions: Vec<RamValContribution> }` and `RamValContribution` is the F-independent descriptor `{ selector: RamValCheckPublic, opening: JoltOpeningId }` (which advice/program-image openings are present).

- [ ] **Step 1:** Implement `input_expression` in public-symbol form:

```rust
fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
    let gamma = challenge(JoltChallengeId::from(RamValCheckChallenge::Gamma));
    let mut init = public(JoltPublicId::from(RamValCheckPublic::InitEval));
    for c in &self.shape.contributions {
        init = init - public(JoltPublicId::from(c.selector)) * opening(c.opening);
    }
    opening(ram_val()) + gamma.clone() * opening(ram_val_final())
        - (JoltExpr::one() + gamma) * init
}
```

`output_expression` is unchanged from today (`public(LtCyclePlusGamma) * opening(ram_inc) * opening(ram_ra)`). `spec()` = `val_check_sumcheck(dimensions)`. `id()` = `RamValCheck`.

- [ ] **Step 2:** Bridge `formulas/ram::val_check` — it currently takes `init: RamValCheckInit<F>`. Change its signature to build the F-independent `RamValCheckShape` from the *structure* of `init` (the present contributions), construct `RamValCheck`, and `JoltRelationClaims::new(…)`. **Caller impact:** `val_check`'s callers (`stage4/verify.rs`, `blindfold/mod.rs`) currently pass `RamValCheckInit<F>` (with values). They must now pass the structure + supply the values via `resolve_public` (Task 7.3) / `SourceValues` (Task 7.4). Do Tasks 7.2–7.4 in one commit so the tree compiles.

- [ ] **Step 3:** unit test: `RamValCheck::new(shape).input_expression::<Fr>()` evaluated with `InitEval`/`InitSelector`/advice resolvers equals the old `ram_val_init_eval`-based input for the same values (full mode AND a decomposed mode). Gate.

#### Task 7.3: `ram_val_check.rs` concrete `resolve_public`

**Files:** `crates/jolt-verifier/src/stages/stage4/ram_val_check.rs`, `stage4/verify.rs`.

- [ ] **Step 1:** `RamValCheck`'s concrete impl holds the `RamValCheck` symbolic object (structure) + the runtime data needed to resolve `InitEval`/`InitSelector` (`r_address`, `public_eval`, the advice selectors). Implement `resolve_public(id, _inputs, _outputs)`:
  - `InitEval` → `reconstruct_full_eval` (standard) / `public_eval` (zk) — exactly today's value.
  - `InitSelector(kind)` → `advice_selector(.., kind, &r_address)` — relocated from `blindfold::ram_val_check_init` / `stage4/verify.rs`.
  - `LtCyclePlusGamma` → as today (needs `outputs.ok_or(…)?`).
- [ ] **Step 2:** Confirm `r_address` is available at `input_claim` time (computed from stage-4 data upfront, not from this sumcheck's point). If not, store it on the instance from upstream output.

#### Task 7.4: BlindFold supplies `val_check` publics

**Files:** `crates/jolt-verifier/src/stages/zk/blindfold/mod.rs` (`ram_val_check_init` → publics), `blindfold/stage4.rs`.

- [ ] **Step 1:** Replace `ram_val_check_init` (which built `RamValCheckInit`) with code that pushes `(VerifierPublicId::Jolt(InitEval), value)` and `(InitSelector(kind), value)` into `SourceValues` for each present contribution — values identical to today's selectors/public_eval. The advice openings remain hidden `VerifierOpeningId` witnesses.
- [ ] **Step 2: Gate hard.** clippy ×2; `muldiv --features host` AND `host,zk`; the `advice` e2e tests; `core-fixtures{,zk}` (sandbox off). All must pass before commit. Commit as one self-contained "val_check remodel" change (Tasks 7.1–7.4).

---

### Phase 8 — Consistency → `validate` (audit)

#### Task 8.1: Audit + ensure every consistency claim has a live `validate`

**Files:** `crates/jolt-verifier/src/stages/stage2/outputs.rs`, `stage3/outputs.rs` (existing `validate`); add others if the audit finds gaps. Keep `*_consistency_openings()` helpers in `jolt-claims`.

- [ ] **Step 1: Enumerate** every former `with_consistency` claim:

```bash
grep -rn -A3 'with_consistency' crates/jolt-claims/src/protocols/jolt --include='*.rs' | grep same_evaluation_as
```

- [ ] **Step 2:** For each `a.same_evaluation_as(b)`, confirm a `validate` method compares the corresponding produced openings (`a == b`) and errors via the `*_consistency_openings()` id-pair helper. Known-covered: bytecode `unexpanded_pc` (stage3), instruction `left`/`right`/`lookup_output` (stage2). For any uncovered claim, **add** the `validate` check on the owning stage's output-claim struct (mirror `Stage3OutputClaims::validate`), and call it from that stage's `verify()` (clear path) — this is a bug fix, not optional.
- [ ] **Step 3:** Add/extend a `validate` unit test per consistency claim (a mismatched pair must `Err`).
- [ ] **Step 4:** clippy ×2; muldiv ×2; `core-fixtures{,zk}` (the `tampered_*` soundness tests are the guard). Commit.

---

### Phase 9 — Delete the old types and bridges

#### Task 9.1: Remove `with_input_challenges` (challenge union replaces it)

- [ ] **Step 1:** Delete every `.with_input_challenges([...])` suffix from the bridge builders. Confirm `required_challenges()` (the symbolic union) still yields the same set per relation (the Phase 0 invariant). Delete `ClaimExpression::pull_challenge(s)_for_transcript_sync`.
- [ ] **Step 2:** clippy ×2; muldiv ×2; the order-locking tests (now on symbolic types). Commit.

#### Task 9.2: Remove `with_consistency`/`ConsistencyClaim`/`SameEvaluation(As)`

- [ ] **Step 1:** Delete every `.with_consistency([...])` suffix (enforcement now lives in `validate`, Phase 8). Remove `ConsistencyClaim`, `SameEvaluation`, `SameEvaluationAs` from `claims.rs` and the `lib.rs` re-exports. Keep `*_consistency_openings()` helpers.
- [ ] **Step 2:** clippy ×2; muldiv ×2. Commit.

#### Task 9.3: Delete `JoltRelationClaims`, `JoltProtocolClaims`, `ClaimExpression`, the free builders, `sumcheck_relation()`

- [ ] **Step 1:** Confirm no remaining references: `grep -rn 'JoltRelationClaims\|JoltProtocolClaims\|ClaimExpression\|sumcheck_relation' crates --include='*.rs'` returns only the definitions to delete + their tests.
- [ ] **Step 2:** Delete `protocols/jolt/relation.rs` (and the `mod`/`pub use` in `protocols/jolt/mod.rs`); delete the now-bridge-only free builders in `formulas/*` (their logic lives in `relations/*`); delete `ClaimExpression`/`InputClaimExpression`/`OutputClaimExpression` from `claims.rs` and `lib.rs`; remove the transition `sumcheck_relation()` provided method from `ConcreteSumcheck` and the `JoltRelationClaims` import in `relations.rs`. Move any still-needed opening-helper fns (`outer_opening`, `*_consistency_openings`, …) from `formulas/*` to `relations/*` or a `relations/openings.rs`.
- [ ] **Step 3:** clippy ×2; `muldiv` ×2; `core-fixtures{,zk}`. Commit.

---

### Phase 10 — `field_inline` symbolic port

#### Task 10.1: Port the 5 field_inline builders; delete `FieldInlineRelationClaims`

**Files:** `crates/jolt-claims/src/protocols/field_inline/relations/*` (new), `formulas/*` (bridge then delete), delete `field_inline/relation.rs`.

- [ ] **Step 1:** Apply the Phase 3 recipe to the 5 builders (`product::field_product`, `registers::val_evaluation`, `claim_reductions/{increments,registers}::claim_reduction`, and any 5th from `grep -rl '\-> FieldInlineRelationClaims<F>' …`). They implement `SymbolicSumcheck` with the `FieldInline*` id types and `FieldInlineSumcheckSpec`. No concrete side exists, so no `ConcreteSumcheck` work.
- [ ] **Step 2:** Delete `FieldInlineRelationClaims`/`FieldInlineProtocolClaims` and `field_inline/relation.rs`; move helpers as in Task 9.3.
- [ ] **Step 3:** clippy ×2 (field_inline has tests but no e2e); `cargo nextest run -p jolt-claims field_inline --cargo-quiet`. Commit.

---

### Phase 11 — jolt-eval invariants (spec request)

#### Task 11.1: Add symbolic↔concrete drift + val_check equivalence invariants

The spec (Invariants) requests these via the repo's `new-invariant` skill. Add them once the migration is in place so they guard regressions.

- [ ] **Step 1:** Invoke `/new-invariant` to add an invariant asserting, for every relation, that `<ConcreteSumcheck>::Symbolic`'s `required_openings::<Fr>()` / `required_challenges::<Fr>()` equal the relation's derive-declared opening set / drawn challenge set (catches symbolic↔concrete drift).
- [ ] **Step 2:** Invoke `/new-invariant` to add an invariant asserting the `val_check` Public-symbol `input_claim` equals the prior baked-coefficient input claim on a fixture (guards the Phase 7 remodel).
- [ ] **Step 3:** Keep `verifier_no_panic` and the existing self-contained-relations invariants green. If prover allocation counts shift from rebuilding expressions per call, add a `/new-objective` to track it (and cache the `Expr` on the concrete instance only if a regression appears).
- [ ] **Step 4:** Run the `jolt-eval` suite per its README; commit.

### Final verification

- [ ] `cargo clippy --all --features host -q --all-targets -- -D warnings` and `--features host,zk` — clean.
- [ ] `cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host` and `--features host,zk` — pass.
- [ ] The `advice` e2e tests — pass (val_check remodel).
- [ ] `jolt-verifier --features core-fixtures` and `core-fixtures,zk` (sandbox off) — completeness + soundness/tampering pass.
- [ ] `grep -rn 'JoltRelationClaims\|JoltProtocolClaims\|with_consistency\|with_input_challenges\|ConsistencyClaim' crates --include='*.rs'` — no hits.
- [ ] Update `specs/symbolic-sumcheck.md` Status → implemented; update the docs listed in the spec's Documentation section.

### Notes / risks (from the spec)

- **val_check (Phase 7)** is the highest-risk step — isolate it and gate on `advice` + `muldiv,zk` before proceeding. The R1CS must be unchanged (baked `Public` ≡ baked literal).
- **Consistency audit (Phase 8)** must find a live `validate` for every former `with_consistency`; an uncovered one is a soundness bug to fix, not drop.
- **Spartan uni-skip** (`outer_uniskip`/`outer_remainder`/`product_uniskip`) are Symbolic-only — no concrete migration; their builders still must port (they return `JoltRelationClaims`).
- The `relations` module may need the `formulas/*` opening-helper fns; move them alongside the types rather than leaving a half-empty `formulas/`.

## References

- [`self-contained-sumcheck-relations.md`](self-contained-sumcheck-relations.md) —
  the predecessor that introduced `SumcheckInstance`, `JoltRelationClaims`, and the
  `Input/OutputClaims` derives this spec builds on.
- [`field-inline-protocol.md`](field-inline-protocol.md) — the isomorphic
  claims-only protocol that also implements `SymbolicSumcheck`.
- `crates/jolt-claims/src/protocols/jolt/ids.rs` — `JoltRelationId`,
  `JoltOpeningId`, `JoltPublicId`, `JoltChallengeId` (the trait's associated id
  types).
- `crates/jolt-claims/src/claims.rs` — `Expr`/`Source`/`Term` (the algebra core,
  reused) and `ClaimExpression` (removed).
- `crates/jolt-verifier/src/stages/relations.rs` — `SumcheckInstance` (→
  `ConcreteSumcheck`) and the `OpeningClaim`/`GetPoint`/`GetValue` cells.
- `crates/jolt-verifier/src/stages/zk/blindfold/mod.rs` — the BlindFold consumer
  (`batched_input_expr`, `ram_val_check_init`, `SourceValues`, `VerifierPublicId`).
- `crates/jolt-verifier-derive` — the `InputClaims`/`OutputClaims` derives (jolt-
  hardwired; unchanged).
- `CLAUDE.md` — "CRITICAL INVARIANT: Sumcheck claim/constraint synchronization"
  (the prover/verifier consistency this design preserves) and the `val_check`
  `init_eval` / `reconstruct_full_eval` decomposition.
