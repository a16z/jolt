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
  argument and subsumes the old `sumcheck: JoltSumcheckSpec` field), and an
  associated `SumcheckSpec`. Provides `new(shape)`, `id()`, `sumcheck()`, and
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
  `claims: JoltRelationClaims<F>`. `input_claim`/`expected_output`/`sumcheck()`
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
- **Not** unifying the two protocols' sumcheck-spec types. `JoltSumcheckSpec` has a
  `domain`; `FieldInlineSumcheckSpec` is `{rounds, degree}`. `SumcheckSpec` is an
  associated type.
- **Not** re-introducing a lowered/runtime `JoltRelationClaims` struct or a
  `Map<JoltOpeningId, _>` claim store. Dispatch stays static; the trait is the
  representation.

## Evaluation

### Acceptance Criteria

- [ ] `SymbolicSumcheck` exists at the `jolt-claims` crate root, generic over the
      four id types + `Shape` + `SumcheckSpec`, with method-generic
      `input_expression<F>`/`output_expression<F>` and provided `required_*<F>`.
- [ ] All ~27 jolt and 4 field_inline former builders are per-relation types
      implementing `SymbolicSumcheck`; the free builder fns are removed.
- [ ] `SumcheckInstance` is renamed `ConcreteSumcheck` with `type Symbolic` +
      `symbolic()`; every impl holds `Self::Symbolic` (not `JoltRelationClaims<F>`);
      `input_claim`/`expected_output`/`sumcheck()` delegate to the symbolic object.
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
    type SumcheckSpec;                       // jolt: {domain,rounds,degree}; field_inline: {rounds,degree}

    fn new(shape: Self::Shape) -> Self;
    fn id() -> Self::RelationId;             // type-level constant; NOT unique (phase/mode splits share one)
    fn sumcheck(&self) -> Self::SumcheckSpec;  // derived from Shape

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
        SumcheckSpec = JoltSumcheckSpec,
    >;
    type Inputs<C>;    // #[derive(InputClaims)]  — unchanged
    type Outputs<C>;   // #[derive(OutputClaims)] — unchanged

    fn symbolic(&self) -> &Self::Symbolic;
    fn id(&self) -> JoltRelationId { Self::Symbolic::id() }
    fn sumcheck(&self) -> JoltSumcheckSpec { self.symbolic().sumcheck() }

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
`sumcheck()` directly (gathered where the instances are already named) — no shared
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
- **Unify `JoltSumcheckSpec`/`FieldInlineSumcheckSpec`.** Rejected: they differ
  structurally (domain). `SumcheckSpec` is associated.

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

Large but mechanical (~31 relation types + ~50 reference sites). Staged to keep the
tree compiling and gated; land each stage as its own behavior-preserving commit.

1. Add `SymbolicSumcheck` + provided methods (no consumers yet).
2. Add per-relation symbolic types for jolt, one formula module at a time
   (spartan, ram, registers, bytecode, instruction, booleanity, ra, claim
   reductions). Builders may delegate to the new types via a short-lived adapter, or
   convert in lockstep with their consumers.
3. Rename `SumcheckInstance` → `ConcreteSumcheck`; add `type Symbolic` +
   `symbolic()`; give `resolve_public` its `Option<outputs>` parameter and route
   `input_claim` through it with `None`; switch
   `input_claim`/`expected_output`/`sumcheck()` to the symbolic object; drop the
   `claims: JoltRelationClaims<F>` field.
4. **`val_check` remodel** — the delicate step; do it isolated with its own gate.
   Add `RamValCheckPublic::{InitEval, InitSelector}`; rewrite `val_check`'s
   expression in `Public`-symbol form with an F-independent `Shape`; implement its
   `resolve_public` (clear + ZK) by relocating `advice_selector`/
   `reconstruct_full_eval`; switch BlindFold to supply those publics. Gate on
   `muldiv` (host,zk) **and** the `advice` e2e tests before proceeding. Confirm
   during the port that `r_address` is available at `input_claim` time (it is —
   computed from stage-4 data, not from this sumcheck's point).
5. Switch BlindFold's batched folds to the symbolic expressions.
6. Audit + finish the consistency `validate` migration; delete `with_consistency`,
   `ConsistencyClaim`, `SameEvaluation(As)`.
7. Delete `JoltRelationClaims`, `JoltProtocolClaims`, `ClaimExpression`/aliases,
   `with_input_challenges`/`pull_challenges_for_transcript_sync`, the free builders.
8. Port `field_inline` symbolic; delete `FieldInlineRelationClaims`/
   `FieldInlineProtocolClaims`.

Gate after each milestone:

```
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host
cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host,zk
```

plus the `advice` e2e tests and `--features core-fixtures{,zk}` for the verifier
(sandbox off — guest builds write `/tmp/jolt-guest-targets`).

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
