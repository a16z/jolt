# jolt-claims review triage

Source: moodlezoup review on PR #1545, `feat: add jolt-claims crate`.

This is a triage note before wiring `jolt-claims` into `jolt-verifier`. The
comments are grouped by whether they need a design call or can be handled in a
mechanical cleanup pass.

## Needs Your Input

These comments affect the protocol API, downstream lowering, or field-register
uniformity. Please mark the `Decision:` line for each item.

### Consistency claims must be consumed

Comment:
`instruction.rs` has input expressions that match core standard-mode
`input_claim()`, while core ZK constraints sometimes bind both opening chains.
The equivalent surface here is `JoltStageClaims::consistency`; downstream
BlindFold/R1CS lowering must not ignore it.

Question:
Should every lowering/verifier consumer be forced to handle consistency claims
as part of the stage contract, or should we fold these same-evaluation bindings
directly into stage input/output expressions where BlindFold needs them?

Recommendation:
Keep consistency claims, but make them impossible to silently drop:

- `jolt-verifier` stage wiring consumes them explicitly.
- BlindFold/R1CS lowering lowers them as equality constraints.
- wrapper lowering lowers them as equality constraints.
- tests assert every consistency claim appears in the lowered verifier R1CS.


I agree. make the conssitency claims hard to drop and expliict so that we get them in jolt-verifier. if we odnt have them we need to add, and we wanna harden this path to make that harder.

Field-register impact:
Apply the same rule to `field_inline`: field-register openings that share an
evaluation point should be represented as stage consistency claims and consumed
by BlindFold/wrapper lowering.

yes.

Decision:
Done: consistency claims stay explicit on relation claims, are included in
dependency aggregation, and the same representation is used by field-inline.

### Challenge vs public taxonomy

Comment:
`InstructionRaVirtualizationChallenge::EqCycle` and
`RamRaVirtualizationPublic::EqCycle` classify the same structural role
differently. This will confuse downstream code that bins ids by source kind.

Question:
Do we want the taxonomy to mean:


They should be Challenge i think, because it is Eq from a challenge type (which is a public computation just over a challenge type)

- `Challenge`: Fiat-Shamir sampled value, even if it is passed through a helper
  as a baked coefficient in BlindFold; and
- `Public`: deterministic verifier-side scalar, public IO/preprocessing value,
  or externally supplied public boundary value?

Recommendation:
Use `Challenge` for transcript-derived values and `Public` for deterministic
non-transcript values. That likely means moving RA-virtualization `EqCycle`
values to challenge IDs consistently.

Field-register impact:
Use the same taxonomy for `FieldInlineChallengeId` and `FieldInlinePublicId`.
For example, `EqCycle` and product-stage challenges should be challenges;
lagrange/equality coefficients computed from already-known points can remain
publics only if they are not themselves transcript samples in that stage.

Decision:
Done: transcript-derived Eq values are classified as challenges across the
affected Jolt and field-inline IDs.

### Forced challenge registration

Comment:
`ClaimExpression::require_challenge` is unclear.

Question:
Do we need an explicit API for transcript challenges that are consumed by a
stage but do not appear syntactically in the claim expression?

Recommendation:
If yes, rename this API to make its role explicit, e.g.
`reserve_challenge`, `with_transcript_challenges`, or
`with_round_challenges`. Keep it only on stage claims, not on raw expressions,
so it reads as stage metadata rather than expression mutation.


Maybe call it, pull_challenge_maintain_transcript_sync to be super explicit with what the api is needed for 

Field-register impact:
Use the same API for field-register stages. Avoid direct mutation of
`ClaimExpression` metadata.

Decision:
Done: the API is named `pull_challenge_for_transcript_sync` /
`pull_challenges_for_transcript_sync`, and relation helpers use it for
transcript-synchronizing challenges.

### SameEvaluation vs EqualExpressions

Comment:
`SameEvaluation` is a special case of `EqualExpressions`; remove it unless
there is a clear need.

Question:
Do we want a semantic `SameEvaluation` variant because it is useful for PCS
opening planning and diagnostics, or should all consistency checks be generic
expression equalities?

Recommendation:
Prefer one representation for lowering. If the semantic marker is useful, keep
a helper constructor but store it as `EqualExpressions(opening(a), opening(b))`
internally.

Keep one representation and make sure it is just explicit though so it's hard to miss

Field-register impact:
Use the same representation for field-register consistency checks so wrapper
and BlindFold lowering do not need protocol-specific branches.

Decision:
Done: `SameEvaluation` remains as a helper, but the stored representation is
the single `EqualExpressions(opening(left), opening(right))` form.

### Stage naming

Comment:
`JoltStageId` is confusing because historically "stage" refers to a batch of
sumchecks.

Question:
Should this be renamed before wiring to something like `JoltRelationId`,
`JoltCheckId`, or `JoltClaimId`?

Just do like JoleRelationId

Recommendation:
Rename now if we want to avoid confusion in `jolt-verifier`; this is easier
before field-inline and Dory-assist add more IDs. If we keep `StageId`, document
that in `jolt-claims` it means "claim-producing verifier component", not
necessarily a numbered prover stage.

Field-register impact:
Whatever name we choose should apply uniformly to `FieldInlineStageId`.

Decision:
Done: `JoltStageId`/`FieldInlineStageId` and stage-claim types were renamed to
`JoltRelationId`/`FieldInlineRelationId` and relation-claim types.

### `pow2` ownership

Comment:
Define `pow2` as a function on `RingCore`.

Question:
Are we willing to extend `jolt-field::RingCore` with a `pow2` helper now?

Recommendation:
If `pow2` is broadly useful in protocol formulas, move it to `RingCore` or a
small field utility trait. If not, keep it local and rename it narrowly. This
is cross-crate API surface, so it is less one-shot than the local cleanups.

Decision:

Yes
Done: `RingCore::pow2` owns the helper and jolt-claims now uses the field API.

## One-Shot Fixes

These are either already handled locally or can be fixed without a protocol
decision. When you are done marking up the section above, these can be applied
in a cleanup pass.

### Claim expression metadata safety

Comment:
`ClaimExpression` should not expose a mutable `expression` field while caching
metadata.

Status: Done. `expression` is private and exposed through `expression()`.

Follow-up:
Keep this invariant when adding field-inline formulas. Do not add APIs that
mutate `ClaimExpression.expression` without rebuilding metadata.

### Remove cached `num_challenges`

Comment:
Remove the `num_challenges` field and expose it as a method.

Action:
Delete `ClaimExpression::num_challenges`, compute from
`required_challenges.len()`, and update `JoltRelationClaims` plus
`FieldInlineRelationClaims` tests.

Status: Done.

### Resolver naming

Comment:
Rename evaluator closures to names like `resolve_opening`.

Action:
Rename `opening_value`, `challenge_value`, and `public_value` parameters in
`Expr::evaluate` / `try_evaluate` to `resolve_opening`, `resolve_challenge`,
and `resolve_public`.

Status: Done.

### Zero-coefficient terms

Comment:
Do not drop terms with zero coefficients if they carry metadata factors.

Status: Done. `From<Term>` only canonicalizes zero constants when
`factors.is_empty()`.

Follow-up:
Add or keep a regression test that zero-coefficient terms with openings still
surface `required_openings`.

### `log2_power_of_two`

Comment:
`debug_assert!` would be stripped in release for non-power-of-two inputs.

Status: Done with `assert!(value.is_power_of_two())`.

### Remove unnecessary `From` impls for dimension structs

Comment:
There are many unnecessary `From` impls; delete them.

Action:
Remove tuple/usize `From` impls for dimension structs and update tests to call
`::new(...)` directly. Apply the same rule to field-inline dimension types.

Status: Done.

### Error enums use `thiserror`

Comment:
Use `thiserror::Error` for error enums.

Action:
Add `thiserror.workspace = true` to `jolt-claims`, derive `Error`, and move
manual `Display` impls into `#[error(...)]` attributes.

Field-register impact:
Any field-inline error enum added later should follow the same style.

Status: Done.

### Move error enums to `error.rs`

Comment:
Pull error enums out of `dimensions.rs` into `error.rs`.

Action:
Create `crates/jolt-claims/src/protocols/jolt/formulas/error.rs`, move
`JoltFormulaPointError` and `JoltFormulaDimensionsError`, and re-export them
from `formulas` or `dimensions` as needed.

Status: Done.

### Move advice-specific dimension types

Comments:
Move `AdviceClaimReductionLayout` and `AdviceClaimReductionDimensions` from
`dimensions.rs` into `advice.rs`.

Action:
Move advice-only layout/dimension structs and helpers into
`formulas/claim_reductions/advice.rs` or a sibling advice dimensions module
owned by that formula.

Status: Done.

### Delete helper constructors on dimensions error

Comment:
Delete helper constructors such as `zero`, `overflow`, and `not_divisible`.

Action:
Inline the enum variants at call sites. With `thiserror`, these helpers add
more indirection than value.

Status: Done.

### Default generic types

Comments:
Default generic parameters for expression types should be `P = (), C = usize`.

Status: Done for `Source`, `Term`, `Expr`, and `ClaimExpression`.

Follow-up:
Preserve this convention in any field-inline-specific type aliases.

### Bytecode constants and XLEN

Comments:
Move the circuit flag constant to `jolt-riscv`; stop propagating const-generic
`XLEN` through `jolt-claims`.

Action:
Expose the canonical 64-bit circuit flag list from `jolt-riscv` and make
bytecode formulas use the ordinary Jolt XLEN. Avoid generic XLEN in
`jolt-claims` unless a formula truly needs the full-hypercube lookup-table test
case.

Status: Done.

### Arithmetic ops for expression refs

Comment:
Implement arithmetic ops on reference types to avoid clones.

Action:
Add `Add`, `Sub`, `Mul`, and `Neg` impls for borrowed `Expr` forms where they
remove clone noise in formulas. This is cleanup, not a protocol blocker.

Status: Done.

### Advice opening arrays

Comment:
Unused advice opening arrays should be removed; arrays are not clearly needed.

Action:
Delete unused helpers or switch to `Vec` only where variable length is real.
Keep fixed arrays only for public helpers that are used in tests or downstream
opening planners.

Status: Done. Advice opening helpers are exercised in tests; the variable
cycle-phase output stays as `Vec`.

### Committed opening order

Comments:
Keep proof/transcript commitment order distinct from final opening order, and
derive final opening IDs from the final polynomial order plus an explicit
polynomial-to-relation mapping.

Action:
Add a helper mapping `JoltCommittedPolynomial -> JoltRelationId` for final
openings, then have `final_opening_ids` map over
`final_opening_polynomial_order(...)`. `proof_commitment_order(...)` remains the
proof payload order (`RdInc`, `RamInc`, `InstructionRa`, `RamRa`, `BytecodeRa`);
it must not call the Stage 8 final-opening order helper. This removes order
drift risk without hiding the fact that the two orders intentionally differ.

Field-register impact:
When `FieldRdInc` enters the final opening/RLC path, use the same explicit
polynomial-to-relation mapping pattern instead of duplicating ordered vectors.

Status: Done for base Jolt; the same mapping pattern is called out for the
future field-register final-opening path.

### RA layout total

Comment:
Make `JoltRaPolynomialLayout::total` a method instead of a stored field.

Action:
Store only `instruction`, `bytecode`, and `ram`; compute checked nonzero total
in `new`, then compute `total()` from the fields. If preserving nonzero as an
invariant is useful, keep the current field and note why.

Status: Done.

### Derive `From`

Comments:
Use `derive_more::From` for ID wrapper `From` impls.

Action:
Add `derive_more.workspace = true` to `jolt-claims` and derive `From` on
wrapper enums where the manual impls are purely boilerplate.

Field-register impact:
Apply this to `FieldInlineChallengeId`, `FieldInlinePolynomialId`, and
`FieldInlinePublicId` too.

Status: Done.

### Unique relation IDs

Comment:
Add debug assertions that relation IDs are unique.

Action:
Add uniqueness checks in `JoltProtocolClaims::new` and
`FieldInlineProtocolClaims::new`, plus tests that duplicate IDs trip the
assertion in debug builds.

Status: Done.

### Deduplicate `extend_unique`

Comment:
`extend_unique` is duplicated in `claims.rs` and relation modules.

Action:
Move it to one crate-private helper and reuse it from Jolt and field-inline
relation modules.

Status: Done.

## Field-Register Uniformity Checklist

When addressing the review, apply the same cleanup to the new field-register
claim code instead of only fixing base Jolt:

- keep `FieldInlineRelationClaims` structurally aligned with `JoltRelationClaims`;
- remove cached challenge counts from both paths;
- use the same challenge/public taxonomy;
- use the same consistency-claim representation and lowering contract;
- use the same relation/claim ID naming choice;
- use the same final-opening mapping pattern for reduced `FieldRdInc`;
- derive or delete boilerplate `From` impls consistently;
- add relation-ID uniqueness checks for field-inline protocol claims too.

Status: Done.
