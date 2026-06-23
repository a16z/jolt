# Spec: Akita Logical Views And Translation

| Field | Value |
|-------|-------|
| Component | logical views and translation |
| Depends On | 00-roadmap.md, 03-prefix-packed-witness.md |
| Unlocks | one-hot increments, packing policy, verifier config |
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-06-08 |
| Status | implemented (transparent verifier/protocol target) |

## Scope

Jolt PIOP relations keep logical polynomial names. Akita opens one proof-owned
physical PackedWitness for packed facts. The integration needs a deterministic
translation from final logical claims either to packed-view claims over `W_pack`
or to separate precommitted openings.

In scope:

```text
- logical claim order vs physical opening targets
- PackedWitness layout
- direct one-hot views
- linear decoded views
- precommitted opening targets
- masked decoded views
- Stage 8 translation rules
- extra sumchecks required by masked views
```

Out of scope:

```text
- concrete increment encoding
- advice byte decomposition
- Akita proof internals
```

Assumptions:

```text
- Prior PIOP stages reduce all final logical trace-domain openings to a common
  row point r.
- Prefix-packed W_pack exists as specified by the PackedWitness component.
- Committed bytecode objects exist through the roadmap background interface.
- TrustedAdvice, BytecodeChunk(i), and ProgramImageInit are precommitted
  commitment objects.
- A precommitted object can only satisfy a final claim through an opening of its
  original commitment, unless a future protocol proves binding between that
  commitment and a packed copy.
- Logical opening IDs include relation IDs.
- Translation layout is public or transcript-bound before translation
  challenges.
```

## Architecture

Layouts:

```text
logical manifest:
  Jolt opening IDs and relation IDs.

PackedWitness layout:
  packed W_pack families and domains.

translation layout:
  map logical opening ID -> physical opening target.
```

Physical opening target:

```text
PackedWitnessView:
  view expression over W_pack.

PrecommittedOpening:
  original commitment handle plus opening relation for a precommitted object.
  This is outside the W_pack statement. A logical precommitted view may expand
  into multiple deterministic component openings when the original commitment is
  chunked or lane-oriented.
  The resulting statement/proof is separate from the PackedWitness packed-view
  proof and binds the original commitment handle.
```

Packed view expression:

```text
View {
  row_point: r
  terms: [(coefficient, fact_id, symbol_or_decoder)]
  kind: Direct | LinearDecoded | MaskedDecoded
}
```

Packed evaluation notation:

```text
W_f,l,s(row) = W_pack(rank(f, row, l, s))

At multilinear point r:
  W_f,l,s(r) = sum_x eq(r, x) * W_f,l,s(x)
```

Packed batch reduction:

```text
Given final logical claims:
  P_i(r) = y_i

Sample packed challenge after claims and layout are transcript-bound:
  rho

Rectangular special case:
  P_pack(r, rho) = sum_i eq(rho, i) * P_i(r)
  y_pack        = sum_i eq(rho, i) * y_i
  lambda_i      = eq(rho, i)
```

Prefix-packed mixed views generalize `eq(rho, i)` into selector/decode weights
derived from the translation layout. The returned `lambda_i` still multiplies
the raw logical claim `y_i`; dummy packed cells have no logical opening ID and
return no coefficient.

Direct one-hot view:

```text
LogicalFact_b(row) = W_pack[prefix, row, limb, b]

Claim at r:
  LogicalFact_b(r) is one packed view opening.

Translation:
  P(r) -> W_f,l,b(r)
```

Linear decoded view:

```text
ByteValue(row) = sum_{b=0}^{255} b * W_pack[prefix, row, limb, b]

Claim at r:
  ByteValue(r) = sum_b b * W_f,limb,b(r)

Translation:
  P(r) -> sum_b b * W_f,limb,b(r)
```

Multi-byte decoded view:

```text
Value(row) = sum_j 256^j * Byte_j(row)

Value(r) =
  sum_j 256^j * sum_b b * W_{byte_j,b}(r)
```

Sign-magnitude decoded view:

```text
Magnitude(row) = sum_j 256^j * Byte_j(row)
Sign(row)      = W_sign,1(row)

Signed(row) = (1 - 2 * Sign(row)) * Magnitude(row)
```

Key fact:

```text
Signed(r) != (1 - 2 * Sign(r)) * Magnitude(r)
```

Sign-magnitude decode is a masked view unless the protocol supplies a separate
linearized relation. Base increments use the masked translation in
`05-onehot-increments.md`.

Masked decoded view:

```text
RamInc(row) = StoreFlag(row) * FusedInc(row)
RdInc(row)  = RdPresent(row) * FusedInc(row)

RamInc(r) = sum_x eq(r, x) * StoreFlag(x) * FusedInc(x)
RdInc(r)  = sum_x eq(r, x) * RdPresent(x) * FusedInc(x)
```

Key fact:

```text
RamInc(r) != StoreFlag(r) * FusedInc(r)
RdInc(r)  != RdPresent(r) * FusedInc(r)
```

Therefore:

```text
direct view:
  no extra row sumcheck.

linear decoded view:
  no extra row sumcheck when row is fixed to r.

masked decoded view:
  requires a row-sum translation relation before final opening.
```

Masked translation relation:

This byte/sign masked translation was the earlier base-increment design. For
base Jolt increments it is superseded by `08-fused-increment-piop.md`, where
Stage 5i proves `IncVirtualization` and Stage 6 binds the `Store` selector
through committed bytecode.

```text
Input claim:
  claimed RamInc(r)

Prove:
  RamInc(r) = sum_x eq(r, x) * StoreFlag(x) * DecodeInc(x)

Output claims:
  proof-owned W_pack openings of packed increment facts at the translation
  sumcheck point rho.
  separate precommitted BytecodeChunk openings for StoreFlag/RdPresent at the
  translation sumcheck point rho.
```

This is not Stage 8-only. It is a PIOP relation because the row variable is
summed out. StoreFlag/RdPresent are not proof-owned W_pack facts unless a
future bound precommitted packed view proves equivalence to the original
BytecodeChunk commitment.

Chunked bytecode lowering:

```text
StoreFlag(rho):
  split the bytecode address point into chunk weights and a chunk-local point.
  open the Store circuit-flag lane of each needed BytecodeChunk(i).
  check the weighted sum equals the StoreFlag source claim.

RdPresent(rho):
  split the bytecode address point the same way.
  open the rd selector lanes of each needed BytecodeChunk(i), unless the
  committed-program interface exposes an explicitly committed rd-present view.
  check the weighted lane/chunk sum equals the RdPresent source claim.
```

These component openings are separate precommitted openings. They are not terms
inside the W_pack packed-view proof.

Validity relation classes:

```text
one-hot validity:
  for each fact family f, row x, and limb l:
    sum_s W_f,l,s(x) = 1
    W_f,l,s(x) * (W_f,l,s(x) - 1) = 0

optional one-hot validity:
  for selector families that may be absent on a row:
    sum_s W_f,l,s(x) is boolean
    W_f,l,s(x) * W_f,l,s'(x) = 0 for s != s'

boolean validity:
  B(x) * (B(x) - 1) = 0

range/decode consistency:
  decoded logical value uses the committed one-hot chunks.
```

Validity placement:

```text
Direct RA facts:
  already covered by existing Jolt RA/booleanity/hamming logic or its lattice
  equivalent.

New increment/advice one-hot facts:
  require explicit validity relations in the increment and packing
  components.

Committed bytecode views:
  use committed-program reductions plus separate openings of BytecodeChunk(i) or
  ProgramImageInit against their original commitments.

Fused increment source views:
  StoreFlag and RdPresent are committed-bytecode views. They are consumed by the
  masked translation relation through separate BytecodeChunk(i) openings, or by
  a future bound precommitted packed view. They are not proof-owned W_pack
  families. RdPresent is a linear view over rd selector lanes unless the
  committed-program interface adds a dedicated committed rd-present object.
```

Stage placement:

```text
Stage 6/7:
  validity of packed facts.
  translation sumchecks for masked views.

Stage 8:
  deterministic direct/linear translation of final logical claims.
  final Akita packed-view opening over W_pack for proof-owned packed claims.
  separate precommitted openings for TrustedAdvice, BytecodeChunk(i), and
  ProgramImageInit.
  separate direct/native openings for precommitted linear views. For current
  base increments, `08-fused-increment-piop.md` binds the Store selector in
  Stage 6 and opens unsigned increment chunk/MSB facts in Stage 8.
```

Stage 8 statement construction:

```text
for each logical opening:
  resolve physical opening target
  add packed physical terms to the W_pack BatchOpeningStatement, or add a
  precommitted opening requirement
  record coefficient mapping logical claim -> physical statement

for precommitted linear/component views:
  add one opening requirement per original committed component
  verify the deterministic recombination equals the logical source claim
  then use that logical claim in the surrounding PIOP relation
  reject if the prover supplies only a W_pack packed-view claim or a backend
  program-commit handle without the original precommitted opening

for direct/linear views:
  no new sumcheck instance

for masked views:
  require prior translation relation output claims
```

Transcript:

```text
1. logical final-opening manifest.
2. PackedWitness layout digest.
3. precommitted commitment handles and opening manifests.
4. logical claims.
5. translation challenges, if any.
6. Akita batch-opening challenges.
```

Proof shape:

```text
PackedWitness opening:
  one batch proof over W_pack for proof-owned packed claims.

Precommitted openings:
  zero or more direct/native proofs, one per precommitted statement after
  deterministic component expansion.
  proof order follows the precommitted opening manifest.
  the verifier rejects missing, extra, reordered, or W_pack-backed proofs.
```

Implementation plan:

```text
jolt-claims:
  define base logical final-opening IDs and relation IDs.
  define lattice-extension view formulas for packed representations.
  define formulas for translation relations that are PIOP-level, such as
  masked views.

jolt-verifier:
  build LogicalOpeningManifest from stage outputs and protocol config.
  enable lattice view resolution only when the selected PCS family is lattice.
  resolve each logical opening through ViewResolver into either PackedWitnessView
  or PrecommittedOpening.
  reject unresolved, unsupported, or misclassified views.
  construct packed BatchOpeningStatement claims and separate precommitted opening
  statements.

jolt-akita:
  accepts PhysicalView formulas.
  verifies direct/linear packed view statements.
  rejects masked views unless they have already been reduced to supported
  physical views.
```

Proposed Rust data model:

```rust
pub struct LogicalOpeningManifest {
    pub claims: Vec<LogicalOpeningClaim>,
    pub digest: [u8; 32],
}

pub struct LogicalOpeningClaim {
    pub id: JoltOpeningId,
    pub relation: JoltRelationId,
    pub point: Vec<F>,
    pub claim_source: ClaimSource,
}

pub enum PhysicalView {
    Direct {
        fact: FactId,
        limb: usize,
        symbol: usize,
    },
    LinearDecode {
        terms: Vec<DecodeTerm>,
    },
    ReducedMasked {
        relation: JoltRelationId,
        output_openings: Vec<JoltOpeningId>,
    },
    Precommitted {
        commitment: CommitmentRef,
        relation: JoltRelationId,
    },
}

pub struct DecodeTerm {
    pub coefficient: F,
    pub fact: FactId,
    pub limb: usize,
    pub symbol: usize,
}
```

View resolver:

```text
fn resolve(id, relation, config, layouts) -> PhysicalView:
  if id is RA fact:
    return Direct(fact, limb=0, symbol)

  if id is fused increment:
    require prior masked translation output
    return ReducedMasked(...)

  if id is BytecodeChunk/ProgramImageInit:
    return Precommitted(original committed-program commitment, relation)

  if id is TrustedAdvice:
    return Precommitted(original trusted-advice commitment, relation)

  if id is UntrustedAdvice:
    return LinearDecode(proof-owned advice PackedWitness facts)
```

Claim coefficient construction:

```text
For direct view:
  logical coefficient maps to one physical term.

For linear decode:
  logical coefficient alpha expands to physical coefficients
    alpha * decode_coeff_j.

For masked view:
  logical coefficient applies to the translation relation output claim, not to
  a pointwise product at r.
```

Software decisions:

```text
owner of decode coefficients:
  jolt-claims should own semantic decode formulas.
  jolt-akita should own physical proof interpretation.

layout fields:
  verifier should derive view formulas from config when possible.
  prover-provided view formulas are allowed only if digest-checked.

unsupported views:
  fail during Stage 8 statement construction.
```

## Invariants

```text
- Each logical final opening resolves to exactly one translation expression.
- Translation expressions are deterministic from verifier config and layout.
- Direct and linear decoded translations do not add hidden row sums.
- Masked decoded translations are proven by explicit PIOP relations.
- Translation coefficients are bound before Akita batch challenges.
- Committed bytecode logical openings resolve through precommitted-program
  openings when lattice mode supports ProgramMode::Committed.
- TrustedAdvice resolves through its precommitted advice commitment; it cannot be
  satisfied only by a W_pack view.
- BytecodeChunk(i), ProgramImageInit, and trusted-advice claims cannot appear in
  the W_pack packed-view statement in the target protocol.
- A direct/linear view may only use facts at the same row point as the logical
  claim.
- A decoded view cannot assume one-hot validity; validity is a separate
  invariant proved by a PIOP relation.
- Optional one-hot families must be marked explicitly; they cannot use the
  exactly-one validity rule.
- Masked views must not be lowered to pointwise multiplication at r.
- Translation layouts are independent of prover witness values.
- Base increment source views use committed bytecode lanes:
    Ram source = Store flag.
    Rd source = sum of rd one-hot lanes.
  These source views resolve through precommitted BytecodeChunk openings unless
  a future bound precommitted packed view is specified.
- Any component openings used to reconstruct a precommitted source claim must be
  checked against that source claim before the claim is used by the fused
  increment translation.
- Backend `Program::Committed` material is not a substitute for the original
  Jolt precommitted commitment handles.
```

## Tests

Targeted tests:

```text
direct_view_translation_matches_packed_eval:
  one-hot fact opening equals packed view opening.

linear_decode_translation_matches_direct_sum:
  decoded byte/limb claim equals weighted one-hot openings.

masked_view_requires_translation_sumcheck:
  verifier rejects masked view without required relation.

translation_layout_digest_mismatch_rejects:
  proof fails under changed logical-to-physical mapping.

decoded_view_without_validity_rejects_or_is_not_enabled:
  config cannot enable decoded facts without corresponding validity relation.

masked_translation_tamper_rejects:
  changing source or decoded inc at rho breaks translation relation.

same_polynomial_different_relation_ids_distinct:
  relation ID participates in view lookup.

precommitted_program_view_uses_original_commitment:
  BytecodeChunk and ProgramImageInit resolve to precommitted opening targets,
  not PackedWitness families.

precommitted_views_are_absent_from_w_pack_statement:
  TrustedAdvice, BytecodeChunk(i), ProgramImageInit, and bytecode-derived
  Store selector components are not emitted as W_pack packed-view claims.

fused_increment_store_uses_precommitted_bytecode_openings:
  Store selector claims resolve through BytecodeChunk openings, not W_pack
  families. Tampering with either component openings or the recombined source
  claim rejects.

precommitted_opening_manifest_is_exact:
  missing, extra, or reordered precommitted proofs reject before the W_pack
  packed-view proof can satisfy the statement.
```

## Performance

Expected:

```text
direct/linear:
  verifier cost proportional to alphabet terms in the decoded view.
  byte decode cost is O(256) terms per opened byte fact unless Akita backend
  supports a more compact internal relation.

masked:
  additional sumcheck cost over row domain.
  output claim count grows with source and decoded limb openings.
```

Rejected:

```text
- treating masked views as pointwise products at r.
- expanding all packed cells just to evaluate a linear decoded view.
- using decoded value claims without proving one-hot/range validity.
- hiding masked translation inside Akita proof without exposing logical
  coefficients needed by Jolt/BlindFold.
- treating byte-limb decode as a point opening unless Akita proves the packed
  linear view relation.
- satisfying bytecode-derived Store selector claims from proof-owned W_pack
  bytecode lanes without a binding to the BytecodeChunk commitment.
```

## Resolved Decisions And Open Questions

```text
resolved:
  jolt-claims owns logical view formulas and relation IDs. jolt-verifier
  resolves those formulas into PhysicalView statements. jolt-akita consumes
  already-resolved packed/direct physical views and does not inspect Jolt PIOP
  semantics.
  the target masked translations are RamInc and RdInc fused-increment
  translations.
  translation layout is derived from verifier config and preprocessing, then
  bound through layout/statement digests; it is not an independent prover-chosen
  proof-header layout.
  current byte-decode views are represented as packing terms. A backend
  optimization may replace this only if it proves the same packed view relation.

open:
  whether future non-increment masked translations are needed.
  which RA validity checks can be simplified once lattice packed-validity
  coverage is fully finalized.
```

## References

```text
- 00-roadmap.md: committed-program interface assumptions.
- 03-prefix-packed-witness.md: physical W_pack.
- ../selected-verifier-integration.md: verifier config context.
```
