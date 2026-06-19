# Spec: Akita Advice And Program Data Opening Policy

| Field | Value |
|-------|-------|
| Component | advice, field-inline, and committed-program opening policy |
| Depends On | 00-roadmap.md, 04-logical-views-and-translation.md, 05-onehot-increments.md |
| Unlocks | verifier config |
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-06-08 |
| Status | implemented (transparent verifier/protocol target) |

## Scope

Lattice mode packs proof-owned non-increment data into the same `PackedWitness`
commitment as RA and fused base increments. Precommitted data stays bound to its
original commitment and is opened separately.

In scope:

```text
- trusted advice opening policy.
- untrusted advice packing policy.
- arbitrary byte/field data decomposition.
- field-inline data and FieldRdInc policy.
- committed bytecode chunk opening policy.
- committed program image opening policy.
- canonical byte ordering.
- ZK rejection policy.
```

Out of scope:

```text
- base fused increment protocol.
- committed bytecode implementation.
- Akita ZK hiding protocol.
```

Assumptions:

```text
- ProgramMode::Committed exists in the target modular stack.
- Committed bytecode chunks encode expanded bytecode rows.
- Program image words are little-endian u64 words.
- Arbitrary field data has no small alphabet unless decomposed.
- Lattice mode targets one Akita proof over one proof-owned PackedWitness, plus
  separate openings for precommitted objects.
- TrustedAdvice, BytecodeChunk(i), and ProgramImageInit are precommitted.
- ZK is rejected for now.
```

Precommitted-object rule:

```text
TrustedAdvice, BytecodeChunk(i), and ProgramImageInit values must not be added
as ordinary proof-owned W_pack families. Their proofs are separate opening
proofs keyed by their original commitments.

A future bound packed-precommitted view may duplicate those values into a
packed object only if the protocol also proves equivalence to the original
commitment. That future view is an additional binding protocol, not the current
target.
```

## Canonical Encoding

Byte order:

```text
All integer-like data uses little-endian byte limbs.
```

Words:

```text
u64 word w:
  byte_j(w) = (w >> 8j) & 0xff
  j in [0, 8)
```

Field elements:

```text
canonical representative:
  integer in [0, p)

byte order:
  fixed-width little-endian bytes.

length:
  ceil(log2(p) / 8)
```

Signed base increments:

```text
handled by 05-onehot-increments.md.
```

No implicit encodings:

```text
Any value represented as one-hot bytes must name:
  byte count.
  byte order.
  domain rows.
  zero padding convention.
```

## Packed Families

Advice:

```text
UntrustedAdviceByte(j)

lifecycle:
  UntrustedAdviceByte is proof-owned.
  TrustedAdvice is preprocessing-owned when present and is not a W_pack family.

domain:
  advice byte index domain, not trace domain.

rows:
  exact committed/proof-declared byte length if transcript-bound.
  otherwise configured maximum length.
```

Advice values are byte data. If a higher-level value is a field element or word,
the owner module must serialize it through the canonical encoding above before
emitting bytes.

Trusted advice final openings are verified against the trusted-advice
precommitment. They may only enter a packed view in a future protocol that also
proves binding to that precommitment.

Field-inline:

```text
FieldRdInc:
  separate PackedWitness family from base IncByte/IncSign.

FieldInlineAdvice:
  byte families if field-inline advice exists.
```

Default field-inline representation:

```text
canonical field bytes:
  field representative in [0, p), little-endian fixed width.

Reason:
  FieldRdInc is field-valued, not base signed-u64-valued.
```

If a future field-inline path proves a smaller structured representation, it
must replace this policy with an explicit decode invariant.

Committed bytecode opening policy:

```text
BytecodeChunk(i) is a precommitted object over committed bytecode chunk rows and
lanes. Its final openings are not W_pack claims.

Lane classes:
  optional register selector lanes:
    rs1, rs2, rd.

  boolean flag lanes:
    circuit flags, instruction flags, raf flag.

  optional lookup selector lanes:
    lookup table selector.

  scalar lanes:
    unexpanded_pc, imm.
```

Committed bytecode representation for separate openings or future bound packed
precommitted views:

```text
selector/flag lanes:
  rs1/rs2/rd and lookup selector use optional one-hot validity.
  circuit flags, instruction flags, and RAF flag use boolean validity.

scalar lanes:
  unexpanded_pc uses 8 little-endian bytes of the u64 address.
  imm uses canonical field bytes for F::from_i128(imm).

Reason:
  current committed-bytecode core stores unexpanded_pc as F::from_u64(address)
  and imm as F::from_i128(imm).
```

Program image:

```text
ProgramImageInit is a precommitted object.
ProgramImageInitWordByte(j), j in [0, 8) is the canonical decomposition if a
future bound packed precommitted view is added.

domain:
  committed program-image word domain.

encoding:
  little-endian bytes of each u64 word.
```

Akita committed-program caveat:

```text
An Akita backend mode that commits program bytecode is not, by itself, the Jolt
precommitted-object set. Jolt still passes explicit original commitment handles
for trusted advice, bytecode chunks, and program image, and each such handle has
its own direct opening path. Copying those values into W_pack, or relying only
on the backend bytecode-commit mode, does not bind the final Jolt claims to the
original precommitted commitments.
```

ZK blinding:

```text
current target:
  rejected when PCS family is lattice.

future:
  a BlindFold-like hiding layer must define its own packed families,
  commitments, and verifier constraints.
```

## Layout

Each proof-owned packed family contributes:

```text
cells_f = rows_f * limbs_f * alphabet_f
```

Examples:

```text
advice bytes:
  rows_f = advice_len_bytes
  limbs_f = 1
  alphabet_f = 256

precommitted program objects:
  TrustedAdvice, BytecodeChunk(i), and ProgramImageInit do not contribute to
  D_pack unless a future bound packed precommitted view is enabled.
```

All proof-owned packed families enter the same global cell sum:

```text
D_pack = ceil_log2(sum_f cells_f)
```

There is no second proof-owned packed dimension in the target lattice protocol.

## Stage Interaction

Committed-program schedule:

```text
Stage 4:
  cache ProgramImageInitContributionRw when ProgramMode::Committed.

Stage 6a:
  run bytecode read-RAF address phase.
  run bytecode booleanity address phase.
  stage BytecodeValStage(i) claims without verifier-held full bytecode.

Stage 6b:
  run bytecode read-RAF cycle phase.
  run bytecode booleanity cycle phase.
  run BytecodeClaimReduction cycle phase when ProgramMode::Committed.
  run ProgramImageClaimReduction cycle phase when ProgramMode::Committed.
  expose committed bytecode Store/RdPresent facts before fused increment
  validity consumes them.

Stage 7:
  run BytecodeClaimReduction address/final phase when ProgramMode::Committed.
  run ProgramImageClaimReduction address/final phase when ProgramMode::Committed.
```

Advice:

```text
Existing advice reductions produce final logical advice openings.
Lattice view resolution maps untrusted advice openings to proof-owned advice
byte families in W_pack.
Trusted advice openings resolve to the trusted-advice precommitment.
```

Field-inline:

```text
Field-inline claim reductions keep their logical names.
Lattice view resolution maps FieldRdInc and field-inline data to separate
PackedWitness families.
```

Committed bytecode:

```text
Committed-program reductions produce BytecodeChunk(i) final logical openings.
Lattice view resolution maps those openings to separate openings against the
BytecodeChunk(i) commitments.
```

Program image:

```text
ProgramImageClaimReduction produces ProgramImageInit final logical openings.
Lattice view resolution maps those openings to a separate opening against the
ProgramImageInit commitment.
```

Stage 8:

```text
Proof-owned final logical claims are packed-view claims over one W_pack.
TrustedAdvice, BytecodeChunk(i), and ProgramImageInit receive separate openings
against their original commitments.
Committed-bytecode source facts used by fused increments, such as StoreFlag and
RdPresent, also resolve through separate BytecodeChunk(i) openings unless a
future bound precommitted packed view is specified.
If a source fact is a linear view over committed bytecode lanes, the verifier
checks the component-opening recombination before accepting the source claim.
```

## Implementation

`jolt-claims`:

```text
Define lattice family IDs for:
  UntrustedAdviceByte.
  FieldRdInc bytes.

Define decode formulas for:
  byte arrays.
  canonical field elements.

Define validity formulas for:
  one-hot byte families.
  canonical field-byte encodings.

Define precommitted opening policies for:
  TrustedAdvice.
  BytecodeChunk(i).
  ProgramImageInit.
```

`jolt-verifier`:

```text
When PCS family is lattice:
  require ProgramMode::Committed.
  enable field-inline only when FieldRdInc byte policy is available.
  enable untrusted advice only when proof-owned advice byte families are
  available.
  enable trusted advice only when a trusted-advice precommitment and separate
  opening path are available.
  require separate precommitted openings for BytecodeChunk(i) and
  ProgramImageInit.
  require fused-increment source claims to use BytecodeChunk(i) openings or an
  explicit bound precommitted packed view.
  reject ZK.
  derive all proof-owned families into one PackedWitnessLayout.
```

`jolt-akita`:

```text
Accept PackedWitnessLayout with heterogeneous alphabets.
Reject dense proof-owned facts that bypass PackedWitness.
Reject byte-decode views unless the backend or adapter proves the linear view.
```

Prover witness source:

```text
Emit nonzero one-hot symbols only.
Use symbol 0 for zero bytes.
Do not emit global dummy cells.
```

## Invariants

```text
- Lattice mode uses one PackedWitness commitment for supported proof-owned
  packed facts.
- Precommitted facts keep separate openings against their original commitments.
- Precommitted commitments and opening proofs must not alias the PackedWitness
  commitment.
- Advice domains are advice domains, not trace domains.
- Field-inline FieldRdInc is separate from base fused increments.
- Field elements use canonical representatives in [0, p).
- All byte decomposition is little-endian.
- Every byte limb is a one-hot byte family unless the family is explicitly
  boolean or optional-selector valued.
- Program image words match current little-endian RAM preprocessing.
- Increment source selection is derived from committed bytecode Store and rd
  presence lanes.
- Increment source openings are precommitted BytecodeChunk openings, not W_pack
  bytecode-family openings.
- BytecodeChunk(i) and ProgramImageInit do not inherit trace-domain row count.
- ZK with lattice is rejected until a hiding protocol is specified.
- No dense proof-owned path bypasses PackedWitness silently.
```

## Tests

Targeted tests:

```text
untrusted_advice_bytes_use_advice_domain:
  untrusted advice family cells are computed from advice byte length, not T.

trusted_advice_uses_precommitted_opening:
  trusted advice final openings are checked against the trusted-advice
  precommitment, not W_pack.

trusted_advice_must_not_alias_w_pack:
  trusted advice precommitment equal to the PackedWitness commitment rejects.

untrusted_advice_encoding_roundtrip:
  untrusted advice bytes decode to the original byte stream.

advice_byte_onehot_validity_rejects:
  malformed UntrustedAdviceByte limb rejects.

field_element_encoding_is_canonical:
  non-canonical field byte representation rejects.

field_rd_inc_uses_field_family:
  FieldRdInc does not reuse base IncByte/IncSign families.

precommitted_program_openings_use_original_commitments:
  BytecodeChunk(i) and ProgramImageInit openings are checked against their
  preprocessing commitments.

precommitted_program_commitments_must_not_alias_w_pack:
  BytecodeChunk(i) or ProgramImageInit commitment equal to the PackedWitness
  commitment rejects.

precommitted_program_not_in_w_pack:
  enabling ProgramMode::Committed does not add BytecodeChunk(i) or
  ProgramImageInit families to the proof-owned PackedWitness layout.

fused_source_facts_not_in_w_pack:
  StoreFlag and RdPresent source claims are checked through BytecodeChunk(i)
  openings or a bound precommitted packed view, not proof-owned bytecode
  families in W_pack.

zk_lattice_rejects:
  lattice family with zk feature fails before proof verification.

single_packed_witness_layout_includes_all_proof_owned_families:
  enabling untrusted advice and field-inline changes D_pack, while
  precommitted-program openings do not.
```

## Performance

Expected:

```text
advice:
  cost scales with advice byte length.

field-inline:
  cost scales with canonical field-byte width unless a smaller representation
  is specified.

bytecode:
  separate opening cost is paid against committed bytecode commitments.

program image:
  separate opening cost is paid against the program-image commitment.

setup:
  one D_pack accounts for proof-owned packed families.
```

Rejected:

```text
- forcing advice into trace-sized rows.
- satisfying precommitted advice or committed program objects only through
  W_pack.
- dense proof-owned commitments that bypass PackedWitness in lattice mode.
- byte-decomposing field data without canonical representative checks.
- using big-endian byte ordering for program image or advice integers.
- silently enabling ZK blinding families.
```

## Resolved Decisions And Open Questions

```text
resolved:
  advice PackedWitness families are sized from the precommitted advice schedule
  and protocol config. Untrusted advice is proof-owned packed data; trusted
  advice is precommitted and uses a separate opening path.
  field-inline FieldRdInc currently uses canonical field-byte one-hot families
  and a canonical-byte packed-validity check against the Akita fp128 modulus.
  byte-decode views currently lower to verifier-facing packed-linear terms.

open:
  whether field-inline FieldRdInc can use a smaller structured encoding than
  canonical field bytes.
  whether a future Akita backend API can prove byte-decode linear views without
  exposing 256 weighted terms in the Jolt-facing statement.
```

## References

```text
- 00-roadmap.md: committed-program interface assumptions.
- 03-prefix-packed-witness.md: PackedWitness dimension rules.
- 04-logical-views-and-translation.md: logical-to-physical views.
- 05-onehot-increments.md: base increment encoding.
- https://github.com/a16z/jolt/blob/main/specs/1344-committed-bytecode-program-image.md:
  committed bytecode and program-image background.
```
