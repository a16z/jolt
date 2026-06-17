# Spec: Akita Advice And Program Data One-Hotting

| Field | Value |
|-------|-------|
| Component | advice, field-inline, and committed-program packing |
| Depends On | 00-roadmap.md, 04-logical-views-and-translation.md, 05-onehot-increments.md |
| Unlocks | verifier config |
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-06-08 |
| Status | draft |

## Scope

Lattice mode packs all supported non-increment data into the same
`PackedWitness` commitment as RA and fused base increments.

In scope:

```text
- trusted advice.
- untrusted advice.
- arbitrary byte/field data decomposition.
- field-inline data and FieldRdInc policy.
- committed bytecode chunks.
- committed program image.
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
- Lattice mode targets one Akita proof over one PackedWitness.
- ZK is rejected for now.
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
TrustedAdviceByte(j)
UntrustedAdviceByte(j)

lifecycle:
  TrustedAdviceByte is preprocessing-owned when trusted advice is present.
  UntrustedAdviceByte is proof-owned.

domain:
  advice byte index domain, not trace domain.

rows:
  exact committed/proof-declared byte length if transcript-bound.
  otherwise configured maximum length.
```

Advice values are byte data. If a higher-level value is a field element or word,
the owner module must serialize it through the canonical encoding above before
emitting bytes.

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

Committed bytecode:

```text
BytecodeChunk(i) families are over committed bytecode chunk rows and lanes.

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

Committed bytecode representation:

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
ProgramImageInitWordByte(j), j in [0, 8)

domain:
  committed program-image word domain.

encoding:
  little-endian bytes of each u64 word.
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

Each family contributes:

```text
cells_f = rows_f * limbs_f * alphabet_f
```

Examples:

```text
advice bytes:
  rows_f = advice_len_bytes
  limbs_f = 1
  alphabet_f = 256

program image u64 words:
  rows_f = program_image_words
  limbs_f = 8
  alphabet_f = 256

bytecode register selector lane:
  rows_f = bytecode_chunk_rows
  limbs_f = 1
  alphabet_f = 32

bytecode flag lane:
  rows_f = bytecode_chunk_rows
  limbs_f = 1
  alphabet_f = 2
```

All families enter the same global cell sum:

```text
D_pack = ceil_log2(sum_f cells_f)
```

There is no second packed dimension in the target lattice protocol.

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
Lattice view resolution maps those openings to advice byte families in W_pack.
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
Lattice view resolution maps those openings to committed-bytecode families.
```

Program image:

```text
ProgramImageClaimReduction produces ProgramImageInit final logical openings.
Lattice view resolution maps those openings to program-image byte families.
```

Stage 8:

```text
All supported final logical claims are packed-view claims over one W_pack.
Akita receives one PackedWitness commitment and one packed-view proof.
```

## Implementation

`jolt-claims`:

```text
Define lattice family IDs for:
  TrustedAdviceByte.
  UntrustedAdviceByte.
  FieldRdInc bytes.
  committed bytecode optional selector lanes.
  committed bytecode boolean flag lanes.
  committed bytecode scalar byte lanes.
  ProgramImageInit word bytes.

Define decode formulas for:
  byte arrays.
  canonical field elements.
  committed bytecode scalar lanes.
  program image words.

Define validity formulas for:
  one-hot byte families.
  optional one-hot committed-bytecode selector lanes.
  boolean committed-bytecode flag lanes.
  canonical field-byte encodings.
  fixed-length program-image word bytes.
```

`jolt-verifier`:

```text
When PCS family is lattice:
  require ProgramMode::Committed.
  enable field-inline only when FieldRdInc byte policy is available.
  enable advice only when advice byte families are available.
  reject ZK.
  derive all families into one PackedWitnessLayout.
```

`jolt-akita`:

```text
Accept PackedWitnessLayout with heterogeneous alphabets.
Reject dense non-PackedWitness commitments.
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
- Lattice mode uses one PackedWitness commitment for supported committed facts.
- Advice domains are advice domains, not trace domains.
- Field-inline FieldRdInc is separate from base fused increments.
- Field elements use canonical representatives in [0, p).
- All byte decomposition is little-endian.
- Every byte limb is a one-hot byte family unless the family is explicitly
  boolean or optional-selector valued.
- Program image words match current little-endian RAM preprocessing.
- Increment source selection is derived from committed bytecode Store and rd
  presence lanes.
- BytecodeChunk(i) and ProgramImageInit do not inherit trace-domain row count.
- ZK with lattice is rejected until a hiding protocol is specified.
- No dense non-PackedWitness path is silently enabled.
```

## Tests

Targeted tests:

```text
advice_bytes_use_advice_domain:
  advice family cells are computed from advice byte length, not T.

trusted_advice_encoding_roundtrip:
  trusted advice bytes decode to the original byte stream.

untrusted_advice_encoding_roundtrip:
  untrusted advice bytes decode to the original byte stream.

advice_byte_onehot_validity_rejects:
  malformed TrustedAdviceByte or UntrustedAdviceByte limb rejects.

field_element_encoding_is_canonical:
  non-canonical field byte representation rejects.

field_rd_inc_uses_field_family:
  FieldRdInc does not reuse base IncByte/IncSign families.

program_image_bytes_are_little_endian:
  u64 word 0x0807060504030201 maps to bytes 1..8.

bytecode_chunk_uses_bytecode_domain:
  committed bytecode family rows are chunk rows/lanes, not trace rows.

bytecode_scalar_lane_byte_roundtrip:
  unexpanded_pc/imm byte limbs decode to the committed scalar lane value.

bytecode_optional_selector_validity_rejects:
  malformed rs1, rs2, rd, or lookup optional one-hot lane rejects.

bytecode_flag_booleanity_rejects:
  malformed circuit flag, instruction flag, or RAF bit rejects.

bytecode_pc_uses_u64_bytes:
  unexpanded_pc lane encodes the u64 address in 8 little-endian bytes.

bytecode_imm_uses_canonical_field_bytes:
  imm lane encodes the field element F::from_i128(imm) canonically.

zk_lattice_rejects:
  lattice family with zk feature fails before proof verification.

single_packed_witness_layout_includes_all_supported_families:
  enabling advice, field-inline, and ProgramMode::Committed changes D_pack,
  not a second packed dimension.
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
  selector/flag lanes are cheap relative to byte-decomposed scalar lanes.

program image:
  cost scales with program-image word count * 8 byte limbs.

setup:
  one D_pack accounts for all families.
```

Rejected:

```text
- forcing advice into trace-sized rows.
- separate Akita proofs for advice or committed program objects.
- dense non-PackedWitness commitments in lattice mode.
- byte-decomposing field data without canonical representative checks.
- using big-endian byte ordering for program image or advice integers.
- silently enabling ZK blinding families.
```

## Questions

```text
1. Should advice use exact byte length or configured maximum length when both
   are transcript-bound?
2. Can field-inline FieldRdInc use a smaller structured encoding than canonical
   field bytes?
3. Can Akita's backend prove byte-decode linear views directly enough to avoid
   expanding 256 terms in verifier-facing statements?
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
