# Spec: Akita One-Hot Increments

> Superseded for base Jolt increments by `08-fused-increment-piop.md`.
> This file records the earlier byte/sign design. The current modular Akita
> base-increment surface is `UnsignedIncChunk(j)` plus trace-domain
> `UnsignedIncMsb`; `FieldRdIncByte` and `FieldRdIncSign` remain only as
> field-inline auxiliary families.

| Field | Value |
|-------|-------|
| Component | fused one-hot increments |
| Depends On | 04-logical-views-and-translation.md |
| Unlocks | verifier config, advice/field-inline packing |
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-06-08 |
| Status | superseded by 08-fused-increment-piop.md |

## Scope

Base Jolt increment claims stay logical:

```text
RamInc(r)
RdInc(r)
```

The lattice path commits one fused increment representation inside `W_pack`.
It does not commit dense `RamInc`/`RdInc`, and it does not commit separate
one-hot `RamInc` and `RdInc` byte families.

In scope:

```text
- fused base increment byte/sign representation.
- sign-magnitude decode matching current core increment semantics.
- committed-bytecode source derivation.
- source-filtered translation for RamInc and RdInc.
- canonical zero encoding.
- Stage 6 placement target.
```

Out of scope:

```text
- field-inline FieldRdInc encoding.
- advice byte decomposition.
- final verifier flag matrix.
- Akita ZK support.
```

Assumptions:

```text
- Base increments are `next(t) - current(t)`.
- Current core computes:
    RamInc(t) from `RAMAccess::Write(post_value - pre_value)`.
    RdInc(t) from `cycle.rd_write(post_value - pre_value)`.
- Current core represents the signed value as 64-bit magnitude plus sign.
- Committed bytecode is over expanded Jolt bytecode rows.
- Committed bytecode lanes include Store and rd one-hot lanes.
- Expanded rows satisfy Store * RdPresent = 0.
```

## Architecture

Fused increment families:

```text
IncByte(j), j in [0, 8):
  one-hot byte limb for the magnitude.

IncSign:
  boolean sign bit.
```

Cell budget over trace rows:

```text
cells_per_row = 8 * 256 + 2 = 2050

base RA cells_per_row = 5888
base + fused increments = 7938

D_pack = n + 13 for the large-trace base shape.
```

Sign-magnitude decode:

```text
Magnitude(x) = sum_{j=0}^{7} 256^j * Byte_j(x)

SignedInc(x) =
  Magnitude(x)   if Sign(x) = 0
  -Magnitude(x)  if Sign(x) = 1
```

Canonical zero:

```text
Magnitude(x) = 0 => Sign(x) = 0
```

Byte validity:

```text
For each row x and byte limb j:
  sum_{b=0}^{255} IncByte(j, x, b) = 1
  IncByte(j, x, b) * (IncByte(j, x, b) - 1) = 0

Sign(x) * (Sign(x) - 1) = 0
```

Source derivation:

```text
StoreFlag(x):
  committed bytecode Store circuit flag for the expanded bytecode row read by
  trace row x.

RdPresent(x):
  sum_{r=0}^{31} committed bytecode rd_lane_r for the expanded bytecode row
  read by trace row x.

Disjointness:
  StoreFlag(x) * RdPresent(x) = 0
```

This is an expanded-row invariant, not a generic instruction-level invariant.
Read-modify-write instruction families must be audited against the actual
expanded `Cycle` sequence before fused increments are enabled.

No base increment selector is committed. Selection is derived from committed
bytecode facts.
Those bytecode facts remain precommitted-program facts; the fused-increment
packed witness does not by itself open the BytecodeChunk commitments.

Logical reconstruction:

```text
RamInc(x) = StoreFlag(x) * SignedInc(x)
RdInc(x)  = RdPresent(x) * SignedInc(x)
```

Inactive rows:

```text
If StoreFlag(x) + RdPresent(x) = 0:
  Magnitude(x) = 0
  Sign(x) = 0
```

This allows a Store/Rd row to have zero delta, but prevents unused rows from
carrying arbitrary hidden increment bytes.

## Translation

Final logical claims are row-MLE claims:

```text
RamInc(r) = sum_x eq(r, x) * StoreFlag(x) * SignedInc(x)
RdInc(r)  = sum_x eq(r, x) * RdPresent(x) * SignedInc(x)
```

These are masked views. They are not equal to:

```text
StoreFlag(r) * SignedInc(r)
RdPresent(r) * SignedInc(r)
```

Therefore the lattice extension must add translation relations before Stage 8.

Translation relation inputs:

```text
RamInc(r) from existing IncClaimReduction.
RdInc(r) from existing IncClaimReduction.
```

Translation relation outputs:

```text
proof-owned W_pack openings:
  IncByte(j, rho, b) views or decoded byte views.
  IncSign(rho).

precommitted bytecode openings:
  StoreFlag(rho) through Store circuit-flag lane openings against the original
  BytecodeChunk(i) commitments.
  RdPresent(rho) through rd selector lane openings against the original
  BytecodeChunk(i) commitments, unless a dedicated committed rd-present view is
  added.
```

The exact output shape depends on whether `jolt-akita` proves byte-decode
linear views directly or whether jolt-claims exposes decoded byte openings.
The bytecode source openings stay outside W_pack unless a future bound
precommitted packed view proves equivalence to the original BytecodeChunk(i)
commitment.

The verifier checks that bytecode component openings recombine to the
StoreFlag/RdPresent source claims before those claims are used in the masked
translation. This check is part of the precommitted opening path, not the
W_pack packed-view proof.

The prover-facing Akita helper follows the same split: it can evaluate
IncByte/IncSign and BytecodeRa claims from the proof-owned PackedWitness, but
StoreFlag/RdPresent source components must be supplied from the committed
bytecode path. The helper recombines those supplied components into the
aggregate Stage 6 source-link claims.

## Committed Bytecode Link

Current committed bytecode lanes encode:

```text
rs1 one-hot
rs2 one-hot
rd one-hot
unexpanded_pc scalar
imm scalar
circuit flags
instruction flags
lookup table selector
raf flag
```

For increments:

```text
StoreFlag:
  circuit flag Store.

RdPresent:
  sum of the rd one-hot lanes.
```

The relation must link trace row `x` to an expanded bytecode row through the
existing bytecode read-RAF path. Source derivation must use the expanded row,
not the unexpanded PC alone, because several expanded rows may share one source
address.

Stage 6 is the target location:

```text
Stage 6:
  bytecode read-RAF has produced committed bytecode row semantics.
  current IncClaimReduction closes RamInc/RdInc logical claims.
  fused increment translation can consume both, but bytecode source facts must
  be opened against their BytecodeChunk commitments or a future bound
  precommitted packed view.
```

## Implementation

`jolt-claims`:

```text
Add lattice fact IDs:
  IncByte(j)
  IncSign

Add formulas:
  byte one-hot validity.
  sign booleanity.
  canonical zero.
  inactive-row zero.
  Store/RdPresent derivation from committed bytecode lanes.
  RamInc/RdInc masked translation.

Reuse existing bytecode read-RAF staged row semantics for Store/RdPresent.
Do not add a new committed selector family unless those staged claims cannot
expose the required Store and rd-present evaluations.
Do not add BytecodeChunk lanes to W_pack for Store/RdPresent. They require
separate component openings against the original BytecodeChunk commitments
unless a future protocol also proves binding between a packed precommitted view
and those commitments.
```

`jolt-verifier`:

```text
When PCS family is lattice:
  require IncrementCommitmentMode::FusedOneHot.
  derive IncByte/IncSign families in PackedWitnessLayout.
  require ProgramMode::Committed.
  reject dense or separate base increment modes.
  schedule fused increment translation after bytecode source data is available.
```

`jolt-akita`:

```text
Expose IncByte and IncSign as PackedWitness families.
Support direct/linear views needed by the translation output.
Reject treating byte-decode as a point opening unless the backend proves the
packing view relation.
```

Prover witness conversion:

```text
for each trace row x:
  store = StoreFlag(x)
  rd = RdPresent(x)
  assert store * rd = 0

  if store = 1:
    delta = RamInc(x)
  else if rd = 1:
    delta = RdInc(x)
  else:
    delta = 0

  encode abs(delta) as 8 little-endian byte limbs.
  encode sign(delta).
  if delta = 0:
    sign = 0
```

## Invariants

```text
- Lattice mode has one fused base increment representation.
- Dense RamInc/RdInc commitments are not used in lattice Stage 8.
- Separate one-hot RamInc/RdInc is not the target protocol.
- Byte limbs are one-hot.
- Sign is boolean.
- Magnitude zero implies sign zero.
- Inactive rows have zero magnitude and sign zero.
- StoreFlag and RdPresent are derived from committed expanded bytecode rows.
- BytecodeChunk commitments are opened or otherwise bound separately from the
  fused-increment W_pack facts.
- StoreFlag and RdPresent source claims are precommitted bytecode openings, not
  proof-owned W_pack openings.
- StoreFlag * RdPresent = 0.
- RamInc/RdInc logical claims are reconstructed by masked translation.
- The signed decode is sign-magnitude, not two's complement.
```

## Tests

Targeted tests:

```text
fused_increment_reconstructs_dense:
  source-filtered decoded increments equal dense RamInc/RdInc witness values.

negative_increment_roundtrip:
  sign=1 and magnitude limbs reconstruct a negative delta.

zero_increment_sign_canonical:
  magnitude zero with sign one rejects.

inactive_increment_row_must_be_zero:
  row with no Store and no rd lane rejects nonzero magnitude or sign.

store_and_rd_present_disjoint:
  expanded bytecode rows reject StoreFlag * RdPresent != 0.

read_modify_write_expansion_is_disjoint:
  AMO/read-modify-write expanded rows never expose StoreFlag=1 and
  RdPresent=1 on the same expanded row.

fused_source_link_tamper_rejects:
  tampered Store or rd source claim breaks RamInc/RdInc translation.

fused_source_link_requires_precommitted_openings:
  Store/RdPresent source claims cannot be satisfied by W_pack bytecode lanes
  without a binding to the original BytecodeChunk commitment. Component
  openings must recombine to the claimed StoreFlag/RdPresent values.

separate_increment_mode_rejects_lattice:
  verifier rejects separate base increments under lattice family.

dense_increment_mode_rejects_lattice:
  verifier rejects dense base increments under lattice family.

byte_limb_not_onehot_rejects:
  malformed IncByte limb fails validity.
```

## Performance

Expected:

```text
fused increments:
  +2050 cells / row over base RA facts.
  D_pack = n + 13 for the large-trace base shape.
  pays committed-bytecode source derivation and masked translation.

separate non-target comparison:
  +4100 cells / row over base RA facts.
  D_pack = n + 14 for the large-trace base shape.
```

Rejected:

```text
- committing a selector for base increments when bytecode facts already
  provide Store/RdPresent.
- committing RamInc and RdInc separately in lattice mode.
- two's-complement base increment decode.
- materializing padded increment cells.
- assuming fused mode is cheaper without accounting for translation cost.
```

## Resolved Decisions And Open Questions

```text
resolved:
  committed-bytecode StoreFlag/RdPresent disjointness is a packed-validity
  requirement for the committed-bytecode source layout.
  Stage 6 derives fused-increment translation and source-link outputs; Stage 8
  consumes those outputs and the separate precommitted BytecodeChunk component
  openings.

open:
  additional expanded row classes may get fixture coverage as committed-bytecode
  prover integration lands.
```

## References

```text
- 00-roadmap.md: committed-program interface assumptions.
- 03-prefix-packed-witness.md: dimension budget.
- 04-logical-views-and-translation.md: masked view translation.
- https://github.com/a16z/jolt/blob/main/specs/1344-committed-bytecode-program-image.md:
  committed bytecode lane layout.
```
