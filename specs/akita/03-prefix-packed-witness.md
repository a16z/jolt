# Spec: Akita Prefix-Packed PackedWitness

| Field | Value |
|-------|-------|
| Component | prefix-packed PackedWitness |
| Depends On | 02-jolt-akita-crate.md |
| Unlocks | logical views, one-hot increments |
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-06-08 |
| Status | draft |

## Scope

Akita commits supported proof-owned lattice facts through one physical object
`W_pack`. `W_pack` is a prefix-packed one-hot object, not the Dory list of
logical polynomials and not a replacement for verifier/preprocessing
commitments.

In scope:

```text
- prefix-packed PackedWitness geometry
- fact alphabets
- cell-count and dimension accounting
- padding rules
- layout digest requirements
```

Out of scope:

```text
- logical claim translation
- increment one-hot semantics
- concrete advice and field-inline byte policies
- verifier proof shape
```

Assumptions:

```text
- Trace length is T = 2^n.
- Packed facts may be trace-domain, proof-owned advice-domain, or other
  proof-owned auxiliary facts.
- TrustedAdvice, BytecodeChunk(i), and ProgramImageInit are precommitted and are
  excluded from W_pack unless a future binding protocol proves equivalence to
  their original commitments.
- Exclusion is by commitment class, not by value shape. A precommitted value
  copied into W_pack is only a proof-owned copy and does not open the original
  precommitted commitment.
- Akita setup cost is sensitive to D_pack = ceil_log2(total packed cells).
- The prover can stream nonzero one-hot facts without materializing W_pack.
- Padding to 2^D_pack is an ambient domain property, not a witness construction
  instruction.
```

## Architecture

Packed object:

```text
W_pack : {0,1}^D_pack -> F
```

Logical indexing before padding:

```text
cell = (family, row, limb, symbol)

family:
  fact family identifier

row:
  row in the family domain

limb:
  byte/sign/subvalue lane inside the family

symbol:
  one-hot alphabet symbol
```

Packed rank:

```text
Each fact family f owns an interval:
  [offset_f, offset_f + rows_f * limbs_f * alphabet_f)

rank(f, row, limb, symbol):
  offset_f + ((row * limbs_f + limb) * alphabet_f) + symbol

W_pack(rank(f, row, limb, symbol)) = Fact_f(row, limb, symbol)
```

Hypercube point:

```text
rank < cells <= 2^D_pack
binary(rank) maps to a point in {0,1}^D_pack

rank >= cells:
  dummy cell
  value = 0
  no logical view may reference it
```

Prefix packing meaning:

```text
The prefix is not necessarily a fixed-width lane.
Small alphabets receive small intervals.

bit fact:
  2T cells

byte fact:
  256T cells

This is the source of the dimension savings over rectangular packing.
```

Cell count:

```text
cells = sum_f rows(f) * limbs(f) * alphabet(f)
D_pack = ceil_log2(cells)
```

For trace-domain byte facts:

```text
rows(f) = T
alphabet(f) = 256
```

For trace-domain bit facts:

```text
rows(f) = T
alphabet(f) = 2
```

Base large-trace RA budget:

```text
instruction RA facts: 16 byte facts
bytecode RA facts:     3 byte facts
RAM RA facts:          4 byte facts

base byte facts = 23
base cells / row = 23 * 256 = 5888
base D_pack = n + 13
```

Representative dimensions:

```text
base only:
  log_T = 20 -> D_pack = 33
  log_T = 25 -> D_pack = 38
  log_T = 30 -> D_pack = 43

non-target split increment model:
  log_T = 20 -> D_pack = 34
  log_T = 25 -> D_pack = 39
  log_T = 30 -> D_pack = 44

fused increments:
  log_T = 20 -> D_pack = 33
  log_T = 25 -> D_pack = 38
  log_T = 30 -> D_pack = 43
```

Interpretation:

```text
Crossing from n + 13 to n + 14 adds one MLE variable.
It does not mean the Jolt trace dimension jumped to 64 variables.
Any Akita backend setup may still have its own internal ring/vector padding;
that padding is backend-specific and must be reported separately.
```

Informative increment budgets:

```text
non-target split increment model:
  base + 2 * (8 byte facts + 1 bit fact)
  cells / row = 5888 + 2 * (8 * 256 + 2) = 9988
  D_pack = n + 14

fused increments:
  base + 8 byte facts + sign bit
  cells / row = 5888 + 8 * 256 + 2 = 7938
  D_pack = n + 13
```

Base increment source is not a PackedWitness family:

```text
Ram source = committed bytecode Store flag.
Rd source  = committed bytecode rd one-hot presence.
These source facts come from the committed-bytecode opening path, not from a
proof-owned W_pack family, unless a future bound precommitted packed view is
specified.
```

Padding:

```text
ambient padding:
  allowed. D rounds to a hypercube.

materialized dummy cells:
  rejected. Prover should not scan or commit explicit dummy cells when Akita
  can avoid it.
```

Layout construction:

```text
1. collect enabled fact families from verifier config.
2. sort by stable PackedFamilyId.
3. assign offset by cumulative cell count.
4. compute cells and D_pack.
5. transcript-bind layout digest before commitment-opening challenges.
```

Fact family record:

```text
FactFamily {
  id,
  domain,
  limbs,
  alphabet,
  offset,
  cell_count,
  view_kind,
}
```

PackedWitness fact classes:

```text
RA one-hot facts:
  InstructionRa(d)
  BytecodeRa(d)
  RamRa(d)

fused base increment facts:
  IncMagnitudeByte(j), j in [0, 8)
  IncSign

field-inline facts:
  FieldRdInc bytes or field-specific canonical limbs.
  field-inline advice/aux bytes when enabled.

advice facts:
  UntrustedAdvice bytes.

excluded precommitted facts:
  TrustedAdvice bytes.
  BytecodeChunk(i) committed bytecode lanes.
  ProgramImageInit little-endian word bytes.
  These are opened through separate statements against their original
  commitments.
  They are never PackedWitness families in the target protocol.
```

Precommitted opening schedule:

```text
TrustedAdvice:
  direct opening against the trusted-advice commitment.

BytecodeChunk(i):
  direct/component openings against the committed-bytecode chunk commitment.

ProgramImageInit:
  direct opening against the committed program-image commitment.

Fused-increment source facts:
  StoreFlag and RdPresent are derived from BytecodeChunk(i) component openings.
  They do not add bytecode selector families to W_pack.
```

Layout:

```text
fact_id
alphabet_size
domain_size
limb_count
offset/range in packed object
dummy-cell convention
dimension D_pack
```

Transcript:

```text
1. layout fields or digest.
2. PackedWitness commitment.
3. later opening point and claims.
```

Implementation plan:

```text
jolt-claims:
  define stable PackedFamilyId values in the lattice extension.
  expose RA fact counts through existing one-hot dimensions.
  expose fused base increment fact families through lattice increment mode.
  expose field-inline/proof-owned-advice families through lattice policy.
  expose trusted-advice and precommitted-program opening IDs separately from
  PackedWitness families.

jolt-akita:
  implement PackedWitnessLayout and PackedFamily.
  implement rank/unrank.
  compute cells and D_pack.
  compute layout digest.
  expose a streaming PackedWitnessSource.

jolt-verifier:
  derives expected PackedWitnessLayout from protocol config.
  stores/binds layout digest in proof/preprocessing path.
  rejects proof layout mismatch.
```

Proposed Rust data model:

```rust
pub struct PackedWitnessLayout {
    pub families: Vec<FactFamily>,
    pub cells: usize,
    pub dimension: usize,
    pub digest: [u8; 32],
}

pub struct FactFamily {
    pub id: FactId,
    pub domain: FactDomain,
    pub limbs: usize,
    pub alphabet: Alphabet,
    pub offset: usize,
    pub cell_count: usize,
    pub view_kind: ViewKind,
}

pub enum Alphabet {
    Bit,
    Byte,
    Fixed { size: usize },
}

pub enum FactDomain {
    TraceRows { log_t: usize },
    AdviceBytes { kind: AdviceKind, log_bytes: usize },
}
```

Planner algorithm:

```text
fn build_packed_witness_layout(config, dimensions) -> PackedWitnessLayout:
  families = []
  append RA families from OneHotConfig dimensions
  append fused base increment magnitude/sign families
  append field-inline families when field-inline is enabled
  append proof-owned advice byte families when advice is enabled
  do not append TrustedAdvice, BytecodeChunk(i), ProgramImageInit, or committed
  bytecode source lanes used by fused increments
  sort families by FactId
  offset = 0
  for family:
    family.offset = offset
    family.cell_count =
      rows(family.domain) * family.limbs * family.alphabet.size()
    offset += family.cell_count
  cells = offset
  dimension = ceil_log2(cells)
  digest = hash(canonical_serialize(families, cells, dimension))
```

Planner audit fields:

```text
fact_count_by_alphabet:
  number of bit, byte, and fixed-alphabet families.

cells_by_domain:
  TraceRows and proof-owned AdviceBytes.

cells_per_trace_row:
  trace-domain contribution normalized by T.

rectangular_lane_equivalent:
  byte-lane count needed by rectangular packing for comparison only.

D_pack:
  packed MLE dimension from total cells.
```

Commit-facing planner output:

```text
PackedWitnessLayout:
  public layout object or verifier-derived equivalent.

PackedWitnessSource:
  prover-owned stream of nonzero packed cells.

AkitaLayoutDigest:
  digest bound before commitment-opening challenges.

ViewCatalog:
  map from logical fact family to direct, linear, or reduced masked view.
```

The planner does not choose protocol semantics. It consumes fact families
defined by `jolt-claims` and emits the physical layout consumed by `jolt-akita`.

Streaming witness source:

```text
trait PackedWitnessSource<F> {
    fn layout(&self) -> &PackedWitnessLayout;
    fn for_each_nonzero(&self, f: impl FnMut(usize, F));
    fn eval_direct_fact(&self, fact: FactId, row: usize, limb: usize, symbol: usize) -> F;
}
```

Software decisions:

```text
layout derivation:
  verifier derives layout from public config.
  prover may include digest, not arbitrary layout fields, unless a later proof
  format explicitly includes audit/debug fields.

rank representation:
  use usize for planner tests.
  reject dimensions/cell counts that overflow platform usize.

dummy cells:
  not emitted by PackedWitnessSource.
  backend receives cells/D and zero convention.

FactId ordering:
  fixed enum order, not insertion order.
```

## Invariants

```text
- The layout uniquely maps every supported proof-owned packed fact to packed cell
  ranges.
- Bit facts cost 2T cells, not 256T cells.
- Dummy cells are zero and not addressable by logical opening IDs.
- D_pack is computed from layout cell count, not from prover witness length.
- Prefix assignment is deterministic.
- offset_f + cell_count_f <= cells for every fact family.
- rank/unrank is injective over non-dummy cells.
- Fact family ordering is stable across prover and verifier.
- The dimension reported to Akita equals layout.D_pack.
- Backend-specific setup padding is reported separately from Jolt cell padding.
- W_pack contains only supported proof-owned packed facts.
- Precommitted facts remain outside the PackedWitness layout unless an explicit
  binding protocol is specified.
- Copying a precommitted fact value into W_pack does not satisfy any
  precommitted opening requirement.
- Any logical opening whose commitment class is precommitted resolves to a
  separate opening statement keyed by the original commitment handle.
- PackedWitnessSource emits only cells owned by PackedWitnessLayout.
- ViewCatalog entries reference existing fact families and cannot create new
  cells.
```

## Tests

Targeted tests:

```text
packed_witness_layout_digest_stable:
  same config gives same digest.

packed_witness_layout_rejects_duplicate_ranges:
  no two fact families own the same cell range.

large_trace_base_cells_are_5888_per_row:
  16 + 3 + 4 byte facts account correctly.

separate_increment_budget_is_n_plus_14:
  separate RamInc/RdInc accounting crosses the expected dimension and is not
  the lattice target.

fused_increment_budget_is_n_plus_13:
  target fused budget remains below the next dimension.

bit_fact_costs_two_cells_per_row:
  sign facts are not byte lanes.

rank_unrank_roundtrip:
  every non-dummy cell maps back to exactly one (fact, row, limb, symbol).

dummy_cells_are_zero_and_unreferenced:
  no logical view maps to rank >= cells.

layout_sort_order_is_stable:
  changing input insertion order does not change digest.

precommitted_program_families_are_excluded:
  TrustedAdvice, BytecodeChunk(i), and ProgramImageInit do not change D_pack
  unless an explicit bound-precommitted packed-view protocol is enabled.

precommitted_values_cannot_enter_packed_source:
  PackedWitnessSource rejects or cannot construct families for TrustedAdvice,
  BytecodeChunk(i), ProgramImageInit, StoreFlag, or RdPresent source lanes.

planner_audit_fields_are_reported:
  layout tests expose fact_count_by_alphabet, cells_by_domain,
  rectangular_lane_equivalent, and D_pack.

packed_witness_source_respects_layout:
  every streamed rank is below cells and maps to the declared family/range.

view_catalog_references_existing_families:
  adding a view for a missing fact family rejects during layout construction.
```

## Performance

Expected:

```text
commit scan:
  proportional to emitted one-hot facts, not padded hypercube size.

setup:
  depends on Akita D_setup.
  backend-specific setup padding must be reported separately.

memory:
  no dense W_pack materialization.

dimension table:
  base D values are n + 13.
  non-target split increment model is n + 14.
  fused increments are n + 13.

log_T = 30 review target:
  fused large-trace target has D_pack = 43.
  non-target split increment model has D_pack = 44.
  if Akita buckets setup by D, D_pack = 44 may be the next setup-memory bucket.
  jolt-akita must report the measured D_setup memory mapping.
```

Rejected:

```text
- representing bit facts as byte facts by default.
- adding PackedWitness families without explicit dimension accounting.
- materializing padding.
- hiding Akita backend padding inside Jolt packed-cell accounting.
- changing fact family order based on prover witness content.
```

## Resolved Decisions And Open Questions

```text
resolved:
  the canonical layout encoding is the jolt-akita PackedWitnessLayout digest.
  family order and offsets are deterministically derived from sorted family
  specs, domains, limbs, and alphabets.
  fact family offsets are derived from config/preprocessing, not supplied as
  independent proof-header data.
  D_pack = ceil_log2(cells) for the packed layout. jolt-akita accepts exact-D
  setup for the current target; setup D must match D_pack.
  no additional public alignment rule is exposed beyond the derived layout
  dimension. The adapter chooses dense or sparse native Akita commitment inputs
  internally.

open:
  whether a future universal/up-to-D Akita setup should be accepted.
```

## References

```text
- ../jolt-verifier-model-crate.md: modular verifier context.
- ../../recursion-paper/techniques.tex: prefix packing technique.
- 02-jolt-akita-crate.md: Akita setup dimension.
```
