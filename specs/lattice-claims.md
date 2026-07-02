# Spec: Lattice (Akita) Claims

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Claude |
| Created | 2026-07-02 |
| Status | draft |
| PR | TBD |

## Purpose

Akita is a lattice PCS: it commits to small-norm coefficient vectors and has no
commitment homomorphism. Two consequences for Jolt:

1. **No RLC of commitments.** The stage-8 discharge (an RLC batch over the
   separate `RamInc`/`RdInc`/`Ra`/advice commitments, each lifted to a unified
   point by an embedding scale) cannot exist. Instead, all committed columns
   live in **one packed witness polynomial `W`** per commitment lifecycle, and
   one opening of `W` discharges every logical claim via the prefix-packing
   reduction (`jolt-openings::PrefixPacking`).
2. **0/1 cells, for efficiency.** Committing 0/1 vectors is where Akita is
   fast, so the packed witness is kept one-hot/boolean throughout — this is a
   performance choice, not a norm-bound requirement. Every committed column
   that is not already one-hot gets a one-hot encoding: the dense signed
   `RdInc`/`RamInc` columns become a **fused, shifted, one-hot chunk
   decomposition**, and all unstructured committed data (trusted/untrusted
   advice, program image, bytecode lanes) is byte/symbol one-hotted. The
   one-hot *validity* relations double as the range checks the decode views
   rely on (byte one-hot ⇒ each decoded byte < 256).

This spec defines the `jolt-claims` surface for both: the canonical packed
witness description, and the additional symbolic relations that make the
one-hot inc representation sound. It describes **additional semantics on top of
the base `jolt/` PIOP** — not a new PIOP. `jolt-verifier` has no lattice
implementation yet; this crate-level design is written so that the later
verifier integration is mechanical (`ConcreteSumcheck` impls + a stage-schedule
swap + a packed-opening stage).

The previous integration attempt is preserved verbatim as `jolt-claims-ref`.
Its **semantics** (the fused-inc chain, the shared-final-point trick) are the
reference; its **structure** (a parallel packing model, untyped
`Lattice{relation, index}` openings, validity-requirement descriptors, digest
machinery) is explicitly rejected here.

## Scope

V1 scope (full prototype parity):

```text
canonical packed-witness description (per-proof + precommitted), consumed by
    jolt-openings::PrefixPacking — jolt-claims defines ids + arities only
inc fusion (RamInc/RdInc -> one Inc stream selected by the bytecode Store flag)
base-2^b one-hot chunk decomposition + msb column (the +2^64 unsigned shift is
    a constant, folded into the chunk reconstruction)
booleanity + hamming-weight coverage of the new one-hot columns
chunk reconstruction relation tying decoded chunks back to the fused Inc claim
store-selector binding to the bytecode Store circuit flag
advice byte one-hot decomposition + its validity relations (untrusted advice)
bytecode sub-column decomposition (register selectors, circuit/instruction
    flags, lookup selector, raf flag, unexpanded-pc bytes, imm bytes) and the
    lane decode-weight views reconstructing BytecodeChunk lane values
program-image byte decomposition
decode-weight views (pure (column, weight) term lists) for every decomposed
    logical polynomial
final-opening discharge map (packed claim | decoded view | virtualized)
```

Out of scope (deferred):

```text
field_inline lattice support (FieldRdInc byte decomposition)
ZK/BlindFold composition with lattice mode
jolt-verifier / jolt-witness / prover implementation
```

All committed unstructured data is one-hot encoded (efficiency: Akita commits
0/1 vectors fast). Where the one-hot *structure* is proven splits by trust:
precommitted public columns (bytecode sub-columns, program image) get their
structural validity (one-hot shape, store/rd disjointness, canonical imm
bytes) checked **offline at preprocessing** — the verifier holds the public
bytecode and program image, so no in-protocol relation is spent on them.
In-protocol validity relations cover prover-supplied columns: the inc
chunks/msb (via the lattice booleanity + reconstruction relations) and
untrusted advice bytes (via dedicated byte-validity relations, which are also
the range checks for the decoded words). Trusted advice is one-hot encoded
like everything else but precommitted by a party the verifier trusts, so its
encoding is not re-proven in-protocol.

## Boundary Contract

| Crate | Owns | Must NOT contain |
|-------|------|------------------|
| `jolt-claims` | ids, arities, symbolic relations, discharge map, decode weights | slot assignment, offsets, digests, transcripts, witnesses, PCS |
| `jolt-openings` | `PrefixPacking` slot assignment, suffix-compat check, eq-prefix reduction, transcript binding | Jolt-specific ids or protocol semantics |
| `jolt-verifier` (future) | `ConcreteSumcheck` impls, stage schedule, packed-opening stage | relation algebra (sourced from `jolt-claims`) |
| `jolt-witness`/prover (future) | packed witness materialization | — |
| `jolt-akita` | PCS transport | — |

`jolt-claims` gains **no dependency** on `jolt-openings`. The packing
description is emitted as `Vec<(Id, usize)>` pairs;
`jolt-openings::PackedPolynomial<Id>` already implements `From<(Id, usize)>`,
so `PrefixPacking::new(lattice::proof_packed_columns(&shape))` works directly.
(A dev-dependency on `jolt-openings` is used for round-trip tests only.)

## Module Layout

```text
crates/jolt-claims/src/protocols/jolt/lattice/
├── mod.rs          re-exports; module doc stating the boundary contract
├── ids.rs          per-relation Challenge/Public leaf enums (aggregated into
│                   the jolt/ids.rs enums as appended variants)
├── geometry.rs     inc chunking (width, count, place values), byte-limb
│                   counts, packed-column arity derivation; pure functions
├── packing.rs      proof_packed_columns(..) / precommitted_packed_columns(..)
│                   -> Vec<(LatticeColumn, usize)>
├── views.rs        decode-weight term lists: bytecode-chunk lane decode,
│                   little-endian byte decode — pure functions
│                   -> Vec<DecodeTerm<F>> (the inc lower-value decode lives in
│                   the reconstruction relation, not a view)
├── discharge.rs    LatticeFinalOpening map:
│                   leaf opening -> packed claim | decoded view | virtualized
└── relations/
    ├── inc_virtualization.rs
    ├── chunk_reconstruction.rs
    ├── booleanity.rs            lattice-mode Booleanity (same relation id)
    └── advice_bytes.rs          untrusted-advice byte validity relation
```

There is no store-binding relation file: the fused-inc destination selector is
the existing `OpFlags(Store)` virtual polynomial, so its opening discharges
through the bytecode read-raf val stages like every other flag consumer (see
relation 1 below).

Placement rationale: the lattice relations are pinned to the Jolt id families
(`ConcreteSumcheck` fixes `RelationId = JoltRelationId`, `OpeningId =
JoltOpeningId`, …), and they consume base-PIOP openings directly. That makes
lattice a **mode of the jolt protocol** (like the committed-bytecode mode), not
a sibling protocol like `field_inline/` (which has fully parallel id families).
Hence `protocols/jolt/lattice/`, not `protocols/lattice/`.

## Id-Space Extension

Minimal, append-only additions to `protocols/jolt/ids.rs` (append-only keeps
index-based serde codecs stable for existing proofs):

```rust
// JoltRelationId — appended variants
IncVirtualization,
UnsignedIncChunkReconstruction,
AdviceBytesValidity,

// JoltCommittedPolynomial — appended variants (committed only in lattice mode,
// as slots of the packed witness; base mode never constructs them)
UnsignedIncChunk(usize),   // one-hot column j of the fused unsigned inc
UnsignedIncMsb,            // boolean msb column
TrustedAdviceBytes,        // byte one-hot advice encodings
UntrustedAdviceBytes,

// JoltVirtualPolynomial — appended variant
FusedInc,                  // gamma-batched RamInc/RdInc stream; its selector
                           // is the existing OpFlags(Store), its +2^64 shift
                           // is folded into the chunk reconstruction
```

WARNING: enum `Ord` is protocol data too — `PrefixPacking` assigns slots by
`(arity, Id)` order, so *reordering* variants (not just inserting) changes the
packed witness layout silently.

Per-relation leaf enums live in `lattice/ids.rs` and are aggregated as
appended `JoltChallengeId`/`JoltDerivedId` variants:

```rust
IncVirtualizationChallenge { Gamma }
IncVirtualizationPublic    { EqRamReadWrite, EqRamValCheck,
                             EqRegistersReadWrite, EqRegistersValEvaluation }
UnsignedIncChunkReconstructionChallenge { Gamma }
UnsignedIncChunkReconstructionPublic    { EqBooleanityAddress, IdentityAtAddress }
```

There is **no** `JoltOpeningId::Lattice { relation, index }` variant. Every
lattice opening is an ordinary `(polynomial, producing-relation)` pair; the
prototype's flat-index ids obscured which polynomial an opening referred to and
allowed the same id to denote claims at two different points.

## Packed Witness Description

The packing id is `LatticeColumn`: a column with in-protocol openings is a
(lattice-mode) committed polynomial wrapped as `Committed(..)`; a column that
is only ever reached through a decode view (precommitted sub-columns) gets its
own variant and **no** `JoltCommittedPolynomial` identity:

```rust
pub enum LatticeColumn {
    /// Has in-protocol openings: Ra families, inc chunks/msb, advice bytes.
    Committed(JoltCommittedPolynomial),
    /// Precommitted bytecode sub-columns; decode-view access only.
    BytecodeRegisterSelector { chunk: usize, selector: usize }, // rs1/rs2/rd
    BytecodeCircuitFlag { chunk: usize, flag: usize },
    BytecodeInstructionFlag { chunk: usize, flag: usize },
    BytecodeLookupSelector { chunk: usize },
    BytecodeRafFlag { chunk: usize },
    BytecodeUnexpandedPcBytes { chunk: usize },
    BytecodeImmBytes { chunk: usize },
    /// Program image words as byte one-hot cells; decode-view access only.
    ProgramImageBytes,
}
```

Two commitment lifecycles, two packings:

```rust
/// Per-proof packed commitment: trace/advice-domain columns.
pub fn proof_packed_columns(shape) -> Vec<(LatticeColumn, usize)>;
// InstructionRa(0..I), BytecodeRa(0..B), RamRa(0..R)   log_k_chunk + log_T each
// UnsignedIncChunk(0..N)                               log_k_chunk + log_T each
//                       (chunk width b = log_k_chunk, N = 64 / b; see invariant)
// UnsignedIncMsb                                       log_T
// UntrustedAdviceBytes                                 8 + 3 + advice_vars

/// Preprocessing-time packed commitment (committed-program mode).
pub fn precommitted_packed_columns(shape) -> Vec<(LatticeColumn, usize)>;
// per chunk c:
//   BytecodeRegisterSelector{c, 0..3}    log2(register_count) + log_bc each
//   BytecodeCircuitFlag{c, 0..NUM_CIRCUIT_FLAGS}          log_bc each (0/1 col)
//   BytecodeInstructionFlag{c, 0..NUM_INSTRUCTION_FLAGS}  log_bc each (0/1 col)
//   BytecodeLookupSelector{c}      log2(next_pow2(TABLE_COUNT)) + log_bc
//   BytecodeRafFlag{c}                                    log_bc  (0/1 col)
//   BytecodeUnexpandedPcBytes{c}                      8 + 3 + log_bc
//   BytecodeImmBytes{c}       8 + ceil_log2(field_byte_width) + log_bc
// ProgramImageBytes                                   8 + 3 + log_words
// TrustedAdvice bytes (Committed(TrustedAdvice) byte encoding, if present)
```

Conventions:

- **Cell order within a column**: `(symbol_bits ‖ limb_bits ‖ row_bits)`,
  high-to-low, so the row coordinates are always the point's suffix (this is
  what makes suffix-compatible packed claims line up on the shared row point).
- **Boolean flag columns** are plain 0/1 columns of arity `log_rows` — *not*
  the prototype's 2-symbol one-hot encoding. `1 − flag` is expression algebra;
  spending a second cell row on it doubles cells for nothing.
- **Non-power-of-two limb counts** (imm bytes) round the limb dimension up;
  dummy limb cells are zero by convention (checked offline with the rest of the
  precommitted validity).

`PrefixPacking::new` sorts by descending arity then id and assigns
power-of-two-aligned prefix slots, so `jolt-claims` performs **no offset
arithmetic, no digesting, no dummy-cell accounting** — all of that (including
transcript binding of the packing) is `jolt-openings`' job. The prototype's
`PackingWitnessLayout`/`PackingFamilySpec`/`PackingFactDomain`/
`PackingAlphabet`/`PackingFamilyId` model (~600 lines + digests) is deleted
with no replacement.

## Decode Views

A decomposed logical value is a **weighted sum** over a column's non-row
cells, not a plain evaluation — e.g. an advice word is
`Σ_limb 256^limb · Σ_sym sym · Bytes(sym ‖ limb ‖ row)`. Weighted sums with
non-`eq` weights are not single `PrefixPackedClaim`s, so every decomposed
logical polynomial carries a **view**: a pure description of the weighted
combination. The verifier discharges a view with a reduction sumcheck over the
cell variables (standalone or folded into the logical polynomial's existing
claim-reduction relation — the weights are per-variable multilinear, so they
sumcheck cleanly), producing the single per-column evaluations the packed
statement needs. Aligned `eq`-weighted blocks (the register-selector lanes)
collapse without extra rounds. Either way, the term list is the semantics both
sides must agree on.

```rust
pub struct DecodeTerm<F> {
    pub column: LatticeColumn,
    /// Index into the column's (symbol ‖ limb) cell prefix, msb-first.
    pub cell: usize,
    pub weight: F,
}
```

`views.rs` provides the term builders (all pure functions):

- `byte_decode_terms(column, limbs)` — little-endian `256^limb · sym` weights.
- `bytecode_chunk_decode_terms(chunk, lane_point)` — reconstructs the
  committed `BytecodeChunk(chunk)` lane value: register-selector cells
  weighted by the lane eq-evals, flag columns by their lane weight, pc/imm
  byte cells by `lane_weight · 256^limb · sym`. Weight computation from a
  point is point algebra, which already lives in `geometry/`
  (`commitment_embedding_scale` precedent).
- `unsigned_inc_lower_value_terms(b)` — `radix^j · sym` weights across chunk
  columns (used by the reconstruction relation's `IdentityAtAddress` leg).
- `advice_word_decode_terms(kind, ..)`, `program_image_decode_terms(..)`.

## The Inc One-Hotting Relation Chain

Committed cells must be 0/1; the signed dense `RdInc`, `RamInc` columns are
replaced by one fused set of one-hot columns. Semantics ported from the
prototype; every relation below is a `SymbolicSumcheck` impl with typed
`#[derive(InputClaims/OutputClaims/SumcheckChallenges)]` structs, exactly like
the existing `relations/**` modules.

### 1. `IncVirtualization` (replaces `IncClaimReduction` in lattice mode)

A cycle writes RAM (store) or a register, never both, so one inc stream
suffices — this halves the one-hot column count ("fused polys").

- **Inputs** (same four consumed claims as base `IncClaimReduction`):
  `RamInc@RamReadWriteChecking`, `RamInc@RamValCheck`,
  `RdInc@RegistersReadWriteChecking`, `RdInc@RegistersValEvaluation`.
- **Outputs**: `FusedInc` (virtual) and `OpFlags(Store)`, both at the bound
  cycle point. Using the existing store circuit-flag polynomial as the
  selector means its opening is bound to the actual bytecode by the same
  read-raf val-stage machinery that discharges every other flag consumer
  (`LATTICE_BYTECODE_VAL_STAGES = NUM_BYTECODE_VAL_STAGES + 1` names the extra
  stage) — no dedicated store-binding relation exists.
- **Input expr**: `ram_rw + γ·ram_val + γ²·rd_rw + γ³·rd_val`.
- **Output expr**:
  `FusedInc · (ram_coeff·store + γ²·rd_coeff·(1 − store))` where
  `ram_coeff = EqRamReadWrite + γ·EqRamValCheck`,
  `rd_coeff = EqRegistersReadWrite + γ·EqRegistersValEvaluation` (deriveds).
- degree 3, `log_T` rounds.

There is no separate unsigned-shift relation: `+2^64` is a constant, hence
free at any opening point, and is folded into the reconstruction's value leg
below. (The prototype spent a `log_T`-round sumcheck on it, with no eq
factor in its output — structurally unsound as written; dropped.)

### 2. Lattice `Booleanity` (same `JoltRelationId::Booleanity`)

Same relation id, lattice-mode shape: the base output sum over `Ra` columns is
extended with the `UnsignedIncChunk(0..N)` columns and the `UnsignedIncMsb`
booleanity term. Precedent: the full/committed bytecode modes already share
relation ids across variant sumcheck structs. The base
`relations/booleanity/*` files stay untouched; the lattice variant reuses the
base geometry helpers. Produces chunk openings at the booleanity
`(r_address, r_cycle)` point.

### 3. `UnsignedIncChunkReconstruction`

One sumcheck over the `b` address bits of a chunk, γ-batching three duties:

- **hamming weight**: each chunk column sums to exactly 1 per cycle row
  (the `γ^{2i}` constant legs),
- **claim reduction**: reduces the chunk openings produced by lattice
  Booleanity to the reconstruction's bound address point (the `γ^{2i+1}` legs
  against the `EqBooleanityAddress` derived),
- **value reconstruction**: `Σ_j place_j · decode(chunk_j) = FusedInc +
  2^64·(1 − msb)` — the folded unsigned shift (the `δ = γ^{2N}` leg against
  the `IdentityAtAddress` derived).

- **Inputs**: `FusedInc@IncVirtualization`, `UnsignedIncMsb@Booleanity`,
  `UnsignedIncChunk(j)@Booleanity`.
- **Outputs**: `UnsignedIncChunk(j)` at the final shared address point.
- degree 2, `b` rounds. Draws `Gamma`. (Per bound variable every leg is at
  most a product of two multilinears — `eq·chunk`, `id·chunk` — so the round
  polynomials are quadratic; the prototype's degree 3 was slack.)

The prototype reused one flat opening id for a chunk's claims at *two
different points* (booleanity-bound and reconstruction-bound). The typed
`from =` attributes make that distinction structural here.

### 4. Store binding (via the existing flag plumbing)

The selector must equal the bytecode `Store` circuit-flag column, else a
malicious prover could route increments to the wrong destination. Because the
selector *is* `OpFlags(Store)`, the binding is the ordinary flag-opening
discharge: the lattice-mode bytecode read-raf batch grows one val stage
(`LATTICE_BYTECODE_VAL_STAGES`) carrying the `OpFlags(Store)@IncVirtualization`
claim at its cycle point, exactly as the spartan-outer/shift flag claims enter
(this is what the prototype's `bind_store` sixth stage was). The offline
store/rd-disjointness check on the public bytecode (`Store` and rd-writing
selectors never co-occur) completes the argument that per cycle exactly one of
RAM/registers receives the fused increment.

### 5. Untrusted advice byte validity (`AdviceBytesValidity`)

The untrusted advice byte column is prover-supplied, so its one-hot structure
is proven in-protocol; this is simultaneously the range check that makes the
advice decode view sound (each decoded byte < 256, each word < 2^64). One
γ-batched sumcheck over the column's `(symbol ‖ limb ‖ word)` domain carrying
the booleanity leg (`cell² − cell`, `EqCell` derived) and the per-(limb, word)
hamming leg (`Σ_sym cell = 1`, `EqLimbWord` derived; claimed input sum `γ`),
producing the byte-column opening that the packed statement consumes.

Integration note: a packed slot admits exactly one claim, so the advice
*decode* (the word claim's reduction onto the byte column) must share this
relation's binding — at verifier-integration time the decode leg joins this
sumcheck as a third γ-leg consuming the advice word claim (weights
`id(symbol) · 256^limb · eq(word)` are multilinear per variable, so it
sumchecks cleanly), rather than producing a second byte-column opening at a
different point.

## Shared-Final-Point Invariant and Discharge

`PrefixPacking` opens `W` at **one** packed point; every logical claim must be
**suffix-compatible** with it (`jolt-openings` errors otherwise — the invariant
is machine-checked). Equal-arity slots occupy the same suffix window of the
packed point, so **equal-arity columns must land their final claims on the
same full point** — not just the same row tail. The relation chain exists
precisely to arrange this:

- Hamming-weight claim reduction already lands every `Ra` column on one shared
  `(r_address ‖ r_cycle)` point.
- The inc chunk width is **fixed at `b = log_k_chunk`** so the chunk columns
  sit in the `Ra` arity class, and the reconstruction relation binds them to
  that same shared point (`EqBooleanityAddress` leg). This is the prototype's
  "optimized packing" made explicit as an invariant.
- `UnsignedIncMsb` (arity `log_T`) claims at the shared `r_cycle` tail.
- Advice/byte columns live in other arity classes; their reductions must land
  each class on one point whose overlap with the trace tail agrees (the same
  anchoring the base advice reduction already performs for the unified point).

**Who arranges this**: `jolt-claims` only names the point equalities; the
relations carry no stage assignments. `jolt-verifier` owns batching, staging,
and constraint ordering, and is responsible for co-batching the lattice
relations so their bound points coincide where the invariant demands (e.g.
`IncVirtualization`'s cycle point with the booleanity cycle tail, the
reconstruction's address rounds with the hamming-weight reduction's).

**Padding invariant**: the hamming legs claim "exactly one hot cell per row"
summed over the *full* Boolean hypercube, so padding rows must be encoded as
the one-hot of value 0 — never as all-zero cells. This applies to inc-chunk
rows beyond the trace length and advice byte positions beyond the actual
advice size (and, by the offline checks, to padded bytecode/program-image
rows). All-zero padding would falsify the claimed input sums (`1` per chunk
row, `γ` for the advice column).

`lattice/discharge.rs` is the single source of truth for the endgame:

```rust
pub enum LatticeFinalOpening {
    /// Becomes one PrefixPackedClaim on the per-proof / precommitted W.
    Packed { column: LatticeColumn, leaf: JoltOpeningId },
    /// A decode view: the logical claim equals a weighted sum of packed
    /// cells; discharged via the view's terms (see Decode Views).
    Decoded { view: LatticeView },
    /// Never PCS-opened: consumed entirely by lattice relations.
    Virtualized,
}

pub fn final_opening(polynomial: JoltCommittedPolynomial) -> LatticeFinalOpening;
// RdInc, RamInc                  -> Virtualized (via IncVirtualization chain)
// InstructionRa/BytecodeRa/RamRa -> Packed, leaf = HammingWeightClaimReduction output
// UnsignedIncChunk(j)            -> Packed, leaf = UnsignedIncChunkReconstruction output
// UnsignedIncMsb                 -> Packed, leaf = Booleanity output
// TrustedAdvice/UntrustedAdvice  -> Decoded (advice byte view)
// BytecodeChunk(i)               -> Decoded (lane decode view over sub-columns)
// ProgramImageInit               -> Decoded (program image byte view)
```

This replaces both the prototype's `final_opening_lattice_requirement` and the
base stage-8 RLC order for lattice mode.

## Verifier Integration Sketch (future work, not this change)

- `JoltProtocolConfig` gains a commitment-mode axis (`Homomorphic` vs
  `Packed`/lattice), mirroring how `ZkConfig` gates BlindFold.
- Stage schedule swaps, all at existing seams: `IncClaimReduction` →
  `IncVirtualization` (+ chain), base `Booleanity` → lattice `Booleanity`,
  bytecode read-raf → store-binding variants, stage-8 RLC batch →
  `PrefixPackedStatement` opening through `jolt-openings` + `jolt-akita`.
- Each lattice relation gets a `ConcreteSumcheck` impl (deriveds: the eq/lt
  and identity evaluations), identical in shape to existing stage relations.

Nothing in `jolt-claims` presumes any of this beyond the `SymbolicSumcheck`
contract.

## Testing Strategy

- Per-relation algebra tests in the existing house style: evaluate
  `input_expression`/`output_expression` against distinct prime values and
  assert the closed-form formula; assert `required_openings/deriveds/
  challenges` sets and opening order.
- Reconstruction identity test: for random u64 values, `Σ place_j ·
  decode(chunk_j) + 2^64·msb − 2^64` round-trips the signed value.
- Packing round-trip test (dev-dependency on `jolt-openings`):
  `PrefixPacking::new(proof_packed_columns(shape))` succeeds, arity classes
  align, and every `discharge::Packed` column has a slot.
- Determinism: packed-column lists are a pure function of shape (golden test).

## Dropped From the Prototype (jolt-claims-ref) and Why

| Prototype construct | Fate | Reason |
|---|---|---|
| `PackingWitnessLayout`/`FamilySpec`/`FactDomain`/`Alphabet`, offsets, digests | deleted | duplicate of `jolt-openings::PrefixPacking`; slot math is transport, not semantics |
| `PackingFamilyId` (namespace/u64/u64 + bit-packed indices) | deleted | the typed `LatticeColumn` enum is the packing id |
| `JoltOpeningId::Lattice { relation, index }` | deleted | untyped; hid polynomial identity and point identity |
| `PackingValidityRequirement`/`PackingValidityKind` + digest | deleted | validity is not metadata; it is the validity **relations** (prover-supplied columns) or an **offline preprocessing check** (public precommitted columns) |
| `PackingViewFormula`/`PackingViewTerm` | restructured | kept as `DecodeTerm` lists from pure builders; `Direct`/`MaskedDecoded` variants and per-view `Proven/Unchecked` tags dropped (provenness follows from the column's validity story) |
| 2-symbol boolean-flag encoding | replaced | plain 0/1 columns of arity `log_rows`; `1 − flag` is expression algebra, and the packed witness stays 0/1 |
| bytecode store-rd-disjoint, canonical imm bytes, sub-column one-hot validity | moved offline | public precommitted data; the verifier checks structure at preprocessing instead of spending relations |
| Public→Challenge enum migration (EqCycle etc.) | not ported | orthogonal to lattice; main's Derived model already covers it |
| eq-polynomial / coefficient materialization helpers | not ported | belongs to verifier/witness crates per the boundary contract |

## Open Questions

1. **Advice byte-column granularity**: one column per advice kind over
   `(symbol ‖ limb ‖ word)` (spec'd here, matching the prototype) vs one
   column per byte limb (8 smaller columns). Per-limb columns shrink each
   slot's arity but multiply slot count; revisit if the packed dimension or
   the validity sumcheck shape prefers one over the other at integration time.
2. **Precommitted trusted-advice openings**: whether the trusted-advice byte
   view discharges against the precommitted `W'` at the bytecode-domain point
   or anchors into the per-proof point set, mirroring the base
   `PrecommittedClaimReduction` scheduling. Decide with the verifier stage
   design; the view terms are identical either way.
