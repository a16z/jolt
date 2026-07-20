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

1. **No RLC of commitments.** The stage-8 final-opening step over homomorphic
   commitments cannot be reused. The per-proof `W_jolt` commitment is therefore
   one native Akita commitment group: every member is a strict `K x T` one-hot
   polynomial, all members open at one canonical point, and one native Akita
   batch proof settles the group. Advice and committed-program objects have
   different domains and remain auxiliary `PrefixPacking` reductions.
2. **0/1 cells, for efficiency.** Committing 0/1 vectors is where Akita is
   fast, so Wjolt and the auxiliary objects stay one-hot/boolean throughout — this is a
   performance choice, not a norm-bound requirement. Every committed column
   that is not already one-hot gets a one-hot encoding: the dense signed
   `RdInc`/`RamInc` columns become a **fused, shifted, one-hot chunk
   decomposition**, and all unstructured committed data (trusted/untrusted
   advice, program image, bytecode lanes) is byte one-hotted. The
   one-hot *validity* relations double as the range checks the reconstructions
   rely on (byte one-hot ⇒ each reconstructed byte < 256).

This spec defines the protocol cutover on top of the base `jolt/` PIOP: the
canonical commitment layout, the additional relations making the fused
increment representation sound, and the verifier/prover stage schedule.

The previous integration attempt was snapshotted as the `jolt-claims-ref`
crate (commit `16727bef9`; deleted from the workspace once this module reached
parity — recover it from git history if needed). Its **semantics** (the
fused-inc chain, the shared-final-point trick) were the reference; its
**structure** (a parallel packing model, untyped `Lattice{relation, index}`
openings, validity-requirement descriptors, digest machinery) is explicitly
rejected here.

## Scope

V1 scope (full prototype parity):

```text
canonical native one-hot Wjolt layout plus auxiliary prefix-packed objects
inc fusion (RamInc/RdInc -> one Inc stream selected by the bytecode Store flag)
base-2^b one-hot chunk decomposition + full one-hot msb column
four fused-inc consumer val stages inside the bytecode read-RAF, discharging
    the reduced inc claims and producing the FusedInc opening at the shared
    stage-6b cycle point
booleanity + hamming-weight coverage of the new one-hot columns
fused-increment decode folded into stage 7 HammingWeightClaimReduction
store-selector binding to the bytecode Store circuit flag
advice byte one-hot decomposition + its reconstruction relations (untrusted:
    validity + word-decode legs; trusted: word-decode leg)
bytecode sub-column decomposition (register selectors, circuit/instruction
    flags, lookup selector, raf flag, unexpanded-pc bytes, imm bytes) and the
    BytecodeChunkReconstruction relation rebuilding chunk lane values
program-image byte decomposition + its reconstruction relation
decode-weight point algebra (the relations' derived semantics)
final-opening map (native Wjolt claim | auxiliary packed claim | virtualized)
```

Out of scope (deferred):

```text
field_inline lattice support (FieldRdInc byte decomposition)
ZK/BlindFold composition with lattice mode
```

All committed unstructured data is one-hot encoded (efficiency: Akita commits
0/1 vectors fast). Where the one-hot *structure* is proven splits by trust:
precommitted public columns (bytecode sub-columns, program image) get their
structural validity (one-hot shape, store/rd disjointness, canonical imm
bytes) checked **offline at preprocessing** — the verifier holds the public
bytecode and program image, so no in-protocol relation is spent on them.
In-protocol validity relations cover prover-supplied columns: the inc
chunks/msb (via lattice Booleanity and HammingWeightClaimReduction) and
untrusted advice bytes (via dedicated byte-validity relations, which are also
the range checks for the reconstructed words). Trusted advice is one-hot encoded
like everything else but precommitted by a party the verifier trusts, so its
encoding is not re-proven in-protocol.

## Boundary Contract

| Crate | Owns | Must NOT contain |
|-------|------|------------------|
| `jolt-claims` | ids, arities, symbolic relations, final-opening map, canonical Wjolt member order and layout digest | transcripts, witnesses, PCS |
| `jolt-openings` | `PrefixPacking` slot assignment, suffix-compat check, eq-prefix reduction, transcript binding | Jolt-specific ids or protocol semantics |
| `jolt-verifier` | `ConcreteSumcheck` impls, stage schedule, native Wjolt opening and auxiliary packed openings | relation algebra (sourced from `jolt-claims`) |
| `jolt-witness`/prover | one-hot witness materialization and matching stage schedule | — |
| `jolt-akita` | PCS transport | — |

`jolt-claims` owns the ordered Wjolt member list and hashes the order, member
identities, and dimensions into a nonzero setup digest. The proof never chooses
this digest. `jolt-openings::PrefixPacking` remains the source of truth only for
the auxiliary advice and committed-program objects.

## Module Layout

```text
crates/jolt-claims/src/protocols/jolt/lattice/
├── mod.rs          re-exports; module doc stating the boundary contract
│                   and the inherited vocabulary (jolt-openings' logical
│                   polynomial / slot / prefix; per-family dimension names)
├── geometry.rs     inc chunking (width, count, place values), byte one-hot
│                   variable counts, and the decode-weight point algebra
│                   (byte_decode_weight, selector_block_weight — the semantic
│                   definitions of the reconstruction relations' deriveds,
│                   composed from jolt-poly's IdentityPolynomial /
│                   eq_index_msb primitives); pure functions
├── packing.rs      wjolt_members(..) -> canonical native member order;
│                   auxiliary precommitted/advice PrefixPacking registration
├── strategy.rs     the one native Wjolt layout, common-point permutation,
│                   setup shape, and canonical nonzero layout digest
└── relations/
    ├── read_raf.rs                      lattice bytecode read-RAF (address
    │                                    fold over the four inc claims; cycle
    │                                    phases carrying the FusedInc factor)
    ├── hamming_weight.rs                stage-7 fused increment reduction
    ├── booleanity.rs                    lattice-mode Booleanity (same id)
    ├── advice_reconstruction.rs         untrusted (validity + decode legs)
    │                                    and trusted (decode leg) advice
    ├── bytecode_reconstruction.rs       chunk -> per-lane polynomial decode
    └── program_image_reconstruction.rs  word -> byte one-hot decode
```

There is no store-binding relation file and no store-selector opening at all:
the fused-inc consumer stages substitute the store column into the read-raf
fold directly (see relation 1 below), so the selector never leaves the
bytecode val-stage machinery.

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
UntrustedAdviceReconstruction,   // byte validity + word-decode legs
TrustedAdviceReconstruction,     // word-decode leg only
ProgramImageReconstruction,      // word-decode leg only
BytecodeChunkReconstruction,     // chunk -> lane sub-column decode

// JoltCommittedPolynomial — appended variants (committed only in lattice mode,
// as native Wjolt members or auxiliary packed columns; base mode never
// constructs them)
UnsignedIncChunk(usize),   // one-hot column j of the fused unsigned inc
UnsignedIncMsb,            // full K x T one-hot column, hot address 0 or 1
TrustedAdviceBytes,        // byte one-hot advice encodings
UntrustedAdviceBytes,
// Precommitted bytecode sub-columns + program image bytes. These are real
// committed polynomials (their claims are produced by the reconstruction
// relations and consumed by the packed opening), so they carry polynomial
// identities — there is no separate LatticeColumn id family.
BytecodeRegisterSelector { chunk, lane },   // one-hot rs1/rs2/rd selectors
BytecodeCircuitFlag { chunk, flag },        // plain 0/1 columns
BytecodeInstructionFlag { chunk, flag },
BytecodeLookupSelector { chunk },           // one-hot table selector
BytecodeRafFlag { chunk },
BytecodeUnexpandedPcBytes { chunk },        // byte one-hot lanes
BytecodeImmBytes { chunk },
ProgramImageBytes,

// JoltVirtualPolynomial — appended variant
FusedInc,                  // gamma-batched RamInc/RdInc stream; opened once,
                           // by the read-raf cycle phase at the shared 6b
                           // cycle point; its +2^64 shift is folded into the
                           // stage-7 hamming reduction
```

WARNING: enum `Ord` is protocol data for auxiliary objects — `PrefixPacking`
assigns slots by `(num_vars, Id)` order. Wjolt does not use this ordering: its
member order is constructed explicitly by `wjolt_members` and hashed into its
canonical layout digest.

Per-relation challenge/public sub-enums live in `protocols/jolt/ids.rs` next to every other
relation's (house convention) and are aggregated as appended
`JoltChallengeId`/`JoltDerivedId` variants:

```rust
BytecodeReadRafPublic::{StageValue, StageCycleEq} index the relation's val
    stages 0..READ_RAF_CYCLE_STAGES (9 in lattice mode; see relation 1)
HammingWeightClaimReductionPublic additionally includes IdentityAtAddress
```

There is **no** `JoltOpeningId::Lattice { relation, index }` variant. Every
lattice opening is an ordinary `(polynomial, producing-relation)` pair; the
prototype's flat-index ids obscured which polynomial an opening referred to and
allowed the same id to denote claims at two different points.

## Commitment Layout

Every native Wjolt member and every auxiliary packed column is identified by
`JoltCommittedPolynomial`. The earlier draft's separate `LatticeColumn` id
family is gone. Wjolt has one relation-produced claim per member at a shared
point; auxiliary columns have one claim per prefix-packed slot.

Wjolt is one native Akita group; the helper below supplies its canonical
ordered member list. Auxiliary objects use separate prefix packings:

```rust
/// Canonical per-proof Wjolt member order.
pub fn wjolt_members(shape) -> Vec<JoltCommittedPolynomial>;
// InstructionRa(0..I), BytecodeRa(0..B), RamRa(0..R)   log_k_chunk + log_T each
// UnsignedIncChunk(0..N)                               log_k_chunk + log_T each
//                       (chunk width b = log_k_chunk, N = 64 / b)
// UnsignedIncMsb                                       log_k_chunk + log_T

// Advice columns are separate auxiliary objects, each over
// 8 + 3 + advice_vars variables.

/// Preprocessing-time packed commitment (committed-program mode).
pub fn precommitted_packing(shape) -> PrefixPacking<JoltCommittedPolynomial>;
// per chunk c:
//   BytecodeRegisterSelector{c, rs1/rs2/rd}  log2(register_count) + log_bc each
//   BytecodeCircuitFlag{c, 0..NUM_CIRCUIT_FLAGS}          log_bc each (0/1 col)
//   BytecodeInstructionFlag{c, 0..NUM_INSTRUCTION_FLAGS}  log_bc each (0/1 col)
//   BytecodeLookupSelector{c}      log2(next_pow2(TABLE_COUNT)) + log_bc
//   BytecodeRafFlag{c}                                    log_bc  (0/1 col)
//   BytecodeUnexpandedPcBytes{c}                      8 + 3 + log_bc
//   BytecodeImmBytes{c}       8 + ceil_log2(field_byte_width) + log_bc
// ProgramImageBytes                                   8 + 3 + log_words
// TrustedAdviceBytes (if present)                     8 + 3 + advice_vars
```

Conventions (vocabulary inherited per `lattice/mod.rs`: jolt-openings'
logical polynomial / slot / prefix, plus each family's own dimension names —
`(address ‖ cycle)`, `(byte ‖ place ‖ word)`, `(lane ‖ row)`):

- **Boolean index order**: `(hot-value bits ‖ instance bits)`, msb-first —
  e.g. `(byte ‖ place ‖ word)` for byte one-hots.
- **Auxiliary public bytecode flag columns** are plain 0/1 columns of
  `log_rows` variables —
  *not* the prototype's two-entry one-hot encoding. `1 − flag` is expression
  algebra; spending a second cell row on it doubles cells for nothing.
- **Non-power-of-two byte counts** (imm bytes) round the place dimension up;
  dummy place cells are zero by convention (checked offline with the rest of
  the precommitted validity).

Every Wjolt member has the same arity. Relation leaves use
`(address || cycle)` order; the native Akita members use row-major
`(cycle || address)` order. Stage 8 applies this one permutation to every
member and rejects unless all mapped points are identical. A protocol-owned,
nonzero digest binds the ordered member identities and dimensions into the
Akita setup; the proof never supplies it. `PrefixPacking` continues to assign
slots and reduction statements for auxiliary objects. The prototype's
`PackingWitnessLayout`/`PackingFamilySpec`/`PackingFactDomain`/
`PackingAlphabet`/`PackingFamilyId` model (~600 lines + digests) is deleted
with no replacement.

## Packed-Column Reconstructions (relations 5–8 below)

A decomposed logical value is a **weighted sum** over a column's non-row
cells, not a plain evaluation — e.g. an advice word is
`Σ_place 256^place · Σ_byte byte · Bytes(byte ‖ place ‖ row)`. Weighted sums
with non-`eq` weights are not single packed evaluation claims, so every
decomposed logical polynomial (`TrustedAdvice`, `UntrustedAdvice`,
`BytecodeChunk(i)`, `ProgramImageInit`) is **virtualized**: a reconstruction
*relation* — an ordinary `SymbolicSumcheck` over the cell variables — consumes
the word/chunk claim produced by the base claim reduction and produces the
single per-column packed claims. The weights are per-variable multilinear, so
they sumcheck cleanly; their bound evaluations are the relations' deriveds,
semantically defined by pure point-algebra helpers in `lattice/geometry.rs`:

- `byte_decode_weight(byte_point, place_point)` — the MLE of
  `(byte, place) ↦ value(byte) · 256^place` (the `ByteDecode` deriveds);
  the value half is `jolt-poly`'s `IdentityPolynomial`, the radix half the
  local `place_value_weight`.
- `selector_block_weight(lane_point, block_start, address_point, addresses)` —
  a selector block's lane-eq weights as an address-variable multilinear (the
  `RegisterSelectorWeight`/`LookupSelectorWeight` deriveds).
- The `LaneWeight(lane)` deriveds of the direct 0/1 flag lanes are plain
  `jolt-poly` `eq_index_msb(lane_point, lane)` — no lattice helper.

The earlier draft's `ReconstructionTerm` lists (pure `(column, cell, weight)`
descriptions with the reduction deferred to "the verifier side") are deleted:
the relations' input/output expressions plus these helpers are the semantics
both sides must agree on, and `tests/lattice_semantics.rs` pins the helpers
against brute-force MLE evaluation.

## The Inc One-Hotting Relation Chain

Committed cells must be 0/1; the signed dense `RdInc`, `RamInc` columns are
replaced by one fused set of one-hot columns. Semantics ported from the
prototype; every relation below is a `SymbolicSumcheck` impl with typed
`#[derive(InputClaims/OutputClaims/SumcheckChallenges)]` structs, exactly like
the existing `relations/**` modules.

### 1. Fused-inc consumer stages inside the bytecode read-RAF

A cycle writes RAM (store) or a register, never both, so one inc stream
suffices — this halves the one-hot column count ("fused polys"). The four
reduced inc claims are discharged *inside* the lattice bytecode read-RAF
(`lattice/relations/read_raf.rs`), with no standalone sumcheck and no
store-selector opening. The load-bearing substitution: bytecode `ra` is
one-hot per cycle, so

```text
Store(j)  = Σ_k val_store(k) · RA(k, j)
RamInc(j) = FusedInc(j) · Store(j)
RdInc(j)  = FusedInc(j) · (1 − Store(j))
```

turns each inc claim into a read-raf-shaped val stage — a bytecode val column
(`store` / `¬store`), the producing relation's cycle point, and the shared RA
product — with one shared `FusedInc` cycle factor. The four consumer stages
join the address-phase input fold at `γ^5..8` (the pc/shift/entry terms shift
to `γ^9..11`):

```text
stage 5: (store,     RamInc@RamReadWriteChecking's cycle point)
stage 6: (store,     RamInc@RamValCheck's cycle point)
stage 7: (1 − store, RdInc@RegistersReadWriteChecking's cycle point)
stage 8: (1 − store, RdInc@RegistersValEvaluation's cycle point)
```

All four input claims exist before stage 6a (stages 2/4/5 produce them), so
the fold has no timing constraint. Two counts, easy to conflate, are kept
distinct in code:

- `READ_RAF_CYCLE_STAGES = 9` in lattice mode (5 base flag stages + the 4
  fused consumers): one cycle point / cycle-eq public per relation stage;
- `NUM_BYTECODE_VAL_STAGES = 6` (5 base staged values + the raw `Store`
  column): the committed-mode staged-value *wire* count. The fused stages
  resolve through the store wire and its complement (`1 − store`), so
  committed mode adds no staged-value wires.

The `FusedInc` factor raises the cycle-phase degree by one (`d + 2` over the
base `d + 1`). The cycle phase binds the shared stage-6b batch challenges, so
its `FusedInc` opening lands directly at the 6b cycle point — the point
stage 7 consumes — as the relation's own output claim alongside the
`BytecodeRa` openings.

There is no separate unsigned-shift relation: `+2^64` is a constant, hence
free at any opening point, and is folded into the Stage 7 decode leg below.
There is also no second point-alignment sumcheck: the opening is born at the
shared point.

### 2. Lattice `Booleanity` (same `JoltRelationId::Booleanity`)

Same relation id, lattice-mode shape: the base output sum over `Ra` columns is
extended with the `UnsignedIncChunk(0..N)` columns and the `UnsignedIncMsb`
booleanity term. Precedent: the full/committed bytecode modes already share
relation ids across variant sumcheck structs. The base
`relations/booleanity/*` files stay untouched; the lattice variant reuses the
base geometry helpers. Every increment column, including the MSB, is a full
`K x T` one-hot column and produces an opening at the common
`(r_address, r_cycle)` point. For each cycle the MSB row is `[1,0,...]` or
`[0,1,0,...]`.

### 3. Lattice `HammingWeightClaimReduction` (Stage 7)

The existing RA hamming-weight claim reduction is extended, on the Akita path
only, with the increment columns. One sumcheck over the `b = log_K` address
bits batches:

- **hamming weight**: every increment chunk and the MSB has exactly one hot
  address per cycle row,
- **claim reduction**: reduces the chunk openings produced by lattice
  Booleanity to Stage 7's bound address point, and
- **value reconstruction**:
  `Σ_j 2^(b*j) * address(chunk_j) + 2^64 * address(msb)
   = FusedInc + 2^64`.

- **Inputs**: `FusedInc@BytecodeReadRaf` (the read-raf cycle phase's own
  output, already at the shared 6b cycle point), `UnsignedIncMsb@Booleanity`,
  `UnsignedIncChunk(j)@Booleanity`.
- **Outputs**: every RA, increment chunk, and MSB opening at the same full
  `(r_address, r_cycle)` point.
- degree 2, `b` rounds, using the existing hamming batching challenge.

There is no standalone increment reconstruction and no additional alignment
sumcheck after Stage 7.

### 4. Store binding (via the existing flag plumbing)

The selector must equal the bytecode `Store` circuit-flag column, else a
malicious prover could route increments to the wrong destination. The binding
is structural: the fused consumer stages never open a selector at all — they
substitute `Σ_k val_store(k)·RA(k, j)` (and its complement) directly, where
`val_store` is the bytecode store column itself. The raw `Store` column stays
the sixth staged-value wire in committed-program mode
(`NUM_BYTECODE_VAL_STAGES = 6`), proven against the committed image by the
bytecode claim reduction like the other staged values; in full-program mode
the verifier folds it from the public bytecode. The offline
store/rd-disjointness check on the public bytecode (`Store` and rd-writing
selectors never co-occur) completes the argument that per cycle exactly one of
RAM/registers receives the fused increment.

### 5. `UntrustedAdviceReconstruction` (validity + word virtualization)

The untrusted advice byte column is prover-supplied, so its one-hot structure
is proven in-protocol; this is simultaneously the range check that makes the
word decode sound (each byte < 256, each word < 2^64). One γ-batched sumcheck
over the column's full `(byte ‖ place ‖ word)` domain carrying three legs:

- **booleanity**: `eq(z, r) · (B(z)² − B(z))`, input `0` (`EqBytePlaceWord`
  derived),
- **hamming**: per-(place, word) `Σ_byte cell = 1`, input `1` (`EqPlaceWord`
  derived),
- **decode**: `Σ_cells eq(word, r_advice) · value(byte) · 256^place · cell`,
  input the `untrusted_advice@AdviceClaimReduction` word claim (`ByteDecode`
  and `EqWord` deriveds).

The legs must share one sumcheck: a packed slot admits exactly one claim, so
a standalone validity relation and a standalone decode reduction would each
pin `UntrustedAdviceBytes` at a different point. Inputs
`untrusted_advice@AdviceClaimReduction`; outputs the byte-column opening the
packed statement consumes. degree 3, `8 + 3 + word_vars` rounds. Draws
`Gamma`.

### 6. `TrustedAdviceReconstruction`

Trusted advice shares the byte encoding but is precommitted by a party the
verifier trusts, so no validity legs are spent — only the decode leg, which
is still required in-protocol (a weighted sum over cells is not an
evaluation, and there is no homomorphism to fold it away). Binds the
`(byte ‖ place)` prefix only; the word point is fixed by the incoming claim.
Inputs
`trusted_advice@AdviceClaimReduction`; outputs `TrustedAdviceBytes` at
`(bound prefix ‖ r_word)` (`ByteDecode` derived). degree 2, `8 + 3` rounds,
no challenges.

### 7. `ProgramImageReconstruction`

Identical single-leg shape to trusted advice over the program image byte
column. Inputs `ProgramImageInit@ProgramImageClaimReduction`; outputs
`ProgramImageBytes` at `(bound prefix ‖ r_word)` (`ByteDecode` derived).
degree 2, `8 + 3` rounds, no challenges.

### 8. `BytecodeChunkReconstruction`

Rebuilds every committed `BytecodeChunk(c)` claim (all chunks share the
`BytecodeClaimReduction` terminus point `(r_lane ‖ r_row)`) from the per-lane
sub-columns, γ-batched across chunks: per chunk,
`BytecodeChunk(r_lane ‖ r_row) = Σ_lane eq(r_lane, lane) · lane_value(r_row)`
where a lane value is a direct 0/1 flag evaluation (`LaneWeight(lane)`
deriveds), a one-hot selector decode (`RegisterSelectorWeight(lane)` /
`LookupSelectorWeight` deriveds), or a byte decode (`PcByteDecode` /
`ImmByteDecode` deriveds, folding the lane weight). One sumcheck over the
widest byte lane's `(byte ‖ place)` prefix with the row point fixed; the
narrower selector legs bind only their suffix rounds and the flag legs none
at all — mixed-round legs are precedented by the lattice booleanity's msb
column. Inputs `Vec<BytecodeChunk@BytecodeClaimReduction>`; outputs one
opening per sub-column slot. degree 2, `8 + max(3, log₂ imm_byte_width)`
rounds. Draws `Gamma`.

The `(chunk, lane/flag)` output families are two-index, which the
`#[derive(OutputClaims)]` `Vec` convention cannot express; the claim-struct
trait impls are hand-written in the same field-declaration order the derive
would emit.

## Packed-Claim Invariants and Final Openings

Wjolt's native batch requires one evaluation per committed member at exactly
one point. Stage 7 therefore lands all RA, increment chunk, and MSB claims at
the same `(address || cycle)` point, after which Stage 8 applies the common
row-major permutation. Any missing member, arity mismatch, or point mismatch
is rejected before the native Akita verifier runs.

Auxiliary packed objects retain the `PrefixPacking` invariant: one claim per
slot, every slot claimed, with the generic reduction handling distinct points.

**Padding invariant**: the hamming legs claim "exactly one hot cell per row"
summed over the *full* Boolean hypercube, so padding rows must be encoded as
the canonical shifted-zero encoding — never as all-zero cells. For fused
increment zero, every chunk is hot at address zero and the MSB is hot at
address one because the encoded value is `2^64`. This also applies to advice
byte positions beyond the actual
advice size (and, by the offline checks, to padded bytecode/program-image
rows). All-zero padding would falsify the claimed input sums (`1` per chunk
row, `γ` for the advice column).

`lattice/packing.rs` is the single source of truth for the endgame:

A polynomial's **final claim** is the one claim left when the relation DAG
bottoms out. Wjolt final claims feed the native batch; auxiliary claims feed
their packed reductions. `Virtualized` polynomials have none.

```rust
pub enum LatticeFinalOpening {
    /// One evaluation claim on the per-proof / precommitted W; `final_claim`
    /// names the producing relation output. Total — no
    /// reconstruction-produced exception.
    Packed { final_claim: JoltOpeningId },
    /// Never PCS-opened: consumed entirely by lattice relations.
    Virtualized,
}

pub fn final_opening(polynomial: JoltCommittedPolynomial) -> LatticeFinalOpening;
// RdInc, RamInc                  -> Virtualized (discharged by the read-raf
//                                   fused consumer stages)
// TrustedAdvice/UntrustedAdvice  -> Virtualized (via relations 6/7)
// BytecodeChunk(i)               -> Virtualized (via relation 9)
// ProgramImageInit               -> Virtualized (via relation 8)
// InstructionRa/BytecodeRa/RamRa -> Packed, from HammingWeightClaimReduction
// UnsignedIncChunk(j)            -> Packed, from HammingWeightClaimReduction
// UnsignedIncMsb                 -> Packed, from HammingWeightClaimReduction
// {Un}trustedAdviceBytes         -> Packed, from relation 6/7
// bytecode lane polynomials      -> Packed, from BytecodeChunkReconstruction
// ProgramImageBytes              -> Packed, from ProgramImageReconstruction
```

The base-mode map (`committed_openings::final_opening_relation`) gained
lattice arms when the committed enum grew and is the single owner of the
polynomial→relation mapping: the final claims are derived from it
(`final_opening_id`).

This replaces both the prototype's `final_opening_lattice_requirement` and the
base stage-8 RLC order for lattice mode.

## Verifier and Prover Schedule

- Stage 6a's bytecode read-RAF address fold consumes the four reduced inc
  claims (stages `γ^5..8`); no phase runs between Stages 5 and 6a.
- Stage 6a/6b runs lattice Booleanity; Stage 6b's read-raf cycle phase carries
  the `FusedInc` factor and produces its opening at the shared cycle point —
  there is no separate fused-inc member.
- Stage 7 extends `HammingWeightClaimReduction` with hamming, Booleanity, and
  shifted-decode terms for all increment one-hot columns.
- Stage 8 opens Wjolt directly with one native same-point Akita batch and runs
  the generic packed-opening reduction only for auxiliary objects.
- The Dory build instantiates the original Booleanity, increment claim
  reduction, HammingWeightClaimReduction, and homomorphic final opening.

## Testing Strategy

- Per-relation algebra tests in the existing house style: evaluate
  `input_expression`/`output_expression` against distinct prime values and
  assert the closed-form formula; assert `required_openings/deriveds/
  challenges` sets and opening order.
- Reconstruction identity test: for random signed values,
  `Σ 2^(b*j)·address(chunk_j) + 2^64·address(msb) − 2^64`
  round-trips the value.
- Semantic integration tests (`tests/lattice_semantics.rs`) over concrete
  witness data: every Wjolt member is a uniform `K x T` one-hot polynomial;
  the chunk/MSB decomposition reconstructs signed increments including
  padding; auxiliary lane/advice reconstruction terms reproduce the committed
  evaluations.
- Determinism: `wjolt_members`/`precommitted_packing` are pure functions of
  shape (golden test), and every packed proof column has a claim source.

## Dropped From the Prototype (jolt-claims-ref) and Why

| Prototype construct | Fate | Reason |
|---|---|---|
| `PackingWitnessLayout`/`FamilySpec`/`FactDomain`/`Alphabet`, offsets, digests | deleted | duplicate of `jolt-openings::PrefixPacking`; slot math is transport, not semantics |
| `PackingFamilyId` (namespace/u64/u64 + bit-packed indices) | deleted | `JoltCommittedPolynomial` is the packing id (the interim `LatticeColumn` enum is deleted too) |
| `JoltOpeningId::Lattice { relation, index }` | deleted | untyped; hid polynomial identity and point identity |
| `PackingValidityRequirement`/`PackingValidityKind` + digest | deleted | validity is not metadata; it is the validity **relations** (prover-supplied columns) or an **offline preprocessing check** (public precommitted columns) |
| `PackingViewFormula`/`PackingViewTerm` | deleted | briefly survived as `ReconstructionTerm` lists, then replaced outright by the reconstruction *relations* (5–8) + the decode-weight point algebra; provenness follows from the column's validity story |
| two-entry one-hot boolean-flag encoding | replaced for auxiliary public bytecode flags | plain 0/1 columns of `log_rows` variables; `1 − flag` is expression algebra, and the auxiliary commitment stays 0/1. The increment MSB is not such a flag: it is a full `K x T` Wjolt one-hot member. |
| bytecode store-rd-disjoint, canonical imm bytes, sub-column one-hot validity | moved offline | public precommitted data; the verifier checks structure at preprocessing instead of spending relations |
| Public→Challenge enum migration (EqCycle etc.) | not ported | orthogonal to lattice; main's Derived model already covers it |
| eq-polynomial / coefficient materialization helpers | not ported | belongs to verifier/witness crates per the boundary contract |

## Open Questions

1. **Advice byte-column granularity**: one column per advice kind over
   `(byte ‖ place ‖ word)` (spec'd here, matching the prototype) vs one
   column per byte place (8 smaller columns). Per-place columns shrink each
   slot's variable count but multiply slot count; revisit if the packed
   dimension or the validity sumcheck shape prefers one over the other at
   integration time.
2. **Precommitted trusted-advice openings** (partially resolved): the
   trusted-advice byte reconstruction is now the `TrustedAdviceReconstruction`
   relation, so *what* is proven is fixed; where its rounds sit in the stage
   schedule (against the precommitted `W'` point set vs anchored into the
   per-proof set, mirroring the base `PrecommittedClaimReduction` scheduling)
   is still the verifier stage design's call — the relation is
   schedule-agnostic.
