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

1. **No RLC of commitments.** The stage-8 final-opening step (an RLC batch over the
   separate `RamInc`/`RdInc`/`Ra`/advice commitments, each lifted to a unified
   point by an embedding scale) cannot exist. Instead, all committed columns
   live in **one packed witness polynomial `W`** per commitment lifecycle, and
   one opening of `W` settles every logical claim via the prefix-packing
   reduction (`jolt-openings::PrefixPacking`).
2. **0/1 cells, for efficiency.** Committing 0/1 vectors is where Akita is
   fast, so the packed witness is kept one-hot/boolean throughout — this is a
   performance choice, not a norm-bound requirement. Every committed column
   that is not already one-hot gets a one-hot encoding: the dense signed
   `RdInc`/`RamInc` columns become a **fused, shifted, one-hot chunk
   decomposition**, and all unstructured committed data (trusted/untrusted
   advice, program image, bytecode lanes) is byte/symbol one-hotted. The
   one-hot *validity* relations double as the range checks the reconstructions
   rely on (byte one-hot ⇒ each reconstructed byte < 256).

This spec defines the `jolt-claims` surface for both: the canonical packed
witness description, and the additional symbolic relations that make the
one-hot inc representation sound. It describes **additional semantics on top of
the base `jolt/` PIOP** — not a new PIOP. `jolt-verifier` has no lattice
implementation yet; this crate-level design is written so that the later
verifier integration is mechanical (`ConcreteSumcheck` impls + a stage-schedule
swap + a packed-opening stage).

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
canonical packed-witness description (per-proof + precommitted), consumed by
    jolt-openings::PrefixPacking — jolt-claims defines ids + arities only
inc fusion (RamInc/RdInc -> one Inc stream selected by the bytecode Store flag)
base-2^b one-hot chunk decomposition + msb column (the +2^64 unsigned shift is
    a constant, folded into the chunk reconstruction)
booleanity + hamming-weight coverage of the new one-hot columns
chunk reconstruction relation tying the chunk columns back to the fused Inc claim
store-selector binding to the bytecode Store circuit flag
advice byte one-hot decomposition + its reconstruction relations (untrusted:
    validity + word-decode legs; trusted: word-decode leg)
bytecode sub-column decomposition (register selectors, circuit/instruction
    flags, lookup selector, raf flag, unexpanded-pc bytes, imm bytes) and the
    BytecodeChunkReconstruction relation rebuilding chunk lane values
program-image byte decomposition + its reconstruction relation
decode-weight point algebra (the relations' derived semantics)
final-opening map (packed claim | virtualized)
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
the range checks for the reconstructed words). Trusted advice is one-hot encoded
like everything else but precommitted by a party the verifier trusts, so its
encoding is not re-proven in-protocol.

## Boundary Contract

| Crate | Owns | Must NOT contain |
|-------|------|------------------|
| `jolt-claims` | ids, arities, symbolic relations, final-opening map, reconstruction terms | slot assignment, offsets, digests, transcripts, witnesses, PCS |
| `jolt-openings` | `PrefixPacking` slot assignment, suffix-compat check, eq-prefix reduction, transcript binding | Jolt-specific ids or protocol semantics |
| `jolt-verifier` (future) | `ConcreteSumcheck` impls, stage schedule, packed-opening stage | relation algebra (sourced from `jolt-claims`) |
| `jolt-witness`/prover (future) | packed witness materialization | — |
| `jolt-akita` | PCS transport | — |

`jolt-claims` depends on `jolt-openings` directly and uses its API as the
**single source of truth** for the packing:
`lattice::proof_packing`/`precommitted_packing` register the canonical column
orderings with `PrefixPacking::new` and return the `PrefixPacking` object
itself — there is no parallel jolt-claims vocabulary for slots, offsets, or
placement (`PrefixSlot::packed_index` is where witness assembly gets cell
positions). The boundary is unchanged in substance: jolt-claims names ids and
arities; jolt-openings owns slot assignment and the eq-prefix reduction;
jolt-claims' API surface remains transcript-free.

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
├── packing.rs      packed-witness registration + endgame:
│                   proof_packing(..)/precommitted_packing(..) ->
│                   PrefixPacking<JoltCommittedPolynomial> (the canonical
│                   registration; every packed polynomial is a lattice-mode
│                   committed polynomial), and the LatticeFinalOpening
│                   final-opening map (packed claim | virtualized)
└── relations/
    ├── inc_virtualization.rs
    ├── chunk_reconstruction.rs
    ├── booleanity.rs                    lattice-mode Booleanity (same id)
    ├── advice_reconstruction.rs         untrusted (validity + decode legs)
    │                                    and trusted (decode leg) advice
    ├── bytecode_reconstruction.rs       chunk -> per-lane polynomial decode
    └── program_image_reconstruction.rs  word -> byte one-hot decode
```

There is no store-binding relation file: the fused-inc destination selector is
the existing `OpFlags(Store)` virtual polynomial, so its opening is settled
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
UntrustedAdviceReconstruction,   // byte validity + word-decode legs
TrustedAdviceReconstruction,     // word-decode leg only
ProgramImageReconstruction,      // word-decode leg only
BytecodeChunkReconstruction,     // chunk -> lane sub-column decode

// JoltCommittedPolynomial — appended variants (committed only in lattice mode,
// as slots of the packed witness; base mode never constructs them)
UnsignedIncChunk(usize),   // one-hot column j of the fused unsigned inc
UnsignedIncMsb,            // boolean msb column
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
FusedInc,                  // gamma-batched RamInc/RdInc stream; its selector
                           // is the existing OpFlags(Store), its +2^64 shift
                           // is folded into the chunk reconstruction
```

WARNING: enum `Ord` is protocol data too — `PrefixPacking` assigns slots by
`(num_vars, Id)` order, so *reordering* variants (not just inserting) changes
the packed witness layout silently.

Per-relation challenge/public sub-enums live in `protocols/jolt/ids.rs` next to every other
relation's (house convention) and are aggregated as appended
`JoltChallengeId`/`JoltDerivedId` variants:

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

The packing id is `JoltCommittedPolynomial` itself: every logical polynomial
of a packed witness — including the precommitted bytecode lane polynomials
and program-image
bytes — is a (lattice-mode) committed polynomial with in-protocol openings
(its packed claim is produced by a relation). The earlier draft's separate
`LatticeColumn` id family, which distinguished "columns with identity" from
"reconstruction-only columns", is gone: once the reconstructions are
relations, every packed column has exactly one relation-produced claim and
the two-tier distinction has nothing left to describe.

Two commitment lifecycles, two packings:

```rust
/// Per-proof packed commitment: trace/advice-domain columns.
pub fn proof_packing(shape) -> PrefixPacking<JoltCommittedPolynomial>;
// InstructionRa(0..I), BytecodeRa(0..B), RamRa(0..R)   log_k_chunk + log_T each
// UnsignedIncChunk(0..N)                               log_k_chunk + log_T each
//                       (chunk width b = log_k_chunk, N = 64 / b)
// UnsignedIncMsb                                       log_T
// UntrustedAdviceBytes                                 8 + 3 + advice_vars

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
  e.g. `(byte ‖ place ‖ word)` for byte one-hots — so the instance bits are
  always the packed point's suffix (this is what lines packed claims up on
  shared suffix points).
- **Boolean flag columns** are plain 0/1 columns of `log_rows` variables —
  *not* the prototype's 2-symbol one-hot encoding. `1 − flag` is expression
  algebra; spending a second cell row on it doubles cells for nothing.
- **Non-power-of-two byte counts** (imm bytes) round the place dimension up;
  dummy place cells are zero by convention (checked offline with the rest of
  the precommitted validity).

`PrefixPacking::new` sorts by descending `num_vars` then id and assigns
power-of-two-aligned prefix slots, so `jolt-claims` performs **no offset
arithmetic, no digesting, no dummy-cell accounting** — all of that (including
transcript binding of the packing) is `jolt-openings`' job. The prototype's
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
- `selector_block_weight(lane_point, block_start, symbol_point, symbols)` —
  a selector block's lane-eq weights as a symbol-variable multilinear (the
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

### 1. `IncVirtualization` (replaces `IncClaimReduction` in lattice mode)

A cycle writes RAM (store) or a register, never both, so one inc stream
suffices — this halves the one-hot column count ("fused polys").

- **Inputs** (same four consumed claims as base `IncClaimReduction`):
  `RamInc@RamReadWriteChecking`, `RamInc@RamValCheck`,
  `RdInc@RegistersReadWriteChecking`, `RdInc@RegistersValEvaluation`.
- **Outputs**: `FusedInc` (virtual) and `OpFlags(Store)`, both at the bound
  cycle point. Using the existing store circuit-flag polynomial as the
  selector means its opening is bound to the actual bytecode by the same
  read-raf val-stage machinery that binds every other flag consumer
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
- **value reconstruction**: `Σ_j place_j · symbol(chunk_j) = FusedInc +
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
plumbing: the lattice-mode bytecode read-raf batch grows one val stage
(`LATTICE_BYTECODE_VAL_STAGES`) carrying the `OpFlags(Store)@IncVirtualization`
claim at its cycle point, exactly as the spartan-outer/shift flag claims enter
(this is what the prototype's `bind_store` sixth stage was). The offline
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
`(byte ‖ place)` prefix only; the word point is fixed by the incoming claim,
mirroring how the chunk reconstruction fixes `r_cycle`. Inputs
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

What `jolt-openings::prepare_statement` machine-checks is exactly: every
claim's point length equals its slot's `num_vars`, **one claim per slot**,
and every slot claimed. The packed-opening reduction itself supports
*distinct* points per slot (the single opening point is fresh sumcheck
randomness), so — correcting this spec's earlier draft — equal-variable-count
columns are **not** required to share a point for correctness.

Two consequences:

- **One claim per slot is the load-bearing invariant.** It is why the
  untrusted-advice validity legs and the word-decode leg are one relation
  (relation 5), and why every packed column has exactly one producing
  relation in the final-opening map below.
- **Point sharing is a prover/verifier optimization**, not a requirement:
  landing the chunk columns on the `Ra` families' shared
  `(r_address ‖ r_cycle)` point (chunk width fixed at `b = log_k_chunk`, the
  `EqBooleanityAddress` leg) keeps the packed-opening eq work down and stays
  the intended schedule. `jolt-verifier` owns batching, staging, and any
  co-binding; `jolt-claims` carries no stage assignments.

**Padding invariant**: the hamming legs claim "exactly one hot cell per row"
summed over the *full* Boolean hypercube, so padding rows must be encoded as
the one-hot of value 0 — never as all-zero cells. This applies to inc-chunk
rows beyond the trace length and advice byte positions beyond the actual
advice size (and, by the offline checks, to padded bytecode/program-image
rows). All-zero padding would falsify the claimed input sums (`1` per chunk
row, `γ` for the advice column).

`lattice/packing.rs` is the single source of truth for the endgame:

A polynomial's **final claim** is the one claim left on it when the relation
DAG bottoms out — consumed by the packed opening (as its slot's single
`EvaluationClaim`) rather than by another relation. `Virtualized` polynomials
have none: every claim on them is consumed by a lattice relation.

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
// RdInc, RamInc                  -> Virtualized (via IncVirtualization chain)
// TrustedAdvice/UntrustedAdvice  -> Virtualized (via relations 5/6)
// BytecodeChunk(i)               -> Virtualized (via relation 8)
// ProgramImageInit               -> Virtualized (via relation 7)
// InstructionRa/BytecodeRa/RamRa -> Packed, from HammingWeightClaimReduction
// UnsignedIncChunk(j)            -> Packed, from UnsignedIncChunkReconstruction
// UnsignedIncMsb                 -> Packed, from Booleanity
// {Un}trustedAdviceBytes         -> Packed, from relation 5/6
// bytecode lane polynomials      -> Packed, from BytecodeChunkReconstruction
// ProgramImageBytes              -> Packed, from ProgramImageReconstruction
```

The base-mode map (`committed_openings::final_opening_relation`) gained
lattice arms when the committed enum grew and is the single owner of the
polynomial→relation mapping: the final claims are derived from it
(`final_opening_id`).

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
  symbol(chunk_j) + 2^64·msb − 2^64` round-trips the signed value.
- Semantic integration tests (`tests/lattice_semantics.rs`) over concrete
  witness data: every packed column reads back through its slot
  (`P_i(x) = W(prefix_i ‖ x)`); the chunk decomposition reconstructs signed
  increments (padding rows included); the lane/advice reconstruction terms reproduce
  the committed evaluations.
- Determinism: `proof_packing`/`precommitted_packing` are pure functions of
  shape (golden test), and every packed proof column has a claim source.

## Dropped From the Prototype (jolt-claims-ref) and Why

| Prototype construct | Fate | Reason |
|---|---|---|
| `PackingWitnessLayout`/`FamilySpec`/`FactDomain`/`Alphabet`, offsets, digests | deleted | duplicate of `jolt-openings::PrefixPacking`; slot math is transport, not semantics |
| `PackingFamilyId` (namespace/u64/u64 + bit-packed indices) | deleted | `JoltCommittedPolynomial` is the packing id (the interim `LatticeColumn` enum is deleted too) |
| `JoltOpeningId::Lattice { relation, index }` | deleted | untyped; hid polynomial identity and point identity |
| `PackingValidityRequirement`/`PackingValidityKind` + digest | deleted | validity is not metadata; it is the validity **relations** (prover-supplied columns) or an **offline preprocessing check** (public precommitted columns) |
| `PackingViewFormula`/`PackingViewTerm` | deleted | briefly survived as `ReconstructionTerm` lists, then replaced outright by the reconstruction *relations* (5–8) + the decode-weight point algebra; provenness follows from the column's validity story |
| 2-symbol boolean-flag encoding | replaced | plain 0/1 columns of `log_rows` variables; `1 − flag` is expression algebra, and the packed witness stays 0/1 |
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
