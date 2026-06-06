# Spec: Akita Integration

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-27 |
| Status | draft / exploratory |
| PR | TBD |

## Purpose

This spec describes the Akita/lattice integration path for modular Jolt. The
north star is a `jolt-akita` PCS crate and a `jolt-verifier` build selected by a
`lattice` protocol axis. The verifier should remain generic over the final PCS
opening path, while Akita owns packed commitment layout, prefix-selector/decode
opening reduction, and lattice-specific proof verification.

The target integration has four coupled changes:

```text
1. Opening trait system:
   replace the stage-8 AdditivelyHomomorphic requirement with a generic
   BatchOpening boundary.

2. Commit path:
   introduce a prefix-packed one-hot fact table so Akita can commit to the
   most efficient physical representation before stage-8 challenges are known.

3. Jolt PIOP/layout:
   keep most Jolt relations over logical polynomials, but define those
   polynomials as decoded views of the packed Akita commitment.

4. Translation and validity:
   add the relations proving that newly one-hotted data is valid and that the
   logical claims emitted by the PIOP equal decoded claims about the packed
   commitment.
```

This spec expands [akita-packed-opening-protocol.md](akita-packed-opening-protocol.md).
That document focuses on the stage-8 opening abstraction; this document owns the
full Jolt/Akita integration plan.

## How To Read This Spec

This is a working architecture and protocol design document, not a finalized
implementation contract. The Akita path changes both PCS mechanics and Jolt
PIOP layout, so several items must be decided with protocol reasoning before
they become code requirements.

Classifications used in this document:

```text
stable:
  conclusions that appear structurally necessary

checkpoint:
  intentionally simple implementation targets useful for integration bring-up

target:
  expected optimized design direction, still requiring protocol details

open:
  decisions that must be resolved before the target architecture is complete
```

Current classification:

```text
stable:
  stage 8 should depend on BatchOpening, not AdditivelyHomomorphic
  Dory can implement BatchOpening through the current homomorphic RLC
  Akita cannot be modeled as post-commitment additive RLC
  Akita commits to a prefix-packed one-hot physical object, not to the Dory
    logical polynomial list
  Jolt relations should continue to speak in logical polynomial/view terms
  final logical claims must be translated to claims about the packed object
  one-hot/range validity for newly one-hotted data belongs before final opening
  Akita setup/prover costs depend on the packed MLE dimension D, so crossing
    D -> D + 1 is the real cliff

checkpoint:
  separate one-hot Rd/RAM increments
  transparent Akita final opening path
  reject or separately handle advice/auxiliary commitments

target:
  prefix-packed main fact table with mixed alphabets
  fused one-hot increment family if expanded-trace support disjointness and
    source linking are made explicit
  explicit auxiliary packing policy by domain/lifecycle
  jolt-akita owns Akita layout, proof, transcript, and field adapters

open:
  formal statement/test of the expanded-trace increment support-disjointness
    invariant
  fused increment source encoding, zero-row convention, and bytecode linking
  advice/bytecode/blinding commitment formats
  first supported Akita field/claim-field mode
  whether Akita ZK composes with BlindFold in the first lattice release
```

## Decision Map

The main architecture/protocol decisions are:

```text
1. Stage-8 verifier boundary:
   BatchOpening is the intended boundary. This is mostly an architecture
   decision and should preserve Dory behavior.

2. Commit-path boundary:
   Akita needs commit-time batch planning. This is architectural, but it is
   constrained by the protocol view/packing manifest and by Akita's native
   layout.

3. Main packed object:
   use prefix packing over typed one-hot fact subtables, not a flat
   fixed-width lane selector, so bit-valued facts cost 2T cells rather than a
   full byte lane.

4. Increment representation:
   separate one-hot increments are the bring-up checkpoint; fused one-hot
   increments are the optimized target. The fused target is blocked on a
   source/linking protocol.

5. Translation staging:
   one-hot validity and semantic linking are PIOP relations, while final
   decoded-claim translation is a stage-8 opening reduction.

6. Bytecode/source linking:
   the fused increment source cannot be a late virtual bytecode value if stage 6
   needs it while bytecode read-RAF is checked in parallel. This is the central
   fused-increment protocol design problem.

7. Auxiliary commitments:
   advice, program bytecode commitments, and ZK blinding commitments need
   explicit domain-specific policies instead of being appended to the main
   packed fact set.
```

The rest of the spec should be read as scaffolding around these decisions.

## Scope

V1 target:

```text
crates/jolt-akita implementing Akita PCS adapters for modular Jolt
generic BatchOpening verifier/prover traits in jolt-openings
generic batch/packed commit planning sufficient for Akita main witness packing
Dory compatibility via AdditivelyHomomorphic-backed batch opening
jolt-verifier stage-8 dispatch through BatchOpening
lattice protocol config axis selecting Akita and lattice final-opening manifest
one-hot increment semantics in jolt-claims
Akita transparent final opening verification for modular jolt-verifier
```

V1 non-goals:

```text
general multipoint Akita incidence exposure in jolt-verifier
wrapping Akita as if it were additively homomorphic
forcing Dory through the packed commitment path
full Akita ZK/BlindFold support unless explicitly selected later
settling all advice, bytecode-commitment, and blinding commitment packing in the
  first checkpoint
preserving legacy jolt-core proof shape for the lattice path
```

The first checkpoint may be intentionally suboptimal if it proves the trait and
modular verifier boundary. The performance target must still be explicit,
because packed cell-count changes can jump the Akita dimension to the next
power-of-two setup size.

## Design Principles

### Stage 8 Is Generic

`jolt-verifier` should not know whether the final opening batch is verified by:

```text
Dory:
  post-commitment homomorphic RLC

Akita:
  prefix-selector/decode opening of a precommitted packed polynomial
```

The verifier constructs a final-opening request and receives logical
coefficients. Those coefficients multiply raw Jolt opening values and are used
for BlindFold/stage-output binding.

### Prefix Packing Is The Commit-Time Layout

Akita's final reduction only makes sense if the prover already committed to the
physical packed object. The target object is not the rectangular selector
packing:

```text
P_pack(x, y) = sum_i eq(y, i) * P_i(x)
```

except as a special case. The target object is a prefix-packed one-hot fact
table:

```text
W : {0,1}^D -> F

subtable g occupies a prefix-coded subcube:
  W[prefix_g, local_coordinates_g]
```

Each logical Jolt object is a view of this table:

```text
RA_d(row, chunk_value):
  direct one-hot view

Inc(row):
  decoded byte/sign view

Advice(row):
  decoded advice-byte view, if advice is packed
```

The stage-8 random packed point is sampled after commitments, but all prefix
codes, subtable dimensions, alphabet sizes, and decoders are fixed and
transcript-bound at commit time. Therefore the trait redesign needs both:

```text
BatchCommitment / packing plan
BatchOpening / final proof verification
```

The planner should optimize total cell count:

```text
D = ceil(log2(sum_g cells(g)))
```

not only the number of same-shaped lanes. Rectangular lane packing is still a
useful implementation checkpoint, but it is not the target abstraction.

Relation to the recursion-paper prefix packing:

```text
same core method:
  put variable-size MLEs into disjoint prefix-coded subcubes
  use prefix selector weights to reduce many same-point claims to one opening

different application:
  recursion paper packs already-meaningful dense MLE claims
  Akita/Jolt packs physical one-hot fact subtables first
  many logical Jolt polys are decoded views of those physical facts
```

The direct-view part of the Akita design is the same sumcheck-free prefix
packing reduction. The decoded-view part adds Jolt-specific translation
relations because the committed data is not always the logical polynomial
itself.

### Logical Views Are Protocol Objects

The PIOP should not reason about raw packed coordinates everywhere. It should
continue to use logical objects:

```text
InstructionRa_d
BytecodeRa_d
RamRa_d
RamInc / RdInc
TrustedAdvice / UntrustedAdvice
auxiliary facts
```

In lattice mode, these are virtual committed views:

```text
P_i(x) = Decode_i(W)(x)
```

The packed manifest is therefore part of the protocol, not an implementation
serialization detail. It must specify:

```text
logical opening id
physical subtable prefix
local domain variables
alphabet size
decoder weights
validity constraints
translation relation class
```

### Validity And Translation Are Separate

Changing the committed representation creates two proof obligations:

```text
validity:
  the packed physical data is well formed
  examples: one-hot rows, boolean source flags, byte/sign range structure

translation:
  the logical PIOP claim P_i(r) equals Decode_i(W)(r)
```

Validity is a PIOP semantic property and belongs in stages 6/7 or adjacent
relations. Translation of the final logical claims belongs in stage 8.

### Main Witness And Auxiliary Commitments Are Different

The main execution-trace one-hot witness facts are large and regular. They are
the best candidates for one packed Akita commitment.

Auxiliary objects may have different domains or lifecycles:

```text
trusted advice
untrusted advice
program bytecode commitment
ZK blinding commitments
future lookup/helper commitments
```

These should not automatically be inserted into the main packed group. Each
class needs an explicit format and packing decision.

## Current Dory Shape

The current modular verifier stage 8 assumes an additive PCS:

```text
final claims:
  P_i(r_i) = v_i

Jolt embedding:
  v'_i = scale_i * v_i

RLC:
  sample gamma_i
  C* = sum_i gamma_i C_i
  v* = sum_i gamma_i v'_i
  verify C*(r) = v*
```

Dory dense increments are shorter than RA one-hot polynomials, so they are
embedded into the full address-cycle point with:

```text
scale_i = eq(r_address, 0)
```

This is a Dory-layout workaround. It should not be inherited by the Akita
lattice path when increments are one-hot committed over the same final point as
the RA views.

## Akita Main Path

Akita commits to a prefix-packed one-hot main witness:

```text
W : {0,1}^D -> F
C_W = AkitaCommit(W)
```

The packed universe is a disjoint union of typed one-hot subtables:

```text
InstructionRa(chunk)  : row x byte-value
BytecodeRa(chunk)     : row x byte-value
RamRa(chunk)          : row x byte-value
IncByte(chunk)        : row x byte-value
IncSign               : row x bit
IncSource, if fused   : row x bit
AdviceByte, if packed : advice-row x byte-value
auxiliary facts       : their own local domains
```

Each logical claim used by the PIOP is a view:

```text
P_i(local_point) = Decode_i(W)(local_point)
```

At stage 8, Jolt has final logical claims:

```text
P_i(r_i) = v_i
```

For the intended common-point path, claim reductions should preserve:

```text
r_i = r for all main trace-domain logical views
```

The batch-opening layer samples packing challenges and proves:

```text
sum_i lambda_i * v_i
  =
linear or structured functional of W
```

For direct one-hot views, the translation is linear and local in the row point:

```text
RA_d(r, k) = W[prefix_RA_d, r, k]

IncByte_j(r) =
  sum_b b * W[prefix_IncByte_j, r, b]
```

The final translation can then reduce over only the packed suffix/prefix
coordinates and end in one Akita opening:

```text
W(r, u*)
```

For structured views such as fused increments, the translation may also range
over the row variable and end at a fresh row challenge:

```text
W(x*, u*)
```

This is still compatible with one Akita opening, but it is a stronger protocol
relation than direct prefix selection.

No Akita implementation should form:

```text
sum_i gamma_i C_i
```

for arbitrary transcript challenges. Linear combination happens in the claim
relation induced by the packed view, not in the commitment group.

## Prefix-Packed Dimension Budget

The relevant Akita cliff is the dimension of the packed MLE:

```text
D = ceil(log2(total packed cells))
```

For same-shaped byte lanes over a trace of length `T = 2^n`, this specializes
to:

```text
D = n + 8 + ceil_log2(byte_lane_count)
```

For prefix-packed mixed alphabets, the accounting is more precise:

```text
byte-valued trace fact: T * 256 cells
bit-valued trace fact:  T * 2 cells
advice byte fact:       advice_rows * 256 cells
```

This is why a source flag or sign bit should not be modeled as a full byte lane
unless the protocol intentionally accepts the wasted cells.

For large-trace RV64 with 8-bit chunks, the expected base byte-valued one-hot
facts are:

```text
instruction RA chunks: 16 byte facts
bytecode RA chunks:    up to 3 byte facts for 16M instructions
RAM RA chunks:         up to 4 byte facts for 4B doublewords / 32GB
base total:            up to 23 byte facts
```

Separate increment one-hotization over a signed 65-bit range gives:

```text
lower 64 bits: 8 byte facts per increment family
sign bit:      1 bit fact per increment family

cells per trace row:
  base RA:       23 * 256 = 5888
  Rd increment:   8 * 256 + 2 = 2050
  RAM increment:  8 * 256 + 2 = 2050
  total:                    9988

dimension:
  n + ceil_log2(9988) = n + 14
```

Fused increment one-hotization gives:

```text
lower 64 bits: 8 byte facts total
sign bit:      1 bit fact
source bit:    1 bit fact

cells per trace row:
  base RA:        23 * 256 = 5888
  fused inc:       8 * 256 = 2048
  sign/source:     2 + 2   = 4
  total:                    7940

dimension:
  n + ceil_log2(7940) = n + 13
```

This is the main packing pressure. With prefix packing, a bit-valued source fact
does not by itself force a 33rd byte lane. With rectangular byte-lane packing,
the same source would be modeled as another `T * 256` lane and would push the
common large-trace case from 32 byte lanes to 33 byte lanes.

The design target is therefore:

```text
use prefix packing for mixed alphabets
track total cells and D, not just lane count
avoid materializing dummy/padding cells
test D for the common large-trace shapes
```

### Concrete Dimension Table

For the common large-trace shape above:

```text
base RA facts:
  23 byte facts -> 5888 cells/row -> D = log_T + 13

separate increments:
  23 base byte facts + 16 inc byte facts + 2 sign bit facts
  -> 9988 cells/row -> D = log_T + 14

fused increments:
  23 base byte facts + 8 inc byte facts + sign/source bit facts
  -> 7940 cells/row -> D = log_T + 13
```

Concrete protocol dimensions:

| log_T | base RA only | separate inc checkpoint | fused inc target |
|-------|--------------|-------------------------|------------------|
| 20 | D = 33 | D = 34 | D = 33 |
| 25 | D = 38 | D = 39 | D = 38 |
| 30 | D = 43 | D = 44 | D = 43 |

Akita consequences:

```text
setup/key memory:
  primarily a function of D or the Akita parameter bucket containing D
  crossing D -> D + 1 is the dangerous cliff

prover commit time:
  should be proportional to emitted one-hot facts plus Akita's D-dependent
  commitment work, not to materialized ambient padding

verifier time:
  should depend on the Akita opening proof and D, not on the number of logical
  Jolt claims after packing; exact constants must be measured in jolt-akita
```

Using the rough Akita setup sizing discussed for the current fork, the
`log_T = 30` fused target at `D = 43` is the desired bucket, while the separate
checkpoint at `D = 44` is roughly the next bucket and should be treated as about
2x setup/key memory until measured. This is an assumption to validate in
`jolt-akita`, not a protocol theorem.

## Increment Strategy

### Checkpoint A: Separate Increments

The simplest Akita checkpoint is the approach from the old Hachi integration:

```text
RdIncByte[0..8)
RdIncSign
RamIncByte[0..8)
RamIncSign
```

Properties:

```text
easy to reason about
matches separate Jolt register/RAM increment semantics
does not require a new disambiguation flag
pushes common large trace packed dimension to n + 14
does not handle auxiliary commitments yet
```

This checkpoint is acceptable as a bring-up target, but it should not be treated
as the final optimized layout if it crosses an Akita setup/prover cliff for the
target parameters.

### Target B: Fused Increments

The optimized target is a fused increment family:

```text
FusedIncByte[0..8)
FusedIncSign
FusedIncSource
```

Conceptually, this reuses one byte/sign decomposition for both register and RAM
increments. The committed physical fact is:

```text
Inc(row)
source(row) in {none/zero convention, register, RAM}
```

The logical views are source-filtered:

```text
RamInc(row) = is_ram_source(row) * Inc(row)
RdInc(row)  = is_rd_source(row)  * Inc(row)
```

This is valid only under the expanded-trace support-disjointness invariant:

```text
for every expanded Jolt trace row t:
  not (RamInc(t) != 0 and RdInc(t) != 0)
```

The support of an increment family is the set of expanded trace rows where that
logical increment polynomial is nonzero. Fusion is valid only if the RAM and
register increment supports are disjoint.

This invariant is not a generic RISC-V source-instruction fact. Source-level
AMOs modify memory and write `rd`. It is expected to hold because Jolt expands
read-modify-write instructions into multiple trace rows, e.g. load old memory,
compute/store new memory, then copy old memory to `rd`. The invariant must be
stated and tested over the actual `Cycle` model before enabling fused
increments.

Rows where both logical increments are zero need a canonical source convention,
or an explicit `none` source value, so the packed source is not underconstrained.

The source cannot be a purely virtual bytecode-derived value if it is needed in
stage 6 while bytecode read-RAF is being checked in parallel. Virtualizing the
source from bytecode would force the bytecode dependency into a later stage,
which is not the desired staging.

Therefore the fused design requires:

```text
a committed source/disambiguation object, or an equivalent committed encoding
a boolean/range proof for source
a stage-6 linking constraint tying the source to bytecode/instruction semantics
a source-filtered decode relation proving the fused increment view equals the
  logical RamInc/RdInc contributions
```

Open design constraint:

```text
The source encoding should be a bit-valued or small-alphabet prefix-packed fact.
It should not be represented as a full byte-valued trace lane unless the
integration explicitly accepts the larger dimension.
```

Possible directions to evaluate:

```text
1. Commit source as a separate small auxiliary commitment outside the main
   packed group.

2. Fold source into a future bytecode commitment/linking proof so it is paid for
   in the bytecode auxiliary domain, not as a main trace fact.

3. Encode source inside the fused increment relation as a small-alphabet
   prefix-packed fact, with a separate consistency proof against bytecode.
```

This spec does not finalize the fused increment protocol. It records it as the
performance target and identifies the stage-6 bytecode-linking obstacle.

## Translation Relation Classes

The packed-view layer should distinguish at least three classes.

### Direct One-Hot Views

Direct views select a subtable and evaluate it at the logical point:

```text
InstructionRa_d(r, k) = W[prefix_InstructionRa_d, r, k]
RamRa_d(r, k)         = W[prefix_RamRa_d, r, k]
```

These are closest to the existing RA commitment semantics. Prefix packing only
changes the physical commitment and final opening reduction.

### Linear Decoded Views

Linear decoded views reconstruct a scalar from one-hot digits:

```text
Inc(row) =
  sum_{j=0}^{7} 256^j * sum_b b * W[prefix_IncByte_j, row, b]
  + sign_decode(W[prefix_IncSign, row, bit])

AdviceWord(row) =
  sum_j 256^j * sum_b b * W[prefix_AdviceByte_j, row, b]
```

At a random point `r`, these remain linear functionals of `W` with the row
fixed to `r`. They can be translated in stage 8 without an additional row
sumcheck.

### Masked Decoded Views

Fused increments are masked views:

```text
RamInc(row) = is_ram_source(row) * Inc(row)
RdInc(row)  = is_rd_source(row)  * Inc(row)
```

For MLEs:

```text
RamInc(r) =
  sum_x eq(r, x) * is_ram_source(x) * Inc(x)
```

and in general:

```text
RamInc(r) != is_ram_source(r) * Inc(r)
```

Therefore fused increments require a real translation/sumcheck over the row
variable. This is the main reason fusion is a protocol optimization rather than
just a prefix-packing layout choice.

## Stage Placement

Commit phase:

```text
construct packed manifest
commit to prefix-packed W and auxiliary groups
transcript-bind commitments and manifest digest/config
```

Stages 1-5:

```text
mostly unchanged
continue to emit logical claims over Jolt virtual and committed-view objects
```

Stage 6:

```text
existing bytecode read-RAF
existing RA booleanity/virtualization
existing increment claim reduction, adapted for one-hot/fused lattice mode
one-hot validity for increment byte/sign/source facts
source-bytecode/instruction linking for fused increments
advice cycle-phase reduction if advice is present
```

Stage 7:

```text
existing hamming-weight claim reduction for RA-style one-hot facts
advice address-phase reduction
one-hot/range checks for advice or auxiliary packed facts when their domain
  naturally fits the stage-7 address phase
```

Stage 8:

```text
collect final logical claims
sample batch/packing challenges
translate logical view claims to packed W claims
verify Akita packed opening
return logical coefficients for standard/BlindFold binding
```

Rule of thumb:

```text
global well-formedness of W -> stage 6/7 PIOP relation
final random decoded claim -> stage 8 translation/opening relation
```

## Auxiliary Commitments

Auxiliary commitments should be classified by domain and lifecycle before they
are packed.

### Advice

Advice has a different lifecycle from main trace polynomials:

```text
trusted advice: may be committed in preprocessing-only context
untrusted advice: committed at prove time
domain: advice size, not necessarily trace length T
```

For large traces, advice may be small relative to `T`. For small traces, advice
can be substantial. A single policy of "put advice in the main packed trace
commitment" is not obviously correct.

If advice is arbitrary field data, the naive one-hot representation is byte or
limb decomposition:

```text
AdviceByte_j(advice_row, byte)
```

Prefix packing makes the cost proportional to the actual advice domain:

```text
cells = advice_rows * bytes_per_advice_value * 256
```

not automatically `T * bytes_per_value * 256`. This is important for small
advice relative to the trace. Conversely, if advice is large for a small trace,
packing it into the main commitment may dominate the dimension and should be a
visible config decision.

V1 policy:

```text
do not require advice to be in the main packed Akita commitment
support a separate advice commitment/proof path or reject advice under lattice
until the policy is implemented
```

Target policy:

```text
commit advice in one-hot form when possible
pack advice by its own domain/size class using prefix packing
batch its final opening through the same BatchOpening trait, either as an
  auxiliary packed group or a separate Akita opening proof
```

### Program Bytecode Commitment

Bytecode RA facts in the main trace are not the same object as a future
precommitted program bytecode commitment. The Akita integration must keep these
separate:

```text
BytecodeRa facts:
  trace-indexed one-hot PC/address witness facts

program bytecode commitment:
  commitment to the program image / bytecode table
  likely smaller and preprocessing-shaped
```

The bytecode commitment may become the natural home for fused-increment source
linking, but that relation is not yet specified.

### ZK Blinding Commitments

If Akita ZK is not implemented in V1, the lattice config must reject `zk`.

If Akita ZK is implemented later, blinding commitments should be modeled as
auxiliary PCS commitments with explicit domain and packing rules. They should
not silently consume main packed cells.

## Trait System

### CommitmentScheme

`CommitmentScheme` remains the singleton PCS contract:

```text
setup
commit one polynomial-like object
open one commitment
verify one opening
```

It must not imply additive homomorphism.

### AdditivelyHomomorphic

`AdditivelyHomomorphic` remains a lower-level capability:

```rust
pub trait AdditivelyHomomorphic: CommitmentScheme {
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output;
    fn combine_hints(hints: Vec<Self::OpeningHint>, scalars: &[Self::Field]) -> Self::OpeningHint;
}
```

Dory implements this as today. Akita does not.

### BatchOpening

Stage 8 depends on a batch-opening trait. The trait receives logical final
opening claims plus enough manifest metadata for the PCS implementation to map
them to a physical opening strategy:

```rust
pub trait BatchOpeningScheme: CommitmentScheme {
    type BatchProof;

    fn verify_batch<T>(
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        request: BatchOpeningRequest<Self::Field, Self::Output>,
        proof: &Self::BatchProof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>;
}
```

Dory can use a simple same-point request shape:

```rust
pub struct BatchOpeningRequest<F, C> {
    pub point: Vec<F>,
    pub items: Vec<BatchOpeningItem<F, C>>,
}

pub struct BatchOpeningItem<F, C> {
    pub commitment: C,
    pub eval: F,
    pub scale: F,
    pub view: PackedViewRef,
}
```

`PackedViewRef` can be trivial for Dory, where each item is physically
committed as its own polynomial. For Akita it identifies:

```text
direct subtable
linear decoded view
masked decoded view
auxiliary group
```

The trait name should use "batch"; it does not need to expose "same point" in
the public name. The intended Jolt invariant is that ordinary final logical
claims are reduced to one common point before stage 8. Structured translations
such as fused increments may internally sample a fresh row point while still
producing a single physical Akita opening.

Return shape:

```rust
pub struct BatchOpeningResult<F, C, H = ()> {
    pub logical_coefficients: Vec<F>,
    pub reduced_eval: Option<F>,
    pub reduced_commitment: Option<C>,
    pub hiding_eval_commitment: Option<H>,
}
```

`logical_coefficients[i]` always multiplies the raw Jolt value `eval_i`, not the
already-scaled value.

Dory implementation:

```text
logical_coefficients[i] = gamma_i * scale_i
```

Akita packed implementation:

```text
direct/linear views:
  logical_coefficients[i] = prefix_selector_i(rho) * decoder/scale terms

masked views:
  logical_coefficients are the coefficients of the stage-8 translation relation
  and may not be a pure local selector at the original row point
```

### BatchCommitment

Akita also needs a generic commit-time planning interface. The exact Rust API
can evolve, but the semantic boundary must exist.

Conceptual data model:

```text
CommitPlan:
  physical commitment groups
  prefix-packed subtables in each group
  logical view ids backed by each subtable
  prefix code / local coordinate layout
  alphabet size and decoder for each view
  domain class for each group
  packing strategy for each group
  proof/verification metadata

BatchCommitmentOutput:
  physical commitments
  per-view commitment references
  prover opening hints / batch hints
  verifier-visible plan digest or config
```

For Dory:

```text
one physical commitment per logical committed polynomial
per-view commitment reference is unique
batch opening later uses homomorphic RLC
```

For Akita main witness:

```text
one physical prefix-packed commitment for all main fact subtables
many logical view references point to the same C_W
batch opening later uses prefix-selector/decode translation
```

Minimal trait sketch:

```rust
pub trait BatchCommitmentScheme: CommitmentScheme {
    type CommitPlan;
    type BatchCommitmentHint;
    type PolynomialBatchSource;

    fn plan_commitment(shape: JoltCommitShape) -> Result<Self::CommitPlan, OpeningsError>;

    fn commit_batch(
        setup: &Self::ProverSetup,
        plan: &Self::CommitPlan,
        source: &Self::PolynomialBatchSource,
    ) -> Result<BatchCommitmentOutput<Self::Output, Self::BatchCommitmentHint>, OpeningsError>;
}
```

`JoltCommitShape` should live at the Jolt protocol layer, likely derived from
`jolt-claims` dimensions and selected protocol config. `jolt-openings` can keep
PCS-specific plan types opaque.

## jolt-akita Crate

Add a `jolt-akita` crate as the modular adapter between Jolt and Akita.

Responsibilities:

```text
define Akita PCS type implementing jolt-openings traits
wrap Akita prover/verifier setup types
wrap Akita commitment and batch proof types
provide Jolt-to-Akita transcript adapter
provide Jolt field bridge for the selected Akita field
own packed main witness layout and subtable order
own prefix-packed one-hot fact-table manifest
own lazy packed one-hot polynomial/source oracle
implement BatchCommitmentScheme for packed main commitments
implement BatchOpeningScheme for prefix-selector/decode opening verification
expose lattice protocol support flags, e.g. transparent-only vs zk-capable
```

Non-responsibilities:

```text
Jolt PIOP claim formulas
Jolt final-opening manifest ownership
legacy jolt-core proof compatibility
Dory assist or Dory RLC logic
```

Field note:

```text
Akita uses a base field and may support extension claim fields.
The first Jolt integration should choose a concrete field mode and make the
ClaimField story explicit. If V1 is base-field-only, the selected Jolt field and
Akita claim field should coincide.
```

## jolt-verifier Changes

`jolt-verifier` should target the modular path only.

Stage 8 should become:

```text
1. derive final logical-opening manifest from protocol config
2. collect typed stage claims
3. resolve each logical opening id to a physical PackedViewRef
4. compute embedding scales / decoder metadata
5. build BatchOpeningRequest
6. call PCS::verify_batch
7. bind PCS-defined opening inputs/transcript outputs
8. return Stage8Output with logical coefficients and optional reduced statement
```

The stage should not call:

```rust
PCS::combine(...)
```

directly.

Proof shape should use:

```rust
pub joint_opening_proof: PCS::BatchProof
```

Commitments should be protocol-configured. Dory native proof shape can preserve:

```text
rd_inc commitment
ram_inc commitment
RA commitment vectors
advice commitments
```

Lattice proof shape should allow:

```text
packed main commitment
auxiliary commitments by class
packed manifest/config metadata or digest
view-resolution metadata needed by the selected PCS
```

This can be modeled as either:

```text
PCS-associated commitment payload type
```

or as a typed Jolt commitment container with an explicit packed-main variant.
The important rule is that Akita should not be forced through legacy Dory
commitment ordering.

## Protocol Config

Add a lattice axis to the selected verifier config.

Conceptual shape:

```rust
pub enum PcsMode {
    Dory,
    AkitaLattice,
}

pub enum IncrementCommitmentMode {
    Dense,
    SeparateOneHot,
    FusedOneHot,
}

pub struct LatticeConfig {
    pub increment_mode: IncrementCommitmentMode,
    pub main_packing: MainPackingConfig,
    pub aux_policy: AuxiliaryCommitmentPolicy,
}
```

Config invariants:

```text
Dory -> Dense increments unless explicitly testing compatibility alternatives
Akita checkpoint -> SeparateOneHot increments
Akita target -> FusedOneHot increments once support disjointness is audited and
  source linking is specified
Akita transparent-only -> reject zk
```

The verifier must reject proofs whose self-described protocol config does not
match the compile-time selected verifier config.

## PIOP Changes

### Final Opening Manifest

Move final-opening list construction into a protocol-owned manifest. The
manifest has two layers in lattice mode:

```text
logical manifest:
  the Jolt claims emitted by stages 6/7, e.g. RamInc(r), RdInc(r), RA_i(r)

physical view manifest:
  how each logical claim is decoded from W or from an auxiliary packed group
```

Dory manifest:

```text
RamInc at IncClaimReduction
RdInc at IncClaimReduction
InstructionRa[*] at HammingWeightClaimReduction
BytecodeRa[*] at HammingWeightClaimReduction
RamRa[*] at HammingWeightClaimReduction
advice if present
field-inline views/facts if enabled
```

Akita separate-increment checkpoint manifest:

```text
logical:
  RdInc at IncClaimReduction
  RamInc at IncClaimReduction
physical views:
  RdIncByte[0..inc_d), RdIncSign
  RamIncByte[0..inc_d), RamIncSign
InstructionRa[*] at HammingWeightClaimReduction
BytecodeRa[*] at HammingWeightClaimReduction
RamRa[*] at HammingWeightClaimReduction
auxiliary views/facts according to aux policy
```

Akita fused-increment target manifest:

```text
logical:
  RdInc at IncClaimReduction
  RamInc at IncClaimReduction
physical views:
  FusedIncByte[0..inc_d), FusedIncSign, FusedIncSource
InstructionRa[*] at HammingWeightClaimReduction
BytecodeRa[*] at HammingWeightClaimReduction
RamRa[*] at HammingWeightClaimReduction
source/linking auxiliary commitments according to fused-inc design
```

### One-Hot Increment Reduction

For separate one-hot increments, the logical final openings remain:

```text
RamInc(r)
RdInc(r)
```

but the committed physical representation changes from dense scalar polys to
one-hot byte/sign facts. The old branch's checkpoint used:

```text
unsigned_inc = inc + 2^XLEN
lower 64 bits -> 8 one-hot chunks when log_k_chunk = 8
bit XLEN      -> one MSB one-hot bit fact with indices 0/1
```

The dense `IncClaimReduction` should be replaced or extended with a lattice
increment view reduction:

```text
stage 6 cycle phase:
  maintain the same logical reduction of four prior increment claims into
  RamInc(r) and RdInc(r)
  add one-hot/range validity for the byte/sign facts

stage 7 address phase:
  evaluate any chunk/address reductions needed to bind logical inc views to
  one-hot physical facts
  fuse with HammingWeight-style reductions where possible

stage 8:
  translate RamInc(r), RdInc(r) into decoded packed views of W
```

This removes the dense stage-8 embedding scale for increments in the lattice
path.

### Fused Increment Linking

The fused increment target requires a stage-6 source-linking relation:

```text
expanded-trace support-disjointness is an invariant:
  not (RamInc(t) != 0 and RdInc(t) != 0)
source flag is committed or otherwise committed-encoded
zero rows have a canonical source/none encoding
source agrees with bytecode/instruction semantics
fused increment value equals the selected register/RAM increment contribution
```

Because bytecode read-RAF is checked in stage 6, a bytecode-derived source flag
cannot be purely virtual if stage 6 needs it. The flag/linking design must be
worked out before fused increments are considered sound.

Open questions:

```text
What exact source encoding avoids adding a full byte fact or dimension bump?
Is the source encoded in a separate auxiliary commitment, bytecode commitment,
or fused relation witness?
What stage-6 input/output claims are needed to link source to bytecode?
Which trace expansion paths must be audited for support disjointness?
Does the linking term interact with field-inline or bytecode commitment PRs?
```

## Packing Policy

Main packed group:

```text
contains large trace-domain one-hot fact subtables selected by the lattice
  manifest
may include small-alphabet bit facts when they are protocol-owned and
  prefix-packed
uses lazy packed one-hot source
tracks total cell count and resulting Akita dimension D
has explicit dimension budget tests
```

Auxiliary groups:

```text
group by domain and lifecycle
do not silently add to the main packed group
may use separate Akita commitments/proofs in V1
may later use one-hot packing when domain shape is favorable
```

The commit planner must expose, at least in debug/test metadata:

```text
main_fact_count_by_alphabet
main_total_cells_per_trace_row
main_dimension_D
rectangular_lane_equivalent, for comparison/debugging
auxiliary_group_count
```

This makes power-of-two regressions visible in tests.

## Phased Implementation

### Phase 1: Trait Boundary With Dory Compatibility

```text
add BatchOpening request/result traits
implement Dory BatchOpening via existing RLC
change jolt-verifier stage 8 to call BatchOpening
preserve current Dory transcript/proof behavior
```

Exit criteria:

```text
muldiv passes in host and host,zk with Dory
jolt-verifier compatibility tests pass
stage8 Dory logical coefficients match current gamma_i * scale_i path
```

### Phase 2: Akita Packed Main Checkpoint

```text
add jolt-akita crate
add prefix-packed main commit planner
implement lazy prefix-packed one-hot source
implement Akita prefix-selector/direct-view BatchOpening
support separate one-hot Rd/RAM increments
reject or separately handle advice/auxiliary commitments
```

Exit criteria:

```text
Akita transparent muldiv verifies on modular path
main packed proof rejects wrong prefix/view manifest
main packed proof rejects tampered selector/decode-reduced claim
main packed source does not materialize padded dummy cells
```

### Phase 3: Auxiliary Commitment Policy

```text
define advice commitment format for lattice mode
define bytecode commitment interaction
define policy for small-domain vs trace-domain packing
add tests for small trace and large trace shapes
```

Exit criteria:

```text
trusted/untrusted advice either verify under lattice or reject at config boundary
bytecode commitment interactions are explicit
auxiliary groups do not accidentally push main dimension over budget
```

### Phase 4: Fused Increment Optimization

```text
prove/test expanded-trace support disjointness for RamInc and RdInc
design source/disambiguation encoding
add committed source/linking relation
integrate source link with stage-6 bytecode read-RAF
switch optimized lattice config from separate one-hot increments to fused
```

Exit criteria:

```text
common large-trace shape has packed dimension n + 13
source-linking tampering tests reject
fused increment claims equal separate increment semantics
```

## Invariants

- The commit-time prefix/view manifest equals the stage-8 final-opening manifest
  consumed by Akita.
- Akita packing challenges are sampled only after commitment, batch shape,
  point, claimed values, and manifest metadata are transcript-bound.
- Dory stage-8 transcript semantics are unchanged by introducing BatchOpening.
- `logical_coefficients` always multiply raw Jolt claim values.
- The lattice config never uses dense increment stage-8 embedding scales for
  one-hot increment views.
- Packed dummy cells are zero and never appear as Jolt logical opening IDs.
- Main packed cell count and Akita dimension are test-visible.
- Auxiliary commitments do not silently alter main packed dimension.
- Fused increments are not enabled until support disjointness, source encoding,
  zero-row convention, and source/linking relation are specified and tested.

## Acceptance Criteria

- [ ] `jolt-openings` exposes a batch-opening trait independent of additive
      homomorphism.
- [ ] Dory implements batch opening through its existing homomorphic RLC path.
- [ ] `jolt-verifier` no longer calls `PCS::combine` directly in stage 8.
- [ ] `jolt-akita` exists and owns Akita setup, commitment, proof, transcript,
      and prefix-packed layout adapters.
- [ ] The lattice final-opening manifest supports separate one-hot increment
      views.
- [ ] Akita packed main commitment verifies final openings through prefix
      selector/decode reduction.
- [ ] Dimension budget tests cover the common large-trace shape:
      `16 instruction + 3 bytecode + 4 RAM`.
- [ ] The separate-increment checkpoint documents the `n + 14` dimension cost,
      and the fused-increment target documents the `n + 13` goal.
- [ ] Fused increment mode is gated on an audited support-disjointness
      invariant over expanded Jolt trace rows.
- [ ] Advice/bytecode/auxiliary commitments either have a specified lattice
      policy or are rejected by config validation.

## Testing Strategy

Dory compatibility:

```bash
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
cargo nextest run -p jolt-verifier --cargo-quiet --features host
```

Trait tests:

```text
dory_batch_opening_matches_existing_rlc
dory_batch_opening_rejects_tampered_joint_proof
batch_opening_returns_coefficients_for_raw_claims
```

Akita packed tests:

```text
packed_commit_manifest_is_stable
prefix_selector_reduction_matches_direct_packed_eval
packed_opening_rejects_wrong_prefix_manifest
packed_opening_rejects_wrong_decode_claim
packed_source_skips_dummy_cells
```

Dimension budget tests:

```text
large_trace_base_byte_fact_count_is_23
separate_increment_dimension_is_n_plus_14
fused_increment_target_dimension_is_n_plus_13
source_flag_costs_two_cells_per_row_not_a_byte_lane
```

PIOP tests:

```text
onehot_increment_chunks_reconstruct_signed_increment
separate_onehot_increment_reduction_matches_dense_increment_claim
expanded_trace_ram_rd_increment_supports_are_disjoint
fused_increment_source_link_rejects_bad_bytecode_flag
fused_increment_matches_separate_increment_semantics
```

The fused source-link test is deferred until the fused protocol is specified.

## Performance

Dory:

```text
no intended regression
keep streaming RLC construction and homomorphic hint combination
```

Akita:

```text
main packed one-hot commit scan is proportional to emitted one-hot facts
not to the dense ambient hypercube and not to materialized padding cells
bit-valued facts cost O(T), not O(256T)
```

Acceptable padding:

```text
ambient hypercube padding from ceil_log2(total packed cells)
Akita ring/setup padding implied by D
Akita layout/planner padding
```

Unacceptable padding:

```text
materializing dummy packed cells
dense materialization of packed one-hot tables
adding auxiliary facts to the main packed group without dimension-budget review
representing bit facts as full byte lanes unless explicitly accepted
enabling fused increments before source/linking constraints justify the
  compression
```

## Open Questions

```text
1. What exact fused-increment source encoding should we use?
2. Can the source/linking term live naturally with bytecode commitment work?
3. What is the canonical source convention for rows with zero fused increment?
4. Which expanded trace paths must be part of the support-disjointness audit?
5. Which auxiliary commitments should be one-hot encoded?
6. Should advice be separate Akita commitments in V1 or rejected under lattice?
7. What is the first supported Akita field/claim-field mode in modular Jolt?
8. Does Akita ZK need to compose with BlindFold immediately, or is transparent
   lattice the first supported path?
```

## References

- [Akita packed opening protocol](akita-packed-opening-protocol.md)
- [jolt-verifier model crate](jolt-verifier-model-crate.md)
- [selected verifier integration](selected-verifier-integration.md)
- `../recursion-paper/techniques.tex` section `Prefix packing`
- `LayerZero-Research/jolt` branch `lz/integrate-hachi`
- `LayerZero-Labs/akita`
