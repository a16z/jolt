# Spec: Akita Packed Opening Protocol

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-27 |
| Status | draft |
| PR | TBD |

## Purpose

Akita packed opening protocol defines the minimal PCS-opening abstraction needed
for a lattice-backed Jolt verifier path. The goal is to let `jolt-verifier`
verify the final stage-8 opening batch without depending on Dory's additive
homomorphism.

Today stage 8 is Dory-shaped:

```text
many final claims P_i(r) = v_i
sample gamma_i
C* = sum_i gamma_i C_i
v* = sum_i gamma_i v_i
verify C*(r) = v*
```

This works for Dory because commitments and opening hints can be linearly
combined after the final Jolt claims are known. Akita does not have this
post-commitment linear-combination interface: its commitment witness is a short
gadget-decomposed lattice object, and arbitrary transcript-scalar linear
combinations do not preserve the short-preimage structure required by the
protocol.

The Akita path instead commits to a packed polynomial:

```text
P_pack(x, y) = sum_i eq(y, i) * P_i(x)
```

Stage 8 then samples a selector point `rho` and reduces many final claims to one
packed opening:

```text
P_pack(r, rho) = sum_i eq(rho, i) * P_i(r)
```

This spec owns the trait split and verifier/prover flow for that replacement.
It also identifies the separate Jolt protocol-layout change needed for
one-hotting increment polynomials in the lattice path.

## Scope

V1 scope:

```text
generic batch-opening verifier/prover trait in jolt-openings
Dory batch-opening implementation via existing additive homomorphic RLC
Akita batch-opening implementation via eq-selector packed openings
jolt-verifier stage-8 dispatch through the generic batch-opening trait
stage-8 return of per-claim logical coefficients for BlindFold constraints
lattice/Akita final-opening manifest using one-hot increment semantics
jolt-akita crate ownership of packed commitment layout and proof verification
```

Out of scope:

```text
general multipoint Akita incidence batching
making jolt-verifier aware of Akita relation internals
requiring Dory to use packed commitments
removing Dory's AdditivelyHomomorphic capability
porting all of lz/integrate-hachi implementation details
ZK support for Akita if the selected lattice PCS initially targets transparent proofs only
```

V1 assumes that prior Jolt PIOP stages reduce final PCS claims to the simplest
shape:

```text
many P_i opened at one common final point r
```

The public trait may use the word "batch", but it does not need to expose a
general multipoint model in the first patch.

## Boundary Model

The ownership split is:

```text
jolt-verifier:
  constructs the typed final-opening manifest for stage 8
  computes Jolt-level embedding scales
  calls a generic BatchOpening verifier trait
  consumes returned per-claim logical coefficients for BlindFold/stage outputs
  does not know whether the PCS used RLC or packed selector reduction

jolt-openings:
  CommitmentScheme base trait
  AdditivelyHomomorphic lower-level capability
  BatchOpening verifier/prover traits
  common request/result data structures
  optional helper/adapter for homomorphic RLC batch openings

jolt-dory:
  implements AdditivelyHomomorphic as today
  implements BatchOpening through the homomorphic RLC helper
  preserves the existing Dory stage-8 transcript semantics

jolt-akita:
  owns packed main commitment construction
  owns lane order and packed layout metadata
  implements BatchOpening through eq-selector reduction
  verifies Akita proof at the packed point (r, rho)

jolt-claims:
  owns protocol-level final-opening manifests
  owns lattice one-hot increment semantics
  owns IDs/dimensions needed to map stage outputs into final opening lanes
```

`jolt-verifier` must not match on an Akita proof shape or call an
Akita-specific selector API directly. The selected PCS implementation owns the
physical batching strategy.

## Protocol Shape

### Dory RLC Batch

Dory commits independently:

```text
C_i = Com(P_i)
```

At stage 8, the verifier has final claims and embedding scales:

```text
P_i(r_i) = v_i
scale_i maps the logical claim into the common PCS opening domain
v'_i = scale_i * v_i
```

For the current standard Dory layout, dense increment polynomials are shorter
than the one-hot RA polynomials. They are embedded into the full address-cycle
domain by:

```text
scale_i = eq(r_address, 0) = product_j (1 - r_address_j)
```

Dory batch verification samples `gamma_i` and verifies:

```text
C* = sum_i gamma_i C_i
v* = sum_i gamma_i v'_i
Verify(C*, r, v*)
```

The logical coefficient returned for claim `i` is:

```text
lambda_i = gamma_i * scale_i
```

These coefficients are the values consumed by BlindFold stage-8 output
constraints.

### Akita Eq-Packed Batch

Akita commits to a packed main witness polynomial. Given a fixed lane order:

```text
lane 0 -> P_0
lane 1 -> P_1
...
lane m-1 -> P_{m-1}
```

the committed object is:

```text
P_pack(x, y) = sum_{i=0}^{m-1} eq(y, i) * P_i(x)
C_pack = AkitaCommit(P_pack)
```

The selector challenge is not known at commit time. Commit time fixes only:

```text
lane order
packed polynomial source
packing bit layout
commitment hint
public commitment C_pack
```

At stage 8, after Jolt has produced claims `P_i(r) = v_i`, Akita batch
verification samples:

```text
rho in F^{ceil(log2(m))}
```

and computes:

```text
theta_i = eq(rho, i)
v_pack = sum_i theta_i * scale_i * v_i
r_pack = (r, rho) after the Akita layout's coordinate ordering
VerifyAkita(C_pack, r_pack, v_pack)
```

The logical coefficient returned for claim `i` is:

```text
lambda_i = theta_i * scale_i
```

For the intended lattice main-witness path, increment polynomials are converted
to one-hot lanes, so their stage-8 scales should be `1` rather than the Dory
dense embedding factor.

## Trait Design

### Batch Opening Request

The verifier-facing request is a one-point batch in V1.

```rust
pub struct BatchOpeningItem<F, C> {
    pub commitment: C,
    pub eval: F,
    pub scale: F,
}

pub struct BatchOpeningRequest<F, C> {
    pub point: Vec<F>,
    pub items: Vec<BatchOpeningItem<F, C>>,
}
```

`items` are flattened in the canonical final-opening manifest order. The batch
trait returns coefficients in exactly this order.

The `commitment` field may repeat. For Dory, each logical lane usually has its
own commitment. For Akita packed main openings, all main witness lanes point to
the same packed commitment. Advice or other non-packed lanes may be separate in
future extensions.

### Batch Opening Result

```rust
pub struct BatchOpeningResult<F, C, H = ()> {
    pub logical_coefficients: Vec<F>,
    pub reduced_eval: Option<F>,
    pub reduced_commitment: Option<C>,
    pub hiding_eval_commitment: Option<H>,
}
```

Semantics:

```text
logical_coefficients[i] multiplies the raw logical Jolt opening value v_i
```

Examples:

```text
Dory:
  logical_coefficients[i] = gamma_i * scale_i
  reduced_eval = Some(sum_i gamma_i * scale_i * v_i)
  reduced_commitment = Some(sum_i gamma_i C_i)

Akita packed:
  logical_coefficients[i] = eq(rho, i) * scale_i
  reduced_eval = Some(sum_i eq(rho, i) * scale_i * v_i)
  reduced_commitment = Some(C_pack)
```

The `hiding_eval_commitment` field is used only when the selected PCS supports
ZK evaluation hiding and the selected verifier path needs to feed that value
into BlindFold.

### Verifier Trait

```rust
pub trait BatchOpeningScheme: CommitmentScheme {
    type BatchProof: Clone + core::fmt::Debug + Eq + Send + Sync + 'static;

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

If the existing proof type is already the batch proof, implementations may set:

```rust
type BatchProof = Self::Proof;
```

`JoltProof` should carry:

```rust
pub joint_opening_proof: PCS::BatchProof
```

instead of assuming the singleton `PCS::Proof` type is the final batch-opening
proof.

### Prover Trait

The prover side needs a corresponding batch-opening API. It may live in the
same trait or in a split prover trait, depending on the final `jolt-openings`
crate direction.

```rust
pub trait BatchOpeningProver: BatchOpeningScheme {
    type BatchOpeningHint;
    type BatchPolynomialSource;

    fn prove_batch<T>(
        setup: &Self::ProverSetup,
        transcript: &mut T,
        source: &Self::BatchPolynomialSource,
        request: ProverBatchOpeningRequest<Self::Field, Self::Output>,
        hint: Self::BatchOpeningHint,
    ) -> Self::BatchProof
    where
        T: Transcript<Challenge = Self::Field>;
}
```

The source type is intentionally abstract. Dory can use a streaming RLC source.
Akita can use a lazy packed one-hot source.

## Homomorphic Implementation Strategy

`AdditivelyHomomorphic` remains a lower-level PCS capability:

```rust
pub trait AdditivelyHomomorphic: CommitmentScheme {
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output;
    fn combine_hints(hints: Vec<Self::OpeningHint>, scalars: &[Self::Field]) -> Self::OpeningHint;
}
```

The batch-opening implementation for homomorphic schemes is:

```text
1. flatten request.items
2. compute effective evals v'_i = scale_i * eval_i
3. absorb v'_i according to the existing Dory transcript labels
4. sample gamma_i
5. combine commitments with gamma_i
6. verify one opening proof at request.point
7. return logical_coefficients_i = gamma_i * scale_i
```

This may be implemented as a blanket impl if Rust coherence permits, or as an
adapter/helper used by `jolt-dory`. The architectural requirement is that Dory's
observable transcript and proof semantics remain unchanged.

## Packed Implementation Strategy

Akita's packed implementation has two layers.

### Packed Commit Layer

The packed commit layer fixes the lane manifest before stage challenges:

```text
PackedMainManifest:
  lane_count
  lane IDs
  lane order
  base point variable count
  selector variable count
  packed coordinate ordering
  commitment arity
```

For V1, main witness lanes should be represented by a single packed commitment:

```text
commitment arity = 1
```

The packed polynomial must be lazy/sparse. The implementation must not
materialize the dense table of size:

```text
T * K * next_power_of_two(lane_count)
```

Instead it should expose per-cycle/per-lane one-hot indices to Akita's sparse
commitment kernels.

### Eq-Selector Opening Layer

Akita `verify_batch` uses the packed manifest to:

```text
1. validate that request.items match the packed lane count/order
2. absorb batch shape, packed commitment, point, and scaled evaluations
3. sample selector point rho
4. compute theta_i = eq(rho, i)
5. compute v_pack = sum_i theta_i * scale_i * eval_i
6. derive packed opening point (r, rho)
7. verify the Akita opening proof against C_pack
8. return logical_coefficients_i = theta_i * scale_i
```

The dummy selector lanes introduced by padding `lane_count` to a power of two
are zero lanes. They do not appear in the Jolt final-opening manifest and do not
produce returned logical coefficients.

## Final Opening Manifest

The final-opening manifest is the Jolt protocol object that tells stage 8 which
logical claims are opened and in what order.

Standard Dory manifest:

```text
RamInc at IncClaimReduction
RdInc at IncClaimReduction
FieldInlineRdInc if enabled
InstructionRa[0..instruction_d) at HammingWeightClaimReduction
BytecodeRa[0..bytecode_d) at HammingWeightClaimReduction
RamRa[0..ram_d) at HammingWeightClaimReduction
TrustedAdvice if present
UntrustedAdvice if present
```

Lattice/Akita manifest:

```text
RdIncRa[0..inc_d) at HammingWeightClaimReduction
RdIncMsb at HammingWeightClaimReduction
RamIncRa[0..inc_d) at HammingWeightClaimReduction
RamIncMsb at HammingWeightClaimReduction
InstructionRa[0..instruction_d) at HammingWeightClaimReduction
BytecodeRa[0..bytecode_d) at HammingWeightClaimReduction
RamRa[0..ram_d) at HammingWeightClaimReduction
TrustedAdvice if supported/present
UntrustedAdvice if supported/present
```

The lattice manifest is selected by a protocol configuration axis, not by
inspecting proof contents at runtime.

## One-Hot Increment Semantics

One-hot increment conversion is not part of the batch-opening trait. It is a
Jolt protocol-layout option required for efficient Akita packing.

Dory currently commits dense signed increment polynomials:

```text
RdInc(cycle)
RamInc(cycle)
```

The lattice path commits one-hot increment chunks. Define:

```text
unsigned_rd_inc = rd_inc + 2^XLEN
unsigned_ram_inc = ram_inc + 2^XLEN
```

Then commit:

```text
RdIncRa[j]      = j-th low chunk of unsigned_rd_inc
RdIncMsb        = bit XLEN of unsigned_rd_inc
RamIncRa[j]     = j-th low chunk of unsigned_ram_inc
RamIncMsb       = bit XLEN of unsigned_ram_inc
```

The exact chunk count and chunk size should be defined in `jolt-claims` so
prover, verifier, and tests derive the same manifest. With one-hot increments,
stage 8 no longer needs the dense Dory embedding scale for increments:

```text
scale = 1
```

## Verifier Flow

Stage 8 verifier flow becomes:

```text
1. derive final-opening manifest from proof protocol config
2. collect typed stage-6/stage-7 output claims
3. compute per-item embedding scale
4. construct BatchOpeningRequest
5. call PCS::verify_batch
6. bind the batch-opening result to the transcript through PCS-defined labels
7. return Stage8Output carrying:
     opening IDs
     logical coefficients
     common opening point
     PCS opening point or packed opening metadata when required
     reduced commitment/evaluation when meaningful
     hiding evaluation commitment when present
```

`jolt-verifier` should no longer call:

```rust
PCS::combine(...)
```

directly in stage 8.

## Proof Shape

The native proof model should be parameterized over the PCS batch proof:

```rust
pub struct JoltProof<PCS, VC, ZkProof, PcsAssist>
where
    PCS: BatchOpeningScheme,
{
    pub joint_opening_proof: PCS::BatchProof,
    ...
}
```

Commitment payload shape must be protocol-configured:

```text
Dory:
  one commitment per final main witness polynomial

Akita:
  one packed main commitment for the packed main witness lanes
  optional separate commitments for advice or other unsupported lanes
```

`compat` remains responsible for translating legacy core proof commitments into
the native proof shape for Dory. Akita should not rely on legacy Dory commitment
ordering.

## Invariants

- The final-opening manifest order is identical for prover and verifier.
- `BatchOpeningResult::logical_coefficients.len() == request.items.len()`.
- Returned logical coefficients multiply raw logical Jolt openings, not already
  scaled values.
- Dory batch opening transcript compatibility is preserved.
- Akita selector challenge `rho` is sampled after the packed commitment and
  final claimed values are transcript-bound.
- Akita commit-time packing fixes lane order before any selector challenge is
  sampled.
- Dummy selector lanes are zero and are not exposed as Jolt logical claims.
- Lattice one-hot increment semantics are selected by protocol config and are
  reflected consistently in prover witness generation, verifier claim
  collection, and final-opening manifests.
- In ZK mode, any stage-8 hidden evaluation commitment must be bound to the same
  logical coefficients used by BlindFold output constraints.

## Acceptance Criteria

- [ ] `jolt-verifier` stage 8 depends on `BatchOpeningScheme` rather than
      directly requiring `AdditivelyHomomorphic`.
- [ ] Dory proofs generated by the existing prover continue to verify with the
      same transcript ordering and final opening proof semantics.
- [ ] The Dory implementation returns the same stage-8 logical coefficients as
      the current `gamma_i * scale_i` path.
- [ ] A lattice/Akita proof can carry a packed main commitment and verify the
      final batch through an eq-selector opening at `(r, rho)`.
- [ ] The lattice final-opening manifest uses one-hot increment lanes rather
      than dense `RdInc`/`RamInc`.
- [ ] Dummy packed selector lanes are not materialized in commitment generation
      and are not exposed as verifier opening IDs.
- [ ] Standard mode and ZK mode either both pass the supported PCS matrix, or
      unsupported combinations reject during proof/config validation.

## Testing Strategy

Required Dory compatibility checks:

```bash
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
cargo nextest run -p jolt-verifier --cargo-quiet --features host
```

New verifier tests:

```text
stage8_batch_opening_dory_matches_existing_rlc
stage8_batch_opening_rejects_tampered_logical_coefficient
stage8_batch_opening_rejects_tampered_batch_proof
stage8_manifest_dory_dense_inc_order_matches_core
stage8_manifest_lattice_onehot_inc_order_is_stable
```

New Akita tests:

```text
packed_eq_reduction_matches_direct_packed_evaluation
packed_opening_rejects_wrong_lane_order
packed_opening_rejects_wrong_selector_claim
packed_opening_does_not_materialize_dummy_lanes
lattice_muldiv_final_opening_manifest_uses_onehot_inc
```

If Akita ZK is not supported in V1, add a config validation test that rejects:

```text
zk + lattice/Akita
```

with a specific error.

## Performance

Dory performance should not regress beyond normal refactor noise. The Dory path
must continue to use the existing streaming RLC polynomial construction and
homomorphic commitment/hint combination.

Akita packed commitment must avoid dense selector-domain materialization. The
expected commit-side scan cost for one-hot main witness lanes is proportional to:

```text
T * lane_count
```

not:

```text
T * K * next_power_of_two(lane_count)
```

Padding costs that remain acceptable in V1:

```text
selector variable count ceil(log2(lane_count))
ring coefficient padding to D
layout/planner padding required by Akita
```

Padding costs that are not acceptable:

```text
materialized dummy selector lanes
dense materialization of the packed one-hot table
per-lane full dense commit work for one-hot witness lanes
```

## Alternatives Considered

### Keep AdditivelyHomomorphic As Stage-8 Bound

This preserves the current Dory flow but excludes Akita. Akita cannot soundly
implement arbitrary transcript-scalar post-commitment RLC without changing the
short-witness relation.

### Expose Akita Incidence Directly In jolt-verifier

Akita's native multipoint incidence model is more general, but exposing it in
`jolt-verifier` is unnecessary for V1 because Jolt final claims are already
reduced to one common point. It would also leak Akita-specific protocol layout
into the generic verifier.

### Make Mega-Polynomial The Generic Trait

A generic packed-polynomial trait would match the Akita main witness path, but
it would force Dory and future schemes to model an implementation strategy they
do not need. The batch-opening trait is the generic boundary; packing is an
Akita implementation strategy beneath it.

### Use Independent Akita Commitments And Batch In The Inner Relation

This is conceptually valid for Akita, but it is likely less efficient for
Jolt's homogeneous one-hot main witness because it carries multiple commitment
witnesses through the relation. The packed main commitment keeps the efficient
one-hot commit path while preserving a generic verifier boundary.

## Documentation

Book updates:

```text
book/src/how/architecture/opening-proof.md:
  explain stage-8 BatchOpening abstraction
  distinguish Dory RLC from Akita eq-packed reduction

book/src/how/appendix/pcs.md:
  add Akita/lattice packed opening description
  document why AdditivelyHomomorphic is not a generic PCS requirement

book/src/how/optimizations/batched-openings.md:
  add one-hot packed commitment notes and padding model
```

If the lattice path is experimental, mark the book section as experimental and
link to this spec.

## Execution

Suggested implementation order:

```text
1. Add BatchOpening request/result structs and verifier trait to jolt-openings.
2. Implement Dory BatchOpening through the existing RLC path.
3. Change jolt-verifier stage 8 to call BatchOpening and consume returned
   logical coefficients.
4. Add final-opening manifest abstraction in jolt-claims.
5. Add lattice manifest variant with one-hot increment lanes.
6. Add jolt-akita packed commitment/proof adapter.
7. Wire lattice feature/config to select Akita PCS and lattice manifest.
8. Add compatibility and tampering tests.
```

The Dory refactor should land before Akita wiring so existing verifier behavior
is protected by tests while the generic trait boundary is introduced.

## References

- [jolt-verifier model crate](jolt-verifier-model-crate.md)
- [selected verifier integration](selected-verifier-integration.md)
- [dory assist protocol](dory-assist-protocol.md)
- `lz/integrate-hachi` branch in `LayerZero-Research/jolt`
- `LayerZero-Labs/akita`, especially `CommitmentProver::batched_commit`,
  `CommitmentProver::batched_prove`, and verifier-side batched claims
