# Spec: Akita Opening Trait System

| Field | Value |
|-------|-------|
| Component | opening trait system |
| Depends On | 00-roadmap.md |
| Unlocks | jolt-akita, verifier config |
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-06-08 |
| Status | draft |

## Scope

Stage 8 should depend on a generic batch-opening trait, not on additive
homomorphism. Dory keeps its current RLC behavior. Akita supplies a different
batch-opening implementation.

In scope:

```text
- BatchOpening trait shape
- prover and verifier statement/output model
- Dory implementation through additive homomorphism
- Akita-compatible statement/output shape
- Stage 8 removal of direct PCS::combine use
- clear-mode coefficient/ID handoff requirements
```

Out of scope:

```text
- Akita packed-view protocol details
- concrete jolt-akita crate internals
- multipoint opening support beyond the final Jolt use case
- Akita ZK opening support
```

Assumptions:

```text
- Jolt final openings are same-point logical openings after prior PIOP
  reductions.
- Dory remains additively homomorphic.
- Akita is not modeled as post-commitment additive RLC.
  Arbitrary transcript-scalar commitment combinations do not preserve the
  Akita short-witness relation.
- Stage 8 owns the logical final-opening manifest.
- PCS implementations own the physical opening strategy.
- Akita lattice mode uses one PackedWitness commitment for all supported
  lattice-visible committed facts.
```

## Architecture

Trait sketch:

```rust
pub trait BatchOpeningScheme: CommitmentScheme {
    type BatchProof;
    type BatchProverHint;

    fn prove_batch<T>(
        setup: &Self::ProverSetup,
        transcript: &mut T,
        prover_input: BatchOpeningProverInput<Self::Field, Self::Output, Self::BatchProverHint>,
    ) -> Result<(Self::BatchProof, BatchOpeningResult<Self::Field, Self::Output>), OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>;

    fn verify_batch<T>(
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        statement: BatchOpeningStatement<Self::Field, Self::Output>,
        proof: &Self::BatchProof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>;
}
```

Verifier statement:

```text
point:
  logical Jolt opening point.

opening claims:
  opening ID
  relation ID
  commitment reference
  claimed value, or hidden claim handle in ZK mode
  physical view formula
  optional embedding scale from prior reductions

layout:
  deterministic logical final-opening order
  physical view resolution table
  protocol config digest
  PackedWitness layout digest when the PCS family is lattice
```

Prover opening input:

```text
all verifier statement fields
opening hints/witness handles for physical commitments
optional PackedWitness source handle for Akita
```

Result:

```text
logical_coefficients:
  coefficients multiplying raw Jolt logical claims.

joint_statement:
  PCS-specific verified statement.

joint_claim:
  optional PCS-reduced evaluation used by Dory-style compatibility checks.

joint_commitment:
  optional PCS-reduced commitment used by homomorphic implementations.

hidden_claim_commitment:
  optional Dory ZK evaluation-hiding commitment. Akita lattice mode rejects ZK
  until a lattice hiding protocol is specified.

transcript_binding:
  PCS-specific opening inputs to bind after verification.

opening_ids:
  IDs used by the clear verifier path and future ZK verifier constraints.
```

Separation of responsibilities:

```text
jolt-verifier:
  derives logical opening IDs.
  derives logical claims from stage outputs.
  validates config.
  calls BatchOpeningScheme.

jolt-openings:
  defines statement/output traits.
  defines generic error and transcript binding hooks.

Dory implementation:
  uses additive homomorphism internally.

Akita implementation:
  uses PackedWitness layouts and view formulas internally.
```

Batching strategy taxonomy:

```text
BatchOpeningScheme:
  PCS-level interface used by Stage 8.

HomomorphicBatching:
  generic implementation for PCS modes that implement AdditivelyHomomorphic.
  reduces m same-point claims to one joint claim by RLC.
  Dory uses this path.

PackedViewBatching:
  generic implementation for PCS modes whose physical commitment already
  encodes many logical objects.
  reduces m same-point logical claims to one packed-view opening relation.
  Akita uses this path.

Stage 8:
  sees BatchOpeningScheme only.
  does not branch on additive homomorphism.
```

Dory implementation:

```text
1. transcript-bind raw opening claims in current order.
2. sample gamma powers.
3. combine commitments homomorphically.
4. combine claimed evaluations linearly.
5. verify one Dory opening.
6. return gamma_i * embedding_scale_i as logical coefficients.
```

Dory formula:

```text
C_joint = sum_i gamma_i * C_i
y_joint = sum_i gamma_i * scale_i * y_i

verify C_joint(r) = y_joint

logical_coeff_i = gamma_i * scale_i
```

Dory dense-increment scale:

```text
scale_i = eq(r_address, 0)
```

This embeds shorter dense increment polynomials into the full address-cycle
domain. Lattice fused increment views are already trace-domain views of
`W_pack`; they do not inherit this Dory dense-increment embedding scale.

Akita implementation:

```text
1. transcript-bind PackedWitness layout, physical views, point, and claims.
2. reduce logical claims to one Akita packed-view opening relation.
3. verify Akita batch proof.
4. return logical coefficients for verifier binding.
```

Akita non-requirements:

```text
- no requirement to compute C_joint = sum_i gamma_i * C_i.
- no requirement that every logical claim maps to a standalone physical
  commitment.
- no requirement that W_pack is materialized.
- no requirement that current Akita public point-opening APIs are sufficient;
  jolt-akita may supply a packed-view adapter.
```

Same-point scope:

```text
Jolt supplies logical same-point openings.
The PCS may internally open:
  the same physical point,
  a derived packed point,
  or a reduced relation point.

The returned logical coefficients still bind the original Jolt claims.
```

Commit planning:

```text
Stage 8 BatchOpening is not the commit-time planner.

Dory:
  one committed logical polynomial per ordinary commitment slot.

Akita:
  one PackedWitness commitment is planned before Stage 8 challenges.
  layout, fact families, offsets, domains, and decode formulas are fixed before
  the packed-opening challenge is sampled.
```

The commit-planning object may live outside `jolt-openings`, but the protocol
contract must exist: Akita cannot discover the packed layout from the final
opening batch alone.

Transcript:

```text
Dory:
  preserve existing Stage 8 transcript ordering unless the verifier integration
  explicitly changes it.

Akita:
  bind PackedWitness layout digest before sampling batch/packing challenges.
```

Implementation plan:

```text
jolt-openings:
  introduce BatchOpeningScheme.
  introduce HomomorphicBatching as the blanket implementation for PCS modes
  with AdditivelyHomomorphic.
  introduce PackedViewBatching as the implementation hook for packed physical
  commitments.
  introduce BatchOpeningStatement and BatchOpeningProverInput.
  introduce BatchOpeningClaim.
  introduce BatchOpeningResult.
  keep AdditivelyHomomorphic as an optional capability.

jolt-verifier:
  replace direct Stage 8 PCS::combine calls with BatchOpeningScheme.
  build the logical final-opening manifest before the PCS call.
  pass relation IDs and view formulas with every opening claim.
  consume returned logical_coefficients for clear mode.

Dory:
  implement BatchOpeningScheme using existing RLC, commitment combine, and
  opening proof.
  preserve current transcript labels and claim order.

Akita:
  implement BatchOpeningScheme without AdditivelyHomomorphic.
  reject unsupported view formulas until PackedWitness layout and logical-view
  formulas exist.
```

Proposed Rust data model:

```rust
pub struct BatchOpeningStatement<F, C> {
    pub logical_point: Vec<F>,
    pub layout_digest: [u8; 32],
    pub claims: Vec<BatchOpeningClaim<F, C>>,
}

pub struct BatchOpeningClaim<F, C> {
    pub id: OpeningId,
    pub relation: RelationId,
    pub commitment: C,
    pub claim: BatchClaim<F>,
    pub view: PhysicalView,
    pub scale: F,
}

pub enum BatchClaim<F> {
    Clear(F),
    Hidden { index: usize },
}

pub struct BatchOpeningResult<F, C, H = ()> {
    pub opening_ids: Vec<OpeningId>,
    pub logical_coefficients: Vec<F>,
    pub verified_commitment: Option<C>,
    pub reduced_claim: Option<F>,
    pub hidden_claim_commitment: Option<H>,
    pub transcript_binding: BatchTranscriptBinding<F, C>,
}
```

Open API decisions:

```text
generic IDs:
  either use protocol-specific associated ID types or a small trait for
  opening/relation IDs.

commitment ownership:
  statement may borrow commitments to avoid clones, but proof serialization may
  require owned payloads elsewhere.

clear vs ZK:
  one statement type is simpler if BatchClaim represents hidden claims.
  separate statement types are stricter if ZK needs different transcript labels.

transcript binding:
  result can return data for Stage 8 to bind, or trait method can perform all
  binding internally. The first option is easier to audit.
```

## Invariants

```text
- Stage 8 does not require AdditivelyHomomorphic.
- Dory Stage 8 accepts and rejects the same proofs as before.
- logical_coefficients multiply raw Jolt logical claim values.
- statement ordering is deterministic.
- PCS-specific proof verification cannot change the Jolt relation ID attached
  to an opening claim.
- Clear-mode opening IDs and coefficients match the logical claims produced by
  the prior stages.
- Prover and verifier derive identical statement layouts from public config and
  transcript-bound commitments.
- Dory's BatchOpening implementation is a refactor of the existing Stage 8
  relation, not a protocol change.
- Akita's BatchOpening implementation cannot require additive commitment
  combination.
- The statement type can represent committed-program openings from
  ProgramMode::Committed as PackedWitness views.
- The lattice statement has one Akita commitment handle unless the verifier
  rejects the feature combination.
```

## Tests

Targeted tests:

```text
dory_batch_opening_matches_current_rlc:
  same joint commitment, joint claim, proof input, and transcript.

dory_dense_increment_scale_is_preserved:
  dense IncClaimReduction openings use eq(r_address, 0) under curve mode.

dory_batch_opening_rejects_tampered_claim:
  altered final claim fails.

stage8_does_not_call_pcs_combine_directly:
  verifier uses BatchOpeningScheme.

batch_result_coefficients_bind_raw_claims:
  returned coefficients reconstruct current Stage 8 relation.

batch_statement_order_is_deterministic:
  identical inputs produce identical opening-claim order.

batch_statement_includes_relation_ids:
  same polynomial opened by different relations remains distinguishable.

clear_opening_ids_match_coefficients:
  every returned coefficient has the corresponding opening ID.

akita_mock_batch_opening_requires_no_combine:
  a non-homomorphic mock PCS can satisfy the trait.

akita_packed_view_statement_has_one_commitment:
  lattice-family statement has one PackedWitness commitment and no Dory-style
  joint commitment.

akita_increment_views_do_not_use_dense_scale:
  fused RamInc/RdInc claims have no Dory dense-increment embedding scale.
```

Targeted command filters:

```bash
cargo nextest run -p jolt-openings batch_opening --cargo-quiet
cargo nextest run -p jolt-verifier stage8 --cargo-quiet --features host
```

## Performance

Expected:

```text
Dory:
  no intended prover/verifier regression.
  no extra commitment clones beyond current Stage 8 order.

Akita:
  no additive-combine requirement.
  physical proof cost is delegated to jolt-akita.
  logical batching cost is one packed-view relation over W_pack.
```

Rejected:

```text
- materializing an RLC polynomial for Akita because the trait shape requires it.
- changing Dory transcript semantics as a side effect.
- exposing packed-view internals in jolt-verifier Stage 8 logic.
- requiring all PCS modes to share one commitment payload layout.
- making a packed-polynomial or PackedWitness trait the generic Stage 8
  interface for all PCS modes.
- exposing Akita multipoint incidence APIs directly in jolt-verifier for the
  same-point Jolt final-opening use case.
- representing Akita byte-decode views as ordinary same-point polynomial
  openings unless the backend proves the corresponding linear view relation.
```

## Questions

```text
1. Should BatchProof replace PCS::Proof, or should CommitmentScheme expose an
   associated final-opening proof type?
2. Should ZK and non-ZK batch statements share one type with hidden claims, or use
   separate statement types?
3. Should `BatchOpeningResult` carry PCS-specific transcript binding data as an
   enum, associated type, or callback?
4. Should prover hints live in the commitment payload or a separate
   `BatchProverHint` map keyed by opening ID?
5. Does the Akita backend expose packed linear-view statements directly, or does
   `jolt-akita` own the reduction from packed views to backend point openings?
```

## References

```text
- ../selected-verifier-integration.md: verifier selection model.
- ../jolt-verifier-model-crate.md: modular verifier target.
- https://github.com/a16z/jolt: current Dory Stage 8 implementation.
```
