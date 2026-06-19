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
- Akita lattice mode uses one proof-owned PackedWitness commitment for
  non-precommitted packed facts.
- Precommitted facts, such as TrustedAdvice, BytecodeChunk(i), and
  ProgramImageInit, keep their original commitments and require separate
  openings unless a future protocol proves an explicit binding to W_pack.
- Separate precommitted opening means a distinct opening statement/proof
  against the original commitment. It does not mean another logical term inside
  the W_pack packed-view batch.
- The packed-view reduction is generic with respect to the PCS family. Akita is
  one backend that can satisfy it; later hash-based PCS modes may also use the
  packed option.
```

## Architecture

Trait sketch:

```rust
pub trait BatchOpeningScheme: CommitmentScheme {
    /// Prove many logical openings at one verifier point.
    fn prove_batch<T, OpeningId, RelationId>(
        setup: &Self::ProverSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
        polynomials: &[Self::Polynomial],
        hints: Vec<Self::OpeningHint>,
    ) -> Result<Self::Proof, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>;

    /// Verify many logical openings at one verifier point.
    fn verify_batch<T, OpeningId, RelationId>(
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
        proof: &Self::Proof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>;
}

pub trait ZkBatchOpeningScheme: BatchOpeningScheme + ZkOpeningScheme {
    fn prove_batch_zk<T, OpeningId, RelationId>(
        setup: &Self::ProverSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId, ()>,
        evals: &[Self::Field],
        polynomials: &[Self::Polynomial],
        hints: Vec<Self::OpeningHint>,
    ) -> Result<(Self::Proof, Self::HidingCommitment, Self::Blind), OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>;

    fn verify_batch_zk<T, OpeningId, RelationId>(
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId, ()>,
        proof: &Self::Proof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output, Self::HidingCommitment>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>;
}
```

Verifier statement:

```text
points:
  logical Jolt opening point.
  PCS physical opening point.

opening claims:
  opening ID
  relation ID
  commitment reference
  commitment class: proof-owned packed witness or precommitted object
  clear claimed value in clear mode; no claim payload in ZK verifier statements
  physical view formula
  optional embedding scale from prior reductions

layout:
  deterministic logical final-opening order
  physical view resolution table
  protocol config digest
  PackedWitness layout digest when the PCS family is lattice
```

Precommitted commitment class:

```text
Precommitted claims are not alternate views of the proof-owned PackedWitness.
Their statement commitment reference is the original TrustedAdvice,
BytecodeChunk(i), ProgramImageInit, or derived BytecodeChunk component
commitment. The batch-opening implementation may group compatible
precommitted claims in a direct/native opening batch, but it must preserve those
original commitment handles and must not satisfy them with W_pack or with only
an Akita backend Program::Committed bytecode handle.
```

Prover method inputs:

```text
verifier statement
polynomials for the logical openings
opening hints or witness handles for the PCS's physical batch-opening strategy
optional PackedWitness source handle for Akita
```

Implementation-owned reductions:

```text
logical_coefficients:
  coefficients multiplying raw Jolt logical claims; returned so Stage 8 can
  keep the existing clear and BlindFold constraint binding.

joint_statement:
  PCS-specific statement verified internally.

joint_claim:
  optional PCS-reduced evaluation used by Dory-style compatibility checks
  returned by clear verification for current Stage 8 compatibility.

joint_commitment:
  PCS-reduced commitment returned for current Stage 8 compatibility.

hidden_claim_commitment:
  Dory ZK evaluation-hiding commitment returned by verify_batch_zk. Akita
  lattice mode rejects ZK until a lattice hiding protocol is specified.

transcript_binding:
  use existing bind_opening_inputs and bind_zk_opening_inputs hooks unless the
  implementation proves they are insufficient.

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
  uses the generic packed-view reduction and supplies the Akita-specific
  physical proof backend.
  verifies precommitted objects through separate direct/native opening
  statements keyed by their original commitments.
  treats precommitted opening proofs as direct openings, not packed-view
  reductions over W_pack.
```

Batching strategy taxonomy:

```text
BatchOpeningScheme:
  PCS-level extension trait used by Stage 8.
  proves and verifies many logical polynomial openings at the same verifier
  point.
  uses the PCS's existing Self::Proof type.

ZkBatchOpeningScheme:
  ZK companion extension trait mirroring ZkOpeningScheme.
  uses Self::Proof, Self::HidingCommitment, and Self::Blind.

HomomorphicBatching:
  implementation pattern for PCS modes that implement AdditivelyHomomorphic.
  reduces m same-point claims to one joint claim by RLC.
  Dory uses this path.
  If implemented as a Rust blanket impl, alternate implementations for an
  additively homomorphic PCS must use a wrapper/newtype to avoid coherence
  conflicts.

PackedViewBatching:
  implementation pattern for PCS modes whose physical commitment already
  encodes many logical objects.
  reduces m same-point logical claims to one packed-view opening relation.
  generic with respect to the PCS family.
  Akita and later hash PCS modes may use this path.
  This is not a required public trait in spec 01.

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
6. derive gamma_i * embedding_scale_i as internal logical coefficients.
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
1. partition final openings by physical commitment class.
2. transcript-bind PackedWitness layout, packed physical views, point, and
   claims.
3. use the generic packed-view reduction to form one packed-view opening
   relation for W_pack claims.
4. verify the Akita physical proof for that relation.
5. build and verify separate precommitted opening statements against the
   original TrustedAdvice, BytecodeChunk(i), and ProgramImageInit commitments.
   These statements use direct/native physical views and carry no PackedWitness
   reduction proof.
6. derive logical coefficients internally for claim binding.
```

Only the W_pack partition is an Akita packed-view relation. The precommitted
partition is required because Akita does not provide the additive homomorphism
needed to batch unrelated commitments into W_pack after the commitments were
already fixed. Copying precommitted values into W_pack can be useful only for a
future bound packed-precommitted view; in this target it is not a proof of the
original precommitted commitment.

Precommitted opening partition:

```text
input:
  final logical opening manifest

for each opening:
  if opening targets proof-owned packed witness data:
    add to the W_pack packed-view statement
  if opening targets TrustedAdvice, BytecodeChunk(i), ProgramImageInit, or a
  source fact derived from BytecodeChunk(i):
    add a precommitted opening statement keyed by the original commitment handle

requirements:
  every precommitted statement has its own direct/native proof material.
  precommitted proofs are ordered by the precommitted opening manifest.
  the verifier rejects missing, extra, reordered, W_pack-backed, or
  backend-program-only proofs for precommitted statements.
```

Akita non-requirements:

```text
1. no requirement to compute C_joint = sum_i gamma_i * C_i.
2. no requirement that every logical claim maps to a standalone physical
   commitment.
3. no requirement that W_pack is materialized.
4. no requirement that current Akita public point-opening APIs are sufficient;
   jolt-akita may supply a packed-view adapter.
```

Akita rejected behavior:

```text
1. satisfying a TrustedAdvice, BytecodeChunk(i), or ProgramImageInit final claim
   only by opening W_pack.
2. using the generic packed-view reduction to form one packed-view opening
   relation that mixes precommitted commitments into the proof-owned packed
   witness without an explicit binding protocol.
3. accepting W_pack openings of copied trusted advice, bytecode, or program
   image values as openings of their original commitments.
```

Same-point scope:

```text
Jolt supplies logical same-point openings.
The PCS may internally open:
  the same physical point,
  a derived packed point,
  or a reduced relation point.

The PCS implementation's logical coefficients still bind the original Jolt
claims.
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
  precommitted commitments are planned and bound separately before Stage 8.
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

Current main integration:

```text
Stage 8 sets BatchOpeningStatement::layout_digest to the verifier
preprocessing digest. This is a deterministic public anchor available on main.
It is not the final PackedWitness layout digest; specs 03/04 replace it once
the PackedWitness planner exists.

PackedCombine binds its layout digest, points, scales, and physical-view
coefficients before delegating to the inner batch-opening PCS. The Dory
homomorphic blanket implementation preserves the legacy Stage 8 transcript
schedule for compatibility with the current core prover.
```

Implementation plan:

```text
jolt-openings:
  define BatchOpeningScheme and ZkBatchOpeningScheme in schemes.rs.
  introduce BatchOpeningStatement.
  introduce BatchOpeningClaim.
  introduce PhysicalView as the only new small enum.
  implement the homomorphic/Dory path behind BatchOpeningScheme for PCS modes
  with AdditivelyHomomorphic.
  keep AdditivelyHomomorphic as an optional capability.
  do not introduce a PackedWitness trait, PackedViewBatching trait, or generic
  ID trait unless a later spec requires one.

jolt-verifier:
  replace direct Stage 8 PCS::combine calls with BatchOpeningScheme.
  build the logical final-opening manifest before the PCS call.
  pass relation IDs and view formulas with every opening claim.

Dory:
  implement BatchOpeningScheme using existing RLC, commitment combine, and
  opening proof.
  preserve current transcript labels and claim order.

Akita:
  implement BatchOpeningScheme without AdditivelyHomomorphic by satisfying the
  generic packed-view reduction.
  reject unsupported view formulas until PackedWitness layout and logical-view
  formulas exist.
```

Implementation sequence:

```text
1. Characterize current Stage 8:
   add or confirm tests for final-opening order, scaled claims, gamma powers,
   dense-increment scale, joint claim, and tampered-claim rejection.
   verify with the narrow Stage 8 and opening tests.

2. Add the minimum batch-opening API:
   define BatchOpeningScheme plus BatchOpeningStatement,
   BatchOpeningClaim, and PhysicalView in jolt-openings/src/schemes.rs.
   define ZkBatchOpeningScheme to mirror ZkOpeningScheme if Dory ZK Stage 8 is
   migrated in the same pass.
   use Self::Proof for batch proofs.
   keep IDs as existing protocol IDs at Stage 8 boundaries; avoid a new ID trait.

3. Implement the homomorphic/Dory-compatible path:
   move the current RLC transcript binding, gamma sampling, commitment combine,
   claim combine, opening verify/open, and transcript binding behind
   BatchOpeningScheme.
   preserve current labels, ordering, and formula.

4. Build the Stage 8 logical manifest:
   reuse stage8_final_opening_order and existing final-claim helpers.
   attach opening ID, relation ID, raw claim, commitment handle,
   physical view, and scale to each logical opening.

5. Wire verifier clear mode:
   call BatchOpeningScheme::verify_batch instead of PCS::combine and PCS::verify.

6. Wire prover clear mode:
   call BatchOpeningScheme::prove_batch instead of directly combining hints and
   opening a Stage 8-owned joint proof.
   mirror the AdditivelyHomomorphic style: the PCS-side batch implementation owns
   how hints or physical opening witnesses are combined.

7. Wire Dory ZK mode:
   use ZkBatchOpeningScheme for open_zk/verify_zk-style batch openings and
   hidden evaluation commitments.
   use BatchOpeningStatement with an empty claim payload for verifier-side ZK
   statements; pass hidden evals only to the prover method.
   Akita lattice ZK remains explicitly rejected.

8. Relax Stage 8 trait bounds:
   remove AdditivelyHomomorphic from Stage 8-facing bounds once both Dory paths
   compile and pass.
   keep AdditivelyHomomorphic bounds only inside the Dory/homomorphic adapter.

9. Add a packed-combine implementation path:
   implement a PCS-generic PackedCombine-style batch adapter that satisfies
   BatchOpeningScheme without exposing AdditivelyHomomorphic to Stage 8.
   use a wrapper/newtype when the underlying PCS also satisfies the homomorphic
   blanket impl.
   use Dory where appropriate to exercise the generic packed interface shape
   before jolt-akita exists: one physical commitment handle, many logical
   claims, deterministic IDs, and clean rejection of unsupported views.
   this does not claim to test Akita's real PackedWitness relation.
```

Minimal abstraction set:

```text
Public trait:
  BatchOpeningScheme.
  ZkBatchOpeningScheme, mirroring ZkOpeningScheme.

Required data types:
  BatchOpeningStatement.
  BatchOpeningClaim.

Small enums/value types:
  PhysicalView.
  BatchOpeningResult, because Stage 8 still needs the verified joint commitment,
  reduced opening, and logical coefficients.

Optional existing capability:
  AdditivelyHomomorphic remains a separate optional PCS capability used by the
  Dory/homomorphic adapter only.

Not introduced in spec 01:
  PackedWitness trait.
  PackedViewBatching public trait.
  HomomorphicBatching public trait, unless Rust coherence forces a newtype.
  generic opening-ID or relation-ID trait.
  multipoint opening abstraction.
  Akita ZK abstraction.
  separate batch proof associated type.
```

Proposed Rust data model:

```rust
pub struct BatchOpeningStatement<F, C> {
    pub logical_point: Vec<F>,
    pub pcs_point: Vec<F>,
    pub layout_digest: [u8; 32],
    pub claims: Vec<BatchOpeningClaim<F, C>>,
}

pub struct BatchOpeningClaim<F, C> {
    pub id: OpeningId,
    pub relation: RelationId,
    pub commitment: C,
    pub claim: F,
    pub view: PhysicalView,
    pub scale: F,
}

// ZK verifier statements use the same struct shape with claim: ().
pub struct BatchOpeningResult<F, C, R = F> {
    pub coefficients: Vec<F>,
    pub joint_commitment: C,
    pub reduced_opening: R,
}
```

Open API decisions:

```text
generic IDs:
  default to existing protocol ID types at Stage 8 boundaries.
  add associated ID types only if jolt-openings must own serialization of
  protocol-independent statements.
  do not add a generic ID trait in spec 01.

commitment ownership:
  statement may borrow commitments to avoid clones, but proof serialization may
  require owned payloads elsewhere.

clear vs ZK:
  one statement type is kept by making the claim payload generic.
  clear verifier statements use claim: F.
  ZK verifier statements use claim: () and receive the hiding commitment from
  verify_batch_zk.

transcript binding:
  use existing CommitmentScheme and ZkOpeningScheme binding hooks first.
  add a returned binding object only if the batch path cannot be expressed with
  those hooks without changing transcript semantics.
```

## Invariants

```text
- Stage 8 does not require AdditivelyHomomorphic.
- Dory Stage 8 accepts and rejects the same proofs as before.
- batch coefficients multiply raw Jolt logical claim values inside the PCS
  implementation.
- statement ordering is deterministic.
- PCS-specific proof verification cannot change the Jolt relation ID attached
  to an opening claim.
- Clear-mode opening IDs match the logical claims produced by the prior stages.
- Prover and verifier derive identical statement layouts from public config and
  transcript-bound commitments.
- Dory's BatchOpening implementation is a refactor of the existing Stage 8
  relation, not a protocol change.
- Akita's BatchOpening implementation cannot require additive commitment
  combination.
- The statement type can represent both packed-view openings and separate
  precommitted openings from ProgramMode::Committed.
- The lattice statement has one Akita commitment handle unless the verifier
  rejects the feature combination; this does not include precommitted object
  commitment handles.
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

batch_coefficients_bind_raw_claims:
  internally derived coefficients reconstruct current Stage 8 relation.

batch_statement_order_is_deterministic:
  identical inputs produce identical opening-claim order.

batch_statement_includes_relation_ids:
  same polynomial opened by different relations remains distinguishable.

clear_opening_ids_match_coefficients:
  every internally derived coefficient has the corresponding opening ID.

non_homomorphic_batch_opening_requires_no_combine:
  a wrapper or real non-homomorphic PCS can satisfy BatchOpeningScheme without
  exposing AdditivelyHomomorphic to Stage 8.

packed_combine_requires_no_stage8_combine:
  a PackedCombine-style adapter exposes no AdditivelyHomomorphic bound to
  Stage 8.

packed_combine_many_claims_one_commitment:
  the packed-style statement can represent many logical claims behind one
  physical commitment handle. Dory may be used to exercise this path before
  jolt-akita exists.

packed_combine_binds_logical_coefficients:
  internally derived coefficients bind the original logical claims even though
  Stage 8 does not see a Dory-style joint commitment.

packed_view_rejects_unsupported_formula:
  unsupported PhysicalView variants fail before proof verification.

akita_packed_view_statement_has_one_commitment:
  lattice-family statement has one PackedWitness commitment and no Stage 8
  Dory-style joint commitment requirement.

akita_precommitted_claims_use_original_commitments:
  TrustedAdvice, BytecodeChunk(i), and ProgramImageInit cannot be satisfied by
  only opening W_pack.

akita_increment_views_do_not_use_dense_scale:
  fused RamInc/RdInc claims have no Dory dense-increment embedding scale.
```

Scope note:

```text
The Dory-backed PackedCombine path tests the packed interface boundary only. It
does not test Akita's real PackedWitness short-witness relation, which requires
later PIOP and jolt-akita specs.
```

Targeted command filters:

```bash
cargo nextest run -p jolt-openings batch_opening --cargo-quiet
cargo nextest run -p jolt-verifier stage8 --cargo-quiet
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
  proof-owned logical batching cost is one packed-view relation over W_pack.
  precommitted claims pay their separate opening cost.

Hash PCS:
  may reuse the same generic packed-view reduction if it can prove the resulting
  packed relation.
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

## Resolved API Decisions

```text
- batch proof type is Self::Proof.
- clear batch openings use BatchOpeningScheme.
- ZK batch openings mirror ZkOpeningScheme through ZkBatchOpeningScheme.
- batch opening APIs live in jolt-openings/src/schemes.rs unless the file
  becomes unwieldy.
- transcript binding uses existing PCS hooks unless proven insufficient.
- no new generic ID types or ID traits in spec 01.
- prover hints/witness handles mirror the AdditivelyHomomorphic extension-trait
  style instead of adding Stage 8-specific plumbing to jolt-openings.
- packed-view reduction is generic with respect to the PCS family; jolt-akita
  does not own this reduction, it supplies an Akita backend for it.
```

## References

```text
- ../selected-verifier-integration.md: verifier selection model.
- ../jolt-verifier-model-crate.md: modular verifier target.
- https://github.com/a16z/jolt: current Dory Stage 8 implementation.
```
