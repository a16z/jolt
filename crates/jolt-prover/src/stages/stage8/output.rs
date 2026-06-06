use {
    jolt_field::Field,
    jolt_poly::{Point, HIGH_TO_LOW},
    jolt_verifier::stages::stage8::outputs::Stage8OpeningId,
};

/// Deterministic Stage 8 final-opening structure (clear path), produced before
/// the PCS opening proof is generated.
///
/// Mirrors the value-level outputs of `jolt-verifier/src/stages/stage8/verify.rs`:
/// the final opening IDs in verifier batch order, the scaled opening-claim values
/// (`opening_claim · scale`) bound to the transcript, the RLC constraint
/// coefficients (`γ^i · scale_i`), the common/PCS opening points, and the joint
/// RLC claim. The `joint_opening_proof` (PCS) and `joint_commitment` are produced
/// in a later slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage8OpeningStructure<F: Field> {
    pub opening_ids: Vec<Stage8OpeningId>,
    pub scaled_opening_values: Vec<F>,
    pub constraint_coefficients: Vec<F>,
    pub opening_point: Point<HIGH_TO_LOW, F>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
    pub joint_claim: F,
}

/// Canonical Stage 8 prover output (clear path): the deterministic opening
/// structure, the generated `joint_opening_proof` (the `JoltProof` PCS artifact),
/// and the joint commitment.
#[derive(Clone, Debug)]
pub struct Stage8ProofOutput<F: Field, Proof, C> {
    pub structure: Stage8OpeningStructure<F>,
    pub joint_opening_proof: Proof,
    pub joint_commitment: C,
}

/// Canonical Stage 8 prover output for ZK mode: the verifier-equivalent opening
/// structure, generated PCS proof, joint commitment, hidden evaluation
/// commitment, and prover-side opening blind retained for BlindFold.
#[derive(Clone, Debug)]
pub struct Stage8ZkProofOutput<F: Field, Proof, C, H, Blind> {
    pub structure: Stage8OpeningStructure<F>,
    pub joint_opening_proof: Proof,
    pub joint_commitment: C,
    pub hiding_evaluation_commitment: H,
    pub hiding_evaluation_blind: Blind,
}
