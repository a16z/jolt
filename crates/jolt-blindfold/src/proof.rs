//! Proof and prover-side data types for committed sumcheck and BlindFold.

use jolt_crypto::JoltCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_spartan::RelaxedSpartanProof;
use serde::{Deserialize, Serialize};

/// Public proof for a committed sumcheck stage.
///
/// Contains only commitments (no polynomial coefficients), making the
/// sumcheck zero-knowledge. Sent from prover to verifier.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommittedSumcheckProof<VC: JoltCommitment> {
    /// One commitment per sumcheck round.
    pub round_commitments: Vec<VC::Commitment>,
    /// Degree of each round polynomial (needed for BlindFold R1CS).
    pub poly_degrees: Vec<usize>,
}

/// Private prover data accumulated during a committed sumcheck stage.
///
/// Contains the polynomial coefficients, blinding factors, and commitments
/// that the prover needs to later construct the BlindFold proof. This is
/// **never** sent to the verifier.
#[derive(Clone, Debug)]
pub struct CommittedRoundData<F: Field, VC: JoltCommitment> {
    /// Commitments to each round polynomial.
    pub round_commitments: Vec<VC::Commitment>,
    /// Cleartext coefficients of each round polynomial.
    pub poly_coeffs: Vec<Vec<F>>,
    /// Blinding factor used for each round's commitment.
    pub blinding_factors: Vec<F>,
    /// Degree of each round polynomial.
    pub poly_degrees: Vec<usize>,
    /// Fiat-Shamir challenges derived at each round.
    pub challenges: Vec<F>,
}

/// Full BlindFold proof tying committed sumcheck stages to a relaxed Spartan proof.
///
/// After all sumcheck stages run with committed rounds, the prover:
/// 1. Builds a verifier R1CS encoding the deferred sumcheck checks.
/// 2. Assigns the witness from accumulated round data.
/// 3. Nova-folds the real instance with a random satisfying instance.
/// 4. Produces a relaxed Spartan proof over the folded instance.
///
/// `PCS` is the polynomial commitment scheme used for the Spartan witness/error
/// openings. The committed sumcheck proofs (with `VC::Commitment`) are sent
/// separately and are not included here.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
#[allow(clippy::type_complexity)]
pub struct BlindFoldProof<F: Field, PCS: CommitmentScheme> {
    /// Commitment to the real R1CS witness.
    pub real_w_commitment: PCS::Output,
    /// Commitment to the real error vector (all zeros for a satisfying instance).
    pub real_e_commitment: PCS::Output,
    /// Relaxation scalar of the random masking instance.
    pub random_u: F,
    /// Commitment to the random instance's witness.
    pub random_w_commitment: PCS::Output,
    /// Commitment to the random instance's error vector.
    pub random_e_commitment: PCS::Output,
    /// Commitment to the cross-term vector from Nova folding.
    pub cross_term_commitment: PCS::Output,
    /// Relaxed Spartan proof over the folded instance.
    pub spartan_proof: RelaxedSpartanProof<F, PCS>,
}

/// Combined output from [`CommittedRoundHandler::finalize`](super::CommittedRoundHandler).
///
/// Splits into a public proof (for the verifier) and private round data
/// (for the BlindFold accumulator).
#[derive(Clone, Debug)]
pub struct CommittedSumcheckOutput<F: Field, VC: JoltCommitment> {
    /// Public proof sent to the verifier.
    pub proof: CommittedSumcheckProof<VC>,
    /// Private data retained by the prover for BlindFold.
    pub round_data: CommittedRoundData<F, VC>,
}
