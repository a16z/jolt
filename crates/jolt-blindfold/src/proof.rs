//! Proof and prover-side data types for committed sumcheck.

use jolt_crypto::JoltCommitment;
use jolt_field::Field;
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
