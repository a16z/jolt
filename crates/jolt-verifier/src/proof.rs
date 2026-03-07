//! Jolt proof and verification key types.
//!
//! [`JoltProof`] is the complete proof object sent from prover to verifier.
//! [`JoltVerifyingKey`] contains PCS setup + preprocessing data needed
//! for verification.

use jolt_openings::CommitmentScheme;
use jolt_sumcheck::SumcheckProof;
use serde::{Deserialize, Serialize};

use crate::config::ProverConfig;

/// Number of sumcheck stages in the Jolt proving pipeline.
pub const NUM_SUMCHECK_STAGES: usize = 7;

/// Complete Jolt proof — sent from prover to verifier.
///
/// Generic over the polynomial commitment scheme. The proof contains:
/// - Commitments to all committed polynomials
/// - One sumcheck proof per stage (7 stages)
/// - Opening proofs for all reduced claims
/// - Configuration used during proving
#[derive(Clone, Serialize, Deserialize)]
pub struct JoltProof<PCS: CommitmentScheme> {
    /// Commitments to all committed polynomials (Inc, RA, etc.).
    pub commitments: Vec<PCS::Output>,
    /// One sumcheck proof per stage.
    pub stage_proofs: Vec<SumcheckProof<PCS::Field>>,
    /// Opening proofs for reduced polynomial claims.
    pub opening_proofs: Vec<PCS::Proof>,
    /// Prover configuration (memory layout, chunk sizes).
    pub config: ProverConfig,
    /// Number of execution cycles in the trace.
    pub trace_length: usize,
}

/// Verification key for a Jolt proof.
///
/// Contains everything the verifier needs besides the proof itself.
/// Typically generated once during preprocessing and reused across
/// multiple proof verifications.
#[derive(Clone, Serialize, Deserialize)]
pub struct JoltVerifyingKey<PCS: CommitmentScheme> {
    /// PCS verifier setup (SRS or structured reference string).
    pub pcs_setup: PCS::VerifierSetup,
}

/// Public inputs to a Jolt proof.
///
/// The verifier uses these to check that the proof corresponds to the
/// claimed program execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoltPublicInput {
    /// Program I/O bytes (stdin/stdout).
    pub program_io: Vec<u8>,
}
