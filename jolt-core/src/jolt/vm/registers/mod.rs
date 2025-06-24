#[cfg(feature = "prover")]
mod prover;
#[cfg(feature = "prover")]
pub use prover::*;
mod verifier;
pub use verifier::*;

use crate::{
    field::JoltField,
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::transcript::{AppendToTranscript, Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RegistersTwistProof<F: JoltField, ProofTranscript: Transcript> {
    /// Proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    read_write_checking_proof: ReadWriteCheckingProof<F, ProofTranscript>,
    /// Proof of the Val-evaluation sumcheck (step 6 of Figure 9).
    val_evaluation_proof: ValEvaluationProof<F, ProofTranscript>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct ReadWriteCheckingProof<F: JoltField, ProofTranscript: Transcript> {
    /// Joint sumcheck proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// The claimed evaluation ra(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    rs1_ra_claim: F,
    /// The claimed evaluation rv(r') proven by the read-checking sumcheck.
    rs1_rv_claim: F,
    /// The claimed evaluation ra(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    rs2_ra_claim: F,
    /// The claimed evaluation rv(r') proven by the read-checking sumcheck.
    rs2_rv_claim: F,
    /// The claimed evaluation wa(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    rd_wa_claim: F,
    /// The claimed evaluation wv(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    rd_wv_claim: F,
    /// The claimed evaluation val(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    val_claim: F,
    /// The claimed evaluation Inc(r, r') proven by the write-checking sumcheck.
    inc_claim: F,
    /// The sumcheck round index at which we switch from binding cycle variables
    /// to binding address variables.
    sumcheck_switch_index: usize,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct ValEvaluationProof<F: JoltField, ProofTranscript: Transcript> {
    /// Sumcheck proof for the Val-evaluation sumcheck (steps 6 of Figure 9).
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// The claimed evaluation Inc(r_address, r_cycle') output by the Val-evaluation sumcheck.
    inc_claim: F,
}
