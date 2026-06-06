#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
#[cfg(feature = "zk")]
use jolt_sumcheck::SumcheckProof;

#[cfg(feature = "zk")]
use jolt_verifier::stages::stage7::outputs::Stage7PublicOutput;
use jolt_verifier::stages::stage7::{inputs::Stage7Claims, outputs::Stage7ClearOutput};

#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckWitness;

/// Canonical Stage 7 prover output (clear path).
///
/// Carries the verifier-owned `stage7_sumcheck_proof` payload, the clear
/// `Stage7Claims`, and the typed `Stage7ClearOutput` that Stage 8 consumes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7ProverOutput<F: Field, Proof> {
    pub stage7_sumcheck_proof: Proof,
    pub claims: Stage7Claims<F>,
    pub verifier_output: Stage7ClearOutput<F>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7CommittedBoundaryOutput<F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub stage7_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub public: Stage7PublicOutput<F>,
    pub output_claim_values: Vec<F>,
    pub verifier_output: Stage7ClearOutput<F>,
    pub(crate) committed_witness: CommittedSumcheckWitness<F>,
}

/// Stage 7 batched-sumcheck input claims, in verifier statement order.
///
/// Mirrors the verifier's private `Stage7BatchInputClaims`: the hamming-weight
/// claim-reduction input is always present; the advice address-phase inputs are
/// present only when the corresponding advice layout has an address phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7RegularBatchInputClaims<F: Field> {
    pub hamming_weight_claim_reduction: F,
    pub trusted_advice_address_phase: Option<F>,
    pub untrusted_advice_address_phase: Option<F>,
}

/// Fiat-Shamir prefix output for the Stage 7 batched sumcheck.
///
/// Produced before the sumcheck rounds run: it draws `hamming_gamma` and derives
/// the per-instance input claims from Stage 6 (and, for advice, the Stage 6
/// advice cycle-phase output claims), exactly as
/// `jolt-verifier/src/stages/stage7/verify.rs` does in clear mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7RegularBatchPrefixOutput<F: Field> {
    pub input_claims: Stage7RegularBatchInputClaims<F>,
    pub hamming_gamma: F,
}
