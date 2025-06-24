#[cfg(feature = "prover")]
mod prover;
#[cfg(feature = "prover")]
pub use prover::*;
mod verifier;
pub use verifier::*;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::marker::PhantomData;

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
    },
    subprotocols::{
        sumcheck::SumcheckInstanceProof,
    },
    utils::{
        transcript::Transcript,
    },
};

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct LookupsProof<const WORD_SIZE: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    read_checking_proof: ReadCheckingProof<F, ProofTranscript>,
    booleanity_proof: BooleanityProof<F, ProofTranscript>,
    hamming_weight_proof: HammingWeightProof<F, ProofTranscript>,
    log_T: usize,
    _marker: PhantomData<PCS>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct ReadCheckingProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    rv_claim: F,
    ra_claims: [F; 4],
    add_sub_mul_flag_claim: F,
    flag_claims: Vec<F>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct BooleanityProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claims: [F; 4],
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct HammingWeightProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claims: [F; 4],
}