use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use tracer::instruction::RV32IMCycle;

use crate::{
    field::JoltField,
    jolt::lookup_table::LookupTables,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        compact_polynomial::CompactPolynomial,
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
    },
    subprotocols::{
        sparse_dense_shout::{prove_sparse_dense_shout, verify_sparse_dense_shout},
        sumcheck::SumcheckInstanceProof,
    },
    utils::{errors::ProofVerifyError, math::Math, transcript::Transcript},
};

pub struct LookupsWitness<F: JoltField> {
    ra: [CompactPolynomial<u16, F>; 4],
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct LookupsProof<const WORD_SIZE: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    read_checking_proof: SumcheckInstanceProof<F, ProofTranscript>,
    rv_claim: F,
    ra_claims: [F; 4],
    flag_claims: Vec<F>,
    log_T: usize,
    _marker: PhantomData<PCS>,
}

impl<const WORD_SIZE: usize, F, PCS, ProofTranscript>
    LookupsProof<WORD_SIZE, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub fn generate_witness(preprocessing: (), lookups: &[LookupTables<WORD_SIZE>]) {}

    #[tracing::instrument(skip_all, name = "LookupsProof::prove")]
    pub fn prove(
        generators: &PCS::Setup,
        trace: &[RV32IMCycle],
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let log_T = trace.len().log_2();
        let r_cycle: Vec<F> = transcript.challenge_vector(log_T);
        let (read_checking_proof, rv_claim, ra_claims, flag_claims) =
            prove_sparse_dense_shout::<WORD_SIZE, _, _>(&trace, r_cycle, transcript);

        Self {
            read_checking_proof,
            rv_claim,
            ra_claims,
            flag_claims,
            log_T,
            _marker: PhantomData,
        }
    }

    pub fn verify(
        &self,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let r_cycle: Vec<F> = transcript.challenge_vector(self.log_T);
        verify_sparse_dense_shout::<WORD_SIZE, _, _>(
            &self.read_checking_proof,
            self.log_T,
            r_cycle,
            self.rv_claim,
            self.ra_claims,
            &self.flag_claims,
            transcript,
        )
    }
}
