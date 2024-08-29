use crate::field::JoltField;
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::utils::transcript::ProofTranscript;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedGrandProductLayerProof<F: JoltField> {
    pub proof: SumcheckInstanceProof<F>,
    pub left_claims: Vec<F>,
    pub right_claims: Vec<F>,
}

impl<F: JoltField> BatchedGrandProductLayerProof<F> {
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> (F, Vec<F>) {
        self.proof
            .verify(claim, num_rounds, degree_bound, transcript)
            .unwrap()
    }
}
