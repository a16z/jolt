// use crate::utils::errors::ProofVerifyError;
// use crate::utils::transcript::ProofTranscript;
//
// pub trait CommitmentScheme {
//     fn verify(
//         proof: &Self::Proof,
//         setup: &Self::Setup,
//         transcript: &mut ProofTranscript,
//         opening_point: &[Self::Field], // point at which the polynomial is evaluated
//         opening: &Self::Field,         // evaluation \widetilde{Z}(r)
//         commitment: &Self::Commitment,
//     ) -> Result<(), ProofVerifyError>;
//
//     fn batch_verify(
//         batch_proof: &Self::BatchedProof,
//         setup: &Self::Setup,
//         opening_point: &[Self::Field],
//         openings: &[Self::Field],
//         commitments: &[&Self::Commitment],
//         transcript: &mut ProofTranscript,
//     ) -> Result<(), ProofVerifyError>;
// }
