use crate::field::JoltField;
use ark_ec::CurveGroup;
use ark_serialize::CanonicalSerialize;
use std::borrow::Borrow;

pub trait Transcript: Default + Clone + Sync + Send + 'static {
    fn new(label: &'static [u8]) -> Self;
    #[cfg(test)]
    fn compare_to(&mut self, other: Self);
    fn append_message(&mut self, msg: &'static [u8]);
    fn append_bytes(&mut self, bytes: &[u8]);
    fn append_u64(&mut self, x: u64);
    fn append_scalar<F: JoltField>(&mut self, scalar: &F);
    fn append_serializable<F: CanonicalSerialize>(&mut self, scalar: &F);
    fn append_scalars<F: JoltField>(&mut self, scalars: &[impl Borrow<F>]);
    fn append_point<G: CurveGroup>(&mut self, point: &G);
    fn append_points<G: CurveGroup>(&mut self, points: &[G]);
    fn challenge_u128(&mut self) -> u128;
    fn challenge_scalar<F: JoltField>(&mut self) -> F;
    fn challenge_scalar_128_bits<F: JoltField>(&mut self) -> F;
    fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F>;
    // Compute powers of scalar q : (1, q, q^2, ..., q^(len-1))
    fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F>;
    // New methods that return F::Challenge
    fn challenge_scalar_optimized<F: JoltField>(&mut self) -> F::Challenge;
    fn challenge_vector_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F::Challenge>;
    fn challenge_scalar_powers_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F>;
}

pub trait AppendToTranscript {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript);
}
