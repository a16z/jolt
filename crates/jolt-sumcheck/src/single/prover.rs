use super::{SingleSumcheckProof, SumcheckError, SumcheckShape};
use crate::pow::grind_round;
use jolt_field::Field;
use jolt_transcript::{Encoding, FieldTranscript, PrivateChannel, ProverTranscript};
use std::marker::PhantomData;

/// Transport for clear or masked prover messages. Errors are classified by the
/// channel, without requiring a foreign error type to implement `From`.
pub trait ProverRoundChannel<E: Field> {
    type Error;
    fn map_sumcheck_error(&self, error: SumcheckError<E>) -> Self::Error;
    /// Use the channel's derived-claim operation; masked transports must not
    /// absorb the private scalar into a clear sub-channel.
    fn bind_input_claim(&mut self, claim: &E) -> Result<(), Self::Error>;
    /// Preserve the clear values in `wire`: the driver evaluates them after sending.
    fn send_round(&mut self, round: usize, wire: &mut [E]) -> Result<(), Self::Error>;
    fn round_challenge(&mut self, round: usize, degree: usize) -> Result<E, Self::Error>;
    /// Take the evaluated wire for proof storage, or drop it for streaming transports.
    fn finish_round(&mut self, wire: Vec<E>) {
        drop(wire);
    }
}

pub struct ClearProverRounds<'a, E: Field, T, Err = SumcheckError<E>> {
    transcript: &'a mut T,
    proof: SingleSumcheckProof<E>,
    error: PhantomData<Err>,
}

impl<'a, E: Field, T, Err> ClearProverRounds<'a, E, T, Err> {
    pub fn new(shape: SumcheckShape, transcript: &'a mut T) -> Self {
        Self {
            transcript,
            proof: SingleSumcheckProof {
                rounds: Vec::with_capacity(shape.num_rounds()),
                pow_nonces: Vec::new(),
            },
            error: PhantomData,
        }
    }
    pub fn into_proof(self) -> SingleSumcheckProof<E> {
        self.proof
    }
}

impl<
        E: Field + Encoding,
        T: FieldTranscript<E> + ProverTranscript,
        Err: From<SumcheckError<E>>,
    > ProverRoundChannel<E> for ClearProverRounds<'_, E, T, Err>
{
    type Error = Err;
    fn map_sumcheck_error(&self, error: SumcheckError<E>) -> Err {
        error.into()
    }
    fn bind_input_claim(&mut self, claim: &E) -> Result<(), Err> {
        self.transcript.bind_derived(claim);
        Ok(())
    }
    fn send_round(&mut self, _round: usize, wire: &mut [E]) -> Result<(), Err> {
        self.transcript.append_private(wire);
        Ok(())
    }
    fn finish_round(&mut self, wire: Vec<E>) {
        self.proof.rounds.push(wire);
    }
    fn round_challenge(&mut self, round: usize, degree: usize) -> Result<E, Err> {
        grind_round(degree, round, &mut self.proof.pow_nonces, self.transcript)?;
        self.transcript
            .challenge()
            .map_err(|e| SumcheckError::from(e).into())
    }
}
