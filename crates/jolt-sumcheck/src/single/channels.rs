use super::{Cow, SingleSumcheckProof, SumcheckError, SumcheckShape};
use crate::pow::RoundNonces;
use jolt_field::Field;
use jolt_transcript::{Encoding, FieldTranscript, PrivateChannel};
#[cfg(feature = "prover")]
use {crate::pow::grind_round, jolt_transcript::ProverTranscript, std::marker::PhantomData};

/// Transport for clear or masked prover messages. Errors are classified by the
/// channel, without requiring a foreign error type to implement `From`.
#[cfg(feature = "prover")]
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

/// Receive, absorb, check proof of work, then draw the challenge and visit the
/// wire. A borrowed decode buffer need only live for the visitor invocation.
pub trait VerifierRoundChannel<E: Field> {
    type Error;
    type Slots;
    /// Reject any mismatch with a transport's predeclared shape before absorption.
    /// Transports whose shape is supplied solely by the driver may accept it directly.
    fn validate_shape(&self, shape: SumcheckShape) -> Result<(), Self::Error>;
    fn map_sumcheck_error(&self, error: SumcheckError<E>) -> Self::Error;
    fn verify_round<R>(
        &mut self,
        round: usize,
        degree: usize,
        visit: impl FnOnce(Cow<'_, [E]>, Self::Slots, E) -> Result<R, Self::Error>,
    ) -> Result<R, Self::Error>;
}

/// Only channels with clear scalar claims implement this capability.
pub trait BindScalarClaim<E: Field>: VerifierRoundChannel<E, Slots = ()> {
    fn bind_input_claim(&mut self, claim: &E) -> Result<(), Self::Error>;
}

#[cfg(feature = "prover")]
pub struct ClearProverRounds<'a, E: Field, T, Err = SumcheckError<E>> {
    transcript: &'a mut T,
    proof: SingleSumcheckProof<E>,
    error: PhantomData<Err>,
}

#[cfg(feature = "prover")]
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

#[cfg(feature = "prover")]
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

pub struct ClearVerifierRounds<'a, E, T> {
    transcript: &'a mut T,
    proof: &'a SingleSumcheckProof<E>,
    pow: RoundNonces<'a>,
    shape: SumcheckShape,
}

impl<'a, E: Field + Encoding, T: FieldTranscript<E>> ClearVerifierRounds<'a, E, T> {
    /// Checks all proof widths and nonce counts before touching the transcript.
    pub fn new(
        shape: SumcheckShape,
        proof: &'a SingleSumcheckProof<E>,
        transcript: &'a mut T,
    ) -> Result<Self, SumcheckError<E>> {
        if proof.rounds.len() != shape.num_rounds() {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected: shape.num_rounds(),
                got: proof.rounds.len(),
            });
        }
        for (round, wire) in proof.rounds.iter().enumerate() {
            if wire.len() != shape.degree_bound() {
                return Err(SumcheckError::WrongRoundWidth {
                    round,
                    expected: shape.degree_bound(),
                    got: wire.len(),
                });
            }
        }
        let pow =
            RoundNonces::new::<E, T>(shape.degree_bound(), shape.num_rounds(), &proof.pow_nonces)?;
        Ok(Self {
            transcript,
            proof,
            pow,
            shape,
        })
    }
}

impl<E: Field + Encoding, T: FieldTranscript<E>> VerifierRoundChannel<E>
    for ClearVerifierRounds<'_, E, T>
{
    type Error = SumcheckError<E>;
    type Slots = ();
    fn validate_shape(&self, shape: SumcheckShape) -> Result<(), Self::Error> {
        if self.shape != shape {
            return Err(SumcheckError::ChannelShapeMismatch {
                expected: shape,
                got: self.shape,
            });
        }
        Ok(())
    }
    fn map_sumcheck_error(&self, error: SumcheckError<E>) -> Self::Error {
        error
    }
    fn verify_round<R>(
        &mut self,
        round: usize,
        degree: usize,
        visit: impl FnOnce(Cow<'_, [E]>, (), E) -> Result<R, Self::Error>,
    ) -> Result<R, Self::Error> {
        let wire = self
            .proof
            .rounds
            .get(round)
            .ok_or(SumcheckError::WrongNumberOfRounds {
                expected: round.saturating_add(1),
                got: self.proof.rounds.len(),
            })?;
        if wire.len() != degree {
            return Err(SumcheckError::WrongRoundWidth {
                round,
                expected: degree,
                got: wire.len(),
            });
        }
        self.transcript.append_private(wire);
        self.pow.check(round, self.transcript)?;
        let challenge = self.transcript.challenge()?;
        visit(Cow::Borrowed(wire), (), challenge)
    }
}

impl<E: Field + Encoding, T: FieldTranscript<E>> BindScalarClaim<E>
    for ClearVerifierRounds<'_, E, T>
{
    fn bind_input_claim(&mut self, claim: &E) -> Result<(), Self::Error> {
        self.transcript.bind_derived(claim);
        Ok(())
    }
}
