use super::{Cow, SingleSumcheckProof, SumcheckError, SumcheckShape};
use crate::pow::RoundNonces;
use jolt_field::Field;
use jolt_transcript::{Encoding, FieldTranscript, PrivateChannel};

/// One round's public transport shape: the number of wire values it carries
/// and the degree bound its proof of work is priced at. A single instance's
/// wire carries exactly `degree` values; a masked batch round in the full
/// format carries `degree + 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundSpec {
    pub width: usize,
    pub degree: usize,
}

/// Receive, absorb, check proof of work, then draw the challenge and visit the
/// wire. A borrowed decode buffer need only live for the visitor invocation.
pub trait VerifierRoundChannel<E: Field> {
    type Error: From<SumcheckError<E>>;
    type Slots;
    /// Reads one round of `spec.width` values, checks its proof of work at
    /// `spec.degree`, draws its challenge, and lends the wire to `visit`.
    fn verify_round<R>(
        &mut self,
        round: usize,
        spec: RoundSpec,
        visit: impl FnOnce(Cow<'_, [E]>, Self::Slots, E) -> Result<R, Self::Error>,
    ) -> Result<R, Self::Error>;
}

/// Only channels with clear scalar claims implement this capability.
pub trait BindScalarClaim<E: Field>: VerifierRoundChannel<E, Slots = ()> {
    fn bind_input_claim(&mut self, claim: &E) -> Result<(), Self::Error>;
}

/// The clear transport over a stored proof. Built only by `verify_single`
/// from the same instance shape the driver reads with, so every round's
/// width and proof-of-work degree are the ones checked here.
pub(crate) struct ClearVerifierRounds<'a, E, T> {
    transcript: &'a mut T,
    proof: &'a SingleSumcheckProof<E>,
    pow: RoundNonces<'a>,
}

impl<'a, E: Field + Encoding, T: FieldTranscript<E>> ClearVerifierRounds<'a, E, T> {
    /// Checks all proof widths and nonce counts before touching the transcript.
    pub(crate) fn new(
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
        })
    }
}

impl<E: Field + Encoding, T: FieldTranscript<E>> VerifierRoundChannel<E>
    for ClearVerifierRounds<'_, E, T>
{
    type Error = SumcheckError<E>;
    type Slots = ();
    fn verify_round<R>(
        &mut self,
        round: usize,
        _spec: RoundSpec,
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
