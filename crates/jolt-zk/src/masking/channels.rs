//! The masked channels: the prover's and the verifier's transport for
//! private messages over committed pads, and the verifier's cursor over a
//! proof's masked tape. Both channels implement the sumcheck engine's round
//! channels ([`ProverRoundChannel`], [`VerifierRoundChannel`]), so the
//! single-instance drivers and `receive_masked_rounds` run over them.

use std::borrow::Cow;

use jolt_field::Field;
#[cfg(feature = "masking-prover")]
use jolt_sumcheck::ProverRoundChannel;
use jolt_sumcheck::{
    round_pow_bits, PadSlots, RoundSpec, SumcheckError, SumcheckShape, VerifierRoundChannel,
};
#[cfg(feature = "masking-prover")]
use jolt_transcript::ProverTranscript;
use jolt_transcript::{Encoding, FieldTranscript, PrivateChannel};

use super::{CommittedPads, CommittedPadsView, MaskedMessages, ZkError};

/// The verifier's cursor over a proof's [`MaskedMessages`].
#[derive(Clone, Debug)]
pub struct MaskedReader<'m, F> {
    source: TapeSource<'m, F>,
    next_value: usize,
    next_nonce: usize,
}

#[derive(Clone, Debug)]
enum TapeSource<'m, F> {
    Proof(&'m MaskedMessages<F>),
    /// An unbounded tape of zeros, for a layout replay over a shape-correct
    /// proof (`ShapeTranscript` prices away every nonce).
    Zeros,
}

impl<'m, F: Field> MaskedReader<'m, F> {
    pub fn new(messages: &'m MaskedMessages<F>) -> Self {
        Self {
            source: TapeSource::Proof(messages),
            next_value: 0,
            next_nonce: 0,
        }
    }

    /// An unbounded tape of zero values and zero nonces, for counting a pad
    /// layout. [`finish`](Self::finish) always succeeds on it.
    pub fn zeros() -> Self {
        Self {
            source: TapeSource::Zeros,
            next_value: 0,
            next_nonce: 0,
        }
    }

    /// The next `len` masked values.
    ///
    /// # Errors
    ///
    /// [`ZkError::MaskedTapeExhausted`] if fewer remain.
    pub fn values(&mut self, len: usize) -> Result<Vec<F>, ZkError> {
        let end = self
            .next_value
            .checked_add(len)
            .ok_or(ZkError::MaskedTapeExhausted)?;
        let values = match self.source {
            TapeSource::Proof(messages) => messages
                .values
                .get(self.next_value..end)
                .ok_or(ZkError::MaskedTapeExhausted)?
                .to_vec(),
            TapeSource::Zeros => vec![F::zero(); len],
        };
        self.next_value = end;
        Ok(values)
    }

    /// The next proof-of-work nonce.
    ///
    /// # Errors
    ///
    /// [`ZkError::MaskedTapeExhausted`] if none remains.
    pub fn nonce(&mut self) -> Result<u64, ZkError> {
        let nonce = match self.source {
            TapeSource::Proof(messages) => *messages
                .nonces
                .get(self.next_nonce)
                .ok_or(ZkError::MaskedTapeExhausted)?,
            TapeSource::Zeros => 0,
        };
        self.next_nonce += 1;
        Ok(nonce)
    }

    /// Checks that every masked value and nonce was read.
    ///
    /// # Errors
    ///
    /// [`ZkError::MaskedTapeTrailing`] otherwise.
    pub fn finish(&self) -> Result<(), ZkError> {
        match self.source {
            TapeSource::Proof(messages)
                if self.next_value != messages.values.len()
                    || self.next_nonce != messages.nonces.len() =>
            {
                Err(ZkError::MaskedTapeTrailing)
            }
            TapeSource::Proof(_) | TapeSource::Zeros => Ok(()),
        }
    }
}

/// The verifier's masked channel: it reads m̂ in place of each private value
/// from the proof's masked tape, absorbs it where the clear verifier absorbs
/// the value, and returns the pad slots the prover used. Derived claims bind
/// nothing, since the masked messages they derive from already do.
pub struct MaskedVerifier<'a, 'm, F, C, T> {
    view: &'a mut CommittedPadsView<C>,
    tape: &'a mut MaskedReader<'m, F>,
    transcript: &'a mut T,
}

impl<'a, 'm, F, C, T> MaskedVerifier<'a, 'm, F, C, T> {
    pub(super) fn new(
        view: &'a mut CommittedPadsView<C>,
        tape: &'a mut MaskedReader<'m, F>,
        transcript: &'a mut T,
    ) -> Self {
        Self {
            view,
            tape,
            transcript,
        }
    }
}

impl<F, C, T> MaskedVerifier<'_, '_, F, C, T>
where
    F: Field + Encoding + 'static,
    T: FieldTranscript<F>,
{
    /// Reads, absorbs and places one private message of `len` values.
    ///
    /// # Errors
    ///
    /// [`ZkError::MaskedTapeExhausted`] or [`ZkError::PadsExhausted`].
    pub fn receive_private(&mut self, len: usize) -> Result<(Vec<F>, PadSlots), ZkError> {
        let values = self.tape.values(len)?;
        self.transcript.append_private(&values);
        let slots = self.view.take(len)?;
        Ok((values, slots))
    }

    /// The transcript, for the public absorbs and challenges a masked stage
    /// interleaves with its private messages.
    pub fn transcript(&mut self) -> &mut T {
        self.transcript
    }

    /// Checks one round's proof of work at degree bound `degree`, its nonce
    /// read from the tape when the challenge set needs one.
    fn check_round_pow(&mut self, round: usize, degree: usize) -> Result<(), ZkError> {
        let bits = round_pow_bits::<F, T>(degree).map_err(ZkError::from)?;
        let nonce = if bits > 0 {
            Some(self.tape.nonce()?)
        } else {
            None
        };
        self.transcript
            .check_public_grind(bits, nonce)
            .map_err(|_| ZkError::from(SumcheckError::<F>::ProofOfWorkRejected { round }))
    }
}

/// A masked round is one private message of `spec.width` values, then the
/// round's proof of work at `spec.degree`, then its challenge; the visitor
/// gets the masked wire and its pad slots.
impl<F, C, T> VerifierRoundChannel<F> for MaskedVerifier<'_, '_, F, C, T>
where
    F: Field + Encoding + 'static,
    T: FieldTranscript<F>,
{
    type Error = ZkError;
    type Slots = PadSlots;

    /// The masked tape has no shape of its own: the driver's shape sets the
    /// width of every read.
    fn validate_shape(&self, _shape: SumcheckShape) -> Result<(), ZkError> {
        Ok(())
    }

    fn map_sumcheck_error(&self, error: SumcheckError<F>) -> ZkError {
        error.into()
    }

    fn verify_round<R>(
        &mut self,
        round: usize,
        spec: RoundSpec,
        visit: impl FnOnce(Cow<'_, [F]>, PadSlots, F) -> Result<R, ZkError>,
    ) -> Result<R, ZkError> {
        let (wire, slots) = self.receive_private(spec.width)?;
        self.check_round_pow(round, spec.degree)?;
        let challenge = self
            .transcript
            .challenge()
            .map_err(|_| ZkError::ChallengeRejected)?;
        visit(Cow::Owned(wire), slots, challenge)
    }
}

/// The prover's masked channel: each private message goes out as
/// m̂ = m + p, absorbed where the clear prover absorbs the value and written
/// to the proof's masked tape, in the order [`MaskedVerifier`] reads them.
pub struct MaskedProver<'a, F, O, T> {
    pads: &'a mut CommittedPads<F, O>,
    tape: &'a mut MaskedMessages<F>,
    transcript: &'a mut T,
}

impl<'a, F, O, T> MaskedProver<'a, F, O, T> {
    pub(super) fn new(
        pads: &'a mut CommittedPads<F, O>,
        tape: &'a mut MaskedMessages<F>,
        transcript: &'a mut T,
    ) -> Self {
        Self {
            pads,
            tape,
            transcript,
        }
    }
}

impl<F, O, T> MaskedProver<'_, F, O, T>
where
    F: Field + Encoding,
    T: FieldTranscript<F>,
{
    /// Masks, absorbs and records one private message.
    ///
    /// # Errors
    ///
    /// [`ZkError::PadsExhausted`] past the end of the schedule.
    pub fn send_private(&mut self, values: &[F]) -> Result<PadSlots, ZkError> {
        self.pads.send(values, self.tape, self.transcript)
    }

    /// The transcript, for public absorbs and challenges.
    pub fn transcript(&mut self) -> &mut T {
        self.transcript
    }
}

/// The prover side of [`MaskedVerifier`]'s rounds. A round's wire goes out as
/// one masked private message; the round's proof of work is ground and its
/// nonce recorded on the tape before the challenge.
#[cfg(feature = "masking-prover")]
impl<F, O, T> ProverRoundChannel<F> for MaskedProver<'_, F, O, T>
where
    F: Field + Encoding + 'static,
    T: FieldTranscript<F> + ProverTranscript,
{
    type Error = ZkError;

    fn map_sumcheck_error(&self, error: SumcheckError<F>) -> ZkError {
        error.into()
    }

    /// Binds nothing: a masked input claim must be a function of masked
    /// values already sent, of challenges and of public data, and the
    /// residual relation pins it through them. The masked verifier, which
    /// reads rounds through `visit_single_rounds`, binds nothing either. A
    /// claim that depends on a value sent later must be sent masked before
    /// the rounds, with [`send_private`](Self::send_private).
    fn bind_input_claim(&mut self, _claim: &F) -> Result<(), ZkError> {
        Ok(())
    }

    fn send_round(&mut self, _round: usize, wire: &mut [F]) -> Result<(), ZkError> {
        self.send_private(wire).map(|_: PadSlots| ())
    }

    fn round_challenge(&mut self, round: usize, degree: usize) -> Result<F, ZkError> {
        let bits = round_pow_bits::<F, T>(degree).map_err(ZkError::from)?;
        let nonce = self
            .transcript
            .public_grind(bits)
            .map_err(|_| ZkError::from(SumcheckError::<F>::ProofOfWorkRejected { round }))?;
        self.tape.nonces.extend(nonce);
        self.transcript
            .challenge()
            .map_err(|_| ZkError::ChallengeRejected)
    }
}

/// A masked verifier's state across one proof: its view of the pad
/// commitment and its cursor over the proof's masked tape. The masked stage
/// verifiers borrow one [`MaskedVerifier`] from it per stage.
pub struct MaskedSession<'m, F, C> {
    pub pads: CommittedPadsView<C>,
    pub tape: MaskedReader<'m, F>,
}

impl<'m, F: Field, C> MaskedSession<'m, F, C> {
    pub fn new(pads: CommittedPadsView<C>, tape: MaskedReader<'m, F>) -> Self {
        Self { pads, tape }
    }

    /// The masked channel over `transcript`.
    pub fn channel<'a, T>(&'a mut self, transcript: &'a mut T) -> MaskedVerifier<'a, 'm, F, C, T> {
        self.pads.channel(&mut self.tape, transcript)
    }

    /// Checks that every scheduled pad and every masked message was used.
    ///
    /// # Errors
    ///
    /// [`ZkError::PadCountMismatch`] or [`ZkError::MaskedTapeTrailing`].
    pub fn finish(&self) -> Result<(), ZkError> {
        self.pads.finish()?;
        self.tape.finish()
    }
}
