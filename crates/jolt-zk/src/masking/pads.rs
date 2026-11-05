//! The pad schedule: the public layout, the prover's committed pads and the
//! verifier's view of their commitment.

use std::ops::Range;

use jolt_field::Field;
use jolt_sumcheck::{round_pow_bits, MaskedRound, MaskedRoundReader, SumcheckError};
#[cfg(feature = "prover")]
use jolt_transcript::ProverTranscript;
use jolt_transcript::{Encoding, FieldTranscript, PadSlots, PrivateChannel};
use serde::{Deserialize, Serialize};

use super::ZkError;
use crate::secret::SecretVec;

/// The position of one pad (one base-field coordinate) in the schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PadIndex(pub usize);

/// Public pad layout, derived from the protocol's public shape by one function
/// that prover and verifier both call. The base messages take the slots
/// before [`auxiliary`](Self::auxiliary); the masked intermediate products
/// take the auxiliary range, which ends the schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PadLayout {
    pub num_pads: usize,
    pub auxiliary: Range<usize>,
    /// The pads of product operands, in first-use order, from the shape
    /// routine (`OperandShapeDomain`). The lattice back end commits them as
    /// constant digit polynomials; BlindFold ignores them.
    pub operand_pads: Vec<PadIndex>,
}

impl PadLayout {
    /// `base` message slots followed by `auxiliary` product slots.
    ///
    /// # Errors
    ///
    /// [`ZkError::InvalidLayout`] if the total overflows `usize`.
    pub fn new(base: usize, auxiliary: usize) -> Result<Self, ZkError> {
        let num_pads = base.checked_add(auxiliary).ok_or(ZkError::InvalidLayout {
            reason: "pad count overflows usize",
        })?;
        Ok(Self {
            num_pads,
            auxiliary: base..num_pads,
            operand_pads: Vec::new(),
        })
    }

    /// This layout with `operand_pads` as its product operands.
    #[must_use]
    pub(crate) fn with_operand_pads(self, operand_pads: Vec<PadIndex>) -> Self {
        Self {
            operand_pads,
            ..self
        }
    }
}

/// A cursor over the schedule: the next unused pad.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Cursor {
    layout: PadLayout,
    next: usize,
}

impl Cursor {
    fn take(&mut self, len: usize) -> Result<PadSlots, ZkError> {
        let end = self
            .next
            .checked_add(len)
            .filter(|&end| end <= self.layout.num_pads)
            .ok_or(ZkError::PadsExhausted {
                requested: len,
                remaining: self.layout.num_pads - self.next,
            })?;
        let slots = PadSlots {
            start: self.next,
            len,
        };
        self.next = end;
        Ok(slots)
    }

    fn finish(&self) -> Result<(), ZkError> {
        if self.next == self.layout.num_pads {
            Ok(())
        } else {
            Err(ZkError::PadCountMismatch {
                scheduled: self.layout.num_pads,
                used: self.next,
            })
        }
    }
}

/// Every masked message of a proof, in transcript order: the masked values
/// m̂ = m + p of each private field message, and the proof-of-work nonces of
/// the masked sumcheck rounds. Both roles absorb each message where the clear
/// protocol absorbs its value, so a masked proof carries this tape in place of
/// the clear proof's round polynomials and output claims.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct MaskedMessages<F> {
    /// The masked values, message after message.
    pub values: Vec<F>,
    /// The nonces of the masked rounds whose challenges need proof of work.
    pub nonces: Vec<u64>,
}

/// Pads committed before the first challenge, with the back end's opening
/// data `O`. Only a [`ZkBackend`](super::ZkBackend) builds one, so masking
/// cannot run before the commitment exists. The pads are zeroed when it is
/// dropped; `O` wipes its own secrets.
pub struct CommittedPads<F, O> {
    pads: SecretVec<F>,
    cursor: Cursor,
    opening: O,
}

impl<F: Field, O> CommittedPads<F, O> {
    #[cfg_attr(
        not(any(
            all(test, feature = "prover"),
            feature = "test-support",
            feature = "whiteout-prover"
        )),
        expect(
            dead_code,
            reason = "only a back end's commit_pads builds pads; BlindFold's lands with the pad-committed BlindFold"
        )
    )]
    pub(crate) fn new(layout: PadLayout, pads: Vec<F>, opening: O) -> Result<Self, ZkError> {
        if pads.len() != layout.num_pads {
            return Err(ZkError::InvalidLayout {
                reason: "pad vector length differs from the layout",
            });
        }
        Ok(Self {
            pads: pads.into(),
            cursor: Cursor { layout, next: 0 },
            opening,
        })
    }

    pub fn layout(&self) -> &PadLayout {
        &self.cursor.layout
    }

    pub fn pads(&self) -> &[F] {
        &self.pads
    }

    pub fn opening(&self) -> &O {
        &self.opening
    }

    /// `values` masked by the next `values.len()` pads, m̂ = m + p, and their
    /// slots. The caller sends the masked values in its own encoding.
    ///
    /// # Errors
    ///
    /// [`ZkError::PadsExhausted`] past the end of the schedule.
    pub fn mask(&mut self, values: &[F]) -> Result<(PadSlots, Vec<F>), ZkError> {
        let slots = self.cursor.take(values.len())?;
        let pads =
            self.pads
                .get(slots.start..slots.start + slots.len)
                .ok_or(ZkError::InvalidLayout {
                    reason: "pad slots outside the pad vector",
                })?;
        Ok((
            slots,
            values.iter().zip(pads).map(|(m, p)| *m + *p).collect(),
        ))
    }

    /// Masks `values`, absorbs the masked values where the clear protocol
    /// absorbs `values` (one [`PrivateChannel::append_private`] message),
    /// and appends them to the proof's masked tape.
    ///
    /// # Errors
    ///
    /// As [`mask`](Self::mask).
    pub fn send<T: PrivateChannel + ?Sized>(
        &mut self,
        values: &[F],
        tape: &mut MaskedMessages<F>,
        transcript: &mut T,
    ) -> Result<PadSlots, ZkError>
    where
        F: Encoding,
    {
        let (slots, masked) = self.mask(values)?;
        transcript.append_private(&masked);
        tape.values.extend_from_slice(&masked);
        Ok(slots)
    }

    /// The prover's masked channel over `transcript`, writing `tape`.
    pub fn channel<'a, T>(
        &'a mut self,
        tape: &'a mut MaskedMessages<F>,
        transcript: &'a mut T,
    ) -> MaskedProver<'a, F, O, T> {
        MaskedProver {
            pads: self,
            tape,
            transcript,
        }
    }

    /// The decoded value m̂ − p of a masked value at `pad`.
    ///
    /// # Errors
    ///
    /// [`ZkError::InvalidLayout`] if `pad` is outside the schedule.
    pub fn decode(&self, masked: F, pad: PadIndex) -> Result<F, ZkError> {
        self.pads
            .get(pad.0)
            .map(|p| masked - *p)
            .ok_or(ZkError::InvalidLayout {
                reason: "pad index outside the schedule",
            })
    }

    /// Checks that every scheduled pad was used.
    ///
    /// # Errors
    ///
    /// [`ZkError::PadCountMismatch`] otherwise.
    pub fn finish(&self) -> Result<(), ZkError> {
        self.cursor.finish()
    }
}

impl<F: Field> CommittedPads<F, ()> {
    /// `pads` under `layout` with no commitment, for tests of protocols
    /// written over the masked channel (Akita's masked levels). It hides
    /// nothing a verifier could check, and no back end opens it.
    ///
    /// # Errors
    ///
    /// [`ZkError::InvalidLayout`] if `pads` does not fill `layout`.
    #[cfg(feature = "test-support")]
    pub fn uncommitted(layout: PadLayout, pads: Vec<F>) -> Result<Self, ZkError> {
        Self::new(layout, pads, ())
    }
}

/// The verifier's view of the pad commitment `C`: the commitment and a
/// cursor, no opening data. Only a [`ZkBackend`](super::ZkBackend) builds
/// one, at the transcript position of the commitment.
pub struct CommittedPadsView<C> {
    commitment: C,
    cursor: Cursor,
}

impl<C> CommittedPadsView<C> {
    pub(crate) fn new(layout: PadLayout, commitment: C) -> Self {
        Self {
            commitment,
            cursor: Cursor { layout, next: 0 },
        }
    }

    pub fn layout(&self) -> &PadLayout {
        &self.cursor.layout
    }

    pub fn commitment(&self) -> &C {
        &self.commitment
    }

    /// The slots of the next masked message of `len` base coordinates.
    ///
    /// # Errors
    ///
    /// [`ZkError::PadsExhausted`] past the end of the schedule.
    pub fn take(&mut self, len: usize) -> Result<PadSlots, ZkError> {
        self.cursor.take(len)
    }

    /// The slots of the next masked message of `len` values of an extension
    /// of degree `degree`.
    ///
    /// # Errors
    ///
    /// As [`take`](Self::take), and [`ZkError::InvalidLayout`] if the pad
    /// count overflows.
    pub fn take_ext(&mut self, len: usize, degree: usize) -> Result<PadSlots, ZkError> {
        let len = len.checked_mul(degree).ok_or(ZkError::InvalidLayout {
            reason: "extension pad count overflows usize",
        })?;
        self.cursor.take(len)
    }

    /// The verifier's masked channel over `transcript`, reading `tape`.
    pub fn channel<'a, 'm, F, T>(
        &'a mut self,
        tape: &'a mut MaskedReader<'m, F>,
        transcript: &'a mut T,
    ) -> MaskedVerifier<'a, 'm, F, C, T> {
        MaskedVerifier {
            view: self,
            tape,
            transcript,
        }
    }

    /// Checks that every scheduled pad was used.
    ///
    /// # Errors
    ///
    /// [`ZkError::PadCountMismatch`] otherwise.
    pub fn finish(&self) -> Result<(), ZkError> {
        self.cursor.finish()
    }

    /// The number of pads the masked messages read so far took.
    pub fn used(&self) -> usize {
        self.cursor.next
    }
}

impl CommittedPadsView<()> {
    /// A view of an unbounded schedule with no commitment, for deriving a
    /// layout: running the masked verifier over it counts the pads its
    /// messages take ([`used`](Self::used)). It opens nothing, and no back end
    /// accepts it.
    pub fn counting() -> Self {
        Self::new(
            PadLayout {
                num_pads: usize::MAX,
                auxiliary: usize::MAX..usize::MAX,
                operand_pads: Vec::new(),
            },
            (),
        )
    }
}

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
    ///
    /// # Errors
    ///
    /// [`ZkError::MaskedTapeExhausted`] or a rejected proof of work.
    pub fn check_round_pow(&mut self, round: usize, degree: usize) -> Result<(), ZkError> {
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

impl<F, C, T> MaskedRoundReader<F> for MaskedVerifier<'_, '_, F, C, T>
where
    F: Field + Encoding + 'static,
    T: FieldTranscript<F>,
{
    type Error = ZkError;

    fn read_round(
        &mut self,
        round: usize,
        width: usize,
        degree: usize,
    ) -> Result<MaskedRound<F>, ZkError> {
        let (wire, slots) = self.receive_private(width)?;
        self.check_round_pow(round, degree)?;
        let challenge = self
            .transcript
            .challenge()
            .map_err(|_| ZkError::ChallengeRejected)?;
        Ok(MaskedRound {
            wire,
            slots,
            challenge,
        })
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

#[cfg(feature = "prover")]
impl<F, O, T> MaskedProver<'_, F, O, T>
where
    F: Field + Encoding + 'static,
    T: FieldTranscript<F> + ProverTranscript,
{
    /// Grinds one round's proof of work at degree bound `degree` and records
    /// its nonce on the tape when the challenge set needs one.
    ///
    /// # Errors
    ///
    /// When the nonce search fails.
    pub fn grind_round(&mut self, round: usize, degree: usize) -> Result<(), ZkError> {
        let bits = round_pow_bits::<F, T>(degree).map_err(ZkError::from)?;
        let nonce = self
            .transcript
            .public_grind(bits)
            .map_err(|_| ZkError::from(SumcheckError::<F>::ProofOfWorkRejected { round }))?;
        self.tape.nonces.extend(nonce);
        Ok(())
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
