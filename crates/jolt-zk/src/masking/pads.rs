//! The pad schedule: the public layout, the prover's committed pads and the
//! verifier's view of their commitment.

use std::ops::Range;

use jolt_field::Field;
use jolt_sumcheck::PadSlots;
use jolt_transcript::{Encoding, PrivateChannel, Transcript};
use serde::{Deserialize, Serialize};

use super::{MaskedProver, MaskedReader, MaskedVerifier, ZkError};
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
    /// routine (`OperandShapeDomain`). A back end under
    /// `OperandRule::MaskedSlot` commits them separately; BlindFold ignores
    /// them.
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
/// data `O`. A back end builds one with [`commit`](Self::commit), which
/// absorbs the commitment, so masking cannot run before the commitment is in
/// the transcript. The pads are zeroed when it is dropped; `O` wipes its own
/// secrets.
pub struct CommittedPads<F, O> {
    pads: SecretVec<F>,
    cursor: Cursor,
    opening: O,
}

impl<F: Field, O> CommittedPads<F, O> {
    /// The pads of `layout` under `commitment`: absorbs `commitment`, the
    /// one absorb [`CommittedPadsView::receive`] mirrors, and returns the
    /// pads ready to mask. A back end's
    /// `ZkBackendProver::commit_pads` calls it before
    /// the first challenge.
    ///
    /// # Errors
    ///
    /// [`ZkError::InvalidLayout`] if `pads` does not fill `layout`; nothing
    /// is absorbed then.
    pub fn commit<C: Encoding + ?Sized, T: Transcript>(
        layout: PadLayout,
        pads: Vec<F>,
        opening: O,
        commitment: &C,
        transcript: &mut T,
    ) -> Result<Self, ZkError> {
        let pads = Self::new(layout, pads, opening)?;
        transcript.append(commitment);
        Ok(pads)
    }

    /// `pads` under `layout`, with nothing absorbed. The pads are wiped on
    /// every path, the length mismatch included.
    pub(crate) fn new(layout: PadLayout, pads: Vec<F>, opening: O) -> Result<Self, ZkError> {
        let pads = SecretVec::from(pads);
        if pads.len() != layout.num_pads {
            return Err(ZkError::InvalidLayout {
                reason: "pad vector length differs from the layout",
            });
        }
        Ok(Self {
            pads,
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
        MaskedProver::new(self, tape, transcript)
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
/// cursor, no opening data. A back end builds one with
/// [`receive`](Self::receive), at the transcript position of the commitment.
pub struct CommittedPadsView<C> {
    commitment: C,
    cursor: Cursor,
}

impl<C> CommittedPadsView<C> {
    /// The view of `commitment` under `layout`: absorbs `commitment` where
    /// [`CommittedPads::commit`] absorbed it. A back end's
    /// [`receive_pad_commitment`](super::ZkBackend::receive_pad_commitment)
    /// checks the commitment's shape against `layout`, then calls it.
    pub fn receive<T: Transcript>(layout: PadLayout, commitment: C, transcript: &mut T) -> Self
    where
        C: Encoding,
    {
        transcript.append(&commitment);
        Self::new(layout, commitment)
    }

    /// The view of `commitment` under `layout`, with nothing absorbed.
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
        MaskedVerifier::new(self, tape, transcript)
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
