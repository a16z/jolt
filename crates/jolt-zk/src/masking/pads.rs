//! The pad schedule: the public layout, the prover's committed pads and the
//! verifier's view of their commitment.

use std::any::Any;
use std::ops::Range;

use jolt_algebra::{CanonicalBytes, CanonicalDecode, CanonicalEncoding, ExtField, Field};
use jolt_transcript::{
    Channel, PadSlots, PrivateChannel, ProverTranscript, SiteId, Sponge, TranscriptError,
    VerifierTranscript,
};

use super::ZkError;

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

/// Pads committed before the first challenge, with the back end's opening
/// data `O`. Only a [`ZkBackend`](super::ZkBackend) builds one, so masking
/// cannot run before the commitment exists.
pub struct CommittedPads<F, O> {
    pads: Vec<F>,
    cursor: Cursor,
    opening: O,
}

impl<F: Field + CanonicalEncoding, O> CommittedPads<F, O> {
    pub(crate) fn new(layout: PadLayout, pads: Vec<F>, opening: O) -> Result<Self, ZkError> {
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

    /// Sends `values` masked by the next `values.len()` pads, m̂ = m + p, and
    /// returns the masked values as sent.
    ///
    /// # Errors
    ///
    /// [`ZkError::PadsExhausted`] past the end of the schedule.
    pub fn send<H: Sponge>(
        &mut self,
        values: &[F],
        transcript: &mut ProverTranscript<H>,
    ) -> Result<Vec<F>, ZkError> {
        let (_, mut masked) = self.mask(values)?;
        transcript.exchange_all(&mut masked)?;
        Ok(masked)
    }

    /// `values` masked by the next `values.len()` pads, and their slots.
    fn mask(&mut self, values: &[F]) -> Result<(PadSlots, Vec<F>), ZkError> {
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

    /// The prover's masked [`PrivateChannel`] over `transcript`.
    pub fn channel<'a, H: Sponge>(
        &'a mut self,
        transcript: &'a mut ProverTranscript<H>,
    ) -> MaskedProver<'a, F, O, H> {
        MaskedProver {
            pads: self,
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

    /// The masked private-message channel over `transcript`.
    pub fn channel<'a, 'p, H: Sponge>(
        &'a mut self,
        transcript: &'a mut VerifierTranscript<'p, H>,
    ) -> MaskedVerifier<'a, 'p, H> {
        MaskedVerifier {
            transcript,
            cursor: &mut self.cursor,
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

impl<C> CommittedPadsView<C> {
    /// The number of pads the masked messages read so far took.
    pub fn used(&self) -> usize {
        self.cursor.next
    }
}

/// The verifier's masked [`PrivateChannel`]: it reads m̂ in place of each
/// private value and returns the pad slots the prover used. Derived claims
/// bind nothing, since the masked messages they derive from already do.
pub struct MaskedVerifier<'a, 'p, H: Sponge> {
    transcript: &'a mut VerifierTranscript<'p, H>,
    cursor: &'a mut Cursor,
}

/// Public messages and challenges pass through to the transcript.
impl<H: Sponge> Channel for MaskedVerifier<'_, '_, H> {
    type Sponge = H;

    fn site(&mut self, site: SiteId) {
        self.transcript.site(site);
    }

    fn record_fork_draw(&mut self, absorbed: usize, squeezed: usize) {
        self.transcript.record_fork_draw(absorbed, squeezed);
    }

    fn public<A: CanonicalBytes>(&mut self, value: &A) {
        self.transcript.public(value);
    }

    fn public_all<A: CanonicalBytes>(&mut self, values: &[A]) {
        self.transcript.public_all(values);
    }

    fn public_bytes(&mut self, bytes: &[u8]) {
        self.transcript.public_bytes(bytes);
    }

    fn exchange<A: CanonicalDecode>(&mut self, value: &mut A) -> Result<(), TranscriptError> {
        self.transcript.exchange(value)
    }

    fn exchange_all<A: CanonicalDecode>(
        &mut self,
        values: &mut [A],
    ) -> Result<(), TranscriptError> {
        self.transcript.exchange_all(values)
    }

    fn challenge<F: Field>(&mut self) -> F {
        self.transcript.challenge()
    }

    fn challenge_small<F: CanonicalEncoding>(&mut self) -> F {
        self.transcript.challenge_small()
    }

    fn challenge_bytes<const N: usize>(&mut self) -> [u8; N] {
        self.transcript.challenge_bytes()
    }
}

impl<H: Sponge> PrivateChannel for MaskedVerifier<'_, '_, H> {
    type Error = ZkError;

    fn exchange_private<F, E>(&mut self, values: &mut [E]) -> Result<PadSlots, ZkError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F> + CanonicalDecode,
    {
        let len = values
            .len()
            .checked_mul(E::DEGREE)
            .ok_or(ZkError::InvalidLayout {
                reason: "extension pad count overflows usize",
            })?;
        self.transcript.exchange_all(values)?;
        self.cursor.take(len)
    }

    fn bind_derived<F, E>(&mut self, _claim: &E) -> Result<(), ZkError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F> + CanonicalDecode,
    {
        Ok(())
    }

    fn exchange_private_base<F>(&mut self, values: &mut [F]) -> Result<PadSlots, ZkError>
    where
        F: Field + CanonicalEncoding,
    {
        self.transcript.exchange_all(values)?;
        self.cursor.take(values.len())
    }

    fn bind_derived_base<F>(&mut self, _claim: &F) -> Result<(), ZkError>
    where
        F: Field + CanonicalEncoding,
    {
        Ok(())
    }
}

/// The prover's masked [`PrivateChannel`], for protocols written once over
/// both roles' channels, such as Akita's masked levels and `prove_single`
/// (Akita `specs/whiteout-akita.md` §4.5). Each private value goes out as
/// m̂ = m + p through [`CommittedPads::send`]'s masking, one pad per base
/// coordinate in the order [`MaskedVerifier`] takes them; derived claims bind
/// nothing.
///
/// The channel's methods are generic over the message field; a release check
/// fails with [`ZkError::MessageFieldMismatch`] unless it is the pad field `F`.
pub struct MaskedProver<'a, F, O, H: Sponge> {
    pads: &'a mut CommittedPads<F, O>,
    transcript: &'a mut ProverTranscript<H>,
}

/// `value` as a `T`, if `S` is `T`.
fn same_field<S: Copy + 'static, T: Copy + 'static>(value: S) -> Result<T, ZkError> {
    (&value as &dyn Any)
        .downcast_ref::<T>()
        .copied()
        .ok_or(ZkError::MessageFieldMismatch)
}

/// Public messages and challenges pass through to the transcript.
impl<F, O, H: Sponge> Channel for MaskedProver<'_, F, O, H> {
    type Sponge = H;

    fn site(&mut self, site: SiteId) {
        self.transcript.site(site);
    }

    fn record_fork_draw(&mut self, absorbed: usize, squeezed: usize) {
        self.transcript.record_fork_draw(absorbed, squeezed);
    }

    fn public<A: CanonicalBytes>(&mut self, value: &A) {
        self.transcript.public(value);
    }

    fn public_all<A: CanonicalBytes>(&mut self, values: &[A]) {
        self.transcript.public_all(values);
    }

    fn public_bytes(&mut self, bytes: &[u8]) {
        self.transcript.public_bytes(bytes);
    }

    fn exchange<A: CanonicalDecode>(&mut self, value: &mut A) -> Result<(), TranscriptError> {
        self.transcript.exchange(value)
    }

    fn exchange_all<A: CanonicalDecode>(
        &mut self,
        values: &mut [A],
    ) -> Result<(), TranscriptError> {
        self.transcript.exchange_all(values)
    }

    fn challenge<C: Field>(&mut self) -> C {
        self.transcript.challenge()
    }

    fn challenge_small<C: CanonicalEncoding>(&mut self) -> C {
        self.transcript.challenge_small()
    }

    fn challenge_bytes<const N: usize>(&mut self) -> [u8; N] {
        self.transcript.challenge_bytes()
    }
}

impl<P, O, H> PrivateChannel for MaskedProver<'_, P, O, H>
where
    P: Field + CanonicalEncoding,
    H: Sponge,
{
    type Error = ZkError;

    /// Masks each base coordinate of each value and sends the masked
    /// elements, so the bytes are `E`'s encoding, as the verifier reads them.
    fn exchange_private<F, E>(&mut self, values: &mut [E]) -> Result<PadSlots, ZkError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F> + CanonicalDecode,
    {
        let coordinates = values
            .iter()
            .flat_map(|value| (0..E::DEGREE).map(|index| value.base_coefficient(index)))
            .map(same_field::<F, P>)
            .collect::<Result<Vec<P>, ZkError>>()?;
        let (slots, masked) = self.pads.mask(&coordinates)?;
        let masked = masked
            .into_iter()
            .map(same_field::<P, F>)
            .collect::<Result<Vec<F>, ZkError>>()?;
        let mut masked: Vec<E> = masked
            .chunks_exact(E::DEGREE)
            .map(|chunk| {
                let mut coordinates = chunk.iter().copied();
                E::from_base_fn(|_| coordinates.next().unwrap_or_else(F::zero))
            })
            .collect();
        self.transcript.exchange_all(&mut masked)?;
        Ok(slots)
    }

    fn bind_derived<F, E>(&mut self, _claim: &E) -> Result<(), ZkError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F> + CanonicalDecode,
    {
        Ok(())
    }

    fn exchange_private_base<F>(&mut self, values: &mut [F]) -> Result<PadSlots, ZkError>
    where
        F: Field + CanonicalEncoding,
    {
        let values = values
            .iter()
            .copied()
            .map(same_field::<F, P>)
            .collect::<Result<Vec<P>, ZkError>>()?;
        let start = self.pads.cursor.next;
        let _ = self.pads.send(&values, self.transcript)?;
        Ok(PadSlots {
            start,
            len: values.len(),
        })
    }

    fn bind_derived_base<F>(&mut self, _claim: &F) -> Result<(), ZkError>
    where
        F: Field + CanonicalEncoding,
    {
        Ok(())
    }
}
