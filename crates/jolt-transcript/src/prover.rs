//! Prover-only operations on `spongefish::ProverState`.

#[cfg(feature = "prover-grinding")]
use crate::codec::RawBytes;
use crate::codec::{NargMessage, SpongeUnit};
#[cfg(feature = "prover-grinding")]
use crate::pow::{Leb128, SpongePow};
use crate::{prover_transcript, ProverState as JoltProverState, Transcript, TAG_BYTES};
#[cfg(feature = "prover-grinding")]
use spongefish::VerificationError;
use spongefish::{DuplexSpongeInit, DuplexSpongeInterface, Encoding, ProverState};

/// The prover's side of a [`Transcript`]: messages written to the NARG string
/// and the proof-of-work nonce searches.
pub trait ProverTranscript: Transcript {
    /// Absorbs `msg`'s [`Encoding`] ([`SpongeUnit::encode_message`]) and
    /// appends its NARG form
    /// ([`NargMessage::write_narg`]) for verifier replay.
    ///
    /// This is not spongefish's inherent `ProverState::prover_message`, which
    /// writes the absorbed encoding to the NARG string: for a [`BytesMsg`]
    /// that is the padded form, which the verifier's [`FromNarg`] does not
    /// read. The distinct name keeps the inherent method from shadowing this
    /// one on a concrete `ProverState`.
    ///
    /// [`BytesMsg`]: crate::BytesMsg
    /// [`FromNarg`]: spongefish::FromNarg
    fn send<M: NargMessage + ?Sized>(&mut self, msg: &M);

    /// Bytes accumulated in the NARG so far.
    fn narg_string(&self) -> &[u8];

    /// An owned callback that writes into its second argument the units the
    /// transcript would squeeze right after absorbing its first, the byte
    /// encoding of a candidate prover message, absorbed as one message
    /// ([`SpongeUnit::encode_message`]). The transcript is unchanged, and
    /// the callback holds a copy of the public sponge only: no private
    /// randomness, no proof bytes. This is the extension oracle of
    /// state-restoration soundness, shaped like spongefish's
    /// `ProverState::verifier_message_oracle` (arkworks-rs/spongefish#306),
    /// which is not in the published 0.8.0.
    ///
    /// A prover uses it to search over the next prover message, a
    /// proof-of-work nonce or a retry counter whose following challenges must
    /// satisfy a prover-side condition, and then sends the chosen message
    /// through the ordinary calls.
    ///
    /// Needs the `prover-grinding` feature, which enables spongefish's
    /// `yolocrypto` to copy the sponge state.
    #[cfg(feature = "prover-grinding")]
    fn verifier_message_oracle(&self) -> impl Fn(&[u8], &mut [Self::Unit]) + Clone + use<Self>;

    /// A proof of work of `bits` bits whose nonce goes into the NARG string.
    ///
    /// For `bits == 0` this does nothing. For `1 <= bits <= MAX_POW_BITS`, the
    /// prover sends the least nonce below `2^(bits + 7)` for which the low
    /// `bits` bits of the next squeezed check are zero, and squeezes the
    /// check: eight bytes read as a little-endian `u64` on a byte sponge, one
    /// element's canonical integer on `PoseidonSponge`. The nonce is one prover
    /// message: canonical LEB128 in the NARG string, one `u64` LE word in the
    /// sponge.
    ///
    /// The step protects the verifier messages drawn after it, up to the next
    /// prover message. The protocol must draw the challenges it protects
    /// directly after this step: a prover message in between would let the
    /// prover change them without redoing the work.
    ///
    /// # Security
    ///
    /// With an ideal sponge, each attempt passes the check with probability
    /// `2^-bits`, and the protected challenges come from the bytes after the
    /// check. For the Keccak duplex sponge, the reduction of Chiesa and Orrù
    /// prices this per permutation call; the Blake2b hash bridge and Poseidon
    /// need their own argument. The difficulty, the position of this step and the
    /// challenges it protects must be fixed by the protocol and bound in its
    /// session.
    ///
    /// # Errors
    /// Returns `VerificationError`, before touching the transcript, if
    /// `bits > MAX_POW_BITS` or the nonce search exhausts the nonce space
    /// (probability at most `e^-128` for an ideal sponge). Also returns it,
    /// after the nonce is sent, if the real check disagrees with the search;
    /// that means the sponge's absorb is not what the oracle replays, a bug.
    #[cfg(feature = "prover-grinding")]
    fn grind(&mut self, bits: u32) -> Result<(), VerificationError>;

    /// A proof of work of `bits` bits whose nonce the proof carries outside the
    /// NARG string. Returns that nonce, or `None` for `bits == 0`, which leaves
    /// the transcript unchanged.
    ///
    /// The nonce is the least one below `2^(bits + 7)` that passes
    /// [`check_public_grind`](Transcript::check_public_grind), which is then
    /// applied to this transcript.
    ///
    /// # Errors
    /// As [`grind`](Self::grind).
    #[cfg(feature = "prover-grinding")]
    fn public_grind(&mut self, bits: u32) -> Result<Option<u64>, VerificationError>;
}

/// A prover transcript that a prover builds itself, bound to the same
/// instance its verifier binds: from the session and the public statement, as
/// [`prover_transcript`] does.
pub trait ProverInit: ProverTranscript + Sized {
    /// The transcript bound to `session` and `statement`.
    fn init(session: &[u8; TAG_BYTES], statement: &(impl Encoding + ?Sized)) -> Self;
}

impl<H> ProverInit for JoltProverState<H>
where
    H: DuplexSpongeInit<U: SpongeUnit>,
{
    fn init(session: &[u8; TAG_BYTES], statement: &(impl Encoding + ?Sized)) -> Self {
        prover_transcript::<H>(session, statement)
    }
}

impl<H, R> ProverTranscript for ProverState<H, R>
where
    H: DuplexSpongeInterface<U: SpongeUnit>,
    R: DuplexSpongeInit<U = u8>,
{
    fn send<M: NargMessage + ?Sized>(&mut self, msg: &M) {
        ProverState::prover_message_with(
            self,
            msg,
            |msg| H::U::encode_message(msg),
            |msg, narg| {
                msg.write_narg(narg);
            },
        );
    }

    fn narg_string(&self) -> &[u8] {
        ProverState::narg_string(self)
    }

    #[cfg(feature = "prover-grinding")]
    fn verifier_message_oracle(&self) -> impl Fn(&[u8], &mut [H::U]) + Clone + use<H, R> {
        let snapshot = self.duplex_sponge_state.clone();
        move |candidate, output| {
            let mut sponge = snapshot.clone();
            let _ = sponge.absorb(H::U::encode_message(&RawBytes(candidate)).as_ref());
            let _ = sponge.squeeze(output);
        }
    }

    #[cfg(feature = "prover-grinding")]
    fn grind(&mut self, bits: u32) -> Result<(), VerificationError> {
        if bits == 0 {
            return Ok(());
        }
        let pow = SpongePow::new(bits).ok_or(VerificationError)?;
        let nonce = pow
            .search(self.verifier_message_oracle())
            .ok_or(VerificationError)?;
        ProverState::prover_message_with(
            self,
            &nonce,
            |&nonce| H::U::encode_word(nonce),
            |&nonce, narg| narg.extend_from_slice(Leb128::new(nonce).as_bytes()),
        );
        if ProverState::verifier_message_as(self, H::U::CHECK_LEN, |check| pow.accepts(check)) {
            Ok(())
        } else {
            Err(VerificationError)
        }
    }

    #[cfg(feature = "prover-grinding")]
    fn public_grind(&mut self, bits: u32) -> Result<Option<u64>, VerificationError> {
        if bits == 0 {
            return Ok(None);
        }
        let pow = SpongePow::new(bits).ok_or(VerificationError)?;
        let nonce = pow
            .search(self.verifier_message_oracle())
            .ok_or(VerificationError)?;
        self.check_public_grind(bits, Some(nonce))?;
        Ok(Some(nonce))
    }
}
