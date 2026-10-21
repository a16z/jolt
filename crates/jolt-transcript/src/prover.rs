//! Prover-only operations on `spongefish::ProverState`.

#[cfg(feature = "prover-grinding")]
use crate::pow::{Leb128, MessageSearch, SpongePow};
use crate::Transcript;
#[cfg(feature = "prover-grinding")]
use spongefish::VerificationError;
use spongefish::{DuplexSpongeInit, DuplexSpongeInterface, Encoding, ProverState};

/// The prover's side of a [`Transcript`]: messages written to the NARG string
/// and the proof-of-work nonce searches.
pub trait ProverTranscript: Transcript {
    /// The duplex sponge behind the transcript.
    type Sponge: DuplexSpongeInterface<U = u8>;

    /// Absorbs `msg` and appends its NARG-serialized form for verifier replay.
    fn prover_message<M: Encoding + ?Sized>(&mut self, msg: &M);

    /// Bytes accumulated in the NARG so far.
    fn narg_string(&self) -> &[u8];

    /// A copy of the public sponge, for searching over the next prover
    /// message: a proof-of-work nonce, or a retry counter whose following
    /// challenges must satisfy a prover-side condition. The prover then sends
    /// the chosen message through the ordinary calls.
    ///
    /// Needs the `prover-grinding` feature, which enables spongefish's
    /// `yolocrypto` to copy the sponge state.
    #[cfg(feature = "prover-grinding")]
    fn message_search(&self) -> MessageSearch<Self::Sponge>;

    /// A proof of work of `bits` bits whose nonce goes into the NARG string.
    ///
    /// For `bits == 0` this does nothing. For `1 <= bits <= MAX_POW_BITS`, the
    /// prover sends the least nonce below `2^(bits + 7)` for which the next
    /// `ceil(bits / 8)` squeezed bytes have their low `bits` bits zero, as
    /// canonical LEB128 with one byte per prover message, and squeezes those
    /// bytes. The layout is the one proposed in arkworks-rs/spongefish#301.
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
    /// that means the sponge's absorb is not what the probe replays, a bug.
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
    fn public_grind(&mut self, bits: u32) -> Result<Option<u32>, VerificationError>;
}

impl<H, R> ProverTranscript for ProverState<H, R>
where
    H: DuplexSpongeInterface<U = u8> + Sync,
    R: DuplexSpongeInit<U = u8>,
{
    type Sponge = H;

    fn prover_message<M: Encoding + ?Sized>(&mut self, msg: &M) {
        ProverState::prover_message(self, msg);
    }

    fn narg_string(&self) -> &[u8] {
        ProverState::narg_string(self)
    }

    #[cfg(feature = "prover-grinding")]
    fn message_search(&self) -> MessageSearch<H> {
        MessageSearch::new(self.duplex_sponge_state.clone())
    }

    #[cfg(feature = "prover-grinding")]
    fn grind(&mut self, bits: u32) -> Result<(), VerificationError> {
        if bits == 0 {
            return Ok(());
        }
        let pow = SpongePow::new(bits).ok_or(VerificationError)?;
        let nonce = pow
            .search(&self.message_search())
            .ok_or(VerificationError)?;
        for byte in Leb128::new(nonce).as_bytes() {
            ProverState::prover_message(self, byte);
        }
        if ProverState::verifier_message_as(self, pow.word_len(), |check| pow.accepts(check)) {
            Ok(())
        } else {
            Err(VerificationError)
        }
    }

    #[cfg(feature = "prover-grinding")]
    fn public_grind(&mut self, bits: u32) -> Result<Option<u32>, VerificationError> {
        if bits == 0 {
            return Ok(None);
        }
        let pow = SpongePow::new(bits).ok_or(VerificationError)?;
        let nonce = pow
            .search(&self.message_search())
            .ok_or(VerificationError)?;
        self.check_public_grind(bits, Some(nonce))?;
        Ok(Some(nonce))
    }
}
