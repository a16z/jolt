//! Spongefish-native [`ProverTranscript`] surface.
//!
//! Implemented directly on `spongefish::ProverState<H, R>` via the orphan
//! rule. Methods are positional, matching spongefish-native usage
//! (WhiR, sigma-rs).

#[cfg(feature = "prover-grinding")]
use crate::pow::{Leb128, MessageSearch, SpongePow};
#[cfg(all(feature = "bn254", feature = "transcript-blake2b"))]
use crate::Blake2b512;
use crate::{challenge::message_bytes, ChallengeCodec, UniformBytes};
#[cfg(feature = "bn254")]
use jolt_field::Fr;
#[cfg(all(feature = "bn254", feature = "transcript-keccak"))]
use spongefish::instantiations::Keccak;
use spongefish::{
    DuplexSpongeInit, DuplexSpongeInterface, Encoding, FromUniform, ProverState, VerificationError,
};

/// Prover-side spongefish transcript.
///
/// `H::U` is the sponge alphabet (`u8` for every sponge in this crate).
pub trait ProverTranscript<H: DuplexSpongeInterface> {
    /// Absorbs `msg` symmetrically with the verifier; emits no NARG bytes.
    fn public_message<T: Encoding<H::U> + ?Sized>(&mut self, msg: &T);

    /// Absorbs `msg` and appends its NARG-serialized form for verifier replay.
    fn prover_message<T: Encoding<H::U> + Encoding + ?Sized>(&mut self, msg: &T);

    /// Squeezes a verifier challenge.
    fn verifier_message<T: FromUniform<H::U>>(&mut self) -> T;

    /// Draws one message with conventional TV ≤ 2^-161 across all `len` elements.
    /// The protocol session/version must bind the codec, field, and length.
    /// `len == 0` returns an empty vector without touching the sponge. Only
    /// sponges with uniform output qualify ([`UniformBytes`]); `PoseidonSponge`
    /// does not.
    ///
    /// # Errors
    /// Returns `VerificationError`, before any sponge operation, when the codec
    /// does not support `len` or the message would exceed
    /// [`MAX_CHALLENGE_BYTES`](crate::MAX_CHALLENGE_BYTES) bytes. Otherwise on
    /// rejection-pool exhaustion (probability at most 2^-161).
    fn field_challenges<F: ChallengeCodec>(
        &mut self,
        len: usize,
    ) -> Result<Vec<F>, VerificationError>
    where
        H: UniformBytes;

    /// A copy of the public sponge, for searching over the next prover
    /// message: a proof-of-work nonce, or a retry counter whose following
    /// challenges must satisfy a prover-side condition. The prover then sends
    /// the chosen message through the ordinary calls.
    ///
    /// Needs the `prover-grinding` feature, which enables spongefish's
    /// `yolocrypto` to copy the sponge state.
    #[cfg(feature = "prover-grinding")]
    fn message_search(&self) -> MessageSearch<H>;

    /// A proof of work of `bits` bits on this transcript's own sponge.
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
    /// Needs the `prover-grinding` feature, which enables spongefish's
    /// `yolocrypto` to copy the sponge state for the nonce search.
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
    fn grind(&mut self, bits: u32) -> Result<(), VerificationError>
    where
        H: DuplexSpongeInterface<U = u8> + Sync;

    /// Bytes accumulated in the NARG so far.
    fn narg_string(&self) -> &[u8];
}

impl<H, R> ProverTranscript<H> for ProverState<H, R>
where
    H: DuplexSpongeInterface,
    R: DuplexSpongeInit<U = u8>,
{
    fn public_message<T: Encoding<H::U> + ?Sized>(&mut self, msg: &T) {
        ProverState::public_message(self, msg);
    }

    fn prover_message<T: Encoding<H::U> + Encoding + ?Sized>(&mut self, msg: &T) {
        ProverState::prover_message(self, msg);
    }

    fn verifier_message<T: FromUniform<H::U>>(&mut self) -> T {
        ProverState::verifier_message::<T>(self)
    }

    fn field_challenges<F: ChallengeCodec>(
        &mut self,
        len: usize,
    ) -> Result<Vec<F>, VerificationError>
    where
        H: UniformBytes,
    {
        let n = message_bytes::<F>(len)?;
        if len == 0 {
            return Ok(Vec::new());
        }
        ProverState::verifier_message_as(self, n, |bytes| F::decode_challenges(bytes, len))
            .ok_or(VerificationError)
    }

    #[cfg(feature = "prover-grinding")]
    fn message_search(&self) -> MessageSearch<H> {
        MessageSearch::new(self.duplex_sponge_state.clone())
    }

    #[cfg(feature = "prover-grinding")]
    fn grind(&mut self, bits: u32) -> Result<(), VerificationError>
    where
        H: DuplexSpongeInterface<U = u8> + Sync,
    {
        if bits == 0 {
            return Ok(());
        }
        let pow = SpongePow::new(bits).ok_or(VerificationError)?;
        let nonce = pow
            .search(&ProverTranscript::message_search(self))
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

    fn narg_string(&self) -> &[u8] {
        ProverState::narg_string(self)
    }
}

/// 128-bit-truncating challenge decoder. Implemented for sponges where the
/// optimization is sound (Blake2b, Keccak); deliberately not implemented
/// for [`PoseidonSponge`](crate::PoseidonSponge), so calling it on a
/// Poseidon-backed state is a compile error.
#[cfg(feature = "bn254")]
pub trait OptimizedChallenge {
    /// Squeezes a 128-bit-truncated challenge as an [`Fr`].
    fn challenge_128(&mut self) -> Fr;
}

#[cfg(all(feature = "bn254", feature = "transcript-blake2b"))]
impl<R> OptimizedChallenge for ProverState<Blake2b512, R>
where
    R: DuplexSpongeInit<U = u8>,
{
    fn challenge_128(&mut self) -> Fr {
        Fr::from(ProverState::verifier_message::<u128>(self))
    }
}

#[cfg(all(feature = "bn254", feature = "transcript-keccak"))]
impl<R> OptimizedChallenge for ProverState<Keccak, R>
where
    R: DuplexSpongeInit<U = u8>,
{
    fn challenge_128(&mut self) -> Fr {
        Fr::from(ProverState::verifier_message::<u128>(self))
    }
}
