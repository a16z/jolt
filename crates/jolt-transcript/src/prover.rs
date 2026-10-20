//! Spongefish-native [`ProverTranscript`] surface.
//!
//! Implemented directly on `spongefish::ProverState<H, R>` via the orphan
//! rule. Methods are positional, matching spongefish-native usage
//! (WhiR, sigma-rs).

#[cfg(all(feature = "bn254", feature = "transcript-blake2b"))]
use crate::Blake2b512;
use crate::{challenge::message_bytes, ChallengeCodec};
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
    /// `len == 0` returns an empty vector without touching the sponge.
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
        H: DuplexSpongeInterface<U = u8>;

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
        H: DuplexSpongeInterface<U = u8>,
    {
        let n = message_bytes::<F>(len)?;
        if len == 0 {
            return Ok(Vec::new());
        }
        ProverState::verifier_message_as(self, n, |bytes| F::decode_challenges(bytes, len))
            .ok_or(VerificationError)
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
