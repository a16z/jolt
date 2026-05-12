//! Spongefish-native [`ProverTranscript`] surface.
//!
//! Implemented directly on `spongefish::ProverState<H, R>` via the orphan
//! rule. Methods are positional, matching spongefish-native usage
//! (WhiR, sigma-rs).

use ark_bn254::Fr;
use rand::{CryptoRng, RngCore};
use spongefish::{Decoding, DuplexSpongeInterface, Encoding, NargSerialize, ProverState};

use crate::codec::FieldElOptimized;

/// Prover-side spongefish transcript.
///
/// `H::U` is the sponge alphabet (`u8` for every sponge in this crate).
pub trait ProverTranscript<H: DuplexSpongeInterface> {
    /// Absorbs `msg` symmetrically with the verifier; emits no NARG bytes.
    fn public_message<T: Encoding<[H::U]> + ?Sized>(&mut self, msg: &T);

    /// Absorbs `msg` and appends its NARG-serialized form for verifier replay.
    fn prover_message<T: Encoding<[H::U]> + NargSerialize + ?Sized>(&mut self, msg: &T);

    /// Squeezes a verifier challenge.
    fn verifier_message<T: Decoding<[H::U]>>(&mut self) -> T;

    /// Bytes accumulated in the NARG so far.
    fn narg_string(&self) -> &[u8];
}

impl<H, R> ProverTranscript<H> for ProverState<H, R>
where
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
{
    fn public_message<T: Encoding<[H::U]> + ?Sized>(&mut self, msg: &T) {
        ProverState::public_message(self, msg);
    }

    fn prover_message<T: Encoding<[H::U]> + NargSerialize + ?Sized>(&mut self, msg: &T) {
        ProverState::prover_message(self, msg);
    }

    fn verifier_message<T: Decoding<[H::U]>>(&mut self) -> T {
        ProverState::verifier_message::<T>(self)
    }

    fn narg_string(&self) -> &[u8] {
        ProverState::narg_string(self)
    }
}

/// 128-bit-truncating challenge decoder. Implemented for sponges where the
/// optimization is sound (Blake2b, Keccak); deliberately not implemented
/// for [`PoseidonSponge`](crate::PoseidonSponge), so calling it on a
/// Poseidon-backed state is a compile error.
pub trait OptimizedChallenge {
    /// Squeezes a 128-bit-truncated challenge as an [`Fr`].
    fn challenge_128(&mut self) -> Fr;
}

#[cfg(feature = "transcript-blake2b")]
impl<R> OptimizedChallenge for ProverState<spongefish::instantiations::Blake2b512, R>
where
    R: RngCore + CryptoRng,
{
    fn challenge_128(&mut self) -> Fr {
        ProverState::verifier_message::<FieldElOptimized>(self).0
    }
}

#[cfg(feature = "transcript-keccak")]
impl<R> OptimizedChallenge for ProverState<spongefish::instantiations::Keccak, R>
where
    R: RngCore + CryptoRng,
{
    fn challenge_128(&mut self) -> Fr {
        ProverState::verifier_message::<FieldElOptimized>(self).0
    }
}
