//! Challenge helpers for spongefish prover states.

use jolt_field::Fr;
use rand::{CryptoRng, RngCore};
use spongefish::ProverState;

/// 128-bit challenge decoder.
///
/// Blake2b/Keccak squeeze a 16-byte `u128` directly — sound because every byte
/// they emit is uniform — yielding a 128-bit value embedded in [`Fr`] for the
/// downstream fast-multiplication path. [`PoseidonSponge`](crate::PoseidonSponge)
/// deliberately does **not** implement a meaningful 128-bit challenge: its impl is
/// `unimplemented!()` because `transcript-poseidon` uses full-field
/// `challenge-254-bit` (truncating defeats Poseidon's recursion purpose). The
/// Poseidon impl exists only so generic-over-sponge bounds resolve.
pub trait OptimizedChallenge {
    /// Squeezes a 128-bit verifier challenge as a raw `u128` — the uniform low
    /// 128 bits of the sponge output.
    ///
    /// This is the primitive: callers that want the fast-multiply field
    /// challenge wrapper (`JoltField::Challenge`, e.g. `MontU128Challenge`)
    /// build it via `Challenge::from(u128)`, which preserves the 128-bit
    /// fast-multiply path. [`challenge_128`](Self::challenge_128) is the
    /// embedded-in-[`Fr`] convenience built on top of it.
    fn challenge_u128(&mut self) -> u128;

    /// Squeezes the same 128-bit challenge embedded in an [`Fr`].
    fn challenge_128(&mut self) -> Fr {
        Fr::from(self.challenge_u128())
    }
}

#[cfg(feature = "transcript-blake2b")]
impl<R> OptimizedChallenge for ProverState<spongefish::instantiations::Blake2b512, R>
where
    R: RngCore + CryptoRng,
{
    fn challenge_u128(&mut self) -> u128 {
        ProverState::verifier_message::<u128>(self)
    }
}

#[cfg(feature = "transcript-keccak")]
impl<R> OptimizedChallenge for ProverState<spongefish::instantiations::Keccak, R>
where
    R: RngCore + CryptoRng,
{
    fn challenge_u128(&mut self) -> u128 {
        ProverState::verifier_message::<u128>(self)
    }
}

// Poseidon `OptimizedChallenge` — deliberately UNIMPLEMENTED (#1586 reviewer / D5b): the
// 128-bit truncation is costly for recursion and defeats Poseidon's purpose, so
// `transcript-poseidon` forces `challenge-254-bit` (genuine full-field challenges) and
// `challenge_u128` is `unimplemented!()` (legacy `challenge_scalar_128_bits`'s analogue).
// The impl is KEPT (not omitted) so generic-over-sponge `OptimizedChallenge` bounds still
// resolve for Poseidon.
#[cfg(feature = "transcript-poseidon")]
impl<R> OptimizedChallenge for ProverState<crate::PoseidonSponge, R>
where
    R: RngCore + CryptoRng,
{
    #[expect(
        clippy::unimplemented,
        reason = "Poseidon uses full-field challenge-254-bit; 128-bit truncation is unsupported (#1586 reviewer)"
    )]
    fn challenge_u128(&mut self) -> u128 {
        unimplemented!(
            "128-bit optimized challenges are unsupported for the Poseidon sponge; \
             transcript-poseidon uses full-field challenge-254-bit"
        )
    }
}
