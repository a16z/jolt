//! Spongefish-native [`ProverTranscript`] surface.
//!
//! Implemented directly on `spongefish::ProverState<H, R>` via the orphan
//! rule. Methods are positional, matching spongefish-native usage
//! (WhiR, sigma-rs).

use jolt_field::Fr;
use rand::{CryptoRng, RngCore};
use spongefish::{Decoding, DuplexSpongeInterface, Encoding, NargSerialize, ProverState};

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

/// 128-bit challenge decoder, defined for all three sponges.
///
/// Blake2b/Keccak squeeze a 16-byte `u128` directly — sound because every byte
/// they emit is uniform. [`PoseidonSponge`](crate::PoseidonSponge) squeezes a
/// *field element* whose high bytes are bounded by the BN254 modulus (so not
/// uniform); it therefore squeezes a whole element and keeps the uniform low
/// 128 bits. Either way the result is a 128-bit value embedded in [`Fr`], so
/// downstream fast-multiplication paths are identical across sponges.
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

// OPTIONAL (remove if not needed) — this impl is NOT required by the transcript
// migration. Poseidon's optimized challenge was historically a FULL 254-bit field
// element (`transcript-poseidon` forces `challenge-254-bit`; legacy jolt-core's
// `challenge_scalar_128_bits` for Poseidon is `unimplemented!`). The migration can
// preserve that by mapping Poseidon's `challenge_scalar_optimized` to
// `verifier_message::<Fr>()` (full field), in which case `challenge_128` is never
// called on Poseidon and this impl is dead code. It exists only to OPTIONALLY give
// Poseidon the same 128-bit fast-multiply path as Blake2b/Keccak, which requires
// decoupling `transcript-poseidon` from `challenge-254-bit` (decision D5b). Since
// `PoseidonSponge` is itself slated for a DSFS rewrite (see the TODO above the
// struct), prefer NOT to pin a 128-bit Poseidon challenge format unless D5b is
// deliberately taken in Phase 3. Decision point: jolt-core wiring of
// `challenge_scalar_optimized` for the Poseidon config.
#[cfg(feature = "transcript-poseidon")]
impl<R> OptimizedChallenge for ProverState<crate::PoseidonSponge, R>
where
    R: RngCore + CryptoRng,
{
    fn challenge_u128(&mut self) -> u128 {
        // Field-native sponge: squeeze a full element (the raw ark type, which
        // spongefish can decode) and keep the uniform low 128 bits — its high
        // bytes are modulus-bounded.
        crate::poseidon::low_128_bits(ProverState::verifier_message::<ark_bn254::Fr>(self))
    }
}
