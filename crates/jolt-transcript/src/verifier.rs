//! Challenge helpers for spongefish verifier states.

use spongefish::VerifierState;

use crate::prover::OptimizedChallenge;

#[cfg(feature = "transcript-blake2b")]
impl OptimizedChallenge for VerifierState<'_, spongefish::instantiations::Blake2b512> {
    fn challenge_u128(&mut self) -> u128 {
        VerifierState::verifier_message::<u128>(self)
    }
}

#[cfg(feature = "transcript-keccak")]
impl OptimizedChallenge for VerifierState<'_, spongefish::instantiations::Keccak> {
    fn challenge_u128(&mut self) -> u128 {
        VerifierState::verifier_message::<u128>(self)
    }
}

// Poseidon `OptimizedChallenge` (verifier side) — deliberately UNIMPLEMENTED, mirror of the
// prover impl (#1586 reviewer / D5b). See the fuller note in `prover.rs`.
#[cfg(feature = "transcript-poseidon")]
impl OptimizedChallenge for VerifierState<'_, crate::PoseidonSponge> {
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
