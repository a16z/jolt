//! Spongefish-native [`VerifierTranscript`] surface.

#[cfg(all(feature = "bn254", feature = "transcript-blake2b"))]
use crate::Blake2b512;
use crate::{
    pow::{PowUnit, SpongePow},
    ChallengeSponge,
};
#[cfg(all(
    feature = "bn254",
    any(feature = "transcript-blake2b", feature = "transcript-keccak")
))]
use jolt_field::Fr;
#[cfg(all(feature = "bn254", feature = "transcript-keccak"))]
use spongefish::instantiations::Keccak;
use spongefish::{
    DuplexSpongeInterface, Encoding, FromNarg, FromUniform, NargReader, Unit, VerificationError,
    VerifierState,
};

#[cfg(all(
    feature = "bn254",
    any(feature = "transcript-blake2b", feature = "transcript-keccak")
))]
use crate::prover::OptimizedChallenge;

/// Verifier-side spongefish transcript.
pub trait VerifierTranscript<H: DuplexSpongeInterface> {
    /// Absorbs `msg` symmetrically with the prover.
    fn public_message<T: Encoding<H::U> + ?Sized>(&mut self, msg: &T);

    /// Reads a prover message from the NARG, absorbing it into the sponge.
    fn prover_message<T: Encoding<H::U> + FromNarg>(&mut self) -> Result<T, VerificationError>;

    /// Squeezes a verifier challenge.
    fn verifier_message<T: FromUniform<H::U>>(&mut self) -> T;

    /// Draws one message with conventional TV ≤ 2^-161 across all `len` elements.
    /// The protocol session/version must bind the codec, field, and length.
    /// `len == 0` returns an empty vector without touching the sponge. Byte
    /// sponges with uniform output ([`UniformBytes`](crate::UniformBytes))
    /// decode every [`ChallengeCodec`](crate::ChallengeCodec) field;
    /// `PoseidonSponge` squeezes `Fr` natively and offers no other field
    /// ([`ChallengeSponge`]).
    ///
    /// # Errors
    /// Returns `VerificationError`, before any sponge operation, when `F` does
    /// not support `len` or the message would exceed
    /// [`MAX_CHALLENGE_BYTES`](crate::MAX_CHALLENGE_BYTES) bytes. Otherwise on
    /// rejection-pool exhaustion (probability at most 2^-161). Every failure
    /// poisons the state, so later reads and `check_eof` fail.
    fn field_challenges<F>(&mut self, len: usize) -> Result<Vec<F>, VerificationError>
    where
        H: ChallengeSponge<F>;

    /// Checks a proof of work of `bits` bits on this transcript's own sponge.
    ///
    /// For `bits == 0` this does nothing. For `1 <= bits <= MAX_POW_BITS`, it
    /// reads a nonce below `2^(bits + 7)` as canonical LEB128, one byte per
    /// prover message, then squeezes the check and requires its low `bits`
    /// bits to be zero. On a byte sponge the check is `ceil(bits / 8)` bytes
    /// read little-endian; on `PoseidonSponge` it is one element's canonical
    /// integer ([`PowUnit`](crate::PowUnit)). The byte-sponge layout is the one
    /// proposed in arkworks-rs/spongefish#301.
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
    /// The verifier accepts any nonce in range that passes, not only the least
    /// one, so proof bytes are not unique; an honest prover sends the least.
    ///
    /// # Errors
    /// Returns `VerificationError`, before any sponge operation, if
    /// `bits > MAX_POW_BITS`. Otherwise on a truncated, non-canonical or
    /// out-of-range nonce, or one whose check bytes are not zero. Every
    /// failure poisons the state, so later reads and `check_eof` fail.
    fn check_grind(&mut self, bits: u32) -> Result<(), VerificationError>
    where
        H::U: PowUnit,
        u8: Encoding<H::U>;

    /// Asserts the NARG was fully consumed.
    ///
    /// Soundness-critical: without this check, `valid_proof || garbage`
    /// verifies as if the trailing bytes were absent, making the top-level
    /// proof bytes malleable. Call once at the end of the top-level verify
    /// function on the success path; error paths skip it (the proof is
    /// already rejected). Composite verifiers follow the same rule at
    /// their own finalize boundary.
    ///
    /// Takes `self` by value to prevent reuse.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the NARG has unread bytes.
    fn check_eof(self) -> Result<(), VerificationError>;
}

/// A prover message that never parses: reading one poisons the verifier
/// without touching the sponge, over any unit.
struct Rejection;

impl<U: Unit> Encoding<U> for Rejection {
    fn encode(&self) -> impl AsRef<[U]> {
        let empty: [U; 0] = [];
        empty
    }
}

impl FromNarg for Rejection {
    fn from_narg(_: &mut NargReader<'_>) -> Result<Self, VerificationError> {
        Err(VerificationError)
    }
}

/// Fails unless `holds` and the state is unpoisoned; a failure poisons the
/// state, so later reads and `check_eof` fail too. Spongefish's
/// `Transcript::check` covers byte sponges only, and neither branch here
/// absorbs or squeezes.
fn check<H: DuplexSpongeInterface>(
    state: &mut VerifierState<'_, H>,
    holds: bool,
) -> Result<(), VerificationError> {
    if holds {
        state.prover_messages_vec::<Rejection>(0).map(drop)
    } else {
        state.prover_message::<Rejection>().map(drop)
    }
}

impl<H> VerifierTranscript<H> for VerifierState<'_, H>
where
    H: DuplexSpongeInterface,
{
    fn public_message<T: Encoding<H::U> + ?Sized>(&mut self, msg: &T) {
        VerifierState::public_message(self, msg);
    }

    fn prover_message<T: Encoding<H::U> + FromNarg>(&mut self) -> Result<T, VerificationError> {
        VerifierState::prover_message::<T>(self)
    }

    fn verifier_message<T: FromUniform<H::U>>(&mut self) -> T {
        VerifierState::verifier_message::<T>(self)
    }

    fn field_challenges<F>(&mut self, len: usize) -> Result<Vec<F>, VerificationError>
    where
        H: ChallengeSponge<F>,
    {
        // A length the sponge rejects poisons the state too, so a caller that
        // drops the error still fails `check_eof`.
        let n = H::challenge_units(len);
        check(self, n.is_some())?;
        let n = n.ok_or(VerificationError)?;
        if len == 0 {
            return Ok(Vec::new());
        }
        let challenges =
            VerifierState::verifier_message_as(self, n, |units| H::decode_challenges(units, len));
        // Exhaustion poisons the state, so later reads and `check_eof` fail too.
        check(self, challenges.is_some())?;
        challenges.ok_or(VerificationError)
    }

    fn check_grind(&mut self, bits: u32) -> Result<(), VerificationError>
    where
        H::U: PowUnit,
        u8: Encoding<H::U>,
    {
        if bits == 0 {
            return Ok(());
        }
        let Some(pow) = SpongePow::new(bits) else {
            return check(self, false);
        };
        // Failed reads poison the state themselves; failed checks poison it
        // through `check`, so `check_eof` fails afterwards too.
        let nonce = pow.read_nonce(|| VerifierState::prover_message::<u8>(self))?;
        check(self, nonce.is_some())?;
        let passes = VerifierState::verifier_message_as(self, H::U::check_len(bits), |check| {
            pow.accepts(check)
        });
        check(self, passes)
    }

    fn check_eof(self) -> Result<(), VerificationError> {
        VerifierState::check_eof(self)
    }
}

#[cfg(all(feature = "bn254", feature = "transcript-blake2b"))]
impl OptimizedChallenge for VerifierState<'_, Blake2b512> {
    fn challenge_128(&mut self) -> Fr {
        Fr::from(VerifierState::verifier_message::<u128>(self))
    }
}

#[cfg(all(feature = "bn254", feature = "transcript-keccak"))]
impl OptimizedChallenge for VerifierState<'_, Keccak> {
    fn challenge_128(&mut self) -> Fr {
        Fr::from(VerifierState::verifier_message::<u128>(self))
    }
}
