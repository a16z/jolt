//! Verifier-only operations on `spongefish::VerifierState`.

use crate::pow::{nonce_word, SpongePow, CHECK_BYTES};
use crate::{verifier_transcript, Transcript, TAG_BYTES};
use spongefish::{
    DuplexSpongeInit, DuplexSpongeInterface, Encoding, FromNarg, Transcript as _,
    VerificationError, VerifierState,
};

/// The verifier's side of a [`Transcript`]: messages read from the NARG
/// string, and its end.
pub trait VerifierTranscript: Transcript {
    /// Reads a prover message from the NARG, absorbing it into the sponge.
    ///
    /// # Errors
    /// `VerificationError` on a truncated or malformed message; the state is
    /// poisoned then.
    fn prover_message<T: Encoding + FromNarg>(&mut self) -> Result<T, VerificationError>;

    /// Checks a proof of work of `bits` bits whose nonce is in the NARG string.
    ///
    /// For `bits == 0` this does nothing. For `1 <= bits <= MAX_POW_BITS`, it
    /// reads a nonce below `2^(bits + 7)` as canonical LEB128 from the NARG
    /// string and absorbs it as one `u64` little-endian word, then squeezes
    /// eight bytes and requires the low `bits` bits of their little-endian
    /// `u64` to be zero.
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
    fn check_grind(&mut self, bits: u32) -> Result<(), VerificationError>;

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
    /// Returns `Err` if the NARG has unread bytes or the state is poisoned.
    fn check_eof(self) -> Result<(), VerificationError>;
}

/// A verifier transcript that a verifier builds itself, so that it binds the
/// instance it checks: from the session, the public statement and the
/// proof's NARG string, as [`verifier_transcript`] does.
pub trait VerifierInit<'a>: VerifierTranscript + Sized {
    /// The transcript bound to `session` and `statement` over `narg`.
    fn init(
        session: &[u8; TAG_BYTES],
        statement: &(impl Encoding + ?Sized),
        narg: &'a [u8],
    ) -> Self;
}

impl<'a, H: DuplexSpongeInit<U = u8>> VerifierInit<'a> for VerifierState<'a, H> {
    fn init(
        session: &[u8; TAG_BYTES],
        statement: &(impl Encoding + ?Sized),
        narg: &'a [u8],
    ) -> Self {
        verifier_transcript::<H>(session, statement, narg)
    }
}

impl<H> VerifierTranscript for VerifierState<'_, H>
where
    H: DuplexSpongeInterface<U = u8>,
{
    fn prover_message<T: Encoding + FromNarg>(&mut self) -> Result<T, VerificationError> {
        VerifierState::prover_message::<T>(self)
    }

    fn check_grind(&mut self, bits: u32) -> Result<(), VerificationError> {
        if bits == 0 {
            return Ok(());
        }
        let Some(pow) = SpongePow::new(bits) else {
            return self.check(|| false);
        };
        // A truncated, non-canonical or out-of-range nonce fails the read,
        // which poisons the state and absorbs nothing; a failed check poisons
        // it through `check`. Either way `check_eof` fails afterwards too.
        let _nonce: u64 = VerifierState::prover_message_with(
            self,
            |reader| {
                pow.read_nonce(|| reader.read::<u8>())?
                    .ok_or(VerificationError)
            },
            |&nonce| nonce_word(nonce),
        )?;
        let passes =
            VerifierState::verifier_message_as(self, CHECK_BYTES, |check| pow.accepts(check));
        self.check(|| passes)
    }

    fn check_eof(self) -> Result<(), VerificationError> {
        VerifierState::check_eof(self)
    }
}
