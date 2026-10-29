//! Per-round proof of work.
//!
//! Each round challenge is preceded by a proof-of-work nonce whose difficulty
//! lifts the round to Jolt's Fiat-Shamir target of about `2^128` hash calls.
//! A false claim survives a round only when the challenge is a root of a
//! nonzero polynomial of degree at most `d`, so the round loses a factor `d`
//! (Schwartz-Zippel) and its difficulty is
//! `grind_bits_for_loss(d, T::CHALLENGE_SET, SECURITY_BITS)`, where
//! `T::CHALLENGE_SET` is the [`FieldTranscript`] challenge set for `F`.
//!
//! `d` is always the verifier's degree bound for the claim or statement, never
//! a degree the prover declares, so the difficulty is fixed by the protocol
//! parameters and the proof cannot lower it. The nonces sit in the proof as a
//! plain list next to the round messages: one per round when the difficulty
//! is positive, none when it is zero.
//!
//! Only clear proofs grind. Committed (zero-knowledge) proofs run over BN254,
//! whose challenges need no proof of work, and check that with
//! [`require_no_pow`](jolt_transcript::require_no_pow) instead of carrying
//! nonces.

use jolt_field::Field;
#[cfg(feature = "prover")]
use jolt_transcript::ProverTranscript;
use jolt_transcript::{grind_bits_for_loss, FieldTranscript, SECURITY_BITS};

use crate::error::SumcheckError;

/// Proof-of-work difficulty for a round whose polynomial has degree at most
/// `degree`, with challenges drawn by [`FieldTranscript::challenge`].
///
/// # Errors
/// [`SumcheckError::PowDifficulty`] if no supported difficulty reaches the
/// target for this degree and challenge set.
pub fn round_pow_bits<F: Field, T: FieldTranscript<F>>(
    degree: usize,
) -> Result<u32, SumcheckError<F>> {
    let loss = u64::try_from(degree.max(1)).map_err(|_| SumcheckError::PowDifficulty { degree })?;
    grind_bits_for_loss(loss, T::CHALLENGE_SET, SECURITY_BITS)
        .map_err(|_| SumcheckError::PowDifficulty { degree })
}

/// The verifier's view of a proof's round nonces, checked for count up front.
pub(crate) struct RoundNonces<'a> {
    bits: u32,
    nonces: &'a [u64],
}

impl<'a> RoundNonces<'a> {
    /// Accepts `nonces` for `rounds` rounds of degree at most `degree`: one
    /// nonce per round when the difficulty is positive, none otherwise.
    pub(crate) fn new<F: Field, T: FieldTranscript<F>>(
        degree: usize,
        rounds: usize,
        nonces: &'a [u64],
    ) -> Result<Self, SumcheckError<F>> {
        let bits = round_pow_bits::<F, T>(degree)?;
        let expected = if bits == 0 { 0 } else { rounds };
        if nonces.len() != expected {
            return Err(SumcheckError::WrongNumberOfPowNonces {
                expected,
                got: nonces.len(),
            });
        }
        Ok(Self { bits, nonces })
    }

    /// Checks round `round`'s nonce against the transcript, after the round
    /// message is absorbed and before the round challenge is drawn.
    pub(crate) fn check<F: Field, T: FieldTranscript<F>>(
        &self,
        round: usize,
        transcript: &mut T,
    ) -> Result<(), SumcheckError<F>> {
        transcript
            .check_public_grind(self.bits, self.nonces.get(round).copied())
            .map_err(|_| SumcheckError::ProofOfWorkRejected { round })
    }
}

/// Grinds round `round`'s nonce after the round message is absorbed and before
/// the round challenge is drawn, appending it to `nonces` when the difficulty
/// is positive.
#[cfg(feature = "prover")]
pub fn grind_round<F: Field, T: FieldTranscript<F> + ProverTranscript>(
    degree: usize,
    round: usize,
    nonces: &mut Vec<u64>,
    transcript: &mut T,
) -> Result<(), SumcheckError<F>> {
    let bits = round_pow_bits::<F, T>(degree)?;
    let nonce = transcript
        .public_grind(bits)
        .map_err(|_| SumcheckError::ProofOfWorkRejected { round })?;
    nonces.extend(nonce);
    Ok(())
}
