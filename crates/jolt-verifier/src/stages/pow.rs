//! Proof of work before the challenges that follow a prover message outside
//! the sumcheck rounds.
//!
//! After each such message the verifier draws a burst of challenges: a stage's
//! relation challenges and batching coefficients, the stage-1 `tau` vector,
//! the stage-2 `tau_high`, and the stage-8 selector points. A false statement
//! survives the burst only when some draw hits a bad set; summed over the
//! burst, the bad sets hold at most `loss · |C|^-1` of the challenge space.
//! One nonce, ground right before the burst's first draw, lifts the burst to
//! `SECURITY_BITS`.
//!
//! Every site is priced at the one flat loss [`SITE_LOSS`] rather than at a
//! per-relation count: overpricing costs the prover a few probes, while a
//! count that drifts below the expressions it describes would be a silent
//! soundness bug. Sumcheck rounds keep their own degree pricing.
//!
//! The nonces ride in [`StagePowNonces`], one optional slot per [`PowSite`].
//! A slot holds a nonce exactly when the site is visited and its difficulty
//! is positive; a site the verifier skips must hold none
//! ([`StagePowNonces::require_absent`]).

use jolt_transcript::{grind_bits_for_loss, FieldTranscript, SECURITY_BITS};
use serde::{Deserialize, Serialize};

use crate::VerifierError;

/// The loss every stage-level burst is priced at.
///
/// Bounds on the bursts' actual losses (Akita with field-inline, the worst
/// case; `B` and `R` are the bytecode and RAM chunk counts, `ℓT` and `ℓR` the
/// trace and RAM address bits):
///
/// | Site | Loss | 64-bit extreme (`B = R = 16`) |
/// |---|---|---|
/// | Stage 6a | `163 + 2(B + R)` | 227 |
/// | Stage 7 | `82 + 3R + 2B` | 162 |
/// | Stage 2 | `ℓR + 8` | 69 |
/// | Stage 1 `tau` | `ℓT + 15` | 55 at `ℓT = 40` |
/// | Stages 1, 2 `tau_high`, 3, 4, 5, 6b | at most 15 | |
/// | Stage 8 | `Σ log₂(slot capacity)` over at most 4 objects | a few |
///
/// Stage 6a is the bytecode read-RAF's 62, up to 4 fresh booleanity
/// coordinates, `2(N − 1)` for the `N = 32 + B + R + 16 + 1` boolean-checked
/// openings, and 1 for the batching coefficients. Stage 7 is 3 `γ` powers per
/// RAM chunk, 2 per instruction and bytecode chunk, 1 per increment digit,
/// plus 2.
///
/// 1023 is 4.5 times the extreme and still covers 2-bit chunks (about 383).
/// It is `2^10 − 1` rather than `2^10`: the field order is below `2^128`, so a
/// power-of-two loss would cost one more bit.
pub const SITE_LOSS: u64 = 1023;

/// The challenge bursts that a proof-of-work nonce protects, in transcript
/// order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowSite {
    /// The stage-1 `tau` vector, after the commitments.
    Stage1Tau,
    /// The stage-1 batching coefficient, after the uni-skip output claim.
    Stage1,
    /// The stage-2 `tau_high`, after the stage-1 openings.
    Stage2TauHigh,
    Stage2,
    Stage3,
    Stage4,
    Stage5,
    Stage6a,
    Stage6b,
    Stage7,
    /// The selector points of the stage-8 opening objects.
    Stage8,
}

impl PowSite {
    /// Every site, in transcript order.
    pub const ALL: [Self; 11] = [
        Self::Stage1Tau,
        Self::Stage1,
        Self::Stage2TauHigh,
        Self::Stage2,
        Self::Stage3,
        Self::Stage4,
        Self::Stage5,
        Self::Stage6a,
        Self::Stage6b,
        Self::Stage7,
        Self::Stage8,
    ];

    /// Position in transcript order, which is also the slot.
    const fn index(self) -> u32 {
        match self {
            Self::Stage1Tau => 0,
            Self::Stage1 => 1,
            Self::Stage2TauHigh => 2,
            Self::Stage2 => 3,
            Self::Stage3 => 4,
            Self::Stage4 => 5,
            Self::Stage5 => 6,
            Self::Stage6a => 7,
            Self::Stage6b => 8,
            Self::Stage7 => 9,
            Self::Stage8 => 10,
        }
    }
}

/// The proof-of-work nonces of every [`PowSite`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StagePowNonces([Option<u64>; PowSite::ALL.len()]);

impl StagePowNonces {
    /// No nonces.
    pub const fn new() -> Self {
        Self([None; PowSite::ALL.len()])
    }

    pub fn get(&self, site: PowSite) -> Option<u64> {
        self.0.get(slot(site)).copied().flatten()
    }

    pub fn set(&mut self, site: PowSite, nonce: Option<u64>) {
        if let Some(entry) = self.0.get_mut(slot(site)) {
            *entry = nonce;
        }
    }

    /// Rejects a nonce at a site the verifier does not visit on this path, so
    /// the slot cannot carry bytes that nothing checks.
    ///
    /// # Errors
    /// [`VerifierError::ProofOfWorkRejected`] if the slot holds a nonce.
    pub fn require_absent(&self, site: PowSite) -> Result<(), VerifierError> {
        match self.get(site) {
            None => Ok(()),
            Some(_) => Err(VerifierError::ProofOfWorkRejected { site }),
        }
    }
}

impl FromIterator<(PowSite, Option<u64>)> for StagePowNonces {
    fn from_iter<I: IntoIterator<Item = (PowSite, Option<u64>)>>(sites: I) -> Self {
        let mut nonces = Self::new();
        for (site, nonce) in sites {
            nonces.set(site, nonce);
        }
        nonces
    }
}

fn slot(site: PowSite) -> usize {
    usize::try_from(site.index()).unwrap_or(usize::MAX)
}

/// Proof-of-work difficulty of every stage-level burst: the difficulty for a
/// loss of [`SITE_LOSS`] over `T::CHALLENGE_SET`.
///
/// # Errors
/// [`VerifierError::PowDifficulty`] if no supported difficulty reaches
/// `SECURITY_BITS` for this challenge set.
pub fn site_pow_bits<F, T: FieldTranscript<F>>() -> Result<u32, VerifierError> {
    grind_bits_for_loss(SITE_LOSS, T::CHALLENGE_SET, SECURITY_BITS)
        .map_err(|_| VerifierError::PowDifficulty)
}

/// Checks the nonce of the burst at `site` right before its first draw.
///
/// # Errors
/// [`VerifierError::PowDifficulty`] as for [`site_pow_bits`], and
/// [`VerifierError::ProofOfWorkRejected`] if the nonce is missing, present
/// when the difficulty is zero, or fails the check.
pub fn check_pow<F, T: FieldTranscript<F>>(
    site: PowSite,
    nonces: &StagePowNonces,
    transcript: &mut T,
) -> Result<(), VerifierError> {
    transcript
        .check_public_grind(site_pow_bits::<F, T>()?, nonces.get(site))
        .map_err(|_| VerifierError::ProofOfWorkRejected { site })
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use jolt_transcript::{grind_bits_for_loss, ChallengeSetSize, SECURITY_BITS};

    use super::SITE_LOSS;

    /// The flat loss costs 10 bits over a 128-bit prime field; a loss of
    /// `2^10` would cost 11.
    #[test]
    fn site_loss_costs_ten_bits_over_a_128_bit_field() {
        // 2^128 − 0xffff_a7f7, the modulus of `Prime128OffsetA7F7`.
        let set = ChallengeSetSize::field_order(u128::MAX - 0xffff_a7f6, 1).unwrap();
        assert_eq!(grind_bits_for_loss(SITE_LOSS, set, SECURITY_BITS), Ok(10));
        assert_eq!(
            grind_bits_for_loss(SITE_LOSS + 1, set, SECURITY_BITS),
            Ok(11)
        );
    }
}
