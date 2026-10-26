//! Proof of work before the challenges that follow a prover message outside
//! the sumcheck rounds.
//!
//! After each such message the verifier draws a burst of challenges: a stage's
//! relation challenges and batching coefficients, the stage-1 `tau` vector,
//! the stage-2 `tau_high`, and the stage-8 selector points. A false statement
//! survives the burst only when some draw hits a bad set; summed over the
//! burst, the bad sets hold at most `loss · |C|^-1` of the challenge space,
//! where `loss` is fixed by the protocol parameters. One nonce, ground right
//! before the burst's first draw, lifts the burst to `SECURITY_BITS`:
//! the difficulty is `grind_bits_for_loss(loss, T::CHALLENGE_SET,
//! SECURITY_BITS)`, the same rule the sumcheck rounds use.
//!
//! The nonces ride in [`StagePowNonces`], one optional slot per [`PowSite`].
//! A slot holds a nonce exactly when its difficulty is positive.

use jolt_claims::{Expr, Source};
use jolt_transcript::{grind_bits_for_loss, FieldTranscript, SECURITY_BITS};
use serde::{Deserialize, Serialize};

use crate::num::u64_from_usize;
use crate::VerifierError;

/// Loss of a batch's coefficients. They combine the members' claims
/// linearly, so a batch with a false member claim passes for at most one
/// value of any one coefficient, whatever the member count.
pub const BATCH_COEFFICIENT_LOSS: u64 = 1;

/// The highest number of challenge factors in any term of `expr`: the total
/// degree of `expr` in the challenges it reads.
pub fn challenge_degree<F, O, P, C>(expr: &Expr<F, O, P, C>) -> u64 {
    expr.terms
        .iter()
        .map(|term| {
            term.factors
                .iter()
                .filter(|factor| matches!(factor, Source::Challenge(_)))
                .count()
        })
        .max()
        .map_or(0, u64_from_usize)
}

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

    /// Position in transcript order, which is also the slot and mask bit.
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

    fn mask_bit(self) -> u16 {
        1u16.checked_shl(self.index()).unwrap_or(0)
    }
}

/// The proof-of-work nonces of every [`PowSite`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "PowNonceWire", try_from = "PowNonceWire")]
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
}

fn slot(site: PowSite) -> usize {
    usize::try_from(site.index()).unwrap_or(usize::MAX)
}

/// Wire form of [`StagePowNonces`]: a bitmask of the occupied slots, then
/// their nonces in site order. With no nonces it costs two bytes.
#[derive(Serialize, Deserialize)]
struct PowNonceWire {
    present: u16,
    nonces: Vec<u64>,
}

impl From<StagePowNonces> for PowNonceWire {
    fn from(slots: StagePowNonces) -> Self {
        let mut present = 0u16;
        let mut nonces = Vec::new();
        for site in PowSite::ALL {
            if let Some(nonce) = slots.get(site) {
                present |= site.mask_bit();
                nonces.push(nonce);
            }
        }
        Self { present, nonces }
    }
}

impl TryFrom<PowNonceWire> for StagePowNonces {
    type Error = &'static str;

    fn try_from(wire: PowNonceWire) -> Result<Self, Self::Error> {
        let known = PowSite::ALL
            .iter()
            .fold(0u16, |mask, site| mask | site.mask_bit());
        if wire.present & !known != 0 {
            return Err("proof-of-work mask names an unknown site");
        }
        if usize::try_from(wire.present.count_ones()).ok() != Some(wire.nonces.len()) {
            return Err("proof-of-work mask and nonce count disagree");
        }
        let mut slots = Self::new();
        let mut nonces = wire.nonces.into_iter();
        for site in PowSite::ALL {
            if wire.present & site.mask_bit() != 0 {
                slots.set(site, nonces.next());
            }
        }
        Ok(slots)
    }
}

/// Proof-of-work difficulty for a burst at `site` whose draws lose `loss`
/// in total; a burst that loses nothing needs none.
///
/// # Errors
/// [`VerifierError::PowDifficulty`] if no supported difficulty reaches
/// `SECURITY_BITS` for this loss and challenge set.
pub fn pow_bits<F, T: FieldTranscript<F>>(site: PowSite, loss: u64) -> Result<u32, VerifierError> {
    if loss == 0 {
        return Ok(0);
    }
    grind_bits_for_loss(loss, T::CHALLENGE_SET, SECURITY_BITS)
        .map_err(|_| VerifierError::PowDifficulty { site, loss })
}

/// Checks the nonce of the burst at `site` right before its first draw.
///
/// # Errors
/// [`VerifierError::PowDifficulty`] as for [`pow_bits`], and
/// [`VerifierError::ProofOfWorkRejected`] if the nonce is missing, present
/// when the difficulty is zero, or fails the check.
pub fn check_pow<F, T: FieldTranscript<F>>(
    site: PowSite,
    loss: u64,
    nonces: &StagePowNonces,
    transcript: &mut T,
) -> Result<(), VerifierError> {
    transcript
        .check_public_grind(pow_bits::<F, T>(site, loss)?, nonces.get(site))
        .map_err(|_| VerifierError::ProofOfWorkRejected { site })
}

/// Rejects a challenge set narrow enough that some burst would need proof of
/// work. Builds whose proofs carry no [`StagePowNonces`] (Dory, ZK) call this
/// before touching the transcript: they draw challenges, such as the stage-8
/// batching coefficients and the BlindFold challenges, that grind nowhere.
///
/// # Errors
/// [`VerifierError::ChallengeSetNeedsProofOfWork`] if a burst of the largest
/// loss needs a positive difficulty.
pub fn require_no_pow<F, T: FieldTranscript<F>>() -> Result<(), VerifierError> {
    match grind_bits_for_loss(u64::MAX, T::CHALLENGE_SET, SECURITY_BITS) {
        Ok(0) => Ok(()),
        _ => Err(VerifierError::ChallengeSetNeedsProofOfWork),
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use super::{PowNonceWire, PowSite, StagePowNonces};

    #[test]
    fn wire_form_round_trips_and_rejects_inconsistent_masks() {
        let mut slots = StagePowNonces::default();
        slots.set(PowSite::Stage1Tau, Some(7));
        slots.set(PowSite::Stage8, Some(0));
        let bytes = bincode::serde::encode_to_vec(slots, bincode::config::standard()).unwrap();
        let (back, _): (StagePowNonces, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(back, slots);

        let empty =
            bincode::serde::encode_to_vec(StagePowNonces::default(), bincode::config::standard())
                .unwrap();
        assert_eq!(empty.len(), 2);

        for (present, nonces) in [(0b1, vec![]), (0b1, vec![1, 2]), (0x800, vec![1])] {
            assert!(StagePowNonces::try_from(PowNonceWire { present, nonces }).is_err());
        }
    }

    /// Nonces were `u32` before the proof-of-work cap rose to 57 bits.
    /// bincode's varint encodes an integer by value, not width, so a proof
    /// whose nonces are below `2^32` keeps its bytes.
    #[test]
    fn widened_nonces_keep_their_wire_bytes_below_u32() {
        #[derive(serde::Serialize)]
        struct U32Wire {
            present: u16,
            nonces: Vec<u32>,
        }

        let config = bincode::config::standard();
        let mut slots = StagePowNonces::default();
        let nonces = [0, 251, 65_536, u32::MAX];
        for (site, nonce) in PowSite::ALL.into_iter().zip(nonces) {
            slots.set(site, Some(u64::from(nonce)));
        }
        let present = PowNonceWire::from(slots).present;
        let old = U32Wire {
            present,
            nonces: nonces.to_vec(),
        };
        assert_eq!(
            bincode::serde::encode_to_vec(slots, config).unwrap(),
            bincode::serde::encode_to_vec(old, config).unwrap()
        );

        slots.set(PowSite::Stage8, Some(u64::MAX));
        let bytes = bincode::serde::encode_to_vec(slots, config).unwrap();
        let (back, _): (StagePowNonces, _) =
            bincode::serde::decode_from_slice(&bytes, config).unwrap();
        assert_eq!(back, slots);
    }
}
