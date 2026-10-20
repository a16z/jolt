//! Proof of work before BlindFold's Fiat-Shamir draws outside its sumchecks.
//!
//! Each draw is a full `Transcript::challenge` draw, so it is priced against
//! `T::CHALLENGE_SET`. The loss of a draw is the most values that can make a
//! false statement pass it, by Schwartz-Zippel over the draw. The folded
//! sumcheck rounds grind inside `jolt_sumcheck` instead.

use jolt_transcript::{grind_bits_for_loss, Transcript, SECURITY_BITS};

/// A BlindFold challenge draw that follows a prover message and grinds first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PowSite {
    /// The folding challenge `r`. The folded error row is
    /// `E₁ + r·T + r²·E₂`, so a bad fold survives at most 2 values of `r`.
    Fold,
    /// The outer zero-check point `τ`. The outer sum is multilinear in `τ`,
    /// so it vanishes for at most `num_vars` values per coordinate.
    OuterPoint { num_vars: usize },
    /// The inner combination `ra, rb, rc`. A false claim survives a random
    /// linear combination with probability `1/|C|`.
    InnerCombination,
}

impl PowSite {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Fold => "folding challenge",
            Self::OuterPoint { .. } => "outer zero-check point",
            Self::InnerCombination => "inner combination",
        }
    }

    /// Grinding bits for this draw, or `None` when the transcript's challenge
    /// set cannot reach the target within the proof-of-work bound.
    pub(crate) fn bits<T: Transcript>(self) -> Option<u32> {
        let loss = match self {
            Self::Fold => 2,
            Self::OuterPoint { num_vars } => u64::try_from(num_vars).ok()?,
            Self::InnerCombination => 1,
        };
        grind_bits_for_loss(loss, T::CHALLENGE_SET, SECURITY_BITS).ok()
    }
}
