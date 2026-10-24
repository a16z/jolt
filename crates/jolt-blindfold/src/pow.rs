//! Proof of work before BlindFold's Fiat-Shamir draws outside its sumchecks.
//!
//! Each draw is a full `FieldTranscript::challenge` draw, so it is priced
//! against `T::CHALLENGE_SET`. The loss of a draw is the most values that can make a
//! false statement pass it, by Schwartz-Zippel over the draw. The folded
//! sumcheck rounds grind inside `jolt_sumcheck` instead.

use jolt_field::JoltField;
use jolt_transcript::{grind_bits_for_loss, FieldTranscript, SECURITY_BITS};

use crate::error::VerificationError;

/// A BlindFold challenge draw that follows a prover message and grinds first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PowSite {
    /// The folding challenge `r`. The folded error row is
    /// `E₁ + r·T + r²·E₂`, so a bad fold survives at most 2 values of `r`.
    Fold,
    /// The outer zero-check point `τ`. The outer sum is multilinear in `τ`,
    /// a polynomial of total degree `num_vars`, so a false claim survives
    /// with probability at most `num_vars/|C|` (Schwartz-Zippel).
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
    pub(crate) fn bits<F, T: FieldTranscript<F>>(self) -> Option<u32> {
        let loss = match self {
            Self::Fold => 2,
            Self::OuterPoint { num_vars } => u64::try_from(num_vars).ok()?,
            Self::InnerCombination => 1,
        };
        grind_bits_for_loss(loss, T::CHALLENGE_SET, SECURITY_BITS).ok()
    }
}

/// The proof's nonces, read in site order by the draws whose difficulty is
/// positive. A draw of difficulty 0 takes no nonce.
pub(crate) struct PowNonces<'a> {
    nonces: &'a [u32],
    next: usize,
}

impl<'a> PowNonces<'a> {
    pub(crate) fn new(nonces: &'a [u32]) -> Self {
        Self { nonces, next: 0 }
    }

    /// Checks the proof of work for `site` right before its draw, reading
    /// the next nonce when the difficulty is positive.
    ///
    /// # Errors
    /// [`VerificationError::ProofOfWorkRejected`] if the difficulty is out of
    /// reach, the nonce is missing, or it fails the check.
    pub(crate) fn check<F: JoltField, T: FieldTranscript<F>>(
        &mut self,
        site: PowSite,
        transcript: &mut T,
    ) -> Result<(), VerificationError<F>> {
        let rejected = || VerificationError::ProofOfWorkRejected { site: site.name() };
        let bits = site.bits::<F, T>().ok_or_else(rejected)?;
        let nonce = if bits == 0 {
            None
        } else {
            let nonce = self.nonces.get(self.next).copied().ok_or_else(rejected)?;
            self.next += 1;
            Some(nonce)
        };
        transcript
            .check_public_grind(bits, nonce)
            .map_err(|_| rejected())
    }

    /// Rejects nonces left over after the last site.
    ///
    /// # Errors
    /// [`VerificationError::WrongNumberOfPowNonces`] if the proof carries
    /// more nonces than its sites read.
    pub(crate) fn finish<F: JoltField>(self) -> Result<(), VerificationError<F>> {
        if self.next == self.nonces.len() {
            Ok(())
        } else {
            Err(VerificationError::WrongNumberOfPowNonces {
                expected: self.next,
                got: self.nonces.len(),
            })
        }
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests should fail loudly")]
mod tests {
    use super::*;
    use jolt_field::{Fr, Prime128OffsetA7F7};
    use jolt_transcript::{prover_transcript, Blake2b512, ProverState, ProverTranscript};

    const SITES: [PowSite; 3] = [
        PowSite::Fold,
        PowSite::OuterPoint { num_vars: 8 },
        PowSite::InnerCombination,
    ];

    #[test]
    fn bn254_sites_read_no_nonces() {
        let mut transcript =
            prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"blindfold-pow"), b"");
        let mut nonces = PowNonces::new(&[]);
        for site in SITES {
            nonces
                .check::<Fr, _>(site, &mut transcript)
                .expect("site check passes");
        }
        assert!(nonces.finish::<Fr>().is_ok());

        let mut nonces = PowNonces::new(&[0]);
        for site in SITES {
            nonces
                .check::<Fr, _>(site, &mut transcript)
                .expect("site check passes");
        }
        assert!(matches!(
            nonces.finish::<Fr>(),
            Err(VerificationError::WrongNumberOfPowNonces {
                expected: 0,
                got: 1
            })
        ));
    }

    #[test]
    fn akita_field_sites_read_one_nonce_each_in_order() {
        type P = Prime128OffsetA7F7;
        type T = ProverState<Blake2b512>;
        assert_eq!(
            SITES.map(PowSite::bits::<P, T>),
            [Some(2), Some(4), Some(1)]
        );

        let mut prover =
            prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"blindfold-pow"), b"");
        let ground = SITES.map(|site| {
            let nonce = prover
                .public_grind(site.bits::<P, T>().expect("difficulty in reach"))
                .expect("grinding succeeds")
                .expect("positive difficulty takes a nonce");
            let _challenge: P = prover.challenge().expect("P supports one challenge");
            nonce
        });

        let mut verifier =
            prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"blindfold-pow"), b"");
        let mut nonces = PowNonces::new(&ground);
        for site in SITES {
            nonces
                .check::<P, _>(site, &mut verifier)
                .expect("site check passes");
            let _challenge: P = verifier.challenge().expect("P supports one challenge");
        }
        assert!(nonces.finish::<P>().is_ok());
        assert_eq!(
            verifier.verifier_message::<[u8; 32]>(),
            prover.verifier_message::<[u8; 32]>()
        );

        let mut verifier =
            prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"blindfold-pow"), b"");
        let mut nonces = PowNonces::new(&ground[..1]);
        nonces
            .check::<P, _>(SITES[0], &mut verifier)
            .expect("site check passes");
        let _challenge: P = verifier.challenge().expect("P supports one challenge");
        assert!(matches!(
            nonces.check::<P, _>(SITES[1], &mut verifier),
            Err(VerificationError::ProofOfWorkRejected {
                site: "outer zero-check point"
            })
        ));
    }
}
