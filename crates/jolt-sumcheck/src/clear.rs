//! Helpers for cleartext sumcheck verifier calls.

use jolt_field::Field;
use jolt_transcript::Transcript;

use crate::{ClearRoundVerifier, SumcheckClaim, SumcheckError, SumcheckProof, SumcheckVerifier};

/// Wire encoding used for cleartext round polynomial transcript absorption.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClearRoundEncoding {
    /// Absorb every round-polynomial coefficient.
    Full,
    /// Absorb every coefficient except the recoverable linear term.
    Compressed,
}

/// Static verifier plan for one cleartext sumcheck instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClearSumcheckPlan {
    /// Number of Boolean variables in the sumcheck claim.
    pub num_vars: usize,
    /// Maximum allowed degree for each round polynomial.
    pub degree: usize,
    /// Domain-separation label absorbed before each round polynomial.
    pub round_label: &'static [u8],
    /// Transcript wire encoding for each round polynomial.
    pub round_encoding: ClearRoundEncoding,
}

/// Successful cleartext sumcheck verification output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClearSumcheckOutput<F: Field> {
    /// Final reduced evaluation claim.
    pub final_eval: F,
    /// Fiat-Shamir challenge point produced by verification.
    pub point: Vec<F>,
}

impl ClearSumcheckPlan {
    /// Verifies a cleartext sumcheck proof and returns its reduced evaluation claim.
    pub fn verify<F, T>(
        &self,
        claimed_sum: F,
        proof: &SumcheckProof<F>,
        transcript: &mut T,
    ) -> Result<ClearSumcheckOutput<F>, SumcheckError>
    where
        F: Field,
        T: Transcript<Challenge = F>,
    {
        let claim = SumcheckClaim {
            num_vars: self.num_vars,
            degree: self.degree,
            claimed_sum,
        };
        let verifier = match self.round_encoding {
            ClearRoundEncoding::Full => ClearRoundVerifier::with_label(self.round_label),
            ClearRoundEncoding::Compressed => {
                ClearRoundVerifier::with_label_compressed(self.round_label)
            }
        };
        let (final_eval, point) =
            SumcheckVerifier::verify(&claim, &proof.round_polynomials, transcript, &verifier)?;
        Ok(ClearSumcheckOutput { final_eval, point })
    }
}

#[cfg(test)]
mod tests {
    use jolt_field::{Field, Fr};
    use jolt_poly::UnivariatePoly;
    use jolt_transcript::Blake2bTranscript;

    use super::*;

    #[test]
    fn verifies_labeled_full_rounds() {
        let proof = SumcheckProof {
            round_polynomials: vec![UnivariatePoly::new(vec![Fr::from_u64(1), Fr::from_u64(1)])],
        };
        let plan = ClearSumcheckPlan {
            num_vars: 1,
            degree: 1,
            round_label: b"round",
            round_encoding: ClearRoundEncoding::Full,
        };
        let mut transcript = Blake2bTranscript::<Fr>::new(b"test");

        let output = plan
            .verify(Fr::from_u64(3), &proof, &mut transcript)
            .expect("valid proof");

        assert_eq!(output.point.len(), 1);
        assert_eq!(
            output.final_eval,
            proof.round_polynomials[0].evaluate(output.point[0])
        );
    }

    #[test]
    fn verifies_labeled_compressed_rounds() {
        let proof = SumcheckProof {
            round_polynomials: vec![UnivariatePoly::new(vec![Fr::from_u64(1), Fr::from_u64(1)])],
        };
        let plan = ClearSumcheckPlan {
            num_vars: 1,
            degree: 1,
            round_label: b"round",
            round_encoding: ClearRoundEncoding::Compressed,
        };
        let mut transcript = Blake2bTranscript::<Fr>::new(b"test");

        let output = plan
            .verify(Fr::from_u64(3), &proof, &mut transcript)
            .expect("valid compressed proof");
        assert_eq!(output.point.len(), 1);
    }
}
