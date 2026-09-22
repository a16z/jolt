//! Fixed full-round sumchecks with private q128 challenges constrained over BN254.
//! Coefficient count, domain and degree bound are public shape parameters. Empty
//! native full polynomials require a different profile and are rejected here.
use jolt_field::{Fr, Prime128OffsetA7F7, Ring};
use jolt_r1cs::fp128_bn254::{Fp128Error, Fp128Var};
use jolt_r1cs::{LinearCombination, R1csBuilder};
use jolt_transcript::r1cs::{Blake2bR1csError, Fp128TranscriptError, LegacyBlake2bVar};
use thiserror::Error;

use crate::{SumcheckDomain, SUMCHECK_ROUND_TRANSCRIPT_LABEL};

#[derive(Debug, Error)]
pub enum Fp128SumcheckError {
    #[error("invalid nonempty round coefficient count {count} for degree bound {degree}")]
    InvalidCount { count: usize, degree: usize },
    #[error("expected {expected} round coefficients, got {actual}")]
    CountMismatch { expected: usize, actual: usize },
    #[error("invalid public domain: {0}")]
    Domain(String),
    #[error(transparent)]
    Field(#[from] Fp128Error),
    #[error(transparent)]
    Transcript(#[from] Fp128TranscriptError),
    #[error(transparent)]
    Blake(#[from] Blake2bR1csError),
}

/// Public geometry for one nonempty full round; shorter native encodings use
/// different shapes. Existing fixed-challenge/committed helpers are unchanged.
pub struct Fp128FullRoundShape {
    weights: Vec<u128>,
    count: u32,
    label: &'static [u8],
}

/// Shared handles for the private transcript challenge and resulting claim.
pub struct Fp128RoundReduction {
    pub challenge: Fp128Var,
    pub claim: Fp128Var,
}

impl Fp128FullRoundShape {
    pub fn new<D: SumcheckDomain<Prime128OffsetA7F7>>(
        count: usize,
        degree: usize,
        domain: D,
        label: &'static [u8],
    ) -> Result<Self, Fp128SumcheckError> {
        let actual_degree = count
            .checked_sub(1)
            .filter(|actual| *actual <= degree)
            .ok_or(Fp128SumcheckError::InvalidCount { count, degree })?;
        let count_u32 = u32::try_from(count)
            .ok()
            .filter(|count| count.checked_add(2).is_some())
            .ok_or(Fp128SumcheckError::InvalidCount { count, degree })?;
        if label.len() > 24 {
            return Err(Blake2bR1csError::LabelTooLong.into());
        }
        let weights = domain
            .round_sum_coefficients(actual_degree)
            .map_err(|error| Fp128SumcheckError::Domain(error.to_string()))?;
        if weights.len() != count {
            return Err(Fp128SumcheckError::CountMismatch {
                expected: count,
                actual: weights.len(),
            });
        }
        Ok(Self {
            weights: weights
                .into_iter()
                .map(|value| value.to_canonical_u128())
                .collect(),
            count: count_u32,
            label,
        })
    }

    /// Validate all public shape and handle indices without circuit allocation.
    pub fn validate(
        &self,
        builder: &R1csBuilder<Fr>,
        coefficients: &[Fp128Var],
        input: &Fp128Var,
    ) -> Result<(), Fp128SumcheckError> {
        if coefficients.len() != self.weights.len() {
            return Err(Fp128SumcheckError::CountMismatch {
                expected: self.weights.len(),
                actual: coefficients.len(),
            });
        }
        input.validate_indices(builder)?;
        for coefficient in coefficients {
            coefficient.validate_indices(builder)?;
        }
        Ok(())
    }

    /// Count framing, each coefficient, then the raw challenge transition.
    pub fn transitions(&self) -> u32 {
        self.count + 2
    }

    /// Constrain the domain sum and Horner evaluation over q; no private field
    /// value becomes a BN254 matrix coefficient. Prefix authentication is external.
    pub fn constrain(
        &self,
        builder: &mut R1csBuilder<Fr>,
        transcript: &mut LegacyBlake2bVar,
        coefficients: &[Fp128Var],
        input: &Fp128Var,
    ) -> Result<Fp128RoundReduction, Fp128SumcheckError> {
        self.validate(builder, coefficients, input)?;
        transcript.check_schedule(self.transitions())?;
        let zero = Fp128Var::allocate(builder, Some(0))?;
        builder.assert_equal(zero.variable(), LinearCombination::zero());
        let mut sum = zero.clone();
        for (coefficient, &weight) in coefficients.iter().zip(&self.weights) {
            let constant = Fp128Var::allocate(builder, Some(weight))?;
            builder.assert_equal(
                constant.variable(),
                LinearCombination::constant(Fr::from_u128(weight)),
            );
            let term = coefficient.multiply(builder, &constant)?;
            sum = sum.add(builder, &term)?;
        }
        builder.assert_equal(sum.variable(), input.variable());
        transcript.append_label_with_count(builder, self.label, u64::from(self.count))?;
        for coefficient in coefficients {
            transcript.append_fp128(builder, coefficient)?;
        }
        let challenge = transcript.challenge_fp128(builder)?;
        let claim = self.evaluate(builder, coefficients, &challenge)?;
        Ok(Fp128RoundReduction { challenge, claim })
    }

    fn evaluate(
        &self,
        builder: &mut R1csBuilder<Fr>,
        coefficients: &[Fp128Var],
        challenge: &Fp128Var,
    ) -> Result<Fp128Var, Fp128SumcheckError> {
        let (highest, rest) =
            coefficients
                .split_last()
                .ok_or(Fp128SumcheckError::CountMismatch {
                    expected: self.weights.len(),
                    actual: 0,
                })?;
        let mut claim = highest.clone();
        for coefficient in rest.iter().rev() {
            claim = claim
                .multiply(builder, challenge)?
                .add(builder, coefficient)?;
        }
        Ok(claim)
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test regenerates canonical challenge and all downstream polynomial auxiliaries"
)]
mod tests {
    use super::*;
    use crate::CenteredIntegerDomain;
    use jolt_field::CanonicalEncoding;

    #[test]
    fn private_challenge_with_valid_local_evaluation_rejects_wrong_hash() {
        let mut builder = R1csBuilder::new();
        let coefficients =
            [0, 1].map(|value| Fp128Var::allocate(&mut builder, Some(value)).unwrap());
        let input = Fp128Var::allocate(&mut builder, Some(0)).unwrap();
        let mut transcript = LegacyBlake2bVar::new(&mut builder, b"challenge-binding").unwrap();
        let shape =
            Fp128FullRoundShape::new(2, 1, CenteredIntegerDomain::new(3), b"round").unwrap();
        let reduction = shape
            .constrain(&mut builder, &mut transcript, &coefficients, &input)
            .unwrap();
        let witness = builder.witness().unwrap();
        let matrices = builder.into_matrices();
        matrices.check_witness(&witness).unwrap();
        let challenge_start = reduction.challenge.variable().index();
        let actual = witness[challenge_start].to_u128_checked().unwrap();
        let forged_value = (Prime128OffsetA7F7::from_u128(actual)
            + Prime128OffsetA7F7::from_u64(1))
        .to_canonical_u128();
        let mut forged = R1csBuilder::new();
        for &value in &witness[1..challenge_start] {
            let _ = forged.alloc(value);
        }
        let forged_challenge = Fp128Var::allocate(&mut forged, Some(forged_value)).unwrap();
        // Reuse the original quotient bit, then regenerate every polynomial auxiliary
        // at the forged challenge with the production evaluator.
        let _ = forged.alloc(witness[forged.num_vars()]);
        let local_claim = shape
            .evaluate(&mut forged, &coefficients, &forged_challenge)
            .unwrap();
        let forged_witness = forged.witness().unwrap();
        assert_eq!(forged_witness.len(), witness.len());
        assert_eq!(local_claim.variable(), reduction.claim.variable());
        assert_eq!(
            forged_witness[local_claim.variable().index()],
            Fr::from_u128(forged_value)
        );
        forged
            .into_matrices()
            .check_witness(&forged_witness)
            .unwrap();
        assert!(matrices.check_witness(&forged_witness).is_err());
    }
}

/// A fixed nonempty compressed Boolean round, storing c0,c2,... in native order.
pub struct Fp128CompressedRoundShape {
    count: u32,
    length: usize,
}
impl Fp128CompressedRoundShape {
    pub fn new(count: usize, degree: usize) -> Result<Self, Fp128SumcheckError> {
        if count == 0 || count > degree {
            return Err(Fp128SumcheckError::InvalidCount { count, degree });
        }
        let length = count;
        let count = u32::try_from(count)
            .ok()
            .filter(|x| x.checked_add(2).is_some())
            .ok_or(Fp128SumcheckError::InvalidCount { count, degree })?;
        Ok(Self { count, length })
    }
    pub fn transitions(&self) -> u32 {
        self.count + 2
    }
    pub fn constrain(
        &self,
        builder: &mut R1csBuilder<Fr>,
        transcript: &mut LegacyBlake2bVar,
        coefficients: &[Fp128Var],
        input: &Fp128Var,
    ) -> Result<Fp128RoundReduction, Fp128SumcheckError> {
        if coefficients.len() != self.length {
            return Err(Fp128SumcheckError::CountMismatch {
                expected: self.length,
                actual: coefficients.len(),
            });
        }
        input.validate_indices(builder)?;
        for value in coefficients {
            value.validate_indices(builder)?;
        }
        transcript.check_schedule(self.transitions())?;
        let (constant, higher) =
            coefficients
                .split_first()
                .ok_or(Fp128SumcheckError::CountMismatch {
                    expected: 1,
                    actual: 0,
                })?;
        let mut linear = input
            .subtract(builder, constant)?
            .subtract(builder, constant)?;
        for value in higher {
            linear = linear.subtract(builder, value)?;
        }
        transcript.append_label_with_count(
            builder,
            SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            u64::from(self.count),
        )?;
        for value in coefficients {
            transcript.append_fp128(builder, value)?;
        }
        let challenge = transcript.challenge_fp128(builder)?;
        let mut full = vec![constant.clone(), linear];
        full.extend_from_slice(higher);
        let claim = jolt_poly::r1cs::evaluate_fp128_bn254(builder, &full, &challenge)?;
        Ok(Fp128RoundReduction { challenge, claim })
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "native compressed round and private-cell regressions"
)]
mod compressed_tests {
    use super::*;
    use crate::{BooleanHypercube, CompressedSumcheckProof, SumcheckClaim};
    use jolt_poly::CompressedPoly;
    use jolt_transcript::{LegacyBlake2bTranscript, Transcript};
    #[test]
    fn private_compressed_round_matches_native_and_rejects_tampering() {
        let values = [3, 5, 7];
        let input_value = 19;
        let mut native = LegacyBlake2bTranscript::<Prime128OffsetA7F7>::new(b"compressed");
        let proof = CompressedSumcheckProof {
            round_polynomials: vec![CompressedPoly::new(
                values.map(Prime128OffsetA7F7::from_u128).to_vec(),
            )],
        };
        let expected = proof
            .verify(
                &SumcheckClaim::new(1, 3, Prime128OffsetA7F7::from_u128(input_value)),
                BooleanHypercube,
                SUMCHECK_ROUND_TRANSCRIPT_LABEL,
                &mut native,
            )
            .unwrap();
        let mut builder = R1csBuilder::new();
        let coefficients = values.map(|v| Fp128Var::allocate(&mut builder, Some(v)).unwrap());
        let input = Fp128Var::allocate(&mut builder, Some(input_value)).unwrap();
        let mut transcript = LegacyBlake2bVar::new(&mut builder, b"compressed").unwrap();
        let result = Fp128CompressedRoundShape::new(3, 3)
            .unwrap()
            .constrain(&mut builder, &mut transcript, &coefficients, &input)
            .unwrap();
        let mut witness = builder.witness().unwrap();
        assert_eq!(
            witness[result.claim.variable().index()],
            Fr::from_u128(expected.value.to_canonical_u128())
        );
        assert_eq!(
            witness[result.challenge.variable().index()],
            Fr::from_u128(expected.point[0].to_canonical_u128())
        );
        let matrices = builder.into_matrices();
        matrices.check_witness(&witness).unwrap();
        witness[result.challenge.variable().index()] += Fr::from_u64(1);
        assert!(matrices.check_witness(&witness).is_err());
        assert!(Fp128CompressedRoundShape::new(0, 3).is_err());
        assert!(Fp128CompressedRoundShape::new(4, 3).is_err());
    }
}
