use jolt_crypto::{Bn254, Bn254G1, JoltGroup, PairingGroup};
use jolt_field::{Fr, One, Zero};
use jolt_transcript::Transcript;

use crate::{HyperKZGError, HyperKZGProverSetup, HyperKZGVerifierSetup};

impl HyperKZGProverSetup {
    pub(crate) fn commit_coefficients(
        &self,
        coefficients: &[Fr],
    ) -> Result<Bn254G1, HyperKZGError> {
        let bases = self
            .g1_powers
            .get(..coefficients.len())
            .ok_or(HyperKZGError::PolynomialShape)?;
        Ok(Bn254G1::msm(bases, coefficients))
    }

    pub(crate) fn open_batch(
        &self,
        polynomials: &[Vec<Fr>],
        points: [Fr; 3],
        transcript: &mut impl Transcript<Challenge = Fr>,
    ) -> Result<([Bn254G1; 3], [Vec<Fr>; 3]), HyperKZGError> {
        let evaluations = points.map(|point| {
            polynomials
                .iter()
                .map(|coefficients| evaluate(coefficients, point))
                .collect::<Vec<_>>()
        });
        let powers = absorb_evaluations(&evaluations, transcript);
        let first = polynomials.first().ok_or(HyperKZGError::PolynomialShape)?;
        let mut combined = vec![Fr::zero(); first.len()];
        for (polynomial, weight) in polynomials.iter().zip(powers) {
            for (value, coefficient) in combined.iter_mut().zip(polynomial) {
                *value += weight * coefficient;
            }
        }
        let [a, b, c] = points.map(|point| self.commit_coefficients(&quotient(&combined, point)));
        let witnesses = [a?, b?, c?];
        let _ = absorb_witnesses(&witnesses, transcript);
        Ok((witnesses, evaluations))
    }
}

impl HyperKZGVerifierSetup {
    pub(crate) fn verify_batch(
        &self,
        commitments: &[Bn254G1],
        points: [Fr; 3],
        evaluations: &[Vec<Fr>; 3],
        witnesses: &[Bn254G1; 3],
        transcript: &mut impl Transcript<Challenge = Fr>,
    ) -> Result<(), HyperKZGError> {
        let powers = absorb_evaluations(evaluations, transcript);
        let d = absorb_witnesses(witnesses, transcript);
        let weights = [Fr::one(), d, d * d];
        let scale = weights.iter().copied().sum::<Fr>();
        let mut lhs = Bn254G1::msm(commitments, &powers).scalar_mul(&scale);
        let mut rhs = Bn254G1::identity();
        let mut evaluation = Fr::zero();
        for (((row, point), witness), weight) in
            evaluations.iter().zip(points).zip(witnesses).zip(weights)
        {
            evaluation += weight
                * row
                    .iter()
                    .zip(&powers)
                    .map(|(value, power)| *value * power)
                    .sum::<Fr>();
            lhs += witness.scalar_mul(&(point * weight));
            rhs += witness.scalar_mul(&weight);
        }
        lhs -= self.g1.scalar_mul(&evaluation);
        if !Bn254::multi_pairing(&[lhs, -rhs], &[self.g2, self.beta_g2]).is_identity() {
            return Err(HyperKZGError::Pairing);
        }
        Ok(())
    }
}

fn absorb_evaluations(
    evaluations: &[Vec<Fr>; 3],
    transcript: &mut impl Transcript<Challenge = Fr>,
) -> Vec<Fr> {
    for row in evaluations {
        for value in row {
            transcript.append(value);
        }
    }
    let q = transcript.challenge();
    let [first, _, _] = evaluations;
    std::iter::successors(Some(Fr::one()), |power| Some(*power * q))
        .take(first.len())
        .collect()
}

fn absorb_witnesses(
    witnesses: &[Bn254G1; 3],
    transcript: &mut impl Transcript<Challenge = Fr>,
) -> Fr {
    for witness in witnesses {
        transcript.append(witness);
    }
    transcript.challenge()
}

fn evaluate(coefficients: &[Fr], point: Fr) -> Fr {
    coefficients
        .iter()
        .rev()
        .fold(Fr::zero(), |value, coefficient| value * point + coefficient)
}

/// Synthetic division: `(f(X) - f(point)) / (X - point)`.
fn quotient(coefficients: &[Fr], point: Fr) -> Vec<Fr> {
    let mut result = Vec::with_capacity(coefficients.len().saturating_sub(1));
    let mut carry = Fr::zero();
    for coefficient in coefficients.iter().skip(1).rev() {
        carry = *coefficient + point * carry;
        result.push(carry);
    }
    result.reverse();
    result
}

#[cfg(test)]
mod tests {
    use jolt_field::Ring;

    use super::*;

    #[test]
    fn synthetic_division_has_independent_coefficients() {
        let coefficients = [1, 2, 3, 4].map(Fr::from_u64);
        assert_eq!(
            quotient(&coefficients, Fr::from_u64(2)),
            [24, 11, 4].map(Fr::from_u64)
        );
        assert_eq!(evaluate(&coefficients, Fr::from_u64(2)), Fr::from_u64(49));
    }
}
