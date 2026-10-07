use jolt_crypto::{Bn254, Bn254G1, JoltGroup, PairingGroup};
use jolt_field::{Fr, One, Zero};
use jolt_transcript::{Channel, ProverTranscript, Sponge, VerifierTranscript};

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

    /// Sends the evaluations at `points` and the batched KZG witnesses.
    pub(crate) fn open_batch<H: Sponge>(
        &self,
        polynomials: &[Vec<Fr>],
        points: [Fr; 3],
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), HyperKZGError> {
        let mut evaluations = points.map(|point| {
            polynomials
                .iter()
                .map(|coefficients| evaluate(coefficients, point))
                .collect::<Vec<_>>()
        });
        let powers = exchange_evaluations(&mut evaluations, transcript)?;
        let first = polynomials.first().ok_or(HyperKZGError::PolynomialShape)?;
        let mut combined = vec![Fr::zero(); first.len()];
        for (polynomial, weight) in polynomials.iter().zip(powers) {
            for (value, coefficient) in combined.iter_mut().zip(polynomial) {
                *value += weight * coefficient;
            }
        }
        let [a, b, c] = points.map(|point| self.commit_coefficients(&quotient(&combined, point)));
        let mut witnesses = [a?, b?, c?];
        let _ = exchange_witnesses(&mut witnesses, transcript)?;
        Ok(())
    }
}

impl HyperKZGVerifierSetup {
    /// Receives the KZG witnesses for `evaluations` (already received, with
    /// their batching `powers`) and checks the batched pairing equation.
    pub(crate) fn verify_batch<H: Sponge>(
        &self,
        commitments: &[Bn254G1],
        points: [Fr; 3],
        evaluations: &[Vec<Fr>; 3],
        powers: &[Fr],
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), HyperKZGError> {
        let mut witnesses = [Bn254G1::identity(); 3];
        let d = exchange_witnesses(&mut witnesses, transcript)?;
        let weights = [Fr::one(), d, d * d];
        let scale = weights.iter().copied().sum::<Fr>();
        let mut lhs = Bn254G1::msm(commitments, powers).scalar_mul(&scale);
        let mut rhs = Bn254G1::identity();
        let mut evaluation = Fr::zero();
        for (((row, point), witness), weight) in
            evaluations.iter().zip(points).zip(&witnesses).zip(weights)
        {
            evaluation += weight
                * row
                    .iter()
                    .zip(powers)
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

/// Exchanges the three evaluation rows, then draws the batching powers.
pub(crate) fn exchange_evaluations<C: Channel>(
    evaluations: &mut [Vec<Fr>; 3],
    channel: &mut C,
) -> Result<Vec<Fr>, HyperKZGError> {
    for row in evaluations.iter_mut() {
        channel.exchange_all(row)?;
    }
    let q: Fr = channel.challenge();
    let [first, _, _] = evaluations;
    Ok(
        std::iter::successors(Some(Fr::one()), |power| Some(*power * q))
            .take(first.len())
            .collect(),
    )
}

/// Exchanges the three KZG witnesses, then draws their combination weight.
fn exchange_witnesses<C: Channel>(
    witnesses: &mut [Bn254G1; 3],
    channel: &mut C,
) -> Result<Fr, HyperKZGError> {
    channel.exchange_all(witnesses)?;
    Ok(channel.challenge())
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
