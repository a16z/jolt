//! BDFG20 section 4 variable-point KZG batching with a batched degree bound.

use jolt_crypto::{JoltGroup, PairingGroup};
use jolt_field::JoltField;
use jolt_transcript::{AppendToTranscript, Transcript};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};

use crate::error::HyperKZGError;
use crate::kzg::{challenge_powers, eval_univariate, kzg_commit};
use crate::{HyperKZGProverSetup, HyperKZGVerifierSetup};

const SUPPORTED_DEGREE: usize = 5;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "P::G1: Serialize",
    deserialize = "P::G1: for<'a> Deserialize<'a>"
))]
pub struct VariableBatchKzgProof<P: PairingGroup> {
    pub shifted_commitment: P::G1,
    pub quotient_commitment: P::G1,
    pub evaluation_witness: P::G1,
}

pub fn open_variable_batch<P, T>(
    polynomials: &[Vec<P::ScalarField>],
    points: &[[P::ScalarField; 3]],
    evaluations: &[[P::ScalarField; 3]],
    degree: usize,
    setup: &HyperKZGProverSetup<P>,
    transcript: &mut T,
) -> Result<VariableBatchKzgProof<P>, HyperKZGError>
where
    P: PairingGroup,
    T: Transcript<Challenge = P::ScalarField>,
    P::ScalarField: AppendToTranscript,
    P::G1: AppendToTranscript,
{
    validate_batch(polynomials, points, evaluations, degree)?;
    if polynomials
        .iter()
        .any(|polynomial| polynomial.len() > degree + 1)
    {
        return Err(HyperKZGError::InvalidBatchShape);
    }
    let rho = transcript.challenge();
    let rho_powers = challenge_powers(rho, polynomials.len());
    let mut combined = vec![P::ScalarField::zero(); degree + 1];
    for (polynomial, &coefficient) in polynomials.iter().zip(&rho_powers) {
        add_scaled(&mut combined, polynomial, coefficient);
    }
    combined.resize(degree + 1, P::ScalarField::zero());
    let shift =
        setup
            .g1_powers
            .len()
            .checked_sub(degree + 1)
            .ok_or(HyperKZGError::SrsTooSmall {
                have: setup.g1_powers.len(),
                need: degree + 1,
            })?;
    let bases =
        setup
            .g1_powers
            .get(shift..shift + degree + 1)
            .ok_or(HyperKZGError::SrsTooSmall {
                have: setup.g1_powers.len(),
                need: shift + degree + 1,
            })?;
    let shifted_commitment = P::g1_affine_msm(bases, &combined);
    transcript.append(&shifted_commitment);

    let remainders = interpolation_remainders(points, evaluations)?;
    let union = point_union(points);
    let vanishing_union = vanishing_polynomial(&union);
    let gamma = transcript.challenge();
    let gamma_powers = challenge_powers(gamma, polynomials.len());
    let mut aggregate = Vec::new();
    for (((polynomial, point_set), remainder), &coefficient) in polynomials
        .iter()
        .zip(points)
        .zip(&remainders)
        .zip(&gamma_powers)
    {
        let complement = union
            .iter()
            .copied()
            .filter(|point| !point_set.contains(point))
            .collect::<Vec<_>>();
        let term = multiply(
            &vanishing_polynomial(&complement),
            &subtract(polynomial, remainder),
        );
        add_scaled(&mut aggregate, &term, coefficient);
    }
    let quotient = divide_exact(&aggregate, &vanishing_union)?;
    let quotient_commitment = kzg_commit(&quotient, setup)?;
    transcript.append(&quotient_commitment);

    let z = transcript.challenge();
    let vanishing_at_z = eval_univariate(&vanishing_union, z);
    let mut evaluation_polynomial = Vec::new();
    for (((polynomial, point_set), remainder), &coefficient) in polynomials
        .iter()
        .zip(points)
        .zip(&remainders)
        .zip(&gamma_powers)
    {
        let complement = union
            .iter()
            .copied()
            .filter(|point| !point_set.contains(point))
            .collect::<Vec<_>>();
        let scale = coefficient * eval_univariate(&vanishing_polynomial(&complement), z);
        let mut centered = polynomial.clone();
        if let Some(constant) = centered.first_mut() {
            *constant -= eval_univariate(remainder, z);
        }
        add_scaled(&mut evaluation_polynomial, &centered, scale);
    }
    add_scaled(&mut evaluation_polynomial, &quotient, -vanishing_at_z);
    let witness = divide_exact(&evaluation_polynomial, &[-z, P::ScalarField::one()])?;
    let evaluation_witness = kzg_commit(&witness, setup)?;
    transcript.append(&evaluation_witness);

    Ok(VariableBatchKzgProof {
        shifted_commitment,
        quotient_commitment,
        evaluation_witness,
    })
}

pub fn verify_variable_batch<P, T>(
    commitments: &[P::G1],
    points: &[[P::ScalarField; 3]],
    evaluations: &[[P::ScalarField; 3]],
    degree: usize,
    proof: &VariableBatchKzgProof<P>,
    setup: &HyperKZGVerifierSetup<P>,
    transcript: &mut T,
) -> Result<(), HyperKZGError>
where
    P: PairingGroup,
    T: Transcript<Challenge = P::ScalarField>,
    P::ScalarField: AppendToTranscript,
    P::G1: AppendToTranscript,
{
    validate_batch(commitments, points, evaluations, degree)?;
    let rho = transcript.challenge();
    let rho_powers = challenge_powers(rho, commitments.len());
    transcript.append(&proof.shifted_commitment);
    let combined_commitment = P::g1_msm(commitments, &rho_powers);
    let degree_check = P::multi_pairing(
        &[proof.shifted_commitment, -combined_commitment],
        &[setup.g2, setup.degree_five_shift_g2],
    );
    if !degree_check.is_identity() {
        return Err(HyperKZGError::DegreeBoundCheckFailed);
    }

    let remainders = interpolation_remainders(points, evaluations)?;
    let union = point_union(points);
    let gamma = transcript.challenge();
    let gamma_powers = challenge_powers(gamma, commitments.len());
    transcript.append(&proof.quotient_commitment);
    let z = transcript.challenge();
    let mut folded_commitment = P::G1::identity();
    for (((&commitment, point_set), remainder), &coefficient) in commitments
        .iter()
        .zip(points)
        .zip(&remainders)
        .zip(&gamma_powers)
    {
        let complement = union
            .iter()
            .copied()
            .filter(|point| !point_set.contains(point))
            .collect::<Vec<_>>();
        let scale = coefficient * eval_univariate(&vanishing_polynomial(&complement), z);
        let remainder_commitment = setup.g1.scalar_mul(&eval_univariate(remainder, z));
        folded_commitment += (commitment - remainder_commitment).scalar_mul(&scale);
    }
    let vanishing_at_z = eval_univariate(&vanishing_polynomial(&union), z);
    folded_commitment -= proof.quotient_commitment.scalar_mul(&vanishing_at_z);
    transcript.append(&proof.evaluation_witness);
    let beta_minus_z = setup.beta_g2 - setup.g2.scalar_mul(&z);
    let opening_check = P::multi_pairing(
        &[folded_commitment, -proof.evaluation_witness],
        &[setup.g2, beta_minus_z],
    );
    if !opening_check.is_identity() {
        return Err(HyperKZGError::VariableBatchCheckFailed);
    }
    Ok(())
}

fn validate_batch<F: JoltField, E>(
    entries: &[E],
    points: &[[F; 3]],
    evaluations: &[[F; 3]],
    degree: usize,
) -> Result<(), HyperKZGError> {
    if degree != SUPPORTED_DEGREE {
        return Err(HyperKZGError::UnsupportedDegreeBound(degree));
    }
    if entries.is_empty() || entries.len() != points.len() || entries.len() != evaluations.len() {
        return Err(HyperKZGError::InvalidBatchShape);
    }
    if points.iter().any(|&[a, b, c]| a == b || a == c || b == c) {
        return Err(HyperKZGError::RepeatedBatchPoint);
    }
    Ok(())
}

fn interpolation_remainders<F: JoltField>(
    points: &[[F; 3]],
    evaluations: &[[F; 3]],
) -> Result<Vec<[F; 3]>, HyperKZGError> {
    points
        .iter()
        .zip(evaluations)
        .map(|(&[a, b, c], &[ya, yb, yc])| {
            let mut result = [F::zero(); 3];
            for (x, y, other_a, other_b) in [(a, ya, b, c), (b, yb, a, c), (c, yc, a, b)] {
                let scale = y
                    * ((x - other_a) * (x - other_b))
                        .inverse()
                        .ok_or(HyperKZGError::RepeatedBatchPoint)?;
                result[0] += scale * other_a * other_b;
                result[1] -= scale * (other_a + other_b);
                result[2] += scale;
            }
            Ok(result)
        })
        .collect()
}

fn point_union<F: JoltField>(points: &[[F; 3]]) -> Vec<F> {
    let mut union = Vec::new();
    for &point in points.iter().flatten() {
        if !union.contains(&point) {
            union.push(point);
        }
    }
    union
}

fn vanishing_polynomial<F: JoltField>(points: &[F]) -> Vec<F> {
    points.iter().fold(vec![F::one()], |polynomial, &point| {
        multiply(&polynomial, &[-point, F::one()])
    })
}

fn subtract<F: JoltField>(left: &[F], right: &[F]) -> Vec<F> {
    let mut result = left.to_vec();
    result.resize(result.len().max(right.len()), F::zero());
    for (value, &subtrahend) in result.iter_mut().zip(right) {
        *value -= subtrahend;
    }
    trim(result)
}

#[expect(
    clippy::indexing_slicing,
    reason = "convolution bounds pin the product index below the allocated length"
)]
fn multiply<F: JoltField>(left: &[F], right: &[F]) -> Vec<F> {
    if left.is_empty() || right.is_empty() {
        return Vec::new();
    }
    let mut product = vec![F::zero(); left.len() + right.len() - 1];
    for (left_degree, &left_coefficient) in left.iter().enumerate() {
        for (right_degree, &right_coefficient) in right.iter().enumerate() {
            product[left_degree + right_degree] += left_coefficient * right_coefficient;
        }
    }
    trim(product)
}

fn add_scaled<F: JoltField>(target: &mut Vec<F>, values: &[F], scale: F) {
    target.resize(target.len().max(values.len()), F::zero());
    for (target, &value) in target.iter_mut().zip(values) {
        *target += scale * value;
    }
    while target.last().is_some_and(F::is_zero) {
        target.truncate(target.len().saturating_sub(1));
    }
}

#[expect(
    clippy::indexing_slicing,
    reason = "descending monic division bounds pin every dividend, divisor, and quotient index"
)]
fn divide_exact<F: JoltField>(dividend: &[F], divisor: &[F]) -> Result<Vec<F>, HyperKZGError> {
    let divisor_degree = divisor
        .len()
        .checked_sub(1)
        .ok_or(HyperKZGError::NonzeroQuotientRemainder)?;
    if dividend.len() < divisor.len() {
        return if dividend.iter().all(F::is_zero) {
            Ok(Vec::new())
        } else {
            Err(HyperKZGError::NonzeroQuotientRemainder)
        };
    }
    let mut remainder = dividend.to_vec();
    let mut quotient = vec![F::zero(); dividend.len() - divisor_degree];
    for degree in (divisor_degree..dividend.len()).rev() {
        let coefficient = remainder[degree];
        let quotient_degree = degree - divisor_degree;
        quotient[quotient_degree] = coefficient;
        for (offset, &divisor_coefficient) in divisor.iter().enumerate() {
            remainder[quotient_degree + offset] -= coefficient * divisor_coefficient;
        }
    }
    if remainder.iter().any(|value| !value.is_zero()) {
        return Err(HyperKZGError::NonzeroQuotientRemainder);
    }
    Ok(trim(quotient))
}

fn trim<F: JoltField>(mut polynomial: Vec<F>) -> Vec<F> {
    while polynomial.last().is_some_and(F::is_zero) {
        polynomial.truncate(polynomial.len().saturating_sub(1));
    }
    polynomial
}
