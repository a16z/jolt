//! Univariate KZG primitives: commit, witness polynomial, batch open/verify.
//!
//! These are the building blocks consumed by the HyperKZG protocol.
//! All operations are generic over `P: PairingGroup`.

use jolt_crypto::{JoltGroup, PairingGroup};
use jolt_field::JoltField;
use jolt_transcript::{AppendToTranscript, Transcript};
use num_traits::Zero;
use rayon::prelude::*;

use crate::error::HyperKZGError;
use crate::types::{HyperKZGProverSetup, HyperKZGVerifierSetup};

/// Commits to a polynomial (given as evaluation/coefficient vector) using MSM against SRS G1 powers.
pub(crate) fn kzg_commit<P: PairingGroup>(
    coeffs: &[P::ScalarField],
    setup: &HyperKZGProverSetup<P>,
) -> Result<P::G1, HyperKZGError> {
    let bases = setup
        .g1_powers
        .get(..coeffs.len())
        .ok_or(HyperKZGError::SrsTooSmall {
            have: setup.g1_powers.len(),
            need: coeffs.len(),
        })?;
    Ok(P::g1_affine_msm(bases, coeffs))
}

/// Evaluates a polynomial (in evaluation/coefficient form) at a point.
///
/// Standard Horner evaluation: `f(u) = f[0] + f[1]*u + f[2]*u^2 + ...`
pub(crate) fn eval_univariate<F: JoltField>(coeffs: &[F], u: F) -> F {
    coeffs
        .iter()
        .rev()
        .fold(F::zero(), |result, &coefficient| result * u + coefficient)
}

/// Batch KZG opening at three points with one degree-three quotient witness.
///
/// Given polynomials `f[0..k]` and evaluation points `u[0..t]`, computes:
/// - `v[i][j]` = f_j(u_i) for all i, j
/// - Linear combination `B = sum_j q^j * f_j` using Fiat-Shamir challenge
/// - Witness commitment `w` = commit(B(x) / product_i(x - u_i))
///
/// Returns `(w, v)`.
#[expect(
    clippy::indexing_slicing,
    reason = "evaluation rows and points are fixed-size arrays of length three"
)]
pub(crate) fn kzg_open_batch<P, T>(
    f: &[Vec<P::ScalarField>],
    u: &[P::ScalarField; 3],
    setup: &HyperKZGProverSetup<P>,
    transcript: &mut T,
) -> (P::G1, [Vec<P::ScalarField>; 3])
where
    P: PairingGroup,
    T: Transcript<Challenge = P::ScalarField>,
    P::ScalarField: AppendToTranscript,
    P::G1: AppendToTranscript,
{
    let k = f.len();

    // Compute evaluations v[t][j] = f_j(u_t)
    let evaluations = f
        .par_iter()
        .map(|fj| (*u).map(|ui| eval_univariate(fj, ui)))
        .collect::<Vec<_>>();
    let v = std::array::from_fn(|i| evaluations.iter().map(|row| row[i]).collect());

    // Absorb all evaluations into transcript
    for row in &v {
        for val in row {
            transcript.append(val);
        }
    }

    // Derive batching challenge and compute powers q, q^2, ..., q^{k-1}
    let q: P::ScalarField = transcript.challenge();
    let q_powers = challenge_powers(q, k);

    // B(x) = sum_j q^j * f_j(x)
    let mut b_poly = f.first().cloned().unwrap_or_default();
    for (fj, &qj) in f.iter().zip(q_powers.iter()).skip(1) {
        b_poly
            .par_iter_mut()
            .zip(fj.par_iter())
            .for_each(|(b, &c)| *b += qj * c);
    }

    let divisor = vanishing_polynomial(u);
    let h = divide_by_monic_cubic(&b_poly, &divisor);
    #[expect(
        clippy::indexing_slicing,
        reason = "prover SRS covers the full polynomial length and the witness polynomial is strictly shorter"
    )]
    let bases = &setup.g1_powers[..h.len()];
    let w = P::g1_affine_msm(bases, &h);

    transcript.append(&w);

    (w, v)
}

/// Batch KZG verification: checks that commitments open correctly at all points.
///
/// Optimized for the t=3 case used by HyperKZG. The pairing check verifies
/// `e(C_B - C_R, g2) == e(W, Z(beta) * g2)`.
pub(crate) fn kzg_verify_batch<P, T>(
    vk: &HyperKZGVerifierSetup<P>,
    com: &[P::G1],
    wit: P::G1,
    u: &[P::ScalarField; 3],
    v: &[Vec<P::ScalarField>; 3],
    transcript: &mut T,
) -> bool
where
    P: PairingGroup,
    T: Transcript<Challenge = P::ScalarField>,
    P::ScalarField: AppendToTranscript,
    P::G1: AppendToTranscript,
{
    let k = com.len();

    if v.iter().any(|row| row.len() != k) {
        return false;
    }

    // Absorb evaluations
    for row in v {
        for val in row {
            transcript.append(val);
        }
    }

    let q: P::ScalarField = transcript.challenge();
    let q_powers = challenge_powers(q, k);

    transcript.append(&wit);

    // B(u_i) = sum_j q^j * v[i][j]
    let b_u: [P::ScalarField; 3] = v.each_ref().map(|v_i| {
        v_i.iter()
            .zip(q_powers.iter())
            .map(|(&a, &b)| a * b)
            .fold(P::ScalarField::zero(), |acc, x| acc + x)
    });

    let Some(remainder) = interpolate_three(u, &b_u) else {
        return false;
    };
    let b_commitment = P::g1_msm(com, &q_powers);
    let remainder_commitment = P::g1_msm(&[vk.g1, vk.beta_g1, vk.beta_sq_g1], &remainder);
    let divisor = vanishing_polynomial(u);
    let divisor_at_beta = vk.g2.scalar_mul(&divisor[0])
        + vk.beta_g2.scalar_mul(&divisor[1])
        + vk.beta_sq_g2.scalar_mul(&divisor[2])
        + vk.beta_cu_g2;

    let result = P::multi_pairing(
        &[b_commitment - remainder_commitment, -wit],
        &[vk.g2, divisor_at_beta],
    );
    result.is_identity()
}

fn vanishing_polynomial<F: JoltField>(u: &[F; 3]) -> [F; 4] {
    [
        -(u[0] * u[1] * u[2]),
        u[0] * u[1] + u[0] * u[2] + u[1] * u[2],
        -(u[0] + u[1] + u[2]),
        F::one(),
    ]
}

#[expect(
    clippy::indexing_slicing,
    reason = "loop bounds pin all polynomial and degree-three divisor indices"
)]
fn divide_by_monic_cubic<F: JoltField>(f: &[F], divisor: &[F; 4]) -> Vec<F> {
    if f.len() <= 3 {
        return vec![];
    }
    let mut quotient = vec![F::zero(); f.len() - 3];
    for i in (0..quotient.len()).rev() {
        let mut coefficient = f[i + 3];
        for offset in 1..=3 {
            if let Some(next) = quotient.get(i + offset) {
                coefficient -= divisor[3 - offset] * *next;
            }
        }
        quotient[i] = coefficient;
    }
    quotient
}

#[expect(
    clippy::indexing_slicing,
    reason = "all indices are reduced modulo fixed-size three-element arrays"
)]
fn interpolate_three<F: JoltField>(u: &[F; 3], y: &[F; 3]) -> Option<[F; 3]> {
    let mut result = [F::zero(); 3];
    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let scale = y[i] * ((u[i] - u[j]) * (u[i] - u[k])).inverse()?;
        result[0] += scale * u[j] * u[k];
        result[1] -= scale * (u[j] + u[k]);
        result[2] += scale;
    }
    Some(result)
}

/// Computes `[1, c, c^2, ..., c^{n-1}]`.
pub(crate) fn challenge_powers<F: JoltField>(c: F, n: usize) -> Vec<F> {
    let mut powers = Vec::with_capacity(n);
    let mut cur = F::one();
    for _ in 0..n {
        powers.push(cur);
        cur *= c;
    }
    powers
}

#[cfg(test)]
mod tests {
    #![expect(clippy::indexing_slicing, reason = "tests index fixture vectors")]

    use super::*;
    use jolt_field::{Fr, Ring};
    use num_traits::Zero;

    #[test]
    fn cubic_quotient_and_remainder() {
        let roots = [Fr::from_u64(2), Fr::from_u64(4), Fr::from_u64(6)];
        let divisor = vanishing_polynomial(&roots);
        let expected = [
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
            Fr::from_u64(11),
        ];
        let remainder = [Fr::from_u64(13), Fr::from_u64(17), Fr::from_u64(19)];
        let mut polynomial = vec![Fr::zero(); expected.len() + divisor.len() - 1];
        for (i, &a) in expected.iter().enumerate() {
            for (j, &b) in divisor.iter().enumerate() {
                polynomial[i + j] += a * b;
            }
        }
        for (coefficient, &value) in polynomial.iter_mut().zip(&remainder) {
            *coefficient += value;
        }

        assert_eq!(divide_by_monic_cubic(&polynomial, &divisor), expected);
        let values = roots.map(|root| eval_univariate(&remainder, root));
        assert_eq!(interpolate_three(&roots, &values), Some(remainder));
    }

    #[test]
    fn eval_univariate_at_zero() {
        let f = vec![Fr::from_u64(42), Fr::from_u64(7), Fr::from_u64(3)];
        assert_eq!(eval_univariate(&f, Fr::zero()), Fr::from_u64(42));
    }

    #[test]
    fn eval_univariate_linear() {
        // f(x) = 3 + 5x, f(2) = 13
        let f = vec![Fr::from_u64(3), Fr::from_u64(5)];
        assert_eq!(eval_univariate(&f, Fr::from_u64(2)), Fr::from_u64(13));
    }
}
