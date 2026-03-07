//! Univariate KZG primitives: commit, witness polynomial, batch open/verify.
//!
//! These are the building blocks consumed by the HyperKZG protocol.
//! All operations are generic over `P: PairingGroup`.

#![allow(non_snake_case)]

use jolt_crypto::{JoltGroup, PairingGroup};
use jolt_field::Field;
use jolt_transcript::{AppendToTranscript, Transcript};
use num_traits::{One, Zero};
use rayon::prelude::*;

use crate::error::HyperKZGError;
use crate::types::{HyperKZGProverSetup, HyperKZGVerifierSetup};

/// Commits to a polynomial (given as evaluation/coefficient vector) using MSM against SRS G1 powers.
pub(crate) fn kzg_commit<P: PairingGroup>(
    coeffs: &[P::ScalarField],
    setup: &HyperKZGProverSetup<P>,
) -> Result<P::G1, HyperKZGError> {
    if setup.g1_powers.len() < coeffs.len() {
        return Err(HyperKZGError::SrsTooSmall {
            have: setup.g1_powers.len(),
            need: coeffs.len(),
        });
    }
    Ok(P::G1::msm(&setup.g1_powers[..coeffs.len()], coeffs))
}

/// Computes the KZG witness polynomial `h(x) = f(x) / (x - u)`.
///
/// Uses Horner's method in reverse: `h[i-1] = f[i] + h[i] * u`.
/// The remainder is `f(u)`, but we don't need it since the verifier
/// can derive it from the evaluation vectors.
pub(crate) fn compute_witness_polynomial<F: Field>(f: &[F], u: F) -> Vec<F> {
    let d = f.len();
    let mut h = vec![F::zero(); d];
    for i in (1..d).rev() {
        h[i - 1] = f[i] + h[i] * u;
    }
    h
}

/// Evaluates a polynomial (in evaluation/coefficient form) at a point.
///
/// Standard Horner evaluation: `f(u) = f[0] + f[1]*u + f[2]*u^2 + ...`
pub(crate) fn eval_univariate<F: Field>(coeffs: &[F], u: F) -> F {
    let mut result = F::zero();
    let mut power = F::one();
    for &c in coeffs {
        result += c * power;
        power *= u;
    }
    result
}

/// Batch KZG opening: commits to witness polynomials for each evaluation point.
///
/// Given polynomials `f[0..k]` and evaluation points `u[0..t]`, computes:
/// - `v[i][j]` = f_j(u_i) for all i, j
/// - Linear combination `B = sum_j q^j * f_j` using Fiat-Shamir challenge
/// - Witness commitments `w[i]` = commit(B(x) / (x - u_i))
///
/// Returns `(w, v)`.
pub(crate) fn kzg_open_batch<P, T>(
    f: &[Vec<P::ScalarField>],
    u: &[P::ScalarField],
    setup: &HyperKZGProverSetup<P>,
    transcript: &mut T,
) -> (Vec<P::G1>, Vec<Vec<P::ScalarField>>)
where
    P: PairingGroup,
    T: Transcript,
    P::ScalarField: AppendToTranscript,
    P::G1: AppendToTranscript,
{
    let k = f.len();

    // Compute evaluations v[i][j] = f_j(u_i)
    let v: Vec<Vec<P::ScalarField>> = u
        .par_iter()
        .map(|ui| f.iter().map(|fj| eval_univariate(fj, *ui)).collect())
        .collect();

    // Absorb all evaluations into transcript
    for row in &v {
        for val in row {
            transcript.append(val);
        }
    }

    // Derive batching challenge and compute powers q, q^2, ..., q^{k-1}
    let q_challenge = transcript.challenge();
    let q_powers = challenge_powers::<P::ScalarField, T>(q_challenge, k);

    // B(x) = sum_j q^j * f_j(x)
    let poly_len = f[0].len();
    let mut b_poly = vec![P::ScalarField::zero(); poly_len];
    for (fj, &qj) in f.iter().zip(q_powers.iter()) {
        for (b, &c) in b_poly.iter_mut().zip(fj.iter()) {
            *b += qj * c;
        }
    }

    // Compute witness polynomials and commit
    let w: Vec<P::G1> = u
        .par_iter()
        .map(|ui| {
            let h = compute_witness_polynomial::<P::ScalarField>(&b_poly, *ui);
            P::G1::msm(&setup.g1_powers[..h.len()], &h)
        })
        .collect();

    // Absorb witness commitments and derive one more challenge to keep
    // prover/verifier transcripts in sync
    for wi in &w {
        transcript.append(wi);
    }
    let _: <T as Transcript>::Challenge = transcript.challenge();

    (w, v)
}

/// Batch KZG verification: checks that commitments open correctly at all points.
///
/// Optimized for the t=3 case used by HyperKZG. The pairing check verifies:
/// `e(L, g2) == e(R, beta_g2)`
pub(crate) fn kzg_verify_batch<P, T>(
    vk: &HyperKZGVerifierSetup<P>,
    com: &[P::G1],
    wit: &[P::G1],
    u: &[P::ScalarField],
    v: &[Vec<P::ScalarField>],
    transcript: &mut T,
) -> bool
where
    P: PairingGroup,
    T: Transcript,
    P::ScalarField: AppendToTranscript,
    P::G1: AppendToTranscript,
{
    let k = com.len();

    // Absorb evaluations
    for row in v {
        for val in row {
            transcript.append(val);
        }
    }

    let q_challenge = transcript.challenge();
    let q_powers = challenge_powers::<P::ScalarField, T>(q_challenge, k);

    // Absorb witness commitments
    for wi in wit {
        transcript.append(wi);
    }
    let d_challenge = transcript.challenge();
    let d_0 = challenge_to_field::<P::ScalarField, T>(d_challenge);
    let d_1 = d_0 * d_0;

    assert_eq!(wit.len(), 3, "HyperKZG requires exactly 3 evaluation points");

    // q_power_multiplier = 1 + d_0 + d_1
    let q_power_multiplier = P::ScalarField::one() + d_0 + d_1;
    let q_powers_multiplied: Vec<P::ScalarField> =
        q_powers.iter().map(|qp| *qp * q_power_multiplier).collect();

    // B(u_i) = sum_j q^j * v[i][j]
    let b_u: Vec<P::ScalarField> = v
        .iter()
        .map(|v_i| {
            v_i.iter()
                .zip(q_powers.iter())
                .map(|(&a, &b)| a * b)
                .fold(P::ScalarField::zero(), |acc, x| acc + x)
        })
        .collect();

    // L = MSM over [C_0..C_{k-1}, W_0, W_1, W_2, g1] with scalars
    //   [q_powers_multiplied, u_0, u_1*d_0, u_2*d_1, -(b_u[0] + d_0*b_u[1] + d_1*b_u[2])]
    let mut bases = Vec::with_capacity(k + 4);
    bases.extend_from_slice(&com[..k]);
    bases.push(wit[0]);
    bases.push(wit[1]);
    bases.push(wit[2]);
    bases.push(vk.g1);

    let mut scalars = Vec::with_capacity(k + 4);
    scalars.extend_from_slice(&q_powers_multiplied[..k]);
    scalars.push(u[0]);
    scalars.push(u[1] * d_0);
    scalars.push(u[2] * d_1);
    scalars.push(-(b_u[0] + d_0 * b_u[1] + d_1 * b_u[2]));

    let lhs = P::G1::msm(&bases, &scalars);

    // R = W[0] + d_0*W[1] + d_1*W[2]
    let rhs = wit[0] + wit[1].scalar_mul(&d_0) + wit[2].scalar_mul(&d_1);

    // e(L, g2) * e(-R, beta_g2) == identity
    let result = P::multi_pairing(&[lhs, -rhs], &[vk.g2, vk.beta_g2]);
    result.is_identity()
}

/// Computes `[1, c, c^2, ..., c^{n-1}]` from a transcript challenge.
pub(crate) fn challenge_powers<F: Field, T: Transcript>(challenge: T::Challenge, n: usize) -> Vec<F> {
    let c = challenge_to_field::<F, T>(challenge);
    let mut powers = Vec::with_capacity(n);
    let mut cur = F::one();
    for _ in 0..n {
        powers.push(cur);
        cur *= c;
    }
    powers
}

/// Converts a transcript challenge to a field element.
///
/// The transcript challenge is a `u128` (for `Blake2bTranscript`).
/// We interpret it as a field element via `from_bytes`.
pub(crate) fn challenge_to_field<F: Field, T: Transcript>(challenge: T::Challenge) -> F {
    // SAFETY: Challenge is Copy + Default, typically u128. Reading as bytes is safe
    // because we only read `size_of::<Challenge>()` bytes from a valid reference.
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(&challenge).cast::<u8>(),
            std::mem::size_of::<T::Challenge>(),
        )
    };
    F::from_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use num_traits::Zero;

    #[test]
    fn witness_polynomial_division() {
        // f(x) = 1 + 2x + 3x^2 + 4x^3
        // f(2) = 1 + 4 + 12 + 32 = 49
        // h(x) = f(x)/(x-2), so f(x) = (x-2)*h(x) + f(2)
        let f = vec![
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
        ];
        let u = Fr::from_u64(2);
        let h = compute_witness_polynomial::<Fr>(&f, u);

        // Verify: (x-u)*h(x) + f(u) should reconstruct f(x)
        for x_val in [0u64, 1, 3, 5, 100] {
            let x = Fr::from_u64(x_val);
            let fx = eval_univariate(&f, x);
            let hx = eval_univariate(&h, x);
            let fu = eval_univariate(&f, u);
            assert_eq!(fx, (x - u) * hx + fu);
        }
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
