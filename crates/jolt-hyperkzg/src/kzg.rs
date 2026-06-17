//! Univariate KZG primitives: commit, witness polynomial, batch open/verify.
//!
//! These are the building blocks consumed by the HyperKZG protocol.
//! All operations are generic over `P: PairingGroup`.

use crate::error::HyperKZGError;
use crate::types::{HyperKZGProverSetup, HyperKZGVerifierSetup};
use jolt_crypto::{JoltGroup, PairingGroup};
use jolt_field::Field;
use jolt_transcript::{AppendToTranscript, Transcript};
use num_traits::{One, Zero};
use rayon::join;

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

/// Adds one scalar hiding shift to a transparent KZG commitment.
#[cfg(feature = "zk")]
pub(crate) fn blind_commitment<P: PairingGroup>(
    commitment: P::G1,
    hiding_base: P::G1,
    blind: P::ScalarField,
) -> P::G1 {
    commitment + hiding_base.scalar_mul(&blind)
}

/// Commits to a V2 hidden scalar evaluation:
/// `value * G + (rho + u*tau) * H - tau * H_1`.
#[cfg(feature = "zk")]
pub(crate) fn randomized_eval_commitment<P: PairingGroup>(
    value_base: P::G1,
    hiding_base: P::G1,
    beta_hiding_base: P::G1,
    value: P::ScalarField,
    rho: P::ScalarField,
    tau: P::ScalarField,
    u: P::ScalarField,
) -> P::G1 {
    value_base.scalar_mul(&value) + hiding_base.scalar_mul(&(rho + u * tau))
        - beta_hiding_base.scalar_mul(&tau)
}

/// Computes the KZG witness polynomial `h(x) = f(x) / (x - u)`.
///
/// Uses Horner's method in reverse: `h[i-1] = f[i] + h[i] * u`.
/// The remainder is `f(u)`, but we don't need it since the verifier
/// can derive it from the evaluation vectors.
pub(crate) fn compute_witness_polynomial<F: Field>(f: &[F], u: F) -> Vec<F> {
    let d = f.len();
    if d <= 1 {
        return vec![];
    }
    let mut h = vec![F::zero(); d - 1];
    h[d - 2] = f[d - 1];
    for i in (1..d - 1).rev() {
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

/// Computes the transparent KZG witness commitment for one polynomial at `u`.
#[cfg(feature = "zk")]
pub(crate) fn kzg_witness_commitment<P: PairingGroup>(
    coeffs: &[P::ScalarField],
    u: P::ScalarField,
    setup: &HyperKZGProverSetup<P>,
) -> Result<P::G1, HyperKZGError> {
    let witness = compute_witness_polynomial::<P::ScalarField>(coeffs, u);
    if setup.g1_powers.len() < witness.len() {
        return Err(HyperKZGError::SrsTooSmall {
            have: setup.g1_powers.len(),
            need: witness.len(),
        });
    }
    Ok(P::G1::msm(&setup.g1_powers[..witness.len()], &witness))
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
    u: &[P::ScalarField; 3],
    setup: &HyperKZGProverSetup<P>,
    transcript: &mut T,
) -> ([P::G1; 3], [Vec<P::ScalarField>; 3])
where
    P: PairingGroup,
    T: Transcript<Challenge = P::ScalarField>,
    P::ScalarField: AppendToTranscript,
    P::G1: AppendToTranscript,
{
    let k = f.len();

    // Compute evaluations v[t][j] = f_j(u_t)
    let eval_row = |u_i| {
        f.iter()
            .map(|fj| eval_univariate(fj, u_i))
            .collect::<Vec<_>>()
    };
    let (v_r, (v_neg_r, v_r2)) = join(
        || eval_row(u[0]),
        || join(|| eval_row(u[1]), || eval_row(u[2])),
    );
    let v = [v_r, v_neg_r, v_r2];

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
    let poly_len = f[0].len();
    let mut b_poly = vec![P::ScalarField::zero(); poly_len];
    for (fj, &qj) in f.iter().zip(q_powers.iter()) {
        for (b, &c) in b_poly.iter_mut().zip(fj.iter()) {
            *b += qj * c;
        }
    }

    // Compute witness polynomials and commit
    let witness = |u_i| {
        let h = compute_witness_polynomial::<P::ScalarField>(&b_poly, u_i);
        P::G1::msm(&setup.g1_powers[..h.len()], &h)
    };
    let (w_r, (w_neg_r, w_r2)) = join(
        || witness(u[0]),
        || join(|| witness(u[1]), || witness(u[2])),
    );
    let w = [w_r, w_neg_r, w_r2];

    // Absorb witness commitments and mirror the verifier's `d_0` challenge
    // to keep prover/verifier transcripts in sync.
    for wi in &w {
        transcript.append(wi);
    }
    let _d_0: P::ScalarField = transcript.challenge();

    (w, v)
}

/// Batch KZG verification: checks that commitments open correctly at all points.
///
/// Optimized for the t=3 case used by HyperKZG. The pairing check verifies:
/// `e(L, g2) == e(R, beta_g2)`
pub(crate) fn kzg_verify_batch<P, T>(
    vk: &HyperKZGVerifierSetup<P>,
    com: &[P::G1],
    wit: &[P::G1; 3],
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

    // Absorb witness commitments
    for wi in wit {
        transcript.append(wi);
    }
    let d_0: P::ScalarField = transcript.challenge();
    let d_1 = d_0 * d_0;

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

/// Computes `[1, c, c^2, ..., c^{n-1}]`.
pub(crate) fn challenge_powers<F: Field>(c: F, n: usize) -> Vec<F> {
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
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};
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
