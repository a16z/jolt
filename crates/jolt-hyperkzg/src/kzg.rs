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
use crate::types::{
    HyperKZGProverSetup, HyperKZGVerifierSetup, NoopVerifierObserver, VerifierObserver,
};

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

/// Evaluates a polynomial (coefficient form) at a point:
/// `f(u) = Σ_j u^(jC) · Horner(chunk_j, u)` with the `C`-coefficient chunks
/// evaluated in parallel.
pub(crate) fn eval_univariate<F: JoltField>(coeffs: &[F], u: F) -> F {
    const CHUNK_LOG: usize = 12;
    if coeffs.len() <= 1 << CHUNK_LOG {
        return horner(coeffs, u);
    }
    let mut u_chunk = u;
    for _ in 0..CHUNK_LOG {
        u_chunk = u_chunk.square();
    }
    let partials: Vec<F> = coeffs
        .par_chunks(1 << CHUNK_LOG)
        .map(|chunk| horner(chunk, u))
        .collect();
    horner(&partials, u_chunk)
}

fn horner<F: JoltField>(coeffs: &[F], u: F) -> F {
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
) -> (P::G1, [Vec<P::ScalarField>; 2], P::ScalarField)
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
    let v: [Vec<P::ScalarField>; 3] =
        std::array::from_fn(|i| evaluations.iter().map(|row| row[i]).collect());

    // P_1(r^2)..P_{ell-1}(r^2) follow from the Gemini fold identities.
    for row in v.iter().take(2) {
        for val in row {
            transcript.append(val);
        }
    }
    let p0_at_r_squared = v[2].first().copied().unwrap_or_default();
    transcript.append(&p0_at_r_squared);

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

    (w, [v[0].clone(), v[1].clone()], p0_at_r_squared)
}

/// Batch KZG verification: checks that commitments open correctly at all points.
///
/// Optimized for the t=3 case used by HyperKZG. Divisor coefficients multiply
/// G1 arguments so verification needs only the four fixed G2 SRS powers.
pub(crate) fn kzg_verify_batch<P, T, O>(
    vk: &HyperKZGVerifierSetup<P>,
    com: &[P::G1],
    wit: P::G1,
    u: &[P::ScalarField; 3],
    v: &[Vec<P::ScalarField>; 3],
    transcript: &mut T,
    observer: &mut O,
) -> bool
where
    P: PairingGroup,
    T: Transcript<Challenge = P::ScalarField>,
    O: VerifierObserver,
    P::ScalarField: AppendToTranscript,
    P::G1: AppendToTranscript,
{
    let k = com.len();

    if v.iter().any(|row| row.len() != k) {
        return false;
    }

    for row in v.iter().take(2) {
        for val in row {
            transcript.append(val);
        }
    }
    let Some(&p0_at_r_squared) = v[2].first() else {
        return false;
    };
    transcript.append(&p0_at_r_squared);

    let q: P::ScalarField = transcript.challenge();
    let q_powers = challenge_powers_observed(q, k, observer);

    transcript.append(&wit);

    // B(u_i) = sum_j q^j * v[i][j]
    let mut b_u = [P::ScalarField::zero(); 3];
    for (value, row) in b_u.iter_mut().zip(v) {
        for (&evaluation, &coefficient) in row.iter().zip(&q_powers) {
            *value += observer.fr_mul(evaluation, coefficient);
        }
    }

    let Some(remainder) = interpolate_three_observed(u, &b_u, observer) else {
        return false;
    };
    let b_commitment = P::g1_msm(com, &q_powers);
    let remainder_commitment = P::g1_msm(&[vk.g1, vk.beta_g1, vk.beta_sq_g1], &remainder);
    let divisor = vanishing_polynomial_observed(u, observer);
    let [z0, z1, z2, _] = divisor;
    observer.ec_mul(k + 6);
    observer.ec_add(k + 5);
    observer.pairing_pairs(4);
    let result = P::multi_pairing(
        &[
            b_commitment - remainder_commitment - wit.scalar_mul(&z0),
            -wit.scalar_mul(&z1),
            -wit.scalar_mul(&z2),
            -wit,
        ],
        &[vk.g2, vk.beta_g2, vk.beta_sq_g2, vk.beta_cu_g2],
    );
    result.is_identity()
}

pub(crate) fn eval_univariate_observed<F, O>(coeffs: &[F], point: F, observer: &mut O) -> F
where
    F: JoltField,
    O: VerifierObserver,
{
    coeffs.iter().rev().fold(F::zero(), |result, &coefficient| {
        observer.fr_mul(result, point) + coefficient
    })
}

fn vanishing_polynomial<F: JoltField>(u: &[F; 3]) -> [F; 4] {
    vanishing_polynomial_observed(u, &mut NoopVerifierObserver)
}

fn vanishing_polynomial_observed<F, O>(u: &[F; 3], observer: &mut O) -> [F; 4]
where
    F: JoltField,
    O: VerifierObserver,
{
    let u0_u1 = observer.fr_mul(u[0], u[1]);
    let u0_u2 = observer.fr_mul(u[0], u[2]);
    let u1_u2 = observer.fr_mul(u[1], u[2]);
    [
        -observer.fr_mul(u0_u1, u[2]),
        u0_u1 + u0_u2 + u1_u2,
        -(u[0] + u[1] + u[2]),
        F::one(),
    ]
}

/// Quotient of `f` by a monic cubic (the remainder is dropped). The top-down
/// recurrence `q[i] = f[i+3] − d2·q[i+1] − d1·q[i+2] − d0·q[i+3]` runs in
/// parallel blocks: every block is solved with a zero incoming state, the true
/// boundary states are propagated with the block transition matrix, and each
/// block then adds its homogeneous correction.
#[expect(
    clippy::indexing_slicing,
    reason = "block slices stay inside f (three longer than the quotient) and incoming has one state per block"
)]
fn divide_by_monic_cubic<F: JoltField>(f: &[F], divisor: &[F; 4]) -> Vec<F> {
    let Some(n) = f.len().checked_sub(3).filter(|&n| n > 0) else {
        return vec![];
    };
    let block = (n / (4 * rayon::current_num_threads())).max(1 << 12);
    let mut quotient = vec![F::zero(); n];
    quotient
        .par_chunks_mut(block)
        .enumerate()
        .for_each(|(index, q)| {
            let lo = index * block;
            divide_block(&f[lo..lo + q.len() + 3], divisor, q, [F::zero(); 3]);
        });
    let blocks = n.div_ceil(block);
    if blocks == 1 {
        return quotient;
    }
    let transition = matrix_power(companion(divisor), block);
    let mut incoming = vec![[F::zero(); 3]; blocks];
    for index in (0..blocks - 1).rev() {
        let above = index + 1;
        let lo = above * block;
        let local: [F; 3] =
            std::array::from_fn(|i| quotient.get(lo + i).copied().unwrap_or(F::zero()));
        incoming[index] = if above == blocks - 1 {
            local
        } else {
            let carried = matvec(&transition, incoming[above]);
            std::array::from_fn(|i| carried[i] + local[i])
        };
    }
    quotient
        .par_chunks_mut(block)
        .zip(incoming.par_iter())
        .for_each(|(q, &state)| {
            if state == [F::zero(); 3] {
                return;
            }
            let mut homogeneous = state;
            for value in q.iter_mut().rev() {
                homogeneous = matvec(&companion(divisor), homogeneous);
                *value += homogeneous[0];
            }
        });
    quotient
}

/// One block of the quotient recurrence; `f.len() == q.len() + 3` and
/// `incoming` holds `q[hi], q[hi+1], q[hi+2]` from the block above.
#[expect(
    clippy::indexing_slicing,
    reason = "f has three more entries than q and incoming covers the three indices past q"
)]
fn divide_block<F: JoltField>(f: &[F], divisor: &[F; 4], q: &mut [F], incoming: [F; 3]) {
    let len = q.len();
    for i in (0..len).rev() {
        let mut coefficient = f[i + 3];
        for offset in 1..=3 {
            let next = if i + offset < len {
                q[i + offset]
            } else {
                incoming[i + offset - len]
            };
            coefficient -= divisor[3 - offset] * next;
        }
        q[i] = coefficient;
    }
}

/// `s_i = A · s_{i+1}` for the state `s_i = (q[i], q[i+1], q[i+2])`.
fn companion<F: JoltField>(divisor: &[F; 4]) -> [[F; 3]; 3] {
    [
        [-divisor[2], -divisor[1], -divisor[0]],
        [F::one(), F::zero(), F::zero()],
        [F::zero(), F::one(), F::zero()],
    ]
}

fn matvec<F: JoltField>(matrix: &[[F; 3]; 3], vector: [F; 3]) -> [F; 3] {
    matrix.map(|row| {
        row.iter()
            .zip(&vector)
            .fold(F::zero(), |acc, (&a, &b)| acc + a * b)
    })
}

#[expect(clippy::indexing_slicing, reason = "fixed-size three-by-three arrays")]
fn matmul<F: JoltField>(left: &[[F; 3]; 3], right: &[[F; 3]; 3]) -> [[F; 3]; 3] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..3).fold(F::zero(), |acc, k| acc + left[i][k] * right[k][j]))
    })
}

fn matrix_power<F: JoltField>(matrix: [[F; 3]; 3], mut exponent: usize) -> [[F; 3]; 3] {
    let mut result = [
        [F::one(), F::zero(), F::zero()],
        [F::zero(), F::one(), F::zero()],
        [F::zero(), F::zero(), F::one()],
    ];
    let mut base = matrix;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = matmul(&result, &base);
        }
        base = matmul(&base, &base);
        exponent >>= 1;
    }
    result
}

#[cfg(test)]
fn interpolate_three<F: JoltField>(u: &[F; 3], y: &[F; 3]) -> Option<[F; 3]> {
    interpolate_three_observed(u, y, &mut NoopVerifierObserver)
}

#[expect(
    clippy::indexing_slicing,
    reason = "all indices are reduced modulo fixed-size three-element arrays"
)]
fn interpolate_three_observed<F, O>(u: &[F; 3], y: &[F; 3], observer: &mut O) -> Option<[F; 3]>
where
    F: JoltField,
    O: VerifierObserver,
{
    let mut result = [F::zero(); 3];
    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let denominator = observer.fr_mul(u[i] - u[j], u[i] - u[k]);
        let denominator_inverse = observer.fr_inv(denominator)?;
        let scale = observer.fr_mul(y[i], denominator_inverse);
        let scale_u_j = observer.fr_mul(scale, u[j]);
        result[0] += observer.fr_mul(scale_u_j, u[k]);
        result[1] -= observer.fr_mul(scale, u[j] + u[k]);
        result[2] += scale;
    }
    Some(result)
}

/// Computes `[1, c, c^2, ..., c^{n-1}]`.
pub(crate) fn challenge_powers<F: JoltField>(c: F, n: usize) -> Vec<F> {
    challenge_powers_observed(c, n, &mut NoopVerifierObserver)
}

pub(crate) fn challenge_powers_observed<F, O>(c: F, n: usize, observer: &mut O) -> Vec<F>
where
    F: JoltField,
    O: VerifierObserver,
{
    let mut powers = Vec::with_capacity(n);
    let mut current = F::one();
    for _ in 0..n {
        powers.push(current);
        current = observer.fr_mul(current, c);
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
    fn cubic_quotient_spans_parallel_blocks() {
        let roots = [Fr::from_u64(2), Fr::from_u64(4), Fr::from_u64(6)];
        let divisor = vanishing_polynomial(&roots);
        let expected: Vec<Fr> = (0..(3 << 12) + 5)
            .map(|i| Fr::from_u64(i as u64 * 7 + 3))
            .collect();
        let mut polynomial = vec![Fr::zero(); expected.len() + 3];
        for (i, &a) in expected.iter().enumerate() {
            for (j, &b) in divisor.iter().enumerate() {
                polynomial[i + j] += a * b;
            }
        }
        polynomial[0] += Fr::from_u64(13);
        polynomial[2] += Fr::from_u64(17);
        assert_eq!(divide_by_monic_cubic(&polynomial, &divisor), expected);
    }

    #[test]
    fn eval_univariate_chunks_match_power_sum() {
        let f: Vec<Fr> = (0..(1 << 13) + 9)
            .map(|i| Fr::from_u64(i as u64 * 5 + 1))
            .collect();
        let u = Fr::from_u64(3);
        let mut power = Fr::from_u64(1);
        let mut expected = Fr::zero();
        for &coefficient in &f {
            expected += coefficient * power;
            power *= u;
        }
        assert_eq!(eval_univariate(&f, u), expected);
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
