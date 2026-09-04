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

type BatchOpening<P> = (
    <P as PairingGroup>::G1,
    [Vec<<P as PairingGroup>::ScalarField>; 4],
    <P as PairingGroup>::ScalarField,
);

pub(crate) struct FoldPoints<F> {
    points: [F; 5],
    root: F,
    residue_scales: [F; 4],
}

impl<F: JoltField> FoldPoints<F> {
    pub(crate) fn new<O: VerifierObserver>(r: F, observer: &mut O) -> Result<Self, HyperKZGError> {
        // BN254 Fr: i = 5^((p-1)/4), p = 1 mod 4.
        const ROOT_LE: [u8; 32] = [
            0x36, 0x36, 0x70, 0x8f, 0x70, 0x04, 0x12, 0x23, 0xec, 0x6b, 0x73, 0xfd, 0xf6, 0x24,
            0xea, 0x5c, 0x04, 0x41, 0xd8, 0x3f, 0x19, 0x6e, 0x8b, 0x04, 0x29, 0xa0, 0x31, 0xe1,
            0x72, 0x4e, 0x64, 0x30,
        ];
        let root = F::from_bytes_le_checked(&ROOT_LE)
            .filter(|&i| observer.fr_mul(i, i) == -F::one() && i != F::one())
            .ok_or(HyperKZGError::MissingFourthRootOfUnity)?;
        let r_squared = observer.fr_mul(r, r);
        let r_cubed = observer.fr_mul(r_squared, r);
        let r_fourth = observer.fr_mul(r_squared, r_squared);
        let ir = observer.fr_mul(root, r);
        if r.is_zero() || [r, ir, -r, -ir].contains(&r_fourth) {
            return Err(HyperKZGError::DegenerateChallenge);
        }
        let twice_r_cubed = r_cubed + r_cubed;
        let scale_3 = observer
            .fr_inv(twice_r_cubed + twice_r_cubed)
            .ok_or(HyperKZGError::DegenerateChallenge)?;
        let scale_2 = observer.fr_mul(scale_3, r);
        let scale_1 = observer.fr_mul(scale_2, r);
        let scale_0 = observer.fr_mul(scale_1, r);
        Ok(Self {
            points: [r, ir, -r, -ir, r_fourth],
            root,
            residue_scales: [scale_0, scale_1, scale_2, scale_3],
        })
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "from_fn visits the four entries of both arrays"
    )]
    pub(crate) fn residues<O: VerifierObserver>(&self, values: [F; 4], observer: &mut O) -> [F; 4] {
        let [a, b, c, d] = values;
        let odd = observer.fr_mul(self.root, d - b);
        let sums = [a + c + b + d, a - c + odd, a + c - b - d, a - c - odd];
        std::array::from_fn(|index| observer.fr_mul(sums[index], self.residue_scales[index]))
    }

    pub(crate) fn binary_residues<O: VerifierObserver>(
        &self,
        at_r: F,
        at_neg_r: F,
        observer: &mut O,
    ) -> [F; 2] {
        let [scale_0, scale_1, _, _] = self.residue_scales;
        [
            observer.fr_mul(at_r + at_neg_r, scale_0 + scale_0),
            observer.fr_mul(at_r - at_neg_r, scale_1 + scale_1),
        ]
    }

    fn divisor<O: VerifierObserver>(&self, observer: &mut O) -> [F; 6] {
        let s = self.points[4];
        [
            observer.fr_mul(s, s),
            -s,
            F::zero(),
            F::zero(),
            -s,
            F::one(),
        ]
    }

    fn interpolate<O: VerifierObserver>(&self, values: [F; 5], observer: &mut O) -> Option<[F; 5]> {
        let [a, b, c, d, y] = values;
        let cubic = self.residues([a, b, c, d], observer);
        let [c0, c1, c2, c3] = cubic;
        let s = self.points[4];
        let cubic_at_s = eval_univariate_observed(&cubic, s, observer);
        let s_squared = observer.fr_mul(s, s);
        let s_fourth = observer.fr_mul(s_squared, s_squared);
        let inverse = observer.fr_inv(s_fourth - s)?;
        let correction = observer.fr_mul(y - cubic_at_s, inverse);
        Some([c0 - observer.fr_mul(correction, s), c1, c2, c3, correction])
    }
}

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

/// Batch KZG opening at five points with one degree-five quotient witness.
///
/// Given polynomials `f[0..k]` and evaluation points `u[0..t]`, computes:
/// - `v[i][j]` = f_j(u_i) for all i, j
/// - Linear combination `B = sum_j q^j * f_j` using Fiat-Shamir challenge
/// - Witness commitment `w` = commit(B(x) / product_i(x - u_i))
///
/// Returns the witness, the first four evaluation rows, and `P_0(r^4)`.
#[expect(
    clippy::indexing_slicing,
    reason = "evaluation rows and points are fixed-size arrays of length five"
)]
pub(crate) fn kzg_open_batch<P, T>(
    f: &[Vec<P::ScalarField>],
    points: &FoldPoints<P::ScalarField>,
    setup: &HyperKZGProverSetup<P>,
    transcript: &mut T,
) -> Result<BatchOpening<P>, HyperKZGError>
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
        .map(|fj| points.points.map(|ui| eval_univariate(fj, ui)))
        .collect::<Vec<_>>();
    let v: [Vec<P::ScalarField>; 5] =
        std::array::from_fn(|i| evaluations.iter().map(|row| row[i]).collect());

    // P_1(r^4).. follow from the two-variable Gemini fold identities.
    for row in v.iter().take(4) {
        for val in row {
            transcript.append(val);
        }
    }
    let p0_at_r_fourth = v[4]
        .first()
        .copied()
        .ok_or(HyperKZGError::InvalidBatchShape)?;
    transcript.append(&p0_at_r_fourth);

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

    let divisor = points.divisor(&mut NoopVerifierObserver);
    let h = divide_by_monic_quintic(&b_poly, &divisor);
    let w = kzg_commit::<P>(&h, setup)?;

    transcript.append(&w);

    let [at_r, at_ir, at_neg_r, at_neg_ir, _] = v;
    Ok((w, [at_r, at_ir, at_neg_r, at_neg_ir], p0_at_r_fourth))
}

/// Batch KZG verification: checks that commitments open correctly at all points.
///
/// Optimized for the t=5 case used by HyperKZG. Divisor coefficients multiply
/// G1 arguments; the sparse divisor needs G2 powers at exponents 0, 1, 4, 5.
pub(crate) fn kzg_verify_batch<P, T, O>(
    vk: &HyperKZGVerifierSetup<P>,
    com: &[P::G1],
    wit: P::G1,
    points: &FoldPoints<P::ScalarField>,
    v: &[Vec<P::ScalarField>; 5],
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

    for row in v.iter().take(4) {
        for val in row {
            transcript.append(val);
        }
    }
    let Some(&p0_at_r_fourth) = v[4].first() else {
        return false;
    };
    transcript.append(&p0_at_r_fourth);

    let q: P::ScalarField = transcript.challenge();
    let q_powers = challenge_powers_observed(q, k, observer);

    transcript.append(&wit);

    // B(u_i) = sum_j q^j * v[i][j]
    let mut b_u = [P::ScalarField::zero(); 5];
    for (value, row) in b_u.iter_mut().zip(v) {
        for (&evaluation, &coefficient) in row.iter().zip(&q_powers) {
            *value += observer.fr_mul(evaluation, coefficient);
        }
    }

    let Some(remainder) = points.interpolate(b_u, observer) else {
        return false;
    };
    let b_commitment = P::g1_msm(com, &q_powers);
    let remainder_commitment = P::g1_msm(
        &[
            vk.g1,
            vk.beta_g1,
            vk.beta_sq_g1,
            vk.beta_cu_g1,
            vk.beta_fourth_g1,
        ],
        &remainder,
    );
    let [z0, z1, _, _, _, _] = points.divisor(observer);
    let scaled_witness = -wit.scalar_mul(&z1);
    observer.ec_mul(k + 7);
    observer.ec_add(k + 5);
    observer.pairing_pairs(4);
    let result = P::multi_pairing(
        &[
            b_commitment - remainder_commitment - wit.scalar_mul(&z0),
            scaled_witness,
            scaled_witness,
            -wit,
        ],
        &[vk.g2, vk.beta_g2, vk.beta_fourth_g2, vk.beta_fifth_g2],
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

/// Quotient of `f` by the monic quintic (the remainder is dropped). The top-down
/// recurrence runs in parallel blocks: each block is solved with zero incoming state, the true
/// boundary states are propagated with the block transition matrix, and each
/// block then adds its homogeneous correction.
#[expect(
    clippy::indexing_slicing,
    reason = "block slices stay inside f (five longer than the quotient) and incoming has one state per block"
)]
fn divide_by_monic_quintic<F: JoltField>(f: &[F], divisor: &[F; 6]) -> Vec<F> {
    let Some(n) = f.len().checked_sub(5).filter(|&n| n > 0) else {
        return vec![];
    };
    let block = (n / (4 * rayon::current_num_threads())).max(1 << 12);
    let mut quotient = vec![F::zero(); n];
    quotient
        .par_chunks_mut(block)
        .enumerate()
        .for_each(|(index, q)| {
            let lo = index * block;
            divide_block(&f[lo..lo + q.len() + 5], divisor, q, [F::zero(); 5]);
        });
    let blocks = n.div_ceil(block);
    if blocks == 1 {
        return quotient;
    }
    let transition = matrix_power(companion(divisor), block);
    let mut incoming = vec![[F::zero(); 5]; blocks];
    for index in (0..blocks - 1).rev() {
        let above = index + 1;
        let lo = above * block;
        let local: [F; 5] =
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
            if state == [F::zero(); 5] {
                return;
            }
            let mut homogeneous = state;
            for value in q.iter_mut().rev() {
                let [a, b, c, d, e] = homogeneous;
                homogeneous = [
                    -divisor[4] * a - divisor[1] * d - divisor[0] * e,
                    a,
                    b,
                    c,
                    d,
                ];
                *value += homogeneous[0];
            }
        });
    quotient
}

/// One block of the quotient recurrence; `f.len() == q.len() + 5` and
/// `incoming` holds the five quotient coefficients above the block.
#[expect(
    clippy::indexing_slicing,
    reason = "f has five more entries than q and incoming covers the five indices past q"
)]
fn divide_block<F: JoltField>(f: &[F], divisor: &[F; 6], q: &mut [F], incoming: [F; 5]) {
    let len = q.len();
    for i in (0..len).rev() {
        let next_1 = if i + 1 < len {
            q[i + 1]
        } else {
            incoming[i + 1 - len]
        };
        let next_4 = if i + 4 < len {
            q[i + 4]
        } else {
            incoming[i + 4 - len]
        };
        let next_5 = if i + 5 < len {
            q[i + 5]
        } else {
            incoming[i + 5 - len]
        };
        let coefficient =
            f[i + 5] - divisor[4] * next_1 - divisor[1] * next_4 - divisor[0] * next_5;
        q[i] = coefficient;
    }
}

/// `s_i = A · s_{i+1}` for five consecutive quotient coefficients.
fn companion<F: JoltField>(divisor: &[F; 6]) -> [[F; 5]; 5] {
    [
        [
            -divisor[4],
            -divisor[3],
            -divisor[2],
            -divisor[1],
            -divisor[0],
        ],
        [F::one(), F::zero(), F::zero(), F::zero(), F::zero()],
        [F::zero(), F::one(), F::zero(), F::zero(), F::zero()],
        [F::zero(), F::zero(), F::one(), F::zero(), F::zero()],
        [F::zero(), F::zero(), F::zero(), F::one(), F::zero()],
    ]
}

fn matvec<F: JoltField>(matrix: &[[F; 5]; 5], vector: [F; 5]) -> [F; 5] {
    matrix.map(|row| {
        row.iter()
            .zip(&vector)
            .fold(F::zero(), |acc, (&a, &b)| acc + a * b)
    })
}

#[expect(clippy::indexing_slicing, reason = "fixed-size five-by-five arrays")]
fn matmul<F: JoltField>(left: &[[F; 5]; 5], right: &[[F; 5]; 5]) -> [[F; 5]; 5] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..5).fold(F::zero(), |acc, k| acc + left[i][k] * right[k][j]))
    })
}

fn matrix_power<F: JoltField>(matrix: [[F; 5]; 5], mut exponent: usize) -> [[F; 5]; 5] {
    let mut result = [
        [F::one(), F::zero(), F::zero(), F::zero(), F::zero()],
        [F::zero(), F::one(), F::zero(), F::zero(), F::zero()],
        [F::zero(), F::zero(), F::one(), F::zero(), F::zero()],
        [F::zero(), F::zero(), F::zero(), F::one(), F::zero()],
        [F::zero(), F::zero(), F::zero(), F::zero(), F::one()],
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
    #![expect(
        clippy::indexing_slicing,
        clippy::unwrap_used,
        reason = "tests index fixture vectors and unwrap valid field parameters"
    )]

    use super::*;
    use jolt_field::{Fr, Ring};
    use num_traits::Zero;

    #[test]
    fn quintic_quotient_and_remainder() {
        let points = FoldPoints::new(Fr::from_u64(3), &mut NoopVerifierObserver).unwrap();
        let divisor = points.divisor(&mut NoopVerifierObserver);
        let expected = [
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
            Fr::from_u64(11),
        ];
        let remainder = [
            Fr::from_u64(13),
            Fr::from_u64(17),
            Fr::from_u64(19),
            Fr::from_u64(23),
            Fr::from_u64(29),
        ];
        let mut polynomial = vec![Fr::zero(); expected.len() + divisor.len() - 1];
        for (i, &a) in expected.iter().enumerate() {
            for (j, &b) in divisor.iter().enumerate() {
                polynomial[i + j] += a * b;
            }
        }
        for (coefficient, &value) in polynomial.iter_mut().zip(&remainder) {
            *coefficient += value;
        }

        assert_eq!(divide_by_monic_quintic(&polynomial, &divisor), expected);
        let values = points
            .points
            .map(|point| eval_univariate(&remainder, point));
        assert_eq!(
            points.interpolate(values, &mut NoopVerifierObserver),
            Some(remainder)
        );
        assert!(points
            .points
            .iter()
            .all(|&point| eval_univariate(&divisor, point).is_zero()));
    }

    #[test]
    fn quintic_quotient_spans_parallel_blocks() {
        let points = FoldPoints::new(Fr::from_u64(3), &mut NoopVerifierObserver).unwrap();
        let divisor = points.divisor(&mut NoopVerifierObserver);
        let expected: Vec<Fr> = (0..(3 << 12) + 3)
            .map(|i| Fr::from_u64(i as u64 * 7 + 3))
            .collect();
        let mut polynomial = vec![Fr::zero(); expected.len() + 5];
        for (i, &a) in expected.iter().enumerate() {
            for (j, &b) in divisor.iter().enumerate() {
                polynomial[i + j] += a * b;
            }
        }
        polynomial[0] += Fr::from_u64(13);
        polynomial[2] += Fr::from_u64(17);
        assert_eq!(divide_by_monic_quintic(&polynomial, &divisor), expected);
    }

    #[test]
    fn four_point_dft_recovers_residues() {
        let points = FoldPoints::new(Fr::from_u64(3), &mut NoopVerifierObserver).unwrap();
        assert_eq!(points.root.square(), -Fr::from_u64(1));
        let polynomial: Vec<_> = (1..=16).map(Fr::from_u64).collect();
        let values = std::array::from_fn(|j| eval_univariate(&polynomial, points.points[j]));
        let residues = points.residues(values, &mut NoopVerifierObserver);
        for (j, residue) in residues.into_iter().enumerate() {
            let coefficients: Vec<_> = polynomial.iter().skip(j).step_by(4).copied().collect();
            assert_eq!(residue, eval_univariate(&coefficients, points.points[4]));
        }
        for r in [
            Fr::zero(),
            Fr::from_u64(1),
            -Fr::from_u64(1),
            points.root,
            -points.root,
        ] {
            assert!(matches!(
                FoldPoints::new(r, &mut NoopVerifierObserver),
                Err(HyperKZGError::DegenerateChallenge)
            ));
        }
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
