//! Stage-1 norm sumcheck prover/verifier for the Akita PCS.
//!
//! The committed witness is a Boolean table
//! `w : {0,1}^{col_bits} x {0,1}^{ring_bits} -> {-half, ..., half-1}` with
//! `half = b/2`. Define the virtual table `S(z) = w(z) * (w(z) + 1)`. For an
//! honest witness every entry of `w` is a valid digit, so `S(z)` lies in the
//! set `{k(k+1) : k = 0, ..., half-1}`. The range-check polynomial
//!
//! `Q(s) = prod_{k=0}^{half-1} (s - k(k+1))`
//!
//! has degree `b/2` and vanishes on exactly that set. The sumcheck proves
//!
//! `0 = sum_z eq(tau0, z) * Q(S(z))`,
//!
//! where the input claim is `0` (an honest prover makes every summand vanish).
//! Stage 1 uses the generic eq-factored sumcheck path: each round writes the
//! full polynomial as `s(X) = l(X) * q(X)`, where `l` is the linear eq factor
//! for the current round and `q` has degree `b/2`. The proof sends the
//! headerless `q` message with its linear term omitted, rather than the full
//! degree-`b/2 + 1` product polynomial. After all rounds, at `stage1_point`, the
//! verifier checks
//!
//! `eq(tau0, stage1_point) * Q(s_claim)`
//!
//! where `s_claim = S(stage1_point) = w(stage1_point) * (w(stage1_point) + 1)` is the
//! carried virtual claim passed into stage 2.
//!
//! ## `b = 8` specialization
//!
//! With `half = 4` the roots are `{0, 2, 6, 12}`, giving
//!
//! `Q(s) = s * (s - 2) * (s - 6) * (s - 12)`,
//!
//! degree 4, so round polynomials have degree 5.

use super::fold_full_prefix_pair;
use super::two_round_prefix::{
    build_stage1_bivariate_skip_proof_from_s_compact, can_use_stage1_two_round_prefix,
    stage1_b4_s_digit_from_compact_s, stage1_b8_s_digit_from_compact_s, Stage1BivariateSkipState,
};
use akita_algebra::split_eq::GruenSplitEq;
use akita_field::parallel::*;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, Zero};
use akita_sumcheck::{
    fold_evals_in_place, CompactPairFoldLut, EqFactoredSumcheckInstanceProver, EqFactoredUniPoly,
};
use std::time::Instant;

const MAX_AFFINE_COEFFS: usize = 17;
const MAX_COMPACT_COEFF_LUT_B: usize = 16;
const MAX_FIELD_COEFF_LUT_B: usize = 32;

#[derive(Clone, Copy, Debug, Default)]
struct CompactCoeffEntry {
    abs_coeff: u64,
    is_neg: bool,
}

fn poly_coeffs_from_roots_int(roots: &[i128]) -> Vec<i128> {
    let mut coeffs = vec![1i128];
    for &root in roots {
        let mut next = vec![0i128; coeffs.len() + 1];
        for (idx, &coeff) in coeffs.iter().enumerate() {
            next[idx] -= coeff * root;
            next[idx + 1] += coeff;
        }
        coeffs = next;
    }
    coeffs
}

#[derive(Clone)]
struct RangeAffineFromSPrecomp<E: FieldCore> {
    dense_coeffs: Vec<E>,
    dense_row_offsets: Vec<usize>,
    degree_q: usize,
    /// `h_i(s_0)` for each valid `s_0` and coefficient index `i`.
    /// Indexed as `compact_idx * num_rows + i`, where `compact_idx` is
    /// obtained from `s_to_compact`.
    small_s_lut: Vec<E>,
    compact_coeff_lut: Option<Vec<CompactCoeffEntry>>,
    field_coeff_lut: Option<Vec<E>>,
    /// Maps raw `s` integer (offset by `min_s`) to a compact index into the
    /// `b/2`-element valid-value set `{k(k+1) : k = 0..half-1}`.
    s_to_compact: Vec<u8>,
    num_valid_s: usize,
    min_s: i16,
}

impl<E: FieldCore + FromPrimitiveInt> RangeAffineFromSPrecomp<E> {
    fn new(b: usize) -> Self {
        assert!(b >= 2, "b must be at least 2");
        let half = (b / 2) as i128;
        let pair_offsets: Vec<i128> = (0..half).map(|k| k * (k + 1)).collect();
        let range_coeffs = poly_coeffs_from_roots_int(&pair_offsets);
        let degree_q = range_coeffs.len() - 1;
        let num_rows = degree_q + 1;

        let total_elems = num_rows * (num_rows + 1) / 2;
        let mut dense_int = Vec::with_capacity(total_elems);
        let mut dense_row_offsets = Vec::with_capacity(num_rows + 1);

        for i in 0..num_rows {
            dense_row_offsets.push(dense_int.len());
            let row_len = degree_q - i + 1;
            let mut binom: i128 = 1;
            for k in 0..row_len {
                let m = i + k;
                let coeff = range_coeffs[m] * binom;
                dense_int.push(coeff);
                if k + 1 < row_len {
                    binom = binom * (m as i128 + 1) / (k as i128 + 1);
                }
            }
        }
        dense_row_offsets.push(dense_int.len());
        let dense_coeffs = dense_int.iter().copied().map(E::from_i128).collect();

        let min_s = 0i16;
        let max_s_i128 = half * (half - 1);
        assert!(
            max_s_i128 <= i16::MAX as i128,
            "compact s range exceeds i16 for b={b}"
        );
        let max_s = max_s_i128 as i16;
        let raw_range = (i32::from(max_s) - i32::from(min_s) + 1) as usize;
        let num_valid_s = half as usize;

        let mut s_to_compact = vec![u8::MAX; raw_range];
        for (compact_idx, &s_val) in pair_offsets.iter().enumerate() {
            s_to_compact[(s_val as i16 - min_s) as usize] = compact_idx as u8;
        }

        let mut small_s_lut = vec![E::zero(); num_valid_s * num_rows];
        let mut small_s_lut_int = vec![0i128; num_valid_s * num_rows];
        for (compact_idx, &s_val) in pair_offsets.iter().enumerate() {
            for i in 0..num_rows {
                let row = &dense_int[dense_row_offsets[i]..dense_row_offsets[i + 1]];
                let mut h: i128 = 0;
                for &c in row.iter().rev() {
                    h = h * s_val + c;
                }
                small_s_lut_int[compact_idx * num_rows + i] = h;
                small_s_lut[compact_idx * num_rows + i] = E::from_i128(h);
            }
        }

        let compact_coeff_lut = if b <= MAX_COMPACT_COEFF_LUT_B {
            let mut lut = Vec::with_capacity(num_valid_s * num_valid_s * num_rows);
            for (s0_ci, &s0_val) in pair_offsets.iter().enumerate() {
                let h_base = s0_ci * num_rows;
                for &s1_val in &pair_offsets {
                    let delta = s1_val - s0_val;
                    let mut delta_pow = 1i128;
                    for &h_i in &small_s_lut_int[h_base..h_base + num_rows] {
                        let coeff = h_i
                            .checked_mul(delta_pow)
                            .expect("compact affine coefficient overflow");
                        let abs_coeff = coeff.unsigned_abs();
                        assert!(
                            abs_coeff <= u64::MAX as u128,
                            "compact affine coefficient exceeds u64"
                        );
                        lut.push(CompactCoeffEntry {
                            abs_coeff: abs_coeff as u64,
                            is_neg: coeff < 0,
                        });
                        delta_pow = delta_pow
                            .checked_mul(delta)
                            .expect("compact affine power overflow");
                    }
                }
            }
            Some(lut)
        } else {
            None
        };
        let field_coeff_lut = if b > MAX_COMPACT_COEFF_LUT_B && b <= MAX_FIELD_COEFF_LUT_B {
            let mut lut = Vec::with_capacity(num_valid_s * num_valid_s * num_rows);
            for (s0_ci, &s0_val) in pair_offsets.iter().enumerate() {
                let h_base = s0_ci * num_rows;
                for &s1_val in &pair_offsets {
                    let delta = E::from_i128(s1_val - s0_val);
                    let mut delta_pow = E::one();
                    for &h_i in &small_s_lut[h_base..h_base + num_rows] {
                        lut.push(h_i * delta_pow);
                        delta_pow *= delta;
                    }
                }
            }
            Some(lut)
        } else {
            None
        };

        Self {
            dense_coeffs,
            dense_row_offsets,
            degree_q,
            small_s_lut,
            compact_coeff_lut,
            field_coeff_lut,
            s_to_compact,
            num_valid_s,
            min_s,
        }
    }
}

impl<E: FieldCore> RangeAffineFromSPrecomp<E> {
    #[inline]
    fn compact_index(&self, s_int: i16) -> usize {
        let raw = (s_int - self.min_s) as usize;
        debug_assert!(raw < self.s_to_compact.len());
        let ci = self.s_to_compact[raw];
        debug_assert_ne!(ci, u8::MAX, "s={s_int} is not a valid w*(w+1) value");
        ci as usize
    }

    fn num_rows(&self) -> usize {
        self.degree_q + 1
    }

    #[inline]
    fn dense_row(&self, i: usize) -> &[E] {
        &self.dense_coeffs[self.dense_row_offsets[i]..self.dense_row_offsets[i + 1]]
    }

    #[inline]
    fn h_i_lut(&self, s_0_int: i16, i: usize) -> E {
        let ci = self.compact_index(s_0_int);
        self.small_s_lut[ci * self.num_rows() + i]
    }

    #[inline]
    fn pair_coeff_lut_start(&self, s_0_int: i16, s_1_int: i16) -> usize {
        let pair_idx = self.compact_index(s_0_int) * self.num_valid_s + self.compact_index(s_1_int);
        pair_idx * self.num_rows()
    }

    #[inline]
    fn compact_coeffs_lut(&self, s_0_int: i16, s_1_int: i16) -> Option<&[CompactCoeffEntry]> {
        let lut = self.compact_coeff_lut.as_ref()?;
        let num_rows = self.num_rows();
        let start = self.pair_coeff_lut_start(s_0_int, s_1_int);
        Some(&lut[start..start + num_rows])
    }

    #[inline]
    fn field_coeffs_lut(&self, s_0_int: i16, s_1_int: i16) -> Option<&[E]> {
        let lut = self.field_coeff_lut.as_ref()?;
        let num_rows = self.num_rows();
        let start = self.pair_coeff_lut_start(s_0_int, s_1_int);
        Some(&lut[start..start + num_rows])
    }
}

#[inline]
fn accumulate_compact_coeff_slot<E: FieldCore + HasUnreducedOps>(
    pos_accum: &mut [E::MulU64Accum],
    neg_accum: &mut [E::MulU64Accum],
    slot: usize,
    e_in: E,
    coeff: &CompactCoeffEntry,
) {
    if coeff.abs_coeff == 0 {
        return;
    }
    let prod = e_in.mul_u64_unreduced(coeff.abs_coeff);
    if coeff.is_neg {
        neg_accum[slot] += prod;
    } else {
        pos_accum[slot] += prod;
    }
}

#[inline]
fn accumulate_compact_coeffs<E: FieldCore + HasUnreducedOps>(
    pos_accum: &mut [E::MulU64Accum],
    neg_accum: &mut [E::MulU64Accum],
    e_in: E,
    coeffs: &[CompactCoeffEntry],
) {
    debug_assert_eq!(pos_accum.len(), neg_accum.len());
    debug_assert!(pos_accum.len() >= coeffs.len());

    for (idx, coeff) in coeffs.iter().enumerate().take(pos_accum.len()) {
        accumulate_compact_coeff_slot(pos_accum, neg_accum, idx, e_in, coeff);
    }
}

#[inline]
fn reduce_small_coeff_accum<E: FieldCore + HasUnreducedOps>(
    pos: E::MulU64Accum,
    neg: E::MulU64Accum,
) -> E {
    E::reduce_mul_u64_accum(pos) - E::reduce_mul_u64_accum(neg)
}

#[inline]
fn accumulate_dense_entry_coeffs<E: FieldCore + HasUnreducedOps>(
    accum: &mut [E::ProductAccum],
    entry_coeffs: &[E],
    e_in: E,
) {
    if accum.is_empty() {
        return;
    }

    for (acc, &entry) in accum.iter_mut().zip(entry_coeffs.iter()) {
        *acc += e_in.mul_to_product_accum(entry);
    }
}

#[inline]
fn compute_entry_coeffs_from_s<E: FieldCore + HasUnreducedOps>(
    out: &mut [E],
    _s_pows: &mut [E],
    precomp: &RangeAffineFromSPrecomp<E>,
    s_0: E,
    a: E,
) {
    let num_rows = precomp.num_rows();
    debug_assert!(out.len() >= num_rows);

    let mut a_pow = E::one();
    for (i, out_i) in out.iter_mut().enumerate().take(num_rows) {
        let mut h_i = E::zero();
        for &coeff in precomp.dense_row(i).iter().rev() {
            h_i = h_i * s_0 + coeff;
        }
        *out_i = a_pow * h_i;
        a_pow *= a;
    }
}

#[inline]
fn compute_entry_coeffs_from_s_x4<E: FieldCore + HasUnreducedOps>(
    out: &mut [[E; MAX_AFFINE_COEFFS]; 4],
    precomp: &RangeAffineFromSPrecomp<E>,
    s_0: [E; 4],
    a: [E; 4],
) {
    let num_rows = precomp.num_rows();

    let mut ap = [E::one(); 4];
    let [out0, out1, out2, out3] = out.each_mut();
    for (i, (((out0_i, out1_i), out2_i), out3_i)) in out0
        .iter_mut()
        .zip(out1.iter_mut())
        .zip(out2.iter_mut())
        .zip(out3.iter_mut())
        .take(num_rows)
        .enumerate()
    {
        let mut h0 = E::zero();
        let mut h1 = E::zero();
        let mut h2 = E::zero();
        let mut h3 = E::zero();
        for &coeff in precomp.dense_row(i).iter().rev() {
            h0 = h0 * s_0[0] + coeff;
            h1 = h1 * s_0[1] + coeff;
            h2 = h2 * s_0[2] + coeff;
            h3 = h3 * s_0[3] + coeff;
        }

        *out0_i = ap[0] * h0;
        *out1_i = ap[1] * h1;
        *out2_i = ap[2] * h2;
        *out3_i = ap[3] * h3;

        ap[0] *= a[0];
        ap[1] *= a[1];
        ap[2] *= a[2];
        ap[3] *= a[3];
    }
}

fn compute_norm_round_eq_poly_from_s<E: FieldCore + FromPrimitiveInt + HasUnreducedOps>(
    split_eq: &GruenSplitEq<E>,
    range_precomp: &RangeAffineFromSPrecomp<E>,
    s_pair: impl Fn(usize) -> (E, E) + Sync,
) -> EqFactoredUniPoly<E> {
    let (e_first, e_second) = split_eq.remaining_eq_tables();
    let num_first = e_first.len();
    let rp = range_precomp;
    let full_num_coeffs_q = rp.degree_q + 1;
    let num_coeffs_q = full_num_coeffs_q;

    let q_coeffs = cfg_fold_reduce!(
        0..e_second.len(),
        || vec![E::ProductAccum::zero(); num_coeffs_q],
        |mut outer_accum, j_high| {
            debug_assert!(full_num_coeffs_q <= MAX_AFFINE_COEFFS);
            let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];
            let base_j = j_high * num_first;
            let full_chunks = e_first.len() / 4;
            let mut batch_out = [[E::zero(); MAX_AFFINE_COEFFS]; 4];

            for chunk in 0..full_chunks {
                let jl = chunk * 4;
                let pairs = [
                    s_pair(base_j + jl),
                    s_pair(base_j + jl + 1),
                    s_pair(base_j + jl + 2),
                    s_pair(base_j + jl + 3),
                ];
                compute_entry_coeffs_from_s_x4(
                    &mut batch_out,
                    rp,
                    [pairs[0].0, pairs[1].0, pairs[2].0, pairs[3].0],
                    [
                        pairs[0].1 - pairs[0].0,
                        pairs[1].1 - pairs[1].0,
                        pairs[2].1 - pairs[2].0,
                        pairs[3].1 - pairs[3].0,
                    ],
                );
                for (b_idx, bo) in batch_out.iter().enumerate() {
                    let e_in = e_first[jl + b_idx];
                    accumulate_dense_entry_coeffs(
                        &mut inner_accum[..num_coeffs_q],
                        &bo[..full_num_coeffs_q],
                        e_in,
                    );
                }
            }

            let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
            let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];
            for (tail_idx, &e_in) in e_first[full_chunks * 4..].iter().enumerate() {
                let j = base_j + full_chunks * 4 + tail_idx;
                let (s_0, s_1) = s_pair(j);
                compute_entry_coeffs_from_s(&mut entry_buf, &mut s_pows_buf, rp, s_0, s_1 - s_0);
                accumulate_dense_entry_coeffs(
                    &mut inner_accum[..num_coeffs_q],
                    &entry_buf[..full_num_coeffs_q],
                    e_in,
                );
            }

            let e_out = e_second[j_high];
            for k in 0..num_coeffs_q {
                let inner_reduced = E::reduce_product_accum(inner_accum[k]);
                outer_accum[k] += e_out.mul_to_product_accum(inner_reduced);
            }
            outer_accum
        },
        |mut a, b_vec| {
            for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                *ai += *bi;
            }
            a
        }
    )
    .into_iter()
    .map(E::reduce_product_accum)
    .collect::<Vec<_>>();

    let _ = split_eq;
    EqFactoredUniPoly::from_q_coeffs(q_coeffs)
}

fn compute_norm_round_eq_poly_from_s_compact_with_pairs<
    E: FieldCore + FromPrimitiveInt + HasUnreducedOps,
>(
    split_eq: &GruenSplitEq<E>,
    range_precomp: &RangeAffineFromSPrecomp<E>,
    s_pair: impl Fn(usize) -> (i16, i16) + Sync,
) -> EqFactoredUniPoly<E> {
    let (e_first, e_second) = split_eq.remaining_eq_tables();
    let num_first = e_first.len();

    let rp = range_precomp;
    let full_num_coeffs_q = rp.degree_q + 1;
    let num_coeffs_q = full_num_coeffs_q;

    let q_coeffs = if rp.compact_coeffs_lut(0, 0).is_some() {
        cfg_fold_reduce!(
            0..e_second.len(),
            || vec![E::ProductAccum::zero(); num_coeffs_q],
            |mut outer_accum, j_high| {
                debug_assert!(full_num_coeffs_q <= MAX_AFFINE_COEFFS);
                let mut inner_pos = [E::MulU64Accum::zero(); MAX_AFFINE_COEFFS];
                let mut inner_neg = [E::MulU64Accum::zero(); MAX_AFFINE_COEFFS];
                for (j_low, &e_in) in e_first.iter().enumerate() {
                    let j = j_high * num_first + j_low;
                    let (s_0_int, s_1_int) = s_pair(j);
                    let coeffs = rp
                        .compact_coeffs_lut(s_0_int, s_1_int)
                        .expect("missing compact coefficient LUT");
                    accumulate_compact_coeffs(
                        &mut inner_pos[..num_coeffs_q],
                        &mut inner_neg[..num_coeffs_q],
                        e_in,
                        coeffs,
                    );
                }
                let e_out = e_second[j_high];
                for k in 0..num_coeffs_q {
                    let inner_reduced = reduce_small_coeff_accum(inner_pos[k], inner_neg[k]);
                    outer_accum[k] += e_out.mul_to_product_accum(inner_reduced);
                }
                outer_accum
            },
            |mut a, b_vec| {
                for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                    *ai += *bi;
                }
                a
            }
        )
        .into_iter()
        .map(E::reduce_product_accum)
        .collect::<Vec<_>>()
    } else if rp.field_coeffs_lut(0, 0).is_some() {
        cfg_fold_reduce!(
            0..e_second.len(),
            || vec![E::ProductAccum::zero(); num_coeffs_q],
            |mut outer_accum, j_high| {
                debug_assert!(full_num_coeffs_q <= MAX_AFFINE_COEFFS);
                let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];
                for (j_low, &e_in) in e_first.iter().enumerate() {
                    let j = j_high * num_first + j_low;
                    let (s_0_int, s_1_int) = s_pair(j);
                    let coeffs = rp
                        .field_coeffs_lut(s_0_int, s_1_int)
                        .expect("missing field coefficient LUT");
                    accumulate_dense_entry_coeffs(&mut inner_accum[..num_coeffs_q], coeffs, e_in);
                }
                let e_out = e_second[j_high];
                for k in 0..num_coeffs_q {
                    let inner_reduced = E::reduce_product_accum(inner_accum[k]);
                    outer_accum[k] += e_out.mul_to_product_accum(inner_reduced);
                }
                outer_accum
            },
            |mut a, b_vec| {
                for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                    *ai += *bi;
                }
                a
            }
        )
        .into_iter()
        .map(E::reduce_product_accum)
        .collect::<Vec<_>>()
    } else {
        cfg_fold_reduce!(
            0..e_second.len(),
            || vec![E::ProductAccum::zero(); num_coeffs_q],
            |mut outer_accum, j_high| {
                debug_assert!(full_num_coeffs_q <= MAX_AFFINE_COEFFS);
                let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];
                for (j_low, &e_in) in e_first.iter().enumerate() {
                    let j = j_high * num_first + j_low;
                    let (s_0_int, s_1_int) = s_pair(j);
                    let s_1 = E::from_i64(i64::from(s_1_int));
                    let a = s_1 - E::from_i64(i64::from(s_0_int));
                    let mut a_pow = E::one();
                    for (coeff_idx, coeff_accum) in
                        inner_accum.iter_mut().take(full_num_coeffs_q).enumerate()
                    {
                        let h_i_s0 = rp.h_i_lut(s_0_int, coeff_idx);
                        let val = a_pow * h_i_s0;
                        *coeff_accum += e_in.mul_to_product_accum(val);
                        a_pow *= a;
                    }
                }
                let e_out = e_second[j_high];
                for k in 0..num_coeffs_q {
                    let inner_reduced = E::reduce_product_accum(inner_accum[k]);
                    outer_accum[k] += e_out.mul_to_product_accum(inner_reduced);
                }
                outer_accum
            },
            |mut a, b_vec| {
                for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                    *ai += *bi;
                }
                a
            }
        )
        .into_iter()
        .map(E::reduce_product_accum)
        .collect::<Vec<_>>()
    };

    let _ = split_eq;
    EqFactoredUniPoly::from_q_coeffs(q_coeffs)
}

fn compute_norm_round_eq_poly_from_s_compact<E: FieldCore + FromPrimitiveInt + HasUnreducedOps>(
    split_eq: &GruenSplitEq<E>,
    s_compact: &[i16],
    range_precomp: &RangeAffineFromSPrecomp<E>,
) -> EqFactoredUniPoly<E> {
    compute_norm_round_eq_poly_from_s_compact_with_pairs(split_eq, range_precomp, |j| {
        (s_compact[2 * j], s_compact[2 * j + 1])
    })
}

enum STable<E: FieldCore> {
    Compact(Vec<i16>),
    Full(Vec<E>),
}

#[inline]
fn compact_s_from_w(w: i8) -> i16 {
    let w = i32::from(w);
    let s = w * (w + 1);
    debug_assert!(s >= 0);
    s as i16
}

fn build_compact_s_table(w_evals_compact: &[i8]) -> Vec<i16> {
    w_evals_compact
        .iter()
        .copied()
        .map(compact_s_from_w)
        .collect()
}

struct Stage1TwoRoundPrefix<E: FieldCore> {
    skip_state: Stage1BivariateSkipState<E>,
    first_challenge: Option<E>,
}

/// Stage-1 norm sumcheck prover over the virtual table `S(x) = w(x)(w(x)+1)`.
pub struct AkitaStage1Prover<E: FieldCore> {
    s_table: STable<E>,
    split_eq: GruenSplitEq<E>,
    range_precomp: RangeAffineFromSPrecomp<E>,
    live_x_cols: usize,
    col_bits: usize,
    num_vars: usize,
    b: usize,
    prefix_tau: Option<Vec<E>>,
    two_round_prefix: Option<Stage1TwoRoundPrefix<E>>,
    cached_round_poly: Option<EqFactoredUniPoly<E>>,
    prefix_time_total: f64,
    dense_time_total: f64,
    fold_time_total: f64,
    rounds_completed: usize,
}

mod lifecycle;
mod round2_prefix;
mod round_flow;
mod sparse_y;
mod x_prefix;

#[cfg(all(test, not(feature = "zk")))]
mod tests;

#[cfg(all(test, not(feature = "zk")))]
pub(crate) use round_flow::{advance_stage1_claim, pad_compact_witness};
