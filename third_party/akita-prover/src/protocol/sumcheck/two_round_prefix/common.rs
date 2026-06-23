use akita_field::unreduced::HasUnreducedOps;
use akita_field::{FieldCore, FromPrimitiveInt};
use akita_sumcheck::{EqFactoredUniPoly, UniPoly};
#[cfg(all(test, not(feature = "zk")))]
use akita_types::range_check_eval_from_s;

#[cfg(all(test, not(feature = "zk")))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrefixPoint<E: FieldCore> {
    Finite(E),
    Infinity,
}

/// Number of cached evaluations in the stage-1 `b = 4` two-round-prefix grid
/// after omitting the four Boolean corners from `{0,1,Infinity}^2`.
pub(crate) const STAGE1_B4_PREFIX_EVAL_COUNT: usize = 5;

pub(crate) const STAGE1_B4_NONBOOLEAN_GRID_INDICES: [usize; STAGE1_B4_PREFIX_EVAL_COUNT] =
    [2, 5, 6, 7, 8];

/// Number of cached evaluations in the stage-1 `b = 8` two-round-prefix grid
/// after omitting the four Boolean corners from `{0,1,-1,2,Infinity}^2`.
pub(crate) const STAGE1_PREFIX_EVAL_COUNT: usize = 21;
pub(crate) const STAGE1_B8_Q_POLY_DEGREE: usize = 4;

pub(crate) const LOOKUP_PREFIX_INF: i64 = i64::MIN;
pub(crate) const STAGE1_B4_S_VALUES: [i64; 2] = [0, 2];
pub(crate) const STAGE1_B8_S_VALUES: [i64; 4] = [0, 2, 6, 12];
pub(crate) const STAGE2_B4_W_VALUES: [i64; 4] = [-2, -1, 0, 1];
pub(crate) const STAGE2_B8_W_VALUES: [i64; 8] = [-4, -3, -2, -1, 0, 1, 2, 3];
pub(crate) const STAGE2_PREFIX_POINT_COUNT: usize = 9;
pub(crate) const STAGE2_COMPRESSED_POINT_COUNT: usize = STAGE2_PREFIX_POINT_COUNT - 1;
pub(crate) const STAGE2_COMPRESSED_POINT_INDICES_BY_OMITTED_CORNER: [[usize;
    STAGE2_COMPRESSED_POINT_COUNT];
    4] = [
    [1, 2, 3, 4, 5, 6, 7, 8],
    [0, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 5, 6, 7, 8],
];

pub(crate) const fn lookup_bilinear_coeffs_from_quad(quad: [i64; 4]) -> [i64; 4] {
    let [t00, t10, t01, t11] = quad;
    [t00, t10 - t00, t01 - t00, t11 - t10 - t01 + t00]
}

pub(crate) const fn lookup_bilinear_eval_on_prefix_points(quad: [i64; 4], x: i64, y: i64) -> i64 {
    let [a, b, c, d] = lookup_bilinear_coeffs_from_quad(quad);
    let x_is_inf = x == LOOKUP_PREFIX_INF;
    let y_is_inf = y == LOOKUP_PREFIX_INF;
    if !x_is_inf && !y_is_inf {
        a + x * (b + y * d) + y * c
    } else if x_is_inf && !y_is_inf {
        b + y * d
    } else if !x_is_inf && y_is_inf {
        c + x * d
    } else {
        d
    }
}

pub(crate) const fn pow_i64(mut base: i64, mut exp: usize) -> i64 {
    let mut out = 1i64;
    while exp > 0 {
        if exp & 1 == 1 {
            out *= base;
        }
        exp >>= 1;
        if exp > 0 {
            base *= base;
        }
    }
    out
}

pub(crate) const fn stage1_b8_range_check_from_s(s: i64) -> i64 {
    s * (s - 2) * (s - 6) * (s - 12)
}

pub(crate) const fn stage1_b4_range_check_from_s(s: i64) -> i64 {
    s * (s - 2)
}

pub(crate) const fn stage1_b4_local_norm_raw_eval_i64(s_quad: [i64; 4], x: i64, y: i64) -> i64 {
    let [_, bx, cy, dxy] = lookup_bilinear_coeffs_from_quad(s_quad);
    let x_is_inf = x == LOOKUP_PREFIX_INF;
    let y_is_inf = y == LOOKUP_PREFIX_INF;
    if !x_is_inf && !y_is_inf {
        stage1_b4_range_check_from_s(lookup_bilinear_eval_on_prefix_points(s_quad, x, y))
    } else if x_is_inf && !y_is_inf {
        let linear = bx + y * dxy;
        linear * linear
    } else if !x_is_inf && y_is_inf {
        let linear = cy + x * dxy;
        linear * linear
    } else {
        dxy * dxy
    }
}

pub(crate) const fn stage1_b8_local_norm_raw_eval_i64(s_quad: [i64; 4], x: i64, y: i64) -> i64 {
    let [_, bx, cy, dxy] = lookup_bilinear_coeffs_from_quad(s_quad);
    let x_is_inf = x == LOOKUP_PREFIX_INF;
    let y_is_inf = y == LOOKUP_PREFIX_INF;
    if !x_is_inf && !y_is_inf {
        stage1_b8_range_check_from_s(lookup_bilinear_eval_on_prefix_points(s_quad, x, y))
    } else if x_is_inf && !y_is_inf {
        pow_i64(bx + y * dxy, 4)
    } else if !x_is_inf && y_is_inf {
        pow_i64(cy + x * dxy, 4)
    } else {
        pow_i64(dxy, 4)
    }
}

pub(crate) const STAGE1_B4_PREFIX_LOOKUP_POINTS_I64: [(i64, i64); STAGE1_B4_PREFIX_EVAL_COUNT] = [
    (0, LOOKUP_PREFIX_INF),
    (1, LOOKUP_PREFIX_INF),
    (LOOKUP_PREFIX_INF, 0),
    (LOOKUP_PREFIX_INF, 1),
    (LOOKUP_PREFIX_INF, LOOKUP_PREFIX_INF),
];

pub(crate) const fn stage1_lookup_points_i64() -> [(i64, i64); STAGE1_PREFIX_EVAL_COUNT] {
    let coords = [0i64, 1, -1, 2, LOOKUP_PREFIX_INF];
    let mut out = [(0i64, 0i64); STAGE1_PREFIX_EVAL_COUNT];
    let mut out_idx = 0usize;
    let mut x_idx = 0usize;
    while x_idx < 5 {
        let mut y_idx = 0usize;
        while y_idx < 5 {
            if !(x_idx < 2 && y_idx < 2) {
                out[out_idx] = (coords[x_idx], coords[y_idx]);
                out_idx += 1;
            }
            y_idx += 1;
        }
        x_idx += 1;
    }
    out
}

pub(crate) const STAGE1_PREFIX_LOOKUP_POINTS_I64: [(i64, i64); STAGE1_PREFIX_EVAL_COUNT] =
    stage1_lookup_points_i64();

#[inline(always)]
pub(crate) const fn stage1_b4_lookup_index_from_digits(digits: [usize; 4]) -> usize {
    digits[0] | (digits[1] << 1) | (digits[2] << 2) | (digits[3] << 3)
}

#[inline(always)]
pub(crate) const fn stage1_b8_lookup_index_from_digits(digits: [usize; 4]) -> usize {
    digits[0] | (digits[1] << 2) | (digits[2] << 4) | (digits[3] << 6)
}

pub(crate) const fn build_stage1_b4_prefix_lookup_table() -> [[i64; STAGE1_B4_PREFIX_EVAL_COUNT]; 16]
{
    let mut table = [[0i64; STAGE1_B4_PREFIX_EVAL_COUNT]; 16];
    let mut d0 = 0usize;
    while d0 < 2 {
        let mut d1 = 0usize;
        while d1 < 2 {
            let mut d2 = 0usize;
            while d2 < 2 {
                let mut d3 = 0usize;
                while d3 < 2 {
                    let quad = [
                        STAGE1_B4_S_VALUES[d0],
                        STAGE1_B4_S_VALUES[d1],
                        STAGE1_B4_S_VALUES[d2],
                        STAGE1_B4_S_VALUES[d3],
                    ];
                    let table_idx = stage1_b4_lookup_index_from_digits([d0, d1, d2, d3]);
                    let mut point_idx = 0usize;
                    while point_idx < STAGE1_B4_PREFIX_EVAL_COUNT {
                        let (x, y) = STAGE1_B4_PREFIX_LOOKUP_POINTS_I64[point_idx];
                        table[table_idx][point_idx] = stage1_b4_local_norm_raw_eval_i64(quad, x, y);
                        point_idx += 1;
                    }
                    d3 += 1;
                }
                d2 += 1;
            }
            d1 += 1;
        }
        d0 += 1;
    }
    table
}

pub(crate) static STAGE1_B4_PREFIX_LOOKUP_TABLE: [[i64; STAGE1_B4_PREFIX_EVAL_COUNT]; 16] =
    build_stage1_b4_prefix_lookup_table();

pub(crate) const fn build_stage1_b8_prefix_lookup_table() -> [[i64; STAGE1_PREFIX_EVAL_COUNT]; 256]
{
    let mut table = [[0i64; STAGE1_PREFIX_EVAL_COUNT]; 256];
    let mut d0 = 0usize;
    while d0 < 4 {
        let mut d1 = 0usize;
        while d1 < 4 {
            let mut d2 = 0usize;
            while d2 < 4 {
                let mut d3 = 0usize;
                while d3 < 4 {
                    let quad = [
                        STAGE1_B8_S_VALUES[d0],
                        STAGE1_B8_S_VALUES[d1],
                        STAGE1_B8_S_VALUES[d2],
                        STAGE1_B8_S_VALUES[d3],
                    ];
                    let table_idx = stage1_b8_lookup_index_from_digits([d0, d1, d2, d3]);
                    let mut point_idx = 0usize;
                    while point_idx < STAGE1_PREFIX_EVAL_COUNT {
                        let (x, y) = STAGE1_PREFIX_LOOKUP_POINTS_I64[point_idx];
                        table[table_idx][point_idx] = stage1_b8_local_norm_raw_eval_i64(quad, x, y);
                        point_idx += 1;
                    }
                    d3 += 1;
                }
                d2 += 1;
            }
            d1 += 1;
        }
        d0 += 1;
    }
    table
}

pub(crate) static STAGE1_B8_PREFIX_LOOKUP_TABLE: [[i64; STAGE1_PREFIX_EVAL_COUNT]; 256] =
    build_stage1_b8_prefix_lookup_table();

pub(crate) const STAGE2_PREFIX_LOOKUP_POINTS_I64: [(i64, i64); STAGE2_PREFIX_POINT_COUNT] = [
    (0, 0),
    (0, 1),
    (0, LOOKUP_PREFIX_INF),
    (1, 0),
    (1, 1),
    (1, LOOKUP_PREFIX_INF),
    (LOOKUP_PREFIX_INF, 0),
    (LOOKUP_PREFIX_INF, 1),
    (LOOKUP_PREFIX_INF, LOOKUP_PREFIX_INF),
];

#[inline(always)]
pub(crate) const fn stage2_b4_lookup_index_from_digits(digits: [usize; 4]) -> usize {
    digits[0] | (digits[1] << 2) | (digits[2] << 4) | (digits[3] << 6)
}

#[inline(always)]
pub(crate) const fn stage2_b8_lookup_index_from_digits(digits: [usize; 4]) -> usize {
    digits[0] | (digits[1] << 3) | (digits[2] << 6) | (digits[3] << 9)
}

pub(crate) const fn stage2_local_norm_raw_eval_i64(w_quad: [i64; 4], x: i64, y: i64) -> i64 {
    let w_eval = lookup_bilinear_eval_on_prefix_points(w_quad, x, y);
    if x == LOOKUP_PREFIX_INF || y == LOOKUP_PREFIX_INF {
        w_eval * w_eval
    } else {
        w_eval * (w_eval + 1)
    }
}

pub(crate) const fn compress_stage2_lookup_values(
    values: [i64; STAGE2_PREFIX_POINT_COUNT],
    omitted_idx: usize,
) -> [i64; STAGE2_COMPRESSED_POINT_COUNT] {
    let mut out = [0i64; STAGE2_COMPRESSED_POINT_COUNT];
    let mut src_idx = 0usize;
    let mut dst_idx = 0usize;
    while src_idx < STAGE2_PREFIX_POINT_COUNT {
        if src_idx != omitted_idx {
            out[dst_idx] = values[src_idx];
            dst_idx += 1;
        }
        src_idx += 1;
    }
    out
}

pub(crate) const fn build_stage2_b4_norm_lookup_table() -> [[i64; STAGE2_PREFIX_POINT_COUNT]; 256] {
    let mut table = [[0i64; STAGE2_PREFIX_POINT_COUNT]; 256];
    let mut d0 = 0usize;
    while d0 < 4 {
        let mut d1 = 0usize;
        while d1 < 4 {
            let mut d2 = 0usize;
            while d2 < 4 {
                let mut d3 = 0usize;
                while d3 < 4 {
                    let quad = [
                        STAGE2_B4_W_VALUES[d0],
                        STAGE2_B4_W_VALUES[d1],
                        STAGE2_B4_W_VALUES[d2],
                        STAGE2_B4_W_VALUES[d3],
                    ];
                    let table_idx = stage2_b4_lookup_index_from_digits([d0, d1, d2, d3]);
                    let mut point_idx = 0usize;
                    while point_idx < STAGE2_PREFIX_POINT_COUNT {
                        let (x, y) = STAGE2_PREFIX_LOOKUP_POINTS_I64[point_idx];
                        table[table_idx][point_idx] = stage2_local_norm_raw_eval_i64(quad, x, y);
                        point_idx += 1;
                    }
                    d3 += 1;
                }
                d2 += 1;
            }
            d1 += 1;
        }
        d0 += 1;
    }
    table
}

pub(crate) static STAGE2_B4_NORM_LOOKUP_TABLE: [[i64; STAGE2_PREFIX_POINT_COUNT]; 256] =
    build_stage2_b4_norm_lookup_table();

pub(crate) const fn build_stage2_b4_relation_weight_table(
) -> [[i64; STAGE2_PREFIX_POINT_COUNT]; 256] {
    let mut table = [[0i64; STAGE2_PREFIX_POINT_COUNT]; 256];
    let mut d0 = 0usize;
    while d0 < 4 {
        let mut d1 = 0usize;
        while d1 < 4 {
            let mut d2 = 0usize;
            while d2 < 4 {
                let mut d3 = 0usize;
                while d3 < 4 {
                    let quad = [
                        STAGE2_B4_W_VALUES[d0],
                        STAGE2_B4_W_VALUES[d1],
                        STAGE2_B4_W_VALUES[d2],
                        STAGE2_B4_W_VALUES[d3],
                    ];
                    let table_idx = stage2_b4_lookup_index_from_digits([d0, d1, d2, d3]);
                    let mut point_idx = 0usize;
                    while point_idx < STAGE2_PREFIX_POINT_COUNT {
                        let (x, y) = STAGE2_PREFIX_LOOKUP_POINTS_I64[point_idx];
                        table[table_idx][point_idx] =
                            lookup_bilinear_eval_on_prefix_points(quad, x, y);
                        point_idx += 1;
                    }
                    d3 += 1;
                }
                d2 += 1;
            }
            d1 += 1;
        }
        d0 += 1;
    }
    table
}

pub(crate) static STAGE2_B4_RELATION_WEIGHT_TABLE: [[i64; STAGE2_PREFIX_POINT_COUNT]; 256] =
    build_stage2_b4_relation_weight_table();

pub(crate) const fn build_stage2_b4_relation_weight_compressed_table(
) -> [[i64; STAGE2_COMPRESSED_POINT_COUNT]; 256] {
    let mut table = [[0i64; STAGE2_COMPRESSED_POINT_COUNT]; 256];
    let mut table_idx = 0usize;
    while table_idx < 256 {
        table[table_idx] =
            compress_stage2_lookup_values(STAGE2_B4_RELATION_WEIGHT_TABLE[table_idx], 0);
        table_idx += 1;
    }
    table
}

pub(crate) static STAGE2_B4_RELATION_WEIGHT_COMPRESSED_TABLE: [[i64;
    STAGE2_COMPRESSED_POINT_COUNT];
    256] = build_stage2_b4_relation_weight_compressed_table();

pub(crate) const fn build_stage2_b8_norm_lookup_table() -> [[i64; STAGE2_PREFIX_POINT_COUNT]; 4096]
{
    let mut table = [[0i64; STAGE2_PREFIX_POINT_COUNT]; 4096];
    let mut d0 = 0usize;
    while d0 < 8 {
        let mut d1 = 0usize;
        while d1 < 8 {
            let mut d2 = 0usize;
            while d2 < 8 {
                let mut d3 = 0usize;
                while d3 < 8 {
                    let quad = [
                        STAGE2_B8_W_VALUES[d0],
                        STAGE2_B8_W_VALUES[d1],
                        STAGE2_B8_W_VALUES[d2],
                        STAGE2_B8_W_VALUES[d3],
                    ];
                    let table_idx = stage2_b8_lookup_index_from_digits([d0, d1, d2, d3]);
                    let mut point_idx = 0usize;
                    while point_idx < STAGE2_PREFIX_POINT_COUNT {
                        let (x, y) = STAGE2_PREFIX_LOOKUP_POINTS_I64[point_idx];
                        table[table_idx][point_idx] = stage2_local_norm_raw_eval_i64(quad, x, y);
                        point_idx += 1;
                    }
                    d3 += 1;
                }
                d2 += 1;
            }
            d1 += 1;
        }
        d0 += 1;
    }
    table
}

pub(crate) static STAGE2_B8_NORM_LOOKUP_TABLE: [[i64; STAGE2_PREFIX_POINT_COUNT]; 4096] =
    build_stage2_b8_norm_lookup_table();

pub(crate) const fn build_stage2_b8_relation_weight_table(
) -> [[i64; STAGE2_PREFIX_POINT_COUNT]; 4096] {
    let mut table = [[0i64; STAGE2_PREFIX_POINT_COUNT]; 4096];
    let mut d0 = 0usize;
    while d0 < 8 {
        let mut d1 = 0usize;
        while d1 < 8 {
            let mut d2 = 0usize;
            while d2 < 8 {
                let mut d3 = 0usize;
                while d3 < 8 {
                    let quad = [
                        STAGE2_B8_W_VALUES[d0],
                        STAGE2_B8_W_VALUES[d1],
                        STAGE2_B8_W_VALUES[d2],
                        STAGE2_B8_W_VALUES[d3],
                    ];
                    let table_idx = stage2_b8_lookup_index_from_digits([d0, d1, d2, d3]);
                    let mut point_idx = 0usize;
                    while point_idx < STAGE2_PREFIX_POINT_COUNT {
                        let (x, y) = STAGE2_PREFIX_LOOKUP_POINTS_I64[point_idx];
                        table[table_idx][point_idx] =
                            lookup_bilinear_eval_on_prefix_points(quad, x, y);
                        point_idx += 1;
                    }
                    d3 += 1;
                }
                d2 += 1;
            }
            d1 += 1;
        }
        d0 += 1;
    }
    table
}

pub(crate) static STAGE2_B8_RELATION_WEIGHT_TABLE: [[i64; STAGE2_PREFIX_POINT_COUNT]; 4096] =
    build_stage2_b8_relation_weight_table();

pub(crate) const fn build_stage2_b8_relation_weight_compressed_table(
) -> [[i64; STAGE2_COMPRESSED_POINT_COUNT]; 4096] {
    let mut table = [[0i64; STAGE2_COMPRESSED_POINT_COUNT]; 4096];
    let mut table_idx = 0usize;
    while table_idx < 4096 {
        table[table_idx] =
            compress_stage2_lookup_values(STAGE2_B8_RELATION_WEIGHT_TABLE[table_idx], 0);
        table_idx += 1;
    }
    table
}

pub(crate) static STAGE2_B8_RELATION_WEIGHT_COMPRESSED_TABLE: [[i64;
    STAGE2_COMPRESSED_POINT_COUNT];
    4096] = build_stage2_b8_relation_weight_compressed_table();

#[inline]
pub(crate) fn accum_lookup_vector_signed<E: FieldCore + HasUnreducedOps, const N: usize>(
    pos: &mut [E::MulU64Accum; N],
    neg: &mut [E::MulU64Accum; N],
    coeff: E,
    values: &[i64; N],
) {
    for (idx, &value) in values.iter().enumerate() {
        if value > 0 {
            pos[idx] += coeff.mul_u64_unreduced(value as u64);
        } else if value < 0 {
            neg[idx] += coeff.mul_u64_unreduced(value.unsigned_abs());
        }
    }
}

#[inline]
pub(crate) fn accum_lookup_vector_signed_selected<
    E: FieldCore + HasUnreducedOps,
    const N: usize,
    const M: usize,
>(
    pos: &mut [E::MulU64Accum; N],
    neg: &mut [E::MulU64Accum; N],
    coeff: E,
    values: &[i64; M],
    selected_indices: &[usize; N],
) {
    for (dst_idx, &src_idx) in selected_indices.iter().enumerate() {
        let value = values[src_idx];
        if value > 0 {
            pos[dst_idx] += coeff.mul_u64_unreduced(value as u64);
        } else if value < 0 {
            neg[dst_idx] += coeff.mul_u64_unreduced(value.unsigned_abs());
        }
    }
}

#[inline]
pub(crate) fn accum_pointwise_signed<E: FieldCore + HasUnreducedOps, const N: usize>(
    pos: &mut [E::MulU64Accum; N],
    neg: &mut [E::MulU64Accum; N],
    coeffs: &[E; N],
    weights: &[i64; N],
) {
    for (idx, (&coeff, &weight)) in coeffs.iter().zip(weights.iter()).enumerate() {
        if weight > 0 {
            pos[idx] += coeff.mul_u64_unreduced(weight as u64);
        } else if weight < 0 {
            neg[idx] += coeff.mul_u64_unreduced(weight.unsigned_abs());
        }
    }
}

#[inline(always)]
pub(crate) fn stage1_b4_s_digit_from_compact_s(s: i16) -> usize {
    match s {
        0 => 0,
        2 => 1,
        other => unreachable!("unexpected compact s value {other}"),
    }
}

#[inline(always)]
pub(crate) fn stage1_b8_s_digit_from_compact_s(s: i16) -> usize {
    match s {
        0 => 0,
        2 => 1,
        6 => 2,
        12 => 3,
        other => unreachable!("unexpected compact s value {other}"),
    }
}

#[inline(always)]
pub(crate) fn stage2_b4_w_digit(w: i8) -> usize {
    let w = i32::from(w);
    debug_assert!((-2..=1).contains(&w));
    (w + 2) as usize
}

#[inline(always)]
pub(crate) fn stage2_b8_w_digit(w: i8) -> usize {
    let w = i32::from(w);
    debug_assert!((-4..=3).contains(&w));
    (w + 4) as usize
}

#[inline]
pub(crate) fn linear_eq_eval<E: FieldCore>(tau: E, x: E) -> E {
    tau * x + (E::one() - tau) * (E::one() - x)
}

#[inline]
pub(crate) fn stage2_relation_m_point_values_compressed<E: FieldCore>(
    m_quad: [E; 4],
) -> [E; STAGE2_COMPRESSED_POINT_COUNT] {
    let m00 = m_quad[0];
    let m10 = m_quad[1];
    let m01 = m_quad[2];
    let m11 = m_quad[3];
    [
        m01,
        m01 - m00,
        m10,
        m11,
        m11 - m10,
        m10 - m00,
        m11 - m01,
        m11 - m10 - m01 + m00,
    ]
}

pub(crate) fn interpolate_eq_factored_q_poly<E: FieldCore + FromPrimitiveInt>(
    evals: &[E],
    degree: usize,
) -> EqFactoredUniPoly<E> {
    let mut q_coeffs = UniPoly::from_evals(evals).coeffs;
    q_coeffs.resize(degree + 1, E::zero());
    EqFactoredUniPoly::from_q_coeffs(q_coeffs)
}

/// Proposed reduced stage-2 domain `{1, Infinity}`.
#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn stage2_reduced_prefix_points<E: FieldCore + FromPrimitiveInt>() -> [PrefixPoint<E>; 2]
{
    [PrefixPoint::Finite(E::one()), PrefixPoint::Infinity]
}

/// Safe full stage-2 fallback domain `{0, 1, Infinity}`.
#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn stage2_full_prefix_points<E: FieldCore + FromPrimitiveInt>() -> [PrefixPoint<E>; 3] {
    [
        PrefixPoint::Finite(E::zero()),
        PrefixPoint::Finite(E::one()),
        PrefixPoint::Infinity,
    ]
}

/// Return the bilinear coefficients for a quad ordered as `[t00, t10, t01, t11]`.
#[inline]
#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn bilinear_coeffs_from_quad<E: FieldCore>(quad: [E; 4]) -> [E; 4] {
    let [t00, t10, t01, t11] = quad;
    [t00, t10 - t00, t01 - t00, t11 - t10 - t01 + t00]
}

/// Evaluate the bilinear multilinear extension of a quad at ordinary field
/// points `(x, y)`.
#[inline]
#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn bilinear_eval<E: FieldCore>(quad: [E; 4], x: E, y: E) -> E {
    let [a, b, c, d] = bilinear_coeffs_from_quad(quad);
    a + x * (b + y * d) + y * c
}

/// Evaluate a quad on a small domain where `Infinity` means "leading
/// coefficient in that coordinate".
#[inline]
#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn bilinear_eval_on_prefix_points<E: FieldCore>(
    quad: [E; 4],
    x: PrefixPoint<E>,
    y: PrefixPoint<E>,
) -> E {
    let [a, b, c, d] = bilinear_coeffs_from_quad(quad);
    match (x, y) {
        (PrefixPoint::Finite(x), PrefixPoint::Finite(y)) => a + x * (b + y * d) + y * c,
        (PrefixPoint::Infinity, PrefixPoint::Finite(y)) => b + y * d,
        (PrefixPoint::Finite(x), PrefixPoint::Infinity) => c + x * d,
        (PrefixPoint::Infinity, PrefixPoint::Infinity) => d,
    }
}

/// Evaluate the stage-1 candidate storage contribution used by the original
/// `{1, -1, 2, Infinity}^2` proposal.
#[inline]
#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn stage1_local_norm_eval<E: FieldCore + FromPrimitiveInt>(
    s_quad: [E; 4],
    x: PrefixPoint<E>,
    y: PrefixPoint<E>,
    b: usize,
) -> E {
    let s_eval = bilinear_eval_on_prefix_points(s_quad, x, y);
    range_check_eval_from_s(s_eval, b)
}

/// Evaluate the raw stage-1 full-domain polynomial on
/// `{0, 1, -1, 2, Infinity}^2`.
///
/// At `Infinity`, we take the leading coefficient in that coordinate of the
/// composed range-check polynomial `range_check(s(X, Y))`, rather than first
/// evaluating `s` at `Infinity` and then applying the range check.
#[inline]
#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn stage1_local_norm_raw_eval<E: FieldCore + FromPrimitiveInt>(
    s_quad: [E; 4],
    x: PrefixPoint<E>,
    y: PrefixPoint<E>,
    b: usize,
) -> E {
    let [_, bx, cy, dxy] = bilinear_coeffs_from_quad(s_quad);
    let degree = b / 2;
    let pow = |base: E| {
        let mut out = E::one();
        for _ in 0..degree {
            out *= base;
        }
        out
    };

    match (x, y) {
        (PrefixPoint::Finite(x), PrefixPoint::Finite(y)) => {
            range_check_eval_from_s(bilinear_eval(s_quad, x, y), b)
        }
        (PrefixPoint::Infinity, PrefixPoint::Finite(y)) => pow(bx + y * dxy),
        (PrefixPoint::Finite(x), PrefixPoint::Infinity) => pow(cy + x * dxy),
        (PrefixPoint::Infinity, PrefixPoint::Infinity) => pow(dxy),
    }
}

/// Evaluate the stage-2 local norm candidate used by the proposed reduced
/// `{1, Infinity}^2` storage: evaluate the bilinear witness first, then apply
/// `w (w + 1)`.
#[inline]
#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn stage2_local_norm_candidate_eval<E: FieldCore>(
    w_quad: [E; 4],
    x: PrefixPoint<E>,
    y: PrefixPoint<E>,
) -> E {
    let w_eval = bilinear_eval_on_prefix_points(w_quad, x, y);
    w_eval * (w_eval + E::one())
}

/// Evaluate the raw degree-`(2,2)` stage-2 norm polynomial on the safe full
/// `{0, 1, Infinity}^2` fallback domain.
///
/// At `Infinity`, we take the leading coefficient in that coordinate of
/// `w(X, Y) * (w(X, Y) + 1)`, so the linear `+w` term drops out.
#[inline]
#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn stage2_local_norm_raw_eval<E: FieldCore>(
    w_quad: [E; 4],
    x: PrefixPoint<E>,
    y: PrefixPoint<E>,
) -> E {
    let w_eval = bilinear_eval_on_prefix_points(w_quad, x, y);
    match (x, y) {
        (PrefixPoint::Finite(_), PrefixPoint::Finite(_)) => w_eval * (w_eval + E::one()),
        _ => w_eval * w_eval,
    }
}

/// Evaluate the stage-2 local relation contribution for one witness quad, one
/// local bilinear factor quad, and one fixed scalar factor.
#[inline]
#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn stage2_local_relation_eval<E: FieldCore>(
    w_quad: [E; 4],
    local_factor_quad: [E; 4],
    fixed_factor: E,
    x: PrefixPoint<E>,
    y: PrefixPoint<E>,
) -> E {
    fixed_factor
        * bilinear_eval_on_prefix_points(w_quad, x, y)
        * bilinear_eval_on_prefix_points(local_factor_quad, x, y)
}

/// Evaluate a quadratic from its values at `{0, 1, Infinity}`.
#[inline]
#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn eval_quadratic_from_01_inf<E: FieldCore>(
    at_zero: E,
    at_one: E,
    at_inf: E,
    x: PrefixPoint<E>,
) -> E {
    match x {
        PrefixPoint::Infinity => at_inf,
        PrefixPoint::Finite(x) => {
            let linear = at_one - at_zero - at_inf;
            at_zero + x * (linear + x * at_inf)
        }
    }
}

#[inline]
pub(crate) fn quadratic_coeffs_from_01_inf<E: FieldCore>(
    at_zero: E,
    at_one: E,
    at_inf: E,
) -> [E; 3] {
    [at_zero, at_one - at_zero - at_inf, at_inf]
}

#[inline]
pub(crate) fn eval_quadratic_from_coeffs<E: FieldCore>(coeffs: [E; 3], x: E) -> E {
    coeffs[0] + x * (coeffs[1] + x * coeffs[2])
}

#[inline]
pub(crate) fn linear_eq_coeffs<E: FieldCore>(tau: E) -> [E; 2] {
    [E::one() - tau, tau + tau - E::one()]
}

#[inline]
pub(crate) fn scale_quadratic_coeffs<E: FieldCore>(coeffs: [E; 3], scale: E) -> [E; 3] {
    [scale * coeffs[0], scale * coeffs[1], scale * coeffs[2]]
}

#[inline]
pub(crate) fn add_quadratic_coeffs<E: FieldCore>(lhs: [E; 3], rhs: [E; 3]) -> [E; 3] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

#[inline]
#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn coeff_array_to_poly<E: FieldCore, const N: usize>(coeffs: [E; N]) -> UniPoly<E> {
    UniPoly::from_coeffs(coeffs.to_vec())
}

#[inline]
pub(crate) fn mul_linear_by_quadratic_coeffs<E: FieldCore>(tau: E, quad: [E; 3]) -> [E; 4] {
    let [l0, l1] = linear_eq_coeffs(tau);
    [
        l0 * quad[0],
        l0 * quad[1] + l1 * quad[0],
        l0 * quad[2] + l1 * quad[1],
        l1 * quad[2],
    ]
}
