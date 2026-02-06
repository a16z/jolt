use alloc::vec::Vec;

use crate::pippenger::{msm_pippenger, msm_pippenger_const};
use crate::traits::GlvCapable;

/// Expand (scalar, point) pairs into GLV-decomposed form.
/// For each (k, P), produces (k1, P') and (k2, φ(P')) where k ≡ k1 + k2·λ.
#[inline(always)]
fn expand_glv_into<G: GlvCapable>(
    scalars: &[G::FullScalar],
    points: &[G],
    out_scalars: &mut [G::HalfScalar],
    out_points: &mut [G],
) {
    debug_assert_eq!(scalars.len(), points.len());
    debug_assert!(out_scalars.len() >= 2 * scalars.len());
    debug_assert!(out_points.len() >= 2 * points.len());

    for (i, (scalar, point)) in scalars.iter().zip(points.iter()).enumerate() {
        let [(sign1, k1), (sign2, k2)] = G::decompose_scalar(scalar);

        let p1 = if sign1 { point.neg() } else { point.clone() };
        let p2_base = point.endomorphism();
        let p2 = if sign2 { p2_base.neg() } else { p2_base };

        let idx = i * 2;
        out_scalars[idx] = k1;
        out_scalars[idx + 1] = k2;
        out_points[idx] = p1;
        out_points[idx + 1] = p2;
    }
}

/// GLV-accelerated MSM: expands input via scalar decomposition, then runs Pippenger.
///
/// # Timing
/// Inherits Pippenger's variable-time behavior. See [`msm_pippenger`] for details.
#[inline(always)]
pub fn msm_glv<G>(scalars: &[G::FullScalar], points: &[G], window_bits: usize) -> G
where
    G: GlvCapable,
{
    assert_eq!(scalars.len(), points.len());
    let n = scalars.len();

    let mut expanded_scalars: Vec<G::HalfScalar> = Vec::with_capacity(2 * n);
    let mut expanded_points: Vec<G> = Vec::with_capacity(2 * n);
    expanded_scalars.resize_with(2 * n, Default::default);
    expanded_points.resize_with(2 * n, G::identity);

    expand_glv_into::<G>(scalars, points, &mut expanded_scalars, &mut expanded_points);

    msm_pippenger(&expanded_scalars, &expanded_points, window_bits)
}

/// GLV-accelerated MSM with compile-time window sizing.
#[inline(always)]
pub fn msm_glv_const<G, const WINDOW_BITS: usize>(scalars: &[G::FullScalar], points: &[G]) -> G
where
    G: GlvCapable,
{
    assert_eq!(scalars.len(), points.len());
    let n = scalars.len();

    let mut expanded_scalars: Vec<G::HalfScalar> = Vec::with_capacity(2 * n);
    let mut expanded_points: Vec<G> = Vec::with_capacity(2 * n);
    expanded_scalars.resize_with(2 * n, Default::default);
    expanded_points.resize_with(2 * n, G::identity);

    expand_glv_into::<G>(scalars, points, &mut expanded_scalars, &mut expanded_points);

    msm_pippenger_const::<G, G::HalfScalar, WINDOW_BITS>(&expanded_scalars, &expanded_points)
}

/// GLV-accelerated MSM using caller-provided scratch buffers (no heap allocation).
#[inline(always)]
pub fn msm_glv_with_scratch_const<G, const WINDOW_BITS: usize>(
    scalars: &[G::FullScalar],
    points: &[G],
    expanded_scalars: &mut [G::HalfScalar],
    expanded_points: &mut [G],
) -> G
where
    G: GlvCapable,
{
    assert_eq!(scalars.len(), points.len());
    let n = scalars.len();
    assert!(
        expanded_scalars.len() >= 2 * n,
        "expanded_scalars buffer too small"
    );
    assert!(
        expanded_points.len() >= 2 * n,
        "expanded_points buffer too small"
    );

    expand_glv_into::<G>(scalars, points, expanded_scalars, expanded_points);

    msm_pippenger_const::<G, G::HalfScalar, WINDOW_BITS>(
        &expanded_scalars[..2 * n],
        &expanded_points[..2 * n],
    )
}
