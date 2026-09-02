//! Host side of the BN254 G1 segment-sum device kernel (`jk_g1_seg_sum`).
//!
//! The kernel computes, per thread, the sum of selected affine bases as a
//! Jacobian point — the tier-1 group operation of the Dory one-hot witness
//! commit. This module owns the host↔device representation contract:
//!
//! - Bases are `ark_bn254::G1Affine` host memory viewed in place as a `uint`
//!   stream (stride [`G1_AFFINE_U32_STRIDE`](super::field) u32s; layout
//!   pinned by const asserts in [`super::field`] and the
//!   `g1_affine_layout_matches_u32_view` test). Bases must not contain the
//!   point at infinity — the shader never reads the `infinity` flag.
//! - Results are Jacobian `(X, Y, Z)` Montgomery limb triples; `Z = 0`
//!   encodes the identity. [`jac_from_device_limbs`] rebuilds the arkworks
//!   projective point (arkworks "projective" for short-Weierstrass curves is
//!   Jacobian, so coordinates carry over directly).

use ark_bn254::{Fq as ArkFq, G1Affine, G1Projective};
use ark_ff::{BigInt, Zero};

use super::buffers::DeviceBuffer;
use super::error::MetalError;
use super::field::{FR_U32_LIMBS, G1_AFFINE_U32_STRIDE};
use super::runtime::{KernelId, MetalContext};

/// u32 words per Jacobian result (X, Y, Z × 8 limbs).
pub const JAC_U32S: usize = 3 * FR_U32_LIMBS;

const _: () = assert!(std::mem::size_of::<G1Projective>() == JAC_U32S * 4);

/// Gather-index flag: sum the NEGATED base `(x, -y)`. Set by the signed
/// MSM entries of the increment-column commit path; base positions must
/// stay below it.
pub const SEG_INDEX_SIGN_BIT: u32 = 1 << 31;

/// View affine bases as the device `uint` stream.
///
/// Callers must ensure no base is the point at infinity (the flag byte is
/// dead to the shader, which would read the stale x/y words of an infinity
/// point as a finite one).
pub fn bases_as_u32s(bases: &[G1Affine]) -> &[u32] {
    // SAFETY: the const asserts in `super::field` pin G1Affine's layout —
    // x at 0, y at FR_U32_LIMBS*4, size a u32 multiple, align ≥ u32 — and
    // every bit pattern is a valid u32, for the same lifetime as `bases`.
    unsafe {
        std::slice::from_raw_parts(
            bases.as_ptr().cast::<u32>(),
            bases.len() * G1_AFFINE_U32_STRIDE,
        )
    }
}

fn fq_from_mont_limbs(limbs: &[u32]) -> ArkFq {
    let mut words = [0u64; 4];
    for (i, w) in words.iter_mut().enumerate() {
        *w = u64::from(limbs[2 * i]) | (u64::from(limbs[2 * i + 1]) << 32);
    }
    ArkFq::new_unchecked(BigInt::new(words))
}

/// Rebuild one kernel result (Montgomery Jacobian limbs) as an arkworks
/// point. `Z = 0` is the device identity encoding, mapped to the canonical
/// arkworks zero.
pub fn jac_from_device_limbs(limbs: &[u32]) -> G1Projective {
    debug_assert_eq!(limbs.len(), JAC_U32S);
    let z_limbs = &limbs[2 * FR_U32_LIMBS..];
    if z_limbs.iter().all(|&w| w == 0) {
        return G1Projective::zero();
    }
    G1Projective {
        x: fq_from_mont_limbs(&limbs[..FR_U32_LIMBS]),
        y: fq_from_mont_limbs(&limbs[FR_U32_LIMBS..2 * FR_U32_LIMBS]),
        z: fq_from_mont_limbs(z_limbs),
    }
}

/// Threadgroup width for the segment-sum dispatches. Default 64: at
/// production segment counts (~20k threads) the 256-wide dispatch loses
/// ~10% to threadgroup packing (measured @2^24 shape: w64 6.75 ms vs w256
/// 7.61 ms isolated). `JOLT_METAL_G1_TG_WIDTH` overrides (kill switch:
/// `256` restores the former width); read once.
pub(super) fn seg_sum_width() -> usize {
    static WIDTH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *WIDTH.get_or_init(|| {
        std::env::var("JOLT_METAL_G1_TG_WIDTH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|width| width.is_power_of_two() && (32..=1024).contains(width))
            .unwrap_or(64)
    })
}

/// Device segment-bounds triples `[start, end, out_slot]` from a prefix
/// array, length-sorted descending: a simdgroup runs as long as its longest
/// segment, so dispatch order groups near-equal trip counts; `out_slot`
/// keeps results in the prefix order the host reducers expect.
/// `JOLT_METAL_G1_SORT=0` keeps bucket-walk order (kill switch; read once).
pub fn seg_bounds_sorted(seg_starts: &[u32]) -> Vec<u32> {
    static SORT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let sort = *SORT
        .get_or_init(|| std::env::var("JOLT_METAL_G1_SORT").map_or(true, |value| value != "0"));
    let n_segs = seg_starts.len().saturating_sub(1);
    let mut order: Vec<u32> = (0..n_segs as u32).collect();
    if sort {
        order.sort_unstable_by_key(|&s| {
            std::cmp::Reverse(seg_starts[s as usize + 1] - seg_starts[s as usize])
        });
    }
    let mut bounds = Vec::with_capacity(3 * n_segs);
    for &s in &order {
        let s = s as usize;
        bounds.extend([seg_starts[s], seg_starts[s + 1], s as u32]);
    }
    bounds
}

/// Encode one dispatch of the segment-sum kernel: thread `t` sums
/// `bases[indices[seg_bounds[3t] .. seg_bounds[3t+1]]]` and writes Jacobian
/// result `seg_bounds[3t+2]`. Synchronous; results are valid in `out` after
/// return.
///
/// `seg_bounds` must hold `3 * n_segs` entries (see [`seg_bounds_sorted`]);
/// `out` must hold `n_segs * JAC_U32S` u32s.
pub fn g1_seg_sum_dispatch(
    ctx: &MetalContext,
    bases: &DeviceBuffer<'_>,
    indices: &DeviceBuffer<'_>,
    seg_bounds: &DeviceBuffer<'_>,
    out: &DeviceBuffer<'_>,
    n_segs: usize,
) -> Result<(), MetalError> {
    debug_assert!(seg_bounds.len_bytes() >= 3 * n_segs * 4);
    debug_assert!(out.len_bytes() >= n_segs * JAC_U32S * 4);
    assert!(n_segs <= u32::MAX as usize, "segment count overflows u32");
    let mut pass = ctx.begin_pass()?;
    pass.dispatch_width(
        KernelId::G1SegSum,
        &[n_segs as u32],
        &[bases, indices, seg_bounds, out],
        n_segs,
        seg_sum_width(),
    );
    pass.run()
}

/// Slice-in, points-out wrapper for tests and small dispatches: builds the
/// device buffers, runs one dispatch, and decodes every segment result.
pub fn g1_seg_sums(
    ctx: &MetalContext,
    bases: &[G1Affine],
    indices: &[u32],
    seg_starts: &[u32],
) -> Result<Vec<G1Projective>, MetalError> {
    assert!(
        !seg_starts.is_empty(),
        "seg_starts needs n_segs + 1 entries"
    );
    let n_segs = seg_starts.len() - 1;
    let bases_buf = ctx.wrap_slice(bases_as_u32s(bases))?;
    let indices_buf = ctx.wrap_slice(indices)?;
    let bounds = seg_bounds_sorted(seg_starts);
    let bounds_buf = ctx.wrap_slice(&bounds)?;
    let out_buf = ctx.alloc_u32s(n_segs * JAC_U32S)?;
    g1_seg_sum_dispatch(ctx, &bases_buf, &indices_buf, &bounds_buf, &out_buf, n_segs)?;
    let mut out = vec![0u32; n_segs * JAC_U32S];
    out_buf.copy_to_u32s(&mut out);
    Ok(out
        .chunks_exact(JAC_U32S)
        .map(jac_from_device_limbs)
        .collect())
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "tests: fail loudly"
)]
mod tests {
    use ark_ec::{AffineRepr, CurveGroup};
    use ark_ff::UniformRand;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    use super::super::testing::gpu_lock;
    use super::*;

    fn ctx() -> &'static MetalContext {
        MetalContext::global().expect("metal context")
    }

    fn random_bases(n: usize, seed: u64) -> Vec<G1Affine> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..n).map(|_| G1Affine::rand(&mut rng)).collect()
    }

    fn host_sum(bases: &[G1Affine], indices: &[u32]) -> G1Projective {
        indices.iter().fold(G1Projective::zero(), |acc, &raw| {
            let base = bases[(raw & !SEG_INDEX_SIGN_BIT) as usize];
            if raw & SEG_INDEX_SIGN_BIT != 0 {
                acc - base
            } else {
                acc + base
            }
        })
    }

    /// The zero-copy contract: an affine point's device words are exactly
    /// its x/y Montgomery limbs at the pinned offsets.
    #[test]
    fn g1_affine_layout_matches_u32_view() {
        let bases = random_bases(3, 7);
        let words = bases_as_u32s(&bases);
        for (i, base) in bases.iter().enumerate() {
            let x = fq_from_mont_limbs(&words[i * G1_AFFINE_U32_STRIDE..][..FR_U32_LIMBS]);
            let y = fq_from_mont_limbs(
                &words[i * G1_AFFINE_U32_STRIDE + FR_U32_LIMBS..][..FR_U32_LIMBS],
            );
            assert_eq!(x, base.x);
            assert_eq!(y, base.y);
        }
    }

    /// Random segments of random sizes match the arkworks sums, including
    /// empty (identity), singleton (Z = 1 copy), and multi-hundred-add
    /// segments.
    #[test]
    fn seg_sums_match_arkworks() {
        let _lock = gpu_lock();
        let bases = random_bases(512, 11);
        let mut rng = ChaCha20Rng::seed_from_u64(12);
        let mut indices = Vec::new();
        let mut seg_starts = vec![0u32];
        for seg in 0..64 {
            let len = match seg % 8 {
                0 => 0,
                1 => 1,
                2 => 2,
                _ => (seg * 37) % 400,
            };
            for _ in 0..len {
                indices.push(rng.next_u32() % 512);
            }
            seg_starts.push(u32::try_from(indices.len()).unwrap());
        }

        let device = g1_seg_sums(ctx(), &bases, &indices, &seg_starts).expect("dispatch");
        for (seg, window) in seg_starts.windows(2).enumerate() {
            let expected = host_sum(&bases, &indices[window[0] as usize..window[1] as usize]);
            assert_eq!(device[seg], expected, "segment {seg} diverged");
        }
    }

    /// Sign-bit gathers subtract the base: random signed segments match the
    /// arkworks signed sums, including all-negative (identity-from-negation
    /// start) and cancelling ±same-base pairs.
    #[test]
    fn signed_seg_sums_match_arkworks() {
        let _lock = gpu_lock();
        let bases = random_bases(256, 17);
        let mut rng = ChaCha20Rng::seed_from_u64(18);
        let mut indices = Vec::new();
        let mut seg_starts = vec![0u32];
        for seg in 0..32 {
            let len = (seg * 29) % 200;
            for _ in 0..len {
                let sign = if rng.next_u32() & 1 == 0 {
                    SEG_INDEX_SIGN_BIT
                } else {
                    0
                };
                indices.push((rng.next_u32() % 256) | sign);
            }
            seg_starts.push(u32::try_from(indices.len()).unwrap());
        }
        // Deterministic edge segments: all-negative, and P - P.
        indices.extend([SEG_INDEX_SIGN_BIT, 1 | SEG_INDEX_SIGN_BIT]);
        seg_starts.push(u32::try_from(indices.len()).unwrap());
        indices.extend([3, 3 | SEG_INDEX_SIGN_BIT]);
        seg_starts.push(u32::try_from(indices.len()).unwrap());

        let device = g1_seg_sums(ctx(), &bases, &indices, &seg_starts).expect("dispatch");
        for (seg, window) in seg_starts.windows(2).enumerate() {
            let expected = host_sum(&bases, &indices[window[0] as usize..window[1] as usize]);
            assert_eq!(device[seg], expected, "signed segment {seg} diverged");
        }
        assert!(device[device.len() - 1].is_zero());
    }

    /// The three g1_madd special cases, unreachable through random data:
    /// P + P (doubling), P + (-P) (identity), and resuming from an identity
    /// accumulator mid-segment.
    #[test]
    fn seg_sum_edge_cases() {
        let _lock = gpu_lock();
        let mut rng = ChaCha20Rng::seed_from_u64(13);
        let p = G1Affine::rand(&mut rng);
        let q = G1Affine::rand(&mut rng);
        let bases = vec![p, (-p.into_group()).into_affine(), q];

        // [P, P] → 2P; [P, -P] → ∞; [P, -P, Q] → Q (identity accumulator
        // then mixed add); [P, P, P] → 3P (double then H≠0 add).
        let indices: Vec<u32> = vec![0, 0, 0, 1, 0, 1, 2, 0, 0, 0];
        let seg_starts: Vec<u32> = vec![0, 2, 4, 7, 10];
        let device = g1_seg_sums(ctx(), &bases, &indices, &seg_starts).expect("dispatch");

        assert_eq!(device[0], p.into_group() + p);
        assert!(device[1].is_zero());
        assert_eq!(device[2], q.into_group());
        assert_eq!(device[3], p.into_group() + p + p);
    }
}
