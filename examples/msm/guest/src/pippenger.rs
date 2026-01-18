use alloc::vec::Vec;

use crate::traits::{MsmGroup, WindowedScalar};

#[inline(always)]
fn double_n<G: MsmGroup>(mut acc: G, n: usize) -> G {
    let mut i = 0;
    while i < n {
        acc = acc.double();
        i += 1;
    }
    acc
}

/// Pippenger MSM: Σ(scalar_i · point_i) with runtime window sizing.
///
/// # Timing
/// This implementation is **variable-time** with respect to scalar values
/// (bucket selection, leading-zero optimization). This is acceptable for zkVM
/// proof generation where the prover already knows all secrets, but should not
/// be used for operations where timing leaks are a concern.
#[inline(always)]
pub fn msm_pippenger<G, S>(scalars: &[S], points: &[G], window_bits: usize) -> G
where
    G: MsmGroup,
    S: WindowedScalar,
{
    assert_eq!(scalars.len(), points.len());
    if scalars.is_empty() {
        return G::identity();
    }
    assert!(window_bits > 0, "window_bits must be positive");
    assert!(window_bits <= 16, "window_bits must be <= 16");

    let scalar_bits = scalars[0].bit_len();
    let num_windows = scalar_bits.div_ceil(window_bits);
    let num_buckets = 1usize << window_bits;

    let mut result = G::identity();
    let mut started = false;

    let mut buckets: Vec<G> = Vec::with_capacity(num_buckets);
    buckets.resize_with(num_buckets, G::identity);

    for window_idx in (0..num_windows).rev() {
        for bucket in buckets.iter_mut() {
            *bucket = G::identity();
        }

        let offset = window_idx * window_bits;
        for (scalar, point) in scalars.iter().zip(points.iter()) {
            let bucket_idx = scalar.window(offset, window_bits) as usize;
            if bucket_idx != 0 {
                buckets[bucket_idx] = buckets[bucket_idx].add(point);
            }
        }

        let mut running = G::identity();
        let mut window_sum = G::identity();
        let mut i = num_buckets;
        while i > 1 {
            i -= 1;
            running = running.add(&buckets[i]);
            window_sum = window_sum.add(&running);
        }

        if started {
            result = double_n(result, window_bits);
        } else if !window_sum.is_identity() {
            started = true;
        }

        if started {
            result = result.add(&window_sum);
        }
    }

    result
}

/// Pippenger MSM: Σ(scalar_i · point_i) with compile-time window sizing.
#[inline(always)]
pub fn msm_pippenger_const<G, S, const WINDOW_BITS: usize>(scalars: &[S], points: &[G]) -> G
where
    G: MsmGroup,
    S: WindowedScalar,
{
    const {
        assert!(WINDOW_BITS > 0, "WINDOW_BITS must be positive");
        assert!(WINDOW_BITS <= 16, "WINDOW_BITS must be <= 16");
    };

    assert_eq!(scalars.len(), points.len());
    if scalars.is_empty() {
        return G::identity();
    }

    let scalar_bits = scalars[0].bit_len();
    let num_windows = scalar_bits.div_ceil(WINDOW_BITS);
    let num_buckets = 1usize << WINDOW_BITS;

    let mut result = G::identity();
    let mut started = false;

    let mut buckets: Vec<G> = Vec::with_capacity(num_buckets);
    buckets.resize_with(num_buckets, G::identity);

    for window_idx in (0..num_windows).rev() {
        for bucket in buckets.iter_mut() {
            *bucket = G::identity();
        }

        let offset = window_idx * WINDOW_BITS;
        for (scalar, point) in scalars.iter().zip(points.iter()) {
            let bucket_idx = scalar.window(offset, WINDOW_BITS) as usize;
            if bucket_idx != 0 {
                buckets[bucket_idx] = buckets[bucket_idx].add(point);
            }
        }

        let mut running = G::identity();
        let mut window_sum = G::identity();
        let mut i = num_buckets;
        while i > 1 {
            i -= 1;
            running = running.add(&buckets[i]);
            window_sum = window_sum.add(&running);
        }

        if started {
            let mut shift = 0;
            while shift < WINDOW_BITS {
                result = result.double();
                shift += 1;
            }
        } else if !window_sum.is_identity() {
            started = true;
        }

        if started {
            result = result.add(&window_sum);
        }
    }

    result
}
