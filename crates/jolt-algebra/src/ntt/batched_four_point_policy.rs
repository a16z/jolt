//! Compile-time gate for the NEON vectorized four-point NTT stages.
//!
//! The fused DIF tail and DIT head transpose four-coefficient transforms across
//! `simd_lanes` independent lanes. One iteration therefore consumes
//! `4 * simd_lanes` coefficients. Callers use scalar stages when the ring degree
//! is not an exact multiple of that block span.

/// Return whether `D` contains an integer number of four-point SIMD blocks.
#[inline]
pub(crate) const fn batched_four_point_eligible<const D: usize>(simd_lanes: usize) -> bool {
    simd_lanes != 0 && D.is_multiple_of(4 * simd_lanes)
}

#[cfg(test)]
mod tests {
    use super::batched_four_point_eligible;

    #[test]
    fn eligibility_tracks_vector_block_span() {
        assert!(batched_four_point_eligible::<16>(4));
        assert!(!batched_four_point_eligible::<16>(8));
        assert!(batched_four_point_eligible::<32>(8));
        assert!(!batched_four_point_eligible::<64>(0));
    }
}
