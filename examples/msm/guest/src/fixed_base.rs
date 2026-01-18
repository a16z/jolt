use alloc::boxed::Box;

use crate::traits::{MsmGroup, WindowedScalar};

/// Fixed-base precomputed table for a single base point.
/// table[window][digit] = digit · 2^(window·width) · base.
///
/// Type parameters:
/// - `G`: The group/point type
/// - `WINDOWS`: Number of windows (= ceil(scalar_bits / window_bits))
/// - `BUCKETS`: Number of buckets per window (= 2^window_bits)
pub struct FixedBaseTable<G, const WINDOWS: usize, const BUCKETS: usize> {
    table: Box<[[G; BUCKETS]; WINDOWS]>,
}

impl<G: MsmGroup, const WINDOWS: usize, const BUCKETS: usize> FixedBaseTable<G, WINDOWS, BUCKETS> {
    /// Window size derived from BUCKETS at compile time.
    pub const WINDOW_BITS: usize = BUCKETS.trailing_zeros() as usize;

    /// Precompute table for a given base point.
    /// Window size is derived from BUCKETS const generic.
    #[inline(always)]
    pub fn new(base: &G) -> Self {
        const {
            assert!(BUCKETS > 1, "BUCKETS must be > 1");
            assert!(BUCKETS.is_power_of_two(), "BUCKETS must be a power of two");
            assert!(
                BUCKETS.trailing_zeros() as usize <= 16,
                "BUCKETS (window size) must be <= 2^16"
            );
        }

        let mut table: Box<[[G; BUCKETS]; WINDOWS]> = Box::new(core::array::from_fn(|_| {
            core::array::from_fn(|_| G::identity())
        }));

        let mut window_base = base.clone();
        for window_table in table.iter_mut() {
            window_table[0] = G::identity();
            window_table[1] = window_base.clone();
            let mut digit = 2;
            while digit < BUCKETS {
                window_table[digit] = window_table[digit - 1].add(&window_base);
                digit += 1;
            }

            let mut shift = 0;
            while shift < Self::WINDOW_BITS {
                window_base = window_base.double();
                shift += 1;
            }
        }

        Self { table }
    }

    /// Scalar multiplication using table lookup + addition.
    #[inline(always)]
    pub fn scalar_mul<S: WindowedScalar>(&self, scalar: &S) -> G {
        let mut result = G::identity();
        for (window_idx, window_table) in self.table.iter().enumerate() {
            let digit = scalar.window(window_idx * Self::WINDOW_BITS, Self::WINDOW_BITS) as usize;
            if digit != 0 {
                result = result.add(&window_table[digit]);
            }
        }
        result
    }
}

/// Fixed-base MSM: Σ(scalar_i · base) using precomputed table.
#[inline(always)]
pub fn msm_fixed_base<G, S, const WINDOWS: usize, const BUCKETS: usize>(
    scalars: &[S],
    table: &FixedBaseTable<G, WINDOWS, BUCKETS>,
) -> G
where
    G: MsmGroup,
    S: WindowedScalar,
{
    let mut result = G::identity();
    for scalar in scalars {
        let term = table.scalar_mul(scalar);
        result = result.add(&term);
    }
    result
}
