#![cfg_attr(feature = "guest", no_std)]

extern crate alloc;

pub mod curves;
pub mod fixed_base;
pub mod glv;
pub mod pippenger;
pub mod scalar;
pub mod traits;

use core::hint::black_box;

use jolt::{end_cycle_tracking, start_cycle_tracking};
use jolt_inlines_grumpkin::{GrumpkinFr, GrumpkinPoint};

use crate::curves::grumpkin::{
    generate_points_fixed_base, generate_scalars, scalar_to_fr, FixedBaseTable, BASELINE_WINDOW,
    GLV_WINDOW,
};
use crate::fixed_base::msm_fixed_base;
use crate::glv::msm_glv_with_scratch_const;
use crate::pippenger::msm_pippenger_const;

// Re-export for convenience.
pub use crate::glv::msm_glv;
pub use crate::pippenger::msm_pippenger;
pub use crate::traits::*;

/// MSM size - adjust this constant to change benchmark size.
/// Recommended values: 16 (quick test), 256, 1024, 2048.
const MSM_SIZE: usize = 1024;

const RUN_FIXED_BASE_ONLY: bool = false;
const RUN_PIPPENGER_ONLY: bool = false;
const RUN_GLV_PIPPENGER_ONLY: bool = true;

const RUN_MICRO_BENCH: bool = false;

#[jolt::provable(
    max_output_size = 1024,
    memory_size = 134217728,   // 128 MB
    stack_size = 67108864,     // 64 MB
    max_trace_length = 500000000
)]
fn msm_bench(seed: u64) -> [u64; 8] {
    let g = GrumpkinPoint::generator();

    // Fixed-base (generator) precompute table.
    start_cycle_tracking("fixed_base_precompute_g_w14");
    let fixed_table_g = FixedBaseTable::new(&g);
    end_cycle_tracking("fixed_base_precompute_g_w14");

    if RUN_MICRO_BENCH {
        let g2 = g.double();

        // Benchmark: Point addition.
        start_cycle_tracking("point_add");
        let _r = black_box(black_box(&g).add(black_box(&g2)));
        end_cycle_tracking("point_add");

        // Benchmark: Point doubling.
        start_cycle_tracking("point_double");
        let _r = black_box(black_box(&g).double());
        end_cycle_tracking("point_double");

        // Benchmark: Double-and-add (2P + Q).
        start_cycle_tracking("point_double_and_add");
        let _r = black_box(black_box(&g).double_and_add(black_box(&g2)));
        end_cycle_tracking("point_double_and_add");
    }

    // Generate test data.
    start_cycle_tracking("msm_setup");
    let scalars = generate_scalars::<MSM_SIZE>(seed);
    let points = generate_points_fixed_base::<MSM_SIZE>(seed.wrapping_add(1), &fixed_table_g);
    let fr_scalars: [GrumpkinFr; MSM_SIZE] = core::array::from_fn(|i| scalar_to_fr(&scalars[i]));
    end_cycle_tracking("msm_setup");

    if RUN_FIXED_BASE_ONLY {
        start_cycle_tracking("msm_fixed_base_table_256_w14");
        let result = black_box(msm_fixed_base(
            black_box(&scalars),
            black_box(&fixed_table_g),
        ));
        end_cycle_tracking("msm_fixed_base_table_256_w14");
        return result.to_u64_arr();
    }

    if RUN_PIPPENGER_ONLY {
        start_cycle_tracking("msm_pippenger");
        let result = black_box(msm_pippenger_const::<
            GrumpkinPoint,
            [u64; 4],
            { BASELINE_WINDOW },
        >(black_box(&scalars), black_box(&points)));
        end_cycle_tracking("msm_pippenger");
        return result.to_u64_arr();
    }

    if RUN_GLV_PIPPENGER_ONLY {
        start_cycle_tracking("msm_glv_pippenger");
        let mut expanded_scalars = [0u128; 2 * MSM_SIZE];
        let mut expanded_points: [GrumpkinPoint; 2 * MSM_SIZE] =
            core::array::from_fn(|_| GrumpkinPoint::infinity());
        let result = black_box(msm_glv_with_scratch_const::<GrumpkinPoint, { GLV_WINDOW }>(
            black_box(&fr_scalars),
            black_box(&points),
            black_box(&mut expanded_scalars),
            black_box(&mut expanded_points),
        ));
        end_cycle_tracking("msm_glv_pippenger");
        return result.to_u64_arr();
    }

    GrumpkinPoint::infinity().to_u64_arr()
}
