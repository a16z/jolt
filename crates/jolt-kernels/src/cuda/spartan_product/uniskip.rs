use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_claims::protocols::jolt::geometry::dimensions::PRODUCT_UNISKIP_DOMAIN_SIZE;
use jolt_field::{Fr, JoltField, Ring};
use jolt_poly::lagrange::{
    centered_domain_start, centered_lagrange_evals, interpolate_to_coeffs, poly_mul,
};
use jolt_poly::UnivariatePoly;

use super::columns::DeviceProductColumns;
use super::witness::{BRANCH_BIT, JUMP_BIT, NEXT_IS_NOOP_BIT, SIGN_BIT_BASE};
use crate::cuda::common::context::{context_for, CudaKernelContext, BLOCK};
use crate::cuda::common::device::{fr_into, LIMBS};
use crate::cuda::common::devices::{fan_out, DeviceTask};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::primitives::reduce_lanes;
use crate::cuda::common::split_eq::split_eq_tables_window;

pub const LANES: usize = PRODUCT_UNISKIP_DOMAIN_SIZE;

pub const EXTENDED_SIZE: usize = 2 * PRODUCT_UNISKIP_DOMAIN_SIZE - 1;

pub const STRIP: usize = 4;

fn extension_coefficients() -> Result<Vec<u32>, CudaError> {
    let start =
        centered_domain_start(EXTENDED_SIZE).map_err(|_| CudaError::InvariantViolation {
            reason: "the Spartan product extension domain is invalid",
        })?;
    let mut coefficients = Vec::with_capacity(EXTENDED_SIZE * LANES);
    for position in 0..EXTENDED_SIZE {
        let weights = centered_lagrange_evals::<Fr>(LANES, Fr::from_i64(start + position as i64))
            .map_err(|_| CudaError::InvariantViolation {
            reason: "the Spartan product uni-skip domain is invalid",
        })?;
        for weight in weights {
            let coefficient = (-3i32..=3)
                .find(|&candidate| Fr::from_i64(i64::from(candidate)) == weight)
                .ok_or(CudaError::NotImplemented {
                    kernel: "CUDA product extension requires integer coefficients of magnitude at most three",
                })?;
            coefficients.push(coefficient as u32);
        }
    }
    Ok(coefficients)
}

pub fn extended_products_windows<F: JoltField>(
    columns: &[(usize, DeviceProductColumns)],
    tau_low: &[F],
) -> Result<Vec<F>, CudaError> {
    let shards = columns.len();
    let tasks: Vec<DeviceTask<'_, Vec<F>, CudaError>> = columns
        .iter()
        .enumerate()
        .map(|(shard, (ordinal, window))| {
            let task: DeviceTask<'_, Vec<F>, CudaError> = Box::new(move || {
                let device = context_for(*ordinal).ok_or(CudaError::InvariantViolation {
                    reason: "a Spartan product window names an absent device",
                })?;
                extended_products(device, window, tau_low, shard, shards)
            });
            task
        })
        .collect();
    let mut total = vec![F::zero(); EXTENDED_SIZE];
    for part in fan_out(tasks)? {
        if part.len() != EXTENDED_SIZE {
            return Err(CudaError::LengthMismatch {
                expected: EXTENDED_SIZE,
                got: part.len(),
            });
        }
        for (slot, value) in total.iter_mut().zip(&part) {
            *slot += *value;
        }
    }
    Ok(total)
}

pub fn extended_products<F: JoltField>(
    context: &CudaKernelContext,
    columns: &DeviceProductColumns,
    tau_low: &[F],
    shard: usize,
    shards: usize,
) -> Result<Vec<F>, CudaError> {
    let (e_in, e_out, in_bits) = split_eq_tables_window(context, tau_low, shard, shards)?;

    let coefficients = context.upload_u32_slice(&extension_coefficients()?)?;
    let cycles = columns.cycles();
    if LANES != 3 || cycles != e_in.len() * e_out.len() {
        return Err(CudaError::InvariantViolation {
            reason:
                "the CUDA product extension requires three lanes and one equality weight per cycle",
        });
    }
    let threads = cycles.div_ceil(STRIP);
    let blocks = u32::try_from(threads.div_ceil(BLOCK as usize).max(1)).map_err(|_| {
        CudaError::InvariantViolation {
            reason: "the Spartan extended products launch exceeds a u32 block count",
        }
    })?;
    let lanes = u32::try_from(EXTENDED_SIZE).map_err(|_| CudaError::InvariantViolation {
        reason: "the Spartan extended products launch exceeds a u32 lane count",
    })?;
    let mut partials = context.alloc(EXTENDED_SIZE * blocks as usize)?;

    let bits = CudaKernelContext::count_of(in_bits)?;
    let count = CudaKernelContext::count_of(cycles)?;
    let strip = CudaKernelContext::count_of(STRIP)?;
    let mut builder = context.stream().launch_builder(context.sp_uniskip());
    let _ = builder.arg(columns.narrow());
    let _ = builder.arg(columns.wide());
    let _ = builder.arg(columns.flags());
    let _ = builder.arg(&coefficients);
    let _ = builder.arg(e_in.limbs());
    let _ = builder.arg(e_out.limbs());
    let _ = builder.arg(&bits);
    let _ = builder.arg(&JUMP_BIT);
    let _ = builder.arg(&BRANCH_BIT);
    let _ = builder.arg(&NEXT_IS_NOOP_BIT);
    let _ = builder.arg(&SIGN_BIT_BASE);
    let _ = builder.arg(&count);
    let _ = builder.arg(&strip);
    let _ = builder.arg(partials.limbs_mut());
    // SAFETY: each node reads three uploaded coefficients and each thread visits
    // at most STRIP cycles, guarded by `t < cycles`. The checked equality-table
    // product covers those cycles. The native columns contain one row per cycle;
    // each block writes one field element within `EXTENDED_SIZE * blocks`.
    // Shared memory holds BLOCK field elements and all threads reach the reduction.
    let _ = unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (blocks, lanes, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
        })
    }?;

    let totals = reduce_lanes(context, partials, lanes, blocks)?;
    let mut values = Vec::with_capacity(EXTENDED_SIZE);
    for value in totals.to_host()? {
        values.push(fr_into(value).ok_or(CudaError::NotImplemented {
            kernel: "CUDA kernels support only the BN254 scalar field",
        })?);
    }
    Ok(values)
}

pub fn first_round_poly<F: JoltField>(
    t1_values: &[F],
    tau_high: F,
) -> Result<UnivariatePoly<F>, CudaError> {
    if t1_values.len() != EXTENDED_SIZE {
        return Err(CudaError::LengthMismatch {
            expected: EXTENDED_SIZE,
            got: t1_values.len(),
        });
    }
    let invalid_domain = || CudaError::InvariantViolation {
        reason: "the Spartan product uni-skip domain is not a valid centered integer domain",
    };
    let domain_start = centered_domain_start(LANES).map_err(|_| invalid_domain())?;
    let extended_start = centered_domain_start(EXTENDED_SIZE).map_err(|_| invalid_domain())?;

    let kernel_values = centered_lagrange_evals::<F>(PRODUCT_UNISKIP_DOMAIN_SIZE, tau_high)
        .map_err(|_| invalid_domain())?;
    let kernel_coefficients = interpolate_to_coeffs(domain_start, &kernel_values);
    let t1_coefficients = interpolate_to_coeffs(extended_start, t1_values);
    Ok(UnivariatePoly::new(poly_mul(
        &kernel_coefficients,
        &t1_coefficients,
    )))
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::spartan::{
        branch_flag_product, jump_flag_product, left_instruction_input_product,
        lookup_output_product, next_is_noop_product, right_instruction_input_product,
    };
    use jolt_claims::protocols::jolt::JoltOneHotConfig;
    use jolt_field::{Fr, Ring};
    use jolt_witness::collect_bundles;
    use jolt_witness::witnesses::ToField;
    use proptest::prelude::*;

    use super::{extended_products, EXTENDED_SIZE, LANES};
    use crate::cuda::common::context::{context_for, shared_context};
    use crate::cuda::common::devices::witness_windows;
    use crate::cuda::common::error::CudaError;
    use crate::cuda::common::testing::{arb_point, with_r1cs_witness};
    use crate::cuda::spartan_product::columns::DeviceProductColumns;
    use crate::cuda::spartan_product::witness::{self, SpartanProductWitness};
    use crate::reference::views::{dense_view, eq_table};
    use jolt_poly::lagrange::{centered_domain_start, centered_lagrange_evals};

    const LOG_T: usize = 7;

    const RAM_K: usize = 1 << 10;

    const fn one_hot() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 32,
        }
    }

    #[test]
    fn extended_products_reject_mismatched_equality_domain() {
        let Some(context) = shared_context() else {
            return;
        };
        let rows = witness::sample_rows(71, 4);
        let columns =
            DeviceProductColumns::new(context, &witness::pack(&rows)).expect("product columns");
        for log_t in [1, 3] {
            let tau = vec![Fr::from_u64(7); log_t];
            assert!(matches!(
                extended_products(context, &columns, &tau, 0, 1),
                Err(CudaError::InvariantViolation { .. })
            ));
        }
    }

    #[test]
    fn uniskip_matches_dense_polynomial_at_extreme_operands() {
        if shared_context().is_none() {
            return;
        }
        for log_t in [1usize, 7, 13] {
            let cycles = 1usize << log_t;
            let mut rows = witness::sample_rows(71, cycles);
            for (i, row) in rows.iter_mut().enumerate() {
                let k = (i / 8) as u64;
                row.left_instruction_input.0 =
                    [u64::MAX - k, 0, 1, k + 1, 0, u64::MAX - k, 1, 0][i % 8];
                row.lookup_output.0 = [0, u64::MAX - k, 0, 1, k + 1, u64::MAX - k, 1, 0][i % 8];
                row.right_instruction_input.0 = [
                    i128::MIN + i128::from(k),
                    i128::MIN + i128::from(k),
                    i128::MAX - i128::from(k),
                    -1,
                    0,
                    1,
                    -1,
                    i128::MAX - i128::from(k),
                ][i % 8];
            }
            let columns: Vec<_> = witness_windows(cycles)
                .iter()
                .enumerate()
                .map(|(ordinal, window)| {
                    let context = context_for(ordinal).expect("configured device");
                    let columns = DeviceProductColumns::new(
                        context,
                        &witness::pack(&rows[window.start..window.end()]),
                    )
                    .expect("product columns");
                    (ordinal, columns)
                })
                .collect();
            for case in 0..3 {
                let tau: Vec<_> = (0..log_t)
                    .map(|i| Fr::from_u64(if case < 2 { case } else { 11 + i as u64 }))
                    .collect();
                let eq = eq_table(&tau);
                let values = super::extended_products_windows::<Fr>(&columns, &tau)
                    .expect("extended products");
                for late_tau in [-1, 0, 1, 37] {
                    let late_tau = Fr::from_i64(late_tau);
                    let poly =
                        super::first_round_poly(&values, late_tau).expect("uni-skip polynomial");
                    let late_weights =
                        centered_lagrange_evals::<Fr>(LANES, late_tau).expect("late weights");
                    for at in [-2, -1, 0, 1, 2, 59] {
                        let at = Fr::from_i64(at);
                        let weights =
                            centered_lagrange_evals::<Fr>(LANES, at).expect("evaluation weights");
                        let product: Fr = rows
                            .iter()
                            .zip(&eq)
                            .map(|(row, e)| {
                                let left = weights[0] * row.left_instruction_input.to_field::<Fr>()
                                    + weights[1] * row.lookup_output.to_field::<Fr>()
                                    + weights[2] * row.jump.to_field::<Fr>();
                                let right = weights[0]
                                    * row.right_instruction_input.to_field::<Fr>()
                                    + weights[1] * row.branch.to_field::<Fr>()
                                    + weights[2]
                                        * (Fr::from_u64(1) - row.next_is_noop.to_field::<Fr>());
                                *e * left * right
                            })
                            .sum();
                        let scale: Fr = weights
                            .iter()
                            .zip(&late_weights)
                            .map(|(a, b)| *a * *b)
                            .sum();
                        assert_eq!(
                            poly.evaluate(at),
                            scale * product,
                            "log_t={log_t}, case={case}, at={at:?}, late_tau={late_tau:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn extended_products_windows_sum_to_the_whole_domain() {
        let Some(context) = shared_context() else {
            return;
        };
        let cycles = 1usize << LOG_T;
        let rows = witness::sample_rows(0x7C0D, cycles);
        let tau_low: Vec<Fr> = (0..LOG_T)
            .map(|i| Fr::from_u64(17 + 4 * i as u64))
            .collect();
        let build = |base: usize, len: usize| {
            DeviceProductColumns::new(context, &witness::pack(&rows[base..base + len]))
                .expect("upload a packed product window")
        };

        let whole = extended_products::<Fr>(context, &build(0, cycles), &tau_low, 0, 1)
            .expect("the whole-domain extended products");
        assert_eq!(whole.len(), EXTENDED_SIZE);
        assert!(
            whole.iter().any(|value| *value != Fr::from_u64(0)),
            "a degenerate fixture would hide a divergence",
        );

        for shards in [2usize, 4, 8] {
            let len = cycles / shards;
            let windows: Vec<(usize, DeviceProductColumns)> = (0..shards)
                .map(|shard| (0usize, build(shard * len, len)))
                .collect();
            let got = super::extended_products_windows::<Fr>(&windows, &tau_low)
                .expect("the windowed extended products");
            assert_eq!(
                got, whole,
                "shards={shards}: the extended products diverged"
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2))]
        #[test]
        fn extended_products_matches_cpu(
            seed in any::<u64>(),
            tau_low in arb_point(LOG_T),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };

            with_r1cs_witness(LOG_T, RAM_K, one_hot(), seed, |witness| {
                let rows = collect_bundles::<SpartanProductWitness>(witness, 1usize << LOG_T)
                    .expect("the fixture serves every product bundle field");
                let packed = witness::pack(&rows);
                let columns = DeviceProductColumns::new(context, &packed)
                    .expect("upload the packed product columns");
                let got = extended_products::<Fr>(context, &columns, &tau_low, 0, 1)
                    .expect("the device extended products");

                let view = |opening| {
                    dense_view::<Fr>(witness, opening).expect("the fixture serves the column")
                };
                let left = [
                    view(left_instruction_input_product()),
                    view(lookup_output_product()),
                    view(jump_flag_product()),
                ];
                let next_is_noop = view(next_is_noop_product());
                let right = [
                    view(right_instruction_input_product()),
                    view(branch_flag_product()),
                    next_is_noop
                        .iter()
                        .map(|value| Fr::from_u64(1) - *value)
                        .collect(),
                ];
                let eq = eq_table(&tau_low);

                let mut expected = vec![Fr::from_u64(0); EXTENDED_SIZE];
                let start = centered_domain_start(EXTENDED_SIZE).expect("extension domain");
                for (node, value) in expected.iter_mut().enumerate() {
                    let weights = centered_lagrange_evals::<Fr>(LANES, Fr::from_i64(start + node as i64))
                        .expect("Lagrange weights");
                    *value = (0..1usize << LOG_T).map(|cycle| {
                        let a: Fr = weights.iter().zip(&left).map(|(w, lane)| *w * lane[cycle]).sum();
                        let b: Fr = weights.iter().zip(&right).map(|(w, lane)| *w * lane[cycle]).sum();
                        eq[cycle] * a * b
                    }).sum();
                }

                prop_assert_eq!(got, expected, "the extended products diverged");
                Ok(())
            })?;
        }

        #[test]
        fn extended_products_matches_cpu_on_signed_rows(
            seed in any::<u64>(),
            tau_low in arb_point(LOG_T),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };

            let rows = witness::sample_rows(seed, 1usize << LOG_T);
            let packed = witness::pack(&rows);
            let columns = DeviceProductColumns::new(context, &packed)
                .expect("upload the packed product columns");
            let got = extended_products::<Fr>(context, &columns, &tau_low, 0, 1)
                .expect("the device extended products");

            let left: [Vec<Fr>; LANES] = [
                rows.iter().map(|row| row.left_instruction_input.to_field()).collect(),
                rows.iter().map(|row| row.lookup_output.to_field()).collect(),
                rows.iter().map(|row| row.jump.to_field()).collect(),
            ];
            let right: [Vec<Fr>; LANES] = [
                rows.iter().map(|row| row.right_instruction_input.to_field()).collect(),
                rows.iter().map(|row| row.branch.to_field()).collect(),
                rows.iter()
                    .map(|row| Fr::from_u64(1) - ToField::to_field::<Fr>(row.next_is_noop))
                    .collect(),
            ];
            let eq = eq_table(&tau_low);

            let mut expected = vec![Fr::from_u64(0); EXTENDED_SIZE];
            let start = centered_domain_start(EXTENDED_SIZE).expect("extension domain");
            for (node, value) in expected.iter_mut().enumerate() {
                let weights = centered_lagrange_evals::<Fr>(LANES, Fr::from_i64(start + node as i64))
                    .expect("Lagrange weights");
                *value = (0..1usize << LOG_T).map(|cycle| {
                    let a: Fr = weights.iter().zip(&left).map(|(w, lane)| *w * lane[cycle]).sum();
                    let b: Fr = weights.iter().zip(&right).map(|(w, lane)| *w * lane[cycle]).sum();
                    eq[cycle] * a * b
                }).sum();
            }

            prop_assert_eq!(got, expected, "the extended products diverged on signed rows");
        }
    }
}
