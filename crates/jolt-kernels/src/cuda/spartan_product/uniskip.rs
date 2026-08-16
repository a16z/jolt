use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_claims::protocols::jolt::geometry::dimensions::PRODUCT_UNISKIP_DOMAIN_SIZE;
use jolt_field::Field;
use jolt_poly::lagrange::{centered_lagrange_evals, interpolate_to_coeffs, poly_mul};
use jolt_poly::UnivariatePoly;

use super::columns::DeviceProductColumns;
use super::witness::{BRANCH_BIT, JUMP_BIT, NEXT_IS_NOOP_BIT, SIGN_BIT_BASE};
use crate::cuda::common::context::{CudaKernelContext, BLOCK};
use crate::cuda::common::dense_product::DeviceDenseProduct;
use crate::cuda::common::device::{fr_into, LIMBS};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::split_eq::split_eq_tables;

pub const LANES: usize = PRODUCT_UNISKIP_DOMAIN_SIZE;

pub const MATRIX_LANES: usize = LANES * LANES;

pub const EXTENDED_SIZE: usize = 2 * PRODUCT_UNISKIP_DOMAIN_SIZE - 1;

pub const STRIP: usize = 4;

pub fn node_field<F: Field>(node: i64) -> F {
    if node >= 0 {
        F::from_u64(node as u64)
    } else {
        -F::from_u64(node.unsigned_abs())
    }
}

pub fn product_matrix<F: Field>(
    context: &CudaKernelContext,
    columns: &DeviceProductColumns,
    tau_low: &[F],
) -> Result<Vec<F>, CudaError> {
    let (e_in, e_out, in_bits) = split_eq_tables(context, tau_low)?;

    let cycles = columns.cycles();
    let threads = cycles.div_ceil(STRIP);
    let blocks = u32::try_from(threads.div_ceil(BLOCK as usize).max(1)).map_err(|_| {
        CudaError::InvariantViolation {
            reason: "the Spartan product matrix launch exceeds a u32 block count",
        }
    })?;
    let left_lanes = u32::try_from(LANES).map_err(|_| CudaError::InvariantViolation {
        reason: "the Spartan product matrix launch exceeds a u32 lane count",
    })?;
    let mut partials = context.alloc(MATRIX_LANES * blocks as usize)?;

    let bits = CudaKernelContext::count_of(in_bits)?;
    let count = CudaKernelContext::count_of(cycles)?;
    let strip = CudaKernelContext::count_of(STRIP)?;
    let mut builder = context.stream().launch_builder(context.sp_matrix());
    let _ = builder.arg(columns.narrow());
    let _ = builder.arg(columns.wide());
    let _ = builder.arg(columns.flags());
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
    // SAFETY: thread `(blockIdx.x, threadIdx.x)` of left factor
    // `blockIdx.y < LANES` covers cycles `base .. base + STRIP` and breaks at
    // `cycles`, so every read is inside `narrow`'s `cycles * NARROW`, `wide`'s
    // `cycles * WIDE * 2` and `flags`' `cycles` entries. The eq lookup is
    // bounded because `in_bits` comes from `e_in`'s length and `t < cycles`
    // equals `e_in_len * e_out_len`. It writes only
    // `partials[(blockIdx.y * LANES + lane) * blocks + blockIdx.x]` for
    // `lane < LANES`, inside the `MATRIX_LANES * blocks` allocation. Shared
    // memory is `BLOCK * LIMBS` u64s, matching `shared_mem_bytes`, and the
    // block reduction sits outside every early exit so all threads reach each
    // `__syncthreads()`.
    let _ = unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (blocks, left_lanes, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
        })
    }?;

    let lanes = u32::try_from(MATRIX_LANES).map_err(|_| CudaError::InvariantViolation {
        reason: "the Spartan product matrix launch exceeds a u32 lane count",
    })?;
    let totals = DeviceDenseProduct::reduce_lanes(context, partials, lanes, blocks)?;
    let mut matrix = Vec::with_capacity(MATRIX_LANES);
    for value in totals.to_host()? {
        matrix.push(fr_into(value).ok_or(CudaError::NotImplemented {
            kernel: "CUDA kernels support only the BN254 scalar field",
        })?);
    }
    Ok(matrix)
}

pub fn first_round_poly<F: Field>(
    matrix: &[F],
    tau_high: F,
) -> Result<UnivariatePoly<F>, CudaError> {
    if matrix.len() != MATRIX_LANES {
        return Err(CudaError::LengthMismatch {
            expected: MATRIX_LANES,
            got: matrix.len(),
        });
    }
    let domain_start = -((PRODUCT_UNISKIP_DOMAIN_SIZE as i64 - 1) / 2);
    let extended_start = -((EXTENDED_SIZE as i64 - 1) / 2);
    let invalid_domain = || CudaError::InvariantViolation {
        reason: "the Spartan product uni-skip domain is not a valid centered integer domain",
    };

    let mut t1_values = vec![F::zero(); EXTENDED_SIZE];
    for (position, value) in t1_values.iter_mut().enumerate() {
        let node = node_field::<F>(extended_start + position as i64);
        let weights = centered_lagrange_evals::<F>(PRODUCT_UNISKIP_DOMAIN_SIZE, node)
            .map_err(|_| invalid_domain())?;
        let mut sum = F::zero();
        for (left, &left_weight) in weights.iter().enumerate() {
            for (right, &right_weight) in weights.iter().enumerate() {
                sum += left_weight * right_weight * matrix[left * LANES + right];
            }
        }
        *value = sum;
    }

    let kernel_values = centered_lagrange_evals::<F>(PRODUCT_UNISKIP_DOMAIN_SIZE, tau_high)
        .map_err(|_| invalid_domain())?;
    let kernel_coefficients = interpolate_to_coeffs(domain_start, &kernel_values);
    let t1_coefficients = interpolate_to_coeffs(extended_start, &t1_values);
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
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_witness::collect_bundles;
    use jolt_witness::witnesses::ToField;
    use proptest::prelude::*;

    use super::{product_matrix, LANES, MATRIX_LANES};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{arb_point, with_r1cs_witness};
    use crate::cuda::spartan_product::columns::DeviceProductColumns;
    use crate::cuda::spartan_product::witness::{self, SpartanProductWitness};
    use crate::reference::views::{dense_view, eq_table};

    const LOG_T: usize = 7;

    const RAM_K: usize = 1 << 10;

    const fn one_hot() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 32,
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2))]
        #[test]
        fn product_matrix_matches_cpu(
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
                let got = product_matrix::<Fr>(context, &columns, &tau_low)
                    .expect("the device product matrix");

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

                let mut expected = vec![Fr::from_u64(0); MATRIX_LANES];
                for (first, left) in left.iter().enumerate() {
                    for (second, right) in right.iter().enumerate() {
                        expected[first * LANES + second] = (0..1usize << LOG_T)
                            .map(|cycle| eq[cycle] * left[cycle] * right[cycle])
                            .sum();
                    }
                }

                prop_assert_eq!(got, expected, "the product matrix diverged");
                Ok(())
            })?;
        }

        #[test]
        fn product_matrix_matches_cpu_on_signed_rows(
            seed in any::<u64>(),
            tau_low in arb_point(LOG_T),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };

            let rows = witness::sample_rows(seed, 1usize << LOG_T);
            let packed = witness::pack(&rows);
            let columns = DeviceProductColumns::new(context, &packed)
                .expect("upload the packed product columns");
            let got = product_matrix::<Fr>(context, &columns, &tau_low)
                .expect("the device product matrix");

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

            let mut expected = vec![Fr::from_u64(0); MATRIX_LANES];
            for (first, left) in left.iter().enumerate() {
                for (second, right) in right.iter().enumerate() {
                    expected[first * LANES + second] = (0..1usize << LOG_T)
                        .map(|cycle| eq[cycle] * left[cycle] * right[cycle])
                        .sum();
                }
            }

            prop_assert_eq!(got, expected, "the product matrix diverged on signed rows");
        }
    }
}
