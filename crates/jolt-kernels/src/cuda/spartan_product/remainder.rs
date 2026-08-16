use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_claims::protocols::jolt::geometry::dimensions::PRODUCT_UNISKIP_DOMAIN_SIZE;
use jolt_claims::protocols::jolt::geometry::spartan::{
    branch_flag_product, jump_flag_product, left_instruction_input_product, lookup_output_product,
    next_is_noop_product, right_instruction_input_product, virtual_instruction_product,
    write_lookup_output_to_rd_product,
};
use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_field::{Field, Fr};
use jolt_poly::lagrange::{centered_lagrange_evals, centered_lagrange_kernel};
use jolt_poly::{BindingOrder, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use std::collections::BTreeMap;

use super::columns::DeviceProductColumns;
use super::witness::{self, BRANCH_BIT, CLAIM_COLUMNS, JUMP_BIT, NEXT_IS_NOOP_BIT, SIGN_BIT_BASE};
use crate::cuda::common::context::{CudaKernelContext, BLOCK};
use crate::cuda::common::dense_product::DeviceDenseProduct;
use crate::cuda::common::device::{fr_into, require_fr, require_fr_slice, DeviceFrVec, LIMBS};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::split_eq::{split_eq_tables, DeviceSplitEq};

pub const CLAIM_LANES: usize = 4;

pub const MESSAGE_STRIP: usize = 1;

pub const CLAIM_STRIP: usize = 16;

pub fn claim_openings() -> [JoltOpeningId; CLAIM_COLUMNS] {
    [
        left_instruction_input_product(),
        right_instruction_input_product(),
        jump_flag_product(),
        write_lookup_output_to_rd_product(),
        lookup_output_product(),
        branch_flag_product(),
        next_is_noop_product(),
        virtual_instruction_product(),
    ]
}

pub struct DeviceProductRemainder<F: Field> {
    columns: DeviceProductColumns,
    left: DeviceFrVec,
    right: DeviceFrVec,
    eq: DeviceSplitEq<F>,
    log_t: usize,
    challenges: Vec<F>,
}

impl<F: Field> DeviceProductRemainder<F> {
    pub fn new(
        context: &CudaKernelContext,
        columns: DeviceProductColumns,
        tau_low: &[F],
        tau_high: F,
        uniskip_challenge: F,
        log_t: usize,
    ) -> Result<Self, CudaError> {
        let invalid_domain = || CudaError::InvariantViolation {
            reason: "the Spartan product uni-skip domain is not a valid centered integer domain",
        };
        let weights = centered_lagrange_evals::<F>(PRODUCT_UNISKIP_DOMAIN_SIZE, uniskip_challenge)
            .map_err(|_| invalid_domain())?;
        let device_weights = context.upload(require_fr_slice(&weights)?)?;

        let cycles = columns.cycles();
        let mut left = context.alloc(cycles)?;
        let mut right = context.alloc(cycles)?;
        let count = CudaKernelContext::count_of(cycles)?;
        let mut builder = context.stream().launch_builder(context.sp_factors());
        let _ = builder.arg(columns.narrow());
        let _ = builder.arg(columns.wide());
        let _ = builder.arg(columns.flags());
        let _ = builder.arg(device_weights.limbs());
        let _ = builder.arg(&JUMP_BIT);
        let _ = builder.arg(&BRANCH_BIT);
        let _ = builder.arg(&NEXT_IS_NOOP_BIT);
        let _ = builder.arg(&SIGN_BIT_BASE);
        let _ = builder.arg(&count);
        let _ = builder.arg(left.limbs_mut());
        let _ = builder.arg(right.limbs_mut());
        // SAFETY: thread `t < cycles` reads row `t` of `narrow`, `wide` and
        // `flags`, all sized `cycles` rows, plus the three uploaded Lagrange
        // weights. It writes `left[t]` and `right[t]`, one slot per thread
        // inside both `cycles`-element allocations; `left` and `right` are
        // fresh and distinct.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;

        let kernel =
            centered_lagrange_kernel::<F>(PRODUCT_UNISKIP_DOMAIN_SIZE, tau_high, uniskip_challenge)
                .map_err(|_| invalid_domain())?;
        let eq =
            DeviceSplitEq::new_with_scaling(context, tau_low, BindingOrder::LowToHigh, kernel)?;

        Ok(Self {
            columns,
            left,
            right,
            eq,
            log_t,
            challenges: Vec::with_capacity(log_t),
        })
    }

    pub const fn rounds(&self) -> usize {
        self.log_t
    }

    pub fn bound_rounds(&self) -> usize {
        self.challenges.len()
    }

    fn bind(&mut self, context: &CudaKernelContext, challenge: F) -> Result<(), CudaError> {
        let scalar = require_fr(challenge)?;
        self.left = context.bind_rows(&self.left, self.left.len(), scalar)?;
        self.right = context.bind_rows(&self.right, self.right.len(), scalar)?;
        self.eq.bind(challenge);
        self.challenges.push(challenge);
        Ok(())
    }

    fn endpoints(&self, context: &CudaKernelContext) -> Result<[F; 2], CudaError> {
        let half = self.left.len() / 2;
        if half == 0 {
            return Err(CudaError::InvariantViolation {
                reason: "the Spartan product remainder has no pairs left to reduce",
            });
        }
        let e_in_len = self.eq.e_in_len();
        let in_bits = e_in_len.max(1).ilog2();
        let threads = half.div_ceil(MESSAGE_STRIP);
        let blocks = u32::try_from(threads.div_ceil(BLOCK as usize).max(1)).map_err(|_| {
            CudaError::InvariantViolation {
                reason: "the Spartan product message launch exceeds a u32 block count",
            }
        })?;
        let mut partials = context.alloc(2 * blocks as usize)?;

        let inner_len = CudaKernelContext::count_of(e_in_len)?;
        let bits = CudaKernelContext::count_of(in_bits as usize)?;
        let count = CudaKernelContext::count_of(half)?;
        let strip = CudaKernelContext::count_of(MESSAGE_STRIP)?;
        let mut builder = context.stream().launch_builder(context.so_message());
        let _ = builder.arg(self.left.limbs());
        let _ = builder.arg(self.right.limbs());
        let _ = builder.arg(self.eq.e_in_current().limbs());
        let _ = builder.arg(self.eq.e_out_current().limbs());
        let _ = builder.arg(&inner_len);
        let _ = builder.arg(&bits);
        let _ = builder.arg(&count);
        let _ = builder.arg(&strip);
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: thread covers pairs `base .. base + MESSAGE_STRIP` and breaks
        // at `half`, so it reads only `left[2p]`, `left[2p + 1]`, `right[2p]`,
        // `right[2p + 1]` for `p < half`, inside both `2 * half`-element
        // tables. The eq lookup reads `e_out[p]` when `e_in_len <= 1` and
        // otherwise `e_in[p & mask]` / `e_out[p >> in_bits]`, bounded because
        // `in_bits` is `e_in`'s log length and `e_in_len * e_out_len == half`
        // for the split-eq at this round. It writes only
        // `partials[lane * blocks + blockIdx.x]` of `2 * blocks`. Shared memory
        // is `BLOCK * LIMBS` u64s, matching `shared_mem_bytes`, and the
        // reduction is outside every early exit.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;

        let totals = DeviceDenseProduct::reduce_lanes(context, partials, 2, blocks)?;
        let host = totals.to_host()?;
        let convert = |value: Fr| {
            fr_into(value).ok_or(CudaError::NotImplemented {
                kernel: "CUDA kernels support only the BN254 scalar field",
            })
        };
        Ok([convert(host[0])?, convert(host[1])?])
    }

    pub fn claims(&self, context: &CudaKernelContext) -> Result<Vec<F>, CudaError> {
        let point: Vec<F> = self.challenges.iter().rev().copied().collect();
        let (e_in, e_out, in_bits) = split_eq_tables(context, &point)?;

        let cycles = self.columns.cycles();
        let threads = cycles.div_ceil(CLAIM_STRIP);
        let blocks = u32::try_from(threads.div_ceil(BLOCK as usize).max(1)).map_err(|_| {
            CudaError::InvariantViolation {
                reason: "the Spartan product claim launch exceeds a u32 block count",
            }
        })?;

        let mut claims = Vec::with_capacity(CLAIM_COLUMNS);
        let mut first = 0usize;
        while first < CLAIM_COLUMNS {
            let lanes = CLAIM_LANES.min(CLAIM_COLUMNS - first);
            let mut partials = context.alloc(lanes * blocks as usize)?;
            let first_arg = CudaKernelContext::count_of(first)?;
            let lanes_arg = CudaKernelContext::count_of(lanes)?;
            let count = CudaKernelContext::count_of(cycles)?;
            let strip = CudaKernelContext::count_of(CLAIM_STRIP)?;
            let bits = CudaKernelContext::count_of(in_bits)?;
            let mut builder = context.stream().launch_builder(context.sp_claims());
            let _ = builder.arg(self.columns.narrow());
            let _ = builder.arg(self.columns.wide());
            let _ = builder.arg(self.columns.flags());
            let _ = builder.arg(self.columns.layout());
            let _ = builder.arg(e_in.limbs());
            let _ = builder.arg(e_out.limbs());
            let _ = builder.arg(&bits);
            let _ = builder.arg(&SIGN_BIT_BASE);
            let _ = builder.arg(&first_arg);
            let _ = builder.arg(&lanes_arg);
            let _ = builder.arg(&count);
            let _ = builder.arg(&strip);
            let _ = builder.arg(partials.limbs_mut());
            // SAFETY: thread covers cycles `base .. base + CLAIM_STRIP` and
            // breaks at `cycles`, so every row read is inside the three column
            // buffers. It reads `layout[first + lane]` for `lane < lanes`, and
            // `first + lanes <= CLAIM_COLUMNS` is the loop's own bound, so the
            // read is inside `layout`'s `CLAIM_COLUMNS` entries. The eq lookup
            // is bounded by `in_bits` coming from `e_in`'s length with
            // `t < cycles`. It writes only `partials[lane * blocks +
            // blockIdx.x]` of `lanes * blocks`, and `lanes <= SP_CLAIM_LANES`
            // bounds the kernel's fixed-size accumulator array. Shared memory
            // is `BLOCK * LIMBS` u64s, matching `shared_mem_bytes`, and the
            // reduction is outside every early exit.
            let _ = unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
                })
            }?;

            let lanes_u32 = CudaKernelContext::count_of(lanes)?;
            let totals = DeviceDenseProduct::reduce_lanes(context, partials, lanes_u32, blocks)?;
            for value in totals.to_host()? {
                claims.push(fr_into(value).ok_or(CudaError::NotImplemented {
                    kernel: "CUDA kernels support only the BN254 scalar field",
                })?);
            }
            first += lanes;
        }
        Ok(claims)
    }

    pub fn opening_values(
        &self,
        context: &CudaKernelContext,
    ) -> Result<BTreeMap<JoltOpeningId, F>, CudaError> {
        let claims = self.claims(context)?;
        if claims.len() != witness::CLAIM_COLUMNS {
            return Err(CudaError::LengthMismatch {
                expected: witness::CLAIM_COLUMNS,
                got: claims.len(),
            });
        }
        Ok(claim_openings().into_iter().zip(claims).collect())
    }
}

pub struct SpartanProductRemainderKernel<F: Field> {
    context: &'static CudaKernelContext,
    state: DeviceProductRemainder<F>,
    degree: usize,
}

impl<F: Field> SpartanProductRemainderKernel<F> {
    pub const fn new(
        context: &'static CudaKernelContext,
        state: DeviceProductRemainder<F>,
        degree: usize,
    ) -> Self {
        Self {
            context,
            state,
            degree,
        }
    }

    pub fn openings(&self) -> Result<BTreeMap<JoltOpeningId, F>, CudaError> {
        self.state.opening_values(self.context)
    }

    pub fn bound_rounds(&self) -> usize {
        self.state.bound_rounds()
    }
}

impl<F: Field> ProveRounds<F> for SpartanProductRemainderKernel<F> {
    fn num_rounds(&self) -> usize {
        self.state.rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let failed = || SumcheckError::MissingEvaluationSource {
            kind: "cuda Spartan product remainder",
        };
        if let Some(challenge) = bind {
            self.state
                .bind(self.context, challenge)
                .map_err(|_| failed())?;
        }
        let [q0, q2] = self.state.endpoints(self.context).map_err(|_| failed())?;
        let mut coefficients = self
            .state
            .eq
            .gruen_poly_deg_3(q0, q2, previous_claim)
            .into_coefficients();
        coefficients.resize(self.degree + 1, F::zero());
        Ok(UnivariatePoly::new(coefficients))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.state
            .bind(self.context, bind)
            .map_err(|_| SumcheckError::MissingEvaluationSource {
                kind: "cuda Spartan product remainder",
            })
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::JoltOneHotConfig;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_witness::collect_bundles;
    use jolt_witness::witnesses::ToField;
    use proptest::prelude::*;

    use super::{
        branch_flag_product, centered_lagrange_evals, jump_flag_product,
        left_instruction_input_product, lookup_output_product, next_is_noop_product,
        right_instruction_input_product, DeviceProductRemainder, PRODUCT_UNISKIP_DOMAIN_SIZE,
    };
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{arb_point, fr, with_r1cs_witness};
    use crate::cuda::spartan_product::columns::DeviceProductColumns;
    use crate::cuda::spartan_product::witness::{self, SpartanProductWitness};
    use crate::reference::views::dense_view;

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
        fn product_factors_match_cpu(
            seed in any::<u64>(),
            tau_low in arb_point(LOG_T),
            tau_high in any::<u64>().prop_map(fr),
            uniskip_challenge in any::<u64>().prop_map(fr),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };

            with_r1cs_witness(LOG_T, RAM_K, one_hot(), seed, |witness| {
                let rows = collect_bundles::<SpartanProductWitness>(witness, 1usize << LOG_T)
                    .expect("the fixture serves every product bundle field");
                let packed = witness::pack(&rows);
                let columns = DeviceProductColumns::new(context, &packed)
                    .expect("upload the packed product columns");
                let device = DeviceProductRemainder::<Fr>::new(
                    context, columns, &tau_low, tau_high, uniskip_challenge, LOG_T,
                ).expect("the device product remainder");

                let weights =
                    centered_lagrange_evals::<Fr>(PRODUCT_UNISKIP_DOMAIN_SIZE, uniskip_challenge)
                        .expect("the centered Lagrange weights");
                let view = |opening| {
                    dense_view::<Fr>(witness, opening).expect("the fixture serves the column")
                };
                let left_columns = [
                    view(left_instruction_input_product()),
                    view(lookup_output_product()),
                    view(jump_flag_product()),
                ];
                let right_columns = [
                    view(right_instruction_input_product()),
                    view(branch_flag_product()),
                    view(next_is_noop_product())
                        .iter()
                        .map(|value| Fr::from_u64(1) - *value)
                        .collect::<Vec<Fr>>(),
                ];
                let collapse = |columns: &[Vec<Fr>; 3]| -> Vec<Fr> {
                    (0..1usize << LOG_T)
                        .map(|cycle| {
                            weights
                                .iter()
                                .zip(columns)
                                .map(|(weight, column)| *weight * column[cycle])
                                .sum()
                        })
                        .collect()
                };

                prop_assert_eq!(
                    device.left.to_host().expect("download the left factor"),
                    collapse(&left_columns),
                    "the left factor table diverged"
                );
                prop_assert_eq!(
                    device.right.to_host().expect("download the right factor"),
                    collapse(&right_columns),
                    "the right factor table diverged"
                );
                Ok(())
            })?;
        }

        #[test]
        fn product_factors_match_cpu_on_signed_rows(
            seed in any::<u64>(),
            tau_low in arb_point(LOG_T),
            tau_high in any::<u64>().prop_map(fr),
            uniskip_challenge in any::<u64>().prop_map(fr),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };

            let rows = witness::sample_rows(seed, 1usize << LOG_T);
            let packed = witness::pack(&rows);
            let columns = DeviceProductColumns::new(context, &packed)
                .expect("upload the packed product columns");
            let device = DeviceProductRemainder::<Fr>::new(
                context, columns, &tau_low, tau_high, uniskip_challenge, LOG_T,
            ).expect("the device product remainder");

            let weights =
                centered_lagrange_evals::<Fr>(PRODUCT_UNISKIP_DOMAIN_SIZE, uniskip_challenge)
                    .expect("the centered Lagrange weights");
            let left_columns = [
                rows.iter().map(|row| row.left_instruction_input.to_field()).collect(),
                rows.iter().map(|row| row.lookup_output.to_field()).collect(),
                rows.iter().map(|row| row.jump.to_field()).collect::<Vec<Fr>>(),
            ];
            let right_columns = [
                rows.iter().map(|row| row.right_instruction_input.to_field()).collect(),
                rows.iter().map(|row| row.branch.to_field()).collect(),
                rows.iter()
                    .map(|row| Fr::from_u64(1) - ToField::to_field::<Fr>(row.next_is_noop))
                    .collect::<Vec<Fr>>(),
            ];
            let collapse = |columns: &[Vec<Fr>; 3]| -> Vec<Fr> {
                (0..1usize << LOG_T)
                    .map(|cycle| {
                        weights
                            .iter()
                            .zip(columns)
                            .map(|(weight, column)| *weight * column[cycle])
                            .sum()
                    })
                    .collect()
            };

            prop_assert_eq!(
                device.left.to_host().expect("download the left factor"),
                collapse(&left_columns),
                "the left factor table diverged on signed rows"
            );
            prop_assert_eq!(
                device.right.to_host().expect("download the right factor"),
                collapse(&right_columns),
                "the right factor table diverged on signed rows"
            );
        }
    }
}
