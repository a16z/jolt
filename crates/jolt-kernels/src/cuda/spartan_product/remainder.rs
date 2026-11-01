#[cfg(feature = "allocative")]
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::Key;
#[cfg(feature = "allocative")]
use allocative::Visitor;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_claims::protocols::jolt::geometry::dimensions::PRODUCT_UNISKIP_DOMAIN_SIZE;
use jolt_claims::protocols::jolt::geometry::spartan::{
    branch_flag_product, jump_flag_product, left_instruction_input_product, lookup_output_product,
    next_is_noop_product, right_instruction_input_product, virtual_instruction_product,
    write_lookup_output_to_rd_product,
};
use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_field::{Fr, JoltField};
use jolt_poly::lagrange::{centered_lagrange_evals, centered_lagrange_kernel};
use jolt_poly::{BindingOrder, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use std::collections::BTreeMap;
use witness::CLAIM_COLUMNS as WitnessClaimColumns;

use super::columns::DeviceProductColumns;
use super::witness::{self, BRANCH_BIT, CLAIM_COLUMNS, JUMP_BIT, NEXT_IS_NOOP_BIT, SIGN_BIT_BASE};
use crate::cuda::common::context::{context_for, CudaKernelContext, BLOCK};
use crate::cuda::common::device::{fr_into, require_fr, require_fr_slice, DeviceFrVec, LIMBS};
use crate::cuda::common::devices::{fan_out, DeviceTask};
use crate::cuda::common::error::backend;
use crate::cuda::common::error::CudaError;
use crate::cuda::common::primitives::reduce_lanes;
use crate::cuda::common::split_eq::{split_eq_tables_window, DeviceSplitEq};

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

pub(crate) struct ProductShard<F: JoltField> {
    pub(crate) ordinal: usize,
    pub(crate) columns: DeviceProductColumns,
    pub(crate) left: DeviceFrVec,
    pub(crate) right: DeviceFrVec,
    pub(crate) eq: Option<DeviceSplitEq<F>>,
}

pub struct DeviceProductRemainder<F: JoltField> {
    shards: Vec<ProductShard<F>>,
    collapsed: Option<(DeviceFrVec, DeviceFrVec)>,
    eq: DeviceSplitEq<F>,
    pending_endpoints: Option<[F; 3]>,
    log_t: usize,
    local_rounds: usize,
    challenges: Vec<F>,
}

impl<F: JoltField> DeviceProductRemainder<F> {
    pub fn new(
        context: &CudaKernelContext,
        columns: Vec<(usize, DeviceProductColumns)>,
        tau_low: &[F],
        tau_high: F,
        uniskip_challenge: F,
        log_t: usize,
    ) -> Result<Self, CudaError> {
        let invalid_domain = || CudaError::InvariantViolation {
            reason: "the Spartan product uni-skip domain is not a valid centered integer domain",
        };
        let count = columns.len();
        if count == 0 || !count.is_power_of_two() {
            return Err(CudaError::InvariantViolation {
                reason: "the Spartan product remainder needs a power-of-two window count",
            });
        }
        let tail_rounds = count.trailing_zeros() as usize;
        if tail_rounds > log_t {
            return Err(CudaError::InvariantViolation {
                reason: "the Spartan product remainder cannot split more windows than cycle rounds",
            });
        }
        let weights = centered_lagrange_evals::<F>(PRODUCT_UNISKIP_DOMAIN_SIZE, uniskip_challenge)
            .map_err(|_| invalid_domain())?;
        let kernel =
            centered_lagrange_kernel::<F>(PRODUCT_UNISKIP_DOMAIN_SIZE, tau_high, uniskip_challenge)
                .map_err(|_| invalid_domain())?;

        let eq =
            DeviceSplitEq::new_with_scaling(context, tau_low, BindingOrder::LowToHigh, kernel)?;
        let whole_eq = &eq;
        let weights = &weights;
        type PreparedShard<F> = (ProductShard<F>, [F; 3]);
        let tasks: Vec<DeviceTask<'_, PreparedShard<F>, CudaError>> = columns
            .into_iter()
            .enumerate()
            .map(|(shard, (ordinal, columns))| {
                let task: DeviceTask<'_, PreparedShard<F>, CudaError> = Box::new(move || {
                    let device = context_for(ordinal).ok_or(absent())?;
                    let eq = if count == 1 {
                        None
                    } else {
                        Some(DeviceSplitEq::new_window_with_scaling(
                            device,
                            tau_low,
                            BindingOrder::LowToHigh,
                            kernel,
                            shard,
                            count,
                        )?)
                    };
                    let (left, right, endpoints) = Self::prepare_window(
                        device,
                        &columns,
                        weights,
                        eq.as_ref().unwrap_or(whole_eq),
                    )?;
                    Ok((
                        ProductShard {
                            ordinal,
                            columns,
                            left,
                            right,
                            eq,
                        },
                        endpoints,
                    ))
                });
                task
            })
            .collect();
        let mut shards = Vec::with_capacity(count);
        let mut endpoints = [F::zero(); 3];
        for (shard, values) in fan_out(tasks)? {
            shards.push(shard);
            for (sum, value) in endpoints.iter_mut().zip(values) {
                *sum += value;
            }
        }

        Ok(Self {
            shards,
            collapsed: None,
            eq,
            pending_endpoints: Some(endpoints),
            log_t,
            local_rounds: log_t - tail_rounds,
            challenges: Vec::with_capacity(log_t),
        })
    }

    fn prepare_window(
        context: &CudaKernelContext,
        columns: &DeviceProductColumns,
        weights: &[F],
        eq: &DeviceSplitEq<F>,
    ) -> Result<(DeviceFrVec, DeviceFrVec, [F; 3]), CudaError> {
        let cycles = columns.cycles();
        if cycles < 2
            || !cycles.is_power_of_two()
            || eq.e_in_current().len() * eq.e_out_current().len() != cycles / 2
        {
            return Err(CudaError::InvariantViolation {
                reason: "the product preparation equality factors must cover the local cycle pairs",
            });
        }
        let device_weights = context.upload(require_fr_slice(weights)?)?;

        let mut left = context.alloc(cycles)?;
        let mut right = context.alloc(cycles)?;
        let count = CudaKernelContext::count_of(cycles)?;
        let blocks = (count / 2).div_ceil(BLOCK);
        let mut partials = context.alloc(3 * blocks as usize)?;
        let inner_len = CudaKernelContext::count_of(eq.e_in_len())?;
        let bits = inner_len.max(1).ilog2();
        let mut builder = context.stream().launch_builder(context.sp_prepare());
        let _ = builder.arg(columns.narrow());
        let _ = builder.arg(columns.wide());
        let _ = builder.arg(columns.flags());
        let _ = builder.arg(device_weights.limbs());
        let _ = builder.arg(&JUMP_BIT);
        let _ = builder.arg(&BRANCH_BIT);
        let _ = builder.arg(&NEXT_IS_NOOP_BIT);
        let _ = builder.arg(&SIGN_BIT_BASE);
        let _ = builder.arg(&count);
        let _ = builder.arg(eq.e_in_current().limbs());
        let _ = builder.arg(eq.e_out_current().limbs());
        let _ = builder.arg(&inner_len);
        let _ = builder.arg(&bits);
        let _ = builder.arg(left.limbs_mut());
        let _ = builder.arg(right.limbs_mut());
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: columns own cycles rows and weights spans the three product
        // lanes. The guard bounds both rows of each pair; each writes distinct
        // entries of the two fresh output tables. The checked equality shape
        // covers every pair. Each block writes three partials, with BLOCK *
        // LIMBS u64 words of shared reduction scratch.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;
        let endpoints = Self::reduce_endpoints(context, partials, blocks)?;
        Ok((left, right, endpoints))
    }

    pub const fn rounds(&self) -> usize {
        self.log_t
    }

    pub fn bound_rounds(&self) -> usize {
        self.challenges.len()
    }

    fn bind(&mut self, context: &CudaKernelContext, challenge: F) -> Result<(), CudaError> {
        let scalar = require_fr(challenge)?;
        if let Some((left, right)) = &mut self.collapsed {
            *left = context.bind_rows(left, left.len(), scalar)?;
            *right = context.bind_rows(right, right.len(), scalar)?;
        } else {
            let tasks: Vec<DeviceTask<'_, (), CudaError>> = self
                .shards
                .iter_mut()
                .map(|shard| {
                    let task: DeviceTask<'_, (), CudaError> = Box::new(move || {
                        let device = context_for(shard.ordinal).ok_or(absent())?;
                        shard.left = device.bind_rows(&shard.left, shard.left.len(), scalar)?;
                        shard.right = device.bind_rows(&shard.right, shard.right.len(), scalar)?;
                        if let Some(eq) = &mut shard.eq {
                            eq.bind(challenge);
                        }
                        Ok(())
                    });
                    task
                })
                .collect();
            let _ = fan_out(tasks)?;
        }
        self.pending_endpoints = None;
        self.eq.bind(challenge);
        self.challenges.push(challenge);
        if self.shards.len() > 1
            && self.collapsed.is_none()
            && self.challenges.len() == self.local_rounds
        {
            self.collapse(context)?;
        }
        Ok(())
    }

    fn collapse(&mut self, context: &CudaKernelContext) -> Result<(), CudaError> {
        type Tables = (Vec<Fr>, Vec<Fr>);
        let tasks: Vec<DeviceTask<'_, Tables, CudaError>> = self
            .shards
            .iter()
            .map(|shard| {
                let task: DeviceTask<'_, Tables, CudaError> = Box::new(move || {
                    let _ = context_for(shard.ordinal).ok_or(absent())?;
                    Ok((shard.left.to_host()?, shard.right.to_host()?))
                });
                task
            })
            .collect();
        let mut left = Vec::new();
        let mut right = Vec::new();
        for (window_left, window_right) in fan_out(tasks)? {
            if window_left.len() != window_right.len() {
                return Err(CudaError::LengthMismatch {
                    expected: window_left.len(),
                    got: window_right.len(),
                });
            }
            left.extend_from_slice(&window_left);
            right.extend_from_slice(&window_right);
        }
        self.collapsed = Some((context.upload(&left)?, context.upload(&right)?));
        for shard in &mut self.shards {
            let device = context_for(shard.ordinal).ok_or(absent())?;
            shard.left = device.alloc(0)?;
            shard.right = device.alloc(0)?;
            shard.eq = None;
        }
        Ok(())
    }

    fn endpoints(&self, context: &CudaKernelContext) -> Result<[F; 3], CudaError> {
        if let Some(endpoints) = self.pending_endpoints {
            return Ok(endpoints);
        }
        if let Some((left, right)) = &self.collapsed {
            return Self::window_endpoints(context, left, right, &self.eq);
        }
        let whole = &self.eq;
        let tasks: Vec<DeviceTask<'_, [F; 3], CudaError>> = self
            .shards
            .iter()
            .map(|shard| {
                let task: DeviceTask<'_, [F; 3], CudaError> = Box::new(move || {
                    let device = context_for(shard.ordinal).ok_or(absent())?;
                    let eq = shard.eq.as_ref().unwrap_or(whole);
                    Self::window_endpoints(device, &shard.left, &shard.right, eq)
                });
                task
            })
            .collect();
        let mut total = [F::zero(); 3];
        for part in fan_out(tasks)? {
            total[0] += part[0];
            total[1] += part[1];
            total[2] += part[2];
        }
        Ok(total)
    }

    fn window_endpoints(
        context: &CudaKernelContext,
        left: &DeviceFrVec,
        right: &DeviceFrVec,
        eq: &DeviceSplitEq<F>,
    ) -> Result<[F; 3], CudaError> {
        let half = left.len() / 2;
        if half == 0 {
            return Err(CudaError::InvariantViolation {
                reason: "the Spartan product remainder has no pairs left to reduce",
            });
        }
        let e_in_len = eq.e_in_len();
        let in_bits = e_in_len.max(1).ilog2();
        let threads = half.div_ceil(MESSAGE_STRIP);
        let blocks = u32::try_from(threads.div_ceil(BLOCK as usize).max(1)).map_err(|_| {
            CudaError::InvariantViolation {
                reason: "the Spartan product message launch exceeds a u32 block count",
            }
        })?;
        let mut partials = context.alloc(3 * blocks as usize)?;

        let inner_len = CudaKernelContext::count_of(e_in_len)?;
        let bits = CudaKernelContext::count_of(in_bits as usize)?;
        let count = CudaKernelContext::count_of(half)?;
        let strip = CudaKernelContext::count_of(MESSAGE_STRIP)?;
        let mut builder = context
            .stream()
            .launch_builder(context.gruen_pair_message());
        let _ = builder.arg(left.limbs());
        let _ = builder.arg(right.limbs());
        let _ = builder.arg(eq.e_in_current().limbs());
        let _ = builder.arg(eq.e_out_current().limbs());
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
        // `partials[lane * blocks + blockIdx.x]` of `3 * blocks`. Shared memory
        // is `BLOCK * LIMBS` u64s, matching `shared_mem_bytes`, and the
        // reduction is outside every early exit.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;

        Self::reduce_endpoints(context, partials, blocks)
    }

    fn reduce_endpoints(
        context: &CudaKernelContext,
        partials: DeviceFrVec,
        blocks: u32,
    ) -> Result<[F; 3], CudaError> {
        let totals = reduce_lanes(context, partials, 3, blocks)?;
        let host = totals.to_host()?;
        let convert = |value: Fr| {
            fr_into(value).ok_or(CudaError::NotImplemented {
                kernel: "CUDA kernels support only the BN254 scalar field",
            })
        };
        Ok([convert(host[0])?, convert(host[1])?, convert(host[2])?])
    }

    pub fn claims(&self, context: &CudaKernelContext) -> Result<Vec<F>, CudaError> {
        let _ = context;
        let point: Vec<F> = self.challenges.iter().rev().copied().collect();
        let point = &point;
        let shards = self.shards.len();
        let tasks: Vec<DeviceTask<'_, Vec<F>, CudaError>> = self
            .shards
            .iter()
            .enumerate()
            .map(|(shard, window)| {
                let task: DeviceTask<'_, Vec<F>, CudaError> = Box::new(move || {
                    let device = context_for(window.ordinal).ok_or(absent())?;
                    Self::window_claims(device, &window.columns, point, shard, shards)
                });
                task
            })
            .collect();
        let mut total = vec![F::zero(); CLAIM_COLUMNS];
        for part in fan_out(tasks)? {
            if part.len() != CLAIM_COLUMNS {
                return Err(CudaError::LengthMismatch {
                    expected: CLAIM_COLUMNS,
                    got: part.len(),
                });
            }
            for (slot, value) in total.iter_mut().zip(&part) {
                *slot += *value;
            }
        }
        Ok(total)
    }

    fn window_claims(
        context: &CudaKernelContext,
        columns: &DeviceProductColumns,
        point: &[F],
        shard: usize,
        shards: usize,
    ) -> Result<Vec<F>, CudaError> {
        let (e_in, e_out, in_bits) = split_eq_tables_window(context, point, shard, shards)?;

        let cycles = columns.cycles();
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
            let _ = builder.arg(columns.narrow());
            let _ = builder.arg(columns.wide());
            let _ = builder.arg(columns.flags());
            let _ = builder.arg(columns.layout());
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
            let totals = reduce_lanes(context, partials, lanes_u32, blocks)?;
            for value in totals.to_host()? {
                claims.push(fr_into(value).ok_or(CudaError::NotImplemented {
                    kernel: "CUDA kernels support only the BN254 scalar field",
                })?);
            }
            first += lanes;
        }
        Ok(claims)
    }

    #[cfg(test)]
    pub(crate) fn window_factors_host(
        &self,
        shard: usize,
    ) -> Result<(Vec<Fr>, Vec<Fr>), CudaError> {
        let window = self
            .shards
            .get(shard)
            .ok_or(CudaError::InvariantViolation {
                reason: "the Spartan product remainder has no such window",
            })?;
        Ok((window.left.to_host()?, window.right.to_host()?))
    }

    pub fn opening_values(
        &self,
        context: &CudaKernelContext,
    ) -> Result<BTreeMap<JoltOpeningId, F>, CudaError> {
        let claims = self.claims(context)?;
        if claims.len() != WitnessClaimColumns {
            return Err(CudaError::LengthMismatch {
                expected: WitnessClaimColumns,
                got: claims.len(),
            });
        }
        Ok(claim_openings().into_iter().zip(claims).collect())
    }
}

const fn absent() -> CudaError {
    CudaError::InvariantViolation {
        reason: "a Spartan product remainder window names an absent device",
    }
}

pub struct SpartanProductRemainderKernel<F: JoltField> {
    context: &'static CudaKernelContext,
    state: DeviceProductRemainder<F>,
    degree: usize,
}

#[cfg(feature = "allocative")]
impl<F: JoltField> Allocative for SpartanProductRemainderKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        let state = &self.state;
        visitor.visit_simple(
            Key::new("columns"),
            state
                .shards
                .iter()
                .map(|shard| shard.columns.device_bytes())
                .sum(),
        );
        visitor.visit_simple(
            Key::new("left_right"),
            state
                .shards
                .iter()
                .map(|shard| shard.left.device_bytes() + shard.right.device_bytes())
                .sum::<usize>()
                + state.collapsed.as_ref().map_or(0, |(left, right)| {
                    left.device_bytes() + right.device_bytes()
                }),
        );
        visitor.visit_simple(
            Key::new("eq"),
            state.eq.device_bytes()
                + state
                    .shards
                    .iter()
                    .map(|shard| shard.eq.as_ref().map_or(0, DeviceSplitEq::device_bytes))
                    .sum::<usize>(),
        );
        visitor.exit();
    }
}

impl<F: JoltField> SpartanProductRemainderKernel<F> {
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

impl<F: JoltField> ProveRounds<F> for SpartanProductRemainderKernel<F> {
    fn num_rounds(&self) -> usize {
        self.state.rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.state
                .bind(self.context, challenge)
                .map_err(backend("cuda Spartan product remainder"))?;
        }
        let [q0, q2, q1] = self
            .state
            .endpoints(self.context)
            .map_err(backend("cuda Spartan product remainder"))?;
        let mut coefficients = self
            .state
            .eq
            .gruen_poly_deg_3(q0, q2, previous_claim, q1)
            .map_err(backend("CUDA quotient interpolation"))?
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
    use jolt_field::{Fr, Ring};
    use jolt_poly::{EqPolynomial, Polynomial};
    use jolt_sumcheck::ProveRounds;
    use jolt_witness::collect_bundles;
    use jolt_witness::witnesses::ToField;
    use proptest::prelude::*;

    use super::{
        branch_flag_product, centered_lagrange_evals, jump_flag_product,
        left_instruction_input_product, lookup_output_product, next_is_noop_product,
        right_instruction_input_product, DeviceProductRemainder, SpartanProductRemainderKernel,
        CLAIM_COLUMNS, PRODUCT_UNISKIP_DOMAIN_SIZE,
    };
    use crate::cuda::common::context::{context_for, shared_context};
    use crate::cuda::common::devices::witness_windows;
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

    #[test]
    fn product_rounds_match_dense_polynomials_at_extreme_operands() {
        let Some(context) = shared_context() else {
            return;
        };
        for log_t in [1usize, 2, 13] {
            let cycles = 1usize << log_t;
            let mut rows = witness::sample_rows(71, cycles);
            for (i, row) in rows.iter_mut().enumerate() {
                let k = (i / 4) as u64;
                row.left_instruction_input.0 = [u64::MAX - k, 0, k + 1, 1][i % 4];
                row.lookup_output.0 = [u64::MAX - k, 1, 0, k + 3][i % 4];
                row.right_instruction_input.0 =
                    [i128::MIN + i128::from(k), i128::MAX - i128::from(k), -1, 0][i % 4];
            }
            for case in 0..4 {
                let tau_low: Vec<_> = (0..log_t)
                    .map(|i| Fr::from_u64(if case < 2 { case } else { 11 + i as u64 }))
                    .collect();
                let (tau_high, uniskip) = if case == 3 {
                    (Fr::from_u64(0), Fr::from_u64(1))
                } else {
                    (Fr::from_u64(61), Fr::from_u64(97))
                };
                let weights = centered_lagrange_evals::<Fr>(PRODUCT_UNISKIP_DOMAIN_SIZE, uniskip)
                    .expect("product Lagrange weights");
                let scaling = super::centered_lagrange_kernel::<Fr>(
                    PRODUCT_UNISKIP_DOMAIN_SIZE,
                    tau_high,
                    uniskip,
                )
                .expect("product scaling");
                if case == 3 {
                    assert_eq!(scaling, Fr::from_u64(0));
                }
                let mut left = Polynomial::new(
                    rows.iter()
                        .map(|r| {
                            weights[0] * r.left_instruction_input.to_field::<Fr>()
                                + weights[1] * r.lookup_output.to_field::<Fr>()
                                + weights[2] * r.jump.to_field::<Fr>()
                        })
                        .collect(),
                );
                let mut right = Polynomial::new(
                    rows.iter()
                        .map(|r| {
                            weights[0] * r.right_instruction_input.to_field::<Fr>()
                                + weights[1] * r.branch.to_field::<Fr>()
                                + weights[2] * (Fr::from_u64(1) - r.next_is_noop.to_field::<Fr>())
                        })
                        .collect(),
                );
                let mut eq = Polynomial::new(EqPolynomial::<Fr>::evals(&tau_low, Some(scaling)));
                let columns = witness_windows(cycles)
                    .iter()
                    .enumerate()
                    .map(|(ordinal, window)| {
                        let device = context_for(ordinal).expect("configured device");
                        let columns = DeviceProductColumns::new(
                            device,
                            &witness::pack(&rows[window.start..window.end()]),
                        )
                        .expect("product columns");
                        (ordinal, columns)
                    })
                    .collect();
                let state = DeviceProductRemainder::new(
                    context, columns, &tau_low, tau_high, uniskip, log_t,
                )
                .expect("product remainder");
                let mut kernel = SpartanProductRemainderKernel::new(context, state, 3);
                let mut previous_challenge = None;
                for round in 0..log_t {
                    let mut expected = [Fr::from_u64(0); 4];
                    for pair in 0..left.len() / 2 {
                        for (x, sum) in expected.iter_mut().enumerate() {
                            let at = |p: &Polynomial<Fr>| {
                                p.evals()[2 * pair]
                                    + Fr::from_u64(x as u64)
                                        * (p.evals()[2 * pair + 1] - p.evals()[2 * pair])
                            };
                            *sum += at(&eq) * at(&left) * at(&right);
                        }
                    }
                    let actual = kernel
                        .prove_round(previous_challenge, round, expected[0] + expected[1])
                        .expect("product round");
                    for (x, value) in expected.into_iter().enumerate() {
                        assert_eq!(
                            actual.evaluate(Fr::from_u64(x as u64)),
                            value,
                            "log_t={log_t}, case={case}, round={round}, x={x}"
                        );
                    }
                    let r = Fr::from_u64(if case < 2 {
                        (round % 2) as u64
                    } else {
                        17 + round as u64
                    });
                    previous_challenge = Some(r);
                    for p in [&mut eq, &mut left, &mut right] {
                        let _ = p.bind_low_to_high_in_place(r);
                    }
                }
                kernel
                    .finish_rounds(previous_challenge.expect("nonempty cycle point"))
                    .expect("final product bind");
                let _ = kernel.openings().expect("fully bound product openings");
            }
        }
    }

    #[test]
    fn windowed_product_remainder_matches_the_whole_domain_round_for_round() {
        let Some(context) = shared_context() else {
            return;
        };
        let cycles = 1usize << LOG_T;
        let rows = witness::sample_rows(0x9E3D, cycles);
        let tau_low: Vec<Fr> = (0..LOG_T).map(|i| fr(23 + 5 * i as u64)).collect();
        let tau_high = fr(61);
        let uniskip_challenge = fr(97);

        let build = |base: usize, len: usize| {
            DeviceProductColumns::new(context, &witness::pack(&rows[base..base + len]))
                .expect("upload a packed product window")
        };

        for shards in [2usize, 4, 8] {
            let mut expected = DeviceProductRemainder::<Fr>::new(
                context,
                vec![(0, build(0, cycles))],
                &tau_low,
                tau_high,
                uniskip_challenge,
                LOG_T,
            )
            .expect("whole-domain product remainder");
            let len = cycles / shards;
            let windows: Vec<(usize, DeviceProductColumns)> = (0..shards)
                .map(|shard| (0usize, build(shard * len, len)))
                .collect();
            let mut got = DeviceProductRemainder::<Fr>::new(
                context,
                windows,
                &tau_low,
                tau_high,
                uniskip_challenge,
                LOG_T,
            )
            .expect("windowed product remainder");

            for round in 0..LOG_T {
                let want = expected.endpoints(context).expect("whole endpoints");
                let have = got.endpoints(context).expect("windowed endpoints");
                assert_eq!(
                    have, want,
                    "shards={shards} round {round}: the endpoint pair diverged",
                );
                let challenge = fr(500 + 23 * round as u64);
                expected.bind(context, challenge).expect("whole bind");
                got.bind(context, challenge).expect("windowed bind");
            }

            let want = expected.claims(context).expect("whole claims");
            let have = got.claims(context).expect("windowed claims");
            assert_eq!(have, want, "shards={shards}: the claim pass diverged");
            assert_eq!(want.len(), CLAIM_COLUMNS);
            assert_ne!(
                want.first().copied(),
                Some(Fr::from_u64(0)),
                "a degenerate fixture would hide a divergence",
            );
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
                    context, vec![(0, columns)], &tau_low, tau_high, uniskip_challenge, LOG_T,
                ).expect("the device product remainder");
                let (got_left, got_right) = device
                    .window_factors_host(0)
                    .expect("download the factor tables");

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
                    got_left,
                    collapse(&left_columns),
                    "the left factor table diverged"
                );
                prop_assert_eq!(
                    got_right,
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
                context, vec![(0, columns)], &tau_low, tau_high, uniskip_challenge, LOG_T,
            ).expect("the device product remainder");
            let (got_left, got_right) = device
                .window_factors_host(0)
                .expect("download the factor tables");

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
                got_left,
                collapse(&left_columns),
                "the left factor table diverged on signed rows"
            );
            prop_assert_eq!(
                got_right,
                collapse(&right_columns),
                "the right factor table diverged on signed rows"
            );
        }
    }
}
