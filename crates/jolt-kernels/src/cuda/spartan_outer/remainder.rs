use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_claims::protocols::jolt::geometry::dimensions::OUTER_UNISKIP_DOMAIN_SIZE;
use jolt_claims::protocols::jolt::geometry::spartan::{outer_opening, SpartanOuterDimensions};
use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_field::{Field, Fr};
use jolt_poly::lagrange::centered_lagrange_kernel;
use jolt_poly::{BindingOrder, UnivariatePoly};
use jolt_r1cs::constraint::ConstraintMatrices;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use std::collections::BTreeMap;

use super::columns::{DeviceR1csInputs, LinearForms};
use super::uniskip::push_forms;
use super::witness::VARIABLES;
use crate::cuda::common::context::{context_for, CudaKernelContext, BLOCK};
use crate::cuda::common::device::{fr_into, require_fr, DeviceFrVec, LIMBS};
use crate::cuda::common::devices::{fan_out, CycleWindow, DeviceTask};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::primitives::reduce_lanes;
use crate::cuda::common::split_eq::split_eq_tables_window;
use crate::cuda::common::split_eq::DeviceSplitEq;

pub const CLAIM_LANES: usize = 4;

pub const MESSAGE_STRIP: usize = 1;

pub const CLAIM_STRIP: usize = 16;

struct RemainderShard<F: Field> {
    ordinal: usize,
    inputs: DeviceR1csInputs,
    cycles: usize,
    az: DeviceFrVec,
    bz: DeviceFrVec,
    eq: DeviceSplitEq<F>,
}

pub struct DeviceRemainder<F: Field> {
    shards: Vec<RemainderShard<F>>,
    collapsed: Option<(DeviceFrVec, DeviceFrVec)>,
    eq: DeviceSplitEq<F>,
    log_t: usize,
    local_rounds: usize,
    challenges: Vec<F>,
}

impl<F: Field> DeviceRemainder<F> {
    pub fn new(
        context: &'static CudaKernelContext,
        windows: Vec<(DeviceR1csInputs, CycleWindow)>,
        matrices: &ConstraintMatrices<F>,
        tau: &[F],
        log_t: usize,
        uniskip_challenge: F,
    ) -> Result<Self, CudaError> {
        let count = windows.len();
        if count == 0 || !count.is_power_of_two() {
            return Err(CudaError::InvariantViolation {
                reason: "a Spartan outer remainder needs a power-of-two window count",
            });
        }
        let tail_rounds = count.trailing_zeros() as usize;
        let columns: Vec<usize> = (1..=VARIABLES).collect();
        let mut forms = LinearForms::new();
        push_forms(&mut forms, matrices, &columns, uniskip_challenge, F::zero())?;
        push_forms(&mut forms, matrices, &columns, uniskip_challenge, F::one())?;

        let kernel = centered_lagrange_kernel::<F>(
            OUTER_UNISKIP_DOMAIN_SIZE,
            tau[log_t + 1],
            uniskip_challenge,
        )
        .map_err(|_| CudaError::InvariantViolation {
            reason: "the Spartan outer uni-skip domain is not a valid centered integer domain",
        })?;
        let cycle_point = &tau[..=log_t];

        let mut shards = Vec::with_capacity(count);
        for (ordinal, (inputs, window)) in windows.into_iter().enumerate() {
            let device = context_for(ordinal).ok_or(absent())?;
            if inputs.cycles() < window.len {
                return Err(CudaError::LengthMismatch {
                    expected: window.len,
                    got: inputs.cycles(),
                });
            }
            let (az, bz) = Self::factors(device, &forms, &inputs, window.len)?;
            shards.push(RemainderShard {
                ordinal,
                inputs,
                cycles: window.len,
                az,
                bz,
                eq: DeviceSplitEq::new_window_with_scaling(
                    device,
                    cycle_point,
                    BindingOrder::LowToHigh,
                    kernel,
                    ordinal,
                    count,
                )?,
            });
        }
        let eq =
            DeviceSplitEq::new_with_scaling(context, cycle_point, BindingOrder::LowToHigh, kernel)?;

        Ok(Self {
            shards,
            collapsed: None,
            eq,
            log_t,
            local_rounds: log_t + 1 - tail_rounds,
            challenges: Vec::with_capacity(log_t + 2),
        })
    }

    fn factors(
        context: &CudaKernelContext,
        forms: &LinearForms,
        inputs: &DeviceR1csInputs,
        cycles: usize,
    ) -> Result<(DeviceFrVec, DeviceFrVec), CudaError> {
        let device_forms = forms.upload(context)?;
        let mut az = context.alloc(2 * cycles)?;
        let mut bz = context.alloc(2 * cycles)?;
        let count = CudaKernelContext::count_of(cycles)?;
        let mut builder = context.stream().launch_builder(context.so_factors());
        let _ = builder.arg(inputs.narrow());
        let _ = builder.arg(inputs.wide());
        let _ = builder.arg(inputs.flags());
        device_forms.bind_args(&mut builder);
        let _ = builder.arg(&count);
        let _ = builder.arg(az.limbs_mut());
        let _ = builder.arg(bz.limbs_mut());
        // SAFETY: thread `t < cycles` reads row `t` of `narrow`, `wide` and
        // `flags`, all sized at least `cycles` rows, and the four uploaded forms
        // indexed `stream * 2 (+ 1)` for `stream < 2`. It writes
        // `az[(t << 1) | stream]` and `bz[(t << 1) | stream]`, one slot per
        // (thread, stream) inside the `2 * cycles` allocations; `az` and `bz` are
        // fresh and distinct.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
        Ok((az, bz))
    }

    fn rounds(&self) -> usize {
        self.log_t + 1
    }

    fn bind(&mut self, challenge: F) -> Result<(), CudaError> {
        let scalar = require_fr(challenge)?;
        let bound = self.challenges.len();
        self.challenges.push(challenge);
        self.eq.bind(challenge);
        if let Some((az, bz)) = &mut self.collapsed {
            let context = context_for(0).ok_or(absent())?;
            *az = context.bind_rows(az, az.len(), scalar)?;
            *bz = context.bind_rows(bz, bz.len(), scalar)?;
            return Ok(());
        }
        let tasks: Vec<DeviceTask<'_, (), CudaError>> = self
            .shards
            .iter_mut()
            .map(|shard| {
                let task: DeviceTask<'_, (), CudaError> = Box::new(move || {
                    let context = context_for(shard.ordinal).ok_or(absent())?;
                    shard.az = context.bind_rows(&shard.az, shard.az.len(), scalar)?;
                    shard.bz = context.bind_rows(&shard.bz, shard.bz.len(), scalar)?;
                    shard.eq.bind(challenge);
                    Ok(())
                });
                task
            })
            .collect();
        let _ = fan_out(tasks)?;
        if bound + 1 == self.local_rounds {
            self.collapse()?;
        }
        Ok(())
    }

    fn collapse(&mut self) -> Result<(), CudaError> {
        let context = context_for(0).ok_or(absent())?;
        let mut az = Vec::with_capacity(self.shards.len());
        let mut bz = Vec::with_capacity(self.shards.len());
        for shard in &self.shards {
            if shard.az.len() != 1 || shard.bz.len() != 1 {
                return Err(CudaError::LengthMismatch {
                    expected: 1,
                    got: shard.az.len(),
                });
            }
            az.push(shard.az.first()?);
            bz.push(shard.bz.first()?);
        }
        self.collapsed = Some((context.upload(&az)?, context.upload(&bz)?));
        for shard in &mut self.shards {
            shard.az = context_for(shard.ordinal).ok_or(absent())?.alloc(0)?;
            shard.bz = context_for(shard.ordinal).ok_or(absent())?.alloc(0)?;
        }
        Ok(())
    }

    fn endpoints(&self) -> Result<[F; 2], CudaError> {
        if let Some((az, bz)) = &self.collapsed {
            let context = context_for(0).ok_or(absent())?;
            return Self::window_endpoints(context, az, bz, &self.eq);
        }
        let tasks: Vec<DeviceTask<'_, [F; 2], CudaError>> = self
            .shards
            .iter()
            .map(|shard| {
                let task: DeviceTask<'_, [F; 2], CudaError> = Box::new(move || {
                    let context = context_for(shard.ordinal).ok_or(absent())?;
                    Self::window_endpoints(context, &shard.az, &shard.bz, &shard.eq)
                });
                task
            })
            .collect();
        let mut total = [F::zero(), F::zero()];
        for part in fan_out(tasks)? {
            total[0] += part[0];
            total[1] += part[1];
        }
        Ok(total)
    }

    fn window_endpoints(
        context: &CudaKernelContext,
        az: &DeviceFrVec,
        bz: &DeviceFrVec,
        eq: &DeviceSplitEq<F>,
    ) -> Result<[F; 2], CudaError> {
        let half = az.len() / 2;
        if half == 0 {
            return Err(CudaError::InvariantViolation {
                reason: "the Spartan outer remainder has no pairs left to reduce",
            });
        }
        let e_in_len = eq.e_in_len();
        let in_bits = e_in_len.max(1).ilog2();
        let threads = half.div_ceil(MESSAGE_STRIP);
        let blocks = u32::try_from(threads.div_ceil(BLOCK as usize).max(1)).map_err(|_| {
            CudaError::InvariantViolation {
                reason: "the Spartan outer message launch exceeds a u32 block count",
            }
        })?;
        let mut partials = context.alloc(2 * blocks as usize)?;

        let inner_len = CudaKernelContext::count_of(e_in_len)?;
        let bits = CudaKernelContext::count_of(in_bits as usize)?;
        let count = CudaKernelContext::count_of(half)?;
        let strip = CudaKernelContext::count_of(MESSAGE_STRIP)?;
        let mut builder = context
            .stream()
            .launch_builder(context.gruen_pair_message());
        let _ = builder.arg(az.limbs());
        let _ = builder.arg(bz.limbs());
        let _ = builder.arg(eq.e_in_current().limbs());
        let _ = builder.arg(eq.e_out_current().limbs());
        let _ = builder.arg(&inner_len);
        let _ = builder.arg(&bits);
        let _ = builder.arg(&count);
        let _ = builder.arg(&strip);
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: thread covers pairs `base .. base + MESSAGE_STRIP` and breaks at
        // `half`, so it reads only `az[2p]`, `az[2p + 1]`, `bz[2p]`, `bz[2p + 1]`
        // for `p < half`, inside both `2 * half`-element tables. The eq lookup
        // reads `e_out[p]` when `e_in_len <= 1` and otherwise
        // `e_in[p & mask]` / `e_out[p >> in_bits]`, bounded because `in_bits` is
        // `e_in`'s log length and `e_in_len * e_out_len == half` for the split-eq
        // at this round. It writes only `partials[lane * blocks + blockIdx.x]` of
        // `2 * blocks`. Shared memory is `BLOCK * LIMBS` u64s, matching
        // `shared_mem_bytes`, and the reduction is outside every early exit.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;

        let totals = reduce_lanes(context, partials, 2, blocks)?;
        let host = totals.to_host()?;
        let convert = |value: Fr| {
            fr_into(value).ok_or(CudaError::NotImplemented {
                kernel: "CUDA kernels support only the BN254 scalar field",
            })
        };
        Ok([convert(host[0])?, convert(host[1])?])
    }

    fn window_claims(
        context: &CudaKernelContext,
        inputs: &DeviceR1csInputs,
        cycles: usize,
        cycle_challenges: &[F],
        shard: usize,
        shards: usize,
    ) -> Result<Vec<F>, CudaError> {
        let (e_in, e_out, in_bits) =
            split_eq_tables_window(context, cycle_challenges, shard, shards)?;
        let threads = cycles.div_ceil(CLAIM_STRIP);
        let blocks = u32::try_from(threads.div_ceil(BLOCK as usize).max(1)).map_err(|_| {
            CudaError::InvariantViolation {
                reason: "the Spartan outer claim launch exceeds a u32 block count",
            }
        })?;

        let mut claims = Vec::with_capacity(VARIABLES);
        let mut first = 0usize;
        while first < VARIABLES {
            let lanes = CLAIM_LANES.min(VARIABLES - first);
            let mut partials = context.alloc(lanes * blocks as usize)?;
            let first_arg = CudaKernelContext::count_of(first)?;
            let lanes_arg = CudaKernelContext::count_of(lanes)?;
            let count = CudaKernelContext::count_of(cycles)?;
            let strip = CudaKernelContext::count_of(CLAIM_STRIP)?;
            let bits = CudaKernelContext::count_of(in_bits)?;
            let mut builder = context.stream().launch_builder(context.so_claims());
            let _ = builder.arg(inputs.narrow());
            let _ = builder.arg(inputs.wide());
            let _ = builder.arg(inputs.flags());
            let _ = builder.arg(inputs.layout());
            let _ = builder.arg(e_in.limbs());
            let _ = builder.arg(e_out.limbs());
            let _ = builder.arg(&bits);
            let _ = builder.arg(&first_arg);
            let _ = builder.arg(&lanes_arg);
            let _ = builder.arg(&count);
            let _ = builder.arg(&strip);
            let _ = builder.arg(partials.limbs_mut());
            // SAFETY: thread covers cycles `base .. base + CLAIM_STRIP` and breaks
            // at `cycles`, so every row read is inside the three column buffers.
            // It reads `layout[first + lane]` for `lane < lanes`, and
            // `first + lanes <= VARIABLES` is the loop's own bound, so the read is
            // inside `layout`'s `VARIABLES` entries. The eq lookup is bounded by
            // `in_bits` coming from `e_in`'s length with `t < cycles`. It writes
            // only `partials[lane * blocks + blockIdx.x]` of `lanes * blocks`,
            // and `lanes <= SO_CLAIM_LANES` bounds the kernel's fixed-size
            // accumulator array. Shared memory is `BLOCK * LIMBS` u64s, matching
            // `shared_mem_bytes`, and the reduction is outside every early exit.
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

    pub fn claims(&self) -> Result<Vec<F>, CudaError> {
        let cycle_challenges: Vec<F> = self.challenges[1..=self.log_t]
            .iter()
            .rev()
            .copied()
            .collect();
        let shards = self.shards.len();
        let tasks: Vec<DeviceTask<'_, Vec<F>, CudaError>> = self
            .shards
            .iter()
            .map(|shard| {
                let cycle_challenges = &cycle_challenges;
                let task: DeviceTask<'_, Vec<F>, CudaError> = Box::new(move || {
                    let context = context_for(shard.ordinal).ok_or(absent())?;
                    Self::window_claims(
                        context,
                        &shard.inputs,
                        shard.cycles,
                        cycle_challenges,
                        shard.ordinal,
                        shards,
                    )
                });
                task
            })
            .collect();
        let parts = fan_out(tasks)?;
        let mut total = parts
            .first()
            .cloned()
            .ok_or(CudaError::InvariantViolation {
                reason: "the Spartan outer claim reduce produced no window",
            })?;
        for part in parts.iter().skip(1) {
            if part.len() != total.len() {
                return Err(CudaError::LengthMismatch {
                    expected: total.len(),
                    got: part.len(),
                });
            }
            for (slot, addend) in total.iter_mut().zip(part) {
                *slot += *addend;
            }
        }
        Ok(total)
    }

    pub fn opening_values(&self, log_t: usize) -> Result<BTreeMap<JoltOpeningId, F>, CudaError> {
        let dimensions = SpartanOuterDimensions::rv64(log_t);
        Ok(dimensions
            .variables()
            .iter()
            .zip(self.claims()?)
            .map(|(&variable, claim)| (outer_opening(variable), claim))
            .collect())
    }
}

pub struct SpartanOuterRemainderKernel<F: Field> {
    state: DeviceRemainder<F>,
    degree: usize,
    log_t: usize,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for SpartanOuterRemainderKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        let state = &self.state;
        visitor.visit_simple(
            allocative::Key::new("inputs"),
            state
                .shards
                .iter()
                .map(|shard| shard.inputs.device_bytes())
                .sum(),
        );
        visitor.visit_simple(
            allocative::Key::new("az_bz"),
            state
                .shards
                .iter()
                .map(|shard| shard.az.device_bytes() + shard.bz.device_bytes())
                .sum::<usize>()
                + state
                    .collapsed
                    .as_ref()
                    .map_or(0, |(az, bz)| az.device_bytes() + bz.device_bytes()),
        );
        visitor.visit_simple(
            allocative::Key::new("eq"),
            state.eq.device_bytes()
                + state
                    .shards
                    .iter()
                    .map(|shard| shard.eq.device_bytes())
                    .sum::<usize>(),
        );
        visitor.exit();
    }
}

impl<F: Field> SpartanOuterRemainderKernel<F> {
    pub const fn new(state: DeviceRemainder<F>, degree: usize, log_t: usize) -> Self {
        Self {
            state,
            degree,
            log_t,
        }
    }

    pub fn openings(&self) -> Result<BTreeMap<JoltOpeningId, F>, CudaError> {
        self.state.opening_values(self.log_t)
    }

    pub fn bound_rounds(&self) -> usize {
        self.state.challenges.len()
    }
}

impl<F: Field> ProveRounds<F> for SpartanOuterRemainderKernel<F> {
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
            kind: "cuda Spartan outer remainder",
        };
        if let Some(challenge) = bind {
            self.state.bind(challenge).map_err(|_| failed())?;
        }
        let [q0, q2] = self.state.endpoints().map_err(|_| failed())?;
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
            .bind(bind)
            .map_err(|_| SumcheckError::MissingEvaluationSource {
                kind: "cuda Spartan outer remainder",
            })
    }
}

const fn absent() -> CudaError {
    CudaError::InvariantViolation {
        reason: "a Spartan outer remainder window names an absent device",
    }
}
