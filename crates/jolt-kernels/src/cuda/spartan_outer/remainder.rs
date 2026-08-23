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
use crate::cuda::common::context::{CudaKernelContext, BLOCK};
use crate::cuda::common::device::{fr_into, require_fr, DeviceFrVec, LIMBS};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::primitives::reduce_lanes;
use crate::cuda::common::split_eq::split_eq_tables;
use crate::cuda::common::split_eq::DeviceSplitEq;

pub const CLAIM_LANES: usize = 4;

pub const MESSAGE_STRIP: usize = 1;

pub const CLAIM_STRIP: usize = 16;

pub struct DeviceRemainder<F: Field> {
    inputs: DeviceR1csInputs,
    az: DeviceFrVec,
    bz: DeviceFrVec,
    eq: DeviceSplitEq<F>,
    log_t: usize,
    challenges: Vec<F>,
}

impl<F: Field> DeviceRemainder<F> {
    pub fn new(
        context: &CudaKernelContext,
        inputs: DeviceR1csInputs,
        matrices: &ConstraintMatrices<F>,
        tau: &[F],
        log_t: usize,
        uniskip_challenge: F,
    ) -> Result<Self, CudaError> {
        let columns: Vec<usize> = (1..=VARIABLES).collect();
        let mut forms = LinearForms::new();
        push_forms(&mut forms, matrices, &columns, uniskip_challenge, F::zero())?;
        push_forms(&mut forms, matrices, &columns, uniskip_challenge, F::one())?;
        let device_forms = forms.upload(context)?;

        let cycles = inputs.cycles();
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
        // `flags`, all sized `cycles` rows, and the four uploaded forms indexed
        // `stream * 2 (+ 1)` for `stream < 2`. It writes `az[(t << 1) | stream]`
        // and `bz[(t << 1) | stream]`, one slot per (thread, stream) inside the
        // `2 * cycles` allocations; `az` and `bz` are fresh and distinct.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;

        let kernel = centered_lagrange_kernel::<F>(
            OUTER_UNISKIP_DOMAIN_SIZE,
            tau[log_t + 1],
            uniskip_challenge,
        )
        .map_err(|_| CudaError::InvariantViolation {
            reason: "the Spartan outer uni-skip domain is not a valid centered integer domain",
        })?;
        let eq = DeviceSplitEq::new_with_scaling(
            context,
            &tau[..=log_t],
            BindingOrder::LowToHigh,
            kernel,
        )?;

        Ok(Self {
            inputs,
            az,
            bz,
            eq,
            log_t,
            challenges: Vec::with_capacity(log_t + 2),
        })
    }

    fn rounds(&self) -> usize {
        self.log_t + 1
    }

    fn bind(&mut self, context: &CudaKernelContext, challenge: F) -> Result<(), CudaError> {
        let scalar = require_fr(challenge)?;
        self.az = context.bind_rows(&self.az, self.az.len(), scalar)?;
        self.bz = context.bind_rows(&self.bz, self.bz.len(), scalar)?;
        self.eq.bind(challenge);
        self.challenges.push(challenge);
        Ok(())
    }

    fn endpoints(&self, context: &CudaKernelContext) -> Result<[F; 2], CudaError> {
        let half = self.az.len() / 2;
        if half == 0 {
            return Err(CudaError::InvariantViolation {
                reason: "the Spartan outer remainder has no pairs left to reduce",
            });
        }
        let e_in_len = self.eq.e_in_len();
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
        let _ = builder.arg(self.az.limbs());
        let _ = builder.arg(self.bz.limbs());
        let _ = builder.arg(self.eq.e_in_current().limbs());
        let _ = builder.arg(self.eq.e_out_current().limbs());
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

    pub fn claims(&self, context: &CudaKernelContext) -> Result<Vec<F>, CudaError> {
        let cycle_challenges: Vec<F> = self.challenges[1..=self.log_t]
            .iter()
            .rev()
            .copied()
            .collect();
        let (e_in, e_out, in_bits) = split_eq_tables(context, &cycle_challenges)?;

        let cycles = self.inputs.cycles();
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
            let _ = builder.arg(self.inputs.narrow());
            let _ = builder.arg(self.inputs.wide());
            let _ = builder.arg(self.inputs.flags());
            let _ = builder.arg(self.inputs.layout());
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

    pub fn opening_values(
        &self,
        context: &CudaKernelContext,
        log_t: usize,
    ) -> Result<BTreeMap<JoltOpeningId, F>, CudaError> {
        let dimensions = SpartanOuterDimensions::rv64(log_t);
        Ok(dimensions
            .variables()
            .iter()
            .zip(self.claims(context)?)
            .map(|(&variable, claim)| (outer_opening(variable), claim))
            .collect())
    }
}

pub struct SpartanOuterRemainderKernel<F: Field> {
    context: &'static CudaKernelContext,
    state: DeviceRemainder<F>,
    degree: usize,
    log_t: usize,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for SpartanOuterRemainderKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let visitor = visitor.enter_self_sized::<Self>();
        visitor.exit();
    }
}

impl<F: Field> SpartanOuterRemainderKernel<F> {
    pub const fn new(
        context: &'static CudaKernelContext,
        state: DeviceRemainder<F>,
        degree: usize,
        log_t: usize,
    ) -> Self {
        Self {
            context,
            state,
            degree,
            log_t,
        }
    }

    pub fn openings(&self) -> Result<BTreeMap<JoltOpeningId, F>, CudaError> {
        self.state.opening_values(self.context, self.log_t)
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
                kind: "cuda Spartan outer remainder",
            })
    }
}
