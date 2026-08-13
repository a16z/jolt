use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_claims::protocols::jolt::JoltChallengeId;
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_poly::{BindingOrder, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckOutputClaims,
};

use super::context::{CudaKernelContext, BLOCK};
use super::device::{fr_into, require_fr, require_fr_slice, DeviceFrVec, LIMBS};
use super::error::CudaError;
use super::lt_poly::DeviceLtPolynomial;
use super::ra_poly::DeviceRaPolynomial;

pub struct DeviceDenseProduct {
    weight: Option<DeviceFrVec>,
    factors: Vec<DeviceFrVec>,
    one_hot: Option<DeviceRaPolynomial>,
    lt: Option<DeviceLtPolynomial>,
    degree: usize,
    rounds: usize,
    rounds_bound: usize,
}

impl DeviceDenseProduct {
    pub fn new<F: Field>(
        context: &CudaKernelContext,
        weights: &[(F, Vec<F>)],
        factors: &[Vec<F>],
        one_hot: Option<DeviceRaPolynomial>,
        lt: Option<DeviceLtPolynomial>,
        rounds: usize,
        degree: usize,
    ) -> Result<Self, CudaError> {
        let expected = 1usize << rounds;
        for table in weights.iter().map(|(_, table)| table).chain(factors.iter()) {
            if table.len() != expected {
                return Err(CudaError::LengthMismatch {
                    expected,
                    got: table.len(),
                });
            }
        }
        if weights.is_empty() && lt.is_none() && one_hot.is_none() && factors.is_empty() {
            return Err(CudaError::InvariantViolation {
                reason: "a dense product needs at least one factor",
            });
        }
        if let Some(one_hot) = &one_hot {
            if one_hot.len() != expected {
                return Err(CudaError::LengthMismatch {
                    expected,
                    got: one_hot.len(),
                });
            }
            if one_hot.order() != BindingOrder::LowToHigh {
                return Err(CudaError::InvariantViolation {
                    reason: "a dense product binds LowToHigh, so its one-hot factor must too",
                });
            }
        }
        if let Some(lt) = &lt {
            if lt.len() != expected {
                return Err(CudaError::LengthMismatch {
                    expected,
                    got: lt.len(),
                });
            }
            if lt.order() != BindingOrder::LowToHigh {
                return Err(CudaError::InvariantViolation {
                    reason: "a dense product binds LowToHigh, so its LT factor must too",
                });
            }
        }

        let weight = if weights.is_empty() {
            None
        } else {
            Some(Self::combine_weights(context, weights, expected)?)
        };
        let mut uploaded = Vec::with_capacity(factors.len());
        for table in factors {
            uploaded.push(context.upload(require_fr_slice(table)?)?);
        }
        Ok(Self {
            weight,
            factors: uploaded,
            one_hot,
            lt,
            degree,
            rounds,
            rounds_bound: 0,
        })
    }

    fn combine_weights<F: Field>(
        context: &CudaKernelContext,
        weights: &[(F, Vec<F>)],
        len: usize,
    ) -> Result<DeviceFrVec, CudaError> {
        let mut accumulator = context.alloc(len)?;
        let count = CudaKernelContext::count_of(len)?;
        for (coefficient, table) in weights {
            let table = context.upload(require_fr_slice(table)?)?;
            let coefficient = context.upload(&[require_fr(*coefficient)?])?;
            let mut builder = context.stream().launch_builder(context.weighted_combine());
            let _ = builder.arg(table.limbs());
            let _ = builder.arg(coefficient.limbs());
            let _ = builder.arg(accumulator.limbs_mut());
            let _ = builder.arg(&count);
            // SAFETY: thread `i < count` reads `weights[i]` plus the
            // single-element `coefficient`, and read-modify-writes only
            // `accumulator[i]` — one thread per element, so uncontended. Both
            // tables hold `count` elements (checked against `len` in `new`) and
            // are distinct allocations.
            let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
            context.stream().synchronize()?;
        }
        Ok(accumulator)
    }

    fn round_tables(
        &self,
        context: &CudaKernelContext,
    ) -> Result<(Vec<&DeviceFrVec>, Option<DeviceFrVec>), CudaError> {
        let mut handles: Vec<&DeviceFrVec> = Vec::with_capacity(2 + self.factors.len());
        handles.extend(self.weight.as_ref());
        handles.extend(self.factors.iter());
        let materialized = match &self.one_hot {
            None => None,
            Some(one_hot) => match one_hot.dense() {
                Some(dense) => {
                    handles.push(dense);
                    None
                }
                None => Some(one_hot.coefficients(context)?),
            },
        };
        Ok((handles, materialized))
    }

    pub fn toom_evals<F: Field>(&self, context: &CudaKernelContext) -> Result<Vec<F>, CudaError> {
        self.round_lanes(context, 1, true, self.degree)
    }

    fn round_lanes<F: Field>(
        &self,
        context: &CudaKernelContext,
        first_point: u32,
        infinity_lane: bool,
        lane_count: usize,
    ) -> Result<Vec<F>, CudaError> {
        let remaining = self.rounds - self.rounds_bound;
        let half = (1usize << remaining) / 2;
        let lanes = CudaKernelContext::count_of(lane_count)?;
        let half_count = CudaKernelContext::count_of(half)?;

        let (mut handles, materialized) = self.round_tables(context)?;
        if let Some(gathered) = &materialized {
            handles.push(gathered);
        }
        let table_count = CudaKernelContext::count_of(handles.len())?;
        let pointers = context.device_pointers(&handles)?;
        let blocks = half_count.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc(lanes as usize * blocks as usize)?;

        let empty = context.alloc(1)?;
        let lt_view = self
            .lt
            .as_ref()
            .map(super::lt_poly::DeviceLtPolynomial::view);
        let has_lt = u32::from(lt_view.is_some());
        let (lt_lo, lt_hi, eq_hi, lt_shift, lo_bits, lo_mask) = match &lt_view {
            Some(view) => (
                view.lt_lo,
                view.lt_hi,
                view.eq_hi,
                view.shift,
                view.lo_bits,
                view.lo_mask,
            ),
            None => (&empty, &empty, &empty, &empty, 0, 0),
        };

        let mut builder = context
            .stream()
            .launch_builder(context.dense_product_round());
        let _ = builder.arg(&pointers);
        let _ = builder.arg(&table_count);
        let _ = builder.arg(&half_count);
        let _ = builder.arg(&lanes);
        let _ = builder.arg(partials.limbs_mut());
        let _ = builder.arg(lt_lo.limbs());
        let _ = builder.arg(lt_hi.limbs());
        let _ = builder.arg(eq_hi.limbs());
        let _ = builder.arg(lt_shift.limbs());
        let _ = builder.arg(&lo_bits);
        let _ = builder.arg(&lo_mask);
        let _ = builder.arg(&has_lt);
        let _ = builder.arg(&first_point);
        let infinity_flag = u32::from(infinity_lane);
        let _ = builder.arg(&infinity_flag);
        // SAFETY: reads, all in bounds — `table[2y]`/`table[2y+1]` for each of
        // `table_count` tables of `2 * half` elements; the LT tables and the
        // single-element `lt_shift` only when
        // `has_lt`, at indices masked below `2^lo_bits` and `2^(len-lo_bits)`.
        // Writes: `partials[c * gridDim.x + blockIdx.x]`, one slot per
        // (lane, block) of `lanes * blocks`. Shared memory is `BLOCK * LIMBS`
        // u64s, matching `shared_mem_bytes`. Every buffer outlives the launch,
        // borrowed through `&self` or owned by this frame.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;
        context.stream().synchronize()?;
        drop(materialized);

        let totals = Self::reduce_lanes(context, partials, lanes, blocks)?;
        totals
            .to_host()?
            .into_iter()
            .map(|value| {
                fr_into(value).ok_or(CudaError::NotImplemented {
                    kernel: "CUDA kernels support only the BN254 scalar field",
                })
            })
            .collect()
    }

    fn reduce_lanes(
        context: &CudaKernelContext,
        mut partials: DeviceFrVec,
        lanes: u32,
        mut width: u32,
    ) -> Result<DeviceFrVec, CudaError> {
        while width > 1 {
            let next = width.div_ceil(2);
            let mut folded = context.alloc(lanes as usize * next as usize)?;
            let mut builder = context.stream().launch_builder(context.lane_sum_reduce());
            let _ = builder.arg(partials.limbs());
            let _ = builder.arg(folded.limbs_mut());
            let _ = builder.arg(&lanes);
            let _ = builder.arg(&width);
            let _ = builder.arg(&next);
            // SAFETY: thread `(i < next, lane < lanes)` reads
            // `in[lane * width + i]` and, when `i + next < width`, its mate at
            // `+ next` — both inside `in`'s `lanes * width` elements — and writes
            // only `out[lane * next + i]` of `lanes * next`. Index sets are
            // pairwise disjoint and `out` is a distinct allocation.
            let _ = unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (next.div_ceil(BLOCK), lanes, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
            context.stream().synchronize()?;
            partials = folded;
            width = next;
        }
        Ok(partials)
    }

    pub fn bind<F: Field>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        let challenge = require_fr(challenge)?;
        if let Some(weight) = &self.weight {
            self.weight = Some(context.bind(weight, challenge, BindingOrder::LowToHigh)?);
        }
        for table in &mut self.factors {
            *table = context.bind(table, challenge, BindingOrder::LowToHigh)?;
        }
        if let Some(one_hot) = &mut self.one_hot {
            one_hot.bind(context, challenge)?;
        }
        if let Some(lt) = &mut self.lt {
            lt.bind(context, challenge)?;
        }
        self.rounds_bound += 1;
        Ok(())
    }

    pub fn factor_finals<F: Field>(
        &self,
        context: &CudaKernelContext,
    ) -> Result<Vec<F>, CudaError> {
        let mut finals = Vec::with_capacity(self.factors.len() + 1);
        for table in &self.factors {
            finals.push(table.first()?);
        }
        if let Some(one_hot) = &self.one_hot {
            finals.push(one_hot.final_claim(context)?);
        }
        finals
            .into_iter()
            .map(|value| {
                fr_into(value).ok_or(CudaError::NotImplemented {
                    kernel: "CUDA kernels support only the BN254 scalar field",
                })
            })
            .collect()
    }

    pub const fn rounds_bound(&self) -> usize {
        self.rounds_bound
    }

    pub fn lt_final<F: Field>(&self, context: &CudaKernelContext) -> Result<Option<F>, CudaError> {
        let Some(lt) = &self.lt else {
            return Ok(None);
        };
        let value = lt.final_claim(context)?;
        fr_into(value)
            .ok_or(CudaError::NotImplemented {
                kernel: "CUDA kernels support only the BN254 scalar field",
            })
            .map(Some)
    }
}

pub struct DenseProductKernel<F: Field, R> {
    pub(crate) state: DeviceDenseProduct,
    pub(crate) relation: R,
    pub(crate) context: &'static CudaKernelContext,
    pub(crate) field: core::marker::PhantomData<F>,
}

impl<F: Field, R> DenseProductKernel<F, R> {
    pub(crate) fn finals(&self) -> Result<Vec<F>, CudaError> {
        self.state.factor_finals(self.context)
    }
}

impl<F, R> ProveRounds<F> for DenseProductKernel<F, R>
where
    F: Field,
    R: ConcreteSumcheck<F>,
    SumcheckInputClaims<F, R>: InputClaims<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, R>: SumcheckChallenges<F, JoltChallengeId>,
{
    fn num_rounds(&self) -> usize {
        self.relation.symbolic().rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.state
                .bind(self.context, challenge)
                .map_err(|_| SumcheckError::MissingEvaluationSource { kind: "cuda bind" })?;
        }
        let evals: Vec<F> = self.state.toom_evals(self.context).map_err(|_| {
            SumcheckError::MissingEvaluationSource {
                kind: "cuda round_evals",
            }
        })?;
        let _ = round;
        let eval_at_0 = previous_claim - evals[0];
        let mut toom = Vec::with_capacity(evals.len() + 1);
        toom.push(eval_at_0);
        toom.extend_from_slice(&evals);
        Ok(UnivariatePoly::from_evals_toom(&toom))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.state
            .bind(self.context, bind)
            .map_err(|_| SumcheckError::MissingEvaluationSource { kind: "cuda bind" })
    }
}
