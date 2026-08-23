use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_claims::protocols::jolt::JoltChallengeId;
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck};
use jolt_field::{Field, Fr};
use jolt_poly::{BindingOrder, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckOutputClaims,
};

use super::context::context_for;
use super::context::{CudaKernelContext, BLOCK};
use super::device::{fr_into, require_fr, DeviceFrVec, LIMBS};
use super::devices::{fan_out, DeviceTask};
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

pub(crate) struct ShardedDenseProduct {
    shards: Vec<(usize, DeviceDenseProduct)>,
    collapsed: Option<(usize, DeviceDenseProduct)>,
    local_rounds: usize,
    tail_rounds: usize,
    bound: usize,
    assembled: bool,
    lt_column: bool,
}

pub(crate) struct WindowScalars {
    weight: Option<Fr>,
    factors: Vec<Fr>,
    one_hot: Option<Fr>,
    lt: Option<Fr>,
}

impl ShardedDenseProduct {
    pub(crate) fn new(shards: Vec<(usize, DeviceDenseProduct)>) -> Result<Self, CudaError> {
        let count = shards.len();
        if count == 0 || !count.is_power_of_two() {
            return Err(CudaError::InvariantViolation {
                reason: "a sharded dense product needs a power-of-two shard count",
            });
        }
        let tail_rounds = count.trailing_zeros() as usize;
        let (local_rounds, lt_column) = shards
            .first()
            .map(|(_, state)| (state.rounds, state.lt.is_some()))
            .ok_or(CudaError::InvariantViolation {
                reason: "a sharded dense product needs at least one shard",
            })?;
        if shards.iter().any(|(_, state)| {
            state.rounds != local_rounds
                || state.rounds_bound != 0
                || state.lt.is_some() != lt_column
        }) {
            return Err(CudaError::InvariantViolation {
                reason: "every dense-product shard starts unbound over the same local domain with \
                         the same factor shape",
            });
        }
        if count == 1 {
            let (ordinal, state) =
                shards
                    .into_iter()
                    .next()
                    .ok_or(CudaError::InvariantViolation {
                        reason: "a single-shard dense product lost its state",
                    })?;
            return Ok(Self {
                shards: Vec::new(),
                collapsed: Some((ordinal, state)),
                local_rounds,
                tail_rounds: 0,
                bound: 0,
                assembled: false,
                lt_column,
            });
        }
        Ok(Self {
            shards,
            collapsed: None,
            local_rounds,
            tail_rounds,
            bound: 0,
            assembled: false,
            lt_column,
        })
    }

    #[cfg(test)]
    pub(crate) fn rounds(&self) -> usize {
        self.local_rounds + self.tail_rounds
    }

    pub(crate) const fn rounds_bound(&self) -> usize {
        self.bound
    }

    pub(crate) fn toom_evals<F: Field>(&self) -> Result<Vec<F>, CudaError> {
        if let Some((ordinal, collapsed)) = &self.collapsed {
            let context = context_for(*ordinal).ok_or(absent())?;
            return collapsed.toom_evals(context);
        }
        let tasks: Vec<DeviceTask<'_, Vec<F>, CudaError>> = self
            .shards
            .iter()
            .map(|(ordinal, state)| {
                let ordinal = *ordinal;
                let task: DeviceTask<'_, Vec<F>, CudaError> = Box::new(move || {
                    let context = context_for(ordinal).ok_or(absent())?;
                    state.toom_evals(context)
                });
                task
            })
            .collect();
        let parts = fan_out(tasks)?;
        let mut total = parts
            .first()
            .cloned()
            .ok_or(CudaError::InvariantViolation {
                reason: "a sharded dense-product round produced no window",
            })?;
        for part in parts.iter().skip(1) {
            if part.len() != total.len() {
                return Err(CudaError::LengthMismatch {
                    expected: total.len(),
                    got: part.len(),
                });
            }
            for (lane, addend) in total.iter_mut().zip(part) {
                *lane += *addend;
            }
        }
        Ok(total)
    }

    pub(crate) fn bind<F: Field>(&mut self, challenge: F) -> Result<(), CudaError> {
        self.bound += 1;
        if let Some((ordinal, collapsed)) = &mut self.collapsed {
            let context = context_for(*ordinal).ok_or(absent())?;
            return collapsed.bind(context, challenge);
        }
        let shards = &mut self.shards;
        let tasks: Vec<DeviceTask<'_, (), CudaError>> = shards
            .iter_mut()
            .map(|(ordinal, state)| {
                let ordinal = *ordinal;
                let task: DeviceTask<'_, (), CudaError> = Box::new(move || {
                    let context = context_for(ordinal).ok_or(absent())?;
                    state.bind(context, challenge)
                });
                task
            })
            .collect();
        let _ = fan_out(tasks)?;
        if self
            .shards
            .first()
            .is_some_and(|(_, state)| state.rounds_bound == self.local_rounds)
        {
            self.collapse()?;
        }
        Ok(())
    }

    /// Once every shard's local domain is fully bound each of its tables holds a
    /// single scalar, so the `n` shards' scalars ARE the `n`-element tables of the
    /// remaining sumcheck over the split bits — which existing single-device code
    /// finishes without any tail-specific round logic.
    fn collapse(&mut self) -> Result<(), CudaError> {
        let context = context_for(0).ok_or(absent())?;
        let degree = self.shards.first().map(|(_, state)| state.degree()).ok_or(
            CudaError::InvariantViolation {
                reason: "a sharded dense product lost its shards before collapsing",
            },
        )?;
        let count = self.shards.len();
        let tasks: Vec<DeviceTask<'_, WindowScalars, CudaError>> = self
            .shards
            .iter()
            .map(|(ordinal, state)| {
                let ordinal = *ordinal;
                let task: DeviceTask<'_, WindowScalars, CudaError> = Box::new(move || {
                    let context = context_for(ordinal).ok_or(absent())?;
                    state.window_scalars(context)
                });
                task
            })
            .collect();
        let mut weights = Vec::with_capacity(count);
        let mut columns: Vec<Vec<Fr>> = Vec::new();
        for scalars in fan_out(tasks)? {
            let mut values = scalars.factors;
            values.extend(scalars.one_hot);
            values.extend(scalars.lt);
            if columns.is_empty() {
                columns = values.iter().map(|_| Vec::new()).collect();
            }
            if values.len() != columns.len() {
                return Err(CudaError::LengthMismatch {
                    expected: columns.len(),
                    got: values.len(),
                });
            }
            for (column, value) in columns.iter_mut().zip(&values) {
                column.push(*value);
            }
            if let Some(weight) = scalars.weight {
                weights.push(weight);
            }
        }
        let weight = if weights.is_empty() {
            None
        } else {
            if weights.len() != count {
                return Err(CudaError::LengthMismatch {
                    expected: count,
                    got: weights.len(),
                });
            }
            Some(context.upload(&weights)?)
        };
        let factors = columns
            .iter()
            .map(|column| context.upload(column))
            .collect::<Result<Vec<_>, _>>()?;
        self.collapsed = Some((
            0,
            DeviceDenseProduct::from_device_factors(
                weight,
                factors,
                None,
                None,
                self.tail_rounds,
                degree,
            )?,
        ));
        self.assembled = true;
        self.shards.clear();
        Ok(())
    }

    fn tail(&self) -> Result<(&'static CudaKernelContext, &DeviceDenseProduct), CudaError> {
        let (ordinal, collapsed) =
            self.collapsed
                .as_ref()
                .ok_or(CudaError::InvariantViolation {
                    reason: "a sharded dense product was asked for finals before its tail rounds",
                })?;
        Ok((context_for(*ordinal).ok_or(absent())?, collapsed))
    }

    pub(crate) fn factor_finals<F: Field>(&self) -> Result<Vec<F>, CudaError> {
        let (context, collapsed) = self.tail()?;
        let mut finals: Vec<F> = collapsed.factor_finals(context)?;
        if self.assembled && self.lt_column {
            let _ = finals.pop().ok_or(CudaError::InvariantViolation {
                reason: "a collapsed dense product lost its LT column",
            })?;
        }
        Ok(finals)
    }

    pub(crate) fn lt_final<F: Field>(&self) -> Result<Option<F>, CudaError> {
        let (context, collapsed) = self.tail()?;
        if !self.assembled {
            return collapsed.lt_final(context);
        }
        if !self.lt_column {
            return Ok(None);
        }
        let finals: Vec<F> = collapsed.factor_finals(context)?;
        finals
            .last()
            .copied()
            .ok_or(CudaError::InvariantViolation {
                reason: "a collapsed dense product lost its LT column",
            })
            .map(Some)
    }
}

const fn absent() -> CudaError {
    CudaError::InvariantViolation {
        reason: "a sharded dense-product window names an absent device",
    }
}

impl DeviceDenseProduct {
    pub fn from_device_factors(
        weight: Option<DeviceFrVec>,
        factors: Vec<DeviceFrVec>,
        one_hot: Option<DeviceRaPolynomial>,
        lt: Option<DeviceLtPolynomial>,
        rounds: usize,
        degree: usize,
    ) -> Result<Self, CudaError> {
        let expected = 1usize << rounds;
        for table in weight.as_ref().into_iter().chain(factors.iter()) {
            if table.len() != expected {
                return Err(CudaError::LengthMismatch {
                    expected,
                    got: table.len(),
                });
            }
        }
        if weight.is_none() && lt.is_none() && one_hot.is_none() && factors.is_empty() {
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

        Ok(Self {
            weight,
            factors,
            one_hot,
            lt,
            degree,
            rounds,
            rounds_bound: 0,
        })
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

    pub(crate) fn reduce_lanes(
        context: &CudaKernelContext,
        partials: DeviceFrVec,
        lanes: u32,
        width: u32,
    ) -> Result<DeviceFrVec, CudaError> {
        if width <= 1 {
            return Ok(partials);
        }
        let mut totals = context.alloc(lanes as usize)?;
        let mut builder = context.stream().launch_builder(context.lane_sum_total());
        let _ = builder.arg(partials.limbs());
        let _ = builder.arg(totals.limbs_mut());
        let _ = builder.arg(&width);
        // SAFETY: block `lane = blockIdx.x < lanes` reads `in[lane * width + i]`
        // for `i` striding from `threadIdx.x` by `blockDim.x` while `i < width`,
        // so every read is inside `in`'s `lanes * width` elements, and writes only
        // `out[lane]` of `lanes`. Shared memory is `BLOCK * LIMBS` u64s, matching
        // `shared_mem_bytes`, every thread reaches each `__syncthreads()` because
        // the strided loop and the tree are outside any early return, and `BLOCK`
        // is a power of two so the tree covers the whole block.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (lanes, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;
        Ok(totals)
    }

    pub fn bind<F: Field>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        let challenge = require_fr(challenge)?;
        if let Some(weight) = &self.weight {
            self.weight = Some(context.bind_rows(weight, weight.len(), challenge)?);
        }
        for table in &mut self.factors {
            *table = context.bind_rows(table, table.len(), challenge)?;
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

    pub(crate) const fn degree(&self) -> usize {
        self.degree
    }

    pub(crate) fn window_scalars(
        &self,
        context: &CudaKernelContext,
    ) -> Result<WindowScalars, CudaError> {
        let scalar = |table: &DeviceFrVec| -> Result<Fr, CudaError> {
            if table.len() != 1 {
                return Err(CudaError::LengthMismatch {
                    expected: 1,
                    got: table.len(),
                });
            }
            table.first()
        };
        Ok(WindowScalars {
            weight: self.weight.as_ref().map(scalar).transpose()?,
            factors: self
                .factors
                .iter()
                .map(scalar)
                .collect::<Result<Vec<_>, _>>()?,
            one_hot: self
                .one_hot
                .as_ref()
                .map(|one_hot| one_hot.final_claim(context))
                .transpose()?,
            lt: self
                .lt
                .as_ref()
                .map(|lt| lt.final_claim(context))
                .transpose()?,
        })
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
    pub(crate) state: ShardedDenseProduct,
    pub(crate) relation: R,
    pub(crate) field: core::marker::PhantomData<F>,
}

#[cfg(feature = "allocative")]
impl<F: Field, R> allocative::Allocative for DenseProductKernel<F, R> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let visitor = visitor.enter_self_sized::<Self>();
        visitor.exit();
    }
}

impl<F: Field, R> DenseProductKernel<F, R> {
    pub(crate) fn finals(&self) -> Result<Vec<F>, CudaError> {
        self.state.factor_finals()
    }

    pub(crate) fn lt_final(&self) -> Result<Option<F>, CudaError> {
        self.state.lt_final()
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
                .bind(challenge)
                .map_err(|_| SumcheckError::MissingEvaluationSource { kind: "cuda bind" })?;
        }
        let evals: Vec<F> =
            self.state
                .toom_evals()
                .map_err(|_| SumcheckError::MissingEvaluationSource {
                    kind: "cuda round_evals",
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
            .bind(bind)
            .map_err(|_| SumcheckError::MissingEvaluationSource { kind: "cuda bind" })
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{BindingOrder, EqPolynomial};

    use super::super::context::shared_context;
    use super::super::lt_poly::DeviceLtPolynomial;
    use super::super::ra_poly::{DeviceRaPolynomial, COLD};
    use super::super::testing::fr;
    use super::{DeviceDenseProduct, ShardedDenseProduct};

    const LOG_T: usize = 8;
    const LOG_K: usize = 3;
    const DEGREE: usize = 2;

    fn tables(cycles: usize) -> (Vec<Fr>, Vec<Vec<Fr>>) {
        let weight: Vec<Fr> = (0..cycles).map(|i| fr(13 + 3 * i as u64)).collect();
        let factors: Vec<Vec<Fr>> = (0..DEGREE)
            .map(|f| {
                (0..cycles)
                    .map(|i| fr(101 + 7 * i as u64 + 31 * f as u64))
                    .collect()
            })
            .collect();
        (weight, factors)
    }

    fn whole(cycles: usize, weight: &[Fr], factors: &[Vec<Fr>]) -> DeviceDenseProduct {
        let context = shared_context().expect("device");
        DeviceDenseProduct::from_device_factors(
            Some(context.upload(weight).expect("upload weight")),
            factors
                .iter()
                .map(|f| context.upload(f).expect("upload factor"))
                .collect(),
            None,
            None,
            cycles.ilog2() as usize,
            DEGREE,
        )
        .expect("whole dense product")
    }

    fn sharded(shards: usize, weight: &[Fr], factors: &[Vec<Fr>]) -> ShardedDenseProduct {
        let context = shared_context().expect("device");
        let len = weight.len() / shards;
        let states = (0..shards)
            .map(|shard| {
                let range = shard * len..(shard + 1) * len;
                let state = DeviceDenseProduct::from_device_factors(
                    Some(
                        context
                            .upload(&weight[range.clone()])
                            .expect("weight window"),
                    ),
                    factors
                        .iter()
                        .map(|f| context.upload(&f[range.clone()]).expect("factor window"))
                        .collect(),
                    None,
                    None,
                    len.ilog2() as usize,
                    DEGREE,
                )
                .expect("shard dense product");
                (0usize, state)
            })
            .collect();
        ShardedDenseProduct::new(states).expect("sharded dense product")
    }

    fn one_hot_words(cycles: usize) -> Vec<u32> {
        (0..cycles)
            .map(|c| {
                if c % 7 == 3 {
                    COLD
                } else {
                    ((c * 5 + 1) % (1 << LOG_K)) as u32
                }
            })
            .collect()
    }

    struct StructuredFixture {
        weight: Vec<Fr>,
        factors: Vec<Vec<Fr>>,
        words: Vec<u32>,
        eq_address: Vec<Fr>,
        cycle_point: Vec<Fr>,
    }

    impl StructuredFixture {
        fn new(cycles: usize) -> Self {
            let (weight, factors) = tables(cycles);
            Self {
                weight,
                factors,
                words: one_hot_words(cycles),
                eq_address: EqPolynomial::new(
                    (0..LOG_K).map(|i| fr(19 + 7 * i as u64)).collect(),
                )
                .evaluations(),
                cycle_point: (0..LOG_T).map(|i| fr(41 + 13 * i as u64)).collect(),
            }
        }

        fn window(&self, shard: usize, shards: usize) -> DeviceDenseProduct {
            let context = shared_context().expect("device");
            let cycles = self.weight.len() / shards;
            let range = shard * cycles..(shard + 1) * cycles;
            let one_hot = DeviceRaPolynomial::from_words(
                context,
                &self.words[range.clone()],
                &self.eq_address,
                BindingOrder::LowToHigh,
            )
            .expect("one-hot window");
            let lt = DeviceLtPolynomial::new(context, &self.cycle_point, BindingOrder::LowToHigh)
                .expect("lt poly");
            let lt = if shards == 1 {
                lt
            } else {
                lt.window(shard, shards).expect("lt window")
            };
            DeviceDenseProduct::from_device_factors(
                Some(
                    context
                        .upload(&self.weight[range.clone()])
                        .expect("upload weight"),
                ),
                self.factors
                    .iter()
                    .map(|f| context.upload(&f[range.clone()]).expect("upload factor"))
                    .collect(),
                Some(one_hot),
                Some(lt),
                cycles.ilog2() as usize,
                DEGREE,
            )
            .expect("structured dense product")
        }
    }

    #[test]
    fn sharded_dense_product_matches_the_whole_domain_with_one_hot_and_lt() {
        let Some(context) = shared_context() else {
            return;
        };
        let fixture = StructuredFixture::new(1usize << LOG_T);

        for shards in [2usize, 4, 8] {
            let mut expected = fixture.window(0, 1);
            let mut got = ShardedDenseProduct::new(
                (0..shards)
                    .map(|shard| (0usize, fixture.window(shard, shards)))
                    .collect(),
            )
            .expect("sharded dense product");
            assert_eq!(got.rounds(), LOG_T, "the shard set must cover every round");

            for round in 0..LOG_T {
                let want: Vec<Fr> = expected.toom_evals(context).expect("whole round evals");
                let have: Vec<Fr> = got.toom_evals().expect("sharded round evals");
                assert_eq!(
                    have, want,
                    "shards={shards} round {round}: a windowed one-hot and a windowed LT must add \
                     across cycle windows exactly as the dense factors do",
                );
                let challenge = fr(2_000 + 23 * round as u64);
                expected.bind(context, challenge).expect("whole bind");
                got.bind(challenge).expect("sharded bind");
            }

            let want: Vec<Fr> = expected.factor_finals(context).expect("whole finals");
            let have: Vec<Fr> = got.factor_finals().expect("sharded finals");
            assert_eq!(
                have, want,
                "shards={shards}: the gathered one-hot final must land in the whole-domain factor \
                 order",
            );
            assert_eq!(want.len(), DEGREE + 1, "the one-hot final must be reported");
            let want_lt: Option<Fr> = expected.lt_final(context).expect("whole lt final");
            let have_lt: Option<Fr> = got.lt_final().expect("sharded lt final");
            assert_eq!(have_lt, want_lt, "shards={shards}: the LT finals diverged");
            assert_ne!(
                want_lt,
                Some(Fr::from_u64(0)),
                "a degenerate fixture would hide a divergence",
            );
        }
    }

    #[test]
    fn sharded_dense_product_matches_the_whole_domain_round_for_round() {
        let Some(context) = shared_context() else {
            return;
        };
        let cycles = 1usize << LOG_T;
        let (weight, factors) = tables(cycles);

        for shards in [2usize, 4, 8] {
            let mut expected = whole(cycles, &weight, &factors);
            let mut got = sharded(shards, &weight, &factors);
            assert_eq!(got.rounds(), LOG_T, "the shard set must cover every round");

            for round in 0..LOG_T {
                let want: Vec<Fr> = expected.toom_evals(context).expect("whole round evals");
                let have: Vec<Fr> = got.toom_evals().expect("sharded round evals");
                assert_eq!(
                    have, want,
                    "shards={shards} round {round}: the sharded round message must equal the \
                     whole-domain one — each lane is a sum over the remaining cycles, so the \
                     windows' partials add, and at the boundary the shards' fully bound scalars \
                     become the n-element tables of the tail sumcheck",
                );
                let challenge = fr(1_000 + 17 * round as u64);
                expected.bind(context, challenge).expect("whole bind");
                got.bind(challenge).expect("sharded bind");
            }

            let want: Vec<Fr> = expected.factor_finals(context).expect("whole finals");
            let have: Vec<Fr> = got.factor_finals().expect("sharded finals");
            assert_eq!(have, want, "shards={shards}: the factor finals diverged");
            assert_ne!(
                want.first().copied(),
                Some(Fr::from_u64(0)),
                "a degenerate fixture would hide a divergence",
            );
        }
    }
}
