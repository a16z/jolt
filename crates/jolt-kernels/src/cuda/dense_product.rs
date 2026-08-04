use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_claims::protocols::jolt::JoltChallengeId;
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck};
use jolt_field::{Field, Fr};
use jolt_poly::{BindingOrder, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckOutputClaims,
};

use super::context::{CudaKernelContext, BLOCK};
use super::device::{as_fr_slice, fr_into, DeviceFrVec, LIMBS};
use super::error::CudaError;

pub struct DeviceDenseProduct {
    tables: Vec<DeviceFrVec>,
    degree: usize,
    rounds: usize,
    rounds_bound: usize,
}

fn only_fr<F: Field>(values: &[F]) -> Result<&[Fr], CudaError> {
    as_fr_slice(values).ok_or(CudaError::NotImplemented {
        kernel: "CUDA kernels support only the BN254 scalar field",
    })
}

fn single<F: Field>(value: F) -> Result<Fr, CudaError> {
    only_fr(std::slice::from_ref(&value))?
        .first()
        .copied()
        .ok_or(CudaError::LengthMismatch {
            expected: 1,
            got: 0,
        })
}

impl DeviceDenseProduct {
    pub fn new<F: Field>(
        context: &CudaKernelContext,
        weights: &[(F, Vec<F>)],
        factors: &[Vec<F>],
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
        if weights.is_empty() {
            return Err(CudaError::InvariantViolation {
                reason: "a dense product needs at least one weight table",
            });
        }

        let mut tables = Vec::with_capacity(1 + factors.len());
        tables.push(Self::combine_weights(context, weights, expected)?);
        for table in factors {
            tables.push(context.upload(only_fr(table)?)?);
        }
        Ok(Self {
            tables,
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
            let table = context.upload(only_fr(table)?)?;
            let coefficient = context.upload(&[single(*coefficient)?])?;
            let mut builder = context.stream().launch_builder(context.weighted_combine());
            let _ = builder.arg(table.limbs());
            let _ = builder.arg(coefficient.limbs());
            let _ = builder.arg(accumulator.limbs_mut());
            let _ = builder.arg(&count);
            // SAFETY: thread `i < count` reads `weights[i]` and the
            // single-element `coefficient` buffer, and read-modify-writes only
            // `accumulator[i]` — one thread per element, so the update is
            // uncontended. All buffers hold `count * LIMBS` u64s (every weight
            // table's length was checked against `len` by the caller), and
            // `table` is a distinct allocation from `accumulator`. Threads with
            // `i >= count` return before any access. The per-weight launches are
            // serialized by the synchronize below.
            let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
            context.stream().synchronize()?;
        }
        Ok(accumulator)
    }

    pub fn round_evals<F: Field>(&self, context: &CudaKernelContext) -> Result<Vec<F>, CudaError> {
        let remaining = self.rounds - self.rounds_bound;
        let half = (1usize << remaining) / 2;
        let lanes = CudaKernelContext::count_of(self.degree + 1)?;
        let half_count = CudaKernelContext::count_of(half)?;
        let table_count = CudaKernelContext::count_of(self.tables.len())?;

        let handles: Vec<&DeviceFrVec> = self.tables.iter().collect();
        let pointers = context.device_pointers(&handles)?;
        let blocks = half_count.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc(lanes as usize * blocks as usize)?;
        let mut builder = context
            .stream()
            .launch_builder(context.dense_product_round());
        let _ = builder.arg(&pointers);
        let _ = builder.arg(&table_count);
        let _ = builder.arg(&half_count);
        let _ = builder.arg(&lanes);
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: for each lane `c`, thread `y < half` reads the pair
        // (`table[2y]`, `table[2y+1]`) of every table addressed by `pointers` —
        // each holds `2 * half * LIMBS` u64s at this round, and `pointers` holds
        // exactly `table_count` addresses of the live `DeviceFrVec`s borrowed
        // through `&self`. The shared-memory tree is sized `BLOCK * LIMBS` u64s
        // by `shared_mem_bytes` below, with `__syncthreads()` between every level
        // and after each lane's write, so lanes never overlap. Only thread 0
        // writes, to `partials[c * gridDim.x + blockIdx.x]`, and `partials` holds
        // `lanes * blocks` elements — one slot per (lane, block), non-aliasing.
        // Threads with `y >= half` contribute field zero without dereferencing a
        // table.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;
        context.stream().synchronize()?;

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
            // SAFETY: thread `(i, lane)` with `i < next` and `lane < lanes` reads
            // `in[lane * width + i]` plus, when `i + next < width`,
            // `in[lane * width + i + next]` — both inside `in`'s
            // `lanes * width * LIMBS` u64s — and writes only
            // `out[lane * next + i]` of `lanes * next * LIMBS`. `out` is a fresh
            // allocation distinct from `in`, and the `(lane, i)` index sets are
            // pairwise disjoint across threads.
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
        let challenge = single(challenge)?;
        for table in &mut self.tables {
            *table = context.bind(table, challenge, BindingOrder::LowToHigh)?;
        }
        self.rounds_bound += 1;
        Ok(())
    }

    pub fn factor_finals<F: Field>(&self) -> Result<Vec<F>, CudaError> {
        self.tables
            .iter()
            .skip(1)
            .map(|table| {
                fr_into(table.first()?).ok_or(CudaError::NotImplemented {
                    kernel: "CUDA kernels support only the BN254 scalar field",
                })
            })
            .collect()
    }

    pub const fn rounds_bound(&self) -> usize {
        self.rounds_bound
    }
}

pub struct DenseProductKernel<F: Field, R> {
    pub(super) state: DeviceDenseProduct,
    pub(super) relation: R,
    pub(super) context: &'static CudaKernelContext,
    pub(super) field: core::marker::PhantomData<F>,
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
        let evals: Vec<F> = self.state.round_evals(self.context).map_err(|_| {
            SumcheckError::MissingEvaluationSource {
                kind: "cuda round_evals",
            }
        })?;
        let round_sum = evals[0] + evals[1];
        if round_sum != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: round_sum,
            });
        }
        Ok(UnivariatePoly::from_evals(&evals))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.state
            .bind(self.context, bind)
            .map_err(|_| SumcheckError::MissingEvaluationSource { kind: "cuda bind" })
    }
}
