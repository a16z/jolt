use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_field::{Fr, JoltField};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, UnivariatePoly};

use crate::cuda::common::context::{context_for, CudaKernelContext, BLOCK};
use crate::cuda::common::device::{fr_into, require_fr, require_fr_slice, DeviceFrVec, LIMBS};
use crate::cuda::common::devices::{fan_out, DeviceTask};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::primitives::reduce_lanes;

const fn absent() -> CudaError {
    CudaError::InvariantViolation {
        reason: "a cycle-round window names an absent device",
    }
}

const MAX_FACTORS: usize = 9;

pub struct CycleShard {
    pub ordinal: usize,
    pub factors: Vec<DeviceFrVec>,
}

pub struct DeviceCycleRounds {
    eq: GruenSplitEqPolynomial<Fr>,
    shards: Vec<CycleShard>,
    collapsed: Option<Vec<DeviceFrVec>>,
    local_rounds: usize,
    rounds: usize,
    rounds_bound: usize,
}

impl DeviceCycleRounds {
    pub fn from_windows<F: JoltField>(
        point: &[F],
        shards: Vec<CycleShard>,
        rounds: usize,
    ) -> Result<Self, CudaError> {
        Self::assemble(require_fr_slice(point)?, shards, rounds)
    }

    fn assemble(point: &[Fr], shards: Vec<CycleShard>, rounds: usize) -> Result<Self, CudaError> {
        if point.len() != rounds {
            return Err(CudaError::LengthMismatch {
                expected: rounds,
                got: point.len(),
            });
        }
        let count = shards.len();
        if count == 0 || !count.is_power_of_two() {
            return Err(CudaError::InvariantViolation {
                reason: "the CUDA cycle rounds need a power-of-two window count",
            });
        }
        let tail_rounds = count.trailing_zeros() as usize;
        if tail_rounds > rounds {
            return Err(CudaError::InvariantViolation {
                reason: "the CUDA cycle rounds cannot split more windows than they have rounds",
            });
        }
        let lanes = shards.first().map(|shard| shard.factors.len()).ok_or(
            CudaError::InvariantViolation {
                reason: "the CUDA cycle rounds need at least one window",
            },
        )?;
        if lanes > MAX_FACTORS {
            return Err(CudaError::NotImplemented {
                kernel: "the CUDA cycle-round quotient supports at most 8 virtual ra polynomials",
            });
        }
        let expected = 1usize << (rounds - tail_rounds);
        for shard in &shards {
            if shard.factors.len() != lanes {
                return Err(CudaError::LengthMismatch {
                    expected: lanes,
                    got: shard.factors.len(),
                });
            }
            for table in &shard.factors {
                if table.len() != expected {
                    return Err(CudaError::LengthMismatch {
                        expected,
                        got: table.len(),
                    });
                }
            }
        }
        Ok(Self {
            eq: GruenSplitEqPolynomial::new(point, BindingOrder::LowToHigh),
            shards,
            collapsed: None,
            local_rounds: rounds - tail_rounds,
            rounds,
            rounds_bound: 0,
        })
    }

    fn quotient_evals(&self, context: &CudaKernelContext) -> Result<Vec<Fr>, CudaError> {
        let half = 1usize << (self.rounds - self.rounds_bound - 1);
        if let Some(factors) = &self.collapsed {
            return Self::window_quotient(
                context,
                factors,
                self.eq.e_in_current(),
                self.eq.e_out_current(),
                half,
            );
        }
        let count = self.shards.len();
        let e_in = self.eq.e_in_current();
        let e_out = self.eq.e_out_current();
        if !e_out.len().is_multiple_of(count) {
            return Err(CudaError::InvariantViolation {
                reason: "a cycle-round window split needs the outer eq factor to divide by the \
                         window count",
            });
        }
        let stride = e_out.len() / count;
        let tasks: Vec<DeviceTask<'_, Vec<Fr>, CudaError>> = self
            .shards
            .iter()
            .enumerate()
            .map(|(index, shard)| {
                let task: DeviceTask<'_, Vec<Fr>, CudaError> = Box::new(move || {
                    let device = context_for(shard.ordinal).ok_or(absent())?;
                    let window = e_out.get(index * stride..(index + 1) * stride).ok_or(
                        CudaError::InvariantViolation {
                            reason: "a cycle-round window lies outside the outer eq factor",
                        },
                    )?;
                    Self::window_quotient(device, &shard.factors, e_in, window, half / count)
                });
                task
            })
            .collect();
        let mut total: Vec<Fr> = Vec::new();
        for part in fan_out(tasks)? {
            if total.is_empty() {
                total = part;
                continue;
            }
            if part.len() != total.len() {
                return Err(CudaError::LengthMismatch {
                    expected: total.len(),
                    got: part.len(),
                });
            }
            for (slot, value) in total.iter_mut().zip(&part) {
                *slot += *value;
            }
        }
        Ok(total)
    }

    fn window_quotient(
        context: &CudaKernelContext,
        factors: &[DeviceFrVec],
        host_in: &[Fr],
        host_out: &[Fr],
        half: usize,
    ) -> Result<Vec<Fr>, CudaError> {
        let lanes = CudaKernelContext::count_of(factors.len())?;
        let half_count = CudaKernelContext::count_of(half)?;

        let e_in = context.upload(host_in)?;
        let e_out = context.upload(host_out)?;
        if e_in.len() * e_out.len() != half {
            return Err(CudaError::LengthMismatch {
                expected: half,
                got: e_in.len() * e_out.len(),
            });
        }
        let in_len = CudaKernelContext::count_of(e_in.len())?;
        let out_len = CudaKernelContext::count_of(e_out.len())?;
        let in_bits = e_in.len().trailing_zeros();

        let handles: Vec<&DeviceFrVec> = factors.iter().collect();
        let pointers = context.device_pointers(&handles)?;
        let blocks = half_count.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc((lanes as usize + 1) * blocks as usize)?;

        let mut builder = context.stream().launch_builder(context.cr_quotient());
        let _ = builder.arg(&pointers);
        let _ = builder.arg(&lanes);
        let _ = builder.arg(e_in.limbs());
        let _ = builder.arg(&in_len);
        let _ = builder.arg(e_out.limbs());
        let _ = builder.arg(&out_len);
        let _ = builder.arg(&in_bits);
        let _ = builder.arg(&half_count);
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: thread `y < half` reads `table[2y]`/`table[2y+1]` for each of
        // `lanes` tables holding `2 * half` elements, plus `e_in[y & (in_len-1)]`
        // and `e_out[y >> in_bits]` — in range because `in_len * out_len == half`
        // is checked in `assemble`. Writes go to
        // `partials[lane * gridDim.x + blockIdx.x]`, one slot per (lane, block)
        // of `lanes * blocks`. Shared memory is `BLOCK * LIMBS` u64s, matching
        // `shared_mem_bytes`. Every buffer outlives the launch.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;
        context.stream().synchronize()?;

        reduce_lanes(context, partials, lanes + 1, blocks)?.to_host()
    }

    pub fn round_message<F: JoltField>(
        &self,
        context: &CudaKernelContext,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, CudaError> {
        let quotient = self.quotient_evals(context)?;
        let poly = self
            .eq
            .gruen_poly_from_evals(
                &quotient[..quotient.len() - 1],
                require_fr(previous_claim)?,
                || quotient[quotient.len() - 1],
            )
            .map_err(|_| CudaError::InvariantViolation {
                reason: "read-RAF quotient disagrees with the input claim",
            })?;
        let coefficients = poly
            .into_coefficients()
            .into_iter()
            .map(|value| {
                fr_into(value).ok_or(CudaError::NotImplemented {
                    kernel: "CUDA kernels support only the BN254 scalar field",
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(UnivariatePoly::new(coefficients))
    }

    pub fn bind<F: JoltField>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        let challenge = require_fr(challenge)?;
        if let Some(factors) = &mut self.collapsed {
            for table in factors.iter_mut() {
                *table = context.bind(table, challenge, BindingOrder::LowToHigh)?;
            }
        } else {
            let tasks: Vec<DeviceTask<'_, (), CudaError>> = self
                .shards
                .iter_mut()
                .map(|shard| {
                    let task: DeviceTask<'_, (), CudaError> = Box::new(move || {
                        let device = context_for(shard.ordinal).ok_or(absent())?;
                        for table in &mut shard.factors {
                            *table = device.bind(table, challenge, BindingOrder::LowToHigh)?;
                        }
                        Ok(())
                    });
                    task
                })
                .collect();
            let _ = fan_out(tasks)?;
        }
        self.eq.bind(challenge);
        self.rounds_bound += 1;
        if self.collapsed.is_none()
            && self.shards.len() > 1
            && self.rounds_bound == self.local_rounds
        {
            self.collapse(context)?;
        }
        Ok(())
    }

    fn collapse(&mut self, context: &CudaKernelContext) -> Result<(), CudaError> {
        let tasks: Vec<DeviceTask<'_, Vec<Fr>, CudaError>> = self
            .shards
            .iter()
            .map(|shard| {
                let task: DeviceTask<'_, Vec<Fr>, CudaError> = Box::new(move || {
                    let _ = context_for(shard.ordinal).ok_or(absent())?;
                    shard
                        .factors
                        .iter()
                        .map(DeviceFrVec::first)
                        .collect::<Result<Vec<Fr>, CudaError>>()
                });
                task
            })
            .collect();
        let mut columns: Vec<Vec<Fr>> = Vec::new();
        for scalars in fan_out(tasks)? {
            if columns.is_empty() {
                columns = scalars.iter().map(|_| Vec::new()).collect();
            }
            if scalars.len() != columns.len() {
                return Err(CudaError::LengthMismatch {
                    expected: columns.len(),
                    got: scalars.len(),
                });
            }
            for (column, value) in columns.iter_mut().zip(&scalars) {
                column.push(*value);
            }
        }
        let mut factors = Vec::with_capacity(columns.len());
        for column in &columns {
            factors.push(context.upload(column)?);
        }
        self.collapsed = Some(factors);
        self.shards.clear();
        Ok(())
    }

    pub fn ra_finals<F: JoltField>(
        &self,
        _context: &CudaKernelContext,
    ) -> Result<Vec<F>, CudaError> {
        let factors = match &self.collapsed {
            Some(factors) => factors.as_slice(),
            None => self
                .shards
                .first()
                .map(|shard| shard.factors.as_slice())
                .ok_or(CudaError::InvariantViolation {
                    reason: "the CUDA cycle rounds lost their tables before the final claims",
                })?,
        };
        factors
            .iter()
            .skip(1)
            .map(|table| {
                let value = table.first()?;
                fr_into(value).ok_or(CudaError::NotImplemented {
                    kernel: "CUDA kernels support only the BN254 scalar field",
                })
            })
            .collect()
    }
}
