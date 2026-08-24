#![expect(
    dead_code,
    reason = "the host-table constructor and round counters are exercised only by the tests"
)]

use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_field::{Field, Fr};
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
    pub fn new<F: Field>(
        context: &CudaKernelContext,
        point: &[F],
        combined_val: &[F],
        ra: &[Vec<F>],
        rounds: usize,
    ) -> Result<Self, CudaError> {
        let mut factors = Vec::with_capacity(ra.len() + 1);
        factors.push(context.upload(require_fr_slice(combined_val)?)?);
        for table in ra {
            factors.push(context.upload(require_fr_slice(table)?)?);
        }
        Self::assemble(
            require_fr_slice(point)?,
            vec![CycleShard {
                ordinal: context.ordinal(),
                factors,
            }],
            rounds,
        )
    }

    pub fn from_windows<F: Field>(
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
        let mut partials = context.alloc(lanes as usize * blocks as usize)?;

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

        reduce_lanes(context, partials, lanes, blocks)?.to_host()
    }

    pub fn round_message<F: Field>(
        &self,
        context: &CudaKernelContext,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, CudaError> {
        let quotient = self.quotient_evals(context)?;
        let poly = self
            .eq
            .gruen_poly_from_evals(&quotient, require_fr(previous_claim)?);
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

    pub fn bind<F: Field>(
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

    pub fn ra_finals<F: Field>(&self, _context: &CudaKernelContext) -> Result<Vec<F>, CudaError> {
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

    pub const fn rounds_bound(&self) -> usize {
        self.rounds_bound
    }

    pub const fn rounds(&self) -> usize {
        self.rounds
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
    use jolt_field::Fr;
    use jolt_lookup_tables::tables::LookupTableKind;
    use jolt_lookup_tables::XLEN as RISCV_XLEN;
    use jolt_poly::UnivariatePoly;
    use jolt_witness::witnesses::{InstructionRafFlag, LookupIndex, TableIndex};
    use proptest::prelude::*;
    use std::num::NonZeroUsize;

    use super::DeviceCycleRounds;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::fr;
    use crate::reference::instruction_read_raf::{
        InstructionReadRafKernel, InstructionReadRafWitness,
    };

    const ADDRESS_BITS: usize = 128;
    const RA_COUNT: usize = 8;

    fn rows(log_t: usize, seed: u64) -> Vec<InstructionReadRafWitness> {
        let tables: Vec<LookupTableKind<RISCV_XLEN>> =
            LookupTableKind::<RISCV_XLEN>::iter().collect();
        (0..1usize << log_t)
            .map(|j| {
                let mixed = (j as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(seed);
                let index = (u128::from(mixed) << 61) | u128::from(mixed.rotate_left(17));
                InstructionReadRafWitness {
                    lookup_index: LookupIndex(index),
                    table_index: TableIndex(if mixed.is_multiple_of(11) {
                        None
                    } else {
                        Some(tables[(mixed % tables.len() as u64) as usize].index())
                    }),
                    raf_flag: InstructionRafFlag(mixed.is_multiple_of(3)),
                }
            })
            .collect()
    }

    fn reference_at_cycle_rounds(
        log_t: usize,
        seed: u64,
        r_reduction: &[Fr],
    ) -> InstructionReadRafKernel<Fr> {
        let dimensions = InstructionReadRafDimensions::new(
            log_t,
            ADDRESS_BITS,
            NonZeroUsize::new(RA_COUNT).unwrap(),
        );
        let mut kernel =
            InstructionReadRafKernel::new(dimensions, r_reduction, rows(log_t, seed), fr(seed + 1))
                .expect("reference kernel");
        for round in 0..ADDRESS_BITS {
            kernel
                .bind(fr(seed + round as u64 + 71))
                .expect("reference bind");
        }
        kernel
    }

    #[test]
    fn windowed_cycle_rounds_match_the_whole_domain_round_for_round() {
        let Some(context) = shared_context() else {
            return;
        };
        const LOG_T: usize = 8;
        let seed = 0x51DEu64;
        let r_reduction: Vec<Fr> = (0..LOG_T).map(|i| fr(seed + i as u64 + 3)).collect();
        let host = reference_at_cycle_rounds(LOG_T, seed, &r_reduction);
        let tables = host.cycle_tables.as_ref().expect("cycle tables");
        let combined: Vec<Fr> = tables.combined_val.evals().to_vec();
        let ra: Vec<Vec<Fr>> = tables.ra.iter().map(|p| p.evals().to_vec()).collect();

        for shards in [2usize, 4, 8] {
            let mut expected = DeviceCycleRounds::new(context, &r_reduction, &combined, &ra, LOG_T)
                .expect("whole-domain cycle rounds");
            let len = (1usize << LOG_T) / shards;
            let windows: Vec<super::CycleShard> = (0..shards)
                .map(|shard| {
                    let range = shard * len..(shard + 1) * len;
                    let mut factors = vec![context
                        .upload(&combined[range.clone()])
                        .expect("upload the combined window")];
                    for table in &ra {
                        factors.push(
                            context
                                .upload(&table[range.clone()])
                                .expect("upload an ra window"),
                        );
                    }
                    super::CycleShard {
                        ordinal: 0,
                        factors,
                    }
                })
                .collect();
            let mut got = DeviceCycleRounds::from_windows(&r_reduction, windows, LOG_T)
                .expect("windowed cycle rounds");

            let mut claim = fr(seed + 909);
            for round in 0..LOG_T {
                let want: UnivariatePoly<Fr> = expected
                    .round_message(context, claim)
                    .expect("whole-domain message");
                let have: UnivariatePoly<Fr> =
                    got.round_message(context, claim).expect("windowed message");
                assert_eq!(
                    have.coefficients(),
                    want.coefficients(),
                    "shards={shards} round {round}: the cycle round polynomial diverged",
                );
                let challenge = fr(seed + round as u64 + 311);
                claim = want.evaluate(challenge);
                expected
                    .bind(context, challenge)
                    .expect("whole-domain bind");
                got.bind(context, challenge).expect("windowed bind");
            }

            let want: Vec<Fr> = expected.ra_finals(context).expect("whole-domain ra finals");
            let have: Vec<Fr> = got.ra_finals(context).expect("windowed ra finals");
            assert_eq!(have, want, "shards={shards}: the ra final claims diverged");
            assert_eq!(want.len(), RA_COUNT);
            assert_ne!(
                want.first().copied(),
                Some(Fr::from(0u64)),
                "a degenerate fixture would hide a divergence",
            );
        }
    }

    proptest! {
        #[test]
        fn cycle_rounds_match_the_reference_round_for_round(
            log_t in 4usize..=8,
            seed in any::<u64>(),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let r_reduction: Vec<Fr> = (0..log_t).map(|i| fr(seed + i as u64 + 3)).collect();
            let mut host = reference_at_cycle_rounds(log_t, seed, &r_reduction);
            let tables = host.cycle_tables.as_ref().expect("cycle tables");
            let ra: Vec<Vec<Fr>> = tables.ra.iter().map(|p| p.evals().to_vec()).collect();
            let mut device = DeviceCycleRounds::new(
                context,
                &r_reduction,
                tables.combined_val.evals(),
                &ra,
                log_t,
            )
            .expect("device cycle rounds");

            let first = host.cycle_message().expect("reference cycle message");
            let mut claim = first[0] + first[1];

            for round in 0..log_t {
                let expected = UnivariatePoly::from_evals(
                    &host.cycle_message().expect("reference cycle message"),
                );
                let got: UnivariatePoly<Fr> = device
                    .round_message(context, claim)
                    .expect("device message");
                prop_assert_eq!(
                    got.coefficients(),
                    expected.coefficients(),
                    "cycle round {} polynomial diverged",
                    round
                );

                let challenge = fr(seed + round as u64 + 211);
                claim = expected.evaluate(challenge);
                host.bind(challenge).expect("reference bind");
                device.bind(context, challenge).expect("device bind");
            }

            let expected: Vec<Fr> = host
                .cycle_tables
                .as_ref()
                .expect("cycle tables")
                .ra
                .iter()
                .map(|ra| ra.evals()[0])
                .collect();
            let got: Vec<Fr> = device.ra_finals(context).expect("device ra finals");
            prop_assert_eq!(got, expected, "ra final claims diverged");
        }
    }
}
