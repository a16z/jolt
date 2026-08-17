use std::sync::Arc;

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use jolt_field::{Field, Fr};

use crate::cuda::common::context::{CudaKernelContext, BLOCK};
use crate::cuda::common::device::{fr_into, require_fr, require_fr_slice, DeviceFrVec, LIMBS};
use crate::cuda::common::error::CudaError;

pub const COLLAPSE_AFTER_ROUNDS: usize = 5;

pub const MAX_LANES: usize = 8;

pub const PACKED_BITS: usize = 32;

pub const MESSAGE_STRIP: usize = 1;

pub struct DeviceBytecodeRa {
    pc: Arc<CudaSlice<u32>>,
    tables: DeviceFrVec,
    dense: Option<DeviceFrVec>,
    polys: usize,
    addresses: usize,
    chunk_bits: usize,
    cycles: usize,
    rounds_bound: usize,
}

impl DeviceBytecodeRa {
    pub fn new<F: Field>(
        context: &CudaKernelContext,
        pc: Arc<CudaSlice<u32>>,
        cycles: usize,
        chunk_bits: usize,
        chunk_points: &[Vec<F>],
    ) -> Result<Self, CudaError> {
        let polys = chunk_points.len();
        if chunk_bits == 0 || chunk_points.iter().any(|point| point.len() != chunk_bits) {
            return Err(CudaError::InvariantViolation {
                reason: "a bytecode one-hot chunk needs one address coordinate per chunk bit",
            });
        }
        if !cycles.is_power_of_two() {
            return Err(CudaError::InvariantViolation {
                reason: "a bytecode one-hot family needs a power-of-two cycle count",
            });
        }
        if polys == 0 || polys + 1 > MAX_LANES || polys * chunk_bits > PACKED_BITS {
            return Err(CudaError::NotImplemented {
                kernel: "the CUDA bytecode read-RAF kernels pack the bytecode PC into one 32-bit \
                         word and evaluate at most eight round-polynomial lanes",
            });
        }
        if pc.len() != cycles {
            return Err(CudaError::LengthMismatch {
                expected: cycles,
                got: pc.len(),
            });
        }

        let flat: Vec<F> = chunk_points.iter().flatten().copied().collect();
        let point = context.upload(require_fr_slice(&flat)?)?;
        let mut tables = context.upload(&vec![Fr::from(1u64); polys])?;
        for level in 0..chunk_bits {
            let prev_len = 1usize << level;
            let mut next = context.alloc(polys * prev_len * 2)?;
            let count = CudaKernelContext::count_of(polys * prev_len)?;
            let poly_count = CudaKernelContext::count_of(polys)?;
            let prev = CudaKernelContext::count_of(prev_len)?;
            let level_arg = CudaKernelContext::count_of(level)?;
            let bits = CudaKernelContext::count_of(chunk_bits)?;
            let mut builder = context.stream().launch_builder(context.irv_eq_double());
            let _ = builder.arg(tables.limbs());
            let _ = builder.arg(point.limbs());
            let _ = builder.arg(next.limbs_mut());
            let _ = builder.arg(&poly_count);
            let _ = builder.arg(&prev);
            let _ = builder.arg(&level_arg);
            let _ = builder.arg(&bits);
            // SAFETY: thread `idx < polys * prev_len` reads `in[idx]` and
            // `point[(idx / prev_len) * chunk_bits + level]` — inside `point`'s
            // `polys * chunk_bits` because `level < chunk_bits` — and writes
            // `out[p * 2 * prev_len + 2 * i]` and `out[... + 1]` for
            // `p = idx / prev_len`, `i = idx % prev_len`, both inside `out`'s
            // `2 * polys * prev_len`. Index sets are disjoint across threads and `out`
            // is a fresh allocation.
            let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
            tables = next;
        }

        Ok(Self {
            pc,
            tables,
            dense: None,
            polys,
            addresses: 1usize << chunk_bits,
            chunk_bits,
            cycles,
            rounds_bound: 0,
        })
    }

    pub const fn lanes(&self) -> usize {
        self.polys + 1
    }

    pub const fn len(&self) -> usize {
        self.cycles >> self.rounds_bound
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the collapse assertion is the round-for-round test's guard against a \
                      config that never leaves the sparse form"
        )
    )]
    pub const fn is_collapsed(&self) -> bool {
        self.dense.is_some()
    }

    const fn slots(&self) -> usize {
        1usize << self.rounds_bound
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the per-round coefficient view is the round-for-round test's entry point"
        )
    )]
    pub fn coefficients(&self, context: &CudaKernelContext) -> Result<DeviceFrVec, CudaError> {
        match &self.dense {
            Some(dense) => dense.try_clone(),
            None => self.gather(context),
        }
    }

    fn gather(&self, context: &CudaKernelContext) -> Result<DeviceFrVec, CudaError> {
        let len = self.len();
        let mut out = context.alloc(self.polys * len)?;
        if len == 0 {
            return Ok(out);
        }

        let addresses = CudaKernelContext::count_of(self.addresses)?;
        let slots = CudaKernelContext::count_of(self.slots())?;
        let bits = CudaKernelContext::count_of(self.chunk_bits)?;
        let polys = CudaKernelContext::count_of(self.polys)?;
        let count = CudaKernelContext::count_of(len)?;
        let mut builder = context.stream().launch_builder(context.brr_gather());
        let _ = builder.arg(self.pc.as_ref());
        let _ = builder.arg(self.tables.limbs());
        let _ = builder.arg(&addresses);
        let _ = builder.arg(&slots);
        let _ = builder.arg(&bits);
        let _ = builder.arg(&polys);
        let _ = builder.arg(out.limbs_mut());
        let _ = builder.arg(&count);
        // SAFETY: thread `(p = blockIdx.y < polys, j < len)` reads
        // `pc[j * slots + s]` for `s < slots` — every cycle index is below
        // `len * slots == cycles`, so inside `pc` — and, only for a cycle whose
        // packed word is not the cold sentinel,
        // `tables[p * slots * addresses + s * addresses + a]` with `a` masked below
        // `addresses`, inside `tables`'s `polys * slots * addresses`. It writes only
        // `out[p * len + j]`, one slot per (poly, cycle) pair, inside `out`'s
        // `polys * len`.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (count.div_ceil(BLOCK).max(1), polys, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        Ok(out)
    }

    fn split_tables(
        &mut self,
        context: &CudaKernelContext,
        challenge: Fr,
    ) -> Result<(), CudaError> {
        let eq_zero = context.upload(&[Fr::from(1u64) - challenge])?;
        let eq_one = context.upload(&[challenge])?;
        let len = self.slots() * self.addresses;
        let mut next = context.alloc(self.polys * len * 2)?;
        let count = CudaKernelContext::count_of(self.polys * len)?;
        let poly_count = CudaKernelContext::count_of(self.polys)?;
        let width = CudaKernelContext::count_of(len)?;
        let mut builder = context.stream().launch_builder(context.irv_tables_split());
        let _ = builder.arg(self.tables.limbs());
        let _ = builder.arg(eq_zero.limbs());
        let _ = builder.arg(eq_one.limbs());
        let _ = builder.arg(next.limbs_mut());
        let _ = builder.arg(&poly_count);
        let _ = builder.arg(&width);
        // SAFETY: thread `idx < polys * len` reads `in[idx]` plus the two
        // single-element challenges, and writes `out[p * 2 * len + i]` and
        // `out[p * 2 * len + len + i]` for `p = idx / len`, `i = idx % len` —
        // disjoint across threads and inside `out`'s `2 * polys * len`. `out` is a
        // fresh allocation, so no thread reads another's write.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
        self.tables = next;
        Ok(())
    }

    pub fn bind<F: Field>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        let challenge = require_fr(challenge)?;
        if let Some(dense) = &self.dense {
            let bound = context.bind_rows(dense, self.len(), challenge)?;
            self.dense = Some(bound);
            self.rounds_bound += 1;
            return Ok(());
        }
        if self.len() < 2 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: self.len(),
            });
        }
        self.split_tables(context, challenge)?;
        self.rounds_bound += 1;
        if self.rounds_bound >= COLLAPSE_AFTER_ROUNDS && self.len() > 1 {
            self.dense = Some(self.gather(context)?);
            self.tables = context.alloc(0)?;
            self.pc = Arc::new(context.alloc_u32(0)?);
        }
        Ok(())
    }

    pub fn final_claims<F: Field>(&self, context: &CudaKernelContext) -> Result<Vec<F>, CudaError> {
        if self.len() != 1 {
            return Err(CudaError::LengthMismatch {
                expected: 1,
                got: self.len(),
            });
        }
        let values = match &self.dense {
            Some(dense) => dense.to_host()?,
            None => self.gather(context)?.to_host()?,
        };
        values
            .into_iter()
            .map(|value| {
                fr_into(value).ok_or(CudaError::NotImplemented {
                    kernel: "CUDA kernels support only the BN254 scalar field",
                })
            })
            .collect()
    }

    pub fn round_evals<F: Field>(
        &self,
        context: &CudaKernelContext,
        coefficient: &DeviceFrVec,
    ) -> Result<Vec<F>, CudaError> {
        let half = self.len() / 2;
        if half == 0 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: self.len(),
            });
        }
        if coefficient.len() != self.len() {
            return Err(CudaError::LengthMismatch {
                expected: self.len(),
                got: coefficient.len(),
            });
        }

        let lanes = self.lanes();
        let threads = half.div_ceil(MESSAGE_STRIP);
        let thread_count = CudaKernelContext::count_of(threads)?;
        let blocks = thread_count.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc(lanes * blocks as usize)?;
        let polys = CudaKernelContext::count_of(self.polys)?;
        let lane_count = CudaKernelContext::count_of(lanes)?;
        let half_count = CudaKernelContext::count_of(half)?;
        let strip = CudaKernelContext::count_of(MESSAGE_STRIP)?;
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
        };

        if let Some(dense) = &self.dense {
            let mut builder = context.stream().launch_builder(context.brr_message_dense());
            let _ = builder.arg(dense.limbs());
            let _ = builder.arg(coefficient.limbs());
            let _ = builder.arg(&polys);
            let _ = builder.arg(&lane_count);
            let _ = builder.arg(&half_count);
            let _ = builder.arg(&strip);
            let _ = builder.arg(partials.limbs_mut());
            // SAFETY: the thread walks `strip` pair indices from
            // `tid * strip`, skipping every `g >= half`, and for each reads
            // `coefficient[2g]`, `coefficient[2g + 1]` — inside its `2 * half`
            // elements — and `dense[p * 2 * half + 2g (+1)]` for every `p < polys`,
            // inside `dense`'s `polys * 2 * half`. It writes only
            // `partials[lane * gridDim.x + blockIdx.x]` for `lane < lanes`, of
            // `lanes * blocks`; `lanes <= MAX_LANES` bounds the register arrays.
            // Shared memory is `BLOCK * LIMBS` u64s, matching `shared_mem_bytes`, and
            // the block reduction sits after the loop with no early return, so every
            // thread reaches each `__syncthreads()`.
            let _ = unsafe { builder.launch(config) }?;
        } else {
            let addresses = CudaKernelContext::count_of(self.addresses)?;
            let slots = CudaKernelContext::count_of(self.slots())?;
            let bits = CudaKernelContext::count_of(self.chunk_bits)?;
            let mut builder = context
                .stream()
                .launch_builder(context.brr_message_sparse());
            let _ = builder.arg(self.pc.as_ref());
            let _ = builder.arg(self.tables.limbs());
            let _ = builder.arg(coefficient.limbs());
            let _ = builder.arg(&addresses);
            let _ = builder.arg(&slots);
            let _ = builder.arg(&bits);
            let _ = builder.arg(&polys);
            let _ = builder.arg(&lane_count);
            let _ = builder.arg(&half_count);
            let _ = builder.arg(&strip);
            let _ = builder.arg(partials.limbs_mut());
            // SAFETY: the thread walks `strip` pair indices from `tid * strip`,
            // skipping every `g >= half`, and for each reads `coefficient[2g (+1)]`
            // inside its `2 * half` elements; `pc` at cycle indices
            // `2 * g * slots + t` for `t < 2 * slots`, all below
            // `2 * half * slots == cycles`; and `tables` at
            // `p * slots * addresses + s * addresses + a` with `a` masked below
            // `addresses` and `p < polys`, inside `tables`'s
            // `polys * slots * addresses`. It writes only
            // `partials[lane * gridDim.x + blockIdx.x]` for `lane < lanes`, of
            // `lanes * blocks`; `lanes <= MAX_LANES` bounds the register arrays.
            // Shared memory matches `shared_mem_bytes` and the block reduction sits
            // after the loop with no early return.
            let _ = unsafe { builder.launch(config) }?;
        }

        let totals = crate::cuda::common::dense_product::DeviceDenseProduct::reduce_lanes(
            context, partials, lane_count, blocks,
        )?;
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
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use std::sync::Arc;

    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{BindingOrder, EqPolynomial, Polynomial};
    use proptest::prelude::*;

    use super::{DeviceBytecodeRa, COLLAPSE_AFTER_ROUNDS};
    use crate::cuda::bytecode_read_raf::witness::COLD;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{arb_point, fr};

    fn mix(seed: u64, cycle: usize) -> u64 {
        let value = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(cycle as u64 + 1);
        value.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ (value >> 29)
    }

    fn column(seed: u64, cycles: usize, span: u32, cold_every: usize) -> Vec<u32> {
        (0..cycles)
            .map(|cycle| {
                if cycle % cold_every == 1 {
                    COLD
                } else {
                    (mix(seed, cycle) % u64::from(span)) as u32
                }
            })
            .collect()
    }

    fn expected_tables(
        pc: &[u32],
        chunk_bits: usize,
        chunk_points: &[Vec<Fr>],
    ) -> Vec<Polynomial<Fr>> {
        let addresses = 1usize << chunk_bits;
        let polys = chunk_points.len();
        chunk_points
            .iter()
            .enumerate()
            .map(|(index, point)| {
                let table = EqPolynomial::<Fr>::evals(point, None);
                let shift = chunk_bits * (polys - 1 - index);
                Polynomial::new(
                    pc.iter()
                        .map(|&word| {
                            if word == COLD {
                                Fr::from_u64(0)
                            } else {
                                table[((word >> shift) as usize) & (addresses - 1)]
                            }
                        })
                        .collect(),
                )
            })
            .collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(6))]
        #[test]
        fn device_bytecode_ra_matches_cpu_round_for_round(
            log_t in (COLLAPSE_AFTER_ROUNDS + 1)..9usize,
            chunk_bits in prop::sample::select(vec![4usize, 8]),
            polys in 1usize..5,
            cold_every in 2usize..5,
            seed in any::<u64>(),
            challenges in arb_point(8),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            if chunk_bits * polys > 32 || polys + 1 > super::MAX_LANES {
                return Ok(());
            }
            let cycles = 1usize << log_t;
            let chunk_points: Vec<Vec<Fr>> = (0..polys)
                .map(|p| {
                    (0..chunk_bits)
                        .map(|i| fr(seed ^ ((p * 97 + i) as u64 * 37 + 11)))
                        .collect()
                })
                .collect();
            let span = 1u32 << (chunk_bits * polys - 1);
            let pc = column(seed, cycles, span, cold_every);

            let mut expected = expected_tables(&pc, chunk_bits, &chunk_points);
            let mut got = DeviceBytecodeRa::new(
                context,
                Arc::new(context.upload_u32_slice(&pc).expect("upload the pc column")),
                cycles,
                chunk_bits,
                &chunk_points,
            )
            .expect("device bytecode one-hot family");

            for (round, &challenge) in challenges.iter().take(log_t).enumerate() {
                let coefficients = got
                    .coefficients(context)
                    .expect("gather coefficients")
                    .to_host()
                    .expect("download coefficients");
                let len = got.len();
                for (p, poly) in expected.iter().enumerate() {
                    prop_assert_eq!(
                        &coefficients[p * len..(p + 1) * len],
                        &poly.evals()[..len],
                        "chunk {} diverged at round {}", p, round
                    );
                }
                for poly in &mut expected {
                    poly.bind_with_order(challenge, BindingOrder::LowToHigh);
                }
                got.bind(context, challenge).expect("device bind");
            }

            prop_assert_eq!(
                got.final_claims::<Fr>(context).expect("device final claims"),
                expected
                    .iter()
                    .map(|poly| poly.evals()[0])
                    .collect::<Vec<Fr>>(),
                "final claims diverged"
            );
            prop_assert!(got.is_collapsed(), "the family never collapsed to dense arrays");
        }
    }
}
