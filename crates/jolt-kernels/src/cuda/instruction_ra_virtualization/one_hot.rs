use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use jolt_field::{Field, Fr};
use jolt_poly::BindingOrder;

use crate::cuda::common::context::{CudaKernelContext, BLOCK};
use crate::cuda::common::device::{require_fr, require_fr_slice, DeviceFrVec, LIMBS};
use crate::cuda::common::error::CudaError;

pub const COLLAPSE_AFTER_ROUNDS: usize = 4;

pub struct DevicePackedRa {
    packed: CudaSlice<u64>,
    tables: DeviceFrVec,
    dense: Vec<DeviceFrVec>,
    polys: usize,
    addresses: usize,
    chunk_bits: usize,
    cycles: usize,
    rounds_bound: usize,
}

impl DevicePackedRa {
    pub fn new<F: Field>(
        context: &CudaKernelContext,
        packed: CudaSlice<u64>,
        cycles: usize,
        chunk_bits: usize,
        address_point: &[F],
        seeds: &[F],
    ) -> Result<Self, CudaError> {
        let polys = seeds.len();
        if !cycles.is_power_of_two() || chunk_bits == 0 || !64usize.is_multiple_of(chunk_bits) {
            return Err(CudaError::InvariantViolation {
                reason: "a packed one-hot family needs a power-of-two cycle count and a chunk \
                         width dividing 64, so no chunk straddles a word boundary",
            });
        }
        if packed.len() != cycles * 2 {
            return Err(CudaError::LengthMismatch {
                expected: cycles * 2,
                got: packed.len(),
            });
        }
        if address_point.len() != polys * chunk_bits {
            return Err(CudaError::LengthMismatch {
                expected: polys * chunk_bits,
                got: address_point.len(),
            });
        }

        let point = context.upload(require_fr_slice(address_point)?)?;
        let mut tables = context.upload(require_fr_slice(seeds)?)?;
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
            // `out[2 * idx]` and `out[2 * idx + 1]` after remapping through the
            // per-poly stride, both inside `out`'s `2 * polys * prev_len`. Index
            // sets are disjoint across threads and `out` is a fresh allocation.
            let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
            context.stream().synchronize()?;
            tables = next;
        }

        Ok(Self {
            packed,
            tables,
            dense: Vec::new(),
            polys,
            addresses: 1usize << chunk_bits,
            chunk_bits,
            cycles,
            rounds_bound: 0,
        })
    }

    pub const fn len(&self) -> usize {
        self.cycles >> self.rounds_bound
    }

    pub const fn polys(&self) -> usize {
        self.polys
    }

    pub const fn rounds_bound(&self) -> usize {
        self.rounds_bound
    }

    pub const fn is_collapsed(&self) -> bool {
        !self.dense.is_empty()
    }

    const fn slots(&self) -> usize {
        1usize << self.rounds_bound
    }

    pub fn coefficients(&self, context: &CudaKernelContext) -> Result<Vec<DeviceFrVec>, CudaError> {
        if self.is_collapsed() {
            return self.dense.iter().map(DeviceFrVec::try_clone).collect();
        }
        self.gather(context)
    }

    fn gather(&self, context: &CudaKernelContext) -> Result<Vec<DeviceFrVec>, CudaError> {
        let len = self.len();
        let mut outputs = Vec::with_capacity(self.polys);
        for _ in 0..self.polys {
            outputs.push(context.alloc(len)?);
        }
        if len == 0 {
            return Ok(outputs);
        }
        let pointers = {
            let refs: Vec<&DeviceFrVec> = outputs.iter().collect();
            context.device_pointers(&refs)?
        };

        let addresses = CudaKernelContext::count_of(self.addresses)?;
        let slots = CudaKernelContext::count_of(self.slots())?;
        let bits = CudaKernelContext::count_of(self.chunk_bits)?;
        let committed = CudaKernelContext::count_of(self.polys)?;
        let count = CudaKernelContext::count_of(len)?;
        let mut builder = context.stream().launch_builder(context.irv_gather());
        let _ = builder.arg(&self.packed);
        let _ = builder.arg(self.tables.limbs());
        let _ = builder.arg(&pointers);
        let _ = builder.arg(&addresses);
        let _ = builder.arg(&slots);
        let _ = builder.arg(&bits);
        let _ = builder.arg(&committed);
        let _ = builder.arg(&count);
        // SAFETY: thread `(p = blockIdx.y < polys, j < len)` reads
        // `packed[2 * (j * slots + s)]` and its `+1` mate for `s < slots` — every
        // cycle index is below `len * slots == cycles`, so inside `packed`'s
        // `2 * cycles` — and `tables[p * slots * addresses + s * addresses + a]`
        // with `a` masked below `addresses`, inside `tables`'s
        // `polys * slots * addresses`. It writes only `out[p][j]`, one slot per
        // (poly, index) pair, and each `out[p]` is a distinct fresh allocation of
        // `len` elements whose device address is `pointers[p]`.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (count.div_ceil(BLOCK).max(1), committed, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        context.stream().synchronize()?;
        Ok(outputs)
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
        context.stream().synchronize()?;
        self.tables = next;
        Ok(())
    }

    pub fn bind<F: Field>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        let challenge = require_fr(challenge)?;
        if self.is_collapsed() {
            for table in &mut self.dense {
                *table = context.bind(table, challenge, BindingOrder::LowToHigh)?;
            }
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
            self.dense = self.gather(context)?;
            self.tables = context.alloc(0)?;
            self.packed = context.alloc_u64(0)?;
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
        let values: Result<Vec<Fr>, CudaError> = if self.is_collapsed() {
            self.dense.iter().map(DeviceFrVec::first).collect()
        } else {
            self.gather(context)?
                .iter()
                .map(DeviceFrVec::first)
                .collect()
        };
        values?
            .into_iter()
            .map(|value| {
                crate::cuda::common::device::fr_into(value).ok_or(CudaError::NotImplemented {
                    kernel: "CUDA kernels support only the BN254 scalar field",
                })
            })
            .collect()
    }

    pub fn round_evals<F: Field>(
        &self,
        context: &CudaKernelContext,
        virtual_polys: usize,
        eq: &crate::cuda::common::split_eq::DeviceSplitEq<F>,
    ) -> Result<[F; 4], CudaError> {
        let half = self.len() / 2;
        if half == 0 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: self.len(),
            });
        }
        let e_in_len = eq.e_in_len();
        if eq.e_out_current().len() * e_in_len != half {
            return Err(CudaError::LengthMismatch {
                expected: half,
                got: eq.e_out_current().len() * e_in_len,
            });
        }

        let half_count = CudaKernelContext::count_of(half)?;
        let blocks = half_count.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc(4 * blocks as usize)?;
        let groups = CudaKernelContext::count_of(virtual_polys)?;
        let e_in_arg = CudaKernelContext::count_of(e_in_len)?;
        let num_x_in_bits = e_in_len.max(1).ilog2();
        let e_in = eq.e_in_current();
        let e_out = eq.e_out_current();
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
        };

        if self.is_collapsed() {
            let refs: Vec<&DeviceFrVec> = self.dense.iter().collect();
            let pointers = context.device_pointers(&refs)?;
            let mut builder = context.stream().launch_builder(context.irv_message_dense());
            let _ = builder.arg(&pointers);
            let _ = builder.arg(&groups);
            let _ = builder.arg(&half_count);
            let _ = builder.arg(e_in.limbs());
            let _ = builder.arg(&e_in_arg);
            let _ = builder.arg(e_out.limbs());
            let _ = builder.arg(&num_x_in_bits);
            let _ = builder.arg(partials.limbs_mut());
            // SAFETY: thread `g < half` reads `dense[p][2g]` and `dense[p][2g+1]`
            // for every `p < 4 * virtual_polys` — each `dense[p]` holds
            // `2 * half` elements — plus `e_in[g & mask]` and
            // `e_out[g >> num_x_in_bits]`, both bounded because
            // `e_out.len() * e_in.len() == half` is checked above. It writes only
            // `partials[lane * gridDim.x + blockIdx.x]` of `4 * blocks`. Shared
            // memory is `BLOCK * LIMBS` u64s, matching `shared_mem_bytes`, and the
            // block reduction sits outside the `g < half` guard so every thread
            // reaches each `__syncthreads()`.
            let _ = unsafe { builder.launch(config) }?;
        } else {
            let addresses = CudaKernelContext::count_of(self.addresses)?;
            let slots = CudaKernelContext::count_of(self.slots())?;
            let bits = CudaKernelContext::count_of(self.chunk_bits)?;
            let committed = CudaKernelContext::count_of(self.polys)?;
            let mut builder = context
                .stream()
                .launch_builder(context.irv_message_sparse());
            let _ = builder.arg(&self.packed);
            let _ = builder.arg(self.tables.limbs());
            let _ = builder.arg(&groups);
            let _ = builder.arg(&addresses);
            let _ = builder.arg(&slots);
            let _ = builder.arg(&bits);
            let _ = builder.arg(&committed);
            let _ = builder.arg(&half_count);
            let _ = builder.arg(e_in.limbs());
            let _ = builder.arg(&e_in_arg);
            let _ = builder.arg(e_out.limbs());
            let _ = builder.arg(&num_x_in_bits);
            let _ = builder.arg(partials.limbs_mut());
            // SAFETY: thread `g < half` reads `packed` at cycle indices
            // `2 * g * slots + t` for `t < 2 * slots`, all below
            // `2 * half * slots == cycles`, and `tables` at
            // `p * slots * addresses + s * addresses + a` with `a` masked below
            // `addresses` and `p < polys`, inside `tables`'s
            // `polys * slots * addresses`. `e_in`/`e_out` are indexed as in the
            // dense arm, bounded by the same check. It writes only
            // `partials[lane * gridDim.x + blockIdx.x]` of `4 * blocks`. Shared
            // memory matches `shared_mem_bytes` and the block reduction sits
            // outside the `g < half` guard.
            let _ = unsafe { builder.launch(config) }?;
        }
        context.stream().synchronize()?;

        let totals = crate::cuda::common::dense_product::DeviceDenseProduct::reduce_lanes(
            context, partials, 4, blocks,
        )?;
        let host = totals.to_host()?;
        let mut evals = [F::from_u64(0); 4];
        for (slot, value) in evals.iter_mut().zip(host) {
            *slot =
                crate::cuda::common::device::fr_into(value).ok_or(CudaError::NotImplemented {
                    kernel: "CUDA kernels support only the BN254 scalar field",
                })?;
        }
        Ok(evals)
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{BindingOrder, EqPolynomial, Polynomial};
    use proptest::prelude::*;

    use super::{DevicePackedRa, COLLAPSE_AFTER_ROUNDS};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{arb_point, fr};

    fn packed_index(seed: u64, cycle: usize) -> u128 {
        let mix = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(cycle as u64 + 1);
        let lo = mix.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ (mix >> 29);
        let hi = mix.wrapping_mul(0x94D0_49BB_1331_11EB) ^ (mix >> 31);
        (u128::from(hi) << 64) | u128::from(lo)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(6))]
        #[test]
        fn device_packed_ra_matches_cpu_round_for_round(
            log_t in (COLLAPSE_AFTER_ROUNDS + 1)..7usize,
            chunk_bits in prop::sample::select(vec![4usize, 8]),
            polys in 1usize..5,
            seed in any::<u64>(),
            challenges in arb_point(8),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let cycles = 1usize << log_t;
            let addresses = 1usize << chunk_bits;

            let words: Vec<u64> = (0..cycles)
                .flat_map(|cycle| {
                    let index = packed_index(seed, cycle);
                    [index as u64, (index >> 64) as u64]
                })
                .collect();
            let point: Vec<Fr> = (0..polys * chunk_bits)
                .map(|i| fr(seed ^ (i as u64 * 37 + 11)))
                .collect();
            let seeds: Vec<Fr> = (0..polys)
                .map(|p| fr(seed ^ (p as u64 * 911 + 3)))
                .collect();

            let mut expected: Vec<Polynomial<Fr>> = (0..polys)
                .map(|p| {
                    let chunk_point = &point[p * chunk_bits..(p + 1) * chunk_bits];
                    let table = EqPolynomial::<Fr>::evals(chunk_point, Some(seeds[p]));
                    let shift = chunk_bits * (polys - 1 - p);
                    Polynomial::new(
                        (0..cycles)
                            .map(|cycle| {
                                let index = packed_index(seed, cycle);
                                let address =
                                    ((index >> shift) & (addresses as u128 - 1)) as usize;
                                table[address]
                            })
                            .collect(),
                    )
                })
                .collect();

            let packed = context.upload_u64_slice(&words).expect("upload packed index");
            let mut got = DevicePackedRa::new(context, packed, cycles, chunk_bits, &point, &seeds)
                .expect("device packed one-hot family");

            for round in 0..log_t {
                let coefficients = got.coefficients(context).expect("gather coefficients");
                for (p, table) in coefficients.iter().enumerate() {
                    prop_assert_eq!(
                        table.to_host().expect("download coefficients"),
                        expected[p].evals()[..got.len()].to_vec(),
                        "poly {} diverged at round {}", p, round
                    );
                }
                let challenge = challenges[round];
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
