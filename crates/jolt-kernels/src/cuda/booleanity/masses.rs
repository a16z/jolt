use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_field::{Field, Fr};

use crate::cuda::common::context::{CudaKernelContext, BLOCK};
use crate::cuda::common::device::{
    fr_into, fr_limbs, require_fr, require_fr_slice, DeviceFrVec, LIMBS,
};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::one_hot_fold::{FoldTuning, OneHotShards};
use crate::cuda::common::primitives::reduce_lanes;
use crate::cuda::common::split_eq::DeviceSplitEq;

const LANES: usize = 2;

pub struct DeviceBooleanityMasses {
    linear: DeviceFrVec,
    squared: DeviceFrVec,
    rho: DeviceFrVec,
    polys: usize,
    len: usize,
}

impl DeviceBooleanityMasses {
    pub fn new<F: Field>(
        context: &CudaKernelContext,
        shards: &OneHotShards,
        cycle_point: &[F],
        gamma: F,
    ) -> Result<Self, CudaError> {
        let columns = shards.whole()?;
        let polys = columns.polys();
        let addresses = columns.addresses();
        if polys == 0 || addresses < 2 {
            return Err(CudaError::InvariantViolation {
                reason: "a booleanity address phase needs at least one checked polynomial and one \
                         chunk bit",
            });
        }

        let linear = shards.fold(cycle_point, FoldTuning::default())?;
        if linear.len() != polys * addresses {
            return Err(CudaError::LengthMismatch {
                expected: polys * addresses,
                got: linear.len(),
            });
        }
        let squared = linear.try_clone()?;

        let square = gamma * gamma;
        let mut powers = Vec::with_capacity(polys);
        let mut power = F::one();
        for _ in 0..polys {
            powers.push(power);
            power *= square;
        }
        let rho = context.upload(require_fr_slice(&powers)?)?;

        Ok(Self {
            linear,
            squared,
            rho,
            polys,
            len: addresses,
        })
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub fn round_lanes<F: Field>(
        &self,
        context: &CudaKernelContext,
        eq: &DeviceSplitEq<F>,
    ) -> Result<(F, F), CudaError> {
        let half = self.len / 2;
        if half == 0 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: self.len,
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
        let mut partials = context.alloc(LANES * blocks as usize)?;
        let poly_count = CudaKernelContext::count_of(self.polys)?;
        let e_in_arg = CudaKernelContext::count_of(e_in_len)?;
        let num_x_in_bits = e_in_len.max(1).ilog2();
        let e_in = eq.e_in_current();
        let e_out = eq.e_out_current();

        let mut builder = context.stream().launch_builder(context.bap_message());
        let _ = builder.arg(self.linear.limbs());
        let _ = builder.arg(self.squared.limbs());
        let _ = builder.arg(self.rho.limbs());
        let _ = builder.arg(&poly_count);
        let _ = builder.arg(&half_count);
        let _ = builder.arg(e_in.limbs());
        let _ = builder.arg(&e_in_arg);
        let _ = builder.arg(e_out.limbs());
        let _ = builder.arg(&num_x_in_bits);
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: thread `y < half` reads `linear[p * 2 * half + 2y]`,
        // `squared[p * 2 * half + 2y]` and `squared[p * 2 * half + 2y + 1]` for every
        // `p < polys` — both buffers hold `polys * 2 * half` elements, because
        // `len == 2 * half` is their current row length — plus `rho[p]` of `polys`,
        // and `e_in[y & mask]` and `e_out[y >> num_x_in_bits]`, both bounded because
        // `e_out.len() * e_in.len() == half` is checked above. It writes only
        // `partials[lane * gridDim.x + blockIdx.x]` for `lane < 2`, of `2 * blocks`.
        // Shared memory is `BLOCK * LIMBS` u64s, matching `shared_mem_bytes`, and the
        // block reduction sits outside the `y < half` guard so every thread reaches
        // each `__syncthreads()`.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;

        let totals = reduce_lanes(
            context,
            partials,
            CudaKernelContext::count_of(LANES)?,
            blocks,
        )?;
        let host = totals.to_host()?;
        let unsupported = || CudaError::NotImplemented {
            kernel: "CUDA kernels support only the BN254 scalar field",
        };
        let at_zero = fr_into(host[0]).ok_or_else(unsupported)?;
        let leading = fr_into(host[1]).ok_or_else(unsupported)?;
        Ok((at_zero, leading))
    }

    pub fn bind<F: Field>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        if self.len < 2 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: self.len,
            });
        }
        let challenge = require_fr(challenge)?;
        let complement = Fr::from(1u64) - challenge;
        let low = fr_limbs(complement * complement);
        let high = fr_limbs(challenge * challenge);

        let half = self.polys * self.len / 2;
        let mut bound = context.alloc(half)?;
        let count = CudaKernelContext::count_of(half)?;
        let mut builder = context.stream().launch_builder(context.bap_bind_squared());
        let _ = builder.arg(self.squared.limbs());
        for limb in &low {
            let _ = builder.arg(limb);
        }
        for limb in &high {
            let _ = builder.arg(limb);
        }
        let _ = builder.arg(bound.limbs_mut());
        let _ = builder.arg(&count);
        // SAFETY: thread `i < half` reads `in[2i]` and `in[2i + 1]`, both inside
        // `in`'s `2 * half` elements, and writes only `out[i]` of `half`. The two
        // squared weights arrive as by-value limbs, so no device buffer backs them.
        // `len` is even, so no pair straddles a row boundary and this is a per-row
        // squared-weight bind whose result is contiguous with stride `len / 2`.
        // `out` is a fresh allocation distinct from `in`, so no thread reads a
        // partially written slot; threads with `i >= half` return first.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;

        self.squared = bound;
        self.linear = context.bind_rows(&self.linear, self.len, challenge)?;
        self.len /= 2;
        Ok(())
    }

    pub fn booleanity_defect<F: Field>(&self) -> Result<F, CudaError> {
        if self.len != 1 {
            return Err(CudaError::LengthMismatch {
                expected: 1,
                got: self.len,
            });
        }
        let linear = self.linear.to_host()?;
        let squared = self.squared.to_host()?;
        let rho = self.rho.to_host()?;
        if linear.len() != self.polys || squared.len() != self.polys || rho.len() != self.polys {
            return Err(CudaError::LengthMismatch {
                expected: self.polys,
                got: linear.len().min(squared.len()).min(rho.len()),
            });
        }
        let mut defect = Fr::from(0u64);
        for poly in 0..self.polys {
            defect += rho[poly] * (squared[poly] - linear[poly]);
        }
        fr_into(defect).ok_or(CudaError::NotImplemented {
            kernel: "CUDA kernels support only the BN254 scalar field",
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial};
    use proptest::prelude::*;

    use super::DeviceBooleanityMasses;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::one_hot_fold::{DeviceOneHotColumns, OneHotShards};
    use crate::cuda::common::one_hot_witness::{packed_columns, OneHotCycleWitness};
    use crate::cuda::common::pack::COLD;
    use crate::cuda::common::split_eq::DeviceSplitEq;
    use crate::cuda::common::testing::fr;

    fn mix(seed: u64, cycle: usize, salt: u64) -> u64 {
        let value = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(cycle as u64 + salt);
        value.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ (value >> 29)
    }

    struct Columns {
        lookup: Vec<u64>,
        pc: Vec<u32>,
        ram: Vec<u32>,
    }

    fn columns(seed: u64, cycles: usize, span: u32, cold_every: usize) -> Columns {
        let mut lookup = Vec::with_capacity(2 * cycles);
        let mut pc = Vec::with_capacity(cycles);
        let mut ram = Vec::with_capacity(cycles);
        for cycle in 0..cycles {
            lookup.push(mix(seed, cycle, 1));
            lookup.push(mix(seed, cycle, 2));
            pc.push(if cycle % cold_every == 1 {
                COLD
            } else {
                (mix(seed, cycle, 3) % u64::from(span)) as u32
            });
            ram.push(if cycle.is_multiple_of(cold_every) {
                COLD
            } else {
                (mix(seed, cycle, 4) % u64::from(span)) as u32
            });
        }
        Columns { lookup, pc, ram }
    }

    fn expected_masses(
        packed: &Columns,
        cycles: usize,
        chunk_bits: usize,
        families: [usize; 3],
        cycle_point: &[Fr],
    ) -> Vec<Vec<Fr>> {
        let eq = EqPolynomial::new(cycle_point.to_vec()).evaluations();
        let addresses = 1usize << chunk_bits;
        let mut masses = Vec::new();
        for (family, count) in families.into_iter().enumerate() {
            for local in 0..count {
                let shift = chunk_bits * (count - 1 - local);
                let mut table = vec![Fr::from_u64(0); addresses];
                for (cycle, weight) in eq.iter().enumerate().take(cycles) {
                    let index = match family {
                        0 => {
                            let wide = u128::from(packed.lookup[2 * cycle])
                                | (u128::from(packed.lookup[2 * cycle + 1]) << 64);
                            Some(((wide >> shift) as usize) & (addresses - 1))
                        }
                        1 => (packed.pc[cycle] != COLD)
                            .then(|| ((packed.pc[cycle] >> shift) as usize) & (addresses - 1)),
                        _ => (packed.ram[cycle] != COLD)
                            .then(|| ((packed.ram[cycle] >> shift) as usize) & (addresses - 1)),
                    };
                    if let Some(index) = index {
                        table[index] += *weight;
                    }
                }
                masses.push(table);
            }
        }
        masses
    }

    fn expanding_table_lanes(
        masses: &[Vec<Fr>],
        rho: &[Fr],
        bound: &[Fr],
        eq: &GruenSplitEqPolynomial<Fr>,
        addresses: usize,
    ) -> (Fr, Fr) {
        let m = bound.len() + 1;
        let mut table = vec![Fr::from_u64(1)];
        for &challenge in bound {
            let mut next = vec![Fr::from_u64(0); table.len() * 2];
            for (index, value) in table.iter().enumerate() {
                next[index] = *value * (Fr::from_u64(1) - challenge);
                next[index + table.len()] = *value * challenge;
            }
            table = next;
        }
        let e_in = eq.e_in_current();
        let e_out = eq.e_out_current();
        let in_bits = e_in.len().trailing_zeros() as usize;
        let mut at_zero = Fr::from_u64(0);
        let mut leading = Fr::from_u64(0);
        for k_prime in 0..(addresses >> m) {
            let mut inner_zero = Fr::from_u64(0);
            let mut inner_lead = Fr::from_u64(0);
            for (poly, mass) in masses.iter().enumerate() {
                let mut sum_zero = Fr::from_u64(0);
                let mut sum_lead = Fr::from_u64(0);
                for k in 0..(1usize << m) {
                    let bit = k >> (m - 1);
                    let factor = table[k & ((1usize << (m - 1)) - 1)];
                    let scaled = mass[(k_prime << m) | k] * factor;
                    let infinity = scaled * factor;
                    sum_lead += infinity;
                    if bit == 0 {
                        sum_zero += infinity - scaled;
                    }
                }
                inner_zero += rho[poly] * sum_zero;
                inner_lead += rho[poly] * sum_lead;
            }
            let weight = e_out[k_prime >> in_bits] * e_in[k_prime & (e_in.len() - 1)];
            at_zero += weight * inner_zero;
            leading += weight * inner_lead;
        }
        (at_zero, leading)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(6))]
        #[test]
        fn booleanity_masses_match_the_expanding_table_form_round_for_round(
            log_t in 4usize..9,
            chunk_bits in prop::sample::select(vec![2usize, 4, 6]),
            bytecode_polys in 1usize..3,
            ram_polys in 1usize..3,
            cold_every in 2usize..5,
            seed in any::<u64>(),
            gamma in any::<u64>().prop_map(fr),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            if chunk_bits * bytecode_polys > 32 || chunk_bits * ram_polys > 32 {
                return Ok(());
            }
            let cycles = 1usize << log_t;
            let addresses = 1usize << chunk_bits;
            let families = [64 / chunk_bits, bytecode_polys, ram_polys];
            let span = 1u32 << (chunk_bits * bytecode_polys.max(ram_polys) - 1).min(31);
            let packed = columns(seed, cycles, span, cold_every);
            let cycle_point: Vec<Fr> =
                (0..log_t).map(|i| fr(seed ^ (i as u64 * 17 + 3))).collect();
            let address_point: Vec<Fr> =
                (0..chunk_bits).map(|i| fr(seed ^ (i as u64 * 53 + 19))).collect();
            let challenges: Vec<Fr> =
                (0..chunk_bits).map(|i| fr(seed ^ (i as u64 * 71 + 29))).collect();

            let masses =
                expected_masses(&packed, cycles, chunk_bits, families, &cycle_point);
            let polys = masses.len();
            let mut rho = Vec::with_capacity(polys);
            let mut power = Fr::from_u64(1);
            let square = gamma * gamma;
            for _ in 0..polys {
                rho.push(power);
                power *= square;
            }

            let uploaded = DeviceOneHotColumns::new(
                context,
                &packed.lookup,
                &packed.pc,
                &packed.ram,
                families,
                chunk_bits,
                cycles,
            )
            .expect("upload one-hot columns");
            let mut got = DeviceBooleanityMasses::new(context, &OneHotShards::single(uploaded), &cycle_point, gamma)
                .expect("device booleanity masses");
            let mut eq = GruenSplitEqPolynomial::<Fr>::new(&address_point, BindingOrder::LowToHigh);
            let mut device_eq =
                DeviceSplitEq::<Fr>::new(context, &address_point, BindingOrder::LowToHigh)
                    .expect("device split-eq");

            for round in 0..chunk_bits {
                let expected = expanding_table_lanes(
                    &masses,
                    &rho,
                    &challenges[..round],
                    &eq,
                    addresses,
                );
                let (at_zero, leading) = got
                    .round_lanes::<Fr>(context, &device_eq)
                    .expect("device round lanes");
                prop_assert_eq!(
                    (at_zero, leading),
                    expected,
                    "round {} lanes diverged", round
                );
                got.bind(context, challenges[round]).expect("device bind");
                eq.bind(challenges[round]);
                device_eq.bind(challenges[round]);
            }

            prop_assert_eq!(got.len(), 1, "the masses never bound down to one address");
        }
    }

    #[test]
    fn booleanity_masses_reject_a_wrong_length_split_eq() {
        let Some(context) = shared_context() else {
            return;
        };
        let log_t = 5;
        let cycles = 1usize << log_t;
        let rows = vec![OneHotCycleWitness::default(); cycles];
        let packed = packed_columns(&rows).expect("pack the columns");
        let uploaded = DeviceOneHotColumns::new(
            context,
            &packed.lookup,
            &packed.pc,
            &packed.ram,
            [1, 0, 0],
            4,
            cycles,
        )
        .expect("upload one-hot columns");
        let cycle_point: Vec<Fr> = (0..log_t).map(|i| fr(i as u64 * 13 + 5)).collect();
        let masses = DeviceBooleanityMasses::new(
            context,
            &OneHotShards::single(uploaded),
            &cycle_point,
            fr(9),
        )
        .expect("device booleanity masses");
        let short: Vec<Fr> = (0..3).map(|i| fr(i as u64 * 3 + 1)).collect();
        let eq = DeviceSplitEq::<Fr>::new(context, &short, BindingOrder::LowToHigh)
            .expect("device split-eq");
        assert!(
            masses.round_lanes::<Fr>(context, &eq).is_err(),
            "a split-eq over the wrong variable count must not silently weight the wrong \
             addresses",
        );
    }
}
