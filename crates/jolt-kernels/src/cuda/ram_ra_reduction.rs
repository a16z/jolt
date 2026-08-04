use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_field::{Field, Fr};
use jolt_poly::BindingOrder;

use super::context::{CudaKernelContext, BLOCK};
use super::device::{require_fr, require_fr_slice, DeviceFrVec, LIMBS};
use super::error::CudaError;
use super::ra_poly::COLD;

pub const TERMS: usize = 3;

const DEGREE: usize = 2;

pub struct CyclePoints<'a, F> {
    pub raf: &'a [F],
    pub read_write: &'a [F],
    pub val_check: &'a [F],
}

impl<F: Field> CyclePoints<'_, F> {
    fn all(&self) -> [&[F]; TERMS] {
        [self.raf, self.read_write, self.val_check]
    }
}

enum Phase {
    Prefix {
        p: Vec<DeviceFrVec>,
        q: Vec<DeviceFrVec>,
    },
    Suffix {
        h_prime: DeviceFrVec,
        eq_hi: Vec<DeviceFrVec>,
    },
}

pub struct DeviceRamRaReduction {
    h: DeviceFrVec,
    suffix_points: Vec<Vec<Fr>>,
    prefix_points: Vec<Vec<Fr>>,
    coefficients: [Fr; TERMS],
    phase: Phase,
    log_t: usize,
    prefix_vars: usize,
    rounds_bound: usize,
    challenges: Vec<Fr>,
}

impl DeviceRamRaReduction {
    pub fn new<F: Field>(
        context: &CudaKernelContext,
        hot_indices: &[Option<usize>],
        eq_address: &[F],
        cycle_points: &CyclePoints<'_, F>,
        gamma: F,
        log_t: usize,
    ) -> Result<Self, CudaError> {
        let cycles = 1usize << log_t;
        if hot_indices.len() != cycles {
            return Err(CudaError::LengthMismatch {
                expected: cycles,
                got: hot_indices.len(),
            });
        }
        let prefix_vars = log_t / 2;
        let suffix_vars = log_t - prefix_vars;
        for point in cycle_points.all() {
            if point.len() != log_t {
                return Err(CudaError::LengthMismatch {
                    expected: log_t,
                    got: point.len(),
                });
            }
        }

        let h = Self::gather_h(context, hot_indices, eq_address, cycles)?;

        let mut suffix_points = Vec::with_capacity(TERMS);
        let mut prefix_points = Vec::with_capacity(TERMS);
        for point in cycle_points.all() {
            let point = require_fr_slice(point)?;
            suffix_points.push(point[..suffix_vars].to_vec());
            prefix_points.push(point[suffix_vars..].to_vec());
        }

        let gamma = require_fr(gamma)?;
        let coefficients = [Fr::from(1u64), gamma, gamma * gamma];

        let mut p = Vec::with_capacity(TERMS);
        let mut q = Vec::with_capacity(TERMS);
        for index in 0..TERMS {
            p.push(context.eq_evals(&prefix_points[index])?);
            let eq_hi = context.eq_evals(&suffix_points[index])?;
            q.push(Self::fold_suffix(context, &h, &eq_hi, prefix_vars, log_t)?);
        }

        Ok(Self {
            h,
            suffix_points,
            prefix_points,
            coefficients,
            phase: Phase::Prefix { p, q },
            log_t,
            prefix_vars,
            rounds_bound: 0,
            challenges: Vec::with_capacity(log_t),
        })
    }

    fn gather_h<F: Field>(
        context: &CudaKernelContext,
        hot_indices: &[Option<usize>],
        eq_address: &[F],
        cycles: usize,
    ) -> Result<DeviceFrVec, CudaError> {
        let addresses = eq_address.len();
        let mut raw = Vec::with_capacity(cycles);
        for hot in hot_indices {
            let encoded = match *hot {
                None => COLD,
                Some(address) if address < addresses => {
                    u32::try_from(address).map_err(|_| CudaError::LengthMismatch {
                        expected: addresses,
                        got: address,
                    })?
                }
                Some(address) => {
                    return Err(CudaError::LengthMismatch {
                        expected: addresses,
                        got: address,
                    })
                }
            };
            raw.push(encoded);
        }
        let indices = context.upload_u32_slice(&raw)?;
        let eq_address = context.upload(require_fr_slice(eq_address)?)?;
        let mut h = context.alloc(cycles)?;
        let count = CudaKernelContext::count_of(cycles)?;
        let mut builder = context.stream().launch_builder(context.ram_ra_gather_h());
        let _ = builder.arg(&indices);
        let _ = builder.arg(eq_address.limbs());
        let _ = builder.arg(h.limbs_mut());
        let _ = builder.arg(&count);
        // SAFETY: thread `c < cycles` reads `indices[c]` and, unless the entry is
        // `COLD`, `eq_address[indices[c]]` — bounded by `addresses` by the check
        // above — and writes only `h[c]` of `cycles` elements. `h` is a distinct
        // allocation.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
        context.stream().synchronize()?;
        Ok(h)
    }

    fn fold_suffix(
        context: &CudaKernelContext,
        h: &DeviceFrVec,
        eq_hi: &DeviceFrVec,
        prefix_vars: usize,
        log_t: usize,
    ) -> Result<DeviceFrVec, CudaError> {
        let prefix_size = 1usize << prefix_vars;
        let suffix_size = 1usize << (log_t - prefix_vars);
        let mut q = context.alloc(prefix_size)?;
        let prefix_count = CudaKernelContext::count_of(prefix_size)?;
        let suffix_count = CudaKernelContext::count_of(suffix_size)?;
        let mut builder = context
            .stream()
            .launch_builder(context.ram_ra_fold_suffix());
        let _ = builder.arg(h.limbs());
        let _ = builder.arg(eq_hi.limbs());
        let _ = builder.arg(q.limbs_mut());
        let _ = builder.arg(&prefix_count);
        let _ = builder.arg(&suffix_count);
        // SAFETY: thread `c_lo < prefix_size` reads `h[c_hi * prefix_size + c_lo]`
        // for `c_hi < suffix_size` — all below `prefix_size * suffix_size == 2^log_t`,
        // `h`'s element count — plus `eq_hi[c_hi]` of `suffix_size`, and writes only
        // `q[c_lo]` of `prefix_size`. `q` is a distinct allocation.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(prefix_count)) }?;
        context.stream().synchronize()?;
        Ok(q)
    }

    fn fold_prefix(
        context: &CudaKernelContext,
        h: &DeviceFrVec,
        eq_prefix: &DeviceFrVec,
        prefix_vars: usize,
        log_t: usize,
    ) -> Result<DeviceFrVec, CudaError> {
        let prefix_size = 1usize << prefix_vars;
        let suffix_size = 1usize << (log_t - prefix_vars);
        let mut h_prime = context.alloc(suffix_size)?;
        let prefix_count = CudaKernelContext::count_of(prefix_size)?;
        let suffix_count = CudaKernelContext::count_of(suffix_size)?;
        let mut builder = context
            .stream()
            .launch_builder(context.ram_ra_fold_prefix());
        let _ = builder.arg(h.limbs());
        let _ = builder.arg(eq_prefix.limbs());
        let _ = builder.arg(h_prime.limbs_mut());
        let _ = builder.arg(&prefix_count);
        let _ = builder.arg(&suffix_count);
        // SAFETY: thread `c_hi < suffix_size` reads `h[c_hi * prefix_size + c_lo]`
        // for `c_lo < prefix_size` — all below `h`'s `2^log_t` elements — plus
        // `eq_prefix[c_lo]` of `prefix_size`, and writes only `h_prime[c_hi]` of
        // `suffix_size`. `h_prime` is a distinct allocation.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(suffix_count)) }?;
        context.stream().synchronize()?;
        Ok(h_prime)
    }

    pub const fn rounds_bound(&self) -> usize {
        self.rounds_bound
    }

    pub const fn in_phase1(&self) -> bool {
        matches!(self.phase, Phase::Prefix { .. })
    }

    #[cfg(test)]
    pub fn q_arrays(&self, context: &CudaKernelContext) -> Result<Vec<DeviceFrVec>, CudaError> {
        let _ = context;
        match &self.phase {
            Phase::Prefix { q, .. } => q.iter().map(DeviceFrVec::try_clone).collect(),
            Phase::Suffix { .. } => Err(CudaError::InvariantViolation {
                reason: "the Q arrays exist only during the prefix phase",
            }),
        }
    }

    #[cfg(test)]
    pub fn h_prime(&self, context: &CudaKernelContext) -> Result<DeviceFrVec, CudaError> {
        let _ = context;
        match &self.phase {
            Phase::Suffix { h_prime, .. } => h_prime.try_clone(),
            Phase::Prefix { .. } => Err(CudaError::InvariantViolation {
                reason: "H' is built at the transition out of the prefix phase",
            }),
        }
    }

    fn enter_suffix_phase(&mut self, context: &CudaKernelContext) -> Result<(), CudaError> {
        let reduced: Vec<Fr> = self.challenges.iter().rev().copied().collect();
        let eq_prefix = context.eq_evals(&reduced)?;
        let h_prime =
            Self::fold_prefix(context, &self.h, &eq_prefix, self.prefix_vars, self.log_t)?;

        let mut eq_hi = Vec::with_capacity(TERMS);
        for index in 0..TERMS {
            let scale = eq_mle(&self.prefix_points[index], &reduced);
            let table = context.eq_evals(&self.suffix_points[index])?;
            eq_hi.push(context.mul_scalar(&table, self.coefficients[index] * scale)?);
        }

        self.h = context.alloc(0)?;
        self.phase = Phase::Suffix { h_prime, eq_hi };
        Ok(())
    }

    pub fn round_evals<F: Field>(&self, context: &CudaKernelContext) -> Result<Vec<F>, CudaError> {
        match &self.phase {
            Phase::Prefix { p, q } => {
                let half = p
                    .first()
                    .ok_or(CudaError::InvariantViolation {
                        reason: "the prefix phase needs at least one P table",
                    })?
                    .len()
                    / 2;
                self.phase1_round_evals(context, p, q, half)
            }
            Phase::Suffix { h_prime, eq_hi } => {
                Self::phase2_round_evals(context, h_prime, eq_hi, h_prime.len() / 2)
            }
        }
    }

    fn phase1_round_evals<F: Field>(
        &self,
        context: &CudaKernelContext,
        p: &[DeviceFrVec],
        q: &[DeviceFrVec],
        half: usize,
    ) -> Result<Vec<F>, CudaError> {
        let lanes = CudaKernelContext::count_of(DEGREE + 1)?;
        let half_count = CudaKernelContext::count_of(half)?;
        let terms = CudaKernelContext::count_of(TERMS)?;
        let p_handles: Vec<&DeviceFrVec> = p.iter().collect();
        let q_handles: Vec<&DeviceFrVec> = q.iter().collect();
        let p_pointers = context.device_pointers(&p_handles)?;
        let q_pointers = context.device_pointers(&q_handles)?;
        let coefficients = context.upload(&self.coefficients)?;
        let blocks = half_count.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc(lanes as usize * blocks as usize)?;
        let mut builder = context
            .stream()
            .launch_builder(context.ram_ra_phase1_round());
        let _ = builder.arg(&p_pointers);
        let _ = builder.arg(&q_pointers);
        let _ = builder.arg(coefficients.limbs());
        let _ = builder.arg(&terms);
        let _ = builder.arg(&half_count);
        let _ = builder.arg(&lanes);
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: for each lane `c`, thread `y < half` reads the pairs
        // (`p[t][2y]`, `p[t][2y+1]`) and (`q[t][2y]`, `q[t][2y+1]`) for each of
        // `terms` tables of `2 * half` elements, plus `coefficients[t]` of
        // `TERMS`. Writes: `partials[c * gridDim.x + blockIdx.x]`, one slot per
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
        Self::reduce_lanes(context, partials, lanes, blocks)
    }

    fn phase2_round_evals<F: Field>(
        context: &CudaKernelContext,
        h_prime: &DeviceFrVec,
        eq_hi: &[DeviceFrVec],
        half: usize,
    ) -> Result<Vec<F>, CudaError> {
        let mut weight = context.alloc(h_prime.len())?;
        for table in eq_hi {
            weight = context.add(&weight, table)?;
        }
        let lanes = CudaKernelContext::count_of(DEGREE + 1)?;
        let half_count = CudaKernelContext::count_of(half)?;
        let handles = [&weight, h_prime];
        let table_count = CudaKernelContext::count_of(handles.len())?;
        let pointers = context.device_pointers(&handles)?;
        let empty = context.alloc(1)?;
        let blocks = half_count.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc(lanes as usize * blocks as usize)?;
        let zero = 0u32;
        let mut builder = context
            .stream()
            .launch_builder(context.dense_product_round());
        let _ = builder.arg(&pointers);
        let _ = builder.arg(&table_count);
        let _ = builder.arg(&half_count);
        let _ = builder.arg(&lanes);
        let _ = builder.arg(partials.limbs_mut());
        let _ = builder.arg(empty.limbs());
        let _ = builder.arg(empty.limbs());
        let _ = builder.arg(empty.limbs());
        let _ = builder.arg(&zero);
        let _ = builder.arg(&zero);
        let _ = builder.arg(&zero);
        // SAFETY: as `dense_product::round_evals` — two tables of `2 * half`
        // elements read at `2y`/`2y+1`, one `partials` slot per (lane, block),
        // shared memory matching `shared_mem_bytes`. `has_lt` is 0, so the LT
        // pointers are never dereferenced and the one-element `empty` placeholder
        // is sound.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;
        context.stream().synchronize()?;
        Self::reduce_lanes(context, partials, lanes, blocks)
    }

    fn reduce_lanes<F: Field>(
        context: &CudaKernelContext,
        mut partials: DeviceFrVec,
        lanes: u32,
        mut width: u32,
    ) -> Result<Vec<F>, CudaError> {
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
        partials
            .to_host()?
            .into_iter()
            .map(|value| {
                super::device::fr_into(value).ok_or(CudaError::NotImplemented {
                    kernel: "CUDA kernels support only the BN254 scalar field",
                })
            })
            .collect()
    }

    pub fn bind<F: Field>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        let challenge = require_fr(challenge)?;
        self.challenges.push(challenge);
        self.rounds_bound += 1;
        match &mut self.phase {
            Phase::Prefix { p, q } => {
                for table in p.iter_mut().chain(q.iter_mut()) {
                    *table = context.bind(table, challenge, BindingOrder::LowToHigh)?;
                }
            }
            Phase::Suffix { h_prime, eq_hi } => {
                *h_prime = context.bind(h_prime, challenge, BindingOrder::LowToHigh)?;
                for table in eq_hi.iter_mut() {
                    *table = context.bind(table, challenge, BindingOrder::LowToHigh)?;
                }
            }
        }
        if self.in_phase1() && self.rounds_bound == self.prefix_vars {
            self.enter_suffix_phase(context)?;
        }
        Ok(())
    }

    pub fn final_claim(&self, context: &CudaKernelContext) -> Result<Fr, CudaError> {
        let _ = context;
        match &self.phase {
            Phase::Suffix { h_prime, .. } => {
                if h_prime.len() != 1 {
                    return Err(CudaError::LengthMismatch {
                        expected: 1,
                        got: h_prime.len(),
                    });
                }
                h_prime.first()
            }
            Phase::Prefix { .. } => Err(CudaError::InvariantViolation {
                reason: "the reduced RA claim is only available after the suffix phase",
            }),
        }
    }
}

fn eq_mle(left: &[Fr], right: &[Fr]) -> Fr {
    let one = Fr::from(1u64);
    left.iter()
        .zip(right)
        .fold(one, |acc, (&x, &y)| acc * (x * y + (one - x) * (one - y)))
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::EqPolynomial;
    use proptest::prelude::*;

    use super::super::context::shared_context;
    use super::super::testing::arb_point;
    use super::{CyclePoints, DeviceRamRaReduction};

    const LOG_T: usize = 6;
    const LOG_K: usize = 3;

    fn hot_indices(seed: u64) -> Vec<Option<usize>> {
        let addresses = 1usize << LOG_K;
        (0..1usize << LOG_T)
            .map(|c| {
                if (c as u64 + seed) % 7 == 2 {
                    None
                } else {
                    Some(((c as u64 * 5 + seed) as usize) % addresses)
                }
            })
            .collect()
    }

    fn host_h(indices: &[Option<usize>], eq_address: &[Fr]) -> Vec<Fr> {
        indices
            .iter()
            .map(|hot| hot.map_or(Fr::from_u64(0), |address| eq_address[address]))
            .collect()
    }

    fn host_q(h: &[Fr], eq_hi: &[Fr], prefix_vars: usize) -> Vec<Fr> {
        let prefix_size = 1usize << prefix_vars;
        let suffix_size = h.len() / prefix_size;
        (0..prefix_size)
            .map(|c_lo| {
                (0..suffix_size)
                    .map(|c_hi| h[c_hi * prefix_size + c_lo] * eq_hi[c_hi])
                    .sum()
            })
            .collect()
    }

    fn host_h_prime(h: &[Fr], eq_prefix: &[Fr], prefix_vars: usize) -> Vec<Fr> {
        let prefix_size = 1usize << prefix_vars;
        let suffix_size = h.len() / prefix_size;
        (0..suffix_size)
            .map(|c_hi| {
                (0..prefix_size)
                    .map(|c_lo| h[c_hi * prefix_size + c_lo] * eq_prefix[c_lo])
                    .sum()
            })
            .collect()
    }

    fn eq_table(point: &[Fr]) -> Vec<Fr> {
        EqPolynomial::new(point.to_vec()).evaluations()
    }

    proptest! {
        #[test]
        fn q_arrays_match_the_suffix_weighted_sums(
            seed in any::<u64>(),
            address in arb_point(LOG_K),
            raf in arb_point(LOG_T),
            read_write in arb_point(LOG_T),
            val_check in arb_point(LOG_T),
            gamma in arb_point(1),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let indices = hot_indices(seed);
            let eq_address = eq_table(&address);
            let points = CyclePoints {
                raf: &raf,
                read_write: &read_write,
                val_check: &val_check,
            };
            let state = DeviceRamRaReduction::new(
                context, &indices, &eq_address, &points, gamma[0], LOG_T,
            ).expect("device ram ra reduction");

            let prefix_vars = LOG_T / 2;
            let suffix_vars = LOG_T - prefix_vars;
            let h = host_h(&indices, &eq_address);
            let expected: Vec<Vec<Fr>> = [&raf, &read_write, &val_check]
                .iter()
                .map(|point| host_q(&h, &eq_table(&point[..suffix_vars]), prefix_vars))
                .collect();

            let got = state.q_arrays(context).expect("device q arrays");
            prop_assert_eq!(got.len(), 3);
            for (index, (device, host)) in got.iter().zip(&expected).enumerate() {
                prop_assert_eq!(
                    &device.to_host().expect("download"),
                    host,
                    "Q array {} diverged",
                    index
                );
            }
        }

        #[test]
        fn h_prime_matches_the_prefix_weighted_sum(
            seed in any::<u64>(),
            address in arb_point(LOG_K),
            raf in arb_point(LOG_T),
            read_write in arb_point(LOG_T),
            val_check in arb_point(LOG_T),
            gamma in arb_point(1),
            challenges in arb_point(LOG_T / 2),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let indices = hot_indices(seed);
            let eq_address = eq_table(&address);
            let points = CyclePoints {
                raf: &raf,
                read_write: &read_write,
                val_check: &val_check,
            };
            let mut state = DeviceRamRaReduction::new(
                context, &indices, &eq_address, &points, gamma[0], LOG_T,
            ).expect("device ram ra reduction");

            for &challenge in &challenges {
                state.bind(context, challenge).expect("device bind");
            }
            prop_assert!(!state.in_phase1());
            prop_assert_eq!(state.rounds_bound(), LOG_T / 2);

            let reversed: Vec<Fr> = challenges.iter().rev().copied().collect();
            let h = host_h(&indices, &eq_address);
            let expected = host_h_prime(&h, &eq_table(&reversed), LOG_T / 2);

            let got = state.h_prime(context).expect("device h prime").to_host().expect("download");
            prop_assert_eq!(got, expected);
        }
    }
}
