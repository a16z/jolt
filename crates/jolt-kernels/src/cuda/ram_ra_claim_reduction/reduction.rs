use std::sync::Arc;

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use jolt_field::{Field, Fr};

use crate::cuda::common::context::{context_for, CudaKernelContext, BLOCK};
use crate::cuda::common::device::{require_fr, require_fr_slice, DeviceFrVec, LIMBS};
use crate::cuda::common::devices::{fan_out, DeviceTask};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::primitives::fold_lanes_by_halving;

pub const TERMS: usize = 3;

const DEGREE: usize = 2;

type SuffixFold = (DeviceFrVec, Vec<Vec<Fr>>);

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
    h: Vec<DeviceFrVec>,
    suffix_points: Vec<Vec<Fr>>,
    prefix_points: Vec<Vec<Fr>>,
    coefficients: [Fr; TERMS],
    phase: Phase,
    log_t: usize,
    prefix_vars: usize,
    tail_rounds: usize,
    rounds_bound: usize,
    challenges: Vec<Fr>,
}

impl DeviceRamRaReduction {
    pub fn new<F: Field>(
        context: &CudaKernelContext,
        words: &CudaSlice<u32>,
        eq_address: &DeviceFrVec,
        cycle_points: &CyclePoints<'_, F>,
        gamma: F,
        log_t: usize,
    ) -> Result<Self, CudaError> {
        let cycles = 1usize << log_t;
        if words.len() != cycles {
            return Err(CudaError::LengthMismatch {
                expected: cycles,
                got: words.len(),
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

        let h = Self::gather_h(context, words, eq_address, cycles)?;

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
            h: vec![h],
            suffix_points,
            prefix_points,
            coefficients,
            phase: Phase::Prefix { p, q },
            log_t,
            prefix_vars,
            tail_rounds: 0,
            rounds_bound: 0,
            challenges: Vec::with_capacity(log_t),
        })
    }

    pub fn new_windowed<F: Field>(
        context: &CudaKernelContext,
        windows: &[(usize, Arc<CudaSlice<u32>>)],
        r_address: &[F],
        cycle_points: &CyclePoints<'_, F>,
        gamma: F,
        log_t: usize,
    ) -> Result<Self, CudaError> {
        let count = windows.len();
        let address = require_fr_slice(r_address)?;
        if count == 1 {
            let (_, words) = windows.first().ok_or(CudaError::InvariantViolation {
                reason: "a single-window RAM RA reduction lost its packed column",
            })?;
            let eq_address = context.eq_evals(address)?;
            return Self::new(context, words, &eq_address, cycle_points, gamma, log_t);
        }
        if count == 0 || !count.is_power_of_two() {
            return Err(CudaError::InvariantViolation {
                reason: "a windowed RAM RA reduction needs a power-of-two window count",
            });
        }
        let tail_rounds = count.trailing_zeros() as usize;
        let prefix_vars = log_t / 2;
        let suffix_vars = log_t - prefix_vars;
        if tail_rounds > suffix_vars {
            return Err(CudaError::InvariantViolation {
                reason: "a windowed RAM RA reduction cannot split more windows than the suffix \
                         phase has cycle variables",
            });
        }
        for point in cycle_points.all() {
            if point.len() != log_t {
                return Err(CudaError::LengthMismatch {
                    expected: log_t,
                    got: point.len(),
                });
            }
        }
        let window_log_t = log_t - tail_rounds;
        let window_cycles = 1usize << window_log_t;
        for (_, words) in windows {
            if words.len() < window_cycles {
                return Err(CudaError::LengthMismatch {
                    expected: window_cycles,
                    got: words.len(),
                });
            }
        }

        let mut suffix_points = Vec::with_capacity(TERMS);
        let mut prefix_points = Vec::with_capacity(TERMS);
        for point in cycle_points.all() {
            let point = require_fr_slice(point)?;
            suffix_points.push(point[..suffix_vars].to_vec());
            prefix_points.push(point[suffix_vars..].to_vec());
        }
        let gamma = require_fr(gamma)?;
        let coefficients = [Fr::from(1u64), gamma, gamma * gamma];

        let suffix = &suffix_points;
        let tasks: Vec<DeviceTask<'_, SuffixFold, CudaError>> = windows
            .iter()
            .enumerate()
            .map(|(shard, (ordinal, words))| {
                let task: DeviceTask<'_, SuffixFold, CudaError> = Box::new(move || {
                    let device = context_for(*ordinal).ok_or(absent())?;
                    let eq_address = device.eq_evals(address)?;
                    let h = Self::gather_h(device, words, &eq_address, window_cycles)?;
                    let mut parts = Vec::with_capacity(TERMS);
                    for point in suffix {
                        let eq_hi = device.eq_evals_shard(point, shard, count)?;
                        parts.push(
                            Self::fold_suffix(device, &h, &eq_hi, prefix_vars, window_log_t)?
                                .to_host()?,
                        );
                    }
                    Ok((h, parts))
                });
                task
            })
            .collect();

        let mut h = Vec::with_capacity(count);
        let mut sums: Option<Vec<Vec<Fr>>> = None;
        for (window, parts) in fan_out(tasks)? {
            match &mut sums {
                None => sums = Some(parts),
                Some(totals) => {
                    if totals.len() != parts.len() {
                        return Err(CudaError::LengthMismatch {
                            expected: totals.len(),
                            got: parts.len(),
                        });
                    }
                    for (total, part) in totals.iter_mut().zip(&parts) {
                        if total.len() != part.len() {
                            return Err(CudaError::LengthMismatch {
                                expected: total.len(),
                                got: part.len(),
                            });
                        }
                        for (slot, value) in total.iter_mut().zip(part) {
                            *slot += *value;
                        }
                    }
                }
            }
            h.push(window);
        }
        let sums = sums.ok_or(CudaError::InvariantViolation {
            reason: "a windowed RAM RA reduction produced no suffix folds",
        })?;

        let mut p = Vec::with_capacity(TERMS);
        let mut q = Vec::with_capacity(TERMS);
        for (point, sum) in prefix_points.iter().zip(&sums) {
            p.push(context.eq_evals(point)?);
            q.push(context.upload(sum)?);
        }
        if q.len() != TERMS {
            return Err(CudaError::LengthMismatch {
                expected: TERMS,
                got: q.len(),
            });
        }

        Ok(Self {
            h,
            suffix_points,
            prefix_points,
            coefficients,
            phase: Phase::Prefix { p, q },
            log_t,
            prefix_vars,
            tail_rounds,
            rounds_bound: 0,
            challenges: Vec::with_capacity(log_t),
        })
    }

    fn gather_h(
        context: &CudaKernelContext,
        words: &CudaSlice<u32>,
        eq_address: &DeviceFrVec,
        cycles: usize,
    ) -> Result<DeviceFrVec, CudaError> {
        let indices = words;
        let mut h = context.alloc(cycles)?;
        let count = CudaKernelContext::count_of(cycles)?;
        let mut builder = context.stream().launch_builder(context.ram_ra_gather_h());
        let _ = builder.arg(indices);
        let _ = builder.arg(eq_address.limbs());
        let _ = builder.arg(h.limbs_mut());
        let _ = builder.arg(&count);
        // SAFETY: thread `c < cycles` reads `indices[c]` and, unless the entry is
        // `COLD`, `eq_address[indices[c]]` — below `eq_address`'s length because
        // every caller encodes through `common::pack::encode_address` with that
        // length as its bound — and writes only `h[c]` of `cycles` elements. `h` is
        // a distinct allocation.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
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
        let h_prime = self.fold_h_prime(context, &reduced)?;

        let mut eq_hi = Vec::with_capacity(TERMS);
        for index in 0..TERMS {
            let scale = eq_mle(&self.prefix_points[index], &reduced);
            let table = context.eq_evals(&self.suffix_points[index])?;
            eq_hi.push(context.mul_scalar(&table, self.coefficients[index] * scale)?);
        }

        self.h = Vec::new();
        self.phase = Phase::Suffix { h_prime, eq_hi };
        Ok(())
    }

    fn fold_h_prime(
        &self,
        context: &CudaKernelContext,
        reduced: &[Fr],
    ) -> Result<DeviceFrVec, CudaError> {
        match self.h.as_slice() {
            [h] => {
                let eq_prefix = context.eq_evals(reduced)?;
                Self::fold_prefix(context, h, &eq_prefix, self.prefix_vars, self.log_t)
            }
            windows => {
                let window_log_t = self.log_t - self.tail_rounds;
                let prefix_vars = self.prefix_vars;
                let tasks: Vec<DeviceTask<'_, Vec<Fr>, CudaError>> = windows
                    .iter()
                    .map(|h| {
                        let task: DeviceTask<'_, Vec<Fr>, CudaError> = Box::new(move || {
                            let device = context_for(h.ordinal()).ok_or(absent())?;
                            let eq_prefix = device.eq_evals(reduced)?;
                            Self::fold_prefix(device, h, &eq_prefix, prefix_vars, window_log_t)?
                                .to_host()
                        });
                        task
                    })
                    .collect();
                let mut flat = Vec::new();
                for part in fan_out(tasks)? {
                    flat.extend_from_slice(&part);
                }
                context.upload(&flat)
            }
        }
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
        let folded = fold_lanes_by_halving(context, partials, lanes, blocks)?;
        folded
            .to_host()?
            .into_iter()
            .map(|value| {
                crate::cuda::common::device::fr_into(value).ok_or(CudaError::NotImplemented {
                    kernel: "CUDA kernels support only the BN254 scalar field",
                })
            })
            .collect()
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
        let _ = builder.arg(empty.limbs());
        let _ = builder.arg(&zero);
        let _ = builder.arg(&zero);
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
        let folded = fold_lanes_by_halving(context, partials, lanes, blocks)?;
        folded
            .to_host()?
            .into_iter()
            .map(|value| {
                crate::cuda::common::device::fr_into(value).ok_or(CudaError::NotImplemented {
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
                    *table = context.bind_rows(table, table.len(), challenge)?;
                }
            }
            Phase::Suffix { h_prime, eq_hi } => {
                *h_prime = context.bind_rows(h_prime, h_prime.len(), challenge)?;
                for table in eq_hi.iter_mut() {
                    *table = context.bind_rows(table, table.len(), challenge)?;
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

const fn absent() -> CudaError {
    CudaError::InvariantViolation {
        reason: "a RAM RA reduction window names an absent device",
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

    use super::{CyclePoints, DeviceRamRaReduction};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::pack::COLD;
    use crate::cuda::common::testing::{arb_point, fr};

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

    fn packed(indices: &[Option<usize>]) -> Vec<u32> {
        indices
            .iter()
            .map(|hot| hot.map_or(COLD, |address| address as u32))
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

    #[test]
    fn windowed_ram_ra_reduction_matches_the_whole_domain_round_for_round() {
        use std::sync::Arc;

        let Some(context) = shared_context() else {
            return;
        };
        let cycles = 1usize << LOG_T;
        let indices = hot_indices(0x5A1D);
        let words = packed(&indices);
        let address: Vec<Fr> = (0..LOG_K).map(|i| fr(11 + 3 * i as u64)).collect();
        let raf: Vec<Fr> = (0..LOG_T).map(|i| fr(29 + 7 * i as u64)).collect();
        let read_write: Vec<Fr> = (0..LOG_T).map(|i| fr(31 + 13 * i as u64)).collect();
        let val_check: Vec<Fr> = (0..LOG_T).map(|i| fr(37 + 17 * i as u64)).collect();
        let gamma = fr(83);
        let points = || CyclePoints {
            raf: &raf,
            read_write: &read_write,
            val_check: &val_check,
        };

        for shards in [2usize, 4, 8] {
            let device_words = context
                .upload_u32_slice(&words)
                .expect("upload the packed ram words");
            let device_eq = context
                .eq_evals(&address)
                .expect("device eq over the address point");
            let mut expected = DeviceRamRaReduction::new(
                context,
                &device_words,
                &device_eq,
                &points(),
                gamma,
                LOG_T,
            )
            .expect("whole-domain ram ra reduction");

            let len = cycles / shards;
            let windows: Vec<(usize, Arc<cudarc::driver::CudaSlice<u32>>)> = (0..shards)
                .map(|shard| {
                    (
                        0usize,
                        Arc::new(
                            context
                                .upload_u32_slice(&words[shard * len..(shard + 1) * len])
                                .expect("upload a packed ram window"),
                        ),
                    )
                })
                .collect();
            let mut got = DeviceRamRaReduction::new_windowed(
                context,
                &windows,
                &address,
                &points(),
                gamma,
                LOG_T,
            )
            .expect("windowed ram ra reduction");

            for round in 0..LOG_T {
                let want: Vec<Fr> = expected.round_evals(context).expect("whole round evals");
                let have: Vec<Fr> = got.round_evals(context).expect("windowed round evals");
                assert_eq!(
                    have, want,
                    "shards={shards} round {round}: the windowed reduction diverged",
                );
                let challenge = fr(600 + 19 * round as u64);
                expected.bind(context, challenge).expect("whole bind");
                got.bind(context, challenge).expect("windowed bind");
            }

            let want = expected.final_claim(context).expect("whole final claim");
            let have = got.final_claim(context).expect("windowed final claim");
            assert_eq!(have, want, "shards={shards}: the reduced claim diverged");
            assert_ne!(
                want,
                Fr::from_u64(0),
                "a degenerate fixture would hide a divergence",
            );
        }
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
            let words = packed(&indices);
            let device_words = context
                .upload_u32_slice(&words)
                .expect("upload the packed ram words");
            let eq_address = eq_table(&address);
            let device_eq = context.upload(&eq_address).expect("upload eq address");
            let points = CyclePoints {
                raf: &raf,
                read_write: &read_write,
                val_check: &val_check,
            };
            let state = DeviceRamRaReduction::new(
                context, &device_words, &device_eq, &points, gamma[0], LOG_T,
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
            let words = packed(&indices);
            let device_words = context
                .upload_u32_slice(&words)
                .expect("upload the packed ram words");
            let eq_address = eq_table(&address);
            let device_eq = context.upload(&eq_address).expect("upload eq address");
            let points = CyclePoints {
                raf: &raf,
                read_write: &read_write,
                val_check: &val_check,
            };
            let mut state = DeviceRamRaReduction::new(
                context, &device_words, &device_eq, &points, gamma[0], LOG_T,
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
