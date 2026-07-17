use jolt_field::Fr;
use jolt_poly::UnivariatePoly;

use crate::cuda::{
    build_schedules, CudaError, CudaSlice, DeviceFrVec, RamRwAddressBindInputs,
    RamRwAddressRoundInputs, RamRwCycleBindInputs, RamRwCycleRoundInputs, RoundPolyTerms,
    RoundSchedule,
};
use crate::split_eq::CudaSplitEqState;

pub(crate) struct CudaDenseState {
    factors: Vec<DeviceFrVec>,
    scratch: Vec<DeviceFrVec>,
    term_coeffs: DeviceFrVec,
    degree: usize,
}

impl CudaDenseState {
    pub(crate) fn new(factors: &[&[Fr]]) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let device_factors = ctx.upload_many(factors).ok()?;
        Self::from_device_factors(device_factors)
    }

    pub(crate) fn from_device_factors(device_factors: Vec<DeviceFrVec>) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let degree = device_factors.len();
        let scratch = (0..degree)
            .map(|_| ctx.upload(&[]).ok())
            .collect::<Option<Vec<DeviceFrVec>>>()?;
        let term_coeffs = ctx.upload(&[Fr::from(1u64)]).ok()?;
        Some(Self {
            factors: device_factors,
            scratch,
            term_coeffs,
            degree,
        })
    }

    pub(crate) fn round_poly(&self) -> Result<UnivariatePoly<Fr>, CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        let factor_refs: Vec<&DeviceFrVec> = self.factors.iter().collect();
        let single_term_offsets = [0u32, self.degree as u32];
        let single_term_indices: Vec<u32> = (0..self.degree as u32).collect();
        let coeffs = ctx.dense_product_round_poly(RoundPolyTerms {
            factors: &factor_refs,
            term_coeffs: &self.term_coeffs,
            term_factor_offsets: &single_term_offsets,
            term_factor_indices: &single_term_indices,
            degree: self.degree,
        })?;
        Ok(UnivariatePoly::new(coeffs))
    }

    pub(crate) fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        for (factor, scratch) in self.factors.iter_mut().zip(&mut self.scratch) {
            ctx.bind(factor, scratch, challenge)?;
        }
        Ok(())
    }

    pub(crate) fn factor_eval(&self, index: usize) -> Result<Fr, CudaError> {
        self.factors[index].first()
    }
}

struct ResidentSchedule {
    even_idx: CudaSlice<i32>,
    odd_idx: CudaSlice<i32>,
    pair: CudaSlice<u32>,
    items: usize,
}

pub(crate) struct CudaRamReadWriteState {
    val_coeff: DeviceFrVec,
    ra_coeff: DeviceFrVec,
    prev_val: DeviceFrVec,
    next_val: DeviceFrVec,
    inc: DeviceFrVec,
    inc_scratch: DeviceFrVec,
    val_init: DeviceFrVec,
    val_init_scratch: DeviceFrVec,
    cycle_eq: CudaSplitEqState<'static>,
    cycle_schedules: Vec<ResidentSchedule>,
    address_schedules: Vec<ResidentSchedule>,
    gamma: Fr,
    log_t: usize,
    round: usize,
}

impl CudaRamReadWriteState {
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn new<F: jolt_field::Field>(
        rows: &[usize],
        cols: &[usize],
        val_coeff: &[F],
        ra_coeff: &[F],
        prev_val: &[u64],
        next_val: &[u64],
        inc: &[F],
        val_init: &[F],
        r_cycle: &[F],
        gamma: F,
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let log_t = r_cycle.len();
        let log_k = val_init.len().trailing_zeros() as usize;

        let (cycle_rounds, cycle_final_cols) = build_schedules(rows, cols, log_t);
        let address_rounds = build_address_schedules(&cycle_final_cols, log_k);

        let cycle_schedules = upload_schedules(ctx, &cycle_rounds)?;
        let address_schedules = upload_schedules(ctx, &address_rounds)?;

        let cycle_eq =
            CudaSplitEqState::new_low_to_high(ctx, crate::cuda::as_fr_slice(r_cycle)?, None).ok()?;

        Some(Self {
            val_coeff: ctx.upload(crate::cuda::as_fr_slice(val_coeff)?).ok()?,
            ra_coeff: ctx.upload(crate::cuda::as_fr_slice(ra_coeff)?).ok()?,
            prev_val: ctx.upload(&fr_from_u64s(prev_val)).ok()?,
            next_val: ctx.upload(&fr_from_u64s(next_val)).ok()?,
            inc: ctx.upload(crate::cuda::as_fr_slice(inc)?).ok()?,
            inc_scratch: ctx.upload(&[]).ok()?,
            val_init: ctx.upload(crate::cuda::as_fr_slice(val_init)?).ok()?,
            val_init_scratch: ctx.upload(&[]).ok()?,
            cycle_eq,
            cycle_schedules,
            address_schedules,
            gamma: crate::cuda::into_fr(gamma)?,
            log_t,
            round: 0,
        })
    }

    #[expect(clippy::too_many_arguments)]
    #[cfg_attr(not(test), expect(dead_code, reason = "used once new_from_raw wiring lands"))]
    pub(crate) fn new_from_raw<F: jolt_field::Field>(
        rows: &[usize],
        cols: &[usize],
        filtered_read: &[u64],
        filtered_write: &[u64],
        all_read: &[u64],
        all_write: &[u64],
        initial_ram: &[u64],
        r_cycle: &[F],
        gamma: F,
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let log_t = r_cycle.len();
        let log_k = initial_ram.len().trailing_zeros() as usize;
        let n = filtered_read.len();

        let (cycle_rounds, cycle_final_cols) = build_schedules(rows, cols, log_t);
        let address_rounds = build_address_schedules(&cycle_final_cols, log_k);
        let cycle_schedules = upload_schedules(ctx, &cycle_rounds)?;
        let address_schedules = upload_schedules(ctx, &address_rounds)?;
        let cycle_eq =
            CudaSplitEqState::new_low_to_high(ctx, crate::cuda::as_fr_slice(r_cycle)?, None).ok()?;

        let filtered_read_dev = ctx.upload_u64_slice(filtered_read).ok()?;
        let filtered_write_dev = ctx.upload_u64_slice(filtered_write).ok()?;
        let val_coeff = ctx.u64_to_mont_dev(&filtered_read_dev, n).ok()?;
        let prev_val = ctx.u64_to_mont_dev(&filtered_read_dev, n).ok()?;
        let next_val = ctx.u64_to_mont_dev(&filtered_write_dev, n).ok()?;

        let ones_dev = ctx.upload_u64_slice(&vec![1u64; n]).ok()?;
        let ra_coeff = ctx.u64_to_mont_dev(&ones_dev, n).ok()?;

        let all_read_dev = ctx.upload_u64_slice(all_read).ok()?;
        let all_write_dev = ctx.upload_u64_slice(all_write).ok()?;
        let all_read_mont = ctx.u64_to_mont_dev(&all_read_dev, all_read.len()).ok()?;
        let mut inc = ctx.u64_to_mont_dev(&all_write_dev, all_write.len()).ok()?;
        ctx.sub(&mut inc, &all_read_mont).ok()?;

        let initial_ram_dev = ctx.upload_u64_slice(initial_ram).ok()?;
        let val_init = ctx.u64_to_mont_dev(&initial_ram_dev, initial_ram.len()).ok()?;

        Some(Self {
            val_coeff,
            ra_coeff,
            prev_val,
            next_val,
            inc,
            inc_scratch: ctx.upload(&[]).ok()?,
            val_init,
            val_init_scratch: ctx.upload(&[]).ok()?,
            cycle_eq,
            cycle_schedules,
            address_schedules,
            gamma: crate::cuda::into_fr(gamma)?,
            log_t,
            round: 0,
        })
    }

    pub(crate) fn round_poly<F: jolt_field::Field>(
        &self,
        previous_claim: F,
    ) -> Option<UnivariatePoly<F>> {
        let previous_claim = crate::cuda::into_fr(previous_claim)?;
        let poly = if self.round < self.log_t {
            self.cycle_round_poly(previous_claim)?
        } else {
            self.address_round_poly(previous_claim)?
        };
        let coeffs = poly
            .coefficients()
            .iter()
            .map(|&c| crate::cuda::fr_into::<F>(c))
            .collect::<Option<Vec<F>>>()?;
        Some(UnivariatePoly::new(coeffs))
    }

    fn cycle_round_poly(&self, previous_claim: Fr) -> Option<UnivariatePoly<Fr>> {
        let ctx = crate::cuda::shared_ctx()?;
        let schedule = self.cycle_schedules.get(self.round)?;
        let e_in = self.cycle_eq.e_in_device();
        let e_out = self.cycle_eq.e_out_device();
        let in_pairs = if e_in.len() > 1 { (e_in.len() / 2) as u32 } else { 0 };
        let (q_constant, q_quadratic) = ctx
            .ram_rw_cycle_round_coefficients(RamRwCycleRoundInputs {
                val_coeff: &self.val_coeff,
                ra_coeff: &self.ra_coeff,
                prev_val: &self.prev_val,
                next_val: &self.next_val,
                even_idx: &schedule.even_idx,
                odd_idx: &schedule.odd_idx,
                pair: &schedule.pair,
                inc: &self.inc,
                e_in,
                e_out,
                gamma: self.gamma,
                in_pairs,
                items: schedule.items,
            })
            .ok()?;
        Some(super::gruen_cubic_poly(
            self.cycle_eq.current_target(),
            q_constant,
            q_quadratic,
            previous_claim,
        ))
    }

    fn address_round_poly(&self, previous_claim: Fr) -> Option<UnivariatePoly<Fr>> {
        let ctx = crate::cuda::shared_ctx()?;
        let schedule = self.address_schedules.get(self.round - self.log_t)?;
        let eq = self.cycle_eq.eval().ok()?;
        let inc0 = self.inc.first().ok()?;
        let (q0, q2) = ctx
            .ram_rw_address_round_coefficients(RamRwAddressRoundInputs {
                ra_coeff: &self.ra_coeff,
                val_coeff: &self.val_coeff,
                val_init: &self.val_init,
                even_idx: &schedule.even_idx,
                odd_idx: &schedule.odd_idx,
                pair: &schedule.pair,
                eq,
                gamma: self.gamma,
                inc0,
                num_groups: schedule.items,
            })
            .ok()?;
        Some(UnivariatePoly::from_evals_and_hint(previous_claim, &[q0, q2]))
    }

    pub(crate) fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        if self.round < self.log_t {
            let schedule = self
                .cycle_schedules
                .get(self.round)
                .ok_or(CudaError::Pool)?;
            let entries = ctx.ram_rw_cycle_bind(RamRwCycleBindInputs {
                val_coeff: &self.val_coeff,
                ra_coeff: &self.ra_coeff,
                prev_val: &self.prev_val,
                next_val: &self.next_val,
                even_idx: &schedule.even_idx,
                odd_idx: &schedule.odd_idx,
                challenge,
                items: schedule.items,
            })?;
            self.val_coeff = entries.val_coeff;
            self.ra_coeff = entries.ra_coeff;
            self.prev_val = entries.prev_val;
            self.next_val = entries.next_val;
            ctx.bind(&mut self.inc, &mut self.inc_scratch, challenge)?;
            self.cycle_eq.bind(challenge)?;
        } else {
            let schedule = self
                .address_schedules
                .get(self.round - self.log_t)
                .ok_or(CudaError::Pool)?;
            let entries = ctx.ram_rw_address_bind(RamRwAddressBindInputs {
                ra_coeff: &self.ra_coeff,
                val_coeff: &self.val_coeff,
                prev_val: &self.prev_val,
                next_val: &self.next_val,
                val_init: &self.val_init,
                even_idx: &schedule.even_idx,
                odd_idx: &schedule.odd_idx,
                pair: &schedule.pair,
                challenge,
                num_groups: schedule.items,
            })?;
            self.ra_coeff = entries.ra_coeff;
            self.val_coeff = entries.val_coeff;
            self.prev_val = entries.prev_val;
            self.next_val = entries.next_val;
            ctx.bind(&mut self.val_init, &mut self.val_init_scratch, challenge)?;
        }
        self.round += 1;
        Ok(())
    }

    pub(crate) fn val_eval(&self) -> Result<Fr, CudaError> {
        if self.val_coeff.is_empty() {
            self.val_init.first()
        } else {
            self.val_coeff.first()
        }
    }

    pub(crate) fn ra_eval(&self) -> Result<Fr, CudaError> {
        if self.ra_coeff.is_empty() {
            Ok(Fr::from(0u64))
        } else {
            self.ra_coeff.first()
        }
    }

    pub(crate) fn inc_eval(&self) -> Result<Fr, CudaError> {
        self.inc.first()
    }
}

fn fr_from_u64s(values: &[u64]) -> Vec<Fr> {
    use jolt_field::Field;
    use rayon::prelude::*;
    values.par_iter().map(|&v| Fr::from_u64(v)).collect()
}

fn upload_schedules(
    ctx: &crate::cuda::CudaKernelContext,
    rounds: &[RoundSchedule],
) -> Option<Vec<ResidentSchedule>> {
    rounds
        .iter()
        .map(|round| {
            Some(ResidentSchedule {
                even_idx: ctx.upload_i32_slice(&round.even_idx).ok()?,
                odd_idx: ctx.upload_i32_slice(&round.odd_idx).ok()?,
                pair: ctx.upload_u32_slice(&round.pair).ok()?,
                items: round.even_idx.len(),
            })
        })
        .collect()
}

fn build_address_schedules(cycle_final_cols: &[usize], log_k: usize) -> Vec<RoundSchedule> {
    let mut order: Vec<usize> = (0..cycle_final_cols.len()).collect();
    order.sort_by_key(|&slot| cycle_final_cols[slot]);
    let sorted_cols: Vec<usize> = order.iter().map(|&slot| cycle_final_cols[slot]).collect();
    let zero_rows = vec![0usize; sorted_cols.len()];
    let (mut rounds, _final_rows) = build_schedules(&sorted_cols, &zero_rows, log_k);
    if let Some(first) = rounds.first_mut() {
        for slot in first.even_idx.iter_mut().chain(first.odd_idx.iter_mut()) {
            if *slot >= 0 {
                *slot = order[*slot as usize] as i32;
            }
        }
    }
    rounds
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::split_eq::SplitEqState;
    use crate::stage2::{RamCycleEntry, RamReadWriteState};
    use jolt_field::Field;
    use num_traits::{One, Zero};

    #[test]
    fn ram_rw_full_relation_matches_cpu() {
        let log_t = 4usize;
        let t = 1usize << log_t;
        let log_k = 3usize;
        let k = 1usize << log_k;
        let gamma = Fr::from_u64(9);
        let r_cycle: Vec<Fr> = (0..log_t).map(|i| Fr::from_u64((i + 3) as u64)).collect();

        let raw: [(usize, usize); 10] = [
            (0, 1),
            (1, 4),
            (2, 2),
            (3, 5),
            (5, 0),
            (6, 2),
            (8, 7),
            (9, 2),
            (12, 6),
            (14, 3),
        ];
        let cycle_entries: Vec<RamCycleEntry<Fr>> = raw
            .iter()
            .enumerate()
            .map(|(g, &(row, col))| RamCycleEntry {
                row,
                col,
                prev_val: (g % 5) as u64,
                next_val: (g % 7) as u64,
                val_coeff: Fr::from_u64((g + 1) as u64),
                ra_coeff: Fr::from_u64((g + 100) as u64),
            })
            .collect();
        let inc: Vec<Fr> = (0..t).map(|row| Fr::from_u64((row + 33) as u64)).collect();
        let val_init: Vec<Fr> = (0..k).map(|i| Fr::from_u64((i + 3) as u64)).collect();

        let rows: Vec<usize> = cycle_entries.iter().map(|e| e.row).collect();
        let cols: Vec<usize> = cycle_entries.iter().map(|e| e.col).collect();
        let val_coeff: Vec<Fr> = cycle_entries.iter().map(|e| e.val_coeff).collect();
        let ra_coeff: Vec<Fr> = cycle_entries.iter().map(|e| e.ra_coeff).collect();
        let prev_val: Vec<u64> = cycle_entries.iter().map(|e| e.prev_val).collect();
        let next_val: Vec<u64> = cycle_entries.iter().map(|e| e.next_val).collect();

        let mut cpu = RamReadWriteState {
            gamma,
            log_t,
            round: 0,
            cycle_eq: SplitEqState::new_low_to_high(&r_cycle, None),
            cycle_entries: cycle_entries.clone(),
            address_entries: Vec::new(),
            address_scratch: Vec::new(),
            inc: inc.clone(),
            inc_scratch: Vec::new(),
            val_init: val_init.clone(),
            val_init_scratch: Vec::new(),
            cuda: None,
        };
        let mut device = CudaRamReadWriteState::new(
            &rows, &cols, &val_coeff, &ra_coeff, &prev_val, &next_val, &inc, &val_init, &r_cycle,
            gamma,
        )
        .unwrap();

        let mut claim = Fr::from_u64(1_234_567);
        let total_rounds = log_t + log_k;
        for round in 0..total_rounds {
            let expected = cpu.round_poly(round, claim).unwrap();
            let got: UnivariatePoly<Fr> = device.round_poly(claim).unwrap();
            assert_eq!(got, expected, "round {round}");
            let challenge = Fr::from_u64((round + 7) as u64);
            cpu.ingest_challenge(challenge).unwrap();
            device.bind(challenge).unwrap();
            claim = expected.evaluate(challenge);
        }

        assert_eq!(device.val_eval().unwrap(), cpu.val_eval().unwrap(), "val_eval");
        assert_eq!(device.ra_eval().unwrap(), cpu.ra_eval().unwrap(), "ra_eval");
        assert_eq!(device.inc_eval().unwrap(), cpu.inc_eval().unwrap(), "inc_eval");
    }

    #[test]
    fn ram_rw_from_raw_matches_cpu() {
        let log_t = 4usize;
        let t = 1usize << log_t;
        let log_k = 3usize;
        let k = 1usize << log_k;
        let gamma = Fr::from_u64(9);
        let r_cycle: Vec<Fr> = (0..log_t).map(|i| Fr::from_u64((i + 3) as u64)).collect();

        // Per-cycle (read, write) and remapped address (None marks a no-op with no column).
        let raw: [(u64, u64, Option<usize>); 16] = [
            (3, 5, Some(1)),
            (7, 7, Some(4)),
            (2, 9, Some(2)),
            (0, 4, Some(5)),
            (1, 1, None),
            (6, 6, Some(0)),
            (8, 2, Some(2)),
            (5, 5, None),
            (4, 10, Some(7)),
            (9, 3, Some(2)),
            (0, 0, None),
            (11, 6, Some(6)),
            (2, 2, Some(3)),
            (13, 1, Some(1)),
            (7, 12, Some(3)),
            (5, 0, Some(4)),
        ];
        assert_eq!(raw.len(), t);
        let initial_ram: Vec<u64> = (0..k as u64).map(|i| if i % 2 == 0 { 0 } else { i + 5 }).collect();

        let all_read: Vec<u64> = raw.iter().map(|&(r, _, _)| r).collect();
        let all_write: Vec<u64> = raw.iter().map(|&(_, w, _)| w).collect();

        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut filtered_read = Vec::new();
        let mut filtered_write = Vec::new();
        for (row, &(read, write, col)) in raw.iter().enumerate() {
            if let Some(col) = col {
                rows.push(row);
                cols.push(col);
                filtered_read.push(read);
                filtered_write.push(write);
            }
        }

        let cycle_entries: Vec<RamCycleEntry<Fr>> = rows
            .iter()
            .zip(&cols)
            .zip(filtered_read.iter().zip(&filtered_write))
            .map(|((&row, &col), (&read, &write))| RamCycleEntry {
                row,
                col,
                prev_val: read,
                next_val: write,
                val_coeff: Fr::from_u64(read),
                ra_coeff: Fr::one(),
            })
            .collect();
        let inc: Vec<Fr> = raw
            .iter()
            .map(|&(read, write, _)| {
                if read == write {
                    Fr::zero()
                } else {
                    Fr::from_u64(write) - Fr::from_u64(read)
                }
            })
            .collect();
        let val_init: Vec<Fr> = initial_ram
            .iter()
            .map(|&v| if v == 0 { Fr::zero() } else { Fr::from_u64(v) })
            .collect();

        let mut cpu = RamReadWriteState {
            gamma,
            log_t,
            round: 0,
            cycle_eq: SplitEqState::new_low_to_high(&r_cycle, None),
            cycle_entries,
            address_entries: Vec::new(),
            address_scratch: Vec::new(),
            inc,
            inc_scratch: Vec::new(),
            val_init,
            val_init_scratch: Vec::new(),
            cuda: None,
        };
        let mut device = CudaRamReadWriteState::new_from_raw(
            &rows,
            &cols,
            &filtered_read,
            &filtered_write,
            &all_read,
            &all_write,
            &initial_ram,
            &r_cycle,
            gamma,
        )
        .unwrap();

        let mut claim = Fr::from_u64(1_234_567);
        for round in 0..(log_t + log_k) {
            let expected = cpu.round_poly(round, claim).unwrap();
            let got: UnivariatePoly<Fr> = device.round_poly(claim).unwrap();
            assert_eq!(got, expected, "round {round}");
            let challenge = Fr::from_u64((round + 7) as u64);
            cpu.ingest_challenge(challenge).unwrap();
            device.bind(challenge).unwrap();
            claim = expected.evaluate(challenge);
        }

        assert_eq!(device.val_eval().unwrap(), cpu.val_eval().unwrap(), "val_eval");
        assert_eq!(device.ra_eval().unwrap(), cpu.ra_eval().unwrap(), "ra_eval");
        assert_eq!(device.inc_eval().unwrap(), cpu.inc_eval().unwrap(), "inc_eval");
    }
}
