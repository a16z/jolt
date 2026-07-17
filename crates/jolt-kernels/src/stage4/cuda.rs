use jolt_field::Fr;
use jolt_poly::UnivariatePoly;

use crate::cuda::{
    build_schedules, CudaError, DeviceFrVec, RoundPolyTerms, RoundSchedule,
    SparseRegisterBindInputs, SparseRegisterEntries, SparseRegisterRoundInputs,
};

pub(crate) struct CudaDenseState {
    factors: Vec<DeviceFrVec>,
    scratch: Vec<DeviceFrVec>,
    term_coeffs: DeviceFrVec,
    term_factor_offsets: Vec<u32>,
    term_factor_indices: Vec<u32>,
    degree: usize,
    active_scale: Fr,
}

impl CudaDenseState {
    pub(crate) fn new(
        factors: &[&[Fr]],
        term_coeffs: Vec<Fr>,
        term_factor_offsets: Vec<u32>,
        term_factor_indices: Vec<u32>,
        degree: usize,
        active_scale: Fr,
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let device_factors = ctx.upload_many(factors).ok()?;
        let scratch = (0..factors.len())
            .map(|_| ctx.upload(&[]).ok())
            .collect::<Option<Vec<DeviceFrVec>>>()?;
        let term_coeffs = ctx.upload(&term_coeffs).ok()?;
        Some(Self {
            factors: device_factors,
            scratch,
            term_coeffs,
            term_factor_offsets,
            term_factor_indices,
            degree,
            active_scale,
        })
    }

    pub(crate) fn from_device_factors(
        factors: Vec<DeviceFrVec>,
        term_coeffs: Vec<Fr>,
        term_factor_offsets: Vec<u32>,
        term_factor_indices: Vec<u32>,
        degree: usize,
        active_scale: Fr,
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let scratch = (0..factors.len())
            .map(|_| ctx.upload(&[]).ok())
            .collect::<Option<Vec<DeviceFrVec>>>()?;
        let term_coeffs = ctx.upload(&term_coeffs).ok()?;
        Some(Self {
            factors,
            scratch,
            term_coeffs,
            term_factor_offsets,
            term_factor_indices,
            degree,
            active_scale,
        })
    }

    pub(crate) fn round_poly(&self) -> Result<UnivariatePoly<Fr>, CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        let factor_refs: Vec<&DeviceFrVec> = self.factors.iter().collect();
        let coeffs = ctx.dense_product_round_poly(RoundPolyTerms {
            factors: &factor_refs,
            term_coeffs: &self.term_coeffs,
            term_factor_offsets: &self.term_factor_offsets,
            term_factor_indices: &self.term_factor_indices,
            degree: self.degree,
        })?;
        let scaled = coeffs.into_iter().map(|c| c * self.active_scale).collect();
        Ok(UnivariatePoly::new(scaled))
    }

    pub(crate) fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        for (factor, scratch) in self.factors.iter_mut().zip(&mut self.scratch) {
            ctx.bind(factor, scratch, challenge)?;
        }
        Ok(())
    }

    pub(crate) fn factor_eval(&self, index: usize) -> Option<Result<Fr, CudaError>> {
        self.factors.get(index).map(DeviceFrVec::first)
    }
}


pub(crate) struct CudaSparseRegistersState {
    entries: SparseRegisterEntries,
    rd_inc: DeviceFrVec,
    rd_inc_scratch: DeviceFrVec,
    eq_cycle: crate::split_eq::CudaSplitEqState<'static>,
    schedules: Vec<RoundSchedule>,
    final_cols: Vec<u8>,
    round: usize,
}

impl CudaSparseRegistersState {
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn new<F: jolt_field::Field>(
        rows: &[usize],
        cols: &[u8],
        val: &[F],
        read_ra: &[F],
        rd_wa: &[F],
        prev_val: &[u64],
        next_val: &[u64],
        rd_inc: &[F],
        trace_point: &[F],
        trace_rounds: usize,
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let entries = SparseRegisterEntries {
            val: ctx.upload(crate::cuda::as_fr_slice(val)?).ok()?,
            read_ra: ctx.upload(crate::cuda::as_fr_slice(read_ra)?).ok()?,
            rd_wa: ctx.upload(crate::cuda::as_fr_slice(rd_wa)?).ok()?,
            prev_val: ctx.upload(&fr_from_u64s(prev_val)).ok()?,
            next_val: ctx.upload(&fr_from_u64s(next_val)).ok()?,
        };
        let eq_cycle = crate::split_eq::CudaSplitEqState::new_low_to_high(
            ctx,
            crate::cuda::as_fr_slice(trace_point)?,
            None,
        )
        .ok()?;
        let (schedules, final_cols) = build_schedules(rows, cols, trace_rounds);
        Some(Self {
            entries,
            rd_inc: ctx.resident_committed_clone(crate::cuda::as_fr_slice(rd_inc)?).ok()?,
            rd_inc_scratch: ctx.upload(&[]).ok()?,
            eq_cycle,
            schedules,
            final_cols,
            round: 0,
        })
    }

    #[expect(clippy::too_many_arguments)]
    pub(crate) fn new_from_raw<F: jolt_field::Field>(
        rows: &[usize],
        cols: &[u8],
        prev_val: &[u64],
        next_val: &[u64],
        rs1_flag: &[u8],
        rs2_flag: &[u8],
        rd_flag: &[u8],
        rd_inc: &[F],
        trace_point: &[F],
        gamma: F,
        gamma2: F,
        trace_rounds: usize,
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let n = rows.len();

        let prev_dev = ctx.upload_u64_slice(prev_val).ok()?;
        let next_dev = ctx.upload_u64_slice(next_val).ok()?;
        let val = ctx.u64_to_mont_dev(&prev_dev, n).ok()?;
        let prev = ctx.u64_to_mont_dev(&prev_dev, n).ok()?;
        let next = ctx.u64_to_mont_dev(&next_dev, n).ok()?;

        let rs1_u64: Vec<u64> = rs1_flag.iter().map(|&f| u64::from(f)).collect();
        let rs2_u64: Vec<u64> = rs2_flag.iter().map(|&f| u64::from(f)).collect();
        let rd_u64: Vec<u64> = rd_flag.iter().map(|&f| u64::from(f)).collect();
        let rs1_dev = ctx.upload_u64_slice(&rs1_u64).ok()?;
        let rs2_dev = ctx.upload_u64_slice(&rs2_u64).ok()?;
        let rd_dev = ctx.upload_u64_slice(&rd_u64).ok()?;

        let rd_wa = ctx.u64_to_mont_dev(&rd_dev, n).ok()?;

        let mut read_ra = ctx.u64_to_mont_dev(&rs1_dev, n).ok()?;
        ctx.mul_scalar(&mut read_ra, crate::cuda::into_fr(gamma)?).ok()?;
        let mut rs2_term = ctx.u64_to_mont_dev(&rs2_dev, n).ok()?;
        ctx.mul_scalar(&mut rs2_term, crate::cuda::into_fr(gamma2)?).ok()?;
        ctx.add(&mut read_ra, &rs2_term).ok()?;

        let entries = SparseRegisterEntries {
            val,
            read_ra,
            rd_wa,
            prev_val: prev,
            next_val: next,
        };
        let eq_cycle = crate::split_eq::CudaSplitEqState::new_low_to_high(
            ctx,
            crate::cuda::as_fr_slice(trace_point)?,
            None,
        )
        .ok()?;
        let (schedules, final_cols) = build_schedules(rows, cols, trace_rounds);
        Some(Self {
            entries,
            rd_inc: ctx.resident_committed_clone(crate::cuda::as_fr_slice(rd_inc)?).ok()?,
            rd_inc_scratch: ctx.upload(&[]).ok()?,
            eq_cycle,
            schedules,
            final_cols,
            round: 0,
        })
    }

    pub(crate) fn round_poly_q(&self) -> Option<[Fr; 2]> {
        let ctx = crate::cuda::shared_ctx()?;
        let schedule = self.schedules.get(self.round)?;
        let e_in_dev = self.eq_cycle.e_in_device();
        let e_out_dev = self.eq_cycle.e_out_device();
        let in_pairs = if e_in_dev.len() > 1 { (e_in_dev.len() / 2) as u32 } else { 0 };
        ctx.sparse_register_round_poly(SparseRegisterRoundInputs {
            val: &self.entries.val,
            read_ra: &self.entries.read_ra,
            rd_wa: &self.entries.rd_wa,
            prev_val: &self.entries.prev_val,
            next_val: &self.entries.next_val,
            even_idx: &schedule.even_idx,
            odd_idx: &schedule.odd_idx,
            pair: &schedule.pair,
            rd_inc: &self.rd_inc,
            e_in: e_in_dev,
            e_out: e_out_dev,
            in_pairs,
        })
        .ok()
    }

    pub(crate) fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        let schedule = self.schedules.get(self.round).ok_or(CudaError::Pool)?;
        let entries = ctx.sparse_register_bind(SparseRegisterBindInputs {
            val: &self.entries.val,
            read_ra: &self.entries.read_ra,
            rd_wa: &self.entries.rd_wa,
            prev_val: &self.entries.prev_val,
            next_val: &self.entries.next_val,
            even_idx: &schedule.even_idx,
            odd_idx: &schedule.odd_idx,
            challenge,
        })?;
        self.entries = entries;
        ctx.bind(&mut self.rd_inc, &mut self.rd_inc_scratch, challenge)?;
        self.eq_cycle.bind(challenge)?;
        self.round += 1;
        Ok(())
    }

    pub(crate) fn rd_inc_first(&self) -> Result<Fr, CudaError> {
        self.rd_inc.first()
    }

    pub(crate) fn materialize<F: jolt_field::Field>(
        &self,
        register_count: usize,
    ) -> Option<(Vec<F>, Vec<F>, Vec<F>)> {
        let val = self.entries.val.to_host().ok()?;
        let read_ra = self.entries.read_ra.to_host().ok()?;
        let rd_wa = self.entries.rd_wa.to_host().ok()?;
        let mut registers_val = vec![F::zero(); register_count];
        let mut read = vec![F::zero(); register_count];
        let mut wa = vec![F::zero(); register_count];
        for (entry, &col) in self.final_cols.iter().enumerate() {
            let col = usize::from(col);
            if col >= register_count {
                return None;
            }
            registers_val[col] = crate::cuda::fr_into::<F>(val[entry])?;
            read[col] = crate::cuda::fr_into::<F>(read_ra[entry])?;
            wa[col] = crate::cuda::fr_into::<F>(rd_wa[entry])?;
        }
        Some((registers_val, read, wa))
    }
}

fn fr_from_u64s(values: &[u64]) -> Vec<Fr> {
    use jolt_field::Field;
    use rayon::prelude::*;
    values.par_iter().map(|&v| Fr::from_u64(v)).collect()
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stage4::{
        append_sparse_register_entries, SparseRegisterEntry, Stage4RegisterAccess,
        Stage4RegisterRead, Stage4RegisterWrite,
    };
    use jolt_field::Field;
    use num_traits::One;

    #[test]
    fn sparse_registers_from_raw_matches_new() {
        let register_count = 8usize;
        let trace_rounds = 4usize;
        let trace_len = 1usize << trace_rounds;
        let gamma = Fr::from_u64(7);
        let gamma2 = gamma * gamma;
        let trace_point: Vec<Fr> =
            (0..trace_rounds).map(|i| Fr::from_u64((i + 2) as u64)).collect();
        let rd_inc: Vec<Fr> = (0..trace_len).map(|r| Fr::from_u64((r + 41) as u64)).collect();

        let reg = |seed: usize, salt: usize| (seed.wrapping_mul(2_654_435_761) + salt) % register_count;
        let accesses: Vec<Stage4RegisterAccess> = (0..trace_len)
            .map(|row| Stage4RegisterAccess {
                rs1: (row % 5 != 0).then(|| Stage4RegisterRead {
                    address: reg(row, 1),
                    value: (row as u64).wrapping_mul(3) + 1,
                }),
                rs2: (row % 3 != 1).then(|| Stage4RegisterRead {
                    address: reg(row, 2),
                    value: (row as u64).wrapping_mul(5) + 2,
                }),
                rd: (row % 4 != 2).then(|| Stage4RegisterWrite {
                    address: reg(row, 3),
                    pre_value: (row as u64).wrapping_mul(7) + 3,
                    post_value: (row as u64).wrapping_mul(11) + 4,
                }),
            })
            .collect();

        let mut entries: Vec<SparseRegisterEntry<Fr>> = Vec::new();
        for (row, access) in accesses.iter().enumerate() {
            append_sparse_register_entries(register_count, row, *access, gamma, gamma2, &mut entries)
                .unwrap();
        }
        let rows: Vec<usize> = entries.iter().map(|e| e.row).collect();
        let cols: Vec<u8> = entries.iter().map(|e| e.col).collect();
        let val: Vec<Fr> = entries.iter().map(|e| e.val).collect();
        let read_ra: Vec<Fr> = entries.iter().map(|e| e.read_ra).collect();
        let rd_wa: Vec<Fr> = entries.iter().map(|e| e.rd_wa).collect();
        let prev_val: Vec<u64> = entries.iter().map(|e| e.prev_val).collect();
        let next_val: Vec<u64> = entries.iter().map(|e| e.next_val).collect();

        let (mut rs1_flag, mut rs2_flag, mut rd_flag) = (Vec::new(), Vec::new(), Vec::new());
        for e in &entries {
            let is_rd = e.rd_wa == Fr::one();
            let has_rs1 = e.read_ra == gamma || e.read_ra == gamma + gamma2;
            let has_rs2 = e.read_ra == gamma2 || e.read_ra == gamma + gamma2;
            rs1_flag.push(u8::from(has_rs1));
            rs2_flag.push(u8::from(has_rs2));
            rd_flag.push(u8::from(is_rd));
        }

        let mut from_new = CudaSparseRegistersState::new(
            &rows, &cols, &val, &read_ra, &rd_wa, &prev_val, &next_val, &rd_inc, &trace_point,
            trace_rounds,
        )
        .unwrap();
        let mut from_raw = CudaSparseRegistersState::new_from_raw(
            &rows,
            &cols,
            &prev_val,
            &next_val,
            &rs1_flag,
            &rs2_flag,
            &rd_flag,
            &rd_inc,
            &trace_point,
            gamma,
            gamma2,
            trace_rounds,
        )
        .unwrap();

        for round in 0..trace_rounds {
            let expected = from_new.round_poly_q().unwrap();
            let got = from_raw.round_poly_q().unwrap();
            assert_eq!(got, expected, "round {round} q");
            let challenge = Fr::from_u64((round + 9) as u64);
            from_new.bind(challenge).unwrap();
            from_raw.bind(challenge).unwrap();
        }

        assert_eq!(
            from_raw.rd_inc_first().unwrap(),
            from_new.rd_inc_first().unwrap(),
            "rd_inc"
        );
        let expected_mat = from_new.materialize::<Fr>(register_count).unwrap();
        let got_mat = from_raw.materialize::<Fr>(register_count).unwrap();
        assert_eq!(got_mat, expected_mat, "materialize");
    }
}
