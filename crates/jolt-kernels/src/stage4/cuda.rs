use jolt_field::Fr;
use jolt_poly::UnivariatePoly;

use crate::cuda::{
    CudaError, DeviceFrVec, RoundPolyTerms, SparseRegisterBindInputs, SparseRegisterEntries,
    SparseRegisterRoundInputs,
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

struct RoundSchedule {
    even_idx: Vec<i32>,
    odd_idx: Vec<i32>,
    pair: Vec<u32>,
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
            rd_inc: ctx.upload(crate::cuda::as_fr_slice(rd_inc)?).ok()?,
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

fn build_schedules(
    rows: &[usize],
    cols: &[u8],
    trace_rounds: usize,
) -> (Vec<RoundSchedule>, Vec<u8>) {
    let mut cur_rows: Vec<usize> = rows.to_vec();
    let mut cur_cols: Vec<u8> = cols.to_vec();
    let mut schedules = Vec::with_capacity(trace_rounds);

    for _ in 0..trace_rounds {
        let mut even_idx = Vec::new();
        let mut odd_idx = Vec::new();
        let mut pair = Vec::new();
        let mut next_rows = Vec::new();
        let mut next_cols = Vec::new();

        let mut cursor = 0usize;
        while cursor < cur_rows.len() {
            let p = cur_rows[cursor] / 2;
            let even_row = 2 * p;
            let odd_row = even_row + 1;
            let even_start = cursor;
            while cursor < cur_rows.len() && cur_rows[cursor] == even_row {
                cursor += 1;
            }
            let even = even_start..cursor;
            let odd_start = cursor;
            while cursor < cur_rows.len() && cur_rows[cursor] == odd_row {
                cursor += 1;
            }
            let odd = odd_start..cursor;

            let mut i = even.start;
            let mut j = odd.start;
            while i < even.end || j < odd.end {
                let (ei, oi, col) = if j >= odd.end
                    || (i < even.end && cur_cols[i] < cur_cols[j])
                {
                    let out = (i as i32, -1i32, cur_cols[i]);
                    i += 1;
                    out
                } else if i >= even.end || cur_cols[j] < cur_cols[i] {
                    let out = (-1i32, j as i32, cur_cols[j]);
                    j += 1;
                    out
                } else {
                    let out = (i as i32, j as i32, cur_cols[i]);
                    i += 1;
                    j += 1;
                    out
                };
                even_idx.push(ei);
                odd_idx.push(oi);
                pair.push(p as u32);
                next_rows.push(p);
                next_cols.push(col);
            }
        }

        schedules.push(RoundSchedule {
            even_idx,
            odd_idx,
            pair,
        });
        cur_rows = next_rows;
        cur_cols = next_cols;
    }

    (schedules, cur_cols)
}
