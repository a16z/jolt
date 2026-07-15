use jolt_field::Fr;
use jolt_poly::UnivariatePoly;

use crate::cuda::{
    CudaError, DeviceFrVec, InstructionRafCycleInputs, InstructionRafCycleSparseInputs,
    PrefixSuffixRoundInputs, RafQScatterInputs, ReadSuffixScatterInputs, ReadTableRoundInputs,
    RoundPolyTerms,
};

pub(crate) struct CudaAddressPhaseState {
    weight: DeviceFrVec,
    lookup_index_lo: crate::cuda::CudaSlice<u64>,
    lookup_index_hi: crate::cuda::CudaSlice<u64>,
    is_interleaved: crate::cuda::CudaSlice<u8>,
    trace_len: usize,
    table_cycle_lists: Vec<crate::cuda::CudaSlice<u32>>,
    table_cycle_lens: Vec<usize>,
    table_suffix_codes: Vec<crate::cuda::CudaSlice<u32>>,
    poly_len: usize,
}

impl CudaAddressPhaseState {
    pub(crate) fn new<F: jolt_field::Field>(
        u_evals: &[F],
        lookup_indices: &[u128],
        lookup_table_indices: &[Option<usize>],
        is_interleaved_operands: &[bool],
        table_suffix_codes: Vec<Vec<u32>>,
        poly_len: usize,
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let weight = ctx.upload(crate::cuda::as_fr_slice(u_evals)?).ok()?;
        use rayon::prelude::*;
        let lo: Vec<u64> = lookup_indices.par_iter().map(|&v| v as u64).collect();
        let hi: Vec<u64> = lookup_indices.par_iter().map(|&v| (v >> 64) as u64).collect();
        let flags: Vec<u8> = is_interleaved_operands.par_iter().map(|&b| u8::from(b)).collect();
        let lookup_index_lo = ctx.upload_u64_slice(&lo).ok()?;
        let lookup_index_hi = ctx.upload_u64_slice(&hi).ok()?;
        let is_interleaved = ctx.upload_u8_slice(&flags).ok()?;

        let mut cycle_lists: Vec<Vec<u32>> = vec![Vec::new(); table_suffix_codes.len()];
        for (cycle, table_index) in lookup_table_indices.iter().enumerate() {
            if let Some(table_index) = table_index {
                cycle_lists[*table_index].push(cycle as u32);
            }
        }
        let table_cycle_lens = cycle_lists.iter().map(Vec::len).collect();
        let table_cycle_lists = cycle_lists
            .iter()
            .map(|list| ctx.upload_u32_slice(list))
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        let table_suffix_codes = table_suffix_codes
            .iter()
            .map(|codes| ctx.upload_u32_slice(codes))
            .collect::<Result<Vec<_>, _>>()
            .ok()?;

        Some(Self {
            weight,
            lookup_index_lo,
            lookup_index_hi,
            is_interleaved,
            trace_len: lookup_indices.len(),
            table_cycle_lists,
            table_cycle_lens,
            table_suffix_codes,
            poly_len,
        })
    }

    pub(crate) fn advance_phase(
        &mut self,
        eq_table: &[Fr],
        shift: usize,
        mask: usize,
    ) -> Result<(), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        ctx.raf_weight_phase_update(
            &mut self.weight,
            eq_table,
            &self.lookup_index_lo,
            &self.lookup_index_hi,
            shift,
            mask,
        )
    }

    pub(crate) fn raf_banks(&self, suffix_len: usize) -> Result<[Vec<Fr>; 5], CudaError> {
        let banks = self.raf_banks_device(suffix_len)?;
        let mut out: Vec<Vec<Fr>> = Vec::with_capacity(5);
        for bank in &banks {
            out.push(bank.to_host()?);
        }
        out.try_into().map_err(|_| CudaError::Pool)
    }

    pub(crate) fn raf_banks_device(
        &self,
        suffix_len: usize,
    ) -> Result<[DeviceFrVec; 5], CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        let banks = ctx.raf_q_scatter(RafQScatterInputs {
            weight: &self.weight,
            lookup_index_lo: &self.lookup_index_lo,
            lookup_index_hi: &self.lookup_index_hi,
            is_interleaved: &self.is_interleaved,
            trace_len: self.trace_len,
            suffix_len,
            poly_len: self.poly_len,
        })?;
        banks.try_into().map_err(|_| CudaError::Pool)
    }

    pub(crate) fn read_suffix_banks_device(
        &self,
        table_index: usize,
        suffix_len: usize,
    ) -> Result<Vec<DeviceFrVec>, CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        ctx.read_suffix_scatter(ReadSuffixScatterInputs {
            weight: &self.weight,
            lookup_index_lo: &self.lookup_index_lo,
            lookup_index_hi: &self.lookup_index_hi,
            cycle_list: &self.table_cycle_lists[table_index],
            suffix_variants: &self.table_suffix_codes[table_index],
            m: self.table_cycle_lens[table_index],
            suffix_len,
            poly_len: self.poly_len,
        })
    }

    pub(crate) fn read_suffix_banks(
        &self,
        table_index: usize,
        suffix_len: usize,
    ) -> Result<Vec<Vec<Fr>>, CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        let banks = ctx.read_suffix_scatter(ReadSuffixScatterInputs {
            weight: &self.weight,
            lookup_index_lo: &self.lookup_index_lo,
            lookup_index_hi: &self.lookup_index_hi,
            cycle_list: &self.table_cycle_lists[table_index],
            suffix_variants: &self.table_suffix_codes[table_index],
            m: self.table_cycle_lens[table_index],
            suffix_len,
            poly_len: self.poly_len,
        })?;
        let mut out = Vec::with_capacity(banks.len());
        for bank in &banks {
            out.push(bank.to_host()?);
        }
        Ok(out)
    }

    pub(crate) fn table_is_empty(&self, table_index: usize) -> bool {
        self.table_cycle_lens[table_index] == 0
    }
}

struct ReadTableSchedule {
    variant: crate::cuda::CudaSlice<u32>,
    suffix_offset: crate::cuda::CudaSlice<u32>,
    suffix_count: crate::cuda::CudaSlice<u32>,
    num_tables: usize,
}

pub(crate) struct CudaAddressPhaseRound {
    left_operand_prefix: DeviceFrVec,
    right_operand_prefix: DeviceFrVec,
    identity_prefix: DeviceFrVec,
    raf_shift_half_q: DeviceFrVec,
    raf_left_q: DeviceFrVec,
    raf_right_q: DeviceFrVec,
    raf_shift_full_q: DeviceFrVec,
    raf_identity_q: DeviceFrVec,
    read_prefix_blob: DeviceFrVec,
    read_prefix_scratch: DeviceFrVec,
    read_prefix_count: usize,
    read_suffix_blob: DeviceFrVec,
    read_suffix_scratch: DeviceFrVec,
    read_suffix_polys_total: usize,
    schedule: ReadTableSchedule,
    gamma: Fr,
    gamma2: Fr,
    active_scale: Fr,
    poly_len: usize,
    scratch: DeviceFrVec,
}

impl CudaAddressPhaseRound {
    #[expect(clippy::too_many_arguments, reason = "device round state gathers all resident polys")]
    pub(crate) fn new(
        operand_prefixes: [DeviceFrVec; 3],
        raf_banks: [DeviceFrVec; 5],
        read_prefix_blob: DeviceFrVec,
        read_suffix_banks: Vec<Vec<DeviceFrVec>>,
        table_variants: Vec<u32>,
        gamma: Fr,
        gamma2: Fr,
        active_scale: Fr,
        poly_len: usize,
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let [left_operand_prefix, right_operand_prefix, identity_prefix] = operand_prefixes;
        let [raf_shift_half_q, raf_left_q, raf_right_q, raf_shift_full_q, raf_identity_q] =
            raf_banks;
        let read_prefix_count = if poly_len == 0 {
            0
        } else {
            read_prefix_blob.len() / poly_len
        };

        let mut suffix_offset = Vec::with_capacity(table_variants.len());
        let mut suffix_count = Vec::with_capacity(table_variants.len());
        let mut offset_polys = 0usize;
        let mut refs: Vec<&DeviceFrVec> = Vec::new();
        for banks in &read_suffix_banks {
            suffix_offset.push(offset_polys as u32);
            suffix_count.push(banks.len() as u32);
            offset_polys += banks.len();
            for bank in banks {
                refs.push(bank);
            }
        }
        let read_suffix_blob = ctx.concat_device(&refs).ok()?;
        let read_suffix_polys_total = offset_polys;

        let schedule = ReadTableSchedule {
            variant: ctx.upload_u32_slice(&table_variants).ok()?,
            suffix_offset: ctx.upload_u32_slice(&suffix_offset).ok()?,
            suffix_count: ctx.upload_u32_slice(&suffix_count).ok()?,
            num_tables: table_variants.len(),
        };

        Some(Self {
            left_operand_prefix,
            right_operand_prefix,
            identity_prefix,
            raf_shift_half_q,
            raf_left_q,
            raf_right_q,
            raf_shift_full_q,
            raf_identity_q,
            read_prefix_blob,
            read_prefix_scratch: ctx.upload(&[]).ok()?,
            read_prefix_count,
            read_suffix_blob,
            read_suffix_scratch: ctx.upload(&[]).ok()?,
            read_suffix_polys_total,
            schedule,
            gamma,
            gamma2,
            active_scale,
            poly_len,
            scratch: ctx.upload(&[]).ok()?,
        })
    }

    fn cur_len(&self) -> usize {
        self.left_operand_prefix.len()
    }

    fn read_table_round_evals(&self) -> Result<[Fr; 2], CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        let len = self.left_operand_prefix.len();
        let half = len / 2;
        if half == 0 || self.schedule.num_tables == 0 {
            use num_traits::Zero;
            return Ok([Fr::zero(); 2]);
        }
        let mut item_table = Vec::with_capacity(self.schedule.num_tables * half);
        let mut item_row = Vec::with_capacity(self.schedule.num_tables * half);
        for t in 0..self.schedule.num_tables {
            for row in 0..half {
                item_table.push(t as u32);
                item_row.push(row as u32);
            }
        }
        let item_table_dev = ctx.upload_u32_slice(&item_table)?;
        let item_row_dev = ctx.upload_u32_slice(&item_row)?;
        ctx.read_table_round_evals(ReadTableRoundInputs {
            prefix_polys: &self.read_prefix_blob,
            suffix_blob: &self.read_suffix_blob,
            table_variant: &self.schedule.variant,
            table_suffix_offset: &self.schedule.suffix_offset,
            table_suffix_count: &self.schedule.suffix_count,
            item_table: &item_table_dev,
            item_row: &item_row_dev,
            len,
            items: item_table.len(),
        })
    }

    pub(crate) fn round_poly(&self, previous_claim: Fr) -> Result<UnivariatePoly<Fr>, CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        let len = self.cur_len();
        let read = self.read_table_round_evals()?;
        let left = ctx.prefix_suffix_round_evals(PrefixSuffixRoundInputs {
            prefix: Some(&self.left_operand_prefix),
            q0: &self.raf_shift_half_q,
            q1: &self.raf_left_q,
            len,
        })?;
        let right = ctx.prefix_suffix_round_evals(PrefixSuffixRoundInputs {
            prefix: Some(&self.right_operand_prefix),
            q0: &self.raf_shift_half_q,
            q1: &self.raf_right_q,
            len,
        })?;
        let identity = ctx.prefix_suffix_round_evals(PrefixSuffixRoundInputs {
            prefix: Some(&self.identity_prefix),
            q0: &self.raf_shift_full_q,
            q1: &self.raf_identity_q,
            len,
        })?;
        let eval_at_0 = (read[0] + self.gamma * left.0 + self.gamma2 * (right.0 + identity.0))
            * self.active_scale;
        let eval_at_2 = (read[1] + self.gamma * left.1 + self.gamma2 * (right.1 + identity.1))
            * self.active_scale;
        Ok(UnivariatePoly::from_evals_and_hint(
            previous_claim,
            &[eval_at_0, eval_at_2],
        ))
    }

    pub(crate) fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        ctx.bind_high_to_low(&mut self.left_operand_prefix, &mut self.scratch, challenge)?;
        ctx.bind_high_to_low(&mut self.right_operand_prefix, &mut self.scratch, challenge)?;
        ctx.bind_high_to_low(&mut self.identity_prefix, &mut self.scratch, challenge)?;
        ctx.bind_high_to_low(&mut self.raf_shift_half_q, &mut self.scratch, challenge)?;
        ctx.bind_high_to_low(&mut self.raf_left_q, &mut self.scratch, challenge)?;
        ctx.bind_high_to_low(&mut self.raf_right_q, &mut self.scratch, challenge)?;
        ctx.bind_high_to_low(&mut self.raf_shift_full_q, &mut self.scratch, challenge)?;
        ctx.bind_high_to_low(&mut self.raf_identity_q, &mut self.scratch, challenge)?;
        ctx.batched_bind_high_to_low(
            &mut self.read_prefix_blob,
            &mut self.read_prefix_scratch,
            self.read_prefix_count,
            challenge,
        )?;
        if self.read_suffix_polys_total > 0 {
            ctx.batched_bind_high_to_low(
                &mut self.read_suffix_blob,
                &mut self.read_suffix_scratch,
                self.read_suffix_polys_total,
                challenge,
            )?;
        }
        self.poly_len /= 2;
        Ok(())
    }

    pub(crate) fn operand_checkpoints(&self) -> Result<(Fr, Fr, Fr), CudaError> {
        Ok((
            self.left_operand_prefix.first()?,
            self.right_operand_prefix.first()?,
            self.identity_prefix.first()?,
        ))
    }

    pub(crate) fn read_prefix_first(&self, num_prefixes: usize) -> Result<Vec<Fr>, CudaError> {
        let host = self.read_prefix_blob.to_host()?;
        let stride = self.poly_len;
        Ok((0..num_prefixes).map(|p| host[p * stride]).collect())
    }
}

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

pub(crate) struct CudaInstructionRafCycleState {
    combined: DeviceFrVec,
    chunks: Vec<DeviceFrVec>,
    scratch: DeviceFrVec,
}

impl CudaInstructionRafCycleState {
    pub(crate) fn new<F: jolt_field::Field>(combined: &[F], chunks: &[Vec<F>]) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        if chunks.len() != 8 {
            return None;
        }
        let mut refs: Vec<&[Fr]> = Vec::with_capacity(chunks.len() + 1);
        refs.push(crate::cuda::as_fr_slice(combined)?);
        for chunk in chunks {
            refs.push(crate::cuda::as_fr_slice(chunk)?);
        }
        let mut uploaded = ctx.upload_many(&refs).ok()?;
        let mut drain = uploaded.drain(..);
        let combined = drain.next()?;
        let device_chunks: Vec<DeviceFrVec> = drain.collect();
        Some(Self {
            combined,
            chunks: device_chunks,
            scratch: ctx.upload(&[]).ok()?,
        })
    }

    fn from_device(combined: DeviceFrVec, chunks: Vec<DeviceFrVec>) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        if chunks.len() != 8 {
            return None;
        }
        Some(Self {
            combined,
            chunks,
            scratch: ctx.upload(&[]).ok()?,
        })
    }

    pub(crate) fn round_poly_evals(
        &self,
        e_in: &DeviceFrVec,
        e_out: &DeviceFrVec,
    ) -> Option<[Fr; 9]> {
        let ctx = crate::cuda::shared_ctx()?;
        let chunk_refs: Vec<&DeviceFrVec> = self.chunks.iter().collect();
        ctx.instruction_raf_cycle_round_poly(InstructionRafCycleInputs {
            combined: &self.combined,
            chunks: &chunk_refs,
            e_in,
            e_out,
        })
        .ok()
    }

    pub(crate) fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        ctx.bind(&mut self.combined, &mut self.scratch, challenge)?;
        for chunk in &mut self.chunks {
            ctx.bind(chunk, &mut self.scratch, challenge)?;
        }
        Ok(())
    }

    pub(crate) fn chunk_first(&self, chunk: usize) -> Option<Result<Fr, CudaError>> {
        self.chunks.get(chunk).map(DeviceFrVec::first)
    }
}

pub(crate) enum CudaInstructionRafCycleSparse {
    Sparse {
        round: u32,
        tables: crate::cuda::CudaSlice<u64>,
        values: crate::cuda::CudaSlice<u16>,
        combined: DeviceFrVec,
        combined_scratch: DeviceFrVec,
        num_chunks: usize,
        chunk_domain: usize,
        source_rows: usize,
    },
    Dense(CudaInstructionRafCycleState),
}

impl CudaInstructionRafCycleSparse {
    pub(crate) fn from_round1<F: jolt_field::Field>(
        tables: &[Vec<F>],
        values: &[u16],
        combined: &[F],
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let num_chunks = tables.len();
        if num_chunks != 8 {
            return None;
        }
        let chunk_domain = tables.first().map_or(0, Vec::len);
        if chunk_domain == 0 || tables.iter().any(|t| t.len() != chunk_domain) {
            return None;
        }
        let source_rows = combined.len();
        if source_rows == 0 || values.len() != num_chunks * source_rows {
            return None;
        }

        let mut flat_tables: Vec<u64> = Vec::with_capacity(num_chunks * chunk_domain * 4);
        for table in tables {
            for v in crate::cuda::as_fr_slice(table)? {
                flat_tables.extend_from_slice(&v.inner_limbs().0);
            }
        }

        Some(Self::Sparse {
            round: 1,
            tables: ctx.upload_u64_slice(&flat_tables).ok()?,
            values: ctx.upload_u16_slice(values).ok()?,
            combined: ctx.upload(crate::cuda::as_fr_slice(combined)?).ok()?,
            combined_scratch: ctx.upload(&[]).ok()?,
            num_chunks,
            chunk_domain,
            source_rows,
        })
    }

    pub(crate) fn round_poly_evals(
        &self,
        e_in: &DeviceFrVec,
        e_out: &DeviceFrVec,
    ) -> Option<[Fr; 9]> {
        match self {
            Self::Sparse {
                round,
                tables,
                values,
                combined,
                num_chunks,
                chunk_domain,
                source_rows,
                ..
            } => {
                let ctx = crate::cuda::shared_ctx()?;
                ctx.instruction_raf_cycle_sparse_round_poly(InstructionRafCycleSparseInputs {
                    tables,
                    values,
                    combined,
                    num_chunks: *num_chunks,
                    chunk_domain: *chunk_domain,
                    source_rows: *source_rows,
                    e_in,
                    e_out,
                    round: *round,
                })
                .ok()
            }
            Self::Dense(dense) => dense.round_poly_evals(e_in, e_out),
        }
    }

    pub(crate) fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        match self {
            Self::Sparse {
                round,
                tables,
                values,
                combined,
                combined_scratch,
                num_chunks,
                chunk_domain,
                source_rows,
            } => {
                let num_sets = 1usize << (*round - 1);
                let set_elems = *num_chunks * *chunk_domain;
                let bound = ctx.ra_virtual_d4_sparse_bind(tables, num_sets, set_elems, challenge)?;
                ctx.bind(combined, combined_scratch, challenge)?;
                if *round < 3 {
                    *tables = bound;
                    *round += 1;
                    Ok(())
                } else {
                    let out_len = *source_rows >> 3;
                    let chunks = ctx.instruction_raf_cycle_sparse_collapse(
                        &bound,
                        values,
                        *num_chunks,
                        *chunk_domain,
                        *source_rows,
                        out_len,
                    )?;
                    let combined = std::mem::replace(combined, ctx.upload(&[])?);
                    let dense = CudaInstructionRafCycleState::from_device(combined, chunks)
                        .ok_or(CudaError::Pool)?;
                    *self = Self::Dense(dense);
                    Ok(())
                }
            }
            Self::Dense(dense) => dense.bind(challenge),
        }
    }

    pub(crate) fn chunk_first(&self, chunk: usize) -> Option<Result<Fr, CudaError>> {
        match self {
            Self::Sparse { .. } => None,
            Self::Dense(dense) => dense.chunk_first(chunk),
        }
    }
}
