use jolt_field::Fr;
use jolt_poly::UnivariatePoly;

use crate::cuda::{
    CudaError, DeviceFrVec, InstructionRafCycleInputs, InstructionRafCycleSparseInputs,
    RafQScatterInputs, ReadSuffixScatterInputs, RoundPolyTerms,
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
        let mut out: Vec<Vec<Fr>> = Vec::with_capacity(5);
        for bank in &banks {
            out.push(bank.to_host()?);
        }
        out.try_into().map_err(|_| CudaError::Pool)
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
