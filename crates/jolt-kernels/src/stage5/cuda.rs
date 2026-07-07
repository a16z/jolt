use jolt_field::Fr;
use jolt_poly::UnivariatePoly;

use crate::cuda::{
    CudaError, DeviceFrVec, InstructionRafCycleInputs, InstructionRafCycleSparseInputs,
    RoundPolyTerms,
};

pub(crate) struct CudaDenseState {
    factors: Vec<DeviceFrVec>,
    scratch: Vec<DeviceFrVec>,
    term_coeffs: Vec<Fr>,
    term_factor_offsets: Vec<u32>,
    term_factor_indices: Vec<u32>,
    degree: usize,
    active_scale: Fr,
}

impl CudaDenseState {
    pub(crate) fn new(
        factors: &[Vec<Fr>],
        term_coeffs: Vec<Fr>,
        term_factor_offsets: Vec<u32>,
        term_factor_indices: Vec<u32>,
        degree: usize,
        active_scale: Fr,
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let mut device_factors = Vec::with_capacity(factors.len());
        let mut scratch = Vec::with_capacity(factors.len());
        for factor in factors {
            device_factors.push(ctx.upload(factor).ok()?);
            scratch.push(ctx.upload(&[]).ok()?);
        }
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
        let device_chunks = chunks
            .iter()
            .map(|chunk| ctx.upload(crate::cuda::as_fr_slice(chunk)?).ok())
            .collect::<Option<Vec<DeviceFrVec>>>()?;
        Some(Self {
            combined: ctx.upload(crate::cuda::as_fr_slice(combined)?).ok()?,
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

    pub(crate) fn round_poly_evals<F: jolt_field::Field>(
        &self,
        e_in: &[F],
        e_out: &[F],
    ) -> Option<[Fr; 9]> {
        let ctx = crate::cuda::shared_ctx()?;
        let e_in_dev = ctx.upload(crate::cuda::as_fr_slice(e_in)?).ok()?;
        let e_out_dev = ctx.upload(crate::cuda::as_fr_slice(e_out)?).ok()?;
        let chunk_refs: Vec<&DeviceFrVec> = self.chunks.iter().collect();
        ctx.instruction_raf_cycle_round_poly(InstructionRafCycleInputs {
            combined: &self.combined,
            chunks: &chunk_refs,
            e_in: &e_in_dev,
            e_out: &e_out_dev,
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

    pub(crate) fn round_poly_evals<F: jolt_field::Field>(
        &self,
        e_in: &[F],
        e_out: &[F],
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
                let e_in_dev = ctx.upload(crate::cuda::as_fr_slice(e_in)?).ok()?;
                let e_out_dev = ctx.upload(crate::cuda::as_fr_slice(e_out)?).ok()?;
                ctx.instruction_raf_cycle_sparse_round_poly(InstructionRafCycleSparseInputs {
                    tables,
                    values,
                    combined,
                    num_chunks: *num_chunks,
                    chunk_domain: *chunk_domain,
                    source_rows: *source_rows,
                    e_in: &e_in_dev,
                    e_out: &e_out_dev,
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
