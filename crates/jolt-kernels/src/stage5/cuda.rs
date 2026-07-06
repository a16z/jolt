use jolt_field::Fr;
use jolt_poly::UnivariatePoly;

use crate::cuda::{CudaError, DeviceFrVec, InstructionRafCycleInputs, RoundPolyTerms};

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
