use jolt_field::Fr;
use jolt_poly::UnivariatePoly;

use crate::cuda::{CudaError, DeviceFrVec, RoundPolyTerms};

pub(crate) struct CudaDenseState {
    factors: Vec<DeviceFrVec>,
    scratch: Vec<DeviceFrVec>,
    degree: usize,
}

impl CudaDenseState {
    pub(crate) fn new(factors: &[Vec<Fr>]) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let degree = factors.len();
        let refs: Vec<&[Fr]> = factors.iter().map(Vec::as_slice).collect();
        let device_factors = ctx.upload_many(&refs).ok()?;
        let scratch = (0..degree)
            .map(|_| ctx.upload(&[]).ok())
            .collect::<Option<Vec<DeviceFrVec>>>()?;
        Some(Self {
            factors: device_factors,
            scratch,
            degree,
        })
    }

    pub(crate) fn round_poly(&self) -> Result<UnivariatePoly<Fr>, CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        let factor_refs: Vec<&DeviceFrVec> = self.factors.iter().collect();
        let single_term_coeffs = [Fr::from(1u64)];
        let single_term_offsets = [0u32, self.degree as u32];
        let single_term_indices: Vec<u32> = (0..self.degree as u32).collect();
        let coeffs = ctx.dense_product_round_poly(RoundPolyTerms {
            factors: &factor_refs,
            term_coeffs: &single_term_coeffs,
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
