use jolt_field::Fr;
use jolt_poly::UnivariatePoly;

use crate::cuda::{CudaError, DeviceFrVec, GruenRoundPolyInputs, RoundPolyTerms};
use crate::split_eq::CudaSplitEqState;

#[derive(Clone, Copy)]
pub(crate) enum CudaGruenKind {
    InstructionInput { gamma: Fr },
    Registers { gamma: Fr, gamma2: Fr },
    Product,
}

fn term_coeffs_for(kind: CudaGruenKind) -> Vec<Fr> {
    match kind {
        CudaGruenKind::InstructionInput { gamma } => {
            vec![Fr::from(1u64), Fr::from(1u64), gamma, gamma]
        }
        CudaGruenKind::Registers { gamma, gamma2 } => vec![Fr::from(1u64), gamma, gamma2],
        CudaGruenKind::Product => vec![Fr::from(1u64)],
    }
}

pub(crate) struct CudaSumOfProductsState {
    factors: Vec<DeviceFrVec>,
    scratch: Vec<DeviceFrVec>,
    split_eq: CudaSplitEqState<'static>,
    kind: CudaGruenKind,
    term_coeffs: DeviceFrVec,
}

impl CudaSumOfProductsState {
    pub(crate) fn new(
        kind: CudaGruenKind,
        factors: &[&[Fr]],
        split_point: &[Fr],
        split_scaling: Option<Fr>,
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let device_factors = ctx.upload_many(factors).ok()?;
        let scratch = (0..factors.len())
            .map(|_| ctx.upload(&[]).ok())
            .collect::<Option<Vec<DeviceFrVec>>>()?;
        let split_eq = CudaSplitEqState::new_low_to_high(ctx, split_point, split_scaling).ok()?;
        let term_coeffs = ctx.upload(&term_coeffs_for(kind)).ok()?;
        Some(Self {
            factors: device_factors,
            scratch,
            split_eq,
            kind,
            term_coeffs,
        })
    }

    pub(crate) fn current_target(&self) -> Fr {
        self.split_eq.current_target()
    }

    fn term_layout(&self) -> (Vec<u32>, Vec<u32>, usize) {
        match self.kind {
            CudaGruenKind::InstructionInput { .. } => {
                (vec![0u32, 2, 4, 6, 8], vec![0u32, 1, 2, 3, 4, 5, 6, 7], 2)
            }
            CudaGruenKind::Registers { .. } => (vec![0u32, 1, 2, 3], vec![0u32, 1, 2], 1),
            CudaGruenKind::Product => (vec![0u32, 2], vec![0u32, 1], 2),
        }
    }

    pub(crate) fn q_coefficients(&self) -> Result<(Fr, Fr), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        let (term_factor_offsets, term_factor_indices, degree) = self.term_layout();
        let factor_refs: Vec<&DeviceFrVec> = self.factors.iter().collect();
        let q = ctx.gruen_round_poly(GruenRoundPolyInputs {
            factors: &factor_refs,
            term_coeffs: &self.term_coeffs,
            term_factor_offsets: &term_factor_offsets,
            term_factor_indices: &term_factor_indices,
            e_in: self.split_eq.e_in_device(),
            e_out: self.split_eq.e_out_device(),
            degree,
        })?;
        Ok((q[0], q[1]))
    }

    pub(crate) fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        for (factor, scratch) in self.factors.iter_mut().zip(&mut self.scratch) {
            ctx.bind(factor, scratch, challenge)?;
        }
        self.split_eq.bind(challenge)
    }

    pub(crate) fn factor_eval(&self, index: usize) -> Result<Fr, CudaError> {
        self.factors[index].first()
    }
}

pub(crate) struct CudaSpartanShiftState {
    factors: Vec<DeviceFrVec>,
    scratch: Vec<DeviceFrVec>,
    term_coeffs: DeviceFrVec,
    term_factor_offsets: Vec<u32>,
    term_factor_indices: Vec<u32>,
    round_factors: usize,
    degree: usize,
}

impl CudaSpartanShiftState {
    pub(crate) fn new(
        factors: &[&[Fr]],
        term_coeffs: &[Fr],
        term_factor_offsets: Vec<u32>,
        term_factor_indices: Vec<u32>,
        round_factors: usize,
        degree: usize,
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let device_factors = ctx.upload_many(factors).ok()?;
        let scratch = (0..factors.len())
            .map(|_| ctx.upload(&[]).ok())
            .collect::<Option<Vec<DeviceFrVec>>>()?;
        let term_coeffs = ctx.upload(term_coeffs).ok()?;
        Some(Self {
            factors: device_factors,
            scratch,
            term_coeffs,
            term_factor_offsets,
            term_factor_indices,
            round_factors,
            degree,
        })
    }

    pub(crate) fn round_poly(&self) -> Result<UnivariatePoly<Fr>, CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        let factor_refs: Vec<&DeviceFrVec> = self.factors[..self.round_factors].iter().collect();
        let coeffs = ctx.dense_product_round_poly(RoundPolyTerms {
            factors: &factor_refs,
            term_coeffs: &self.term_coeffs,
            term_factor_offsets: &self.term_factor_offsets,
            term_factor_indices: &self.term_factor_indices,
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

    #[expect(dead_code, reason = "used by SpartanShift Phase2 finals (next step)")]
    pub(crate) fn factor_eval(&self, index: usize) -> Result<Fr, CudaError> {
        self.factors[index].first()
    }
}
