use jolt_field::Fr;

use crate::cuda::{CudaError, DeviceFrVec, GruenRoundPolyInputs};
use crate::split_eq::CudaSplitEqState;

pub(crate) enum CudaGruenKind {
    InstructionInput { gamma: Fr },
    Registers { gamma: Fr, gamma2: Fr },
}

pub(crate) struct CudaSumOfProductsState {
    factors: Vec<DeviceFrVec>,
    scratch: Vec<DeviceFrVec>,
    split_eq: CudaSplitEqState<'static>,
    kind: CudaGruenKind,
}

impl CudaSumOfProductsState {
    pub(crate) fn new(
        kind: CudaGruenKind,
        factors: &[Vec<Fr>],
        split_point: &[Fr],
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let refs: Vec<&[Fr]> = factors.iter().map(Vec::as_slice).collect();
        let device_factors = ctx.upload_many(&refs).ok()?;
        let scratch = (0..factors.len())
            .map(|_| ctx.upload(&[]).ok())
            .collect::<Option<Vec<DeviceFrVec>>>()?;
        let split_eq = CudaSplitEqState::new_low_to_high(ctx, split_point, None).ok()?;
        Some(Self {
            factors: device_factors,
            scratch,
            split_eq,
            kind,
        })
    }

    pub(crate) fn current_target(&self) -> Fr {
        self.split_eq.current_target()
    }

    fn term_table(&self) -> (Vec<Fr>, Vec<u32>, Vec<u32>, usize) {
        match self.kind {
            CudaGruenKind::InstructionInput { gamma } => {
                let coeffs = vec![Fr::from(1u64), Fr::from(1u64), gamma, gamma];
                let offsets = vec![0u32, 2, 4, 6, 8];
                let indices = vec![0u32, 1, 2, 3, 4, 5, 6, 7];
                (coeffs, offsets, indices, 2)
            }
            CudaGruenKind::Registers { gamma, gamma2 } => {
                let coeffs = vec![Fr::from(1u64), gamma, gamma2];
                let offsets = vec![0u32, 1, 2, 3];
                let indices = vec![0u32, 1, 2];
                (coeffs, offsets, indices, 1)
            }
        }
    }

    pub(crate) fn q_coefficients(&self) -> Result<(Fr, Fr), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        let (term_coeffs, term_factor_offsets, term_factor_indices, degree) = self.term_table();
        let factor_refs: Vec<&DeviceFrVec> = self.factors.iter().collect();
        let q = ctx.gruen_round_poly(GruenRoundPolyInputs {
            factors: &factor_refs,
            term_coeffs: &term_coeffs,
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
