use jolt_field::Fr;
use jolt_poly::UnivariatePoly;

use crate::cuda::{CudaError, CudaKernelContext, DeviceFrVec};
use crate::dense::bind_dense_evals_reuse_cuda;

pub struct CudaDenseOuterState<'a> {
    ctx: &'a CudaKernelContext,
    eq: DeviceFrVec,
    az: DeviceFrVec,
    bz: DeviceFrVec,
    eq_scratch: DeviceFrVec,
    az_scratch: DeviceFrVec,
    bz_scratch: DeviceFrVec,
}

impl<'a> CudaDenseOuterState<'a> {
    pub fn from_host(
        ctx: &'a CudaKernelContext,
        eq: &[Fr],
        az: &[Fr],
        bz: &[Fr],
    ) -> Result<Self, CudaError> {
        Ok(Self {
            ctx,
            eq: ctx.upload(eq)?,
            az: ctx.upload(az)?,
            bz: ctx.upload(bz)?,
            eq_scratch: ctx.upload(&[])?,
            az_scratch: ctx.upload(&[])?,
            bz_scratch: ctx.upload(&[])?,
        })
    }

    pub fn round_poly(&self) -> Result<UnivariatePoly<Fr>, CudaError> {
        let coeffs = self.ctx.cubic_accumulate(&self.eq, &self.az, &self.bz)?;
        Ok(UnivariatePoly::new(coeffs.to_vec()))
    }

    pub fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        bind_dense_evals_reuse_cuda(self.ctx, &mut self.eq, &mut self.eq_scratch, challenge)?;
        bind_dense_evals_reuse_cuda(self.ctx, &mut self.az, &mut self.az_scratch, challenge)?;
        bind_dense_evals_reuse_cuda(self.ctx, &mut self.bz, &mut self.bz_scratch, challenge)?;
        Ok(())
    }

    pub fn eq(&self) -> Result<Vec<Fr>, CudaError> {
        self.eq.to_host()
    }

    pub fn az(&self) -> Result<Vec<Fr>, CudaError> {
        self.az.to_host()
    }

    pub fn bz(&self) -> Result<Vec<Fr>, CudaError> {
        self.bz.to_host()
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stage1::DenseOuterState;
    use jolt_field::Field;
    use proptest::prelude::*;

    fn fr_strategy() -> impl Strategy<Value = Fr> {
        any::<[u8; 32]>().prop_map(|bytes| Fr::from_bytes(&bytes))
    }

    fn triple_strategy(max_vars: usize) -> impl Strategy<Value = (Vec<Fr>, Vec<Fr>, Vec<Fr>)> {
        (1usize..max_vars).prop_flat_map(|num_vars| {
            let len = 1usize << num_vars;
            (
                prop::collection::vec(fr_strategy(), len),
                prop::collection::vec(fr_strategy(), len),
                prop::collection::vec(fr_strategy(), len),
            )
        })
    }

    proptest! {
        #[test]
        fn cuda_dense_outer_matches_cpu((eq, az, bz) in triple_strategy(10)) {
            let ctx = CudaKernelContext::new(0).unwrap();
            let mut cpu = DenseOuterState::from_raw(eq.clone(), az.clone(), bz.clone());
            let mut gpu = CudaDenseOuterState::from_host(&ctx, &eq, &az, &bz).unwrap();

            for round in 0..eq.len().trailing_zeros() {
                let gpu_poly = gpu.round_poly().unwrap();
                let cpu_poly = cpu.round_poly();
                prop_assert_eq!(gpu_poly.coefficients(), cpu_poly.coefficients());
                prop_assert_eq!(gpu.eq().unwrap(), cpu.eq().to_vec());
                prop_assert_eq!(gpu.az().unwrap(), cpu.az().to_vec());
                prop_assert_eq!(gpu.bz().unwrap(), cpu.bz().to_vec());

                let challenge = Fr::from_u64((round + 1) as u64);
                cpu.bind(challenge);
                gpu.bind(challenge).unwrap();
            }

            prop_assert_eq!(gpu.eq().unwrap(), cpu.eq().to_vec());
        }
    }
}
