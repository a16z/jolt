use jolt_field::Field;
use jolt_poly::EqPolynomial;

use crate::dense::bind_dense_evals_reuse;

#[derive(Clone)]
pub struct SplitEqState<F: Field> {
    low_point: Vec<F>,
    high_point: Vec<F>,
    e_in: Vec<F>,
    e_out: Vec<F>,
    e_in_scratch: Vec<F>,
    e_out_scratch: Vec<F>,
}

impl<F: Field> SplitEqState<F> {
    #[inline]
    pub fn new_low_to_high(point: &[F], scaling: Option<F>) -> Self {
        let (high_point, low_point) = point.split_at(point.len() / 2);
        Self {
            low_point: low_point.to_vec(),
            high_point: high_point.to_vec(),
            e_in: EqPolynomial::<F>::evals(low_point, scaling),
            e_out: EqPolynomial::<F>::evals(high_point, None),
            e_in_scratch: Vec::new(),
            e_out_scratch: Vec::new(),
        }
    }

    #[inline]
    pub fn e_in(&self) -> &[F] {
        &self.e_in
    }

    #[inline]
    pub fn e_out(&self) -> &[F] {
        &self.e_out
    }

    #[inline]
    pub fn current_target(&self) -> F {
        debug_assert!(self.e_in.len() > 1 || self.e_out.len() > 1);
        if self.e_in.len() > 1 {
            let remaining = self.e_in.len().trailing_zeros() as usize;
            self.low_point[remaining - 1]
        } else {
            let remaining = self.e_out.len().trailing_zeros() as usize;
            self.high_point[remaining - 1]
        }
    }

    #[inline]
    pub fn eval(&self) -> F {
        self.e_in[0] * self.e_out[0]
    }

    #[inline]
    pub fn bind(&mut self, challenge: F) {
        if self.e_in.len() > 1 {
            bind_dense_evals_reuse(&mut self.e_in, &mut self.e_in_scratch, challenge);
        } else {
            bind_dense_evals_reuse(&mut self.e_out, &mut self.e_out_scratch, challenge);
        }
    }
}

#[cfg(feature = "cuda")]
use crate::cuda::{CudaError, CudaKernelContext, DeviceFrVec};
#[cfg(feature = "cuda")]
use jolt_field::Fr;

#[cfg(feature = "cuda")]
#[expect(dead_code)]
pub struct CudaSplitEqState<'a> {
    ctx: &'a CudaKernelContext,
    low_point: Vec<Fr>,
    high_point: Vec<Fr>,
    e_in: DeviceFrVec,
    e_out: DeviceFrVec,
    e_in_scratch: DeviceFrVec,
    e_out_scratch: DeviceFrVec,
}

#[cfg(feature = "cuda")]
#[expect(clippy::todo, unused_variables)]
impl<'a> CudaSplitEqState<'a> {
    pub fn new_low_to_high(
        ctx: &'a CudaKernelContext,
        point: &[Fr],
        scaling: Option<Fr>,
    ) -> Result<Self, CudaError> {
        todo!()
    }

    pub fn e_in(&self) -> Result<Vec<Fr>, CudaError> {
        todo!()
    }

    pub fn e_out(&self) -> Result<Vec<Fr>, CudaError> {
        todo!()
    }

    pub fn current_target(&self) -> Fr {
        todo!()
    }

    pub fn eval(&self) -> Result<Fr, CudaError> {
        todo!()
    }

    pub fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        todo!()
    }
}

#[cfg(all(test, feature = "cuda"))]
#[expect(clippy::unwrap_used)]
mod cuda_tests {
    use super::*;
    use proptest::prelude::*;

    fn fr_strategy() -> impl Strategy<Value = Fr> {
        any::<[u8; 32]>().prop_map(|bytes| Fr::from_bytes(&bytes))
    }

    proptest! {
        #[test]
        #[ignore = "CudaSplitEqState methods are todo!()"]
        fn cuda_split_eq_matches_cpu(
            num_vars in 1usize..10,
            point_seed in fr_strategy(),
            scaling in proptest::option::of(fr_strategy()),
            challenge_seed in fr_strategy(),
        ) {
            let point: Vec<Fr> = (0..num_vars)
                .map(|i| point_seed + Fr::from_u64(i as u64))
                .collect();

            let ctx = CudaKernelContext::new(0).unwrap();
            let mut cpu = SplitEqState::<Fr>::new_low_to_high(&point, scaling);
            let mut gpu = CudaSplitEqState::new_low_to_high(&ctx, &point, scaling).unwrap();

            for round in 0..num_vars {
                prop_assert_eq!(gpu.e_in().unwrap(), cpu.e_in().to_vec());
                prop_assert_eq!(gpu.e_out().unwrap(), cpu.e_out().to_vec());
                prop_assert_eq!(gpu.current_target(), cpu.current_target());
                prop_assert_eq!(gpu.eval().unwrap(), cpu.eval());

                let challenge = challenge_seed + Fr::from_u64(round as u64);
                cpu.bind(challenge);
                gpu.bind(challenge).unwrap();
            }

            prop_assert_eq!(gpu.eval().unwrap(), cpu.eval());
        }
    }
}
