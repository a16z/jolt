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
use crate::dense::bind_dense_evals_reuse_cuda;
#[cfg(feature = "cuda")]
use jolt_field::Fr;

#[cfg(feature = "cuda")]
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
impl<'a> CudaSplitEqState<'a> {
    pub fn new_low_to_high(
        ctx: &'a CudaKernelContext,
        point: &[Fr],
        scaling: Option<Fr>,
    ) -> Result<Self, CudaError> {
        let (high_point, low_point) = point.split_at(point.len() / 2);
        Ok(Self {
            ctx,
            low_point: low_point.to_vec(),
            high_point: high_point.to_vec(),
            e_in: ctx.upload(&EqPolynomial::<Fr>::evals(low_point, scaling))?,
            e_out: ctx.upload(&EqPolynomial::<Fr>::evals(high_point, None))?,
            e_in_scratch: ctx.upload(&[])?,
            e_out_scratch: ctx.upload(&[])?,
        })
    }

    pub fn e_in(&self) -> Result<Vec<Fr>, CudaError> {
        self.e_in.to_host()
    }

    pub fn e_out(&self) -> Result<Vec<Fr>, CudaError> {
        self.e_out.to_host()
    }

    pub fn e_in_device(&self) -> &DeviceFrVec {
        &self.e_in
    }

    pub fn e_out_device(&self) -> &DeviceFrVec {
        &self.e_out
    }

    pub fn current_target(&self) -> Fr {
        debug_assert!(self.e_in.len() > 1 || self.e_out.len() > 1);
        if self.e_in.len() > 1 {
            let remaining = self.e_in.len().trailing_zeros() as usize;
            self.low_point[remaining - 1]
        } else {
            let remaining = self.e_out.len().trailing_zeros() as usize;
            self.high_point[remaining - 1]
        }
    }

    pub fn eval(&self) -> Result<Fr, CudaError> {
        Ok(self.e_in.first()? * self.e_out.first()?)
    }

    pub fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        if self.e_in.len() > 1 {
            bind_dense_evals_reuse_cuda(self.ctx, &mut self.e_in, &mut self.e_in_scratch, challenge)
        } else {
            bind_dense_evals_reuse_cuda(
                self.ctx,
                &mut self.e_out,
                &mut self.e_out_scratch,
                challenge,
            )
        }
    }
}

#[cfg(feature = "cuda")]
use jolt_poly::GruenSplitEqPolynomial;

#[cfg(feature = "cuda")]
pub struct CudaGruenSplitEq {
    e_in_levels: Vec<DeviceFrVec>,
    e_out_levels: Vec<DeviceFrVec>,
    live_e_in: usize,
    live_e_out: usize,
}

#[cfg(feature = "cuda")]
impl CudaGruenSplitEq {
    pub fn new<F: Field>(ctx: &CudaKernelContext, host: &GruenSplitEqPolynomial<F>) -> Option<Self> {
        let e_in_refs: Vec<&[Fr]> = host
            .e_in_levels()
            .iter()
            .map(|level| crate::cuda::as_fr_slice(level))
            .collect::<Option<Vec<&[Fr]>>>()?;
        let e_out_refs: Vec<&[Fr]> = host
            .e_out_levels()
            .iter()
            .map(|level| crate::cuda::as_fr_slice(level))
            .collect::<Option<Vec<&[Fr]>>>()?;
        let e_in_levels = ctx.upload_many(&e_in_refs).ok()?;
        let e_out_levels = ctx.upload_many(&e_out_refs).ok()?;
        Some(Self {
            live_e_in: host.e_in_num_levels(),
            live_e_out: host.e_out_num_levels(),
            e_in_levels,
            e_out_levels,
        })
    }

    pub fn e_in_device(&self) -> &DeviceFrVec {
        &self.e_in_levels[self.live_e_in - 1]
    }

    pub fn e_out_device(&self) -> &DeviceFrVec {
        &self.e_out_levels[self.live_e_out - 1]
    }

    pub fn sync_to_host<F: Field>(&mut self, host: &GruenSplitEqPolynomial<F>) {
        self.live_e_in = host.e_in_num_levels();
        self.live_e_out = host.e_out_num_levels();
    }
}

#[cfg(all(test, feature = "cuda"))]
#[expect(clippy::unwrap_used)]
mod cuda_tests {
    use super::*;
    use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};
    use proptest::prelude::*;

    fn fr_strategy() -> impl Strategy<Value = Fr> {
        any::<[u8; 32]>().prop_map(|bytes| Fr::from_bytes(&bytes))
    }

    proptest! {
        #[test]
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

        #[test]
        fn cuda_gruen_split_eq_matches_host_levels(
            num_vars in 1usize..12,
            point_seed in fr_strategy(),
            scaling in proptest::option::of(fr_strategy()),
            challenge_seed in fr_strategy(),
        ) {
            let point: Vec<Fr> = (0..num_vars)
                .map(|i| point_seed + Fr::from_u64(i as u64))
                .collect();

            let ctx = CudaKernelContext::new(0).unwrap();
            let mut host =
                GruenSplitEqPolynomial::<Fr>::new_with_scaling(&point, BindingOrder::LowToHigh, scaling);
            let mut gpu = CudaGruenSplitEq::new(&ctx, &host).unwrap();

            for round in 0..num_vars {
                prop_assert_eq!(
                    gpu.e_in_device().to_host().unwrap(),
                    host.e_in_current().to_vec(),
                    "round {}", round
                );
                prop_assert_eq!(
                    gpu.e_out_device().to_host().unwrap(),
                    host.e_out_current().to_vec(),
                    "round {}", round
                );

                let challenge = challenge_seed + Fr::from_u64(round as u64);
                host.bind(challenge);
                gpu.sync_to_host(&host);
            }
        }
    }
}
