use ark_bn254::{G1Projective}; 
use ark_ec::{CurveGroup, ScalarMul};
use ark_ff::{BigInteger, Field, PrimeField};
use icicle_bn254::curve::CurveCfg as IcicleBn254;
use icicle_bn254::curve::G1Projective as GPUG1;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    msm::{msm, MSMConfig, MSM},
    traits::{ArkConvertible, FieldImpl},
};
use icicle_cuda_runtime::{
    memory::{DeviceVec, HostSlice},
    stream::CudaStream,
};
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelRefIterator;

use crate::msm::VariableBaseMSM;

use once_cell::sync::OnceCell;

pub static ICICLE_BASES: OnceCell<Vec<GPUG1>> = OnceCell::new();

impl Icicle for G1Projective {
    type C = IcicleBn254;

    fn to_ark_projective(point: &Projective<Self::C>) -> Self {
        let proj_x =
            <Self as CurveGroup>::BaseField::from_random_bytes(&point.x.to_bytes_le()).unwrap();
        let proj_y =
            <Self as CurveGroup>::BaseField::from_random_bytes(&point.y.to_bytes_le()).unwrap();
        let proj_z =
            <Self as CurveGroup>::BaseField::from_random_bytes(&point.z.to_bytes_le()).unwrap();

        let proj_x = proj_x * proj_z;
        let proj_y = proj_y * proj_z * proj_z;
        Self::new_unchecked(proj_x, proj_y, proj_z)
    }

    fn from_ark_affine(point: &Self::MulBase) -> Affine<Self::C> {
        let x_bytes: Vec<u8> = point
            .x
            .to_base_prime_field_elements()
            .flat_map(|x| x.into_bigint().to_bytes_le())
            .collect();
        let y_bytes: Vec<u8> = point
            .y
            .to_base_prime_field_elements()
            .flat_map(|x| x.into_bigint().to_bytes_le())
            .collect();
        let x = <Self::C as Curve>::BaseField::from_bytes_le(&x_bytes);
        let y = <Self::C as Curve>::BaseField::from_bytes_le(&y_bytes);
        Affine::<Self::C> { x, y }
    }
}

pub trait Icicle: ScalarMul {
    type C: Curve<ScalarField: ArkConvertible<ArkEquivalent = Self::ScalarField>> + MSM<Self::C>;

    // Note: To prevent excessive trait the arkworks conversion functions within icicle are reimplemented
    fn to_ark_projective(point: &Projective<Self::C>) -> Self;

    fn from_ark_affine(point: &Self::MulBase) -> Affine<Self::C>;
}

#[tracing::instrument(skip_all, name = "icicle_msm")]
pub fn icicle_msm<V: VariableBaseMSM + Icicle>(
    bases: &[Affine<V::C>],
    scalars: &[V::ScalarField],
) -> V {
    // let span = tracing::span!(tracing::Level::INFO, "convert_bases");
    // let _guard = span.enter();

    // let bases = bases.par_iter().map(|base| V::from_ark_affine(base)).collect::<Vec<_>>();
    let mut bases_slice = DeviceVec::<Affine<V::C>>::cuda_malloc(bases.len()).unwrap();
    // drop(_guard);
    // drop(span);

    let span = tracing::span!(tracing::Level::INFO, "convert_scalars");
    let _guard = span.enter();

    let mut scalars_slice = DeviceVec::<<<V as Icicle>::C as Curve>::ScalarField>::cuda_malloc(scalars.len()).unwrap();
    let scalars_mont = unsafe { &*(&scalars[..] as *const _ as *const [<<V as Icicle>::C as Curve>::ScalarField]) };

    drop(_guard);
    drop(span);

    let stream = CudaStream::create().unwrap();
    bases_slice.copy_from_host_async(HostSlice::from_slice(&bases), &stream).unwrap();
    scalars_slice.copy_from_host_async(HostSlice::from_slice(&scalars_mont), &stream).unwrap();
    let mut msm_result = DeviceVec::<Projective<V::C>>::cuda_malloc(1).unwrap();
    let mut cfg = MSMConfig::default();
    cfg.ctx.stream = &stream;
    cfg.is_async = false;
    cfg.are_scalars_montgomery_form = true;

    let span = tracing::span!(tracing::Level::INFO, "msm");
    let _guard = span.enter();

    msm(&scalars_slice[..], &bases_slice[..], &cfg, &mut msm_result[..]).unwrap();

    drop(_guard);
    drop(span);
    let mut msm_host_result = [Projective::<V::C>::zero(); 1];

    let span = tracing::span!(tracing::Level::INFO, "copy_msm_result");
    let _guard = span.enter();

    msm_result
        .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
        .unwrap();

    drop(_guard);
    drop(span);

    stream.synchronize().unwrap();
    stream.destroy().unwrap();
    V::to_ark_projective(&msm_host_result[0])
}

#[tracing::instrument(skip_all, name = "icicle_batch_msm")]
pub fn icicle_batch_msm<V: VariableBaseMSM + Icicle>(
    bases: &[Affine<V::C>],
    scalar_batches: &[&[V::ScalarField]],
) -> Vec<V> {
    let len = bases.len();
    let batch_size = scalar_batches.len();
    assert!(scalar_batches.iter().all(|s| s.len() == len));

    let mut bases_slice = DeviceVec::<Affine<V::C>>::cuda_malloc(bases.len()).unwrap();

    let span = tracing::span!(tracing::Level::INFO, "convert_scalars");
    let _guard = span.enter();

    // Scalar slices are non-contiguous in host memory, but need to be contiguous in device memory
    let mut scalars_alloc= DeviceVec::<<<V as Icicle>::C as Curve>::ScalarField>::cuda_malloc(len * batch_size).unwrap();
    let stream = CudaStream::create().unwrap();
    for (batch_i, scalars) in scalar_batches.iter().enumerate() {
        let scalars_mont = unsafe { &*(&scalars[..] as *const _ as *const [<<V as Icicle>::C as Curve>::ScalarField]) };
        scalars_alloc[batch_i*len .. (batch_i + 1)*len].copy_from_host_async(HostSlice::from_slice(&scalars_mont), &stream).unwrap();
    }

    drop(_guard);
    drop(span);

    bases_slice.copy_from_host_async(HostSlice::from_slice(&bases), &stream).unwrap();
    let mut msm_result = DeviceVec::<Projective<V::C>>::cuda_malloc(batch_size).unwrap();
    let mut cfg = MSMConfig::default();
    cfg.ctx.stream = &stream;
    cfg.is_async = false;
    cfg.are_scalars_montgomery_form = true;
    // cfg.batch_size = batch_size; // TODO(sragsss): Should happen automatically?

    let span = tracing::span!(tracing::Level::INFO, "msm");
    let _guard = span.enter();

    msm(&scalars_alloc[..], &bases_slice[..], &cfg, &mut msm_result[..]).unwrap();

    drop(_guard);
    drop(span);
    let mut msm_host_result = vec![Projective::<V::C>::zero(); batch_size];

    let span = tracing::span!(tracing::Level::INFO, "copy_msm_result");
    let _guard = span.enter();

    msm_result
        .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
        .unwrap();

    drop(_guard);
    drop(span);

    stream.synchronize().unwrap();
    stream.destroy().unwrap();
    msm_host_result.into_iter().map(|gpu_form| V::to_ark_projective(&gpu_form)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::{Fr, G1Affine, G1Projective};
    use icicle_bn254::curve::{ScalarField as GPUScalar, G1Projective as GPUG1, CurveCfg as GPUCurve};
    use ark_ec::VariableBaseMSM as ark_VariableBaseMSM;
    use ark_std::UniformRand;
    use rand_core::SeedableRng;
    use icicle_core::traits::ArkConvertible;

    // Note due to contention for the gpu device testing using multiple threads leads to unit tests that intermittently fail.
    // To avoid this run `cargo t --features icicle -- --test-threads=1`
    #[test]
    fn icicle_msm_consistency() {
        for pow in 1..10 {
            let n = 1 << pow;
            let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(n as u64);
            for _ in 0..100 {
                let scalars: Vec<Fr> = std::iter::repeat_with(|| Fr::rand(&mut rng))
                    .take(n)
                    .collect();
                let bases: Vec<G1Affine> = std::iter::repeat_with(|| G1Affine::rand(&mut rng))
                    .take(n)
                    .collect();

                let gpu_bases = bases.par_iter().map(|base| <G1Projective as Icicle>::from_ark_affine(base)).collect::<Vec<_>>();
                let icicle_res = icicle_msm::<G1Projective>(&gpu_bases, &scalars);
                let arkworks_res: G1Projective =
                    ark_VariableBaseMSM::msm(&bases, &scalars).unwrap();
                let msm_res: G1Projective = VariableBaseMSM::msm(&bases, &scalars).unwrap();
                assert_eq!(icicle_res, arkworks_res);
                assert_eq!(icicle_res, msm_res);
            }
        }
    }

    #[test]
    fn casting() {
        let ark = Fr::from(100);
        let gpu = GPUScalar::from_ark(ark);

        // TODO(sragss): This doesn't work because GPUScalar::from_ark converts back to the integer repr.
        let ark_bytes: [u8; 32] = unsafe { std::mem::transmute(ark) };
        let gpu_bytes: [u8; 32] = unsafe { std::mem::transmute(gpu) };
        assert_eq!(ark_bytes, gpu_bytes);
    }
}
