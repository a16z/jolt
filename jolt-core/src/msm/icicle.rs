use ark_bn254::G1Projective;
use ark_ec::{CurveGroup, ScalarMul};
use ark_ff::{BigInteger, Field, PrimeField};
use icicle_bn254::curve::CurveCfg as IcicleBn254;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    msm::{msm, MSMConfig, MSM},
    traits::{ArkConvertible, FieldImpl},
};
use icicle_cuda_runtime::{
    memory::{DeviceVec, HostSlice},
    stream::CudaStream,
};

use crate::msm::VariableBaseMSM;

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

    //Note: To prevent excessive trait the arkworks conversion functions within icicle are reimplemented
    fn to_ark_projective(point: &Projective<Self::C>) -> Self;

    fn from_ark_affine(point: &Self::MulBase) -> Affine<Self::C>;
}

#[tracing::instrument(skip_all, name = "icicle_msm")]
pub fn icicle_msm<V: VariableBaseMSM + Icicle>(
    bases: &[V::MulBase],
    scalars: &[V::ScalarField],
) -> V {
    let bases = bases
        .iter()
        .map(|base| V::from_ark_affine(base))
        .collect::<Vec<_>>();
    let bases_slice = HostSlice::from_slice(&bases);
    let scalars = &scalars
        .iter()
        .map(|scalar| <<V as Icicle>::C as Curve>::ScalarField::from_ark(*scalar))
        .collect::<Vec<_>>();
    let scalars_slice = HostSlice::from_slice(scalars);
    let mut msm_result = DeviceVec::<Projective<V::C>>::cuda_malloc(1).unwrap();
    let stream = CudaStream::create().unwrap();
    let mut cfg = MSMConfig::default();
    cfg.ctx.stream = &stream;
    cfg.is_async = true;
    msm(scalars_slice, bases_slice, &cfg, &mut msm_result[..]).unwrap();
    let mut msm_host_result = [Projective::<V::C>::zero(); 1];
    msm_result
        .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
        .unwrap();
    stream.synchronize().unwrap();
    stream.destroy().unwrap();
    V::to_ark_projective(&msm_host_result[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::{Fr, G1Affine, G1Projective};
    use ark_ec::VariableBaseMSM as ark_VariableBaseMSM;
    use ark_std::UniformRand;
    use rand_core::SeedableRng;

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

                let icicle_res = icicle_msm::<G1Projective>(&bases, &scalars);
                let arkworks_res: G1Projective =
                    ark_VariableBaseMSM::msm(&bases, &scalars).unwrap();
                let msm_res: G1Projective = VariableBaseMSM::msm(&bases, &scalars).unwrap();
                assert_eq!(icicle_res, arkworks_res);
                assert_eq!(icicle_res, msm_res);
            }
        }
    }
}
