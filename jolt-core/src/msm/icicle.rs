use ark_bn254::G1Projective;
use ark_ec::{CurveGroup, ScalarMul};
use ark_ff::{BigInteger, Field, PrimeField};
use icicle_bn254::curve::CurveCfg;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    msm::{msm, MSMConfig, MSM},
    traits::FieldImpl,
};
use icicle_cuda_runtime::{
    memory::{DeviceVec, HostSlice},
    stream::CudaStream,
};

use crate::msm::VariableBaseMSM;

impl Icicle for G1Projective {
    type IC = CurveCfg;

    //To avoid adding a convoluted trait bound to every CurveGroup in jolt-core we reimplement the conversion methods in icicle.
    fn from_ark_base_field(
        field: &<Self as CurveGroup>::BaseField,
    ) -> <Self::IC as Curve>::BaseField
    where
        Self: CurveGroup,
    {
        let ark_bytes: Vec<u8> = field
            .to_base_prime_field_elements()
            .map(|x| x.into_bigint().to_bytes_le())
            .flatten()
            .collect();
        <Self::IC as Curve>::BaseField::from_bytes_le(&ark_bytes)
    }

    //To avoid adding a convoluted trait bound to every CurveGroup in jolt-core we reimplement the conversion methods in icicle.
    fn from_ark_scalar(scalar: &Self::ScalarField) -> <Self::IC as Curve>::ScalarField {
        let ark_bytes: Vec<u8> = scalar
            .to_base_prime_field_elements()
            .map(|x| x.into_bigint().to_bytes_le())
            .flatten()
            .collect();
        <Self::IC as Curve>::ScalarField::from_bytes_le(&ark_bytes)
    }

    fn to_ark_projective(point: &Projective<Self::IC>) -> Self {
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

    fn from_ark_affine(point: &Self::MulBase) -> Affine<Self::IC> {
        Affine::<Self::IC> {
            x: Self::from_ark_base_field(&point.x),
            y: Self::from_ark_base_field(&point.y),
        }
    }
}

pub trait Icicle: ScalarMul {
    type IC: Curve + MSM<Self::IC>;

    //Note: To prevent excessive trait the arkworks conversion functions within icicle are reimplemented
    fn from_ark_base_field(
        field: &<Self as CurveGroup>::BaseField,
    ) -> <Self::IC as Curve>::BaseField
    where
        Self: CurveGroup;

    fn from_ark_scalar(scalar: &Self::ScalarField) -> <Self::IC as Curve>::ScalarField;

    fn to_ark_projective(point: &Projective<Self::IC>) -> Self;

    fn from_ark_affine(point: &Self::MulBase) -> Affine<Self::IC>;
}
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
        .map(|scalar| V::from_ark_scalar(scalar))
        .collect::<Vec<_>>();
    let scalars_slice = HostSlice::from_slice(&scalars);
    let mut msm_result = DeviceVec::<Projective<V::IC>>::cuda_malloc(1).unwrap();
    let stream = CudaStream::create().unwrap();
    let mut cfg = MSMConfig::default();
    cfg.ctx.stream = &stream;
    cfg.is_async = true;
    msm(scalars_slice, bases_slice, &cfg, &mut msm_result[..]).unwrap();
    let mut msm_host_result = vec![Projective::<V::IC>::zero(); 1];
    stream.synchronize().unwrap();
    msm_result
        .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
        .unwrap();
    // Note: in a batched setting this could be removed to a separate method
    stream.destroy().unwrap();
    V::to_ark_projective(&msm_host_result[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::msm::{map_field_elements_to_u64, msm_bigint, msm_binary, msm_u64_wnaf};
    use ark_bn254::{Fr, G1Affine, G1Projective};
    use ark_std::{test_rng, UniformRand, Zero, rand::{distributions::Uniform, Rng}};

    #[test]
    fn icicle_consistency() {
        let mut rng = test_rng();
        let n = 20;
        let scalars = vec![Fr::rand(&mut rng); n];
        let bases = vec![G1Affine::rand(&mut rng); n];
        let max_num_bits = scalars
            .iter()
            .map(|s| s.into_bigint().num_bits())
            .max()
            .unwrap();

        let icicle_res = icicle_msm::<G1Projective>(&bases, &scalars);
        let scalars_u64 = &map_field_elements_to_u64::<G1Projective>(&scalars);
        let msm_binary = msm_binary::<G1Projective>(&bases, &scalars_u64);
        let msm_u64_res = msm_u64_wnaf::<G1Projective>(&bases, scalars_u64, max_num_bits as usize);
        let scalars = scalars.iter().map(|s| s.into_bigint()).collect::<Vec<_>>();
        let msm_bigint_res = msm_bigint::<G1Projective>(&bases, &scalars, max_num_bits as usize);

        assert_eq!(icicle_res, msm_bigint_res);
        //Note: Results of Icicle msm and msm_bigint are inconsistent with msm_u64, msm_small, and msm_binary
        assert_ne!(icicle_res, msm_u64_res);
        assert_ne!(icicle_res, msm_binary);
        assert_ne!(msm_u64_res, msm_binary);
    }

    #[test]
    fn msm_consistency_scalars_all_0() {
        let mut rng = test_rng();
        let n = 20;
        let scalars = vec![Fr::zero(); n];
        let bases = vec![G1Affine::rand(&mut rng); n];

        let icicle_res = icicle_msm::<G1Projective>(&bases, &scalars);
        let msm_res: G1Projective = VariableBaseMSM::msm(&bases, &scalars).unwrap();

        assert_eq!(icicle_res, msm_res);
    }

    #[test]
    fn msm_consistency_scalars_random_0_1() {
        let mut rng = test_rng();
        let range = Uniform::new(0, 1);
        let n = 20;
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::from(rng.sample(&range))).collect();
        let bases = vec![G1Affine::rand(&mut rng); n];

        let icicle_res = icicle_msm::<G1Projective>(&bases, &scalars);
        let msm_res: G1Projective = VariableBaseMSM::msm(&bases, &scalars).unwrap();

        assert_eq!(icicle_res, msm_res);
    }

    #[test]
    fn msm_consistency_scalars_random_0_2_9() {
        let mut rng = test_rng();
        let n = 2_i32.pow(9) as usize;
        let scalars = vec![Fr::rand(&mut rng); n];
        let bases = vec![G1Affine::rand(&mut rng); n];

        let icicle_res = icicle_msm::<G1Projective>(&bases, &scalars);
        let msm_res: G1Projective = VariableBaseMSM::msm(&bases, &scalars).unwrap();

        assert_eq!(icicle_res, msm_res);
    }

    #[test]
    fn msm_consistency_scalars_random_0_2_63() {
        let mut rng = test_rng();
        let n = 2_i32.pow(63) as usize;
        let scalars = vec![Fr::rand(&mut rng); n];
        let bases = vec![G1Affine::rand(&mut rng); n];

        let icicle_res = icicle_msm::<G1Projective>(&bases, &scalars);
        let msm_res: G1Projective = VariableBaseMSM::msm(&bases, &scalars).unwrap();

        assert_eq!(icicle_res, msm_res);
    }
    #[test]
    fn msm_consistency_scalars_random_0_2_253() {
        let mut rng = test_rng();
        let n = 2_i32.pow(253) as usize;
        let scalars = vec![Fr::rand(&mut rng); n];
        let bases = vec![G1Affine::rand(&mut rng); n];

        let icicle_res = icicle_msm::<G1Projective>(&bases, &scalars);
        let msm_res: G1Projective = VariableBaseMSM::msm(&bases, &scalars).unwrap();

        assert_eq!(icicle_res, msm_res);
    }
}
