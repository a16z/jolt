use ark_bn254::G1Projective;
use ark_ec::{CurveGroup, ScalarMul, VariableBaseMSM as ark_VariableBaseMSM};
use ark_ff::{BigInteger, BigInt, Field, PrimeField};
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

    fn to_ark_scalar(scalar: &<Self::IC as Curve>::ScalarField ) -> Self::ScalarField {
        Self::ScalarField::from_random_bytes(&scalar.to_bytes_le()).unwrap()
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
            y: Self::from_ark_base_field(&point.y)
        }
    }

    fn from_ark_projective(ark: &Self) -> Projective<Self::IC> {
        let proj_x = ark.x * ark.z;
        let proj_z = ark.z * ark.z * ark.z;
        Projective::<Self::IC> {
            x: Self::from_ark_base_field(&proj_x),
            y: Self::from_ark_base_field(&ark.y),
            z: Self::from_ark_base_field(&proj_z)
        }
    }

    fn to_ark_affine(affine: &Affine<Self::IC>) -> Self::MulBase {
        let ark_x = <Self as CurveGroup>::BaseField::from_random_bytes(&affine.x.to_bytes_le()).unwrap();
        let ark_y = <Self as CurveGroup>::BaseField::from_random_bytes(&affine.y.to_bytes_le()).unwrap();
        Self::MulBase::new_unchecked(ark_x, ark_y,)
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

    fn to_ark_scalar(scalar: &<Self::IC as Curve>::ScalarField ) -> Self::ScalarField;

    fn to_ark_projective(point: &Projective<Self::IC>) -> Self;

    fn from_ark_projective(ark: &Self) -> Projective<Self::IC>;

    fn from_ark_affine(point: &Self::MulBase) -> Affine<Self::IC>;

    fn to_ark_affine(ark: &Affine<Self::IC>) -> Self::MulBase;
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
    use ark_bn254::{Fr, Fq, G1Affine, G1Projective};
    use ark_std::{test_rng, UniformRand, Zero, One, rand::{distributions::Uniform, Rng}};
    use icicle_bn254::curve::{CurveCfg, ScalarCfg};
    use icicle_core::traits::GenerateRandom;

    #[test]
    fn icicle_consistency() {
        let mut rng = test_rng();
        let n = 3;
        let scalars = vec![Fr::rand(&mut rng); n];
        let bases = vec![G1Affine::rand(&mut rng); n];
        let max_num_bits = scalars
            .iter()
            .map(|s| s.into_bigint().num_bits())
            .max()
            .unwrap();

        let arkworks_res: G1Projective = ark_VariableBaseMSM::msm(&bases, &scalars).unwrap();
        let icicle_res = icicle_msm::<G1Projective>(&bases, &scalars);
        let scalars_u64 = &map_field_elements_to_u64::<G1Projective>(&scalars);
        let msm_binary = msm_binary::<G1Projective>(&bases, &scalars_u64);
        let msm_u64_res = msm_u64_wnaf::<G1Projective>(&bases, scalars_u64, max_num_bits as usize);
        let scalars = scalars.iter().map(|s| s.into_bigint()).collect::<Vec<_>>();
        let msm_bigint_res = msm_bigint::<G1Projective>(&bases, &scalars, max_num_bits as usize);

        assert_eq!(arkworks_res, msm_bigint_res);
        assert_eq!(icicle_res, arkworks_res);
        assert_eq!(icicle_res, msm_bigint_res);
        //Note: Results of Icicle msm and msm_bigint are inconsistent with msm_u64, msm_small, and msm_binary
        assert_ne!(icicle_res, msm_u64_res);
        assert_ne!(arkworks_res, msm_u64_res);
        assert_ne!(icicle_res, msm_binary);
        assert_ne!(arkworks_res, msm_binary);
        assert_ne!(msm_u64_res, msm_binary);
    }

    fn icicle_test<V: VariableBaseMSM + Icicle + ark_ec::VariableBaseMSM>(
    bases: &[V::MulBase],
    scalars: &[V::ScalarField],
    ) {
        let max_num_bits = scalars
            .iter()
            .map(|s| s.into_bigint().num_bits())
            .max()
            .unwrap();

        let icicle_res = icicle_msm::<V>(&bases, &scalars);
        let msm_res: V  = VariableBaseMSM::msm(&bases, &scalars).unwrap();
        let arkworks_res: V = ark_VariableBaseMSM::msm(&bases, &scalars).unwrap();

        assert_eq!(icicle_res, arkworks_res);
        assert_eq!(icicle_res, msm_res);
    }

    #[test]
    fn msm_consistency_scalars_all_0() {
        for _ in 0..100 {
            let mut rng = test_rng();
            let n = 20;
            let scalars = vec![Fr::zero(); n];
            let bases = vec![G1Affine::rand(&mut rng); n];

            icicle_test::<G1Projective>(&bases, &scalars)
        }
    }

    #[test]
    fn msm_consistency_scalars_random_0_1() {
        for _ in 0..100 {
            let mut rng = test_rng();
            let range = Uniform::new(0, 1);
            let n = 20;
            let scalars: Vec<Fr> = vec![Fr::from(rng.sample(&range));n];
            let bases = vec![G1Affine::rand(&mut rng); n];

            icicle_test::<G1Projective>(&bases, &scalars)
        }
    }

    #[test]
    fn msm_consistency_scalars_random_0_2_9() {
        for _ in 0..100 {
            let mut rng = test_rng();
            let n = 20;
            let range = Uniform::new(0,2u128.pow(9u32));
            let scalars: Vec<Fr> = vec![Fr::from(rng.sample(&range));n];
            let bases = vec![G1Affine::rand(&mut rng); n];

            icicle_test::<G1Projective>(&bases, &scalars)
        }
    }

    #[test]
    fn msm_consistency_scalars_random_0_2_63() {
        for _ in 0..100 {
            let mut rng = test_rng();
            let n = 3;
            let range = Uniform::new(0,2u128.pow(63u32));
            let scalars = vec![Fr::from(rng.sample(&range)); n];
            let bases = vec![G1Affine::rand(&mut rng); n];
            icicle_test::<G1Projective>(&bases, &scalars)
        }
    }

    #[test]
    fn msm_consistency_scalars_random_0_2_253() {
        for _ in 0..100 {
            let mut rng = test_rng();
            let n = 3;
            let scalars = vec![Fr::rand(&mut rng); n];
            let bases = vec![G1Affine::rand(&mut rng); n];

            icicle_test::<G1Projective>(&bases, &scalars)
        }
    }

    //TODO: rework to be inline with past test
    #[test]
    fn icicle_arkworks_conversion() {
        for _ in 0..100 {
            let mut rng = test_rng();
            let icicle_affine = CurveCfg::generate_random_affine_points(1)[0];
            let icicle_projective = CurveCfg::generate_random_projective_points(1)[0];
            let icicle_scalar = ScalarCfg::generate_random(1)[0];

            // Icicle -> Arkworks -> Icicle
            let ark_scalar = G1Projective::to_ark_scalar(&icicle_scalar);
            let scalar_res = G1Projective::from_ark_scalar(&ark_scalar);
            assert_eq!(scalar_res, icicle_scalar);

            let ark_projective = G1Projective::to_ark_projective(&icicle_projective);
            let projective_res = G1Projective::from_ark_projective(&ark_projective);
            assert_eq!(projective_res, icicle_projective);

            let ark_affine = G1Projective::to_ark_affine(&icicle_affine);
            let affine_res = G1Projective::from_ark_affine(&ark_affine);
            assert_eq!(affine_res, icicle_affine);

            // Arkworks -> Icicle Affine -> Icicle Proj -> Icicle Proj 
            // This avoids using added helper methods
            let rand_affine = G1Affine::rand(&mut rng);
            let icicle_affine = G1Projective::from_ark_affine(&rand_affine);
            let icicle_to_proj = icicle_affine.to_projective();
            let proj_res = G1Projective::to_ark_projective(&icicle_to_proj);
            assert_eq!(rand_affine, proj_res);

            // Arkworks -> Icicle -> Arkworks
            let rand_affine = G1Affine::rand(&mut rng);
            let rand_projective = G1Projective::rand(&mut rng);
            let rand_scalar = Fr::rand(&mut rng);

            let icicle_scalar = G1Projective::from_ark_scalar(&rand_scalar);
            let scalar_res = G1Projective::to_ark_scalar(&icicle_scalar);
            assert_eq!(rand_scalar, scalar_res);

            let icicle_projective = G1Projective::from_ark_projective(&rand_projective);
            let proj_res = G1Projective::to_ark_projective(&icicle_projective);
            assert_eq!(proj_res, rand_projective);

            let icicle_affine = G1Projective::from_ark_affine(&rand_affine);
            let affine_res = G1Projective::to_ark_affine(&icicle_affine);
            assert_eq!(affine_res, rand_affine);
        }
    }

    fn zero_pad() {
        for _ in 0..100 {
            let mut rng = test_rng();
            let n = 3;
            let mut scalars = vec![Fr::rand(&mut rng); n];
            //Pad scalars to length 4 with 0 vector -> We should see this succeed when 2_253 (aka msm of length 3) fails
            scalars.push(Fr::zero());
            let bases = vec![G1Affine::rand(&mut rng); n + 1];
            icicle_test::<G1Projective>(&bases, &scalars)
        }
    }
    
    #[test]
    fn point_doubling_1() {
        for _ in 0..100 {
            let mut rng = test_rng();
            let scalars = vec![Fr::rand(&mut rng); 4];
            let rand_base = G1Affine::rand(&mut rng);
            let bases = vec![rand_base, rand_base, rand_base, (rand_base + rand_base).into()];
            icicle_test::<G1Projective>(&bases, &scalars)
        }
    }

    #[test]
    fn point_doubling_2() {
        for _ in 0..100 {
            let mut rng = test_rng();
            let n = 3;
            let scalars = vec![Fr::rand(&mut rng); n];
            let rand_base = G1Affine::rand(&mut rng);
            let bases = vec![rand_base, (rand_base + rand_base).into(), (rand_base + rand_base + rand_base).into()];
            icicle_test::<G1Projective>(&bases, &scalars)
        }
    }

}
