//! Implements `dory_pcs` traits for jolt-crypto group types, allowing jolt-dory
//! to call `dory::prove`/`dory::verify` with [`Bn254G1`], [`Bn254G2`], [`Bn254GT`],
//! and [`Bn254`] directly — no arkworks-to-jolt conversion layer needed.

use std::io::{Read, Write};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use dory::primitives::arithmetic;
use dory::primitives::serialization::{
    Compress, DoryDeserialize, DorySerialize, SerializationError, Valid, Validate,
};

use jolt_field::Fr;

use crate::arkworks::bn254::{Bn254, Bn254G1, Bn254G2, Bn254GT};
use crate::{JoltGroup, PairingGroup};

// Serialization bridge helpers

#[inline(always)]
fn to_ark_compress(c: Compress) -> ark_serialize::Compress {
    match c {
        Compress::Yes => ark_serialize::Compress::Yes,
        Compress::No => ark_serialize::Compress::No,
    }
}

#[inline(always)]
fn to_ark_validate(v: Validate) -> ark_serialize::Validate {
    match v {
        Validate::Yes => ark_serialize::Validate::Yes,
        Validate::No => ark_serialize::Validate::No,
    }
}

fn ark_err_to_dory(e: ark_serialize::SerializationError) -> SerializationError {
    match e {
        ark_serialize::SerializationError::IoError(io) => SerializationError::IoError(io),
        ark_serialize::SerializationError::InvalidData => {
            SerializationError::InvalidData("arkworks: invalid data".into())
        }
        ark_serialize::SerializationError::UnexpectedFlags => SerializationError::UnexpectedData,
        ark_serialize::SerializationError::NotEnoughSpace => {
            SerializationError::InvalidData("arkworks: not enough space".into())
        }
    }
}

// Macro for serialization trait impls (G1, G2, GT all follow the same pattern)

macro_rules! impl_dory_serialization {
    ($ty:ty, $inner:ty) => {
        impl Valid for $ty {
            fn check(&self) -> Result<(), SerializationError> {
                ark_serialize::Valid::check(&self.0).map_err(ark_err_to_dory)
            }
        }

        impl DorySerialize for $ty {
            fn serialize_with_mode<W: Write>(
                &self,
                writer: W,
                compress: Compress,
            ) -> Result<(), SerializationError> {
                self.0
                    .serialize_with_mode(writer, to_ark_compress(compress))
                    .map_err(ark_err_to_dory)
            }

            fn serialized_size(&self, compress: Compress) -> usize {
                self.0.serialized_size(to_ark_compress(compress))
            }
        }

        impl DoryDeserialize for $ty {
            fn deserialize_with_mode<R: Read>(
                reader: R,
                compress: Compress,
                validate: Validate,
            ) -> Result<Self, SerializationError> {
                <$inner>::deserialize_with_mode(
                    reader,
                    to_ark_compress(compress),
                    to_ark_validate(validate),
                )
                .map(Self)
                .map_err(ark_err_to_dory)
            }
        }
    };
}

impl_dory_serialization!(Bn254G1, ark_bn254::G1Projective);
impl_dory_serialization!(Bn254G2, ark_bn254::G2Projective);
impl_dory_serialization!(Bn254GT, ark_bn254::Fq12);

// Fr * Group operator impls (required by dory::Group::Scalar bound)

impl std::ops::Mul<Bn254G1> for Fr {
    type Output = Bn254G1;
    #[inline(always)]
    fn mul(self, rhs: Bn254G1) -> Bn254G1 {
        rhs.scalar_mul(&self)
    }
}

impl<'a> std::ops::Mul<&'a Bn254G1> for Fr {
    type Output = Bn254G1;
    #[inline(always)]
    fn mul(self, rhs: &'a Bn254G1) -> Bn254G1 {
        rhs.scalar_mul(&self)
    }
}

impl std::ops::Mul<Bn254G2> for Fr {
    type Output = Bn254G2;
    #[inline(always)]
    fn mul(self, rhs: Bn254G2) -> Bn254G2 {
        rhs.scalar_mul(&self)
    }
}

impl<'a> std::ops::Mul<&'a Bn254G2> for Fr {
    type Output = Bn254G2;
    #[inline(always)]
    fn mul(self, rhs: &'a Bn254G2) -> Bn254G2 {
        rhs.scalar_mul(&self)
    }
}

impl std::ops::Mul<Bn254GT> for Fr {
    type Output = Bn254GT;
    #[inline(always)]
    fn mul(self, rhs: Bn254GT) -> Bn254GT {
        rhs.scalar_mul(&self)
    }
}

impl<'a> std::ops::Mul<&'a Bn254GT> for Fr {
    type Output = Bn254GT;
    #[inline(always)]
    fn mul(self, rhs: &'a Bn254GT) -> Bn254GT {
        rhs.scalar_mul(&self)
    }
}

// dory::Group impls

impl arithmetic::Group for Bn254G1 {
    type Scalar = Fr;

    #[inline(always)]
    fn identity() -> Self {
        JoltGroup::identity()
    }

    #[inline(always)]
    fn add(&self, rhs: &Self) -> Self {
        *self + *rhs
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        std::ops::Neg::neg(*self)
    }

    #[inline]
    fn scale(&self, k: &Fr) -> Self {
        self.scalar_mul(k)
    }

    #[inline]
    fn random() -> Self {
        Bn254::random_g1(&mut rand_core::OsRng)
    }
}

impl arithmetic::Group for Bn254G2 {
    type Scalar = Fr;

    #[inline(always)]
    fn identity() -> Self {
        JoltGroup::identity()
    }

    #[inline(always)]
    fn add(&self, rhs: &Self) -> Self {
        *self + *rhs
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        std::ops::Neg::neg(*self)
    }

    #[inline]
    fn scale(&self, k: &Fr) -> Self {
        self.scalar_mul(k)
    }

    #[inline]
    fn random() -> Self {
        use ark_std::UniformRand;
        Bn254G2(ark_bn254::G2Projective::rand(&mut rand_core::OsRng))
    }
}

impl arithmetic::Group for Bn254GT {
    type Scalar = Fr;

    #[inline(always)]
    fn identity() -> Self {
        JoltGroup::identity()
    }

    #[inline(always)]
    fn add(&self, rhs: &Self) -> Self {
        *self + *rhs
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        std::ops::Neg::neg(*self)
    }

    #[inline]
    fn scale(&self, k: &Fr) -> Self {
        self.scalar_mul(k)
    }

    #[inline]
    fn random() -> Self {
        // Generate a random GT element via pairing(g1^r, g2_gen)
        let r_g1 = Bn254::random_g1(&mut rand_core::OsRng);
        let g2 = Bn254::g2_generator();
        <Bn254 as PairingGroup>::pairing(&r_g1, &g2)
    }
}

// dory::PairingCurve

impl arithmetic::PairingCurve for Bn254 {
    type G1 = Bn254G1;
    type G2 = Bn254G2;
    type GT = Bn254GT;

    #[inline]
    fn pair(p: &Bn254G1, q: &Bn254G2) -> Bn254GT {
        <Bn254 as PairingGroup>::pairing(p, q)
    }

    #[inline]
    fn multi_pair(ps: &[Bn254G1], qs: &[Bn254G2]) -> Bn254GT {
        <Bn254 as PairingGroup>::multi_pairing(ps, qs)
    }

    #[inline]
    fn multi_pair_g2_setup(ps: &[Bn254G1], qs: &[Bn254G2]) -> Bn254GT {
        <Bn254 as PairingGroup>::multi_pairing(ps, qs)
    }

    #[inline]
    fn multi_pair_g1_setup(ps: &[Bn254G1], qs: &[Bn254G2]) -> Bn254GT {
        <Bn254 as PairingGroup>::multi_pairing(ps, qs)
    }
}

// DoryRoutines

/// GLV-optimized MSM and vector operations for G1 elements.
pub struct OptimizedG1Routines;

impl arithmetic::DoryRoutines<Bn254G1> for OptimizedG1Routines {
    #[inline]
    fn msm(bases: &[Bn254G1], scalars: &[Fr]) -> Bn254G1 {
        JoltGroup::msm(bases, scalars)
    }

    fn fixed_base_vector_scalar_mul(base: &Bn254G1, scalars: &[Fr]) -> Vec<Bn254G1> {
        crate::arkworks::bn254::glv::fixed_base_vector_msm_g1(base, scalars)
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[Bn254G1], vs: &mut [Bn254G1], scalar: &Fr) {
        crate::arkworks::bn254::glv::vector_add_scalar_mul_g1(vs, bases, *scalar);
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [Bn254G1], addends: &[Bn254G1], scalar: &Fr) {
        crate::arkworks::bn254::glv::vector_scalar_mul_add_gamma_g1(vs, *scalar, addends);
    }
}

/// GLV-optimized MSM and vector operations for G2 elements.
pub struct OptimizedG2Routines;

impl arithmetic::DoryRoutines<Bn254G2> for OptimizedG2Routines {
    #[inline]
    fn msm(bases: &[Bn254G2], scalars: &[Fr]) -> Bn254G2 {
        JoltGroup::msm(bases, scalars)
    }

    fn fixed_base_vector_scalar_mul(base: &Bn254G2, scalars: &[Fr]) -> Vec<Bn254G2> {
        scalars
            .iter()
            .map(|s| {
                crate::arkworks::bn254::glv::glv_four_scalar_mul(
                    *s,
                    std::slice::from_ref(base),
                )[0]
            })
            .collect()
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[Bn254G2], vs: &mut [Bn254G2], scalar: &Fr) {
        crate::arkworks::bn254::glv::vector_add_scalar_mul_g2(vs, bases, *scalar);
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [Bn254G2], addends: &[Bn254G2], scalar: &Fr) {
        crate::arkworks::bn254::glv::vector_scalar_mul_add_gamma_g2(vs, *scalar, addends);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dory::primitives::arithmetic::{
        DoryRoutines, Field as DoryField, Group as DoryGroup, PairingCurve as DoryPairing,
    };

    #[test]
    fn g1_identity_and_neg() {
        let id = <Bn254G1 as DoryGroup>::identity();
        let g = <Bn254G1 as DoryGroup>::random();
        assert_eq!(DoryGroup::add(&g, &id), g);
        assert_eq!(DoryGroup::add(&g, &DoryGroup::neg(&g)), id);
    }

    #[test]
    fn g2_identity_and_neg() {
        let id = <Bn254G2 as DoryGroup>::identity();
        let g = <Bn254G2 as DoryGroup>::random();
        assert_eq!(DoryGroup::add(&g, &id), g);
        assert_eq!(DoryGroup::add(&g, &DoryGroup::neg(&g)), id);
    }

    #[test]
    fn gt_identity_and_neg() {
        let id = <Bn254GT as DoryGroup>::identity();
        let g = <Bn254GT as DoryGroup>::random();
        assert_eq!(DoryGroup::add(&g, &id), g);
        assert_eq!(DoryGroup::add(&g, &DoryGroup::neg(&g)), id);
    }

    #[test]
    fn g1_scale() {
        let g = <Bn254G1 as DoryGroup>::random();
        let two = <Fr as DoryField>::from_u64(2);
        let doubled = DoryGroup::scale(&g, &two);
        assert_eq!(doubled, DoryGroup::add(&g, &g));
    }

    #[test]
    fn pairing_bilinearity() {
        let g1 = Bn254::g1_generator();
        let g2 = Bn254::g2_generator();
        let a = <Fr as DoryField>::from_u64(3);
        let b = <Fr as DoryField>::from_u64(5);

        // e(a*G1, b*G2) == e(G1, G2)^(a*b)
        let lhs =
            <Bn254 as DoryPairing>::pair(&DoryGroup::scale(&g1, &a), &DoryGroup::scale(&g2, &b));
        let rhs = DoryGroup::scale(
            &<Bn254 as DoryPairing>::pair(&g1, &g2),
            &DoryField::mul(&a, &b),
        );
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn multi_pair_matches_sequential() {
        let g1 = Bn254::g1_generator();
        let g2 = Bn254::g2_generator();
        let a = <Fr as DoryField>::from_u64(7);
        let b = <Fr as DoryField>::from_u64(11);

        let g1s = vec![DoryGroup::scale(&g1, &a), DoryGroup::scale(&g1, &b)];
        let g2s = vec![g2, g2];

        let multi = <Bn254 as DoryPairing>::multi_pair(&g1s, &g2s);
        let sequential = DoryGroup::add(
            &<Bn254 as DoryPairing>::pair(&g1s[0], &g2s[0]),
            &<Bn254 as DoryPairing>::pair(&g1s[1], &g2s[1]),
        );
        assert_eq!(multi, sequential);
    }

    #[test]
    fn g1_msm_matches_jolt() {
        let bases: Vec<Bn254G1> = (0..4).map(|_| <Bn254G1 as DoryGroup>::random()).collect();
        let scalars: Vec<Fr> = (1..=4).map(|i| <Fr as DoryField>::from_u64(i)).collect();

        let via_routines = OptimizedG1Routines::msm(&bases, &scalars);
        let via_jolt = JoltGroup::msm(&bases, &scalars);
        assert_eq!(via_routines, via_jolt);
    }

    #[test]
    fn g1_fixed_base_vector_scalar_mul() {
        let base = <Bn254G1 as DoryGroup>::random();
        let scalars: Vec<Fr> = (1..=3).map(|i| <Fr as DoryField>::from_u64(i)).collect();
        let result = OptimizedG1Routines::fixed_base_vector_scalar_mul(&base, &scalars);
        for (r, s) in result.iter().zip(scalars.iter()) {
            assert_eq!(*r, DoryGroup::scale(&base, s));
        }
    }

    #[test]
    fn g1_fixed_scalar_mul_bases_then_add() {
        let bases: Vec<Bn254G1> = (0..3).map(|_| <Bn254G1 as DoryGroup>::random()).collect();
        let scalar = <Fr as DoryField>::from_u64(5);
        let mut vs: Vec<Bn254G1> = (0..3).map(|_| <Bn254G1 as DoryGroup>::random()).collect();
        let vs_orig = vs.clone();

        OptimizedG1Routines::fixed_scalar_mul_bases_then_add(&bases, &mut vs, &scalar);
        for i in 0..3 {
            let expected = DoryGroup::add(&vs_orig[i], &DoryGroup::scale(&bases[i], &scalar));
            assert_eq!(vs[i], expected);
        }
    }

    #[test]
    fn g2_msm_matches_jolt() {
        let bases: Vec<Bn254G2> = (0..4).map(|_| <Bn254G2 as DoryGroup>::random()).collect();
        let scalars: Vec<Fr> = (1..=4).map(|i| <Fr as DoryField>::from_u64(i)).collect();

        let via_routines = OptimizedG2Routines::msm(&bases, &scalars);
        let via_jolt = JoltGroup::msm(&bases, &scalars);
        assert_eq!(via_routines, via_jolt);
    }

    #[test]
    fn g1_serialization_roundtrip() {
        let g = <Bn254G1 as DoryGroup>::random();
        let mut buf = Vec::new();
        DorySerialize::serialize_compressed(&g, &mut buf).unwrap();
        let recovered: Bn254G1 = DoryDeserialize::deserialize_compressed(&buf[..]).unwrap();
        assert_eq!(g, recovered);
    }

    #[test]
    fn g2_serialization_roundtrip() {
        let g = <Bn254G2 as DoryGroup>::random();
        let mut buf = Vec::new();
        DorySerialize::serialize_compressed(&g, &mut buf).unwrap();
        let recovered: Bn254G2 = DoryDeserialize::deserialize_compressed(&buf[..]).unwrap();
        assert_eq!(g, recovered);
    }

    #[test]
    fn gt_serialization_roundtrip() {
        let g = <Bn254GT as DoryGroup>::random();
        let mut buf = Vec::new();
        DorySerialize::serialize_compressed(&g, &mut buf).unwrap();
        let recovered: Bn254GT = DoryDeserialize::deserialize_compressed(&buf[..]).unwrap();
        assert_eq!(g, recovered);
    }

    #[test]
    fn fr_mul_g1() {
        let g = <Bn254G1 as DoryGroup>::random();
        let s = <Fr as DoryField>::from_u64(42);
        let via_mul: Bn254G1 = s * g;
        let via_scale = DoryGroup::scale(&g, &s);
        assert_eq!(via_mul, via_scale);

        // Reference variant
        let via_mul_ref: Bn254G1 = s * &g;
        assert_eq!(via_mul_ref, via_scale);
    }
}
