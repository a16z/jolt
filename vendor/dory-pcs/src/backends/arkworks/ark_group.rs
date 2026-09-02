//! Group implementations for BN254 curve (G1, G2, GT)

#![allow(missing_docs)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use super::ark_field::ArkFr;
use crate::primitives::arithmetic::{DoryRoutines, Group};
use ark_bn254::{Bn254, Fq12, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::{Field as ArkField, One, PrimeField, UniformRand, Zero as ArkZero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::ops::{Add, Mul, Neg, Sub};

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug, CanonicalSerialize, CanonicalDeserialize)]
#[repr(transparent)]
pub struct ArkG1(pub G1Projective);

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug, CanonicalSerialize, CanonicalDeserialize)]
#[repr(transparent)]
pub struct ArkG2(pub G2Projective);

#[derive(Clone, Copy, PartialEq, Eq, Debug, CanonicalSerialize, CanonicalDeserialize)]
#[repr(transparent)]
pub struct ArkGT(pub PairingOutput<Bn254>);

impl Default for ArkGT {
    fn default() -> Self {
        ArkGT(PairingOutput(Fq12::one()))
    }
}

impl Group for ArkG1 {
    type Scalar = ArkFr;

    fn identity() -> Self {
        ArkG1(ArkZero::zero())
    }

    fn add(&self, rhs: &Self) -> Self {
        ArkG1(self.0 + rhs.0)
    }

    fn neg(&self) -> Self {
        ArkG1(-self.0)
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        ArkG1(self.0 * k.0)
    }

    fn random() -> Self {
        ArkG1(G1Projective::rand(&mut rand_core::OsRng))
    }
}

impl Add for ArkG1 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        ArkG1(self.0 + rhs.0)
    }
}

impl Sub for ArkG1 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        ArkG1(self.0 - rhs.0)
    }
}

impl Neg for ArkG1 {
    type Output = Self;
    fn neg(self) -> Self {
        ArkG1(-self.0)
    }
}

impl<'a> Add<&'a ArkG1> for ArkG1 {
    type Output = ArkG1;
    fn add(self, rhs: &'a ArkG1) -> ArkG1 {
        ArkG1(self.0 + rhs.0)
    }
}

impl<'a> Sub<&'a ArkG1> for ArkG1 {
    type Output = ArkG1;
    fn sub(self, rhs: &'a ArkG1) -> ArkG1 {
        ArkG1(self.0 - rhs.0)
    }
}

impl Mul<ArkG1> for ArkFr {
    type Output = ArkG1;
    fn mul(self, rhs: ArkG1) -> ArkG1 {
        ArkG1(rhs.0 * self.0)
    }
}

impl<'a> Mul<&'a ArkG1> for ArkFr {
    type Output = ArkG1;
    fn mul(self, rhs: &'a ArkG1) -> ArkG1 {
        ArkG1(rhs.0 * self.0)
    }
}

impl Group for ArkG2 {
    type Scalar = ArkFr;

    fn identity() -> Self {
        ArkG2(ArkZero::zero())
    }

    fn add(&self, rhs: &Self) -> Self {
        ArkG2(self.0 + rhs.0)
    }

    fn neg(&self) -> Self {
        ArkG2(-self.0)
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        ArkG2(self.0 * k.0)
    }

    fn random() -> Self {
        ArkG2(G2Projective::rand(&mut rand_core::OsRng))
    }
}

impl Add for ArkG2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        ArkG2(self.0 + rhs.0)
    }
}

impl Sub for ArkG2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        ArkG2(self.0 - rhs.0)
    }
}

impl Neg for ArkG2 {
    type Output = Self;
    fn neg(self) -> Self {
        ArkG2(-self.0)
    }
}

impl<'a> Add<&'a ArkG2> for ArkG2 {
    type Output = ArkG2;
    fn add(self, rhs: &'a ArkG2) -> ArkG2 {
        ArkG2(self.0 + rhs.0)
    }
}

impl<'a> Sub<&'a ArkG2> for ArkG2 {
    type Output = ArkG2;
    fn sub(self, rhs: &'a ArkG2) -> ArkG2 {
        ArkG2(self.0 - rhs.0)
    }
}

impl Mul<ArkG2> for ArkFr {
    type Output = ArkG2;
    fn mul(self, rhs: ArkG2) -> ArkG2 {
        ArkG2(rhs.0 * self.0)
    }
}

impl<'a> Mul<&'a ArkG2> for ArkFr {
    type Output = ArkG2;
    fn mul(self, rhs: &'a ArkG2) -> ArkG2 {
        ArkG2(rhs.0 * self.0)
    }
}

impl Group for ArkGT {
    type Scalar = ArkFr;

    fn identity() -> Self {
        ArkGT(PairingOutput(Fq12::one()))
    }

    fn add(&self, rhs: &Self) -> Self {
        ArkGT(self.0 + rhs.0)
    }

    fn neg(&self) -> Self {
        ArkGT(PairingOutput(
            ArkField::inverse(&self.0 .0).expect("GT inverse"),
        ))
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        ArkGT(PairingOutput(self.0 .0.pow(k.0.into_bigint())))
    }

    fn random() -> Self {
        ArkGT(Bn254::pairing(
            G1Affine::rand(&mut rand_core::OsRng),
            G2Affine::rand(&mut rand_core::OsRng),
        ))
    }
}

impl Add for ArkGT {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        ArkGT(self.0 + rhs.0)
    }
}

impl Sub for ArkGT {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        ArkGT(self.0 - rhs.0)
    }
}

impl Neg for ArkGT {
    type Output = Self;
    fn neg(self) -> Self {
        ArkGT(-self.0)
    }
}

impl<'a> Add<&'a ArkGT> for ArkGT {
    type Output = ArkGT;
    fn add(self, rhs: &'a ArkGT) -> ArkGT {
        ArkGT(self.0 + rhs.0)
    }
}

impl<'a> Sub<&'a ArkGT> for ArkGT {
    type Output = ArkGT;
    fn sub(self, rhs: &'a ArkGT) -> ArkGT {
        ArkGT(self.0 - rhs.0)
    }
}

impl Mul<ArkGT> for ArkFr {
    type Output = ArkGT;
    fn mul(self, rhs: ArkGT) -> ArkGT {
        ArkGT(PairingOutput(rhs.0 .0.pow(self.0.into_bigint())))
    }
}

impl<'a> Mul<&'a ArkGT> for ArkFr {
    type Output = ArkGT;
    fn mul(self, rhs: &'a ArkGT) -> ArkGT {
        ArkGT(PairingOutput(rhs.0 .0.pow(self.0.into_bigint())))
    }
}

/// left\[i\] = left\[i\] * scalar + right\[i\], parallelized when the `parallel`
/// feature is enabled.
fn fold_field_vectors_impl(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
    assert_eq!(left.len(), right.len(), "Lengths must match");

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        // Per-element work is a single field mul-add (~tens of ns); without a
        // minimum split length, rayon overhead dominates the short vectors of
        // the late reduce-fold rounds.
        left.par_iter_mut()
            .zip(right.par_iter())
            .with_min_len(1 << 10)
            .for_each(|(l, r)| *l = *l * *scalar + *r);
    }
    #[cfg(not(feature = "parallel"))]
    for (l, r) in left.iter_mut().zip(right.iter()) {
        *l = *l * *scalar + *r;
    }
}

pub struct G1Routines;

impl DoryRoutines<ArkG1> for G1Routines {
    #[tracing::instrument(skip_all, name = "G1::msm", fields(len = bases.len()))]
    fn msm(bases: &[ArkG1], scalars: &[ArkFr]) -> ArkG1 {
        assert_eq!(
            bases.len(),
            scalars.len(),
            "MSM requires equal length vectors"
        );

        if bases.is_empty() {
            return ArkG1::identity();
        }

        // Already-normalized bases (z = 1, e.g. setup generators) convert for
        // free via `into_affine`'s short-circuit; otherwise one shared
        // Montgomery batch inversion replaces one field inversion per point.
        let bases_affine: Vec<G1Affine> = if bases.iter().all(|b| b.0.z.is_one()) {
            bases.iter().map(|b| b.0.into_affine()).collect()
        } else {
            let bases_proj: Vec<G1Projective> = bases.iter().map(|b| b.0).collect();
            G1Projective::normalize_batch(&bases_proj)
        };
        let scalars_fr: Vec<ark_bn254::Fr> = scalars.iter().map(|s| s.0).collect();

        ArkG1(G1Projective::msm(&bases_affine, &scalars_fr).expect("MSM failed"))
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG1, scalars: &[ArkFr]) -> Vec<ArkG1> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            scalars.par_iter().map(|s| base.scale(s)).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            scalars.iter().map(|s| base.scale(s)).collect()
        }
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG1], vs: &mut [ArkG1], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "Lengths must match");

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            vs.par_iter_mut()
                .zip(bases.par_iter())
                .for_each(|(v, base)| *v = v.add(&base.scale(scalar)));
        }
        #[cfg(not(feature = "parallel"))]
        for (v, base) in vs.iter_mut().zip(bases.iter()) {
            *v = v.add(&base.scale(scalar));
        }
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG1], addends: &[ArkG1], scalar: &ArkFr) {
        assert_eq!(vs.len(), addends.len(), "Lengths must match");

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            vs.par_iter_mut()
                .zip(addends.par_iter())
                .for_each(|(v, addend)| *v = v.scale(scalar).add(addend));
        }
        #[cfg(not(feature = "parallel"))]
        for (v, addend) in vs.iter_mut().zip(addends.iter()) {
            *v = v.scale(scalar).add(addend);
        }
    }

    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        fold_field_vectors_impl(left, right, scalar);
    }
}

pub struct G2Routines;

impl DoryRoutines<ArkG2> for G2Routines {
    #[tracing::instrument(skip_all, name = "G2::msm", fields(len = bases.len()))]
    fn msm(bases: &[ArkG2], scalars: &[ArkFr]) -> ArkG2 {
        assert_eq!(
            bases.len(),
            scalars.len(),
            "MSM requires equal length vectors"
        );

        if bases.is_empty() {
            return ArkG2::identity();
        }

        // Already-normalized bases (z = 1, e.g. setup generators) convert for
        // free via `into_affine`'s short-circuit; otherwise one shared
        // Montgomery batch inversion replaces one field inversion per point.
        let bases_affine: Vec<G2Affine> = if bases.iter().all(|b| b.0.z.is_one()) {
            bases.iter().map(|b| b.0.into_affine()).collect()
        } else {
            let bases_proj: Vec<G2Projective> = bases.iter().map(|b| b.0).collect();
            G2Projective::normalize_batch(&bases_proj)
        };
        let scalars_fr: Vec<ark_bn254::Fr> = scalars.iter().map(|s| s.0).collect();

        ArkG2(G2Projective::msm(&bases_affine, &scalars_fr).expect("MSM failed"))
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG2, scalars: &[ArkFr]) -> Vec<ArkG2> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            scalars.par_iter().map(|s| base.scale(s)).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            scalars.iter().map(|s| base.scale(s)).collect()
        }
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG2], vs: &mut [ArkG2], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "Lengths must match");

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            vs.par_iter_mut()
                .zip(bases.par_iter())
                .for_each(|(v, base)| *v = v.add(&base.scale(scalar)));
        }
        #[cfg(not(feature = "parallel"))]
        for (v, base) in vs.iter_mut().zip(bases.iter()) {
            *v = v.add(&base.scale(scalar));
        }
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG2], addends: &[ArkG2], scalar: &ArkFr) {
        assert_eq!(vs.len(), addends.len(), "Lengths must match");

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            vs.par_iter_mut()
                .zip(addends.par_iter())
                .for_each(|(v, addend)| *v = v.scale(scalar).add(addend));
        }
        #[cfg(not(feature = "parallel"))]
        for (v, addend) in vs.iter_mut().zip(addends.iter()) {
            *v = v.scale(scalar).add(addend);
        }
    }

    fn fold_field_vectors(left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        fold_field_vectors_impl(left, right, scalar);
    }
}
