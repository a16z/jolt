#![allow(missing_docs)]
use ark_bn254::{Bn254, Fq12, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{pairing::Pairing as ArkPairing, AffineRepr, CurveGroup};
use ark_ff::{Field as ArkField, One, PrimeField, UniformRand, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError};
use ark_serialize::{Read, Valid, Validate, Write};
use ark_std::rand::{rngs::StdRng, RngCore, SeedableRng};

use crate::arithmetic::*;

/// Create a fixed RNG for deterministic tests
pub fn test_rng() -> StdRng {
    let seed = [
        1, 0, 0, 30, 23, 0, 0, 0, 200, 1, 0, 0, 210, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];
    StdRng::from_seed(seed)
}

/* --------- Field trait for Fr --------------------------------------- */
impl Field for Fr {
    fn zero() -> Self {
        Zero::zero()
    }
    fn one() -> Self {
        One::one()
    }

    fn add(&self, rhs: &Self) -> Self {
        *self + *rhs
    }
    fn sub(&self, rhs: &Self) -> Self {
        *self - *rhs
    }
    fn mul(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
    fn inv(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(self.inverse().unwrap())
        }
    }
    fn random<R: RngCore>(_rng: &mut R) -> Self {
        // We use our own fixed RNG for testing
        let mut rng = test_rng();
        Fr::rand(&mut rng)
    }
}

/* --------- Group trait for G1Affine -------------------------------- */
impl Group for G1Affine {
    type Scalar = Fr;

    fn identity() -> Self {
        G1Affine::identity()
    }

    fn add(&self, rhs: &Self) -> Self {
        (self.into_group() + rhs.into_group()).into_affine()
    }

    fn neg(&self) -> Self {
        (-self.into_group()).into_affine()
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        self.mul_bigint((*k).into_bigint()).into_affine()
    }

    fn random<R: RngCore>(_rng: &mut R) -> Self {
        let mut rng = test_rng();
        G1Projective::rand(&mut rng).into_affine()
    }
}

/* --------- Wrapper for G2Affine to avoid conflict ----------------- */

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct G2AffineWrapper(G2Affine);

impl From<G2Affine> for G2AffineWrapper {
    fn from(value: G2Affine) -> Self {
        G2AffineWrapper(value)
    }
}

impl From<G2AffineWrapper> for G2Affine {
    fn from(value: G2AffineWrapper) -> Self {
        value.0
    }
}

// Implementations for ark-serialize
impl CanonicalSerialize for G2AffineWrapper {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(compress)
    }

    fn serialize_compressed<W: std::io::Write>(
        &self,
        writer: W,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.serialize_with_mode(writer, ark_serialize::Compress::Yes)
    }

    fn compressed_size(&self) -> usize {
        self.serialized_size(ark_serialize::Compress::Yes)
    }

    fn serialize_uncompressed<W: std::io::Write>(
        &self,
        writer: W,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.serialize_with_mode(writer, ark_serialize::Compress::No)
    }

    fn uncompressed_size(&self) -> usize {
        self.serialized_size(ark_serialize::Compress::No)
    }
}
impl CanonicalDeserialize for G2AffineWrapper {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        G2Affine::deserialize_with_mode(reader, compress, validate).map(G2AffineWrapper)
    }

    fn deserialize_compressed<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(reader, Compress::Yes, Validate::Yes)
    }

    fn deserialize_compressed_unchecked<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(reader, Compress::Yes, Validate::No)
    }

    fn deserialize_uncompressed<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(reader, Compress::No, Validate::Yes)
    }

    fn deserialize_uncompressed_unchecked<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(reader, Compress::No, Validate::No)
    }
}
impl Valid for G2AffineWrapper {
    fn check(&self) -> Result<(), SerializationError> {
        self.0.check()
    }
}

/* --------- Group trait for G2AffineWrapper ------------------------ */
impl Group for G2AffineWrapper {
    type Scalar = Fr;

    fn identity() -> Self {
        G2AffineWrapper(G2Affine::identity())
    }

    fn add(&self, rhs: &Self) -> Self {
        G2AffineWrapper((self.0.into_group() + rhs.0.into_group()).into_affine())
    }

    fn neg(&self) -> Self {
        G2AffineWrapper((-self.0.into_group()).into_affine())
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        G2AffineWrapper(self.0.mul_bigint((*k).into_bigint()).into_affine())
    }

    fn random<R: RngCore>(_rng: &mut R) -> Self {
        // We use our own fixed RNG for testing
        let mut rng = test_rng();
        G2AffineWrapper(G2Projective::rand(&mut rng).into_affine())
    }
}

/* --------- Group trait for Fq12 (GT) ------------------------------- */
impl Group for Fq12 {
    type Scalar = Fr;

    fn identity() -> Self {
        Self::one()
    }

    fn add(&self, rhs: &Self) -> Self {
        *self * *rhs // Multiplicative group
    }

    fn neg(&self) -> Self {
        if self.is_zero() {
            *self
        } else {
            self.inverse().unwrap()
        }
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        // We convert to BigInt representation suitable for powering
        let repr = (*k).into_bigint();
        self.pow(repr)
    }

    fn random<R: RngCore>(_rng: &mut R) -> Self {
        // We use our own fixed RNG for testing
        let mut rng = test_rng();
        Self::rand(&mut rng)
    }
}

/* --------- lightweight Pairing wrapper ----------------------------- */
#[derive(Clone, Debug)]
pub struct ArkBn254Pairing;

impl Pairing for ArkBn254Pairing {
    type G1 = G1Affine;
    type G2 = G2AffineWrapper;
    type GT = Fq12;

    fn pair(p: &Self::G1, q: &Self::G2) -> Self::GT {
        Bn254::pairing(*p, q.0).0
    }

    fn multi_pair(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair requires equal length vectors"
        );

        if ps.is_empty() {
            return Self::GT::identity();
        }

        // Extract G1 and G2 elements separately
        let g1_elements: Vec<G1Affine> = ps.iter().copied().collect();
        let g2_elements: Vec<G2Affine> = qs.iter().map(|q| q.0).collect();

        // Use the optimized multi-pairing from arkworks (takes two separate iterators)
        Bn254::multi_pairing(g1_elements, g2_elements).0
    }
}

// Optimized MSM implementation using ark-ec's VariableBaseMSM for G1
pub struct OptimizedMsmG1;

impl MultiScalarMul<G1Affine> for OptimizedMsmG1 {
    fn msm(bases: &[G1Affine], scalars: &[Fr]) -> G1Affine {
        if bases.is_empty() {
            return G1Affine::identity();
        }
        use ark_ec::VariableBaseMSM;
        G1Projective::msm(bases, scalars)
            .unwrap_or_else(|_| G1Projective::zero())
            .into_affine()
    }
}

// Optimized MSM implementation using ark-ec's VariableBaseMSM for G2
pub struct OptimizedMsmG2;

impl MultiScalarMul<G2AffineWrapper> for OptimizedMsmG2 {
    fn msm(bases: &[G2AffineWrapper], scalars: &[Fr]) -> G2AffineWrapper {
        if bases.is_empty() {
            return G2AffineWrapper::identity();
        }

        // Convert wrappers to native G2Affine
        let native_bases: Vec<G2Affine> = bases.iter().map(|b| b.0).collect();

        // Use ark-ec's optimized MSM
        use ark_ec::VariableBaseMSM;
        let result = G2Projective::msm(&native_bases, scalars)
            .unwrap_or_else(|_| G2Projective::zero())
            .into_affine();

        G2AffineWrapper(result)
    }
}

// Implementation of MultiScalarMul for GT (Fq12) - fallback to dummy since no native MSM
pub struct DummyMsm<G: Group> {
    _phantom: std::marker::PhantomData<G>,
}

impl<G: Group> MultiScalarMul<G> for DummyMsm<G> {
    fn msm(bases: &[G], scalars: &[G::Scalar]) -> G {
        assert_eq!(
            bases.len(),
            scalars.len(),
            "msm requires equal length inputs"
        );
        if bases.is_empty() {
            return G::identity();
        }

        bases
            .iter()
            .zip(scalars.iter())
            .fold(G::identity(), |acc, (base, scalar)| {
                acc.add(&base.scale(scalar))
            })
    }
}
