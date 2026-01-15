//! Curve traits for Jolt's cryptographic operations.
//!
//! This module defines the `JoltCurve` trait which abstracts over pairing-friendly
//! elliptic curves used for polynomial commitments and zero-knowledge proofs.

use crate::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

/// A group element suitable for cryptographic operations.
///
/// This trait abstracts over elliptic curve group operations needed for
/// Pedersen commitments, polynomial commitments, and other cryptographic primitives.
pub trait JoltGroupElement:
    Clone
    + Copy
    + Debug
    + Default
    + Eq
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + AddAssign
    + SubAssign
    + CanonicalSerialize
    + CanonicalDeserialize
{
    fn zero() -> Self;

    fn is_zero(&self) -> bool;

    fn double(&self) -> Self;

    fn scalar_mul<F: JoltField>(&self, scalar: &F) -> Self;
}

/// A pairing-friendly curve suitable for Dory PCS and ZK operations.
///
/// The scalar field is passed as a generic parameter to functions rather than
/// being an associated type, allowing flexibility in which field is used with
/// the curve operations.
pub trait JoltCurve: Clone + Sync + Send + 'static {
    /// G1 group element type
    type G1: JoltGroupElement;

    /// G2 group element type
    type G2: JoltGroupElement;

    /// Target group element type (result of pairing)
    type GT: Clone
        + Debug
        + Default
        + Eq
        + Send
        + Sync
        + 'static
        + Add<Output = Self::GT>
        + for<'a> Add<&'a Self::GT, Output = Self::GT>
        + AddAssign
        + CanonicalSerialize
        + CanonicalDeserialize;

    /// Returns the generator of G1
    fn g1_generator() -> Self::G1;

    /// Returns the generator of G2
    fn g2_generator() -> Self::G2;

    /// Compute pairing e(g1, g2)
    fn pairing(g1: &Self::G1, g2: &Self::G2) -> Self::GT;

    /// Multi-pairing: ∏ᵢ e(g1s[i], g2s[i])
    fn multi_pairing(g1s: &[Self::G1], g2s: &[Self::G2]) -> Self::GT;

    /// Multi-scalar multiplication in G1: Σᵢ scalars[i] * bases[i]
    fn g1_msm<F: JoltField>(bases: &[Self::G1], scalars: &[F]) -> Self::G1;

    /// Multi-scalar multiplication in G2: Σᵢ scalars[i] * bases[i]
    fn g2_msm<F: JoltField>(bases: &[Self::G2], scalars: &[F]) -> Self::G2;

    /// Generate a random G1 element
    fn random_g1<R: rand_core::RngCore>(rng: &mut R) -> Self::G1;

    /// Hash to a G1 curve point with unknown discrete log.
    ///
    /// SECURITY: The returned point must have an unknown discrete log relationship
    /// to the standard generator and any other points derived from scalar multiplication.
    /// This is critical for Pedersen commitment security.
    fn hash_to_g1(domain: &[u8]) -> Self::G1;
}

// ============================================================================
// BN254 Implementation
// ============================================================================

use ark_bn254::{Bn254, Fq12, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{pairing::Pairing, AdditiveGroup, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, Zero};
use std::ops::MulAssign;

/// Wrapper around BN254 G1 projective points
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Bn254G1(pub G1Projective);

impl Add for Bn254G1 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Bn254G1(self.0 + rhs.0)
    }
}

impl<'a> Add<&'a Bn254G1> for Bn254G1 {
    type Output = Self;
    fn add(self, rhs: &'a Bn254G1) -> Self::Output {
        Bn254G1(self.0 + rhs.0)
    }
}

impl Sub for Bn254G1 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Bn254G1(self.0 - rhs.0)
    }
}

impl<'a> Sub<&'a Bn254G1> for Bn254G1 {
    type Output = Self;
    fn sub(self, rhs: &'a Bn254G1) -> Self::Output {
        Bn254G1(self.0 - rhs.0)
    }
}

impl Neg for Bn254G1 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Bn254G1(-self.0)
    }
}

impl AddAssign for Bn254G1 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for Bn254G1 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl<F: JoltField> Mul<F> for Bn254G1 {
    type Output = Self;
    fn mul(mut self, rhs: F) -> Self::Output {
        let scalar = jolt_field_to_fr(&rhs);
        self.0.mul_assign(scalar);
        self
    }
}

impl JoltGroupElement for Bn254G1 {
    fn zero() -> Self {
        Bn254G1(G1Projective::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    fn double(&self) -> Self {
        Bn254G1(AdditiveGroup::double(&self.0))
    }

    fn scalar_mul<F: JoltField>(&self, scalar: &F) -> Self {
        let fr_scalar = jolt_field_to_fr(scalar);
        Bn254G1(self.0 * fr_scalar)
    }
}

/// Wrapper around BN254 G2 projective points
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Bn254G2(pub G2Projective);

impl Add for Bn254G2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Bn254G2(self.0 + rhs.0)
    }
}

impl<'a> Add<&'a Bn254G2> for Bn254G2 {
    type Output = Self;
    fn add(self, rhs: &'a Bn254G2) -> Self::Output {
        Bn254G2(self.0 + rhs.0)
    }
}

impl Sub for Bn254G2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Bn254G2(self.0 - rhs.0)
    }
}

impl<'a> Sub<&'a Bn254G2> for Bn254G2 {
    type Output = Self;
    fn sub(self, rhs: &'a Bn254G2) -> Self::Output {
        Bn254G2(self.0 - rhs.0)
    }
}

impl Neg for Bn254G2 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Bn254G2(-self.0)
    }
}

impl AddAssign for Bn254G2 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for Bn254G2 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl<F: JoltField> Mul<F> for Bn254G2 {
    type Output = Self;
    fn mul(mut self, rhs: F) -> Self::Output {
        let scalar = jolt_field_to_fr(&rhs);
        self.0.mul_assign(scalar);
        self
    }
}

impl JoltGroupElement for Bn254G2 {
    fn zero() -> Self {
        Bn254G2(G2Projective::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    fn double(&self) -> Self {
        Bn254G2(AdditiveGroup::double(&self.0))
    }

    fn scalar_mul<F: JoltField>(&self, scalar: &F) -> Self {
        let fr_scalar = jolt_field_to_fr(scalar);
        Bn254G2(self.0 * fr_scalar)
    }
}

/// Wrapper around BN254 target group (Fq12)
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Bn254GT(pub Fq12);

impl Add for Bn254GT {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Bn254GT(self.0 + rhs.0)
    }
}

impl<'a> Add<&'a Bn254GT> for Bn254GT {
    type Output = Self;
    fn add(self, rhs: &'a Bn254GT) -> Self::Output {
        Bn254GT(self.0 + rhs.0)
    }
}

impl AddAssign for Bn254GT {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

/// The BN254 pairing curve implementation
#[derive(Clone, Debug, Default)]
pub struct Bn254Curve;

impl JoltCurve for Bn254Curve {
    type G1 = Bn254G1;
    type G2 = Bn254G2;
    type GT = Bn254GT;

    fn g1_generator() -> Self::G1 {
        Bn254G1(G1Affine::generator().into())
    }

    fn g2_generator() -> Self::G2 {
        Bn254G2(G2Affine::generator().into())
    }

    fn pairing(g1: &Self::G1, g2: &Self::G2) -> Self::GT {
        Bn254GT(Bn254::pairing(g1.0, g2.0).0)
    }

    fn multi_pairing(g1s: &[Self::G1], g2s: &[Self::G2]) -> Self::GT {
        debug_assert_eq!(g1s.len(), g2s.len());

        let g1_affines: Vec<G1Affine> = g1s.iter().map(|g| g.0.into_affine()).collect();
        let g2_affines: Vec<G2Affine> = g2s.iter().map(|g| g.0.into_affine()).collect();

        Bn254GT(Bn254::multi_pairing(&g1_affines, &g2_affines).0)
    }

    fn g1_msm<F: JoltField>(bases: &[Self::G1], scalars: &[F]) -> Self::G1 {
        debug_assert_eq!(bases.len(), scalars.len());

        let affine_bases: Vec<G1Affine> = bases.iter().map(|b| b.0.into_affine()).collect();
        let fr_scalars: Vec<Fr> = scalars.iter().map(jolt_field_to_fr).collect();
        let bigint_scalars: Vec<_> = fr_scalars.iter().map(|s| s.into_bigint()).collect();

        Bn254G1(G1Projective::msm_bigint(&affine_bases, &bigint_scalars))
    }

    fn g2_msm<F: JoltField>(bases: &[Self::G2], scalars: &[F]) -> Self::G2 {
        debug_assert_eq!(bases.len(), scalars.len());

        let affine_bases: Vec<G2Affine> = bases.iter().map(|b| b.0.into_affine()).collect();
        let fr_scalars: Vec<Fr> = scalars.iter().map(jolt_field_to_fr).collect();
        let bigint_scalars: Vec<_> = fr_scalars.iter().map(|s| s.into_bigint()).collect();

        Bn254G2(G2Projective::msm_bigint(&affine_bases, &bigint_scalars))
    }

    fn random_g1<R: rand_core::RngCore>(rng: &mut R) -> Self::G1 {
        use ark_std::UniformRand;
        Bn254G1(G1Projective::rand(rng))
    }

    fn hash_to_g1(domain: &[u8]) -> Self::G1 {
        bn254_hash_to_g1(domain)
    }
}

/// BN254 hash-to-curve using try-and-increment.
///
/// For BN254 G1, curve equation is y² = x³ + 3.
/// Repeatedly hashes domain with counter until finding valid curve point.
fn bn254_hash_to_g1(domain: &[u8]) -> Bn254G1 {
    use ark_bn254::Fq;
    use ark_ff::{BigInteger, Field};
    use sha3::{Digest, Sha3_512};

    let b = Fq::from(3u64);

    for counter in 0u64.. {
        let mut hasher = Sha3_512::new();
        hasher.update(domain);
        hasher.update(counter.to_le_bytes());
        let hash = hasher.finalize();

        let mut x_bytes = [0u8; 32];
        x_bytes.copy_from_slice(&hash[..32]);
        let sign_bit = (hash[32] & 1) == 1;

        let x = Fq::from_le_bytes_mod_order(&x_bytes);
        let x_cubed = x * x * x;
        let y_squared = x_cubed + b;

        if let Some(y) = y_squared.sqrt() {
            let y_final = if sign_bit == y.into_bigint().is_odd() {
                y
            } else {
                -y
            };

            let point = G1Affine::new_unchecked(x, y_final);
            if point.is_on_curve() && !point.is_zero() {
                return Bn254G1(point.into());
            }
        }
    }
    unreachable!("Hash-to-curve should find a valid point")
}

/// Convert a JoltField element to BN254 Fr.
///
/// This assumes the JoltField is compatible with BN254's scalar field.
/// For ark_bn254::Fr, this is a direct conversion.
#[inline]
fn jolt_field_to_fr<F: JoltField>(f: &F) -> Fr {
    // Serialize the field element and deserialize as Fr
    // This is safe because JoltField elements are assumed to be in the same field
    let mut bytes = [0u8; 32];
    f.serialize_uncompressed(&mut bytes[..])
        .expect("serialization should succeed");
    Fr::from_le_bytes_mod_order(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::UniformRand;
    use rand::thread_rng;

    #[test]
    fn test_g1_operations() {
        let g = Bn254Curve::g1_generator();
        let zero = Bn254G1::zero();

        assert!(zero.is_zero());
        assert!(!g.is_zero());
        assert_eq!(g + zero, g);
        assert_eq!(g - g, zero);
    }

    #[test]
    fn test_g2_operations() {
        let g = Bn254Curve::g2_generator();
        let zero = Bn254G2::zero();

        assert!(zero.is_zero());
        assert!(!g.is_zero());
        assert_eq!(g + zero, g);
        assert_eq!(g - g, zero);
    }

    #[test]
    fn test_pairing_bilinearity() {
        let mut rng = thread_rng();
        let a = Fr::rand(&mut rng);
        let b = Fr::rand(&mut rng);

        let g1 = Bn254Curve::g1_generator();
        let g2 = Bn254Curve::g2_generator();

        let g1_a = g1.scalar_mul(&a);
        let g2_b = g2.scalar_mul(&b);

        // e(a*G1, b*G2) should relate to e(G1, G2)^(a*b)
        let pairing1 = Bn254Curve::pairing(&g1_a, &g2_b);
        let pairing2 = Bn254Curve::pairing(&g1.scalar_mul(&(a * b)), &g2);

        assert_eq!(pairing1, pairing2);
    }

    #[test]
    fn test_g1_msm() {
        let g = Bn254Curve::g1_generator();
        let scalars = vec![Fr::from(2u64), Fr::from(3u64)];
        let bases = vec![g, g];

        let result = Bn254Curve::g1_msm(&bases, &scalars);
        let expected = g.scalar_mul(&Fr::from(5u64));

        assert_eq!(result, expected);
    }
}
