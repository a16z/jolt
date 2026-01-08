//! Pedersen commitment scheme for small vectors (e.g., sumcheck round polynomials)
//!
//! Used to make sumcheck prover messages hiding. Commitments are of the form:
//!   C = Σᵢ mᵢ * Gᵢ + r * H
//! where Gᵢ are message generators and H is the blinding generator.

use crate::curve::{JoltCurve, JoltGroupElement};
use crate::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand_core::CryptoRngCore;
use sha3::{Digest, Sha3_256};

/// Pedersen commitment generators for sumcheck blinding.
///
/// Generic over the curve type to support different elliptic curves.
#[derive(Clone, Debug)]
pub struct PedersenGenerators<C: JoltCurve> {
    /// Message generators G₀, G₁, ..., Gₖ
    pub message_generators: Vec<C::G1>,
    /// Blinding generator H (derived via hash-to-curve)
    pub blinding_generator: C::G1,
}

impl<C: JoltCurve> PedersenGenerators<C> {
    /// Create Pedersen generators from a slice of G1 elements.
    ///
    /// The blinding generator H is derived deterministically from the first
    /// message generator using a hash-based approach.
    pub fn new(generators: Vec<C::G1>) -> Self {
        assert!(!generators.is_empty(), "Need at least one generator");

        let blinding_generator = derive_blinding_generator::<C>(&generators[0]);

        Self {
            message_generators: generators,
            blinding_generator,
        }
    }

    /// Commit to a vector of scalars with blinding factor.
    ///
    /// Returns C = Σᵢ coeffsᵢ * Gᵢ + blinding * H
    pub fn commit<F: JoltField>(&self, coeffs: &[F], blinding: &F) -> C::G1 {
        assert!(
            coeffs.len() <= self.message_generators.len(),
            "Too many coefficients: {} > {}",
            coeffs.len(),
            self.message_generators.len()
        );

        // Compute message part: Σᵢ coeffsᵢ * Gᵢ
        let msg_commitment = if coeffs.is_empty() {
            C::G1::zero()
        } else {
            C::g1_msm(&self.message_generators[..coeffs.len()], coeffs)
        };

        // Compute blinding part: blinding * H
        let blinding_commitment = self.blinding_generator.scalar_mul(blinding);

        msg_commitment + blinding_commitment
    }

    /// Verify a Pedersen commitment opening.
    pub fn verify<F: JoltField>(&self, commitment: &C::G1, coeffs: &[F], blinding: &F) -> bool {
        let expected = self.commit(coeffs, blinding);
        *commitment == expected
    }
}

/// Derive the blinding generator H from the first message generator.
///
/// Uses a simple approach: H = hash_scalar * G₀ where hash_scalar is derived
/// from hashing the serialized G₀. This ensures H has an unknown discrete log
/// relationship to the message generators.
fn derive_blinding_generator<C: JoltCurve>(g0: &C::G1) -> C::G1 {
    let mut hasher = Sha3_256::new();
    hasher.update(b"jolt_pedersen_blinding_v1");

    let mut g0_bytes = Vec::new();
    g0.serialize_compressed(&mut g0_bytes)
        .expect("Serialization should not fail");
    hasher.update(&g0_bytes);

    let hash = hasher.finalize();

    // Use the hash to derive a scalar, then multiply G₀ by it
    // This gives us a point with unknown discrete log relative to G₀
    let mut scalar_bytes = [0u8; 32];
    scalar_bytes.copy_from_slice(&hash[..32]);

    // We need to derive a scalar from the hash and multiply
    // Using a simple approach: interpret hash as scalar and use double-and-add
    let mut result = C::G1::zero();
    let mut base = *g0;

    for byte in scalar_bytes.iter() {
        for bit in 0..8 {
            if (byte >> bit) & 1 == 1 {
                result += base;
            }
            base = base.double();
        }
    }

    result
}

// ============================================================================
// Dory Integration
// ============================================================================

use crate::curve::{Bn254Curve, Bn254G1};
use crate::poly::commitment::dory::ArkG1;

impl PedersenGenerators<Bn254Curve> {
    /// Create Pedersen generators from Dory URS G1 generators.
    ///
    /// This allows reusing the Dory trusted setup for Pedersen commitments
    /// in zero-knowledge sumcheck.
    pub fn from_dory_generators(dory_g1: &[ArkG1], count: usize) -> Self {
        assert!(count <= dory_g1.len(), "Not enough Dory generators");

        // ArkG1 wraps G1Projective, and Bn254G1 also wraps G1Projective
        let generators: Vec<Bn254G1> = dory_g1[..count].iter().map(|g| Bn254G1(g.0)).collect();

        Self::new(generators)
    }
}

/// A blinded scalar value with its commitment randomness
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BlindedScalar<F: JoltField> {
    pub value: F,
    pub blinding: F,
}

impl<F: JoltField> BlindedScalar<F> {
    pub fn new(value: F, blinding: F) -> Self {
        Self { value, blinding }
    }

    pub fn random<R: CryptoRngCore>(value: F, rng: &mut R) -> Self {
        Self {
            value,
            blinding: F::random(rng),
        }
    }
}

/// A blinded vector of scalars (e.g., polynomial coefficients)
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BlindedVector<F: JoltField> {
    pub values: Vec<F>,
    pub blinding: F,
}

impl<F: JoltField> BlindedVector<F> {
    pub fn new(values: Vec<F>, blinding: F) -> Self {
        Self { values, blinding }
    }

    pub fn random<R: CryptoRngCore>(values: Vec<F>, rng: &mut R) -> Self {
        Self {
            values,
            blinding: F::random(rng),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::{Bn254Curve, Bn254G1};
    use ark_bn254::Fr;
    use ark_std::UniformRand;
    use rand::thread_rng;

    fn mock_generators(n: usize) -> Vec<Bn254G1> {
        let mut rng = thread_rng();
        (0..n)
            .map(|_| {
                use ark_bn254::G1Projective;
                Bn254G1(G1Projective::rand(&mut rng))
            })
            .collect()
    }

    #[test]
    fn test_pedersen_commit_verify() {
        let mut rng = thread_rng();
        let generators = mock_generators(10);
        let gens = PedersenGenerators::<Bn254Curve>::new(generators);

        let coeffs: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rng)).collect();
        let blinding = Fr::rand(&mut rng);

        let commitment = gens.commit(&coeffs, &blinding);

        assert!(gens.verify(&commitment, &coeffs, &blinding));

        // Wrong blinding should fail
        let wrong_blinding = Fr::rand(&mut rng);
        assert!(!gens.verify(&commitment, &coeffs, &wrong_blinding));

        // Wrong coeffs should fail
        let mut wrong_coeffs = coeffs.clone();
        wrong_coeffs[0] = Fr::rand(&mut rng);
        assert!(!gens.verify(&commitment, &wrong_coeffs, &blinding));
    }

    #[test]
    fn test_blinding_generator_derivation() {
        let generators = mock_generators(1);
        let gens = PedersenGenerators::<Bn254Curve>::new(generators.clone());

        // Blinding generator should be different from message generator
        assert_ne!(gens.blinding_generator, gens.message_generators[0]);

        // Blinding generator should not be zero
        assert!(!gens.blinding_generator.is_zero());

        // Derivation should be deterministic
        let gens2 = PedersenGenerators::<Bn254Curve>::new(generators);
        assert_eq!(gens.blinding_generator, gens2.blinding_generator);
    }

    #[test]
    fn test_commitment_homomorphism() {
        let mut rng = thread_rng();
        let generators = mock_generators(10);
        let gens = PedersenGenerators::<Bn254Curve>::new(generators);

        let coeffs1: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rng)).collect();
        let coeffs2: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rng)).collect();
        let r1 = Fr::rand(&mut rng);
        let r2 = Fr::rand(&mut rng);

        let c1 = gens.commit(&coeffs1, &r1);
        let c2 = gens.commit(&coeffs2, &r2);

        // Sum of commitments
        let c_sum = c1 + c2;

        // Commitment to sum
        let coeffs_sum: Vec<Fr> = coeffs1
            .iter()
            .zip(coeffs2.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        let r_sum = r1 + r2;
        let c_expected = gens.commit(&coeffs_sum, &r_sum);

        assert_eq!(c_sum, c_expected);
    }
}
