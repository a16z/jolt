//! Pedersen commitment scheme for small vectors (e.g., sumcheck round polynomials)
//!
//! Used to make sumcheck prover messages hiding. Commitments are of the form:
//!   C = Σᵢ mᵢ * Gᵢ + r * H
//! where Gᵢ are message generators and H is the blinding generator.

use crate::curve::{JoltCurve, JoltGroupElement};
use crate::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand_core::CryptoRngCore;

#[derive(Clone, Debug)]
pub struct PedersenGenerators<C: JoltCurve> {
    pub message_generators: Vec<C::G1>,
    pub blinding_generator: C::G1,
}

impl<C: JoltCurve> PedersenGenerators<C> {
    pub fn new(generators: Vec<C::G1>) -> Self {
        assert!(!generators.is_empty(), "Need at least one generator");
        let blinding_generator = C::hash_to_g1(b"jolt_pedersen_blinding_h2c_v1");
        Self {
            message_generators: generators,
            blinding_generator,
        }
    }

    pub fn commit<F: JoltField>(&self, coeffs: &[F], blinding: &F) -> C::G1 {
        assert!(
            coeffs.len() <= self.message_generators.len(),
            "Too many coefficients: {} > {}",
            coeffs.len(),
            self.message_generators.len()
        );

        let msg_commitment = if coeffs.is_empty() {
            C::G1::zero()
        } else {
            C::g1_msm(&self.message_generators[..coeffs.len()], coeffs)
        };

        let blinding_commitment = self.blinding_generator.scalar_mul(blinding);
        msg_commitment + blinding_commitment
    }

    pub fn verify<F: JoltField>(&self, commitment: &C::G1, coeffs: &[F], blinding: &F) -> bool {
        let expected = self.commit(coeffs, blinding);
        *commitment == expected
    }

    pub fn deterministic(count: usize) -> Self {
        let generators: Vec<C::G1> = (0..count)
            .map(|i| {
                let mut domain = b"jolt_pedersen_msg_gen_v1_".to_vec();
                domain.extend_from_slice(&(i as u64).to_le_bytes());
                C::hash_to_g1(&domain)
            })
            .collect();
        Self::new(generators)
    }
}

use crate::curve::{Bn254Curve, Bn254G1};
use crate::poly::commitment::dory::ArkG1;

impl PedersenGenerators<Bn254Curve> {
    pub fn from_dory_generators(dory_g1: &[ArkG1], count: usize) -> Self {
        assert!(count <= dory_g1.len(), "Not enough Dory generators");
        let generators: Vec<Bn254G1> = dory_g1[..count].iter().map(|g| Bn254G1(g.0)).collect();
        Self::new(generators)
    }
}

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
    use crate::curve::Bn254Curve;
    use ark_bn254::Fr;
    use ark_std::UniformRand;
    use rand::thread_rng;

    #[test]
    fn test_pedersen_commit_verify() {
        let mut rng = thread_rng();
        let gens = PedersenGenerators::<Bn254Curve>::deterministic(10);

        let coeffs: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rng)).collect();
        let blinding = Fr::rand(&mut rng);

        let commitment = gens.commit(&coeffs, &blinding);

        assert!(gens.verify(&commitment, &coeffs, &blinding));

        let wrong_blinding = Fr::rand(&mut rng);
        assert!(!gens.verify(&commitment, &coeffs, &wrong_blinding));

        let mut wrong_coeffs = coeffs.clone();
        wrong_coeffs[0] = Fr::rand(&mut rng);
        assert!(!gens.verify(&commitment, &wrong_coeffs, &blinding));
    }

    #[test]
    fn test_blinding_generator_derivation() {
        let gens = PedersenGenerators::<Bn254Curve>::deterministic(3);

        // Blinding generator should be different from all message generators
        for msg_gen in &gens.message_generators {
            assert_ne!(gens.blinding_generator, *msg_gen);
        }

        assert!(!gens.blinding_generator.is_zero());

        // Deterministic
        let gens2 = PedersenGenerators::<Bn254Curve>::deterministic(3);
        assert_eq!(gens.blinding_generator, gens2.blinding_generator);
        assert_eq!(gens.message_generators, gens2.message_generators);
    }

    #[test]
    fn test_commitment_homomorphism() {
        let mut rng = thread_rng();
        let gens = PedersenGenerators::<Bn254Curve>::deterministic(10);

        let coeffs1: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rng)).collect();
        let coeffs2: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rng)).collect();
        let r1 = Fr::rand(&mut rng);
        let r2 = Fr::rand(&mut rng);

        let c1 = gens.commit(&coeffs1, &r1);
        let c2 = gens.commit(&coeffs2, &r2);

        let c_sum = c1 + c2;

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
