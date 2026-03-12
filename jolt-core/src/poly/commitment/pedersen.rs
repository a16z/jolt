//! Pedersen commitment scheme for small vectors (e.g., sumcheck round polynomials)
//!
//! Used to make sumcheck prover messages hiding. Commitments are of the form:
//!   C = Σᵢ mᵢ * Gᵢ + r * H
//! where Gᵢ are message generators and H is the blinding generator.

use crate::curve::JoltCurve;
use crate::field::JoltField;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use rand_core::CryptoRngCore;

#[derive(Clone, Debug)]
pub struct PedersenGenerators<C: JoltCurve> {
    pub message_generators: Vec<C::G1>,
    pub blinding_generator: C::G1,
    /// Pre-converted affine bases: [msg_0, msg_1, ..., msg_{n-1}, blinding]
    /// Avoids per-commit field inversion from projective→affine conversion.
    affine_bases: Vec<C::G1Affine>,
}

impl<C: JoltCurve> CanonicalSerialize for PedersenGenerators<C> {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.message_generators
            .serialize_with_mode(&mut writer, compress)?;
        self.blinding_generator
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.message_generators.serialized_size(compress)
            + self.blinding_generator.serialized_size(compress)
    }
}

impl<C: JoltCurve> Valid for PedersenGenerators<C> {
    fn check(&self) -> Result<(), SerializationError> {
        self.message_generators.check()?;
        self.blinding_generator.check()
    }
}

impl<C: JoltCurve> CanonicalDeserialize for PedersenGenerators<C> {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let message_generators =
            Vec::<C::G1>::deserialize_with_mode(&mut reader, compress, validate)?;
        let blinding_generator = C::G1::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(Self::new(message_generators, blinding_generator))
    }
}

impl<C: JoltCurve> PedersenGenerators<C> {
    pub fn new(message_generators: Vec<C::G1>, blinding_generator: C::G1) -> Self {
        assert!(
            !message_generators.is_empty(),
            "Need at least one generator"
        );
        let mut affine_bases: Vec<C::G1Affine> =
            message_generators.iter().map(C::g1_to_affine).collect();
        affine_bases.push(C::g1_to_affine(&blinding_generator));
        Self {
            message_generators,
            blinding_generator,
            affine_bases,
        }
    }

    /// Single MSM including blinding — no separate scalar_mul + add.
    pub fn commit(&self, coeffs: &[C::F], blinding: &C::F) -> C::G1 {
        let n = coeffs.len();
        assert!(
            n <= self.message_generators.len(),
            "Too many coefficients: {} > {}",
            n,
            self.message_generators.len()
        );

        // Build combined scalar vector: [coeffs..., blinding]
        // Uses affine_bases[0..n] for message gens, affine_bases[last] for blinding
        let blinding_affine_idx = self.message_generators.len();
        let mut combined_bases = Vec::with_capacity(n + 1);
        combined_bases.extend_from_slice(&self.affine_bases[..n]);
        combined_bases.push(self.affine_bases[blinding_affine_idx]);

        let mut combined_scalars = Vec::with_capacity(n + 1);
        combined_scalars.extend_from_slice(coeffs);
        combined_scalars.push(*blinding);

        C::g1_affine_msm(&combined_bases, &combined_scalars)
    }

    pub fn commit_chunked<R: CryptoRngCore>(
        &self,
        values: &[C::F],
        rng: &mut R,
    ) -> Vec<(C::G1, C::F)> {
        values
            .chunks(self.message_generators.len())
            .map(|chunk| {
                let blinding = C::F::random(rng);
                let commitment = self.commit(chunk, &blinding);
                (commitment, blinding)
            })
            .collect()
    }

    pub fn verify(&self, commitment: &C::G1, coeffs: &[C::F], blinding: &C::F) -> bool {
        let expected = self.commit(coeffs, blinding);
        *commitment == expected
    }
}

#[cfg(test)]
impl PedersenGenerators<crate::curve::Bn254Curve> {
    /// Test-only: derives generators from hash-to-curve. Production code uses Dory URS.
    pub fn deterministic(count: usize) -> Self {
        use ark_bn254::G1Projective;
        use ark_std::UniformRand;
        use rand_chacha::ChaCha20Rng;
        use rand_core::SeedableRng;
        use sha3::Digest;

        let hash_to_g1 = |domain: &[u8]| -> crate::curve::Bn254G1 {
            let hash = sha3::Sha3_256::digest(domain);
            let mut rng = ChaCha20Rng::from_seed(hash.into());
            crate::curve::Bn254G1(G1Projective::rand(&mut rng))
        };

        let generators = (0..count)
            .map(|i| {
                let mut domain = b"jolt_pedersen_msg_gen_v1_".to_vec();
                domain.extend_from_slice(&(i as u64).to_le_bytes());
                hash_to_g1(&domain)
            })
            .collect();
        let blinding_generator = hash_to_g1(b"jolt_pedersen_blinding_h2c_v1");
        Self::new(generators, blinding_generator)
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
    use crate::curve::{Bn254Curve, JoltGroupElement};
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
