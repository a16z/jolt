//! GLV-vector-op invariants for `jolt-crypto`.
//!
//! Covers `vector_add_scalar_mul_{g1,g2}` and `vector_scalar_mul_add_gamma_{g1,g2}`
//! plus `glv_four_scalar_mul`.

use arbitrary::{Arbitrary, Unstructured};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use jolt_crypto::ec::bn254::glv;
use jolt_crypto::{Bn254, Bn254G1, Bn254G2, JoltGroup};
use jolt_field::{Field, Fr};

use super::{CheckError, Invariant, InvariantViolation};

const MAX_LEN: usize = 32;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct GlvVectorInput {
    pub seed: u64,
    pub len: u8,
}

impl<'a> Arbitrary<'a> for GlvVectorInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self {
            seed: u.arbitrary()?,
            len: u.int_in_range(0u8..=MAX_LEN as u8)?,
        })
    }
}

#[jolt_eval_macros::invariant(Test, Fuzz)]
#[derive(Default)]
pub struct JoltCryptoGlvVectorInvariant;

impl Invariant for JoltCryptoGlvVectorInvariant {
    type Setup = ();
    type Input = GlvVectorInput;

    fn name(&self) -> &str {
        "jolt_crypto_glv_vector_matches_naive"
    }

    fn description(&self) -> String {
        "GLV vector operations (vector_add_scalar_mul_{g1,g2}, \
         vector_scalar_mul_add_gamma_{g1,g2}, glv_four_scalar_mul) must \
         match the naive scalar_mul/add formulation for all inputs."
            .to_string()
    }

    fn setup(&self) {}

    fn check(&self, _setup: &(), input: GlvVectorInput) -> Result<(), CheckError> {
        let n = input.len as usize;
        let mut rng = ChaCha20Rng::seed_from_u64(input.seed);
        let scalar = Fr::random(&mut rng);

        // G1: vector_add_scalar_mul_g1
        let generators: Vec<Bn254G1> = (0..n).map(|_| Bn254::random_g1(&mut rng)).collect();
        let initial: Vec<Bn254G1> = (0..n).map(|_| Bn254::random_g1(&mut rng)).collect();
        let mut result = initial.clone();
        glv::vector_add_scalar_mul_g1(&mut result, &generators, scalar);
        for i in 0..n {
            let expected = initial[i] + generators[i].scalar_mul(&scalar);
            if result[i] != expected {
                return Err(CheckError::Violation(InvariantViolation::with_details(
                    "vector_add_scalar_mul_g1 mismatch",
                    format!("i={i}, len={n}"),
                )));
            }
        }

        // G1: vector_scalar_mul_add_gamma_g1
        let gamma_g1: Vec<Bn254G1> = (0..n).map(|_| Bn254::random_g1(&mut rng)).collect();
        let initial_g1: Vec<Bn254G1> = (0..n).map(|_| Bn254::random_g1(&mut rng)).collect();
        let mut result = initial_g1.clone();
        glv::vector_scalar_mul_add_gamma_g1(&mut result, scalar, &gamma_g1);
        for i in 0..n {
            let expected = initial_g1[i].scalar_mul(&scalar) + gamma_g1[i];
            if result[i] != expected {
                return Err(CheckError::Violation(InvariantViolation::with_details(
                    "vector_scalar_mul_add_gamma_g1 mismatch",
                    format!("i={i}, len={n}"),
                )));
            }
        }

        // G2: derive bases from generator (random_g2 helper is not public).
        let g2 = Bn254::g2_generator();
        let generators_g2: Vec<Bn254G2> = (0..n)
            .map(|_| g2.scalar_mul(&Fr::random(&mut rng)))
            .collect();
        let initial_g2: Vec<Bn254G2> = (0..n)
            .map(|_| g2.scalar_mul(&Fr::random(&mut rng)))
            .collect();
        let mut result = initial_g2.clone();
        glv::vector_add_scalar_mul_g2(&mut result, &generators_g2, scalar);
        for i in 0..n {
            let expected = initial_g2[i] + generators_g2[i].scalar_mul(&scalar);
            if result[i] != expected {
                return Err(CheckError::Violation(InvariantViolation::with_details(
                    "vector_add_scalar_mul_g2 mismatch",
                    format!("i={i}, len={n}"),
                )));
            }
        }

        // G2: vector_scalar_mul_add_gamma_g2
        let gamma_g2: Vec<Bn254G2> = (0..n)
            .map(|_| g2.scalar_mul(&Fr::random(&mut rng)))
            .collect();
        let initial_g2: Vec<Bn254G2> = (0..n)
            .map(|_| g2.scalar_mul(&Fr::random(&mut rng)))
            .collect();
        let mut result = initial_g2.clone();
        glv::vector_scalar_mul_add_gamma_g2(&mut result, scalar, &gamma_g2);
        for i in 0..n {
            let expected = initial_g2[i].scalar_mul(&scalar) + gamma_g2[i];
            if result[i] != expected {
                return Err(CheckError::Violation(InvariantViolation::with_details(
                    "vector_scalar_mul_add_gamma_g2 mismatch",
                    format!("i={i}, len={n}"),
                )));
            }
        }

        // G2: glv_four_scalar_mul
        let points: Vec<Bn254G2> = (0..n)
            .map(|_| g2.scalar_mul(&Fr::random(&mut rng)))
            .collect();
        let results = glv::glv_four_scalar_mul(scalar, &points);
        for (i, (r, p)) in results.iter().zip(points.iter()).enumerate() {
            let expected = p.scalar_mul(&scalar);
            if *r != expected {
                return Err(CheckError::Violation(InvariantViolation::with_details(
                    "glv_four_scalar_mul mismatch",
                    format!("i={i}, len={n}"),
                )));
            }
        }

        // G1: fixed_base_vector_msm_g1
        if n > 0 {
            let base = Bn254::random_g1(&mut rng);
            let scalars: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            let results = glv::fixed_base_vector_msm_g1(&base, &scalars);
            for (i, (r, s)) in results.iter().zip(scalars.iter()).enumerate() {
                let expected = base.scalar_mul(s);
                if *r != expected {
                    return Err(CheckError::Violation(InvariantViolation::with_details(
                        "fixed_base_vector_msm_g1 mismatch",
                        format!("i={i}, len={n}"),
                    )));
                }
            }
        }

        Ok(())
    }

    fn seed_corpus(&self) -> Vec<GlvVectorInput> {
        [0u8, 1, 2, 3, 4, 8, 16, 17, 32]
            .into_iter()
            .enumerate()
            .map(|(i, len)| GlvVectorInput {
                seed: i as u64 + 100,
                len,
            })
            .collect()
    }
}
