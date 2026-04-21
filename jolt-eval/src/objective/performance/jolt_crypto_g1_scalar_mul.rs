//! `Bn254G1::scalar_mul` — GLV-path canary.

use jolt_crypto::{Bn254, Bn254G1, JoltGroup};
use jolt_field::{Field, Fr};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use crate::objective::{Objective, OptimizationObjective, PerformanceObjective};

pub const JOLT_CRYPTO_G1_SCALAR_MUL: OptimizationObjective = OptimizationObjective::Performance(
    PerformanceObjective::JoltCryptoG1ScalarMul(JoltCryptoG1ScalarMulObjective),
);

pub struct G1ScalarMulSetup {
    pub base: Bn254G1,
    pub scalar: Fr,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct JoltCryptoG1ScalarMulObjective;

impl Objective for JoltCryptoG1ScalarMulObjective {
    type Setup = G1ScalarMulSetup;

    fn name(&self) -> &str {
        "jolt_crypto_g1_scalar_mul"
    }

    fn description(&self) -> String {
        "Wall-clock time of Bn254G1::scalar_mul with a random G1 point and random scalar"
            .to_string()
    }

    fn setup(&self) -> G1ScalarMulSetup {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        G1ScalarMulSetup {
            base: Bn254::random_g1(&mut rng),
            scalar: Fr::random(&mut rng),
        }
    }

    fn run(&self, setup: G1ScalarMulSetup) {
        let result = setup.base.scalar_mul(&setup.scalar);
        std::hint::black_box(&result);
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}
