//! `Bn254GT::scalar_mul` — GT exponentiation hot path in Dory opening verify.

use jolt_crypto::{Bn254, Bn254GT, JoltGroup, PairingGroup};
use jolt_field::{Field, Fr};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use crate::objective::{Objective, OptimizationObjective, PerformanceObjective};

pub const JOLT_CRYPTO_GT_SCALAR_MUL: OptimizationObjective = OptimizationObjective::Performance(
    PerformanceObjective::JoltCryptoGtScalarMul(JoltCryptoGtScalarMulObjective),
);

pub struct GtScalarMulSetup {
    pub base: Bn254GT,
    pub scalar: Fr,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct JoltCryptoGtScalarMulObjective;

impl Objective for JoltCryptoGtScalarMulObjective {
    type Setup = GtScalarMulSetup;

    fn name(&self) -> &str {
        "jolt_crypto_gt_scalar_mul"
    }

    fn description(&self) -> String {
        "Wall-clock time of Bn254GT::scalar_mul with a random non-trivial GT element and random scalar"
            .to_string()
    }

    fn setup(&self) -> GtScalarMulSetup {
        let mut rng = ChaCha20Rng::seed_from_u64(2);
        let base = Bn254::pairing(&Bn254::g1_generator(), &Bn254::g2_generator());
        GtScalarMulSetup {
            base,
            scalar: Fr::random(&mut rng),
        }
    }

    fn run(&self, setup: GtScalarMulSetup) {
        let result = setup.base.scalar_mul(&setup.scalar);
        std::hint::black_box(&result);
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}
