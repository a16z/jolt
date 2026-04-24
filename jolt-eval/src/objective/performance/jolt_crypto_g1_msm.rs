//! `Bn254G1::msm` with 1024 bases/scalars — dominant Dory tier-1 cost.

use jolt_crypto::{Bn254, Bn254G1, JoltGroup};
use jolt_field::{Field, Fr};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use crate::objective::{Objective, OptimizationObjective, PerformanceObjective};

pub const JOLT_CRYPTO_G1_MSM_1024: OptimizationObjective = OptimizationObjective::Performance(
    PerformanceObjective::JoltCryptoG1Msm1024(JoltCryptoG1Msm1024Objective),
);

const MSM_SIZE: usize = 1024;

pub struct G1MsmSetup {
    pub bases: Vec<Bn254G1>,
    pub scalars: Vec<Fr>,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct JoltCryptoG1Msm1024Objective;

impl Objective for JoltCryptoG1Msm1024Objective {
    type Setup = G1MsmSetup;

    fn name(&self) -> &str {
        "jolt_crypto_g1_msm_1024"
    }

    fn description(&self) -> String {
        "Wall-clock time of Bn254G1::msm with 1024 random bases and 1024 random scalars".to_string()
    }

    fn setup(&self) -> G1MsmSetup {
        let mut rng = ChaCha20Rng::seed_from_u64(0x1024);
        G1MsmSetup {
            bases: (0..MSM_SIZE).map(|_| Bn254::random_g1(&mut rng)).collect(),
            scalars: (0..MSM_SIZE).map(|_| Fr::random(&mut rng)).collect(),
        }
    }

    fn run(&self, setup: G1MsmSetup) {
        let result = Bn254G1::msm(&setup.bases, &setup.scalars);
        std::hint::black_box(&result);
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}
