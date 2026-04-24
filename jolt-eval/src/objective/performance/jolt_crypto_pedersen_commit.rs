//! `Pedersen::<Bn254G1>::commit` with 1024 generators — BlindFold hot path.

use jolt_crypto::{Bn254, Bn254G1, Pedersen, PedersenSetup, VectorCommitment};
use jolt_field::{Field, Fr};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use crate::objective::{Objective, OptimizationObjective, PerformanceObjective};

pub const JOLT_CRYPTO_PEDERSEN_COMMIT_1024: OptimizationObjective =
    OptimizationObjective::Performance(PerformanceObjective::JoltCryptoPedersenCommit1024(
        JoltCryptoPedersenCommit1024Objective,
    ));

const COMMIT_SIZE: usize = 1024;

pub struct PedersenCommitSetup {
    pub setup: PedersenSetup<Bn254G1>,
    pub values: Vec<Fr>,
    pub blinding: Fr,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct JoltCryptoPedersenCommit1024Objective;

impl Objective for JoltCryptoPedersenCommit1024Objective {
    type Setup = PedersenCommitSetup;

    fn name(&self) -> &str {
        "jolt_crypto_pedersen_commit_1024"
    }

    fn description(&self) -> String {
        "Wall-clock time of Pedersen::<Bn254G1>::commit with 1024 generators + 1024 values + blinding"
            .to_string()
    }

    fn setup(&self) -> PedersenCommitSetup {
        let mut rng = ChaCha20Rng::seed_from_u64(3);
        let gens: Vec<Bn254G1> = (0..COMMIT_SIZE)
            .map(|_| Bn254::random_g1(&mut rng))
            .collect();
        let blinding_gen = Bn254::random_g1(&mut rng);
        let setup = PedersenSetup::new(gens, blinding_gen);
        let values: Vec<Fr> = (0..COMMIT_SIZE).map(|_| Fr::random(&mut rng)).collect();
        let blinding = Fr::random(&mut rng);
        PedersenCommitSetup {
            setup,
            values,
            blinding,
        }
    }

    fn run(&self, setup: PedersenCommitSetup) {
        let result = Pedersen::<Bn254G1>::commit(&setup.setup, &setup.values, &setup.blinding);
        std::hint::black_box(&result);
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}
