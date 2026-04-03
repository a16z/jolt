use ark_bn254::Fr;
use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;

use super::PerfObjective;

type Challenge = <Fr as JoltField>::Challenge;

/// Number of variables for the benchmark polynomial (2^NUM_VARS evaluations).
const NUM_VARS: usize = 20;

/// Per-iteration state: a fresh polynomial and a challenge to bind with.
pub struct BindSetup {
    pub poly: DensePolynomial<Fr>,
    pub challenge: Challenge,
}

/// Shared state used to produce per-iteration setups cheaply.
struct BindShared {
    evals: Vec<Fr>,
    challenge: Challenge,
}

impl BindShared {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            evals: (0..1 << NUM_VARS)
                .map(|_| Fr::random(&mut rng))
                .collect(),
            challenge: Challenge::random(&mut rng),
        }
    }

    fn make_setup(&self) -> BindSetup {
        BindSetup {
            poly: DensePolynomial::new(self.evals.clone()),
            challenge: self.challenge,
        }
    }
}

/// Benchmark `DensePolynomial::bind_parallel` with `LowToHigh` binding.
#[derive(Default)]
pub struct BindLowToHighObjective;

impl BindLowToHighObjective {
    pub const NAME: &str = "bind_parallel_low_to_high";
}

impl PerfObjective for BindLowToHighObjective {
    type Setup = BindSetup;

    fn name(&self) -> &str {
        Self::NAME
    }

    fn setup(&self) -> BindSetup {
        // Thread-local shared state so we only generate random evals once.
        thread_local! {
            static SHARED: BindShared = BindShared::new();
        }
        SHARED.with(|s| s.make_setup())
    }

    fn run(&self, mut setup: BindSetup) {
        setup
            .poly
            .bind_parallel(setup.challenge, BindingOrder::LowToHigh);
        std::hint::black_box(&setup.poly);
    }
}

/// Benchmark `DensePolynomial::bind_parallel` with `HighToLow` binding.
#[derive(Default)]
pub struct BindHighToLowObjective;

impl BindHighToLowObjective {
    pub const NAME: &str = "bind_parallel_high_to_low";
}

impl PerfObjective for BindHighToLowObjective {
    type Setup = BindSetup;

    fn name(&self) -> &str {
        Self::NAME
    }

    fn setup(&self) -> BindSetup {
        thread_local! {
            static SHARED: BindShared = BindShared::new();
        }
        SHARED.with(|s| s.make_setup())
    }

    fn run(&self, mut setup: BindSetup) {
        setup
            .poly
            .bind_parallel(setup.challenge, BindingOrder::HighToLow);
        std::hint::black_box(&setup.poly);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bind_low_to_high_runs() {
        let obj = BindLowToHighObjective;
        let setup = obj.setup();
        obj.run(setup);
    }

    #[test]
    fn bind_high_to_low_runs() {
        let obj = BindHighToLowObjective;
        let setup = obj.setup();
        obj.run(setup);
    }
}
