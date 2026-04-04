use ark_bn254::Fr;
use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;

use crate::objective::{Objective, OptimizationObjective, PerformanceObjective};

pub const BIND_LOW_TO_HIGH: OptimizationObjective =
    OptimizationObjective::Performance(PerformanceObjective::BindLowToHigh(BindLowToHighObjective));
pub const BIND_HIGH_TO_LOW: OptimizationObjective =
    OptimizationObjective::Performance(PerformanceObjective::BindHighToLow(BindHighToLowObjective));

type Challenge = <Fr as JoltField>::Challenge;

const NUM_VARS: usize = 20;

pub struct BindSetup {
    pub poly: DensePolynomial<Fr>,
    pub challenge: Challenge,
}

struct BindShared {
    evals: Vec<Fr>,
    challenge: Challenge,
}

impl BindShared {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            evals: (0..1 << NUM_VARS).map(|_| Fr::random(&mut rng)).collect(),
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
#[derive(Clone, Copy, PartialEq, Hash)]
pub struct BindLowToHighObjective;

impl Objective for BindLowToHighObjective {
    type Setup = BindSetup;

    fn name(&self) -> &str {
        "bind_parallel_low_to_high"
    }

    fn description(&self) -> String {
        "Wall-clock time of DensePolynomial::bind_parallel with LowToHigh binding (2^20 evaluations)".to_string()
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
            .bind_parallel(setup.challenge, BindingOrder::LowToHigh);
        std::hint::black_box(&setup.poly);
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}

/// Benchmark `DensePolynomial::bind_parallel` with `HighToLow` binding.
#[derive(Clone, Copy, PartialEq, Hash)]
pub struct BindHighToLowObjective;

impl Objective for BindHighToLowObjective {
    type Setup = BindSetup;

    fn name(&self) -> &str {
        "bind_parallel_high_to_low"
    }

    fn description(&self) -> String {
        "Wall-clock time of DensePolynomial::bind_parallel with HighToLow binding (2^20 evaluations)".to_string()
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

    fn units(&self) -> Option<&str> {
        Some("s")
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
