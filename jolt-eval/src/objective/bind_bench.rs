use ark_bn254::Fr;
use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;

use super::PerfObjective;

type Challenge = <Fr as JoltField>::Challenge;

/// Number of variables for the benchmark polynomial (2^NUM_VARS evaluations).
const NUM_VARS: usize = 20;

pub struct BindSetup {
    /// Original evaluations (cloned into a fresh poly each iteration).
    pub evals: Vec<Fr>,
    pub challenge: Challenge,
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
        let mut rng = rand::thread_rng();
        BindSetup {
            evals: (0..1 << NUM_VARS)
                .map(|_| Fr::random(&mut rng))
                .collect(),
            challenge: Challenge::random(&mut rng),
        }
    }

    fn run(&self, setup: &mut BindSetup) {
        let mut poly = DensePolynomial::new(setup.evals.clone());
        poly.bind_parallel(setup.challenge, BindingOrder::LowToHigh);
        std::hint::black_box(poly);
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
        let mut rng = rand::thread_rng();
        BindSetup {
            evals: (0..1 << NUM_VARS)
                .map(|_| Fr::random(&mut rng))
                .collect(),
            challenge: Challenge::random(&mut rng),
        }
    }

    fn run(&self, setup: &mut BindSetup) {
        let mut poly = DensePolynomial::new(setup.evals.clone());
        poly.bind_parallel(setup.challenge, BindingOrder::HighToLow);
        std::hint::black_box(poly);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bind_low_to_high_runs() {
        let obj = BindLowToHighObjective;
        let mut setup = obj.setup();
        obj.run(&mut setup);
    }

    #[test]
    fn bind_high_to_low_runs() {
        let obj = BindHighToLowObjective;
        let mut setup = obj.setup();
        obj.run(&mut setup);
    }
}
