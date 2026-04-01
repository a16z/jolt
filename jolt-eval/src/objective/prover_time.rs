use std::sync::Arc;
use std::time::Instant;

use super::{AbstractObjective, Direction, MeasurementError, ObjectiveEntry};
use crate::{ProverPreprocessing, TestCase};

inventory::submit! {
    ObjectiveEntry {
        name: "prover_time",
        direction: Direction::Minimize,
        needs_guest: true,
        build: |s, inputs| { let setup = s.unwrap(); Box::new(ProverTimeObjective::new(
            setup.test_case.clone(), setup.prover_preprocessing.clone(), inputs,
            )) },
    }
}

/// Measures wall-clock prover time in seconds.
pub struct ProverTimeObjective {
    pub test_case: Arc<TestCase>,
    pub prover_preprocessing: Arc<ProverPreprocessing>,
    pub inputs: Vec<u8>,
}

impl ProverTimeObjective {
    pub fn new(
        test_case: Arc<TestCase>,
        prover_preprocessing: Arc<ProverPreprocessing>,
        inputs: Vec<u8>,
    ) -> Self {
        Self {
            test_case,
            prover_preprocessing,
            inputs,
        }
    }
}

impl AbstractObjective for ProverTimeObjective {
    fn name(&self) -> &str {
        "prover_time"
    }

    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        let start = Instant::now();
        let (_proof, _io) = self
            .test_case
            .prove(&self.prover_preprocessing, &self.inputs);
        Ok(start.elapsed().as_secs_f64())
    }

    fn recommended_samples(&self) -> usize {
        3
    }

    fn regression_threshold(&self) -> Option<f64> {
        Some(0.05)
    }

    fn direction(&self) -> Direction {
        Direction::Minimize
    }
}
