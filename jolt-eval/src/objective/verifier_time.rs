use std::sync::Arc;
use std::time::Instant;

use super::{AbstractObjective, Direction, MeasurementError, ObjectiveEntry};
use crate::{ProverPreprocessing, TestCase, VerifierPreprocessing};

inventory::submit! {
    ObjectiveEntry {
        name: "verifier_time",
        direction: Direction::Minimize,
        needs_guest: true,
        build: |s, inputs| { let setup = s.unwrap(); Box::new(VerifierTimeObjective::new(
            setup.test_case.clone(), setup.prover_preprocessing.clone(),
            setup.verifier_preprocessing.clone(), inputs,
            )) },
    }
}

/// Measures wall-clock verifier time in seconds.
pub struct VerifierTimeObjective {
    pub test_case: Arc<TestCase>,
    pub prover_preprocessing: Arc<ProverPreprocessing>,
    pub verifier_preprocessing: Arc<VerifierPreprocessing>,
    pub inputs: Vec<u8>,
}

impl VerifierTimeObjective {
    pub fn new(
        test_case: Arc<TestCase>,
        prover_preprocessing: Arc<ProverPreprocessing>,
        verifier_preprocessing: Arc<VerifierPreprocessing>,
        inputs: Vec<u8>,
    ) -> Self {
        Self {
            test_case,
            prover_preprocessing,
            verifier_preprocessing,
            inputs,
        }
    }
}

impl AbstractObjective for VerifierTimeObjective {
    fn name(&self) -> &str {
        "verifier_time"
    }

    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        let (proof, io_device) = self
            .test_case
            .prove(&self.prover_preprocessing, &self.inputs);

        let start = Instant::now();
        TestCase::verify(&self.verifier_preprocessing, proof, &io_device)
            .map_err(|e| MeasurementError::new(format!("Verification failed: {e}")))?;
        Ok(start.elapsed().as_secs_f64())
    }

    fn recommended_samples(&self) -> usize {
        5
    }

    fn regression_threshold(&self) -> Option<f64> {
        Some(0.05)
    }

    fn direction(&self) -> Direction {
        Direction::Minimize
    }
}
