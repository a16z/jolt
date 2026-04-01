use std::sync::Arc;

use super::{AbstractObjective, Direction, MeasurementError, ObjectiveEntry};
use crate::{serialize_proof, ProverPreprocessing, TestCase};

inventory::submit! {
    ObjectiveEntry {
        name: "proof_size",
        direction: Direction::Minimize,
        build: |setup, inputs| Box::new(ProofSizeObjective::new(
            setup.test_case.clone(), setup.prover_preprocessing.clone(), inputs,
        )),
    }
}

/// Measures serialized proof size in bytes.
pub struct ProofSizeObjective {
    pub test_case: Arc<TestCase>,
    pub prover_preprocessing: Arc<ProverPreprocessing>,
    pub inputs: Vec<u8>,
}

impl ProofSizeObjective {
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

impl AbstractObjective for ProofSizeObjective {
    fn name(&self) -> &str {
        "proof_size"
    }

    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        let (proof, _io) = self
            .test_case
            .prove(&self.prover_preprocessing, &self.inputs);
        let bytes = serialize_proof(&proof);
        Ok(bytes.len() as f64)
    }

    fn recommended_samples(&self) -> usize {
        1
    }

    fn direction(&self) -> Direction {
        Direction::Minimize
    }
}
