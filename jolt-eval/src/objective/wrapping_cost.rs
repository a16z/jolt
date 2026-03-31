use std::sync::Arc;

use super::{AbstractObjective, Direction, MeasurementError};
use crate::{ProverPreprocessing, TestCase};

/// Measures the "wrapping cost" as the total number of constraints in the R1CS.
///
/// This is derived from the preprocessing data which encodes the constraint
/// structure. Lower constraint counts mean cheaper verification.
pub struct WrappingCostObjective {
    pub test_case: Arc<TestCase>,
    pub prover_preprocessing: Arc<ProverPreprocessing>,
}

impl WrappingCostObjective {
    pub fn new(test_case: Arc<TestCase>, prover_preprocessing: Arc<ProverPreprocessing>) -> Self {
        Self {
            test_case,
            prover_preprocessing,
        }
    }
}

impl AbstractObjective for WrappingCostObjective {
    fn name(&self) -> &str {
        "wrapping_cost"
    }

    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        // The padded trace length from preprocessing reflects the constraint
        // system size, which is the dominant factor in wrapping cost.
        let max_padded = self.prover_preprocessing.shared.max_padded_trace_length;
        Ok(max_padded as f64)
    }

    fn recommended_samples(&self) -> usize {
        1
    }

    fn direction(&self) -> Direction {
        Direction::Minimize
    }
}
