use std::sync::Arc;

use super::{AbstractObjective, Direction, MeasurementError, ObjectiveEntry};
use crate::TestCase;

inventory::submit! {
    ObjectiveEntry {
        name: "guest_cycle_count",
        direction: Direction::Minimize,
        build: |setup, inputs| Box::new(GuestCycleCountObjective::new(
            setup.test_case.clone(), inputs,
        )),
    }
}

/// Measures guest instruction cycle count via program tracing.
pub struct GuestCycleCountObjective {
    pub test_case: Arc<TestCase>,
    pub inputs: Vec<u8>,
}

impl GuestCycleCountObjective {
    pub fn new(test_case: Arc<TestCase>, inputs: Vec<u8>) -> Self {
        Self { test_case, inputs }
    }
}

impl AbstractObjective for GuestCycleCountObjective {
    fn name(&self) -> &str {
        "guest_cycle_count"
    }

    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        let program = self.test_case.make_program();
        let (_lazy_trace, trace, _memory, _io) = program.trace(&self.inputs, &[], &[]);
        Ok(trace.len() as f64)
    }

    fn recommended_samples(&self) -> usize {
        1
    }

    fn direction(&self) -> Direction {
        Direction::Minimize
    }
}
