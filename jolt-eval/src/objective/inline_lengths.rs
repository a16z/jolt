use std::sync::Arc;

use super::{AbstractObjective, Direction, MeasurementError, ObjectiveEntry};
use crate::TestCase;

inventory::submit! {
    ObjectiveEntry {
        name: "inline_lengths",
        direction: Direction::Maximize,
        build: |setup, _inputs| Box::new(InlineLengthsObjective::new(setup.test_case.clone())),
    }
}

/// Measures total virtual/inline sequence length in the decoded bytecode.
///
/// Inline sequences replace guest-side computation with constraint-native
/// implementations, so their total length reflects how much of the program
/// is handled by optimized inline instructions.
pub struct InlineLengthsObjective {
    pub test_case: Arc<TestCase>,
}

impl InlineLengthsObjective {
    pub fn new(test_case: Arc<TestCase>) -> Self {
        Self { test_case }
    }
}

impl AbstractObjective for InlineLengthsObjective {
    fn name(&self) -> &str {
        "inline_lengths"
    }

    fn collect_measurement(&self) -> Result<f64, MeasurementError> {
        let program = self.test_case.make_program();
        let (instructions, _memory_init, _program_size, _entry) = program.decode();

        // Count INLINE instructions (optimized constraint-native operations)
        let total_inline_length: usize = instructions
            .iter()
            .filter(|instr| matches!(instr, tracer::instruction::Instruction::INLINE(_)))
            .count();

        Ok(total_inline_length as f64)
    }

    fn recommended_samples(&self) -> usize {
        1
    }

    fn direction(&self) -> Direction {
        // More inlines generally means more efficient execution
        Direction::Maximize
    }
}
