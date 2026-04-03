use std::{collections::HashMap, fs::File, io, path::PathBuf};

use serde::{Deserialize, Serialize};
use tracer::{
    instruction::{Cycle, Instruction},
    JoltDevice,
};

use crate::field::JoltField;

#[derive(Serialize, Deserialize)]
pub struct ProgramSummary {
    pub trace: Vec<Cycle>,
    pub bytecode: Vec<Instruction>,
    pub memory_init: Vec<(u64, u8)>,
    pub io_device: JoltDevice,
}

/// Detailed analysis result from tracing a guest program.
pub struct AnalysisReport {
    pub total_cycles: usize,
    pub padded_trace_length: usize,
    pub unique_bytecode_instructions: usize,
    pub instruction_counts: Vec<(&'static str, usize)>,
    pub panicked: bool,
}

impl ProgramSummary {
    pub fn trace_len(&self) -> usize {
        self.trace.len()
    }

    pub fn analyze<F: JoltField>(&self) -> Vec<(&'static str, usize)> {
        let mut counts = HashMap::<&'static str, usize>::new();
        for cycle in self.trace.iter() {
            let instruction_name: &'static str = cycle.into();
            if let Some(count) = counts.get(instruction_name) {
                counts.insert(instruction_name, count + 1);
            } else {
                counts.insert(instruction_name, 1);
            }
        }

        let mut counts: Vec<_> = counts.into_iter().collect();
        counts.sort_by_key(|v| v.1);
        counts.reverse();

        counts
    }

    pub fn detailed_analyze<F: JoltField>(&self) -> AnalysisReport {
        let instruction_counts = self.analyze::<F>();
        let total_cycles = self.trace.len();
        let padded_trace_length = total_cycles.next_power_of_two();
        let unique_bytecode_instructions = self.bytecode.len();
        let panicked = self.io_device.panic;

        AnalysisReport {
            total_cycles,
            padded_trace_length,
            unique_bytecode_instructions,
            instruction_counts,
            panicked,
        }
    }

    pub fn write_to_file(self, path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(path)?;
        let data = bincode::serde::encode_to_vec(&self, bincode::config::standard())?;
        io::Write::write_all(&mut file, &data)?;
        Ok(())
    }
}
