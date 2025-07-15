use std::{collections::HashMap, fs::File, io, path::PathBuf};

use serde::{Deserialize, Serialize};
use tracer::{
    instruction::{RV32IMCycle, RV32IMInstruction},
    JoltDevice,
};

use crate::field::JoltField;

#[derive(Serialize, Deserialize)]
pub struct ProgramSummary {
    pub trace: Vec<RV32IMCycle>,
    pub bytecode: Vec<RV32IMInstruction>,
    pub memory_init: Vec<(u64, u8)>,
    pub io_device: JoltDevice,
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

    pub fn write_to_file(self, path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(path)?;
        let data = bincode::serialize(&self)?;
        io::Write::write_all(&mut file, &data)?;
        Ok(())
    }

    pub fn write_trace_analysis<F: JoltField>(
        &self,
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let instruction_counts = self.analyze::<F>();

        let mut output = String::new();

        // Basic statistics
        output.push_str("=== BASIC STATISTICS ===\n");
        output.push_str(&format!("Trace length: {}\n", self.trace_len()));
        output.push_str(&format!("Bytecode length: {}\n", self.bytecode.len()));
        output.push_str(&format!(
            "Memory init entries: {}\n",
            self.memory_init.len()
        ));
        output.push_str("\n");

        // Instruction frequency analysis
        output.push_str("=== INSTRUCTION FREQUENCY ANALYSIS ===\n");
        output.push_str(&format!(
            "{:<20} | {:>10} | {:>8}\n",
            "Instruction", "Count", "Percent"
        ));
        output.push_str(&format!("{:-<20}-+-{:->10}-+-{:->8}\n", "", "", ""));

        for (instruction, count) in instruction_counts.iter() {
            let percentage = (*count as f64 / self.trace_len() as f64) * 100.0;
            output.push_str(&format!(
                "{:<20} | {:>10} | {:>7.2}%\n",
                instruction, count, percentage
            ));
        }
        output.push_str("\n");

        // All bytecode instructions
        output.push_str("=== ALL BYTECODE INSTRUCTIONS ===\n");
        for (i, instruction) in self.bytecode.iter().enumerate() {
            output.push_str(&format!("{:4}: {:?}\n", i, instruction));
        }
        output.push_str("\n");

        // All trace instructions
        output.push_str("=== ALL TRACE INSTRUCTIONS ===\n");
        for (i, cycle) in self.trace.iter().enumerate() {
            output.push_str(&format!("{:4}: {:?}\n", i, cycle));
        }

        // Write to file
        std::fs::write(filename, output)?;
        Ok(())
    }
}
