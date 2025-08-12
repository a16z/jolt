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
        let mut between_ecalls = false;
        
        for cycle in self.trace.iter() {
            let instruction_name: &'static str = cycle.into();
            
            if instruction_name == "ECALL" {
                // Toggle the flag - we're either entering or exiting a segment
                between_ecalls = !between_ecalls;
                continue; // Don't count the ECALL itself
            }
            
            if between_ecalls {
                *counts.entry(instruction_name).or_insert(0) += 1;
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

    pub fn write_trace_analysis_ecalls<F: JoltField>(
        &self,
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let instruction_counts = self.analyze::<F>();

        let mut output = String::new();

        // Count instructions between ECALLs (all segments)
        let mut trace_between_ecalls_count = 0;
        let mut between_ecalls = false;
        
        for cycle in self.trace.iter() {
            let instruction_name: &'static str = cycle.into();
            if instruction_name == "ECALL" {
                between_ecalls = !between_ecalls;
                continue;
            }
            if between_ecalls {
                trace_between_ecalls_count += 1;
            }
        }
        
        // Count bytecode instructions between ECALLs (all segments)
        let mut bytecode_between_ecalls_count = 0;
        let mut between_ecalls = false;
        
        for instruction in self.bytecode.iter() {
            let instr_str = format!("{:?}", instruction);
            if instr_str.contains("ECALL") {
                between_ecalls = !between_ecalls;
                continue;
            }
            if between_ecalls {
                bytecode_between_ecalls_count += 1;
            }
        }

        // Basic statistics
        output.push_str("=== BASIC STATISTICS ===\n");
        output.push_str(&format!("Total trace length: {}\n", self.trace_len()));
        output.push_str(&format!("Trace instructions between ECALLs: {}\n", trace_between_ecalls_count));
        output.push_str(&format!("Total bytecode length: {}\n", self.bytecode.len()));
        output.push_str(&format!("Bytecode instructions between ECALLs: {}\n", bytecode_between_ecalls_count));
        output.push_str(&format!(
            "Memory init entries: {}\n",
            self.memory_init.len()
        ));
        output.push_str("\n");

        // Instruction frequency analysis (between ECALLs)
        output.push_str("=== INSTRUCTION FREQUENCY ANALYSIS (BETWEEN ECALLS) ===\n");
        output.push_str(&format!(
            "{:<20} | {:>10} | {:>8}\n",
            "Instruction", "Count", "Percent"
        ));
        output.push_str(&format!("{:-<20}-+-{:->10}-+-{:->8}\n", "", "", ""));

        for (instruction, count) in instruction_counts.iter() {
            let percentage = if trace_between_ecalls_count > 0 {
                (*count as f64 / trace_between_ecalls_count as f64) * 100.0
            } else {
                0.0
            };
            output.push_str(&format!(
                "{:<20} | {:>10} | {:>7.2}%\n",
                instruction, count, percentage
            ));
        }
        output.push_str("\n");

        // All bytecode instructions between ECALLs (all segments)
        output.push_str("=== ALL BYTECODE INSTRUCTIONS BETWEEN ECALLS ===\n");
        let mut between_ecalls = false;
        let mut segment_num = 0;
        
        for (i, instruction) in self.bytecode.iter().enumerate() {
            // Check if this is an ECALL instruction in bytecode
            let instr_str = format!("{:?}", instruction);
            if instr_str.contains("ECALL") {
                if between_ecalls {
                    output.push_str(&format!("--- End of segment {} ---\n", segment_num));
                }
                between_ecalls = !between_ecalls;
                if between_ecalls {
                    segment_num += 1;
                    output.push_str(&format!("--- Segment {} ---\n", segment_num));
                }
                continue; // Skip the ECALL itself
            }
            
            if between_ecalls {
                output.push_str(&format!("{:4}: {:?}\n", i, instruction));
            }
        }
        output.push_str("\n");

        // All trace instructions between ECALLs (all segments)
        output.push_str("=== ALL TRACE INSTRUCTIONS BETWEEN ECALLS ===\n");
        let mut between_ecalls = false;
        let mut segment_num = 0;
        
        for (i, cycle) in self.trace.iter().enumerate() {
            let instruction_name: &'static str = cycle.into();
            
            if instruction_name == "ECALL" {
                if between_ecalls {
                    output.push_str(&format!("--- End of segment {} ---\n", segment_num));
                }
                between_ecalls = !between_ecalls;
                if between_ecalls {
                    segment_num += 1;
                    output.push_str(&format!("--- Segment {} ---\n", segment_num));
                }
                continue; // Skip the ECALL itself
            }
            
            if between_ecalls {
                output.push_str(&format!("{:4}: {:?}\n", i, cycle));
            }
        }

        // Write to file
        std::fs::write(filename, output)?;
        Ok(())
    }
}
