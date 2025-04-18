use std::{collections::HashMap, fs::File, io, path::PathBuf};

use serde::{Deserialize, Serialize};
use tracer::{
    instruction::{RV32IMCycle, RV32IMInstruction},
    JoltDevice, RV32IM,
};

use crate::{field::JoltField, jolt::vm::JoltTraceStep};

#[derive(Serialize, Deserialize)]
pub struct ProgramSummary {
    pub raw_trace: Vec<RV32IMCycle>,
    pub bytecode: Vec<RV32IMInstruction>,
    pub memory_init: Vec<(u64, u8)>,
    pub io_device: JoltDevice,
    pub processed_trace: Vec<JoltTraceStep<32>>,
}

impl ProgramSummary {
    pub fn trace_len(&self) -> usize {
        self.processed_trace.len()
    }

    pub fn analyze<F: JoltField>(&self) -> Vec<(RV32IM, usize)> {
        let mut counts = HashMap::<RV32IM, usize>::new();
        for row in self.raw_trace.iter() {
            let op = row.instruction.opcode;
            if let Some(count) = counts.get(&op) {
                counts.insert(op, count + 1);
            } else {
                counts.insert(op, 1);
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
}
