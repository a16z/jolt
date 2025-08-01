use std::{collections::HashMap, fs::File, io, path::PathBuf};

use serde::{Deserialize, Serialize};
use tracer::{
    instruction::{RV32IMCycle, RV32IMInstruction},
    JoltDevice,
};

use crate::field::JoltField;

#[derive(Serialize, Deserialize, Debug)]
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
}
