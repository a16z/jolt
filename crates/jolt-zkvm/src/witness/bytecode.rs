//! Bytecode preprocessing for PC expansion.
//!
//! Maps instruction memory addresses to expanded (sequential) program counter
//! values. Virtual instruction sequences share the same unexpanded PC but get
//! consecutive expanded PCs.

use std::collections::BTreeMap;

use tracer::instruction::{Cycle, NormalizedInstruction};

/// Maps `(address, virtual_sequence_remaining)` pairs to expanded PCs.
///
/// Each unique instruction execution point in the trace gets a sequential
/// expanded PC. Virtual instruction sequences share the same `address` but
/// differ in `virtual_sequence_remaining`, giving each step its own expanded PC.
pub struct BytecodePreprocessing {
    pc_map: BTreeMap<(usize, Option<u16>), u64>,
}

impl BytecodePreprocessing {
    /// Builds the PC map from a complete execution trace.
    ///
    /// Assigns sequential expanded PCs (0, 1, 2, ...) to each unique
    /// `(address, virtual_sequence_remaining)` pair seen in the trace.
    pub fn new(trace: &[Cycle]) -> Self {
        let mut pc_map = BTreeMap::new();
        let mut next_pc = 0u64;

        for cycle in trace {
            let norm = Self::normalize_cycle(cycle);
            let key = (norm.address, norm.virtual_sequence_remaining);
            let _ = pc_map.entry(key).or_insert_with(|| {
                let pc = next_pc;
                next_pc += 1;
                pc
            });
        }

        Self { pc_map }
    }

    /// Returns the expanded PC for a cycle.
    ///
    /// # Panics
    ///
    /// Panics if the cycle's `(address, virtual_sequence_remaining)` was not
    /// seen during preprocessing.
    pub fn get_pc(&self, cycle: &Cycle) -> u64 {
        // NoOp (padding) cycles map to PC 0, matching jolt-core's convention
        if matches!(cycle, Cycle::NoOp) {
            return 0;
        }
        let norm = Self::normalize_cycle(cycle);
        let key = (norm.address, norm.virtual_sequence_remaining);
        *self
            .pc_map
            .get(&key)
            .unwrap_or_else(|| panic!("unknown PC for address={}, vsr={:?}", key.0, key.1))
    }

    /// Number of unique instruction points (expanded PC count).
    pub fn code_size(&self) -> usize {
        self.pc_map.len()
    }

    fn normalize_cycle(cycle: &Cycle) -> NormalizedInstruction {
        cycle.instruction().normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_trace() {
        let prep = BytecodePreprocessing::new(&[]);
        assert_eq!(prep.code_size(), 0);
    }
}
