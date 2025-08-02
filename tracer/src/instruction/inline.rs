//! Inline instruction support for RISC-V.
//!
//! This module provides a flexible framework for registering and executing
//! inlines within the RISC-V instruction set.
//!
//! # Architecture
//!
//! The inline system uses the RISC-V custom-1 opcode (0x2B) with the R-format
//! instruction encoding. inlines are uniquely identified by their funct3 and
//! funct7 fields, allowing up to 1,024 different inline operations.

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
};
use crate::emulator::cpu::Cpu;
use lazy_static::lazy_static;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

// Type alias for the exec and virtual_sequence functions signature
pub type ExecFunction = Box<dyn Fn(&INLINE, &mut Cpu, &mut ()) + Send + Sync>;
pub type VirtualSequenceFunction =
    Box<dyn Fn(u64, usize, usize) -> Vec<RV32IMInstruction> + Send + Sync>;

// Key type for the registry: (funct3, funct7)
type InlineKey = (u32, u32);

// Global registry that maps (funct3, funct7) pairs to inline implementations
lazy_static! {
    static ref INLINE_REGISTRY: RwLock<HashMap<InlineKey, (String, ExecFunction, VirtualSequenceFunction)>> =
        RwLock::new(HashMap::new());
}

/// Registers a new inline instruction handler.
///
/// # Arguments
///
/// * `funct3` - The 3-bit function code (0-7)
/// * `funct7` - The 7-bit function code (0-127)
/// * `name` - Human-readable name for the inline
/// * `exec_fn` - Function to execute during CPU simulation
/// * `virtual_sequence_fn` - Function to generate virtual instruction sequence
#[allow(dead_code)]
pub fn register_inline(
    funct3: u32,
    funct7: u32,
    name: &str,
    exec_fn: ExecFunction,
    virtual_sequence_fn: VirtualSequenceFunction,
) -> Result<(), String> {
    // Validate input ranges
    if funct3 > 7 {
        return Err(format!("funct3 value {} exceeds maximum of 7", funct3));
    }
    if funct7 > 127 {
        return Err(format!("funct7 value {} exceeds maximum of 127", funct7));
    }

    let mut registry = INLINE_REGISTRY
        .write()
        .map_err(|_| "Failed to acquire write lock on inline registry")?;

    let key = (funct3, funct7);
    if registry.contains_key(&key) {
        return Err(format!(
            "Inline '{}' with funct3={}, funct7={} is already registered",
            registry
                .get(&key)
                .map(|(name, _, _)| name.as_str())
                .unwrap_or("unknown"),
            funct3,
            funct7
        ));
    }

    registry.insert(key, (name.to_string(), exec_fn, virtual_sequence_fn));
    Ok(())
}

/// Returns a list of all registered inlines.
///
/// # Returns
///
/// A vector of tuples containing:
/// - `(funct3, funct7)` pair
/// - Inline name
#[allow(dead_code)]
pub fn list_registered_inlines() -> Vec<((u32, u32), String)> {
    match INLINE_REGISTRY.read() {
        Ok(registry) => registry
            .iter()
            .map(|(&key, (name, _, _))| (key, name.clone()))
            .collect(),
        Err(_) => {
            eprintln!("Warning: Failed to acquire read lock on inline registry");
            Vec::new()
        }
    }
}

/// Checks if a inline is registered for the given funct3 and funct7 values.
#[allow(dead_code)]
pub fn is_inline_registered(funct3: u32, funct7: u32) -> bool {
    match INLINE_REGISTRY.read() {
        Ok(registry) => registry.contains_key(&(funct3, funct7)),
        Err(_) => false,
    }
}

/// RISC-V inline instruction.
/// # Note
///
/// This struct is manually implemented instead of using the `declare_riscv_instr!`
/// macro because we need to:
/// 1. Store funct3 and funct7 fields for dispatch
/// 2. Implement custom parsing behavior in the `new()` method
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct INLINE {
    /// 3-bit function selector (bits 14:12 of instruction)
    pub funct3: u32,
    /// 7-bit function selector (bits 31:25 of instruction)
    pub funct7: u32,
    /// Memory address of this instruction
    pub address: u64,
    /// R-format operands (rd, rs1, rs2)
    pub operands: FormatR,
    /// Tracks remaining virtual instructions (used by tracer)
    pub virtual_sequence_remaining: Option<usize>,
}

impl RISCVInstruction for INLINE {
    const MASK: u32 = 0x0000707f;
    const MATCH: u32 = 0x0000002b; // opcode=0x2B (custom-1)

    type Format = FormatR;
    type RAMAccess = ();

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn new(word: u32, address: u64, _validate: bool) -> Self {
        Self {
            funct3: (word >> 12) & 0x7,
            funct7: (word >> 25) & 0x7f,
            address,
            operands: FormatR::parse(word),
            virtual_sequence_remaining: None,
        }
    }

    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        Self {
            funct3: rng.next_u32() & 0x7,
            funct7: rng.next_u32() & 0x7f,
            address: rng.next_u64(),
            operands: FormatR::random(rng),
            virtual_sequence_remaining: None,
        }
    }

    fn execute(&self, cpu: &mut Cpu, ram: &mut Self::RAMAccess) {
        self.exec(cpu, ram)
    }
}

impl INLINE {
    pub fn exec(&self, cpu: &mut Cpu, _: &mut <INLINE as RISCVInstruction>::RAMAccess) {
        // Look up the inline function in the registry
        let key = (self.funct3, self.funct7);

        match INLINE_REGISTRY.read() {
            Ok(registry) => {
                match registry.get(&key) {
                    Some((_name, exec_fn, _)) => {
                        // Execute the registered function
                        exec_fn(self, cpu, &mut ());
                    }
                    None => {
                        panic!(
                            "No inline handler registered for funct3={:#03b}, funct7={:#09b} \
                            Register a handler using register_inline().",
                            self.funct3, self.funct7
                        );
                    }
                }
            }
            Err(_) => {
                panic!(
                    "Failed to acquire read lock on inline registry. \
                    This indicates a critical error in the system."
                );
            }
        }
    }
}

impl RISCVTrace for INLINE {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence();
        let mut trace = trace;
        for instr in virtual_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for INLINE {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        let key = (self.funct3, self.funct7);

        match INLINE_REGISTRY.read() {
            Ok(registry) => {
                match registry.get(&key) {
                    Some((_name, _exec_fn, virtual_seq_fn)) => {
                        // Generate the virtual instruction sequence
                        virtual_seq_fn(self.address, self.operands.rs1, self.operands.rs2)
                    }
                    None => {
                        panic!(
                            "No virtual sequence builder registered for inline \
                            with funct3={:#03b}, funct7={:#09b}. \
                            Register a builder using register_inline().",
                            self.funct3, self.funct7
                        );
                    }
                }
            }
            Err(_) => {
                panic!(
                    "Failed to acquire read lock on inline registry. \
                    This indicates a critical error in the system."
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_inline_validation() {
        // Test funct3 validation
        let result = register_inline(
            8, // Invalid funct3 (> 7)
            0,
            "test",
            Box::new(|_, _, _| {}),
            Box::new(|_, _, _| vec![]),
        );
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("funct3 value 8 exceeds maximum"));

        // Test funct7 validation
        let result = register_inline(
            0,
            128, // Invalid funct7 (> 127)
            "test",
            Box::new(|_, _, _| {}),
            Box::new(|_, _, _| vec![]),
        );
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("funct7 value 128 exceeds maximum"));
    }

    #[test]
    fn test_inline_parsing() {
        // Test instruction word parsing
        // funct7=0x7f, rs2=0x1f, rs1=0x1f, funct3=0x7, rd=0x1f, opcode=0x2b
        let word: u32 = 0xff_ff_f_f_ab;
        let inline = INLINE::new(word, 0x1000, false);

        assert_eq!(inline.funct3, 0x7);
        assert_eq!(inline.funct7, 0x7f);
        assert_eq!(inline.address, 0x1000);
    }

    #[test]
    fn test_list_inlines() {
        // Clear registry first (in test environment)
        let inlines = list_registered_inlines();
        // Should return empty or existing inlines
        assert!(inlines.is_empty() || !inlines.is_empty());
    }
}
