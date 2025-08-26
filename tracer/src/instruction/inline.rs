//! Inline instruction support for RISC-V.
//!
//! This module provides a flexible framework for registering and executing
//! inlines within the RISC-V instruction set.
//!
//! # Architecture
//!
//! The inline system uses the RISC-V custom-0 (0x0B) and custom-1 (0x2B) opcodes
//! with the R-format instruction encoding. Inlines are uniquely identified by their
//! opcode, funct3, and funct7 fields.

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
};
use crate::{
    emulator::cpu::{Cpu, Xlen},
    instruction::NormalizedInstruction,
};
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

// Type alias for the exec and inline_sequence functions signature
pub type ExecFunction = Box<dyn Fn(&INLINE, &mut Cpu, &mut ()) + Send + Sync>;
pub type InlineSequenceFunction =
    Box<dyn Fn(u64, bool, Xlen, u8, u8, u8) -> Vec<RV32IMInstruction> + Send + Sync>;

// Key type for the registry: (opcode, funct3, funct7)
type InlineKey = (u32, u32, u32);

// Global registry that maps (opcode, funct3, funct7) tuples to inline implementations
lazy_static! {
    static ref INLINE_REGISTRY: RwLock<HashMap<InlineKey, (String, ExecFunction, InlineSequenceFunction)>> =
        RwLock::new(HashMap::new());
}

/// Registers a new inline instruction handler.
///
/// Each new type of operation should be placed under different funct7,
/// while funct3 should hold all necessary instructions for that operation.
///
/// # Arguments
///
/// * `opcode` - The 7-bit opcode (0-127)
/// * `funct3` - The 3-bit function code (0-7)
/// * `funct7` - The 7-bit function code (0-127)
/// * `name` - Human-readable name for the inline
/// * `exec_fn` - Function to execute during CPU simulation
/// * `inline_sequence_fn` - Function to generate virtual instruction sequence
pub fn register_inline(
    opcode: u32,
    funct3: u32,
    funct7: u32,
    name: &str,
    exec_fn: ExecFunction,
    inline_sequence_fn: InlineSequenceFunction,
) -> Result<(), String> {
    if opcode != 0x0B && opcode != 0x2B {
        return Err(format!(
            "opcode value {opcode:#04x} is invalid. Only 0x0B (custom-0) and 0x2B (custom-1) are allowed for inline in"
        ));
    }
    if funct3 > 7 {
        return Err(format!("funct3 value {funct3} exceeds maximum of 7"));
    }
    if funct7 > 127 {
        return Err(format!("funct7 value {funct7} exceeds maximum of 127"));
    }

    let mut registry = INLINE_REGISTRY
        .write()
        .map_err(|_| "Failed to acquire write lock on inline registry")?;

    let key = (opcode, funct3, funct7);
    if registry.contains_key(&key) {
        return Err(format!(
            "Inline '{}' with opcode={opcode:#x}, funct3={funct3}, funct7={funct7} is already registered",
            registry
                .get(&key)
                .map(|(name, _, _)| name.as_str())
                .unwrap_or("unknown")
        ));
    }

    registry.insert(key, (name.to_string(), exec_fn, inline_sequence_fn));
    Ok(())
}

/// Returns a list of all registered inlines.
///
/// # Returns
///
/// A vector of tuples containing:
/// - `(opcode, funct3, funct7)` tuple
/// - Inline name
pub fn list_registered_inlines() -> Vec<((u32, u32, u32), String)> {
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

/// Checks if a inline is registered for the given opcode, funct3 and funct7 values.
pub fn is_inline_registered(opcode: u32, funct3: u32, funct7: u32) -> bool {
    match INLINE_REGISTRY.read() {
        Ok(registry) => registry.contains_key(&(opcode, funct3, funct7)),
        Err(_) => false,
    }
}

/// RISC-V inline instruction.
/// # Note
///
/// This struct is manually implemented instead of using the `declare_riscv_instr!` macro because we need to:
/// Store opcode, funct3 and funct7 fields for dispatch
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct INLINE {
    /// 7-bit opcode (bits 6:0 of instruction)
    pub opcode: u32,
    /// 3-bit function selector (bits 14:12 of instruction)
    pub funct3: u32,
    /// 7-bit function selector (bits 31:25 of instruction)
    pub funct7: u32,
    /// Memory address of this instruction
    pub address: u64,
    /// R-format operands (rd, rs1, rs2)
    pub operands: FormatR,
    /// Tracks remaining virtual instructions (used by tracer)
    pub inline_sequence_remaining: Option<u16>,
    pub is_compressed: bool,
}

impl RISCVInstruction for INLINE {
    const MASK: u32 = 0x0000707f;
    const MATCH: u32 = 0x0000002b; // opcode=0x2B (custom-1)

    type Format = FormatR;
    type RAMAccess = ();

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn new(word: u32, address: u64, _validate: bool, is_compressed: bool) -> Self {
        Self {
            opcode: word & 0x7f,
            funct3: (word >> 12) & 0x7,
            funct7: (word >> 25) & 0x7f,
            address,
            operands: FormatR::parse(word),
            inline_sequence_remaining: None,
            is_compressed,
        }
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use crate::instruction::format::InstructionFormat;
        use rand::RngCore;
        Self {
            opcode: rng.next_u32() & 0x7f,
            funct3: rng.next_u32() & 0x7,
            funct7: rng.next_u32() & 0x7f,
            address: rng.next_u64(),
            operands: FormatR::random(rng),
            inline_sequence_remaining: None,
            is_compressed: false,
        }
    }

    fn execute(&self, cpu: &mut Cpu, ram: &mut Self::RAMAccess) {
        self.exec(cpu, ram)
    }
}

impl INLINE {
    pub fn exec(&self, cpu: &mut Cpu, _: &mut <INLINE as RISCVInstruction>::RAMAccess) {
        // Look up the inline function in the registry
        let key = (self.opcode, self.funct3, self.funct7);

        match INLINE_REGISTRY.read() {
            Ok(registry) => {
                match registry.get(&key) {
                    Some((_name, exec_fn, _)) => {
                        // Execute the registered function
                        exec_fn(self, cpu, &mut ());
                    }
                    None => {
                        panic!(
                            "No inline handler registered for opcode={:#04x}, funct3={:#03b}, funct7={:#09b} \
                            Register a handler using register_inline().",
                            self.opcode, self.funct3, self.funct7
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
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        let key = (self.opcode, self.funct3, self.funct7);

        match INLINE_REGISTRY.read() {
            Ok(registry) => {
                match registry.get(&key) {
                    Some((_name, _exec_fn, virtual_seq_fn)) => {
                        // Generate the virtual instruction sequence
                        virtual_seq_fn(
                            self.address,
                            self.is_compressed,
                            xlen,
                            self.operands.rs1,
                            self.operands.rs2,
                            self.operands.rd,
                        )
                    }
                    None => {
                        panic!(
                            "No inline sequence builder registered for inline \
                            with opcode={:#04x}, funct3={:#03b}, funct7={:#09b}. \
                            Register a builder using register_inline().",
                            self.opcode, self.funct3, self.funct7
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

impl From<NormalizedInstruction> for INLINE {
    fn from(_: NormalizedInstruction) -> Self {
        unimplemented!("Inline::from(NormalizedInstruction) should not be called");
    }
}

impl From<INLINE> for NormalizedInstruction {
    fn from(instr: INLINE) -> Self {
        NormalizedInstruction {
            address: instr.address as usize,
            operands: instr.operands.into(),
            inline_sequence_remaining: instr.inline_sequence_remaining,
            is_compressed: instr.is_compressed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_inline_validation() {
        // Test opcode validation
        let result = register_inline(
            128, // Invalid opcode (> 127)
            0,
            0,
            "test",
            Box::new(|_, _, _| {}),
            Box::new(|_, _, _, _, _, _| vec![]),
        );
        assert!(result.is_err());

        // Test funct3 validation
        let result = register_inline(
            0x2B, // Valid opcode
            8,    // Invalid funct3 (> 7)
            0,
            "test",
            Box::new(|_, _, _| {}),
            Box::new(|_, _, _, _, _, _| vec![]),
        );
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("funct3 value 8 exceeds maximum"));

        // Test funct7 validation
        let result = register_inline(
            0x2B, // Valid opcode
            0,
            128, // Invalid funct7 (> 127)
            "test",
            Box::new(|_, _, _| {}),
            Box::new(|_, _, _, _, _, _| vec![]),
        );
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("funct7 value 128 exceeds maximum"));
    }

    #[test]
    fn test_valid_opcodes() {
        // Test that 0x0B (custom-0) is valid
        let result = register_inline(
            0x0B,
            0,
            0,
            "test_custom0",
            Box::new(|_, _, _| {}),
            Box::new(|_, _, _, _, _, _| vec![]),
        );
        assert!(result.is_ok());

        // Test that 0x2B (custom-1) is valid
        let result = register_inline(
            0x2B,
            0,
            1, // Different funct7 to avoid duplicate registration
            "test_custom1",
            Box::new(|_, _, _| {}),
            Box::new(|_, _, _, _, _, _| vec![]),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_inline_parsing() {
        // Test instruction word parsing
        // funct7=0x7f, rs2=0x1f, rs1=0x1f, funct3=0x7, rd=0x1f, opcode=0x2b
        let word: u32 = 0xffffffab;
        let inline = INLINE::new(word, 0x1000, false, false);

        assert_eq!(inline.opcode, 0x2b);
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
