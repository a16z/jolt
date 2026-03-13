//! Inline instruction support for RISC-V.
//!
//! The inline system uses the RISC-V custom-0 (0x0B) and custom-1 (0x2B) opcodes
//! with the Inline-format instruction encoding. Inlines are uniquely identified by their
//! opcode, funct3, and funct7 fields.
//!
//! Inline implementations register themselves at link time via `inventory::submit!`.
//! The INLINE instruction iterates these registrations to find the matching builder.

use super::{
    format::{format_inline::FormatInline, InstructionFormat},
    Cycle, Instruction, RISCVInstruction, RISCVTrace,
};
use crate::{
    emulator::cpu::{Cpu, Xlen},
    instruction::NormalizedInstruction,
    utils::{inline_helpers::InstrAssembler, virtual_registers::VirtualRegisterAllocator},
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

pub type InlineSequenceFn = fn(InstrAssembler, FormatInline) -> Vec<Instruction>;

pub type AdviceFn = fn(InstrAssembler, FormatInline, &mut Cpu) -> Option<VecDeque<u64>>;

/// A registered inline implementation, discovered at link time via `inventory`.
pub struct InlineRegistration {
    pub opcode: u32,
    pub funct3: u32,
    pub funct7: u32,
    pub name: &'static str,
    pub build_sequence: InlineSequenceFn,
    pub build_advice: AdviceFn,
}

inventory::collect!(InlineRegistration);

fn find_inline(opcode: u32, funct3: u32, funct7: u32) -> &'static InlineRegistration {
    inventory::iter::<InlineRegistration>
        .into_iter()
        .find(|r| r.opcode == opcode && r.funct3 == funct3 && r.funct7 == funct7)
        .unwrap_or_else(|| {
            panic!(
                "No inline registered for opcode={opcode:#04x}, funct3={funct3:#03b}, funct7={funct7:#09b}."
            )
        })
}

pub fn list_registered_inlines() -> Vec<((u32, u32, u32), String)> {
    inventory::iter::<InlineRegistration>
        .into_iter()
        .map(|r| ((r.opcode, r.funct3, r.funct7), r.name.to_string()))
        .collect()
}

pub fn is_inline_registered(opcode: u32, funct3: u32, funct7: u32) -> bool {
    inventory::iter::<InlineRegistration>
        .into_iter()
        .any(|r| r.opcode == opcode && r.funct3 == funct3 && r.funct7 == funct7)
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
    pub operands: FormatInline,
    /// Tracks remaining virtual instructions (used by tracer)
    pub virtual_sequence_remaining: Option<u16>,
    pub is_first_in_sequence: bool,
    pub is_compressed: bool,
}

impl RISCVInstruction for INLINE {
    const MASK: u32 = 0x0000707f;
    const MATCH: u32 = 0x0000002b; // opcode=0x2B (custom-1)

    type Format = FormatInline;
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
            operands: FormatInline::parse(word),
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
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
            operands: FormatInline::random(rng),
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    fn execute(&self, cpu: &mut Cpu, ram: &mut Self::RAMAccess) {
        self.exec(cpu, ram)
    }

    fn has_side_effects(&self) -> bool {
        true
    }
}

impl INLINE {
    pub fn exec(&self, _cpu: &mut Cpu, _: &mut <INLINE as RISCVInstruction>::RAMAccess) {
        panic!("Inline instructions must use trace(), not exec()");
    }
}

impl RISCVTrace for INLINE {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // If rd (rs3 in FormatInline) is x0, remap to a virtual register so the
        // constraint system never sees rd=x0.
        if self.operands.rs3 == 0 {
            let vr = cpu.vr_allocator.allocate();
            let mut remapped = *self;
            remapped.operands.rs3 = *vr;
            remapped.trace(cpu, trace);
            return;
        }

        let reg = find_inline(self.opcode, self.funct3, self.funct7);
        let asm = InstrAssembler::new_inline(
            self.address,
            self.is_compressed,
            cpu.xlen,
            &cpu.vr_allocator,
        );
        if let Some(mut advice) = (reg.build_advice)(asm, self.operands, cpu) {
            let mut inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
            let mut trace = trace;
            for instr in inline_sequence.iter_mut() {
                if let Instruction::VirtualAdvice(va) = instr {
                    va.advice = match advice.pop_front() {
                        Some(val) => val,
                        None => panic!(
                            "Inline advice for opcode={:#04x}, funct3={:#03b}, funct7={:#09b} \
                            did not provide enough values",
                            self.opcode, self.funct3, self.funct7
                        ),
                    };
                }
                instr.trace(cpu, trace.as_deref_mut());
            }
            assert!(
                advice.is_empty(),
                "Inline advice for opcode={:#04x}, funct3={:#03b}, funct7={:#09b} \
                provided too many values",
                self.opcode,
                self.funct3,
                self.funct7
            );
        } else {
            let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
            let mut trace = trace;
            for instr in inline_sequence {
                instr.trace(cpu, trace.as_deref_mut());
            }
        }
    }

    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let reg = find_inline(self.opcode, self.funct3, self.funct7);
        let asm = InstrAssembler::new_inline(self.address, self.is_compressed, xlen, allocator);
        (reg.build_sequence)(asm, self.operands)
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
            virtual_sequence_remaining: instr.virtual_sequence_remaining,
            is_first_in_sequence: instr.is_first_in_sequence,
            is_compressed: instr.is_compressed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inline_parsing() {
        let word: u32 = 0xffffffab;
        let inline = INLINE::new(word, 0x1000, false, false);

        assert_eq!(inline.opcode, 0x2b);
        assert_eq!(inline.funct3, 0x7);
        assert_eq!(inline.funct7, 0x7f);
        assert_eq!(inline.address, 0x1000);
    }
}
