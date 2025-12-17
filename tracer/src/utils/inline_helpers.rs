//! InstrAssembler
//!
//! Builds and owns a vector of virtual RISC-V instructions (`Instruction`).
//! The struct provides small helper methods so that higher-level builders can emit
//! common instructions without repeating encoding boiler-plate.
//!
//! This module focuses purely on the mechanical task of
//! pushing the correct instruction variant into `sequence`, and is meant to be used
//! by higher-level builders.
//!
//! ## Generic Emission
//!
//! The assembler provides format-aware generic helpers:
//! * `emit_r::<XOR>(rd, rs1, rs2)` - R-type (register-register)
//! * `emit_i::<ADDI>(rd, rs1, imm)` - I-type (immediate)
//! * `emit_s::<SW>(rs1, rs2, offset)` - S-type (store)
//!
//! And a generic binary operation helper:
//! * `bin::<XOR, XORI>(a, b, rd, |x, y| x ^ y)` - handles constant folding
//!
//! Assumptions:
//! * The helpers are infallible (in terms of error handling) and run in constant time.
//! * Composite helpers such as `xor64` and `rotl64` expand into multiple 32-bit
//!   operations; the exact policy is documented where they are defined.

/// Common opcode for all inline instructions
pub const INLINE_OPCODE: u32 = 0x0B;
use crate::instruction::add::ADD;
use crate::instruction::addi::ADDI;
use crate::instruction::and::AND;
use crate::instruction::andi::ANDI;
use crate::instruction::srli::SRLI;
use crate::instruction::srliw::SRLIW;

use crate::instruction::format::format_assert_align::AssertAlignFormat;
use crate::instruction::format::format_b::FormatB;
use crate::instruction::format::format_i::FormatI;
use crate::instruction::format::format_j::FormatJ;
use crate::instruction::format::format_load::FormatLoad;
use crate::instruction::format::format_r::FormatR;
use crate::instruction::format::format_s::FormatS;
use crate::instruction::format::format_u::FormatU;
use crate::instruction::format::format_virtual_right_shift_i::FormatVirtualRightShiftI;
use crate::instruction::format::format_virtual_right_shift_r::FormatVirtualRightShiftR;
use crate::instruction::format::NormalizedOperands;

use crate::emulator::cpu::Xlen;
use crate::instruction::virtual_rotri::VirtualROTRI;
use crate::instruction::virtual_rotriw::VirtualROTRIW;
use crate::instruction::xor::XOR;
use crate::instruction::xori::XORI;
use crate::instruction::Cycle;
use crate::instruction::Instruction;
use crate::instruction::NormalizedInstruction;
use crate::instruction::RISCVCycle;
use crate::instruction::RISCVInstruction;
use crate::instruction::RISCVTrace;
use crate::utils::virtual_registers::VirtualRegisterAllocator;

use common::constants::RISCV_REGISTER_COUNT;
use common::constants::VIRTUAL_INSTRUCTION_RESERVED_REGISTER_COUNT;
/// Operand that can be either an immediate or a register.
#[derive(Clone, Copy, Debug)]
pub enum Value {
    Imm(u64),
    Reg(u8),
}
use Value::{Imm, Reg};

/// Convenience assembler for building a sequence of `Instruction`s.
/// Includes common instruction emitters to be used by higher-level builders for inlines.
///
/// The assembler stores `address` and `is_compressed` so that
/// each emitted instruction carries the right metadata.
#[derive(Debug)]
pub struct InstrAssembler {
    /// Program counter associated with the *first* instruction; builders are
    /// responsible for post-processing if they need exact PCs per-instruction.
    pub address: u64,
    /// Whether to use RVC encodings.
    pub is_compressed: bool,
    /// Xlen of the CPU.
    pub xlen: Xlen,
    /// Accumulated instruction buffer.
    sequence: Vec<Instruction>,
    /// Whether the instruction uses the the FormatInline instruction format.
    has_inline_instr_format: bool,
    /// Virtual register allocator
    pub allocator: VirtualRegisterAllocator,
}

impl InstrAssembler {
    /// Create a new assembler with an empty instruction buffer.
    pub(crate) fn new(
        address: u64,
        is_compressed: bool,
        xlen: Xlen,
        allocator: &VirtualRegisterAllocator,
    ) -> Self {
        Self {
            address,
            is_compressed,
            xlen,
            sequence: Vec::new(),
            has_inline_instr_format: false,
            allocator: allocator.clone(),
        }
    }

    /// Create a new assembler with an empty instruction buffer.
    pub fn new_inline(
        address: u64,
        is_compressed: bool,
        xlen: Xlen,
        allocator: &VirtualRegisterAllocator,
    ) -> Self {
        Self {
            address,
            is_compressed,
            xlen,
            sequence: Vec::new(),
            has_inline_instr_format: true,
            allocator: allocator.clone(),
        }
    }

    /// Finalize the instruction buffer: back-fill `virtual_sequence_remaining`
    /// and return ownership of the underlying `Vec`.
    pub(crate) fn finalize(mut self) -> Vec<Instruction> {
        let len = self.sequence.len();
        for (i, instr) in self.sequence.iter_mut().enumerate() {
            instr.set_is_first_in_sequence(i == 0);
            instr.set_virtual_sequence_remaining(Some((len - i - 1) as u16));
        }
        self.sequence
            .last_mut()
            .expect("sequence should not be empty")
            .set_is_compressed(self.is_compressed);
        self.sequence
    }

    /// Finalize inline instructions by zeroing virtual registers, then calling finalize.
    pub fn finalize_inline(mut self) -> Vec<Instruction> {
        let register = self.allocator.get_registers_for_reset();
        // Zero inline virtual registers using ADDI rd, x0, 0
        for reg in register {
            self.emit_i::<ADDI>(reg, 0, 0);
        }
        self.finalize()
    }

    /// Validates that rd is an inline virtual register (not RISC-V or reserved virtual registers).
    #[inline]
    fn is_valid_virtual_rd(virtual_rd: u8) -> bool {
        virtual_rd == 0
            || virtual_rd >= RISCV_REGISTER_COUNT + VIRTUAL_INSTRUCTION_RESERVED_REGISTER_COUNT
    }

    #[inline]
    fn add_to_sequence<I: RISCVInstruction + RISCVTrace>(&mut self, inst: I)
    where
        RISCVCycle<I>: Into<Cycle>,
    {
        if self.has_inline_instr_format {
            let normalized: NormalizedInstruction = inst.into();
            if !Self::is_valid_virtual_rd(normalized.operands.rd) {
                const MIN_INLINE_REG: u8 =
                    RISCV_REGISTER_COUNT + VIRTUAL_INSTRUCTION_RESERVED_REGISTER_COUNT;
                panic!(
                    "Inline instruction attempted to write to register {}, but must use registers >= {}",
                    normalized.operands.rd, MIN_INLINE_REG
                );
            }
        }
        self.sequence
            .extend(inst.inline_sequence(&self.allocator, self.xlen));
    }

    /// Emit any R-type instruction (rd, rs1, rs2).
    #[track_caller]
    #[inline]
    pub fn emit_r<Op: RISCVInstruction<Format = FormatR> + RISCVTrace>(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
    ) where
        RISCVCycle<Op>: Into<Cycle>,
    {
        self.add_to_sequence(Op::from(NormalizedInstruction {
            address: self.address as usize,
            operands: NormalizedOperands {
                rd,
                rs1,
                rs2,
                imm: 0,
            },
            is_compressed: false,
            is_first_in_sequence: false,
            virtual_sequence_remaining: Some(0),
        }));
    }

    /// Emit any I-type instruction (rd, rs1, imm).
    #[track_caller]
    #[inline]
    pub fn emit_i<Op: RISCVInstruction<Format = FormatI> + RISCVTrace>(
        &mut self,
        rd: u8,
        rs1: u8,
        imm: u64,
    ) where
        RISCVCycle<Op>: Into<Cycle>,
    {
        self.add_to_sequence(Op::from(NormalizedInstruction {
            address: self.address as usize,
            operands: NormalizedOperands {
                rd,
                rs1,
                rs2: 0,
                imm: imm as i128,
            },
            is_compressed: false,
            is_first_in_sequence: false,
            virtual_sequence_remaining: Some(0),
        }));
    }

    /// Emit any S-type instruction (rs1, rs2, imm).
    #[track_caller]
    #[inline]
    pub fn emit_s<Op: RISCVInstruction<Format = FormatS> + RISCVTrace>(
        &mut self,
        rs1: u8,
        rs2: u8,
        imm: i64,
    ) where
        RISCVCycle<Op>: Into<Cycle>,
    {
        self.add_to_sequence(Op::from(NormalizedInstruction {
            address: self.address as usize,
            operands: NormalizedOperands {
                rd: 0,
                rs1,
                rs2,
                imm: imm as i128,
            },
            is_compressed: false,
            is_first_in_sequence: false,
            virtual_sequence_remaining: Some(0),
        }));
    }

    /// Emit any Load-type instruction (rd, rs1, imm) - like FormatI but with signed imm.
    #[track_caller]
    #[inline]
    pub fn emit_ld<Op: RISCVInstruction<Format = FormatLoad> + RISCVTrace>(
        &mut self,
        rd: u8,
        rs1: u8,
        imm: i64,
    ) where
        RISCVCycle<Op>: Into<Cycle>,
    {
        self.add_to_sequence(Op::from(NormalizedInstruction {
            address: self.address as usize,
            operands: NormalizedOperands {
                rd,
                rs1,
                rs2: 0,
                imm: imm as i128,
            },
            is_compressed: false,
            is_first_in_sequence: false,
            virtual_sequence_remaining: Some(0),
        }));
    }

    /// Emit any B-type instruction (rs1, rs2, imm) - branch instructions.
    #[track_caller]
    #[inline]
    pub fn emit_b<Op: RISCVInstruction<Format = FormatB> + RISCVTrace>(
        &mut self,
        rs1: u8,
        rs2: u8,
        imm: i64,
    ) where
        RISCVCycle<Op>: Into<Cycle>,
    {
        self.add_to_sequence(Op::from(NormalizedInstruction {
            address: self.address as usize,
            operands: NormalizedOperands {
                rd: 0,
                rs1,
                rs2,
                imm: imm as i128,
            },
            is_compressed: false,
            is_first_in_sequence: false,
            virtual_sequence_remaining: Some(0),
        }));
    }

    /// Emit any J-type instruction (rd, imm) - jump instructions.
    #[track_caller]
    #[inline]
    pub fn emit_j<Op: RISCVInstruction<Format = FormatJ> + RISCVTrace>(&mut self, rd: u8, imm: u64)
    where
        RISCVCycle<Op>: Into<Cycle>,
    {
        self.add_to_sequence(Op::from(NormalizedInstruction {
            address: self.address as usize,
            operands: NormalizedOperands {
                rd,
                rs1: 0,
                rs2: 0,
                imm: imm as i128,
            },
            is_compressed: false,
            is_first_in_sequence: false,
            virtual_sequence_remaining: Some(0),
        }));
    }

    /// Emit any U-type instruction (rd, imm) - upper immediate instructions.
    #[track_caller]
    #[inline]
    pub fn emit_u<Op: RISCVInstruction<Format = FormatU> + RISCVTrace>(&mut self, rd: u8, imm: u64)
    where
        RISCVCycle<Op>: Into<Cycle>,
    {
        self.add_to_sequence(Op::from(NormalizedInstruction {
            address: self.address as usize,
            operands: NormalizedOperands {
                rd,
                rs1: 0,
                rs2: 0,
                imm: imm as i128,
            },
            is_compressed: false,
            is_first_in_sequence: false,
            virtual_sequence_remaining: Some(0),
        }));
    }

    /// Emit any virtual right shift I-type instruction (rd, rs1, imm).
    #[track_caller]
    #[inline]
    pub fn emit_vshift_i<Op: RISCVInstruction<Format = FormatVirtualRightShiftI> + RISCVTrace>(
        &mut self,
        rd: u8,
        rs1: u8,
        imm: u64,
    ) where
        RISCVCycle<Op>: Into<Cycle>,
    {
        self.add_to_sequence(Op::from(NormalizedInstruction {
            address: self.address as usize,
            operands: NormalizedOperands {
                rd,
                rs1,
                rs2: 0,
                imm: imm as i128,
            },
            is_compressed: false,
            is_first_in_sequence: false,
            virtual_sequence_remaining: Some(0),
        }));
    }

    /// Emit any virtual right shift R-type instruction (rd, rs1, rs2).
    #[track_caller]
    #[inline]
    pub fn emit_vshift_r<Op: RISCVInstruction<Format = FormatVirtualRightShiftR> + RISCVTrace>(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
    ) where
        RISCVCycle<Op>: Into<Cycle>,
    {
        self.add_to_sequence(Op::from(NormalizedInstruction {
            address: self.address as usize,
            operands: NormalizedOperands {
                rd,
                rs1,
                rs2,
                imm: 0,
            },
            is_compressed: false,
            is_first_in_sequence: false,
            virtual_sequence_remaining: Some(0),
        }));
    }

    /// Emit any halfword alignment instruction (rs1, imm).
    #[track_caller]
    #[inline]
    pub fn emit_halign<Op: RISCVInstruction<Format = AssertAlignFormat> + RISCVTrace>(
        &mut self,
        rs1: u8,
        imm: i64,
    ) where
        RISCVCycle<Op>: Into<Cycle>,
    {
        self.add_to_sequence(Op::from(NormalizedInstruction {
            address: self.address as usize,
            operands: NormalizedOperands {
                rd: 0,
                rs1,
                rs2: 0,
                imm: imm as i128,
            },
            is_compressed: false,
            is_first_in_sequence: false,
            virtual_sequence_remaining: Some(0),
        }));
    }

    /// Generic binary operation with constant folding.
    /// Automatically selects R-type vs I-type encoding based on operand types.
    pub fn bin<
        OR: RISCVInstruction<Format = FormatR> + RISCVTrace,
        OI: RISCVInstruction<Format = FormatI> + RISCVTrace,
    >(
        &mut self,
        rs1: Value,
        rs2: Value,
        rd: u8,
        fold: fn(u64, u64) -> u64,
    ) -> Value
    where
        RISCVCycle<OR>: Into<Cycle>,
        RISCVCycle<OI>: Into<Cycle>,
    {
        match (rs1, rs2) {
            (Reg(r1), Reg(r2)) => {
                self.emit_r::<OR>(rd, r1, r2);
                Reg(rd)
            }
            (Reg(r1), Imm(imm)) => {
                self.emit_i::<OI>(rd, r1, imm);
                Reg(rd)
            }
            (Imm(_), Reg(_)) => self.bin::<OR, OI>(rs2, rs1, rd, fold),
            (Imm(i1), Imm(i2)) => Imm(fold(i1, i2)),
        }
    }

    pub fn add(&mut self, rs1: Value, rs2: Value, rd: u8) -> Value {
        self.bin::<ADD, ADDI>(rs1, rs2, rd, |x, y| (x).wrapping_add(y))
    }

    /// Logical right-shift immediate on a 32-bit word.
    pub fn srli(&mut self, rs1: Value, shamt: u32, rd: u8) -> Value {
        if shamt == 0 {
            return self.xor(rs1, Imm(0), rd);
        }
        match rs1 {
            Reg(rs1) => {
                match self.xlen {
                    Xlen::Bit32 => self.emit_i::<SRLI>(rd, rs1, shamt as u64),
                    Xlen::Bit64 => self.emit_i::<SRLIW>(rd, rs1, (shamt & 0x1f) as u64),
                }
                Reg(rd)
            }
            Imm(val) => Imm(((val as u32) >> shamt) as u64),
        }
    }

    /// Rotate-right by amount on a 32-bit word.
    pub fn rotri32(&mut self, rs1: Value, shamt: u32, rd: u8) -> Value {
        if shamt == 0 || shamt == 32 {
            // Rotating by 0 or 32 bits on a 32-bit value is a no-op.
            return self.xor(rs1, Imm(0), rd);
        }
        let ones = (1u64 << (32 - shamt)) - 1;
        let mask = ones << shamt;
        match rs1 {
            Reg(rs1_reg) => {
                match self.xlen {
                    Xlen::Bit32 => self.emit_vshift_i::<VirtualROTRI>(rd, rs1_reg, mask),
                    Xlen::Bit64 => self.emit_vshift_i::<VirtualROTRIW>(rd, rs1_reg, mask),
                }
                Reg(rd)
            }
            Imm(val) => Imm(((val as u32).rotate_right(shamt)) as u64),
        }
    }

    /// Composite ROTRᵢ ⊕ ROTRⱼ used by SHA-256.
    pub fn rotri_xor_rotri32(
        &mut self,
        rs1: Value,
        imm1: u32,
        imm2: u32,
        rd: u8,
        scratch: u8,
    ) -> Value {
        let r1 = self.rotri32(rs1, imm1, scratch);
        let r2 = self.rotri32(rs1, imm2, rd);
        self.xor(r1, r2, rd)
    }

    /// Emit `XOR rd, rs1, rs2` / `XORI rd, rs1, imm` and return `Reg(rd)` or
    /// an `Imm` result when both operands are immediates.
    pub fn xor(&mut self, rs1: Value, rs2: Value, rd: u8) -> Value {
        self.bin::<XOR, XORI>(rs1, rs2, rd, |x, y| x ^ y)
    }

    /// Emit `AND`/`ANDI` similarly to `xor`.
    pub fn and(&mut self, rs1: Value, rs2: Value, rd: u8) -> Value {
        self.bin::<AND, ANDI>(rs1, rs2, rd, |x, y| x & y)
    }

    /// Emit virtual `ROTRI` (rotate-right immediate) or compute constant fold.
    pub fn rotri(&mut self, rs1: Value, imm: u64, rd: u8) -> Value {
        match rs1 {
            Reg(rs1) => {
                self.emit_vshift_i::<VirtualROTRI>(rd, rs1, imm);
                Reg(rd)
            }
            Imm(val) => {
                // `imm` is a rotate-right bit-mask (trailing zeros denote shift amount).
                let shift = imm.trailing_zeros();
                Imm(((val as u32).rotate_right(shift)) as u64)
            }
        }
    }

    /// Rotate left on a 64-bit value.
    pub fn rotr64(&mut self, rs1: Value, amount: u32, rd: u8) -> Value {
        if amount == 0 || amount == 64 {
            // Identity rotation: emit XOR with zero to copy value.
            // Rotating by 0 or 64 bits on a 64-bit value is a no-op.
            return self.xor(rs1, Imm(0), rd);
        }

        match rs1 {
            Reg(rs1_reg) => {
                // The VirtualROTRI instruction encodes the rotation via a bitmask
                // whose trailing zeros indicate the shift amount.
                // For rotr(n), we need n trailing zeros.
                let ones = (1u64 << (64 - amount as u64)) - 1;
                let imm = ones << amount as u64;
                self.rotri(Reg(rs1_reg), imm, rd)
            }
            Imm(val) => Imm(val.rotate_right(amount)),
        }
    }

    /// Rotate left on a 64-bit value.
    pub fn rotl64(&mut self, rs1: Value, amount: u32, rd: u8) -> Value {
        // rotl(n) == rotr(64 - n)
        self.rotr64(rs1, 64 - amount, rd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotl64_immediate_paths() {
        let allocator = VirtualRegisterAllocator::new();
        let mut asm = InstrAssembler::new(0, false, Xlen::Bit64, &allocator);
        let dest = 0;
        let vectors: &[(u64, u32, u64)] = &[
            (0x0000000000000001, 1, 0x0000000000000002),
            (0x8000000000000000, 1, 0x0000000000000001),
            (0x0123456789ABCDEF, 4, 0x123456789ABCDEF0),
            (0x0123456789ABCDEF, 32, 0x89ABCDEF01234567),
            (0x0123456789ABCDEF, 36, 0x9ABCDEF012345678),
        ];
        for (val, amount, expected) in vectors {
            let out = match asm.rotl64(Imm(*val), *amount, dest) {
                Imm(v) => v,
                _ => panic!("rotl64 immediate path should constant-fold"),
            };
            assert_eq!(out, *expected, "rotl64({val:#018x}, {amount})");
        }
    }
}
