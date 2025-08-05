//! InstrAssembler
//!
//! Builds and owns a vector of virtual RISC-V instructions (`RV32IMInstruction`).
//! The struct provides small helper methods so that higher-level builders can emit
//! common instructions without repeating encoding boiler-plate.
//!
//! This module focuses purely on the mechanical task of
//! pushing the correct instruction variant into `sequence`, and is meant to be used
//! by higher-level builders.
//!
//! Assumptions:
//! * The helpers are infallible (in terms of error handling) and run in constant time.
//! * Composite helpers such as `xor64` and `rotl64` expand into multiple 32-bit
//!   operations; the exact policy is documented where they are defined.
use crate::instruction::and::AND;
use crate::instruction::andi::ANDI;
use crate::instruction::andn::ANDN;
use crate::instruction::format::format_i::FormatI;
use crate::instruction::format::format_r::FormatR;
use crate::instruction::format::format_s::FormatS;
use crate::instruction::format::format_virtual_right_shift_i::FormatVirtualRightShiftI;
use crate::instruction::ld::LD;
use crate::instruction::sd::SD;
use crate::instruction::virtual_rotri::VirtualROTRI;
use crate::instruction::xor::XOR;
use crate::instruction::xori::XORI;
use crate::instruction::RV32IMInstruction;

/// Operand that can be either an immediate or a register.
#[derive(Clone, Copy, Debug)]
pub enum Value {
    Imm(u64),
    Reg(usize),
}
use Value::{Imm, Reg};

/// Convenience assembler for building a sequence of `RV32IMInstruction`s.
/// Includes common instruction emitters to be used by higher-level builders for inlines.
///
/// The assembler stores `address` and `is_compressed` so that
/// each emitted instruction carries the right metadata.
#[derive(Debug, Default)]
pub struct InstrAssembler {
    /// Program counter associated with the *first* instruction; builders are
    /// responsible for post-processing if they need exact PCs per-instruction.
    pub address: u64,
    /// Whether to use RVC encodings.
    pub is_compressed: bool,
    /// Accumulated instruction buffer.
    sequence: Vec<RV32IMInstruction>,
}

impl InstrAssembler {
    /// Create a new assembler with an empty instruction buffer.
    pub fn new(address: u64, is_compressed: bool) -> Self {
        Self {
            address,
            is_compressed,
            sequence: Vec::new(),
        }
    }

    /// Finalize the instruction buffer: back-fill `virtual_sequence_remaining`
    /// and return ownership of the underlying `Vec`.
    pub fn finalize(mut self) -> Vec<RV32IMInstruction> {
        let len = self.sequence.len();
        for (i, instr) in self.sequence.iter_mut().enumerate() {
            instr.set_virtual_sequence_remaining(Some(len - i - 1));
        }
        self.sequence
    }

    /// Consume the assembler and return the internal sequence **without**
    /// back-filling `virtual_sequence_remaining`.  Prefer `finalize()` unless you
    /// have a specific reason.
    pub fn into_inner(self) -> Vec<RV32IMInstruction> {
        self.sequence
    }

    // ---------------------------------------------------------------------
    // RV64 Load/Store helpers (Keccak needs 64-bit variants)
    // ---------------------------------------------------------------------

    /// Emit `LD rd, offset(rs1)` where `offset` is in *lanes* (8-byte units).
    #[inline]
    pub fn ld(&mut self, rs1: usize, offset_lanes: i64, rd: usize) {
        let inst = LD {
            address: self.address,
            operands: FormatI {
                rd,
                rs1,
                imm: (offset_lanes * 8) as u64,
            },
            virtual_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        self.sequence.push(inst.into());
    }

    /// Emit `SD rs2, offset(rs1)` where `offset` is in *lanes* (8-byte units).
    #[inline]
    pub fn sd(&mut self, rs1: usize, rs2: usize, offset_lanes: i64) {
        let inst = SD {
            address: self.address,
            operands: FormatS {
                rs1,
                rs2,
                imm: offset_lanes * 8,
            },
            virtual_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        self.sequence.push(inst.into());
    }

    // ---------------------------------------------------------------------
    // Logical & immediate helpers
    // ---------------------------------------------------------------------

    /// Emit `XOR rd, rs1, rs2` / `XORI rd, rs1, imm` and return `Reg(rd)` or
    /// an `Imm` result when both operands are immediates.
    pub fn xor(&mut self, rs1: Value, rs2: Value, rd: usize) -> Value {
        match (rs1, rs2) {
            (Reg(rs1), Reg(rs2)) => {
                let inst = XOR {
                    address: self.address,
                    operands: FormatR { rd, rs1, rs2 },
                    virtual_sequence_remaining: Some(0),
                    is_compressed: self.is_compressed,
                };
                self.sequence.push(inst.into());
                Reg(rd)
            }
            (Reg(rs1), Imm(imm)) => {
                let inst = XORI {
                    address: self.address,
                    operands: FormatI { rd, rs1, imm },
                    virtual_sequence_remaining: Some(0),
                    is_compressed: self.is_compressed,
                };
                self.sequence.push(inst.into());
                Reg(rd)
            }
            (Imm(_), Reg(_)) => self.xor(rs2, rs1, rd), // swap & reuse logic
            (Imm(imm1), Imm(imm2)) => Imm(imm1 ^ imm2),
        }
    }

    /// Emit `AND`/`ANDI` similarly to `xor`.
    pub fn and(&mut self, rs1: Value, rs2: Value, rd: usize) -> Value {
        match (rs1, rs2) {
            (Reg(rs1), Reg(rs2)) => {
                let inst = AND {
                    address: self.address,
                    operands: FormatR { rd, rs1, rs2 },
                    virtual_sequence_remaining: Some(0),
                    is_compressed: self.is_compressed,
                };
                self.sequence.push(inst.into());
                Reg(rd)
            }
            (Reg(rs1), Imm(imm)) => {
                let inst = ANDI {
                    address: self.address,
                    operands: FormatI { rd, rs1, imm },
                    virtual_sequence_remaining: Some(0),
                    is_compressed: self.is_compressed,
                };
                self.sequence.push(inst.into());
                Reg(rd)
            }
            (Imm(_), Reg(_)) => self.and(rs2, rs1, rd),
            (Imm(imm1), Imm(imm2)) => Imm(imm1 & imm2),
        }
    }

    /// Emit virtual `ROTRI` (rotate-right immediate) or compute constant fold.
    pub fn rotri(&mut self, rs1: Value, imm: u64, rd: usize) -> Value {
        match rs1 {
            Reg(rs1) => {
                let inst = VirtualROTRI {
                    address: self.address,
                    operands: FormatVirtualRightShiftI { rd, rs1, imm },
                    virtual_sequence_remaining: Some(0),
                    is_compressed: self.is_compressed,
                };
                self.sequence.push(inst.into());
                Reg(rd)
            }
            Imm(val) => Imm(val.rotate_right(imm as u32)),
        }
    }

    // ---------------------------------------------------------------------
    // 64-bit composite helpers (policy fixed, documented here)
    // ---------------------------------------------------------------------

    /// A & ~B via the `ANDN` (bit-clear) instruction when both operands are
    /// registers.  When immediates are involved we fall back to constant-fold
    /// or NOT+AND sequences.  Scratch-register handling is left to the caller.
    pub fn andn64(&mut self, rs1: Value, rs2: Value, rd: usize) -> Value {
        match (rs1, rs2) {
            (Reg(rs1), Reg(rs2)) => {
                let inst = ANDN {
                    address: self.address,
                    operands: FormatR { rd, rs1, rs2 },
                    virtual_sequence_remaining: Some(0),
                    is_compressed: self.is_compressed,
                };
                self.sequence.push(inst.into());
                Reg(rd)
            }
            (Imm(imm1), Imm(imm2)) => Imm(imm1 & !imm2),
            // Mixed case: caller must provide scratch register management.
            (Reg(_), Imm(_)) | (Imm(_), Reg(_)) => {
                panic!("andn64 with mixed operands requires caller-provided NOT+AND sequence")
            }
        }
    }

    /// NOT a 64-bit value by XORing with `u64::MAX`.
    #[inline]
    pub fn not64(&mut self, rs1: Value, rd: usize) -> Value {
        self.xor(rs1, Imm(u64::MAX), rd)
    }

    /// Rotate left on a 64-bit value.
    pub fn rotl64(&mut self, rs1: Value, amount: u32, rd: usize) -> Value {
        if amount == 0 {
            // Identity rotation: emit XOR with zero to copy value.
            return self.xor(rs1, Imm(0), rd);
        }

        match rs1 {
            Reg(rs1_reg) => {
                // rotl(n) == rotr(64 − n).  The VirtualROTRI instruction encodes
                // the rotation via a bitmask whose trailing zeros indicate the
                // shift amount.  We construct that mask and delegate to `rotri`
                // to avoid duplication.
                let ones = (1u64 << amount as u64) - 1;
                let imm = ones << (64 - amount as u64);
                self.rotri(Reg(rs1_reg), imm, rd)
            }
            Imm(val) => Imm(val.rotate_left(amount)),
        }
    }
}
