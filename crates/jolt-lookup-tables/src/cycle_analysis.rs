//! Flag and lookup-table-aware computations over `tracer::Cycle`.
//!
//! These extend `CycleRow` (defined in `jolt-trace`) with methods that need
//! `LookupTableKind` / `interleave_bits`, which is why they live here rather
//! than in jolt-trace.

use jolt_trace::flags::{CircuitFlagSet, CircuitFlags, Flags, InstructionFlagSet, InstructionFlags};
use jolt_trace::{instructions, CycleRow};
use tracer::instruction::{Cycle, Instruction, NormalizedInstruction};

use crate::{interleave_bits, InstructionLookupTable, LookupTableKind};

/// Cycle-level computations that depend on lookup tables and instruction flags.
pub trait CycleAnalysis: CycleRow {
    /// R1CS circuit flags (packed bitfield, indexed by `CircuitFlags`).
    fn circuit_flags(&self) -> CircuitFlagSet;

    /// Non-R1CS instruction flags (packed bitfield, indexed by `InstructionFlags`).
    fn instruction_flags(&self) -> InstructionFlagSet;

    /// Combined lookup index for RA polynomial construction (128-bit).
    fn lookup_index(&self) -> u128;

    /// Lookup table evaluation result.
    ///
    /// For arithmetic: the computation result (e.g., rs1 + rs2 for ADD).
    /// For branches: the comparison result (0 or 1).
    /// For stores: zero.
    /// For no-ops: zero.
    ///
    /// This is the value of V_LOOKUP_OUTPUT in the R1CS witness.
    fn lookup_output(&self) -> u64;

    /// Index of the lookup table this instruction uses, or `None` for no-ops.
    ///
    /// The index corresponds to `LookupTableFlag(i)` — a per-cycle boolean
    /// that's 1 iff this cycle uses table `i`. Used by BytecodeReadRaf's
    /// multi-stage input claim.
    fn lookup_table_index(&self) -> Option<usize>;
}

/// Map an `Instruction` variant to its ISA struct, bind it to `$i`, evaluate `$body`.
/// The `noop =>` arm handles `Instruction::NoOp` separately.
macro_rules! with_isa_struct {
    ($instr:expr, |$i:ident| $body:expr, noop => $noop:expr) => {{
        match $instr {
            Instruction::ADD(_) => {
                let $i = instructions::Add;
                $body
            }
            Instruction::ADDI(_) => {
                let $i = instructions::Addi;
                $body
            }
            Instruction::SUB(_) => {
                let $i = instructions::Sub;
                $body
            }
            Instruction::LUI(_) => {
                let $i = instructions::Lui;
                $body
            }
            Instruction::AUIPC(_) => {
                let $i = instructions::Auipc;
                $body
            }
            Instruction::MUL(_) => {
                let $i = instructions::Mul;
                $body
            }
            Instruction::MULHU(_) => {
                let $i = instructions::MulHU;
                $body
            }
            Instruction::AND(_) => {
                let $i = instructions::And;
                $body
            }
            Instruction::ANDI(_) => {
                let $i = instructions::AndI;
                $body
            }
            Instruction::ANDN(_) => {
                let $i = instructions::Andn;
                $body
            }
            Instruction::OR(_) => {
                let $i = instructions::Or;
                $body
            }
            Instruction::ORI(_) => {
                let $i = instructions::OrI;
                $body
            }
            Instruction::XOR(_) => {
                let $i = instructions::Xor;
                $body
            }
            Instruction::XORI(_) => {
                let $i = instructions::XorI;
                $body
            }
            Instruction::SLT(_) => {
                let $i = instructions::Slt;
                $body
            }
            Instruction::SLTI(_) => {
                let $i = instructions::SltI;
                $body
            }
            Instruction::SLTIU(_) => {
                let $i = instructions::SltIU;
                $body
            }
            Instruction::SLTU(_) => {
                let $i = instructions::SltU;
                $body
            }
            Instruction::BEQ(_) => {
                let $i = instructions::Beq;
                $body
            }
            Instruction::BGE(_) => {
                let $i = instructions::Bge;
                $body
            }
            Instruction::BGEU(_) => {
                let $i = instructions::BgeU;
                $body
            }
            Instruction::BLT(_) => {
                let $i = instructions::Blt;
                $body
            }
            Instruction::BLTU(_) => {
                let $i = instructions::BltU;
                $body
            }
            Instruction::BNE(_) => {
                let $i = instructions::Bne;
                $body
            }
            Instruction::JAL(_) => {
                let $i = instructions::Jal;
                $body
            }
            Instruction::JALR(_) => {
                let $i = instructions::Jalr;
                $body
            }
            Instruction::LD(_) => {
                let $i = instructions::Ld;
                $body
            }
            Instruction::SD(_) => {
                let $i = instructions::Sd;
                $body
            }
            Instruction::EBREAK(_) => {
                let $i = instructions::Ebreak;
                $body
            }
            Instruction::ECALL(_) => {
                let $i = instructions::Ecall;
                $body
            }
            Instruction::FENCE(_) => {
                let $i = instructions::Fence;
                $body
            }
            Instruction::VirtualAdvice(_) => {
                let $i = instructions::VirtualAdvice;
                $body
            }
            Instruction::VirtualAdviceLen(_) => {
                let $i = instructions::VirtualAdviceLen;
                $body
            }
            Instruction::VirtualAdviceLoad(_) => {
                let $i = instructions::VirtualAdviceLoad;
                $body
            }
            Instruction::VirtualHostIO(_) => {
                let $i = instructions::VirtualHostIO;
                $body
            }
            Instruction::VirtualMULI(_) => {
                let $i = instructions::MulI;
                $body
            }
            Instruction::VirtualPow2(_) => {
                let $i = instructions::Pow2;
                $body
            }
            Instruction::VirtualPow2I(_) => {
                let $i = instructions::Pow2I;
                $body
            }
            Instruction::VirtualPow2W(_) => {
                let $i = instructions::Pow2W;
                $body
            }
            Instruction::VirtualPow2IW(_) => {
                let $i = instructions::Pow2IW;
                $body
            }
            Instruction::VirtualAssertEQ(_) => {
                let $i = instructions::AssertEq;
                $body
            }
            Instruction::VirtualAssertLTE(_) => {
                let $i = instructions::AssertLte;
                $body
            }
            Instruction::VirtualAssertValidDiv0(_) => {
                let $i = instructions::AssertValidDiv0;
                $body
            }
            Instruction::VirtualAssertValidUnsignedRemainder(_) => {
                let $i = instructions::AssertValidUnsignedRemainder;
                $body
            }
            Instruction::VirtualAssertMulUNoOverflow(_) => {
                let $i = instructions::AssertMulUNoOverflow;
                $body
            }
            Instruction::VirtualAssertWordAlignment(_) => {
                let $i = instructions::AssertWordAlignment;
                $body
            }
            Instruction::VirtualAssertHalfwordAlignment(_) => {
                let $i = instructions::AssertHalfwordAlignment;
                $body
            }
            Instruction::VirtualMovsign(_) => {
                let $i = instructions::MovSign;
                $body
            }
            Instruction::VirtualRev8W(_) => {
                let $i = instructions::VirtualRev8W;
                $body
            }
            Instruction::VirtualChangeDivisor(_) => {
                let $i = instructions::VirtualChangeDivisor;
                $body
            }
            Instruction::VirtualChangeDivisorW(_) => {
                let $i = instructions::VirtualChangeDivisorW;
                $body
            }
            Instruction::VirtualZeroExtendWord(_) => {
                let $i = instructions::VirtualZeroExtendWord;
                $body
            }
            Instruction::VirtualSignExtendWord(_) => {
                let $i = instructions::VirtualSignExtendWord;
                $body
            }
            Instruction::VirtualSRL(_) => {
                let $i = instructions::VirtualSrl;
                $body
            }
            Instruction::VirtualSRLI(_) => {
                let $i = instructions::VirtualSrli;
                $body
            }
            Instruction::VirtualSRA(_) => {
                let $i = instructions::VirtualSra;
                $body
            }
            Instruction::VirtualSRAI(_) => {
                let $i = instructions::VirtualSrai;
                $body
            }
            Instruction::VirtualShiftRightBitmask(_) => {
                let $i = instructions::VirtualShiftRightBitmask;
                $body
            }
            Instruction::VirtualShiftRightBitmaskI(_) => {
                let $i = instructions::VirtualShiftRightBitmaski;
                $body
            }
            Instruction::VirtualROTRI(_) => {
                let $i = instructions::VirtualRotri;
                $body
            }
            Instruction::VirtualROTRIW(_) => {
                let $i = instructions::VirtualRotriw;
                $body
            }
            Instruction::VirtualXORROT32(_) => {
                let $i = instructions::VirtualXorRot32;
                $body
            }
            Instruction::VirtualXORROT24(_) => {
                let $i = instructions::VirtualXorRot24;
                $body
            }
            Instruction::VirtualXORROT16(_) => {
                let $i = instructions::VirtualXorRot16;
                $body
            }
            Instruction::VirtualXORROT63(_) => {
                let $i = instructions::VirtualXorRot63;
                $body
            }
            Instruction::VirtualXORROTW16(_) => {
                let $i = instructions::VirtualXorRotW16;
                $body
            }
            Instruction::VirtualXORROTW12(_) => {
                let $i = instructions::VirtualXorRotW12;
                $body
            }
            Instruction::VirtualXORROTW8(_) => {
                let $i = instructions::VirtualXorRotW8;
                $body
            }
            Instruction::VirtualXORROTW7(_) => {
                let $i = instructions::VirtualXorRotW7;
                $body
            }
            Instruction::NoOp => $noop,
            Instruction::INLINE(x) => panic!(
                "INLINE reached CycleAnalysis: opcode={}, funct3={}, funct7={}",
                x.opcode, x.funct3, x.funct7
            ),
            _ => panic!("unsupported instruction: {:?}", $instr),
        }
    }};
}

impl CycleAnalysis for Cycle {
    fn circuit_flags(&self) -> CircuitFlagSet {
        let instr = self.instruction();
        let mut flags = static_circuit_flags(&instr);
        let norm = instr.normalize();
        flags = apply_dynamic_flags(flags, &norm);
        if matches!(instr, Instruction::JALR(_)) && norm.virtual_sequence_remaining == Some(0) {
            flags = flags.set(CircuitFlags::IsLastInSequence);
        }
        flags
    }

    fn instruction_flags(&self) -> InstructionFlagSet {
        let instr = self.instruction();
        let mut flags = static_instruction_flags(&instr);
        let norm = instr.normalize();
        if matches!(norm.operands.rd, Some(rd) if rd != 0) {
            flags = flags.set(InstructionFlags::IsRdNotZero);
        }
        flags
    }

    fn lookup_index(&self) -> u128 {
        let cflags = CycleAnalysis::circuit_flags(self);
        let iflags = CycleAnalysis::instruction_flags(self);
        let (left, right) = instruction_inputs(self, &iflags);

        if cflags[CircuitFlags::AddOperands] {
            (left as u128).wrapping_add(right)
        } else if cflags[CircuitFlags::SubtractOperands] {
            (1u128 << 64).wrapping_sub(right).wrapping_add(left as u128)
        } else if cflags[CircuitFlags::MultiplyOperands] {
            (left as u128).wrapping_mul(right)
        } else if cflags[CircuitFlags::Advice] {
            self.rd_write().map_or(0, |(_, _, post)| post as u128)
        } else if self.is_noop() {
            0
        } else {
            interleave_bits(left, right as u64)
        }
    }

    fn lookup_table_index(&self) -> Option<usize> {
        if self.is_noop() {
            return None;
        }
        lookup_table_kind(&self.instruction()).map(|k| k as usize)
    }

    fn lookup_output(&self) -> u64 {
        if self.is_noop() {
            return 0;
        }
        let cflags = CycleAnalysis::circuit_flags(self);
        let iflags = CycleAnalysis::instruction_flags(self);

        if cflags[CircuitFlags::Jump] {
            // JAL/JALR: lookup output = jump target, not return address.
            let left = if iflags[InstructionFlags::LeftOperandIsPC] {
                self.unexpanded_pc()
            } else {
                self.rs1_read().map_or(0, |(_, v)| v)
            };
            let target = (left as i64).wrapping_add(self.imm() as i64) as u64;
            if iflags[InstructionFlags::LeftOperandIsRs1Value] {
                target & !1 // JALR aligns to 2-byte boundary
            } else {
                target
            }
        } else if cflags[CircuitFlags::Assert] {
            1
        } else if iflags[InstructionFlags::Branch] {
            let rs1 = self.rs1_read().map_or(0, |(_, v)| v);
            let rs2 = self.rs2_read().map_or(0, |(_, v)| v);
            branch_result(self.instruction(), rs1, rs2)
        } else if cflags[CircuitFlags::WriteLookupOutputToRD] {
            self.rd_write().map_or(0, |(_, _, post)| post)
        } else {
            0
        }
    }
}

fn instruction_inputs(cycle: &impl CycleRow, iflags: &InstructionFlagSet) -> (u64, u128) {
    let left = if iflags[InstructionFlags::LeftOperandIsPC] {
        cycle.unexpanded_pc()
    } else if iflags[InstructionFlags::LeftOperandIsRs1Value] {
        cycle.rs1_read().map_or(0, |(_, v)| v)
    } else {
        0
    };

    let right: i128 = if iflags[InstructionFlags::RightOperandIsImm] {
        cycle.imm()
    } else if iflags[InstructionFlags::RightOperandIsRs2Value] {
        cycle.rs2_read().map_or(0, |(_, v)| v as i128)
    } else {
        0
    };

    (left, right as u64 as u128)
}

fn static_circuit_flags(instr: &Instruction) -> CircuitFlagSet {
    with_isa_struct!(instr, |i| Flags::circuit_flags(&i), noop => {
        CircuitFlagSet::default().set(CircuitFlags::DoNotUpdateUnexpandedPC)
    })
}

fn static_instruction_flags(instr: &Instruction) -> InstructionFlagSet {
    with_isa_struct!(instr, |i| Flags::instruction_flags(&i), noop => {
        InstructionFlagSet::default().set(InstructionFlags::IsNoop)
    })
}

fn lookup_table_kind(instr: &Instruction) -> Option<LookupTableKind> {
    with_isa_struct!(instr, |i| InstructionLookupTable::lookup_table(&i), noop => None)
}

fn branch_result(instr: Instruction, rs1: u64, rs2: u64) -> u64 {
    let taken = match instr {
        Instruction::BEQ(_) => rs1 == rs2,
        Instruction::BNE(_) => rs1 != rs2,
        Instruction::BLT(_) => (rs1 as i64) < (rs2 as i64),
        Instruction::BGE(_) => (rs1 as i64) >= (rs2 as i64),
        Instruction::BLTU(_) => rs1 < rs2,
        Instruction::BGEU(_) => rs1 >= rs2,
        _ => false,
    };
    taken as u64
}

fn apply_dynamic_flags(mut flags: CircuitFlagSet, norm: &NormalizedInstruction) -> CircuitFlagSet {
    if norm.virtual_sequence_remaining.is_some() {
        flags = flags.set(CircuitFlags::VirtualInstruction);
    }
    if norm.virtual_sequence_remaining.unwrap_or(0) != 0 {
        flags = flags.set(CircuitFlags::DoNotUpdateUnexpandedPC);
    }
    if norm.is_first_in_sequence {
        flags = flags.set(CircuitFlags::IsFirstInSequence);
    }
    if norm.is_compressed {
        flags = flags.set(CircuitFlags::IsCompressed);
    }
    flags
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_flags_and_lookup() {
        let noop = <Cycle as CycleRow>::noop();
        let cflags = CycleAnalysis::circuit_flags(&noop);
        assert!(cflags[CircuitFlags::DoNotUpdateUnexpandedPC]);

        let iflags = CycleAnalysis::instruction_flags(&noop);
        assert!(iflags[InstructionFlags::IsNoop]);

        assert_eq!(CycleAnalysis::lookup_index(&noop), 0);
    }
}
