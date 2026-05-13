//! `CycleRow` implementation for `tracer::Cycle`.
//!
//! Maps `Instruction` variants to ISA structs via [`with_isa_struct!`], then
//! derives circuit flags and instruction flags.

use jolt_riscv::{CircuitFlagSet, CircuitFlags, Flags, InstructionFlagSet, InstructionFlags};
use jolt_witness::FrCycleBytecode;
use tracer::instruction::field_op;
use tracer::instruction::{Cycle, Instruction, RAMAccess};

use crate::CycleRow;

/// Map an `Instruction` variant to its ISA struct, bind it to `$i`, evaluate `$body`.
/// The `noop =>` arm handles `Instruction::NoOp` separately.
///
/// Exported so downstream crates (e.g. `jolt-host`) can dispatch a
/// `&Instruction` to its concrete ISA struct for traits whose impls live
/// in `jolt-lookup-tables` etc.
#[macro_export]
macro_rules! with_isa_struct {
    ($instr:expr, |$i:ident| $body:expr, noop => $noop:expr) => {{
        use jolt_riscv::instructions::*;
        match $instr {
            Instruction::ADD(value) => {
                let $i = Add(*value);
                $body
            }
            Instruction::ADDI(value) => {
                let $i = Addi(*value);
                $body
            }
            Instruction::SUB(value) => {
                let $i = Sub(*value);
                $body
            }
            Instruction::LUI(value) => {
                let $i = Lui(*value);
                $body
            }
            Instruction::AUIPC(value) => {
                let $i = Auipc(*value);
                $body
            }
            Instruction::MUL(value) => {
                let $i = Mul(*value);
                $body
            }
            Instruction::MULHU(value) => {
                let $i = MulHU(*value);
                $body
            }
            Instruction::AND(value) => {
                let $i = And(*value);
                $body
            }
            Instruction::ANDI(value) => {
                let $i = AndI(*value);
                $body
            }
            Instruction::ANDN(value) => {
                let $i = Andn(*value);
                $body
            }
            Instruction::OR(value) => {
                let $i = Or(*value);
                $body
            }
            Instruction::ORI(value) => {
                let $i = OrI(*value);
                $body
            }
            Instruction::XOR(value) => {
                let $i = Xor(*value);
                $body
            }
            Instruction::XORI(value) => {
                let $i = XorI(*value);
                $body
            }
            Instruction::SLT(value) => {
                let $i = Slt(*value);
                $body
            }
            Instruction::SLTI(value) => {
                let $i = SltI(*value);
                $body
            }
            Instruction::SLTIU(value) => {
                let $i = SltIU(*value);
                $body
            }
            Instruction::SLTU(value) => {
                let $i = SltU(*value);
                $body
            }
            Instruction::BEQ(value) => {
                let $i = Beq(*value);
                $body
            }
            Instruction::BGE(value) => {
                let $i = Bge(*value);
                $body
            }
            Instruction::BGEU(value) => {
                let $i = BgeU(*value);
                $body
            }
            Instruction::BLT(value) => {
                let $i = Blt(*value);
                $body
            }
            Instruction::BLTU(value) => {
                let $i = BltU(*value);
                $body
            }
            Instruction::BNE(value) => {
                let $i = Bne(*value);
                $body
            }
            Instruction::JAL(value) => {
                let $i = Jal(*value);
                $body
            }
            Instruction::JALR(value) => {
                let $i = Jalr(*value);
                $body
            }
            Instruction::LD(value) => {
                let $i = Ld(*value);
                $body
            }
            Instruction::SD(value) => {
                let $i = Sd(*value);
                $body
            }
            Instruction::EBREAK(value) => {
                let $i = Ebreak(*value);
                $body
            }
            Instruction::ECALL(value) => {
                let $i = Ecall(*value);
                $body
            }
            Instruction::FENCE(value) => {
                let $i = Fence(*value);
                $body
            }
            Instruction::VirtualAdvice(value) => {
                let $i = VirtualAdvice(*value);
                $body
            }
            Instruction::VirtualAdviceLen(value) => {
                let $i = VirtualAdviceLen(*value);
                $body
            }
            Instruction::VirtualAdviceLoad(value) => {
                let $i = VirtualAdviceLoad(*value);
                $body
            }
            Instruction::VirtualHostIO(value) => {
                let $i = VirtualHostIO(*value);
                $body
            }
            Instruction::VirtualMULI(value) => {
                let $i = MulI(*value);
                $body
            }
            Instruction::VirtualPow2(value) => {
                let $i = Pow2(*value);
                $body
            }
            Instruction::VirtualPow2I(value) => {
                let $i = Pow2I(*value);
                $body
            }
            Instruction::VirtualPow2W(value) => {
                let $i = Pow2W(*value);
                $body
            }
            Instruction::VirtualPow2IW(value) => {
                let $i = Pow2IW(*value);
                $body
            }
            Instruction::VirtualAssertEQ(value) => {
                let $i = AssertEq(*value);
                $body
            }
            Instruction::VirtualAssertLTE(value) => {
                let $i = AssertLte(*value);
                $body
            }
            Instruction::VirtualAssertValidDiv0(value) => {
                let $i = AssertValidDiv0(*value);
                $body
            }
            Instruction::VirtualAssertValidUnsignedRemainder(value) => {
                let $i = AssertValidUnsignedRemainder(*value);
                $body
            }
            Instruction::VirtualAssertMulUNoOverflow(value) => {
                let $i = AssertMulUNoOverflow(*value);
                $body
            }
            Instruction::VirtualAssertWordAlignment(value) => {
                let $i = AssertWordAlignment(*value);
                $body
            }
            Instruction::VirtualAssertHalfwordAlignment(value) => {
                let $i = AssertHalfwordAlignment(*value);
                $body
            }
            Instruction::VirtualMovsign(value) => {
                let $i = MovSign(*value);
                $body
            }
            Instruction::VirtualRev8W(value) => {
                let $i = VirtualRev8W(*value);
                $body
            }
            Instruction::VirtualChangeDivisor(value) => {
                let $i = VirtualChangeDivisor(*value);
                $body
            }
            Instruction::VirtualChangeDivisorW(value) => {
                let $i = VirtualChangeDivisorW(*value);
                $body
            }
            Instruction::VirtualZeroExtendWord(value) => {
                let $i = VirtualZeroExtendWord(*value);
                $body
            }
            Instruction::VirtualSignExtendWord(value) => {
                let $i = VirtualSignExtendWord(*value);
                $body
            }
            Instruction::VirtualSRL(value) => {
                let $i = VirtualSrl(*value);
                $body
            }
            Instruction::VirtualSRLI(value) => {
                let $i = VirtualSrli(*value);
                $body
            }
            Instruction::VirtualSRA(value) => {
                let $i = VirtualSra(*value);
                $body
            }
            Instruction::VirtualSRAI(value) => {
                let $i = VirtualSrai(*value);
                $body
            }
            Instruction::VirtualShiftRightBitmask(value) => {
                let $i = VirtualShiftRightBitmask(*value);
                $body
            }
            Instruction::VirtualShiftRightBitmaskI(value) => {
                let $i = VirtualShiftRightBitmaski(*value);
                $body
            }
            Instruction::VirtualROTRI(value) => {
                let $i = VirtualRotri(*value);
                $body
            }
            Instruction::VirtualROTRIW(value) => {
                let $i = VirtualRotriw(*value);
                $body
            }
            Instruction::VirtualXORROT32(value) => {
                let $i = VirtualXorRot32(*value);
                $body
            }
            Instruction::VirtualXORROT24(value) => {
                let $i = VirtualXorRot24(*value);
                $body
            }
            Instruction::VirtualXORROT16(value) => {
                let $i = VirtualXorRot16(*value);
                $body
            }
            Instruction::VirtualXORROT63(value) => {
                let $i = VirtualXorRot63(*value);
                $body
            }
            Instruction::VirtualXORROTW16(value) => {
                let $i = VirtualXorRotW16(*value);
                $body
            }
            Instruction::VirtualXORROTW12(value) => {
                let $i = VirtualXorRotW12(*value);
                $body
            }
            Instruction::VirtualXORROTW8(value) => {
                let $i = VirtualXorRotW8(*value);
                $body
            }
            Instruction::VirtualXORROTW7(value) => {
                let $i = VirtualXorRotW7(*value);
                $body
            }
            Instruction::FieldOp(value) => {
                // FieldOp is the shared tracer struct for FMUL/FADD/FSUB/FINV
                // discriminated by funct3. Dispatch to the matching jolt-riscv
                // wrapper so per-variant circuit_flags fire correctly.
                use tracer::instruction::field_op::{
                    FUNCT3_FADD, FUNCT3_FINV, FUNCT3_FMUL, FUNCT3_FSUB,
                };
                match value.funct3 {
                    FUNCT3_FMUL => {
                        let $i = FieldMul(*value);
                        $body
                    }
                    FUNCT3_FADD => {
                        let $i = FieldAdd(*value);
                        $body
                    }
                    FUNCT3_FSUB => {
                        let $i = FieldSub(*value);
                        $body
                    }
                    FUNCT3_FINV => {
                        let $i = FieldInv(*value);
                        $body
                    }
                    _ => panic!(
                        "unsupported FieldOp funct3 = 0x{:02x}",
                        value.funct3
                    ),
                }
            }
            Instruction::FieldAssertEq(value) => {
                let $i = FieldAssertEq(*value);
                $body
            }
            Instruction::FieldMov(value) => {
                let $i = FieldMov(*value);
                $body
            }
            Instruction::FieldSLL64(value) => {
                let $i = FieldSLL64(*value);
                $body
            }
            Instruction::FieldSLL128(value) => {
                let $i = FieldSLL128(*value);
                $body
            }
            Instruction::FieldSLL192(value) => {
                let $i = FieldSLL192(*value);
                $body
            }
            Instruction::NoOp => $noop,
            Instruction::INLINE(x) => panic!(
                "INLINE reached CycleRow: opcode={}, funct3={}, funct7={}",
                x.opcode, x.funct3, x.funct7
            ),
            _ => panic!("unsupported instruction: {:?}", $instr),
        }
    }};
}

impl CycleRow for Cycle {
    fn noop() -> Self {
        Cycle::NoOp
    }

    fn is_noop(&self) -> bool {
        matches!(self, Cycle::NoOp)
    }

    fn unexpanded_pc(&self) -> u64 {
        match self {
            Cycle::NoOp => 0,
            _ => self.instruction().normalize().address as u64,
        }
    }

    fn virtual_sequence_remaining(&self) -> Option<u16> {
        match self {
            Cycle::NoOp => None,
            _ => self.instruction().normalize().virtual_sequence_remaining,
        }
    }

    fn is_first_in_sequence(&self) -> bool {
        match self {
            Cycle::NoOp => false,
            _ => self.instruction().normalize().is_first_in_sequence,
        }
    }

    fn is_virtual(&self) -> bool {
        self.virtual_sequence_remaining().is_some()
    }

    fn rs1_read(&self) -> Option<(u8, u64)> {
        self.rs1_read()
    }

    fn rs2_read(&self) -> Option<(u8, u64)> {
        self.rs2_read()
    }

    fn rd_write(&self) -> Option<(u8, u64, u64)> {
        self.rd_write()
    }

    fn rd_operand(&self) -> Option<u8> {
        match self {
            Cycle::NoOp => None,
            _ => self.instruction().normalize().operands.rd,
        }
    }

    fn ram_access_address(&self) -> Option<u64> {
        match self.ram_access() {
            RAMAccess::Read(r) => Some(r.address),
            RAMAccess::Write(w) => Some(w.address),
            RAMAccess::NoOp => None,
        }
    }

    fn ram_read_value(&self) -> Option<u64> {
        match self.ram_access() {
            RAMAccess::Read(r) => Some(r.value),
            RAMAccess::Write(w) => Some(w.pre_value),
            RAMAccess::NoOp => None,
        }
    }

    fn ram_write_value(&self) -> Option<u64> {
        match self.ram_access() {
            RAMAccess::Read(r) => Some(r.value),
            RAMAccess::Write(w) => Some(w.post_value),
            RAMAccess::NoOp => None,
        }
    }

    fn imm(&self) -> i128 {
        match self {
            Cycle::NoOp => 0,
            _ => self.instruction().normalize().operands.imm,
        }
    }

    fn circuit_flags(&self) -> CircuitFlagSet {
        static_circuit_flags(&self.instruction())
    }

    fn instruction_flags(&self) -> InstructionFlagSet {
        static_instruction_flags(&self.instruction())
    }

    fn lookup_index(&self) -> u128 {
        let cflags = self.circuit_flags();
        let iflags = self.instruction_flags();
        let (left, right) = instruction_inputs(self, iflags);

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

    fn lookup_output(&self) -> u64 {
        if self.is_noop() {
            return 0;
        }
        let cflags = self.circuit_flags();
        let iflags = self.instruction_flags();

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

    fn fr_meta(&self) -> FrCycleBytecode {
        // FR slot indices are 5 bits in the encoding; the high bit must be 0
        // (slots are 0..=15). Mask once at the producer per audit N4.
        const SLOT_MASK: u8 = 0x0F;
        match self.instruction() {
            Instruction::FieldOp(op) => {
                let reads_frs2 = op.funct3 != field_op::FUNCT3_FINV;
                FrCycleBytecode {
                    frs1: op.operands.rs1 & SLOT_MASK,
                    frs2: op.operands.rs2 & SLOT_MASK,
                    frd: op.operands.rd & SLOT_MASK,
                    reads_frs1: true,
                    reads_frs2,
                    writes_frd: true,
                }
            }
            Instruction::FieldAssertEq(op) => FrCycleBytecode {
                frs1: op.operands.rs1 & SLOT_MASK,
                frs2: op.operands.rs2 & SLOT_MASK,
                frd: 0,
                reads_frs1: true,
                reads_frs2: true,
                writes_frd: false,
            },
            Instruction::FieldMov(op) => FrCycleBytecode {
                frs1: 0,
                frs2: 0,
                frd: op.operands.rd & SLOT_MASK,
                reads_frs1: false,
                reads_frs2: false,
                writes_frd: true,
            },
            Instruction::FieldSLL64(op) => FrCycleBytecode {
                frs1: 0,
                frs2: 0,
                frd: op.operands.rd & SLOT_MASK,
                reads_frs1: false,
                reads_frs2: false,
                writes_frd: true,
            },
            Instruction::FieldSLL128(op) => FrCycleBytecode {
                frs1: 0,
                frs2: 0,
                frd: op.operands.rd & SLOT_MASK,
                reads_frs1: false,
                reads_frs2: false,
                writes_frd: true,
            },
            Instruction::FieldSLL192(op) => FrCycleBytecode {
                frs1: 0,
                frs2: 0,
                frd: op.operands.rd & SLOT_MASK,
                reads_frs1: false,
                reads_frs2: false,
                writes_frd: true,
            },
            _ => FrCycleBytecode::default(),
        }
    }
}

fn instruction_inputs(cycle: &impl CycleRow, iflags: InstructionFlagSet) -> (u64, u128) {
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

pub fn instruction_circuit_flags(instr: &Instruction) -> CircuitFlagSet {
    with_isa_struct!(instr, |i| Flags::circuit_flags(&i), noop => {
        CircuitFlagSet::default().set(CircuitFlags::DoNotUpdateUnexpandedPC)
    })
}

pub fn instruction_instruction_flags(instr: &Instruction) -> InstructionFlagSet {
    with_isa_struct!(instr, |i| Flags::instruction_flags(&i), noop => {
        InstructionFlagSet::default().set(InstructionFlags::IsNoop)
    })
}

fn static_circuit_flags(instr: &Instruction) -> CircuitFlagSet {
    instruction_circuit_flags(instr)
}

fn static_instruction_flags(instr: &Instruction) -> InstructionFlagSet {
    instruction_instruction_flags(instr)
}

#[inline]
fn interleave_bits(x: u64, y: u64) -> u128 {
    let mut x_bits = x as u128;
    x_bits = (x_bits | (x_bits << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    x_bits = (x_bits | (x_bits << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    x_bits = (x_bits | (x_bits << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    x_bits = (x_bits | (x_bits << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    x_bits = (x_bits | (x_bits << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x_bits = (x_bits | (x_bits << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    let mut y_bits = y as u128;
    y_bits = (y_bits | (y_bits << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    y_bits = (y_bits | (y_bits << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    y_bits = (y_bits | (y_bits << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    y_bits = (y_bits | (y_bits << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    y_bits = (y_bits | (y_bits << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    y_bits = (y_bits | (y_bits << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    (x_bits << 1) | y_bits
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_trait_methods() {
        let noop = Cycle::noop();
        assert!(noop.is_noop());
        assert_eq!(noop.unexpanded_pc(), 0);
        assert!(noop.ram_access_address().is_none());
        assert!(noop.rs1_read().is_none());
        assert!(noop.rd_write().is_none());

        let cflags = CycleRow::circuit_flags(&noop);
        assert!(cflags[CircuitFlags::DoNotUpdateUnexpandedPC]);

        let iflags = CycleRow::instruction_flags(&noop);
        assert!(iflags[InstructionFlags::IsNoop]);
    }

    #[test]
    fn noop_lookup_index_is_zero() {
        assert_eq!(Cycle::noop().lookup_index(), 0);
    }
}
