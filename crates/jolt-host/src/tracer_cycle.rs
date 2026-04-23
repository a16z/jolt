//! `CycleRow` implementation for `tracer::Cycle`.
//!
//! Maps `Instruction` variants to ISA structs via [`with_isa_struct!`], then
//! derives circuit flags, instruction flags, and lookup table kinds.

use jolt_instructions::flags::{
    CircuitFlags, Flags, InstructionFlags, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
};
use jolt_instructions::traits::Instruction as JoltInstructionTrait;
use jolt_instructions::LookupTableKind;
use tracer::instruction::field_op::{FUNCT3_FADD, FUNCT3_FINV, FUNCT3_FMUL, FUNCT3_FSUB};
use tracer::instruction::{Cycle, Instruction, NormalizedInstruction, RAMAccess};

use crate::CycleRow;

/// Map an `Instruction` variant to its ISA struct, bind it to `$i`, evaluate `$body`.
/// The `noop =>` arm handles `Instruction::NoOp` separately.
macro_rules! with_isa_struct {
    ($instr:expr, |$i:ident| $body:expr, noop => $noop:expr) => {{
        use jolt_instructions::rv::{
            arithmetic, branch, compare, jump, load, logic, store, system,
        };
        use jolt_instructions::virtual_::{
            advice, arithmetic as varith, assert as vassert, bitwise, byte, division, extension,
            shift, xor_rotate,
        };
        match $instr {
            Instruction::ADD(_) => {
                let $i = arithmetic::Add;
                $body
            }
            Instruction::ADDI(_) => {
                let $i = arithmetic::Addi;
                $body
            }
            Instruction::SUB(_) => {
                let $i = arithmetic::Sub;
                $body
            }
            Instruction::LUI(_) => {
                let $i = arithmetic::Lui;
                $body
            }
            Instruction::AUIPC(_) => {
                let $i = arithmetic::Auipc;
                $body
            }
            Instruction::MUL(_) => {
                let $i = arithmetic::Mul;
                $body
            }
            Instruction::MULHU(_) => {
                let $i = arithmetic::MulHU;
                $body
            }
            Instruction::AND(_) => {
                let $i = logic::And;
                $body
            }
            Instruction::ANDI(_) => {
                let $i = logic::AndI;
                $body
            }
            Instruction::ANDN(_) => {
                let $i = logic::Andn;
                $body
            }
            Instruction::OR(_) => {
                let $i = logic::Or;
                $body
            }
            Instruction::ORI(_) => {
                let $i = logic::OrI;
                $body
            }
            Instruction::XOR(_) => {
                let $i = logic::Xor;
                $body
            }
            Instruction::XORI(_) => {
                let $i = logic::XorI;
                $body
            }
            Instruction::SLT(_) => {
                let $i = compare::Slt;
                $body
            }
            Instruction::SLTI(_) => {
                let $i = compare::SltI;
                $body
            }
            Instruction::SLTIU(_) => {
                let $i = compare::SltIU;
                $body
            }
            Instruction::SLTU(_) => {
                let $i = compare::SltU;
                $body
            }
            Instruction::BEQ(_) => {
                let $i = branch::Beq;
                $body
            }
            Instruction::BGE(_) => {
                let $i = branch::Bge;
                $body
            }
            Instruction::BGEU(_) => {
                let $i = branch::BgeU;
                $body
            }
            Instruction::BLT(_) => {
                let $i = branch::Blt;
                $body
            }
            Instruction::BLTU(_) => {
                let $i = branch::BltU;
                $body
            }
            Instruction::BNE(_) => {
                let $i = branch::Bne;
                $body
            }
            Instruction::JAL(_) => {
                let $i = jump::Jal;
                $body
            }
            Instruction::JALR(_) => {
                let $i = jump::Jalr;
                $body
            }
            Instruction::LD(_) => {
                let $i = load::Ld;
                $body
            }
            Instruction::SD(_) => {
                let $i = store::Sd;
                $body
            }
            Instruction::EBREAK(_) => {
                let $i = system::Ebreak;
                $body
            }
            Instruction::ECALL(_) => {
                let $i = system::Ecall;
                $body
            }
            Instruction::FENCE(_) => {
                let $i = system::Fence;
                $body
            }
            Instruction::VirtualAdvice(_) => {
                let $i = advice::VirtualAdvice;
                $body
            }
            Instruction::VirtualAdviceLen(_) => {
                let $i = advice::VirtualAdviceLen;
                $body
            }
            Instruction::VirtualAdviceLoad(_) => {
                let $i = advice::VirtualAdviceLoad;
                $body
            }
            Instruction::VirtualHostIO(_) => {
                let $i = advice::VirtualHostIO;
                $body
            }
            Instruction::VirtualMULI(_) => {
                let $i = varith::MulI;
                $body
            }
            Instruction::VirtualPow2(_) => {
                let $i = varith::Pow2;
                $body
            }
            Instruction::VirtualPow2I(_) => {
                let $i = varith::Pow2I;
                $body
            }
            Instruction::VirtualPow2W(_) => {
                let $i = varith::Pow2W;
                $body
            }
            Instruction::VirtualPow2IW(_) => {
                let $i = varith::Pow2IW;
                $body
            }
            Instruction::VirtualAssertEQ(_) => {
                let $i = vassert::AssertEq;
                $body
            }
            Instruction::VirtualAssertLTE(_) => {
                let $i = vassert::AssertLte;
                $body
            }
            Instruction::VirtualAssertValidDiv0(_) => {
                let $i = vassert::AssertValidDiv0;
                $body
            }
            Instruction::VirtualAssertValidUnsignedRemainder(_) => {
                let $i = vassert::AssertValidUnsignedRemainder;
                $body
            }
            Instruction::VirtualAssertMulUNoOverflow(_) => {
                let $i = vassert::AssertMulUNoOverflow;
                $body
            }
            Instruction::VirtualAssertWordAlignment(_) => {
                let $i = vassert::AssertWordAlignment;
                $body
            }
            Instruction::VirtualAssertHalfwordAlignment(_) => {
                let $i = vassert::AssertHalfwordAlignment;
                $body
            }
            Instruction::VirtualMovsign(_) => {
                let $i = bitwise::MovSign;
                $body
            }
            Instruction::VirtualRev8W(_) => {
                let $i = byte::VirtualRev8W;
                $body
            }
            Instruction::VirtualChangeDivisor(_) => {
                let $i = division::VirtualChangeDivisor;
                $body
            }
            Instruction::VirtualChangeDivisorW(_) => {
                let $i = division::VirtualChangeDivisorW;
                $body
            }
            Instruction::VirtualZeroExtendWord(_) => {
                let $i = extension::VirtualZeroExtendWord;
                $body
            }
            Instruction::VirtualSignExtendWord(_) => {
                let $i = extension::VirtualSignExtendWord;
                $body
            }
            Instruction::VirtualSRL(_) => {
                let $i = shift::VirtualSrl;
                $body
            }
            Instruction::VirtualSRLI(_) => {
                let $i = shift::VirtualSrli;
                $body
            }
            Instruction::VirtualSRA(_) => {
                let $i = shift::VirtualSra;
                $body
            }
            Instruction::VirtualSRAI(_) => {
                let $i = shift::VirtualSrai;
                $body
            }
            Instruction::VirtualShiftRightBitmask(_) => {
                let $i = shift::VirtualShiftRightBitmask;
                $body
            }
            Instruction::VirtualShiftRightBitmaskI(_) => {
                let $i = shift::VirtualShiftRightBitmaski;
                $body
            }
            Instruction::VirtualROTRI(_) => {
                let $i = shift::VirtualRotri;
                $body
            }
            Instruction::VirtualROTRIW(_) => {
                let $i = shift::VirtualRotriw;
                $body
            }
            Instruction::VirtualXORROT32(_) => {
                let $i = xor_rotate::VirtualXorRot32;
                $body
            }
            Instruction::VirtualXORROT24(_) => {
                let $i = xor_rotate::VirtualXorRot24;
                $body
            }
            Instruction::VirtualXORROT16(_) => {
                let $i = xor_rotate::VirtualXorRot16;
                $body
            }
            Instruction::VirtualXORROT63(_) => {
                let $i = xor_rotate::VirtualXorRot63;
                $body
            }
            Instruction::VirtualXORROTW16(_) => {
                let $i = xor_rotate::VirtualXorRotW16;
                $body
            }
            Instruction::VirtualXORROTW12(_) => {
                let $i = xor_rotate::VirtualXorRotW12;
                $body
            }
            Instruction::VirtualXORROTW8(_) => {
                let $i = xor_rotate::VirtualXorRotW8;
                $body
            }
            Instruction::VirtualXORROTW7(_) => {
                let $i = xor_rotate::VirtualXorRotW7;
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

    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let instr = self.instruction();
        let mut flags = static_circuit_flags(&instr);
        let norm = instr.normalize();
        apply_dynamic_flags(&mut flags, &norm);
        if matches!(instr, Instruction::JALR(_)) && norm.virtual_sequence_remaining == Some(0) {
            flags[CircuitFlags::IsLastInSequence] = true;
        }
        flags
    }

    fn instruction_flags(&self) -> [bool; NUM_INSTRUCTION_FLAGS] {
        let instr = self.instruction();
        let mut flags = static_instruction_flags(&instr);
        let norm = instr.normalize();
        flags[InstructionFlags::IsRdNotZero] = matches!(norm.operands.rd, Some(rd) if rd != 0);
        flags
    }

    fn lookup_index(&self) -> u128 {
        let cflags = self.circuit_flags();
        let iflags = self.instruction_flags();
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
            jolt_instructions::interleave_bits(left, right as u64)
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
}

fn instruction_inputs(
    cycle: &impl CycleRow,
    iflags: &[bool; NUM_INSTRUCTION_FLAGS],
) -> (u64, u128) {
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

fn static_circuit_flags(instr: &Instruction) -> [bool; NUM_CIRCUIT_FLAGS] {
    // BN254 Fr coprocessor instructions don't live in the generic ISA-struct
    // dispatch (their static Flags are dynamic in funct3 for FieldOp, and
    // FMov{I2F,F2I} are identity with respect to the R1CS flag set). Handle
    // them up-front before hitting the macro's `_ => panic!` catch-all.
    if let Some(flags) = field_op_circuit_flags(instr) {
        return flags;
    }
    with_isa_struct!(instr, |i| Flags::circuit_flags(&i), noop => {
        let mut f = [false; NUM_CIRCUIT_FLAGS];
        f[CircuitFlags::DoNotUpdateUnexpandedPC] = true;
        f
    })
}

fn static_instruction_flags(instr: &Instruction) -> [bool; NUM_INSTRUCTION_FLAGS] {
    if field_op_is_fr(instr) {
        // FR coprocessor cycles: not a noop, no branch, no lookup — all flags
        // default to false. (IsRdNotZero is set downstream by `circuit_flags`
        // via the normalized-operand `rd`.)
        return [false; NUM_INSTRUCTION_FLAGS];
    }
    with_isa_struct!(instr, |i| Flags::instruction_flags(&i), noop => {
        let mut f = [false; NUM_INSTRUCTION_FLAGS];
        f[InstructionFlags::IsNoop] = true;
        f
    })
}

fn lookup_table_kind(instr: &Instruction) -> Option<LookupTableKind> {
    if field_op_is_fr(instr) {
        return None;
    }
    with_isa_struct!(instr, |i| i.lookup_table(), noop => None)
}

/// True if `instr` is a BN254 Fr coprocessor instruction (`FieldOp` or the
/// two `FMov{I2F,F2I}` limb bridges). Used to short-circuit flag / lookup
/// dispatch before the generic ISA-struct macro.
fn field_op_is_fr(instr: &Instruction) -> bool {
    matches!(
        instr,
        Instruction::FieldOp(_)
            | Instruction::FMovIntToFieldLimb(_)
            | Instruction::FMovFieldToIntLimb(_)
    )
}

/// Circuit flags for BN254 Fr coprocessor cycles. Returns `Some(flags)` for
/// FieldOp / FMov* cycles, `None` otherwise (let the caller fall through).
///
/// For `FieldOp`, funct3 selects one of `IsFieldMul/Add/Sub/Inv`. FMov cycles
/// set no circuit flags — they're plain R-type instructions that advance PC
/// but don't participate in any R1CS gate.
fn field_op_circuit_flags(instr: &Instruction) -> Option<[bool; NUM_CIRCUIT_FLAGS]> {
    match instr {
        Instruction::FieldOp(op) => {
            let mut f = [false; NUM_CIRCUIT_FLAGS];
            match op.funct3 {
                FUNCT3_FMUL => f[CircuitFlags::IsFieldMul] = true,
                FUNCT3_FADD => f[CircuitFlags::IsFieldAdd] = true,
                FUNCT3_FSUB => f[CircuitFlags::IsFieldSub] = true,
                FUNCT3_FINV => f[CircuitFlags::IsFieldInv] = true,
                other => panic!("invalid FieldOp funct3 in CycleRow: {other:#x}"),
            }
            Some(f)
        }
        Instruction::FMovIntToFieldLimb(_) => {
            let mut f = [false; NUM_CIRCUIT_FLAGS];
            f[CircuitFlags::IsFMovI2F] = true;
            Some(f)
        }
        Instruction::FMovFieldToIntLimb(_) => {
            let mut f = [false; NUM_CIRCUIT_FLAGS];
            f[CircuitFlags::IsFMovF2I] = true;
            Some(f)
        }
        _ => None,
    }
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

fn apply_dynamic_flags(flags: &mut [bool; NUM_CIRCUIT_FLAGS], norm: &NormalizedInstruction) {
    if norm.virtual_sequence_remaining.is_some() {
        flags[CircuitFlags::VirtualInstruction] = true;
    }
    if norm.virtual_sequence_remaining.unwrap_or(0) != 0 {
        flags[CircuitFlags::DoNotUpdateUnexpandedPC] = true;
    }
    if norm.is_first_in_sequence {
        flags[CircuitFlags::IsFirstInSequence] = true;
    }
    if norm.is_compressed {
        flags[CircuitFlags::IsCompressed] = true;
    }
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

    /// Constructing a `Cycle::FieldOp` by hand and routing it through the
    /// CycleRow methods (pre-fix this panicked at `with_isa_struct!`'s
    /// catch-all arm).
    fn make_field_op_cycle(funct3: u8) -> Cycle {
        use tracer::instruction::field_op::FieldOp;
        use tracer::instruction::format::format_r::FormatR;
        use tracer::instruction::RISCVCycle;

        let op = FieldOp {
            address: 0x1000,
            operands: FormatR { rd: 3, rs1: 1, rs2: 2 },
            funct3,
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        Cycle::FieldOp(RISCVCycle {
            instruction: op,
            register_state: Default::default(),
            ram_access: (),
        })
    }

    #[test]
    fn field_op_cycle_row_sets_funct3_flag() {
        for (funct3, want) in [
            (FUNCT3_FMUL, CircuitFlags::IsFieldMul),
            (FUNCT3_FADD, CircuitFlags::IsFieldAdd),
            (FUNCT3_FSUB, CircuitFlags::IsFieldSub),
            (FUNCT3_FINV, CircuitFlags::IsFieldInv),
        ] {
            let cycle = make_field_op_cycle(funct3);
            let cflags = CycleRow::circuit_flags(&cycle);
            assert!(cflags[want], "funct3 {funct3:#x}: expected {want:?} set");
            // Exactly one FieldOp flag should be set on a FieldOp cycle.
            let field_flag_count = [
                cflags[CircuitFlags::IsFieldMul],
                cflags[CircuitFlags::IsFieldAdd],
                cflags[CircuitFlags::IsFieldSub],
                cflags[CircuitFlags::IsFieldInv],
            ]
            .iter()
            .filter(|&&b| b)
            .count();
            assert_eq!(field_flag_count, 1, "exactly one FieldOp flag per cycle");
            // Not a noop, not virtual, and no lookup.
            assert!(!cycle.is_noop());
            let iflags = CycleRow::instruction_flags(&cycle);
            assert!(!iflags[InstructionFlags::IsNoop]);
            assert!(cycle.lookup_table_index().is_none());
        }
    }
}
