use ark_ff::PrimeField;
use strum_macros::FromRepr;

use crate::jolt::vm::pc::ELFRow;

// Reference: https://www.cs.sfu.ca/~ashriram/Courses/CS295/assets/notebooks/RISCV/RISCV_CARD.pdf
#[derive(Debug, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum RV32IM {
    ADD,
    SUB,
    XOR,
    OR,
    AND,
    SLL,
    SRL,
    SRA,
    SLT,
    SLTU,
    ADDI,
    XORI,
    ORI,
    ANDI,
    SLLI,
    SRLI,
    SRAI,
    SLTI,
    SLTIU,
    LB,
    LH,
    LW,
    LBU,
    LHU,
    SB,
    SH,
    SW,
    BEQ,
    BNE,
    BLT,
    BGE,
    BLTU,
    BGEU,
    JAL,
    JALR,
    LUI,
    AUIPC,
    ECALL,
    EBREAK,
    MUL,
    MULH,
    MULSU,
    MULU,
    DIV,
    DIVU,
    REM,
    REMU
}

trait R1CSFlags {
  fn to_circuit_flags<F: PrimeField>(&self) -> Vec<F>;
}

impl R1CSFlags for RV32IM {
  fn to_circuit_flags<F: PrimeField>(&self) -> Vec<F> {

    // Jolt Appendix A.1
    // 0: first_operand == rs1 (1 if PC)
    // 1: second_operand == rs2 (1 if imm)
    // 2: Load instruction
    // 3: Store instruciton
    // 4: Jump instruction 
    // 5: Branch instruciton
    // 6: Instruction writes lookup output to rd
    // 7: Instruction adds operands (ie, and uses the ADD lookup table)
    // 8: Instruction subtracts operands
    // 9: Instruction multiplies operands
    // 10: Instruction involves non-deterministic advice?
    // 11: Instruction asserts lookup output as false
    // 12: Instruction asserts lookup output as true
    // 13: Sign-bit of imm
    // 14: Instruction is lui 

    let flag_0 = match self {
        RV32IM::JAL | RV32IM::JALR | RV32IM::LUI | RV32IM::AUIPC => true,
        _ => false
    };

    let flag_1 = match self {
        RV32IM::ADDI | RV32IM::XORI | RV32IM::ORI | RV32IM::ANDI | RV32IM::SLLI 
            | RV32IM::SRLI | RV32IM::SRAI | RV32IM::SLTI | RV32IM::SLTIU => true,
        _ => false,
    };

    let flag_2 = match self {
        RV32IM::LB | RV32IM::LH | RV32IM::LW | RV32IM::LBU | RV32IM::LHU => true,
        _ => false
    };  

    let flag_3 = match self {
        RV32IM::SB | RV32IM::SH | RV32IM::SW => true,
        _ => false
    };

    let flag_4 = match self {
        RV32IM::JAL | RV32IM::JALR => true,
        _ => false
    };

    let flag_5 = match self {
        RV32IM::BEQ | RV32IM::BNE | RV32IM::BLT | RV32IM::BGE | RV32IM::BLTU | RV32IM::BGEU => true,
        _ => false
    };

    // loads, stores, branches, jumps do not store the lookup output to rd (they may update rd in other ways)
    let flag_6 = match self {
        RV32IM::LB | RV32IM::LH | RV32IM::LW | RV32IM::LBU | RV32IM::LHU | RV32IM::SB | RV32IM::SH | RV32IM::SW |
        RV32IM::BEQ | RV32IM::BNE | RV32IM::BLT | RV32IM::BGE | RV32IM::BLTU | RV32IM::BGEU |
        RV32IM::JAL | RV32IM::JALR | RV32IM::LUI  => false,
        _ => true,
    };

    let flag_7 = match self {
        RV32IM::ADD | RV32IM::ADDI | RV32IM::JAL | RV32IM::JALR | RV32IM::AUIPC => true,
        _ => false
    };

    let flag_8 = match self {
        RV32IM::SUB => true,
        _ => false
    };

    let flag_9 = match self {
        RV32IM::MUL | RV32IM::MULU | RV32IM::MULH | RV32IM::MULSU => true,
        _ => false
    };

    // not incorporating advice instructions yet
    let flag_10 = match self {
        _ => false
    };

    // not incorporating assert true instructions yet
    let flag_11 = match self {
        _ => false
    };

    // not incorporating assert false instructions yet
    let flag_12 = match self {
        _ => false
    };

    // not incorporating advice instructions yet
    let flag_13 = match self {
        _ => false
    };

    let flag_14 = match self {
        RV32IM::LUI => true,
        _ => false
    };

    vec![
        F::from(flag_0), 
        F::from(flag_1), 
        F::from(flag_2), 
        F::from(flag_3), 
        F::from(flag_4), 
        F::from(flag_5), 
        F::from(flag_6), 
        F::from(flag_7), 
        F::from(flag_8), 
        F::from(flag_9), 
        F::from(flag_10), 
        F::from(flag_11), 
        F::from(flag_12), 
        F::from(flag_13), 
        F::from(flag_13),
        F::from(flag_14)
    ]
  }
}