use ark_ff::PrimeField;
use strum_macros::FromRepr;

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

    // TODO(sragss):
    // - What are we doing with imm if we have a flag for the sign bit?
    // - What is 10 below?

    // Jolt Appendix A.1
    // 0: first_operand == rs2
    // 1: second_operand == imm
    // 2: Load instruction
    // 3: Store instruciton
    // 4: Jump instruction 
    // 5: Branch instruciton
    // 6: Instruction writes to rd
    // 7: Instruction adds operands
    // 8: Instruction subtracts operands
    // 9: Instruction multiplies operands
    // 10: Instruction involves non-deterministic advice?
    // 11: Instruction asserts lookup output as false
    // 12: Instruction asserts lookup output as true
    // 13: Sign-bit of imm

    let flag_0 = match self {
        _ => true
    };

    let flag_1 = match self {
        _ => true
    }

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

    let flag_6 = match self {
        RV32IM::ADD | RV32IM::SUB | RV32IM::SLL | RV32IM::SLT | RV32IM::SLTU | RV32IM::XOR | RV32IM::SRL | RV32IM::SRA | RV32IM::OR | RV32IM::AND => true,
        _ => false
    };

    let flag_7 = match self {
        // RV32IM::ADD | RV32IM::SUB | RV32IM::SLL | RV32IM::SLT | RV32IM::SLTU | RV32IM::XOR | RV32IM::SRL | RV32IM::SRA | RV32IM::OR | RV32IM::AND => true,
        // _ => false
    };
    todo!("fill out  all flags")
  }
}