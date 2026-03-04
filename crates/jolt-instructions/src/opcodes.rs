//! Sequential opcode assignments for all Jolt instructions.
//!
//! Opcodes are assigned contiguously starting from 0 to enable efficient
//! array-based dispatch in the instruction set registry.

// RV64I arithmetic
pub const ADD: u32 = 0;
pub const SUB: u32 = 1;
pub const LUI: u32 = 2;
pub const AUIPC: u32 = 3;

// RV64M multiply/divide
pub const MUL: u32 = 4;
pub const MULH: u32 = 5;
pub const MULHSU: u32 = 6;
pub const MULHU: u32 = 7;
pub const DIV: u32 = 8;
pub const DIVU: u32 = 9;
pub const REM: u32 = 10;
pub const REMU: u32 = 11;

// RV64I arithmetic W-suffix (32-bit on RV64)
pub const ADDW: u32 = 12;
pub const SUBW: u32 = 13;

// RV64M W-suffix
pub const MULW: u32 = 14;
pub const DIVW: u32 = 15;
pub const DIVUW: u32 = 16;
pub const REMW: u32 = 17;
pub const REMUW: u32 = 18;

// RV64I logic
pub const AND: u32 = 19;
pub const OR: u32 = 20;
pub const XOR: u32 = 21;
pub const ANDI: u32 = 22;
pub const ORI: u32 = 23;
pub const XORI: u32 = 24;

// RV64I shifts
pub const SLL: u32 = 25;
pub const SRL: u32 = 26;
pub const SRA: u32 = 27;
pub const SLLI: u32 = 28;
pub const SRLI: u32 = 29;
pub const SRAI: u32 = 30;

// RV64I shifts W-suffix
pub const SLLW: u32 = 31;
pub const SRLW: u32 = 32;
pub const SRAW: u32 = 33;
pub const SLLIW: u32 = 34;
pub const SRLIW: u32 = 35;
pub const SRAIW: u32 = 36;

// RV64I compare
pub const SLT: u32 = 37;
pub const SLTU: u32 = 38;
pub const SLTI: u32 = 39;
pub const SLTIU: u32 = 40;

// RV64I branch
pub const BEQ: u32 = 41;
pub const BNE: u32 = 42;
pub const BLT: u32 = 43;
pub const BGE: u32 = 44;
pub const BLTU: u32 = 45;
pub const BGEU: u32 = 46;

// RV64I load
pub const LB: u32 = 47;
pub const LBU: u32 = 48;
pub const LH: u32 = 49;
pub const LHU: u32 = 50;
pub const LW: u32 = 51;
pub const LWU: u32 = 52;
pub const LD: u32 = 53;

// RV64I store
pub const SB: u32 = 54;
pub const SH: u32 = 55;
pub const SW: u32 = 56;
pub const SD: u32 = 57;

// RV64I system
pub const ECALL: u32 = 58;
pub const EBREAK: u32 = 59;
pub const FENCE: u32 = 60;
pub const NOOP: u32 = 61;

// RV64I immediate aliases
pub const ADDI: u32 = 62;
pub const ADDIW: u32 = 63;

// Virtual instructions
pub const ASSERT_EQ: u32 = 64;
pub const ASSERT_LTE: u32 = 65;
pub const POW2: u32 = 66;
pub const MOVSIGN: u32 = 67;

/// Total number of opcodes in the instruction set.
pub const COUNT: u32 = 68;
