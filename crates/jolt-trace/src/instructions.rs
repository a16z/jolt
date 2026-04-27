//! Jolt RISC-V instruction types and their static metadata.
//!
//! Each unit struct represents an instruction kind. `#[derive(Flags)]` declares
//! the R1CS circuit and witness-generation flags. The `InstructionLookupTable`
//! impls (in `jolt-lookup-tables`) map instructions to lookup tables.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I ADD: `rd = rs1 + rs2` (wrapping).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Add;

/// RV64I ADDI: `rd = rs1 + imm` (wrapping). Immediate already decoded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct Addi;

/// RV64I SUB: `rd = rs1 - rs2` (wrapping).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(SubtractOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Sub;

/// RV64I LUI: load upper immediate. Result is the immediate value itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(RightOperandIsImm)]
pub struct Lui;

/// RV64I AUIPC: add upper immediate to PC. `rd = PC + imm`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsPC, RightOperandIsImm)]
pub struct Auipc;

/// RV64M MUL: signed multiply, lower 64 bits of the 128-bit product.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(MultiplyOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Mul;

/// RV64M MULH: signed×signed multiply, upper 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct MulH;

/// RV64M MULHSU: signed×unsigned multiply, upper 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct MulHSU;

/// RV64M MULHU: unsigned×unsigned multiply, upper 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(MultiplyOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct MulHU;

/// RV64M DIV: signed division with RISC-V overflow handling.
///
/// Special cases per the RISC-V spec:
/// - Division by zero returns `u64::MAX` (all bits set, i.e. -1 unsigned).
/// - `i64::MIN / -1` returns `i64::MIN` (overflow wraps).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Div;

/// RV64M DIVU: unsigned division. Returns `u64::MAX` on division by zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct DivU;

/// RV64M REM: signed remainder. Returns `x` on division by zero,
/// returns 0 when `x == i64::MIN && y == -1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Rem;

/// RV64M REMU: unsigned remainder. Returns `x` on division by zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct RemU;

/// RV64I ADDW: 32-bit add, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct AddW;

/// RV64I ADDIW: 32-bit add immediate, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct AddiW;

/// RV64I SUBW: 32-bit subtract, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(SubtractOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct SubW;

/// RV64M MULW: 32-bit multiply, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(MultiplyOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct MulW;

/// RV64M DIVW: 32-bit signed division, sign-extended to 64 bits.
///
/// Division by zero returns `u64::MAX`. Overflow (`i32::MIN / -1`) returns `i32::MIN` sign-extended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct DivW;

/// RV64M DIVUW: 32-bit unsigned division, sign-extended to 64 bits.
/// Returns `u64::MAX` on division by zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct DivUW;

/// RV64M REMW: 32-bit signed remainder, sign-extended to 64 bits.
/// Returns `x` (truncated to 32 bits, sign-extended) on division by zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct RemW;

/// RV64M REMUW: 32-bit unsigned remainder, sign-extended to 64 bits.
/// Returns `x` (truncated to 32 bits, sign-extended) on division by zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct RemUW;

/// RV64I AND: bitwise AND of two registers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct And;

/// RV64I ANDI: bitwise AND with sign-extended immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct AndI;

/// RV64I OR: bitwise OR of two registers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Or;

/// RV64I ORI: bitwise OR with sign-extended immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct OrI;

/// RV64I XOR: bitwise exclusive OR of two registers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Xor;

/// RV64I XORI: bitwise exclusive OR with sign-extended immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct XorI;

/// Zbb ANDN: bitwise AND-NOT. `rd = rs1 & ~rs2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Andn;

/// RV64I SLL: shift left logical. Shift amount from lower 6 bits of `y`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Sll;

/// RV64I SLLI: shift left logical by immediate. Immediate already masked.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SllI;

/// RV64I SRL: shift right logical. Shift amount from lower 6 bits of `y`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Srl;

/// RV64I SRLI: shift right logical by immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SrlI;

/// RV64I SRA: shift right arithmetic. Preserves sign bit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Sra;

/// RV64I SRAI: shift right arithmetic by immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SraI;

/// RV64I SLLW: 32-bit shift left logical, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct SllW;

/// RV64I SLLIW: 32-bit shift left logical by immediate, sign-extended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SllIW;

/// RV64I SRLW: 32-bit shift right logical, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct SrlW;

/// RV64I SRLIW: 32-bit shift right logical by immediate, sign-extended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SrlIW;

/// RV64I SRAW: 32-bit shift right arithmetic, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct SraW;

/// RV64I SRAIW: 32-bit shift right arithmetic by immediate, sign-extended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SraIW;

/// RV64I SLT: set if less than (signed). `rd = (rs1 < rs2) ? 1 : 0`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Slt;

/// RV64I SLTI: set if less than immediate (signed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SltI;

/// RV64I SLTU: set if less than (unsigned). `rd = (rs1 < rs2) ? 1 : 0`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct SltU;

/// RV64I SLTIU: set if less than immediate (unsigned).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SltIU;

/// RV64I BEQ: branch if equal. Returns 1 when `rs1 == rs2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch)]
pub struct Beq;

/// RV64I BNE: branch if not equal. Returns 1 when `rs1 != rs2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch)]
pub struct Bne;

/// RV64I BLT: branch if less than (signed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch)]
pub struct Blt;

/// RV64I BGE: branch if greater than or equal (signed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch)]
pub struct Bge;

/// RV64I BLTU: branch if less than (unsigned).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch)]
pub struct BltU;

/// RV64I BGEU: branch if greater than or equal (unsigned).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch)]
pub struct BgeU;

/// RV64I LB: load byte, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Load)]
pub struct Lb;

/// RV64I LBU: load byte, zero-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Load)]
pub struct Lbu;

/// RV64I LH: load halfword (16 bits), sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Load)]
pub struct Lh;

/// RV64I LHU: load halfword, zero-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Load)]
pub struct Lhu;

/// RV64I LW: load word (32 bits), sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Load)]
pub struct Lw;

/// RV64I LWU: load word, zero-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Load)]
pub struct Lwu;

/// RV64I LD: load doubleword (64 bits). Identity operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Load)]
pub struct Ld;

/// RV64I SB: store byte (lowest 8 bits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Store)]
pub struct Sb;

/// RV64I SH: store halfword (lowest 16 bits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Store)]
pub struct Sh;

/// RV64I SW: store word (lowest 32 bits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Store)]
pub struct Sw;

/// RV64I SD: store doubleword (full 64 bits). Identity operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Store)]
pub struct Sd;

/// RV64I ECALL: environment call (syscall).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
pub struct Ecall;

/// RV64I EBREAK: breakpoint trap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
pub struct Ebreak;

/// RV64I FENCE: memory ordering fence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
pub struct Fence;

/// No-operation pseudo-instruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(IsNoop)]
pub struct Noop;

/// RV64I JAL: jump and link. `rd = PC + 4; PC = PC + imm`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, Jump)]
#[instruction(LeftOperandIsPC, RightOperandIsImm)]
pub struct Jal;

/// RV64I JALR: jump and link register. `rd = PC + 4; PC = (rs1 + imm) & !1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, Jump)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct Jalr;

/// Virtual ASSERT_EQ: returns 1 if operands are equal, 0 otherwise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct AssertEq;

/// Virtual ASSERT_LTE: returns 1 if `x <= y` (unsigned), 0 otherwise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct AssertLte;

/// Virtual ASSERT_VALID_DIV0: validates `(divisor, quotient)` for division-by-zero handling.
/// Returns 1 if the divisor is nonzero, or if the divisor is 0 and the quotient is MAX.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct AssertValidDiv0;

/// Virtual ASSERT_VALID_UNSIGNED_REMAINDER: validates unsigned remainder.
/// Returns 1 if divisor is 0 or remainder < divisor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct AssertValidUnsignedRemainder;

/// Virtual ASSERT_MULU_NO_OVERFLOW: checks unsigned multiply doesn't overflow.
/// Returns 1 if the upper XLEN bits of `x * y` are all zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(MultiplyOperands, Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct AssertMulUNoOverflow;

/// Virtual ASSERT_WORD_ALIGNMENT: checks whether `rs1 + imm` is 4-byte aligned.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct AssertWordAlignment;

/// Virtual ASSERT_HALFWORD_ALIGNMENT: checks whether `rs1 + imm` is 2-byte aligned.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct AssertHalfwordAlignment;

/// Virtual POW2: computes `2^rs1` using the low 6 bits of `rs1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct Pow2;

/// Virtual POW2I: computes `2^imm` with immediate exponent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(RightOperandIsImm)]
pub struct Pow2I;

/// Virtual POW2W: computes `2^(rs1 mod 32)` for 32-bit mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value)]
pub struct Pow2W;

/// Virtual POW2IW: computes `2^(imm mod 32)` for 32-bit immediate mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(RightOperandIsImm)]
pub struct Pow2IW;

/// Virtual MULI: multiply by immediate. `rd = rs1 * imm`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(MultiplyOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct MulI;

/// Virtual MOVSIGN: returns all-ones if `x` is negative (signed), otherwise zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct MovSign;

/// Virtual REV8W: byte-reverse within the lower 32 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value)]
pub struct VirtualRev8W;

/// Virtual CHANGE_DIVISOR: transforms divisor for signed division overflow.
/// Returns the divisor unchanged, unless dividend == MIN && divisor == -1,
/// in which case returns 1 to avoid overflow.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualChangeDivisor;

/// Virtual CHANGE_DIVISOR_W: 32-bit version of change divisor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualChangeDivisorW;

/// Virtual SIGN_EXTEND_WORD: sign-extends a 32-bit value to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value)]
pub struct VirtualSignExtendWord;

/// Virtual ZERO_EXTEND_WORD: zero-extends a 32-bit value to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value)]
pub struct VirtualZeroExtendWord;

/// Virtual SRL: logical right shift using a bitmask-encoded shift amount.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualSrl;

/// Virtual SRLI: logical right shift using a bitmask immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct VirtualSrli;

/// Virtual SRA: arithmetic right shift using a bitmask-encoded shift amount.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualSra;

/// Virtual SRAI: arithmetic right shift using a bitmask immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct VirtualSrai;

/// Virtual SHIFT_RIGHT_BITMASK: bitmask for the shift amount stored in `rs1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct VirtualShiftRightBitmask;

/// Virtual SHIFT_RIGHT_BITMASKI: bitmask for the shift amount stored in the immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(RightOperandIsImm)]
pub struct VirtualShiftRightBitmaski;

/// Virtual ROTRI: rotate right using a bitmask immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct VirtualRotri;

/// Virtual ROTRIW: 32-bit rotate right using a bitmask immediate, zero-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct VirtualRotriw;

/// Virtual XOR then rotate right by 32 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualXorRot32;

/// Virtual XOR then rotate right by 24 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualXorRot24;

/// Virtual XOR then rotate right by 16 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualXorRot16;

/// Virtual XOR then rotate right by 63 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualXorRot63;

/// Virtual XOR then rotate right word (32-bit) by 16 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualXorRotW16;

/// Virtual XOR then rotate right word by 12 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualXorRotW12;

/// Virtual XOR then rotate right word by 8 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualXorRotW8;

/// Virtual XOR then rotate right word by 7 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualXorRotW7;

/// Virtual ADVICE: runtime-provided advice value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Advice, WriteLookupOutputToRD)]
pub struct VirtualAdvice;

/// Virtual ADVICE_LEN: advice-tape length query.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Advice, WriteLookupOutputToRD)]
pub struct VirtualAdviceLen;

/// Virtual ADVICE_LOAD: advice-tape read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Advice, WriteLookupOutputToRD)]
pub struct VirtualAdviceLoad;

/// Virtual HOST_IO: host I/O side-effect instruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
pub struct VirtualHostIO;

/// Enum with one variant per Jolt instruction.
///
/// Each variant carries the corresponding unit struct, enabling trait-based
/// dispatch (e.g. via `Flags` or `InstructionLookupTable`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JoltInstructions {
    Add(Add),
    Addi(Addi),
    Sub(Sub),
    Lui(Lui),
    Auipc(Auipc),
    Mul(Mul),
    MulH(MulH),
    MulHSU(MulHSU),
    MulHU(MulHU),
    Div(Div),
    DivU(DivU),
    Rem(Rem),
    RemU(RemU),
    AddW(AddW),
    AddiW(AddiW),
    SubW(SubW),
    MulW(MulW),
    DivW(DivW),
    DivUW(DivUW),
    RemW(RemW),
    RemUW(RemUW),
    And(And),
    AndI(AndI),
    Or(Or),
    OrI(OrI),
    Xor(Xor),
    XorI(XorI),
    Andn(Andn),
    Sll(Sll),
    SllI(SllI),
    Srl(Srl),
    SrlI(SrlI),
    Sra(Sra),
    SraI(SraI),
    SllW(SllW),
    SllIW(SllIW),
    SrlW(SrlW),
    SrlIW(SrlIW),
    SraW(SraW),
    SraIW(SraIW),
    Slt(Slt),
    SltI(SltI),
    SltU(SltU),
    SltIU(SltIU),
    Beq(Beq),
    Bne(Bne),
    Blt(Blt),
    Bge(Bge),
    BltU(BltU),
    BgeU(BgeU),
    Lb(Lb),
    Lbu(Lbu),
    Lh(Lh),
    Lhu(Lhu),
    Lw(Lw),
    Lwu(Lwu),
    Ld(Ld),
    Sb(Sb),
    Sh(Sh),
    Sw(Sw),
    Sd(Sd),
    Ecall(Ecall),
    Ebreak(Ebreak),
    Fence(Fence),
    Noop(Noop),
    Jal(Jal),
    Jalr(Jalr),
    AssertEq(AssertEq),
    AssertLte(AssertLte),
    AssertValidDiv0(AssertValidDiv0),
    AssertValidUnsignedRemainder(AssertValidUnsignedRemainder),
    AssertMulUNoOverflow(AssertMulUNoOverflow),
    AssertWordAlignment(AssertWordAlignment),
    AssertHalfwordAlignment(AssertHalfwordAlignment),
    Pow2(Pow2),
    Pow2I(Pow2I),
    Pow2W(Pow2W),
    Pow2IW(Pow2IW),
    MulI(MulI),
    MovSign(MovSign),
    VirtualRev8W(VirtualRev8W),
    VirtualChangeDivisor(VirtualChangeDivisor),
    VirtualChangeDivisorW(VirtualChangeDivisorW),
    VirtualSignExtendWord(VirtualSignExtendWord),
    VirtualZeroExtendWord(VirtualZeroExtendWord),
    VirtualSrl(VirtualSrl),
    VirtualSrli(VirtualSrli),
    VirtualSra(VirtualSra),
    VirtualSrai(VirtualSrai),
    VirtualShiftRightBitmask(VirtualShiftRightBitmask),
    VirtualShiftRightBitmaski(VirtualShiftRightBitmaski),
    VirtualRotri(VirtualRotri),
    VirtualRotriw(VirtualRotriw),
    VirtualXorRot32(VirtualXorRot32),
    VirtualXorRot24(VirtualXorRot24),
    VirtualXorRot16(VirtualXorRot16),
    VirtualXorRot63(VirtualXorRot63),
    VirtualXorRotW16(VirtualXorRotW16),
    VirtualXorRotW12(VirtualXorRotW12),
    VirtualXorRotW8(VirtualXorRotW8),
    VirtualXorRotW7(VirtualXorRotW7),
    VirtualAdvice(VirtualAdvice),
    VirtualAdviceLen(VirtualAdviceLen),
    VirtualAdviceLoad(VirtualAdviceLoad),
    VirtualHostIO(VirtualHostIO),
}
