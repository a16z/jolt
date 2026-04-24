//! BN254 Fr native-field coprocessor instructions.
//!
//! Nine ops under custom-0 opcode `0x0B`:
//! - Arithmetic (FR-FR): `FieldMul`, `FieldAdd`, `FieldSub`, `FieldInv`, `FieldAssertEq`
//! - Integer→field bridge: `FieldMov`, `FieldSLL64`, `FieldSLL128`, `FieldSLL192`
//!
//! The `execute` body is a placeholder — FR ops operate on 256-bit field
//! elements, not u64 operands, and don't feed into the RV lookup argument.
//! `LookupTable` is `None` for all FR instructions; the R1CS rows in
//! `jolt-r1cs/src/constraints/rv64.rs` gate directly on the circuit flags.
//!
//! Bridge instructions (FieldMov/SLL*) set `LeftOperandIsRs1Value` so
//! `V_RS1_VALUE` is populated from the integer register — the R1CS bridge
//! rows (28–31) reference it.

use crate::opcodes;

define_instruction!(
    /// BN254 Fr FieldMul: `FReg[frd] = FReg[frs1] · FReg[frs2]`.
    FieldMul, opcodes::FIELD_MUL, "FIELD_MUL",
    |_x, _y| 0,
    circuit: [IsFieldMul],
);

define_instruction!(
    /// BN254 Fr FieldAdd: `FReg[frd] = FReg[frs1] + FReg[frs2]`.
    FieldAdd, opcodes::FIELD_ADD, "FIELD_ADD",
    |_x, _y| 0,
    circuit: [IsFieldAdd],
);

define_instruction!(
    /// BN254 Fr FieldSub: `FReg[frd] = FReg[frs1] − FReg[frs2]`.
    FieldSub, opcodes::FIELD_SUB, "FIELD_SUB",
    |_x, _y| 0,
    circuit: [IsFieldSub],
);

define_instruction!(
    /// BN254 Fr FieldInv: `FReg[frd] = FReg[frs1]⁻¹`.
    FieldInv, opcodes::FIELD_INV, "FIELD_INV",
    |_x, _y| 0,
    circuit: [IsFieldInv],
);

define_instruction!(
    /// BN254 Fr FieldAssertEq: `assert FReg[frs1] == FReg[frs2]`; no write.
    FieldAssertEq, opcodes::FIELD_ASSERT_EQ, "FIELD_ASSERT_EQ",
    |_x, _y| 0,
    circuit: [IsFieldAssertEq],
);

define_instruction!(
    /// Integer→field FieldMov: `FReg[frd] = XReg[rs1] as Fr`.
    FieldMov, opcodes::FIELD_MOV, "FIELD_MOV",
    |_x, _y| 0,
    circuit: [IsFieldMov],
    instruction: [LeftOperandIsRs1Value],
);

define_instruction!(
    /// Integer→field FieldSLL64: `FReg[frd] = XReg[rs1] · 2⁶⁴`.
    FieldSLL64, opcodes::FIELD_SLL64, "FIELD_SLL64",
    |_x, _y| 0,
    circuit: [IsFieldSLL64],
    instruction: [LeftOperandIsRs1Value],
);

define_instruction!(
    /// Integer→field FieldSLL128: `FReg[frd] = XReg[rs1] · 2¹²⁸`.
    FieldSLL128, opcodes::FIELD_SLL128, "FIELD_SLL128",
    |_x, _y| 0,
    circuit: [IsFieldSLL128],
    instruction: [LeftOperandIsRs1Value],
);

define_instruction!(
    /// Integer→field FieldSLL192: `FReg[frd] = XReg[rs1] · 2¹⁹²`.
    FieldSLL192, opcodes::FIELD_SLL192, "FIELD_SLL192",
    |_x, _y| 0,
    circuit: [IsFieldSLL192],
    instruction: [LeftOperandIsRs1Value],
);
