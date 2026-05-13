//! BN254 Fr native-field coprocessor v2 instruction kinds.
//!
//! Nine ops under custom-0 opcode `0x0B`:
//! - Arithmetic (FR-FR): `FieldMul`, `FieldAdd`, `FieldSub`, `FieldInv`,
//!   `FieldAssertEq`
//! - Integer→field bridge: `FieldMov`, `FieldSLL64`, `FieldSLL128`,
//!   `FieldSLL192`
//!
//! FR ops operate on 256-bit Fr elements and do not feed the RV lookup
//! argument — `InstructionLookupTable` impls are absent. The R1CS rows in
//! `jolt-r1cs/src/constraints/rv64.rs` gate directly on circuit flags.
//!
//! Bridge instructions (FieldMov/SLL*) set `LeftOperandIsRs1Value` so
//! `V_RS1_VALUE` is populated from the integer register — the R1CS bridge
//! rows reference it.

mod field_add;
mod field_assert_eq;
mod field_inv;
mod field_mov;
mod field_mul;
mod field_sll128;
mod field_sll192;
mod field_sll64;
mod field_sub;

pub use field_add::FieldAdd;
pub use field_assert_eq::FieldAssertEq;
pub use field_inv::FieldInv;
pub use field_mov::FieldMov;
pub use field_mul::FieldMul;
pub use field_sll128::FieldSLL128;
pub use field_sll192::FieldSLL192;
pub use field_sll64::FieldSLL64;
pub use field_sub::FieldSub;
