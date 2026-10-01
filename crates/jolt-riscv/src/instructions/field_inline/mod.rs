//! Native field-inline instructions.

pub mod add;
pub mod advice_limb;
pub mod assert_eq;
pub mod assert_zero;
pub mod inv;
pub mod load_accumulate_from_memory;
pub mod load_accumulate_from_register;
pub mod load_imm;
pub mod mul;
pub mod sub;

pub use add::FieldAdd;
pub use advice_limb::FieldAdviceLimb;
pub use assert_eq::FieldAssertEq;
pub use assert_zero::FieldAssertZero;
pub use inv::FieldInv;
pub use load_accumulate_from_memory::FieldLoadAccumulateFromMemory;
pub use load_accumulate_from_register::FieldLoadAccumulateFromRegister;
pub use load_imm::FieldLoadImm;
pub use mul::FieldMul;
pub use sub::FieldSub;
