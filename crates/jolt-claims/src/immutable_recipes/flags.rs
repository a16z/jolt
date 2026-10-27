use jolt_riscv::{CircuitFlags, InstructionFlags};

#[cfg(feature = "immutable-recipes")]
crate::protocols::jolt::geometry::spartan::circuit_flags_recipe_id!();
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { jolt_riscv; InstructionFlags {
    LeftOperandIsPC = 0,
    RightOperandIsImm = 1,
    LeftOperandIsRs1Value = 2,
    RightOperandIsRs2Value = 3,
    Branch = 4,
    IsNoop = 5,
} }
