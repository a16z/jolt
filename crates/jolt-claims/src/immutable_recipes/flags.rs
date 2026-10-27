use jolt_riscv::{CircuitFlags, InstructionFlags};

#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { jolt_riscv; CircuitFlags {
    AddOperands = 0,
    SubtractOperands = 1,
    MultiplyOperands = 2,
    Load = 3,
    Store = 4,
    Jump = 5,
    WriteLookupOutputToRD = 6,
    VirtualInstruction = 7,
    Assert = 8,
    DoNotUpdateUnexpandedPC = 9,
    Advice = 10,
    IsCompressed = 11,
    IsFirstInSequence = 12,
    IsLastInSequence = 13,
    #[cfg(feature = "field-inline")]
    FieldAdd = 14,
    #[cfg(feature = "field-inline")]
    FieldSub = 15,
    #[cfg(feature = "field-inline")]
    FieldMul = 16,
    #[cfg(feature = "field-inline")]
    FieldInv = 17,
    #[cfg(feature = "field-inline")]
    FieldAssertEq = 18,
    #[cfg(feature = "field-inline")]
    FieldLoadAccumulateFromRegister = 19,
    #[cfg(feature = "field-inline")]
    FieldAssertZero = 20,
    #[cfg(feature = "field-inline")]
    FieldLoadImm = 21,
    #[cfg(feature = "field-inline")]
    FieldLoadAccumulateFromMemory = 22,
    #[cfg(feature = "field-inline")]
    FieldAdviceLimb = 23,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { jolt_riscv; InstructionFlags {
    LeftOperandIsPC = 0,
    RightOperandIsImm = 1,
    LeftOperandIsRs1Value = 2,
    RightOperandIsRs2Value = 3,
    Branch = 4,
    IsNoop = 5,
} }
