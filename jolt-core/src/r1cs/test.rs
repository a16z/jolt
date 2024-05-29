use crate::impl_r1cs_input_lc_conversions;

use super::ops::ConstraintInput;

#[allow(non_camel_case_types)]
#[derive(
    strum_macros::EnumIter,
    strum_macros::EnumCount,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
)]
#[repr(usize)]
pub enum TestInputs {
    PcIn,
    PcOut,
    BytecodeA,
    BytecodeVOpcode,
    BytecodeVRS1,
    BytecodeVRS2,
    BytecodeVRD,
    BytecodeVImm,
    RAMA,
    RAMRS1,
    RAMRS2,
    RAMByte0,
    RAMByte1,
    RAMByte2,
    RAMByte3,
    OpFlags0,
    OpFlags1,
    OpFlags2,
    OpFlags3,
    OpFlags_SignImm,
}
impl ConstraintInput for TestInputs {}
impl_r1cs_input_lc_conversions!(TestInputs);
