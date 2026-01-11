use crate::emulator::cpu::{Cpu, Xlen};
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;

pub mod format_amo;
pub mod format_assert_align;
pub mod format_b;
pub mod format_fence;
pub mod format_i;
pub mod format_inline;
pub mod format_j;
pub mod format_load;
pub mod format_r;
pub mod format_s;
pub mod format_u;
pub mod format_virtual_right_shift_i;
pub mod format_virtual_right_shift_r;

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct NormalizedOperands {
    pub rs1: Option<u8>,
    pub rs2: Option<u8>,
    pub rd: Option<u8>,
    pub imm: i128,
}

pub trait InstructionFormat:
    Default + Debug + From<NormalizedOperands> + Into<NormalizedOperands>
{
    type RegisterState: InstructionRegisterState + PartialEq;

    fn parse(word: u32) -> Self;
    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu);
    fn capture_post_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu);
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self;
}

pub trait InstructionRegisterState:
    Default + Copy + Clone + Serialize + DeserializeOwned + Debug
{
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng, operands: &NormalizedOperands) -> Self;
    fn rs1_value(&self) -> Option<u64> {
        None
    }
    fn rs2_value(&self) -> Option<u64> {
        None
    }
    fn rd_values(&self) -> Option<(u64, u64)> {
        None
    }
}

pub fn normalize_register_value(value: i64, xlen: &Xlen) -> u64 {
    match xlen {
        Xlen::Bit32 => value as u32 as u64,
        Xlen::Bit64 => value as u64,
    }
}

pub fn normalize_imm(imm: u64, xlen: &Xlen) -> i64 {
    match xlen {
        Xlen::Bit32 => imm as i32 as i64,
        Xlen::Bit64 => imm as i64,
    }
}
