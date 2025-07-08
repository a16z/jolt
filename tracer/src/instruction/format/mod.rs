use crate::emulator::cpu::{Cpu, Xlen};
use rand::rngs::StdRng;
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;

pub mod format_b;
pub mod format_i;
pub mod format_j;
pub mod format_load;
pub mod format_r;
pub mod format_s;
pub mod format_u;
pub mod format_virtual_halfword_alignment;
pub mod format_virtual_right_shift_i;
pub mod format_virtual_right_shift_r;

#[derive(Default)]
pub struct NormalizedOperands {
    pub rs1: usize,
    pub rs2: usize,
    pub rd: usize,
    pub imm: i64,
}

pub trait InstructionFormat: Default + Debug {
    type RegisterState: InstructionRegisterState + PartialEq;

    fn parse(word: u32) -> Self;
    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu);
    fn capture_post_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu);
    fn random(rng: &mut StdRng) -> Self;
    fn normalize(&self) -> NormalizedOperands;
}

pub trait InstructionRegisterState:
    Default + Copy + Clone + Serialize + DeserializeOwned + Debug
{
    fn random(rng: &mut StdRng) -> Self;
    fn rs1_value(&self) -> u64 {
        0
    }
    fn rs2_value(&self) -> u64 {
        0
    }
    fn rd_values(&self) -> (u64, u64) {
        (0, 0)
    }
}

pub fn normalize_register_value(value: i64, xlen: &Xlen) -> u64 {
    match xlen {
        Xlen::Bit32 => value as u32 as u64,
        Xlen::Bit64 => value as u64,
    }
}
