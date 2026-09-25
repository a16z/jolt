use crate::emulator::cpu::Cpu;
#[cfg(any(feature = "test-utils", test))]
use jolt_riscv::NormalizedOperands;
#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;

use super::RISCVInstruction;

pub mod advice_load_i;
pub mod amo;
pub mod assert_align;
pub mod b;
pub mod fence;
#[cfg(feature = "field-inline")]
pub mod field_inline;
pub mod i;
pub mod inline;
pub mod j;
pub mod load;
pub mod r;
pub mod s;
pub mod t;
pub mod u;
pub mod virtual_right_shift_i;
pub mod virtual_right_shift_r;

pub trait InstructionRegisterState:
    Default + Copy + Clone + Serialize + DeserializeOwned + Debug
{
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut StdRng, operands: &NormalizedOperands) -> Self;
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

pub(crate) fn normalize_register_value(cpu: &Cpu, reg: usize) -> u64 {
    let value = match reg {
        0 => {
            debug_assert_eq!(cpu.x[reg], 0);
            0
        }
        _ => cpu.x[reg],
    };
    value as u64
}

/// Capture register inputs before execution and complete the record afterwards.
/// Destination post-values initially equal their pre-values; the trace driver
/// must call `capture_post` before publishing the cycle.
pub trait RegisterSnapshot<I: RISCVInstruction>: InstructionRegisterState {
    fn capture_pre(instruction: &I, cpu: &Cpu) -> Self;
    fn capture_post(&mut self, instruction: &I, cpu: &Cpu);
}
