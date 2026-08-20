use super::*;
use jolt_riscv::JoltInstructionKind as Kind;

mod mulh;
mod mulhsu;
pub(super) use mulh::expand_mulh;
pub(super) use mulhsu::expand_mulhsu;
