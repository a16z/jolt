pub use jolt_riscv::NormalizedOperands;
use std::fmt::Debug;

pub mod format_advice_load_i;
pub mod format_amo;
pub mod format_assert_align;
pub mod format_b;
pub mod format_fence;
#[cfg(feature = "field-inline")]
pub mod format_field_inline;
pub mod format_i;
pub mod format_inline;
pub mod format_j;
pub mod format_load;
pub mod format_r;
pub mod format_s;
pub mod format_t;
pub mod format_u;
pub mod format_virtual_right_shift_i;
pub mod format_virtual_right_shift_r;

pub trait InstructionFormat:
    Default + Debug + From<NormalizedOperands> + Into<NormalizedOperands>
{
    fn parse(word: u32) -> Self;
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self;

    /// Overwrite the destination register. Default is a no-op for formats
    /// without a destination register (branches, stores).
    fn set_rd(&mut self, _rd: u8) {}
}

pub fn normalize_imm(imm: u64) -> i64 {
    imm as i64
}
