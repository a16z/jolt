use super::guest::{load, read_out};
use super::{emit, r_word, FUNCT3_ADD, REG_A, REG_B, REG_OUT};

/// No field callback runs while the scale occupies REG_B. Products stay in
/// REG_OUT until their addition; the caller checks each final canonical value.
#[inline(always)]
pub(crate) fn add_scaled_tile(scale: &[u64; 2], src: &[[u64; 2]], dst: &mut [[u64; 2]]) {
    load(REG_B, scale);
    for (value, slot) in src.iter().zip(dst) {
        load(REG_A, value);
        emit::mul_out();
        load(REG_A, slot);
        // SAFETY: a fixed field operation within the arithmetic register lease.
        unsafe {
            core::arch::asm!(
                ".word {word}",
                word = const r_word(FUNCT3_ADD, REG_OUT, REG_OUT, REG_A),
                options(nostack)
            );
        }
        *slot = read_out(REG_OUT);
    }
}
