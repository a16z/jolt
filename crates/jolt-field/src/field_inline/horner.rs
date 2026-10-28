use super::guest::{load, read_out};
use super::{emit, r_word, FUNCT3_ADD, FUNCT3_MUL, REG_A, REG_B, REG_SUM};

/// Evaluate rows at `alpha` with five register-resident Horner accumulators.
/// The caller checks that its power vector is exactly `[1, alpha, ...]`.
#[rustfmt::skip] pub(crate) fn weighted<const N: usize>(
    rows: &[&[[u64; N]]],
    weights: &[[u64; N]],
    alpha: &[u64; N],
    width: usize,
) -> [u64; N] {
    assert_eq!(rows.len(), weights.len(), "one weight per row");
    assert!(width > 0 && rows.iter().all(|row| row.len() == width));
    macro_rules! word {
        ($op:expr, $rd:expr, $a:expr, $b:expr) => {
            // SAFETY: a fixed field-inline operation; no Rust memory is accessed.
            unsafe {
                core::arch::asm!(
                    ".word {w}",
                    w = const r_word($op, $rd, $a, $b),
                    options(nostack)
                );
            }
        };
    }
    macro_rules! block {
        ($($row:ident => $k:literal),+) => {{
            $(let $row: &[[u64; N]] = *$row;)+
            // Initializing from the leading coefficient avoids multiplication
            // by zero. Each block uses only the existing field register lease.
            $(
                emit::row_acc_zero($k);
                // SAFETY: the nonempty width and all row lengths were checked.
                unsafe { load(REG_B, $row.get_unchecked(width - 1)); }
                word!(FUNCT3_ADD, 9 + $k, 9 + $k, REG_B);
            )+
            for j in (0..width - 1).rev() {
                $(
                    // SAFETY: j is below the validated width of every row.
                    unsafe { load(REG_B, $row.get_unchecked(j)); }
                    word!(FUNCT3_MUL, 9 + $k, 9 + $k, REG_A);
                    word!(FUNCT3_ADD, 9 + $k, 9 + $k, REG_B);
                )+
            }
        }};
    }
    emit::sum_zero();
    load(REG_A, alpha);
    for (block_rows, block_weights) in rows.chunks(5).zip(weights.chunks(5)) {
        match block_rows {
            [r0, r1, r2, r3, r4] => { #[cfg(feature = "field-inline-horner-triplet")] if N == 2 && width == 64 { triplet_block([r0, r1, r2, r3, r4]); } else { block!(r0 => 0, r1 => 1, r2 => 2, r3 => 3, r4 => 4); } #[cfg(not(feature = "field-inline-horner-triplet"))] block!(r0 => 0, r1 => 1, r2 => 2, r3 => 3, r4 => 4); },
            [r0, r1, r2, r3] => block!(r0 => 0, r1 => 1, r2 => 2, r3 => 3),
            [r0, r1, r2] => block!(r0 => 0, r1 => 1, r2 => 2),
            [r0, r1] => block!(r0 => 0, r1 => 1),
            [r0] => block!(r0 => 0),
            _ => {}
        }
        for (k, weight) in block_weights.iter().enumerate() {
            load(REG_B, weight);
            emit::mul_out_row_acc_b(k);
            emit::sum_add_out();
        }
    }
    read_out(REG_SUM)
}

#[cfg(feature = "field-inline-horner-triplet")]
use jolt_riscv::field_inline_load_accumulate_from_memory_funct7;

/// The caller has checked five rows of 64 two-limb elements. REG_A holds
/// alpha; registers 9..=13 belong to these row accumulators. The existing
/// memory instruction's offsets cover three coefficients from one base.
#[cfg(feature = "field-inline-horner-triplet")]
#[inline(always)]
fn triplet_block<const N: usize>(rows: [&[[u64; N]]; 5]) {
    let [r0, r1, r2, r3, r4] = rows;
    macro_rules! word {
        ($op:expr, $rd:expr, $a:expr, $b:expr) => {
            // SAFETY: fixed field operations within the caller's register lease.
            unsafe {
                core::arch::asm!(
                    ".word {w}",
                    w = const r_word($op, $rd, $a, $b),
                    options(nostack)
                );
            }
        };
    }
    macro_rules! initialize {
        ($($row:ident => $k:literal),+) => {$(
            emit::row_acc_zero($k);
            // SAFETY: the caller validated all row lengths as 64.
            unsafe { load(REG_B, $row.get_unchecked(63)); }
            word!(FUNCT3_ADD, 9 + $k, 9 + $k, REG_B);
        )+};
    }
    macro_rules! advance {
        ($base:ident, $k:literal, $coefficient:literal) => {
            emit::clear(REG_B);
            // SAFETY: base identifies elements j..j+2 in a validated row,
            // N == 2, and coefficient is 0, 1 or 2. The scratch output is
            // early-clobber: its first write must not overwrite the base
            // needed by the second load. Both reads are bound by the same
            // field-memory relation as the ordinary load helper.
            unsafe {
                core::arch::asm!(
                    ".insn r {opcode}, {funct3}, {high}, {scratch}, {base}, x{dst}",
                    ".insn r {opcode}, {funct3}, {low}, {scratch}, {base}, x{dst}",
                    opcode = const super::OPCODE,
                    funct3 = const super::FUNCT3_LOAD_ACCUMULATE,
                    high = const field_inline_load_accumulate_from_memory_funct7(2 * $coefficient + 1),
                    low = const field_inline_load_accumulate_from_memory_funct7(2 * $coefficient),
                    dst = const REG_B,
                    base = in(reg) $base,
                    scratch = out(reg) _,
                    options(nostack)
                );
            }
            word!(FUNCT3_MUL, 9 + $k, 9 + $k, REG_A);
            word!(FUNCT3_ADD, 9 + $k, 9 + $k, REG_B);
        };
    }
    macro_rules! group {
        ($j:expr; $($row:ident => $k:literal),+) => {$(
            // SAFETY: j ranges from 60 down to 0 in steps of 3; each pair
            // of limb loads remains inside the row's first 63 elements.
            let base = unsafe { $row.as_ptr().add($j) };
            advance!(base, $k, 2);
            advance!(base, $k, 1);
            advance!(base, $k, 0);
        )+};
    }
    initialize!(r0 => 0, r1 => 1, r2 => 2, r3 => 3, r4 => 4);
    for j in (0..63).step_by(3).rev() {
        group!(j; r0 => 0, r1 => 1, r2 => 2, r3 => 3, r4 => 4);
    }
}
