use super::guest::{load, read_out};
use super::{emit, r_word, FUNCT3_ADD, FUNCT3_MUL, REG_A, REG_B, REG_SUM};

/// Evaluate rows at `alpha` with five register-resident Horner accumulators.
/// The caller checks that its power vector is exactly `[1, alpha, ...]`.
pub(crate) fn weighted<const N: usize>(
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
            [r0, r1, r2, r3, r4] => block!(r0 => 0, r1 => 1, r2 => 2, r3 => 3, r4 => 4),
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
