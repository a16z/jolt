//! Six-product Montgomery pointwise accumulation.

#[cfg(target_arch = "riscv64")]
use crate::OPCODE;
use crate::{reduce, DEGREE};

pub const DOT_PRODUCTS: usize = 6;
#[cfg(any(feature = "host", target_arch = "riscv64"))]
pub const DOT_FUNCT7: u32 = 9;

/// Add six pointwise products to a 64-coefficient Montgomery accumulator.
///
/// With canonical inputs and an odd prime `0 < p < 2^30`, computes
/// `acc[i] += sum(lhs[j][i] * rhs[j][i]) / 2^32 (mod p)`, returning canonical
/// residues. `pinv` is `p^-1 mod 2^32`. A shorter dot can pad with zero arrays.
/// Other inputs follow explicit wrapping integer arithmetic. Unaligned arrays
/// use the portable path; the inline requires doubleword-aligned paired loads.
#[inline]
pub fn pointwise_dot64(
    acc: &mut [i32; DEGREE],
    lhs: [&[i32; DEGREE]; DOT_PRODUCTS],
    rhs: [&[i32; DEGREE]; DOT_PRODUCTS],
    p: i32,
    pinv: i32,
) {
    #[cfg(target_arch = "riscv64")]
    if acc.as_ptr() as usize & 7 == 0
        && lhs
            .iter()
            .chain(&rhs)
            .all(|row| row.as_ptr() as usize & 7 == 0)
    {
        let mut params = [0u64; 2 * DOT_PRODUCTS + 1];
        for (slot, row) in params.iter_mut().zip(lhs.iter().chain(&rhs)) {
            *slot = row.as_ptr() as u64;
        }
        params[2 * DOT_PRODUCTS] = u64::from(p as u32) | (u64::from(pinv as u32) << 32);
        // SAFETY: every paired access is within a live, doubleword-aligned
        // array. The inline reads operands and writes only the accumulator.
        unsafe {
            core::arch::asm!(
                ".insn r {opcode}, 0, {funct7}, x0, {acc}, {params}",
                opcode = const OPCODE,
                funct7 = const DOT_FUNCT7,
                acc = in(reg) acc.as_mut_ptr(),
                params = in(reg) params.as_ptr(),
                options(nostack),
            );
        }
        return;
    }
    portable_dot(acc, lhs, rhs, p, pinv);
}

fn portable_dot(
    acc: &mut [i32; DEGREE],
    lhs: [&[i32; DEGREE]; DOT_PRODUCTS],
    rhs: [&[i32; DEGREE]; DOT_PRODUCTS],
    p: i32,
    pinv: i32,
) {
    for (lane, coefficient) in acc.iter_mut().enumerate() {
        let sum = lhs.iter().zip(rhs).fold(0i64, |sum, (a, b)| {
            sum.wrapping_add(i64::from(a[lane]) * i64::from(b[lane]))
        });
        let correction = (sum as i32).wrapping_mul(pinv);
        let batch = (sum.wrapping_sub(i64::from(correction) * i64::from(p)) >> 32) as i32;
        *coefficient = reduce(coefficient.wrapping_add(reduce(batch, p)), p);
    }
}
