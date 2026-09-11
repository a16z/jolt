//! A 64-point Montgomery NTT expanded into existing proved integer instructions.
#![cfg_attr(not(feature = "host"), no_std)]

pub const DEGREE: usize = 64;
pub const OPCODE: u32 = 0x0b;
pub const FUNCT3: u32 = 0;
pub const FUNCT7: u32 = 8;

#[cfg(feature = "host")]
pub mod sequence_builder;
#[cfg(feature = "host")]
use jolt_inlines_sdk::host::InlineExtension;
#[cfg(feature = "host")]
use sequence_builder::ForwardNtt64;
#[cfg(feature = "host")]
jolt_inlines_sdk::register_inlines! {
    trace_file: "ntt_trace.joltinline",
    extension: InlineExtension::Ntt,
    ops: [ForwardNtt64],
}

/// Twist, then forward DIF NTT, with bit-reversed output in Montgomery form.
///
/// `psi[i]` is the Montgomery form of psi^i. Stage twiddles occupy
/// `twiddles[len - 1..2*len - 1]`, for half-lengths 32 down to 1.
/// For NTT semantics, the caller supplies an odd prime `0 < p < 2^30`,
/// `pinv = p^-1 mod 2^32`, valid roots, and coefficients in `(-p, p)`.
/// Arithmetic outside that domain still follows signed wrapping i32/i64
/// operations; the inline introduces no trusted advice or unchecked equation.
/// Arrays without doubleword alignment use aligned stack buffers on RISC-V.
#[inline]
pub fn forward_ntt64(
    state: &mut [i32; DEGREE],
    psi: &[i32; DEGREE],
    twiddles: &[i32; DEGREE],
    p: i32,
    pinv: i32,
) {
    #[cfg(target_arch = "riscv64")]
    {
        if (state.as_ptr() as usize | psi.as_ptr() as usize | twiddles.as_ptr() as usize) & 7 == 0 {
            // SAFETY: the guard establishes alignment for all paired accesses.
            unsafe { forward_aligned(state, psi, twiddles, p, pinv) };
        } else {
            #[repr(align(8))]
            struct Aligned([i32; DEGREE]);
            let mut aligned_state = Aligned(*state);
            let aligned_psi = Aligned(*psi);
            let aligned_twiddles = Aligned(*twiddles);
            // SAFETY: the stack buffers have explicit doubleword alignment.
            unsafe {
                forward_aligned(
                    &mut aligned_state.0,
                    &aligned_psi.0,
                    &aligned_twiddles.0,
                    p,
                    pinv,
                )
            };
            *state = aligned_state.0;
        }
    }
    #[cfg(not(target_arch = "riscv64"))]
    scalar_forward(state, psi, twiddles, p, pinv);
}

#[cfg(target_arch = "riscv64")]
#[inline]
unsafe fn forward_aligned(
    state: &mut [i32; DEGREE],
    psi: &[i32; DEGREE],
    twiddles: &[i32; DEGREE],
    p: i32,
    pinv: i32,
) {
    let params = [
        psi.as_ptr() as u64,
        twiddles.as_ptr() as u64,
        u64::from(p as u32) | (u64::from(pinv as u32) << 32),
    ];
    // SAFETY: the caller supplies aligned live arrays. No virtual row writes
    // the tables; all coefficient accesses stay within the state array.
    unsafe {
        core::arch::asm!(
            ".insn r {opcode}, {funct3}, {funct7}, x0, {state}, {params}",
            opcode = const OPCODE,
            funct3 = const FUNCT3,
            funct7 = const FUNCT7,
            state = in(reg) state.as_mut_ptr(),
            params = in(reg) params.as_ptr(),
            options(nostack),
        );
    }
}

#[cfg(not(target_arch = "riscv64"))]
fn mont_mul(a: i32, b: i32, p: i32, pinv: i32) -> i32 {
    let c = i64::from(a) * i64::from(b);
    let t = (c as i32).wrapping_mul(pinv);
    (c.wrapping_sub(i64::from(t) * i64::from(p)) >> 32) as i32
}

#[cfg(not(target_arch = "riscv64"))]
fn reduce(a: i32, p: i32) -> i32 {
    let p = i64::from(p);
    let diff = i64::from(a) - p;
    let a = diff + ((diff >> 63) & p);
    (a + ((a >> 63) & p)) as i32
}

#[cfg(not(target_arch = "riscv64"))]
fn scalar_forward(
    state: &mut [i32; DEGREE],
    psi: &[i32; DEGREE],
    twiddles: &[i32; DEGREE],
    p: i32,
    pinv: i32,
) {
    for (a, w) in state.iter_mut().zip(psi) {
        *a = mont_mul(*a, *w, p, pinv);
    }
    let mut len = DEGREE / 2;
    while len != 0 {
        for start in (0..DEGREE).step_by(2 * len) {
            for j in 0..len {
                let u = state[start + j];
                let v = state[start + j + len];
                state[start + j] = reduce(u.wrapping_add(v), p);
                state[start + j + len] =
                    mont_mul(u.wrapping_sub(v), twiddles[len - 1 + j], p, pinv);
            }
        }
        len /= 2;
    }
    for a in state {
        *a = reduce(*a, p);
    }
}

#[cfg(all(test, feature = "host"))]
mod tests;
