#![cfg_attr(feature = "guest", no_std)]
use core::hint::black_box;

#[cfg(target_arch = "riscv64")]
fn srlw(lhs: i64, rhs: i64) -> i64 {
    let result;
    // SAFETY: Register-only arithmetic with no memory or control-flow effects.
    unsafe {
        core::arch::asm!(
            "srlw {result}, {lhs}, {rhs}",
            result = lateout(reg) result,
            lhs = in(reg) lhs,
            rhs = in(reg) rhs,
            options(nomem, nostack)
        );
    }
    result
}

#[cfg(not(target_arch = "riscv64"))]
fn srlw(lhs: i64, rhs: i64) -> i64 {
    ((lhs as u32) >> (rhs as u32 & 31)) as i32 as i64
}

#[cfg(target_arch = "riscv64")]
fn sraw(lhs: i64, rhs: i64) -> i64 {
    let result;
    // SAFETY: Register-only arithmetic with no memory or control-flow effects.
    unsafe {
        core::arch::asm!(
            "sraw {result}, {lhs}, {rhs}",
            result = lateout(reg) result,
            lhs = in(reg) lhs,
            rhs = in(reg) rhs,
            options(nomem, nostack)
        );
    }
    result
}

#[cfg(not(target_arch = "riscv64"))]
fn sraw(lhs: i64, rhs: i64) -> i64 {
    ((lhs as i32) >> (rhs as u32 & 31)) as i64
}

#[cfg(target_arch = "riscv64")]
fn srliw<const SHIFT: u32>(lhs: i64) -> i64 {
    let result;
    // SAFETY: Register-only arithmetic with no memory or control-flow effects.
    unsafe {
        core::arch::asm!(
            "srliw {result}, {lhs}, {shift}",
            result = lateout(reg) result,
            lhs = in(reg) lhs,
            shift = const SHIFT,
            options(nomem, nostack)
        );
    }
    result
}

#[cfg(not(target_arch = "riscv64"))]
fn srliw<const SHIFT: u32>(lhs: i64) -> i64 {
    ((lhs as u32) >> SHIFT) as i32 as i64
}

#[cfg(target_arch = "riscv64")]
fn sraiw<const SHIFT: u32>(lhs: i64) -> i64 {
    let result;
    // SAFETY: Register-only arithmetic with no memory or control-flow effects.
    unsafe {
        core::arch::asm!(
            "sraiw {result}, {lhs}, {shift}",
            result = lateout(reg) result,
            lhs = in(reg) lhs,
            shift = const SHIFT,
            options(nomem, nostack)
        );
    }
    result
}

#[cfg(not(target_arch = "riscv64"))]
fn sraiw<const SHIFT: u32>(lhs: i64) -> i64 {
    ((lhs as i32) >> SHIFT) as i64
}

#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn muldiv(a: u32, b: u32, c: u32) -> u32 {
    use jolt::{end_cycle_tracking, start_cycle_tracking};

    start_cycle_tracking("muldiv");
    let lhs = 0xa5a5_a5a5_8000_0001u64 as i64;
    let shift_checksum = srlw(lhs, 0)
        .wrapping_add(srlw(lhs, 31))
        .wrapping_add(sraw(lhs, 0))
        .wrapping_add(sraw(lhs, 31))
        .wrapping_add(srliw::<0>(lhs))
        .wrapping_add(srliw::<31>(lhs))
        .wrapping_add(sraiw::<0>(lhs))
        .wrapping_add(sraiw::<31>(lhs));
    let result = black_box(a * b / c).wrapping_add(black_box(shift_checksum as u32));
    end_cycle_tracking("muldiv");
    result
}
