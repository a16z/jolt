#![cfg_attr(feature = "guest", no_std)]
use core::hint::black_box;

#[cfg(target_arch = "riscv64")]
fn signed_div(lhs: i64, rhs: i64) -> i64 {
    let result;
    // SAFETY: Register-only arithmetic with no memory or control-flow effects.
    unsafe {
        core::arch::asm!(
            "div {result}, {lhs}, {rhs}",
            result = lateout(reg) result,
            lhs = in(reg) lhs,
            rhs = in(reg) rhs,
            options(nomem, nostack)
        );
    }
    result
}

#[cfg(not(target_arch = "riscv64"))]
fn signed_div(lhs: i64, rhs: i64) -> i64 {
    if rhs == 0 {
        -1
    } else if lhs == i64::MIN && rhs == -1 {
        i64::MIN
    } else {
        lhs / rhs
    }
}

#[cfg(target_arch = "riscv64")]
fn signed_rem(lhs: i64, rhs: i64) -> i64 {
    let result;
    // SAFETY: Register-only arithmetic with no memory or control-flow effects.
    unsafe {
        core::arch::asm!(
            "rem {result}, {lhs}, {rhs}",
            result = lateout(reg) result,
            lhs = in(reg) lhs,
            rhs = in(reg) rhs,
            options(nomem, nostack)
        );
    }
    result
}

#[cfg(not(target_arch = "riscv64"))]
fn signed_rem(lhs: i64, rhs: i64) -> i64 {
    if rhs == 0 {
        lhs
    } else if lhs == i64::MIN && rhs == -1 {
        0
    } else {
        lhs % rhs
    }
}

#[cfg(target_arch = "riscv64")]
fn signed_divw(lhs: i64, rhs: i64) -> i64 {
    let result;
    // SAFETY: Register-only arithmetic with no memory or control-flow effects.
    unsafe {
        core::arch::asm!(
            "divw {result}, {lhs}, {rhs}",
            result = lateout(reg) result,
            lhs = in(reg) lhs,
            rhs = in(reg) rhs,
            options(nomem, nostack)
        );
    }
    result
}

#[cfg(not(target_arch = "riscv64"))]
fn signed_divw(lhs: i64, rhs: i64) -> i64 {
    let lhs = lhs as i32;
    let rhs = rhs as i32;
    if rhs == 0 {
        -1
    } else if lhs == i32::MIN && rhs == -1 {
        i32::MIN as i64
    } else {
        (lhs / rhs) as i64
    }
}

#[cfg(target_arch = "riscv64")]
fn signed_remw(lhs: i64, rhs: i64) -> i64 {
    let result;
    // SAFETY: Register-only arithmetic with no memory or control-flow effects.
    unsafe {
        core::arch::asm!(
            "remw {result}, {lhs}, {rhs}",
            result = lateout(reg) result,
            lhs = in(reg) lhs,
            rhs = in(reg) rhs,
            options(nomem, nostack)
        );
    }
    result
}

#[cfg(not(target_arch = "riscv64"))]
fn signed_remw(lhs: i64, rhs: i64) -> i64 {
    let lhs = lhs as i32;
    let rhs = rhs as i32;
    if rhs == 0 {
        lhs as i64
    } else if lhs == i32::MIN && rhs == -1 {
        0
    } else {
        (lhs % rhs) as i64
    }
}

#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn muldiv(a: u32, b: u32, c: u32) -> u32 {
    use jolt::{end_cycle_tracking, start_cycle_tracking};

    start_cycle_tracking("muldiv");
    black_box(signed_div(-37, 0));
    black_box(signed_rem(-37, 0));
    black_box(signed_div(i64::MIN, -1));
    black_box(signed_rem(i64::MIN, -1));
    black_box(signed_divw(-37, 0));
    black_box(signed_remw(-37, 0));
    black_box(signed_divw(i32::MIN as i64, -1));
    black_box(signed_remw(i32::MIN as i64, -1));
    let result = black_box(a * b / c); // use black_box to keep code in place
    end_cycle_tracking("muldiv");
    result
}
