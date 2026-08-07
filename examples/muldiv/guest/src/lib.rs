#![cfg_attr(feature = "guest", no_std)]
use core::hint::black_box;

#[cfg(target_arch = "riscv64")]
fn addw(lhs: i64, rhs: i64) -> i64 {
    let result;
    // SAFETY: Register-only arithmetic with no memory or control-flow effects.
    unsafe {
        core::arch::asm!(
            "addw {result}, {lhs}, {rhs}",
            result = lateout(reg) result,
            lhs = in(reg) lhs,
            rhs = in(reg) rhs,
            options(nomem, nostack)
        );
    }
    result
}

#[cfg(not(target_arch = "riscv64"))]
fn addw(lhs: i64, rhs: i64) -> i64 {
    lhs.wrapping_add(rhs) as i32 as i64
}

#[cfg(target_arch = "riscv64")]
fn addiw<const IMM: i32>(lhs: i64) -> i64 {
    let result;
    // SAFETY: Register-only arithmetic with no memory or control-flow effects.
    unsafe {
        core::arch::asm!(
            "addiw {result}, {lhs}, {imm}",
            result = lateout(reg) result,
            lhs = in(reg) lhs,
            imm = const IMM,
            options(nomem, nostack)
        );
    }
    result
}

#[cfg(not(target_arch = "riscv64"))]
fn addiw<const IMM: i32>(lhs: i64) -> i64 {
    lhs.wrapping_add(IMM as i64) as i32 as i64
}

#[cfg(target_arch = "riscv64")]
fn subw(lhs: i64, rhs: i64) -> i64 {
    let result;
    // SAFETY: Register-only arithmetic with no memory or control-flow effects.
    unsafe {
        core::arch::asm!(
            "subw {result}, {lhs}, {rhs}",
            result = lateout(reg) result,
            lhs = in(reg) lhs,
            rhs = in(reg) rhs,
            options(nomem, nostack)
        );
    }
    result
}

#[cfg(not(target_arch = "riscv64"))]
fn subw(lhs: i64, rhs: i64) -> i64 {
    lhs.wrapping_sub(rhs) as i32 as i64
}

#[cfg(target_arch = "riscv64")]
fn mulw(lhs: i64, rhs: i64) -> i64 {
    let result;
    // SAFETY: Register-only arithmetic with no memory or control-flow effects.
    unsafe {
        core::arch::asm!(
            "mulw {result}, {lhs}, {rhs}",
            result = lateout(reg) result,
            lhs = in(reg) lhs,
            rhs = in(reg) rhs,
            options(nomem, nostack)
        );
    }
    result
}

#[cfg(not(target_arch = "riscv64"))]
fn mulw(lhs: i64, rhs: i64) -> i64 {
    lhs.wrapping_mul(rhs) as i32 as i64
}

#[cfg(target_arch = "riscv64")]
fn slliw<const SHIFT: u32>(lhs: i64) -> i64 {
    let result;
    // SAFETY: Register-only arithmetic with no memory or control-flow effects.
    unsafe {
        core::arch::asm!(
            "slliw {result}, {lhs}, {shift}",
            result = lateout(reg) result,
            lhs = in(reg) lhs,
            shift = const SHIFT,
            options(nomem, nostack)
        );
    }
    result
}

#[cfg(not(target_arch = "riscv64"))]
fn slliw<const SHIFT: u32>(lhs: i64) -> i64 {
    ((lhs as u32) << SHIFT) as i32 as i64
}

#[cfg(target_arch = "riscv64")]
fn sllw(lhs: i64, rhs: i64) -> i64 {
    let result;
    // SAFETY: Register-only arithmetic with no memory or control-flow effects.
    unsafe {
        core::arch::asm!(
            "sllw {result}, {lhs}, {rhs}",
            result = lateout(reg) result,
            lhs = in(reg) lhs,
            rhs = in(reg) rhs,
            options(nomem, nostack)
        );
    }
    result
}

#[cfg(not(target_arch = "riscv64"))]
fn sllw(lhs: i64, rhs: i64) -> i64 {
    ((lhs as u32) << (rhs as u32 & 31)) as i32 as i64
}

fn fused_word_ops_checksum() -> u32 {
    let word_positive = 0xdead_beef_7fff_ffff_u64 as i64;
    let word_negative = 0xa5a5_5a5a_8000_0001_u64 as i64;
    let garbage_one = 0xffff_ffff_0000_0001_u64 as i64;
    let garbage_minus_one = 0x0123_4567_ffff_ffff_u64 as i64;

    [
        addw(word_positive, garbage_one),
        addw(word_negative, garbage_minus_one),
        addiw::<1>(word_positive),
        addiw::<2047>(word_negative),
        subw(word_negative, garbage_one),
        subw(word_positive, garbage_minus_one),
        mulw(word_negative, garbage_minus_one),
        mulw(word_positive, garbage_one),
        slliw::<0>(word_negative),
        slliw::<31>(word_positive),
        sllw(word_negative, 0),
        sllw(word_positive, 31),
    ]
    .into_iter()
    .fold(0_u64, |checksum, value| {
        checksum.wrapping_add(black_box(value) as u64)
    }) as u32
}

#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn muldiv(a: u32, b: u32, c: u32) -> u32 {
    use jolt::{end_cycle_tracking, start_cycle_tracking};

    start_cycle_tracking("muldiv");
    let result = black_box(a * b / c).wrapping_add(black_box(fused_word_ops_checksum()));
    end_cycle_tracking("muldiv");
    result
}
