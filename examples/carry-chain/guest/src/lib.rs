//! Exercises every `{ADD, MUL} -> {ADDC, MULC}` implicit-carry pairing via raw
//! `.insn` assembly. Dependent chains stay inside one `asm!` block: any
//! intervening non-carry-producing instruction clears the carry to zero.
#![cfg_attr(feature = "guest", no_std)]

/// The widening results of the four carry chains, folded so the host can
/// compare against a `u128`-arithmetic reference.
#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn carry_chain(a: u64, b: u64, c: u64) -> u64 {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    {
        let (add_lo, add_hi): (u64, u64);
        let (mul_lo, mul_hi): (u64, u64);
        let (addc_lo, addc_hi): (u64, u64);
        let (mulc_lo, mulc_hi): (u64, u64);
        unsafe {
            // ADD -> ADDC: 128-bit add (a + b as the low limb, c + a with
            // carry-in as the high limb).
            core::arch::asm!(
                "add {lo}, {a}, {b}",
                ".insn r 0x5B, 0x0, 0x06, {hi}, {c}, {a}",
                lo = out(reg) add_lo,
                hi = out(reg) add_hi,
                a = in(reg) a,
                b = in(reg) b,
                c = in(reg) c,
                options(nomem, nostack)
            );
            // MUL -> ADDC: widening multiply; ADDC x0 + x0 + carry extracts
            // the high half.
            core::arch::asm!(
                "mul {lo}, {a}, {b}",
                ".insn r 0x5B, 0x0, 0x06, {hi}, x0, x0",
                lo = out(reg) mul_lo,
                hi = out(reg) mul_hi,
                a = in(reg) a,
                b = in(reg) b,
                options(nomem, nostack)
            );
            // ADD -> MULC: the add's carry feeds a multiply-accumulate low
            // half; a trailing ADDC extracts the multiply's carry-out.
            core::arch::asm!(
                "add {lo}, {a}, {b}",
                ".insn r 0x5B, 0x0, 0x07, {lo}, {b}, {c}",
                ".insn r 0x5B, 0x0, 0x06, {hi}, x0, x0",
                lo = out(reg) addc_lo,
                hi = out(reg) addc_hi,
                a = in(reg) a,
                b = in(reg) b,
                c = in(reg) c,
                options(nomem, nostack)
            );
            // MUL -> MULC: carry-chained product limbs.
            core::arch::asm!(
                "mul {lo}, {a}, {c}",
                ".insn r 0x5B, 0x0, 0x07, {hi}, {b}, {c}",
                lo = out(reg) mulc_lo,
                hi = out(reg) mulc_hi,
                a = in(reg) a,
                b = in(reg) b,
                c = in(reg) c,
                options(nomem, nostack)
            );
        }
        add_lo
            .wrapping_mul(3)
            .wrapping_add(add_hi.wrapping_mul(5))
            .wrapping_add(mul_lo.wrapping_mul(7))
            .wrapping_add(mul_hi.wrapping_mul(11))
            .wrapping_add(addc_lo.wrapping_mul(13))
            .wrapping_add(addc_hi.wrapping_mul(17))
            .wrapping_add(mulc_lo.wrapping_mul(19))
            .wrapping_add(mulc_hi.wrapping_mul(23))
    }
    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    {
        carry_chain_reference(a, b, c)
    }
}

/// 256x256 -> 512-bit schoolbook multiplication built on MULC/ADDC carry
/// chains, folded to a `u64` checksum. Each row is one `asm!` block holding
/// two chains (row product, then accumulation), each opened by a
/// carry-producing `mul`/`add` so no stale carry leaks in.
#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn mul256_carry(a: [u64; 4], b: [u64; 4]) -> u64 {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    {
        let (a0, a1, a2, a3) = (a[0], a[1], a[2], a[3]);
        let (r0, mut r1, mut r2, mut r3, mut r4, mut r5, mut r6, r7): (
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
        );
        unsafe {
            // Row 0: r[0..5] = a * b[0].
            core::arch::asm!(
                "mul {r0}, {a0}, {bi}",
                ".insn r 0x5B, 0x0, 0x07, {r1}, {a1}, {bi}",
                ".insn r 0x5B, 0x0, 0x07, {r2}, {a2}, {bi}",
                ".insn r 0x5B, 0x0, 0x07, {r3}, {a3}, {bi}",
                ".insn r 0x5B, 0x0, 0x06, {r4}, x0, x0",
                r0 = out(reg) r0,
                r1 = out(reg) r1,
                r2 = out(reg) r2,
                r3 = out(reg) r3,
                r4 = out(reg) r4,
                a0 = in(reg) a0,
                a1 = in(reg) a1,
                a2 = in(reg) a2,
                a3 = in(reg) a3,
                bi = in(reg) b[0],
                options(nomem, nostack)
            );
            // Row 1: r[1..6] += a * b[1].
            core::arch::asm!(
                "mul {t0}, {a0}, {bi}",
                ".insn r 0x5B, 0x0, 0x07, {t1}, {a1}, {bi}",
                ".insn r 0x5B, 0x0, 0x07, {t2}, {a2}, {bi}",
                ".insn r 0x5B, 0x0, 0x07, {t3}, {a3}, {bi}",
                ".insn r 0x5B, 0x0, 0x06, {t4}, x0, x0",
                "add {r1}, {r1}, {t0}",
                ".insn r 0x5B, 0x0, 0x06, {r2}, {r2}, {t1}",
                ".insn r 0x5B, 0x0, 0x06, {r3}, {r3}, {t2}",
                ".insn r 0x5B, 0x0, 0x06, {r4}, {r4}, {t3}",
                ".insn r 0x5B, 0x0, 0x06, {r5}, {t4}, x0",
                t0 = out(reg) _,
                t1 = out(reg) _,
                t2 = out(reg) _,
                t3 = out(reg) _,
                t4 = out(reg) _,
                r1 = inout(reg) r1,
                r2 = inout(reg) r2,
                r3 = inout(reg) r3,
                r4 = inout(reg) r4,
                r5 = out(reg) r5,
                a0 = in(reg) a0,
                a1 = in(reg) a1,
                a2 = in(reg) a2,
                a3 = in(reg) a3,
                bi = in(reg) b[1],
                options(nomem, nostack)
            );
            // Row 2: r[2..7] += a * b[2].
            core::arch::asm!(
                "mul {t0}, {a0}, {bi}",
                ".insn r 0x5B, 0x0, 0x07, {t1}, {a1}, {bi}",
                ".insn r 0x5B, 0x0, 0x07, {t2}, {a2}, {bi}",
                ".insn r 0x5B, 0x0, 0x07, {t3}, {a3}, {bi}",
                ".insn r 0x5B, 0x0, 0x06, {t4}, x0, x0",
                "add {r2}, {r2}, {t0}",
                ".insn r 0x5B, 0x0, 0x06, {r3}, {r3}, {t1}",
                ".insn r 0x5B, 0x0, 0x06, {r4}, {r4}, {t2}",
                ".insn r 0x5B, 0x0, 0x06, {r5}, {r5}, {t3}",
                ".insn r 0x5B, 0x0, 0x06, {r6}, {t4}, x0",
                t0 = out(reg) _,
                t1 = out(reg) _,
                t2 = out(reg) _,
                t3 = out(reg) _,
                t4 = out(reg) _,
                r2 = inout(reg) r2,
                r3 = inout(reg) r3,
                r4 = inout(reg) r4,
                r5 = inout(reg) r5,
                r6 = out(reg) r6,
                a0 = in(reg) a0,
                a1 = in(reg) a1,
                a2 = in(reg) a2,
                a3 = in(reg) a3,
                bi = in(reg) b[2],
                options(nomem, nostack)
            );
            // Row 3: r[3..8] += a * b[3].
            core::arch::asm!(
                "mul {t0}, {a0}, {bi}",
                ".insn r 0x5B, 0x0, 0x07, {t1}, {a1}, {bi}",
                ".insn r 0x5B, 0x0, 0x07, {t2}, {a2}, {bi}",
                ".insn r 0x5B, 0x0, 0x07, {t3}, {a3}, {bi}",
                ".insn r 0x5B, 0x0, 0x06, {t4}, x0, x0",
                "add {r3}, {r3}, {t0}",
                ".insn r 0x5B, 0x0, 0x06, {r4}, {r4}, {t1}",
                ".insn r 0x5B, 0x0, 0x06, {r5}, {r5}, {t2}",
                ".insn r 0x5B, 0x0, 0x06, {r6}, {r6}, {t3}",
                ".insn r 0x5B, 0x0, 0x06, {r7}, {t4}, x0",
                t0 = out(reg) _,
                t1 = out(reg) _,
                t2 = out(reg) _,
                t3 = out(reg) _,
                t4 = out(reg) _,
                r3 = inout(reg) r3,
                r4 = inout(reg) r4,
                r5 = inout(reg) r5,
                r6 = inout(reg) r6,
                r7 = out(reg) r7,
                a0 = in(reg) a0,
                a1 = in(reg) a1,
                a2 = in(reg) a2,
                a3 = in(reg) a3,
                bi = in(reg) b[3],
                options(nomem, nostack)
            );
        }
        mul256_checksum(&[r0, r1, r2, r3, r4, r5, r6, r7])
    }
    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    {
        mul256_checksum(&mul256_portable(&a, &b))
    }
}

/// The same 256-bit multiplication in the portable style bigint code uses
/// today (u128 multiply-accumulate rows the compiler lowers to
/// `mul`/`mulhu`/`add`/`sltu` sequences).
#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn mul256_baseline(a: [u64; 4], b: [u64; 4]) -> u64 {
    mul256_checksum(&mul256_portable(&a, &b))
}

/// Control for the trace-length comparison: identical signature and IO,
/// no multiplication work.
#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn mul256_noop(a: [u64; 4], b: [u64; 4]) -> u64 {
    a[0] ^ a[1] ^ a[2] ^ a[3] ^ b[0] ^ b[1] ^ b[2] ^ b[3]
}

/// Textbook schoolbook multiplication with u128 rows.
pub fn mul256_portable(a: &[u64; 4], b: &[u64; 4]) -> [u64; 8] {
    let mut r = [0u64; 8];
    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            let acc = r[i + j] as u128 + (a[j] as u128) * (b[i] as u128) + carry;
            r[i + j] = acc as u64;
            carry = acc >> 64;
        }
        r[i + 4] = carry as u64;
    }
    r
}

/// Folds the eight product limbs so the host can compare a single output.
pub fn mul256_checksum(r: &[u64; 8]) -> u64 {
    r.iter()
        .zip([3u64, 5, 7, 11, 13, 17, 19, 23])
        .fold(0u64, |acc, (limb, weight)| {
            acc.wrapping_add(limb.wrapping_mul(weight))
        })
}

/// Host-side reference of the guest computation, in `u128` arithmetic.
pub fn carry_chain_reference(a: u64, b: u64, c: u64) -> u64 {
    // ADD -> ADDC
    let sum = a as u128 + b as u128;
    let add_lo = sum as u64;
    let carry = sum >> 64;
    let sum_hi = c as u128 + a as u128 + carry;
    let add_hi = sum_hi as u64;

    // MUL -> ADDC
    let product = a as u128 * b as u128;
    let mul_lo = product as u64;
    let mul_hi = (product >> 64) as u64;

    // ADD -> MULC (-> ADDC)
    let sum = a as u128 + b as u128;
    let mac = b as u128 * c as u128 + (sum >> 64);
    let addc_lo = mac as u64;
    let addc_hi = (mac >> 64) as u64;

    // MUL -> MULC
    let product = a as u128 * c as u128;
    let mulc_lo = product as u64;
    let chained = b as u128 * c as u128 + (product >> 64);
    let mulc_hi = chained as u64;

    add_lo
        .wrapping_mul(3)
        .wrapping_add(add_hi.wrapping_mul(5))
        .wrapping_add(mul_lo.wrapping_mul(7))
        .wrapping_add(mul_hi.wrapping_mul(11))
        .wrapping_add(addc_lo.wrapping_mul(13))
        .wrapping_add(addc_hi.wrapping_mul(17))
        .wrapping_add(mulc_lo.wrapping_mul(19))
        .wrapping_add(mulc_hi.wrapping_mul(23))
}
