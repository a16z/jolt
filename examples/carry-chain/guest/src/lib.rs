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
