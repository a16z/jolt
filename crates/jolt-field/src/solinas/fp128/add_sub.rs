//! Architecture-specific Fp128 addition and subtraction kernels.

use super::{join, pack, split, Fp128};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use std::arch::asm;

impl<const P: u128> Fp128<P> {
    #[inline(always)]
    pub(super) fn add_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        #[cfg(target_arch = "aarch64")]
        {
            // Keep the reduction predicate in flags through `ccmp`.
            Self::add_raw_aarch64_dispatch(a, b)
        }

        #[cfg(target_arch = "x86_64")]
        {
            // Materialize the carry as a mask, then use `cmovne` for the
            // final branchless selection.
            Self::add_raw_x86_64_dispatch(a, b)
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::add_raw_portable(a, b)
        }
    }

    #[cfg_attr(
        all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            not(all(test, target_arch = "aarch64"))
        ),
        expect(
            dead_code,
            reason = "target-specific helper is intentionally unused on some architectures"
        )
    )]
    #[inline(always)]
    pub(super) fn add_raw_portable(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        // Compute s = a + b as two limbs.
        let (s0, carry0) = a[0].overflowing_add(b[0]);
        let (s1a, carry1a) = a[1].overflowing_add(b[1]);
        let (s1, carry1b) = s1a.overflowing_add(carry0 as u64);
        let overflow = carry1a | carry1b;

        // Since p = 2^128 - C and C < 2^64, reduction adds C to
        // the low limb and propagates the carry.
        let (r0, carry2) = s0.overflowing_add(Self::C_LO);
        let (r1, carry3) = s1.overflowing_add(carry2 as u64);

        pack(
            if overflow | carry3 { r0 } else { s0 },
            if overflow | carry3 { r1 } else { s1 },
        )
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn add_raw_aarch64_dispatch(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        // AArch64 add immediates are limited to 12 bits. List the built-in
        // small offsets here and use a register for every other offset.
        match Self::C_LO {
            275 => Self::add_raw_aarch64_imm::<275>(a, b),
            159 => Self::add_raw_aarch64_imm::<159>(a, b),
            2355 => Self::add_raw_aarch64_imm::<2355>(a, b),
            0xffff_a7f7 => Self::add_raw_aarch64_a7f7(a, b),
            _ => Self::add_raw_aarch64_reg(a, b, Self::C_LO),
        }
    }

    /// A7F7 addition through the exact instruction body proved in HOL Light.
    ///
    /// Register contract for `fp128_add_body.inc`:
    ///
    /// - `x0:x1` starts with `a` and finishes with the result.
    /// - `x2:x3` contains `b` and is not changed.
    /// - `x4` contains `C = 2^128 - P = 0xffff_a7f7` and is not changed.
    /// - `x5:x9` and the condition flags are temporary state.
    /// - The body does not access memory or the stack.
    ///
    /// The field representation requires canonical inputs. The machine theorem
    /// uses the same assumption. This function adds no runtime range check.
    /// The formal verification workflow checks the exact object and optimized
    /// public operation witness words.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn add_raw_aarch64_a7f7(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let [mut out_lo, mut out_hi] = a;
        let [b_lo, b_hi] = b;
        // SAFETY: The register contract is listed above. The body declares
        // every clobber and does not access memory or the stack.
        unsafe {
            asm!(
                include_str!("../../../asm/aarch64/fp128_add_body.inc"),
                inout("x0") out_lo,
                inout("x1") out_hi,
                in("x2") b_lo,
                in("x3") b_hi,
                in("x4") Self::C_LO,
                out("x5") _,
                out("x6") _,
                out("x7") _,
                out("x8") _,
                out("x9") _,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn add_raw_aarch64_imm<const C: u64>(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        // SAFETY: All inputs, outputs, and temporary registers are declared.
        // The instruction sequence does not access memory or the stack.
        unsafe {
            // `carry1` records overflow from a + b. The carry from s + C says
            // whether an unwrapped sum was at least p. `ccmp` combines them.
            asm!(
                "adds {s_lo}, {a_lo}, {b_lo}",
                "adcs {s_hi}, {a_hi}, {b_hi}",
                "cset {carry1:w}, hs",
                "adds {t_lo}, {s_lo}, #{c}",
                "adcs {t_hi}, {s_hi}, xzr",
                "ccmp {carry1:w}, #0, #0, lo",
                "csel {out_lo}, {t_lo}, {s_lo}, ne",
                "csel {out_hi}, {t_hi}, {s_hi}, ne",
                c = const C,
                a_lo = in(reg) a[0],
                a_hi = in(reg) a[1],
                b_lo = in(reg) b[0],
                b_hi = in(reg) b[1],
                s_lo = out(reg) _,
                s_hi = out(reg) _,
                t_lo = out(reg) _,
                t_hi = out(reg) _,
                carry1 = out(reg) _,
                out_lo = lateout(reg) out_lo,
                out_hi = lateout(reg) out_hi,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn add_raw_aarch64_reg(a: [u64; 2], b: [u64; 2], c: u64) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        // SAFETY: All inputs, outputs, and temporary registers are declared.
        // The instruction sequence does not access memory or the stack.
        unsafe {
            // This is the same flag flow as the immediate path, with C in a
            // register because it is not encodable as an immediate.
            asm!(
                "adds {s_lo}, {a_lo}, {b_lo}",
                "adcs {s_hi}, {a_hi}, {b_hi}",
                "cset {carry1:w}, hs",
                "adds {t_lo}, {s_lo}, {c}",
                "adcs {t_hi}, {s_hi}, xzr",
                "ccmp {carry1:w}, #0, #0, lo",
                "csel {out_lo}, {t_lo}, {s_lo}, ne",
                "csel {out_hi}, {t_hi}, {s_hi}, ne",
                c = in(reg) c,
                a_lo = in(reg) a[0],
                a_hi = in(reg) a[1],
                b_lo = in(reg) b[0],
                b_hi = in(reg) b[1],
                s_lo = out(reg) _,
                s_hi = out(reg) _,
                t_lo = out(reg) _,
                t_hi = out(reg) _,
                carry1 = out(reg) _,
                out_lo = lateout(reg) out_lo,
                out_hi = lateout(reg) out_hi,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn add_raw_x86_64_dispatch(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        // x86-64 sign extends 32 bit immediate operands. A7F7 and any other
        // offset at least 2^31 must use the register form.
        match Self::C_LO {
            275 => Self::add_raw_x86_64_imm::<275>(a, b),
            159 => Self::add_raw_x86_64_imm::<159>(a, b),
            2355 => Self::add_raw_x86_64_imm::<2355>(a, b),
            _ => Self::add_raw_x86_64_reg(a, b, Self::C_LO),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn add_raw_x86_64_imm<const C: i32>(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let mut out_lo = a[0];
        let mut out_hi = a[1];
        // SAFETY: All inputs, outputs, and temporary registers are declared.
        // The instruction sequence does not access memory or the stack.
        unsafe {
            // `sbb mask, mask` turns the first carry into 0 or -1. After
            // t = s + C, `adc mask, mask` sets ZF exactly when no reduction
            // is needed. `cmovne` selects t when either carry was set.
            asm!(
                "add {out_lo}, {b_lo}",
                "adc {out_hi}, {b_hi}",
                "sbb {mask}, {mask}",
                "mov {t_lo}, {out_lo}",
                "mov {t_hi}, {out_hi}",
                "add {t_lo}, {c}",
                "adc {t_hi}, 0",
                "adc {mask}, {mask}",
                "cmovne {out_lo}, {t_lo}",
                "cmovne {out_hi}, {t_hi}",
                out_lo = inout(reg) out_lo,
                out_hi = inout(reg) out_hi,
                b_lo = in(reg) b[0],
                b_hi = in(reg) b[1],
                mask = out(reg) _,
                t_lo = out(reg) _,
                t_hi = out(reg) _,
                c = const C,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn add_raw_x86_64_reg(a: [u64; 2], b: [u64; 2], c: u64) -> [u64; 2] {
        let mut out_lo = a[0];
        let mut out_hi = a[1];
        // SAFETY: All inputs, outputs, and temporary registers are declared.
        // The instruction sequence does not access memory or the stack.
        unsafe {
            asm!(
                "add {out_lo}, {b_lo}",
                "adc {out_hi}, {b_hi}",
                "sbb {mask}, {mask}",
                "mov {t_lo}, {out_lo}",
                "mov {t_hi}, {out_hi}",
                "add {t_lo}, {c}",
                "adc {t_hi}, 0",
                "adc {mask}, {mask}",
                "cmovne {out_lo}, {t_lo}",
                "cmovne {out_hi}, {t_hi}",
                out_lo = inout(reg) out_lo,
                out_hi = inout(reg) out_hi,
                b_lo = in(reg) b[0],
                b_hi = in(reg) b[1],
                c = in(reg) c,
                mask = out(reg) _,
                t_lo = out(reg) _,
                t_hi = out(reg) _,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[inline(always)]
    pub(super) fn sub_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        #[cfg(target_arch = "aarch64")]
        {
            // Keep the borrow in flags and reduce by subtracting C.
            Self::sub_raw_aarch64_dispatch(a, b)
        }

        #[cfg(target_arch = "x86_64")]
        {
            // Turn the borrow into a mask, select 0 or C, and subtract it.
            Self::sub_raw_x86_64_dispatch(a, b)
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::sub_raw_portable(a, b)
        }
    }

    #[cfg_attr(
        all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            not(all(test, target_arch = "aarch64"))
        ),
        expect(
            dead_code,
            reason = "target-specific helper is intentionally unused on some architectures"
        )
    )]
    #[inline(always)]
    pub(super) const fn sub_raw_portable(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let (diff, borrow) = join(a).overflowing_sub(join(b));
        split(if borrow { diff.wrapping_add(P) } else { diff })
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn sub_raw_aarch64_dispatch(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        // List the built-in small immediates and use a register otherwise.
        match Self::C_LO {
            275 => Self::sub_raw_aarch64_imm::<275>(a, b),
            159 => Self::sub_raw_aarch64_imm::<159>(a, b),
            2355 => Self::sub_raw_aarch64_imm::<2355>(a, b),
            0xffff_a7f7 => Self::sub_raw_aarch64_a7f7(a, b),
            _ => Self::sub_raw_aarch64_reg(a, b, Self::C_LO),
        }
    }

    /// A7F7 subtraction through the exact instruction body proved in HOL Light.
    ///
    /// Register contract for `fp128_sub_body.inc`:
    ///
    /// - `x0:x1` starts with `a` and finishes with the result.
    /// - `x2:x3` contains `b` and is not changed.
    /// - `x4` contains `C = 2^128 - P = 0xffff_a7f7` and is not changed.
    /// - `x5:x7` and the condition flags are temporary state.
    /// - The body does not access memory or the stack.
    ///
    /// The field representation requires canonical inputs. The machine theorem
    /// uses the same assumption. This function adds no runtime range check.
    /// The formal verification workflow checks the exact object and optimized
    /// public operation witness words.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn sub_raw_aarch64_a7f7(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let [mut out_lo, mut out_hi] = a;
        let [b_lo, b_hi] = b;
        // SAFETY: The register contract is listed above. The body declares
        // every clobber and does not access memory or the stack.
        unsafe {
            asm!(
                include_str!("../../../asm/aarch64/fp128_sub_body.inc"),
                inout("x0") out_lo,
                inout("x1") out_hi,
                in("x2") b_lo,
                in("x3") b_hi,
                in("x4") Self::C_LO,
                out("x5") _,
                out("x6") _,
                out("x7") _,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn sub_raw_aarch64_imm<const C: u64>(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        // SAFETY: All inputs, outputs, and temporary registers are declared.
        // The instruction sequence does not access memory or the stack.
        unsafe {
            // If a - b borrows, reduction subtracts C from the wrapped result.
            // Select 0 or C directly from the borrow flag.
            asm!(
                "mov {c_tmp}, #{c}",
                "subs {out_lo}, {a_lo}, {b_lo}",
                "sbcs {out_hi}, {a_hi}, {b_hi}",
                "csel {c_tmp}, xzr, {c_tmp}, hs",
                "subs {out_lo}, {out_lo}, {c_tmp}",
                "sbc {out_hi}, {out_hi}, xzr",
                c = const C,
                a_lo = in(reg) a[0],
                a_hi = in(reg) a[1],
                b_lo = in(reg) b[0],
                b_hi = in(reg) b[1],
                c_tmp = out(reg) _,
                out_lo = out(reg) out_lo,
                out_hi = out(reg) out_hi,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn sub_raw_aarch64_reg(a: [u64; 2], b: [u64; 2], c: u64) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        // SAFETY: All inputs, outputs, and temporary registers are declared.
        // The instruction sequence does not access memory or the stack.
        unsafe {
            asm!(
                "subs {out_lo}, {a_lo}, {b_lo}",
                "sbcs {out_hi}, {a_hi}, {b_hi}",
                "csel {c_tmp}, xzr, {c}, hs",
                "subs {out_lo}, {out_lo}, {c_tmp}",
                "sbc {out_hi}, {out_hi}, xzr",
                c = in(reg) c,
                a_lo = in(reg) a[0],
                a_hi = in(reg) a[1],
                b_lo = in(reg) b[0],
                b_hi = in(reg) b[1],
                c_tmp = out(reg) _,
                out_lo = out(reg) out_lo,
                out_hi = out(reg) out_hi,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn sub_raw_x86_64_dispatch(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        // The immediate form is valid for the built-in small offsets. Larger
        // offsets use a register because x86-64 sign extends imm32 operands.
        match Self::C_LO {
            275 => Self::sub_raw_x86_64_imm::<275>(a, b),
            159 => Self::sub_raw_x86_64_imm::<159>(a, b),
            2355 => Self::sub_raw_x86_64_imm::<2355>(a, b),
            _ => Self::sub_raw_x86_64_reg(a, b, Self::C_LO),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn sub_raw_x86_64_imm<const C: i32>(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let mut out_lo = a[0];
        let mut out_hi = a[1];
        // SAFETY: All inputs, outputs, and temporary registers are declared.
        // The instruction sequence does not access memory or the stack.
        unsafe {
            asm!(
                "sub {out_lo}, {b_lo}",
                "sbb {out_hi}, {b_hi}",
                "sbb {mask}, {mask}",
                "and {mask}, {c}",
                "sub {out_lo}, {mask}",
                "sbb {out_hi}, 0",
                out_lo = inout(reg) out_lo,
                out_hi = inout(reg) out_hi,
                b_lo = in(reg) b[0],
                b_hi = in(reg) b[1],
                mask = out(reg) _,
                c = const C,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn sub_raw_x86_64_reg(a: [u64; 2], b: [u64; 2], c: u64) -> [u64; 2] {
        let mut out_lo = a[0];
        let mut out_hi = a[1];
        // SAFETY: All inputs, outputs, and temporary registers are declared.
        // The instruction sequence does not access memory or the stack.
        unsafe {
            asm!(
                "sub {out_lo}, {b_lo}",
                "sbb {out_hi}, {b_hi}",
                "sbb {mask}, {mask}",
                "and {mask}, {c}",
                "sub {out_lo}, {mask}",
                "sbb {out_hi}, 0",
                out_lo = inout(reg) out_lo,
                out_hi = inout(reg) out_hi,
                b_lo = in(reg) b[0],
                b_hi = in(reg) b[1],
                c = in(reg) c,
                mask = out(reg) _,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }
}
