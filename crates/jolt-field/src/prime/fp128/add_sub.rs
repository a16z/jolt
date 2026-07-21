#![cfg_attr(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    expect(
        clippy::undocumented_unsafe_blocks,
        reason = "ported inline-assembly kernels retain their audited flag-flow invariants"
    )
)]

use super::*;

impl<const P: u128> Fp128<P> {
    #[inline(always)]
    pub(super) fn add_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        #[cfg(target_arch = "aarch64")]
        {
            // On AArch64 we can keep the reduction predicate in flags via `ccmp`,
            // which is materially better than the generic `u128` lowering.
            Self::add_raw_aarch64_dispatch(a, b)
        }

        #[cfg(target_arch = "x86_64")]
        {
            // On x86-64, `sbb reg, reg` turns carry1 into a 0/-1 mask without
            // leaving flags. After computing `s + C`, one more `adc mask, mask`
            // makes ZF encode "need reduction", so the final select stays on
            // the flag path via `cmovne`.
            Self::add_raw_x86_64_dispatch(a, b)
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::add_raw_portable(a, b)
        }
    }

    #[cfg_attr(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        expect(
            dead_code,
            reason = "target-specific helper is intentionally unused on some architectures"
        )
    )]
    #[inline(always)]
    fn add_raw_portable(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        // Compute s = a + b as two limbs.
        let (s0, carry0) = a[0].overflowing_add(b[0]);
        let (s1a, carry1a) = a[1].overflowing_add(b[1]);
        let (s1, carry1b) = s1a.overflowing_add(carry0 as u64);
        let overflow = carry1a | carry1b;

        // Since p = 2^128 - C and C < 2^64, reducing s modulo p is just
        // adding C into the low limb and propagating that carry.
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
        // The immediate form is best when C < 4096 (the AArch64 add-immediate
        // encoding limit). Stable Rust does not let us feed `Self::C_LO`
        // directly into an `asm!(..., const ...)` operand, so the known
        // built-in offsets are spelled out here and everything else uses the
        // register form.
        match Self::C_LO {
            275 => Self::add_raw_aarch64_imm::<275>(a, b),
            159 => Self::add_raw_aarch64_imm::<159>(a, b),
            2355 => Self::add_raw_aarch64_imm::<2355>(a, b),
            _ => Self::add_raw_aarch64_reg(a, b, Self::C_LO),
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn add_raw_aarch64_imm<const C: u64>(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        unsafe {
            // carry1 is the overflow bit from a + b.
            // carry2 is the overflow bit from s + C, equivalently s >= p.
            // `ccmp` folds `carry1 | carry2` back into flags so the final
            // select stays branchless and never round-trips through GPR logic.
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
        unsafe {
            // Same flag flow as the immediate path above, but with C supplied in
            // a register for offsets that are not encodable as add immediates.
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
        // As on AArch64, stable Rust does not let us feed `Self::C_LO`
        // directly into a const asm operand. The built-in offsets get the
        // immediate form and everything else uses the register form.
        match Self::C_LO {
            275 => Self::add_raw_x86_64_imm::<275>(a, b),
            159 => Self::add_raw_x86_64_imm::<159>(a, b),
            2355 => Self::add_raw_x86_64_imm::<2355>(a, b),
            // For C >= 2^31 the i32 immediate form is unusable: `add r64,
            // imm32` sign-extends the immediate, which would silently
            // corrupt the high limb. Such offsets fall through to the
            // register form below (`Prime128OffsetA7F7` lands here).
            _ => Self::add_raw_x86_64_reg(a, b, Self::C_LO),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn add_raw_x86_64_imm<const C: i32>(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let mut out_lo = a[0];
        let mut out_hi = a[1];
        let _mask: u64;
        let _t_lo: u64;
        let _t_hi: u64;
        unsafe {
            // After `s = a + b`, `sbb mask, mask` materializes carry1 as 0/-1.
            // After `t = s + C`, `adc mask, mask` leaves ZF=1 iff neither
            // carry1 nor carry2 was set. `cmovne` then picks `t` exactly when
            // reduction is needed.
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
                mask = out(reg) _mask,
                t_lo = out(reg) _t_lo,
                t_hi = out(reg) _t_hi,
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
        let _mask: u64;
        let _t_lo: u64;
        let _t_hi: u64;
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
                mask = out(reg) _mask,
                t_lo = out(reg) _t_lo,
                t_hi = out(reg) _t_hi,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[inline(always)]
    pub(super) fn sub_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        #[cfg(target_arch = "aarch64")]
        {
            // The const path still uses `sub_raw_portable`, but at runtime on
            // AArch64 we can keep subtraction in limbs and reduce with `-C`
            // instead of materializing `P = 2^128 - C`.
            Self::sub_raw_aarch64_dispatch(a, b)
        }

        #[cfg(target_arch = "x86_64")]
        {
            // On x86-64, `sbb reg, reg` turns the final borrow into a 0/-1 mask.
            // Masking that with C lets us keep the same "select 0 or C, then do
            // one final subtract" structure that worked well on AArch64.
            Self::sub_raw_x86_64_dispatch(a, b)
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::sub_raw_portable(a, b)
        }
    }

    #[inline(always)]
    pub(super) const fn sub_raw_portable(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let (diff, borrow) = to_u128(a).overflowing_sub(to_u128(b));
        from_u128(if borrow { diff.wrapping_add(P) } else { diff })
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn sub_raw_aarch64_dispatch(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        // As in add_raw, stable Rust cannot feed `Self::C_LO` directly into a
        // `const` asm operand, so the built-in offsets get immediate forms and
        // everything else falls back to the register form.
        match Self::C_LO {
            275 => Self::sub_raw_aarch64_imm::<275>(a, b),
            159 => Self::sub_raw_aarch64_imm::<159>(a, b),
            2355 => Self::sub_raw_aarch64_imm::<2355>(a, b),
            _ => Self::sub_raw_aarch64_reg(a, b, Self::C_LO),
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn sub_raw_aarch64_imm<const C: u64>(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        unsafe {
            // If `a - b` borrows, then modulo `p = 2^128 - C` we need
            // `diff + p = diff - C (mod 2^128)`. Instead of round-tripping the
            // borrow bit through a GPR with `cset`/`cmp`, select the subtrahend
            // (`0` or `C`) directly from flags and do one final subtract.
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
        // The immediate form keeps C out of the input register set for the
        // built-in offsets. Stable Rust does not let us pass `Self::C_LO`
        // directly as a const asm operand, so the known built-ins are spelled
        // out here and everything else uses the register form.
        match Self::C_LO {
            275 => Self::sub_raw_x86_64_imm::<275>(a, b),
            159 => Self::sub_raw_x86_64_imm::<159>(a, b),
            2355 => Self::sub_raw_x86_64_imm::<2355>(a, b),
            // See the matching note in `add_raw_x86_64_dispatch`: offsets
            // with C >= 2^31 cannot use the i32 immediate form because the
            // sign-extended `and r64, imm32` would corrupt the mask, so
            // they fall through to the register path here.
            _ => Self::sub_raw_x86_64_reg(a, b, Self::C_LO),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn sub_raw_x86_64_imm<const C: i32>(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let mut out_lo = a[0];
        let mut out_hi = a[1];
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
