#![cfg_attr(
    target_arch = "aarch64",
    expect(
        clippy::undocumented_unsafe_blocks,
        reason = "ported inline-assembly kernels retain their audited carry-flow invariants"
    )
)]

use super::*;

impl<const P: u128> Fp128<P> {
    #[inline(always)]
    pub(super) fn mul_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        #[cfg(target_arch = "aarch64")]
        {
            Self::mul_raw_aarch64(a, b)
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            Self::mul_raw_portable(a, b)
        }
    }

    #[cfg_attr(
        target_arch = "aarch64",
        expect(
            dead_code,
            reason = "target-specific helper is intentionally unused on some architectures"
        )
    )]
    #[inline(always)]
    fn mul_raw_portable(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let [r0, r1, r2, r3] = Self(a).mul_wide(Self(b));
        Self::reduce_4(r0, r1, r2, r3)
    }

    #[inline(always)]
    fn mul_add_raw(a: [u64; 2], b: [u64; 2], addend: [u64; 2]) -> [u64; 2] {
        #[cfg(target_arch = "aarch64")]
        {
            Self::mul_add_raw_aarch64(a, b, addend)
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            Self::mul_add_raw_portable(a, b, addend)
        }
    }

    #[cfg_attr(
        target_arch = "aarch64",
        expect(
            dead_code,
            reason = "target-specific helper is intentionally unused on some architectures"
        )
    )]
    #[inline(always)]
    fn mul_add_raw_portable(a: [u64; 2], b: [u64; 2], addend: [u64; 2]) -> [u64; 2] {
        let prod = Self(a).mul_wide(Self(b));
        let [s0, s1, s2, s3] = Self::add_128_into_256(prod, addend);
        Self::reduce_4(s0, s1, s2, s3)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn mul_add_raw_aarch64(a: [u64; 2], b: [u64; 2], addend: [u64; 2]) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        unsafe {
            asm!(
                // Schoolbook 2×2 → 256-bit product [r0,r1,r2,r3]
                "mul     {p00l}, {a0}, {b0}",
                "umulh   {p00h}, {a0}, {b0}",
                "mul     {p01l}, {a0}, {b1}",
                "umulh   {p01h}, {a0}, {b1}",
                "mul     {p10l}, {a1}, {b0}",
                "umulh   {p10h}, {a1}, {b0}",
                "mul     {p11l}, {a1}, {b1}",
                "umulh   {p11h}, {a1}, {b1}",

                // Carry accumulation into [r0=p00l, r1=p00h, r2=p01h, r3=p11h]
                "adds   {p00h}, {p00h}, {p01l}",
                "cset   {p01l:w}, hs",
                "adds   {p01h}, {p01h}, {p10h}",
                "cset   {p10h:w}, hs",
                "adds   {p01h}, {p01h}, {p11l}",
                "cinc   {p10h}, {p10h}, hs",
                "adds   {p00h}, {p00h}, {p10l}",
                "adcs   {p01h}, {p01h}, {p01l}",
                "adc    {p11h}, {p11h}, {p10h}",

                // Fuse the addend into the low 128 bits before the Solinas fold.
                "adds   {p00l}, {p00l}, {add_lo}",
                "adcs   {p00h}, {p00h}, {add_hi}",
                "adcs   {p01h}, {p01h}, xzr",
                "adc    {p11h}, {p11h}, xzr",

                // Fold-1: [t0,t1,t2] = [r0,r1] + C·[r2,r3]
                "mul    {p01l}, {p01h}, {c}",
                "umulh  {p10l}, {p01h}, {c}",
                "mul    {p10h}, {p11h}, {c}",
                "umulh  {p11l}, {p11h}, {c}",

                "adds   {p00l}, {p00l}, {p01l}",
                "adcs   {p00h}, {p00h}, {p10l}",
                "cset   {p01h:w}, hs",
                "adds   {p00h}, {p00h}, {p10h}",
                "adc    {p11h}, {p11l}, {p01h}",

                // Fold-2 + canonicalize via ccmp
                "mul    {p01l}, {p11h}, {c}",
                "adds   {p00l}, {p00l}, {p01l}",
                "adcs   {p00h}, {p00h}, xzr",
                "cset   {p01l:w}, hs",
                "adds   {p10l}, {p00l}, {c}",
                "adcs   {p10h}, {p00h}, xzr",
                "ccmp   {p01l:w}, #0, #0, lo",
                "csel   {out_lo}, {p10l}, {p00l}, ne",
                "csel   {out_hi}, {p10h}, {p00h}, ne",

                a0 = in(reg) a[0],
                a1 = in(reg) a[1],
                b0 = in(reg) b[0],
                b1 = in(reg) b[1],
                add_lo = in(reg) addend[0],
                add_hi = in(reg) addend[1],
                c = in(reg) Self::C_LO,
                p00l = out(reg) _,
                p00h = out(reg) _,
                p01l = out(reg) _,
                p01h = out(reg) _,
                p10l = out(reg) _,
                p10h = out(reg) _,
                p11l = out(reg) _,
                p11h = out(reg) _,
                out_lo = lateout(reg) out_lo,
                out_hi = lateout(reg) out_hi,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    /// 35-instruction AArch64 inline-asm multiply with Solinas reduction.
    ///
    /// Saves 6 instructions vs LLVM's codegen by:
    ///   - Fold-1 carry chain: direct adds/adcs/adc (5 vs 8 instructions),
    ///     avoiding intermediate cset/cinc shuttling of carries.
    ///   - Fold-2 + canonicalize: `ccmp` folds the overflow predicate with
    ///     the ≥p check (8 vs 10 instructions).
    ///
    /// Benchmarked at 1.29x throughput improvement on Apple M4.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn mul_raw_aarch64(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        unsafe {
            asm!(
                // Schoolbook 2×2 → 256-bit product [r0,r1,r2,r3]
                "mul     {p00l}, {a0}, {b0}",
                "umulh   {p00h}, {a0}, {b0}",
                "mul     {p01l}, {a0}, {b1}",
                "umulh   {p01h}, {a0}, {b1}",
                "mul     {p10l}, {a1}, {b0}",
                "umulh   {p10h}, {a1}, {b0}",
                "mul     {p11l}, {a1}, {b1}",
                "umulh   {p11h}, {a1}, {b1}",

                // Carry accumulation into [r0=p00l, r1=p00h, r2=p01h, r3=p11h]
                "adds   {p00h}, {p00h}, {p01l}",
                "cset   {p01l:w}, hs",
                "adds   {p01h}, {p01h}, {p10h}",
                "cset   {p10h:w}, hs",
                "adds   {p01h}, {p01h}, {p11l}",
                "cinc   {p10h}, {p10h}, hs",
                "adds   {p00h}, {p00h}, {p10l}",
                "adcs   {p01h}, {p01h}, {p01l}",
                "adc    {p11h}, {p11h}, {p10h}",

                // Fold-1: [t0,t1,t2] = [r0,r1] + C·[r2,r3]
                "mul    {p01l}, {p01h}, {c}",
                "umulh  {p10l}, {p01h}, {c}",
                "mul    {p10h}, {p11h}, {c}",
                "umulh  {p11l}, {p11h}, {c}",

                "adds   {p00l}, {p00l}, {p01l}",
                "adcs   {p00h}, {p00h}, {p10l}",
                "cset   {p01h:w}, hs",
                "adds   {p00h}, {p00h}, {p10h}",
                "adc    {p11h}, {p11l}, {p01h}",

                // Fold-2 + canonicalize via ccmp (C < 2^32 ⇒ C·t2 fits in 64 bits)
                "mul    {p01l}, {p11h}, {c}",
                "adds   {p00l}, {p00l}, {p01l}",
                "adcs   {p00h}, {p00h}, xzr",
                "cset   {p01l:w}, hs",
                "adds   {p10l}, {p00l}, {c}",
                "adcs   {p10h}, {p00h}, xzr",
                "ccmp   {p01l:w}, #0, #0, lo",
                "csel   {out_lo}, {p10l}, {p00l}, ne",
                "csel   {out_hi}, {p10h}, {p00h}, ne",

                a0 = in(reg) a[0],
                a1 = in(reg) a[1],
                b0 = in(reg) b[0],
                b1 = in(reg) b[1],
                c = in(reg) Self::C_LO,
                p00l = out(reg) _,
                p00h = out(reg) _,
                p01l = out(reg) _,
                p01h = out(reg) _,
                p10l = out(reg) _,
                p10h = out(reg) _,
                p11l = out(reg) _,
                p11h = out(reg) _,
                out_lo = lateout(reg) out_lo,
                out_hi = lateout(reg) out_hi,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[inline(always)]
    fn sqr_wide(self) -> [u64; 4] {
        let (a0, a1) = (self.0[0], self.0[1]);
        let (p00_lo, p00_hi) = mul64_wide(a0, a0);
        let (p01_lo, p01_hi) = mul64_wide(a0, a1);
        let (p11_lo, p11_hi) = mul64_wide(a1, a1);

        let row1 = p00_hi as u128 + (p01_lo as u128) * 2;
        let r0 = p00_lo;
        let r1 = row1 as u64;
        let carry1 = (row1 >> 64) as u64;

        let row2 = (p01_hi as u128) * 2 + p11_lo as u128 + carry1 as u128;
        let r2 = row2 as u64;
        let carry2 = (row2 >> 64) as u64;

        let row3 = p11_hi as u128 + carry2 as u128;
        let r3 = row3 as u64;
        debug_assert_eq!(row3 >> 64, 0);

        [r0, r1, r2, r3]
    }

    #[inline(always)]
    fn sqr_raw(a: [u64; 2]) -> [u64; 2] {
        #[cfg(target_arch = "aarch64")]
        {
            Self::sqr_raw_aarch64(a)
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            Self::sqr_raw_portable(a)
        }
    }

    #[cfg_attr(
        target_arch = "aarch64",
        expect(
            dead_code,
            reason = "target-specific helper is intentionally unused on some architectures"
        )
    )]
    #[inline(always)]
    fn sqr_raw_portable(a: [u64; 2]) -> [u64; 2] {
        let [r0, r1, r2, r3] = Self(a).sqr_wide();
        Self::reduce_4(r0, r1, r2, r3)
    }

    /// 31-instruction AArch64 inline-asm squaring with Solinas reduction.
    ///
    /// Uses 3 widening multiplies (vs 4 for general mul) and doubles the
    /// cross term via shifted-register operands. Same fold-1 + ccmp
    /// canonicalize as `mul_raw_aarch64`.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn sqr_raw_aarch64(a: [u64; 2]) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        unsafe {
            asm!(
                // Squaring schoolbook: 3 widening muls
                "mul     {p00l}, {a0}, {a0}",
                "umulh   {p00h}, {a0}, {a0}",
                "mul     {p01l}, {a0}, {a1}",
                "umulh   {p01h}, {a0}, {a1}",
                "mul     {p11l}, {a1}, {a1}",
                "umulh   {p11h}, {a1}, {a1}",

                // Carry accumulation with doubled cross term
                // row1 = p00h + 2*p01l, row2 = 2*p01h + p11l, r3 = p11h + carries
                "lsr    {t0}, {p01l}, #63",
                "lsr    {t1}, {p01h}, #63",
                "adds   {p01h}, {p11l}, {p01h}, lsl #1",
                "cinc   {t1}, {t1}, hs",
                "adds   {p00h}, {p00h}, {p01l}, lsl #1",
                "adcs   {p01h}, {p01h}, {t0}",
                "adc    {p11h}, {p11h}, {t1}",

                // At this point: r0=p00l, r1=p00h, r2=p01h, r3=p11h

                // Fold-1: [t0,t1,t2] = [r0,r1] + C·[r2,r3]
                "mul    {t0}, {p01h}, {c}",
                "umulh  {t1}, {p01h}, {c}",
                "mul    {p01l}, {p11h}, {c}",
                "umulh  {p11l}, {p11h}, {c}",

                "adds   {p00l}, {p00l}, {t0}",
                "adcs   {p00h}, {p00h}, {t1}",
                "cset   {t0:w}, hs",
                "adds   {p00h}, {p00h}, {p01l}",
                "adc    {p11h}, {p11l}, {t0}",

                // Fold-2 + canonicalize via ccmp (C < 2^32 ⇒ C·t2 fits in 64 bits)
                "mul    {t0}, {p11h}, {c}",
                "adds   {p00l}, {p00l}, {t0}",
                "adcs   {p00h}, {p00h}, xzr",
                "cset   {t0:w}, hs",
                "adds   {t1}, {p00l}, {c}",
                "adcs   {p01l}, {p00h}, xzr",
                "ccmp   {t0:w}, #0, #0, lo",
                "csel   {out_lo}, {t1}, {p00l}, ne",
                "csel   {out_hi}, {p01l}, {p00h}, ne",

                a0 = in(reg) a[0],
                a1 = in(reg) a[1],
                c = in(reg) Self::C_LO,
                p00l = out(reg) _,
                p00h = out(reg) _,
                p01l = out(reg) _,
                p01h = out(reg) _,
                p11l = out(reg) _,
                p11h = out(reg) _,
                t0 = out(reg) _,
                t1 = out(reg) _,
                out_lo = lateout(reg) out_lo,
                out_hi = lateout(reg) out_hi,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    /// Squaring, equivalent to `self * self`.
    #[inline(always)]
    pub fn square(self) -> Self {
        Self(Self::sqr_raw(self.0))
    }

    /// Fused multiply-add, equivalent to `self * rhs + addend`.
    ///
    /// This widens the product, adds the canonical addend before reduction,
    /// and performs a single final Solinas reduction.
    #[inline(always)]
    pub fn mul_add(self, rhs: Self, addend: Self) -> Self {
        Self(Self::mul_add_raw(self.0, rhs.0, addend.0))
    }

    pub(super) fn pow_u128(self, mut exp: u128) -> Self {
        let mut base = self;
        let mut acc = Self::one();
        while exp > 0 {
            if (exp & 1) == 1 {
                acc *= base;
            }
            base = Self(Self::sqr_raw(base.0));
            exp >>= 1;
        }
        acc
    }
}
