//! Two-limb pseudo-Mersenne prime field: `p = 2^128 − C` with `C < 2^32`,
//! stored as `[u64; 2]` little-endian limbs ([`Fp128`]).
//!
//! Solinas-style two-fold reduction, no Montgomery form: a 256-bit product
//! is folded twice through `2^128 ≡ C (mod p)` and canonicalized with one
//! conditional add. `C < 2^32` keeps every fold term inside two limbs and is
//! const-asserted in exactly one place ([`Fp128::C`]), together with
//! `C(C+1) < P` (implied by `C < 2^32`, kept as belt-and-suspenders).
//!
//! Unlike the word-sized fields, the modulus is NOT primality-checked at
//! compile time (2^128 trial division is not CTFE-viable); instantiating a
//! composite odd modulus yields a ring whose `inverse` is meaningless.
//!
//! With the `asm` feature, every valid offset uses architecture-specific add,
//! subtract, and multiply kernels. AArch64 also uses assembly for fused
//! multiply-add and squaring. AArch64 multiplication was benchmarked at 1.29x
//! the portable throughput on Apple M4. Without `asm`, all Fp128 operations
//! use portable Rust.

use super::word::mul64_wide;
use crate::PseudoMersenne;
use crate::{CanonicalBytes, CanonicalEncoding, Field, Ring, WithAccumulator};
use rand_core::RngCore;
#[cfg(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64")))]
use std::arch::asm;
#[cfg(all(feature = "fuzzing", target_arch = "x86_64"))]
use std::sync::atomic::{AtomicU8, Ordering};

mod add_sub;

#[cfg(any(
    all(feature = "asm", target_arch = "x86_64"),
    all(test, feature = "asm", target_arch = "aarch64")
))]
const A7F7_OFFSET: u64 = 0xffff_a7f7;
#[cfg(all(feature = "fuzzing", target_arch = "x86_64"))]
const X86_64_BASELINE_BACKEND: u8 = 1;
#[cfg(all(
    feature = "fuzzing",
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
const X86_64_BMI2_ADX_BACKEND: u8 = 2;
#[cfg(all(feature = "fuzzing", target_arch = "x86_64"))]
static LAST_X86_64_MUL_BACKEND: AtomicU8 = AtomicU8::new(0);

/// Pack two `u64` limbs into little-endian `[lo, hi]`.
#[inline(always)]
const fn pack(lo: u64, hi: u64) -> [u64; 2] {
    [lo, hi]
}

/// Split a `u128` into little-endian `[u64; 2]` limbs.
#[inline(always)]
const fn split(x: u128) -> [u64; 2] {
    [x as u64, (x >> 64) as u64]
}

/// Join little-endian `[u64; 2]` limbs into a `u128`.
#[inline(always)]
const fn join(x: [u64; 2]) -> u128 {
    x[0] as u128 | (x[1] as u128) << 64
}

/// 128-bit prime field element for primes of the form `p = 2^128 − c`,
/// stored as `[u64; 2]` little-endian limbs holding the canonical
/// representative in `[0, p)`.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Fp128<const P: u128>(pub(crate) [u64; 2]);

impl<const P: u128> Fp128<P> {
    /// Offset `c = 2^128 − P`. Instantiating with a modulus that violates
    /// the Solinas preconditions is a compile-time error.
    pub const C: u128 = {
        let c = 0u128.wrapping_sub(P);
        assert!(P != 0, "modulus must be nonzero");
        assert!(P & 1 == 1, "modulus must be odd");
        assert!(c < (1 << 32), "C must be < 2^32 (two-limb fold terms)");
        assert!(
            c * (c + 1) < P,
            "C(C+1) < P required for fused canonicalize"
        );
        c
    };

    /// Low 64 bits of `C` (always equals `C` since `C < 2^32`).
    pub const C_LO: u64 = Self::C as u64;

    /// `+1` means `C = 2^a + 1`, `-1` means `C = 2^a − 1`, `0` means generic.
    /// (`C < 2^32` is const-asserted, so `C + 1` cannot overflow and the
    /// shift is at most 32.)
    const C_SHIFT_KIND: i8 = {
        let c = Self::C_LO;
        if c > 1 && (c - 1).is_power_of_two() {
            1
        } else if (c + 1).is_power_of_two() {
            -1
        } else {
            0
        }
    };
    const C_SHIFT: u32 = if Self::C_SHIFT_KIND == 1 {
        (Self::C_LO - 1).trailing_zeros()
    } else if Self::C_SHIFT_KIND == -1 {
        (Self::C_LO + 1).trailing_zeros()
    } else {
        0
    };

    /// Widening multiply by `C`: returns `C·x` as `(lo, hi)`.
    ///
    /// For `C = 2^a ± 1` this is shift/add or shift/sub only; otherwise a
    /// generic widening multiply. Bound: `C·x < 2^32 · 2^64 = 2^96`, so
    /// `hi ≤ C − 1 < 2^32`.
    #[inline(always)]
    fn mul_c_wide(x: u64) -> (u64, u64) {
        if Self::C_SHIFT_KIND == 1 {
            let v = ((x as u128) << Self::C_SHIFT) + x as u128;
            (v as u64, (v >> 64) as u64)
        } else if Self::C_SHIFT_KIND == -1 {
            let v = ((x as u128) << Self::C_SHIFT) - x as u128;
            (v as u64, (v >> 64) as u64)
        } else {
            mul64_wide(Self::C_LO, x)
        }
    }

    /// Fold 2 + canonicalize: reduce `[t0, t1] + t2·2^128` into `[0, p)`.
    ///
    /// Valid for ANY `u64` `t2` given `C < 2^32`. Let `v = t + C·t2`
    /// (mathematical, not mod 2^128) with `t = [t0, t1] < 2^128`:
    ///
    /// - `v < 2^128 + C·(2^64 − 1) < 2^128 + 2^96 < 2^129`, so the two-limb
    ///   add wraps at most once (`overflow` is a single bit).
    /// - **No overflow** (`v < 2^128`): `s = v` and the standard
    ///   canonicalize applies — `s + C` carries (`carry3`) iff `s ≥ p`,
    ///   in which case the wrapped `s + C` equals `s − p < C < p` (since
    ///   `s < 2^128 = p + C`).
    /// - **Overflow** (`v ≥ 2^128`): `s = v − 2^128 < C·t2`, and the correct
    ///   residue is `s + C` (because `2^128 ≡ C mod p`). Since
    ///   `s + C < C·(t2 + 1) ≤ C·2^64 < 2^96 < p`, that value is already
    ///   canonical and the add does not carry.
    ///
    /// Hence `if overflow | carry3 { s + C } else { s }` is correct in both
    /// cases, fusing the overflow correction with canonicalization.
    #[inline(always)]
    fn fold2_canonicalize(t0: u64, t1: u64, t2: u64) -> [u64; 2] {
        let (ct2_lo, ct2_hi) = Self::mul_c_wide(t2);

        let (s0, carry0) = t0.overflowing_add(ct2_lo);
        let (s1a, carry1a) = t1.overflowing_add(ct2_hi);
        let (s1, carry1b) = s1a.overflowing_add(carry0 as u64);
        let overflow = carry1a | carry1b;

        let (r0, carry2) = s0.overflowing_add(Self::C_LO);
        let (r1, carry3) = s1.overflowing_add(carry2 as u64);

        pack(
            if overflow | carry3 { r0 } else { s0 },
            if overflow | carry3 { r1 } else { s1 },
        )
    }

    /// Fold 1 for exactly 4 limbs: `[r0,r1] + C·[r2,r3]` → 3 limbs `[t0,t1,t2]`,
    /// then [`fold2_canonicalize`](Self::fold2_canonicalize).
    ///
    /// Bounds (any 256-bit input): `t = lo128 + C·hi128 ≤ (C+1)(2^128 − 1)
    /// < (C+1)·2^128`, so `t2 = t >> 128 ≤ C`. Per limb: `t0_sum ≤
    /// 2(2^64 − 1)` (carry ≤ 1); `t1_sum ≤ 3(2^64 − 1) + 1 < 2^66` (its high
    /// part ≤ 2, kept via the full `u128` shift — never truncated);
    /// `t2_sum ≤ (C − 1) + 2 < 2^64` (no fourth limb, debug-asserted).
    #[inline(always)]
    fn reduce_4(r0: u64, r1: u64, r2: u64, r3: u64) -> [u64; 2] {
        let (cr2_lo, cr2_hi) = Self::mul_c_wide(r2);
        let (cr3_lo, cr3_hi) = Self::mul_c_wide(r3);

        let t0_sum = r0 as u128 + cr2_lo as u128;
        let t0 = t0_sum as u64;
        let carryf = (t0_sum >> 64) as u64;

        let t1_sum = r1 as u128 + cr2_hi as u128 + cr3_lo as u128 + carryf as u128;
        let t1 = t1_sum as u64;

        let t2_sum = cr3_hi as u128 + (t1_sum >> 64);
        let t2 = t2_sum as u64;
        debug_assert_eq!(t2_sum >> 64, 0);

        Self::fold2_canonicalize(t0, t1, t2)
    }

    /// Adds a canonical 128-bit value to a 256-bit product.
    #[cfg(any(
        test,
        feature = "fuzzing",
        not(all(feature = "asm", target_arch = "aarch64"))
    ))]
    #[inline(always)]
    fn add_128_into_256(prod: [u64; 4], addend: [u64; 2]) -> [u64; 4] {
        let (s0, carry0) = prod[0].overflowing_add(addend[0]);
        let (s1a, carry1a) = prod[1].overflowing_add(addend[1]);
        let (s1, carry1b) = s1a.overflowing_add(carry0 as u64);
        let (s2, carry2) = prod[2].overflowing_add((carry1a | carry1b) as u64);
        let (s3, carry3) = prod[3].overflowing_add(carry2 as u64);
        debug_assert!(!carry3);
        [s0, s1, s2, s3]
    }

    #[inline(always)]
    fn mul_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        #[cfg(all(feature = "asm", target_arch = "aarch64"))]
        {
            Self::mul_raw_aarch64_dispatch(a, b)
        }
        #[cfg(all(feature = "asm", target_arch = "x86_64"))]
        {
            Self::mul_raw_x86_64_dispatch(a, b)
        }
        #[cfg(not(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64"))))]
        {
            Self::mul_raw_portable(a, b)
        }
    }

    #[inline(always)]
    fn mul_add_raw(a: [u64; 2], b: [u64; 2], addend: [u64; 2]) -> [u64; 2] {
        #[cfg(all(feature = "asm", target_arch = "aarch64"))]
        {
            Self::mul_add_raw_aarch64(a, b, addend)
        }
        #[cfg(not(all(feature = "asm", target_arch = "aarch64")))]
        {
            Self::mul_add_raw_portable(a, b, addend)
        }
    }

    #[cfg(any(
        test,
        feature = "fuzzing",
        not(all(feature = "asm", target_arch = "aarch64"))
    ))]
    #[inline(always)]
    fn mul_add_raw_portable(a: [u64; 2], b: [u64; 2], addend: [u64; 2]) -> [u64; 2] {
        let product = Self(a).mul_wide(Self(b));
        let [s0, s1, s2, s3] = Self::add_128_into_256(product, addend);
        Self::reduce_4(s0, s1, s2, s3)
    }

    #[cfg(all(feature = "asm", target_arch = "aarch64"))]
    #[inline(always)]
    fn mul_add_raw_aarch64(a: [u64; 2], b: [u64; 2], addend: [u64; 2]) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        // SAFETY: this register-only assembly implements the same widening
        // product, carry-chain add, and Solinas reduction as
        // `mul_add_raw_portable`. It does not access memory or the stack.
        unsafe {
            asm!(
                "mul     {p00l}, {a0}, {b0}",
                "umulh   {p00h}, {a0}, {b0}",
                "mul     {p01l}, {a0}, {b1}",
                "umulh   {p01h}, {a0}, {b1}",
                "mul     {p10l}, {a1}, {b0}",
                "umulh   {p10h}, {a1}, {b0}",
                "mul     {p11l}, {a1}, {b1}",
                "umulh   {p11h}, {a1}, {b1}",
                "adds   {p00h}, {p00h}, {p01l}",
                "cset   {p01l:w}, hs",
                "adds   {p01h}, {p01h}, {p10h}",
                "cset   {p10h:w}, hs",
                "adds   {p01h}, {p01h}, {p11l}",
                "cinc   {p10h}, {p10h}, hs",
                "adds   {p00h}, {p00h}, {p10l}",
                "adcs   {p01h}, {p01h}, {p01l}",
                "adc    {p11h}, {p11h}, {p10h}",
                "adds   {p00l}, {p00l}, {add_lo}",
                "adcs   {p00h}, {p00h}, {add_hi}",
                "adcs   {p01h}, {p01h}, xzr",
                "adc    {p11h}, {p11h}, xzr",
                "mul    {p01l}, {p01h}, {c}",
                "umulh  {p10l}, {p01h}, {c}",
                "mul    {p10h}, {p11h}, {c}",
                "umulh  {p11l}, {p11h}, {c}",
                "adds   {p00l}, {p00l}, {p01l}",
                "adcs   {p00h}, {p00h}, {p10l}",
                "cset   {p01h:w}, hs",
                "adds   {p00h}, {p00h}, {p10h}",
                "adc    {p11h}, {p11l}, {p01h}",
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

    /// Portable multiply: schoolbook 2×2 widening product, then the two
    /// Solinas folds. Assembly builds retain it for tests and fuzzing as the
    /// differential oracle for the architecture kernel.
    #[cfg(any(
        test,
        feature = "fuzzing",
        not(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64")))
    ))]
    #[inline(always)]
    fn mul_raw_portable(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let [r0, r1, r2, r3] = Self(a).mul_wide(Self(b));
        Self::reduce_4(r0, r1, r2, r3)
    }

    /// x86-64 multiplication dispatch. Builds that enable both BMI2 and ADX
    /// use the matching A7F7 specialization. Every other case uses the
    /// parameterized baseline assembly sequence.
    #[cfg(all(feature = "asm", target_arch = "x86_64"))]
    #[inline(always)]
    fn mul_raw_x86_64_dispatch(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        if Self::C_LO == A7F7_OFFSET {
            #[cfg(all(target_feature = "bmi2", target_feature = "adx"))]
            {
                Self::mul_raw_x86_64_a7f7_bmi2_adx(a, b)
            }
            #[cfg(not(all(target_feature = "bmi2", target_feature = "adx")))]
            {
                Self::mul_raw_x86_64_baseline(a, b)
            }
        } else {
            Self::mul_raw_x86_64_baseline(a, b)
        }
    }

    /// A7F7 multiplication for x86-64 builds with BMI2 and ADX enabled.
    ///
    /// `rdi:rsi` starts with `a`, and `rdx:rcx` starts with `b`. The body
    /// finishes directly in the System V result registers `rax:rdx`. It uses
    /// only caller saved registers, flags, and no memory or stack.
    #[cfg(all(
        feature = "asm",
        target_arch = "x86_64",
        target_feature = "bmi2",
        target_feature = "adx"
    ))]
    #[inline(always)]
    fn mul_raw_x86_64_a7f7_bmi2_adx(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        #[cfg(feature = "fuzzing")]
        LAST_X86_64_MUL_BACKEND.store(X86_64_BMI2_ADX_BACKEND, Ordering::Relaxed);
        let [a_lo, a_hi] = a;
        let [b_lo, b_hi] = b;
        let out_lo: u64;
        let out_hi: u64;
        // SAFETY: The compile-time target-feature gate guarantees instruction
        // availability. Every changed register is declared, and the body
        // accesses neither memory nor stack.
        unsafe {
            asm!(
                include_str!("../../asm/x86_64/fp128_mul_bmi2_adx_body.inc"),
                inout("rdi") a_lo => _,
                inout("rsi") a_hi => _,
                inout("rdx") b_lo => out_hi,
                inout("rcx") b_hi => _,
                lateout("rax") out_lo,
                out("r8") _,
                out("r9") _,
                out("r10") _,
                out("r11") _,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    /// Parameterized multiplication through the baseline x86-64 instruction
    /// body proved generically over every valid offset in HOL Light.
    ///
    /// `rdi:rsi` starts with `a`, and `rdx:rcx` starts with `b`. The result
    /// finishes in `rdi:rcx`, which Rust binds to the two returned limbs. `r8`
    /// contains `C = 2^128 - P`. The body uses `rax`, `rdx`, `r9`, `r10`,
    /// `r11`, and the flags as temporary state. It uses no memory or stack and
    /// requires no optional x86 target features.
    #[cfg(all(feature = "asm", target_arch = "x86_64"))]
    #[inline(always)]
    fn mul_raw_x86_64_baseline(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        #[cfg(feature = "fuzzing")]
        LAST_X86_64_MUL_BACKEND.store(X86_64_BASELINE_BACKEND, Ordering::Relaxed);
        let [mut out_lo, a_hi] = a;
        let [b_lo, mut out_hi] = b;
        // SAFETY: The register contract is listed above. Every changed
        // register is declared, and the body accesses neither memory nor stack.
        unsafe {
            asm!(
                include_str!("../../asm/x86_64/fp128_mul_body.inc"),
                inout("rdi") out_lo,
                in("rsi") a_hi,
                inout("rdx") b_lo => _,
                inout("rcx") out_hi,
                in("r8") Self::C_LO,
                out("rax") _,
                out("r9") _,
                out("r10") _,
                out("r11") _,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    /// 35-instruction AArch64 inline-asm multiply with Solinas reduction.
    ///
    /// Saves 6 instructions vs LLVM's codegen of the portable path by:
    ///   - Fold-1 carry chain: direct adds/adcs/adc (5 vs 8 instructions),
    ///     avoiding intermediate cset/cinc shuttling of carries.
    ///   - Fold-2 + canonicalize: `ccmp` folds the overflow predicate with
    ///     the ≥p check (8 vs 10 instructions).
    ///
    /// Benchmarked at 1.29x throughput improvement on Apple M4.
    #[cfg(all(feature = "asm", target_arch = "aarch64"))]
    #[inline(always)]
    fn mul_raw_aarch64_dispatch(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        Self::mul_raw_aarch64_asm(a, b)
    }

    /// Parameterized multiplication through the instruction body proved
    /// generically over every valid offset in HOL Light.
    ///
    /// `x0:x1` starts with `a` and finishes with the result. `x2:x3` contains
    /// `b`. `x4` contains `C = 2^128 - P`. The body uses `x5:x12` and the
    /// condition flags as temporary state. It does not access memory or the
    /// stack.
    #[cfg(all(feature = "asm", target_arch = "aarch64"))]
    #[inline(always)]
    fn mul_raw_aarch64_asm(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let [mut out_lo, mut out_hi] = a;
        let [b_lo, b_hi] = b;
        // SAFETY: The register contract is listed above. Every temporary
        // register is declared, and the body does not access memory or stack.
        unsafe {
            asm!(
                include_str!("../../asm/aarch64/fp128_mul_body.inc"),
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
                out("x10") _,
                out("x11") _,
                out("x12") _,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[inline(always)]
    fn sqr_raw(a: [u64; 2]) -> [u64; 2] {
        #[cfg(all(feature = "asm", target_arch = "aarch64"))]
        {
            Self::sqr_raw_aarch64(a)
        }
        #[cfg(not(all(feature = "asm", target_arch = "aarch64")))]
        {
            Self::sqr_raw_portable(a)
        }
    }

    /// Portable squaring (see [`mul_raw_portable`](Self::mul_raw_portable)
    /// for the AArch64 `cfg(test)` role).
    #[cfg(any(
        test,
        feature = "fuzzing",
        not(all(feature = "asm", target_arch = "aarch64"))
    ))]
    #[inline(always)]
    fn sqr_raw_portable(a: [u64; 2]) -> [u64; 2] {
        let [r0, r1, r2, r3] = Self(a).sqr_wide();
        Self::reduce_4(r0, r1, r2, r3)
    }

    /// Squaring schoolbook with the cross term doubled: 3 widening muls.
    ///
    /// Row bounds: `row1 = p00_hi + 2·p01_lo ≤ 3(2^64 − 1) < 2^66` (carry ≤
    /// 2), `row2 = 2·p01_hi + p11_lo + carry1 < 2^66` (carry ≤ 2), and the
    /// top limb is exact because `a² < 2^256` (debug-asserted).
    #[cfg(any(
        test,
        feature = "fuzzing",
        not(all(feature = "asm", target_arch = "aarch64"))
    ))]
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

    /// 31-instruction AArch64 inline-asm squaring with Solinas reduction:
    /// 3 widening multiplies (vs 4 for general mul), the cross term doubled
    /// via shifted-register operands, then the same fold-1 + ccmp
    /// canonicalize as the parameterized multiplication kernel.
    #[cfg(all(feature = "asm", target_arch = "aarch64"))]
    #[inline(always)]
    fn sqr_raw_aarch64(a: [u64; 2]) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        // SAFETY: register-only inline asm (pure, nomem, nostack) over plain
        // integer operands; implements exactly `sqr_wide` + `reduce_4`, with
        // the same `C < 2^32` fold-2 invariant as the multiplication kernel.
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

    fn pow_u128(self, mut exp: u128) -> Self {
        let mut base = self;
        let mut acc = <Self as num_traits::One>::one();
        while exp > 0 {
            if (exp & 1) == 1 {
                acc *= base;
            }
            base = Self(Self::sqr_raw(base.0));
            exp >>= 1;
        }
        acc
    }

    /// Create from a canonical representative in `[0, P)` without checking it.
    ///
    /// # Safety
    ///
    /// `x` must be less than `P`. Violating this condition breaks the private
    /// canonical representation invariant used by the arithmetic kernels.
    #[inline]
    pub const unsafe fn from_canonical_u128(x: u128) -> Self {
        debug_assert!(x < P);
        Self(split(x))
    }

    /// Return the canonical representative in `[0, P)`.
    #[inline]
    pub fn to_canonical_u128(self) -> u128 {
        join(self.0)
    }

    /// Extract the canonical `[lo, hi]` limb representation.
    #[inline(always)]
    pub fn to_limbs(self) -> [u64; 2] {
        self.0
    }

    /// 128×128 → 256-bit widening multiply, **no reduction**.
    ///
    /// Returns `[r0, r1, r2, r3]`: the schoolbook 2×2 portion of the Solinas
    /// multiply without the reduction fold. Cost: 4 widening `mul64`. Row
    /// bounds: each row sums at most three limb halves plus a carry ≤ 2, so
    /// every row fits `u128` with its high part ≤ 2 (kept via the full
    /// `u128` shift); the top limb is exact because `a·b < 2^256`
    /// (debug-asserted).
    #[inline(always)]
    pub fn mul_wide(self, other: Self) -> [u64; 4] {
        let (a0, a1) = (self.0[0], self.0[1]);
        let (b0, b1) = (other.0[0], other.0[1]);
        let (p00_lo, p00_hi) = mul64_wide(a0, b0);
        let (p01_lo, p01_hi) = mul64_wide(a0, b1);
        let (p10_lo, p10_hi) = mul64_wide(a1, b0);
        let (p11_lo, p11_hi) = mul64_wide(a1, b1);

        let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
        let r0 = p00_lo;
        let r1 = row1 as u64;
        let carry1 = (row1 >> 64) as u64;

        let row2 = p01_hi as u128 + p10_hi as u128 + p11_lo as u128 + carry1 as u128;
        let r2 = row2 as u64;
        let carry2 = (row2 >> 64) as u64;

        let row3 = p11_hi as u128 + carry2 as u128;
        let r3 = row3 as u64;
        debug_assert_eq!(row3 >> 64, 0);

        [r0, r1, r2, r3]
    }

    /// 128×64 → 192-bit widening multiply, **no reduction**.
    ///
    /// Returns `[lo, mid, hi]`. Cost: 2 widening `mul64`. Bounds:
    /// `mid ≤ 2(2^64 − 1)` (carry ≤ 1) and `hi ≤ (2^64 − 2) + 1` cannot
    /// overflow because `a·b < 2^192`.
    #[inline(always)]
    pub fn mul_wide_u64(self, other: u64) -> [u64; 3] {
        let (a0, a1) = (self.0[0], self.0[1]);
        let (p0_lo, p0_hi) = mul64_wide(a0, other);
        let (p1_lo, p1_hi) = mul64_wide(a1, other);
        let mid = p0_hi as u128 + p1_lo as u128;
        let hi = p1_hi + (mid >> 64) as u64;
        [p0_lo, mid as u64, hi]
    }

    /// 128×128 → 256-bit widening multiply with a raw `u128` operand,
    /// **no reduction**.
    #[inline(always)]
    pub fn mul_wide_u128(self, other: u128) -> [u64; 4] {
        self.mul_wide(Self(split(other)))
    }

    /// 128×(64×`M`) → (64×`OUT`) widening multiply, **no reduction**.
    ///
    /// Multiplies a canonical field value by an arbitrary little-endian limb
    /// array and returns the little-endian product truncated or extended to
    /// `OUT` limbs. The common three- and four-limb accumulator shapes use
    /// straight-line schedules; other shapes use the generic schoolbook path.
    #[inline(always)]
    pub fn mul_wide_limbs<const M: usize, const OUT: usize>(self, other: [u64; M]) -> [u64; OUT] {
        let (a0, a1) = (self.0[0], self.0[1]);

        if M == 3 && OUT == 5 {
            let (b0, b1, b2) = (other[0], other[1], other[2]);
            let (p00_lo, p00_hi) = mul64_wide(a0, b0);
            let (p01_lo, p01_hi) = mul64_wide(a0, b1);
            let (p02_lo, p02_hi) = mul64_wide(a0, b2);
            let (p10_lo, p10_hi) = mul64_wide(a1, b0);
            let (p11_lo, p11_hi) = mul64_wide(a1, b1);
            let (p12_lo, p12_hi) = mul64_wide(a1, b2);

            let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
            let row2 =
                p01_hi as u128 + p02_lo as u128 + p10_hi as u128 + p11_lo as u128 + (row1 >> 64);
            let row3 = p02_hi as u128 + p11_hi as u128 + p12_lo as u128 + (row2 >> 64);
            let row4 = p12_hi as u128 + (row3 >> 64);
            debug_assert_eq!(row4 >> 64, 0);

            let mut out = [0u64; OUT];
            out[0] = p00_lo;
            out[1] = row1 as u64;
            out[2] = row2 as u64;
            out[3] = row3 as u64;
            out[4] = row4 as u64;
            return out;
        }
        if M == 3 && OUT == 4 {
            let (b0, b1, b2) = (other[0], other[1], other[2]);
            let (p00_lo, p00_hi) = mul64_wide(a0, b0);
            let (p01_lo, p01_hi) = mul64_wide(a0, b1);
            let (p02_lo, p02_hi) = mul64_wide(a0, b2);
            let (p10_lo, p10_hi) = mul64_wide(a1, b0);
            let (p11_lo, p11_hi) = mul64_wide(a1, b1);
            let p12_lo = a1.wrapping_mul(b2);

            let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
            let row2 =
                p01_hi as u128 + p02_lo as u128 + p10_hi as u128 + p11_lo as u128 + (row1 >> 64);
            let row3 = p02_hi as u128 + p11_hi as u128 + p12_lo as u128 + (row2 >> 64);

            let mut out = [0u64; OUT];
            out[0] = p00_lo;
            out[1] = row1 as u64;
            out[2] = row2 as u64;
            out[3] = row3 as u64;
            return out;
        }
        if M == 4 && OUT == 6 {
            let (b0, b1, b2, b3) = (other[0], other[1], other[2], other[3]);
            let (p00_lo, p00_hi) = mul64_wide(a0, b0);
            let (p01_lo, p01_hi) = mul64_wide(a0, b1);
            let (p02_lo, p02_hi) = mul64_wide(a0, b2);
            let (p03_lo, p03_hi) = mul64_wide(a0, b3);
            let (p10_lo, p10_hi) = mul64_wide(a1, b0);
            let (p11_lo, p11_hi) = mul64_wide(a1, b1);
            let (p12_lo, p12_hi) = mul64_wide(a1, b2);
            let (p13_lo, p13_hi) = mul64_wide(a1, b3);

            let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
            let row2 =
                p01_hi as u128 + p02_lo as u128 + p10_hi as u128 + p11_lo as u128 + (row1 >> 64);
            let row3 =
                p02_hi as u128 + p03_lo as u128 + p11_hi as u128 + p12_lo as u128 + (row2 >> 64);
            let row4 = p03_hi as u128 + p12_hi as u128 + p13_lo as u128 + (row3 >> 64);
            let row5 = p13_hi as u128 + (row4 >> 64);
            debug_assert_eq!(row5 >> 64, 0);

            let mut out = [0u64; OUT];
            out[0] = p00_lo;
            out[1] = row1 as u64;
            out[2] = row2 as u64;
            out[3] = row3 as u64;
            out[4] = row4 as u64;
            out[5] = row5 as u64;
            return out;
        }
        if M == 4 && OUT == 5 {
            let (b0, b1, b2, b3) = (other[0], other[1], other[2], other[3]);
            let (p00_lo, p00_hi) = mul64_wide(a0, b0);
            let (p01_lo, p01_hi) = mul64_wide(a0, b1);
            let (p02_lo, p02_hi) = mul64_wide(a0, b2);
            let (p03_lo, p03_hi) = mul64_wide(a0, b3);
            let (p10_lo, p10_hi) = mul64_wide(a1, b0);
            let (p11_lo, p11_hi) = mul64_wide(a1, b1);
            let (p12_lo, p12_hi) = mul64_wide(a1, b2);
            let p13_lo = a1.wrapping_mul(b3);

            let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
            let row2 =
                p01_hi as u128 + p02_lo as u128 + p10_hi as u128 + p11_lo as u128 + (row1 >> 64);
            let row3 =
                p02_hi as u128 + p03_lo as u128 + p11_hi as u128 + p12_lo as u128 + (row2 >> 64);
            let row4 = p03_hi as u128 + p12_hi as u128 + p13_lo as u128 + (row3 >> 64);

            let mut out = [0u64; OUT];
            out[0] = p00_lo;
            out[1] = row1 as u64;
            out[2] = row2 as u64;
            out[3] = row3 as u64;
            out[4] = row4 as u64;
            return out;
        }
        if M == 4 && OUT == 4 {
            let (b0, b1, b2, b3) = (other[0], other[1], other[2], other[3]);
            let (p00_lo, p00_hi) = mul64_wide(a0, b0);
            let (p01_lo, p01_hi) = mul64_wide(a0, b1);
            let (p02_lo, p02_hi) = mul64_wide(a0, b2);
            let p03_lo = a0.wrapping_mul(b3);
            let (p10_lo, p10_hi) = mul64_wide(a1, b0);
            let (p11_lo, p11_hi) = mul64_wide(a1, b1);
            let p12_lo = a1.wrapping_mul(b2);

            let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
            let row2 =
                p01_hi as u128 + p02_lo as u128 + p10_hi as u128 + p11_lo as u128 + (row1 >> 64);
            let row3 =
                p02_hi as u128 + p03_lo as u128 + p11_hi as u128 + p12_lo as u128 + (row2 >> 64);

            let mut out = [0u64; OUT];
            out[0] = p00_lo;
            out[1] = row1 as u64;
            out[2] = row2 as u64;
            out[3] = row3 as u64;
            return out;
        }

        let mut out = [0u64; OUT];
        for (i, &b) in other.iter().enumerate() {
            if i >= OUT {
                break;
            }

            let (p0_lo, p0_hi) = mul64_wide(a0, b);
            let (p1_lo, p1_hi) = mul64_wide(a1, b);
            let s0 = out[i] as u128 + p0_lo as u128;
            out[i] = s0 as u64;
            let mut carry = s0 >> 64;

            if i + 1 >= OUT {
                continue;
            }
            let s1 = out[i + 1] as u128 + p0_hi as u128 + p1_lo as u128 + carry;
            out[i + 1] = s1 as u64;
            carry = s1 >> 64;

            if i + 2 >= OUT {
                continue;
            }
            let s2 = out[i + 2] as u128 + p1_hi as u128 + carry;
            out[i + 2] = s2 as u64;

            let mut carry_hi = s2 >> 64;
            let mut j = i + 3;
            while carry_hi != 0 && j < OUT {
                let sj = out[j] as u128 + carry_hi;
                out[j] = sj as u64;
                carry_hi = sj >> 64;
                j += 1;
            }
        }
        out
    }

    /// Reduce an arbitrary-width little-endian limb array to a canonical
    /// field element via iterated Solinas folding.
    ///
    /// Each fold splits at the 128-bit boundary and replaces `hi · 2^128`
    /// with `hi · C`, shrinking the value by one limb per iteration.
    /// Supports 0–10 input limbs (up to 640 bits).
    ///
    /// # Panics
    ///
    /// Panics if `limbs.len() > 10`.
    #[inline(always)]
    pub fn solinas_reduce(limbs: &[u64]) -> Self {
        match limbs.len() {
            0 => Self(pack(0, 0)),
            // A single limb is always canonical: 2^64 < p.
            1 => Self(pack(limbs[0], 0)),
            // Any u128 < 2^128 = p + C needs at most one subtraction of p.
            2 => Self::from_u128_reduced(join([limbs[0], limbs[1]])),
            // fold2_canonicalize accepts any u64 third limb (see its bounds).
            3 => Self(Self::fold2_canonicalize(limbs[0], limbs[1], limbs[2])),
            4 => Self(Self::reduce_4(limbs[0], limbs[1], limbs[2], limbs[3])),
            5 => {
                // One fold 320 → 256 bits, then reduce_4. Limb bounds:
                // s0 ≤ 2(2^64−1) (carry ≤ 1); s1 ≤ 2(2^64−1) + (C−1) + 1
                // (carry ≤ 2); s2 ≤ (C−1) + (2^64−1) + 2 (carry ≤ 1);
                // s3 = c4_hi + carry ≤ C < 2^64, so no fifth limb
                // (debug-asserted). All carries use full u128 shifts.
                let (l0, l1, l2, l3, l4) = (limbs[0], limbs[1], limbs[2], limbs[3], limbs[4]);
                let (c2_lo, c2_hi) = Self::mul_c_wide(l2);
                let (c3_lo, c3_hi) = Self::mul_c_wide(l3);
                let (c4_lo, c4_hi) = Self::mul_c_wide(l4);

                let s0 = l0 as u128 + c2_lo as u128;
                let s1 = l1 as u128 + c2_hi as u128 + c3_lo as u128 + (s0 >> 64);
                let s2 = c3_hi as u128 + c4_lo as u128 + (s1 >> 64);
                let s3 = c4_hi as u128 + (s2 >> 64);
                debug_assert_eq!(s3 >> 64, 0);

                Self(Self::reduce_4(s0 as u64, s1 as u64, s2 as u64, s3 as u64))
            }
            n => {
                assert!(n <= 10, "solinas_reduce supports at most 10 limbs");
                let mut buf = [0u64; 11];
                buf[..n].copy_from_slice(limbs);
                let mut len = n;
                let c = Self::C_LO;

                // Each pass computes C·buf[2..len] + buf[0..2]. With
                // H < 2^{64(len−2)} and L < 2^128, the result is
                // < 2^{64(len−2)+32} + 2^128 < 2^{64(len−1)} for len ≥ 4,
                // so it fits len−1 limbs and the carry chain dies out
                // (debug-asserted below).
                while len > 5 {
                    let high_len = len - 2;
                    let mut next = [0u64; 11];

                    let mut carry: u64 = 0;
                    for i in 0..high_len {
                        let wide = c as u128 * buf[i + 2] as u128 + carry as u128;
                        next[i] = wide as u64;
                        carry = (wide >> 64) as u64;
                    }
                    // carry ≤ C − 1: the top partial product's high half.
                    next[high_len] = carry;

                    let s0 = next[0] as u128 + buf[0] as u128;
                    next[0] = s0 as u64;
                    let s1 = next[1] as u128 + buf[1] as u128 + (s0 >> 64);
                    next[1] = s1 as u64;
                    let mut c_out = (s1 >> 64) as u64;
                    for limb in &mut next[2..=high_len] {
                        if c_out == 0 {
                            break;
                        }
                        let s = *limb as u128 + c_out as u128;
                        *limb = s as u64;
                        c_out = (s >> 64) as u64;
                    }
                    debug_assert_eq!(c_out, 0);

                    buf = next;
                    len -= 1;
                    while len > 5 && buf[len - 1] == 0 {
                        len -= 1;
                    }
                }

                Self::solinas_reduce(&buf[..len])
            }
        }
    }

    /// Cross-checks every architecture-specific kernel against its portable
    /// implementation. This is public only for the out-of-crate fuzz target.
    #[cfg(all(
        feature = "fuzzing",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    #[doc(hidden)]
    pub fn assert_asm_matches_portable_for_fuzzing(self, rhs: Self, _addend: Self) {
        assert_eq!(
            Self::add_raw(self.0, rhs.0),
            Self::add_raw_portable(self.0, rhs.0)
        );
        assert_eq!(
            Self::sub_raw(self.0, rhs.0),
            Self::sub_raw_portable(self.0, rhs.0)
        );
        let asm_mul = Self::mul_raw(self.0, rhs.0);
        assert_eq!(asm_mul, Self::mul_raw_portable(self.0, rhs.0));

        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(all(target_feature = "bmi2", target_feature = "adx"))]
            let expected_backend = if Self::C_LO == A7F7_OFFSET {
                X86_64_BMI2_ADX_BACKEND
            } else {
                X86_64_BASELINE_BACKEND
            };
            #[cfg(not(all(target_feature = "bmi2", target_feature = "adx")))]
            let expected_backend = X86_64_BASELINE_BACKEND;
            assert_eq!(
                LAST_X86_64_MUL_BACKEND.load(Ordering::Relaxed),
                expected_backend,
                "the requested x86-64 multiplication backend was not executed"
            );
        }

        #[cfg(target_arch = "aarch64")]
        {
            assert_eq!(
                Self::mul_add_raw(self.0, rhs.0, _addend.0),
                Self::mul_add_raw_portable(self.0, rhs.0, _addend.0)
            );
            assert_eq!(Self::sqr_raw(self.0), Self::sqr_raw_portable(self.0));
        }
    }
}

crate::impl_ring_ops!(impl[const P: u128] Fp128<P> {
    add(a, b): Fp128(Self::add_raw(a.0, b.0)),
    sub(a, b): Fp128(Self::sub_raw(a.0, b.0)),
    mul(a, b): Fp128(Self::mul_raw(a.0, b.0)),
    neg(a): Fp128(Self::sub_raw(pack(0, 0), a.0)),
    zero: Fp128(pack(0, 0)),
    // P > 1 is implied by the C asserts (odd and C(C+1) < P).
    one: Fp128(pack(1, 0)),
});

impl<const P: u128> std::fmt::Display for Fp128<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_canonical_u128())
    }
}

impl<const P: u128> Ring for Fp128<P> {
    /// Any u64 is canonical: `p = 2^128 − C > 2^64`, so no reduction path.
    #[inline(always)]
    fn from_u64(v: u64) -> Self {
        Self(pack(v, 0))
    }

    #[inline(always)]
    fn from_i64(v: i64) -> Self {
        if v >= 0 {
            Self::from_u64(v as u64)
        } else {
            -Self::from_u64(v.unsigned_abs())
        }
    }

    #[inline(always)]
    fn from_u128(v: u128) -> Self {
        Self::from_u128_reduced(v)
    }

    #[inline(always)]
    fn from_i128(v: i128) -> Self {
        if v >= 0 {
            Self::from_u128(v as u128)
        } else {
            -Self::from_u128(v.unsigned_abs())
        }
    }

    #[inline(always)]
    fn square(&self) -> Self {
        Self(Self::sqr_raw(self.0))
    }
}

impl<const P: u128> Field for Fp128<P> {
    #[inline(always)]
    fn inverse(&self) -> Option<Self> {
        let inv = self.inv_or_zero();
        if num_traits::Zero::is_zero(self) {
            None
        } else {
            Some(inv)
        }
    }

    /// Fermat inversion with branchless zero-masking.
    #[inline(always)]
    fn inv_or_zero(self) -> Self {
        let candidate = self.pow_u128(P.wrapping_sub(2));
        let v = join(self.0);
        let nz = ((v | v.wrapping_neg()) >> 127) & 1;
        let mask = 0u128.wrapping_sub(nz);
        Self(split(join(candidate.0) & mask))
    }

    /// Canonical rejection sampling: each attempt reads exactly 16
    /// little-endian bytes and rejects non-canonical candidates (probability
    /// `C / 2^128 < 2^-96` per draw).
    #[inline(always)]
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self(split(super::sample_uniform_below(rng, P, u128::BITS)))
    }

    #[inline(always)]
    fn mul_add(self, rhs: Self, addend: Self) -> Self {
        Self(Self::mul_add_raw(self.0, rhs.0, addend.0))
    }

    /// Halving via shift: `(x + (x odd)·p) / 2`, computed as
    /// `(x >> 1) + (x & 1)·(p + 1)/2`, which stays below `p` (no overflow):
    /// for odd `x ≤ p − 2`, the sum is at most `(p − 3)/2 + (p + 1)/2 = p − 1`.
    #[inline]
    fn half(self) -> Self {
        let x = join(self.0);
        Self(split((x >> 1) + (x & 1) * ((P >> 1) + 1)))
    }

    #[inline]
    fn two_inv() -> Self {
        <Self as num_traits::One>::one().half()
    }
}

impl<const P: u128> CanonicalBytes for Fp128<P> {
    const NUM_BYTES: usize = 16;

    #[inline(always)]
    fn to_bytes_le(&self, out: &mut [u8]) {
        assert_eq!(out.len(), Self::NUM_BYTES);
        out.copy_from_slice(&join(self.0).to_le_bytes());
    }
}

impl<const P: u128> CanonicalEncoding for Fp128<P> {
    // C < 2^32 implies p > 2^127, so the modulus is exactly 128 bits.
    const MODULUS_BITS: u32 = 128;

    #[inline(always)]
    fn from_bytes_le_reduced(bytes: &[u8]) -> Self {
        if bytes.len() <= 16 {
            let mut padded = [0u8; 16];
            padded[..bytes.len()].copy_from_slice(bytes);
            return Self::from_u128(u128::from_le_bytes(padded));
        }
        crate::solinas::reduce_le_bytes_mod_order(bytes)
    }

    #[inline]
    fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
        let arr: [u8; 16] = bytes.try_into().ok()?;
        Self::from_u128_checked(u128::from_le_bytes(arr))
    }

    #[inline]
    fn to_u128_checked(&self) -> Option<u128> {
        Some(join(self.0))
    }

    #[inline]
    fn from_u128_checked(v: u128) -> Option<Self> {
        (v < P).then(|| Self(split(v)))
    }

    /// Any u128 is below `2^128 = p + C < 2p`, so a single conditional
    /// subtraction canonicalizes (and `v − p < C < p`).
    #[inline]
    fn from_u128_reduced(v: u128) -> Self {
        let (sub, borrow) = v.overflowing_sub(P);
        Self(split(if borrow { v } else { sub }))
    }

    #[inline]
    fn num_bits(&self) -> u32 {
        u128::BITS - join(self.0).leading_zeros()
    }

    #[inline]
    fn from_scalar_challenge_bytes(bytes: &[u8]) -> Self {
        Self::from_bytes_le_reduced(bytes)
    }
}

crate::impl_serde_bytes!(impl[const P: u128] Fp128<P>, 16);

impl<const P: u128> WithAccumulator for Fp128<P> {
    type Accumulator = crate::Fp128Accumulator<P>;
    type SmallScalarAccumulator = crate::Fp128SignedAccumulator<P>;
    type SignedProductAccumulator = crate::Fp128SignedAccumulator<P>;
}

impl<const P: u128> PseudoMersenne for Fp128<P> {
    const OFFSET: u128 = Self::C;
}

// Cross-check the inline-asm kernels against the portable arithmetic on every
// supported architecture.
#[cfg(all(
    test,
    feature = "asm",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    fn cases(p: u128) -> Vec<[u64; 2]> {
        let mut rng = ChaCha20Rng::seed_from_u64(0xf128_a5a5);
        let mut v: Vec<[u64; 2]> = [0u128, 1, 2, p / 2, p / 2 + 1, p - 2, p - 1]
            .iter()
            .map(|&x| split(x))
            .collect();
        v.extend((0..500).map(|_| split(rng.gen::<u128>() % p)));
        v
    }

    fn check<const P: u128>() {
        for a in cases(P) {
            #[cfg(target_arch = "aarch64")]
            assert_eq!(
                Fp128::<P>::sqr_raw_aarch64(a),
                Fp128::<P>::sqr_raw_portable(a),
                "sqr asm vs portable, a={a:?}"
            );
            for b in cases(P) {
                assert_eq!(
                    Fp128::<P>::add_raw(a, b),
                    Fp128::<P>::add_raw_portable(a, b),
                    "add asm vs portable, a={a:?} b={b:?}"
                );
                assert_eq!(
                    Fp128::<P>::sub_raw(a, b),
                    Fp128::<P>::sub_raw_portable(a, b),
                    "sub asm vs portable, a={a:?} b={b:?}"
                );
                assert_eq!(
                    Fp128::<P>::mul_raw(a, b),
                    Fp128::<P>::mul_raw_portable(a, b),
                    "mul asm vs portable, a={a:?} b={b:?}"
                );
                #[cfg(target_arch = "aarch64")]
                {
                    let addend = split(join(a).wrapping_add(join(b)) % P);
                    assert_eq!(
                        Fp128::<P>::mul_add_raw_aarch64(a, b, addend),
                        Fp128::<P>::mul_add_raw_portable(a, b, addend),
                        "mul-add asm vs portable, a={a:?} b={b:?} addend={addend:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn fp128_asm_matches_portable() {
        check::<{ u128::MAX - 172 }>(); // C = 173, outside the published aliases
        check::<{ u128::MAX - 274 }>(); // C = 275
        check::<{ u128::MAX - (A7F7_OFFSET as u128 - 1) }>();
    }
}

#[cfg(test)]
mod wide_tests {
    use super::*;
    use crate::solinas::Prime128Offset275;
    use rand_chacha::ChaCha20Rng;
    use rand_core::RngCore;
    use rand_core::SeedableRng;

    #[test]
    fn mul_wide_limbs_roundtrips_through_reduction() {
        type F = Prime128Offset275;
        let mut rng = ChaCha20Rng::seed_from_u64(0x1bad_f00d_0ddc_afe1);
        for _ in 0..1000 {
            let a = F::random(&mut rng);
            let b3 = [rng.next_u64(), rng.next_u64(), rng.next_u64()];
            let b4 = [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
            ];

            let got3_full = a.mul_wide_limbs::<3, 5>(b3);
            let got3_trunc = a.mul_wide_limbs::<3, 4>(b3);
            assert_eq!(got3_trunc, got3_full[..4]);
            assert_eq!(F::solinas_reduce(&got3_full), a * F::solinas_reduce(&b3));

            let got4_full = a.mul_wide_limbs::<4, 6>(b4);
            let got4_trunc = a.mul_wide_limbs::<4, 4>(b4);
            assert_eq!(got4_trunc, got4_full[..4]);
            assert_eq!(F::solinas_reduce(&got4_full), a * F::solinas_reduce(&b4));
        }
    }
}
