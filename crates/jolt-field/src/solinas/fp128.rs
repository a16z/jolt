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
//! On AArch64 the multiply and squaring kernels use inline assembly
//! (benchmarked at 1.29x throughput vs the portable path on Apple M4); every
//! other path, and every path on other architectures, is portable Rust.

use super::word::mul64_wide;
use crate::PseudoMersenne;
use crate::{CanonicalBytes, CanonicalEncoding, Field, NaiveAccumulator, Ring, WithAccumulator};
use rand_core::RngCore;
#[cfg(target_arch = "aarch64")]
use std::arch::asm;

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

    /// Carry-chain add with fused reduction.
    ///
    /// For `a, b < p`: if the two-limb add wraps (`overflow`), the real sum
    /// is `s + 2^128 ≡ s + C`, and `s = a + b − 2^128 < 2p − 2^128 = p − C`,
    /// so `s + C < p` is already canonical (and `carry3 = 0`). Without wrap,
    /// `s + C` carries iff `s ≥ p`, and then the wrapped value is
    /// `s − p ≤ p − 2`. Both cases select `r` on `overflow | carry3`.
    #[inline(always)]
    fn add_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let (s0, carry0) = a[0].overflowing_add(b[0]);
        let (s1a, carry1a) = a[1].overflowing_add(b[1]);
        let (s1, carry1b) = s1a.overflowing_add(carry0 as u64);
        let overflow = carry1a | carry1b;

        let (r0, carry2) = s0.overflowing_add(Self::C_LO);
        let (r1, carry3) = s1.overflowing_add(carry2 as u64);

        pack(
            if overflow | carry3 { r0 } else { s0 },
            if overflow | carry3 { r1 } else { s1 },
        )
    }

    /// Subtract with borrow-conditional modulus add-back (`a − b + p` when
    /// `a < b`; the wrapped difference plus `p` cannot wrap again since
    /// `a − b + 2^128 + p − 2^128 = a − b + p < p`).
    #[inline(always)]
    fn sub_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let (diff, borrow) = join(a).overflowing_sub(join(b));
        split(if borrow { diff.wrapping_add(P) } else { diff })
    }

    #[inline(always)]
    fn mul_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        #[cfg(target_arch = "aarch64")]
        {
            Self::mul_raw_aarch64(a, b)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self::mul_raw_portable(a, b)
        }
    }

    /// Portable multiply: schoolbook 2×2 widening product, then the two
    /// Solinas folds. On AArch64 this is compiled only under `cfg(test)`
    /// as the differential oracle for the assembly kernel.
    #[cfg(any(not(target_arch = "aarch64"), test))]
    #[inline(always)]
    fn mul_raw_portable(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let [r0, r1, r2, r3] = Self(a).mul_wide(Self(b));
        Self::reduce_4(r0, r1, r2, r3)
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
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn mul_raw_aarch64(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        // SAFETY: register-only inline asm (pure, nomem, nostack) over plain
        // integer operands; the carry/flag flow implements exactly the
        // portable `mul_wide` + `reduce_4` algorithm, and `C < 2^32` (const-
        // asserted) guarantees the fold-2 `mul {p11h}, {c}` cannot overflow.
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

    /// Portable squaring (see [`mul_raw_portable`](Self::mul_raw_portable)
    /// for the AArch64 `cfg(test)` role).
    #[cfg(any(not(target_arch = "aarch64"), test))]
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
    #[cfg(any(not(target_arch = "aarch64"), test))]
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
    /// canonicalize as [`mul_raw_aarch64`](Self::mul_raw_aarch64).
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn sqr_raw_aarch64(a: [u64; 2]) -> [u64; 2] {
        let out_lo: u64;
        let out_hi: u64;
        // SAFETY: register-only inline asm (pure, nomem, nostack) over plain
        // integer operands; implements exactly `sqr_wide` + `reduce_4`, with
        // the same `C < 2^32` fold-2 invariant as `mul_raw_aarch64`.
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

    /// Create from a canonical representative in `[0, P)`.
    #[inline]
    pub fn from_canonical_u128(x: u128) -> Self {
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
}

crate::impl_serde_bytes!(impl[const P: u128] Fp128<P>, 16);

impl<const P: u128> WithAccumulator for Fp128<P> {
    type Accumulator = NaiveAccumulator<Self>;
    type SmallScalarAccumulator = NaiveAccumulator<Self>;
    type SignedProductAccumulator = NaiveAccumulator<Self>;
}

impl<const P: u128> PseudoMersenne for Fp128<P> {
    const OFFSET: u128 = Self::C;
}

// AArch64-only: the inline-asm kernels against the portable fold, so a
// machine running the asm still exercises (and cross-checks) both paths.
#[cfg(test)]
#[cfg(target_arch = "aarch64")]
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
            assert_eq!(
                Fp128::<P>::sqr_raw_aarch64(a),
                Fp128::<P>::sqr_raw_portable(a),
                "sqr asm vs portable, a={a:?}"
            );
            for b in cases(P) {
                assert_eq!(
                    Fp128::<P>::mul_raw_aarch64(a, b),
                    Fp128::<P>::mul_raw_portable(a, b),
                    "mul asm vs portable, a={a:?} b={b:?}"
                );
            }
        }
    }

    #[test]
    fn asm_matches_portable() {
        check::<{ u128::MAX - 274 }>(); // C = 275
        check::<{ u128::MAX - 158 }>(); // C = 159
        check::<{ u128::MAX - 2354 }>(); // C = 2355
        check::<{ u128::MAX - 0xFFFF_A7F6 }>(); // C = 0xFFFF_A7F7
    }
}
