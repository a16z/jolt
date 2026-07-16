use super::*;

impl<const P: u128> Fp128<P> {
    /// Extract the canonical `[lo, hi]` limb representation.
    #[inline(always)]
    pub fn to_limbs(self) -> [u64; 2] {
        self.0
    }

    /// 128×64 → 192-bit widening multiply, **no reduction**.
    ///
    /// Returns `[lo, mid, hi]` representing `self · other` as a 192-bit
    /// integer.  Cost: 2 widening `mul64`.
    #[inline(always)]
    pub fn mul_wide_u64(self, other: u64) -> [u64; 3] {
        let (a0, a1) = (self.0[0], self.0[1]);
        let (p0_lo, p0_hi) = mul64_wide(a0, other);
        let (p1_lo, p1_hi) = mul64_wide(a1, other);
        let mid = p0_hi as u128 + p1_lo as u128;
        let hi = p1_hi + (mid >> 64) as u64;
        [p0_lo, mid as u64, hi]
    }

    /// 128×128 → 256-bit widening multiply, **no reduction**.
    ///
    /// Returns `[r0, r1, r2, r3]` representing `self · other` as a 256-bit
    /// integer.  This is the schoolbook 2×2 portion of the Solinas multiply,
    /// without the reduction fold.  Cost: 4 widening `mul64`.
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

    /// 128×128 → 256-bit widening multiply with a raw `u128` operand,
    /// **no reduction**.
    #[inline(always)]
    pub fn mul_wide_u128(self, other: u128) -> [u64; 4] {
        self.mul_wide(Self(from_u128(other)))
    }

    /// 128×(64*M) → (64*OUT) widening multiply, **no reduction**.
    ///
    /// Multiplies a canonical Fp128 value (`[u64; 2]`) by an arbitrary
    /// little-endian limb array and returns the little-endian product
    /// truncated/extended to `OUT` limbs.
    #[inline(always)]
    pub fn mul_wide_limbs<const M: usize, const OUT: usize>(self, other: [u64; M]) -> [u64; OUT] {
        let (a0, a1) = (self.0[0], self.0[1]);

        // Hot-path specializations used by Jolt (M in {3,4}, OUT in {4,5}).
        // These avoid loop/control-flow overhead in tight sumcheck FMAs.
        if M == 3 && OUT == 5 {
            let b0 = other[0];
            let b1 = other[1];
            let b2 = other[2];

            let (p00_lo, p00_hi) = mul64_wide(a0, b0);
            let (p01_lo, p01_hi) = mul64_wide(a0, b1);
            let (p02_lo, p02_hi) = mul64_wide(a0, b2);
            let (p10_lo, p10_hi) = mul64_wide(a1, b0);
            let (p11_lo, p11_hi) = mul64_wide(a1, b1);
            let (p12_lo, p12_hi) = mul64_wide(a1, b2);

            let r0 = p00_lo;

            let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
            let r1 = row1 as u64;
            let carry1 = row1 >> 64;

            let row2 = p01_hi as u128 + p02_lo as u128 + p10_hi as u128 + p11_lo as u128 + carry1;
            let r2 = row2 as u64;
            let carry2 = row2 >> 64;

            let row3 = p02_hi as u128 + p11_hi as u128 + p12_lo as u128 + carry2;
            let r3 = row3 as u64;
            let carry3 = row3 >> 64;

            let row4 = p12_hi as u128 + carry3;
            let r4 = row4 as u64;
            debug_assert_eq!(row4 >> 64, 0);

            let mut out = [0u64; OUT];
            out[0] = r0;
            out[1] = r1;
            out[2] = r2;
            out[3] = r3;
            out[4] = r4;
            return out;
        }
        if M == 3 && OUT == 4 {
            let b0 = other[0];
            let b1 = other[1];
            let b2 = other[2];

            let (p00_lo, p00_hi) = mul64_wide(a0, b0);
            let (p01_lo, p01_hi) = mul64_wide(a0, b1);
            let (p02_lo, p02_hi) = mul64_wide(a0, b2);
            let (p10_lo, p10_hi) = mul64_wide(a1, b0);
            let (p11_lo, p11_hi) = mul64_wide(a1, b1);
            let p12_lo = a1.wrapping_mul(b2);

            let r0 = p00_lo;

            let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
            let r1 = row1 as u64;
            let carry1 = row1 >> 64;

            let row2 = p01_hi as u128 + p02_lo as u128 + p10_hi as u128 + p11_lo as u128 + carry1;
            let r2 = row2 as u64;
            let carry2 = row2 >> 64;

            let row3 = p02_hi as u128 + p11_hi as u128 + p12_lo as u128 + carry2;
            let r3 = row3 as u64;

            let mut out = [0u64; OUT];
            out[0] = r0;
            out[1] = r1;
            out[2] = r2;
            out[3] = r3;
            return out;
        }
        if M == 4 && OUT == 6 {
            let b0 = other[0];
            let b1 = other[1];
            let b2 = other[2];
            let b3 = other[3];

            let (p00_lo, p00_hi) = mul64_wide(a0, b0);
            let (p01_lo, p01_hi) = mul64_wide(a0, b1);
            let (p02_lo, p02_hi) = mul64_wide(a0, b2);
            let (p03_lo, p03_hi) = mul64_wide(a0, b3);
            let (p10_lo, p10_hi) = mul64_wide(a1, b0);
            let (p11_lo, p11_hi) = mul64_wide(a1, b1);
            let (p12_lo, p12_hi) = mul64_wide(a1, b2);
            let (p13_lo, p13_hi) = mul64_wide(a1, b3);

            let r0 = p00_lo;

            let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
            let r1 = row1 as u64;
            let carry1 = row1 >> 64;

            let row2 = p01_hi as u128 + p02_lo as u128 + p10_hi as u128 + p11_lo as u128 + carry1;
            let r2 = row2 as u64;
            let carry2 = row2 >> 64;

            let row3 = p02_hi as u128 + p03_lo as u128 + p11_hi as u128 + p12_lo as u128 + carry2;
            let r3 = row3 as u64;
            let carry3 = row3 >> 64;

            let row4 = p03_hi as u128 + p12_hi as u128 + p13_lo as u128 + carry3;
            let r4 = row4 as u64;
            let carry4 = row4 >> 64;

            let row5 = p13_hi as u128 + carry4;
            let r5 = row5 as u64;
            debug_assert_eq!(row5 >> 64, 0);

            let mut out = [0u64; OUT];
            out[0] = r0;
            out[1] = r1;
            out[2] = r2;
            out[3] = r3;
            out[4] = r4;
            out[5] = r5;
            return out;
        }
        if M == 4 && OUT == 5 {
            let b0 = other[0];
            let b1 = other[1];
            let b2 = other[2];
            let b3 = other[3];

            let (p00_lo, p00_hi) = mul64_wide(a0, b0);
            let (p01_lo, p01_hi) = mul64_wide(a0, b1);
            let (p02_lo, p02_hi) = mul64_wide(a0, b2);
            let (p03_lo, p03_hi) = mul64_wide(a0, b3);
            let (p10_lo, p10_hi) = mul64_wide(a1, b0);
            let (p11_lo, p11_hi) = mul64_wide(a1, b1);
            let (p12_lo, p12_hi) = mul64_wide(a1, b2);
            let p13_lo = a1.wrapping_mul(b3);

            let r0 = p00_lo;

            let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
            let r1 = row1 as u64;
            let carry1 = row1 >> 64;

            let row2 = p01_hi as u128 + p02_lo as u128 + p10_hi as u128 + p11_lo as u128 + carry1;
            let r2 = row2 as u64;
            let carry2 = row2 >> 64;

            let row3 = p02_hi as u128 + p03_lo as u128 + p11_hi as u128 + p12_lo as u128 + carry2;
            let r3 = row3 as u64;
            let carry3 = row3 >> 64;

            let row4 = p03_hi as u128 + p12_hi as u128 + p13_lo as u128 + carry3;
            let r4 = row4 as u64;

            let mut out = [0u64; OUT];
            out[0] = r0;
            out[1] = r1;
            out[2] = r2;
            out[3] = r3;
            out[4] = r4;
            return out;
        }
        if M == 4 && OUT == 4 {
            let b0 = other[0];
            let b1 = other[1];
            let b2 = other[2];
            let b3 = other[3];

            let (p00_lo, p00_hi) = mul64_wide(a0, b0);
            let (p01_lo, p01_hi) = mul64_wide(a0, b1);
            let (p02_lo, p02_hi) = mul64_wide(a0, b2);
            let p03_lo = a0.wrapping_mul(b3);
            let (p10_lo, p10_hi) = mul64_wide(a1, b0);
            let (p11_lo, p11_hi) = mul64_wide(a1, b1);
            let p12_lo = a1.wrapping_mul(b2);

            let r0 = p00_lo;

            let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
            let r1 = row1 as u64;
            let carry1 = row1 >> 64;

            let row2 = p01_hi as u128 + p02_lo as u128 + p10_hi as u128 + p11_lo as u128 + carry1;
            let r2 = row2 as u64;
            let carry2 = row2 >> 64;

            let row3 = p02_hi as u128 + p03_lo as u128 + p11_hi as u128 + p12_lo as u128 + carry2;
            let r3 = row3 as u64;

            let mut out = [0u64; OUT];
            out[0] = r0;
            out[1] = r1;
            out[2] = r2;
            out[3] = r3;
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
}
