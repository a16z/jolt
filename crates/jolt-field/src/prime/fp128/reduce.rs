use super::*;

impl<const P: u128> Fp128<P> {
    /// +1 means `C = 2^a + 1`, -1 means `C = 2^a - 1`, 0 means generic.
    const C_SHIFT_KIND: i8 = {
        let c = Self::C_LO;
        if c > 1 && is_pow2_u64(c - 1) {
            1
        } else if c == u64::MAX || is_pow2_u64(c + 1) {
            -1
        } else {
            0
        }
    };
    const C_SHIFT: u32 = {
        let c = Self::C_LO;
        if Self::C_SHIFT_KIND == 1 {
            log2_pow2_u64(c - 1)
        } else if Self::C_SHIFT_KIND == -1 {
            if c == u64::MAX {
                64
            } else {
                log2_pow2_u64(c + 1)
            }
        } else {
            0
        }
    };

    /// Multiply by `C = 2^128 - P`. For `C = 2^a ± 1`, this is shift/add or
    /// shift/sub only; otherwise it falls back to generic widening multiply.
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
    /// Correctness argument for the fused overflow+canonicalize:
    ///
    /// Let `v = base + C·t2` (mathematical, not mod 2^128).
    /// From the fold-1 mac chain, `t2 ≤ C`, so `C·t2 ≤ C²`.
    ///
    /// - **No overflow** (`v < 2^128`): `s = v`, and the standard
    ///   canonicalize applies — `s + C` carries iff `s ≥ P`.
    /// - **Overflow** (`v ≥ 2^128`): `s = v − 2^128`, so `s < C·t2 ≤ C²`.
    ///   The correct reduced value is `s + C` (since `2^128 ≡ C mod P`).
    ///   Because `s + C < C² + C = C(C+1)` and `C(C+1) < P` for all
    ///   `C < 2^64`, the value `s + C` is already in `[0, P)` — no
    ///   further canonicalization is needed, and `s + C < 2^128` so the
    ///   add does NOT carry.
    ///
    /// Therefore `if (overflow | carry) { s + C } else { s }` is correct
    /// in both cases, fusing the overflow correction with canonicalization.
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

    /// Solinas fold for exactly 4 limbs: `[r0,r1] + C·[r2,r3]` → 3 limbs,
    /// then `fold2_canonicalize`.
    #[inline(always)]
    pub(super) fn reduce_4(r0: u64, r1: u64, r2: u64, r3: u64) -> [u64; 2] {
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

    /// Add a canonical 128-bit value into a 256-bit little-endian limb array.
    ///
    /// Since both multiplicands and addends are canonical field elements,
    /// `a * b + c < 2^256`, so the top carry is guaranteed to be zero.
    #[inline(always)]
    pub(super) fn add_128_into_256(prod: [u64; 4], addend: [u64; 2]) -> [u64; 4] {
        let (s0, carry0) = prod[0].overflowing_add(addend[0]);
        let (s1a, carry1a) = prod[1].overflowing_add(addend[1]);
        let (s1, carry1b) = s1a.overflowing_add(carry0 as u64);
        let carry1 = carry1a | carry1b;
        let (s2, carry2) = prod[2].overflowing_add(carry1 as u64);
        let (s3, carry3) = prod[3].overflowing_add(carry2 as u64);
        debug_assert!(!carry3);
        [s0, s1, s2, s3]
    }

    /// Reduce an arbitrary-width little-endian limb array to a canonical
    /// field element via iterated Solinas folding.
    ///
    /// Each fold splits at the 128-bit boundary and replaces
    /// `hi · 2^128` with `hi · C`, reducing width by one limb per
    /// iteration.  Supports 0–10 input limbs (up to 640 bits).
    ///
    /// # Panics
    ///
    /// Panics if `limbs.len() > 10`.
    #[inline(always)]
    pub fn solinas_reduce(limbs: &[u64]) -> Self {
        match limbs.len() {
            0 => Self::zero(),
            1 => Self(pack(limbs[0], 0)),
            2 => Self::from_canonical_u128_reduced(to_u128([limbs[0], limbs[1]])),
            3 => Self(Self::fold2_canonicalize(limbs[0], limbs[1], limbs[2])),
            4 => Self(Self::reduce_4(limbs[0], limbs[1], limbs[2], limbs[3])),
            5 => {
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

                while len > 5 {
                    let high_len = len - 2;
                    let mut next = [0u64; 11];

                    let mut carry: u64 = 0;
                    for i in 0..high_len {
                        let wide = c as u128 * buf[i + 2] as u128 + carry as u128;
                        next[i] = wide as u64;
                        carry = (wide >> 64) as u64;
                    }
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
