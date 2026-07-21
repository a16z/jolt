use super::*;

/// Number of packed `Fp32` lanes.
pub(crate) const FP32_WIDTH: usize = 4;

/// NEON packed `Fp32` backend: 4 lanes in `uint32x4_t`.
#[derive(Clone, Copy)]
pub struct PackedFp32Neon<const P: u32> {
    vals: [u32; 4],
}

#[inline(always)]
fn to_vec32(x: [u32; 4]) -> uint32x4_t {
    unsafe { transmute::<[u32; 4], uint32x4_t>(x) }
}

#[inline(always)]
fn from_vec32(v: uint32x4_t) -> [u32; 4] {
    unsafe { transmute::<uint32x4_t, [u32; 4]>(v) }
}

impl<const P: u32> PackedFp32Neon<P> {
    const BITS: u32 = 32 - P.leading_zeros();

    const C: u32 = {
        let c = if Self::BITS == 32 {
            0u32.wrapping_sub(P)
        } else {
            (1u32 << Self::BITS) - P
        };
        assert!(P != 0, "modulus must be nonzero");
        assert!(P & 1 == 1, "modulus must be odd");
        assert!(
            (c as u64) * (c as u64 + 1) < P as u64,
            "C(C+1) < P required for fused canonicalize"
        );
        c
    };

    const MASK_U64: u64 = if Self::BITS == 32 {
        u32::MAX as u64
    } else {
        (1u64 << Self::BITS) - 1
    };

    const TWO_FOLD_FOUR_PRODUCT_OK: bool = {
        let c = Self::C as u64;
        4 * c * c + 3 * c <= (1u64 << Self::BITS)
    };

    #[inline(always)]
    fn to_vec(self) -> uint32x4_t {
        to_vec32(self.vals)
    }

    #[inline(always)]
    fn from_vec(v: uint32x4_t) -> Self {
        Self {
            vals: from_vec32(v),
        }
    }

    #[inline(always)]
    fn add_vec(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe {
            let p = vdupq_n_u32(P);
            if Self::BITS <= 31 {
                let t = vaddq_u32(a, b);
                vminq_u32(t, vsubq_u32(t, p))
            } else {
                let c = vdupq_n_u32(Self::C);
                let t = vaddq_u32(a, b);
                let overflow = vcltq_u32(t, a);
                let folded = vaddq_u32(t, vandq_u32(overflow, c));
                vminq_u32(folded, vsubq_u32(folded, p))
            }
        }
    }

    #[inline(always)]
    fn sub_vec(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe {
            let p = vdupq_n_u32(P);
            if Self::BITS <= 31 {
                let t = vsubq_u32(a, b);
                vminq_u32(t, vaddq_u32(t, p))
            } else {
                let t = vsubq_u32(a, b);
                let underflow = vcltq_u32(a, b);
                vsubq_u32(t, vandq_u32(underflow, vdupq_n_u32(Self::C)))
            }
        }
    }

    #[inline(always)]
    fn mul_vec(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe {
            if Self::BITS == 31 {
                return if Self::C == 1 {
                    Self::mul_mersenne31_vec(a, b)
                } else {
                    Self::mul_pmersenne31_vec(a, b)
                };
            }
            let prod_lo = vmull_u32(vget_low_u32(a), vget_low_u32(b));
            let prod_hi = vmull_high_u32(a, b);
            Self::solinas_reduce(prod_lo, prod_hi)
        }
    }

    #[inline(always)]
    unsafe fn mul_mersenne31_vec(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe {
            let p = vdupq_n_u32(P);
            let prod_hi31 = vreinterpretq_u32_s32(vqdmulhq_s32(
                vreinterpretq_s32_u32(a),
                vreinterpretq_s32_u32(b),
            ));
            let prod_lo32 = vmulq_u32(a, b);
            let folded = vmlsq_u32(prod_lo32, prod_hi31, p);
            vminq_u32(folded, vsubq_u32(folded, p))
        }
    }

    /// Packed multiply for 31-bit pseudo-Mersenne primes `P = 2^31 - C`
    /// (`BITS == 31`, `C > 1`), reducing entirely in 32-bit lanes.
    ///
    /// This generalises the `C == 1` Mersenne kernel
    /// ([`Self::mul_mersenne31_vec`]) to any small `C` admitted by the
    /// `Fp32<P>` invariant `C(C+1) < P`, replacing the 64-bit-widening
    /// [`Self::solinas_reduce`] path. It keeps all four lanes in `uint32x4_t`
    /// and uses two `vqdmulhq_s32` high-multiplies (the same instruction the
    /// Mersenne path uses) to extract Solinas fold high words without ever
    /// forming a 64-bit intermediate.
    ///
    /// # Correctness (exact, no estimation)
    ///
    /// Precondition: lanes `a, b ∈ [0, P)` (the `Add`/`Sub`/`Mul` impls all
    /// return canonical lanes, so every `mul_vec` input is canonical). Write
    /// `z = a*b`, so `0 ≤ z ≤ (P-1)^2 < 2^62`. All steps are exact integer
    /// identities; the only inequality used is the compile-time invariant
    /// `C(C+1) < P`, which gives `C^2 < P < 2^31` and `C(C+2) < 2^31`.
    ///
    /// 1. `h = sqdmulh(a,b) = floor(2z / 2^32) = floor(z / 2^31)`, exact
    ///    because `2z < 2^63` (no saturation), and `h ∈ [0, 2^31)`.
    /// 2. `z_lo31 = (z mod 2^32) & (2^31-1) = z mod 2^31`, so
    ///    `z = h·2^31 + z_lo31` exactly.
    /// 3. Since `2^31 = P + C ≡ C (mod P)`, `z ≡ C·h + z_lo31 =: t (mod P)`.
    /// 4. Fold `t`: `hh = sqdmulh(h, C) = floor(C·h / 2^31) ∈ [0, C)` (exact,
    ///    `2·h·C < 2^63`), and `ch_lo31 = (C·h mod 2^32) & (2^31-1)
    ///    = C·h mod 2^31`, so `C·h = hh·2^31 + ch_lo31`.
    /// 5. `s = ch_lo31 + z_lo31 < 2^32` (sum of two sub-`2^31` values, no u32
    ///    overflow). With `hp = hh + (s >> 31)` and `lo31p = s & (2^31-1)`,
    ///    `t = hh·2^31 + s = hp·2^31 + lo31p`, and `hp ≤ (C-1) + 1 = C`.
    /// 6. Fold again: `t ≡ C·hp + lo31p =: t' (mod P)`. Since `hp ≤ C`,
    ///    `C·hp ≤ C^2 < 2^31` (so `vmulq_u32(hp, C)` is exact, no wrap), and
    ///    `t' = C·hp + lo31p < C^2 + 2^31 < 2^32` (no u32 overflow).
    /// 7. `t' < C^2 + 2^31 ≤ 2P` because `C(C+2) < 2^31 ⇔ C^2 + 2^31 < 2P`.
    ///    Thus `t' ≡ z (mod P)` and `t' ∈ [0, 2P)`, i.e. `t' ∈ {r, r+P}` for
    ///    `r = z mod P`. The final `vminq_u32(t', t' - P)` (wrapping sub)
    ///    returns the canonical `r ∈ [0, P)`.
    #[inline(always)]
    unsafe fn mul_pmersenne31_vec(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe {
            let mask31 = vdupq_n_u32((1u32 << 31) - 1);
            let cvec = vdupq_n_u32(Self::C);
            let p = vdupq_n_u32(P);

            // Step 1-2: high/low split of z = a*b.
            let h = vreinterpretq_u32_s32(vqdmulhq_s32(
                vreinterpretq_s32_u32(a),
                vreinterpretq_s32_u32(b),
            ));
            let z_lo31 = vandq_u32(vmulq_u32(a, b), mask31);

            // Step 3-5: first Solinas fold t = C*h + z_lo31 = hp*2^31 + lo31p.
            let hh = vreinterpretq_u32_s32(vqdmulhq_s32(
                vreinterpretq_s32_u32(h),
                vreinterpretq_s32_u32(cvec),
            ));
            let ch_lo31 = vandq_u32(vmulq_u32(h, cvec), mask31);
            let s = vaddq_u32(ch_lo31, z_lo31);
            let hp = vaddq_u32(hh, vshrq_n_u32::<31>(s));
            let lo31p = vandq_u32(s, mask31);

            // Step 6-7: second fold t' = C*hp + lo31p in [0, 2P), canonicalize.
            let tprime = vaddq_u32(vmulq_u32(hp, cvec), lo31p);
            vminq_u32(tprime, vsubq_u32(tprime, p))
        }
    }

    #[inline(always)]
    fn add_u64_with_carry(
        sum: uint64x2_t,
        rhs: uint64x2_t,
        carry: uint64x2_t,
    ) -> (uint64x2_t, uint64x2_t) {
        unsafe {
            let next = vaddq_u64(sum, rhs);
            let overflow = vcltq_u64(next, sum);
            (next, vaddq_u64(carry, mask_to_bit(overflow)))
        }
    }

    #[inline(always)]
    fn carry_correction(carry: uint64x2_t) -> uint64x2_t {
        unsafe { vmull_u32(vmovn_u64(carry), vdup_n_u32(Fp32::<P>::SHIFT64_MOD_P)) }
    }

    #[inline(always)]
    fn dot_product_4_vec(a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> uint32x4_t {
        unsafe {
            let mut sum_lo = vmull_u32(vget_low_u32(a[0]), vget_low_u32(b[0]));
            let mut sum_hi = vmull_high_u32(a[0], b[0]);

            if Self::BITS <= 31 {
                sum_lo = vaddq_u64(sum_lo, vmull_u32(vget_low_u32(a[1]), vget_low_u32(b[1])));
                sum_hi = vaddq_u64(sum_hi, vmull_high_u32(a[1], b[1]));
                sum_lo = vaddq_u64(sum_lo, vmull_u32(vget_low_u32(a[2]), vget_low_u32(b[2])));
                sum_hi = vaddq_u64(sum_hi, vmull_high_u32(a[2], b[2]));
                sum_lo = vaddq_u64(sum_lo, vmull_u32(vget_low_u32(a[3]), vget_low_u32(b[3])));
                sum_hi = vaddq_u64(sum_hi, vmull_high_u32(a[3], b[3]));

                return Self::solinas_reduce(sum_lo, sum_hi);
            }

            let mut carry_lo = vdupq_n_u64(0);
            let mut carry_hi = vdupq_n_u64(0);

            let prod_lo_1 = vmull_u32(vget_low_u32(a[1]), vget_low_u32(b[1]));
            let prod_hi_1 = vmull_high_u32(a[1], b[1]);
            (sum_lo, carry_lo) = Self::add_u64_with_carry(sum_lo, prod_lo_1, carry_lo);
            (sum_hi, carry_hi) = Self::add_u64_with_carry(sum_hi, prod_hi_1, carry_hi);

            let prod_lo_2 = vmull_u32(vget_low_u32(a[2]), vget_low_u32(b[2]));
            let prod_hi_2 = vmull_high_u32(a[2], b[2]);
            (sum_lo, carry_lo) = Self::add_u64_with_carry(sum_lo, prod_lo_2, carry_lo);
            (sum_hi, carry_hi) = Self::add_u64_with_carry(sum_hi, prod_hi_2, carry_hi);

            let prod_lo_3 = vmull_u32(vget_low_u32(a[3]), vget_low_u32(b[3]));
            let prod_hi_3 = vmull_high_u32(a[3], b[3]);
            (sum_lo, carry_lo) = Self::add_u64_with_carry(sum_lo, prod_lo_3, carry_lo);
            (sum_hi, carry_hi) = Self::add_u64_with_carry(sum_hi, prod_hi_3, carry_hi);

            Self::solinas_reduce_with_carry(sum_lo, sum_hi, carry_lo, carry_hi)
        }
    }

    #[inline(always)]
    fn dot_product_3_vec(a: [uint32x4_t; 3], b: [uint32x4_t; 3]) -> uint32x4_t {
        unsafe {
            let mut sum_lo = vmull_u32(vget_low_u32(a[0]), vget_low_u32(b[0]));
            let mut sum_hi = vmull_high_u32(a[0], b[0]);

            if Self::BITS <= 31 {
                sum_lo = vaddq_u64(sum_lo, vmull_u32(vget_low_u32(a[1]), vget_low_u32(b[1])));
                sum_hi = vaddq_u64(sum_hi, vmull_high_u32(a[1], b[1]));
                sum_lo = vaddq_u64(sum_lo, vmull_u32(vget_low_u32(a[2]), vget_low_u32(b[2])));
                sum_hi = vaddq_u64(sum_hi, vmull_high_u32(a[2], b[2]));

                return Self::solinas_reduce(sum_lo, sum_hi);
            }

            let mut carry_lo = vdupq_n_u64(0);
            let mut carry_hi = vdupq_n_u64(0);

            let prod_lo_1 = vmull_u32(vget_low_u32(a[1]), vget_low_u32(b[1]));
            let prod_hi_1 = vmull_high_u32(a[1], b[1]);
            (sum_lo, carry_lo) = Self::add_u64_with_carry(sum_lo, prod_lo_1, carry_lo);
            (sum_hi, carry_hi) = Self::add_u64_with_carry(sum_hi, prod_hi_1, carry_hi);

            let prod_lo_2 = vmull_u32(vget_low_u32(a[2]), vget_low_u32(b[2]));
            let prod_hi_2 = vmull_high_u32(a[2], b[2]);
            (sum_lo, carry_lo) = Self::add_u64_with_carry(sum_lo, prod_lo_2, carry_lo);
            (sum_hi, carry_hi) = Self::add_u64_with_carry(sum_hi, prod_hi_2, carry_hi);

            Self::solinas_reduce_with_carry(sum_lo, sum_hi, carry_lo, carry_hi)
        }
    }

    #[inline(always)]
    fn mul_nr_vec<C>(x: uint32x4_t) -> uint32x4_t
    where
        C: FpExt2Config<Fp32<P>>,
    {
        if C::IS_NEG_ONE {
            Self::sub_vec(unsafe { vdupq_n_u32(0) }, x)
        } else if C::non_residue().0 == 2 {
            Self::add_vec(x, x)
        } else {
            C::mul_non_residue(Self::from_vec(x), Self::broadcast).to_vec()
        }
    }

    #[inline(always)]
    fn mul_c_u64(hi: uint64x2_t, c: uint32x2_t) -> uint64x2_t {
        unsafe {
            if Self::C == 1 {
                return hi;
            }
            if Self::C == 3 {
                return vaddq_u64(vshlq_n_u64::<1>(hi), hi);
            }
            if Self::C == 19 {
                return vaddq_u64(vaddq_u64(vshlq_n_u64::<4>(hi), vshlq_n_u64::<1>(hi)), hi);
            }
            if Self::C == 35 {
                return vaddq_u64(vaddq_u64(vshlq_n_u64::<5>(hi), vshlq_n_u64::<1>(hi)), hi);
            }
            if Self::C == 99 {
                return vaddq_u64(
                    vaddq_u64(vshlq_n_u64::<6>(hi), vshlq_n_u64::<5>(hi)),
                    vaddq_u64(vshlq_n_u64::<1>(hi), hi),
                );
            }
            let lo = vmull_u32(vmovn_u64(hi), c);
            let hi = vmull_u32(vmovn_u64(vshrq_n_u64::<32>(hi)), c);
            vaddq_u64(lo, vshlq_n_u64::<32>(hi))
        }
    }

    #[inline(always)]
    fn solinas_reduce(prod_lo: uint64x2_t, prod_hi: uint64x2_t) -> uint32x4_t {
        unsafe {
            if Self::BITS == 31 {
                return Self::solinas_reduce_bits31(prod_lo, prod_hi);
            }

            let mask = vdupq_n_u64(Self::MASK_U64);
            let neg_bits = vdupq_n_s64(-(Self::BITS as i64));
            let c = vdup_n_u32(Self::C);

            let f1_lo = vaddq_u64(
                vandq_u64(prod_lo, mask),
                Self::mul_c_u64(vshlq_u64(prod_lo, neg_bits), c),
            );
            let f1_hi = vaddq_u64(
                vandq_u64(prod_hi, mask),
                Self::mul_c_u64(vshlq_u64(prod_hi, neg_bits), c),
            );

            let f2_lo = vaddq_u64(
                vandq_u64(f1_lo, mask),
                Self::mul_c_u64(vshlq_u64(f1_lo, neg_bits), c),
            );
            let f2_hi = vaddq_u64(
                vandq_u64(f1_hi, mask),
                Self::mul_c_u64(vshlq_u64(f1_hi, neg_bits), c),
            );

            if Self::BITS < 32 {
                let (reduced_lo, reduced_hi) = if Self::TWO_FOLD_FOUR_PRODUCT_OK {
                    (f2_lo, f2_hi)
                } else {
                    (
                        vaddq_u64(
                            vandq_u64(f2_lo, mask),
                            Self::mul_c_u64(vshlq_u64(f2_lo, neg_bits), c),
                        ),
                        vaddq_u64(
                            vandq_u64(f2_hi, mask),
                            Self::mul_c_u64(vshlq_u64(f2_hi, neg_bits), c),
                        ),
                    )
                };

                let result = vcombine_u32(vmovn_u64(reduced_lo), vmovn_u64(reduced_hi));
                let p = vdupq_n_u32(P);
                vminq_u32(result, vsubq_u32(result, p))
            } else {
                let p_u64 = vdupq_n_u64(P as u64);

                let red_lo = vsubq_u64(f2_lo, p_u64);
                let keep_lo = vcltq_u64(f2_lo, p_u64);
                let out_lo = vbslq_u64(keep_lo, f2_lo, red_lo);

                let red_hi = vsubq_u64(f2_hi, p_u64);
                let keep_hi = vcltq_u64(f2_hi, p_u64);
                let out_hi = vbslq_u64(keep_hi, f2_hi, red_hi);

                vcombine_u32(vmovn_u64(out_lo), vmovn_u64(out_hi))
            }
        }
    }

    #[inline(always)]
    fn solinas_reduce_bits31(prod_lo: uint64x2_t, prod_hi: uint64x2_t) -> uint32x4_t {
        unsafe {
            let mask = vdupq_n_u64((1u64 << 31) - 1);
            let c = vdup_n_u32(Self::C);

            let f1_lo = vaddq_u64(
                vandq_u64(prod_lo, mask),
                Self::mul_c_u64(vshrq_n_u64::<31>(prod_lo), c),
            );
            let f1_hi = vaddq_u64(
                vandq_u64(prod_hi, mask),
                Self::mul_c_u64(vshrq_n_u64::<31>(prod_hi), c),
            );

            let f2_lo = vaddq_u64(
                vandq_u64(f1_lo, mask),
                Self::mul_c_u64(vshrq_n_u64::<31>(f1_lo), c),
            );
            let f2_hi = vaddq_u64(
                vandq_u64(f1_hi, mask),
                Self::mul_c_u64(vshrq_n_u64::<31>(f1_hi), c),
            );

            let (reduced_lo, reduced_hi) = if Self::TWO_FOLD_FOUR_PRODUCT_OK {
                (f2_lo, f2_hi)
            } else {
                (
                    vaddq_u64(
                        vandq_u64(f2_lo, mask),
                        Self::mul_c_u64(vshrq_n_u64::<31>(f2_lo), c),
                    ),
                    vaddq_u64(
                        vandq_u64(f2_hi, mask),
                        Self::mul_c_u64(vshrq_n_u64::<31>(f2_hi), c),
                    ),
                )
            };

            let result = vcombine_u32(vmovn_u64(reduced_lo), vmovn_u64(reduced_hi));
            let p = vdupq_n_u32(P);
            vminq_u32(result, vsubq_u32(result, p))
        }
    }

    #[inline(always)]
    fn solinas_reduce_with_carry(
        prod_lo: uint64x2_t,
        prod_hi: uint64x2_t,
        carry_lo: uint64x2_t,
        carry_hi: uint64x2_t,
    ) -> uint32x4_t {
        unsafe {
            if Self::BITS == 31 {
                return Self::solinas_reduce_with_carry_bits31(
                    prod_lo, prod_hi, carry_lo, carry_hi,
                );
            }

            let mask = vdupq_n_u64(Self::MASK_U64);
            let neg_bits = vdupq_n_s64(-(Self::BITS as i64));
            let c = vdup_n_u32(Self::C);

            let f1_lo = vaddq_u64(
                vaddq_u64(
                    vandq_u64(prod_lo, mask),
                    Self::mul_c_u64(vshlq_u64(prod_lo, neg_bits), c),
                ),
                Self::carry_correction(carry_lo),
            );
            let f1_hi = vaddq_u64(
                vaddq_u64(
                    vandq_u64(prod_hi, mask),
                    Self::mul_c_u64(vshlq_u64(prod_hi, neg_bits), c),
                ),
                Self::carry_correction(carry_hi),
            );

            let f2_lo = vaddq_u64(
                vandq_u64(f1_lo, mask),
                Self::mul_c_u64(vshlq_u64(f1_lo, neg_bits), c),
            );
            let f2_hi = vaddq_u64(
                vandq_u64(f1_hi, mask),
                Self::mul_c_u64(vshlq_u64(f1_hi, neg_bits), c),
            );

            if Self::BITS < 32 {
                let (reduced_lo, reduced_hi) = if Self::TWO_FOLD_FOUR_PRODUCT_OK {
                    (f2_lo, f2_hi)
                } else {
                    (
                        vaddq_u64(
                            vandq_u64(f2_lo, mask),
                            Self::mul_c_u64(vshlq_u64(f2_lo, neg_bits), c),
                        ),
                        vaddq_u64(
                            vandq_u64(f2_hi, mask),
                            Self::mul_c_u64(vshlq_u64(f2_hi, neg_bits), c),
                        ),
                    )
                };

                let result = vcombine_u32(vmovn_u64(reduced_lo), vmovn_u64(reduced_hi));
                let p = vdupq_n_u32(P);
                vminq_u32(result, vsubq_u32(result, p))
            } else {
                let p_u64 = vdupq_n_u64(P as u64);

                let red_lo = vsubq_u64(f2_lo, p_u64);
                let keep_lo = vcltq_u64(f2_lo, p_u64);
                let out_lo = vbslq_u64(keep_lo, f2_lo, red_lo);

                let red_hi = vsubq_u64(f2_hi, p_u64);
                let keep_hi = vcltq_u64(f2_hi, p_u64);
                let out_hi = vbslq_u64(keep_hi, f2_hi, red_hi);

                vcombine_u32(vmovn_u64(out_lo), vmovn_u64(out_hi))
            }
        }
    }

    /// `solinas_reduce_with_carry` specialised for `BITS == 31` (Mersenne31
    /// and any pseudo-Mersenne `Fp32<P>` with `P = 2^31 - C`). Sibling of
    /// `solinas_reduce_bits31`: a separate function that swaps the
    /// variable-amount `vshlq_u64(.., neg_bits)` for the immediate-shift
    /// `vshrq_n_u64::<31>`, reducing shift-count register pressure and
    /// dispatch port pressure. Since `BITS == 31` implies `BITS < 32`, the
    /// `else` branch of the canonicalisation can be dropped.
    #[inline(always)]
    fn solinas_reduce_with_carry_bits31(
        prod_lo: uint64x2_t,
        prod_hi: uint64x2_t,
        carry_lo: uint64x2_t,
        carry_hi: uint64x2_t,
    ) -> uint32x4_t {
        unsafe {
            let mask = vdupq_n_u64((1u64 << 31) - 1);
            let c = vdup_n_u32(Self::C);

            // Fold 1 with carry correction
            let f1_lo = vaddq_u64(
                vaddq_u64(
                    vandq_u64(prod_lo, mask),
                    Self::mul_c_u64(vshrq_n_u64::<31>(prod_lo), c),
                ),
                Self::carry_correction(carry_lo),
            );
            let f1_hi = vaddq_u64(
                vaddq_u64(
                    vandq_u64(prod_hi, mask),
                    Self::mul_c_u64(vshrq_n_u64::<31>(prod_hi), c),
                ),
                Self::carry_correction(carry_hi),
            );

            // Fold 2
            let f2_lo = vaddq_u64(
                vandq_u64(f1_lo, mask),
                Self::mul_c_u64(vshrq_n_u64::<31>(f1_lo), c),
            );
            let f2_hi = vaddq_u64(
                vandq_u64(f1_hi, mask),
                Self::mul_c_u64(vshrq_n_u64::<31>(f1_hi), c),
            );

            let (reduced_lo, reduced_hi) = if Self::TWO_FOLD_FOUR_PRODUCT_OK {
                (f2_lo, f2_hi)
            } else {
                (
                    vaddq_u64(
                        vandq_u64(f2_lo, mask),
                        Self::mul_c_u64(vshrq_n_u64::<31>(f2_lo), c),
                    ),
                    vaddq_u64(
                        vandq_u64(f2_hi, mask),
                        Self::mul_c_u64(vshrq_n_u64::<31>(f2_hi), c),
                    ),
                )
            };

            let result = vcombine_u32(vmovn_u64(reduced_lo), vmovn_u64(reduced_hi));
            let p = vdupq_n_u32(P);
            vminq_u32(result, vsubq_u32(result, p))
        }
    }
}

impl<const P: u32> Default for PackedFp32Neon<P> {
    #[inline]
    fn default() -> Self {
        Self { vals: [0; 4] }
    }
}

impl<const P: u32> fmt::Debug for PackedFp32Neon<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackedFp32Neon").field(&self.vals).finish()
    }
}

impl<const P: u32> PartialEq for PackedFp32Neon<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.vals == other.vals
    }
}

impl<const P: u32> Eq for PackedFp32Neon<P> {}

impl<const P: u32> Add for PackedFp32Neon<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_vec(Self::add_vec(self.to_vec(), rhs.to_vec()))
    }
}

impl<const P: u32> Sub for PackedFp32Neon<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_vec(Self::sub_vec(self.to_vec(), rhs.to_vec()))
    }
}

impl<const P: u32> Mul for PackedFp32Neon<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_vec(Self::mul_vec(self.to_vec(), rhs.to_vec()))
    }
}

impl<const P: u32> PackedValue for PackedFp32Neon<P> {
    type Value = Fp32<P>;
    const WIDTH: usize = FP32_WIDTH;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        Self {
            vals: [f(0).0, f(1).0, f(2).0, f(3).0],
        }
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Value {
        debug_assert!(lane < FP32_WIDTH);
        Fp32(self.vals[lane])
    }
}

impl<const P: u32> AddAssign for PackedFp32Neon<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u32> SubAssign for PackedFp32Neon<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u32> MulAssign for PackedFp32Neon<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u32> PackedField for PackedFp32Neon<P> {
    type Scalar = Fp32<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self { vals: [value.0; 4] }
    }

    #[inline(always)]
    fn fp_ext2_mul<C>(a0: Self, a1: Self, b0: Self, b1: Self) -> (Self, Self)
    where
        C: FpExt2Config<Self::Scalar>,
    {
        let a0 = a0.to_vec();
        let a1 = a1.to_vec();
        let b0 = b0.to_vec();
        let b1 = b1.to_vec();

        let v0 = Self::mul_vec(a0, b0);
        let v1 = Self::mul_vec(a1, b1);
        let cross = Self::mul_vec(Self::add_vec(a0, a1), Self::add_vec(b0, b1));

        (
            Self::from_vec(Self::add_vec(v0, Self::mul_nr_vec::<C>(v1))),
            Self::from_vec(Self::sub_vec(Self::sub_vec(cross, v0), v1)),
        )
    }

    #[inline(always)]
    fn fp_ext4_mul(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        let [a0, a1, a2, a3] = a.map(Self::to_vec);
        let [b0, b1, b2, b3] = b.map(Self::to_vec);
        let two_b1 = Self::add_vec(b1, b1);
        let two_b2 = Self::add_vec(b2, b2);
        let two_b3 = Self::add_vec(b3, b3);
        let b0_plus_b2 = Self::add_vec(b0, b2);
        let b1_plus_b3 = Self::add_vec(b1, b3);
        let b1_minus_b3 = Self::sub_vec(b1, b3);
        let b0_minus_b2 = Self::sub_vec(b0, b2);
        [
            Self::from_vec(Self::dot_product_4_vec(
                [a0, a1, a2, a3],
                [b0, two_b1, two_b2, two_b3],
            )),
            Self::from_vec(Self::dot_product_4_vec(
                [a0, a1, a2, a3],
                [b1, b0_plus_b2, b1_plus_b3, b2],
            )),
            Self::from_vec(Self::dot_product_4_vec(
                [a0, a1, a2, a3],
                [b2, b1_plus_b3, b0, b1_minus_b3],
            )),
            Self::from_vec(Self::dot_product_4_vec(
                [a0, a1, a2, a3],
                [b3, b2, b1_minus_b3, b0_minus_b2],
            )),
        ]
    }

    #[inline(always)]
    fn fp_ext4_square(a: [Self; 4]) -> [Self; 4] {
        let [a0, a1, a2, a3] = a.map(Self::to_vec);
        let zero = unsafe { vdupq_n_u32(0) };
        let two_a1 = Self::add_vec(a1, a1);
        let two_a2 = Self::add_vec(a2, a2);
        let two_a3 = Self::add_vec(a3, a3);
        let neg_a3 = Self::sub_vec(zero, a3);
        let neg_two_a3 = Self::sub_vec(zero, two_a3);
        [
            Self::from_vec(Self::dot_product_4_vec(
                [a0, a1, a2, a3],
                [a0, two_a1, two_a2, two_a3],
            )),
            Self::from_vec(Self::dot_product_3_vec(
                [a0, a1, a2],
                [two_a1, two_a2, two_a3],
            )),
            Self::from_vec(Self::dot_product_4_vec(
                [a0, a1, a1, a3],
                [two_a2, a1, two_a3, neg_a3],
            )),
            Self::from_vec(Self::dot_product_3_vec(
                [a0, a1, a2],
                [two_a3, two_a2, neg_two_a3],
            )),
        ]
    }

    #[inline(always)]
    fn fp_ext4_inverse(a: [Self; 4]) -> Option<[Self; 4]>
    where
        Self::Scalar: Invertible,
    {
        let [a0, a1, a2, a3] = a.map(Self::to_vec);
        let zero = unsafe { vdupq_n_u32(0) };
        let x0 = a0;
        let x1 = a2;
        let y0 = Self::sub_vec(a1, a3);
        let y1 = a3;

        let x1_square = Self::mul_vec(x1, x1);
        let y1_square = Self::mul_vec(y1, y1);
        let aa0 = Self::add_vec(Self::mul_vec(x0, x0), Self::add_vec(x1_square, x1_square));
        let aa1 = {
            let x0x1 = Self::mul_vec(x0, x1);
            Self::add_vec(x0x1, x0x1)
        };
        let bb0 = Self::add_vec(Self::mul_vec(y0, y0), Self::add_vec(y1_square, y1_square));
        let bb1 = {
            let y0y1 = Self::mul_vec(y0, y1);
            Self::add_vec(y0y1, y0y1)
        };
        let nr_bb0 = Self::add_vec(Self::add_vec(bb0, bb0), Self::add_vec(bb1, bb1));
        let nr_bb1 = Self::add_vec(bb0, Self::add_vec(bb1, bb1));
        let norm0 = Self::sub_vec(aa0, nr_bb0);
        let norm1 = Self::sub_vec(aa1, nr_bb1);

        let inv_norm_base = {
            let norm1_square = Self::mul_vec(norm1, norm1);
            let norm_base = Self::sub_vec(
                Self::mul_vec(norm0, norm0),
                Self::add_vec(norm1_square, norm1_square),
            );
            Self::from_vec(norm_base).inverse()?.to_vec()
        };
        let inv_norm0 = Self::mul_vec(norm0, inv_norm_base);
        let inv_norm1 = Self::mul_vec(Self::sub_vec(zero, norm1), inv_norm_base);

        let v0 = Self::mul_vec(x0, inv_norm0);
        let v1 = Self::mul_vec(x1, inv_norm1);
        let constant0 = Self::add_vec(v0, Self::add_vec(v1, v1));
        let constant1 = Self::sub_vec(
            Self::sub_vec(
                Self::mul_vec(Self::add_vec(x0, x1), Self::add_vec(inv_norm0, inv_norm1)),
                v0,
            ),
            v1,
        );

        let neg_y0 = Self::sub_vec(zero, y0);
        let neg_y1 = Self::sub_vec(zero, y1);
        let w0 = Self::mul_vec(neg_y0, inv_norm0);
        let w1 = Self::mul_vec(neg_y1, inv_norm1);
        let e1_coeff0 = Self::add_vec(w0, Self::add_vec(w1, w1));
        let e1_coeff1 = Self::sub_vec(
            Self::sub_vec(
                Self::mul_vec(
                    Self::add_vec(neg_y0, neg_y1),
                    Self::add_vec(inv_norm0, inv_norm1),
                ),
                w0,
            ),
            w1,
        );

        Some([
            Self::from_vec(constant0),
            Self::from_vec(Self::add_vec(e1_coeff0, e1_coeff1)),
            Self::from_vec(constant1),
            Self::from_vec(e1_coeff1),
        ])
    }
}
