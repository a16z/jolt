use super::*;

/// Number of `Fp32` lanes in an AVX-512 packed vector.
pub(crate) const FP32_WIDTH: usize = 16;

/// AVX-512 packed arithmetic for `Fp32<P>`, processing 16 lanes.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PackedFp32Avx512<const P: u32>(pub [Fp32<P>; FP32_WIDTH]);

impl<const P: u32> PackedFp32Avx512<P> {
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

    /// Whether two Solinas folds suffice to bring the sum of four
    /// `(P-1)^2` products into `[0, 2*P)` for the final canonicalize step.
    /// Mirrors `PackedFp32Neon::TWO_FOLD_FOUR_PRODUCT_OK`. When `false`,
    /// `solinas_reduce` must do a third fold before handing off to
    /// `pack_and_canonicalize`.
    const TWO_FOLD_FOUR_PRODUCT_OK: bool = {
        let c = Self::C as u64;
        4 * c * c + 3 * c <= (1u64 << Self::BITS)
    };

    #[inline(always)]
    fn to_vec(self) -> __m512i {
        unsafe { transmute(self) }
    }

    #[inline(always)]
    unsafe fn from_vec(v: __m512i) -> Self {
        unsafe { transmute(v) }
    }

    /// Multiply each `u64` lane by `C`. Building block of Solinas reduction;
    /// the `C == 1` fast path skips the multiply entirely for Mersenne-like
    /// primes.
    ///
    /// Uses `_mm512_mullo_epi64` (AVX-512DQ, single `vpmullq`) for full
    /// 64-bit width. The previous implementation used `_mm512_mul_epu32`
    /// which only reads the *low 32 bits* of each lane and silently dropped
    /// bit 32+ of the input — fine for `BITS == 32` (where the caller's
    /// `prod >> 32` always fits in 32 bits) but wrong for `BITS == 31` and
    /// `C != 1` where `prod >> 31` can occupy 33 bits.
    #[inline(always)]
    unsafe fn mul_c_u64(x: __m512i) -> __m512i {
        if Self::C == 1 {
            x
        } else {
            let c_vec = _mm512_set1_epi64(Self::C as i64);
            _mm512_mullo_epi64(x, c_vec)
        }
    }

    /// One Solinas fold of a single 64-bit product lane (BITS == 32 only):
    /// `(x & (2^32-1)) + C*(x >> 32)`. For a single product `x < 2^64` the
    /// high word `x >> 32 < 2^32`, so the result is `< 2^32 + C*2^32 < 2^40`.
    /// Used by the `BITS == 32` dot-product path to pre-fold each product so
    /// that up to four folded terms (each `< 2^40`) sum below `2^42` without
    /// overflowing a `u64`, removing the per-product carry tracking.
    #[inline(always)]
    unsafe fn fold_product_once(x: __m512i) -> __m512i {
        let lo = _mm512_and_si512(x, _mm512_set1_epi64(Self::MASK_U64 as i64));
        let hi = _mm512_srli_epi64::<32>(x);
        _mm512_add_epi64(lo, Self::mul_c_u64(hi))
    }

    /// Plonky3-style Mersenne31 multiply (P = 2^31 - 1). Specialized using
    /// `_mm512_srli_epi64::<31>` shifts and 16-lane mask blends. Used by
    /// the `Mul` impl when `Self::BITS == 31 && Self::C == 1`.
    #[inline(always)]
    unsafe fn mul_mersenne31_vec(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            const EVENS: __mmask16 = 0b0101_0101_0101_0101;
            const ODDS: __mmask16 = 0b1010_1010_1010_1010;

            let lhs_evn_dbl = _mm512_add_epi32(a, a);
            let rhs_odd = movehdup_epi32_512(b);
            let lhs_odd_dbl = _mm512_srli_epi64::<31>(a);

            let prod_odd_dbl = _mm512_mul_epu32(lhs_odd_dbl, rhs_odd);
            let prod_evn_dbl = _mm512_mul_epu32(lhs_evn_dbl, b);

            let prod_lo_dbl =
                _mm512_mask_blend_epi32(ODDS, prod_evn_dbl, moveldup_epi32_512(prod_odd_dbl));
            let prod_hi =
                _mm512_mask_blend_epi32(EVENS, prod_odd_dbl, movehdup_epi32_512(prod_evn_dbl));
            let prod_lo = _mm512_srli_epi32::<1>(prod_lo_dbl);

            let p = _mm512_set1_epi32(P as i32);
            let folded = _mm512_add_epi32(prod_lo, prod_hi);
            _mm512_min_epu32(folded, _mm512_sub_epi32(folded, p))
        }
    }

    /// Vector form of field add: 16-lane add + canonicalize to `[0, P)`.
    /// Mirrors `PackedFp32Avx2::add_vec` with native AVX-512 mask compares.
    #[inline(always)]
    unsafe fn add_vec(a: __m512i, b: __m512i) -> __m512i {
        let p = _mm512_set1_epi32(P as i32);
        if Self::BITS <= 31 {
            let t = _mm512_add_epi32(a, b);
            let u = _mm512_sub_epi32(t, p);
            _mm512_min_epu32(t, u)
        } else {
            let c = _mm512_set1_epi32(Self::C as i32);
            let t = _mm512_add_epi32(a, b);
            let overflow = _mm512_cmplt_epu32_mask(t, a);
            let t2 = _mm512_mask_add_epi32(t, overflow, t, c);
            let geq_p = _mm512_cmpge_epu32_mask(t2, p);
            _mm512_mask_sub_epi32(t2, geq_p, t2, p)
        }
    }

    /// Vector form of field sub: 16-lane sub + canonicalize to `[0, P)`.
    /// Mirrors `PackedFp32Avx2::sub_vec` with native AVX-512 mask compares.
    #[inline(always)]
    unsafe fn sub_vec(a: __m512i, b: __m512i) -> __m512i {
        let p = _mm512_set1_epi32(P as i32);
        if Self::BITS <= 31 {
            let t = _mm512_sub_epi32(a, b);
            let u = _mm512_add_epi32(t, p);
            _mm512_min_epu32(t, u)
        } else {
            let t = _mm512_sub_epi32(a, b);
            let underflow = _mm512_cmplt_epu32_mask(a, b);
            _mm512_mask_add_epi32(t, underflow, t, p)
        }
    }

    /// Vector form of field mul: 16-lane Solinas multiply + canonicalize.
    #[inline(always)]
    unsafe fn mul_vec(a: __m512i, b: __m512i) -> __m512i {
        let prod_evn = _mm512_mul_epu32(a, b);
        let a_odd = movehdup_epi32_512(a);
        let b_odd = movehdup_epi32_512(b);
        let prod_odd = _mm512_mul_epu32(a_odd, b_odd);
        Self::solinas_reduce(prod_evn, prod_odd)
    }

    /// 4-way fused multiply-accumulate with a single end-reduction.
    /// Mirrors `PackedFp32Avx2::dot_product_4_vec` at 16 lanes. For
    /// `BITS <= 31`, four `(2^31 - 1)^2` products sum below `2^64`, so the
    /// raw products accumulate without overflow. For `BITS == 32`, each
    /// product is pre-folded once (`< 2^40`) so four folds sum below `2^42`,
    /// again overflow-free. Both branches end in a single carry-free
    /// `solinas_reduce`; the `if` is a const condition resolved at compile
    /// time.
    #[inline(always)]
    unsafe fn dot_product_4_vec(a: [__m512i; 4], b: [__m512i; 4]) -> __m512i {
        let mut sum_evn = _mm512_mul_epu32(a[0], b[0]);
        let mut sum_odd = _mm512_mul_epu32(movehdup_epi32_512(a[0]), movehdup_epi32_512(b[0]));

        if Self::BITS <= 31 {
            for i in 1..4 {
                let prod_evn = _mm512_mul_epu32(a[i], b[i]);
                let prod_odd = _mm512_mul_epu32(movehdup_epi32_512(a[i]), movehdup_epi32_512(b[i]));
                sum_evn = _mm512_add_epi64(sum_evn, prod_evn);
                sum_odd = _mm512_add_epi64(sum_odd, prod_odd);
            }
            return Self::solinas_reduce(sum_evn, sum_odd);
        }

        // BITS == 32: four 32-bit products overflow a `u64` sum, so pre-fold each
        // product once (`< 2^40`) and accumulate the folds (`< 4*2^40 < 2^42`),
        // which is carry-free, then a single carry-free `solinas_reduce`.
        let mut sum_evn = Self::fold_product_once(sum_evn);
        let mut sum_odd = Self::fold_product_once(sum_odd);
        for i in 1..4 {
            let prod_evn = Self::fold_product_once(_mm512_mul_epu32(a[i], b[i]));
            let prod_odd = Self::fold_product_once(_mm512_mul_epu32(
                movehdup_epi32_512(a[i]),
                movehdup_epi32_512(b[i]),
            ));
            sum_evn = _mm512_add_epi64(sum_evn, prod_evn);
            sum_odd = _mm512_add_epi64(sum_odd, prod_odd);
        }
        Self::solinas_reduce(sum_evn, sum_odd)
    }

    /// 3-way fused multiply-accumulate with a single end-reduction.
    #[inline(always)]
    unsafe fn dot_product_3_vec(a: [__m512i; 3], b: [__m512i; 3]) -> __m512i {
        let mut sum_evn = _mm512_mul_epu32(a[0], b[0]);
        let mut sum_odd = _mm512_mul_epu32(movehdup_epi32_512(a[0]), movehdup_epi32_512(b[0]));

        if Self::BITS <= 31 {
            for i in 1..3 {
                let prod_evn = _mm512_mul_epu32(a[i], b[i]);
                let prod_odd = _mm512_mul_epu32(movehdup_epi32_512(a[i]), movehdup_epi32_512(b[i]));
                sum_evn = _mm512_add_epi64(sum_evn, prod_evn);
                sum_odd = _mm512_add_epi64(sum_odd, prod_odd);
            }
            return Self::solinas_reduce(sum_evn, sum_odd);
        }

        // BITS == 32: pre-fold (see `dot_product_4_vec`).
        let mut sum_evn = Self::fold_product_once(sum_evn);
        let mut sum_odd = Self::fold_product_once(sum_odd);
        for i in 1..3 {
            let prod_evn = Self::fold_product_once(_mm512_mul_epu32(a[i], b[i]));
            let prod_odd = Self::fold_product_once(_mm512_mul_epu32(
                movehdup_epi32_512(a[i]),
                movehdup_epi32_512(b[i]),
            ));
            sum_evn = _mm512_add_epi64(sum_evn, prod_evn);
            sum_odd = _mm512_add_epi64(sum_odd, prod_odd);
        }
        Self::solinas_reduce(sum_evn, sum_odd)
    }

    /// Multiply by an `FpExt2` non-residue. Recognizes `nr == -1` and `nr == 2`
    /// fast paths.
    #[inline(always)]
    unsafe fn mul_nr_vec<C>(x: __m512i) -> __m512i
    where
        C: FpExt2Config<Fp32<P>>,
    {
        if C::IS_NEG_ONE {
            Self::sub_vec(_mm512_setzero_si512(), x)
        } else if C::non_residue().0 == 2 {
            Self::add_vec(x, x)
        } else {
            C::mul_non_residue(Self::from_vec(x), Self::broadcast).to_vec()
        }
    }

    /// Two-or-three-fold Solinas reduction of 8+8 `u64` products → 16 `u32`
    /// lanes.
    ///
    /// The `Self::BITS == 31` branches use immediate-shift
    /// `_mm512_srli_epi64::<31>` instead of the generic variable-shift
    /// `_mm512_srl_epi64(.., shift)`, mirroring the same specialisation
    /// the base-field `Mul` impl uses on Mersenne31, so extension-field
    /// operations on Mersenne31 get the same per-shift win.
    ///
    /// Two folds always suffice when `Self::TWO_FOLD_FOUR_PRODUCT_OK`. When
    /// it doesn't (large `C` such that `4*C^2 + 3*C > 2^BITS`), we run a
    /// third fold so `pack_and_canonicalize`'s single subtract-and-min step
    /// is enough to land in `[0, P)`. Mirrors `PackedFp32Neon::solinas_reduce`.
    #[inline(always)]
    unsafe fn solinas_reduce(prod_evn: __m512i, prod_odd: __m512i) -> __m512i {
        let mask = _mm512_set1_epi64(Self::MASK_U64 as i64);
        let shift = _mm_set_epi64x(0, Self::BITS as i64);

        // Fold 1
        let evn_lo = _mm512_and_si512(prod_evn, mask);
        let evn_hi = if Self::BITS == 31 {
            _mm512_srli_epi64::<31>(prod_evn)
        } else {
            _mm512_srl_epi64(prod_evn, shift)
        };
        let evn_f1 = _mm512_add_epi64(evn_lo, Self::mul_c_u64(evn_hi));

        let odd_lo = _mm512_and_si512(prod_odd, mask);
        let odd_hi = if Self::BITS == 31 {
            _mm512_srli_epi64::<31>(prod_odd)
        } else {
            _mm512_srl_epi64(prod_odd, shift)
        };
        let odd_f1 = _mm512_add_epi64(odd_lo, Self::mul_c_u64(odd_hi));

        // Fold 2
        let evn_f1_lo = _mm512_and_si512(evn_f1, mask);
        let evn_f1_hi = if Self::BITS == 31 {
            _mm512_srli_epi64::<31>(evn_f1)
        } else {
            _mm512_srl_epi64(evn_f1, shift)
        };
        let evn_f2 = _mm512_add_epi64(evn_f1_lo, Self::mul_c_u64(evn_f1_hi));

        let odd_f1_lo = _mm512_and_si512(odd_f1, mask);
        let odd_f1_hi = if Self::BITS == 31 {
            _mm512_srli_epi64::<31>(odd_f1)
        } else {
            _mm512_srl_epi64(odd_f1, shift)
        };
        let odd_f2 = _mm512_add_epi64(odd_f1_lo, Self::mul_c_u64(odd_f1_hi));

        // Optional third fold for large-C primes (e.g. Generic31Offset32787)
        // where two folds leave residue > 2*P.
        let (evn_final, odd_final) = if Self::TWO_FOLD_FOUR_PRODUCT_OK {
            (evn_f2, odd_f2)
        } else {
            let evn_f2_lo = _mm512_and_si512(evn_f2, mask);
            let evn_f2_hi = if Self::BITS == 31 {
                _mm512_srli_epi64::<31>(evn_f2)
            } else {
                _mm512_srl_epi64(evn_f2, shift)
            };
            let odd_f2_lo = _mm512_and_si512(odd_f2, mask);
            let odd_f2_hi = if Self::BITS == 31 {
                _mm512_srli_epi64::<31>(odd_f2)
            } else {
                _mm512_srl_epi64(odd_f2, shift)
            };
            (
                _mm512_add_epi64(evn_f2_lo, Self::mul_c_u64(evn_f2_hi)),
                _mm512_add_epi64(odd_f2_lo, Self::mul_c_u64(odd_f2_hi)),
            )
        };

        Self::pack_and_canonicalize(evn_final, odd_final)
    }

    /// Combine 8+8 `u64` lanes into 16 `u32` lanes canonicalized to `[0, P)`.
    /// AVX-512 uses native unsigned compare masks and `_mm512_min_epu64`,
    /// simplifying both branches vs the AVX2 implementation.
    #[inline(always)]
    unsafe fn pack_and_canonicalize(evn_f2: __m512i, odd_f2: __m512i) -> __m512i {
        if Self::BITS < 32 {
            let odd_shifted = _mm512_slli_epi64::<32>(odd_f2);
            let combined = _mm512_mask_blend_epi32(0b1010_1010_1010_1010, evn_f2, odd_shifted);
            let p = _mm512_set1_epi32(P as i32);
            let reduced = _mm512_sub_epi32(combined, p);
            _mm512_min_epu32(combined, reduced)
        } else {
            let p_u64 = _mm512_set1_epi64(P as i64);

            let red_evn = _mm512_sub_epi64(evn_f2, p_u64);
            let out_evn = _mm512_min_epu64(evn_f2, red_evn);

            let red_odd = _mm512_sub_epi64(odd_f2, p_u64);
            let out_odd = _mm512_min_epu64(odd_f2, red_odd);

            let odd_shifted = _mm512_slli_epi64::<32>(out_odd);
            _mm512_mask_blend_epi32(0b1010_1010_1010_1010, out_evn, odd_shifted)
        }
    }
}

impl<const P: u32> Default for PackedFp32Avx512<P> {
    #[inline]
    fn default() -> Self {
        Self([Fp32(0); FP32_WIDTH])
    }
}

impl<const P: u32> fmt::Debug for PackedFp32Avx512<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackedFp32Avx512").field(&self.0).finish()
    }
}

impl<const P: u32> PartialEq for PackedFp32Avx512<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const P: u32> Eq for PackedFp32Avx512<P> {}

impl<const P: u32> Add for PackedFp32Avx512<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self::from_vec(Self::add_vec(self.to_vec(), rhs.to_vec())) }
    }
}

impl<const P: u32> Sub for PackedFp32Avx512<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self::from_vec(Self::sub_vec(self.to_vec(), rhs.to_vec())) }
    }
}

impl<const P: u32> Mul for PackedFp32Avx512<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let a = self.to_vec();
            let b = rhs.to_vec();

            if Self::BITS == 31 && Self::C == 1 {
                return Self::from_vec(Self::mul_mersenne31_vec(a, b));
            }

            let prod_evn = _mm512_mul_epu32(a, b);
            let a_odd = movehdup_epi32_512(a);
            let b_odd = movehdup_epi32_512(b);
            let prod_odd = _mm512_mul_epu32(a_odd, b_odd);

            let mask = _mm512_set1_epi64(Self::MASK_U64 as i64);
            let shift = _mm_set_epi64x(0, Self::BITS as i64);

            // Fold 1
            let evn_lo = _mm512_and_si512(prod_evn, mask);
            let evn_hi = if Self::BITS == 31 {
                _mm512_srli_epi64::<31>(prod_evn)
            } else {
                _mm512_srl_epi64(prod_evn, shift)
            };
            let evn_f1 = _mm512_add_epi64(evn_lo, Self::mul_c_u64(evn_hi));

            let odd_lo = _mm512_and_si512(prod_odd, mask);
            let odd_hi = if Self::BITS == 31 {
                _mm512_srli_epi64::<31>(prod_odd)
            } else {
                _mm512_srl_epi64(prod_odd, shift)
            };
            let odd_f1 = _mm512_add_epi64(odd_lo, Self::mul_c_u64(odd_hi));

            // Fold 2
            let evn_f1_lo = _mm512_and_si512(evn_f1, mask);
            let evn_f1_hi = if Self::BITS == 31 {
                _mm512_srli_epi64::<31>(evn_f1)
            } else {
                _mm512_srl_epi64(evn_f1, shift)
            };
            let evn_f2 = _mm512_add_epi64(evn_f1_lo, Self::mul_c_u64(evn_f1_hi));

            let odd_f1_lo = _mm512_and_si512(odd_f1, mask);
            let odd_f1_hi = if Self::BITS == 31 {
                _mm512_srli_epi64::<31>(odd_f1)
            } else {
                _mm512_srl_epi64(odd_f1, shift)
            };
            let odd_f2 = _mm512_add_epi64(odd_f1_lo, Self::mul_c_u64(odd_f1_hi));

            // Recombine + canonicalize. For `BITS == 32` the two-fold residue
            // can land in `[2^32, 2*P)` (up to `2^32 + C^2`), so the subtract
            // must happen on the full 64-bit lanes before packing; a 32-bit
            // recombine would drop bit 32. `pack_and_canonicalize` does the
            // 64-bit subtract for `BITS == 32` and is identical to the inline
            // 32-bit recombine for `BITS < 32`.
            Self::from_vec(Self::pack_and_canonicalize(evn_f2, odd_f2))
        }
    }
}

impl<const P: u32> PackedValue for PackedFp32Avx512<P> {
    type Value = Fp32<P>;
    const WIDTH: usize = FP32_WIDTH;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        Self([
            f(0),
            f(1),
            f(2),
            f(3),
            f(4),
            f(5),
            f(6),
            f(7),
            f(8),
            f(9),
            f(10),
            f(11),
            f(12),
            f(13),
            f(14),
            f(15),
        ])
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Value {
        debug_assert!(lane < FP32_WIDTH);
        self.0[lane]
    }
}

impl<const P: u32> AddAssign for PackedFp32Avx512<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u32> SubAssign for PackedFp32Avx512<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u32> MulAssign for PackedFp32Avx512<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u32> PackedField for PackedFp32Avx512<P> {
    type Scalar = Fp32<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self([value; FP32_WIDTH])
    }

    #[inline(always)]
    fn fp_ext2_mul<C>(a0: Self, a1: Self, b0: Self, b1: Self) -> (Self, Self)
    where
        C: FpExt2Config<Self::Scalar>,
    {
        unsafe {
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
    }

    #[inline(always)]
    fn fp_ext4_mul(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        unsafe {
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
    }

    #[inline(always)]
    fn fp_ext4_square(a: [Self; 4]) -> [Self; 4] {
        unsafe {
            let [a0, a1, a2, a3] = a.map(Self::to_vec);
            let zero = _mm512_setzero_si512();
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
    }

    #[inline(always)]
    fn fp_ext4_inverse(a: [Self; 4]) -> Option<[Self; 4]>
    where
        Self::Scalar: Invertible,
    {
        unsafe {
            let [a0, a1, a2, a3] = a.map(Self::to_vec);
            let zero = _mm512_setzero_si512();
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
}
