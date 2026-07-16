use super::*;

/// Number of `Fp64` lanes in an AVX-512 packed vector.
pub(crate) const FP64_WIDTH: usize = 8;

/// AVX-512 packed arithmetic for `Fp64<P>`, processing 8 lanes.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PackedFp64Avx512<const P: u64>(pub [Fp64<P>; FP64_WIDTH]);

impl<const P: u64> PackedFp64Avx512<P> {
    const BITS: u32 = 64 - P.leading_zeros();

    const C_LO: u64 = {
        let c = if Self::BITS == 64 {
            0u64.wrapping_sub(P)
        } else {
            (1u64 << Self::BITS) - P
        };
        assert!(P != 0, "modulus must be nonzero");
        assert!(P & 1 == 1, "modulus must be odd");
        c
    };

    const MASK64: u64 = if Self::BITS < 64 {
        (1u64 << Self::BITS) - 1
    } else {
        u64::MAX
    };

    #[inline(always)]
    fn to_vec(self) -> __m512i {
        unsafe { transmute(self) }
    }

    #[inline(always)]
    unsafe fn from_vec(v: __m512i) -> Self {
        unsafe { transmute(v) }
    }

    /// Vectorized 128-bit Solinas reduction for p = 2^BITS - C.
    /// Given (hi, lo) = 128-bit product, computes result ≡ (hi*2^64 + lo) mod p.
    #[inline]
    unsafe fn reduce128_vec(hi: __m512i, lo: __m512i) -> __m512i {
        if Self::BITS < 64 {
            Self::reduce128_small_k(hi, lo)
        } else {
            Self::reduce128_full_k(hi, lo)
        }
    }

    /// Reduction for BITS < 64 (e.g. 40-bit prime). No overflow issues: all
    /// intermediates fit in u64.
    #[inline]
    unsafe fn reduce128_small_k(hi: __m512i, lo: __m512i) -> __m512i {
        let mask_k = _mm512_set1_epi64(Self::MASK64 as i64);
        let c_vec = _mm512_set1_epi64(Self::C_LO as i64);
        let p_vec = _mm512_set1_epi64(P as i64);
        let shift_k = _mm_set_epi64x(0, Self::BITS as i64);
        let shift_64mk = _mm_set_epi64x(0, (64 - Self::BITS) as i64);

        let lo_k = _mm512_and_si512(lo, mask_k);
        let lo_upper = _mm512_srl_epi64(lo, shift_k);
        let hi_shifted = _mm512_sll_epi64(hi, shift_64mk);
        let hi_k = _mm512_or_si512(lo_upper, hi_shifted);

        // c * hi_k: hi_k may exceed 32 bits, split into lo32 and top
        let c_hi_lo = _mm512_mul_epu32(c_vec, hi_k);
        let hi_k_top = _mm512_srli_epi64::<32>(hi_k);
        let c_hi_top = _mm512_mul_epu32(c_vec, hi_k_top);
        let c_hi_top_shifted = _mm512_slli_epi64::<32>(c_hi_top);
        let c_hi_full = _mm512_add_epi64(c_hi_lo, c_hi_top_shifted);

        let fold1 = _mm512_add_epi64(lo_k, c_hi_full);

        let fold1_lo_k = _mm512_and_si512(fold1, mask_k);
        let fold1_hi = _mm512_srl_epi64(fold1, shift_k);
        let c_fold1_hi = _mm512_mul_epu32(c_vec, fold1_hi);
        let fold2 = _mm512_add_epi64(fold1_lo_k, c_fold1_hi);

        let reduced = _mm512_sub_epi64(fold2, p_vec);
        _mm512_min_epu64(fold2, reduced)
    }

    /// Reduction for BITS == 64 (e.g. p = 2^64 - 87). Tracks overflow from
    /// c*hi exceeding 64 bits, using native unsigned comparisons.
    #[inline]
    unsafe fn reduce128_full_k(hi: __m512i, lo: __m512i) -> __m512i {
        let c_vec = _mm512_set1_epi64(Self::C_LO as i64);
        let p_vec = _mm512_set1_epi64(P as i64);
        let one = _mm512_set1_epi64(1);

        // c * hi_lo32
        let c_hi_lo = _mm512_mul_epu32(c_vec, hi);
        // c * hi_hi32
        let hi_hi = _mm512_srli_epi64::<32>(hi);
        let c_hi_hi = _mm512_mul_epu32(c_vec, hi_hi);

        let c_hi_hi_lo32 = _mm512_slli_epi64::<32>(c_hi_hi);
        let c_hi_carry = _mm512_srli_epi64::<32>(c_hi_hi);

        // Lower 64 bits of c * hi
        let sum_lo = _mm512_add_epi64(c_hi_lo, c_hi_hi_lo32);
        let carry0 = _mm512_cmplt_epu64_mask(sum_lo, c_hi_lo);
        let overflow = _mm512_mask_add_epi64(c_hi_carry, carry0, c_hi_carry, one);

        // lo + sum_lo
        let s = _mm512_add_epi64(lo, sum_lo);
        let carry1 = _mm512_cmplt_epu64_mask(s, lo);
        let total_overflow = _mm512_mask_add_epi64(overflow, carry1, overflow, one);

        // Fold overflow: total_overflow * c (at most ~2^15)
        let final_corr = _mm512_mul_epu32(c_vec, total_overflow);
        let result = _mm512_add_epi64(s, final_corr);
        let carry_f = _mm512_cmplt_epu64_mask(result, s);
        let result = _mm512_mask_add_epi64(result, carry_f, result, c_vec);

        let ge_mask = _mm512_cmpge_epu64_mask(result, p_vec);
        _mm512_mask_sub_epi64(result, ge_mask, result, p_vec)
    }
}

impl<const P: u64> Default for PackedFp64Avx512<P> {
    #[inline]
    fn default() -> Self {
        Self([Fp64(0); FP64_WIDTH])
    }
}

impl<const P: u64> fmt::Debug for PackedFp64Avx512<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackedFp64Avx512").field(&self.0).finish()
    }
}

impl<const P: u64> PartialEq for PackedFp64Avx512<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const P: u64> Eq for PackedFp64Avx512<P> {}

impl<const P: u64> Add for PackedFp64Avx512<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            let a = self.to_vec();
            let b = rhs.to_vec();
            let p = _mm512_set1_epi64(P as i64);

            let result = if Self::BITS <= 62 {
                let s = _mm512_add_epi64(a, b);
                let geq_p = _mm512_cmpge_epu64_mask(s, p);
                _mm512_mask_sub_epi64(s, geq_p, s, p)
            } else {
                let s = _mm512_add_epi64(a, b);
                let overflow = _mm512_cmplt_epu64_mask(s, a);
                let c = _mm512_set1_epi64(Self::C_LO as i64);
                let geq_p = _mm512_cmpge_epu64_mask(s, p);
                let no_of = _mm512_mask_sub_epi64(s, geq_p, s, p);
                let s_plus_c = _mm512_add_epi64(s, c);
                _mm512_mask_blend_epi64(overflow, no_of, s_plus_c)
            };

            Self::from_vec(result)
        }
    }
}

impl<const P: u64> Sub for PackedFp64Avx512<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            let a = self.to_vec();
            let b = rhs.to_vec();
            let p = _mm512_set1_epi64(P as i64);
            let d = _mm512_sub_epi64(a, b);
            let underflow = _mm512_cmplt_epu64_mask(a, b);
            Self::from_vec(_mm512_mask_add_epi64(d, underflow, d, p))
        }
    }
}

impl<const P: u64> Mul for PackedFp64Avx512<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let (hi, lo) = mul64_64_512(self.to_vec(), rhs.to_vec());
            Self::from_vec(Self::reduce128_vec(hi, lo))
        }
    }
}

impl<const P: u64> PackedValue for PackedFp64Avx512<P> {
    type Value = Fp64<P>;
    const WIDTH: usize = FP64_WIDTH;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        Self([f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7)])
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Value {
        debug_assert!(lane < FP64_WIDTH);
        self.0[lane]
    }
}

impl<const P: u64> AddAssign for PackedFp64Avx512<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u64> SubAssign for PackedFp64Avx512<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u64> MulAssign for PackedFp64Avx512<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u64> PackedField for PackedFp64Avx512<P> {
    type Scalar = Fp64<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self([value; FP64_WIDTH])
    }
}
