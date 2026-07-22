use super::*;

/// Number of `Fp64` lanes in an AVX2 packed vector.
pub(crate) const FP64_WIDTH: usize = 4;

/// AVX2 packed arithmetic for `Fp64<P>`, processing 4 lanes.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PackedFp64Avx2<const P: u64>(pub [Fp64<P>; FP64_WIDTH]);

impl<const P: u64> PackedFp64Avx2<P> {
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
    fn to_vec(self) -> __m256i {
        unsafe { transmute(self) }
    }

    #[inline(always)]
    unsafe fn from_vec(v: __m256i) -> Self {
        unsafe { transmute(v) }
    }

    #[inline]
    unsafe fn reduce128_vec(hi: __m256i, lo: __m256i) -> __m256i {
        if Self::BITS < 64 {
            Self::reduce128_small_k(hi, lo)
        } else {
            Self::reduce128_full_k(hi, lo)
        }
    }

    /// Reduction for BITS < 64. All intermediates fit in u64 — no overflow.
    #[inline]
    unsafe fn reduce128_small_k(hi: __m256i, lo: __m256i) -> __m256i {
        let mask_k = _mm256_set1_epi64x(Self::MASK64 as i64);
        let c_vec = _mm256_set1_epi64x(Self::C_LO as i64);
        let p_vec = _mm256_set1_epi64x(P as i64);
        let shift_k = _mm_set_epi64x(0, Self::BITS as i64);
        let shift_64mk = _mm_set_epi64x(0, (64 - Self::BITS) as i64);

        let lo_k = _mm256_and_si256(lo, mask_k);
        let lo_upper = _mm256_srl_epi64(lo, shift_k);
        let hi_shifted = _mm256_sll_epi64(hi, shift_64mk);
        let hi_k = _mm256_or_si256(lo_upper, hi_shifted);

        let c_hi_lo = _mm256_mul_epu32(c_vec, hi_k);
        let hi_k_top = _mm256_srli_epi64::<32>(hi_k);
        let c_hi_top = _mm256_mul_epu32(c_vec, hi_k_top);
        let c_hi_top_shifted = _mm256_slli_epi64::<32>(c_hi_top);
        let c_hi_full = _mm256_add_epi64(c_hi_lo, c_hi_top_shifted);

        let fold1 = _mm256_add_epi64(lo_k, c_hi_full);

        let fold1_lo_k = _mm256_and_si256(fold1, mask_k);
        let fold1_hi = _mm256_srl_epi64(fold1, shift_k);
        let c_fold1_hi = _mm256_mul_epu32(c_vec, fold1_hi);
        let fold2 = _mm256_add_epi64(fold1_lo_k, c_fold1_hi);

        let reduced = _mm256_sub_epi64(fold2, p_vec);
        let sign = _mm256_set1_epi64x(i64::MIN);
        let fold2_s = _mm256_xor_si256(fold2, sign);
        let reduced_s = _mm256_xor_si256(reduced, sign);
        let fold2_lt = _mm256_cmpgt_epi64(reduced_s, fold2_s);
        _mm256_blendv_epi8(reduced, fold2, fold2_lt)
    }

    /// Reduction for BITS == 64. Uses XOR-with-SIGN_BIT trick for unsigned
    /// overflow detection.
    #[inline]
    unsafe fn reduce128_full_k(hi: __m256i, lo: __m256i) -> __m256i {
        let c_vec = _mm256_set1_epi64x(Self::C_LO as i64);
        let p_vec = _mm256_set1_epi64x(P as i64);
        let sign = _mm256_set1_epi64x(i64::MIN);
        let c_hi_lo = _mm256_mul_epu32(c_vec, hi);
        let hi_hi = _mm256_srli_epi64::<32>(hi);
        let c_hi_hi = _mm256_mul_epu32(c_vec, hi_hi);

        let c_hi_hi_lo32 = _mm256_slli_epi64::<32>(c_hi_hi);
        let c_hi_carry = _mm256_srli_epi64::<32>(c_hi_hi);

        let sum_lo = _mm256_add_epi64(c_hi_lo, c_hi_hi_lo32);
        let c_hi_lo_s = _mm256_xor_si256(c_hi_lo, sign);
        let sum_lo_s = _mm256_xor_si256(sum_lo, sign);
        let carry0 = _mm256_cmpgt_epi64(c_hi_lo_s, sum_lo_s);
        let overflow = _mm256_sub_epi64(c_hi_carry, carry0);

        let s = _mm256_add_epi64(lo, sum_lo);
        let lo_s = _mm256_xor_si256(lo, sign);
        let s_s = _mm256_xor_si256(s, sign);
        let carry1 = _mm256_cmpgt_epi64(lo_s, s_s);
        let total_overflow = _mm256_sub_epi64(overflow, carry1);

        let final_corr = _mm256_mul_epu32(c_vec, total_overflow);
        let result = _mm256_add_epi64(s, final_corr);
        let s2_s = _mm256_xor_si256(s, sign);
        let result_s = _mm256_xor_si256(result, sign);
        let carry_f = _mm256_cmpgt_epi64(s2_s, result_s);
        let corr_f = _mm256_and_si256(carry_f, c_vec);
        let result = _mm256_add_epi64(result, corr_f);

        let result_s2 = _mm256_xor_si256(result, sign);
        let p_s = _mm256_xor_si256(p_vec, sign);
        let lt_p = _mm256_cmpgt_epi64(p_s, result_s2);
        let sub_amt = _mm256_andnot_si256(lt_p, p_vec);
        _mm256_sub_epi64(result, sub_amt)
    }
}

impl<const P: u64> Default for PackedFp64Avx2<P> {
    #[inline]
    fn default() -> Self {
        Self([Fp64(0); FP64_WIDTH])
    }
}

impl<const P: u64> fmt::Debug for PackedFp64Avx2<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackedFp64Avx2").field(&self.0).finish()
    }
}

impl<const P: u64> PartialEq for PackedFp64Avx2<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const P: u64> Eq for PackedFp64Avx2<P> {}

impl<const P: u64> Add for PackedFp64Avx2<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            let a = self.to_vec();
            let b = rhs.to_vec();
            let p = _mm256_set1_epi64x(P as i64);

            let result = if Self::BITS <= 62 {
                // a + b < 2P < 2^63: no overflow.
                let s = _mm256_add_epi64(a, b);
                let r = _mm256_sub_epi64(s, p);
                // s < P? Use signed compare after shift trick.
                let sign = _mm256_set1_epi64x(i64::MIN);
                let s_s = _mm256_xor_si256(s, sign);
                let p_s = _mm256_xor_si256(p, sign);
                let borrow = _mm256_cmpgt_epi64(p_s, s_s);
                _mm256_blendv_epi8(r, s, borrow)
            } else {
                // a + b can overflow u64.
                let s = _mm256_add_epi64(a, b);
                let sign = _mm256_set1_epi64x(i64::MIN);
                let a_s = _mm256_xor_si256(a, sign);
                let s_s = _mm256_xor_si256(s, sign);
                let overflow = _mm256_cmpgt_epi64(a_s, s_s);
                let c = _mm256_set1_epi64x(Self::C_LO as i64);
                let s_plus_c = _mm256_add_epi64(s, c);
                let s_minus_p = _mm256_sub_epi64(s, p);
                let p_s = _mm256_xor_si256(p, sign);
                let lt_p = _mm256_cmpgt_epi64(p_s, s_s);
                let no_of = _mm256_blendv_epi8(s_minus_p, s, lt_p);
                _mm256_blendv_epi8(no_of, s_plus_c, overflow)
            };

            Self::from_vec(result)
        }
    }
}

impl<const P: u64> Sub for PackedFp64Avx2<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            let a = self.to_vec();
            let b = rhs.to_vec();
            let p = _mm256_set1_epi64x(P as i64);
            let d = _mm256_sub_epi64(a, b);

            let sign = _mm256_set1_epi64x(i64::MIN);
            let a_s = _mm256_xor_si256(a, sign);
            let b_s = _mm256_xor_si256(b, sign);
            let underflow = _mm256_cmpgt_epi64(b_s, a_s);
            let corrected = _mm256_add_epi64(d, p);
            Self::from_vec(_mm256_blendv_epi8(d, corrected, underflow))
        }
    }
}

impl<const P: u64> Mul for PackedFp64Avx2<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let (hi, lo) = mul64_64_256(self.to_vec(), rhs.to_vec());
            Self::from_vec(Self::reduce128_vec(hi, lo))
        }
    }
}

impl<const P: u64> AddAssign for PackedFp64Avx2<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u64> SubAssign for PackedFp64Avx2<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u64> MulAssign for PackedFp64Avx2<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u64> PackedField for PackedFp64Avx2<P> {
    const WIDTH: usize = FP64_WIDTH;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Scalar,
    {
        Self([f(0), f(1), f(2), f(3)])
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Scalar {
        debug_assert!(lane < FP64_WIDTH);
        self.0[lane]
    }

    type Scalar = Fp64<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self([value; FP64_WIDTH])
    }
}
