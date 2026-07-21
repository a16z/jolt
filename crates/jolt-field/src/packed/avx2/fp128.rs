use super::*;

/// Number of `Fp128` lanes in an AVX2 packed vector.
pub(crate) const FP128_WIDTH: usize = 4;

/// AVX2 packed arithmetic for `Fp128<P>`, 4 lanes in SoA layout.
///
/// Stores 4 elements as separate `lo` and `hi` `u64` arrays, enabling
/// vectorized add/sub via `__m256i`.  Mul remains scalar per-lane.
#[derive(Clone, Copy)]
pub struct PackedFp128Avx2<const P: u128> {
    lo: [u64; FP128_WIDTH],
    hi: [u64; FP128_WIDTH],
}

impl<const P: u128> PackedFp128Avx2<P> {
    const P_LO: u64 = P as u64;
    const P_HI: u64 = (P >> 64) as u64;
}

impl<const P: u128> Default for PackedFp128Avx2<P> {
    #[inline]
    fn default() -> Self {
        Self {
            lo: [0; FP128_WIDTH],
            hi: [0; FP128_WIDTH],
        }
    }
}

impl<const P: u128> fmt::Debug for PackedFp128Avx2<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elems: Vec<_> = (0..FP128_WIDTH).map(|i| self.extract(i)).collect();
        f.debug_tuple("PackedFp128Avx2").field(&elems).finish()
    }
}

impl<const P: u128> PartialEq for PackedFp128Avx2<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.lo == other.lo && self.hi == other.hi
    }
}

impl<const P: u128> Eq for PackedFp128Avx2<P> {}

impl<const P: u128> Add for PackedFp128Avx2<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            let a_lo = _mm256_loadu_si256(self.lo.as_ptr().cast());
            let a_hi = _mm256_loadu_si256(self.hi.as_ptr().cast());
            let b_lo = _mm256_loadu_si256(rhs.lo.as_ptr().cast());
            let b_hi = _mm256_loadu_si256(rhs.hi.as_ptr().cast());
            let p_lo = _mm256_set1_epi64x(Self::P_LO as i64);
            let p_hi = _mm256_set1_epi64x(Self::P_HI as i64);
            let sign = _mm256_set1_epi64x(i64::MIN);
            let one = _mm256_set1_epi64x(1);

            // 128-bit add with unsigned compare emulation (XOR sign bit)
            let sum_lo = _mm256_add_epi64(a_lo, b_lo);
            let carry_lo =
                _mm256_cmpgt_epi64(_mm256_xor_si256(a_lo, sign), _mm256_xor_si256(sum_lo, sign));
            let carry_lo_bit = _mm256_and_si256(carry_lo, one);

            let hi_tmp = _mm256_add_epi64(a_hi, b_hi);
            let ov1 =
                _mm256_cmpgt_epi64(_mm256_xor_si256(a_hi, sign), _mm256_xor_si256(hi_tmp, sign));
            let sum_hi = _mm256_add_epi64(hi_tmp, carry_lo_bit);
            let ov2 = _mm256_cmpgt_epi64(
                _mm256_xor_si256(hi_tmp, sign),
                _mm256_xor_si256(sum_hi, sign),
            );
            let carry_128 = _mm256_or_si256(ov1, ov2);

            // 128-bit subtract P
            let red_lo = _mm256_sub_epi64(sum_lo, p_lo);
            let borrow_lo =
                _mm256_cmpgt_epi64(_mm256_xor_si256(p_lo, sign), _mm256_xor_si256(sum_lo, sign));
            let borrow_lo_bit = _mm256_and_si256(borrow_lo, one);

            let red_hi_tmp = _mm256_sub_epi64(sum_hi, p_hi);
            let bw1 =
                _mm256_cmpgt_epi64(_mm256_xor_si256(p_hi, sign), _mm256_xor_si256(sum_hi, sign));
            let red_hi = _mm256_sub_epi64(red_hi_tmp, borrow_lo_bit);
            let bw2 = _mm256_cmpgt_epi64(
                _mm256_xor_si256(borrow_lo_bit, sign),
                _mm256_xor_si256(red_hi_tmp, sign),
            );
            let borrow = _mm256_or_si256(bw1, bw2);

            // use_reduced = carry_128 | !borrow
            let not_borrow = _mm256_xor_si256(borrow, _mm256_set1_epi64x(-1));
            let use_reduced = _mm256_or_si256(carry_128, not_borrow);
            let out_lo = _mm256_blendv_epi8(sum_lo, red_lo, use_reduced);
            let out_hi = _mm256_blendv_epi8(sum_hi, red_hi, use_reduced);

            let mut result = Self::default();
            _mm256_storeu_si256(result.lo.as_mut_ptr().cast(), out_lo);
            _mm256_storeu_si256(result.hi.as_mut_ptr().cast(), out_hi);
            result
        }
    }
}

impl<const P: u128> Sub for PackedFp128Avx2<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            let a_lo = _mm256_loadu_si256(self.lo.as_ptr().cast());
            let a_hi = _mm256_loadu_si256(self.hi.as_ptr().cast());
            let b_lo = _mm256_loadu_si256(rhs.lo.as_ptr().cast());
            let b_hi = _mm256_loadu_si256(rhs.hi.as_ptr().cast());
            let p_lo = _mm256_set1_epi64x(Self::P_LO as i64);
            let p_hi = _mm256_set1_epi64x(Self::P_HI as i64);
            let sign = _mm256_set1_epi64x(i64::MIN);
            let one = _mm256_set1_epi64x(1);

            // 128-bit sub
            let diff_lo = _mm256_sub_epi64(a_lo, b_lo);
            let borrow_lo =
                _mm256_cmpgt_epi64(_mm256_xor_si256(b_lo, sign), _mm256_xor_si256(a_lo, sign));
            let borrow_lo_bit = _mm256_and_si256(borrow_lo, one);

            let hi_tmp = _mm256_sub_epi64(a_hi, b_hi);
            let bw1 =
                _mm256_cmpgt_epi64(_mm256_xor_si256(b_hi, sign), _mm256_xor_si256(a_hi, sign));
            let diff_hi = _mm256_sub_epi64(hi_tmp, borrow_lo_bit);
            let bw2 = _mm256_cmpgt_epi64(
                _mm256_xor_si256(borrow_lo_bit, sign),
                _mm256_xor_si256(hi_tmp, sign),
            );
            let borrow_128 = _mm256_or_si256(bw1, bw2);

            // Correction: add P back where underflow occurred
            let corr_lo = _mm256_add_epi64(diff_lo, p_lo);
            let carry_lo = _mm256_cmpgt_epi64(
                _mm256_xor_si256(diff_lo, sign),
                _mm256_xor_si256(corr_lo, sign),
            );
            let carry_lo_bit = _mm256_and_si256(carry_lo, one);
            let corr_hi = _mm256_add_epi64(diff_hi, p_hi);
            let corr_hi = _mm256_add_epi64(corr_hi, carry_lo_bit);

            let out_lo = _mm256_blendv_epi8(diff_lo, corr_lo, borrow_128);
            let out_hi = _mm256_blendv_epi8(diff_hi, corr_hi, borrow_128);

            let mut result = Self::default();
            _mm256_storeu_si256(result.lo.as_mut_ptr().cast(), out_lo);
            _mm256_storeu_si256(result.hi.as_mut_ptr().cast(), out_hi);
            result
        }
    }
}

impl<const P: u128> Mul for PackedFp128Avx2<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut out = Self::default();
        for i in 0..FP128_WIDTH {
            let a = Fp128::<P>([self.lo[i], self.hi[i]]);
            let b = Fp128::<P>([rhs.lo[i], rhs.hi[i]]);
            let r = a * b;
            out.lo[i] = r.0[0];
            out.hi[i] = r.0[1];
        }
        out
    }
}

impl<const P: u128> PackedValue for PackedFp128Avx2<P> {
    type Value = Fp128<P>;
    const WIDTH: usize = FP128_WIDTH;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        let mut lo = [0u64; FP128_WIDTH];
        let mut hi = [0u64; FP128_WIDTH];
        for i in 0..FP128_WIDTH {
            let v = f(i);
            lo[i] = v.0[0];
            hi[i] = v.0[1];
        }
        Self { lo, hi }
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Value {
        debug_assert!(lane < FP128_WIDTH);
        Fp128([self.lo[lane], self.hi[lane]])
    }
}

impl<const P: u128> AddAssign for PackedFp128Avx2<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u128> SubAssign for PackedFp128Avx2<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u128> MulAssign for PackedFp128Avx2<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u128> PackedField for PackedFp128Avx2<P> {
    type Scalar = Fp128<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self {
            lo: [value.0[0]; FP128_WIDTH],
            hi: [value.0[1]; FP128_WIDTH],
        }
    }
}
