use super::*;

/// Number of `Fp128` lanes in an AVX-512 packed vector.
pub(crate) const FP128_WIDTH: usize = 8;

/// AVX-512 packed arithmetic for `Fp128<P>`, 8 lanes in SoA layout.
///
/// Stores 8 elements as separate `lo` and `hi` `u64` arrays, enabling
/// vectorized add/sub via `__m512i`.  Mul remains scalar per-lane.
#[derive(Clone, Copy)]
pub struct PackedFp128Avx512<const P: u128> {
    lo: [u64; FP128_WIDTH],
    hi: [u64; FP128_WIDTH],
}

impl<const P: u128> PackedFp128Avx512<P> {
    const P_LO: u64 = P as u64;
    const P_HI: u64 = (P >> 64) as u64;
}

impl<const P: u128> Default for PackedFp128Avx512<P> {
    #[inline]
    fn default() -> Self {
        Self {
            lo: [0; FP128_WIDTH],
            hi: [0; FP128_WIDTH],
        }
    }
}

impl<const P: u128> fmt::Debug for PackedFp128Avx512<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elems: Vec<_> = (0..FP128_WIDTH).map(|i| self.extract(i)).collect();
        f.debug_tuple("PackedFp128Avx512").field(&elems).finish()
    }
}

impl<const P: u128> PartialEq for PackedFp128Avx512<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.lo == other.lo && self.hi == other.hi
    }
}

impl<const P: u128> Eq for PackedFp128Avx512<P> {}

impl<const P: u128> Add for PackedFp128Avx512<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            let a_lo = _mm512_loadu_si512(self.lo.as_ptr().cast());
            let a_hi = _mm512_loadu_si512(self.hi.as_ptr().cast());
            let b_lo = _mm512_loadu_si512(rhs.lo.as_ptr().cast());
            let b_hi = _mm512_loadu_si512(rhs.hi.as_ptr().cast());
            let p_lo = _mm512_set1_epi64(Self::P_LO as i64);
            let p_hi = _mm512_set1_epi64(Self::P_HI as i64);
            let one = _mm512_set1_epi64(1);

            // 128-bit add: (sum_hi, sum_lo) = (a_hi, a_lo) + (b_hi, b_lo)
            let sum_lo = _mm512_add_epi64(a_lo, b_lo);
            let carry_lo = _mm512_cmplt_epu64_mask(sum_lo, a_lo);
            let hi_tmp = _mm512_add_epi64(a_hi, b_hi);
            let ov1 = _mm512_cmplt_epu64_mask(hi_tmp, a_hi);
            let sum_hi = _mm512_mask_add_epi64(hi_tmp, carry_lo, hi_tmp, one);
            let ov2 = _mm512_cmplt_epu64_mask(sum_hi, hi_tmp);
            let carry_128 = ov1 | ov2;

            // 128-bit subtract P: (red_hi, red_lo) = (sum_hi, sum_lo) - P
            let red_lo = _mm512_sub_epi64(sum_lo, p_lo);
            let borrow_lo = _mm512_cmplt_epu64_mask(sum_lo, p_lo);
            let red_hi_tmp = _mm512_sub_epi64(sum_hi, p_hi);
            let bw1 = _mm512_cmplt_epu64_mask(sum_hi, p_hi);
            let red_hi = _mm512_mask_sub_epi64(red_hi_tmp, borrow_lo, red_hi_tmp, one);
            let bw2 = _mm512_cmplt_epu64_mask(red_hi_tmp, _mm512_maskz_mov_epi64(borrow_lo, one));
            let borrow = bw1 | bw2;

            // Use reduced if: overflow happened OR subtraction didn't borrow
            let use_reduced = carry_128 | !borrow;
            let out_lo = _mm512_mask_blend_epi64(use_reduced, sum_lo, red_lo);
            let out_hi = _mm512_mask_blend_epi64(use_reduced, sum_hi, red_hi);

            let mut result = Self::default();
            _mm512_storeu_si512(result.lo.as_mut_ptr().cast(), out_lo);
            _mm512_storeu_si512(result.hi.as_mut_ptr().cast(), out_hi);
            result
        }
    }
}

impl<const P: u128> Sub for PackedFp128Avx512<P> {
    type Output = Self;
    // `bw1 | bw2` below is correct 128-bit borrow wiring (mask OR), not an
    // arithmetic bug; suppress the lint locally rather than module-wide.
    #[expect(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            let a_lo = _mm512_loadu_si512(self.lo.as_ptr().cast());
            let a_hi = _mm512_loadu_si512(self.hi.as_ptr().cast());
            let b_lo = _mm512_loadu_si512(rhs.lo.as_ptr().cast());
            let b_hi = _mm512_loadu_si512(rhs.hi.as_ptr().cast());
            let p_lo = _mm512_set1_epi64(Self::P_LO as i64);
            let p_hi = _mm512_set1_epi64(Self::P_HI as i64);
            let one = _mm512_set1_epi64(1);

            // 128-bit sub: (diff_hi, diff_lo) = (a_hi, a_lo) - (b_hi, b_lo)
            let diff_lo = _mm512_sub_epi64(a_lo, b_lo);
            let borrow_lo = _mm512_cmplt_epu64_mask(a_lo, b_lo);
            let hi_tmp = _mm512_sub_epi64(a_hi, b_hi);
            let bw1 = _mm512_cmplt_epu64_mask(a_hi, b_hi);
            let diff_hi = _mm512_mask_sub_epi64(hi_tmp, borrow_lo, hi_tmp, one);
            let bw2 = _mm512_cmplt_epu64_mask(hi_tmp, _mm512_maskz_mov_epi64(borrow_lo, one));
            let borrow_128 = bw1 | bw2;

            // Correction: add P back where underflow occurred
            let corr_lo = _mm512_add_epi64(diff_lo, p_lo);
            let carry_lo = _mm512_cmplt_epu64_mask(corr_lo, diff_lo);
            let corr_hi = _mm512_add_epi64(diff_hi, p_hi);
            let corr_hi = _mm512_mask_add_epi64(corr_hi, carry_lo, corr_hi, one);

            let out_lo = _mm512_mask_blend_epi64(borrow_128, diff_lo, corr_lo);
            let out_hi = _mm512_mask_blend_epi64(borrow_128, diff_hi, corr_hi);

            let mut result = Self::default();
            _mm512_storeu_si512(result.lo.as_mut_ptr().cast(), out_lo);
            _mm512_storeu_si512(result.hi.as_mut_ptr().cast(), out_hi);
            result
        }
    }
}

impl<const P: u128> Mul for PackedFp128Avx512<P> {
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

impl<const P: u128> AddAssign for PackedFp128Avx512<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u128> SubAssign for PackedFp128Avx512<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u128> MulAssign for PackedFp128Avx512<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u128> PackedField for PackedFp128Avx512<P> {
    const WIDTH: usize = FP128_WIDTH;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Scalar,
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
    fn extract(&self, lane: usize) -> Self::Scalar {
        debug_assert!(lane < FP128_WIDTH);
        Fp128([self.lo[lane], self.hi[lane]])
    }

    type Scalar = Fp128<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self {
            lo: [value.0[0]; FP128_WIDTH],
            hi: [value.0[1]; FP128_WIDTH],
        }
    }
}
