use super::*;

impl<F: Field, const D: usize> CyclotomicRing<F, D> {
    /// Samples a ring element with uniformly random coefficients.
    pub fn random<R: RngCore>(rng: &mut R) -> Self {
        Self {
            coeffs: from_fn(|_| F::random(rng)),
        }
    }
}

impl<F: Field, const D: usize> AddAssign for CyclotomicRing<F, D> {
    fn add_assign(&mut self, rhs: Self) {
        for (dst, src) in self.coeffs.iter_mut().zip(rhs.coeffs.iter()) {
            *dst = *dst + *src;
        }
    }
}

impl<F: Field, const D: usize> SubAssign for CyclotomicRing<F, D> {
    fn sub_assign(&mut self, rhs: Self) {
        for (dst, src) in self.coeffs.iter_mut().zip(rhs.coeffs.iter()) {
            *dst = *dst - *src;
        }
    }
}

impl<F: Field, const D: usize> Add for CyclotomicRing<F, D> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        self += rhs;
        self
    }
}

impl<F: Field, const D: usize> Sub for CyclotomicRing<F, D> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self {
        self -= rhs;
        self
    }
}

impl<F: Field, const D: usize> Neg for CyclotomicRing<F, D> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut out = self.coeffs;
        for c in &mut out {
            *c = -*c;
        }
        Self { coeffs: out }
    }
}

impl<F: Field, const D: usize> MulAssign for CyclotomicRing<F, D> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'a, F: Field, const D: usize> Add<&'a Self> for CyclotomicRing<F, D> {
    type Output = Self;
    fn add(self, rhs: &'a Self) -> Self {
        self + *rhs
    }
}

impl<'a, F: Field, const D: usize> Sub<&'a Self> for CyclotomicRing<F, D> {
    type Output = Self;
    fn sub(self, rhs: &'a Self) -> Self {
        self - *rhs
    }
}

impl<'a, F: Field, const D: usize> Mul<&'a Self> for CyclotomicRing<F, D> {
    type Output = Self;
    fn mul(self, rhs: &'a Self) -> Self {
        self * *rhs
    }
}

/// Schoolbook negacyclic convolution: O(D^2).
///
/// For each pair `(i, j)`:
/// - If `i + j < D`: accumulate `a_i * b_j` at index `i + j`.
/// - If `i + j >= D`: accumulate `-(a_i * b_j)` at index `(i + j) - D`.
impl<F: Field, const D: usize> Mul for CyclotomicRing<F, D> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut out = [F::zero(); D];
        for i in 0..D {
            for j in 0..D {
                let product = self.coeffs[i] * rhs.coeffs[j];
                let idx = i + j;
                if idx < D {
                    out[idx] += product;
                } else {
                    out[idx - D] -= product;
                }
            }
        }
        Self { coeffs: out }
    }
}

impl<F: Field, const D: usize> Zero for CyclotomicRing<F, D> {
    #[inline]
    fn zero() -> Self {
        Self {
            coeffs: [F::zero(); D],
        }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(Zero::is_zero)
    }
}

impl<F: Field, const D: usize> One for CyclotomicRing<F, D> {
    #[inline]
    fn one() -> Self {
        let mut coeffs = [F::zero(); D];
        coeffs[0] = F::one();
        Self { coeffs }
    }
}

impl<F: Field, const D: usize> fmt::Display for CyclotomicRing<F, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CyclotomicRing")
            .field(&self.coeffs.as_slice())
            .finish()
    }
}

impl<F: Field, const D: usize> Sum for CyclotomicRing<F, D> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a, F: Field, const D: usize> Sum<&'a Self> for CyclotomicRing<F, D> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + *x)
    }
}

impl<F: Field, const D: usize> Product for CyclotomicRing<F, D> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a, F: Field, const D: usize> Product<&'a Self> for CyclotomicRing<F, D> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * *x)
    }
}

impl<F: Field, const D: usize> AdditiveGroup for CyclotomicRing<F, D> {}
impl<F: Field, const D: usize> Ring for CyclotomicRing<F, D> {
    fn from_u64(v: u64) -> Self {
        Self::from_slice(&[F::from_u64(v)])
    }

    fn from_i64(v: i64) -> Self {
        Self::from_slice(&[F::from_i64(v)])
    }

    fn from_u128(v: u128) -> Self {
        Self::from_slice(&[F::from_u128(v)])
    }

    fn from_i128(v: i128) -> Self {
        Self::from_slice(&[F::from_i128(v)])
    }
}

impl<F: Field + Valid, const D: usize> Valid for CyclotomicRing<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        for x in self.coeffs.iter() {
            x.check()?;
        }
        Ok(())
    }
}

impl<F: Field + AkitaSerialize, const D: usize> AkitaSerialize for CyclotomicRing<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for x in self.coeffs.iter() {
            x.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coeffs
            .iter()
            .map(|x| x.serialized_size(compress))
            .sum()
    }
}

impl<F: Field + Valid + AkitaDeserialize<Context = ()>, const D: usize> AkitaDeserialize
    for CyclotomicRing<F, D>
{
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let mut coeffs = [F::zero(); D];
        for c in &mut coeffs {
            *c = F::deserialize_with_mode(&mut reader, compress, validate, &())?;
        }
        let out = Self { coeffs };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: Field, const D: usize> Default for CyclotomicRing<F, D> {
    fn default() -> Self {
        Self::zero()
    }
}
