//! The extension-field abstraction over a base field.
//!
//! [`FpExt4`] and [`FpExt8`] use the cyclotomic ring-subfield basis aligned with
//! trace reduction and production fp32 presets.

#![expect(
    clippy::expect_used,
    reason = "registered pseudo-Mersenne parameters are a field-type invariant"
)]

use crate::ext::{ExtMulBackend, FpExt2, FpExt2Config, FpExt4, FpExt8};
use crate::unreduced::HasUnreducedOps;
use crate::{
    pseudo_mersenne_modulus, FieldCore, FieldError, FromPrimitiveInt, PseudoMersenneField,
};

/// An algebraic extension of base field `F`.
///
/// Provides the extension degree, embedding of and multiplication by base
/// elements, coefficient access in the canonical basis `{1, u, u^2, ...}`,
/// and Frobenius powers.
pub trait ExtField<F: FieldCore>: FieldCore + FromPrimitiveInt {
    /// Extension degree: `[Self : F]`.
    const EXT_DEGREE: usize;

    /// Embed `x ∈ F` as a constant in `Self`.
    ///
    /// This is intentionally small: for extension towers we embed into the
    /// constant term.
    fn lift_base(x: F) -> Self;

    /// Return `self * x`, where `x` is interpreted as a base-field scalar.
    ///
    /// This avoids materializing the base scalar as an extension element and
    /// then using a full extension multiply. For tower extensions this scales
    /// each base-field coordinate directly.
    fn mul_base(self, x: F) -> Self;

    /// Construct from a coefficient slice `[c0, c1, ..., c_{d-1}]`.
    ///
    /// # Panics
    /// Panics if `coeffs.len() != Self::EXT_DEGREE`.
    fn from_base_slice(coeffs: &[F]) -> Self;

    /// Return base-field coefficients in the canonical basis.
    fn to_base_vec(&self) -> Vec<F>;

    /// Apply `x -> x^(q^power)`, where `q = |F|`.
    ///
    /// The provided implementations are intentionally algebraic rather than
    /// basis-specific: they raise to powers of the base-field modulus.
    /// Specialized extension types can add cheaper implementations later, but
    /// this gives the protocol a single auditable contract first.
    fn frobenius_pow(self, power: usize) -> Self;

    /// Apply the inverse Frobenius power. Since `x -> x^q` has order
    /// `[Self:F]` on `Self`, this is `frobenius_pow(EXT_DEGREE - power)`.
    fn frobenius_inv_pow(self, power: usize) -> Self {
        let degree = Self::EXT_DEGREE;
        if degree == 0 {
            return self;
        }
        self.frobenius_pow((degree - (power % degree)) % degree)
    }
}

/// Deferred-reduction extension-times-base multiply.
///
/// `mul_base_to_product_accum` scales `self` by a base scalar `x` and writes the
/// result into [`HasUnreducedOps::ProductAccum`] without reducing, so a batch of
/// `E × F` products can be summed and reduced once. When
/// [`HasUnreducedOps::DELAYED_PRODUCT_SUM_IS_EXACT`] holds, the reduced sum equals
/// the per-term [`ExtField::mul_base`] sum within the accumulator's headroom.
///
/// `E × F` has no cross terms, so the default body (lift `x` and reuse
/// [`HasUnreducedOps::mul_to_product_accum`]) is correct everywhere; extensions
/// whose product-accumulator layout admits cheaper coordinate scaling override it.
pub trait MulBaseUnreduced<F: FieldCore>: ExtField<F> + HasUnreducedOps {
    /// Accumulate `self * x` (extension times base scalar) without reducing.
    #[inline]
    fn mul_base_to_product_accum(self, x: F) -> Self::ProductAccum {
        self.mul_to_product_accum(Self::lift_base(x))
    }
}

impl<F: PseudoMersenneField + HasUnreducedOps> MulBaseUnreduced<F> for F {}

#[inline]
fn field_pow_u128<E: FieldCore>(mut base: E, mut exp: u128) -> E {
    let mut acc = E::one();
    while exp > 0 {
        if (exp & 1) == 1 {
            acc *= base;
        }
        base *= base;
        exp >>= 1;
    }
    acc
}

#[inline]
fn base_modulus<F: PseudoMersenneField>() -> u128 {
    pseudo_mersenne_modulus(F::MODULUS_BITS, F::MODULUS_OFFSET)
        .expect("pseudo-Mersenne modulus parameters must be valid")
}

fn frobenius_pow_via_base_modulus<F, E>(value: E, power: usize) -> E
where
    F: PseudoMersenneField,
    E: ExtField<F>,
{
    let q = base_modulus::<F>();
    let mut out = value;
    for _ in 0..(power % E::EXT_DEGREE.max(1)) {
        out = field_pow_u128(out, q);
    }
    out
}

/// Return the first `width` elements of the canonical extension basis.
///
/// For [`FpExt4`] and [`FpExt8`] this is the fixed
/// ring-subfield basis `[1, e1, ...]`, so the chosen Moore-type theta family
/// is aligned with the coefficient packing basis used by `embed_subfield`.
///
/// # Errors
///
/// Returns an error if `width > E::EXT_DEGREE`.
pub fn canonical_frobenius_thetas<F, E>(width: usize) -> Result<Vec<E>, FieldError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    if width > E::EXT_DEGREE {
        return Err(FieldError::InvalidInput(format!(
            "Frobenius theta width {width} exceeds extension degree {}",
            E::EXT_DEGREE
        )));
    }
    Ok((0..width)
        .map(|idx| {
            let mut coeffs = vec![F::zero(); E::EXT_DEGREE];
            coeffs[idx] = F::one();
            E::from_base_slice(&coeffs)
        })
        .collect())
}

/// Solve `M_t(theta) z = r`, where
/// `M_t(theta)_{j,h} = theta_h^(q^-j)`.
///
/// This intentionally uses dense elimination: supported Frobenius widths are
/// tiny (`<= [E:F]`) and explicit validation is more valuable here than a
/// clever specialized solver.
///
/// # Errors
///
/// Returns an error if the matrix is not square, the dimensions do not match,
/// or the Moore-type matrix is singular.
pub fn solve_frobenius_moore<F, E>(thetas: &[E], rhs: &[E]) -> Result<Vec<E>, FieldError>
where
    F: PseudoMersenneField,
    E: ExtField<F>,
{
    let n = thetas.len();
    if rhs.len() != n {
        return Err(FieldError::InvalidSize {
            expected: n,
            actual: rhs.len(),
        });
    }
    let mut matrix = (0..n)
        .map(|row| {
            thetas
                .iter()
                .map(|&theta| theta.frobenius_inv_pow(row))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut values = rhs.to_vec();

    for col in 0..n {
        let pivot = (col..n)
            .find(|&row| !matrix[row][col].is_zero())
            .ok_or_else(|| {
                FieldError::InvalidInput("singular Frobenius Moore-type matrix".to_string())
            })?;
        if pivot != col {
            matrix.swap(col, pivot);
            values.swap(col, pivot);
        }
        let inv = matrix[col][col].inverse().ok_or_else(|| {
            FieldError::InvalidInput("singular Frobenius Moore-type matrix".to_string())
        })?;
        for entry in &mut matrix[col][col..] {
            *entry *= inv;
        }
        values[col] *= inv;

        let pivot_tail = matrix[col][col..].to_vec();
        let pivot_value = values[col];
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = matrix[row][col];
            if factor.is_zero() {
                continue;
            }
            for (entry, &pivot_entry) in matrix[row][col..].iter_mut().zip(pivot_tail.iter()) {
                *entry -= factor * pivot_entry;
            }
            values[row] -= factor * pivot_value;
        }
    }
    Ok(values)
}

/// Validate that the canonical theta family gives a nonsingular Moore-type
/// matrix for `width`.
///
/// # Errors
///
/// Returns an error if theta construction fails or the Moore solve rejects.
pub fn validate_canonical_frobenius_thetas<F, E>(width: usize) -> Result<(), FieldError>
where
    F: PseudoMersenneField,
    E: ExtField<F>,
{
    let thetas = canonical_frobenius_thetas::<F, E>(width)?;
    let rhs = (0..width)
        .map(|idx| E::lift_base(F::from_u64((idx + 1) as u64)))
        .collect::<Vec<_>>();
    solve_frobenius_moore::<F, E>(&thetas, &rhs).map(|_| ())
}

impl<F: PseudoMersenneField> ExtField<F> for F {
    const EXT_DEGREE: usize = 1;

    #[inline]
    fn lift_base(x: F) -> Self {
        x
    }

    #[inline]
    fn mul_base(self, x: F) -> Self {
        self * x
    }

    #[inline]
    fn from_base_slice(coeffs: &[F]) -> Self {
        assert_eq!(coeffs.len(), 1);
        coeffs[0]
    }

    #[inline]
    fn to_base_vec(&self) -> Vec<F> {
        vec![*self]
    }

    #[inline]
    fn frobenius_pow(self, power: usize) -> Self {
        let _ = power;
        self
    }
}

impl<F, C> ExtField<F> for FpExt2<F, C>
where
    F: PseudoMersenneField,
    C: FpExt2Config<F>,
{
    const EXT_DEGREE: usize = 2;

    #[inline]
    fn lift_base(x: F) -> Self {
        Self::new(x, F::zero())
    }

    #[inline]
    fn mul_base(self, x: F) -> Self {
        Self::new(self.coeffs[0] * x, self.coeffs[1] * x)
    }

    #[inline]
    fn from_base_slice(coeffs: &[F]) -> Self {
        assert_eq!(coeffs.len(), 2);
        Self::new(coeffs[0], coeffs[1])
    }

    #[inline]
    fn to_base_vec(&self) -> Vec<F> {
        vec![self.coeffs[0], self.coeffs[1]]
    }

    #[inline]
    fn frobenius_pow(self, power: usize) -> Self {
        frobenius_pow_via_base_modulus::<F, Self>(self, power)
    }
}

impl<F> ExtField<F> for FpExt4<F>
where
    F: PseudoMersenneField + ExtMulBackend,
{
    const EXT_DEGREE: usize = 4;

    #[inline]
    fn lift_base(x: F) -> Self {
        Self::new([x, F::zero(), F::zero(), F::zero()])
    }

    #[inline]
    fn mul_base(self, x: F) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] * x))
    }

    #[inline]
    fn from_base_slice(coeffs: &[F]) -> Self {
        assert_eq!(coeffs.len(), 4);
        Self::new([coeffs[0], coeffs[1], coeffs[2], coeffs[3]])
    }

    #[inline]
    fn to_base_vec(&self) -> Vec<F> {
        self.coeffs.to_vec()
    }

    #[inline]
    fn frobenius_pow(self, power: usize) -> Self {
        frobenius_pow_via_base_modulus::<F, Self>(self, power)
    }
}

impl<F> ExtField<F> for FpExt8<F>
where
    F: PseudoMersenneField + ExtMulBackend,
{
    const EXT_DEGREE: usize = 8;

    #[inline]
    fn lift_base(x: F) -> Self {
        Self::new([
            x,
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
        ])
    }

    #[inline]
    fn mul_base(self, x: F) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] * x))
    }

    #[inline]
    fn from_base_slice(coeffs: &[F]) -> Self {
        assert_eq!(coeffs.len(), 8);
        Self::new([
            coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5], coeffs[6], coeffs[7],
        ])
    }

    #[inline]
    fn to_base_vec(&self) -> Vec<F> {
        self.coeffs.to_vec()
    }

    #[inline]
    fn frobenius_pow(self, power: usize) -> Self {
        frobenius_pow_via_base_modulus::<F, Self>(self, power)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fp32, NegOneNr};

    type F = Fp32<251>;
    type E2 = FpExt2<F, NegOneNr>;
    type E4 = FpExt4<F>;

    #[test]
    fn mul_base_matches_full_multiply_for_base_field() {
        let x = F::from_u64(7);
        let scalar = F::from_u64(11);

        assert_eq!(x.mul_base(scalar), x * scalar);
    }

    #[test]
    fn mul_base_matches_full_multiply_for_fp_ext2() {
        let x = E2::new(F::from_u64(3), F::from_u64(5));
        let scalar = F::from_u64(11);

        assert_eq!(x.mul_base(scalar), x * E2::lift_base(scalar));
    }

    #[test]
    fn mul_base_matches_full_multiply_for_fp_ext4() {
        let x = E4::new([
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(13),
        ]);
        let scalar = F::from_u64(11);

        assert_eq!(x.mul_base(scalar), x * E4::lift_base(scalar));
    }
}
