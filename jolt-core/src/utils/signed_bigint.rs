
use ark_ff::{BigInt, BigInteger};
use std::ops::{Add, Sub, Mul, Neg, AddAssign, SubAssign, MulAssign};

/// A signed big integer using arkworks BigInt for magnitude and a sign bit
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignedBigInt<const N: usize> {
    pub magnitude: BigInt<N>,
    pub is_positive: bool,
}

impl<const N: usize> SignedBigInt<N> {
    /// Create a new SignedBigInt from limbs and sign
    pub fn new(limbs: [u64; N], is_positive: bool) -> Self {
        Self { 
            magnitude: BigInt::new(limbs),
            is_positive 
        }
    }

    /// Create a new SignedBigInt from BigInt magnitude and sign
    pub fn from_bigint(magnitude: BigInt<N>, is_positive: bool) -> Self {
        Self { magnitude, is_positive }
    }

    /// Create zero
    pub fn zero() -> Self {
        Self {
            magnitude: BigInt::from(0u64),
            is_positive: true,
        }
    }

    /// Create one
    pub fn one() -> Self {
        Self {
            magnitude: BigInt::from(1u64),
            is_positive: true,
        }
    }

    /// Check if the value is zero
    pub fn is_zero(&self) -> bool {
        self.magnitude.is_zero()
    }

    /// Get the magnitude (absolute value) as BigInt
    pub fn magnitude(&self) -> BigInt<N> {
        self.magnitude
    }

    /// Get the magnitude as limbs array
    pub fn magnitude_limbs(&self) -> [u64; N] {
        self.magnitude.0
    }

    /// Get the sign
    pub fn sign(&self) -> bool {
        self.is_positive
    }

    /// Add two SignedBigInt values
    pub fn add(self, other: Self) -> Self {
        if self.is_positive == other.is_positive {
            // Same sign: add magnitudes
            let mut result_magnitude = self.magnitude;
            let _carry = result_magnitude.add_with_carry(&other.magnitude);
            // TODO: Handle carry/overflow properly
            Self::from_bigint(result_magnitude, self.is_positive)
        } else {
            // Different signs: subtract magnitudes
            match self.magnitude.cmp(&other.magnitude) {
                std::cmp::Ordering::Greater => {
                    let mut result_magnitude = self.magnitude;
                    let _borrow = result_magnitude.sub_with_borrow(&other.magnitude);
                    Self::from_bigint(result_magnitude, self.is_positive)
                }
                std::cmp::Ordering::Less => {
                    let mut result_magnitude = other.magnitude;
                    let _borrow = result_magnitude.sub_with_borrow(&self.magnitude);
                    Self::from_bigint(result_magnitude, other.is_positive)
                }
                std::cmp::Ordering::Equal => {
                    Self::zero()
                }
            }
        }
    }

    /// Subtract two SignedBigInt values
    pub fn sub(self, other: Self) -> Self {
        self.add(other.neg())
    }

    /// Multiply two SignedBigInt values
    pub fn mul(self, other: Self) -> Self {
        // Multiply magnitudes and XOR signs - simple!
        let (result_magnitude, _overflow) = self.magnitude.mul(&other.magnitude);
        let result_positive = self.is_positive == other.is_positive;
        // TODO: Handle overflow properly in a more complete implementation
        Self::from_bigint(result_magnitude, result_positive)
    }

    /// Negate the SignedBigInt
    pub fn neg(self) -> Self {
        Self::from_bigint(self.magnitude, !self.is_positive)
    }
}

// Specializations for common sizes
impl SignedBigInt<1> {
    /// Create from a u64 magnitude
    pub fn from_u64(value: u64, is_positive: bool) -> Self {
        Self::from_bigint(BigInt::from(value), is_positive)
    }

    /// Convert to i128 if possible
    pub fn to_i128(&self) -> Option<i128> {
        let magnitude = self.magnitude.0[0];
        
        // Check if magnitude fits within i128 range
        // Note: i128::MIN.unsigned_abs() > u64::MAX, so we can never represent i128::MIN in a single u64
        if self.is_positive {
            if magnitude <= i128::MAX as u64 {
                Some(magnitude as i128)
            } else {
                None
            }
        } else {
            if magnitude <= i128::MAX as u64 {
                Some(-(magnitude as i128))
            } else {
                None
            }
        }
    }
}

impl SignedBigInt<2> {
    /// Create from a u128 magnitude
    pub fn from_u128(value: u128, is_positive: bool) -> Self {
        Self::from_bigint(BigInt::from(value), is_positive)
    }

    /// Convert to i128 if the high limb is zero
    pub fn to_i128(&self) -> Option<i128> {
        if self.magnitude.0[1] == 0 {
            let single_limb = SignedBigInt::<1>::from_bigint(
                BigInt::from(self.magnitude.0[0]), 
                self.is_positive
            );
            single_limb.to_i128()
        } else {
            None
        }
    }

    /// Get as u128 magnitude
    pub fn magnitude_as_u128(&self) -> u128 {
        (self.magnitude.0[1] as u128) << 64 | (self.magnitude.0[0] as u128)
    }
}


/// Helper function for single u64 signed arithmetic (moved from types.rs)
pub fn add_with_sign_u64(a_mag: u64, a_pos: bool, b_mag: u64, b_pos: bool) -> (u64, bool) {
    let a = SignedBigInt::<1>::from_u64(a_mag, a_pos);
    let b = SignedBigInt::<1>::from_u64(b_mag, b_pos);
    let result = a + b;  // Now using the operator trait!
    (result.magnitude.0[0], result.is_positive)
}

// ===============================================
// Standard operator trait implementations
// ===============================================

impl<const N: usize> Add for SignedBigInt<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        SignedBigInt::add(self, rhs)
    }
}

impl<const N: usize> Sub for SignedBigInt<N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        SignedBigInt::sub(self, rhs)
    }
}

impl<const N: usize> Mul for SignedBigInt<N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        SignedBigInt::mul(self, rhs)
    }
}

impl<const N: usize> Neg for SignedBigInt<N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        SignedBigInt::neg(self)
    }
}

impl<const N: usize> AddAssign for SignedBigInt<N> {
    fn add_assign(&mut self, rhs: Self) {
        *self = SignedBigInt::add(*self, rhs);
    }
}

impl<const N: usize> SubAssign for SignedBigInt<N> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = SignedBigInt::sub(*self, rhs);
    }
}

impl<const N: usize> MulAssign for SignedBigInt<N> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = SignedBigInt::mul(*self, rhs);
    }
}

// Reference variants for efficiency
impl<const N: usize> Add<&SignedBigInt<N>> for SignedBigInt<N> {
    type Output = SignedBigInt<N>;

    fn add(self, rhs: &SignedBigInt<N>) -> Self::Output {
        SignedBigInt::add(self, *rhs)
    }
}

impl<const N: usize> Sub<&SignedBigInt<N>> for SignedBigInt<N> {
    type Output = SignedBigInt<N>;

    fn sub(self, rhs: &SignedBigInt<N>) -> Self::Output {
        SignedBigInt::sub(self, *rhs)
    }
}

impl<const N: usize> Mul<&SignedBigInt<N>> for SignedBigInt<N> {
    type Output = SignedBigInt<N>;

    fn mul(self, rhs: &SignedBigInt<N>) -> Self::Output {
        SignedBigInt::mul(self, *rhs)
    }
}

impl<const N: usize> AddAssign<&SignedBigInt<N>> for SignedBigInt<N> {
    fn add_assign(&mut self, rhs: &SignedBigInt<N>) {
        *self = SignedBigInt::add(*self, *rhs);
    }
}

impl<const N: usize> SubAssign<&SignedBigInt<N>> for SignedBigInt<N> {
    fn sub_assign(&mut self, rhs: &SignedBigInt<N>) {
        *self = SignedBigInt::sub(*self, *rhs);
    }
}

impl<const N: usize> MulAssign<&SignedBigInt<N>> for SignedBigInt<N> {
    fn mul_assign(&mut self, rhs: &SignedBigInt<N>) {
        *self = SignedBigInt::mul(*self, *rhs);
    }
}

// ===============================================
// SmallScalar conversion trait implementations
// ===============================================

impl From<crate::utils::small_scalar::SmallScalar> for SignedBigInt<2> {
    /// Infallible conversion from SmallScalar to SignedBigInt<2> (128-bit)
    /// All SmallScalar values fit within 128 bits
    fn from(scalar: crate::utils::small_scalar::SmallScalar) -> Self {
        use crate::utils::small_scalar::SmallScalar;
        match scalar {
            SmallScalar::Bool(v) => SignedBigInt::new([v as u64, 0], true),
            SmallScalar::U8(v) => SignedBigInt::new([v as u64, 0], true),
            SmallScalar::U64(v) => SignedBigInt::new([v, 0], true),
            SmallScalar::I64(v) => {
                if v >= 0 {
                    SignedBigInt::new([v as u64, 0], true)
                } else {
                    SignedBigInt::new([(-v) as u64, 0], false)
                }
            }
            SmallScalar::U128(v) => {
                let low = v as u64;
                let high = (v >> 64) as u64;
                SignedBigInt::new([low, high], true)
            }
            SmallScalar::I128(v) => {
                if v >= 0 {
                    let v_u128 = v as u128;
                    let low = v_u128 as u64;
                    let high = (v_u128 >> 64) as u64;
                    SignedBigInt::new([low, high], true)
                } else {
                    let v_u128 = (-v) as u128;
                    let low = v_u128 as u64;
                    let high = (v_u128 >> 64) as u64;
                    SignedBigInt::new([low, high], false)
                }
            }
        }
    }
}

impl TryFrom<crate::utils::small_scalar::SmallScalar> for SignedBigInt<1> {
    type Error = crate::utils::small_scalar::SmallScalarConversionError;

    /// Fallible conversion from SmallScalar to SignedBigInt<1> (64-bit)
    /// Fails if the value doesn't fit in 64 bits
    fn try_from(scalar: crate::utils::small_scalar::SmallScalar) -> Result<Self, Self::Error> {
        use crate::utils::small_scalar::{SmallScalar, SmallScalarConversionError};
        
        match scalar {
            SmallScalar::Bool(v) => Ok(SignedBigInt::from_u64(v as u64, true)),
            SmallScalar::U8(v) => Ok(SignedBigInt::from_u64(v as u64, true)),
            SmallScalar::U64(v) => Ok(SignedBigInt::from_u64(v, true)),
            SmallScalar::I64(v) => {
                if v >= 0 {
                    Ok(SignedBigInt::from_u64(v as u64, true))
                } else {
                    // Use wrapping_neg to handle i64::MIN case
                    Ok(SignedBigInt::from_u64(v.wrapping_neg() as u64, false))
                }
            }
            SmallScalar::U128(v) => {
                if v <= u64::MAX as u128 {
                    Ok(SignedBigInt::from_u64(v as u64, true))
                } else {
                    Err(SmallScalarConversionError::OutOfRange {
                        value: v.to_string(),
                        target_type: "SignedBigInt<1>",
                    })
                }
            }
            SmallScalar::I128(v) => {
                let abs_val = v.unsigned_abs();
                if abs_val <= u64::MAX as u128 {
                    Ok(SignedBigInt::from_u64(abs_val as u64, v >= 0))
                } else {
                    Err(SmallScalarConversionError::OutOfRange {
                        value: v.to_string(),
                        target_type: "SignedBigInt<1>",
                    })
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::small_scalar::SmallScalar;

    #[test]
    fn test_construction() {
        // Test zero
        let zero = SignedBigInt::<1>::zero();
        assert!(zero.is_zero());
        assert_eq!(zero.magnitude.0[0], 0);
        assert!(zero.is_positive);

        // Test one
        let one = SignedBigInt::<1>::one();
        assert!(!one.is_zero());
        assert_eq!(one.magnitude.0[0], 1);
        assert!(one.is_positive);

        // Test from_u64
        let pos = SignedBigInt::<1>::from_u64(42, true);
        assert_eq!(pos.magnitude.0[0], 42);
        assert!(pos.is_positive);

        let neg = SignedBigInt::<1>::from_u64(42, false);
        assert_eq!(neg.magnitude.0[0], 42);
        assert!(!neg.is_positive);
    }

    #[test]
    fn test_construction_128() {
        // Test with a value that spans both limbs
        let val = SignedBigInt::<2>::from_u128(0x123456789abcdef0fedcba9876543210, true);
        // BigInt stores in little-endian format
        assert_eq!(val.magnitude.0[0], 0xfedcba9876543210); // low 64 bits
        assert_eq!(val.magnitude.0[1], 0x123456789abcdef0); // high 64 bits
        assert!(val.is_positive);
    }

    #[test]
    fn test_addition() {
        // Same sign addition
        let a = SignedBigInt::<1>::from_u64(10, true);
        let b = SignedBigInt::<1>::from_u64(20, true);
        let result = a.add(b);
        assert_eq!(result.magnitude.0[0], 30);
        assert!(result.is_positive);

        // Different sign subtraction (positive - negative = positive)
        let a = SignedBigInt::<1>::from_u64(30, true);
        let b = SignedBigInt::<1>::from_u64(20, false);
        let result = a.add(b);
        assert_eq!(result.magnitude.0[0], 10);
        assert!(result.is_positive);

        // Different sign subtraction (negative - positive = negative)
        let a = SignedBigInt::<1>::from_u64(20, false);
        let b = SignedBigInt::<1>::from_u64(30, true);
        let result = a.add(b);
        assert_eq!(result.magnitude.0[0], 10);
        assert!(result.is_positive);

        // Equal magnitudes different signs = zero
        let a = SignedBigInt::<1>::from_u64(42, true);
        let b = SignedBigInt::<1>::from_u64(42, false);
        let result = a.add(b);
        assert!(result.is_zero());
    }

    #[test]
    fn test_subtraction() {
        let a = SignedBigInt::<1>::from_u64(50, true);
        let b = SignedBigInt::<1>::from_u64(30, true);
        let result = a.sub(b);
        assert_eq!(result.magnitude.0[0], 20);
        assert!(result.is_positive);

        // Subtracting a negative is addition
        let a = SignedBigInt::<1>::from_u64(50, true);
        let b = SignedBigInt::<1>::from_u64(30, false);
        let result = a.sub(b);
        assert_eq!(result.magnitude.0[0], 80);
        assert!(result.is_positive);
    }

    #[test]
    fn test_multiplication() {
        // Positive * positive = positive
        let a = SignedBigInt::<1>::from_u64(6, true);
        let b = SignedBigInt::<1>::from_u64(7, true);
        let result = a.mul(b);
        assert_eq!(result.magnitude.0[0], 42);
        assert!(result.is_positive);

        // Positive * negative = negative
        let a = SignedBigInt::<1>::from_u64(6, true);
        let b = SignedBigInt::<1>::from_u64(7, false);
        let result = a.mul(b);
        assert_eq!(result.magnitude.0[0], 42);
        assert!(!result.is_positive);

        // Negative * negative = positive
        let a = SignedBigInt::<1>::from_u64(6, false);
        let b = SignedBigInt::<1>::from_u64(7, false);
        let result = a.mul(b);
        assert_eq!(result.magnitude.0[0], 42);
        assert!(result.is_positive);
    }

    #[test]
    fn test_negation() {
        let pos = SignedBigInt::<1>::from_u64(42, true);
        let neg = pos.neg();
        assert_eq!(neg.magnitude.0[0], 42);
        assert!(!neg.is_positive);

        let neg_orig = SignedBigInt::<1>::from_u64(42, false);
        let pos_result = neg_orig.neg();
        assert_eq!(pos_result.magnitude.0[0], 42);
        assert!(pos_result.is_positive);
    }

    #[test]
    fn test_operator_traits() {
        let a = SignedBigInt::<1>::from_u64(10, true);
        let b = SignedBigInt::<1>::from_u64(5, true);

        // Add trait
        let sum = a + b;
        assert_eq!(sum.magnitude.0[0], 15);
        assert!(sum.is_positive);

        // Sub trait
        let diff = a - b;
        assert_eq!(diff.magnitude.0[0], 5);
        assert!(diff.is_positive);

        // Mul trait
        let prod = a * b;
        assert_eq!(prod.magnitude.0[0], 50);
        assert!(prod.is_positive);

        // Neg trait
        let neg_a = -a;
        assert_eq!(neg_a.magnitude.0[0], 10);
        assert!(!neg_a.is_positive);
    }

    #[test]
    fn test_assign_operators() {
        let mut a = SignedBigInt::<1>::from_u64(10, true);
        let b = SignedBigInt::<1>::from_u64(5, true);

        // AddAssign
        a += b;
        assert_eq!(a.magnitude.0[0], 15);

        // SubAssign
        a -= b;
        assert_eq!(a.magnitude.0[0], 10);

        // MulAssign
        a *= b;
        assert_eq!(a.magnitude.0[0], 50);
    }

    #[test]
    fn test_reference_operators() {
        let a = SignedBigInt::<1>::from_u64(10, true);
        let b = SignedBigInt::<1>::from_u64(5, true);

        // Test reference variants
        let sum = a + &b;
        assert_eq!(sum.magnitude.0[0], 15);

        let diff = a - &b;
        assert_eq!(diff.magnitude.0[0], 5);

        let prod = a * &b;
        assert_eq!(prod.magnitude.0[0], 50);

        // Test assignment with references
        let mut c = a;
        c += &b;
        assert_eq!(c.magnitude.0[0], 15);
    }

    #[test]
    fn test_to_i128_single_limb() {
        // Positive values within i128 range
        let pos = SignedBigInt::<1>::from_u64(100, true);
        assert_eq!(pos.to_i128(), Some(100));

        // Negative values within i128 range
        let neg = SignedBigInt::<1>::from_u64(100, false);
        assert_eq!(neg.to_i128(), Some(-100));

        // Maximum positive value that fits in u64 range of i128
        let max_pos_u64 = SignedBigInt::<1>::from_u64(i64::MAX as u64, true);
        assert_eq!(max_pos_u64.to_i128(), Some(i64::MAX as i128));

        // i128::MIN doesn't fit in SignedBigInt<1> since its abs value > u64::MAX
        // Test with i64::MIN instead which does fit 
        let min_neg = SignedBigInt::<1>::from_u64(i64::MIN.unsigned_abs(), false);
        assert_eq!(min_neg.to_i128(), Some(i64::MIN as i128));

        // Value too large for i128
        let too_large = SignedBigInt::<1>::from_u64(u64::MAX, true);
        assert_eq!(too_large.to_i128(), None);
    }

    #[test]
    fn test_to_i128_double_limb() {
        // Value that fits in single limb
        let small = SignedBigInt::<2>::new([100, 0], true);
        assert_eq!(small.to_i128(), Some(100));

        // Value with high limb set should fail
        let large = SignedBigInt::<2>::new([100, 1], true);
        assert_eq!(large.to_i128(), None);
    }

    #[test]
    fn test_magnitude_as_u128() {
        let val = SignedBigInt::<2>::new([0x123456789abcdef0, 0xfedcba9876543210], true);
        let expected = (0xfedcba9876543210u128 << 64) | 0x123456789abcdef0u128;
        assert_eq!(val.magnitude_as_u128(), expected);
    }

    #[test]
    fn test_small_scalar_conversion_to_128bit() {
        // Test all SmallScalar variants convert to SignedBigInt<2>
        let bool_val = SignedBigInt::<2>::from(SmallScalar::Bool(true));
        assert_eq!(bool_val.magnitude.0[0], 1);
        assert_eq!(bool_val.magnitude.0[1], 0);
        assert!(bool_val.is_positive);

        let u8_val = SignedBigInt::<2>::from(SmallScalar::U8(255));
        assert_eq!(u8_val.magnitude.0[0], 255);
        assert!(u8_val.is_positive);

        let i64_neg = SignedBigInt::<2>::from(SmallScalar::I64(-42));
        assert_eq!(i64_neg.magnitude.0[0], 42);
        assert!(!i64_neg.is_positive);

        let i128_pos = SignedBigInt::<2>::from(SmallScalar::I128(i128::MAX));
        assert!(i128_pos.is_positive);
    }

    #[test]
    fn test_small_scalar_conversion_to_64bit() {
        // Successful conversions
        assert!(SignedBigInt::<1>::try_from(SmallScalar::Bool(false)).is_ok());
        assert!(SignedBigInt::<1>::try_from(SmallScalar::U8(100)).is_ok());
        assert!(SignedBigInt::<1>::try_from(SmallScalar::U64(u64::MAX)).is_ok());
        assert!(SignedBigInt::<1>::try_from(SmallScalar::I64(i64::MIN)).is_ok());

        // Should succeed for small U128
        assert!(SignedBigInt::<1>::try_from(SmallScalar::U128(1000)).is_ok());

        // Should fail for large U128
        assert!(SignedBigInt::<1>::try_from(SmallScalar::U128(u128::MAX)).is_err());

        // Should fail for large I128
        assert!(SignedBigInt::<1>::try_from(SmallScalar::I128(i128::MAX)).is_err());
    }

    #[test]
    fn test_helper_function() {
        // Test add_with_sign_u64 helper
        let (result_mag, result_sign) = add_with_sign_u64(10, true, 5, true);
        assert_eq!(result_mag, 15);
        assert!(result_sign);

        let (result_mag, result_sign) = add_with_sign_u64(10, true, 5, false);
        assert_eq!(result_mag, 5);
        assert!(result_sign);

        let (result_mag, result_sign) = add_with_sign_u64(5, true, 10, false);
        assert_eq!(result_mag, 5);
        assert!(!result_sign);
    }

    #[test]
    fn test_zero_operations() {
        let zero = SignedBigInt::<1>::zero();
        let five = SignedBigInt::<1>::from_u64(5, true);

        // Zero + anything = that thing
        assert_eq!(zero + five, five);
        assert_eq!(five + zero, five);

        // Zero * anything = zero
        assert_eq!(zero * five, zero);
        assert_eq!(five * zero, zero);

        // Anything - itself = zero
        assert_eq!(five - five, zero);
    }

    #[test]
    fn test_arithmetic_properties() {
        let a = SignedBigInt::<1>::from_u64(3, true);
        let b = SignedBigInt::<1>::from_u64(5, true);
        let c = SignedBigInt::<1>::from_u64(7, false);

        // Commutativity
        assert_eq!(a + b, b + a);
        assert_eq!(a * b, b * a);

        // Associativity 
        assert_eq!((a + b) + c, a + (b + c));
        assert_eq!((a * b) * c, a * (b * c));

        // Distributivity: a * (b + c) = a * b + a * c
        assert_eq!(a * (b + c), a * b + a * c);
    }
}