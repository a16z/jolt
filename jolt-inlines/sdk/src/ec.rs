use core::marker::PhantomData;

/// Shared interface for field elements used in EC point arithmetic.
pub trait ECField: Clone + PartialEq + core::fmt::Debug + Sized {
    type Error;

    fn zero() -> Self;
    fn is_zero(&self) -> bool;
    fn neg(&self) -> Self;
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn dbl(&self) -> Self;
    fn tpl(&self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn square(&self) -> Self;
    fn div(&self, other: &Self) -> Self;
    fn div_assume_nonzero(&self, other: &Self) -> Self;
    fn to_u64_arr(&self) -> [u64; 4];
    fn from_u64_arr(arr: &[u64; 4]) -> Result<Self, Self::Error>;
    fn from_u64_arr_unchecked(arr: &[u64; 4]) -> Self;
}

/// Curve-specific constants for a short Weierstrass curve y² = x³ + ax + b.
pub trait CurveParams<F: ECField>: Clone {
    type Error: core::fmt::Debug;

    /// Returns `Some(a)` for curves with a ≠ 0 (e.g. P-256 where a = -3).
    /// Returns `None` (default) for a = 0 curves (secp256k1, grumpkin),
    /// which lets `double` and `is_on_curve` skip the extra multiply.
    fn curve_a() -> Option<F> {
        None
    }

    fn curve_b() -> F;

    /// When true, `double_and_add` checks for divisor == 0 (needed for grumpkin).
    /// Secp256k1 can skip this check.
    const DOUBLE_AND_ADD_DIVISOR_CHECK: bool = false;

    fn not_on_curve_error() -> Self::Error;
}

/// Affine point on a short Weierstrass curve y² = x³ + ax + b.
/// Infinity represented as (zero, zero) since that point is not on any curve with b != 0.
#[derive(Clone, PartialEq, Debug)]
pub struct AffinePoint<F: ECField, C: CurveParams<F>> {
    x: F,
    y: F,
    _curve: PhantomData<C>,
}

impl<F: ECField, C: CurveParams<F>> AffinePoint<F, C> {
    #[inline(always)]
    pub fn new(x: F, y: F) -> Result<Self, C::Error> {
        let p = Self {
            x,
            y,
            _curve: PhantomData,
        };
        if p.is_on_curve() {
            Ok(p)
        } else {
            Err(C::not_on_curve_error())
        }
    }

    #[inline(always)]
    pub fn new_unchecked(x: F, y: F) -> Self {
        Self {
            x,
            y,
            _curve: PhantomData,
        }
    }

    #[inline(always)]
    pub fn infinity() -> Self {
        Self {
            x: F::zero(),
            y: F::zero(),
            _curve: PhantomData,
        }
    }

    #[inline(always)]
    pub fn is_infinity(&self) -> bool {
        self.x.is_zero() && self.y.is_zero()
    }

    #[inline(always)]
    pub fn is_on_curve(&self) -> bool {
        if self.is_infinity() {
            return true;
        }
        let rhs = self.x.square().mul(&self.x).add(&C::curve_b());
        let rhs = match C::curve_a() {
            Some(a) => rhs.add(&a.mul(&self.x)),
            None => rhs,
        };
        self.y.square() == rhs
    }

    #[inline(always)]
    pub fn x(&self) -> F {
        self.x.clone()
    }

    #[inline(always)]
    pub fn y(&self) -> F {
        self.y.clone()
    }

    #[inline(always)]
    pub fn to_u64_arr(&self) -> [u64; 8] {
        let mut arr = [0u64; 8];
        arr[0..4].copy_from_slice(&self.x.to_u64_arr());
        arr[4..8].copy_from_slice(&self.y.to_u64_arr());
        arr
    }

    #[inline(always)]
    pub fn from_u64_arr(arr: &[u64; 8]) -> Result<Self, C::Error>
    where
        C::Error: From<F::Error>,
    {
        let x = F::from_u64_arr(&[arr[0], arr[1], arr[2], arr[3]])?;
        let y = F::from_u64_arr(&[arr[4], arr[5], arr[6], arr[7]])?;
        Self::new(x, y)
    }

    #[inline(always)]
    pub fn from_u64_arr_unchecked(arr: &[u64; 8]) -> Self {
        let x = F::from_u64_arr_unchecked(&[arr[0], arr[1], arr[2], arr[3]]);
        let y = F::from_u64_arr_unchecked(&[arr[4], arr[5], arr[6], arr[7]]);
        Self {
            x,
            y,
            _curve: PhantomData,
        }
    }

    #[inline(always)]
    pub fn neg(&self) -> Self {
        if self.is_infinity() {
            Self::infinity()
        } else {
            Self::new_unchecked(self.x.clone(), self.y.neg())
        }
    }

    #[inline(always)]
    pub fn double(&self) -> Self {
        if self.y.is_zero() {
            Self::infinity()
        } else {
            let num = self.x.square().tpl();
            let num = match C::curve_a() {
                Some(a) => num.add(&a),
                None => num,
            };
            let s = num.div_assume_nonzero(&self.y.dbl());
            let x2 = s.square().sub(&self.x.dbl());
            let y2 = s.mul(&self.x.sub(&x2)).sub(&self.y);
            Self::new_unchecked(x2, y2)
        }
    }

    #[inline(always)]
    pub fn add(&self, other: &Self) -> Self {
        if self.is_infinity() {
            other.clone()
        } else if other.is_infinity() {
            self.clone()
        } else if self.x == other.x && self.y == other.y {
            self.double()
        } else if self.x == other.x {
            Self::infinity()
        } else {
            let s = self
                .y
                .sub(&other.y)
                .div_assume_nonzero(&self.x.sub(&other.x));
            let x2 = s.square().sub(&self.x.add(&other.x));
            let y2 = s.mul(&self.x.sub(&x2)).sub(&self.y);
            Self::new_unchecked(x2, y2)
        }
    }

    /// 2*self + other using an optimized formula that saves one field operation.
    #[inline(always)]
    pub fn double_and_add(&self, other: &Self) -> Self {
        if self.is_infinity() {
            other.clone()
        } else if other.is_infinity() {
            self.add(self)
        } else if self.x == other.x && self.y == other.y {
            self.add(self).add(other)
        } else if self.x == other.x && self.y != other.y {
            self.clone()
        } else {
            let ns = self
                .y
                .sub(&other.y)
                .div_assume_nonzero(&other.x.sub(&self.x));
            let nx2 = other.x.sub(&ns.square());
            let divisor = self.x.dbl().add(&nx2);
            if C::DOUBLE_AND_ADD_DIVISOR_CHECK && divisor.is_zero() {
                return Self::infinity();
            }
            let t = self.y.dbl().div_assume_nonzero(&divisor).add(&ns);
            let x3 = t.square().add(&nx2);
            let y3 = t.mul(&self.x.sub(&x3)).sub(&self.y);
            Self::new_unchecked(x3, y3)
        }
    }
}
