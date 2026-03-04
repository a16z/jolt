// Ported from a16z/arkworks-algebra fork (dev/twist-shout branch).
// Extended Jacobian (XYZZ) coordinates for efficient bucket accumulation in MSM.
// Formulas: <https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html>
//
// Original code dual-licensed under Apache-2.0 and MIT.

use ark_ec::models::short_weierstrass::{Affine, Projective, SWCurveConfig};
use ark_ff::{AdditiveGroup, Field};
use std::ops::{AddAssign, Neg, SubAssign};

#[must_use]
pub struct Bucket<P: SWCurveConfig> {
    pub x: P::BaseField,
    pub y: P::BaseField,
    pub zz: P::BaseField,
    pub zzz: P::BaseField,
}

impl<P: SWCurveConfig> Copy for Bucket<P> {}
impl<P: SWCurveConfig> Clone for Bucket<P> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: SWCurveConfig> Default for Bucket<P> {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<P: SWCurveConfig> Bucket<P> {
    pub const ZERO: Self = Self {
        x: P::BaseField::ONE,
        y: P::BaseField::ONE,
        zz: P::BaseField::ZERO,
        zzz: P::BaseField::ZERO,
    };

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.zz == P::BaseField::ZERO && self.zzz == P::BaseField::ZERO
    }

    /// <https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-dbl-2008-s-1>
    pub fn double_in_place(&mut self) {
        let mut u = self.y;
        u.double_in_place();

        let mut v = u;
        v.square_in_place();

        let mut w = u;
        w *= &v;

        let mut s = self.x;
        s *= &v;

        let mut m = self.x.square();
        m += m.double();
        if P::COEFF_A != P::BaseField::ZERO {
            m += P::mul_by_a(self.zz.square());
        }

        self.x = m;
        self.x.square_in_place();
        self.x -= &s.double();

        self.y = P::BaseField::sum_of_products(&[m, -w], &[(s - &self.x), self.y]);

        self.zz *= v;
        self.zzz *= &w;
    }
}

/// Doubles an affine point directly into XYZZ bucket form.
/// Uses mdbl-2008-s-1 (optimized for ZZ=ZZZ=1).
pub fn double_affine_to_bucket<P: SWCurveConfig>(p: &Affine<P>) -> Bucket<P> {
    if p.infinity {
        return Bucket::ZERO;
    }

    let u = p.y.double();
    let v = u.square();
    let w = u * &v;
    let s = p.x * &v;

    let mut m = p.x.square();
    m += m.double();
    if !P::COEFF_A.is_zero() {
        m += P::COEFF_A;
    }

    let x = m.square() - s.double();
    let y = m * (s - x) - w * p.y;

    Bucket { x, y, zz: v, zzz: w }
}

impl<P: SWCurveConfig> Neg for Bucket<P> {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self {
        self.y = -self.y;
        self
    }
}

/// Mixed addition: bucket += affine point.
/// <https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s>
impl<P: SWCurveConfig> AddAssign<&Affine<P>> for Bucket<P> {
    fn add_assign(&mut self, other: &Affine<P>) {
        if other.infinity {
            return;
        }

        if self.is_zero() {
            self.x = other.x;
            self.y = other.y;
            self.zz = P::BaseField::one();
            self.zzz = P::BaseField::one();
            return;
        }

        let z1z1 = self.zz;

        let mut u2 = other.x;
        u2 *= &z1z1;

        let s2 = other.y * self.zzz;

        if self.x == u2 {
            if self.y == s2 {
                *self = double_affine_to_bucket(other);
            } else {
                *self = Self::ZERO;
            }
        } else {
            let mut p = u2;
            p -= &self.x;

            let mut r = s2;
            r -= &self.y;

            let mut pp = p;
            pp.square_in_place();

            let mut ppp = pp;
            ppp *= &p;

            let mut q = self.x;
            q *= &pp;

            self.x = r.square();
            self.x -= &ppp;
            self.x -= &q.double();

            q -= &self.x;
            self.y = P::BaseField::sum_of_products(&[r, -self.y], &[q, ppp]);

            self.zz *= &pp;
            self.zzz *= &ppp;
        }
    }
}

impl<P: SWCurveConfig> AddAssign<Affine<P>> for Bucket<P> {
    #[inline]
    fn add_assign(&mut self, other: Affine<P>) {
        *self += &other;
    }
}

impl<P: SWCurveConfig> SubAssign<&Affine<P>> for Bucket<P> {
    #[inline]
    fn sub_assign(&mut self, other: &Affine<P>) {
        *self += &(-*other);
    }
}

impl<P: SWCurveConfig> SubAssign<Affine<P>> for Bucket<P> {
    #[inline]
    fn sub_assign(&mut self, other: Affine<P>) {
        *self -= &other;
    }
}

/// Full addition: bucket += bucket.
/// <https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s>
impl<P: SWCurveConfig> AddAssign<&Bucket<P>> for Bucket<P> {
    fn add_assign(&mut self, other: &Self) {
        if self.is_zero() {
            *self = *other;
            return;
        }
        if other.is_zero() {
            return;
        }

        let z1z1 = self.zz;
        let z2z2 = other.zz;

        let mut u1 = self.x;
        u1 *= &z2z2;

        let mut u2 = other.x;
        u2 *= &z1z1;

        let s1 = self.y * other.zzz;
        let s2 = other.y * self.zzz;

        if u1 == u2 {
            if s1 == s2 {
                self.double_in_place();
            } else {
                *self = Self::ZERO;
            }
        } else {
            let mut p = u2;
            p -= &u1;

            let mut r = s2;
            r -= &s1;

            let mut pp = p;
            pp.square_in_place();

            let mut ppp = pp;
            ppp *= &p;

            let mut q = u1;
            q *= &pp;

            self.x = r.square();
            self.x -= &ppp;
            self.x -= &q.double();

            q -= &self.x;
            self.y = P::BaseField::sum_of_products(&[r, -s1], &[q, ppp]);

            self.zz *= &pp;
            self.zz *= &other.zz;

            self.zzz *= &ppp;
            self.zzz *= &other.zzz;
        }
    }
}

impl<P: SWCurveConfig> SubAssign<&Bucket<P>> for Bucket<P> {
    #[inline]
    fn sub_assign(&mut self, other: &Self) {
        *self += &(-*other);
    }
}

impl<P: SWCurveConfig> From<Bucket<P>> for Projective<P> {
    #[inline]
    fn from(p: Bucket<P>) -> Self {
        if p.is_zero() {
            Self::zero()
        } else {
            Self::new_unchecked(p.x * &p.zz, p.y * &p.zzz, p.zz)
        }
    }
}

impl<P: SWCurveConfig> AddAssign<&Bucket<P>> for Projective<P> {
    fn add_assign(&mut self, other: &Bucket<P>) {
        if self.is_zero() {
            *self = (*other).into();
            return;
        }
        if other.is_zero() {
            return;
        }
        *self += Self::from(*other);
    }
}
