//! Claim domains: the values a verifier check is written over, so one
//! formula serves every verifier that runs it.

use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

use jolt_field::Field;

/// The values a verifier check is written over.
///
/// A check is written once as a generic function over this trait so that
/// every verifier that runs it evaluates the same formula. [`ClearDomain`] is
/// the only implementation here; a masked domain (values as affine forms over
/// pads, each check a residual row) does not exist yet.
///
/// Implementations must allocate auxiliary slots ([`product`](Self::product))
/// from a formula's structure alone, never from values, so that every domain
/// running one formula allocates the same slots in the same order, and must
/// reject an `x` operand that is not a single slot. [`ClearDomain`] does not
/// enforce either rule: it has no slots and only compares field values, so a
/// formula that breaks them still passes there.
pub trait ClaimDomain<F> {
    /// One value of the domain.
    type Value: Clone;
    /// Why a check or an allocation failed.
    type Error;

    /// The public constant `value`.
    fn constant(&self, value: F) -> Self::Value;

    /// `lhs + rhs`.
    fn add(&self, lhs: &Self::Value, rhs: &Self::Value) -> Self::Value;

    /// `value · scale` for a public `scale`.
    fn scale(&self, value: &Self::Value, scale: F) -> Self::Value;

    /// A new auxiliary slot holding `x · y`, where `x` is a single masked or
    /// auxiliary slot; returns the slot.
    ///
    /// # Errors
    ///
    /// The domain's error if `x` is not a single slot or no slot is left.
    fn product(&mut self, x: Self::Value, y: Self::Value) -> Result<Self::Value, Self::Error>;

    /// The check `x · y = target`, with no new slot: the product lands on an
    /// existing value, such as a sumcheck's running claim. `x` is as for
    /// [`product`](Self::product).
    ///
    /// # Errors
    ///
    /// The domain's error if the check fails where it is evaluated or `x` is
    /// not a single slot.
    fn product_into(
        &mut self,
        x: Self::Value,
        y: Self::Value,
        target: Self::Value,
    ) -> Result<(), Self::Error>;
}

/// A clear check failed: the values it compares differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClaimMismatch;

impl Display for ClaimMismatch {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("a clear claim check failed")
    }
}

impl Error for ClaimMismatch {}

/// The clear verifier's domain: values are field elements, products are
/// computed directly, and checks compare.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClearDomain;

impl<F: Field> ClaimDomain<F> for ClearDomain {
    type Value = F;
    type Error = ClaimMismatch;

    fn constant(&self, value: F) -> F {
        value
    }

    fn add(&self, lhs: &F, rhs: &F) -> F {
        *lhs + *rhs
    }

    fn scale(&self, value: &F, scale: F) -> F {
        *value * scale
    }

    fn product(&mut self, x: F, y: F) -> Result<F, ClaimMismatch> {
        Ok(x * y)
    }

    fn product_into(&mut self, x: F, y: F, target: F) -> Result<(), ClaimMismatch> {
        if x * y == target {
            Ok(())
        } else {
            Err(ClaimMismatch)
        }
    }
}

#[cfg(all(test, feature = "solinas"))]
mod tests {
    use jolt_field::{Prime128Offset275 as Fp, Ring};

    use super::{ClaimDomain, ClaimMismatch, ClearDomain};

    /// A degree-4 claim `Σ_{k≤4} c_k·r^k` lowered with one product slot:
    /// `u = r·r`, then `u·(c₄u + c₃r + c₂) = claim − c₁r − c₀` lands on the
    /// claim, the shape a masked verifier needs where the degree exceeds two.
    fn degree_four_leaf<D: ClaimDomain<Fp>>(
        domain: &mut D,
        r: D::Value,
        c: [Fp; 5],
        claim: D::Value,
    ) -> Result<(), D::Error> {
        let u = domain.product(r.clone(), r.clone())?;
        let y = domain.add(
            &domain.add(&domain.scale(&u, c[4]), &domain.scale(&r, c[3])),
            &domain.constant(c[2]),
        );
        let target = domain.add(
            &claim,
            &domain.add(&domain.scale(&r, -c[1]), &domain.constant(-c[0])),
        );
        domain.product_into(u, y, target)
    }

    #[test]
    fn clear_domain_compares_checks() {
        let c = [3, 5, 7, 11, 13].map(Fp::from_u64);
        let r = Fp::from_u64(17);
        let claim = c
            .iter()
            .rev()
            .fold(Fp::from_u64(0), |sum, &coefficient| sum * r + coefficient);
        assert_eq!(degree_four_leaf(&mut ClearDomain, r, c, claim), Ok(()));
        assert_eq!(
            degree_four_leaf(&mut ClearDomain, r, c, claim + Fp::from_u64(1)),
            Err(ClaimMismatch)
        );
    }
}
