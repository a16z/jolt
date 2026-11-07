//! Claim domains: the values a verifier check is written over, so one
//! formula serves every verifier that runs it.

use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

use jolt_field::Field;

/// The values a verifier check is written over.
///
/// A check is written once as a generic function over this trait so that
/// every verifier that runs it evaluates the same formula: the clear verifier
/// ([`ClearDomain`], where values are field elements and checks compare), a
/// masked verifier (values are affine forms over pads and each check becomes
/// a residual row) and a pad layout (values are symbolic and only slots are
/// counted). The masked and layout domains live in `jolt-zk`'s masking
/// lowering.
///
/// Implementations must allocate auxiliary slots
/// ([`define_auxiliary`](Self::define_auxiliary), [`product`](Self::product))
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

    /// A new auxiliary slot defined by the affine `value`; returns the slot.
    ///
    /// # Errors
    ///
    /// The domain's error if no slot is left.
    fn define_auxiliary(&mut self, value: Self::Value) -> Result<Self::Value, Self::Error>;

    /// A new auxiliary slot holding `x · y`, where `x` is a single masked or
    /// auxiliary slot; returns the slot.
    ///
    /// # Errors
    ///
    /// The domain's error if `x` is not a single slot or no slot is left.
    fn product(&mut self, x: Self::Value, y: Self::Value) -> Result<Self::Value, Self::Error>;

    /// The check `x · y = target`, with no new slot: the product lands on an
    /// existing value, such as a sumcheck's running claim. `x` is as for
    /// [`product`](Self::product). One pair of
    /// [`products_into`](Self::products_into).
    ///
    /// # Errors
    ///
    /// As [`products_into`](Self::products_into).
    fn product_into(
        &mut self,
        x: Self::Value,
        y: Self::Value,
        target: Self::Value,
    ) -> Result<(), Self::Error> {
        self.products_into((x, y), Vec::new(), target)
    }

    /// The check `Σ_k x_k · y_k = target` over the pairs `first` and
    /// `rest`, with no new slot: several products land on one existing
    /// value, such as a stage's running claim. Each `x_k` is as for
    /// [`product`](Self::product).
    ///
    /// # Errors
    ///
    /// The domain's error if the check fails where it is evaluated or an
    /// `x_k` is not a single slot.
    fn products_into(
        &mut self,
        first: (Self::Value, Self::Value),
        rest: Vec<(Self::Value, Self::Value)>,
        target: Self::Value,
    ) -> Result<(), Self::Error>;

    /// The check `value = 0`.
    ///
    /// # Errors
    ///
    /// The domain's error if the check fails where it is evaluated.
    fn check_zero(&mut self, value: Self::Value) -> Result<(), Self::Error>;
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

/// The clear verifier's domain: values are field elements, auxiliary slots
/// are the values themselves, and checks compare.
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

    fn define_auxiliary(&mut self, value: F) -> Result<F, ClaimMismatch> {
        Ok(value)
    }

    fn product(&mut self, x: F, y: F) -> Result<F, ClaimMismatch> {
        Ok(x * y)
    }

    fn products_into(
        &mut self,
        first: (F, F),
        rest: Vec<(F, F)>,
        target: F,
    ) -> Result<(), ClaimMismatch> {
        let sum = rest
            .iter()
            .fold(first.0 * first.1, |sum, (x, y)| sum + *x * *y);
        self.check_zero(sum - target)
    }

    fn check_zero(&mut self, value: F) -> Result<(), ClaimMismatch> {
        if value.is_zero() {
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
        assert_eq!(
            ClaimDomain::<Fp>::check_zero(&mut ClearDomain, Fp::from_u64(0)),
            Ok(())
        );
        assert_eq!(ClearDomain.check_zero(Fp::from_u64(1)), Err(ClaimMismatch));
    }

    #[test]
    fn clear_domain_checks_a_sum_of_products() {
        let pairs = [(2, 3), (5, 7), (11, 13)].map(|(x, y)| (Fp::from_u64(x), Fp::from_u64(y)));
        let target = Fp::from_u64(2 * 3 + 5 * 7 + 11 * 13);
        assert_eq!(
            ClearDomain.products_into(pairs[0], pairs[1..].to_vec(), target),
            Ok(())
        );
        assert_eq!(
            ClearDomain.products_into(pairs[0], pairs[1..].to_vec(), target + Fp::from_u64(1)),
            Err(ClaimMismatch)
        );
        assert_eq!(
            ClearDomain.products_into(pairs[0], pairs[1..2].to_vec(), target),
            Err(ClaimMismatch)
        );
    }
}
