//! Shared scalar helpers for R1CS verifier equations.
//!
//! These helpers let verifier equations be written once over either native
//! builder-field scalars or non-native scalars represented inside the builder.

use jolt_field::{Field, Fq, Fr};
use jolt_poly::r1cs::PolynomialScalarGadget;
use num_traits::{One, Zero};

use crate::{AssignedScalar, FqVar, LinearCombination, R1csBuilder};

pub trait ScalarGadget: Clone {
    type BuilderField: Field;
    type Scalar: Field;

    fn constant(scalar: Self::Scalar) -> Self;
    fn alloc(builder: &mut R1csBuilder<Self::BuilderField>, scalar: Self::Scalar) -> Self;
    fn assert_equal(&self, builder: &mut R1csBuilder<Self::BuilderField>, rhs: &Self);
    fn add(&self, builder: &mut R1csBuilder<Self::BuilderField>, rhs: &Self) -> Self;
    fn sub(&self, builder: &mut R1csBuilder<Self::BuilderField>, rhs: &Self) -> Self;
    fn mul(&self, builder: &mut R1csBuilder<Self::BuilderField>, rhs: &Self) -> Self;
    fn scale_by_constant(
        &self,
        builder: &mut R1csBuilder<Self::BuilderField>,
        scalar: Self::Scalar,
    ) -> Self;
    fn select(
        builder: &mut R1csBuilder<Self::BuilderField>,
        selector: &AssignedScalar<Self::BuilderField>,
        when_true: &Self,
        when_false: &Self,
    ) -> Self;
}

pub fn scalar_affine_combination<'a, S>(
    builder: &mut R1csBuilder<S::BuilderField>,
    constant: S::Scalar,
    terms: impl IntoIterator<Item = (S::Scalar, &'a S)>,
) -> S
where
    S: ScalarGadget + 'a,
{
    let mut result = S::constant(constant);
    for (coefficient, scalar) in terms {
        let term = scalar.scale_by_constant(builder, coefficient);
        result = result.add(builder, &term);
    }
    result
}

pub fn scalar_dot_product<'a, S>(
    builder: &mut R1csBuilder<S::BuilderField>,
    terms: impl IntoIterator<Item = (&'a S, &'a S)>,
) -> S
where
    S: ScalarGadget + 'a,
{
    let mut result = S::constant(S::Scalar::zero());
    for (lhs, rhs) in terms {
        let term = lhs.mul(builder, rhs);
        result = result.add(builder, &term);
    }
    result
}

impl<F> ScalarGadget for AssignedScalar<F>
where
    F: Field,
{
    type BuilderField = F;
    type Scalar = F;

    fn constant(scalar: Self::Scalar) -> Self {
        AssignedScalar::constant(scalar)
    }

    fn alloc(builder: &mut R1csBuilder<Self::BuilderField>, scalar: Self::Scalar) -> Self {
        AssignedScalar::alloc(builder, scalar)
    }

    fn assert_equal(&self, builder: &mut R1csBuilder<Self::BuilderField>, rhs: &Self) {
        builder.assert_equal(self.lc.clone(), rhs.lc.clone());
    }

    fn add(&self, _builder: &mut R1csBuilder<Self::BuilderField>, rhs: &Self) -> Self {
        Self::new(self.value + rhs.value, self.lc.clone() + rhs.lc.clone())
    }

    fn sub(&self, _builder: &mut R1csBuilder<Self::BuilderField>, rhs: &Self) -> Self {
        Self::new(self.value - rhs.value, self.lc.clone() - rhs.lc.clone())
    }

    fn mul(&self, builder: &mut R1csBuilder<Self::BuilderField>, rhs: &Self) -> Self {
        Self::new(
            self.value * rhs.value,
            builder.multiply(self.lc.clone(), rhs.lc.clone()),
        )
    }

    fn scale_by_constant(
        &self,
        _builder: &mut R1csBuilder<Self::BuilderField>,
        scalar: Self::Scalar,
    ) -> Self {
        self.clone().scale(scalar)
    }

    fn select(
        builder: &mut R1csBuilder<Self::BuilderField>,
        selector: &AssignedScalar<Self::BuilderField>,
        when_true: &Self,
        when_false: &Self,
    ) -> Self {
        assert_boolean(builder, selector);

        let selected_delta = builder.multiply(
            selector.lc.clone(),
            when_true.lc.clone() - when_false.lc.clone(),
        );
        Self::new(
            when_false.value + selector.value * (when_true.value - when_false.value),
            when_false.lc.clone() + selected_delta,
        )
    }
}

impl ScalarGadget for FqVar {
    type BuilderField = Fr;
    type Scalar = Fq;

    fn constant(scalar: Self::Scalar) -> Self {
        FqVar::constant(scalar)
    }

    fn alloc(builder: &mut R1csBuilder<Self::BuilderField>, scalar: Self::Scalar) -> Self {
        FqVar::alloc(builder, scalar)
    }

    fn assert_equal(&self, builder: &mut R1csBuilder<Self::BuilderField>, rhs: &Self) {
        self.assert_equal(builder, rhs);
    }

    fn add(&self, builder: &mut R1csBuilder<Self::BuilderField>, rhs: &Self) -> Self {
        self.add(builder, rhs)
    }

    fn sub(&self, builder: &mut R1csBuilder<Self::BuilderField>, rhs: &Self) -> Self {
        self.sub(builder, rhs)
    }

    fn mul(&self, builder: &mut R1csBuilder<Self::BuilderField>, rhs: &Self) -> Self {
        self.mul(builder, rhs)
    }

    fn scale_by_constant(
        &self,
        builder: &mut R1csBuilder<Self::BuilderField>,
        scalar: Self::Scalar,
    ) -> Self {
        if scalar.is_zero() {
            Self::constant(Fq::zero())
        } else if scalar == Fq::one() {
            self.clone()
        } else {
            self.mul(builder, &Self::constant(scalar))
        }
    }

    fn select(
        builder: &mut R1csBuilder<Self::BuilderField>,
        selector: &AssignedScalar<Self::BuilderField>,
        when_true: &Self,
        when_false: &Self,
    ) -> Self {
        FqVar::select(builder, selector, when_true, when_false)
    }
}

impl<F> PolynomialScalarGadget for AssignedScalar<F>
where
    F: Field,
{
    type ConstraintBuilder = R1csBuilder<F>;
    type Scalar = F;

    fn constant(scalar: Self::Scalar) -> Self {
        <Self as ScalarGadget>::constant(scalar)
    }

    fn add(&self, builder: &mut Self::ConstraintBuilder, rhs: &Self) -> Self {
        <Self as ScalarGadget>::add(self, builder, rhs)
    }

    fn sub(&self, builder: &mut Self::ConstraintBuilder, rhs: &Self) -> Self {
        <Self as ScalarGadget>::sub(self, builder, rhs)
    }

    fn mul(&self, builder: &mut Self::ConstraintBuilder, rhs: &Self) -> Self {
        <Self as ScalarGadget>::mul(self, builder, rhs)
    }
}

impl PolynomialScalarGadget for FqVar {
    type ConstraintBuilder = R1csBuilder<Fr>;
    type Scalar = Fq;

    fn constant(scalar: Self::Scalar) -> Self {
        <Self as ScalarGadget>::constant(scalar)
    }

    fn add(&self, builder: &mut Self::ConstraintBuilder, rhs: &Self) -> Self {
        <Self as ScalarGadget>::add(self, builder, rhs)
    }

    fn sub(&self, builder: &mut Self::ConstraintBuilder, rhs: &Self) -> Self {
        <Self as ScalarGadget>::sub(self, builder, rhs)
    }

    fn mul(&self, builder: &mut Self::ConstraintBuilder, rhs: &Self) -> Self {
        <Self as ScalarGadget>::mul(self, builder, rhs)
    }
}

fn assert_boolean<F>(builder: &mut R1csBuilder<F>, value: &AssignedScalar<F>)
where
    F: Field,
{
    builder.assert_product(
        value.lc.clone(),
        value.lc.clone() - LinearCombination::one(),
        LinearCombination::zero(),
    );
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use jolt_field::{Fq, Fr, FromPrimitiveInt};

    use super::*;
    use crate::Variable;

    #[test]
    fn native_scalar_relation_accepts_valid_witnesses() {
        for selector in [false, true] {
            let mut builder = R1csBuilder::<Fr>::new();
            let x = AssignedScalar::alloc(&mut builder, Fr::from_u64(9));
            let y = AssignedScalar::alloc(&mut builder, Fr::from_u64(12));
            let z = AssignedScalar::alloc(&mut builder, Fr::from_u64(5));
            let w = AssignedScalar::alloc(&mut builder, Fr::from_u64(4));
            let selector = AssignedScalar::alloc(&mut builder, Fr::from_bool(selector));

            let result = shared_relation(&mut builder, &selector, &x, &y, &z, &w);
            result.assert_equal(
                &mut builder,
                &AssignedScalar::constant(expected_relation(
                    selector.value == Fr::one(),
                    x.value,
                    y.value,
                    z.value,
                    w.value,
                )),
            );

            assert!(builder_accepts(builder));
        }
    }

    #[test]
    fn native_scalar_relation_rejects_single_variable_tampering() {
        let mut builder = R1csBuilder::<Fr>::new();
        let x = AssignedScalar::alloc(&mut builder, Fr::from_u64(9));
        let y = AssignedScalar::alloc(&mut builder, Fr::from_u64(12));
        let z = AssignedScalar::alloc(&mut builder, Fr::from_u64(5));
        let w = AssignedScalar::alloc(&mut builder, Fr::from_u64(4));
        let selector = AssignedScalar::alloc(&mut builder, Fr::one());

        let result = shared_relation(&mut builder, &selector, &x, &y, &z, &w);
        result.assert_equal(
            &mut builder,
            &AssignedScalar::constant(expected_relation(true, x.value, y.value, z.value, w.value)),
        );

        assert_single_variable_tampering_rejected(builder);
    }

    #[test]
    fn nonnative_scalar_relation_accepts_valid_witnesses() {
        for selector in [false, true] {
            let mut builder = R1csBuilder::<Fr>::new();
            let x = FqVar::alloc(&mut builder, Fq::from_u64(9));
            let y = FqVar::alloc(&mut builder, Fq::from_u64(12));
            let z = FqVar::alloc(&mut builder, Fq::from_u64(5));
            let w = FqVar::alloc(&mut builder, Fq::from_u64(4));
            let selector = AssignedScalar::alloc(&mut builder, Fr::from_bool(selector));

            let result = shared_relation(&mut builder, &selector, &x, &y, &z, &w);
            result.assert_equal(
                &mut builder,
                &FqVar::constant(expected_relation(
                    selector.value == Fr::one(),
                    Fq::from_u64(9),
                    Fq::from_u64(12),
                    Fq::from_u64(5),
                    Fq::from_u64(4),
                )),
            );

            assert!(builder_accepts(builder));
        }
    }

    #[test]
    fn nonnative_scalar_relation_rejects_single_variable_tampering() {
        let mut builder = R1csBuilder::<Fr>::new();
        let x = FqVar::alloc(&mut builder, Fq::from_u64(9));
        let y = FqVar::alloc(&mut builder, Fq::from_u64(12));
        let z = FqVar::alloc(&mut builder, Fq::from_u64(5));
        let w = FqVar::alloc(&mut builder, Fq::from_u64(4));
        let selector = AssignedScalar::alloc(&mut builder, Fr::one());

        let result = shared_relation(&mut builder, &selector, &x, &y, &z, &w);
        result.assert_equal(
            &mut builder,
            &FqVar::constant(expected_relation(
                true,
                Fq::from_u64(9),
                Fq::from_u64(12),
                Fq::from_u64(5),
                Fq::from_u64(4),
            )),
        );

        assert_variable_tampering_rejected(
            builder,
            [
                variable(&selector),
                variable(&x.limbs()[0]),
                variable(&result.limbs()[0]),
            ],
        );
    }

    fn shared_relation<S>(
        builder: &mut R1csBuilder<S::BuilderField>,
        selector: &AssignedScalar<S::BuilderField>,
        x: &S,
        y: &S,
        z: &S,
        w: &S,
    ) -> S
    where
        S: ScalarGadget,
    {
        let dot_product = scalar_dot_product(builder, [(x, y), (z, w)]);
        let true_branch = scalar_affine_combination(
            builder,
            S::Scalar::from_u64(7),
            [
                (S::Scalar::from_u64(3), &dot_product),
                (S::Scalar::from_u64(5), x),
            ],
        );

        let product = y.mul(builder, z);
        let difference = product.sub(builder, w);
        let false_branch = scalar_affine_combination(
            builder,
            -S::Scalar::from_u64(11),
            [(S::Scalar::from_u64(2), &difference), (S::Scalar::one(), z)],
        );

        S::select(builder, selector, &true_branch, &false_branch)
    }

    fn expected_relation<F>(selector: bool, x: F, y: F, z: F, w: F) -> F
    where
        F: Field,
    {
        let dot_product = x * y + z * w;
        let true_branch = F::from_u64(7) + F::from_u64(3) * dot_product + F::from_u64(5) * x;
        let false_branch = -F::from_u64(11) + F::from_u64(2) * (y * z - w) + z;
        if selector {
            true_branch
        } else {
            false_branch
        }
    }

    fn builder_accepts<F>(builder: R1csBuilder<F>) -> bool
    where
        F: Field,
    {
        let witness = builder.witness().expect("witness is assigned");
        builder.into_matrices().check_witness(&witness).is_ok()
    }

    fn assert_single_variable_tampering_rejected<F>(builder: R1csBuilder<F>)
    where
        F: Field,
    {
        let variables = (1..builder.num_vars())
            .map(Variable::new)
            .collect::<Vec<_>>();
        assert_variable_tampering_rejected(builder, variables);
    }

    fn assert_variable_tampering_rejected<F>(
        builder: R1csBuilder<F>,
        variables: impl IntoIterator<Item = Variable>,
    ) where
        F: Field,
    {
        let witness = builder.witness().expect("witness is assigned");
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());

        for variable in variables {
            let mut tampered = witness.clone();
            let index = variable.index();
            tampered[index] += F::one();
            assert!(
                matrices.check_witness(&tampered).is_err(),
                "variable {index} accepted after single-variable tampering"
            );
        }
    }

    fn variable<F>(scalar: &AssignedScalar<F>) -> Variable
    where
        F: Field,
    {
        assert_eq!(scalar.lc.terms.len(), 1);
        let (variable, coefficient) = scalar
            .lc
            .terms
            .first()
            .copied()
            .expect("linear combination has one term");
        assert_eq!(coefficient, F::one());
        variable
    }
}
