//! R1CS helpers for cryptographic group operations.

use ark_ec::CurveGroup;
use jolt_field::{Fr, FromPrimitiveInt, Invertible};
use jolt_r1cs::{AssignedScalar, FqVar, LinearCombination, R1csBuilder, ScalarGadget};
use num_traits::{One, Zero};
use thiserror::Error;

use crate::{GrumpkinPoint, JoltGroup, Pedersen, PedersenSetup};

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CryptoR1csError {
    #[error("affine Grumpkin point gadget does not represent the identity")]
    IdentityPoint,
    #[error("non-exceptional affine addition requires distinct input x-coordinates")]
    ExceptionalAffineAddition,
    #[error("affine doubling requires non-zero y-coordinate")]
    ExceptionalAffineDoubling,
    #[error("fixed-base MSM length mismatch: bases={bases}, scalars={scalars}")]
    FixedBaseMsmLengthMismatch { bases: usize, scalars: usize },
    #[error("vector commitment opening length {values} exceeds setup capacity {capacity}")]
    VectorCommitmentCapacityExceeded { capacity: usize, values: usize },
}

pub trait GroupElementVar: Clone {
    type BuilderField: jolt_field::Field;
    type ScalarVar: jolt_poly::r1cs::PolynomialScalarGadget<
        ConstraintBuilder = R1csBuilder<Self::BuilderField>,
    >;
    type Error;

    fn assert_valid(
        &self,
        builder: &mut R1csBuilder<Self::BuilderField>,
    ) -> Result<(), Self::Error>;

    fn assert_equal(&self, builder: &mut R1csBuilder<Self::BuilderField>, rhs: &Self);
}

pub trait NonExceptionalAddGroupVar: GroupElementVar {
    fn assert_nonexceptional_add(
        builder: &mut R1csBuilder<Self::BuilderField>,
        lhs: &Self,
        rhs: &Self,
        output: &Self,
    ) -> Result<(), Self::Error>;
}

pub trait CompleteAddGroupVar: GroupElementVar {
    fn complete_add(
        builder: &mut R1csBuilder<Self::BuilderField>,
        lhs: &Self,
        rhs: &Self,
    ) -> Result<Self, Self::Error>;
}

pub trait DoubleGroupVar: GroupElementVar {
    fn assert_double(
        builder: &mut R1csBuilder<Self::BuilderField>,
        input: &Self,
        output: &Self,
    ) -> Result<(), Self::Error>;
}

pub trait VariableBaseScalarMulGroupVar: CompleteAddGroupVar {
    fn variable_base_scalar_mul(
        builder: &mut R1csBuilder<Self::BuilderField>,
        base: &Self,
        scalar: &Self::ScalarVar,
    ) -> Result<Self, Self::Error>;
}

pub trait FixedBaseScalarMulGroupVar: GroupElementVar {
    type Constant;

    fn fixed_base_scalar_mul(
        builder: &mut R1csBuilder<Self::BuilderField>,
        base: &Self::Constant,
        scalar: &Self::ScalarVar,
    ) -> Result<Self, Self::Error>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VectorCommitmentOpeningVar<S>
where
    S: jolt_poly::r1cs::PolynomialScalarGadget,
{
    pub values: Vec<S>,
    pub blinding: S,
}

impl<S> VectorCommitmentOpeningVar<S>
where
    S: jolt_poly::r1cs::PolynomialScalarGadget,
{
    pub fn new(values: Vec<S>, blinding: S) -> Self {
        Self { values, blinding }
    }
}

pub trait VectorCommitmentR1cs {
    type BuilderField: jolt_field::Field;
    type ScalarVar: jolt_poly::r1cs::PolynomialScalarGadget<
        ConstraintBuilder = R1csBuilder<Self::BuilderField>,
    >;
    type OutputVar: GroupElementVar<BuilderField = Self::BuilderField, ScalarVar = Self::ScalarVar>;
    type SetupVar;
    type Error;

    fn capacity(setup: &Self::SetupVar) -> usize;

    fn linear_combine_commitments(
        builder: &mut R1csBuilder<Self::BuilderField>,
        commitments: &[Self::OutputVar],
        coefficients: &[Self::ScalarVar],
    ) -> Result<Self::OutputVar, Self::Error>;

    fn verify_opening(
        builder: &mut R1csBuilder<Self::BuilderField>,
        setup: &Self::SetupVar,
        commitment: &Self::OutputVar,
        opening: &VectorCommitmentOpeningVar<Self::ScalarVar>,
    ) -> Result<(), Self::Error>;
}

/// Affine Grumpkin point with coordinates in the BN254 scalar field `Fr`.
///
/// This is an affine non-identity representation. It is useful for early
/// Grumpkin equation gadgets, but a full Pedersen verifier should move to a
/// complete representation before accepting arbitrary commitment outputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrumpkinPointVar {
    pub x: AssignedScalar<Fr>,
    pub y: AssignedScalar<Fr>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrumpkinPointWithIdentityVar {
    pub x: AssignedScalar<Fr>,
    pub y: AssignedScalar<Fr>,
    pub is_identity: AssignedScalar<Fr>,
}

impl GrumpkinPointVar {
    pub fn new(x: AssignedScalar<Fr>, y: AssignedScalar<Fr>) -> Self {
        Self { x, y }
    }

    pub fn alloc(
        builder: &mut R1csBuilder<Fr>,
        point: &GrumpkinPoint,
    ) -> Result<Self, CryptoR1csError> {
        let (x, y) = grumpkin_coordinates(point)?;
        Ok(Self {
            x: AssignedScalar::alloc(builder, x),
            y: AssignedScalar::alloc(builder, y),
        })
    }

    pub fn constant(point: &GrumpkinPoint) -> Result<Self, CryptoR1csError> {
        let (x, y) = grumpkin_coordinates(point)?;
        Ok(Self {
            x: AssignedScalar::constant(x),
            y: AssignedScalar::constant(y),
        })
    }

    pub fn assert_equal(&self, builder: &mut R1csBuilder<Fr>, rhs: &Self) {
        self.x.assert_equal(builder, &rhs.x);
        self.y.assert_equal(builder, &rhs.y);
    }

    pub fn assert_on_curve(&self, builder: &mut R1csBuilder<Fr>) {
        let y_squared = self.y.mul(builder, &self.y);
        let x_squared = self.x.mul(builder, &self.x);
        let x_cubed = x_squared.mul(builder, &self.x);
        let rhs = x_cubed.add(builder, &AssignedScalar::constant(-Fr::from_u64(17)));
        y_squared.assert_equal(builder, &rhs);
    }
}

impl GrumpkinPointWithIdentityVar {
    pub fn new(
        x: AssignedScalar<Fr>,
        y: AssignedScalar<Fr>,
        is_identity: AssignedScalar<Fr>,
    ) -> Self {
        Self { x, y, is_identity }
    }

    pub fn identity() -> Self {
        Self {
            x: AssignedScalar::constant(Fr::zero()),
            y: AssignedScalar::constant(Fr::zero()),
            is_identity: AssignedScalar::constant(Fr::one()),
        }
    }

    pub fn alloc(builder: &mut R1csBuilder<Fr>, point: &GrumpkinPoint) -> Self {
        let affine = point.0.into_affine();
        if affine.infinity {
            return Self {
                x: AssignedScalar::alloc(builder, Fr::zero()),
                y: AssignedScalar::alloc(builder, Fr::zero()),
                is_identity: AssignedScalar::alloc(builder, Fr::one()),
            };
        }

        Self {
            x: AssignedScalar::alloc(builder, Fr::from(affine.x)),
            y: AssignedScalar::alloc(builder, Fr::from(affine.y)),
            is_identity: AssignedScalar::alloc(builder, Fr::zero()),
        }
    }

    pub fn constant(point: &GrumpkinPoint) -> Self {
        let affine = point.0.into_affine();
        if affine.infinity {
            return Self::identity();
        }

        Self {
            x: AssignedScalar::constant(Fr::from(affine.x)),
            y: AssignedScalar::constant(Fr::from(affine.y)),
            is_identity: AssignedScalar::constant(Fr::zero()),
        }
    }

    pub fn from_nonidentity(point: GrumpkinPointVar) -> Self {
        Self {
            x: point.x,
            y: point.y,
            is_identity: AssignedScalar::constant(Fr::zero()),
        }
    }

    pub fn assert_valid(&self, builder: &mut R1csBuilder<Fr>) {
        assert_boolean(builder, &self.is_identity);
        builder.assert_product(
            self.is_identity.lc.clone(),
            self.x.lc.clone(),
            LinearCombination::zero(),
        );
        builder.assert_product(
            self.is_identity.lc.clone(),
            self.y.lc.clone(),
            LinearCombination::zero(),
        );

        let y_squared = self.y.mul(builder, &self.y);
        let x_squared = self.x.mul(builder, &self.x);
        let x_cubed = x_squared.mul(builder, &self.x);
        let rhs = x_cubed.add(builder, &AssignedScalar::constant(-Fr::from_u64(17)));
        let curve_residual = y_squared.sub(builder, &rhs);
        let not_identity = AssignedScalar::constant(Fr::one()).sub(builder, &self.is_identity);
        builder.assert_product(
            not_identity.lc,
            curve_residual.lc,
            LinearCombination::zero(),
        );
    }

    pub fn assert_equal(&self, builder: &mut R1csBuilder<Fr>, rhs: &Self) {
        self.x.assert_equal(builder, &rhs.x);
        self.y.assert_equal(builder, &rhs.y);
        self.is_identity.assert_equal(builder, &rhs.is_identity);
    }

    pub fn select(
        builder: &mut R1csBuilder<Fr>,
        selector: &AssignedScalar<Fr>,
        when_true: &Self,
        when_false: &Self,
    ) -> Self {
        Self {
            x: <AssignedScalar<Fr> as ScalarGadget>::select(
                builder,
                selector,
                &when_true.x,
                &when_false.x,
            ),
            y: <AssignedScalar<Fr> as ScalarGadget>::select(
                builder,
                selector,
                &when_true.y,
                &when_false.y,
            ),
            is_identity: <AssignedScalar<Fr> as ScalarGadget>::select(
                builder,
                selector,
                &when_true.is_identity,
                &when_false.is_identity,
            ),
        }
    }

    pub fn witness_value(&self) -> GrumpkinPoint {
        if self.is_identity.value == Fr::one() {
            GrumpkinPoint::identity()
        } else {
            grumpkin_point_from_coordinates(self.x.value, self.y.value)
        }
    }
}

impl GroupElementVar for GrumpkinPointVar {
    type BuilderField = Fr;
    type ScalarVar = jolt_r1cs::FqVar;
    type Error = CryptoR1csError;

    fn assert_valid(
        &self,
        builder: &mut R1csBuilder<Self::BuilderField>,
    ) -> Result<(), Self::Error> {
        self.assert_on_curve(builder);
        Ok(())
    }

    fn assert_equal(&self, builder: &mut R1csBuilder<Self::BuilderField>, rhs: &Self) {
        GrumpkinPointVar::assert_equal(self, builder, rhs);
    }
}

impl GroupElementVar for GrumpkinPointWithIdentityVar {
    type BuilderField = Fr;
    type ScalarVar = FqVar;
    type Error = CryptoR1csError;

    fn assert_valid(
        &self,
        builder: &mut R1csBuilder<Self::BuilderField>,
    ) -> Result<(), Self::Error> {
        self.assert_valid(builder);
        Ok(())
    }

    fn assert_equal(&self, builder: &mut R1csBuilder<Self::BuilderField>, rhs: &Self) {
        GrumpkinPointWithIdentityVar::assert_equal(self, builder, rhs);
    }
}

impl CompleteAddGroupVar for GrumpkinPointWithIdentityVar {
    fn complete_add(
        builder: &mut R1csBuilder<Self::BuilderField>,
        lhs: &Self,
        rhs: &Self,
    ) -> Result<Self, Self::Error> {
        complete_grumpkin_add(builder, lhs, rhs)
    }
}

impl VariableBaseScalarMulGroupVar for GrumpkinPointWithIdentityVar {
    fn variable_base_scalar_mul(
        builder: &mut R1csBuilder<Self::BuilderField>,
        base: &Self,
        scalar: &Self::ScalarVar,
    ) -> Result<Self, Self::Error> {
        variable_base_grumpkin_scalar_mul(builder, base, scalar)
    }
}

impl FixedBaseScalarMulGroupVar for GrumpkinPointWithIdentityVar {
    type Constant = GrumpkinPoint;

    fn fixed_base_scalar_mul(
        builder: &mut R1csBuilder<Self::BuilderField>,
        base: &Self::Constant,
        scalar: &Self::ScalarVar,
    ) -> Result<Self, Self::Error> {
        fixed_base_grumpkin_scalar_mul(builder, base, scalar)
    }
}

impl NonExceptionalAddGroupVar for GrumpkinPointVar {
    fn assert_nonexceptional_add(
        builder: &mut R1csBuilder<Self::BuilderField>,
        lhs: &Self,
        rhs: &Self,
        output: &Self,
    ) -> Result<(), Self::Error> {
        assert_nonexceptional_grumpkin_add(builder, lhs, rhs, output)
    }
}

impl DoubleGroupVar for GrumpkinPointVar {
    fn assert_double(
        builder: &mut R1csBuilder<Self::BuilderField>,
        input: &Self,
        output: &Self,
    ) -> Result<(), Self::Error> {
        assert_grumpkin_double(builder, input, output)
    }
}

impl VectorCommitmentR1cs for Pedersen<GrumpkinPoint> {
    type BuilderField = Fr;
    type ScalarVar = FqVar;
    type OutputVar = GrumpkinPointWithIdentityVar;
    type SetupVar = PedersenSetup<GrumpkinPoint>;
    type Error = CryptoR1csError;

    fn capacity(setup: &Self::SetupVar) -> usize {
        setup.message_generators.len()
    }

    fn linear_combine_commitments(
        builder: &mut R1csBuilder<Self::BuilderField>,
        commitments: &[Self::OutputVar],
        coefficients: &[Self::ScalarVar],
    ) -> Result<Self::OutputVar, Self::Error> {
        linear_combine_grumpkin_commitments(builder, commitments, coefficients)
    }

    fn verify_opening(
        builder: &mut R1csBuilder<Self::BuilderField>,
        setup: &Self::SetupVar,
        commitment: &Self::OutputVar,
        opening: &VectorCommitmentOpeningVar<Self::ScalarVar>,
    ) -> Result<(), Self::Error> {
        verify_grumpkin_pedersen_opening(builder, setup, commitment, opening)
    }
}

/// Constrains `output = 2 * input` for affine Grumpkin points.
///
/// The formula is valid for non-identity affine points with non-zero
/// y-coordinate. In the Grumpkin prime-order subgroup this is the ordinary
/// doubling case used by scalar multiplication.
pub fn assert_grumpkin_double(
    builder: &mut R1csBuilder<Fr>,
    input: &GrumpkinPointVar,
    output: &GrumpkinPointVar,
) -> Result<(), CryptoR1csError> {
    input.assert_on_curve(builder);
    output.assert_on_curve(builder);

    let two_y = input.y.scale_by_constant(builder, Fr::from_u64(2));
    if two_y.value.is_zero() {
        return Err(CryptoR1csError::ExceptionalAffineDoubling);
    }
    let two_y_inverse = AssignedScalar::alloc(
        builder,
        two_y
            .value
            .inverse()
            .ok_or(CryptoR1csError::ExceptionalAffineDoubling)?,
    );
    builder.assert_product(
        two_y.lc.clone(),
        two_y_inverse.lc.clone(),
        LinearCombination::one(),
    );

    let x_squared = input.x.mul(builder, &input.x);
    let three_x_squared = x_squared.scale_by_constant(builder, Fr::from_u64(3));
    let slope = three_x_squared.mul(builder, &two_y_inverse);
    let slope_squared = slope.mul(builder, &slope);
    let two_x = input.x.scale_by_constant(builder, Fr::from_u64(2));
    let expected_slope_squared = output.x.add(builder, &two_x);
    slope_squared.assert_equal(builder, &expected_slope_squared);

    let x_delta = input.x.sub(builder, &output.x);
    let y_delta = output.y.add(builder, &input.y);
    let expected_y_delta = slope.mul(builder, &x_delta);
    y_delta.assert_equal(builder, &expected_y_delta);

    Ok(())
}

/// Constrains `output = lhs + rhs` for identity-aware affine points.
///
/// The non-identity/non-identity case still uses the ordinary affine addition
/// relation, so doubling and inverse-pair addition are intentionally rejected.
pub fn assert_grumpkin_add_with_identity(
    builder: &mut R1csBuilder<Fr>,
    lhs: &GrumpkinPointWithIdentityVar,
    rhs: &GrumpkinPointWithIdentityVar,
    output: &GrumpkinPointWithIdentityVar,
) -> Result<(), CryptoR1csError> {
    lhs.assert_valid(builder);
    rhs.assert_valid(builder);
    output.assert_valid(builder);

    let not_lhs_identity = AssignedScalar::constant(Fr::one()).sub(builder, &lhs.is_identity);
    let not_rhs_identity = AssignedScalar::constant(Fr::one()).sub(builder, &rhs.is_identity);
    let rhs_only_gate = not_lhs_identity.mul(builder, &rhs.is_identity);
    let active = not_lhs_identity.mul(builder, &not_rhs_identity);
    let expected_output_identity = lhs.is_identity.mul(builder, &rhs.is_identity);
    output
        .is_identity
        .assert_equal(builder, &expected_output_identity);

    gated_assert_equal(builder, &lhs.is_identity, &output.x, &rhs.x);
    gated_assert_equal(builder, &lhs.is_identity, &output.y, &rhs.y);
    gated_assert_equal(builder, &rhs_only_gate, &output.x, &lhs.x);
    gated_assert_equal(builder, &rhs_only_gate, &output.y, &lhs.y);

    let dx = rhs.x.sub(builder, &lhs.x);
    if !active.value.is_zero() && dx.value.is_zero() {
        return Err(CryptoR1csError::ExceptionalAffineAddition);
    }
    let dx_inverse_value = if active.value.is_zero() {
        Fr::zero()
    } else {
        dx.value
            .inverse()
            .ok_or(CryptoR1csError::ExceptionalAffineAddition)?
    };
    let dx_inverse = AssignedScalar::alloc(builder, dx_inverse_value);
    let dx_times_inverse = dx.mul(builder, &dx_inverse);
    let inverse_residual = dx_times_inverse.sub(builder, &AssignedScalar::constant(Fr::one()));
    gated_assert_zero(builder, &active, &inverse_residual);

    let dy = rhs.y.sub(builder, &lhs.y);
    let slope = dy.mul(builder, &dx_inverse);
    let slope_squared = slope.mul(builder, &slope);
    let expected_slope_squared = output.x.add(builder, &lhs.x).add(builder, &rhs.x);
    let slope_residual = slope_squared.sub(builder, &expected_slope_squared);
    gated_assert_zero(builder, &active, &slope_residual);

    let x_delta = lhs.x.sub(builder, &output.x);
    let y_delta = output.y.add(builder, &lhs.y);
    let expected_y_delta = slope.mul(builder, &x_delta);
    let y_residual = y_delta.sub(builder, &expected_y_delta);
    gated_assert_zero(builder, &active, &y_residual);

    Ok(())
}

/// Allocates and constrains `lhs + rhs` for all affine Grumpkin cases.
pub fn complete_grumpkin_add(
    builder: &mut R1csBuilder<Fr>,
    lhs: &GrumpkinPointWithIdentityVar,
    rhs: &GrumpkinPointWithIdentityVar,
) -> Result<GrumpkinPointWithIdentityVar, CryptoR1csError> {
    let output =
        GrumpkinPointWithIdentityVar::alloc(builder, &(lhs.witness_value() + rhs.witness_value()));
    assert_complete_grumpkin_add(builder, lhs, rhs, &output)?;
    Ok(output)
}

/// Constrains `output = lhs + rhs` for identity-aware affine Grumpkin points.
///
/// This relation covers identity cases, ordinary affine addition, doubling,
/// and inverse-pair addition to the identity. Case flags are constrained with
/// zero tests and gates, so callers do not need to preclude exceptional inputs.
pub fn assert_complete_grumpkin_add(
    builder: &mut R1csBuilder<Fr>,
    lhs: &GrumpkinPointWithIdentityVar,
    rhs: &GrumpkinPointWithIdentityVar,
    output: &GrumpkinPointWithIdentityVar,
) -> Result<(), CryptoR1csError> {
    lhs.assert_valid(builder);
    rhs.assert_valid(builder);
    output.assert_valid(builder);

    let one = AssignedScalar::constant(Fr::one());
    let zero = AssignedScalar::constant(Fr::zero());
    let not_lhs_identity = one.sub(builder, &lhs.is_identity);
    let not_rhs_identity = one.sub(builder, &rhs.is_identity);
    let rhs_only_gate = not_lhs_identity.mul(builder, &rhs.is_identity);
    let both_nonidentity = not_lhs_identity.mul(builder, &not_rhs_identity);

    gated_assert_equal(builder, &lhs.is_identity, &output.x, &rhs.x);
    gated_assert_equal(builder, &lhs.is_identity, &output.y, &rhs.y);
    gated_assert_equal(
        builder,
        &lhs.is_identity,
        &output.is_identity,
        &rhs.is_identity,
    );
    gated_assert_equal(builder, &rhs_only_gate, &output.x, &lhs.x);
    gated_assert_equal(builder, &rhs_only_gate, &output.y, &lhs.y);
    gated_assert_zero(builder, &rhs_only_gate, &output.is_identity);

    let dx = rhs.x.sub(builder, &lhs.x);
    let dx_zero = zero_check(builder, &dx);
    let dy = rhs.y.sub(builder, &lhs.y);
    let dy_zero = zero_check(builder, &dy);
    let y_sum = rhs.y.add(builder, &lhs.y);
    let y_sum_zero = zero_check(builder, &y_sum);

    let not_dx_zero = one.sub(builder, &dx_zero.is_zero);
    let ordinary_gate = both_nonidentity.mul(builder, &not_dx_zero);
    let same_x_gate = both_nonidentity.mul(builder, &dx_zero.is_zero);
    let same_point_gate = same_x_gate.mul(builder, &dy_zero.is_zero);
    let not_dy_zero = one.sub(builder, &dy_zero.is_zero);
    let inverse_candidate_gate = same_x_gate.mul(builder, &not_dy_zero);
    let inverse_gate = inverse_candidate_gate.mul(builder, &y_sum_zero.is_zero);
    let not_y_sum_zero = one.sub(builder, &y_sum_zero.is_zero);
    let invalid_same_x_gate = inverse_candidate_gate.mul(builder, &not_y_sum_zero);
    invalid_same_x_gate.assert_equal(builder, &zero);

    let two_y = lhs.y.scale_by_constant(builder, Fr::from_u64(2));
    let two_y_zero = zero_check(builder, &two_y);
    let not_two_y_zero = one.sub(builder, &two_y_zero.is_zero);
    let double_gate = same_point_gate.mul(builder, &not_two_y_zero);
    let double_to_identity_gate = same_point_gate.mul(builder, &two_y_zero.is_zero);

    gated_assert_zero(builder, &ordinary_gate, &output.is_identity);
    let ordinary_slope = dy.mul(builder, &dx_zero.inverse);
    let ordinary_slope_squared = ordinary_slope.mul(builder, &ordinary_slope);
    let expected_ordinary_slope_squared = output.x.add(builder, &lhs.x).add(builder, &rhs.x);
    let ordinary_x_residual = ordinary_slope_squared.sub(builder, &expected_ordinary_slope_squared);
    gated_assert_zero(builder, &ordinary_gate, &ordinary_x_residual);
    let ordinary_x_delta = lhs.x.sub(builder, &output.x);
    let ordinary_y_delta = output.y.add(builder, &lhs.y);
    let expected_ordinary_y_delta = ordinary_slope.mul(builder, &ordinary_x_delta);
    let ordinary_y_residual = ordinary_y_delta.sub(builder, &expected_ordinary_y_delta);
    gated_assert_zero(builder, &ordinary_gate, &ordinary_y_residual);

    gated_assert_zero(builder, &double_gate, &output.is_identity);
    let x_squared = lhs.x.mul(builder, &lhs.x);
    let three_x_squared = x_squared.scale_by_constant(builder, Fr::from_u64(3));
    let double_slope = three_x_squared.mul(builder, &two_y_zero.inverse);
    let double_slope_squared = double_slope.mul(builder, &double_slope);
    let two_x = lhs.x.scale_by_constant(builder, Fr::from_u64(2));
    let expected_double_slope_squared = output.x.add(builder, &two_x);
    let double_x_residual = double_slope_squared.sub(builder, &expected_double_slope_squared);
    gated_assert_zero(builder, &double_gate, &double_x_residual);
    let double_x_delta = lhs.x.sub(builder, &output.x);
    let double_y_delta = output.y.add(builder, &lhs.y);
    let expected_double_y_delta = double_slope.mul(builder, &double_x_delta);
    let double_y_residual = double_y_delta.sub(builder, &expected_double_y_delta);
    gated_assert_zero(builder, &double_gate, &double_y_residual);

    assert_output_identity(builder, &inverse_gate, output);
    assert_output_identity(builder, &double_to_identity_gate, output);

    Ok(())
}

pub fn fixed_base_grumpkin_scalar_mul(
    builder: &mut R1csBuilder<Fr>,
    base: &GrumpkinPoint,
    scalar: &FqVar,
) -> Result<GrumpkinPointWithIdentityVar, CryptoR1csError> {
    let mut accumulator = GrumpkinPointWithIdentityVar::identity();
    let mut base_power = *base;
    let identity = GrumpkinPointWithIdentityVar::identity();

    for bit in scalar.bits_le(builder) {
        let selected = GrumpkinPointWithIdentityVar::select(
            builder,
            &bit,
            &GrumpkinPointWithIdentityVar::constant(&base_power),
            &identity,
        );
        accumulator = complete_grumpkin_add(builder, &accumulator, &selected)?;
        base_power = base_power.double();
    }

    Ok(accumulator)
}

/// Constrains a fixed-base MSM over Grumpkin.
///
pub fn fixed_base_grumpkin_msm(
    builder: &mut R1csBuilder<Fr>,
    bases: &[GrumpkinPoint],
    scalars: &[FqVar],
) -> Result<GrumpkinPointWithIdentityVar, CryptoR1csError> {
    if bases.len() != scalars.len() {
        return Err(CryptoR1csError::FixedBaseMsmLengthMismatch {
            bases: bases.len(),
            scalars: scalars.len(),
        });
    }

    let mut accumulator = GrumpkinPointWithIdentityVar::identity();
    for (base, scalar) in bases.iter().zip(scalars) {
        let term = fixed_base_grumpkin_scalar_mul(builder, base, scalar)?;
        accumulator = complete_grumpkin_add(builder, &accumulator, &term)?;
    }

    Ok(accumulator)
}

pub fn variable_base_grumpkin_scalar_mul(
    builder: &mut R1csBuilder<Fr>,
    base: &GrumpkinPointWithIdentityVar,
    scalar: &FqVar,
) -> Result<GrumpkinPointWithIdentityVar, CryptoR1csError> {
    base.assert_valid(builder);

    let mut accumulator = GrumpkinPointWithIdentityVar::identity();
    let mut base_power = base.clone();
    let identity = GrumpkinPointWithIdentityVar::identity();

    for bit in scalar.bits_le(builder) {
        let selected = GrumpkinPointWithIdentityVar::select(builder, &bit, &base_power, &identity);
        accumulator = complete_grumpkin_add(builder, &accumulator, &selected)?;
        base_power = complete_grumpkin_add(builder, &base_power, &base_power)?;
    }

    Ok(accumulator)
}

pub fn linear_combine_grumpkin_commitments(
    builder: &mut R1csBuilder<Fr>,
    commitments: &[GrumpkinPointWithIdentityVar],
    coefficients: &[FqVar],
) -> Result<GrumpkinPointWithIdentityVar, CryptoR1csError> {
    if commitments.len() != coefficients.len() {
        return Err(CryptoR1csError::FixedBaseMsmLengthMismatch {
            bases: commitments.len(),
            scalars: coefficients.len(),
        });
    }

    let mut accumulator = GrumpkinPointWithIdentityVar::identity();
    for (commitment, coefficient) in commitments.iter().zip(coefficients) {
        let term = variable_base_grumpkin_scalar_mul(builder, commitment, coefficient)?;
        accumulator = complete_grumpkin_add(builder, &accumulator, &term)?;
    }

    Ok(accumulator)
}

pub fn grumpkin_pedersen_opening_commitment(
    builder: &mut R1csBuilder<Fr>,
    setup: &PedersenSetup<GrumpkinPoint>,
    opening: &VectorCommitmentOpeningVar<FqVar>,
) -> Result<GrumpkinPointWithIdentityVar, CryptoR1csError> {
    let capacity = setup.message_generators.len();
    if opening.values.len() > capacity {
        return Err(CryptoR1csError::VectorCommitmentCapacityExceeded {
            capacity,
            values: opening.values.len(),
        });
    }

    let mut bases = setup.message_generators[..opening.values.len()].to_vec();
    bases.push(setup.blinding_generator);
    let mut scalars = opening.values.clone();
    scalars.push(opening.blinding.clone());

    fixed_base_grumpkin_msm(builder, &bases, &scalars)
}

pub fn verify_grumpkin_pedersen_opening(
    builder: &mut R1csBuilder<Fr>,
    setup: &PedersenSetup<GrumpkinPoint>,
    commitment: &GrumpkinPointWithIdentityVar,
    opening: &VectorCommitmentOpeningVar<FqVar>,
) -> Result<(), CryptoR1csError> {
    commitment.assert_valid(builder);
    let computed = grumpkin_pedersen_opening_commitment(builder, setup, opening)?;
    computed.assert_equal(builder, commitment);
    Ok(())
}

/// Constrains `output = lhs + rhs` for ordinary affine additions.
///
/// This helper intentionally rejects exceptional cases (`lhs.x == rhs.x`).
/// Doubling and adding inverse points need separate formulas so callers cannot
/// accidentally use an incomplete relation for those cases.
pub fn assert_nonexceptional_grumpkin_add(
    builder: &mut R1csBuilder<Fr>,
    lhs: &GrumpkinPointVar,
    rhs: &GrumpkinPointVar,
    output: &GrumpkinPointVar,
) -> Result<(), CryptoR1csError> {
    lhs.assert_on_curve(builder);
    rhs.assert_on_curve(builder);
    output.assert_on_curve(builder);

    let dx = rhs.x.sub(builder, &lhs.x);
    if dx.value.is_zero() {
        return Err(CryptoR1csError::ExceptionalAffineAddition);
    }
    let dy = rhs.y.sub(builder, &lhs.y);
    let dx_inverse = AssignedScalar::alloc(
        builder,
        dx.value
            .inverse()
            .ok_or(CryptoR1csError::ExceptionalAffineAddition)?,
    );
    builder.assert_product(
        dx.lc.clone(),
        dx_inverse.lc.clone(),
        LinearCombination::one(),
    );

    let slope = dy.mul(builder, &dx_inverse);
    let slope_squared = slope.mul(builder, &slope);
    let expected_slope_squared = output.x.add(builder, &lhs.x).add(builder, &rhs.x);
    slope_squared.assert_equal(builder, &expected_slope_squared);

    let x_delta = lhs.x.sub(builder, &output.x);
    let y_delta = output.y.add(builder, &lhs.y);
    let expected_y_delta = slope.mul(builder, &x_delta);
    y_delta.assert_equal(builder, &expected_y_delta);

    Ok(())
}

fn grumpkin_coordinates(point: &GrumpkinPoint) -> Result<(Fr, Fr), CryptoR1csError> {
    let affine = point.0.into_affine();
    if affine.infinity {
        return Err(CryptoR1csError::IdentityPoint);
    }
    Ok((Fr::from(affine.x), Fr::from(affine.y)))
}

fn grumpkin_point_from_coordinates(x: Fr, y: Fr) -> GrumpkinPoint {
    GrumpkinPoint(ark_grumpkin::Affine::new_unchecked(x.into(), y.into()).into())
}

#[derive(Clone, Debug)]
struct ZeroCheck {
    is_zero: AssignedScalar<Fr>,
    inverse: AssignedScalar<Fr>,
}

fn zero_check(builder: &mut R1csBuilder<Fr>, value: &AssignedScalar<Fr>) -> ZeroCheck {
    let is_zero = AssignedScalar::alloc(builder, Fr::from_bool(value.value.is_zero()));
    assert_boolean(builder, &is_zero);
    let inverse = AssignedScalar::alloc(builder, value.value.inverse().unwrap_or_else(Fr::zero));
    builder.assert_product(
        value.lc.clone(),
        is_zero.lc.clone(),
        LinearCombination::zero(),
    );
    let product = value.mul(builder, &inverse);
    let one_minus_is_zero = AssignedScalar::constant(Fr::one()).sub(builder, &is_zero);
    product.assert_equal(builder, &one_minus_is_zero);

    ZeroCheck { is_zero, inverse }
}

fn assert_boolean(builder: &mut R1csBuilder<Fr>, value: &AssignedScalar<Fr>) {
    builder.assert_product(
        value.lc.clone(),
        value.lc.clone() - LinearCombination::one(),
        LinearCombination::zero(),
    );
}

fn gated_assert_equal(
    builder: &mut R1csBuilder<Fr>,
    gate: &AssignedScalar<Fr>,
    lhs: &AssignedScalar<Fr>,
    rhs: &AssignedScalar<Fr>,
) {
    let difference = lhs.sub(builder, rhs);
    gated_assert_zero(builder, gate, &difference);
}

fn gated_assert_zero(
    builder: &mut R1csBuilder<Fr>,
    gate: &AssignedScalar<Fr>,
    value: &AssignedScalar<Fr>,
) {
    builder.assert_product(gate.lc.clone(), value.lc.clone(), LinearCombination::zero());
}

fn assert_output_identity(
    builder: &mut R1csBuilder<Fr>,
    gate: &AssignedScalar<Fr>,
    output: &GrumpkinPointWithIdentityVar,
) {
    gated_assert_zero(builder, gate, &output.x);
    gated_assert_zero(builder, gate, &output.y);
    gated_assert_equal(
        builder,
        gate,
        &output.is_identity,
        &AssignedScalar::constant(Fr::one()),
    );
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use jolt_field::Fq;
    use jolt_r1cs::Variable;

    use super::*;
    use crate::{Grumpkin, JoltGroup, Pedersen, VectorCommitment};

    #[test]
    fn grumpkin_on_curve_accepts_valid_point() {
        let mut builder = R1csBuilder::<Fr>::new();
        let point = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let point = GrumpkinPointVar::alloc(&mut builder, &point).expect("non-identity point");

        point.assert_on_curve(&mut builder);

        assert!(builder_accepts(builder));
    }

    #[test]
    fn grumpkin_on_curve_rejects_tampered_coordinate() {
        let mut builder = R1csBuilder::<Fr>::new();
        let point = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let point = GrumpkinPointVar::alloc(&mut builder, &point).expect("non-identity point");
        let targets = [
            ("x-coordinate", variable(&point.x)),
            ("y-coordinate", variable(&point.y)),
        ];

        point.assert_on_curve(&mut builder);

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn grumpkin_on_curve_rejects_invalid_point() {
        let mut builder = R1csBuilder::<Fr>::new();
        let point = GrumpkinPointVar::new(
            AssignedScalar::alloc(&mut builder, Fr::from_u64(1)),
            AssignedScalar::alloc(&mut builder, Fr::from_u64(1)),
        );

        point.assert_on_curve(&mut builder);

        assert!(builder_rejects(builder));
    }

    #[test]
    fn group_element_trait_accepts_grumpkin_point_var() {
        let mut builder = R1csBuilder::<Fr>::new();
        let point = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let point = GrumpkinPointVar::alloc(&mut builder, &point).expect("non-identity point");

        assert_group_var_valid(&mut builder, &point).expect("valid group variable");
        GroupElementVar::assert_equal(&point, &mut builder, &point);

        assert!(builder_accepts(builder));
    }

    #[test]
    fn grumpkin_nonexceptional_add_accepts_valid_sum() {
        let mut builder = R1csBuilder::<Fr>::new();
        let p = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let q = Grumpkin::generator().scalar_mul(&Fq::from_u64(7));
        let sum = p + q;
        let p = GrumpkinPointVar::alloc(&mut builder, &p).expect("non-identity point");
        let q = GrumpkinPointVar::alloc(&mut builder, &q).expect("non-identity point");
        let sum = GrumpkinPointVar::alloc(&mut builder, &sum).expect("non-identity point");

        assert_nonexceptional_grumpkin_add(&mut builder, &p, &q, &sum).expect("ordinary addition");

        assert!(builder_accepts(builder));
    }

    #[test]
    fn nonexceptional_add_trait_accepts_valid_grumpkin_sum() {
        let mut builder = R1csBuilder::<Fr>::new();
        let p = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let q = Grumpkin::generator().scalar_mul(&Fq::from_u64(7));
        let sum = p + q;
        let p = GrumpkinPointVar::alloc(&mut builder, &p).expect("non-identity point");
        let q = GrumpkinPointVar::alloc(&mut builder, &q).expect("non-identity point");
        let sum = GrumpkinPointVar::alloc(&mut builder, &sum).expect("non-identity point");

        assert_generic_nonexceptional_add(&mut builder, &p, &q, &sum).expect("ordinary addition");

        assert!(builder_accepts(builder));
    }

    #[test]
    fn grumpkin_double_accepts_valid_double() {
        let mut builder = R1csBuilder::<Fr>::new();
        let point = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let doubled = point.double();
        let point = GrumpkinPointVar::alloc(&mut builder, &point).expect("non-identity point");
        let doubled = GrumpkinPointVar::alloc(&mut builder, &doubled).expect("non-identity point");

        assert_grumpkin_double(&mut builder, &point, &doubled).expect("ordinary doubling");

        assert!(builder_accepts(builder));
    }

    #[test]
    fn grumpkin_double_trait_accepts_valid_double() {
        let mut builder = R1csBuilder::<Fr>::new();
        let point = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let doubled = point.double();
        let point = GrumpkinPointVar::alloc(&mut builder, &point).expect("non-identity point");
        let doubled = GrumpkinPointVar::alloc(&mut builder, &doubled).expect("non-identity point");

        assert_generic_double(&mut builder, &point, &doubled).expect("ordinary doubling");

        assert!(builder_accepts(builder));
    }

    #[test]
    fn grumpkin_double_rejects_tampered_output() {
        let mut builder = R1csBuilder::<Fr>::new();
        let point = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let doubled = point.double();
        let point = GrumpkinPointVar::alloc(&mut builder, &point).expect("non-identity point");
        let doubled = GrumpkinPointVar::alloc(&mut builder, &doubled).expect("non-identity point");
        let targets = [
            ("doubled x-coordinate", variable(&doubled.x)),
            ("doubled y-coordinate", variable(&doubled.y)),
        ];

        assert_grumpkin_double(&mut builder, &point, &doubled).expect("ordinary doubling");

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn grumpkin_double_rejects_zero_y_input() {
        let mut builder = R1csBuilder::<Fr>::new();
        let input = GrumpkinPointVar::new(
            AssignedScalar::alloc(&mut builder, Fr::from_u64(1)),
            AssignedScalar::alloc(&mut builder, Fr::zero()),
        );
        let output = GrumpkinPointVar::new(
            AssignedScalar::alloc(&mut builder, Fr::from_u64(1)),
            AssignedScalar::alloc(&mut builder, Fr::from_u64(1)),
        );

        assert_eq!(
            assert_grumpkin_double(&mut builder, &input, &output),
            Err(CryptoR1csError::ExceptionalAffineDoubling)
        );
    }

    #[test]
    fn grumpkin_nonexceptional_add_rejects_tampered_output() {
        let mut builder = R1csBuilder::<Fr>::new();
        let p = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let q = Grumpkin::generator().scalar_mul(&Fq::from_u64(7));
        let sum = p + q;
        let p = GrumpkinPointVar::alloc(&mut builder, &p).expect("non-identity point");
        let q = GrumpkinPointVar::alloc(&mut builder, &q).expect("non-identity point");
        let sum = GrumpkinPointVar::alloc(&mut builder, &sum).expect("non-identity point");
        let targets = [
            ("sum x-coordinate", variable(&sum.x)),
            ("sum y-coordinate", variable(&sum.y)),
        ];

        assert_nonexceptional_grumpkin_add(&mut builder, &p, &q, &sum).expect("ordinary addition");

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn grumpkin_nonexceptional_add_rejects_equal_x_inputs() {
        let mut builder = R1csBuilder::<Fr>::new();
        let p = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let doubled = p + p;
        let p = GrumpkinPointVar::alloc(&mut builder, &p).expect("non-identity point");
        let doubled = GrumpkinPointVar::alloc(&mut builder, &doubled).expect("non-identity point");

        assert_eq!(
            assert_nonexceptional_grumpkin_add(&mut builder, &p, &p, &doubled),
            Err(CryptoR1csError::ExceptionalAffineAddition)
        );
    }

    #[test]
    fn grumpkin_affine_gadget_rejects_identity() {
        let mut builder = R1csBuilder::<Fr>::new();

        assert_eq!(
            GrumpkinPointVar::alloc(&mut builder, &GrumpkinPoint::identity()),
            Err(CryptoR1csError::IdentityPoint)
        );
    }

    #[test]
    fn grumpkin_identity_aware_point_accepts_identity() {
        let mut builder = R1csBuilder::<Fr>::new();
        let identity =
            GrumpkinPointWithIdentityVar::alloc(&mut builder, &GrumpkinPoint::identity());

        identity.assert_valid(&mut builder);

        assert!(builder_accepts(builder));
    }

    #[test]
    fn grumpkin_identity_aware_point_rejects_malformed_identity() {
        let mut builder = R1csBuilder::<Fr>::new();
        let point = GrumpkinPointWithIdentityVar::new(
            AssignedScalar::alloc(&mut builder, Fr::from_u64(1)),
            AssignedScalar::alloc(&mut builder, Fr::zero()),
            AssignedScalar::alloc(&mut builder, Fr::one()),
        );

        point.assert_valid(&mut builder);

        assert!(builder_rejects(builder));
    }

    #[test]
    fn grumpkin_add_with_identity_accepts_left_identity() {
        let mut builder = R1csBuilder::<Fr>::new();
        let identity =
            GrumpkinPointWithIdentityVar::alloc(&mut builder, &GrumpkinPoint::identity());
        let point = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let point = GrumpkinPointWithIdentityVar::alloc(&mut builder, &point);
        let output = point.clone();

        assert_grumpkin_add_with_identity(&mut builder, &identity, &point, &output)
            .expect("identity plus point");

        assert!(builder_accepts(builder));
    }

    #[test]
    fn grumpkin_add_with_identity_accepts_right_identity() {
        let mut builder = R1csBuilder::<Fr>::new();
        let point = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let point = GrumpkinPointWithIdentityVar::alloc(&mut builder, &point);
        let identity =
            GrumpkinPointWithIdentityVar::alloc(&mut builder, &GrumpkinPoint::identity());
        let output = point.clone();

        assert_grumpkin_add_with_identity(&mut builder, &point, &identity, &output)
            .expect("point plus identity");

        assert!(builder_accepts(builder));
    }

    #[test]
    fn grumpkin_add_with_identity_accepts_both_identities() {
        let mut builder = R1csBuilder::<Fr>::new();
        let lhs = GrumpkinPointWithIdentityVar::alloc(&mut builder, &GrumpkinPoint::identity());
        let rhs = GrumpkinPointWithIdentityVar::alloc(&mut builder, &GrumpkinPoint::identity());
        let output = GrumpkinPointWithIdentityVar::alloc(&mut builder, &GrumpkinPoint::identity());

        assert_grumpkin_add_with_identity(&mut builder, &lhs, &rhs, &output)
            .expect("identity plus identity");

        assert!(builder_accepts(builder));
    }

    #[test]
    fn grumpkin_add_with_identity_accepts_ordinary_sum() {
        let mut builder = R1csBuilder::<Fr>::new();
        let p = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let q = Grumpkin::generator().scalar_mul(&Fq::from_u64(7));
        let sum = p + q;
        let p = GrumpkinPointWithIdentityVar::alloc(&mut builder, &p);
        let q = GrumpkinPointWithIdentityVar::alloc(&mut builder, &q);
        let sum = GrumpkinPointWithIdentityVar::alloc(&mut builder, &sum);

        assert_grumpkin_add_with_identity(&mut builder, &p, &q, &sum).expect("ordinary addition");

        assert!(builder_accepts(builder));
    }

    #[test]
    fn grumpkin_add_with_identity_rejects_tampered_identity_case_output() {
        let mut builder = R1csBuilder::<Fr>::new();
        let identity =
            GrumpkinPointWithIdentityVar::alloc(&mut builder, &GrumpkinPoint::identity());
        let point = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let point = GrumpkinPointWithIdentityVar::alloc(&mut builder, &point);
        let output = point.clone();
        let targets = [("identity-case output x", variable(&output.x))];

        assert_grumpkin_add_with_identity(&mut builder, &identity, &point, &output)
            .expect("identity plus point");

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn grumpkin_add_with_identity_rejects_tampered_ordinary_output() {
        let mut builder = R1csBuilder::<Fr>::new();
        let p = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let q = Grumpkin::generator().scalar_mul(&Fq::from_u64(7));
        let sum = p + q;
        let p = GrumpkinPointWithIdentityVar::alloc(&mut builder, &p);
        let q = GrumpkinPointWithIdentityVar::alloc(&mut builder, &q);
        let sum = GrumpkinPointWithIdentityVar::alloc(&mut builder, &sum);
        let targets = [
            ("ordinary output x", variable(&sum.x)),
            ("ordinary output y", variable(&sum.y)),
        ];

        assert_grumpkin_add_with_identity(&mut builder, &p, &q, &sum).expect("ordinary addition");

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn grumpkin_add_with_identity_rejects_nonidentity_exceptional_addition() {
        let mut builder = R1csBuilder::<Fr>::new();
        let point = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let doubled = point.double();
        let point = GrumpkinPointWithIdentityVar::alloc(&mut builder, &point);
        let doubled = GrumpkinPointWithIdentityVar::alloc(&mut builder, &doubled);

        assert_eq!(
            assert_grumpkin_add_with_identity(&mut builder, &point, &point, &doubled),
            Err(CryptoR1csError::ExceptionalAffineAddition)
        );
    }

    #[test]
    fn complete_grumpkin_add_accepts_identity_cases() {
        let mut builder = R1csBuilder::<Fr>::new();
        let identity =
            GrumpkinPointWithIdentityVar::alloc(&mut builder, &GrumpkinPoint::identity());
        let point = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let point = GrumpkinPointWithIdentityVar::alloc(&mut builder, &point);

        let left = complete_grumpkin_add(&mut builder, &identity, &point).expect("O + P");
        let right = complete_grumpkin_add(&mut builder, &point, &identity).expect("P + O");
        let both = complete_grumpkin_add(&mut builder, &identity, &identity).expect("O + O");

        left.assert_equal(&mut builder, &point);
        right.assert_equal(&mut builder, &point);
        both.assert_equal(&mut builder, &identity);
        assert!(builder_accepts(builder));
    }

    #[test]
    fn complete_grumpkin_add_accepts_ordinary_sum() {
        let mut builder = R1csBuilder::<Fr>::new();
        let p = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let q = Grumpkin::generator().scalar_mul(&Fq::from_u64(7));
        let expected = GrumpkinPointWithIdentityVar::constant(&(p + q));
        let p = GrumpkinPointWithIdentityVar::alloc(&mut builder, &p);
        let q = GrumpkinPointWithIdentityVar::alloc(&mut builder, &q);

        let sum = complete_grumpkin_add(&mut builder, &p, &q).expect("ordinary sum");

        sum.assert_equal(&mut builder, &expected);
        assert!(builder_accepts(builder));
    }

    #[test]
    fn complete_grumpkin_add_accepts_doubling() {
        let mut builder = R1csBuilder::<Fr>::new();
        let p = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let expected = GrumpkinPointWithIdentityVar::constant(&p.double());
        let p = GrumpkinPointWithIdentityVar::alloc(&mut builder, &p);

        let doubled = complete_grumpkin_add(&mut builder, &p, &p).expect("doubling");

        doubled.assert_equal(&mut builder, &expected);
        assert!(builder_accepts(builder));
    }

    #[test]
    fn complete_grumpkin_add_accepts_inverse_to_identity() {
        let mut builder = R1csBuilder::<Fr>::new();
        let p = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let neg_p = -p;
        let p = GrumpkinPointWithIdentityVar::alloc(&mut builder, &p);
        let neg_p = GrumpkinPointWithIdentityVar::alloc(&mut builder, &neg_p);

        let sum = complete_grumpkin_add(&mut builder, &p, &neg_p).expect("inverse sum");

        sum.assert_equal(&mut builder, &GrumpkinPointWithIdentityVar::identity());
        assert!(builder_accepts(builder));
    }

    #[test]
    fn complete_grumpkin_add_trait_accepts_valid_sum() {
        let mut builder = R1csBuilder::<Fr>::new();
        let p = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let q = Grumpkin::generator().scalar_mul(&Fq::from_u64(7));
        let expected = GrumpkinPointWithIdentityVar::constant(&(p + q));
        let p = GrumpkinPointWithIdentityVar::alloc(&mut builder, &p);
        let q = GrumpkinPointWithIdentityVar::alloc(&mut builder, &q);

        let sum = assert_generic_complete_add::<GrumpkinPointWithIdentityVar>(&mut builder, &p, &q)
            .expect("complete add");

        sum.assert_equal(&mut builder, &expected);
        assert!(builder_accepts(builder));
    }

    #[test]
    fn complete_grumpkin_add_rejects_tampered_ordinary_output() {
        let mut builder = R1csBuilder::<Fr>::new();
        let p = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let q = Grumpkin::generator().scalar_mul(&Fq::from_u64(7));
        let expected = p + q;
        let p = GrumpkinPointWithIdentityVar::alloc(&mut builder, &p);
        let q = GrumpkinPointWithIdentityVar::alloc(&mut builder, &q);
        let output = GrumpkinPointWithIdentityVar::alloc(&mut builder, &expected);
        let targets = [
            ("complete ordinary output x", variable(&output.x)),
            ("complete ordinary output y", variable(&output.y)),
        ];

        assert_complete_grumpkin_add(&mut builder, &p, &q, &output).expect("complete add");

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn complete_grumpkin_add_rejects_tampered_inverse_output() {
        let mut builder = R1csBuilder::<Fr>::new();
        let p = Grumpkin::generator().scalar_mul(&Fq::from_u64(5));
        let neg_p = -p;
        let p = GrumpkinPointWithIdentityVar::alloc(&mut builder, &p);
        let neg_p = GrumpkinPointWithIdentityVar::alloc(&mut builder, &neg_p);
        let output = GrumpkinPointWithIdentityVar::alloc(&mut builder, &GrumpkinPoint::identity());
        let targets = [(
            "complete inverse output identity flag",
            variable(&output.is_identity),
        )];

        assert_complete_grumpkin_add(&mut builder, &p, &neg_p, &output).expect("complete add");

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn fixed_base_grumpkin_scalar_mul_accepts_zero_scalar() {
        let mut builder = R1csBuilder::<Fr>::new();
        let base = Grumpkin::generator();
        let scalar = FqVar::alloc(&mut builder, Fq::zero());

        let result =
            fixed_base_grumpkin_scalar_mul(&mut builder, &base, &scalar).expect("scalar mul");

        result.assert_equal(&mut builder, &GrumpkinPointWithIdentityVar::identity());
        assert!(builder_accepts(builder));
    }

    #[test]
    fn fixed_base_grumpkin_scalar_mul_accepts_nonzero_scalar() {
        let mut builder = R1csBuilder::<Fr>::new();
        let base = Grumpkin::generator();
        let scalar_value = Fq::from_u64(13);
        let scalar = FqVar::alloc(&mut builder, scalar_value);

        let result =
            fixed_base_grumpkin_scalar_mul(&mut builder, &base, &scalar).expect("scalar mul");
        let expected = GrumpkinPointWithIdentityVar::constant(&base.scalar_mul(&scalar_value));

        result.assert_equal(&mut builder, &expected);
        assert!(builder_accepts(builder));
    }

    #[test]
    fn fixed_base_scalar_mul_trait_accepts_nonzero_scalar() {
        let mut builder = R1csBuilder::<Fr>::new();
        let base = Grumpkin::generator();
        let scalar_value = Fq::from_u64(19);
        let scalar = FqVar::alloc(&mut builder, scalar_value);

        let result = assert_generic_fixed_base_scalar_mul::<GrumpkinPointWithIdentityVar>(
            &mut builder,
            &base,
            &scalar,
        )
        .expect("scalar mul");
        let expected = GrumpkinPointWithIdentityVar::constant(&base.scalar_mul(&scalar_value));

        result.assert_equal(&mut builder, &expected);
        assert!(builder_accepts(builder));
    }

    #[test]
    fn fixed_base_grumpkin_scalar_mul_rejects_tampered_output() {
        let mut builder = R1csBuilder::<Fr>::new();
        let base = Grumpkin::generator();
        let scalar = FqVar::alloc(&mut builder, Fq::from_u64(13));

        let result =
            fixed_base_grumpkin_scalar_mul(&mut builder, &base, &scalar).expect("scalar mul");
        let targets = [("scalar mul output x", variable(&result.x))];

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn variable_base_grumpkin_scalar_mul_accepts_zero_scalar() {
        let mut builder = R1csBuilder::<Fr>::new();
        let base = Grumpkin::generator().scalar_mul(&Fq::from_u64(11));
        let base = GrumpkinPointWithIdentityVar::alloc(&mut builder, &base);
        let scalar = FqVar::alloc(&mut builder, Fq::zero());

        let result =
            variable_base_grumpkin_scalar_mul(&mut builder, &base, &scalar).expect("scalar mul");

        result.assert_equal(&mut builder, &GrumpkinPointWithIdentityVar::identity());
        assert!(builder_accepts(builder));
    }

    #[test]
    fn variable_base_grumpkin_scalar_mul_accepts_nonzero_scalar() {
        let mut builder = R1csBuilder::<Fr>::new();
        let base_value = Grumpkin::generator().scalar_mul(&Fq::from_u64(11));
        let scalar_value = Fq::from_u64(13);
        let base = GrumpkinPointWithIdentityVar::alloc(&mut builder, &base_value);
        let scalar = FqVar::alloc(&mut builder, scalar_value);

        let result =
            variable_base_grumpkin_scalar_mul(&mut builder, &base, &scalar).expect("scalar mul");
        let expected =
            GrumpkinPointWithIdentityVar::constant(&base_value.scalar_mul(&scalar_value));

        result.assert_equal(&mut builder, &expected);
        assert!(builder_accepts(builder));
    }

    #[test]
    fn variable_base_scalar_mul_trait_accepts_nonzero_scalar() {
        let mut builder = R1csBuilder::<Fr>::new();
        let base_value = Grumpkin::generator().scalar_mul(&Fq::from_u64(11));
        let scalar_value = Fq::from_u64(19);
        let base = GrumpkinPointWithIdentityVar::alloc(&mut builder, &base_value);
        let scalar = FqVar::alloc(&mut builder, scalar_value);

        let result = assert_generic_variable_base_scalar_mul::<GrumpkinPointWithIdentityVar>(
            &mut builder,
            &base,
            &scalar,
        )
        .expect("scalar mul");
        let expected =
            GrumpkinPointWithIdentityVar::constant(&base_value.scalar_mul(&scalar_value));

        result.assert_equal(&mut builder, &expected);
        assert!(builder_accepts(builder));
    }

    #[test]
    fn variable_base_grumpkin_scalar_mul_rejects_tampered_base() {
        let mut builder = R1csBuilder::<Fr>::new();
        let base = Grumpkin::generator().scalar_mul(&Fq::from_u64(11));
        let base = GrumpkinPointWithIdentityVar::alloc(&mut builder, &base);
        let scalar = FqVar::alloc(&mut builder, Fq::from_u64(13));
        let result =
            variable_base_grumpkin_scalar_mul(&mut builder, &base, &scalar).expect("scalar mul");
        result.assert_valid(&mut builder);
        let targets = [("variable-base scalar mul base x", variable(&base.x))];

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn variable_base_grumpkin_scalar_mul_rejects_tampered_output() {
        let mut builder = R1csBuilder::<Fr>::new();
        let base = Grumpkin::generator().scalar_mul(&Fq::from_u64(11));
        let base = GrumpkinPointWithIdentityVar::alloc(&mut builder, &base);
        let scalar = FqVar::alloc(&mut builder, Fq::from_u64(13));

        let result =
            variable_base_grumpkin_scalar_mul(&mut builder, &base, &scalar).expect("scalar mul");
        let targets = [("variable-base scalar mul output x", variable(&result.x))];

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn fixed_base_grumpkin_msm_accepts_nonzero_scalars() {
        let mut builder = R1csBuilder::<Fr>::new();
        let bases = vec![
            Grumpkin::generator().scalar_mul(&Fq::from_u64(11)),
            Grumpkin::generator().scalar_mul(&Fq::from_u64(17)),
            Grumpkin::generator().scalar_mul(&Fq::from_u64(23)),
        ];
        let scalar_values = vec![Fq::from_u64(3), Fq::from_u64(5), Fq::from_u64(7)];
        let scalars = scalar_values
            .iter()
            .copied()
            .map(|scalar| FqVar::alloc(&mut builder, scalar))
            .collect::<Vec<_>>();

        let result =
            fixed_base_grumpkin_msm(&mut builder, &bases, &scalars).expect("fixed-base MSM");
        let expected =
            GrumpkinPointWithIdentityVar::constant(&GrumpkinPoint::msm(&bases, &scalar_values));

        result.assert_equal(&mut builder, &expected);
        assert!(builder_accepts(builder));
    }

    #[test]
    fn fixed_base_grumpkin_msm_rejects_length_mismatch() {
        let mut builder = R1csBuilder::<Fr>::new();
        let bases = vec![Grumpkin::generator()];
        let scalars = vec![
            FqVar::alloc(&mut builder, Fq::from_u64(3)),
            FqVar::alloc(&mut builder, Fq::from_u64(5)),
        ];

        assert_eq!(
            fixed_base_grumpkin_msm(&mut builder, &bases, &scalars),
            Err(CryptoR1csError::FixedBaseMsmLengthMismatch {
                bases: 1,
                scalars: 2,
            })
        );
    }

    #[test]
    fn linear_combine_grumpkin_commitments_accepts_valid_combination() {
        let mut builder = R1csBuilder::<Fr>::new();
        let commitment_values = [
            Grumpkin::generator().scalar_mul(&Fq::from_u64(11)),
            Grumpkin::generator().scalar_mul(&Fq::from_u64(17)),
        ];
        let coefficient_values = [Fq::from_u64(3), Fq::from_u64(5)];
        let commitments = commitment_values
            .iter()
            .map(|commitment| GrumpkinPointWithIdentityVar::alloc(&mut builder, commitment))
            .collect::<Vec<_>>();
        let coefficients = coefficient_values
            .iter()
            .copied()
            .map(|coefficient| FqVar::alloc(&mut builder, coefficient))
            .collect::<Vec<_>>();

        let combined =
            linear_combine_grumpkin_commitments(&mut builder, &commitments, &coefficients)
                .expect("linear combination");
        let expected = GrumpkinPointWithIdentityVar::constant(&GrumpkinPoint::msm(
            &commitment_values,
            &coefficient_values,
        ));

        combined.assert_equal(&mut builder, &expected);
        assert!(builder_accepts(builder));
    }

    #[test]
    fn vector_commitment_r1cs_trait_combines_grumpkin_commitments() {
        let mut builder = R1csBuilder::<Fr>::new();
        let commitment_values = [
            Grumpkin::generator().scalar_mul(&Fq::from_u64(11)),
            Grumpkin::generator().scalar_mul(&Fq::from_u64(17)),
        ];
        let coefficient_values = [Fq::from_u64(3), Fq::from_u64(5)];
        let commitments = commitment_values
            .iter()
            .map(|commitment| GrumpkinPointWithIdentityVar::alloc(&mut builder, commitment))
            .collect::<Vec<_>>();
        let coefficients = coefficient_values
            .iter()
            .copied()
            .map(|coefficient| FqVar::alloc(&mut builder, coefficient))
            .collect::<Vec<_>>();

        let combined =
            <Pedersen<GrumpkinPoint> as VectorCommitmentR1cs>::linear_combine_commitments(
                &mut builder,
                &commitments,
                &coefficients,
            )
            .expect("linear combination");
        let expected = GrumpkinPointWithIdentityVar::constant(&GrumpkinPoint::msm(
            &commitment_values,
            &coefficient_values,
        ));

        combined.assert_equal(&mut builder, &expected);
        assert!(builder_accepts(builder));
    }

    #[test]
    fn linear_combine_grumpkin_commitments_rejects_tampered_coefficient() {
        let mut builder = R1csBuilder::<Fr>::new();
        let commitment_values = [
            Grumpkin::generator().scalar_mul(&Fq::from_u64(11)),
            Grumpkin::generator().scalar_mul(&Fq::from_u64(17)),
        ];
        let coefficient_values = [Fq::from_u64(3), Fq::from_u64(5)];
        let commitments = commitment_values
            .iter()
            .map(|commitment| GrumpkinPointWithIdentityVar::alloc(&mut builder, commitment))
            .collect::<Vec<_>>();
        let coefficients = coefficient_values
            .iter()
            .copied()
            .map(|coefficient| FqVar::alloc(&mut builder, coefficient))
            .collect::<Vec<_>>();

        let combined =
            linear_combine_grumpkin_commitments(&mut builder, &commitments, &coefficients)
                .expect("linear combination");
        combined.assert_valid(&mut builder);
        let targets = [(
            "linear-combine coefficient limb",
            variable(&coefficients[0].limbs()[0]),
        )];

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn linear_combine_grumpkin_commitments_rejects_length_mismatch() {
        let mut builder = R1csBuilder::<Fr>::new();
        let commitments = vec![GrumpkinPointWithIdentityVar::alloc(
            &mut builder,
            &Grumpkin::generator(),
        )];
        let coefficients = vec![
            FqVar::alloc(&mut builder, Fq::from_u64(3)),
            FqVar::alloc(&mut builder, Fq::from_u64(5)),
        ];

        assert_eq!(
            linear_combine_grumpkin_commitments(&mut builder, &commitments, &coefficients),
            Err(CryptoR1csError::FixedBaseMsmLengthMismatch {
                bases: 1,
                scalars: 2,
            })
        );
    }

    #[test]
    fn grumpkin_pedersen_opening_accepts_valid_opening() {
        let mut builder = R1csBuilder::<Fr>::new();
        let setup = grumpkin_pedersen_setup();
        let value_scalars = vec![Fq::from_u64(3), Fq::from_u64(5)];
        let blinding_scalar = Fq::from_u64(7);
        let commitment_value =
            Pedersen::<GrumpkinPoint>::commit(&setup, &value_scalars, &blinding_scalar);
        let commitment = GrumpkinPointWithIdentityVar::alloc(&mut builder, &commitment_value);
        let opening = grumpkin_pedersen_opening_var(&mut builder, &value_scalars, blinding_scalar);

        verify_grumpkin_pedersen_opening(&mut builder, &setup, &commitment, &opening)
            .expect("valid Pedersen opening");

        assert!(builder_accepts(builder));
    }

    #[test]
    fn vector_commitment_r1cs_trait_verifies_grumpkin_pedersen_opening() {
        let mut builder = R1csBuilder::<Fr>::new();
        let setup = grumpkin_pedersen_setup();
        let value_scalars = vec![Fq::from_u64(3), Fq::from_u64(5)];
        let blinding_scalar = Fq::from_u64(7);
        let commitment_value =
            Pedersen::<GrumpkinPoint>::commit(&setup, &value_scalars, &blinding_scalar);
        let commitment = GrumpkinPointWithIdentityVar::alloc(&mut builder, &commitment_value);
        let opening = grumpkin_pedersen_opening_var(&mut builder, &value_scalars, blinding_scalar);

        <Pedersen<GrumpkinPoint> as VectorCommitmentR1cs>::verify_opening(
            &mut builder,
            &setup,
            &commitment,
            &opening,
        )
        .expect("valid Pedersen opening");

        assert_eq!(
            <Pedersen<GrumpkinPoint> as VectorCommitmentR1cs>::capacity(&setup),
            setup.message_generators.len()
        );
        assert!(builder_accepts(builder));
    }

    #[test]
    fn grumpkin_pedersen_opening_rejects_tampered_value() {
        let mut builder = R1csBuilder::<Fr>::new();
        let setup = grumpkin_pedersen_setup();
        let value_scalars = vec![Fq::from_u64(3), Fq::from_u64(5)];
        let blinding_scalar = Fq::from_u64(7);
        let commitment_value =
            Pedersen::<GrumpkinPoint>::commit(&setup, &value_scalars, &blinding_scalar);
        let commitment = GrumpkinPointWithIdentityVar::alloc(&mut builder, &commitment_value);
        let opening = grumpkin_pedersen_opening_var(&mut builder, &value_scalars, blinding_scalar);
        let targets = [(
            "Pedersen opening value limb",
            variable(&opening.values[0].limbs()[0]),
        )];

        verify_grumpkin_pedersen_opening(&mut builder, &setup, &commitment, &opening)
            .expect("valid Pedersen opening before tampering");

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn grumpkin_pedersen_opening_rejects_tampered_blinding() {
        let mut builder = R1csBuilder::<Fr>::new();
        let setup = grumpkin_pedersen_setup();
        let value_scalars = vec![Fq::from_u64(3), Fq::from_u64(5)];
        let blinding_scalar = Fq::from_u64(7);
        let commitment_value =
            Pedersen::<GrumpkinPoint>::commit(&setup, &value_scalars, &blinding_scalar);
        let commitment = GrumpkinPointWithIdentityVar::alloc(&mut builder, &commitment_value);
        let opening = grumpkin_pedersen_opening_var(&mut builder, &value_scalars, blinding_scalar);
        let targets = [(
            "Pedersen opening blinding limb",
            variable(&opening.blinding.limbs()[0]),
        )];

        verify_grumpkin_pedersen_opening(&mut builder, &setup, &commitment, &opening)
            .expect("valid Pedersen opening before tampering");

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn grumpkin_pedersen_opening_rejects_tampered_commitment() {
        let mut builder = R1csBuilder::<Fr>::new();
        let setup = grumpkin_pedersen_setup();
        let value_scalars = vec![Fq::from_u64(3), Fq::from_u64(5)];
        let blinding_scalar = Fq::from_u64(7);
        let commitment_value =
            Pedersen::<GrumpkinPoint>::commit(&setup, &value_scalars, &blinding_scalar);
        let commitment = GrumpkinPointWithIdentityVar::alloc(&mut builder, &commitment_value);
        let opening = grumpkin_pedersen_opening_var(&mut builder, &value_scalars, blinding_scalar);
        let targets = [("Pedersen commitment x-coordinate", variable(&commitment.x))];

        verify_grumpkin_pedersen_opening(&mut builder, &setup, &commitment, &opening)
            .expect("valid Pedersen opening before tampering");

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn grumpkin_pedersen_opening_rejects_capacity_exceeded() {
        let mut builder = R1csBuilder::<Fr>::new();
        let setup = PedersenSetup::new(
            vec![Grumpkin::generator().scalar_mul(&Fq::from_u64(11))],
            Grumpkin::generator().scalar_mul(&Fq::from_u64(23)),
        );
        let opening = grumpkin_pedersen_opening_var(
            &mut builder,
            &[Fq::from_u64(3), Fq::from_u64(5)],
            Fq::from_u64(7),
        );

        assert_eq!(
            grumpkin_pedersen_opening_commitment(&mut builder, &setup, &opening),
            Err(CryptoR1csError::VectorCommitmentCapacityExceeded {
                capacity: 1,
                values: 2,
            })
        );
    }

    #[test]
    fn vector_commitment_opening_var_preserves_values_and_blinding() {
        let mut builder = R1csBuilder::<Fr>::new();
        let values = vec![
            jolt_r1cs::FqVar::alloc(&mut builder, Fq::from_u64(3)),
            jolt_r1cs::FqVar::alloc(&mut builder, Fq::from_u64(5)),
        ];
        let blinding = jolt_r1cs::FqVar::alloc(&mut builder, Fq::from_u64(7));

        let opening = VectorCommitmentOpeningVar::new(values.clone(), blinding.clone());

        assert_eq!(opening.values, values);
        assert_eq!(opening.blinding, blinding);
        assert!(builder_accepts(builder));
    }

    fn assert_group_var_valid<G>(
        builder: &mut R1csBuilder<Fr>,
        point: &G,
    ) -> Result<(), CryptoR1csError>
    where
        G: GroupElementVar<BuilderField = Fr, Error = CryptoR1csError>,
    {
        point.assert_valid(builder)
    }

    fn assert_generic_nonexceptional_add<G>(
        builder: &mut R1csBuilder<Fr>,
        lhs: &G,
        rhs: &G,
        output: &G,
    ) -> Result<(), CryptoR1csError>
    where
        G: NonExceptionalAddGroupVar<BuilderField = Fr, Error = CryptoR1csError>,
    {
        G::assert_nonexceptional_add(builder, lhs, rhs, output)
    }

    fn assert_generic_double<G>(
        builder: &mut R1csBuilder<Fr>,
        input: &G,
        output: &G,
    ) -> Result<(), CryptoR1csError>
    where
        G: DoubleGroupVar<BuilderField = Fr, Error = CryptoR1csError>,
    {
        G::assert_double(builder, input, output)
    }

    fn assert_generic_complete_add<G>(
        builder: &mut R1csBuilder<Fr>,
        lhs: &G,
        rhs: &G,
    ) -> Result<G, CryptoR1csError>
    where
        G: CompleteAddGroupVar<BuilderField = Fr, Error = CryptoR1csError>,
    {
        G::complete_add(builder, lhs, rhs)
    }

    fn assert_generic_variable_base_scalar_mul<G>(
        builder: &mut R1csBuilder<Fr>,
        base: &G,
        scalar: &FqVar,
    ) -> Result<G, CryptoR1csError>
    where
        G: VariableBaseScalarMulGroupVar<
            BuilderField = Fr,
            ScalarVar = FqVar,
            Error = CryptoR1csError,
        >,
    {
        G::variable_base_scalar_mul(builder, base, scalar)
    }

    fn assert_generic_fixed_base_scalar_mul<G>(
        builder: &mut R1csBuilder<Fr>,
        base: &GrumpkinPoint,
        scalar: &FqVar,
    ) -> Result<G, CryptoR1csError>
    where
        G: FixedBaseScalarMulGroupVar<
            BuilderField = Fr,
            ScalarVar = FqVar,
            Constant = GrumpkinPoint,
            Error = CryptoR1csError,
        >,
    {
        G::fixed_base_scalar_mul(builder, base, scalar)
    }

    fn grumpkin_pedersen_setup() -> PedersenSetup<GrumpkinPoint> {
        PedersenSetup::new(
            vec![
                Grumpkin::generator().scalar_mul(&Fq::from_u64(11)),
                Grumpkin::generator().scalar_mul(&Fq::from_u64(17)),
            ],
            Grumpkin::generator().scalar_mul(&Fq::from_u64(23)),
        )
    }

    fn grumpkin_pedersen_opening_var(
        builder: &mut R1csBuilder<Fr>,
        values: &[Fq],
        blinding: Fq,
    ) -> VectorCommitmentOpeningVar<FqVar> {
        VectorCommitmentOpeningVar::new(
            values
                .iter()
                .copied()
                .map(|value| FqVar::alloc(builder, value))
                .collect(),
            FqVar::alloc(builder, blinding),
        )
    }

    fn builder_accepts(builder: R1csBuilder<Fr>) -> bool {
        let witness = builder.witness().expect("witness is assigned");
        builder.into_matrices().check_witness(&witness).is_ok()
    }

    fn builder_rejects(builder: R1csBuilder<Fr>) -> bool {
        let witness = builder.witness().expect("witness is assigned");
        builder.into_matrices().check_witness(&witness).is_err()
    }

    fn assert_tampering_rejected(
        builder: R1csBuilder<Fr>,
        targets: impl IntoIterator<Item = (&'static str, Variable)>,
    ) {
        let witness = builder.witness().expect("witness is assigned");
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());

        for (label, variable) in targets {
            let mut tampered = witness.clone();
            tampered[variable.index()] += Fr::from_u64(1);
            assert!(
                matrices.check_witness(&tampered).is_err(),
                "{label} accepted after tampering variable {}",
                variable.index()
            );
        }
    }

    fn variable(scalar: &AssignedScalar<Fr>) -> Variable {
        scalar
            .lc
            .terms
            .first()
            .copied()
            .expect("expected scalar backed by one variable")
            .0
    }
}
