#![expect(clippy::expect_used, reason = "integration tests may fail by panic")]

use jolt_field::{Fq, Fr, FromPrimitiveInt};
use jolt_poly::{
    r1cs::{eq_evals, multilinear_eval},
    EqPolynomial, Polynomial,
};
use jolt_r1cs::{AssignedScalar, FqVar, R1csBuilder, Variable};

#[test]
fn native_poly_r1cs_multilinear_eval_accepts_valid_witness() {
    let mut builder = R1csBuilder::<Fr>::new();
    let evaluation_values = (0..8)
        .map(|index| Fr::from_u64((3 * index + 2) as u64))
        .collect::<Vec<_>>();
    let point_values = [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
    let evaluations = evaluation_values
        .iter()
        .copied()
        .map(|value| AssignedScalar::alloc(&mut builder, value))
        .collect::<Vec<_>>();
    let point = point_values
        .iter()
        .copied()
        .map(|value| AssignedScalar::alloc(&mut builder, value))
        .collect::<Vec<_>>();

    let result = multilinear_eval(&mut builder, &evaluations, &point).expect("evaluation succeeds");
    let expected = Polynomial::new(evaluation_values).evaluate(&point_values);
    builder.assert_equal(result.lc, AssignedScalar::constant(expected).lc);

    assert!(builder_accepts(builder));
}

#[test]
fn native_poly_r1cs_rejects_single_variable_tampering() {
    let mut builder = R1csBuilder::<Fr>::new();
    let evaluation_values = (0..8)
        .map(|index| Fr::from_u64((5 * index + 1) as u64))
        .collect::<Vec<_>>();
    let point_values = [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
    let evaluations = evaluation_values
        .iter()
        .copied()
        .map(|value| AssignedScalar::alloc(&mut builder, value))
        .collect::<Vec<_>>();
    let point = point_values
        .iter()
        .copied()
        .map(|value| AssignedScalar::alloc(&mut builder, value))
        .collect::<Vec<_>>();

    let result = multilinear_eval(&mut builder, &evaluations, &point).expect("evaluation succeeds");
    let expected = Polynomial::new(evaluation_values).evaluate(&point_values);
    builder.assert_equal(result.lc, AssignedScalar::constant(expected).lc);

    assert_single_variable_tampering_rejected(builder);
}

#[test]
fn nonnative_poly_r1cs_multilinear_eval_accepts_valid_witness() {
    let mut builder = R1csBuilder::<Fr>::new();
    let evaluation_values = (0..8)
        .map(|index| Fq::from_u64((7 * index + 4) as u64))
        .collect::<Vec<_>>();
    let point_values = [Fq::from_u64(2), Fq::from_u64(3), Fq::from_u64(5)];
    let evaluations = evaluation_values
        .iter()
        .copied()
        .map(|value| FqVar::alloc(&mut builder, value))
        .collect::<Vec<_>>();
    let point = point_values
        .iter()
        .copied()
        .map(|value| FqVar::alloc(&mut builder, value))
        .collect::<Vec<_>>();

    let result = multilinear_eval(&mut builder, &evaluations, &point).expect("evaluation succeeds");
    let expected = Polynomial::new(evaluation_values).evaluate(&point_values);
    result.assert_equal(&mut builder, &FqVar::constant(expected));

    assert!(builder_accepts(builder));
}

#[test]
fn nonnative_poly_r1cs_rejects_single_variable_tampering() {
    let mut builder = R1csBuilder::<Fr>::new();
    let evaluation_values = (0..8)
        .map(|index| Fq::from_u64((11 * index + 4) as u64))
        .collect::<Vec<_>>();
    let point_values = [Fq::from_u64(2), Fq::from_u64(3), Fq::from_u64(5)];
    let evaluations = evaluation_values
        .iter()
        .copied()
        .map(|value| FqVar::alloc(&mut builder, value))
        .collect::<Vec<_>>();
    let point = point_values
        .iter()
        .copied()
        .map(|value| FqVar::alloc(&mut builder, value))
        .collect::<Vec<_>>();

    let result = multilinear_eval(&mut builder, &evaluations, &point).expect("evaluation succeeds");
    let expected = Polynomial::new(evaluation_values).evaluate(&point_values);
    result.assert_equal(&mut builder, &FqVar::constant(expected));

    let tamper_targets = [
        variable(&evaluations[0].limbs()[0]).index(),
        variable(&point[0].limbs()[0]).index(),
        variable(&result.limbs()[0]).index(),
    ];
    assert_selected_variable_tampering_rejected(builder, &tamper_targets);
}

#[test]
fn native_eq_table_order_matches_jolt_poly_plain_order() {
    let mut builder = R1csBuilder::<Fr>::new();
    let point_values = [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
    let point = point_values
        .iter()
        .copied()
        .map(|value| AssignedScalar::alloc(&mut builder, value))
        .collect::<Vec<_>>();

    let actual = eq_evals(&mut builder, &point);
    let expected = EqPolynomial::new(point_values.to_vec()).evaluations();
    for (actual, expected) in actual.into_iter().zip(expected) {
        builder.assert_equal(actual.lc, AssignedScalar::constant(expected).lc);
    }

    assert!(builder_accepts(builder));
}

fn builder_accepts<F>(builder: R1csBuilder<F>) -> bool
where
    F: jolt_field::Field,
{
    let witness = builder.witness().expect("witness is assigned");
    builder.into_matrices().check_witness(&witness).is_ok()
}

fn assert_single_variable_tampering_rejected<F>(builder: R1csBuilder<F>)
where
    F: jolt_field::Field,
{
    let witness = builder.witness().expect("witness is assigned");
    let matrices = builder.into_matrices();
    assert!(matrices.check_witness(&witness).is_ok());

    for variable in 1..witness.len() {
        let mut tampered = witness.clone();
        tampered[variable] += F::one();
        assert!(
            matrices.check_witness(&tampered).is_err(),
            "tampering variable {variable} was accepted"
        );
    }
}

fn assert_selected_variable_tampering_rejected<F>(builder: R1csBuilder<F>, targets: &[usize])
where
    F: jolt_field::Field,
{
    let witness = builder.witness().expect("witness is assigned");
    let matrices = builder.into_matrices();
    assert!(matrices.check_witness(&witness).is_ok());

    for &variable in targets {
        let mut tampered = witness.clone();
        tampered[variable] += F::one();
        assert!(
            matrices.check_witness(&tampered).is_err(),
            "tampering variable {variable} was accepted"
        );
    }
}

fn variable<F>(scalar: &AssignedScalar<F>) -> Variable
where
    F: jolt_field::Field,
{
    assert_eq!(scalar.lc.terms.len(), 1);
    let (variable, coefficient) = scalar
        .lc
        .terms
        .first()
        .copied()
        .expect("assigned scalar should be backed by one variable");
    assert_eq!(coefficient, F::one());
    variable
}
