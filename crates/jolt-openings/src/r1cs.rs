//! R1CS helpers for generic opening-claim preparation.
//!
//! This module only constrains scheme-independent opening algebra. Concrete
//! commitment checks belong to the R1CS module of the selected PCS.

use thiserror::Error;

use jolt_r1cs::{R1csBuilder, ScalarGadget};

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum OpeningR1csError {
    #[error("opening claim reduction requires at least one opening claim")]
    EmptyOpeningClaims,
    #[error("opening point length mismatch: expected {expected}, got {got}")]
    OpeningPointLengthMismatch { expected: usize, got: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningClaimVar<S, C>
where
    S: ScalarGadget,
{
    pub commitment: C,
    pub point: Vec<S>,
    pub opening_claim: S,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedOpeningClaimScalars<S>
where
    S: ScalarGadget,
{
    pub point: Vec<S>,
    pub opening_claim: S,
}

impl<S, C> OpeningClaimVar<S, C>
where
    S: ScalarGadget,
{
    pub fn new(commitment: C, point: Vec<S>, opening_claim: S) -> Self {
        Self {
            commitment,
            point,
            opening_claim,
        }
    }
}

pub fn reduce_same_point_opening_claims<S, C>(
    builder: &mut R1csBuilder<S::BuilderField>,
    claims: &[OpeningClaimVar<S, C>],
    batching_challenge: &S,
) -> Result<ReducedOpeningClaimScalars<S>, OpeningR1csError>
where
    S: ScalarGadget,
{
    let point = assert_same_opening_point(builder, claims)?;
    let opening_claim = reduce_opening_claim_scalars(
        builder,
        claims.iter().map(|claim| &claim.opening_claim),
        batching_challenge,
    )?;

    Ok(ReducedOpeningClaimScalars {
        point,
        opening_claim,
    })
}

pub fn assert_same_opening_point<S, C>(
    builder: &mut R1csBuilder<S::BuilderField>,
    claims: &[OpeningClaimVar<S, C>],
) -> Result<Vec<S>, OpeningR1csError>
where
    S: ScalarGadget,
{
    let Some((first, rest)) = claims.split_first() else {
        return Err(OpeningR1csError::EmptyOpeningClaims);
    };

    for claim in rest {
        if claim.point.len() != first.point.len() {
            return Err(OpeningR1csError::OpeningPointLengthMismatch {
                expected: first.point.len(),
                got: claim.point.len(),
            });
        }

        for (actual, expected) in claim.point.iter().zip(&first.point) {
            actual.assert_equal(builder, expected);
        }
    }

    Ok(first.point.clone())
}

pub fn reduce_opening_claim_scalars<'a, S>(
    builder: &mut R1csBuilder<S::BuilderField>,
    opening_claims: impl IntoIterator<Item = &'a S>,
    batching_challenge: &S,
) -> Result<S, OpeningR1csError>
where
    S: ScalarGadget + 'a,
{
    let opening_claims = opening_claims.into_iter().collect::<Vec<_>>();
    let Some((last, rest)) = opening_claims.split_last() else {
        return Err(OpeningR1csError::EmptyOpeningClaims);
    };

    let mut reduced = (*last).clone();
    for opening_claim in rest.iter().rev() {
        reduced = reduced.mul(builder, batching_challenge);
        reduced = reduced.add(builder, opening_claim);
    }

    Ok(reduced)
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use jolt_field::{Fq, Fr, FromPrimitiveInt};
    use jolt_r1cs::{AssignedScalar, FqVar, Variable};
    use jolt_transcript::r1cs::{PoseidonR1csTranscript, R1csJoltTranscript, R1csTranscript};

    use super::*;
    use crate::rlc_combine_scalars;

    #[test]
    fn native_reduces_same_point_opening_claims() {
        let mut builder = R1csBuilder::<Fr>::new();
        let claims = native_claims(&mut builder);
        let gamma = AssignedScalar::alloc(&mut builder, Fr::from_u64(2));

        let reduced = reduce_same_point_opening_claims(&mut builder, &claims, &gamma)
            .expect("same-point opening claims reduce");

        reduced
            .opening_claim
            .assert_equal(&mut builder, &AssignedScalar::constant(Fr::from_u64(170)));
        assert_eq!(reduced.point.len(), 2);
        assert!(builder_accepts(builder));
    }

    #[test]
    fn native_reduction_matches_opening_rlc_helper() {
        let mut builder = R1csBuilder::<Fr>::new();
        let values = [
            Fr::from_u64(10),
            Fr::from_u64(20),
            Fr::from_u64(30),
            Fr::from_u64(40),
        ];
        let opening_claims = values
            .iter()
            .copied()
            .map(|value| AssignedScalar::alloc(&mut builder, value))
            .collect::<Vec<_>>();
        let gamma = AssignedScalar::alloc(&mut builder, Fr::from_u64(7));
        let expected = rlc_combine_scalars(&values, gamma.value);

        let reduced = reduce_opening_claim_scalars(&mut builder, &opening_claims, &gamma)
            .expect("opening claims reduce");
        reduced.assert_equal(&mut builder, &AssignedScalar::constant(expected));

        assert!(builder_accepts(builder));
    }

    #[test]
    fn native_rejects_opening_claim_challenge_and_point_tampering() {
        let mut builder = R1csBuilder::<Fr>::new();
        let a = AssignedScalar::alloc(&mut builder, Fr::from_u64(10));
        let b = AssignedScalar::alloc(&mut builder, Fr::from_u64(20));
        let c = AssignedScalar::alloc(&mut builder, Fr::from_u64(30));
        let gamma = AssignedScalar::alloc(&mut builder, Fr::from_u64(2));
        let point0 = AssignedScalar::alloc(&mut builder, Fr::from_u64(7));
        let point1 = AssignedScalar::alloc(&mut builder, Fr::from_u64(8));
        let point1_copy = AssignedScalar::alloc(&mut builder, Fr::from_u64(8));
        let targets = [
            ("opening claim", variable(&a)),
            ("batching challenge", variable(&gamma)),
            ("opening point", variable(&point1_copy)),
        ];
        let claims = vec![
            OpeningClaimVar::new(0usize, vec![point0.clone(), point1], a),
            OpeningClaimVar::new(1usize, vec![point0.clone(), point1_copy], b),
            OpeningClaimVar::new(
                2usize,
                vec![point0, AssignedScalar::constant(Fr::from_u64(8))],
                c,
            ),
        ];

        let reduced = reduce_same_point_opening_claims(&mut builder, &claims, &gamma)
            .expect("same-point opening claims reduce");
        reduced
            .opening_claim
            .assert_equal(&mut builder, &AssignedScalar::constant(Fr::from_u64(170)));

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn native_rejects_wrong_opening_point() {
        let mut builder = R1csBuilder::<Fr>::new();
        let gamma = AssignedScalar::alloc(&mut builder, Fr::from_u64(2));
        let claims = vec![
            OpeningClaimVar::new(
                0usize,
                vec![AssignedScalar::alloc(&mut builder, Fr::from_u64(7))],
                AssignedScalar::alloc(&mut builder, Fr::from_u64(10)),
            ),
            OpeningClaimVar::new(
                1usize,
                vec![AssignedScalar::alloc(&mut builder, Fr::from_u64(8))],
                AssignedScalar::alloc(&mut builder, Fr::from_u64(20)),
            ),
        ];

        let _ = reduce_same_point_opening_claims(&mut builder, &claims, &gamma)
            .expect("point equality constraints are emitted");

        assert!(builder_rejects(builder));
    }

    #[test]
    fn native_accepts_independent_same_point_groups() {
        let mut builder = R1csBuilder::<Fr>::new();
        let gamma = AssignedScalar::alloc(&mut builder, Fr::from_u64(3));
        let first_group = vec![
            OpeningClaimVar::new(
                0usize,
                vec![AssignedScalar::alloc(&mut builder, Fr::from_u64(7))],
                AssignedScalar::alloc(&mut builder, Fr::from_u64(10)),
            ),
            OpeningClaimVar::new(
                1usize,
                vec![AssignedScalar::alloc(&mut builder, Fr::from_u64(7))],
                AssignedScalar::alloc(&mut builder, Fr::from_u64(20)),
            ),
        ];
        let second_group = vec![
            OpeningClaimVar::new(
                2usize,
                vec![AssignedScalar::alloc(&mut builder, Fr::from_u64(99))],
                AssignedScalar::alloc(&mut builder, Fr::from_u64(30)),
            ),
            OpeningClaimVar::new(
                3usize,
                vec![AssignedScalar::alloc(&mut builder, Fr::from_u64(99))],
                AssignedScalar::alloc(&mut builder, Fr::from_u64(40)),
            ),
        ];

        let first = reduce_same_point_opening_claims(&mut builder, &first_group, &gamma)
            .expect("first point group reduces");
        let second = reduce_same_point_opening_claims(&mut builder, &second_group, &gamma)
            .expect("second point group reduces");

        first
            .opening_claim
            .assert_equal(&mut builder, &AssignedScalar::constant(Fr::from_u64(70)));
        second
            .opening_claim
            .assert_equal(&mut builder, &AssignedScalar::constant(Fr::from_u64(150)));

        assert!(builder_accepts(builder));
    }

    #[test]
    fn transcript_challenge_composes_with_native_reduction() {
        let mut builder = R1csBuilder::<Fr>::new();
        let mut transcript = PoseidonR1csTranscript::new(&mut builder, b"OpeningR1cs");
        let claims = [
            AssignedScalar::alloc(&mut builder, Fr::from_u64(11)),
            AssignedScalar::alloc(&mut builder, Fr::from_u64(13)),
            AssignedScalar::alloc(&mut builder, Fr::from_u64(17)),
        ];
        transcript.append_scalars(&mut builder, b"opening_claim", &claims);
        let gamma = transcript.challenge_scalar(&mut builder);
        let expected_values: [Fr; 3] = std::array::from_fn(|index| claims[index].value);
        let expected = rlc_combine_scalars(&expected_values, gamma.value);
        let targets = [
            ("transcript opening claim", variable(&claims[0])),
            ("transcript challenge", variable(&gamma)),
        ];

        let reduced =
            reduce_opening_claim_scalars(&mut builder, &claims, &gamma).expect("claims reduce");
        reduced.assert_equal(&mut builder, &AssignedScalar::constant(expected));

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn nonnative_reduces_same_point_opening_claims() {
        let mut builder = R1csBuilder::<Fr>::new();
        let claims = nonnative_claims(&mut builder);
        let gamma = FqVar::alloc(&mut builder, Fq::from_u64(2));

        let reduced = reduce_same_point_opening_claims(&mut builder, &claims, &gamma)
            .expect("same-point opening claims reduce");

        reduced
            .opening_claim
            .assert_equal(&mut builder, &FqVar::constant(Fq::from_u64(170)));
        assert_eq!(reduced.point.len(), 2);
        assert!(builder_accepts(builder));
    }

    #[test]
    fn nonnative_reduction_matches_opening_rlc_helper() {
        let mut builder = R1csBuilder::<Fr>::new();
        let values = [
            Fq::from_u64(10),
            Fq::from_u64(20),
            Fq::from_u64(30),
            Fq::from_u64(40),
        ];
        let opening_claims = values
            .iter()
            .copied()
            .map(|value| FqVar::alloc(&mut builder, value))
            .collect::<Vec<_>>();
        let gamma_value = Fq::from_u64(7);
        let gamma = FqVar::alloc(&mut builder, gamma_value);
        let expected = rlc_combine_scalars(&values, gamma_value);

        let reduced = reduce_opening_claim_scalars(&mut builder, &opening_claims, &gamma)
            .expect("opening claims reduce");
        reduced.assert_equal(&mut builder, &FqVar::constant(expected));

        assert!(builder_accepts(builder));
    }

    #[test]
    fn nonnative_rejects_opening_claim_challenge_and_point_tampering() {
        let mut builder = R1csBuilder::<Fr>::new();
        let a = FqVar::alloc(&mut builder, Fq::from_u64(10));
        let b = FqVar::alloc(&mut builder, Fq::from_u64(20));
        let c = FqVar::alloc(&mut builder, Fq::from_u64(30));
        let gamma = FqVar::alloc(&mut builder, Fq::from_u64(2));
        let point0 = FqVar::alloc(&mut builder, Fq::from_u64(7));
        let point1 = FqVar::alloc(&mut builder, Fq::from_u64(8));
        let point1_copy = FqVar::alloc(&mut builder, Fq::from_u64(8));
        let targets = [
            ("opening claim limb", variable(&a.limbs()[0])),
            ("batching challenge limb", variable(&gamma.limbs()[0])),
            ("opening point limb", variable(&point1_copy.limbs()[0])),
        ];
        let claims = vec![
            OpeningClaimVar::new(0usize, vec![point0.clone(), point1], a),
            OpeningClaimVar::new(1usize, vec![point0.clone(), point1_copy], b),
            OpeningClaimVar::new(2usize, vec![point0, FqVar::constant(Fq::from_u64(8))], c),
        ];

        let reduced = reduce_same_point_opening_claims(&mut builder, &claims, &gamma)
            .expect("same-point opening claims reduce");
        reduced
            .opening_claim
            .assert_equal(&mut builder, &FqVar::constant(Fq::from_u64(170)));

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn nonnative_rejects_wrong_opening_claim() {
        let mut builder = R1csBuilder::<Fr>::new();
        let claims = nonnative_claims(&mut builder);
        let gamma = FqVar::alloc(&mut builder, Fq::from_u64(2));

        let reduced = reduce_same_point_opening_claims(&mut builder, &claims, &gamma)
            .expect("same-point opening claims reduce");
        reduced
            .opening_claim
            .assert_equal(&mut builder, &FqVar::constant(Fq::from_u64(171)));

        assert!(builder_rejects(builder));
    }

    #[test]
    fn empty_opening_claims_are_typed_errors() {
        let mut builder = R1csBuilder::<Fr>::new();
        let gamma = AssignedScalar::constant(Fr::from_u64(2));

        assert_eq!(
            reduce_same_point_opening_claims::<AssignedScalar<Fr>, usize>(
                &mut builder,
                &[],
                &gamma
            ),
            Err(OpeningR1csError::EmptyOpeningClaims)
        );
        assert_eq!(
            reduce_opening_claim_scalars::<AssignedScalar<Fr>>(&mut builder, &[], &gamma),
            Err(OpeningR1csError::EmptyOpeningClaims)
        );
    }

    #[test]
    fn opening_point_length_mismatch_is_a_typed_error() {
        let mut builder = R1csBuilder::<Fr>::new();
        let gamma = AssignedScalar::constant(Fr::from_u64(2));
        let claims = vec![
            OpeningClaimVar::new(
                0usize,
                vec![AssignedScalar::constant(Fr::from_u64(7))],
                AssignedScalar::constant(Fr::from_u64(10)),
            ),
            OpeningClaimVar::new(
                1usize,
                vec![
                    AssignedScalar::constant(Fr::from_u64(7)),
                    AssignedScalar::constant(Fr::from_u64(8)),
                ],
                AssignedScalar::constant(Fr::from_u64(20)),
            ),
        ];

        assert_eq!(
            reduce_same_point_opening_claims(&mut builder, &claims, &gamma),
            Err(OpeningR1csError::OpeningPointLengthMismatch {
                expected: 1,
                got: 2
            })
        );
    }

    fn native_claims(
        builder: &mut R1csBuilder<Fr>,
    ) -> Vec<OpeningClaimVar<AssignedScalar<Fr>, usize>> {
        let point0 = AssignedScalar::alloc(builder, Fr::from_u64(7));
        let point1 = AssignedScalar::alloc(builder, Fr::from_u64(8));
        vec![
            OpeningClaimVar::new(
                0,
                vec![point0.clone(), point1.clone()],
                AssignedScalar::alloc(builder, Fr::from_u64(10)),
            ),
            OpeningClaimVar::new(
                1,
                vec![point0.clone(), point1.clone()],
                AssignedScalar::alloc(builder, Fr::from_u64(20)),
            ),
            OpeningClaimVar::new(
                2,
                vec![point0, point1],
                AssignedScalar::alloc(builder, Fr::from_u64(30)),
            ),
        ]
    }

    fn nonnative_claims(builder: &mut R1csBuilder<Fr>) -> Vec<OpeningClaimVar<FqVar, usize>> {
        let point0 = FqVar::alloc(builder, Fq::from_u64(7));
        let point1 = FqVar::alloc(builder, Fq::from_u64(8));
        vec![
            OpeningClaimVar::new(
                0,
                vec![point0.clone(), point1.clone()],
                FqVar::alloc(builder, Fq::from_u64(10)),
            ),
            OpeningClaimVar::new(
                1,
                vec![point0.clone(), point1.clone()],
                FqVar::alloc(builder, Fq::from_u64(20)),
            ),
            OpeningClaimVar::new(
                2,
                vec![point0, point1],
                FqVar::alloc(builder, Fq::from_u64(30)),
            ),
        ]
    }

    fn builder_accepts<F>(builder: R1csBuilder<F>) -> bool
    where
        F: jolt_field::Field,
    {
        let witness = builder.witness().expect("witness is assigned");
        builder.into_matrices().check_witness(&witness).is_ok()
    }

    fn builder_rejects<F>(builder: R1csBuilder<F>) -> bool
    where
        F: jolt_field::Field,
    {
        let witness = builder.witness().expect("witness is assigned");
        builder.into_matrices().check_witness(&witness).is_err()
    }

    fn assert_tampering_rejected<F>(
        builder: R1csBuilder<F>,
        targets: impl IntoIterator<Item = (&'static str, Variable)>,
    ) where
        F: jolt_field::Field,
    {
        let witness = builder.witness().expect("witness is assigned");
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());

        for (label, variable) in targets {
            let mut tampered = witness.clone();
            tampered[variable.index()] += F::one();
            assert!(
                matrices.check_witness(&tampered).is_err(),
                "{label} accepted after tampering variable {}",
                variable.index()
            );
        }
    }

    fn variable<F>(scalar: &AssignedScalar<F>) -> Variable
    where
        F: jolt_field::Field,
    {
        scalar
            .lc
            .terms
            .first()
            .copied()
            .expect("expected scalar backed by one variable")
            .0
    }
}
