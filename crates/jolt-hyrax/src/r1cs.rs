use jolt_crypto::r1cs::{VectorCommitmentOpeningVar, VectorCommitmentR1cs};
use jolt_poly::r1cs::{self as poly_r1cs, PolyR1csError};
use jolt_r1cs::{R1csBuilder, ScalarGadget};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HyraxOpeningR1csInput<S, C> {
    pub row_commitments: Vec<C>,
    pub row_point: Vec<S>,
    pub entry_point: Vec<S>,
    pub combined_row: Vec<S>,
    pub combined_blinding: S,
    pub claimed_eval: S,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum HyraxR1csError<VCError> {
    #[error("dimension {dimension} is too large to fit in usize")]
    DimensionTooLarge { dimension: usize },
    #[error("row commitment count mismatch: expected {expected}, got {got}")]
    RowCommitmentCountMismatch { expected: usize, got: usize },
    #[error("combined row length mismatch: expected {expected}, got {got}")]
    CombinedRowLengthMismatch { expected: usize, got: usize },
    #[error("polynomial R1CS failed: {0}")]
    Polynomial(#[from] PolyR1csError),
    #[error("vector commitment R1CS failed: {0}")]
    VectorCommitment(VCError),
}

pub fn verify_opening<VC>(
    builder: &mut R1csBuilder<VC::BuilderField>,
    setup: &VC::SetupVar,
    input: &HyraxOpeningR1csInput<VC::ScalarVar, VC::OutputVar>,
) -> Result<(), HyraxR1csError<VC::Error>>
where
    VC: VectorCommitmentR1cs,
    VC::ScalarVar: ScalarGadget<BuilderField = VC::BuilderField>,
{
    let expected_rows = hypercube_len::<VC::Error>(input.row_point.len())?;
    if input.row_commitments.len() != expected_rows {
        return Err(HyraxR1csError::RowCommitmentCountMismatch {
            expected: expected_rows,
            got: input.row_commitments.len(),
        });
    }

    let expected_row_len = hypercube_len::<VC::Error>(input.entry_point.len())?;
    if input.combined_row.len() != expected_row_len {
        return Err(HyraxR1csError::CombinedRowLengthMismatch {
            expected: expected_row_len,
            got: input.combined_row.len(),
        });
    }

    let row_weights = poly_r1cs::eq_evals(builder, &input.row_point);
    let combined_commitment =
        VC::linear_combine_commitments(builder, &input.row_commitments, &row_weights)
            .map_err(HyraxR1csError::VectorCommitment)?;

    VC::verify_opening(
        builder,
        setup,
        &combined_commitment,
        &VectorCommitmentOpeningVar::new(
            input.combined_row.clone(),
            input.combined_blinding.clone(),
        ),
    )
    .map_err(HyraxR1csError::VectorCommitment)?;

    let entry_weights = poly_r1cs::eq_evals(builder, &input.entry_point);
    let opened_eval = poly_r1cs::inner_product(builder, &input.combined_row, &entry_weights)?;
    opened_eval.assert_equal(builder, &input.claimed_eval);

    Ok(())
}

fn hypercube_len<VCError>(dimension: usize) -> Result<usize, HyraxR1csError<VCError>> {
    if dimension >= usize::BITS as usize {
        return Err(HyraxR1csError::DimensionTooLarge { dimension });
    }
    Ok(1usize << dimension)
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use jolt_crypto::{
        r1cs::{CryptoR1csError, GrumpkinPointWithIdentityVar},
        Grumpkin, GrumpkinPoint, JoltGroup, Pedersen, PedersenSetup, VectorCommitment,
    };
    use jolt_field::{Fq, Fr, FromPrimitiveInt};
    use jolt_poly::EqPolynomial;
    use jolt_r1cs::{AssignedScalar, FqVar, Variable};

    use super::*;

    type TestVc = Pedersen<GrumpkinPoint>;
    type TestError = HyraxR1csError<CryptoR1csError>;

    #[derive(Clone)]
    struct NativeCase {
        setup: PedersenSetup<GrumpkinPoint>,
        row_commitments: Vec<GrumpkinPoint>,
        row_point: Vec<Fq>,
        entry_point: Vec<Fq>,
        combined_row: Vec<Fq>,
        combined_blinding: Fq,
        claimed_eval: Fq,
    }

    struct AllocatedCase {
        input: HyraxOpeningR1csInput<FqVar, GrumpkinPointWithIdentityVar>,
    }

    #[test]
    fn hyrax_r1cs_accepts_valid_opening() {
        let mut builder = R1csBuilder::<Fr>::new();
        let case = native_case();
        let allocated = allocate_case(&mut builder, &case);

        verify_opening::<TestVc>(&mut builder, &case.setup, &allocated.input)
            .expect("valid Hyrax opening");

        assert!(builder_accepts(builder));
    }

    #[test]
    fn hyrax_r1cs_rejects_tampered_row_commitment() {
        let mut builder = R1csBuilder::<Fr>::new();
        let case = native_case();
        let allocated = allocate_case(&mut builder, &case);
        let targets = [(
            "row commitment x-coordinate",
            variable(&allocated.input.row_commitments[0].x),
        )];

        verify_opening::<TestVc>(&mut builder, &case.setup, &allocated.input)
            .expect("valid Hyrax opening before tampering");

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn hyrax_r1cs_rejects_tampered_row_point() {
        let mut builder = R1csBuilder::<Fr>::new();
        let case = native_case();
        let allocated = allocate_case(&mut builder, &case);
        let targets = [(
            "row point limb",
            variable(&allocated.input.row_point[0].limbs()[0]),
        )];

        verify_opening::<TestVc>(&mut builder, &case.setup, &allocated.input)
            .expect("valid Hyrax opening before tampering");

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn hyrax_r1cs_rejects_tampered_entry_point() {
        let mut builder = R1csBuilder::<Fr>::new();
        let case = native_case();
        let allocated = allocate_case(&mut builder, &case);
        let targets = [(
            "entry point limb",
            variable(&allocated.input.entry_point[0].limbs()[0]),
        )];

        verify_opening::<TestVc>(&mut builder, &case.setup, &allocated.input)
            .expect("valid Hyrax opening before tampering");

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn hyrax_r1cs_rejects_tampered_combined_row() {
        let mut builder = R1csBuilder::<Fr>::new();
        let case = native_case();
        let allocated = allocate_case(&mut builder, &case);
        let targets = [(
            "combined row limb",
            variable(&allocated.input.combined_row[0].limbs()[0]),
        )];

        verify_opening::<TestVc>(&mut builder, &case.setup, &allocated.input)
            .expect("valid Hyrax opening before tampering");

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn hyrax_r1cs_rejects_tampered_combined_blinding() {
        let mut builder = R1csBuilder::<Fr>::new();
        let case = native_case();
        let allocated = allocate_case(&mut builder, &case);
        let targets = [(
            "combined blinding limb",
            variable(&allocated.input.combined_blinding.limbs()[0]),
        )];

        verify_opening::<TestVc>(&mut builder, &case.setup, &allocated.input)
            .expect("valid Hyrax opening before tampering");

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn hyrax_r1cs_rejects_tampered_claimed_eval() {
        let mut builder = R1csBuilder::<Fr>::new();
        let case = native_case();
        let allocated = allocate_case(&mut builder, &case);
        let targets = [(
            "claimed eval limb",
            variable(&allocated.input.claimed_eval.limbs()[0]),
        )];

        verify_opening::<TestVc>(&mut builder, &case.setup, &allocated.input)
            .expect("valid Hyrax opening before tampering");

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn hyrax_r1cs_rejects_row_commitment_count_mismatch() {
        let mut builder = R1csBuilder::<Fr>::new();
        let case = native_case();
        let mut allocated = allocate_case(&mut builder, &case);
        let _ = allocated.input.row_commitments.pop();

        assert_eq!(
            verify_opening::<TestVc>(&mut builder, &case.setup, &allocated.input),
            Err(TestError::RowCommitmentCountMismatch {
                expected: 4,
                got: 3,
            })
        );
    }

    #[test]
    fn hyrax_r1cs_rejects_combined_row_length_mismatch() {
        let mut builder = R1csBuilder::<Fr>::new();
        let case = native_case();
        let mut allocated = allocate_case(&mut builder, &case);
        let _ = allocated.input.combined_row.pop();

        assert_eq!(
            verify_opening::<TestVc>(&mut builder, &case.setup, &allocated.input),
            Err(TestError::CombinedRowLengthMismatch {
                expected: 4,
                got: 3,
            })
        );
    }

    fn native_case() -> NativeCase {
        let generator = Grumpkin::generator();
        let setup = PedersenSetup::new(
            (1..=4)
                .map(|index| generator.scalar_mul(&Fq::from_u64(10 + index)))
                .collect(),
            generator.scalar_mul(&Fq::from_u64(99)),
        );
        let rows = [
            [2, 3, 5, 7],
            [11, 13, 17, 19],
            [23, 29, 31, 37],
            [41, 43, 47, 53],
        ]
        .map(|row| row.map(Fq::from_u64).to_vec());
        let row_point = vec![Fq::from_u64(3), Fq::from_u64(5)];
        let entry_point = vec![Fq::from_u64(7), Fq::from_u64(11)];
        let row_blindings = [
            Fq::from_u64(101),
            Fq::from_u64(103),
            Fq::from_u64(107),
            Fq::from_u64(109),
        ];

        let row_commitments = rows
            .iter()
            .zip(row_blindings)
            .map(|(row, blinding)| TestVc::commit(&setup, row, &blinding))
            .collect::<Vec<_>>();
        let row_weights = EqPolynomial::new(row_point.clone()).evaluations();
        let entry_weights = EqPolynomial::new(entry_point.clone()).evaluations();
        let zero = Fq::from_u64(0);
        let mut combined_row = vec![zero; 4];
        for (row, row_weight) in rows.iter().zip(&row_weights) {
            for (combined, row_entry) in combined_row.iter_mut().zip(row) {
                *combined += *row_weight * *row_entry;
            }
        }
        let combined_blinding = row_blindings
            .iter()
            .zip(&row_weights)
            .fold(zero, |acc, (blinding, row_weight)| {
                acc + *blinding * *row_weight
            });
        let claimed_eval = combined_row
            .iter()
            .zip(&entry_weights)
            .fold(zero, |acc, (entry, entry_weight)| {
                acc + *entry * *entry_weight
            });

        NativeCase {
            setup,
            row_commitments,
            row_point,
            entry_point,
            combined_row,
            combined_blinding,
            claimed_eval,
        }
    }

    fn allocate_case(builder: &mut R1csBuilder<Fr>, case: &NativeCase) -> AllocatedCase {
        AllocatedCase {
            input: HyraxOpeningR1csInput {
                row_commitments: case
                    .row_commitments
                    .iter()
                    .map(|commitment| GrumpkinPointWithIdentityVar::alloc(builder, commitment))
                    .collect(),
                row_point: alloc_fq_vec(builder, &case.row_point),
                entry_point: alloc_fq_vec(builder, &case.entry_point),
                combined_row: alloc_fq_vec(builder, &case.combined_row),
                combined_blinding: FqVar::alloc(builder, case.combined_blinding),
                claimed_eval: FqVar::alloc(builder, case.claimed_eval),
            },
        }
    }

    fn alloc_fq_vec(builder: &mut R1csBuilder<Fr>, values: &[Fq]) -> Vec<FqVar> {
        values
            .iter()
            .copied()
            .map(|value| FqVar::alloc(builder, value))
            .collect()
    }

    fn builder_accepts(builder: R1csBuilder<Fr>) -> bool {
        let witness = builder.witness().expect("witness is assigned");
        builder.into_matrices().check_witness(&witness).is_ok()
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
