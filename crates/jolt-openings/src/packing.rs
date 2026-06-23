mod batch;
mod encoding;
mod reduction;
mod selector;
mod transcript;
mod types;
mod util;

pub use reduction::{
    has_packing_view, prove_packing_reduction, prove_sparse_packing_reduction,
    validate_packing_statement, verify_packing_reduction,
};
pub use types::{
    PackingAddress, PackingBatch, PackingBatchProof, PackingFamily, PackingLayout,
    PackingProverReduction, PackingProverSetup, PackingReductionProof, PackingSetupParams,
    PackingVerifierReduction, PackingVerifierSetup, PackingWitnessSource,
};

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        clippy::panic,
        reason = "tests assert successful packed reduction setup and fail loudly on malformed fixtures"
    )]

    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::mock::{MockCommitment, MockCommitmentScheme, MockProof};
    use crate::{
        BatchOpeningClaim, BatchOpeningResult, BatchOpeningScheme, BatchOpeningStatement,
        CommitmentScheme, OpeningsError, PackedFamilyRef, PackingTerm, PhysicalView,
    };
    use jolt_crypto::Commitment;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{EqPolynomial, MultilinearPoly, Polynomial};
    use jolt_transcript::{Blake2bTranscript, Transcript};

    const FAMILY: PackedFamilyRef = PackedFamilyRef {
        namespace: 0x6a6f_6c74,
        id: 7,
        index: 0,
    };

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct TestPackedPcs;

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct TestLayout {
        digest: [u8; 32],
    }

    impl PackingLayout for TestLayout {
        fn digest(&self) -> [u8; 32] {
            self.digest
        }

        fn dimension(&self) -> usize {
            3
        }

        fn cells(&self) -> usize {
            8
        }

        fn family(&self, family: PackedFamilyRef) -> Result<Option<PackingFamily>, OpeningsError> {
            Ok((family == FAMILY).then_some(PackingFamily {
                id: FAMILY,
                offset: 0,
                rows: 4,
                limbs: 1,
                alphabet_size: 2,
            }))
        }

        fn rank(&self, address: PackingAddress) -> Result<usize, OpeningsError> {
            if address.family != FAMILY
                || address.limb != 0
                || address.row >= 4
                || address.symbol >= 2
            {
                return Err(OpeningsError::InvalidBatch(
                    "test packed address out of range".to_string(),
                ));
            }
            Ok(address.row * 2 + address.symbol)
        }
    }

    impl Commitment for TestPackedPcs {
        type Output = MockCommitment<Fr>;
    }

    impl CommitmentScheme for TestPackedPcs {
        type Field = Fr;
        type Proof = MockProof<Fr>;
        type ProverSetup = TestLayout;
        type VerifierSetup = TestLayout;
        type Polynomial = Polynomial<Fr>;
        type OpeningHint = ();
        type SetupParams = TestLayout;

        fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
            (params.clone(), params)
        }

        fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
            prover_setup.clone()
        }

        fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
            poly: &P,
            _setup: &Self::ProverSetup,
        ) -> (Self::Output, Self::OpeningHint) {
            MockCommitmentScheme::<Fr>::commit(poly, &())
        }

        fn open(
            poly: &Self::Polynomial,
            point: &[Self::Field],
            eval: Self::Field,
            _setup: &Self::ProverSetup,
            hint: Option<Self::OpeningHint>,
            transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Self::Proof {
            MockCommitmentScheme::<Fr>::open(poly, point, eval, &(), hint, transcript)
        }

        fn verify(
            commitment: &Self::Output,
            point: &[Self::Field],
            eval: Self::Field,
            proof: &Self::Proof,
            _setup: &Self::VerifierSetup,
            transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Result<(), OpeningsError> {
            MockCommitmentScheme::<Fr>::verify(commitment, point, eval, proof, &(), transcript)
        }

        fn bind_opening_inputs(
            transcript: &mut impl Transcript<Challenge = Self::Field>,
            point: &[Self::Field],
            eval: &Self::Field,
        ) {
            MockCommitmentScheme::<Fr>::bind_opening_inputs(transcript, point, eval);
        }
    }

    impl BatchOpeningScheme for TestPackedPcs {
        fn prove_batch<T, OpeningId, RelationId>(
            _setup: &Self::ProverSetup,
            _transcript: &mut T,
            _statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
            _polynomials: &[Self::Polynomial],
            _hints: Vec<Self::OpeningHint>,
        ) -> Result<Self::Proof, OpeningsError>
        where
            T: Transcript<Challenge = Self::Field>,
        {
            Err(OpeningsError::InvalidBatch(
                "test PCS only supports packed adapter openings".to_string(),
            ))
        }

        fn verify_batch<T, OpeningId, RelationId>(
            _setup: &Self::VerifierSetup,
            _transcript: &mut T,
            _statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
            _proof: &Self::Proof,
        ) -> Result<BatchOpeningResult<Self::Field, Self::Output>, OpeningsError>
        where
            T: Transcript<Challenge = Self::Field>,
        {
            Err(OpeningsError::InvalidBatch(
                "test PCS only supports packed adapter openings".to_string(),
            ))
        }
    }

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn layout() -> TestLayout {
        TestLayout { digest: [17; 32] }
    }

    fn packed_setup(
        layout: TestLayout,
    ) -> (
        PackingProverSetup<TestLayout, TestLayout>,
        PackingVerifierSetup<TestLayout, TestLayout>,
    ) {
        (
            PackingProverSetup {
                pcs: layout.clone(),
                layout: layout.clone(),
            },
            PackingVerifierSetup {
                pcs: layout.clone(),
                layout,
            },
        )
    }

    fn packed_polynomial() -> Polynomial<Fr> {
        Polynomial::new((0..8).map(|index| fr(3 + 2 * index)).collect())
    }

    fn packed_term(coefficient: Fr, symbol: usize, row_point: &[Fr]) -> PackingTerm<Fr> {
        PackingTerm::new(coefficient, FAMILY, 0, symbol).with_row_point(row_point.to_vec())
    }

    fn packed_view_eval(poly: &Polynomial<Fr>, row_point: &[Fr], terms: &[PackingTerm<Fr>]) -> Fr {
        let row_weights = EqPolynomial::new(row_point.to_vec()).evaluations();
        terms.iter().fold(fr(0), |acc, term| {
            acc + row_weights.iter().copied().enumerate().fold(
                fr(0),
                |term_acc, (row, row_weight)| {
                    term_acc
                        + term.coefficient * row_weight * poly.evaluations()[row * 2 + term.symbol]
                },
            )
        })
    }

    fn statement(
        commitment: MockCommitment<Fr>,
        poly: &Polynomial<Fr>,
        row_point: &[Fr],
    ) -> BatchOpeningStatement<Fr, MockCommitment<Fr>, usize, usize> {
        let first_terms = vec![packed_term(fr(2), 1, row_point)];
        let second_terms = vec![
            packed_term(fr(5), 0, row_point),
            packed_term(fr(7), 1, row_point),
        ];
        BatchOpeningStatement {
            logical_point: row_point.to_vec(),
            pcs_point: row_point.to_vec(),
            layout_digest: [17; 32],
            claims: vec![
                BatchOpeningClaim {
                    id: 0,
                    relation: 0,
                    commitment: commitment.clone(),
                    claim: packed_view_eval(poly, row_point, &first_terms),
                    view: PhysicalView::Packing {
                        layout_digest: [17; 32],
                        terms: first_terms,
                    },
                    scale: fr(11),
                },
                BatchOpeningClaim {
                    id: 1,
                    relation: 1,
                    commitment,
                    claim: packed_view_eval(poly, row_point, &second_terms),
                    view: PhysicalView::Packing {
                        layout_digest: [17; 32],
                        terms: second_terms,
                    },
                    scale: fr(13),
                },
            ],
        }
    }

    #[test]
    fn packing_batch_roundtrip_many_views_one_commitment() {
        type PackedTestPcs = PackingBatch<TestPackedPcs, TestLayout>;

        let layout = layout();
        let (prover_setup, verifier_setup) = packed_setup(layout.clone());
        let poly = packed_polynomial();
        let (commitment, hint) = TestPackedPcs::commit(&poly, &layout);
        let row_point = vec![fr(2), fr(5)];
        let statement = statement(commitment.clone(), &poly, &row_point);

        let mut prover_transcript = Blake2bTranscript::new(b"generic-packing");
        let proof = <PackedTestPcs as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            std::slice::from_ref(&poly),
            vec![hint],
        )
        .expect("generic packed proof should prove");
        assert!(proof.reduction.is_some());

        let mut verifier_transcript = Blake2bTranscript::new(b"generic-packing");
        let result = <PackedTestPcs as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("generic packed proof should verify");

        assert_eq!(result.joint_commitment, commitment);
        assert_eq!(result.coefficients.len(), statement.claims.len());
        assert_eq!(
            result.reduced_opening,
            result
                .coefficients
                .iter()
                .zip(&statement.claims)
                .fold(fr(0), |acc, (coefficient, claim)| {
                    acc + *coefficient * claim.claim
                })
        );
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }

    #[test]
    fn packing_batch_rejects_tampered_view_address() {
        type PackedTestPcs = PackingBatch<TestPackedPcs, TestLayout>;

        let layout = layout();
        let (prover_setup, verifier_setup) = packed_setup(layout.clone());
        let poly = packed_polynomial();
        let (commitment, hint) = TestPackedPcs::commit(&poly, &layout);
        let row_point = vec![fr(2), fr(5)];
        let statement = statement(commitment, &poly, &row_point);
        let mut prover_transcript = Blake2bTranscript::new(b"generic-packing-tamper");
        let proof = <PackedTestPcs as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            std::slice::from_ref(&poly),
            vec![hint],
        )
        .expect("generic packed proof should prove");

        let mut tampered = statement;
        let PhysicalView::Packing { terms, .. } = &mut tampered.claims[0].view else {
            panic!("test statement uses packed views");
        };
        terms[0].symbol = 0;

        let mut verifier_transcript = Blake2bTranscript::new(b"generic-packing-tamper");
        let result = <PackedTestPcs as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &tampered,
            &proof,
        );
        assert!(result.is_err(), "tampered packed address should reject");
    }
}
