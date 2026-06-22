#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

use jolt_akita::{
    AkitaCommitInput, AkitaCommitment, AkitaField, AkitaPackedScheme, AkitaScheme,
    AkitaSetupParams, AKITA_FIELD_MODULUS,
};
use jolt_openings::{
    BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme, OpeningsError,
    PackedAlphabet, PackedCellAddress, PackedCombine, PackedFactDomain, PackedFamilyId,
    PackedFamilySpec, PackedLayoutError, PackedLinearTerm, PackedWitnessLayout,
    PackedWitnessSource, PhysicalView, SparsePackedWitness, ZkOpeningScheme,
};
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_transcript::{Blake2bTranscript, Transcript};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpeningId {
    A,
    B,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RelationId {
    Packed,
}

type PackedAkita = PackedCombine<AkitaScheme>;

fn f(value: u64) -> AkitaField {
    AkitaField::from_u64(value)
}

fn layout(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn polynomial(offset: u64) -> Polynomial<AkitaField> {
    Polynomial::new((0..16).map(|value| f(value + offset)).collect())
}

fn setup() -> (
    <AkitaScheme as CommitmentScheme>::ProverSetup,
    <AkitaScheme as CommitmentScheme>::VerifierSetup,
) {
    AkitaScheme::setup(AkitaSetupParams::new(4, 2, layout(7)))
}

fn packed_layout() -> PackedWitnessLayout {
    PackedWitnessLayout::new([PackedFamilySpec::direct(
        PackedFamilyId::Custom {
            namespace: 1,
            index: 0,
        },
        PackedFactDomain::TraceRows { log_t: 1 },
        1,
        PackedAlphabet::Fixed { size: 3 },
    )])
    .expect("packed layout should be valid")
}

fn ring_sized_packed_layout() -> PackedWitnessLayout {
    PackedWitnessLayout::new([PackedFamilySpec::direct(
        PackedFamilyId::Custom {
            namespace: 3,
            index: 0,
        },
        PackedFactDomain::TraceRows { log_t: 4 },
        1,
        PackedAlphabet::Fixed { size: 4 },
    )])
    .expect("ring-sized packed layout should be valid")
}

fn packed_reduction_family() -> PackedFamilyId {
    PackedFamilyId::Custom {
        namespace: 2,
        index: 0,
    }
}

fn packed_reduction_layout() -> PackedWitnessLayout {
    PackedWitnessLayout::new([PackedFamilySpec::direct(
        packed_reduction_family(),
        PackedFactDomain::TraceRows { log_t: 2 },
        2,
        PackedAlphabet::Bit,
    )])
    .expect("packed reduction layout should be valid")
}

fn packed_address(row: usize, symbol: usize) -> PackedCellAddress {
    PackedCellAddress {
        family: PackedFamilyId::Custom {
            namespace: 1,
            index: 0,
        },
        row,
        limb: 0,
        symbol,
    }
}

fn ring_sized_packed_address(row: usize, symbol: usize) -> PackedCellAddress {
    PackedCellAddress {
        family: PackedFamilyId::Custom {
            namespace: 3,
            index: 0,
        },
        row,
        limb: 0,
        symbol,
    }
}

fn packed_reduction_address(row: usize, limb: usize, symbol: usize) -> PackedCellAddress {
    PackedCellAddress {
        family: packed_reduction_family(),
        row,
        limb,
        symbol,
    }
}

fn packed_term(coefficient: AkitaField) -> PackedLinearTerm<AkitaField> {
    packed_term_at(coefficient, 0)
}

fn packed_term_at(coefficient: AkitaField, symbol: usize) -> PackedLinearTerm<AkitaField> {
    PackedLinearTerm::new(
        coefficient,
        (PackedFamilyId::Custom {
            namespace: 1,
            index: 0,
        })
        .physical_ref(),
        0,
        symbol,
    )
}

fn packed_reduction_term(
    coefficient: AkitaField,
    limb: usize,
    symbol: usize,
    row_point: &[AkitaField],
) -> PackedLinearTerm<AkitaField> {
    PackedLinearTerm::new(
        coefficient,
        packed_reduction_family().physical_ref(),
        limb,
        symbol,
    )
    .with_row_point(row_point.to_vec())
}

fn ring_sized_packed_term(
    coefficient: AkitaField,
    symbol: usize,
    row_point: &[AkitaField],
) -> PackedLinearTerm<AkitaField> {
    PackedLinearTerm::new(
        coefficient,
        (PackedFamilyId::Custom {
            namespace: 3,
            index: 0,
        })
        .physical_ref(),
        0,
        symbol,
    )
    .with_row_point(row_point.to_vec())
}

fn packed_polynomial(
    layout: &PackedWitnessLayout,
    entries: &[(usize, AkitaField)],
) -> Polynomial<AkitaField> {
    let mut evals = vec![AkitaField::zero(); 1usize << layout.dimension];
    for &(rank, value) in entries {
        evals[rank] = value;
    }
    Polynomial::new(evals)
}

fn packed_view_eval(
    layout: &PackedWitnessLayout,
    witness: &SparsePackedWitness<AkitaField>,
    terms: &[PackedLinearTerm<AkitaField>],
) -> AkitaField {
    terms.iter().fold(AkitaField::zero(), |acc, term| {
        let family = layout
            .families
            .iter()
            .find(|family| family.id.physical_ref() == term.family)
            .expect("term family should exist");
        let row_weights = EqPolynomial::new(term.row_point.clone()).evaluations();
        let contribution = row_weights.iter().copied().enumerate().fold(
            AkitaField::zero(),
            |acc, (row, row_weight)| {
                let value = witness
                    .eval_direct_fact(&PackedCellAddress {
                        family: family.id.clone(),
                        row,
                        limb: term.limb,
                        symbol: term.symbol,
                    })
                    .expect("packed address should be valid");
                acc + row_weight * value
            },
        );
        acc + term.coefficient * contribution
    })
}

struct EmittingPackedSource {
    layout: PackedWitnessLayout,
    entries: Vec<(usize, AkitaField)>,
}

impl PackedWitnessSource<AkitaField> for EmittingPackedSource {
    fn layout(&self) -> &PackedWitnessLayout {
        &self.layout
    }

    fn for_each_nonzero(&self, mut f: impl FnMut(usize, AkitaField)) {
        for &(rank, value) in &self.entries {
            f(rank, value);
        }
    }

    fn eval_direct_fact(
        &self,
        address: &PackedCellAddress,
    ) -> Result<AkitaField, PackedLayoutError> {
        let rank = self.layout.rank(address)?;
        Ok(self
            .entries
            .iter()
            .find(|(entry_rank, _)| *entry_rank == rank)
            .map_or_else(AkitaField::zero, |(_, value)| *value))
    }
}

fn direct_statement(
    commitment: AkitaCommitment,
    point: &[AkitaField],
    eval_a: AkitaField,
    eval_b: AkitaField,
) -> BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId> {
    BatchOpeningStatement {
        logical_point: point.to_vec(),
        pcs_point: point.to_vec(),
        layout_digest: layout(7),
        claims: vec![
            BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::Packed,
                commitment: commitment.clone(),
                claim: eval_a,
                view: PhysicalView::Direct,
                scale: f(2),
            },
            BatchOpeningClaim {
                id: OpeningId::B,
                relation: RelationId::Packed,
                commitment,
                claim: eval_b,
                view: PhysicalView::Direct,
                scale: f(5),
            },
        ],
    }
}

fn unit_packed_statement(
    commitment: AkitaCommitment,
    point: &[AkitaField],
    eval_a: AkitaField,
    eval_b: AkitaField,
) -> BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId> {
    BatchOpeningStatement {
        logical_point: point.to_vec(),
        pcs_point: point.to_vec(),
        layout_digest: layout(7),
        claims: vec![
            BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::Packed,
                commitment: commitment.clone(),
                claim: eval_a,
                view: PhysicalView::PackedLinear {
                    layout_digest: layout(7),
                    terms: vec![packed_term(f(1))],
                },
                scale: f(2),
            },
            BatchOpeningClaim {
                id: OpeningId::B,
                relation: RelationId::Packed,
                commitment,
                claim: eval_b,
                view: PhysicalView::PackedLinear {
                    layout_digest: layout(7),
                    terms: vec![packed_term(f(1))],
                },
                scale: f(5),
            },
        ],
    }
}

fn packed_reduction_statement(
    layout: &PackedWitnessLayout,
    commitment: AkitaCommitment,
    row_point: &[AkitaField],
    terms_a: Vec<PackedLinearTerm<AkitaField>>,
    claim_a: AkitaField,
    terms_b: Vec<PackedLinearTerm<AkitaField>>,
    claim_b: AkitaField,
) -> BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId> {
    BatchOpeningStatement {
        logical_point: row_point.to_vec(),
        pcs_point: row_point.to_vec(),
        layout_digest: layout.digest,
        claims: vec![
            BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::Packed,
                commitment: commitment.clone(),
                claim: claim_a,
                view: PhysicalView::PackedLinear {
                    layout_digest: layout.digest,
                    terms: terms_a,
                },
                scale: f(3),
            },
            BatchOpeningClaim {
                id: OpeningId::B,
                relation: RelationId::Packed,
                commitment,
                claim: claim_b,
                view: PhysicalView::PackedLinear {
                    layout_digest: layout.digest,
                    terms: terms_b,
                },
                scale: f(7),
            },
        ],
    }
}

fn run_on_large_stack(test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(test)
        .expect("failed to spawn test thread")
        .join()
        .expect("test thread panicked");
}

#[test]
fn setup_params_report_packed_dimension_and_digest() {
    let layout = packed_layout();
    let params = AkitaSetupParams::from_packed_layout(&layout, 1);
    assert_eq!(params.max_num_vars, layout.dimension);
    assert_eq!(params.default_layout_digest, layout.digest);
    assert_eq!(params.packed_layout.as_ref(), Some(&layout));
}

#[test]
fn akita_commit_packed_source_roundtrip() {
    run_on_large_stack(|| {
        let layout = packed_layout();
        assert!(layout.dummy_cell_count() > 0);
        let witness = SparsePackedWitness::try_from_cells(
            layout.clone(),
            [(packed_address(0, 1), f(7)), (packed_address(1, 2), f(11))],
        )
        .expect("sparse witness should be valid");
        let poly = packed_polynomial(&layout, witness.entries());
        let point = vec![f(2), f(3), f(5)];
        let eval = poly.evaluate(&point);
        let (prover_setup, verifier_setup) =
            AkitaScheme::setup(AkitaSetupParams::from_packed_layout(&layout, 1));

        let (commitment, hint) = AkitaScheme::commit_packed_source(&prover_setup, &witness)
            .expect("source commit should succeed");
        assert_eq!(commitment.layout_digest, layout.digest);
        assert_eq!(commitment.num_vars, layout.dimension);
        assert_eq!(commitment.poly_count, 1);

        let mut prover_transcript = Blake2bTranscript::new(b"akita-packed-source");
        let proof = AkitaScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prover_transcript,
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"akita-packed-source");
        AkitaScheme::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
        )
        .expect("source proof should verify");
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    });
}

#[test]
fn akita_commit_packed_source_roundtrip_with_sparse_unit_source() {
    run_on_large_stack(|| {
        let layout = ring_sized_packed_layout();
        assert_eq!(layout.dimension, 6);
        assert_eq!(layout.dummy_cell_count(), 0);
        let witness = SparsePackedWitness::try_from_cells(
            layout.clone(),
            [
                (ring_sized_packed_address(0, 1), AkitaField::one()),
                (ring_sized_packed_address(15, 3), AkitaField::one()),
            ],
        )
        .expect("sparse one-hot witness should be valid");
        let poly = packed_polynomial(&layout, witness.entries());
        let point = vec![f(2), f(3), f(5), f(7), f(11), f(13)];
        let eval = poly.evaluate(&point);
        let (prover_setup, verifier_setup) =
            AkitaScheme::setup(AkitaSetupParams::from_packed_layout(&layout, 1));

        let (commitment, hint) = AkitaScheme::commit_packed_source(&prover_setup, &witness)
            .expect("source commit should succeed");
        assert_eq!(commitment.layout_digest, layout.digest);
        assert_eq!(commitment.num_vars, layout.dimension);
        assert_eq!(commitment.poly_count, 1);

        let statement = BatchOpeningStatement {
            logical_point: point.clone(),
            pcs_point: point.clone(),
            layout_digest: layout.digest,
            claims: vec![BatchOpeningClaim {
                id: (),
                relation: (),
                commitment: commitment.clone(),
                claim: eval,
                view: PhysicalView::Direct,
                scale: AkitaField::one(),
            }],
        };

        let mut prover_transcript = Blake2bTranscript::new(b"akita-packed-source-sparse");
        let proof = AkitaPackedScheme::prove_packed_source_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            &witness,
            hint,
        )
        .expect("source proof should be produced");

        let mut verifier_transcript = Blake2bTranscript::new(b"akita-packed-source-sparse");
        let _ = AkitaPackedScheme::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("source proof should verify");
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    });
}

#[test]
fn akita_packed_source_sparse_unit_views_roundtrip() {
    run_on_large_stack(|| {
        let layout = ring_sized_packed_layout();
        assert_eq!(layout.dimension, 6);
        let witness = SparsePackedWitness::try_from_cells(
            layout.clone(),
            [
                (ring_sized_packed_address(0, 1), AkitaField::one()),
                (ring_sized_packed_address(15, 3), AkitaField::one()),
            ],
        )
        .expect("sparse one-hot witness should be valid");
        let (prover_setup, verifier_setup) =
            AkitaPackedScheme::setup(AkitaSetupParams::from_packed_layout(&layout, 1));
        let (commitment, hint) = AkitaPackedScheme::commit_packed_source(&prover_setup, &witness)
            .expect("source commit should succeed");

        let row_point = vec![f(2), f(3), f(5), f(7)];
        let terms_a = vec![ring_sized_packed_term(f(1), 1, &row_point)];
        let terms_b = vec![
            ring_sized_packed_term(f(2), 1, &row_point),
            ring_sized_packed_term(f(3), 3, &row_point),
        ];
        let claim_a = packed_view_eval(&layout, &witness, &terms_a);
        let claim_b = packed_view_eval(&layout, &witness, &terms_b);
        let statement = packed_reduction_statement(
            &layout,
            commitment.clone(),
            &row_point,
            terms_a,
            claim_a,
            terms_b,
            claim_b,
        );

        let mut prover_transcript = Blake2bTranscript::new(b"akita-packed-sparse-reduction");
        let proof = AkitaPackedScheme::prove_packed_source_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            &witness,
            hint,
        )
        .expect("sparse packed reduction proof should be produced");
        assert!(
            proof.reduction.is_some(),
            "packed views should still produce a reduction proof"
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"akita-packed-sparse-reduction");
        let result = AkitaPackedScheme::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("sparse packed reduction proof should verify");

        assert_eq!(result.joint_commitment, commitment);
        assert_eq!(
            result.reduced_opening,
            result.coefficients[0] * claim_a + result.coefficients[1] * claim_b
        );
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    });
}

#[test]
fn akita_commit_packed_source_rejects_malformed_emitters() {
    let layout = packed_layout();
    let (prover_setup, _) = AkitaScheme::setup(AkitaSetupParams::from_packed_layout(&layout, 1));

    let dummy_rank = layout.cells;
    let dummy_result = AkitaScheme::commit_packed_source(
        &prover_setup,
        &EmittingPackedSource {
            layout: layout.clone(),
            entries: vec![(dummy_rank, f(1))],
        },
    );
    assert!(matches!(dummy_result, Err(OpeningsError::InvalidBatch(_))));

    let duplicate_result = AkitaScheme::commit_packed_source(
        &prover_setup,
        &EmittingPackedSource {
            layout: layout.clone(),
            entries: vec![(0, f(1)), (0, f(2))],
        },
    );
    assert!(matches!(
        duplicate_result,
        Err(OpeningsError::InvalidBatch(_))
    ));

    let zero_result = AkitaScheme::commit_packed_source(
        &prover_setup,
        &EmittingPackedSource {
            layout,
            entries: vec![(0, AkitaField::zero())],
        },
    );
    assert!(matches!(zero_result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn akita_commit_packed_source_rejects_non_unit_sparse_values() {
    let layout = ring_sized_packed_layout();
    let (prover_setup, _) = AkitaScheme::setup(AkitaSetupParams::from_packed_layout(&layout, 1));

    let result = AkitaScheme::commit_packed_source(
        &prover_setup,
        &EmittingPackedSource {
            layout,
            entries: vec![(0, f(7))],
        },
    );

    assert!(
        matches!(result, Err(OpeningsError::InvalidBatch(reason)) if reason.contains("non-unit")),
        "real-size packed W_pack sources must be unit one-hot entries"
    );
}

#[test]
fn akita_packed_scheme_rejects_generic_dense_single_opening_path() {
    run_on_large_stack(|| {
        let (prover_setup, _) = setup();
        let poly = polynomial(1);
        let point = vec![f(2), f(3), f(5), f(7)];
        let eval = poly.evaluate(&point);

        let commit_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            <AkitaPackedScheme as CommitmentScheme>::commit(&poly, &prover_setup)
        }));
        assert!(
            commit_result.is_err(),
            "AkitaPackedScheme generic commit must not bypass PackedWitnessSource"
        );

        let (_, hint) =
            AkitaScheme::commit_group(&prover_setup, layout(7), std::slice::from_ref(&poly))
                .expect("direct commitment should commit");
        let open_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut transcript = Blake2bTranscript::new(b"akita-packed-dense-open");
            <AkitaPackedScheme as CommitmentScheme>::open(
                &poly,
                &point,
                eval,
                &prover_setup,
                Some(hint),
                &mut transcript,
            )
        }));
        assert!(
            open_result.is_err(),
            "AkitaPackedScheme generic open must not bypass PackedWitnessSource"
        );

        let zk_commit_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            <AkitaPackedScheme as ZkOpeningScheme>::commit_zk(&poly, &prover_setup)
        }));
        assert!(
            zk_commit_result.is_err(),
            "AkitaPackedScheme generic ZK commit must not bypass PackedWitnessSource"
        );
    });
}

#[test]
fn akita_packed_scheme_keeps_direct_batch_opening_path() {
    run_on_large_stack(|| {
        let (prover_setup, verifier_setup) =
            AkitaPackedScheme::setup(AkitaSetupParams::new(4, 2, layout(7)));
        let poly_a = polynomial(1);
        let poly_b = polynomial(20);
        let point = vec![f(2), f(3), f(5), f(7)];
        let eval_a = poly_a.evaluate(&point);
        let eval_b = poly_b.evaluate(&point);
        let (commitment, hint) =
            AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
                .expect("grouped commit should succeed");
        let statement = direct_statement(commitment.clone(), &point, eval_a, eval_b);

        let mut prover_transcript = Blake2bTranscript::new(b"akita-packed-direct");
        let proof = <AkitaPackedScheme as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            &[poly_a, poly_b],
            vec![hint],
        )
        .expect("direct batch proof should be produced");
        assert!(
            proof.reduction.is_none(),
            "direct precommitted-style openings must not use a PackedWitness reduction"
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"akita-packed-direct");
        let result = <AkitaPackedScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("direct batch proof should verify");

        assert_eq!(result.joint_commitment, commitment);
        assert_eq!(result.coefficients, vec![f(2), f(5)]);
        assert_eq!(result.reduced_opening, f(2) * eval_a + f(5) * eval_b);
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    });
}

#[test]
fn packed_combine_akita_unit_packed_views_roundtrip() {
    run_on_large_stack(|| {
        let (prover_setup, verifier_setup) = setup();
        let poly_a = polynomial(1);
        let poly_b = polynomial(20);
        let point = vec![f(2), f(3), f(5), f(7)];
        let eval_a = poly_a.evaluate(&point);
        let eval_b = poly_b.evaluate(&point);
        let (commitment, hint) =
            AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
                .expect("grouped commit should succeed");
        let statement = unit_packed_statement(commitment.clone(), &point, eval_a, eval_b);

        let mut prover_transcript = Blake2bTranscript::new(b"akita-packed-combine");
        let proof = <PackedAkita as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            &[poly_a, poly_b],
            vec![hint],
        )
        .expect("packed wrapper should produce an Akita proof");

        let mut verifier_transcript = Blake2bTranscript::new(b"akita-packed-combine");
        let result = <PackedAkita as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("packed wrapper should verify through Akita");

        assert_eq!(result.joint_commitment, commitment);
        assert_eq!(result.coefficients, vec![f(2), f(5)]);
        assert_eq!(result.reduced_opening, f(2) * eval_a + f(5) * eval_b);
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    });
}

#[test]
fn akita_packed_scheme_reduces_packed_views_to_native_opening() {
    run_on_large_stack(|| {
        let layout = packed_reduction_layout();
        let witness = SparsePackedWitness::try_from_cells(
            layout.clone(),
            [
                (packed_reduction_address(0, 0, 1), f(11)),
                (packed_reduction_address(1, 0, 1), f(13)),
                (packed_reduction_address(2, 0, 1), f(17)),
                (packed_reduction_address(3, 0, 1), f(19)),
                (packed_reduction_address(0, 1, 0), f(23)),
                (packed_reduction_address(1, 1, 0), f(29)),
                (packed_reduction_address(2, 1, 0), f(31)),
                (packed_reduction_address(3, 1, 0), f(37)),
            ],
        )
        .expect("packed witness should be valid");
        let (prover_setup, verifier_setup) =
            AkitaPackedScheme::setup(AkitaSetupParams::from_packed_layout(&layout, 1));
        let (commitment, hint) = AkitaScheme::commit_packed_source(&prover_setup, &witness)
            .expect("source commit should succeed");
        let row_point = vec![f(2), f(5)];
        let terms_a = vec![packed_reduction_term(f(1), 0, 1, &row_point)];
        let terms_b = vec![
            packed_reduction_term(f(2), 0, 1, &row_point),
            packed_reduction_term(f(3), 1, 0, &row_point),
        ];
        let claim_a = packed_view_eval(&layout, &witness, &terms_a);
        let claim_b = packed_view_eval(&layout, &witness, &terms_b);
        let statement = packed_reduction_statement(
            &layout,
            commitment.clone(),
            &row_point,
            terms_a,
            claim_a,
            terms_b,
            claim_b,
        );

        let mut prover_transcript = Blake2bTranscript::new(b"akita-packed-reduction");
        let proof = AkitaPackedScheme::prove_packed_source_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            &witness,
            hint,
        )
        .expect("packed reduction proof should be produced");
        assert!(
            proof.reduction.is_some(),
            "packed views should produce a reduction proof"
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"akita-packed-reduction");
        let result = <AkitaPackedScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("packed reduction proof should verify");

        assert_eq!(result.joint_commitment, commitment);
        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(
            result.reduced_opening,
            result.coefficients[0] * claim_a + result.coefficients[1] * claim_b
        );
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    });
}

#[test]
fn akita_packed_scheme_rejects_tampered_packed_statement() {
    run_on_large_stack(|| {
        let layout = packed_reduction_layout();
        let witness = SparsePackedWitness::try_from_cells(
            layout.clone(),
            [
                (packed_reduction_address(0, 0, 1), f(11)),
                (packed_reduction_address(1, 0, 1), f(13)),
                (packed_reduction_address(2, 0, 1), f(17)),
                (packed_reduction_address(3, 0, 1), f(19)),
                (packed_reduction_address(0, 1, 0), f(23)),
                (packed_reduction_address(1, 1, 0), f(29)),
                (packed_reduction_address(2, 1, 0), f(31)),
                (packed_reduction_address(3, 1, 0), f(37)),
            ],
        )
        .expect("packed witness should be valid");
        let (prover_setup, verifier_setup) =
            AkitaPackedScheme::setup(AkitaSetupParams::from_packed_layout(&layout, 1));
        let (commitment, hint) = AkitaScheme::commit_packed_source(&prover_setup, &witness)
            .expect("source commit should succeed");
        let row_point = vec![f(2), f(5)];
        let terms_a = vec![packed_reduction_term(f(1), 0, 1, &row_point)];
        let terms_b = vec![
            packed_reduction_term(f(2), 0, 1, &row_point),
            packed_reduction_term(f(3), 1, 0, &row_point),
        ];
        let claim_a = packed_view_eval(&layout, &witness, &terms_a);
        let claim_b = packed_view_eval(&layout, &witness, &terms_b);
        let statement = packed_reduction_statement(
            &layout, commitment, &row_point, terms_a, claim_a, terms_b, claim_b,
        );

        let mut prover_transcript = Blake2bTranscript::new(b"akita-packed-tamper");
        let proof = AkitaPackedScheme::prove_packed_source_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            &witness,
            hint,
        )
        .expect("packed reduction proof should be produced");

        let mut noncanonical_proof = proof.clone();
        noncanonical_proof
            .reduction
            .as_mut()
            .expect("packed proof should contain a reduction")
            .opening_eval = AKITA_FIELD_MODULUS.to_le_bytes().to_vec();
        let mut verifier_transcript = Blake2bTranscript::new(b"akita-packed-tamper");
        let result = <AkitaPackedScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &statement,
            &noncanonical_proof,
        );
        assert!(
            result.is_err(),
            "noncanonical packed reduction field bytes should reject"
        );

        let mut claim_tampered = statement.clone();
        claim_tampered.claims[0].claim += f(1);
        let mut verifier_transcript = Blake2bTranscript::new(b"akita-packed-tamper");
        let result = <AkitaPackedScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &claim_tampered,
            &proof,
        );
        assert!(result.is_err(), "changed packed claim should reject");

        let mut row_point_tampered = statement;
        assert!(matches!(
            &row_point_tampered.claims[0].view,
            PhysicalView::PackedLinear { .. }
        ));
        if let PhysicalView::PackedLinear { terms, .. } = &mut row_point_tampered.claims[0].view {
            terms[0].row_point[0] += f(1);
        }
        let mut verifier_transcript = Blake2bTranscript::new(b"akita-packed-tamper");
        let result = <AkitaPackedScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &row_point_tampered,
            &proof,
        );
        assert!(result.is_err(), "changed packed row point should reject");
    });
}

#[test]
fn akita_packed_scheme_requires_one_packed_witness_commitment() {
    run_on_large_stack(|| {
        let layout = packed_reduction_layout();
        let witness = SparsePackedWitness::try_from_cells(
            layout.clone(),
            [
                (packed_reduction_address(0, 0, 1), f(11)),
                (packed_reduction_address(1, 0, 1), f(13)),
                (packed_reduction_address(2, 0, 1), f(17)),
                (packed_reduction_address(3, 0, 1), f(19)),
                (packed_reduction_address(0, 1, 0), f(23)),
                (packed_reduction_address(1, 1, 0), f(29)),
                (packed_reduction_address(2, 1, 0), f(31)),
                (packed_reduction_address(3, 1, 0), f(37)),
            ],
        )
        .expect("packed witness should be valid");
        let (prover_setup, verifier_setup) =
            AkitaPackedScheme::setup(AkitaSetupParams::from_packed_layout(&layout, 2));
        let (commitment, hint) = AkitaScheme::commit_packed_source(&prover_setup, &witness)
            .expect("source commit should succeed");
        let row_point = vec![f(2), f(5)];
        let terms_a = vec![packed_reduction_term(f(1), 0, 1, &row_point)];
        let terms_b = vec![
            packed_reduction_term(f(2), 0, 1, &row_point),
            packed_reduction_term(f(3), 1, 0, &row_point),
        ];
        let claim_a = packed_view_eval(&layout, &witness, &terms_a);
        let claim_b = packed_view_eval(&layout, &witness, &terms_b);
        let statement = packed_reduction_statement(
            &layout, commitment, &row_point, terms_a, claim_a, terms_b, claim_b,
        );

        let mut prover_transcript = Blake2bTranscript::new(b"akita-packed-single-witness");
        let proof = AkitaPackedScheme::prove_packed_source_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            &witness,
            hint,
        )
        .expect("packed reduction proof should be produced");

        let (group_commitment, _) = AkitaScheme::commit_group(
            &prover_setup,
            layout.digest,
            &[polynomial(100), polynomial(200)],
        )
        .expect("grouped commitment should succeed");
        let mut group_commitment_statement = statement.clone();
        for claim in &mut group_commitment_statement.claims {
            claim.commitment = group_commitment.clone();
        }
        let mut verifier_transcript = Blake2bTranscript::new(b"akita-packed-single-witness");
        let result = <AkitaPackedScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &group_commitment_statement,
            &proof,
        );
        assert!(
            matches!(result, Err(OpeningsError::InvalidBatch(_))),
            "packed view statements must reject grouped commitments"
        );

        let (other_commitment, _) =
            AkitaScheme::commit_group(&prover_setup, layout.digest, &[polynomial(300)])
                .expect("alternate commitment should succeed");
        let mut mixed_commitment_statement = statement;
        mixed_commitment_statement.claims[1].commitment = other_commitment;
        let mut verifier_transcript = Blake2bTranscript::new(b"akita-packed-single-witness");
        let result = <AkitaPackedScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &mixed_commitment_statement,
            &proof,
        );
        assert!(
            matches!(result, Err(OpeningsError::InvalidBatch(_))),
            "packed view statements must use one packed witness commitment"
        );
    });
}

#[test]
fn akita_packed_source_hint_layout_mismatch_rejects() {
    run_on_large_stack(|| {
        let layout_a = packed_reduction_layout();
        let family_b = PackedFamilyId::Custom {
            namespace: 3,
            index: 0,
        };
        let layout_b = PackedWitnessLayout::new([PackedFamilySpec::direct(
            family_b.clone(),
            PackedFactDomain::TraceRows { log_t: 2 },
            2,
            PackedAlphabet::Bit,
        )])
        .expect("alternate packed layout should be valid");
        assert_eq!(layout_a.dimension, layout_b.dimension);
        assert_ne!(layout_a.digest, layout_b.digest);

        let witness_a = SparsePackedWitness::try_from_cells(
            layout_a.clone(),
            [(packed_reduction_address(0, 0, 1), f(11))],
        )
        .expect("packed witness A should be valid");
        let witness_b = SparsePackedWitness::try_from_cells(
            layout_b.clone(),
            [(
                PackedCellAddress {
                    family: family_b.clone(),
                    row: 0,
                    limb: 0,
                    symbol: 1,
                },
                f(11),
            )],
        )
        .expect("packed witness B should be valid");
        let (prover_setup_a, _) =
            AkitaPackedScheme::setup(AkitaSetupParams::from_packed_layout(&layout_a, 1));
        let (prover_setup_b, _) =
            AkitaPackedScheme::setup(AkitaSetupParams::from_packed_layout(&layout_b, 1));
        let (_, hint_a) = AkitaPackedScheme::commit_packed_source(&prover_setup_a, &witness_a)
            .expect("source A commit should succeed");
        let (commitment_b, _) =
            AkitaPackedScheme::commit_packed_source(&prover_setup_b, &witness_b)
                .expect("source B commit should succeed");

        let row_point = vec![f(2), f(5)];
        let terms = vec![PackedLinearTerm::new(f(1), family_b.physical_ref(), 0, 1)
            .with_row_point(row_point.clone())];
        let claim = packed_view_eval(&layout_b, &witness_b, &terms);
        let statement = BatchOpeningStatement {
            logical_point: row_point.clone(),
            pcs_point: row_point,
            layout_digest: layout_b.digest,
            claims: vec![BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::Packed,
                commitment: commitment_b,
                claim,
                view: PhysicalView::PackedLinear {
                    layout_digest: layout_b.digest,
                    terms,
                },
                scale: f(1),
            }],
        };

        let mut transcript = Blake2bTranscript::new(b"akita-packed-hint-layout");
        let result = AkitaPackedScheme::prove_packed_source_batch(
            &prover_setup_b,
            &mut transcript,
            &statement,
            &witness_b,
            hint_a,
        );
        assert!(
            matches!(result, Err(OpeningsError::InvalidBatch(message)) if message.contains("hint")),
            "packed witness hint generated for another layout should reject"
        );
    });
}

#[test]
fn packed_combine_akita_binds_packed_coefficients_to_native_proof() {
    run_on_large_stack(|| {
        let (prover_setup, verifier_setup) = setup();
        let poly_a = polynomial(1);
        let poly_b = polynomial(20);
        let point = vec![f(2), f(3), f(5), f(7)];
        let eval_a = poly_a.evaluate(&point);
        let eval_b = poly_b.evaluate(&point);
        let (commitment, hint) =
            AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
                .expect("grouped commit should succeed");
        let statement = unit_packed_statement(commitment, &point, eval_a, eval_b);

        let mut prover_transcript = Blake2bTranscript::new(b"akita-packed-coeff-tamper");
        let proof = <PackedAkita as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            &[poly_a, poly_b],
            vec![hint],
        )
        .expect("packed proof should be produced");

        let mut tampered = statement;
        tampered.claims[0].view = PhysicalView::PackedLinear {
            layout_digest: layout(7),
            terms: vec![packed_term(f(0)), packed_term_at(f(1), 1)],
        };
        let mut verifier_transcript = Blake2bTranscript::new(b"akita-packed-coeff-tamper");
        let result = <PackedAkita as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &tampered,
            &proof,
        );
        assert!(result.is_err(), "changed packed coefficients should reject");
    });
}

#[test]
fn akita_native_adapter_rejects_packed_linear_view_until_lowered() {
    let (prover_setup, _) = setup();
    let poly = polynomial(1);
    let point = vec![f(2), f(3), f(5), f(7)];
    let eval = poly.evaluate(&point);
    let (commitment, hint) = AkitaScheme::commit_packed_witness(
        &prover_setup,
        AkitaCommitInput {
            layout_digest: layout(7),
            polynomial: poly.clone(),
        },
    )
    .expect("commit should succeed");
    let statement = BatchOpeningStatement {
        logical_point: point.clone(),
        pcs_point: point,
        layout_digest: layout(7),
        claims: vec![BatchOpeningClaim {
            id: OpeningId::A,
            relation: RelationId::Packed,
            commitment,
            claim: eval,
            view: PhysicalView::PackedLinear {
                layout_digest: layout(7),
                terms: vec![packed_term(f(1))],
            },
            scale: f(1),
        }],
    };

    let mut transcript = Blake2bTranscript::new(b"akita-packed");
    let result = <AkitaScheme as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        &mut transcript,
        &statement,
        &[poly],
        vec![hint],
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}
