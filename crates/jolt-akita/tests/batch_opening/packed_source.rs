use super::support::*;

#[test]
fn setup_params_report_packed_dimension_and_digest() {
    let layout = packed_layout();
    let params = akita_params_for_layout(&layout, 1);
    assert_eq!(params.max_num_vars, layout.dimension);
    assert_eq!(params.default_layout_digest, layout.digest);
    let packed_params = packed_akita_params(&layout, 1);
    assert_eq!(packed_params.layout, layout);
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
            AkitaScheme::setup(akita_params_for_layout(&layout, 1));

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
            AkitaPackedScheme::setup(packed_akita_params(&layout, 1));

        let (commitment, hint) = AkitaPackedScheme::commit_packed_source(&prover_setup, &witness)
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
            AkitaPackedScheme::setup(packed_akita_params(&layout, 1));
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
    let (prover_setup, _) = AkitaScheme::setup(akita_params_for_layout(&layout, 1));

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
    let (prover_setup, _) = AkitaScheme::setup(akita_params_for_layout(&layout, 1));

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
        let (prover_setup_a, _) = AkitaPackedScheme::setup(packed_akita_params(&layout_a, 1));
        let (prover_setup_b, _) = AkitaPackedScheme::setup(packed_akita_params(&layout_b, 1));
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
