use super::support::*;

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
