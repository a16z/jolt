use super::support::*;

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
