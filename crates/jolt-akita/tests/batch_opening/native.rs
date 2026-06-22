use super::support::*;

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
