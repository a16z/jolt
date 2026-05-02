#[test]
fn generated_role_crates_expose_matching_stage_prefix() {
    let prover_stages = jolt_prover::generated_stage_names().collect::<Vec<_>>();
    let verifier_stages = jolt_verifier::generated_stage_names().collect::<Vec<_>>();

    assert_eq!(prover_stages, verifier_stages);
    assert_eq!(
        prover_stages,
        vec!["commitment", "stage1_outer", "stage2", "stage3"]
    );

    assert!(!jolt_prover::stages::commitment::TRANSCRIPT_PLAN.is_empty());
    assert!(!jolt_verifier::stages::commitment::TRANSCRIPT_PLAN.is_empty());
    assert!(!jolt_prover::stages::stage1_outer::STAGE1_PROGRAM
        .drivers
        .is_empty());
    assert!(!jolt_verifier::stages::stage1_outer::STAGE1_PROGRAM
        .drivers
        .is_empty());
    assert!(!jolt_prover::stages::stage2::STAGE2_PROGRAM
        .drivers
        .is_empty());
    assert!(!jolt_verifier::stages::stage2::STAGE2_PROGRAM
        .drivers
        .is_empty());
    assert!(!jolt_prover::stages::stage3::STAGE3_PROGRAM
        .drivers
        .is_empty());
    assert!(!jolt_verifier::stages::stage3::STAGE3_PROGRAM
        .drivers
        .is_empty());
    let _proof = jolt_verifier::JoltProof {
        commitments: Vec::new(),
        stage1_outer: jolt_verifier::JoltStageProof::default(),
        stage2: jolt_verifier::JoltStageProof::default(),
        stage3: jolt_verifier::JoltStageProof::default(),
    };
}
