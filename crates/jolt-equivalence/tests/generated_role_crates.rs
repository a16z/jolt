use jolt_core::transcripts::{
    Blake2bTranscript as CoreBlake2bTranscript, Transcript as CoreTranscript,
};
use jolt_core::zkvm::lookup_table::LookupTables as CoreLookupTables;
use jolt_field::Fr;
use jolt_lookup_tables::LookupTableKind;
use jolt_transcript::{Blake2bTranscript, Transcript as GeneratedTranscript};
use strum::{EnumCount, IntoEnumIterator};

#[test]
fn generated_role_crates_expose_matching_stage_prefix() {
    let prover_stages = jolt_prover::generated_stage_names().collect::<Vec<_>>();
    let verifier_stages = jolt_verifier::generated_stage_names().collect::<Vec<_>>();

    assert_eq!(prover_stages, verifier_stages);
    assert_eq!(
        prover_stages,
        vec![
            "commitment",
            "stage1_outer",
            "stage2",
            "stage3",
            "stage4",
            "stage5",
            "stage6",
            "stage7",
            "stage8"
        ]
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
    assert!(!jolt_prover::stages::stage4::STAGE4_PROGRAM
        .drivers
        .is_empty());
    assert!(!jolt_verifier::stages::stage4::STAGE4_PROGRAM
        .drivers
        .is_empty());
    assert!(!jolt_prover::stages::stage5::STAGE5_PROGRAM
        .drivers
        .is_empty());
    assert!(!jolt_verifier::stages::stage5::STAGE5_PROGRAM
        .drivers
        .is_empty());
    assert!(!jolt_prover::stages::stage6::STAGE6_PROGRAM
        .drivers
        .is_empty());
    assert!(!jolt_verifier::stages::stage6::STAGE6_PROGRAM
        .drivers
        .is_empty());
    assert!(!jolt_prover::stages::stage7::STAGE7_PROGRAM
        .drivers
        .is_empty());
    assert!(!jolt_verifier::stages::stage7::STAGE7_PROGRAM
        .drivers
        .is_empty());
    assert!(!jolt_prover::stages::stage8::STAGE8_PROGRAM
        .opening_claims
        .is_empty());
    assert!(!jolt_verifier::stages::stage8::STAGE8_PROGRAM
        .opening_claims
        .is_empty());
    assert_eq!(
        jolt_prover::stages::stage8::STAGE8_PROGRAM.pcs_proof.mode,
        "open"
    );
    assert_eq!(
        jolt_verifier::stages::stage8::STAGE8_PROGRAM.pcs_proof.mode,
        "verify"
    );
    let _proof = jolt_verifier::JoltProof {
        commitments: Vec::new(),
        stage1_outer: jolt_verifier::JoltStageProof::default(),
        stage2: jolt_verifier::JoltStageProof::default(),
        stage3: jolt_verifier::JoltStageProof::default(),
        stage4: jolt_verifier::JoltStageProof::default(),
        stage5: jolt_verifier::JoltStageProof::default(),
        stage6: jolt_verifier::JoltStageProof::default(),
        stage7: jolt_verifier::JoltStageProof::default(),
        evaluation: None,
    };
}

#[test]
fn monolithic_transcript_challenges_match_core_full_field_path() {
    let mut core = <CoreBlake2bTranscript as CoreTranscript>::new(b"Jolt");
    let mut generated = <Blake2bTranscript<Fr> as GeneratedTranscript>::new(b"Jolt");

    let core_challenge: ark_bn254::Fr = core.challenge_scalar_optimized::<ark_bn254::Fr>();
    let generated_challenge: ark_bn254::Fr = generated.challenge().into();

    assert_eq!(core_challenge, generated_challenge);
    assert_eq!(core.state_history[1], *generated.state());
}

#[test]
fn modular_lookup_table_list_matches_core_order() {
    const XLEN: usize = 64;

    let modular_tables = LookupTableKind::<XLEN>::all();
    assert_eq!(modular_tables.len(), CoreLookupTables::<XLEN>::COUNT);
    assert_eq!(
        modular_tables.len(),
        <LookupTableKind<XLEN> as EnumCount>::COUNT
    );

    for (index, (core_table, modular_table)) in CoreLookupTables::<XLEN>::iter()
        .zip(modular_tables)
        .enumerate()
    {
        let core_name: &'static str = core_table.into();
        assert_eq!(
            modular_table.name(),
            core_name,
            "table name at index {index}"
        );
        assert_eq!(
            LookupTableKind::index(&modular_table),
            index,
            "modular table index for {core_name}"
        );
        assert_eq!(
            CoreLookupTables::enum_index(&core_table),
            index,
            "core table index for {core_name}"
        );
    }
}
