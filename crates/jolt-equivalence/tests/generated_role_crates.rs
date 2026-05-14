use jolt_core::transcripts::{
    Blake2bTranscript as CoreBlake2bTranscript, Transcript as CoreTranscript,
};
use jolt_core::zkvm::lookup_table::LookupTables as CoreLookupTables;
use jolt_equivalence::core_oracle::core_lookup_table_to_modular_index;
use jolt_field::Fr;
use jolt_lookup_tables::LookupTableKind;
use jolt_transcript::{Blake2bTranscript, Transcript as GeneratedTranscript};
use strum::EnumCount;

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
    macro_rules! assert_stage_drivers {
        ($(($prover:expr, $verifier:expr)),+ $(,)?) => {
            $(
            assert!(!$prover.drivers.is_empty());
            assert!(!$verifier.drivers.is_empty());
            )+
        };
    }
    assert_stage_drivers!(
        (
            jolt_prover::stages::stage1_outer::STAGE1_PROGRAM,
            jolt_verifier::stages::stage1_outer::STAGE1_PROGRAM
        ),
        (
            jolt_prover::stages::stage2::STAGE2_PROGRAM,
            jolt_verifier::stages::stage2::STAGE2_PROGRAM
        ),
        (
            jolt_prover::stages::stage3::STAGE3_PROGRAM,
            jolt_verifier::stages::stage3::STAGE3_PROGRAM
        ),
        (
            jolt_prover::stages::stage4::STAGE4_PROGRAM,
            jolt_verifier::stages::stage4::STAGE4_PROGRAM
        ),
        (
            jolt_prover::stages::stage5::STAGE5_PROGRAM,
            jolt_verifier::stages::stage5::STAGE5_PROGRAM
        ),
        (
            jolt_prover::stages::stage6::STAGE6_PROGRAM,
            jolt_verifier::stages::stage6::STAGE6_PROGRAM
        ),
        (
            jolt_prover::stages::stage7::STAGE7_PROGRAM,
            jolt_verifier::stages::stage7::STAGE7_PROGRAM
        ),
    );
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

    let core_challenge: ark_bn254::Fr = core.challenge_scalar_optimized::<ark_bn254::Fr>().into();
    let generated_challenge: ark_bn254::Fr = generated.challenge_optimized().into();

    assert_eq!(core_challenge, generated_challenge);
    assert_eq!(core.state, *generated.state());
}

#[test]
fn modular_lookup_table_list_matches_generated_order() {
    const XLEN: usize = 64;

    let modular_tables = LookupTableKind::<XLEN>::all();
    assert_eq!(modular_tables.len(), CoreLookupTables::<XLEN>::COUNT + 1);
    assert_eq!(
        modular_tables.len(),
        <LookupTableKind<XLEN> as EnumCount>::COUNT
    );

    for (index, modular_table) in modular_tables.into_iter().enumerate() {
        assert_eq!(
            LookupTableKind::index(&modular_table),
            index,
            "modular table index at position {index}"
        );
    }
}

#[test]
fn core_lookup_table_indices_translate_to_modular_order() {
    const XLEN: usize = 64;

    macro_rules! assert_mapping {
        ($core:expr, $modular:expr) => {
            assert_eq!(
                core_lookup_table_to_modular_index::<XLEN>(&$core),
                $modular.index()
            );
        };
    }

    assert_mapping!(
        CoreLookupTables::<XLEN>::RangeCheck(Default::default()),
        LookupTableKind::<XLEN>::RangeCheck(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::RangeCheckAligned(Default::default()),
        LookupTableKind::<XLEN>::RangeCheckAligned(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::And(Default::default()),
        LookupTableKind::<XLEN>::And(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::Andn(Default::default()),
        LookupTableKind::<XLEN>::Andn(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::Or(Default::default()),
        LookupTableKind::<XLEN>::Or(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::Xor(Default::default()),
        LookupTableKind::<XLEN>::Xor(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::Equal(Default::default()),
        LookupTableKind::<XLEN>::Equal(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::SignedGreaterThanEqual(Default::default()),
        LookupTableKind::<XLEN>::SignedGreaterThanEqual(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::UnsignedGreaterThanEqual(Default::default()),
        LookupTableKind::<XLEN>::UnsignedGreaterThanEqual(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::NotEqual(Default::default()),
        LookupTableKind::<XLEN>::NotEqual(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::SignedLessThan(Default::default()),
        LookupTableKind::<XLEN>::SignedLessThan(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::UnsignedLessThan(Default::default()),
        LookupTableKind::<XLEN>::UnsignedLessThan(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::Movsign(Default::default()),
        LookupTableKind::<XLEN>::SignMask(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::UpperWord(Default::default()),
        LookupTableKind::<XLEN>::UpperWord(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::LessThanEqual(Default::default()),
        LookupTableKind::<XLEN>::UnsignedLessThanEqual(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::ValidUnsignedRemainder(Default::default()),
        LookupTableKind::<XLEN>::ValidUnsignedRemainder(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::ValidDiv0(Default::default()),
        LookupTableKind::<XLEN>::ValidDiv0(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::HalfwordAlignment(Default::default()),
        LookupTableKind::<XLEN>::HalfwordAlignment(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::WordAlignment(Default::default()),
        LookupTableKind::<XLEN>::WordAlignment(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::LowerHalfWord(Default::default()),
        LookupTableKind::<XLEN>::LowerHalfWord(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::SignExtendHalfWord(Default::default()),
        LookupTableKind::<XLEN>::SignExtendHalfWord(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::Pow2(Default::default()),
        LookupTableKind::<XLEN>::Pow2(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::Pow2W(Default::default()),
        LookupTableKind::<XLEN>::Pow2W(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::ShiftRightBitmask(Default::default()),
        LookupTableKind::<XLEN>::ShiftRightBitmask(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::VirtualRev8W(Default::default()),
        LookupTableKind::<XLEN>::VirtualRev8W(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::VirtualSRL(Default::default()),
        LookupTableKind::<XLEN>::VirtualSRL(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::VirtualSRA(Default::default()),
        LookupTableKind::<XLEN>::VirtualSRA(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::VirtualROTR(Default::default()),
        LookupTableKind::<XLEN>::VirtualROTR(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::VirtualROTRW(Default::default()),
        LookupTableKind::<XLEN>::VirtualROTRW(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::VirtualChangeDivisor(Default::default()),
        LookupTableKind::<XLEN>::VirtualChangeDivisor(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::VirtualChangeDivisorW(Default::default()),
        LookupTableKind::<XLEN>::VirtualChangeDivisorW(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::MulUNoOverflow(Default::default()),
        LookupTableKind::<XLEN>::MulUNoOverflow(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::VirtualXORROT32(Default::default()),
        LookupTableKind::<XLEN>::VirtualXORROT32(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::VirtualXORROT24(Default::default()),
        LookupTableKind::<XLEN>::VirtualXORROT24(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::VirtualXORROT16(Default::default()),
        LookupTableKind::<XLEN>::VirtualXORROT16(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::VirtualXORROT63(Default::default()),
        LookupTableKind::<XLEN>::VirtualXORROT63(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::VirtualXORROTW16(Default::default()),
        LookupTableKind::<XLEN>::VirtualXORROTW16(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::VirtualXORROTW12(Default::default()),
        LookupTableKind::<XLEN>::VirtualXORROTW12(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::VirtualXORROTW8(Default::default()),
        LookupTableKind::<XLEN>::VirtualXORROTW8(Default::default())
    );
    assert_mapping!(
        CoreLookupTables::<XLEN>::VirtualXORROTW7(Default::default()),
        LookupTableKind::<XLEN>::VirtualXORROTW7(Default::default())
    );
}
