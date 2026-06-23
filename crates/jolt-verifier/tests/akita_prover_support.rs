#![cfg(feature = "akita")]
#![expect(
    clippy::expect_used,
    reason = "integration tests assert successful artifact construction"
)]

use common::jolt_device::JoltDevice;
use jolt_akita::{AkitaField, AkitaScheme, AkitaSetupParams};
use jolt_claims::protocols::jolt::{
    formulas::dimensions::REGISTER_ADDRESS_BITS, lattice_packed_validity_digest,
};
#[cfg(feature = "field-inline")]
use jolt_field::FixedByteSize;
use jolt_openings::{
    CommitmentScheme, PackingAdviceKind, PackingAlphabet, PackingCellAddress, PackingFactDomain,
    PackingFamilyId, PackingFamilySpec, PackingProverSetup, PackingVerifierSetup,
    PackingWitnessLayout, PackingWitnessSource, SparsePackingWitness,
};
use jolt_riscv::{
    CapturedState, JoltInstructionKind, JoltInstructionRow, JoltTraceRow, NonMemoryState,
    NormalizedOperands, StoreState,
};
use jolt_verifier::{
    akita::{
        build_akita_packing_jolt_witness, commit_akita_packing_jolt_witness,
        commit_akita_packing_witness, commit_akita_packing_witness_with_config, verify_akita_clear,
        AkitaJoltProof, AkitaPackingBatchProof, AkitaPackingJoltWitnessInput,
        AkitaPackingProverSetup, AkitaPackingValidityProofArtifacts, AkitaPackingVerifierSetup,
        AkitaPackingWitnessArtifacts, AkitaVerifierPreprocessing,
    },
    stages::stage8::{
        lattice_protocol_config_for_packed_witness_layout,
        lattice_validity_requirements_for_packed_witness_layout,
    },
    IncrementCommitmentMode, JoltProtocolConfig, ProgramMode, VerifierError,
};

fn tiny_layout() -> PackingWitnessLayout {
    let specs = vec![
        PackingFamilySpec::direct(
            PackingFamilyId::InstructionRa { index: 0 },
            PackingFactDomain::TraceRows { log_t: 0 },
            1,
            PackingAlphabet::Byte,
        ),
        PackingFamilySpec::direct(
            PackingFamilyId::UnsignedIncMsb,
            PackingFactDomain::TraceRows { log_t: 0 },
            1,
            PackingAlphabet::Bit,
        ),
    ];
    #[cfg(feature = "field-inline")]
    let specs = {
        let mut specs = specs;
        specs.extend((0..AkitaField::NUM_BYTES).map(|index| {
            PackingFamilySpec::direct(
                PackingFamilyId::FieldRdIncByte { index },
                PackingFactDomain::TraceRows { log_t: 0 },
                1,
                PackingAlphabet::Byte,
            )
        }));
        specs
    };
    PackingWitnessLayout::new(specs).expect("layout should build")
}

fn packed_cell_at(
    family: PackingFamilyId,
    row: usize,
    limb: usize,
    symbol: usize,
) -> PackingCellAddress {
    PackingCellAddress {
        family,
        row,
        limb,
        symbol,
    }
}

fn instruction(
    kind: JoltInstructionKind,
    address: usize,
    operands: NormalizedOperands,
) -> JoltInstructionRow {
    JoltInstructionRow {
        instruction_kind: kind,
        address,
        operands,
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: false,
    }
}

fn trace_row(
    kind: JoltInstructionKind,
    operands: NormalizedOperands,
    state: CapturedState,
    bytecode_pc: u32,
) -> JoltTraceRow {
    JoltTraceRow::from_components(
        state,
        &instruction(kind, 0x8000_0000 + (bytecode_pc as usize * 4), operands),
        bytecode_pc,
    )
    .expect("trace row should build")
}

fn akita_packing_setup(
    layout: &PackingWitnessLayout,
    max_num_polys_per_commitment_group: usize,
) -> (AkitaPackingProverSetup, AkitaPackingVerifierSetup) {
    let (pcs, verifier_pcs) = AkitaScheme::setup(AkitaSetupParams::new(
        layout.dimension,
        max_num_polys_per_commitment_group,
        layout.digest,
    ));
    (
        PackingProverSetup {
            pcs,
            layout: layout.clone(),
        },
        PackingVerifierSetup {
            pcs: verifier_pcs,
            layout: layout.clone(),
        },
    )
}

#[test]
fn protocol_config_binds_layout_digest_and_dimension() {
    let layout = tiny_layout();

    let config = lattice_protocol_config_for_packed_witness_layout(&layout);

    assert_eq!(
        config.lattice.packed_witness.layout_digest,
        Some(layout.digest)
    );
    assert_eq!(config.lattice.packed_witness.d_pack, Some(layout.dimension));
    assert_eq!(
        config.lattice.packed_witness.validity_digest,
        Some(lattice_packed_validity_digest(
            &lattice_validity_requirements_for_packed_witness_layout(&layout)
        ))
    );
    assert_eq!(config.lattice.program_mode, ProgramMode::Committed);
    assert_eq!(
        config.lattice.increment_mode,
        IncrementCommitmentMode::FusedOneHot
    );
}

#[test]
fn commits_packed_witness_and_returns_verifier_payload() {
    let layout = tiny_layout();
    let (prover_setup, _) = akita_packing_setup(&layout, 1);
    let source = SparsePackingWitness::try_new(
        layout.clone(),
        vec![(0, AkitaField::from_u64(1)), (256, AkitaField::from_u64(1))],
    )
    .expect("source should build");

    let artifact =
        commit_akita_packing_witness(&prover_setup, &source).expect("packed witness should commit");

    assert_eq!(artifact.layout, layout);
    let payload = artifact
        .payload()
        .expect("artifact should carry lattice payload");
    assert_eq!(payload.layout_digest, layout.digest);
    assert_eq!(payload.d_pack, layout.dimension);
    assert_eq!(payload.packed_witness.layout_digest, layout.digest);
    assert_eq!(payload.packed_witness.num_vars, layout.dimension);
    assert_eq!(
        artifact.protocol.lattice.packed_witness.layout_digest,
        Some(layout.digest)
    );
}

#[test]
fn commits_jolt_packed_witness_inputs_with_padding() {
    let specs = vec![
        PackingFamilySpec::direct(
            PackingFamilyId::InstructionRa { index: 0 },
            PackingFactDomain::TraceRows { log_t: 1 },
            1,
            PackingAlphabet::Byte,
        ),
        PackingFamilySpec::direct(
            PackingFamilyId::BytecodeRa { index: 0 },
            PackingFactDomain::TraceRows { log_t: 1 },
            1,
            PackingAlphabet::Byte,
        ),
        PackingFamilySpec::direct(
            PackingFamilyId::RamRa { index: 0 },
            PackingFactDomain::TraceRows { log_t: 1 },
            1,
            PackingAlphabet::Byte,
        ),
        PackingFamilySpec::direct(
            PackingFamilyId::UnsignedIncChunk { index: 0 },
            PackingFactDomain::TraceRows { log_t: 1 },
            1,
            PackingAlphabet::Byte,
        ),
        PackingFamilySpec::direct(
            PackingFamilyId::AdviceBytes {
                kind: PackingAdviceKind::Untrusted,
                index: 0,
            },
            PackingFactDomain::AdviceBytes {
                kind: PackingAdviceKind::Untrusted,
                log_bytes: 2,
            },
            1,
            PackingAlphabet::Byte,
        ),
    ];
    #[cfg(feature = "field-inline")]
    let specs = {
        let mut specs = specs;
        specs.extend((0..AkitaField::NUM_BYTES).map(|index| {
            PackingFamilySpec::direct(
                PackingFamilyId::FieldRdIncByte { index },
                PackingFactDomain::TraceRows { log_t: 1 },
                1,
                PackingAlphabet::Byte,
            )
        }));
        specs
    };
    let layout = PackingWitnessLayout::new(specs).expect("layout should build");
    let rows = [
        trace_row(
            JoltInstructionKind::ADD,
            NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: Some(3),
                imm: 0,
            },
            CapturedState::NonMemory(NonMemoryState {
                rs1_value: 1,
                rs2_value: 2,
                rd_pre_value: 4,
                rd_write_value: 7,
            }),
            0,
        ),
        trace_row(
            JoltInstructionKind::SD,
            NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: None,
                imm: 8,
            },
            CapturedState::Store(StoreState {
                rs1_value: 1,
                rs2_value: 11,
                ram_read_value: 10,
                ram_address: 0x34,
            }),
            1,
        ),
    ];
    let (prover_setup, _) = akita_packing_setup(&layout, 1);

    let committed = commit_akita_packing_jolt_witness(
        &prover_setup,
        AkitaPackingJoltWitnessInput {
            layout: layout.clone(),
            trace_rows: &rows,
            log_k_chunk: 8,
            instruction_lookup_indices: &[0xaa, 0xbb],
            untrusted_advice: Some(&[7, 8]),
        },
    )
    .expect("Jolt packed witness should build and commit");

    assert_eq!(committed.artifacts.layout, layout);
    let payload = committed
        .artifacts
        .payload()
        .expect("artifact should carry lattice payload");
    assert_eq!(payload.layout_digest, layout.digest);

    let witness = &committed.witness;
    assert_eq!(
        witness
            .eval_direct_fact(&packed_cell_at(
                PackingFamilyId::InstructionRa { index: 0 },
                0,
                0,
                0xaa,
            ))
            .expect("instruction RA cell should exist"),
        AkitaField::one()
    );
    assert_eq!(
        witness
            .eval_direct_fact(&packed_cell_at(
                PackingFamilyId::BytecodeRa { index: 0 },
                1,
                0,
                1,
            ))
            .expect("bytecode RA cell should exist"),
        AkitaField::one()
    );
    assert_eq!(
        witness
            .eval_direct_fact(&packed_cell_at(
                PackingFamilyId::RamRa { index: 0 },
                1,
                0,
                0x34
            ))
            .expect("RAM RA cell should exist"),
        AkitaField::one()
    );
    assert_eq!(
        witness
            .eval_direct_fact(&packed_cell_at(
                PackingFamilyId::UnsignedIncChunk { index: 0 },
                0,
                0,
                3,
            ))
            .expect("increment cell should exist"),
        AkitaField::one()
    );
    assert_eq!(
        witness
            .eval_direct_fact(&packed_cell_at(
                PackingFamilyId::AdviceBytes {
                    kind: PackingAdviceKind::Untrusted,
                    index: 0,
                },
                2,
                0,
                0,
            ))
            .expect("padded untrusted advice byte should exist"),
        AkitaField::one()
    );
}

#[test]
fn build_jolt_packed_witness_rejects_precommitted_layout_families() {
    let forbidden_specs = [
        PackingFamilySpec::direct(
            PackingFamilyId::AdviceBytes {
                kind: PackingAdviceKind::Trusted,
                index: 0,
            },
            PackingFactDomain::AdviceBytes {
                kind: PackingAdviceKind::Trusted,
                log_bytes: 0,
            },
            1,
            PackingAlphabet::Byte,
        ),
        PackingFamilySpec::direct(
            PackingFamilyId::BytecodeChunk { index: 0 },
            PackingFactDomain::BytecodeRows { log_bytecode: 0 },
            1,
            PackingAlphabet::Byte,
        ),
        PackingFamilySpec::direct(
            PackingFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 2,
            },
            PackingFactDomain::BytecodeRows { log_bytecode: 0 },
            1,
            PackingAlphabet::Fixed {
                size: 1 << REGISTER_ADDRESS_BITS,
            },
        ),
        PackingFamilySpec::direct(
            PackingFamilyId::ProgramImageInit,
            PackingFactDomain::ProgramImageWords { log_words: 0 },
            8,
            PackingAlphabet::Byte,
        ),
    ];

    for spec in forbidden_specs {
        let layout =
            PackingWitnessLayout::new([spec]).expect("forbidden layout should still parse");

        let error = build_akita_packing_jolt_witness(AkitaPackingJoltWitnessInput {
            layout,
            trace_rows: &[],
            log_k_chunk: 8,
            instruction_lookup_indices: &[],
            untrusted_advice: None,
        })
        .expect_err("precommitted packed-witness layout should reject");

        assert!(
            matches!(
                error,
                VerifierError::InvalidProtocolConfig { ref reason }
                    if reason.contains("precommitted family")
            ),
            "unexpected error: {error:?}"
        );
    }
}

#[test]
fn configured_layout_mismatch_rejects_before_commit() {
    let layout = tiny_layout();
    let (prover_setup, _) = akita_packing_setup(&layout, 1);
    let source = SparsePackingWitness::try_new(layout.clone(), Vec::new())
        .expect("empty source should build");
    let mut config = lattice_protocol_config_for_packed_witness_layout(&layout);
    config.lattice.packed_witness.layout_digest = Some([9; 32]);

    let error = commit_akita_packing_witness_with_config(config, &prover_setup, &source)
        .expect_err("layout mismatch should reject");

    assert!(matches!(error, VerifierError::InvalidProtocolConfig { .. }));
}

#[test]
fn akita_clear_verifier_surface_is_nameable() {
    type TestTranscript = jolt_transcript::Blake2bTranscript<AkitaField>;
    type VerifyFn = fn(
        &AkitaVerifierPreprocessing,
        &JoltDevice,
        &AkitaJoltProof,
        Option<&jolt_akita::AkitaCommitment>,
        &JoltProtocolConfig,
    ) -> Result<(), VerifierError>;
    let _verify: VerifyFn = verify_akita_clear::<TestTranscript>;
    type ProveFn = fn(
        &AkitaPackingProverSetup,
        &AkitaVerifierPreprocessing,
        &JoltDevice,
        &AkitaJoltProof,
        Option<&jolt_akita::AkitaCommitment>,
        &AkitaPackingWitnessArtifacts,
        &SparsePackingWitness<AkitaField>,
    ) -> Result<AkitaPackingBatchProof, VerifierError>;
    let _prove: ProveFn = jolt_verifier::akita::prove_akita_jolt_final_openings::<
        TestTranscript,
        SparsePackingWitness<AkitaField>,
    >;
    type ProveValidityFn = fn(
        &AkitaPackingProverSetup,
        &AkitaVerifierPreprocessing,
        &JoltDevice,
        &AkitaJoltProof,
        Option<&jolt_akita::AkitaCommitment>,
        &AkitaPackingWitnessArtifacts,
        &SparsePackingWitness<AkitaField>,
    ) -> Result<AkitaPackingValidityProofArtifacts, VerifierError>;
    let _prove_validity: ProveValidityFn = jolt_verifier::akita::prove_akita_jolt_packed_validity::<
        TestTranscript,
        SparsePackingWitness<AkitaField>,
    >;
}
