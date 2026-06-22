
#![expect(
    clippy::expect_used,
    reason = "tests assert successful artifact construction"
)]

use super::*;
use crate::stages::stage8::{
    Stage8BatchStatement, Stage8ClearBatchStatement, Stage8LogicalManifest, Stage8OpeningId,
    Stage8PhysicalManifest,
};
use crate::stages::{CommittedProgramSchedule, PrecommittedSchedule};
use jolt_akita::{AkitaScheme, AkitaSetupParams, AKITA_FIELD_MODULUS};
use jolt_claims::protocols::jolt::{
    bytecode_imm_canonical_bytes_requirement,
    formulas::{
        dimensions::{TracePolynomialOrder, REGISTER_ADDRESS_BITS},
        ra::JoltRaPolynomialLayout,
    },
    unsigned_inc_msb_opening, JoltCommittedPolynomial, JoltOpeningId, JoltRelationId,
};
use jolt_field::FixedByteSize;
use jolt_openings::{
    BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme, PackedAlphabet,
    PackedCellAddress, PackedFactDomain, PackedFamilySpec, PackedLinearTerm, PhysicalView,
    SparsePackedWitness,
};
use jolt_poly::Point;
use jolt_riscv::{
    CapturedState, CircuitFlags, JoltInstructionKind, JoltInstructionRow, JoltTraceRow,
    NonMemoryState, NormalizedOperands, StoreState,
};
use jolt_transcript::{Blake2bTranscript, Transcript};

fn tiny_layout() -> PackedWitnessLayout {
    let specs = vec![
        PackedFamilySpec::direct(
            PackedFamilyId::InstructionRa { index: 0 },
            PackedFactDomain::TraceRows { log_t: 0 },
            1,
            PackedAlphabet::Byte,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::UnsignedIncMsb,
            PackedFactDomain::TraceRows { log_t: 0 },
            1,
            PackedAlphabet::Bit,
        ),
    ];
    #[cfg(feature = "field-inline")]
    let specs = {
        let mut specs = specs;
        specs.extend((0..AkitaField::NUM_BYTES).map(|index| {
            PackedFamilySpec::direct(
                PackedFamilyId::FieldRdIncByte { index },
                PackedFactDomain::TraceRows { log_t: 0 },
                1,
                PackedAlphabet::Byte,
            )
        }));
        specs
    };
    PackedWitnessLayout::new(specs).expect("layout should build")
}

fn packed_cell(family: PackedFamilyId, symbol: usize) -> PackedCellAddress {
    packed_cell_at(family, 0, 0, symbol)
}

fn packed_cell_at(
    family: PackedFamilyId,
    row: usize,
    limb: usize,
    symbol: usize,
) -> PackedCellAddress {
    PackedCellAddress {
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

fn af(value: u64) -> AkitaField {
    AkitaField::from_u64(value)
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
fn protocol_config_binds_layout_digest_and_dimension() {
    let layout = tiny_layout();

    let config = akita_lattice_protocol_config_for_layout(&layout);

    assert_eq!(
        config.lattice.packed_witness.layout_digest,
        Some(layout.digest)
    );
    assert_eq!(config.lattice.packed_witness.d_pack, Some(layout.dimension));
    assert_eq!(
        config.lattice.packed_witness.validity_digest,
        Some(lattice_packed_validity_digest(
            &akita_lattice_validity_requirements_for_layout(&layout)
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
    let params = AkitaSetupParams::from_packed_layout(&layout, 1);
    let (prover_setup, _) = AkitaPackedScheme::setup(params);
    let source = SparsePackedWitness::try_new(
        layout.clone(),
        vec![(0, AkitaField::from_u64(1)), (256, AkitaField::from_u64(1))],
    )
    .expect("source should build");

    let artifact =
        commit_akita_packed_witness(&prover_setup, &source).expect("packed witness should commit");

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
        PackedFamilySpec::direct(
            PackedFamilyId::InstructionRa { index: 0 },
            PackedFactDomain::TraceRows { log_t: 1 },
            1,
            PackedAlphabet::Byte,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::BytecodeRa { index: 0 },
            PackedFactDomain::TraceRows { log_t: 1 },
            1,
            PackedAlphabet::Byte,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::RamRa { index: 0 },
            PackedFactDomain::TraceRows { log_t: 1 },
            1,
            PackedAlphabet::Byte,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::UnsignedIncChunk { index: 0 },
            PackedFactDomain::TraceRows { log_t: 1 },
            1,
            PackedAlphabet::Byte,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::AdviceBytes {
                kind: PackedAdviceKind::Untrusted,
                index: 0,
            },
            PackedFactDomain::AdviceBytes {
                kind: PackedAdviceKind::Untrusted,
                log_bytes: 2,
            },
            1,
            PackedAlphabet::Byte,
        ),
    ];
    #[cfg(feature = "field-inline")]
    let specs = {
        let mut specs = specs;
        specs.extend((0..AkitaField::NUM_BYTES).map(|index| {
            PackedFamilySpec::direct(
                PackedFamilyId::FieldRdIncByte { index },
                PackedFactDomain::TraceRows { log_t: 1 },
                1,
                PackedAlphabet::Byte,
            )
        }));
        specs
    };
    let layout = PackedWitnessLayout::new(specs).expect("layout should build");
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
    let params = AkitaSetupParams::from_packed_layout(&layout, 1);
    let (prover_setup, _) = AkitaPackedScheme::setup(params);

    let committed = commit_akita_packed_jolt_witness(
        &prover_setup,
        AkitaPackedJoltWitnessInput {
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
                PackedFamilyId::InstructionRa { index: 0 },
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
                PackedFamilyId::BytecodeRa { index: 0 },
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
                PackedFamilyId::RamRa { index: 0 },
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
                PackedFamilyId::UnsignedIncChunk { index: 0 },
                0,
                0,
                3
            ))
            .expect("increment cell should exist"),
        AkitaField::one()
    );
    assert_eq!(
        witness
            .eval_direct_fact(&packed_cell_at(
                PackedFamilyId::AdviceBytes {
                    kind: PackedAdviceKind::Untrusted,
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
        PackedFamilySpec::direct(
            PackedFamilyId::AdviceBytes {
                kind: PackedAdviceKind::Trusted,
                index: 0,
            },
            PackedFactDomain::AdviceBytes {
                kind: PackedAdviceKind::Trusted,
                log_bytes: 0,
            },
            1,
            PackedAlphabet::Byte,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::BytecodeChunk { index: 0 },
            PackedFactDomain::BytecodeRows { log_bytecode: 0 },
            1,
            PackedAlphabet::Byte,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 2,
            },
            PackedFactDomain::BytecodeRows { log_bytecode: 0 },
            1,
            PackedAlphabet::Fixed {
                size: 1 << REGISTER_ADDRESS_BITS,
            },
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::ProgramImageInit,
            PackedFactDomain::ProgramImageWords { log_words: 0 },
            8,
            PackedAlphabet::Byte,
        ),
    ];

    for spec in forbidden_specs {
        let layout = PackedWitnessLayout::new([spec]).expect("forbidden layout should still parse");

        let error = build_akita_packed_jolt_witness(AkitaPackedJoltWitnessInput {
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
fn packed_witness_artifacts_feed_akita_packed_batch_verifier() {
    let layout = tiny_layout();
    let params = AkitaSetupParams::from_packed_layout(&layout, 1);
    let (prover_setup, verifier_setup) = AkitaPackedScheme::setup(params);
    let instruction_family = PackedFamilyId::InstructionRa { index: 0 };
    let sign_family = PackedFamilyId::UnsignedIncMsb;
    let source = SparsePackedWitness::try_from_cells(
        layout.clone(),
        [
            (
                packed_cell(instruction_family.clone(), 7),
                AkitaField::one(),
            ),
            (packed_cell(sign_family.clone(), 1), AkitaField::one()),
        ],
    )
    .expect("source should build");
    let artifact =
        commit_akita_packed_witness(&prover_setup, &source).expect("packed witness should commit");
    let commitment = artifact
        .payload()
        .expect("artifact should carry lattice payload")
        .packed_witness
        .clone();
    let instruction_claim = AkitaField::from_u64(2);
    let sign_claim = AkitaField::from_u64(3);
    let instruction_id = Stage8OpeningId::from(JoltOpeningId::committed(
        JoltCommittedPolynomial::InstructionRa(0),
        JoltRelationId::HammingWeightClaimReduction,
    ));
    let sign_id = Stage8OpeningId::from(unsigned_inc_msb_opening());
    let statement = BatchOpeningStatement {
        logical_point: Vec::new(),
        pcs_point: Vec::new(),
        layout_digest: layout.digest,
        claims: vec![
            BatchOpeningClaim {
                id: instruction_id,
                relation: instruction_id,
                commitment: commitment.clone(),
                claim: instruction_claim,
                view: PhysicalView::PackedLinear {
                    layout_digest: layout.digest,
                    terms: vec![PackedLinearTerm::new(
                        AkitaField::from_u64(2),
                        instruction_family.physical_ref(),
                        0,
                        7,
                    )
                    .with_row_point(Vec::new())],
                },
                scale: AkitaField::from_u64(3),
            },
            BatchOpeningClaim {
                id: sign_id,
                relation: sign_id,
                commitment: commitment.clone(),
                claim: sign_claim,
                view: PhysicalView::PackedLinear {
                    layout_digest: layout.digest,
                    terms: vec![PackedLinearTerm::new(
                        AkitaField::from_u64(3),
                        sign_family.physical_ref(),
                        0,
                        1,
                    )
                    .with_row_point(Vec::new())],
                },
                scale: AkitaField::from_u64(7),
            },
        ],
    };
    let stage8_statement = Stage8BatchStatement::Clear(Stage8ClearBatchStatement {
        logical_manifest: Stage8LogicalManifest {
            openings: Vec::new(),
            pcs_opening_point: Point::high_to_low(Vec::<AkitaField>::new()),
        },
        physical_manifest: Stage8PhysicalManifest {
            openings: Vec::new(),
            layout_digest: layout.digest,
        },
        opening_ids: vec![instruction_id, sign_id],
        opening_claims: Vec::new(),
        pcs_opening_point: Point::high_to_low(Vec::<AkitaField>::new()),
        statement: statement.clone(),
        precommitted_statements: Vec::new(),
    });

    let mut prover_transcript = Blake2bTranscript::new(b"verifier-akita-packed");
    let proof = prove_akita_stage8_clear_openings(
        &prover_setup,
        &mut prover_transcript,
        &artifact,
        &source,
        &stage8_statement,
    )
    .expect("packed batch proof should be produced");
    validate_akita_opening_proof_payload_shape(&artifact.commitments, &proof)
        .expect("fresh packed batch proof shape should pass preflight");

    let mut wrong_stage8_statement = stage8_statement.clone();
    let Stage8BatchStatement::Clear(wrong_statement) = &mut wrong_stage8_statement else {
        unreachable!("test statement is clear");
    };
    wrong_statement.statement.claims[0].commitment.layout_digest = [9; 32];
    let mut wrong_transcript = Blake2bTranscript::new(b"verifier-akita-packed");
    let error = prove_akita_stage8_clear_openings(
        &prover_setup,
        &mut wrong_transcript,
        &artifact,
        &source,
        &wrong_stage8_statement,
    )
    .expect_err("non-artifact commitment should reject");
    assert!(matches!(
        error,
        VerifierError::FinalOpeningBatchFailed { .. }
    ));

    let mut wrong_commitment_proof = proof.clone();
    wrong_commitment_proof.native.commitment.layout_digest = [9; 32];
    assert!(matches!(
        validate_akita_opening_proof_payload_shape(
            &artifact.commitments,
            &wrong_commitment_proof,
        ),
        Err(VerifierError::InvalidProtocolConfig { reason })
            if reason.contains("opening proof commitment")
    ));

    let mut missing_native_proof = proof.clone();
    missing_native_proof.native.proof.clear();
    assert!(matches!(
        validate_akita_opening_proof_payload_shape(&artifact.commitments, &missing_native_proof),
        Err(VerifierError::InvalidProtocolConfig { reason })
            if reason.contains("native proof bytes")
    ));

    let mut missing_reduction = proof.clone();
    missing_reduction.reduction = None;
    assert!(matches!(
        validate_akita_packed_opening_proof_payload_shape(
            &artifact.commitments,
            &missing_reduction,
            "Akita joint opening proof",
        ),
        Err(VerifierError::InvalidProtocolConfig { reason })
            if reason.contains("packed reduction")
    ));

    let mut missing_reduction_eval = proof.clone();
    missing_reduction_eval
        .reduction
        .as_mut()
        .expect("packed proof should contain a reduction")
        .opening_eval
        .clear();
    assert!(matches!(
        validate_akita_opening_proof_payload_shape(
            &artifact.commitments,
            &missing_reduction_eval,
        ),
        Err(VerifierError::InvalidProtocolConfig { reason })
            if reason.contains("packed reduction opening eval")
    ));

    let mut noncanonical_reduction_eval = proof.clone();
    noncanonical_reduction_eval
        .reduction
        .as_mut()
        .expect("packed proof should contain a reduction")
        .opening_eval = AKITA_FIELD_MODULUS.to_le_bytes().to_vec();
    assert!(matches!(
        validate_akita_packed_opening_proof_payload_shape(
            &artifact.commitments,
            &noncanonical_reduction_eval,
            "Akita joint opening proof",
        ),
        Err(VerifierError::InvalidProtocolConfig { reason })
            if reason.contains("canonical Akita field encoding")
    ));

    let mut verifier_transcript = Blake2bTranscript::new(b"verifier-akita-packed");
    let result = <AkitaPackedScheme as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &mut verifier_transcript,
        &statement,
        &proof,
    )
    .expect("packed batch proof should verify");

    assert_eq!(result.joint_commitment, commitment);
    assert_eq!(result.coefficients.len(), 2);
    assert_eq!(
        result.reduced_opening,
        result.coefficients[0] * instruction_claim + result.coefficients[1] * sign_claim
    );
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn stage8_clear_openings_prove_separate_precommitted_batches() {
    let layout = tiny_layout();
    let params = AkitaSetupParams::from_packed_layout(&layout, 1);
    let (prover_setup, verifier_setup) = AkitaPackedScheme::setup(params);
    let sign_family = PackedFamilyId::UnsignedIncMsb;
    let source = SparsePackedWitness::try_from_cells(
        layout.clone(),
        [(packed_cell(sign_family.clone(), 1), AkitaField::one())],
    )
    .expect("source should build");
    let artifact =
        commit_akita_packed_witness(&prover_setup, &source).expect("packed witness should commit");
    let packed_commitment = artifact
        .payload()
        .expect("artifact should carry lattice payload")
        .packed_witness
        .clone();
    let sign_id = Stage8OpeningId::from(unsigned_inc_msb_opening());
    let packed_statement = BatchOpeningStatement {
        logical_point: Vec::new(),
        pcs_point: Vec::new(),
        layout_digest: layout.digest,
        claims: vec![BatchOpeningClaim {
            id: sign_id,
            relation: sign_id,
            commitment: packed_commitment.clone(),
            claim: AkitaField::from_u64(3),
            view: PhysicalView::PackedLinear {
                layout_digest: layout.digest,
                terms: vec![PackedLinearTerm::new(
                    AkitaField::from_u64(3),
                    sign_family.physical_ref(),
                    0,
                    1,
                )
                .with_row_point(Vec::new())],
            },
            scale: AkitaField::from_u64(7),
        }],
    };

    let precommitted_point = vec![AkitaField::zero(); layout.dimension];
    let mut precommitted_evals = vec![AkitaField::zero(); 1usize << layout.dimension];
    precommitted_evals[0] = AkitaField::from_u64(19);
    let precommitted_poly = Polynomial::new(precommitted_evals);
    let precommitted_digest = [11; 32];
    let (precommitted_commitment, precommitted_hint) = AkitaScheme::commit_group(
        &prover_setup,
        precommitted_digest,
        std::slice::from_ref(&precommitted_poly),
    )
    .expect("precommitted commitment should commit");
    let precommitted_id = Stage8OpeningId::from(JoltOpeningId::committed(
        JoltCommittedPolynomial::TrustedAdvice,
        JoltRelationId::AdviceClaimReduction,
    ));
    let precommitted_statement = BatchOpeningStatement {
        logical_point: precommitted_point.clone(),
        pcs_point: precommitted_point,
        layout_digest: precommitted_digest,
        claims: vec![BatchOpeningClaim {
            id: precommitted_id,
            relation: precommitted_id,
            commitment: precommitted_commitment.clone(),
            claim: AkitaField::from_u64(19),
            view: PhysicalView::Direct,
            scale: AkitaField::from_u64(2),
        }],
    };
    assert_eq!(
        precommitted_statement.layout_digest,
        precommitted_commitment.layout_digest
    );
    let stage8_statement = Stage8BatchStatement::Clear(Stage8ClearBatchStatement {
        logical_manifest: Stage8LogicalManifest {
            openings: Vec::new(),
            pcs_opening_point: Point::high_to_low(Vec::<AkitaField>::new()),
        },
        physical_manifest: Stage8PhysicalManifest {
            openings: Vec::new(),
            layout_digest: layout.digest,
        },
        opening_ids: vec![sign_id, precommitted_id],
        opening_claims: Vec::new(),
        pcs_opening_point: Point::high_to_low(Vec::<AkitaField>::new()),
        statement: packed_statement.clone(),
        precommitted_statements: vec![precommitted_statement.clone()],
    });
    let precommitted_inputs = [AkitaPrecommittedOpeningInput {
        polynomial: &precommitted_poly,
        hint: &precommitted_hint,
    }];

    let mut prover_transcript = Blake2bTranscript::new(b"verifier-akita-precommitted");
    let proofs = prove_akita_stage8_clear_openings_with_precommitted(
        &prover_setup,
        &mut prover_transcript,
        &artifact,
        &source,
        &stage8_statement,
        &precommitted_inputs,
    )
    .expect("stage8 proofs should be produced");
    assert_eq!(proofs.precommitted.len(), 1);
    validate_akita_precommitted_opening_proof_payload_shapes(
        &artifact.commitments,
        &proofs.precommitted,
    )
    .expect("fresh precommitted proof payload should pass preflight");

    let mut packed_target_precommitted_proof = proofs.precommitted[0].clone();
    packed_target_precommitted_proof.native.commitment = packed_commitment.clone();
    assert!(matches!(
        validate_akita_precommitted_opening_proof_payload_shapes(
            &artifact.commitments,
            std::slice::from_ref(&packed_target_precommitted_proof),
        ),
        Err(VerifierError::InvalidProtocolConfig { reason })
            if reason.contains("precommitted commitment")
    ));

    let mut packed_reduction_precommitted_proof = proofs.precommitted[0].clone();
    packed_reduction_precommitted_proof.reduction = Some(jolt_akita::AkitaPackedReductionProof {
        rounds: Vec::new(),
        opening_eval: vec![0; AkitaField::NUM_BYTES],
    });
    assert!(matches!(
        validate_akita_precommitted_opening_proof_payload_shapes(
            &artifact.commitments,
            std::slice::from_ref(&packed_reduction_precommitted_proof),
        ),
        Err(VerifierError::InvalidProtocolConfig { reason })
            if reason.contains("packed reduction")
    ));

    let mut packed_target_statement = stage8_statement.clone();
    let Stage8BatchStatement::Clear(clear_statement) = &mut packed_target_statement else {
        unreachable!("test statement is clear");
    };
    clear_statement.precommitted_statements[0].claims[0].commitment = packed_commitment.clone();
    let mut packed_target_transcript = Blake2bTranscript::new(b"verifier-akita-precommitted");
    let error = prove_akita_stage8_clear_openings_with_precommitted(
        &prover_setup,
        &mut packed_target_transcript,
        &artifact,
        &source,
        &packed_target_statement,
        &precommitted_inputs,
    )
    .expect_err("precommitted statement targeting W_pack should fail");
    assert!(matches!(
        error,
        VerifierError::FinalOpeningBatchFailed { reason }
            if reason.contains("separate precommitted commitment")
    ));

    let packed_hint_inputs = [AkitaPrecommittedOpeningInput {
        polynomial: &precommitted_poly,
        hint: &artifact.hint,
    }];
    let mut packed_hint_transcript = Blake2bTranscript::new(b"verifier-akita-precommitted");
    let error = prove_akita_stage8_clear_openings_with_precommitted(
        &prover_setup,
        &mut packed_hint_transcript,
        &artifact,
        &source,
        &stage8_statement,
        &packed_hint_inputs,
    )
    .expect_err("precommitted input using W_pack hint should fail");
    assert!(matches!(
        error,
        VerifierError::FinalOpeningBatchFailed { reason }
            if reason.contains("packed witness hint")
    ));

    let mut verifier_transcript = Blake2bTranscript::new(b"verifier-akita-precommitted");
    let _ = <AkitaPackedScheme as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &mut verifier_transcript,
        &packed_statement,
        &proofs.packed,
    )
    .expect("packed proof should verify");
    let _ = <AkitaPackedScheme as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &mut verifier_transcript,
        &precommitted_statement,
        &proofs.precommitted[0],
    )
    .expect("precommitted proof should verify");
    assert_eq!(prover_transcript.state(), verifier_transcript.state());

    let mut missing_input_transcript = Blake2bTranscript::new(b"verifier-akita-precommitted");
    let error = prove_akita_stage8_clear_openings(
        &prover_setup,
        &mut missing_input_transcript,
        &artifact,
        &source,
        &stage8_statement,
    )
    .expect_err("precommitted statement requires input");
    assert!(matches!(
        error,
        VerifierError::FinalOpeningBatchFailed { reason }
            if reason.contains("expected 1 Akita precommitted opening inputs")
    ));
}

#[test]
#[cfg_attr(
    feature = "field-inline",
    ignore = "field-inline canonical-byte validity makes the real Akita proof fixture expensive; run explicitly with --run-ignored"
)]
fn packed_validity_helper_proves_real_akita_opening_proof() {
    run_on_large_stack(|| {
        let log_t = 0;
        let log_k_chunk = 1;
        let precommitted = PrecommittedSchedule::new(
            TracePolynomialOrder::CycleMajor,
            log_t,
            log_k_chunk,
            None,
            None,
            Some(CommittedProgramSchedule {
                bytecode_len: 1,
                bytecode_chunk_count: 1,
                program_image_len_words: 1,
                program_image_start_index: 0,
            }),
        )
        .expect("precommitted schedule should build");
        let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
        config.lattice.program_mode = ProgramMode::Committed;
        config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
        config.lattice.packed_witness.layout_digest = Some([0; 32]);
        config.lattice.packed_witness.d_pack = Some(0);
        config.lattice.packed_witness.validity_digest = Some([0; 32]);
        #[cfg(feature = "field-inline")]
        {
            config.lattice.field_inline.enabled = true;
            config.lattice.packed_witness.field_rd_inc_family = true;
        }

        let layout = crate::stages::stage8::derive_lattice_packed_witness_layout(
            &config,
            log_t,
            log_k_chunk,
            JoltRaPolynomialLayout::new(1, 1, 1).expect("RA layout should build"),
            &precommitted,
        )
        .expect("layout should derive");
        config.lattice.packed_witness.layout_digest = Some(layout.digest);
        config.lattice.packed_witness.d_pack = Some(layout.dimension);
        let requirements =
            derive_lattice_packed_validity_requirements(&config, log_k_chunk, &precommitted)
                .expect("validity requirements should derive");
        config.lattice.packed_witness.validity_digest =
            Some(lattice_packed_validity_digest(&requirements));
        let source = validity_default_source(&layout, &requirements);
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, verifier_setup) = AkitaPackedScheme::setup(params);
        let artifacts = commit_akita_packed_witness_with_config(config, &prover_setup, &source)
            .expect("valid packed witness should commit");

        let mut prover_transcript = Blake2bTranscript::new(b"akita-validity");
        let validity = prove_akita_packed_validity(
            &prover_setup,
            &mut prover_transcript,
            &artifacts,
            &source,
            log_k_chunk,
            &precommitted,
        )
        .expect("validity proof should prove");

        let mut verifier_transcript = Blake2bTranscript::new(b"akita-validity");
        verify_validity_artifacts(
            &verifier_setup,
            &mut verifier_transcript,
            &artifacts,
            log_k_chunk,
            &precommitted,
            &validity,
        )
        .expect("validity proof should verify");
        assert_eq!(prover_transcript.state(), verifier_transcript.state());

        let mut tampered = validity.clone();
        tampered.opening_claims.opening_claims[0] += AkitaField::one();
        let mut tampered_transcript = Blake2bTranscript::new(b"akita-validity");
        let error = verify_validity_artifacts(
            &verifier_setup,
            &mut tampered_transcript,
            &artifacts,
            log_k_chunk,
            &precommitted,
            &tampered,
        )
        .expect_err("tampered validity opening claim should reject");
        assert!(matches!(
            error,
            VerifierError::LatticePackedValidityOutputMismatch
                | VerifierError::LatticePackedValidityOpeningVerificationFailed { .. }
        ));
    });
}

#[cfg(feature = "field-inline")]
#[test]
#[ignore = "real Akita negative canonical-byte proof takes over two minutes; run explicitly with --run-ignored"]
fn packed_validity_rejects_noncanonical_field_rd_inc_bytes() {
    run_on_large_stack(|| {
        let log_t = 0;
        let log_k_chunk = 1;
        let precommitted = PrecommittedSchedule::new(
            TracePolynomialOrder::CycleMajor,
            log_t,
            log_k_chunk,
            None,
            None,
            Some(CommittedProgramSchedule {
                bytecode_len: 1,
                bytecode_chunk_count: 1,
                program_image_len_words: 1,
                program_image_start_index: 0,
            }),
        )
        .expect("precommitted schedule should build");
        let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
        config.lattice.program_mode = ProgramMode::Committed;
        config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
        config.lattice.field_inline.enabled = true;
        config.lattice.packed_witness.field_rd_inc_family = true;
        config.lattice.packed_witness.layout_digest = Some([0; 32]);
        config.lattice.packed_witness.d_pack = Some(0);
        config.lattice.packed_witness.validity_digest = Some([0; 32]);

        let layout = crate::stages::stage8::derive_lattice_packed_witness_layout(
            &config,
            log_t,
            log_k_chunk,
            JoltRaPolynomialLayout::new(1, 1, 1).expect("RA layout should build"),
            &precommitted,
        )
        .expect("layout should derive");
        config.lattice.packed_witness.layout_digest = Some(layout.digest);
        config.lattice.packed_witness.d_pack = Some(layout.dimension);
        let requirements =
            derive_lattice_packed_validity_requirements(&config, log_k_chunk, &precommitted)
                .expect("validity requirements should derive");
        config.lattice.packed_witness.validity_digest =
            Some(lattice_packed_validity_digest(&requirements));

        let modulus_bytes = AKITA_FIELD_MODULUS.to_le_bytes();
        let source =
            validity_source_with_field_rd_inc_bytes(&layout, &requirements, &modulus_bytes);
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, verifier_setup) = AkitaPackedScheme::setup(params);
        let artifacts = commit_akita_packed_witness_with_config(config, &prover_setup, &source)
            .expect("packed witness should commit");

        let mut prover_transcript = Blake2bTranscript::new(b"akita-validity");
        let validity = prove_akita_packed_validity(
            &prover_setup,
            &mut prover_transcript,
            &artifacts,
            &source,
            log_k_chunk,
            &precommitted,
        )
        .expect("invalid packed witness can still produce a proof transcript");

        let mut verifier_transcript = Blake2bTranscript::new(b"akita-validity");
        let error = verify_validity_artifacts(
            &verifier_setup,
            &mut verifier_transcript,
            &artifacts,
            log_k_chunk,
            &precommitted,
            &validity,
        )
        .expect_err("noncanonical field bytes should reject");
        assert!(matches!(
            error,
            VerifierError::LatticePackedValidityOutputMismatch
                | VerifierError::LatticePackedValiditySumcheckFailed { .. }
                | VerifierError::LatticePackedValidityOpeningVerificationFailed { .. }
        ));
    });
}

#[cfg(feature = "field-inline")]
#[test]
fn packed_validity_value_detects_noncanonical_field_rd_inc_bytes() {
    let log_t = 0;
    let log_k_chunk = 1;
    let precommitted = PrecommittedSchedule::new(
        TracePolynomialOrder::CycleMajor,
        log_t,
        log_k_chunk,
        None,
        None,
        Some(CommittedProgramSchedule {
            bytecode_len: 1,
            bytecode_chunk_count: 1,
            program_image_len_words: 1,
            program_image_start_index: 0,
        }),
    )
    .expect("precommitted schedule should build");
    let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
    config.lattice.program_mode = ProgramMode::Committed;
    config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
    config.lattice.field_inline.enabled = true;
    config.lattice.packed_witness.field_rd_inc_family = true;
    config.lattice.packed_witness.layout_digest = Some([0; 32]);
    config.lattice.packed_witness.d_pack = Some(0);
    config.lattice.packed_witness.validity_digest = Some([0; 32]);

    let layout = crate::stages::stage8::derive_lattice_packed_witness_layout(
        &config,
        log_t,
        log_k_chunk,
        JoltRaPolynomialLayout::new(1, 1, 1).expect("RA layout should build"),
        &precommitted,
    )
    .expect("layout should derive");
    let requirements =
        derive_lattice_packed_validity_requirements(&config, log_k_chunk, &precommitted)
            .expect("validity requirements should derive");
    let statements = derive_lattice_packed_validity_statements(&layout, &requirements)
        .expect("validity statements should derive");
    let source = validity_source_with_field_rd_inc_bytes(
        &layout,
        &requirements,
        &AKITA_FIELD_MODULUS.to_le_bytes(),
    );
    let statement = statements
        .iter()
        .find(|statement| {
            matches!(
                statement.requirement.family,
                LatticePackedFamilyId::FieldRdIncByte { index: 0 }
            ) && statement.kind == LatticePackedValidityStatementKind::FieldElementCanonicalBytes
        })
        .expect("FieldRdInc canonical-byte statement should exist");
    let point = vec![AkitaField::zero(); statement.num_vars];
    let value =
        validity_value(&source, statement, &point, &point).expect("validity value should evaluate");

    assert_ne!(value, AkitaField::zero());
}

#[test]
fn packed_validity_value_detects_noncanonical_bytecode_imm_bytes() {
    let (layout, statements, requirements) = small_bytecode_validity_context();
    let source = validity_source_with_bytecode_imm_bytes(
        &layout,
        &requirements,
        &AKITA_FIELD_MODULUS.to_le_bytes(),
    );
    let statement = statements
        .iter()
        .find(|statement| {
            matches!(
                statement.requirement.family,
                LatticePackedFamilyId::BytecodeImmBytes { chunk: 0 }
            ) && statement.kind == LatticePackedValidityStatementKind::FieldElementCanonicalBytes
        })
        .expect("bytecode immediate canonical-byte statement should exist");
    let point = vec![AkitaField::zero(); statement.num_vars];
    let value =
        validity_value(&source, statement, &point, &point).expect("validity value should evaluate");

    assert_ne!(value, AkitaField::zero());
}

#[test]
fn packed_validity_value_detects_malformed_advice_byte_onehot() {
    let (layout, statements) = small_validity_context();
    let family = PackedFamilyId::AdviceBytes {
        kind: PackedAdviceKind::Untrusted,
        index: 0,
    };
    let source = SparsePackedWitness::try_from_cells(
        layout,
        [
            (packed_cell_at(family.clone(), 0, 0, 7), AkitaField::one()),
            (packed_cell_at(family, 0, 0, 8), AkitaField::one()),
        ],
    )
    .expect("malformed advice source should build");
    let statement = validity_statement(
        &statements,
        LatticePackedFamilyId::AdviceBytes {
            kind: JoltAdviceKind::Untrusted,
            index: 0,
        },
        LatticePackedValidityStatementKind::ExactOneHotRowSum,
    );

    assert_ne!(
        validity_value_at_zero(&source, statement),
        AkitaField::zero()
    );
}

#[test]
fn packed_validity_value_detects_malformed_bytecode_optional_selector() {
    let (layout, statements, _) = small_bytecode_validity_context();
    let family = PackedFamilyId::BytecodeRegisterSelector {
        chunk: 0,
        selector: 0,
    };
    let source = SparsePackedWitness::try_from_cells(
        layout,
        [
            (packed_cell_at(family.clone(), 0, 0, 3), AkitaField::one()),
            (packed_cell_at(family, 0, 0, 4), AkitaField::one()),
        ],
    )
    .expect("malformed bytecode selector source should build");
    let statement = validity_statement(
        &statements,
        LatticePackedFamilyId::BytecodeRegisterSelector {
            chunk: 0,
            selector: 0,
        },
        LatticePackedValidityStatementKind::OptionalOneHotRowSum,
    );

    assert_ne!(
        validity_value_at_zero(&source, statement),
        AkitaField::zero()
    );
}

#[test]
fn packed_validity_value_detects_malformed_bytecode_boolean_flag() {
    let (layout, statements, _) = small_bytecode_validity_context();
    let flag = CircuitFlags::Store as usize;
    let family = PackedFamilyId::BytecodeCircuitFlag { chunk: 0, flag };
    let source =
        SparsePackedWitness::try_from_cells(layout, [(packed_cell_at(family, 0, 0, 1), af(2))])
            .expect("malformed bytecode flag source should build");
    let statement = validity_statement(
        &statements,
        LatticePackedFamilyId::BytecodeCircuitFlag { chunk: 0, flag },
        LatticePackedValidityStatementKind::BooleanIndicator,
    );

    assert_ne!(
        validity_value_at_zero(&source, statement),
        AkitaField::zero()
    );
}

#[test]
fn packed_validity_rejects_precommitted_bytecode_layout_config() {
    let (layout, _, requirements) = small_bytecode_validity_context();
    let source = validity_source_with_bytecode_imm_bytes(
        &layout,
        &requirements,
        &AKITA_FIELD_MODULUS.to_le_bytes(),
    );
    let mut config = akita_lattice_protocol_config_for_layout(&layout);
    config.lattice.packed_witness.validity_digest =
        Some(lattice_packed_validity_digest(&requirements));
    let params = AkitaSetupParams::from_packed_layout(&layout, 1);
    let (prover_setup, _) = AkitaPackedScheme::setup(params);

    let error = commit_akita_packed_witness_with_config(config, &prover_setup, &source)
        .expect_err("precommitted bytecode families should reject");

    assert!(matches!(
        error,
        VerifierError::InvalidProtocolConfig { reason }
            if reason.contains("precommitted family")
    ));
}

fn validity_default_source(
    layout: &PackedWitnessLayout,
    requirements: &[LatticePackedValidityRequirement],
) -> SparsePackedWitness<AkitaField> {
    validity_source_with_symbols(layout, requirements, |_, _| 0)
}

fn small_validity_context() -> (PackedWitnessLayout, Vec<LatticePackedValidityStatement>) {
    let log_t = 0;
    let log_k_chunk = 1;
    let precommitted = PrecommittedSchedule::new(
        TracePolynomialOrder::CycleMajor,
        log_t,
        log_k_chunk,
        None,
        Some(1),
        Some(CommittedProgramSchedule {
            bytecode_len: 1,
            bytecode_chunk_count: 1,
            program_image_len_words: 1,
            program_image_start_index: 0,
        }),
    )
    .expect("precommitted schedule should build");
    let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
    config.lattice.program_mode = ProgramMode::Committed;
    config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
    config.lattice.advice.untrusted = true;
    config.lattice.packed_witness.untrusted_advice_family = true;
    config.lattice.packed_witness.layout_digest = Some([0; 32]);
    config.lattice.packed_witness.d_pack = Some(0);
    config.lattice.packed_witness.validity_digest = Some([0; 32]);
    #[cfg(feature = "field-inline")]
    {
        config.lattice.field_inline.enabled = true;
        config.lattice.packed_witness.field_rd_inc_family = true;
    }

    let layout = crate::stages::stage8::derive_lattice_packed_witness_layout(
        &config,
        log_t,
        log_k_chunk,
        JoltRaPolynomialLayout::new(1, 1, 1).expect("RA layout should build"),
        &precommitted,
    )
    .expect("layout should derive");
    let requirements =
        derive_lattice_packed_validity_requirements(&config, log_k_chunk, &precommitted)
            .expect("validity requirements should derive");
    let statements = derive_lattice_packed_validity_statements(&layout, &requirements)
        .expect("validity statements should derive");
    (layout, statements)
}

fn small_bytecode_validity_context() -> (
    PackedWitnessLayout,
    Vec<LatticePackedValidityStatement>,
    Vec<LatticePackedValidityRequirement>,
) {
    let specs = vec![
        PackedFamilySpec::direct(
            PackedFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 0,
            },
            PackedFactDomain::BytecodeRows { log_bytecode: 0 },
            1,
            PackedAlphabet::Fixed {
                size: 1 << REGISTER_ADDRESS_BITS,
            },
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 2,
            },
            PackedFactDomain::BytecodeRows { log_bytecode: 0 },
            1,
            PackedAlphabet::Fixed {
                size: 1 << REGISTER_ADDRESS_BITS,
            },
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::BytecodeCircuitFlag {
                chunk: 0,
                flag: CircuitFlags::Store as usize,
            },
            PackedFactDomain::BytecodeRows { log_bytecode: 0 },
            1,
            PackedAlphabet::Bit,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::BytecodeImmBytes { chunk: 0 },
            PackedFactDomain::BytecodeRows { log_bytecode: 0 },
            AkitaField::NUM_BYTES,
            PackedAlphabet::Byte,
        ),
    ];
    #[cfg(feature = "field-inline")]
    let specs = {
        let mut specs = specs;
        specs.extend((0..AkitaField::NUM_BYTES).map(|index| {
            PackedFamilySpec::direct(
                PackedFamilyId::FieldRdIncByte { index },
                PackedFactDomain::TraceRows { log_t: 0 },
                1,
                PackedAlphabet::Byte,
            )
        }));
        specs
    };
    let layout =
        PackedWitnessLayout::new(specs).expect("manual bytecode validity layout should build");
    let mut requirements = akita_lattice_validity_requirements_for_layout(&layout);
    requirements.push(bytecode_imm_canonical_bytes_requirement(
        0,
        AkitaField::NUM_BYTES,
        AKITA_FIELD_MODULUS,
    ));
    let statements = derive_lattice_packed_validity_statements(&layout, &requirements)
        .expect("manual bytecode validity statements should derive");
    (layout, statements, requirements)
}

fn validity_statement(
    statements: &[LatticePackedValidityStatement],
    family: LatticePackedFamilyId,
    kind: LatticePackedValidityStatementKind,
) -> &LatticePackedValidityStatement {
    statements
        .iter()
        .find(|statement| statement.requirement.family == family && statement.kind == kind)
        .expect("validity statement should exist")
}

fn validity_value_at_zero(
    source: &SparsePackedWitness<AkitaField>,
    statement: &LatticePackedValidityStatement,
) -> AkitaField {
    let point = vec![AkitaField::zero(); statement.num_vars];
    validity_value(source, statement, &point, &point).expect("validity value should evaluate")
}

#[cfg(feature = "field-inline")]
fn validity_source_with_field_rd_inc_bytes(
    layout: &PackedWitnessLayout,
    requirements: &[LatticePackedValidityRequirement],
    bytes: &[u8],
) -> SparsePackedWitness<AkitaField> {
    validity_source_with_symbols(layout, requirements, |family, _| match family {
        LatticePackedFamilyId::FieldRdIncByte { index } => bytes[*index] as usize,
        _ => 0,
    })
}

fn validity_source_with_bytecode_imm_bytes(
    layout: &PackedWitnessLayout,
    requirements: &[LatticePackedValidityRequirement],
    bytes: &[u8],
) -> SparsePackedWitness<AkitaField> {
    validity_source_with_symbols(layout, requirements, |family, limb| match family {
        LatticePackedFamilyId::BytecodeImmBytes { .. } => bytes[limb] as usize,
        _ => 0,
    })
}

fn validity_source_with_symbols(
    layout: &PackedWitnessLayout,
    requirements: &[LatticePackedValidityRequirement],
    mut symbol_for: impl FnMut(&LatticePackedFamilyId, usize) -> usize,
) -> SparsePackedWitness<AkitaField> {
    let mut cells = Vec::new();
    for requirement in requirements {
        let family_id = lattice_packing_family_id(&requirement.family);
        let family = layout
            .family(&family_id)
            .expect("validity family should exist");
        let rows = family.domain.rows().expect("family rows should derive");
        if !matches!(requirement.kind, LatticePackedValidityKind::ExactOneHot) {
            continue;
        }
        for row in 0..rows {
            for limb in 0..requirement.limbs {
                let symbol = symbol_for(&requirement.family, limb);
                cells.push((
                    PackedCellAddress {
                        family: family_id.clone(),
                        row,
                        limb,
                        symbol,
                    },
                    AkitaField::one(),
                ));
            }
        }
    }
    SparsePackedWitness::try_from_cells(layout.clone(), cells)
        .expect("validity source should build")
}

fn verify_validity_artifacts<T>(
    setup: &AkitaVerifierSetup,
    transcript: &mut T,
    artifacts: &AkitaPackedWitnessArtifacts,
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
    validity: &AkitaPackedValidityProofArtifacts,
) -> Result<(), VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
{
    crate::stages::stage8::verify_lattice_packed_validity_proof::<
        AkitaField,
        AkitaPackedScheme,
        T,
        ClearOnlyCommitment,
    >(
        setup,
        transcript,
        &artifacts.protocol,
        log_k_chunk,
        precommitted,
        &artifacts.layout,
        artifacts
            .payload()
            .expect("artifact should carry lattice payload")
            .packed_witness
            .clone(),
        &validity.sumcheck_proof,
        &validity.opening_claims.opening_claims,
        &validity.opening_proof,
    )
}

#[test]
fn akita_clear_verifier_surface_is_nameable() {
    type TestTranscript = Blake2bTranscript<AkitaField>;
    type VerifyFn = fn(
        &AkitaVerifierPreprocessing,
        &JoltDevice,
        &AkitaJoltProof,
        Option<&AkitaCommitment>,
        &JoltProtocolConfig,
    ) -> Result<(), VerifierError>;
    let _verify: VerifyFn = verify_akita_clear::<TestTranscript>;
    type ProveFn = fn(
        &AkitaProverSetup,
        &AkitaVerifierPreprocessing,
        &JoltDevice,
        &AkitaJoltProof,
        Option<&AkitaCommitment>,
        &AkitaPackedWitnessArtifacts,
        &SparsePackedWitness<AkitaField>,
    ) -> Result<AkitaPackedBatchProof, VerifierError>;
    let _prove: ProveFn =
        prove_akita_jolt_final_openings::<TestTranscript, SparsePackedWitness<AkitaField>>;
    type ProveValidityFn = fn(
        &AkitaProverSetup,
        &AkitaVerifierPreprocessing,
        &JoltDevice,
        &AkitaJoltProof,
        Option<&AkitaCommitment>,
        &AkitaPackedWitnessArtifacts,
        &SparsePackedWitness<AkitaField>,
    ) -> Result<AkitaPackedValidityProofArtifacts, VerifierError>;
    let _prove_validity: ProveValidityFn =
        prove_akita_jolt_packed_validity::<TestTranscript, SparsePackedWitness<AkitaField>>;
    type AttachOpeningsFn = fn(
        &AkitaProverSetup,
        &AkitaVerifierPreprocessing,
        &JoltDevice,
        &mut AkitaJoltProof,
        Option<&AkitaCommitment>,
        &AkitaPackedWitnessArtifacts,
        &SparsePackedWitness<AkitaField>,
    ) -> Result<(), VerifierError>;
    let _attach_openings: AttachOpeningsFn =
        prove_and_attach_akita_opening_proofs::<TestTranscript, SparsePackedWitness<AkitaField>>;
}

#[test]
fn akita_verifier_setup_binds_protocol_config() {
    let layout = tiny_layout();
    let params = AkitaSetupParams::from_packed_layout(&layout, 1);
    let (_, verifier_setup) = AkitaPackedScheme::setup(params);
    let config = akita_lattice_protocol_config_for_layout(&layout);

    validate_akita_verifier_setup_config(&verifier_setup, &config)
        .expect("setup should match generated Akita protocol config");

    let mut wrong_digest = config;
    let mut digest = layout.digest;
    digest[0] ^= 1;
    wrong_digest.lattice.packed_witness.layout_digest = Some(digest);
    assert!(matches!(
        validate_akita_verifier_setup_config(&verifier_setup, &wrong_digest),
        Err(VerifierError::InvalidProtocolConfig { .. })
    ));

    let mut wrong_dimension = config;
    wrong_dimension.lattice.packed_witness.d_pack = Some(layout.dimension + 1);
    assert!(matches!(
        validate_akita_verifier_setup_config(&verifier_setup, &wrong_dimension),
        Err(VerifierError::InvalidProtocolConfig { .. })
    ));

    let mut missing_layout = verifier_setup.clone();
    missing_layout.packed_layout = None;
    assert!(matches!(
        validate_akita_verifier_setup_config(&missing_layout, &config),
        Err(VerifierError::InvalidProtocolConfig { .. })
    ));

    let mut missing_native = verifier_setup;
    missing_native.native.clear();
    assert!(matches!(
        validate_akita_verifier_setup_config(&missing_native, &config),
        Err(VerifierError::InvalidProtocolConfig { reason })
            if reason.contains("native setup bytes")
    ));
}

#[test]
fn akita_verifier_setup_binds_artifact_layout() {
    let layout = tiny_layout();
    let params = AkitaSetupParams::from_packed_layout(&layout, 1);
    let (_, verifier_setup) = AkitaPackedScheme::setup(params);

    validate_akita_verifier_setup_layout(&verifier_setup, &layout)
        .expect("setup should match generated Akita packed layout");

    let other_layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
        PackedFamilyId::InstructionRa { index: 1 },
        PackedFactDomain::TraceRows { log_t: 0 },
        1,
        PackedAlphabet::Byte,
    )])
    .expect("layout should build");
    assert!(matches!(
        validate_akita_verifier_setup_layout(&verifier_setup, &other_layout),
        Err(VerifierError::InvalidProtocolConfig { .. })
    ));

    let mut zero_group_setup = verifier_setup;
    zero_group_setup.max_num_polys_per_commitment_group = 0;
    assert!(matches!(
        validate_akita_verifier_setup_layout(&zero_group_setup, &layout),
        Err(VerifierError::InvalidProtocolConfig { .. })
    ));
}

#[test]
fn akita_verifier_payload_shape_binds_inner_commitment_metadata() {
    let layout = tiny_layout();
    let params = AkitaSetupParams::from_packed_layout(&layout, 1);
    let (prover_setup, verifier_setup) = AkitaPackedScheme::setup(params);
    let source = SparsePackedWitness::try_new(layout.clone(), Vec::new())
        .expect("empty sparse source should build");
    let artifacts =
        commit_akita_packed_witness(&prover_setup, &source).expect("packed witness should commit");
    validate_akita_proof_payload_shape(&verifier_setup, &artifacts.commitments)
        .expect("matching payload shape should pass");
    let payload = artifacts
        .commitments
        .as_lattice()
        .expect("artifact should carry lattice payload");

    let mut wrong_commitment_digest = payload.clone();
    wrong_commitment_digest.packed_witness.layout_digest = [9; 32];
    assert!(matches!(
        validate_akita_proof_payload_shape(
            &verifier_setup,
            &CommitmentPayload::Lattice(wrong_commitment_digest),
        ),
        Err(VerifierError::InvalidProtocolConfig { reason })
            if reason.contains("commitment layout digest")
    ));

    let mut wrong_commitment_dimension = payload.clone();
    wrong_commitment_dimension.packed_witness.num_vars = layout.dimension + 1;
    assert!(matches!(
        validate_akita_proof_payload_shape(
            &verifier_setup,
            &CommitmentPayload::Lattice(wrong_commitment_dimension),
        ),
        Err(VerifierError::InvalidProtocolConfig { reason })
            if reason.contains("commitment dimension")
    ));

    let mut wrong_poly_count = payload.clone();
    wrong_poly_count.packed_witness.poly_count = 2;
    assert!(matches!(
        validate_akita_proof_payload_shape(
            &verifier_setup,
            &CommitmentPayload::Lattice(wrong_poly_count),
        ),
        Err(VerifierError::InvalidProtocolConfig { reason })
            if reason.contains("exactly one polynomial")
    ));

    let mut missing_native_commitment = payload.clone();
    missing_native_commitment.packed_witness.native.clear();
    assert!(matches!(
        validate_akita_proof_payload_shape(
            &verifier_setup,
            &CommitmentPayload::Lattice(missing_native_commitment),
        ),
        Err(VerifierError::InvalidProtocolConfig { reason })
            if reason.contains("native commitment bytes")
    ));
}

#[test]
fn akita_untrusted_advice_aliases_packed_witness_but_trusted_must_be_separate() {
    let layout = tiny_layout();
    let params = AkitaSetupParams::from_packed_layout(&layout, 1);
    let (prover_setup, _) = AkitaPackedScheme::setup(params);
    let source =
        SparsePackedWitness::try_new(layout, Vec::new()).expect("empty sparse source should build");
    let artifacts =
        commit_akita_packed_witness(&prover_setup, &source).expect("packed witness should commit");
    let payload = artifacts
        .commitments
        .as_lattice()
        .expect("artifact should carry lattice payload");
    let packed_witness = &payload.packed_witness;
    validate_akita_advice_commitment_aliases(&artifacts.commitments, None, None)
        .expect("absent advice commitments should pass");
    validate_akita_advice_commitment_aliases(&artifacts.commitments, Some(packed_witness), None)
        .expect("packed-witness untrusted advice alias should pass");
    assert!(matches!(
        validate_akita_advice_commitment_aliases(
            &artifacts.commitments,
            None,
            Some(packed_witness),
        ),
        Err(VerifierError::InvalidProtocolConfig { reason })
            if reason.contains("trusted advice commitment must be separate")
    ));

    let mut other_commitment = packed_witness.clone();
    other_commitment.layout_digest[0] ^= 1;
    assert!(matches!(
        validate_akita_advice_commitment_aliases(
            &artifacts.commitments,
            Some(&other_commitment),
            None,
        ),
        Err(VerifierError::InvalidProtocolConfig { reason })
            if reason.contains("untrusted advice commitment")
    ));
    validate_akita_advice_commitment_aliases(&artifacts.commitments, None, Some(&other_commitment))
        .expect("trusted advice may use a separate precommitted commitment");
}

#[test]
fn akita_precommitted_commitments_must_not_alias_packed_witness() {
    let layout = tiny_layout();
    let params = AkitaSetupParams::from_packed_layout(&layout, 1);
    let (prover_setup, _) = AkitaPackedScheme::setup(params);
    let source =
        SparsePackedWitness::try_new(layout, Vec::new()).expect("empty sparse source should build");
    let artifacts =
        commit_akita_packed_witness(&prover_setup, &source).expect("packed witness should commit");
    let packed_witness = &artifacts
        .commitments
        .as_lattice()
        .expect("artifact should carry lattice payload")
        .packed_witness;

    assert!(matches!(
        validate_akita_precommitted_commitment_is_separate(
            packed_witness,
            packed_witness,
            "bytecode chunk",
        ),
        Err(VerifierError::InvalidProtocolConfig { reason })
            if reason.contains("bytecode chunk commitment must be separate")
    ));

    let mut separate_commitment = packed_witness.clone();
    separate_commitment.layout_digest[0] ^= 1;
    validate_akita_precommitted_commitment_is_separate(
        packed_witness,
        &separate_commitment,
        "program image",
    )
    .expect("separate precommitted commitment should pass");
}

#[test]
fn configured_layout_mismatch_rejects_before_commit() {
    let layout = tiny_layout();
    let params = AkitaSetupParams::from_packed_layout(&layout, 1);
    let (prover_setup, _) = AkitaPackedScheme::setup(params);
    let source = SparsePackedWitness::try_new(layout.clone(), Vec::new())
        .expect("empty sparse source should build");
    let mut config = akita_lattice_protocol_config_for_layout(&layout);
    config.lattice.packed_witness.layout_digest = Some([9; 32]);

    let error = commit_akita_packed_witness_with_config(config, &prover_setup, &source)
        .expect_err("layout mismatch should reject");

    assert!(matches!(error, VerifierError::InvalidProtocolConfig { .. }));
}

#[test]
fn akita_artifact_preflight_rejects_stale_protocol_and_commitments() {
    let layout = tiny_layout();
    let params = AkitaSetupParams::from_packed_layout(&layout, 1);
    let (prover_setup, verifier_setup) = AkitaPackedScheme::setup(params);
    let source = SparsePackedWitness::try_from_cells(
        layout.clone(),
        [
            (
                packed_cell(PackedFamilyId::InstructionRa { index: 0 }, 7),
                AkitaField::one(),
            ),
            (
                packed_cell(PackedFamilyId::UnsignedIncMsb, 1),
                AkitaField::one(),
            ),
        ],
    )
    .expect("source should build");
    let other_source = SparsePackedWitness::try_from_cells(
        layout.clone(),
        [
            (
                packed_cell(PackedFamilyId::InstructionRa { index: 0 }, 8),
                AkitaField::one(),
            ),
            (
                packed_cell(PackedFamilyId::UnsignedIncMsb, 0),
                AkitaField::one(),
            ),
        ],
    )
    .expect("other source should build");
    let artifacts =
        commit_akita_packed_witness(&prover_setup, &source).expect("packed witness should commit");
    let other_artifacts = commit_akita_packed_witness(&prover_setup, &other_source)
        .expect("other packed witness should commit");

    validate_akita_artifacts_for_proof(
        &verifier_setup,
        &artifacts.protocol,
        &artifacts.commitments,
        &artifacts,
    )
    .expect("matching artifacts should pass preflight");

    let mut stale_protocol = artifacts.protocol;
    stale_protocol.lattice.packed_witness.d_pack = Some(layout.dimension + 1);
    assert!(matches!(
        validate_akita_artifacts_for_proof(
            &verifier_setup,
            &stale_protocol,
            &artifacts.commitments,
            &artifacts,
        ),
        Err(VerifierError::ProtocolConfigMismatch { expected, got })
            if expected == artifacts.protocol && got == stale_protocol
    ));

    assert!(matches!(
        validate_akita_artifacts_for_proof(
            &verifier_setup,
            &artifacts.protocol,
            &other_artifacts.commitments,
            &artifacts,
        ),
        Err(VerifierError::InvalidProtocolConfig { reason })
            if reason.contains("do not match packed witness artifacts")
    ));
}
