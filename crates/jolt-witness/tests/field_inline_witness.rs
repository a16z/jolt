#![cfg(feature = "field-inline")]
#![cfg_attr(feature = "field-inline", expect(clippy::unwrap_used))]

use common::constants::RAM_START_ADDRESS;
use jolt_claims::protocols::{
    field_inline::{
        FieldInlineCommittedPolynomial, FieldInlineOpFlag, FieldInlineVirtualPolynomial,
    },
    jolt::{JoltCommittedPolynomial, JoltOneHotConfig},
};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_program::{
    execution::{
        JoltProgram, OwnedTrace, RegisterRead, RegisterState, RegisterWrite, TraceOutput, TraceRow,
    },
    field_inline::{
        FieldEncodedValue, FieldInlineBridge, FieldInlineTraceData, FieldRegisterRead,
        FieldRegisterWrite,
    },
    preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing},
};
use jolt_riscv::{
    FieldInlineOp, JoltInstructionKind, JoltInstructionProfile, JoltInstructionRow,
    NormalizedOperands, RV64IMAC_JOLT, RV64IMAC_JOLT_FIELD_INLINE,
};
use jolt_witness::{
    protocols::jolt_vm::{
        field_inline::{FieldInlineNamespace, FIELD_INLINE_NAMESPACE},
        JoltVmWitnessConfig, JoltVmWitnessInputs,
    },
    CommittedWitnessProvider, OracleRef, OracleViewRequest, PolynomialChunk, PolynomialStream,
    PolynomialView, RetentionHint, WitnessError, WitnessProvider,
};

const ENTRY: u64 = RAM_START_ADDRESS;

fn config(log_t: usize) -> JoltVmWitnessConfig {
    JoltVmWitnessConfig::new(
        log_t,
        64,
        JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        },
    )
}

fn instruction(
    instruction_kind: JoltInstructionKind,
    offset: usize,
    rd: Option<u8>,
    rs1: Option<u8>,
    rs2: Option<u8>,
    imm: i128,
) -> JoltInstructionRow {
    JoltInstructionRow {
        instruction_kind,
        address: ENTRY as usize + offset * 4,
        operands: NormalizedOperands { rd, rs1, rs2, imm },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: false,
    }
}

fn preprocessing(
    bytecode: Vec<JoltInstructionRow>,
    profile: JoltInstructionProfile,
) -> JoltProgramPreprocessing {
    JoltProgramPreprocessing {
        bytecode: BytecodePreprocessing::preprocess(bytecode, ENTRY, profile).unwrap(),
        ram: RAMPreprocessing::default(),
        memory_layout: Default::default(),
        max_padded_trace_length: 8,
    }
}

fn program(bytecode: Vec<JoltInstructionRow>, profile: JoltInstructionProfile) -> JoltProgram {
    JoltProgram::from_parts_with_profile(
        Vec::new(),
        bytecode,
        Vec::new(),
        ENTRY + 4,
        ENTRY,
        profile,
    )
}

fn witness<'a>(
    program: &'a JoltProgram,
    preprocessing: &'a JoltProgramPreprocessing,
    rows: Vec<TraceRow>,
    log_t: usize,
) -> jolt_witness::protocols::jolt_vm::TraceBackedJoltVmWitness<'a, OwnedTrace> {
    jolt_witness::protocols::jolt_vm::TraceBackedJoltVmWitness::new(
        config(log_t),
        JoltVmWitnessInputs::new(
            program,
            preprocessing,
            TraceOutput::new(OwnedTrace::new(rows), Default::default(), None),
        ),
    )
}

fn enc(value: u64) -> FieldEncodedValue {
    FieldEncodedValue::from_u64(value)
}

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

fn field_row(instruction: JoltInstructionRow, data: FieldInlineTraceData) -> TraceRow {
    TraceRow {
        instruction,
        field_inline: Some(data),
        ..TraceRow::default()
    }
}

fn public_fixture() -> (Vec<JoltInstructionRow>, Vec<TraceRow>) {
    let load_a = instruction(
        JoltInstructionKind::FIELD_LOAD_IMM,
        0,
        Some(1),
        None,
        None,
        13,
    );
    let row0 = field_row(
        load_a,
        FieldInlineTraceData {
            op: Some(FieldInlineOp::LoadImm),
            rd: Some(FieldRegisterWrite {
                register: 1,
                pre_value: enc(0),
                post_value: enc(13),
            }),
            ..FieldInlineTraceData::default()
        },
    );
    let load_b = instruction(
        JoltInstructionKind::FIELD_LOAD_IMM,
        1,
        Some(2),
        None,
        None,
        17,
    );
    let row1 = field_row(
        load_b,
        FieldInlineTraceData {
            op: Some(FieldInlineOp::LoadImm),
            rd: Some(FieldRegisterWrite {
                register: 2,
                pre_value: enc(0),
                post_value: enc(17),
            }),
            ..FieldInlineTraceData::default()
        },
    );
    let mul = instruction(
        JoltInstructionKind::FIELD_MUL,
        2,
        Some(3),
        Some(1),
        Some(2),
        0,
    );
    let row2 = field_row(
        mul,
        FieldInlineTraceData {
            op: Some(FieldInlineOp::Mul),
            rs1: Some(FieldRegisterRead {
                register: 1,
                value: enc(13),
            }),
            rs2: Some(FieldRegisterRead {
                register: 2,
                value: enc(17),
            }),
            rd: Some(FieldRegisterWrite {
                register: 3,
                pre_value: enc(0),
                post_value: enc(221),
            }),
            product: Some(enc(221)),
            ..FieldInlineTraceData::default()
        },
    );
    (vec![load_a, load_b, mul], vec![row0, row1, row2])
}

fn owned_view(
    provider: &jolt_witness::protocols::jolt_vm::field_inline::TraceBackedFieldInlineWitness<
        '_,
        '_,
        OwnedTrace,
    >,
    oracle: OracleRef<FieldInlineNamespace>,
) -> Vec<Fr> {
    let requirement =
        <jolt_witness::protocols::jolt_vm::field_inline::TraceBackedFieldInlineWitness<
            '_,
            '_,
            OwnedTrace,
        > as WitnessProvider<Fr, FieldInlineNamespace>>::view_requirements(provider, oracle)
        .unwrap()
        .remove(0);
    let view: PolynomialView<'_, Fr, FieldInlineNamespace> =
        <jolt_witness::protocols::jolt_vm::field_inline::TraceBackedFieldInlineWitness<
            '_,
            '_,
            OwnedTrace,
        > as WitnessProvider<Fr, FieldInlineNamespace>>::oracle_view(
            provider,
            OracleViewRequest::new(requirement),
        )
        .unwrap();
    match view {
        PolynomialView::Owned { values, .. } => values,
        PolynomialView::Borrowed { values, .. } => values.to_vec(),
        PolynomialView::Deferred { .. } => Vec::new(),
    }
}

#[test]
fn field_inline_public_provider_streams_and_materializes_views() {
    let (bytecode, rows) = public_fixture();
    let program = program(bytecode.clone(), RV64IMAC_JOLT_FIELD_INLINE);
    let preprocessing = preprocessing(bytecode, RV64IMAC_JOLT_FIELD_INLINE);
    let witness = witness(&program, &preprocessing, rows, 2);
    let provider = witness.field_inline_witness().unwrap();

    let order: Vec<FieldInlineCommittedPolynomial> =
        <jolt_witness::protocols::jolt_vm::field_inline::TraceBackedFieldInlineWitness<
            '_,
            '_,
            OwnedTrace,
        > as CommittedWitnessProvider<Fr, FieldInlineNamespace>>::committed_oracle_order(
            &provider
        )
        .unwrap();
    assert_eq!(order, vec![FieldInlineCommittedPolynomial::FieldRdInc]);

    let mut stream = <_ as WitnessProvider<Fr, FieldInlineNamespace>>::committed_stream(
        &provider,
        FieldInlineCommittedPolynomial::FieldRdInc,
        4,
    )
    .unwrap();
    assert_eq!(
        stream.next_chunk(),
        Ok(Some(PolynomialChunk::Dense(vec![
            fr(13),
            fr(17),
            fr(221),
            fr(0),
        ])))
    );

    let products = owned_view(
        &provider,
        OracleRef::virtual_polynomial(FieldInlineVirtualPolynomial::FieldProduct),
    );
    assert_eq!(&products[..4], &[fr(0), fr(0), fr(221), fr(0)]);

    let mul_flags = owned_view(
        &provider,
        OracleRef::virtual_polynomial(FieldInlineVirtualPolynomial::FieldOpFlag(
            FieldInlineOpFlag::Mul,
        )),
    );
    assert_eq!(&mul_flags[..4], &[fr(0), fr(0), fr(1), fr(0)]);

    let requirement =
        <jolt_witness::protocols::jolt_vm::field_inline::TraceBackedFieldInlineWitness<
            '_,
            '_,
            OwnedTrace,
        > as WitnessProvider<Fr, FieldInlineNamespace>>::view_requirements(
            &provider,
            OracleRef::virtual_polynomial(FieldInlineVirtualPolynomial::FieldRegistersVal),
        )
        .unwrap()
        .remove(0);
    assert_eq!(requirement.retention, RetentionHint::ThroughBlindFold);
}

#[test]
fn field_inline_public_provider_is_absent_for_fr_off_programs() {
    let bytecode = vec![instruction(
        JoltInstructionKind::ADDI,
        0,
        Some(1),
        Some(2),
        None,
        3,
    )];
    let program = program(bytecode.clone(), RV64IMAC_JOLT);
    let preprocessing = preprocessing(bytecode, RV64IMAC_JOLT);
    let witness = witness(
        &program,
        &preprocessing,
        vec![TraceRow {
            instruction: instruction(JoltInstructionKind::ADDI, 0, Some(1), Some(2), None, 3),
            ..TraceRow::default()
        }],
        2,
    );

    assert_eq!(
        witness.field_inline_witness().err(),
        Some(WitnessError::UnavailableView {
            namespace: FIELD_INLINE_NAMESPACE.name,
        })
    );
}

#[test]
fn public_bridge_rows_keep_x_register_and_field_register_witnesses_disjoint() {
    let load = instruction(
        JoltInstructionKind::FIELD_LOAD_FROM_X,
        0,
        Some(1),
        Some(5),
        None,
        0,
    );
    let load_row = TraceRow {
        instruction: load,
        registers: RegisterState {
            rs1: Some(RegisterRead {
                register: 5,
                value: 19,
            }),
            ..RegisterState::default()
        },
        field_inline: Some(FieldInlineTraceData {
            op: Some(FieldInlineOp::LoadFromX),
            rd: Some(FieldRegisterWrite {
                register: 1,
                pre_value: enc(0),
                post_value: enc(19),
            }),
            bridge: Some(FieldInlineBridge::LoadFromX {
                x_register: 5,
                x_value: 19,
                field_value: enc(19),
            }),
            ..FieldInlineTraceData::default()
        }),
        ..TraceRow::default()
    };

    let store = instruction(
        JoltInstructionKind::FIELD_STORE_TO_X,
        1,
        Some(6),
        Some(1),
        None,
        0,
    );
    let store_row = TraceRow {
        instruction: store,
        registers: RegisterState {
            rd: Some(RegisterWrite {
                register: 6,
                pre_value: 0,
                post_value: 19,
            }),
            ..RegisterState::default()
        },
        field_inline: Some(FieldInlineTraceData {
            op: Some(FieldInlineOp::StoreToX),
            rs1: Some(FieldRegisterRead {
                register: 1,
                value: enc(19),
            }),
            bridge: Some(FieldInlineBridge::StoreToX {
                field_register: 1,
                field_value: enc(19),
                x_register: 6,
                x_value: 19,
            }),
            ..FieldInlineTraceData::default()
        }),
        ..TraceRow::default()
    };

    let bytecode = vec![load, store];
    let program = program(bytecode.clone(), RV64IMAC_JOLT_FIELD_INLINE);
    let preprocessing = preprocessing(bytecode, RV64IMAC_JOLT_FIELD_INLINE);
    let witness = witness(&program, &preprocessing, vec![load_row, store_row], 2);
    let provider = witness.field_inline_witness().unwrap();

    let mut ordinary_stream = witness
        .committed_stream(JoltCommittedPolynomial::RdInc, 4)
        .unwrap();
    assert_eq!(
        ordinary_stream.next_chunk(),
        Ok(Some(PolynomialChunk::<Fr>::I128(vec![0, 19, 0, 0])))
    );

    let mut field_stream = <_ as WitnessProvider<Fr, FieldInlineNamespace>>::committed_stream(
        &provider,
        FieldInlineCommittedPolynomial::FieldRdInc,
        4,
    )
    .unwrap();
    assert_eq!(
        field_stream.next_chunk(),
        Ok(Some(PolynomialChunk::Dense(vec![
            fr(19),
            fr(0),
            fr(0),
            fr(0),
        ])))
    );
}
