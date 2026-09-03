#![cfg(feature = "field-inline")]
#![cfg_attr(feature = "field-inline", expect(clippy::unwrap_used))]

use std::sync::Arc;

use common::constants::RAM_START_ADDRESS;
use jolt_claims::protocols::{
    field_inline::{
        FieldInlineCommittedPolynomial, FieldInlineOpFlag, FieldInlinePolynomialId,
        FieldInlineVirtualPolynomial,
    },
    jolt::{JoltCommittedPolynomial, JoltOneHotConfig, JoltPolynomialId},
};
use jolt_field::{Fr, Ring};
use jolt_program::{
    execution::{
        JoltProgram, OwnedTrace, RamAccess, RegisterRead, RegisterState, RegisterWrite,
        TraceOutput, TraceRow,
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
    field_inline::{TraceBackedFieldInlineWitness, FIELD_INLINE_LABEL},
    JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessOracle, TraceBackend, WitnessError,
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
) -> Arc<JoltProgramPreprocessing> {
    Arc::new(JoltProgramPreprocessing {
        bytecode: BytecodePreprocessing::preprocess(bytecode, ENTRY, profile).unwrap(),
        ram: RAMPreprocessing::default(),
        memory_layout: Default::default(),
        max_padded_trace_length: 8,
    })
}

fn program(bytecode: Vec<JoltInstructionRow>, profile: JoltInstructionProfile) -> Arc<JoltProgram> {
    Arc::new(JoltProgram::from_parts_with_profile(
        Vec::new(),
        bytecode,
        Vec::new(),
        ENTRY + 4,
        ENTRY,
        profile,
    ))
}

fn witness(
    program: &Arc<JoltProgram>,
    preprocessing: &Arc<JoltProgramPreprocessing>,
    rows: Vec<TraceRow>,
    log_t: usize,
) -> TraceBackend<OwnedTrace> {
    TraceBackend::new(
        config(log_t),
        JoltVmWitnessInputs::new(
            program,
            preprocessing,
            TraceOutput::new(OwnedTrace::new(rows), Default::default(), None, None),
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
    field_row_with_registers(instruction, RegisterState::default(), data)
}

fn field_row_with_registers(
    instruction: JoltInstructionRow,
    registers: RegisterState,
    data: FieldInlineTraceData,
) -> TraceRow {
    let mut row = TraceRow::new(instruction, registers, RamAccess::NoOp).unwrap();
    row.field_inline = Some(data.into());
    row
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
    provider: &TraceBackedFieldInlineWitness,
    id: impl Into<FieldInlinePolynomialId>,
) -> Vec<Fr> {
    provider.oracle_table::<Fr>(id.into()).unwrap()
}

fn field_rd_inc_column(provider: &TraceBackedFieldInlineWitness) -> Vec<Fr> {
    owned_view(
        provider,
        FieldInlinePolynomialId::Committed(FieldInlineCommittedPolynomial::FieldRdInc),
    )
}

#[test]
fn field_inline_public_provider_materializes_views() {
    let (bytecode, rows) = public_fixture();
    let program = program(bytecode.clone(), RV64IMAC_JOLT_FIELD_INLINE);
    let preprocessing = preprocessing(bytecode, RV64IMAC_JOLT_FIELD_INLINE);
    let witness = witness(&program, &preprocessing, rows, 2);
    let provider = witness.field_inline_witness().unwrap();

    let order = provider.committed_order();
    assert_eq!(order, vec![FieldInlineCommittedPolynomial::FieldRdInc]);

    assert_eq!(
        field_rd_inc_column(&provider),
        vec![fr(13), fr(17), fr(221), fr(0)]
    );

    let products = owned_view(
        &provider,
        FieldInlinePolynomialId::Virtual(FieldInlineVirtualPolynomial::FieldProduct),
    );
    assert_eq!(&products[..4], &[fr(0), fr(0), fr(221), fr(0)]);

    let mul_flags = owned_view(
        &provider,
        FieldInlinePolynomialId::Virtual(FieldInlineVirtualPolynomial::FieldOpFlag(
            FieldInlineOpFlag::Mul,
        )),
    );
    assert_eq!(&mul_flags[..4], &[fr(0), fr(0), fr(1), fr(0)]);
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
        vec![TraceRow::from_instruction(instruction(
            JoltInstructionKind::ADDI,
            0,
            Some(1),
            Some(2),
            None,
            3,
        ))
        .unwrap()],
        2,
    );

    assert_eq!(
        witness.field_inline_witness().err(),
        Some(WitnessError::UnavailableView {
            label: FIELD_INLINE_LABEL,
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
    let load_row = field_row_with_registers(
        load,
        RegisterState {
            rs1: Some(RegisterRead {
                register: 5,
                value: 19,
            }),
            ..RegisterState::default()
        },
        FieldInlineTraceData {
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
        },
    );

    let store = instruction(
        JoltInstructionKind::FIELD_STORE_TO_X,
        1,
        Some(6),
        Some(1),
        None,
        0,
    );
    let store_row = field_row_with_registers(
        store,
        RegisterState {
            rd: Some(RegisterWrite {
                register: 6,
                pre_value: 0,
                post_value: 19,
            }),
            ..RegisterState::default()
        },
        FieldInlineTraceData {
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
        },
    );

    let bytecode = vec![load, store];
    let program = program(bytecode.clone(), RV64IMAC_JOLT_FIELD_INLINE);
    let preprocessing = preprocessing(bytecode, RV64IMAC_JOLT_FIELD_INLINE);
    let witness = witness(&program, &preprocessing, vec![load_row, store_row], 2);
    let provider = witness.field_inline_witness().unwrap();

    let ordinary = JoltWitnessOracle::<Fr>::oracle_table(
        &witness,
        JoltPolynomialId::Committed(JoltCommittedPolynomial::RdInc),
    )
    .unwrap();
    assert_eq!(ordinary, vec![fr(0), fr(19), fr(0), fr(0)]);

    assert_eq!(
        field_rd_inc_column(&provider),
        vec![fr(19), fr(0), fr(0), fr(0)]
    );
}

#[test]
fn plane_accessor_serves_the_attached_field_inline_view() {
    let (bytecode, rows) = public_fixture();
    let program = program(bytecode.clone(), RV64IMAC_JOLT_FIELD_INLINE);
    let preprocessing = preprocessing(bytecode, RV64IMAC_JOLT_FIELD_INLINE);

    // Fail-closed default: an FR-profile backend without the attached view
    // serves no field-inline oracle.
    let detached = witness(&program, &preprocessing, rows.clone(), 2);
    assert!(JoltWitnessOracle::<Fr>::field_inline(&detached).is_none());

    let attached = witness(&program, &preprocessing, rows, 2)
        .with_field_inline()
        .unwrap();
    let oracle: &dyn JoltWitnessOracle<Fr> = &attached;
    let provider = oracle.field_inline().unwrap();
    assert_eq!(
        provider.committed_order(),
        vec![FieldInlineCommittedPolynomial::FieldRdInc]
    );
    // The dyn seam serves the same column as the inherent view.
    assert_eq!(
        provider
            .oracle_table(FieldInlinePolynomialId::Committed(
                FieldInlineCommittedPolynomial::FieldRdInc,
            ))
            .unwrap(),
        vec![fr(13), fr(17), fr(221), fr(0)]
    );
}

#[test]
fn plane_accessor_stays_absent_for_fr_off_profile() {
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
    let backend = witness(
        &program,
        &preprocessing,
        vec![TraceRow::from_instruction(instruction(
            JoltInstructionKind::ADDI,
            0,
            Some(1),
            Some(2),
            None,
            3,
        ))
        .unwrap()],
        2,
    );

    assert!(JoltWitnessOracle::<Fr>::field_inline(&backend).is_none());
    // The view cannot be attached for an FR-off guest, so the accessor can
    // never become Some.
    assert!(matches!(
        backend.with_field_inline(),
        Err(WitnessError::UnavailableView {
            label: FIELD_INLINE_LABEL,
        })
    ));
}
