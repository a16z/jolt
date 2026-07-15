use common::{
    constants::RAM_START_ADDRESS,
    jolt_device::{JoltDevice, MemoryConfig, MemoryLayout},
};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_program::{
    execution::{
        JoltProgram, MemoryImage, OwnedTrace, RamAccess, RamRead, RamWrite, RegisterRead,
        RegisterState, RegisterWrite, TraceOutput, TraceRow,
    },
    preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing},
};
use jolt_riscv::{
    CircuitFlags, InstructionFlags, JoltInstructionKind, JoltInstructionRow, NormalizedOperands,
    RV64IMAC_JOLT,
};

use super::*;
use crate::witnesses::{
    Extract, ExtractIndexed, Imm, InstructionFlag, LeftInstructionInput, LookupOutput, NextIsNoop,
    OpFlag, Pc, Product, RamAddress, RamReadValue, RamWriteValue, RightInstructionInput, Rs1Value,
    ShouldJump, ToField, UnexpandedPc, WitnessEnv,
};
use crate::{JoltWitnessOracle, PolynomialChunk, PolynomialEncoding, PolynomialStream, Shape};

fn preprocessing() -> JoltProgramPreprocessing {
    let bytecode = BytecodePreprocessing {
        code_size: 32,
        ..Default::default()
    };
    let mut preprocessing = JoltProgramPreprocessing {
        bytecode,
        ram: RAMPreprocessing::default(),
        memory_layout: Default::default(),
        max_padded_trace_length: 16,
    };
    preprocessing.memory_layout.max_trusted_advice_size = 64;
    preprocessing.memory_layout.max_untrusted_advice_size = 128;
    preprocessing
}

fn preprocessing_with_bytecode(bytecode: BytecodePreprocessing) -> JoltProgramPreprocessing {
    JoltProgramPreprocessing {
        bytecode,
        ..preprocessing()
    }
}

fn preprocessing_with_memory_layout(memory_layout: MemoryLayout) -> JoltProgramPreprocessing {
    JoltProgramPreprocessing {
        memory_layout,
        ..preprocessing()
    }
}

fn config() -> JoltVmWitnessConfig {
    JoltVmWitnessConfig::new(
        4,
        64,
        JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        },
    )
}

fn trace_output() -> TraceOutput<OwnedTrace> {
    TraceOutput::new(OwnedTrace::default(), Default::default(), None)
}

fn trace_output_with_rows(rows: Vec<TraceRow>) -> TraceOutput<OwnedTrace> {
    TraceOutput::new(OwnedTrace::new(rows), Default::default(), None)
}

fn trace_output_with_device(device: JoltDevice) -> TraceOutput<OwnedTrace> {
    TraceOutput::new(OwnedTrace::default(), device, None)
}

fn trace_output_with_device_and_final_memory(
    device: JoltDevice,
    final_memory: MemoryImage,
) -> TraceOutput<OwnedTrace> {
    TraceOutput::new(OwnedTrace::default(), device, Some(final_memory))
}

fn instruction(address: usize) -> JoltInstructionRow {
    JoltInstructionRow {
        instruction_kind: JoltInstructionKind::ADDI,
        address,
        operands: NormalizedOperands {
            rd: Some(1),
            rs1: Some(2),
            rs2: None,
            imm: 3,
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: false,
    }
}

fn compact_memory_layout() -> MemoryLayout {
    MemoryLayout::new(&MemoryConfig {
        max_input_size: 0,
        max_trusted_advice_size: 0,
        max_untrusted_advice_size: 0,
        max_output_size: 0,
        stack_size: 0,
        heap_size: 0,
        program_size: Some(64),
    })
}

fn shape(
    witness: &TraceBackend<'_, OwnedTrace>,
    id: impl Into<JoltPolynomialId>,
) -> Result<Shape, WitnessError> {
    witness.shape_of(id.into())
}

#[test]
fn witness_keeps_jolt_program_execution_boundary() {
    let program = JoltProgram::default();
    let preprocessing = preprocessing();
    let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output());
    let config = config().retain_trace_rows(true);

    let witness = TraceBackend::new(config.clone(), inputs);

    assert_eq!(witness.config, config);
    assert_eq!(witness.program.elf_bytes(), program.elf_bytes());
    assert_eq!(
        witness.preprocessing.max_padded_trace_length,
        preprocessing.max_padded_trace_length
    );
}

#[test]
fn committed_polynomial_order_uses_proof_payload_order() {
    let program = JoltProgram::default();
    let preprocessing = preprocessing();
    let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output());
    let witness = TraceBackend::new(
        config()
            .include_trusted_advice(true)
            .include_untrusted_advice(true),
        inputs,
    );
    let mut expected = vec![
        JoltCommittedPolynomial::RdInc,
        JoltCommittedPolynomial::RamInc,
    ];
    expected.extend((0..32).map(JoltCommittedPolynomial::InstructionRa));
    expected.extend((0..2).map(JoltCommittedPolynomial::RamRa));
    expected.extend((0..2).map(JoltCommittedPolynomial::BytecodeRa));
    expected.push(JoltCommittedPolynomial::TrustedAdvice);
    expected.push(JoltCommittedPolynomial::UntrustedAdvice);

    assert_eq!(witness.committed_polynomial_order(), Ok(expected));
}

#[test]
fn committed_oracle_descriptors_report_dimensions_and_encoding() {
    let program = JoltProgram::default();
    let preprocessing = preprocessing();
    let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output());
    let witness = TraceBackend::new(config().include_trusted_advice(true), inputs);

    assert_eq!(
        shape(&witness, JoltCommittedPolynomial::RamInc),
        Ok(Shape::new(4, PolynomialEncoding::Compact))
    );
    assert_eq!(
        shape(&witness, JoltCommittedPolynomial::InstructionRa(0)),
        Ok(Shape::new(8, PolynomialEncoding::OneHot))
    );
    assert_eq!(
        shape(&witness, JoltCommittedPolynomial::TrustedAdvice),
        Ok(Shape::new(3, PolynomialEncoding::Compact))
    );
}

#[test]
fn descriptors_reject_disabled_advice_and_out_of_range_ra() {
    let program = JoltProgram::default();
    let preprocessing = preprocessing();
    let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output());
    let witness = TraceBackend::new(config(), inputs);

    assert_eq!(
        shape(&witness, JoltCommittedPolynomial::TrustedAdvice),
        Err(WitnessError::UnknownOracle {
            label: JOLT_VM_LABEL,
        })
    );
    assert_eq!(
        shape(&witness, JoltCommittedPolynomial::BytecodeRa(2)),
        Err(WitnessError::UnknownOracle {
            label: JOLT_VM_LABEL,
        })
    );
}

#[test]
fn virtual_oracle_descriptors_report_stage1_trace_columns() -> Result<(), String> {
    let program = JoltProgram::default();
    let preprocessing = preprocessing();
    let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output());
    let witness = TraceBackend::new(config().with_log_t(2), inputs);

    assert_eq!(
        shape(&witness, JoltVirtualPolynomial::Product).map_err(|error| error.to_string())?,
        Shape::new(2, PolynomialEncoding::Dense)
    );
    assert_eq!(
        shape(&witness, JoltVirtualPolynomial::RamRa),
        Ok(Shape::new(8, PolynomialEncoding::Dense))
    );
    assert_eq!(
        shape(&witness, JoltVirtualPolynomial::RamValFinal),
        Ok(Shape::new(6, PolynomialEncoding::Dense))
    );
    Ok(())
}

#[test]
fn virtual_oracle_views_materialize_stage1_r1cs_inputs() -> Result<(), String> {
    let instruction_row = instruction(0x8000_0000);
    let bytecode = BytecodePreprocessing::preprocess(
        vec![instruction_row],
        instruction_row.address as u64,
        RV64IMAC_JOLT,
    )
    .map_err(|error| error.to_string())?;
    let preprocessing = preprocessing_with_bytecode(bytecode);
    let program = JoltProgram::default();
    let rows = vec![
        TraceRow {
            instruction: instruction_row,
            registers: RegisterState {
                rs1: Some(RegisterRead {
                    register: 2,
                    value: 5,
                }),
                rd: Some(RegisterWrite {
                    register: 1,
                    pre_value: 0,
                    post_value: 8,
                }),
                ..Default::default()
            },
            ram_access: RamAccess::Read(RamRead {
                address: RAM_START_ADDRESS,
                value: 7,
            }),
            #[cfg(feature = "field-inline")]
            field_inline: None,
        },
        TraceRow::default(),
    ];
    let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_rows(rows));
    let witness = TraceBackend::new(config().with_log_t(2), inputs);

    assert_virtual_values(
        &witness,
        JoltVirtualPolynomial::LeftInstructionInput,
        &[5, 0, 0, 0],
    )?;
    assert_virtual_values(
        &witness,
        JoltVirtualPolynomial::RightInstructionInput,
        &[3, 0, 0, 0],
    )?;
    assert_virtual_values(&witness, JoltVirtualPolynomial::Product, &[15, 0, 0, 0])?;
    assert_virtual_values(&witness, JoltVirtualPolynomial::LookupOutput, &[8, 0, 0, 0])?;
    assert_virtual_values(&witness, JoltVirtualPolynomial::PC, &[1, 0, 0, 0])?;
    // The last cycle's missing successor counts as a no-op (the
    // product/shift-family convention; see the trace materialization).
    assert_virtual_values(&witness, JoltVirtualPolynomial::NextIsNoop, &[1, 1, 1, 1])?;
    assert_virtual_values(
        &witness,
        JoltVirtualPolynomial::RamAddress,
        &[RAM_START_ADDRESS, 0, 0, 0],
    )?;
    assert_virtual_values(&witness, JoltVirtualPolynomial::RamReadValue, &[7, 0, 0, 0])?;
    assert_virtual_values(
        &witness,
        JoltVirtualPolynomial::RamWriteValue,
        &[7, 0, 0, 0],
    )?;
    assert_virtual_values(
        &witness,
        JoltVirtualPolynomial::OpFlags(CircuitFlags::AddOperands),
        &[1, 0, 0, 0],
    )?;
    assert_virtual_values(
        &witness,
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsImm),
        &[1, 0, 0, 0],
    )?;
    Ok(())
}

#[test]
fn ram_read_write_virtual_views_materialize_address_major_state() -> Result<(), String> {
    let program = JoltProgram::default();
    let memory_layout = compact_memory_layout();
    let access_address = memory_layout.stack_end;
    let preprocessing = preprocessing_with_memory_layout(memory_layout);
    let rows = vec![
        TraceRow {
            ram_access: RamAccess::Write(RamWrite {
                address: access_address,
                pre_value: 3,
                post_value: 9,
            }),
            ..Default::default()
        },
        TraceRow {
            ram_access: RamAccess::NoOp,
            ..Default::default()
        },
        TraceRow {
            ram_access: RamAccess::Read(RamRead {
                address: access_address,
                value: 9,
            }),
            ..Default::default()
        },
    ];
    let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_rows(rows));
    let witness = TraceBackend::new(JoltVmWitnessConfig::new(2, 16, config().one_hot), inputs);

    let val = materialized_virtual_view(&witness, JoltVirtualPolynomial::RamVal)?;
    let ra = materialized_virtual_view(&witness, JoltVirtualPolynomial::RamRa)?;
    let base = 10 * 4;
    assert_eq!(val.len(), 64);
    assert_eq!(ra.len(), 64);
    assert_eq!(val[base], Fr::from_u64(3));
    assert_eq!(val[base + 1], Fr::from_u64(9));
    assert_eq!(val[base + 2], Fr::from_u64(9));
    assert_eq!(val[base + 3], Fr::from_u64(9));
    assert_eq!(ra[base], Fr::from_u64(1));
    assert_eq!(ra[base + 1], Fr::from_u64(0));
    assert_eq!(ra[base + 2], Fr::from_u64(1));
    assert_eq!(ra[base + 3], Fr::from_u64(0));
    Ok(())
}

#[test]
fn register_read_write_virtual_views_materialize_address_major_state() -> Result<(), String> {
    let program = JoltProgram::default();
    let preprocessing = preprocessing();
    let rows = vec![
        TraceRow {
            registers: RegisterState {
                rd: Some(RegisterWrite {
                    register: 1,
                    pre_value: 0,
                    post_value: 5,
                }),
                ..Default::default()
            },
            ..Default::default()
        },
        TraceRow {
            registers: RegisterState {
                rs1: Some(RegisterRead {
                    register: 1,
                    value: 5,
                }),
                rd: Some(RegisterWrite {
                    register: 2,
                    pre_value: 0,
                    post_value: 7,
                }),
                ..Default::default()
            },
            ..Default::default()
        },
        TraceRow {
            registers: RegisterState {
                rs2: Some(RegisterRead {
                    register: 2,
                    value: 7,
                }),
                ..Default::default()
            },
            ..Default::default()
        },
    ];
    let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_rows(rows));
    let witness = TraceBackend::new(config().with_log_t(2), inputs);

    let val = materialized_virtual_view(&witness, JoltVirtualPolynomial::RegistersVal)?;
    let rs1_ra = materialized_virtual_view(&witness, JoltVirtualPolynomial::Rs1Ra)?;
    let rs2_ra = materialized_virtual_view(&witness, JoltVirtualPolynomial::Rs2Ra)?;
    let rd_wa = materialized_virtual_view(&witness, JoltVirtualPolynomial::RdWa)?;

    assert_eq!(val.len(), 128 * 4);
    assert_eq!(&val[4..8], &[0, 5, 5, 5].map(Fr::from_u64));
    assert_eq!(&val[8..12], &[0, 0, 7, 7].map(Fr::from_u64));
    assert_eq!(rs1_ra[4 + 1], Fr::from_u64(1));
    assert_eq!(rs2_ra[8 + 2], Fr::from_u64(1));
    assert_eq!(rd_wa[4], Fr::from_u64(1));
    assert_eq!(rd_wa[8 + 1], Fr::from_u64(1));
    Ok(())
}

#[test]
fn ram_val_final_virtual_view_materializes_final_memory_and_public_io() -> Result<(), String> {
    let program = JoltProgram::default();
    let memory_layout = MemoryLayout::new(&MemoryConfig {
        max_input_size: 8,
        max_trusted_advice_size: 8,
        max_untrusted_advice_size: 8,
        max_output_size: 8,
        stack_size: 0,
        heap_size: 0,
        program_size: Some(64),
    });
    let preprocessing = preprocessing_with_memory_layout(memory_layout.clone());
    let device = JoltDevice {
        memory_layout,
        trusted_advice: vec![0x11],
        untrusted_advice: vec![0x22],
        inputs: vec![0x33],
        outputs: vec![0x44, 0x55],
        ..Default::default()
    };
    let final_memory = MemoryImage {
        bytes: vec![(64, 0x66), (65, 0x77)],
    };
    let inputs = JoltVmWitnessInputs::new(
        &program,
        &preprocessing,
        trace_output_with_device_and_final_memory(device, final_memory),
    );
    let witness = TraceBackend::new(JoltVmWitnessConfig::new(2, 32, config().one_hot), inputs);

    let val_final = materialized_virtual_view(&witness, JoltVirtualPolynomial::RamValFinal)?;
    assert_eq!(val_final.len(), 32);
    assert_eq!(val_final[0], Fr::from_u64(0x11));
    assert_eq!(val_final[1], Fr::from_u64(0x22));
    assert_eq!(val_final[2], Fr::from_u64(0x33));
    assert_eq!(val_final[3], Fr::from_u64(0x5544));
    assert_eq!(val_final[4], Fr::from_u64(0));
    assert_eq!(val_final[5], Fr::from_u64(1));
    assert_eq!(val_final[16], Fr::from_u64(0x7766));
    Ok(())
}

#[test]
fn atomic_extractors_derive_named_witnesses() -> Result<(), String> {
    let instruction_row = instruction(0x8000_0000);
    let bytecode = BytecodePreprocessing::preprocess(
        vec![instruction_row],
        instruction_row.address as u64,
        RV64IMAC_JOLT,
    )
    .map_err(|error| error.to_string())?;
    let preprocessing = preprocessing_with_bytecode(bytecode);
    let row = TraceRow {
        instruction: instruction_row,
        registers: RegisterState {
            rs1: Some(RegisterRead {
                register: 2,
                value: 5,
            }),
            rd: Some(RegisterWrite {
                register: 1,
                pre_value: 0,
                post_value: 8,
            }),
            ..Default::default()
        },
        ram_access: RamAccess::Read(RamRead {
            address: RAM_START_ADDRESS,
            value: 7,
        }),
        #[cfg(feature = "field-inline")]
        field_inline: None,
    };
    let next = TraceRow::default();
    let env = WitnessEnv {
        preprocessing: &preprocessing,
    };

    assert_eq!(
        LeftInstructionInput::extract(&row, Some(&next), &env),
        Ok(LeftInstructionInput(5))
    );
    assert_eq!(
        RightInstructionInput::extract(&row, Some(&next), &env),
        Ok(RightInstructionInput(3))
    );
    assert_eq!(
        Product::extract(&row, Some(&next), &env).map(|product| product.to_field::<Fr>()),
        Ok(Fr::from_u64(15))
    );
    assert_eq!(
        LookupOutput::extract(&row, Some(&next), &env),
        Ok(LookupOutput(8))
    );
    assert_eq!(Pc::extract(&row, Some(&next), &env), Ok(Pc(1)));
    assert_eq!(
        UnexpandedPc::extract(&row, Some(&next), &env),
        Ok(UnexpandedPc(0x8000_0000))
    );
    assert_eq!(Imm::extract(&row, Some(&next), &env), Ok(Imm(3)));
    assert_eq!(Rs1Value::extract(&row, Some(&next), &env), Ok(Rs1Value(5)));
    assert_eq!(
        RamAddress::extract(&row, Some(&next), &env),
        Ok(RamAddress(RAM_START_ADDRESS))
    );
    assert_eq!(
        RamReadValue::extract(&row, Some(&next), &env),
        Ok(RamReadValue(7))
    );
    // Reads write back the read value.
    assert_eq!(
        RamWriteValue::extract(&row, Some(&next), &env),
        Ok(RamWriteValue(7))
    );
    assert_eq!(
        OpFlag::extract_indexed(CircuitFlags::AddOperands, &row, Some(&next), &env),
        Ok(OpFlag(true))
    );
    assert_eq!(
        InstructionFlag::extract_indexed(
            InstructionFlags::RightOperandIsImm,
            &row,
            Some(&next),
            &env
        ),
        Ok(InstructionFlag(true))
    );
    Ok(())
}

#[test]
fn lookahead_witnesses_pad_the_final_cycle() {
    let preprocessing = preprocessing();
    let env = WitnessEnv {
        preprocessing: &preprocessing,
    };
    let row = TraceRow {
        instruction: instruction(0x8000_0000),
        ..Default::default()
    };
    let noop_next = TraceRow::default();

    // A missing successor counts as a no-op for the shift family, exactly
    // like a present no-op successor.
    assert_eq!(NextIsNoop::extract(&row, None, &env), Ok(NextIsNoop(true)));
    assert_eq!(
        NextIsNoop::extract(&row, Some(&noop_next), &env),
        Ok(NextIsNoop(true))
    );
    assert_eq!(
        crate::witnesses::NextPc::extract(&row, None, &env),
        Ok(crate::witnesses::NextPc(0))
    );
    assert_eq!(
        crate::witnesses::NextUnexpandedPc::extract(&row, None, &env),
        Ok(crate::witnesses::NextUnexpandedPc(0))
    );
    // ShouldJump suppresses the jump only for a PRESENT no-op successor: a
    // missing successor does not count as a no-op here (ADDI has no jump
    // flag, so both are false; the semantics are pinned by the oracle-table
    // assertions on real traces).
    assert_eq!(ShouldJump::extract(&row, None, &env), Ok(ShouldJump(false)));
}

fn assert_virtual_values(
    witness: &TraceBackend<'_, OwnedTrace>,
    id: JoltVirtualPolynomial,
    expected: &[u64],
) -> Result<(), String> {
    let actual = materialized_virtual_view(witness, id)?;
    let expected = expected
        .iter()
        .copied()
        .map(Fr::from_u64)
        .collect::<Vec<_>>();
    assert_eq!(actual.as_slice(), expected.as_slice());
    Ok(())
}

fn materialized_virtual_view(
    witness: &TraceBackend<'_, OwnedTrace>,
    id: JoltVirtualPolynomial,
) -> Result<Vec<Fr>, String> {
    <TraceBackend<'_, OwnedTrace> as JoltWitnessOracle<Fr>>::oracle_table(
        witness,
        JoltPolynomialId::Virtual(id),
    )
    .map_err(|error| error.to_string())
}

#[test]
fn rd_inc_streams_register_write_deltas_and_padding() {
    let program = JoltProgram::default();
    let preprocessing = preprocessing();
    let rows = vec![
        TraceRow {
            registers: RegisterState {
                rd: Some(RegisterWrite {
                    register: 1,
                    pre_value: 10,
                    post_value: 4,
                }),
                ..Default::default()
            },
            ..Default::default()
        },
        TraceRow {
            registers: RegisterState {
                rd: Some(RegisterWrite {
                    register: 2,
                    pre_value: 2,
                    post_value: 11,
                }),
                ..Default::default()
            },
            ..Default::default()
        },
    ];
    let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_rows(rows));
    let witness = TraceBackend::new(config().with_log_t(2), inputs);
    let stream_result = witness.committed_stream(JoltCommittedPolynomial::RdInc, 3);
    let mut stream = match stream_result {
        Ok(stream) => stream,
        Err(error) => {
            assert_eq!(
                error,
                WitnessError::UnsupportedView {
                    view: "stream should be supported",
                }
            );
            return;
        }
    };

    assert_eq!(
        stream.next_chunk(),
        Ok(Some(PolynomialChunk::<i128>::I128(vec![-6, 9, 0])))
    );
    assert_eq!(
        stream.next_chunk(),
        Ok(Some(PolynomialChunk::<i128>::I128(vec![0])))
    );
    assert_eq!(stream.next_chunk(), Ok(None::<PolynomialChunk<i128>>));
}

#[test]
fn ram_inc_streams_write_deltas_only() {
    let program = JoltProgram::default();
    let preprocessing = preprocessing();
    let rows = vec![
        TraceRow {
            ram_access: RamAccess::Write(RamWrite {
                address: 10,
                pre_value: 5,
                post_value: 12,
            }),
            ..Default::default()
        },
        TraceRow {
            ram_access: RamAccess::Read(RamRead {
                address: 10,
                value: 12,
            }),
            ..Default::default()
        },
    ];
    let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_rows(rows));
    let witness = TraceBackend::new(config().with_log_t(2), inputs);
    let stream_result = witness.committed_stream(JoltCommittedPolynomial::RamInc, 2);
    let mut stream = match stream_result {
        Ok(stream) => stream,
        Err(error) => {
            assert_eq!(
                error,
                WitnessError::UnsupportedView {
                    view: "stream should be supported",
                }
            );
            return;
        }
    };

    assert_eq!(
        stream.next_chunk(),
        Ok(Some(PolynomialChunk::<i128>::I128(vec![7, 0])))
    );
    assert_eq!(
        stream.next_chunk(),
        Ok(Some(PolynomialChunk::<i128>::I128(vec![0, 0])))
    );
    assert_eq!(stream.next_chunk(), Ok(None::<PolynomialChunk<i128>>));
}

#[test]
fn bytecode_ra_streams_pc_chunks_and_noop_padding() {
    let program = JoltProgram::default();
    let first = instruction(RAM_START_ADDRESS as usize);
    let second = instruction(RAM_START_ADDRESS as usize + 4);
    let bytecode_result =
        BytecodePreprocessing::preprocess(vec![first, second], RAM_START_ADDRESS, RV64IMAC_JOLT);
    assert!(
        bytecode_result.is_ok(),
        "bytecode preprocessing failed: {bytecode_result:?}"
    );
    let Ok(bytecode) = bytecode_result else {
        return;
    };
    let preprocessing = preprocessing_with_bytecode(bytecode);
    let rows = vec![
        TraceRow {
            instruction: first,
            ..Default::default()
        },
        TraceRow {
            instruction: second,
            ..Default::default()
        },
    ];
    let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_rows(rows));
    let witness = TraceBackend::new(config().with_log_t(2), inputs);
    let stream_result = witness.committed_stream(JoltCommittedPolynomial::BytecodeRa(0), 3);
    let mut stream = match stream_result {
        Ok(stream) => stream,
        Err(error) => {
            assert_eq!(
                error,
                WitnessError::UnsupportedView {
                    view: "stream should be supported",
                }
            );
            return;
        }
    };

    assert_eq!(
        stream.next_chunk(),
        Ok(Some(PolynomialChunk::<i128>::OneHot(vec![
            Some(1),
            Some(2),
            Some(0)
        ])))
    );
    assert_eq!(
        stream.next_chunk(),
        Ok(Some(PolynomialChunk::<i128>::OneHot(vec![Some(0)])))
    );
    assert_eq!(stream.next_chunk(), Ok(None::<PolynomialChunk<i128>>));
}

#[test]
fn ram_ra_streams_remapped_address_chunks_and_noop_padding() {
    let program = JoltProgram::default();
    let memory_layout = compact_memory_layout();
    let access_address = memory_layout.stack_end;
    let remapped = memory_layout.remap_word_address(access_address);
    assert_eq!(remapped, Ok(Some(10)));
    let preprocessing = preprocessing_with_memory_layout(memory_layout);
    let rows = vec![
        TraceRow {
            ram_access: RamAccess::Read(RamRead {
                address: access_address,
                value: 12,
            }),
            ..Default::default()
        },
        TraceRow {
            ram_access: RamAccess::NoOp,
            ..Default::default()
        },
    ];
    let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_rows(rows));
    let witness = TraceBackend::new(config().with_log_t(2), inputs);
    let stream_result = witness.committed_stream(JoltCommittedPolynomial::RamRa(1), 4);
    let mut stream = match stream_result {
        Ok(stream) => stream,
        Err(error) => {
            assert_eq!(
                error,
                WitnessError::UnsupportedView {
                    view: "stream should be supported",
                }
            );
            return;
        }
    };

    assert_eq!(
        stream.next_chunk(),
        Ok(Some(PolynomialChunk::<i128>::OneHot(vec![
            Some(10),
            None,
            None,
            None
        ])))
    );
    assert_eq!(stream.next_chunk(), Ok(None::<PolynomialChunk<i128>>));
}

#[test]
fn instruction_ra_streams_lookup_index_chunks_and_noop_padding() {
    let program = JoltProgram::default();
    let preprocessing = preprocessing();
    let mut instruction_row = instruction(RAM_START_ADDRESS as usize);
    instruction_row.operands.imm = -1;
    let rows = vec![TraceRow {
        instruction: instruction_row,
        registers: RegisterState {
            rs1: Some(RegisterRead {
                register: 2,
                value: 10,
            }),
            ..Default::default()
        },
        ..Default::default()
    }];
    let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_rows(rows));
    let witness = TraceBackend::new(config().with_log_t(2), inputs);
    let stream_result = witness.committed_stream(JoltCommittedPolynomial::InstructionRa(15), 2);
    let mut stream = match stream_result {
        Ok(stream) => stream,
        Err(error) => {
            assert_eq!(
                error,
                WitnessError::UnsupportedView {
                    view: "stream should be supported",
                }
            );
            return;
        }
    };

    assert_eq!(
        stream.next_chunk(),
        Ok(Some(PolynomialChunk::<i128>::OneHot(vec![
            Some(1),
            Some(0)
        ])))
    );
    assert_eq!(
        stream.next_chunk(),
        Ok(Some(PolynomialChunk::<i128>::OneHot(vec![
            Some(0),
            Some(0)
        ])))
    );
    assert_eq!(stream.next_chunk(), Ok(None::<PolynomialChunk<i128>>));
}

#[test]
fn committed_batch_stream_preserves_single_pass_core_shape() {
    let program = JoltProgram::default();
    let instruction_row = instruction(RAM_START_ADDRESS as usize);
    let bytecode_result =
        BytecodePreprocessing::preprocess(vec![instruction_row], RAM_START_ADDRESS, RV64IMAC_JOLT);
    assert!(
        bytecode_result.is_ok(),
        "bytecode preprocessing failed: {bytecode_result:?}"
    );
    let Ok(bytecode) = bytecode_result else {
        return;
    };
    let memory_layout = compact_memory_layout();
    let access_address = memory_layout.stack_end;
    let mut preprocessing = preprocessing_with_bytecode(bytecode);
    preprocessing.memory_layout = memory_layout;
    let rows = vec![
        TraceRow {
            instruction: instruction_row,
            registers: RegisterState {
                rs1: Some(RegisterRead {
                    register: 2,
                    value: 10,
                }),
                rd: Some(RegisterWrite {
                    register: 1,
                    pre_value: 4,
                    post_value: 9,
                }),
                ..Default::default()
            },
            ram_access: RamAccess::Write(RamWrite {
                address: access_address,
                pre_value: 7,
                post_value: 11,
            }),
            #[cfg(feature = "field-inline")]
            field_inline: None,
        },
        TraceRow {
            instruction: instruction_row,
            ram_access: RamAccess::NoOp,
            ..Default::default()
        },
    ];
    let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_rows(rows));
    let witness = TraceBackend::new(config().with_log_t(2), inputs);
    let ids = [
        JoltCommittedPolynomial::RdInc,
        JoltCommittedPolynomial::RamInc,
        JoltCommittedPolynomial::InstructionRa(15),
        JoltCommittedPolynomial::BytecodeRa(0),
        JoltCommittedPolynomial::RamRa(1),
    ];
    let stream_result =
        <TraceBackend<'_, OwnedTrace> as JoltWitnessOracle<Fr>>::committed_batch_stream(
            &witness, &ids, 3,
        );
    let mut stream = match stream_result {
        Ok(stream) => stream,
        Err(error) => {
            assert_eq!(
                error,
                WitnessError::UnsupportedView {
                    view: "stream should be supported",
                }
            );
            return;
        }
    };

    let first_result = stream.next_batch();
    assert!(first_result.is_ok(), "first batch failed: {first_result:?}");
    let Ok(Some(first)) = first_result else {
        return;
    };
    assert_eq!(first.len(), 3);
    assert_eq!(
        first.chunks,
        vec![
            (
                JoltCommittedPolynomial::RdInc,
                PolynomialChunk::I128(vec![5, 0, 0])
            ),
            (
                JoltCommittedPolynomial::RamInc,
                PolynomialChunk::I128(vec![4, 0, 0])
            ),
            (
                JoltCommittedPolynomial::InstructionRa(15),
                PolynomialChunk::OneHot(vec![Some(0), Some(0), Some(0)])
            ),
            (
                JoltCommittedPolynomial::BytecodeRa(0),
                PolynomialChunk::OneHot(vec![Some(1), Some(1), Some(0)])
            ),
            (
                JoltCommittedPolynomial::RamRa(1),
                PolynomialChunk::OneHot(vec![Some(10), None, None])
            ),
        ]
    );

    let second_result = stream.next_batch();
    assert!(
        second_result.is_ok(),
        "second batch failed: {second_result:?}"
    );
    let Ok(Some(second)) = second_result else {
        return;
    };
    assert_eq!(
        second.chunks,
        vec![
            (
                JoltCommittedPolynomial::RdInc,
                PolynomialChunk::I128(vec![0])
            ),
            (
                JoltCommittedPolynomial::RamInc,
                PolynomialChunk::I128(vec![0])
            ),
            (
                JoltCommittedPolynomial::InstructionRa(15),
                PolynomialChunk::OneHot(vec![Some(0)])
            ),
            (
                JoltCommittedPolynomial::BytecodeRa(0),
                PolynomialChunk::OneHot(vec![Some(0)])
            ),
            (
                JoltCommittedPolynomial::RamRa(1),
                PolynomialChunk::OneHot(vec![None])
            ),
        ]
    );
    assert_eq!(stream.next_batch(), Ok(None));
}

#[test]
fn advice_streams_pack_device_bytes_as_little_endian_words() {
    let program = JoltProgram::default();
    let preprocessing = preprocessing();
    let device = JoltDevice {
        trusted_advice: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        untrusted_advice: vec![0xaa, 0xbb],
        ..Default::default()
    };
    let inputs =
        JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_device(device));
    let witness = TraceBackend::new(
        config()
            .include_trusted_advice(true)
            .include_untrusted_advice(true),
        inputs,
    );

    let trusted_result = witness.committed_stream(JoltCommittedPolynomial::TrustedAdvice, 3);
    assert!(
        trusted_result.is_ok(),
        "trusted advice stream failed: {trusted_result:?}"
    );
    let Ok(mut trusted) = trusted_result else {
        return;
    };
    assert_eq!(
        trusted.next_chunk(),
        Ok(Some(PolynomialChunk::<i128>::U64(vec![
            0x0807_0605_0403_0201,
            0x0a09,
            0,
        ])))
    );
    assert_eq!(
        trusted.next_chunk(),
        Ok(Some(PolynomialChunk::<i128>::U64(vec![0, 0, 0])))
    );
    assert_eq!(
        trusted.next_chunk(),
        Ok(Some(PolynomialChunk::<i128>::U64(vec![0, 0])))
    );
    assert_eq!(trusted.next_chunk(), Ok(None::<PolynomialChunk<i128>>));

    let untrusted_result = witness.committed_stream(JoltCommittedPolynomial::UntrustedAdvice, 5);
    assert!(
        untrusted_result.is_ok(),
        "untrusted advice stream failed: {untrusted_result:?}"
    );
    let Ok(mut untrusted) = untrusted_result else {
        return;
    };
    assert_eq!(
        untrusted.next_chunk(),
        Ok(Some(PolynomialChunk::<i128>::U64(vec![0xbbaa, 0, 0, 0, 0])))
    );
}

#[test]
fn advice_streams_reject_disabled_and_oversized_advice() {
    let program = JoltProgram::default();
    let preprocessing = preprocessing();
    let device = JoltDevice {
        trusted_advice: vec![0; 65],
        ..Default::default()
    };
    let inputs =
        JoltVmWitnessInputs::new(&program, &preprocessing, trace_output_with_device(device));
    let witness = TraceBackend::new(config().include_trusted_advice(true), inputs);

    assert!(matches!(
        witness.committed_stream(JoltCommittedPolynomial::TrustedAdvice, 1),
        Err(WitnessError::InvalidWitnessData {
            label: "jolt_vm",
            ..
        })
    ));

    let inputs = JoltVmWitnessInputs::new(
        &program,
        &preprocessing,
        trace_output_with_device(Default::default()),
    );
    let disabled = TraceBackend::new(config(), inputs);
    assert!(matches!(
        disabled.committed_stream(JoltCommittedPolynomial::TrustedAdvice, 1),
        Err(WitnessError::UnknownOracle { label: "jolt_vm" })
    ));
}

#[test]
fn committed_batch_stream_rejects_advice_until_variable_length_batches_land() {
    let program = JoltProgram::default();
    let preprocessing = preprocessing();
    let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output());
    let witness = TraceBackend::new(config().include_trusted_advice(true), inputs);

    assert!(matches!(
        witness.committed_batch_stream(&[JoltCommittedPolynomial::TrustedAdvice], 1),
        Err(WitnessError::UnsupportedView {
            view: "batched Jolt VM advice streams",
        })
    ));
}

#[test]
fn committed_stream_rejects_unsupported_oracles_and_empty_chunks() {
    let program = JoltProgram::default();
    let preprocessing = preprocessing();
    let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output());
    let witness = TraceBackend::new(config(), inputs);

    assert!(matches!(
        witness.committed_stream(JoltCommittedPolynomial::TrustedAdvice, 1),
        Err(WitnessError::UnknownOracle { label: "jolt_vm" })
    ));
    assert!(matches!(
        witness.committed_stream(JoltCommittedPolynomial::RdInc, 0),
        Err(WitnessError::InvalidDimensions { .. })
    ));
}

#[test]
fn excluded_ids_report_their_classification() {
    use super::oracle::{
        COMMITTED_PROGRAM_REASON, LATTICE_REASON, PROTOCOL_INTERMEDIATE_REASON, UNSERVED_REASON,
    };
    let program = JoltProgram::default();
    let preprocessing = preprocessing();
    let inputs = JoltVmWitnessInputs::new(&program, &preprocessing, trace_output());
    let witness = TraceBackend::new(config(), inputs);

    let assert_reason = |id: JoltPolynomialId, reason: &'static str| {
        for result in [
            witness.shape_of(id).map(|_| ()),
            <TraceBackend<'_, OwnedTrace> as JoltWitnessOracle<Fr>>::oracle_table(&witness, id)
                .map(|_| ()),
        ] {
            assert_eq!(
                result,
                Err(WitnessError::NotServed {
                    oracle: format!("{id:?}"),
                    reason,
                }),
                "classification mismatch for {id:?}"
            );
        }
    };

    for id in [
        JoltCommittedPolynomial::BytecodeChunk(0),
        JoltCommittedPolynomial::ProgramImageInit,
    ] {
        assert_reason(JoltPolynomialId::Committed(id), COMMITTED_PROGRAM_REASON);
    }
    for id in [
        JoltCommittedPolynomial::UnsignedIncChunk(0),
        JoltCommittedPolynomial::UnsignedIncMsb,
        JoltCommittedPolynomial::TrustedAdviceBytes,
        JoltCommittedPolynomial::UntrustedAdviceBytes,
        JoltCommittedPolynomial::BytecodeLookupSelector { chunk: 0 },
        JoltCommittedPolynomial::ProgramImageBytes,
    ] {
        assert_reason(JoltPolynomialId::Committed(id), LATTICE_REASON);
    }
    for id in [
        JoltVirtualPolynomial::Rd,
        JoltVirtualPolynomial::InstructionRaf,
        JoltVirtualPolynomial::RamValInit,
    ] {
        assert_reason(JoltPolynomialId::Virtual(id), UNSERVED_REASON);
    }
    for id in [
        JoltVirtualPolynomial::UnivariateSkip,
        JoltVirtualPolynomial::BytecodeValStage(0),
        JoltVirtualPolynomial::BytecodeReadRafAddrClaim,
        JoltVirtualPolynomial::BooleanityAddrClaim,
        JoltVirtualPolynomial::BytecodeClaimReductionIntermediate,
        JoltVirtualPolynomial::ProgramImageInitContributionRw,
    ] {
        assert_reason(JoltPolynomialId::Virtual(id), PROTOCOL_INTERMEDIATE_REASON);
    }
    assert_reason(
        JoltPolynomialId::Virtual(JoltVirtualPolynomial::FusedInc),
        LATTICE_REASON,
    );
}
