//! Sample-trace fixtures for the derive-generated bundle consistency tests.

use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltPolynomialId};
use jolt_field::Fr;
use jolt_program::{
    execution::{
        JoltProgram, OwnedTrace, RamAccess, RamWrite, RegisterRead, RegisterState, RegisterWrite,
        TraceOutput, TraceRow,
    },
    preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing},
};
use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RV64IMAC_JOLT};

use crate::backend::trace::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
use crate::{BundleSource, JoltWitnessOracle, WitnessBundle};

/// Runs `f` against a small canned backend: an ADDI and a store, padded to `2^2`.
pub fn with_sample_backend<R>(f: impl FnOnce(&TraceBackend<'_, OwnedTrace>) -> R) -> R {
    with_sample_backend_at_log_t(2, 4, f)
}

/// Runs the canned trace against a caller-selected padded cycle domain.
pub fn with_sample_backend_at_log_t<R>(
    log_t: usize,
    log_k_chunk: u8,
    f: impl FnOnce(&TraceBackend<'_, OwnedTrace>) -> R,
) -> R {
    with_sample_backend_at_geometry(log_t, 2, log_k_chunk, f)
}

/// Runs the canned trace with caller-selected cycle and bytecode domains.
#[expect(clippy::unwrap_used, reason = "test fixture construction")]
pub fn with_sample_backend_at_geometry<R>(
    log_t: usize,
    log_k: usize,
    log_k_chunk: u8,
    f: impl FnOnce(&TraceBackend<'_, OwnedTrace>) -> R,
) -> R {
    let instruction = JoltInstructionRow {
        instruction_kind: JoltInstructionKind::ADDI,
        address: 0x8000_0000,
        operands: NormalizedOperands {
            rd: Some(1),
            rs1: Some(2),
            rs2: None,
            imm: 3,
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: false,
    };
    let store = JoltInstructionRow {
        instruction_kind: JoltInstructionKind::SD,
        address: instruction.address + 4,
        operands: NormalizedOperands {
            rd: None,
            rs1: Some(2),
            rs2: Some(3),
            imm: 0,
        },
        ..Default::default()
    };
    let mut bytecode = BytecodePreprocessing::preprocess(
        vec![instruction, store],
        instruction.address as u64,
        RV64IMAC_JOLT,
    )
    .unwrap();
    let bytecode_rows = 1usize.checked_shl(log_k as u32).unwrap();
    assert!(bytecode_rows >= bytecode.bytecode.len());
    let padding = *bytecode.bytecode.last().unwrap();
    bytecode.bytecode.resize(bytecode_rows, padding);
    bytecode.code_size = bytecode_rows;
    let preprocessing = JoltProgramPreprocessing {
        bytecode,
        ram: RAMPreprocessing::default(),
        memory_layout: Default::default(),
        max_padded_trace_length: 4.max(1usize << log_t),
    };
    let rows = vec![
        TraceRow {
            instruction,
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
            ram_access: RamAccess::NoOp,
            #[cfg(feature = "field-inline")]
            field_inline: None,
        },
        TraceRow {
            instruction: store,
            registers: RegisterState {
                rs1: Some(RegisterRead {
                    register: 2,
                    value: 0x8000_1008,
                }),
                rs2: Some(RegisterRead {
                    register: 3,
                    value: 11,
                }),
                ..Default::default()
            },
            ram_access: RamAccess::Write(RamWrite {
                address: 0x8000_1008,
                pre_value: 7,
                post_value: 11,
            }),
            #[cfg(feature = "field-inline")]
            field_inline: None,
        },
    ];
    with_backend(log_t, log_k_chunk, preprocessing, rows, f)
}

/// Runs a compact but address-diverse trace over a full bytecode domain.
#[expect(clippy::unwrap_used, reason = "test fixture construction")]
pub fn with_diverse_sample_backend_at_geometry<R>(
    log_t: usize,
    log_k: usize,
    log_k_chunk: u8,
    f: impl FnOnce(&TraceBackend<'_, OwnedTrace>) -> R,
) -> R {
    let bytecode_rows = 1usize.checked_shl(log_k as u32).unwrap();
    assert!(log_t >= 1 && bytecode_rows >= 4);
    let instructions = (0..bytecode_rows - 1)
        .map(|index| {
            let address = 0x8000_0000usize + 4 * index;
            if index % 4 == 3 {
                JoltInstructionRow {
                    instruction_kind: JoltInstructionKind::SD,
                    address,
                    operands: NormalizedOperands {
                        rd: None,
                        rs1: Some(2),
                        rs2: Some(3),
                        imm: 0,
                    },
                    ..Default::default()
                }
            } else {
                let magnitude = (index % 31 + 1) as i128;
                JoltInstructionRow {
                    instruction_kind: JoltInstructionKind::ADDI,
                    address,
                    operands: NormalizedOperands {
                        rd: Some(1),
                        rs1: Some(2),
                        rs2: None,
                        imm: if index % 8 >= 4 {
                            -magnitude
                        } else {
                            magnitude
                        },
                    },
                    virtual_sequence_remaining: None,
                    is_first_in_sequence: false,
                    is_compressed: false,
                }
            }
        })
        .collect::<Vec<_>>();
    let bytecode = BytecodePreprocessing::preprocess(
        instructions.clone(),
        instructions[0].address as u64,
        RV64IMAC_JOLT,
    )
    .unwrap();
    assert_eq!(bytecode.bytecode.len(), bytecode_rows);
    let preprocessing = JoltProgramPreprocessing {
        bytecode,
        ram: RAMPreprocessing::default(),
        memory_layout: Default::default(),
        max_padded_trace_length: 4.max(1usize << log_t),
    };
    let trace_rows = (1usize << log_t).min(instructions.len());
    let rows = (0..trace_rows)
        .map(|cycle| {
            let index = cycle * (instructions.len() - 1) / (trace_rows - 1);
            diverse_trace_row(instructions[index], index)
        })
        .collect();
    with_backend(log_t, log_k_chunk, preprocessing, rows, f)
}

fn diverse_trace_row(instruction: JoltInstructionRow, index: usize) -> TraceRow {
    let delta = (index % 31 + 1) as u64;
    let base = 1_000 + 64 * index as u64;
    let post_value = if index % 8 >= 4 {
        base - delta
    } else {
        base + delta
    };
    if instruction.instruction_kind == JoltInstructionKind::SD {
        TraceRow {
            instruction,
            registers: RegisterState {
                rs1: Some(RegisterRead {
                    register: 2,
                    value: 0x8000_1008,
                }),
                rs2: Some(RegisterRead {
                    register: 3,
                    value: post_value,
                }),
                ..Default::default()
            },
            ram_access: RamAccess::Write(RamWrite {
                address: 0x8000_1008,
                pre_value: base,
                post_value,
            }),
            #[cfg(feature = "field-inline")]
            field_inline: None,
        }
    } else {
        TraceRow {
            instruction,
            registers: RegisterState {
                rs1: Some(RegisterRead {
                    register: 2,
                    value: base,
                }),
                rd: Some(RegisterWrite {
                    register: 1,
                    pre_value: base,
                    post_value,
                }),
                ..Default::default()
            },
            ram_access: RamAccess::NoOp,
            #[cfg(feature = "field-inline")]
            field_inline: None,
        }
    }
}

fn with_backend<R>(
    log_t: usize,
    log_k_chunk: u8,
    preprocessing: JoltProgramPreprocessing,
    rows: Vec<TraceRow>,
    f: impl FnOnce(&TraceBackend<'_, OwnedTrace>) -> R,
) -> R {
    let program = JoltProgram::default();
    let config = JoltVmWitnessConfig::new(
        log_t,
        64,
        JoltOneHotConfig {
            log_k_chunk,
            lookups_ra_virtual_log_k_chunk: 16,
        },
    );
    let inputs = JoltVmWitnessInputs::new(
        &program,
        &preprocessing,
        TraceOutput::new(OwnedTrace::new(rows), Default::default(), None),
    );
    let backend = TraceBackend::new(config, inputs);
    f(&backend)
}

/// Asserts that one annotated bundle field's column (extracted by `value`)
/// equals the backend's `oracle_table` for `id` — the typed path and the id
/// path meeting at the `Extract` impls. Driven by the derive-generated
/// per-field consistency tests.
#[expect(clippy::unwrap_used, reason = "test assertion helper")]
pub fn assert_bundle_column_matches<B>(id: JoltPolynomialId, value: impl Fn(&B) -> Fr)
where
    B: WitnessBundle + Clone + Send + Sync,
{
    with_sample_backend(|backend| {
        assert!(
            B::annotated_ids().contains(&id),
            "{id:?} is not in the bundle's annotated id set"
        );
        let rows: Vec<B> = backend.bundles().unwrap();
        let column: Vec<Fr> = rows.iter().map(value).collect();
        let table = JoltWitnessOracle::<Fr>::oracle_table(backend, id).unwrap();
        assert_eq!(
            column, table,
            "bundle column diverges from oracle_table for {id:?}"
        );
    });
}

#[cfg(test)]
mod tests {
    use jolt_claims::protocols::jolt::JoltVirtualPolynomial;
    use jolt_field::FromPrimitiveInt;

    use super::*;

    #[test]
    #[expect(clippy::unwrap_used, reason = "test assertion")]
    fn diverse_fixture_covers_both_pc_chunks_and_increment_signs() {
        with_diverse_sample_backend_at_geometry(9, 9, 8, |backend| {
            let pcs =
                JoltWitnessOracle::<Fr>::oracle_table(backend, JoltVirtualPolynomial::PC.into())
                    .unwrap();
            let increments = JoltWitnessOracle::<Fr>::oracle_table(
                backend,
                JoltVirtualPolynomial::FusedInc.into(),
            )
            .unwrap();

            assert!(pcs.contains(&Fr::from_u64(511)));
            assert!(increments.contains(&Fr::from_u64(1)));
            assert!(increments.contains(&-Fr::from_u64(5)));
        });
    }
}
