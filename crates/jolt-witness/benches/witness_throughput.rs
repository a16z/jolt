#![expect(unused_results)]
#![expect(clippy::unwrap_used)]

use common::constants::RAM_START_ADDRESS;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltOneHotConfig};
use jolt_field::Fr;
use jolt_program::{
    execution::{
        JoltProgram, OwnedTrace, RamAccess, RamWrite, RegisterRead, RegisterState, RegisterWrite,
        TraceOutput, TraceRow,
    },
    preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing},
};
use jolt_riscv::{
    JoltInstructionKind, JoltInstructionProfile, JoltInstructionRow, NormalizedOperands,
    RV64IMAC_JOLT,
};
use jolt_witness::{
    protocols::jolt_vm::{JoltVmNamespace, JoltVmWitnessConfig, JoltVmWitnessInputs},
    PolynomialBatchStream, PolynomialStream, WitnessProvider,
};

#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{
    FieldInlineCommittedPolynomial, FieldInlineOpFlag, FieldInlineVirtualPolynomial,
};
#[cfg(feature = "field-inline")]
use jolt_program::field_inline::{
    FieldEncodedValue, FieldInlineTraceData, FieldRegisterRead, FieldRegisterWrite,
};
#[cfg(feature = "field-inline")]
use jolt_riscv::{FieldInlineOp, RV64IMAC_JOLT_FIELD_INLINE};
#[cfg(feature = "field-inline")]
use jolt_witness::protocols::jolt_vm::field_inline::FieldInlineNamespace;
#[cfg(feature = "field-inline")]
use jolt_witness::{OracleRef, OracleViewRequest, PolynomialView};

const ENTRY: u64 = RAM_START_ADDRESS;
const CHUNK_SIZE: usize = 1 << 12;
const ROW_COUNTS: [usize; 3] = [1 << 12, 1 << 16, 1 << 18];

fn bench_rv64_streams(c: &mut Criterion) {
    let mut group = c.benchmark_group("jolt_witness/rv64_streams");
    for row_count in ROW_COUNTS {
        let fixture = Rv64Fixture::new(row_count);
        group.throughput(Throughput::Elements(row_count as u64));

        group.bench_with_input(
            BenchmarkId::new("rd_inc", row_count),
            &fixture,
            |bench, fixture| {
                bench.iter_batched(
                    || fixture.witness(),
                    |witness| {
                        let mut stream =
                            <_ as WitnessProvider<Fr, JoltVmNamespace>>::committed_stream(
                                &witness,
                                JoltCommittedPolynomial::RdInc,
                                CHUNK_SIZE,
                            )
                            .unwrap();
                        drain_stream(stream.as_mut())
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ram_inc", row_count),
            &fixture,
            |bench, fixture| {
                bench.iter_batched(
                    || fixture.witness(),
                    |witness| {
                        let mut stream =
                            <_ as WitnessProvider<Fr, JoltVmNamespace>>::committed_stream(
                                &witness,
                                JoltCommittedPolynomial::RamInc,
                                CHUNK_SIZE,
                            )
                            .unwrap();
                        drain_stream(stream.as_mut())
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("instruction_ra_chunk", row_count),
            &fixture,
            |bench, fixture| {
                bench.iter_batched(
                    || fixture.witness(),
                    |witness| {
                        let mut stream =
                            <_ as WitnessProvider<Fr, JoltVmNamespace>>::committed_stream(
                                &witness,
                                JoltCommittedPolynomial::InstructionRa(15),
                                CHUNK_SIZE,
                            )
                            .unwrap();
                        drain_stream(stream.as_mut())
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bytecode_ra_chunk", row_count),
            &fixture,
            |bench, fixture| {
                bench.iter_batched(
                    || fixture.witness(),
                    |witness| {
                        let mut stream =
                            <_ as WitnessProvider<Fr, JoltVmNamespace>>::committed_stream(
                                &witness,
                                JoltCommittedPolynomial::BytecodeRa(0),
                                CHUNK_SIZE,
                            )
                            .unwrap();
                        drain_stream(stream.as_mut())
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("committed_batch_all", row_count),
            &fixture,
            |bench, fixture| {
                bench.iter_batched(
                    || fixture.witness(),
                    |witness| {
                        let order = witness.committed_polynomial_order().unwrap();
                        let mut stream =
                            <_ as WitnessProvider<Fr, JoltVmNamespace>>::committed_batch_stream(
                                &witness, &order, CHUNK_SIZE,
                            )
                            .unwrap();
                        drain_batch_stream(stream.as_mut())
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

#[cfg(feature = "field-inline")]
fn bench_field_inline(c: &mut Criterion) {
    let mut group = c.benchmark_group("jolt_witness/field_inline");
    for row_count in ROW_COUNTS {
        let fixture = FieldInlineFixture::new(row_count);
        group.throughput(Throughput::Elements(row_count as u64));

        group.bench_with_input(
            BenchmarkId::new("field_rd_inc", row_count),
            &fixture,
            |bench, fixture| {
                bench.iter_batched(
                    || fixture.witness(),
                    |witness| {
                        let provider = witness.field_inline_witness().unwrap();
                        let mut stream =
                            <_ as WitnessProvider<Fr, FieldInlineNamespace>>::committed_stream(
                                &provider,
                                FieldInlineCommittedPolynomial::FieldRdInc,
                                CHUNK_SIZE,
                            )
                            .unwrap();
                        drain_stream(stream.as_mut())
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("field_product_view", row_count),
            &fixture,
            |bench, fixture| {
                bench.iter_batched(
                    || fixture.witness(),
                    |witness| {
                        let provider = witness.field_inline_witness().unwrap();
                        materialized_view_len(
                            &provider,
                            OracleRef::virtual_polynomial(
                                FieldInlineVirtualPolynomial::FieldProduct,
                            ),
                        )
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("field_mul_flag_view", row_count),
            &fixture,
            |bench, fixture| {
                bench.iter_batched(
                    || fixture.witness(),
                    |witness| {
                        let provider = witness.field_inline_witness().unwrap();
                        materialized_view_len(
                            &provider,
                            OracleRef::virtual_polynomial(
                                FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::Mul),
                            ),
                        )
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

        group.throughput(Throughput::Elements(
            (row_count * FIELD_REGISTER_COUNT) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::new("field_registers_val_view", row_count),
            &fixture,
            |bench, fixture| {
                bench.iter_batched(
                    || fixture.witness(),
                    |witness| {
                        let provider = witness.field_inline_witness().unwrap();
                        materialized_view_len(
                            &provider,
                            OracleRef::virtual_polynomial(
                                FieldInlineVirtualPolynomial::FieldRegistersVal,
                            ),
                        )
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

#[cfg(not(feature = "field-inline"))]
fn bench_field_inline(_c: &mut Criterion) {}

struct Rv64Fixture {
    program: JoltProgram,
    preprocessing: JoltProgramPreprocessing,
    rows: Vec<TraceRow>,
    log_t: usize,
}

impl Rv64Fixture {
    fn new(row_count: usize) -> Self {
        let instruction = rv64_instruction();
        let program = program(vec![instruction], RV64IMAC_JOLT);
        let preprocessing = preprocessing(vec![instruction], RV64IMAC_JOLT);
        let rows = (0..row_count)
            .map(|index| TraceRow {
                instruction,
                registers: RegisterState {
                    rs1: Some(RegisterRead {
                        register: 2,
                        value: index as u64,
                    }),
                    rd: Some(RegisterWrite {
                        register: 1,
                        pre_value: index as u64,
                        post_value: index as u64 + 1,
                    }),
                    ..RegisterState::default()
                },
                ram_access: RamAccess::Write(RamWrite {
                    address: ENTRY + ((index % 1024) as u64) * 8,
                    pre_value: index as u64,
                    post_value: index as u64 + 1,
                }),
                field_inline: None,
            })
            .collect();

        Self {
            program,
            preprocessing,
            rows,
            log_t: row_count.ilog2() as usize,
        }
    }

    fn witness(
        &self,
    ) -> jolt_witness::protocols::jolt_vm::TraceBackedJoltVmWitness<'_, OwnedTrace> {
        jolt_witness::protocols::jolt_vm::TraceBackedJoltVmWitness::new(
            config(self.log_t),
            JoltVmWitnessInputs::new(
                &self.program,
                &self.preprocessing,
                TraceOutput::new(OwnedTrace::new(self.rows.clone()), Default::default(), None),
            ),
        )
    }
}

#[cfg(feature = "field-inline")]
struct FieldInlineFixture {
    program: JoltProgram,
    preprocessing: JoltProgramPreprocessing,
    rows: Vec<TraceRow>,
    log_t: usize,
}

#[cfg(feature = "field-inline")]
impl FieldInlineFixture {
    fn new(row_count: usize) -> Self {
        let load_a = field_load_imm(0, 1, 1);
        let load_b = field_load_imm(1, 2, 1);
        let mul = field_mul(2, 3, 1, 2);
        let bytecode = vec![load_a, load_b, mul];
        let program = program(bytecode.clone(), RV64IMAC_JOLT_FIELD_INLINE);
        let preprocessing = preprocessing(bytecode, RV64IMAC_JOLT_FIELD_INLINE);
        let rows = (0..row_count)
            .map(|index| match index {
                0 => TraceRow {
                    instruction: load_a,
                    field_inline: Some(FieldInlineTraceData {
                        op: Some(FieldInlineOp::LoadImm),
                        rd: Some(FieldRegisterWrite {
                            register: 1,
                            pre_value: field_value(0),
                            post_value: field_value(1),
                        }),
                        ..FieldInlineTraceData::default()
                    }),
                    ..TraceRow::default()
                },
                1 => TraceRow {
                    instruction: load_b,
                    field_inline: Some(FieldInlineTraceData {
                        op: Some(FieldInlineOp::LoadImm),
                        rd: Some(FieldRegisterWrite {
                            register: 2,
                            pre_value: field_value(0),
                            post_value: field_value(1),
                        }),
                        ..FieldInlineTraceData::default()
                    }),
                    ..TraceRow::default()
                },
                2 => field_mul_row(mul, 0),
                _ => field_mul_row(mul, 1),
            })
            .collect();

        Self {
            program,
            preprocessing,
            rows,
            log_t: row_count.ilog2() as usize,
        }
    }

    fn witness(
        &self,
    ) -> jolt_witness::protocols::jolt_vm::TraceBackedJoltVmWitness<'_, OwnedTrace> {
        jolt_witness::protocols::jolt_vm::TraceBackedJoltVmWitness::new(
            config(self.log_t),
            JoltVmWitnessInputs::new(
                &self.program,
                &self.preprocessing,
                TraceOutput::new(OwnedTrace::new(self.rows.clone()), Default::default(), None),
            ),
        )
    }
}

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

fn preprocessing(
    bytecode: Vec<JoltInstructionRow>,
    profile: JoltInstructionProfile,
) -> JoltProgramPreprocessing {
    JoltProgramPreprocessing {
        bytecode: BytecodePreprocessing::preprocess(bytecode, ENTRY, profile).unwrap(),
        ram: RAMPreprocessing::default(),
        memory_layout: Default::default(),
        max_padded_trace_length: 1 << 18,
    }
}

fn rv64_instruction() -> JoltInstructionRow {
    JoltInstructionRow {
        instruction_kind: JoltInstructionKind::ADDI,
        address: ENTRY as usize,
        operands: NormalizedOperands {
            rd: Some(1),
            rs1: Some(2),
            rs2: None,
            imm: -1,
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: false,
    }
}

#[cfg(feature = "field-inline")]
fn field_load_imm(offset: usize, rd: u8, value: u64) -> JoltInstructionRow {
    JoltInstructionRow {
        instruction_kind: JoltInstructionKind::FIELD_LOAD_IMM,
        address: ENTRY as usize + offset * 4,
        operands: NormalizedOperands {
            rd: Some(rd),
            rs1: None,
            rs2: None,
            imm: i128::from(value),
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: false,
    }
}

#[cfg(feature = "field-inline")]
fn field_mul(offset: usize, rd: u8, rs1: u8, rs2: u8) -> JoltInstructionRow {
    JoltInstructionRow {
        instruction_kind: JoltInstructionKind::FIELD_MUL,
        address: ENTRY as usize + offset * 4,
        operands: NormalizedOperands {
            rd: Some(rd),
            rs1: Some(rs1),
            rs2: Some(rs2),
            imm: 0,
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: false,
    }
}

#[cfg(feature = "field-inline")]
fn field_mul_row(instruction: JoltInstructionRow, rd_pre_value: u64) -> TraceRow {
    TraceRow {
        instruction,
        field_inline: Some(FieldInlineTraceData {
            op: Some(FieldInlineOp::Mul),
            rs1: Some(FieldRegisterRead {
                register: 1,
                value: field_value(1),
            }),
            rs2: Some(FieldRegisterRead {
                register: 2,
                value: field_value(1),
            }),
            rd: Some(FieldRegisterWrite {
                register: 3,
                pre_value: field_value(rd_pre_value),
                post_value: field_value(1),
            }),
            product: Some(field_value(1)),
            ..FieldInlineTraceData::default()
        }),
        ..TraceRow::default()
    }
}

#[cfg(feature = "field-inline")]
fn field_value(value: u64) -> FieldEncodedValue {
    FieldEncodedValue::from_u64(value)
}

fn drain_stream<F>(stream: &mut dyn PolynomialStream<F>) -> usize {
    let mut rows = 0usize;
    while let Some(chunk) = stream.next_chunk().unwrap() {
        rows += std::hint::black_box(chunk.len());
    }
    std::hint::black_box(rows)
}

fn drain_batch_stream<F>(stream: &mut dyn PolynomialBatchStream<F, JoltVmNamespace>) -> usize {
    let mut rows = 0usize;
    while let Some(batch) = stream.next_batch().unwrap() {
        rows += std::hint::black_box(batch.len());
        std::hint::black_box(batch.chunks.len());
    }
    std::hint::black_box(rows)
}

#[cfg(feature = "field-inline")]
fn materialized_view_len(
    provider: &jolt_witness::protocols::jolt_vm::field_inline::TraceBackedFieldInlineWitness<
        '_,
        '_,
        OwnedTrace,
    >,
    oracle: OracleRef<FieldInlineNamespace>,
) -> usize {
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
    std::hint::black_box(view.len())
}

#[cfg(feature = "field-inline")]
const FIELD_REGISTER_COUNT: usize = 1 << 4;

criterion_group!(benches, bench_rv64_streams, bench_field_inline);
criterion_main!(benches);
