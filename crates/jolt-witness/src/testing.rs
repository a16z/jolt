use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltPolynomialId};
use jolt_field::Fr;
use jolt_program::{
    execution::{JoltProgram, RowReader, RowReplay, TraceData, TraceError, TraceOutput},
    preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing},
};
use jolt_riscv::{
    JoltInstructionKind, JoltInstructionRow, JoltTraceRow, NormalizedOperands, RamAccess, RamWrite,
    RegisterRead, RegisterState, RegisterWrite, RV64IMAC_JOLT,
};
use std::ops::Range;
use std::sync::{Arc, Mutex};

use crate::backend::trace::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
use crate::{BundleSource, JoltWitnessOracle, WitnessBundle};

/// Runs `f` against a small canned backend: an ADDI and a store, padded to `2^2`.
#[expect(clippy::unwrap_used, reason = "test fixture construction")]
pub fn with_sample_backend<R>(f: impl FnOnce(&TraceBackend) -> R) -> R {
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
    let preprocessing = Arc::new(JoltProgramPreprocessing {
        bytecode: BytecodePreprocessing::preprocess(
            vec![instruction, store],
            instruction.address as u64,
            RV64IMAC_JOLT,
        )
        .unwrap(),
        ram: RAMPreprocessing::default(),
        memory_layout: Default::default(),
        max_padded_trace_length: 4,
    });
    let program = Arc::new(JoltProgram::default());
    let rows = vec![
        JoltTraceRow::new(
            instruction,
            RegisterState {
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
            RamAccess::NoOp,
            u32::try_from(preprocessing.bytecode.get_pc(&instruction).unwrap()).unwrap(),
        )
        .unwrap(),
        JoltTraceRow::new(
            store,
            RegisterState {
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
            RamAccess::Write(RamWrite {
                address: 0x8000_1008,
                pre_value: 7,
                post_value: 11,
            }),
            u32::try_from(preprocessing.bytecode.get_pc(&store).unwrap()).unwrap(),
        )
        .unwrap(),
    ];
    let config = JoltVmWitnessConfig::new(
        2,
        64,
        JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        },
    );
    let inputs = JoltVmWitnessInputs::new(
        &program,
        &preprocessing,
        TraceOutput::new(TraceData::new(rows), Default::default(), None, None),
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

/// A [`RowReplay`] serving retained rows, for tests of replay-backed
/// witnesses: readers log their opens and reads, a read past the proof prefix
/// fails, and a read covering `fail_row` fails like a diverged replay.
pub struct LoggedReplay {
    rows: TraceData,
    segment_rows: usize,
    fail_row: Option<usize>,
    log: Mutex<Vec<ReplayEvent>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayEvent {
    Open(usize),
    Read(Range<usize>),
}

impl LoggedReplay {
    /// Replays the proof prefix of `rows` (trailing canonical padding is
    /// trimmed exactly as [`TraceData::new`] trims it).
    pub fn new(rows: Vec<JoltTraceRow>, segment_rows: usize) -> Self {
        Self {
            rows: TraceData::new(rows),
            segment_rows,
            fail_row: None,
            log: Mutex::new(Vec::new()),
        }
    }

    pub fn failing_at(mut self, row: usize) -> Self {
        self.fail_row = Some(row);
        self
    }

    #[expect(clippy::unwrap_used, reason = "test helper")]
    pub fn take_log(&self) -> Vec<ReplayEvent> {
        std::mem::take(&mut *self.log.lock().unwrap())
    }

    #[expect(clippy::unwrap_used, reason = "test helper")]
    fn record(&self, event: ReplayEvent) {
        self.log.lock().unwrap().push(event);
    }
}

impl RowReplay for LoggedReplay {
    fn segment_rows(&self) -> usize {
        self.segment_rows
    }

    fn proof_len(&self) -> usize {
        self.rows.proof_len()
    }

    fn reader(&self, start: usize) -> Result<Box<dyn RowReader + '_>, TraceError> {
        self.record(ReplayEvent::Open(start));
        Ok(Box::new(LoggedReader {
            replay: self,
            next: start,
        }))
    }
}

struct LoggedReader<'a> {
    replay: &'a LoggedReplay,
    next: usize,
}

impl RowReader for LoggedReader<'_> {
    fn read(&mut self, rows: &mut [JoltTraceRow]) -> Result<(), TraceError> {
        let range = self.next..self.next + rows.len();
        self.replay.record(ReplayEvent::Read(range.clone()));
        let Some(source) = self.replay.rows.proof_rows().get(range.clone()) else {
            return Err(TraceError::ReplayOutOfRange {
                start: range.start,
                end: range.end,
                extent: self.replay.rows.proof_len(),
            });
        };
        if let Some(row) = self.replay.fail_row.filter(|row| range.contains(row)) {
            return Err(TraceError::ReplayDiverged {
                row,
                reason: "injected failure".to_owned(),
            });
        }
        rows.copy_from_slice(source);
        self.next = range.end;
        Ok(())
    }
}
