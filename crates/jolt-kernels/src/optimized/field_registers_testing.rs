//! FR-profile trace fixtures for the optimized field-registers kernels'
//! parity tests: register-consistent field-inline executions behind a full
//! `TraceBackend` witness plane with the field-inline view attached (the
//! [`super::registers_read_write::test_support::TraceFixture`] discipline at
//! the FR instruction family).
//!
//! Reads return the running FR register-file state and writes advance it, so
//! the witness view's build-time replay validation holds by construction.
//! Bridge ops (`FIELD_LOAD_FROM_X`/`FIELD_STORE_TO_X`) are deliberately not
//! modeled — their payloads couple to the x-register file, and the FR kernel
//! surface under test never distinguishes bridge writes from ordinary ones
//! (the e2e's eq-MLE guest covers them at the proof level).

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test support module: fail loudly"
)]

use std::sync::Arc;

use common::constants::RAM_START_ADDRESS;
use jolt_claims::protocols::jolt::JoltOneHotConfig;
use jolt_field::{CanonicalBytes, Fr, Ring};
use jolt_program::execution::{JoltProgram, OwnedTrace, TraceOutput, TraceRow};
use jolt_program::field_inline::{
    FieldEncodedValue, FieldInlineTraceData, FieldRegisterRead, FieldRegisterWrite,
};
use jolt_program::preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing};
use jolt_riscv::{
    FieldInlineOp, JoltInstructionKind, JoltInstructionRow, NormalizedOperands,
    RV64IMAC_JOLT_FIELD_INLINE,
};
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

const ENTRY: u64 = RAM_START_ADDRESS;

fn encode(value: Fr) -> FieldEncodedValue {
    let mut bytes_le = [0u8; 32];
    value.to_bytes_le(&mut bytes_le);
    FieldEncodedValue { bytes_le }
}

/// A register-consistent FR trace builder over the 16-slot FR register file.
pub(crate) struct FrTraceFixture {
    rows: Vec<TraceRow>,
    bytecode: Vec<JoltInstructionRow>,
    state: [Fr; 16],
    counter: u64,
}

impl FrTraceFixture {
    pub(crate) fn new() -> Self {
        Self {
            rows: Vec::new(),
            bytecode: Vec::new(),
            state: [Fr::from_u64(0); 16],
            counter: 0x0DDF_00D5_EED0_25EC,
        }
    }

    fn instruction(
        &mut self,
        kind: JoltInstructionKind,
        rd: Option<u8>,
        rs1: Option<u8>,
        rs2: Option<u8>,
        imm: i128,
    ) -> JoltInstructionRow {
        let instruction = JoltInstructionRow {
            instruction_kind: kind,
            address: ENTRY as usize + self.bytecode.len() * 4,
            operands: NormalizedOperands { rd, rs1, rs2, imm },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        self.bytecode.push(instruction);
        instruction
    }

    /// A fresh pseudo-random full-width field value (squaring pushes the
    /// value past the u64 range, exercising real field arithmetic).
    fn fresh_value(&mut self) -> Fr {
        self.counter = self
            .counter
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let seed = Fr::from_u64(self.counter | 1);
        seed * seed + seed
    }

    fn read(&self, register: u8) -> FieldRegisterRead {
        FieldRegisterRead {
            register,
            value: encode(self.state[usize::from(register)]),
        }
    }

    fn write(&mut self, register: u8, post: Fr) -> FieldRegisterWrite {
        let pre = self.state[usize::from(register)];
        self.state[usize::from(register)] = post;
        FieldRegisterWrite {
            register,
            pre_value: encode(pre),
            post_value: encode(post),
        }
    }

    /// An ordinary (FR-inactive) row: an ADDI with no register traffic.
    pub(crate) fn noop(&mut self) {
        let instruction = self.instruction(JoltInstructionKind::ADDI, Some(1), Some(0), None, 0);
        self.rows.push(TraceRow {
            instruction,
            ..TraceRow::default()
        });
    }

    pub(crate) fn load_imm(&mut self, rd: u8, imm: u64) {
        let instruction =
            self.instruction(JoltInstructionKind::FIELD_LOAD_IMM, Some(rd), None, None, {
                imm as i128
            });
        let rd = self.write(rd, Fr::from_u64(imm));
        self.rows.push(TraceRow {
            instruction,
            field_inline: Some(Arc::new(FieldInlineTraceData {
                op: Some(FieldInlineOp::LoadImm),
                rd: Some(rd),
                ..FieldInlineTraceData::default()
            })),
            ..TraceRow::default()
        });
    }

    /// One FR arithmetic row (`Add`/`Sub`/`Mul`): reads both operands off the
    /// running state, writes a fresh pseudo-random destination value plus the
    /// op-required product payload.
    pub(crate) fn arithmetic(&mut self, op: FieldInlineOp, rd: u8, rs1: u8, rs2: u8) {
        let kind = match op {
            FieldInlineOp::Add => JoltInstructionKind::FIELD_ADD,
            FieldInlineOp::Sub => JoltInstructionKind::FIELD_SUB,
            FieldInlineOp::Mul => JoltInstructionKind::FIELD_MUL,
            _ => panic!("arithmetic fixture rows are Add/Sub/Mul only"),
        };
        let instruction = self.instruction(kind, Some(rd), Some(rs1), Some(rs2), 0);
        let product = (op == FieldInlineOp::Mul)
            .then(|| encode(self.state[usize::from(rs1)] * self.state[usize::from(rs2)]));
        let rs1 = self.read(rs1);
        let rs2 = self.read(rs2);
        let post = self.fresh_value();
        let rd = self.write(rd, post);
        self.rows.push(TraceRow {
            instruction,
            field_inline: Some(Arc::new(FieldInlineTraceData {
                op: Some(op),
                rs1: Some(rs1),
                rs2: Some(rs2),
                rd: Some(rd),
                product,
                ..FieldInlineTraceData::default()
            })),
            ..TraceRow::default()
        });
    }

    pub(crate) fn assert_eq_row(&mut self, rs1: u8, rs2: u8) {
        let instruction = self.instruction(
            JoltInstructionKind::FIELD_ASSERT_EQ,
            None,
            Some(rs1),
            Some(rs2),
            0,
        );
        let rs1 = self.read(rs1);
        let rs2 = self.read(rs2);
        self.rows.push(TraceRow {
            instruction,
            field_inline: Some(Arc::new(FieldInlineTraceData {
                op: Some(FieldInlineOp::AssertEq),
                rs1: Some(rs1),
                rs2: Some(rs2),
                ..FieldInlineTraceData::default()
            })),
            ..TraceRow::default()
        });
    }

    pub(crate) fn inv(&mut self, rd: u8, rs1: u8) {
        let instruction =
            self.instruction(JoltInstructionKind::FIELD_INV, Some(rd), Some(rs1), None, 0);
        let post = self.fresh_value();
        let inv_product = encode(self.state[usize::from(rs1)] * post);
        let rs1 = self.read(rs1);
        let rd = self.write(rd, post);
        self.rows.push(TraceRow {
            instruction,
            field_inline: Some(Arc::new(FieldInlineTraceData {
                op: Some(FieldInlineOp::Inv),
                rs1: Some(rs1),
                rd: Some(rd),
                inv_product: Some(inv_product),
                ..FieldInlineTraceData::default()
            })),
            ..TraceRow::default()
        });
    }

    /// Run `f` against an FR-profile trace backend padded to `2^log_t`
    /// cycles, with the field-inline witness view attached.
    pub(crate) fn with_plane<R>(
        self,
        log_t: usize,
        f: impl FnOnce(&TraceBackend<OwnedTrace>) -> R,
    ) -> R {
        assert!(self.rows.len() <= 1 << log_t, "fixture overflows 2^log_t");
        let preprocessing = Arc::new(JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::preprocess(
                self.bytecode.clone(),
                ENTRY,
                RV64IMAC_JOLT_FIELD_INLINE,
            )
            .unwrap(),
            ram: RAMPreprocessing::default(),
            memory_layout: Default::default(),
            max_padded_trace_length: 1 << log_t,
        });
        let program = Arc::new(JoltProgram::from_parts_with_profile(
            Vec::new(),
            self.bytecode,
            Vec::new(),
            ENTRY + 4,
            ENTRY,
            RV64IMAC_JOLT_FIELD_INLINE,
        ));
        let config = JoltVmWitnessConfig::new(
            log_t,
            64,
            JoltOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
        );
        let inputs = JoltVmWitnessInputs::new(
            &program,
            &preprocessing,
            TraceOutput::new(OwnedTrace::new(self.rows), Default::default(), None, None),
        );
        let backend = TraceBackend::new(config, inputs)
            .with_field_inline()
            .unwrap();
        f(&backend)
    }
}

/// A structured FR workload: seed loads, add/sub/mul/inv chains, `rs1 == rs2`
/// and `rd == rs1` aliasing, an assert-eq, repeated writes to one register,
/// high slot indices, and interleaved FR-inactive rows. Emits at most
/// `cycles` rows.
pub(crate) fn structured_fr_fixture(cycles: usize) -> FrTraceFixture {
    let mut fixture = FrTraceFixture::new();
    for step in 0..cycles {
        match step % 8 {
            0 => fixture.load_imm(3, 17 + step as u64),
            1 => fixture.arithmetic(FieldInlineOp::Add, 5, 3, 15),
            2 => fixture.arithmetic(FieldInlineOp::Mul, 5, 5, 3),
            3 => fixture.noop(),
            4 => fixture.arithmetic(FieldInlineOp::Sub, 15, 5, 5),
            5 => fixture.inv(7, 15),
            6 => fixture.assert_eq_row(5, 7),
            _ => fixture.arithmetic(FieldInlineOp::Mul, 0, 7, 0),
        }
    }
    fixture
}

/// An FR-profile fixture executing zero FR instructions — every FR column is
/// identically zero (the uniform-shape degenerate case).
pub(crate) fn inactive_fr_fixture(cycles: usize) -> FrTraceFixture {
    let mut fixture = FrTraceFixture::new();
    for _ in 0..cycles {
        fixture.noop();
    }
    fixture
}
