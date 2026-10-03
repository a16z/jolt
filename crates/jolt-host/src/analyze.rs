use std::{
    collections::HashMap,
    fmt::{Formatter, Result as FmtResult},
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::Arc,
};

use common::jolt_device::JoltDevice;
use jolt_program::{
    execution::{TraceData, TraceDataSeed},
    preprocess::BytecodePreprocessing,
};
use jolt_riscv::JoltInstructionRow;
#[cfg(not(feature = "field-inline"))]
use jolt_riscv::RV64IMAC_JOLT_ALL_INLINES;
#[cfg(feature = "field-inline")]
use jolt_riscv::RV64IMAC_JOLT_FIELD_INLINE;
use serde::{
    de::{Error, SeqAccess, Visitor},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};

const SUMMARY_MAGIC: [u8; 8] = *b"JOLTTRCE";
const SUMMARY_VERSION: u16 = 1;
const FIELD_PAYLOAD_FORMAT: bool = cfg!(feature = "field-inline");

/// Full execution analysis, retaining source rows and any field payloads.
///
/// The serde wire is a seven-element sequence: magic, schema version, field
/// payload format, bytecode, trace events, initial memory, and public I/O.
/// Import validates each row against the preceding bytecode. Formats from
/// different field-inline feature modes are rejected before reading rows.
#[derive(Debug)]
pub struct ProgramSummary {
    pub trace: Arc<TraceData>,
    pub bytecode: Vec<JoltInstructionRow>,
    pub memory_init: Vec<(u64, u8)>,
    pub io_device: JoltDevice,
}

impl Serialize for ProgramSummary {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut wire = serializer.serialize_tuple(7)?;
        wire.serialize_element(&SUMMARY_MAGIC)?;
        wire.serialize_element(&SUMMARY_VERSION)?;
        wire.serialize_element(&FIELD_PAYLOAD_FORMAT)?;
        wire.serialize_element(&self.bytecode)?;
        wire.serialize_element(self.trace.as_ref())?;
        wire.serialize_element(&self.memory_init)?;
        wire.serialize_element(&self.io_device)?;
        wire.end()
    }
}

impl<'de> Deserialize<'de> for ProgramSummary {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_tuple(7, SummaryVisitor)
    }
}

struct SummaryVisitor;

impl<'de> Visitor<'de> for SummaryVisitor {
    type Value = ProgramSummary;

    fn expecting(&self, formatter: &mut Formatter) -> FmtResult {
        formatter.write_str("a versioned Jolt program-summary envelope")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let magic: [u8; 8] = sequence
            .next_element()?
            .ok_or_else(|| A::Error::invalid_length(0, &self))?;
        if magic != SUMMARY_MAGIC {
            return Err(A::Error::custom("invalid program-summary magic"));
        }
        let version: u16 = sequence
            .next_element()?
            .ok_or_else(|| A::Error::invalid_length(1, &self))?;
        if version != SUMMARY_VERSION {
            return Err(A::Error::custom(
                "unsupported program-summary schema version",
            ));
        }
        let field_payload_format: bool = sequence
            .next_element()?
            .ok_or_else(|| A::Error::invalid_length(2, &self))?;
        if field_payload_format != FIELD_PAYLOAD_FORMAT {
            return Err(A::Error::custom(
                "program-summary field payload format does not match this build",
            ));
        }
        let bytecode: Vec<JoltInstructionRow> = sequence
            .next_element()?
            .ok_or_else(|| A::Error::invalid_length(3, &self))?;
        #[cfg(not(feature = "field-inline"))]
        let profile = RV64IMAC_JOLT_ALL_INLINES;
        #[cfg(feature = "field-inline")]
        let profile = RV64IMAC_JOLT_FIELD_INLINE;
        let preprocessing = BytecodePreprocessing::preprocess(bytecode.clone(), 0, profile)
            .map_err(A::Error::custom)?;
        let trace = sequence
            .next_element_seed(TraceDataSeed {
                bytecode: &preprocessing,
            })?
            .ok_or_else(|| A::Error::invalid_length(4, &self))?;
        let memory_init = sequence
            .next_element()?
            .ok_or_else(|| A::Error::invalid_length(5, &self))?;
        let io_device = sequence
            .next_element()?
            .ok_or_else(|| A::Error::invalid_length(6, &self))?;
        Ok(ProgramSummary {
            trace: Arc::new(trace),
            bytecode,
            memory_init,
            io_device,
        })
    }
}

impl ProgramSummary {
    pub fn trace_len(&self) -> usize {
        self.trace.len()
    }

    pub fn analyze(&self) -> Vec<(&'static str, usize)> {
        let mut counts = HashMap::<&'static str, usize>::new();
        for row in self.trace.rows() {
            let instruction_name = row.instruction().instruction_kind.name();
            if let Some(count) = counts.get(instruction_name) {
                let _ = counts.insert(instruction_name, count + 1);
            } else {
                let _ = counts.insert(instruction_name, 1);
            }
        }

        let mut counts: Vec<_> = counts.into_iter().collect();
        counts.sort_by_key(|v| v.1);
        counts.reverse();

        counts
    }

    pub fn write_to_file(self, path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = BufWriter::new(File::create(path)?);
        let _ =
            bincode::serde::encode_into_std_write(&self, &mut file, bincode::config::standard())?;
        file.flush()?;
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use super::*;
    use common::constants::RAM_START_ADDRESS;
    use jolt_riscv::{
        JoltInstructionKind, JoltTraceRow, NormalizedOperands, RamAccess, RegisterRead,
        RegisterState, RegisterWrite,
    };

    fn summary() -> ProgramSummary {
        let instruction = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::ADDI,
            address: RAM_START_ADDRESS as usize,
            operands: NormalizedOperands {
                rs1: Some(2),
                rd: Some(1),
                imm: 3,
                ..Default::default()
            },
            ..Default::default()
        };
        let row = JoltTraceRow::new(
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
            1,
        )
        .unwrap();
        ProgramSummary {
            trace: Arc::new(TraceData::new(vec![
                row,
                JoltTraceRow::default(),
                JoltTraceRow::default(),
            ])),
            bytecode: vec![instruction],
            memory_init: vec![(RAM_START_ADDRESS, 0x13)],
            io_device: JoltDevice::default(),
        }
    }

    fn roundtrip(summary: &ProgramSummary) -> ProgramSummary {
        let bytes = bincode::serde::encode_to_vec(summary, bincode::config::standard()).unwrap();
        let (decoded, consumed) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(consumed, bytes.len());
        decoded
    }

    #[test]
    fn archive_roundtrip_retains_full_execution_including_padding() {
        let summary = summary();
        let decoded = roundtrip(&summary);
        assert_eq!(decoded.trace.as_ref(), summary.trace.as_ref());
        assert_eq!(decoded.bytecode, summary.bytecode);
        assert_eq!(decoded.memory_init, summary.memory_init);
        assert_eq!(decoded.io_device, summary.io_device);
        assert_eq!(decoded.trace_len(), 3);
        assert_eq!(decoded.trace.proof_len(), 1);
        assert_eq!(decoded.analyze(), vec![("NoOp", 2), ("ADDI", 1)]);
    }

    #[test]
    fn archive_rejects_wrong_header_and_instruction_identity() {
        let summary = summary();
        for (magic, version, field_format) in [
            (*b"NOTJOLT!", SUMMARY_VERSION, FIELD_PAYLOAD_FORMAT),
            (SUMMARY_MAGIC, SUMMARY_VERSION + 1, FIELD_PAYLOAD_FORMAT),
            (SUMMARY_MAGIC, SUMMARY_VERSION, !FIELD_PAYLOAD_FORMAT),
        ] {
            let invalid = (
                magic,
                version,
                field_format,
                &summary.bytecode,
                summary.trace.as_ref(),
                &summary.memory_init,
                &summary.io_device,
            );
            let bytes =
                bincode::serde::encode_to_vec(invalid, bincode::config::standard()).unwrap();
            assert!(bincode::serde::decode_from_slice::<ProgramSummary, _>(
                &bytes,
                bincode::config::standard()
            )
            .is_err());
        }
        let mut mismatch = summary;
        mismatch.bytecode.first_mut().unwrap().operands.imm += 1;
        let bytes = bincode::serde::encode_to_vec(&mismatch, bincode::config::standard()).unwrap();
        assert!(bincode::serde::decode_from_slice::<ProgramSummary, _>(
            &bytes,
            bincode::config::standard()
        )
        .is_err());
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn archive_roundtrip_retains_field_payloads() {
        use jolt_program::{
            execution::TraceEvent,
            field_inline::{FieldEncodedValue, FieldInlineTraceData, FieldRegisterWrite},
        };
        use jolt_riscv::FieldInlineOp;

        let instruction = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::FIELD_LOAD_IMM,
            address: RAM_START_ADDRESS as usize,
            operands: NormalizedOperands {
                rd: Some(1),
                imm: 7,
                ..Default::default()
            },
            ..Default::default()
        };
        let row =
            JoltTraceRow::new(instruction, RegisterState::default(), RamAccess::NoOp, 1).unwrap();
        let payload = FieldInlineTraceData {
            op: Some(FieldInlineOp::LoadImm),
            rd: Some(FieldRegisterWrite {
                register: 1,
                pre_value: FieldEncodedValue::zero(),
                post_value: FieldEncodedValue::from_u64(7),
            }),
            ..Default::default()
        };
        let mut trace = TraceData::default();
        trace.push(TraceEvent {
            row,
            field_inline: Some(Arc::new(payload)),
        });
        trace.push(JoltTraceRow::default().into());
        let summary = ProgramSummary {
            trace: Arc::new(trace),
            bytecode: vec![instruction],
            memory_init: Vec::new(),
            io_device: JoltDevice::default(),
        };
        let decoded = roundtrip(&summary);
        assert_eq!(decoded.trace.as_ref(), summary.trace.as_ref());
        assert_eq!(decoded.trace.field_inline(0), Some(&payload));
        assert_eq!(decoded.trace_len(), 2);
    }
}
