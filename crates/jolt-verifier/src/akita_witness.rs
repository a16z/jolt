use jolt_field::{CanonicalBytes, FromPrimitiveInt};
use jolt_riscv::JoltTraceRow;
use thiserror::Error;

use jolt_akita::{
    AkitaField, PackedAdviceKind, PackedCellAddress, PackedFactDomain, PackedFamilyId,
    PackedLayoutError, PackedWitnessLayout, SparsePackedWitness,
};
use jolt_claims::protocols::jolt::unsigned_inc_lower_chunk_count;

#[derive(Clone, Debug)]
pub struct JoltPackedWitnessBuilder {
    layout: PackedWitnessLayout,
    entries: Vec<(usize, AkitaField)>,
}

impl JoltPackedWitnessBuilder {
    pub fn new(layout: PackedWitnessLayout) -> Self {
        Self {
            layout,
            entries: Vec::new(),
        }
    }

    pub fn layout(&self) -> &PackedWitnessLayout {
        &self.layout
    }

    pub fn pack_trace_rows(
        &mut self,
        rows: &[JoltTraceRow],
        log_k_chunk: usize,
        mut lookup_index: impl FnMut(usize, &JoltTraceRow) -> u128,
        mut ram_address: impl FnMut(usize, &JoltTraceRow) -> Option<u64>,
    ) -> Result<&mut Self, JoltPackedWitnessError> {
        let expected = self
            .trace_row_count()?
            .ok_or(JoltPackedWitnessError::MissingDomain {
                domain: "trace rows",
            })?;
        if rows.len() != expected {
            return Err(JoltPackedWitnessError::LengthMismatch {
                domain: "trace rows",
                expected,
                got: rows.len(),
            });
        }
        let instruction_chunks = self.max_trace_family_index(|family| {
            matches!(family, PackedFamilyId::InstructionRa { .. })
        });
        let bytecode_chunks = self
            .max_trace_family_index(|family| matches!(family, PackedFamilyId::BytecodeRa { .. }));
        let ram_chunks =
            self.max_trace_family_index(|family| matches!(family, PackedFamilyId::RamRa { .. }));

        for (row_index, row) in rows.iter().enumerate() {
            let lookup_index = lookup_index(row_index, row);
            for index in 0..instruction_chunks {
                let symbol = chunk(lookup_index, index, instruction_chunks, log_k_chunk)?;
                self.emit_one(
                    PackedFamilyId::InstructionRa { index },
                    row_index,
                    0,
                    symbol,
                )?;
            }

            let pc = row.pc() as u128;
            for index in 0..bytecode_chunks {
                let symbol = chunk(pc, index, bytecode_chunks, log_k_chunk)?;
                self.emit_one(PackedFamilyId::BytecodeRa { index }, row_index, 0, symbol)?;
            }

            if let Some(address) = ram_address(row_index, row) {
                for index in 0..ram_chunks {
                    let symbol = chunk(address as u128, index, ram_chunks, log_k_chunk)?;
                    self.emit_one(PackedFamilyId::RamRa { index }, row_index, 0, symbol)?;
                }
            }

            self.pack_increment_row(row_index, row, log_k_chunk)?;
        }
        Ok(self)
    }

    pub fn pack_untrusted_advice_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<&mut Self, JoltPackedWitnessError> {
        let kind = PackedAdviceKind::Untrusted;
        let domain = "untrusted advice bytes";
        let expected = self
            .advice_byte_count(kind)?
            .ok_or(JoltPackedWitnessError::MissingDomain { domain })?;
        if bytes.len() != expected {
            return Err(JoltPackedWitnessError::LengthMismatch {
                domain,
                expected,
                got: bytes.len(),
            });
        }
        for (row, byte) in bytes.iter().copied().enumerate() {
            self.emit_byte(PackedFamilyId::AdviceBytes { kind, index: 0 }, row, 0, byte)?;
        }
        Ok(self)
    }

    pub fn finish(self) -> Result<SparsePackedWitness<AkitaField>, JoltPackedWitnessError> {
        SparsePackedWitness::try_new(self.layout, self.entries).map_err(Into::into)
    }

    fn pack_increment_row(
        &mut self,
        row_index: usize,
        row: &JoltTraceRow,
        log_k_chunk: usize,
    ) -> Result<(), JoltPackedWitnessError> {
        let is_store = row.is_store();
        let has_rd = row.rd_index().is_some();
        if is_store && has_rd {
            return Err(JoltPackedWitnessError::IncrementSourceConflict { row: row_index });
        }

        let delta = if is_store {
            row.ram_write_value() as i128 - row.ram_read_value() as i128
        } else if has_rd {
            row.rd_write_value() as i128 - row.rd_pre_value() as i128
        } else {
            0
        };
        self.emit_unsigned_increment(row_index, delta, log_k_chunk)?;

        if self
            .layout
            .family(&PackedFamilyId::FieldRdIncByte { index: 0 })
            .is_some()
        {
            let rd_delta = if row.rd_index().is_some() {
                row.rd_write_value() as i128 - row.rd_pre_value() as i128
            } else {
                0
            };
            let encoded = AkitaField::from_i128(rd_delta).to_bytes_le_vec();
            for (index, byte) in encoded.into_iter().enumerate() {
                self.emit_byte(PackedFamilyId::FieldRdIncByte { index }, row_index, 0, byte)?;
            }
        }
        Ok(())
    }

    fn emit_unsigned_increment(
        &mut self,
        row: usize,
        delta: i128,
        log_k_chunk: usize,
    ) -> Result<(), JoltPackedWitnessError> {
        let chunk_count = unsigned_inc_lower_chunk_count(log_k_chunk).ok_or(
            JoltPackedWitnessError::InvalidChunkGeometry {
                chunks: 0,
                index: 0,
                log_k_chunk,
            },
        )?;
        let shifted = (1u128 << 64)
            .checked_add_signed(delta)
            .ok_or(JoltPackedWitnessError::DimensionOverflow)?;
        let lower_mask = (1u128 << 64) - 1;
        let lower = shifted & lower_mask;
        let msb = shifted >> 64;
        for index in 0..chunk_count {
            self.emit_one(
                PackedFamilyId::UnsignedIncChunk { index },
                row,
                0,
                little_endian_chunk(lower, index, log_k_chunk)?,
            )?;
        }
        if msb == 1 {
            self.emit_one(PackedFamilyId::UnsignedIncMsb, row, 0, 1)?;
        }
        Ok(())
    }

    fn emit_byte(
        &mut self,
        family: PackedFamilyId,
        row: usize,
        limb: usize,
        byte: u8,
    ) -> Result<(), JoltPackedWitnessError> {
        self.emit_one(family, row, limb, byte as usize)
    }

    fn emit_one(
        &mut self,
        family: PackedFamilyId,
        row: usize,
        limb: usize,
        symbol: usize,
    ) -> Result<(), JoltPackedWitnessError> {
        if self.layout.family(&family).is_none() {
            return Ok(());
        }
        let rank = self.layout.rank(&PackedCellAddress {
            family,
            row,
            limb,
            symbol,
        })?;
        self.entries.push((rank, AkitaField::one()));
        Ok(())
    }

    fn trace_row_count(&self) -> Result<Option<usize>, JoltPackedWitnessError> {
        let mut rows = None;
        for family in &self.layout.families {
            if matches!(family.domain, PackedFactDomain::TraceRows { .. }) {
                merge_domain_rows(&mut rows, family.domain, "trace rows")?;
            }
        }
        Ok(rows)
    }

    fn advice_byte_count(
        &self,
        kind: PackedAdviceKind,
    ) -> Result<Option<usize>, JoltPackedWitnessError> {
        let mut rows = None;
        for family in &self.layout.families {
            if family.id == (PackedFamilyId::AdviceBytes { kind, index: 0 }) {
                merge_domain_rows(&mut rows, family.domain, "untrusted advice bytes")?;
            }
        }
        Ok(rows)
    }

    fn max_trace_family_index(&self, is_family: impl Fn(&PackedFamilyId) -> bool) -> usize {
        self.layout
            .families
            .iter()
            .filter(|family| matches!(family.domain, PackedFactDomain::TraceRows { .. }))
            .filter(|family| is_family(&family.id))
            .filter_map(|family| match family.id {
                PackedFamilyId::InstructionRa { index }
                | PackedFamilyId::BytecodeRa { index }
                | PackedFamilyId::RamRa { index } => Some(index + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }
}

#[derive(Debug, Error)]
pub enum JoltPackedWitnessError {
    #[error("packed witness is missing {domain} layout families")]
    MissingDomain { domain: &'static str },
    #[error("{domain} length mismatch: expected {expected}, got {got}")]
    LengthMismatch {
        domain: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("{domain} layout rows are inconsistent: expected {expected}, got {got}")]
    InconsistentDomain {
        domain: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("packed witness dimension overflows usize")]
    DimensionOverflow,
    #[error("invalid chunk geometry: chunks={chunks}, index={index}, log_k_chunk={log_k_chunk}")]
    InvalidChunkGeometry {
        chunks: usize,
        index: usize,
        log_k_chunk: usize,
    },
    #[error("increment row {row} exposes both store and rd-present sources")]
    IncrementSourceConflict { row: usize },
    #[error(transparent)]
    Layout(#[from] PackedLayoutError),
}

fn merge_domain_rows(
    rows: &mut Option<usize>,
    domain: PackedFactDomain,
    name: &'static str,
) -> Result<(), JoltPackedWitnessError> {
    let got = domain_rows(domain)?;
    match *rows {
        Some(expected) if expected != got => Err(JoltPackedWitnessError::InconsistentDomain {
            domain: name,
            expected,
            got,
        }),
        Some(_) => Ok(()),
        None => {
            *rows = Some(got);
            Ok(())
        }
    }
}

fn domain_rows(domain: PackedFactDomain) -> Result<usize, JoltPackedWitnessError> {
    let log_rows = match domain {
        PackedFactDomain::TraceRows { log_t } => log_t,
        PackedFactDomain::BytecodeRows { log_bytecode } => log_bytecode,
        PackedFactDomain::ProgramImageWords { log_words } => log_words,
        PackedFactDomain::AdviceBytes { log_bytes, .. } => log_bytes,
    };
    1usize
        .checked_shl(log_rows as u32)
        .ok_or(JoltPackedWitnessError::DimensionOverflow)
}

fn chunk(
    value: u128,
    index: usize,
    chunks: usize,
    log_k_chunk: usize,
) -> Result<usize, JoltPackedWitnessError> {
    if chunks == 0 || index >= chunks || log_k_chunk == 0 || log_k_chunk >= usize::BITS as usize {
        return Err(JoltPackedWitnessError::InvalidChunkGeometry {
            chunks,
            index,
            log_k_chunk,
        });
    }
    let shift = log_k_chunk
        .checked_mul(chunks - 1 - index)
        .ok_or(JoltPackedWitnessError::DimensionOverflow)?;
    let mask = (1u128 << log_k_chunk) - 1;
    Ok(((value >> shift) & mask) as usize)
}

fn little_endian_chunk(
    value: u128,
    index: usize,
    log_k_chunk: usize,
) -> Result<usize, JoltPackedWitnessError> {
    if log_k_chunk == 0 || log_k_chunk >= usize::BITS as usize {
        return Err(JoltPackedWitnessError::InvalidChunkGeometry {
            chunks: 0,
            index,
            log_k_chunk,
        });
    }
    let shift = log_k_chunk
        .checked_mul(index)
        .ok_or(JoltPackedWitnessError::DimensionOverflow)?;
    let mask = (1u128 << log_k_chunk) - 1;
    Ok(((value >> shift) & mask) as usize)
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "tests assert successful witness construction"
    )]

    use super::*;
    use jolt_akita::{PackedAlphabet, PackedFamilySpec, PackedWitnessSource};
    use jolt_field::FixedByteSize;
    use jolt_riscv::{
        CapturedState, JoltInstructionKind, JoltInstructionRow, NormalizedOperands, StoreState,
    };

    fn trace_domain() -> PackedFactDomain {
        PackedFactDomain::TraceRows { log_t: 1 }
    }

    fn trace_row(
        instruction_kind: JoltInstructionKind,
        operands: NormalizedOperands,
        state: CapturedState,
        bytecode_pc: u32,
    ) -> JoltTraceRow {
        let instruction = JoltInstructionRow {
            instruction_kind,
            address: 0x8000_0000,
            operands,
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        JoltTraceRow::from_components(state, &instruction, bytecode_pc)
            .expect("trace row should build")
    }

    fn get(
        witness: &SparsePackedWitness<AkitaField>,
        family: PackedFamilyId,
        row: usize,
        limb: usize,
        symbol: usize,
    ) -> AkitaField {
        witness
            .eval_direct_fact(&PackedCellAddress {
                family,
                row,
                limb,
                symbol,
            })
            .expect("address should be in layout")
    }

    #[test]
    fn packs_trace_ra_and_unsigned_increment_facts() {
        let layout = PackedWitnessLayout::new([
            PackedFamilySpec::direct(
                PackedFamilyId::InstructionRa { index: 0 },
                trace_domain(),
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeRa { index: 0 },
                trace_domain(),
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::RamRa { index: 0 },
                trace_domain(),
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::UnsignedIncChunk { index: 0 },
                trace_domain(),
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::UnsignedIncMsb,
                trace_domain(),
                1,
                PackedAlphabet::Bit,
            ),
        ])
        .expect("layout should build");
        let rows = [
            trace_row(
                JoltInstructionKind::ADD,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(2),
                    rd: Some(3),
                    imm: 0,
                },
                CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                    rs1_value: 1,
                    rs2_value: 2,
                    rd_pre_value: 10,
                    rd_write_value: 3,
                }),
                9,
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
                    rs2_value: 30,
                    ram_read_value: 10,
                    ram_address: 0x42,
                }),
                11,
            ),
        ];

        let mut builder = JoltPackedWitnessBuilder::new(layout);
        let _ = builder
            .pack_trace_rows(
                &rows,
                8,
                |index, _| [0x7f, 0x80][index],
                |index, _| [None, Some(0x42)][index],
            )
            .expect("trace packing should succeed");
        let witness = builder.finish().expect("source should build");

        assert_eq!(
            get(
                &witness,
                PackedFamilyId::InstructionRa { index: 0 },
                0,
                0,
                0x7f
            ),
            AkitaField::one()
        );
        assert_eq!(
            get(&witness, PackedFamilyId::BytecodeRa { index: 0 }, 1, 0, 11),
            AkitaField::one()
        );
        assert_eq!(
            get(&witness, PackedFamilyId::RamRa { index: 0 }, 1, 0, 0x42),
            AkitaField::one()
        );
        assert_eq!(
            get(
                &witness,
                PackedFamilyId::UnsignedIncChunk { index: 0 },
                0,
                0,
                249
            ),
            AkitaField::one()
        );
        assert!(get(&witness, PackedFamilyId::UnsignedIncMsb, 0, 0, 1).is_zero());
        assert_eq!(
            get(
                &witness,
                PackedFamilyId::UnsignedIncChunk { index: 0 },
                1,
                0,
                20
            ),
            AkitaField::one()
        );
        assert_eq!(
            get(&witness, PackedFamilyId::UnsignedIncMsb, 1, 0, 1),
            AkitaField::one()
        );
    }

    #[test]
    fn zero_increment_emits_zero_lower_chunks_and_set_msb() {
        let layout = increment_layout();
        let rows = [
            trace_row(
                JoltInstructionKind::ADD,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(2),
                    rd: Some(3),
                    imm: 0,
                },
                CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                    rs1_value: 1,
                    rs2_value: 2,
                    rd_pre_value: 10,
                    rd_write_value: 10,
                }),
                9,
            ),
            trace_row(
                JoltInstructionKind::ADD,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(2),
                    rd: Some(3),
                    imm: 0,
                },
                CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                    rs1_value: 1,
                    rs2_value: 2,
                    rd_pre_value: 7,
                    rd_write_value: 7,
                }),
                11,
            ),
        ];

        let mut builder = JoltPackedWitnessBuilder::new(layout);
        let _ = builder
            .pack_trace_rows(&rows, 8, |_, _| 0, |_, _| None)
            .expect("trace packing should succeed");
        let witness = builder.finish().expect("source should build");

        for index in 0..8 {
            assert_eq!(
                get(
                    &witness,
                    PackedFamilyId::UnsignedIncChunk { index },
                    0,
                    0,
                    0
                ),
                AkitaField::one()
            );
        }
        assert_eq!(
            get(&witness, PackedFamilyId::UnsignedIncMsb, 0, 0, 1),
            AkitaField::one()
        );
    }

    #[test]
    fn unsigned_increment_ignores_rd_slots_without_rd_destination() {
        let layout = increment_layout();
        let rows = [
            trace_row(
                JoltInstructionKind::BEQ,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(2),
                    rd: None,
                    imm: 4,
                },
                CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                    rs1_value: 1,
                    rs2_value: 2,
                    rd_pre_value: 10,
                    rd_write_value: 3,
                }),
                9,
            ),
            JoltTraceRow::no_op(),
        ];

        let mut builder = JoltPackedWitnessBuilder::new(layout);
        let _ = builder
            .pack_trace_rows(&rows, 8, |_, _| 0, |_, _| None)
            .expect("trace packing should succeed");
        let witness = builder.finish().expect("source should build");

        for index in 0..8 {
            assert_eq!(
                get(
                    &witness,
                    PackedFamilyId::UnsignedIncChunk { index },
                    0,
                    0,
                    0
                ),
                AkitaField::one()
            );
        }
        assert_eq!(
            get(&witness, PackedFamilyId::UnsignedIncMsb, 0, 0, 1),
            AkitaField::one()
        );
    }

    #[test]
    fn unsigned_increment_rejects_store_with_rd_destination() {
        let layout = increment_layout();
        let rows = [
            trace_row(
                JoltInstructionKind::SD,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(2),
                    rd: Some(3),
                    imm: 8,
                },
                CapturedState::Store(StoreState {
                    rs1_value: 1,
                    rs2_value: 11,
                    ram_read_value: 10,
                    ram_address: 0x34,
                }),
                9,
            ),
            JoltTraceRow::no_op(),
        ];

        let mut builder = JoltPackedWitnessBuilder::new(layout);
        let error = builder
            .pack_trace_rows(&rows, 8, |_, _| 0, |index, _| [Some(0x34), None][index])
            .expect_err("ambiguous increment source should reject");

        assert!(matches!(
            error,
            JoltPackedWitnessError::IncrementSourceConflict { row: 0 }
        ));
    }

    #[test]
    fn negative_increment_emits_offset_lower_chunks_and_clear_msb() {
        let layout = increment_layout();
        let rows = [
            trace_row(
                JoltInstructionKind::ADD,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(2),
                    rd: Some(3),
                    imm: 0,
                },
                CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                    rs1_value: 1,
                    rs2_value: 2,
                    rd_pre_value: 10,
                    rd_write_value: 3,
                }),
                9,
            ),
            trace_row(
                JoltInstructionKind::ADD,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(2),
                    rd: Some(3),
                    imm: 0,
                },
                CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                    rs1_value: 1,
                    rs2_value: 2,
                    rd_pre_value: 7,
                    rd_write_value: 7,
                }),
                11,
            ),
        ];

        let mut builder = JoltPackedWitnessBuilder::new(layout);
        let _ = builder
            .pack_trace_rows(&rows, 8, |_, _| 0, |_, _| None)
            .expect("trace packing should succeed");
        let witness = builder.finish().expect("source should build");

        assert_eq!(
            get(
                &witness,
                PackedFamilyId::UnsignedIncChunk { index: 0 },
                0,
                0,
                249
            ),
            AkitaField::one()
        );
        for index in 1..8 {
            assert_eq!(
                get(
                    &witness,
                    PackedFamilyId::UnsignedIncChunk { index },
                    0,
                    0,
                    255
                ),
                AkitaField::one()
            );
        }
        assert!(get(&witness, PackedFamilyId::UnsignedIncMsb, 0, 0, 1).is_zero());
    }

    #[test]
    fn field_rd_inc_uses_canonical_field_bytes() {
        let layout = field_rd_inc_layout();
        let rows = [
            trace_row(
                JoltInstructionKind::ADD,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(2),
                    rd: Some(3),
                    imm: 0,
                },
                CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                    rs1_value: 1,
                    rs2_value: 2,
                    rd_pre_value: 10,
                    rd_write_value: 3,
                }),
                9,
            ),
            trace_row(
                JoltInstructionKind::ADD,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(2),
                    rd: Some(3),
                    imm: 0,
                },
                CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                    rs1_value: 1,
                    rs2_value: 2,
                    rd_pre_value: 7,
                    rd_write_value: 7,
                }),
                11,
            ),
        ];

        let mut builder = JoltPackedWitnessBuilder::new(layout);
        let _ = builder
            .pack_trace_rows(&rows, 8, |_, _| 0, |_, _| None)
            .expect("trace packing should succeed");
        let witness = builder.finish().expect("source should build");
        let encoded = AkitaField::from_i128(-7).to_bytes_le_vec();

        assert_eq!(encoded.len(), AkitaField::NUM_BYTES);
        for (index, byte) in encoded.into_iter().enumerate() {
            assert_eq!(
                get(
                    &witness,
                    PackedFamilyId::FieldRdIncByte { index },
                    0,
                    0,
                    byte as usize
                ),
                AkitaField::one()
            );
        }
        assert!(witness
            .layout()
            .family(&PackedFamilyId::FieldRdIncSign)
            .is_none());
    }

    #[test]
    fn field_rd_inc_ignores_rd_slots_without_rd_destination() {
        let layout = field_rd_inc_layout();
        let rows = [
            trace_row(
                JoltInstructionKind::BEQ,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(2),
                    rd: None,
                    imm: 4,
                },
                CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                    rs1_value: 1,
                    rs2_value: 2,
                    rd_pre_value: 10,
                    rd_write_value: 3,
                }),
                9,
            ),
            JoltTraceRow::no_op(),
        ];

        let mut builder = JoltPackedWitnessBuilder::new(layout);
        let _ = builder
            .pack_trace_rows(&rows, 8, |_, _| 0, |_, _| None)
            .expect("trace packing should succeed");
        let witness = builder.finish().expect("source should build");

        for index in 0..AkitaField::NUM_BYTES {
            assert_eq!(
                get(&witness, PackedFamilyId::FieldRdIncByte { index }, 0, 0, 0),
                AkitaField::one()
            );
        }
    }

    #[test]
    fn untrusted_advice_encoding_roundtrip() {
        let bytes = [255, 0, 7, 8];
        let layout = advice_layout(PackedAdviceKind::Untrusted);

        let mut builder = JoltPackedWitnessBuilder::new(layout);
        let _ = builder
            .pack_untrusted_advice_bytes(&bytes)
            .expect("advice packing should succeed");
        let witness = builder.finish().expect("source should build");

        for (row, byte) in bytes.iter().copied().enumerate() {
            assert_eq!(
                get(
                    &witness,
                    PackedFamilyId::AdviceBytes {
                        kind: PackedAdviceKind::Untrusted,
                        index: 0,
                    },
                    row,
                    0,
                    byte as usize
                ),
                AkitaField::one()
            );
        }
    }

    fn advice_layout(kind: PackedAdviceKind) -> PackedWitnessLayout {
        PackedWitnessLayout::new([PackedFamilySpec::direct(
            PackedFamilyId::AdviceBytes { kind, index: 0 },
            PackedFactDomain::AdviceBytes { kind, log_bytes: 2 },
            1,
            PackedAlphabet::Byte,
        )])
        .expect("layout should build")
    }

    fn field_rd_inc_layout() -> PackedWitnessLayout {
        PackedWitnessLayout::new((0..AkitaField::NUM_BYTES).map(|index| {
            PackedFamilySpec::direct(
                PackedFamilyId::FieldRdIncByte { index },
                trace_domain(),
                1,
                PackedAlphabet::Byte,
            )
        }))
        .expect("layout should build")
    }

    fn increment_layout() -> PackedWitnessLayout {
        let mut specs = (0..8)
            .map(|index| {
                PackedFamilySpec::direct(
                    PackedFamilyId::UnsignedIncChunk { index },
                    trace_domain(),
                    1,
                    PackedAlphabet::Byte,
                )
            })
            .collect::<Vec<_>>();
        specs.push(PackedFamilySpec::direct(
            PackedFamilyId::UnsignedIncMsb,
            trace_domain(),
            1,
            PackedAlphabet::Bit,
        ));
        PackedWitnessLayout::new(specs).expect("layout should build")
    }
}
