use std::collections::BTreeMap;

use jolt_field::{CanonicalBytes, FromPrimitiveInt};
use jolt_lookup_tables::{InstructionLookupTable, XLEN};
use jolt_riscv::{
    CircuitFlagSet, CircuitFlags, Flags, InstructionFlagSet, InstructionFlags,
    InterleavedBitsMarker, JoltInstruction, JoltInstructionKind, JoltInstructionRow, JoltTraceRow,
    NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
};
use thiserror::Error;

use jolt_akita::{
    AkitaField, PackedAdviceKind, PackedCellAddress, PackedFactDomain, PackedFamilyId,
    PackedLayoutError, PackedWitnessLayout, SparsePackedWitness,
};

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

            self.pack_increment_row(row_index, row)?;
        }
        Ok(self)
    }

    pub fn pack_bytecode_rows(
        &mut self,
        bytecode: &[JoltInstructionRow],
    ) -> Result<&mut Self, JoltPackedWitnessError> {
        let chunks = self.bytecode_chunk_rows()?;
        if chunks.is_empty() {
            return Err(JoltPackedWitnessError::MissingDomain {
                domain: "bytecode rows",
            });
        }
        let chunk_count = chunks
            .keys()
            .next_back()
            .map_or(0, |chunk| chunk.saturating_add(1));
        for chunk in 0..chunk_count {
            if !chunks.contains_key(&chunk) {
                return Err(JoltPackedWitnessError::MissingChunk {
                    domain: "bytecode rows",
                    chunk,
                });
            }
        }
        let chunk_rows = *chunks
            .values()
            .next()
            .ok_or(JoltPackedWitnessError::MissingDomain {
                domain: "bytecode rows",
            })?;
        let expected = chunk_count
            .checked_mul(chunk_rows)
            .ok_or(JoltPackedWitnessError::DimensionOverflow)?;
        if bytecode.len() != expected {
            return Err(JoltPackedWitnessError::LengthMismatch {
                domain: "bytecode rows",
                expected,
                got: bytecode.len(),
            });
        }

        for (global_row, instruction) in bytecode.iter().enumerate() {
            let chunk = global_row / chunk_rows;
            let row = global_row % chunk_rows;
            self.pack_bytecode_row(chunk, row, instruction)?;
        }
        Ok(self)
    }

    pub fn pack_program_image_words(
        &mut self,
        words: &[u64],
    ) -> Result<&mut Self, JoltPackedWitnessError> {
        let expected =
            self.program_image_word_count()?
                .ok_or(JoltPackedWitnessError::MissingDomain {
                    domain: "program image words",
                })?;
        if words.len() != expected {
            return Err(JoltPackedWitnessError::LengthMismatch {
                domain: "program image words",
                expected,
                got: words.len(),
            });
        }
        for (row, word) in words.iter().copied().enumerate() {
            self.emit_little_endian_bytes(
                PackedFamilyId::ProgramImageInit,
                row,
                &word.to_le_bytes(),
            )?;
        }
        Ok(self)
    }

    pub fn pack_advice_bytes(
        &mut self,
        kind: PackedAdviceKind,
        bytes: &[u8],
    ) -> Result<&mut Self, JoltPackedWitnessError> {
        let domain = advice_domain_name(kind);
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
    ) -> Result<(), JoltPackedWitnessError> {
        let is_store = row.is_store();
        let has_rd = row.rd_index().is_some();
        if is_store && has_rd {
            return Err(JoltPackedWitnessError::FusedIncrementSourceConflict { row: row_index });
        }

        let delta = if is_store {
            row.ram_write_value() as i128 - row.ram_read_value() as i128
        } else if has_rd {
            row.rd_write_value() as i128 - row.rd_pre_value() as i128
        } else {
            0
        };
        self.emit_signed_increment(
            PackedFamilyId::IncByte { index: 0 },
            PackedFamilyId::IncSign,
            row_index,
            delta,
        )?;

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

    fn emit_signed_increment(
        &mut self,
        first_byte_family: PackedFamilyId,
        sign_family: PackedFamilyId,
        row: usize,
        delta: i128,
    ) -> Result<(), JoltPackedWitnessError> {
        let magnitude = delta.unsigned_abs() as u64;
        let bytes = magnitude.to_le_bytes();
        match first_byte_family {
            PackedFamilyId::IncByte { .. } => {
                for (index, byte) in bytes.iter().copied().enumerate() {
                    self.emit_byte(PackedFamilyId::IncByte { index }, row, 0, byte)?;
                }
            }
            PackedFamilyId::FieldRdIncByte { .. } => {
                for (index, byte) in bytes.iter().copied().enumerate() {
                    self.emit_byte(PackedFamilyId::FieldRdIncByte { index }, row, 0, byte)?;
                }
            }
            _ => {}
        }
        if delta < 0 {
            self.emit_one(sign_family, row, 0, 1)?;
        }
        Ok(())
    }

    fn pack_bytecode_row(
        &mut self,
        chunk: usize,
        row: usize,
        instruction: &JoltInstructionRow,
    ) -> Result<(), JoltPackedWitnessError> {
        let instruction = JoltInstruction::try_from(*instruction)
            .map_err(|kind| JoltPackedWitnessError::InvalidInstructionKind { kind })?;
        let circuit_flags = instruction.circuit_flags();
        let instruction_flags = instruction.instruction_flags();
        let source_row = JoltInstructionRow::from(instruction);
        if circuit_flags.get(CircuitFlags::Store) && source_row.operands.rd.is_some() {
            return Err(JoltPackedWitnessError::BytecodeSourceConflict { chunk, row });
        }

        if let Some(register) = source_row.operands.rs1 {
            self.emit_one(
                PackedFamilyId::BytecodeRegisterSelector { chunk, selector: 0 },
                row,
                0,
                register as usize,
            )?;
        }
        if let Some(register) = source_row.operands.rs2 {
            self.emit_one(
                PackedFamilyId::BytecodeRegisterSelector { chunk, selector: 1 },
                row,
                0,
                register as usize,
            )?;
        }
        if let Some(register) = source_row.operands.rd {
            self.emit_one(
                PackedFamilyId::BytecodeRegisterSelector { chunk, selector: 2 },
                row,
                0,
                register as usize,
            )?;
        }

        self.emit_circuit_flags(chunk, row, circuit_flags)?;
        self.emit_instruction_flags(chunk, row, instruction_flags)?;
        if let Some(table) = InstructionLookupTable::<XLEN>::lookup_table(&instruction) {
            self.emit_one(
                PackedFamilyId::BytecodeLookupSelector { chunk },
                row,
                0,
                table.index(),
            )?;
        }
        if !circuit_flags.is_interleaved_operands() {
            self.emit_one(PackedFamilyId::BytecodeRafFlag { chunk }, row, 0, 1)?;
        }

        self.emit_little_endian_bytes(
            PackedFamilyId::BytecodeUnexpandedPcBytes { chunk },
            row,
            &(source_row.address as u64).to_le_bytes(),
        )?;
        let imm = AkitaField::from_i128(source_row.operands.imm);
        self.emit_little_endian_bytes(
            PackedFamilyId::BytecodeImmBytes { chunk },
            row,
            &imm.to_bytes_le_vec(),
        )?;
        Ok(())
    }

    fn emit_circuit_flags(
        &mut self,
        chunk: usize,
        row: usize,
        flags: CircuitFlagSet,
    ) -> Result<(), JoltPackedWitnessError> {
        for flag in 0..NUM_CIRCUIT_FLAGS {
            if flags.get(circuit_flag(flag)?) {
                self.emit_one(
                    PackedFamilyId::BytecodeCircuitFlag { chunk, flag },
                    row,
                    0,
                    1,
                )?;
            }
        }
        Ok(())
    }

    fn emit_instruction_flags(
        &mut self,
        chunk: usize,
        row: usize,
        flags: InstructionFlagSet,
    ) -> Result<(), JoltPackedWitnessError> {
        for flag in 0..NUM_INSTRUCTION_FLAGS {
            if flags.get(instruction_flag(flag)?) {
                self.emit_one(
                    PackedFamilyId::BytecodeInstructionFlag { chunk, flag },
                    row,
                    0,
                    1,
                )?;
            }
        }
        Ok(())
    }

    fn emit_little_endian_bytes(
        &mut self,
        family: PackedFamilyId,
        row: usize,
        bytes: &[u8],
    ) -> Result<(), JoltPackedWitnessError> {
        for (limb, byte) in bytes.iter().copied().enumerate() {
            self.emit_byte(family.clone(), row, limb, byte)?;
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

    fn bytecode_chunk_rows(&self) -> Result<BTreeMap<usize, usize>, JoltPackedWitnessError> {
        let mut chunks = BTreeMap::new();
        for family in &self.layout.families {
            let chunk = match family.id {
                PackedFamilyId::BytecodeChunk { index }
                | PackedFamilyId::BytecodeRegisterSelector { chunk: index, .. }
                | PackedFamilyId::BytecodeCircuitFlag { chunk: index, .. }
                | PackedFamilyId::BytecodeInstructionFlag { chunk: index, .. }
                | PackedFamilyId::BytecodeLookupSelector { chunk: index }
                | PackedFamilyId::BytecodeRafFlag { chunk: index }
                | PackedFamilyId::BytecodeUnexpandedPcBytes { chunk: index }
                | PackedFamilyId::BytecodeImmBytes { chunk: index } => index,
                _ => continue,
            };
            let rows = domain_rows(family.domain)?;
            match chunks.get(&chunk) {
                Some(existing) if *existing != rows => {
                    return Err(JoltPackedWitnessError::InconsistentDomain {
                        domain: "bytecode rows",
                        expected: *existing,
                        got: rows,
                    });
                }
                Some(_) => {}
                None => {
                    let _ = chunks.insert(chunk, rows);
                }
            }
        }
        Ok(chunks)
    }

    fn program_image_word_count(&self) -> Result<Option<usize>, JoltPackedWitnessError> {
        let mut rows = None;
        for family in &self.layout.families {
            if family.id == PackedFamilyId::ProgramImageInit {
                merge_domain_rows(&mut rows, family.domain, "program image words")?;
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
                merge_domain_rows(&mut rows, family.domain, advice_domain_name(kind))?;
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

fn advice_domain_name(kind: PackedAdviceKind) -> &'static str {
    match kind {
        PackedAdviceKind::Trusted => "trusted advice bytes",
        PackedAdviceKind::Untrusted => "untrusted advice bytes",
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
    #[error("{domain} chunk {chunk} is missing from the packed witness layout")]
    MissingChunk { domain: &'static str, chunk: usize },
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
    #[error("invalid Jolt instruction kind {kind:?}")]
    InvalidInstructionKind { kind: JoltInstructionKind },
    #[error("fused increment row {row} exposes both store and rd-present sources")]
    FusedIncrementSourceConflict { row: usize },
    #[error("bytecode row {chunk}:{row} exposes both store and rd-present sources")]
    BytecodeSourceConflict { chunk: usize, row: usize },
    #[error("unknown circuit flag index {index}")]
    UnknownCircuitFlag { index: usize },
    #[error("unknown instruction flag index {index}")]
    UnknownInstructionFlag { index: usize },
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

fn circuit_flag(index: usize) -> Result<CircuitFlags, JoltPackedWitnessError> {
    match index {
        0 => Ok(CircuitFlags::AddOperands),
        1 => Ok(CircuitFlags::SubtractOperands),
        2 => Ok(CircuitFlags::MultiplyOperands),
        3 => Ok(CircuitFlags::Load),
        4 => Ok(CircuitFlags::Store),
        5 => Ok(CircuitFlags::Jump),
        6 => Ok(CircuitFlags::WriteLookupOutputToRD),
        7 => Ok(CircuitFlags::VirtualInstruction),
        8 => Ok(CircuitFlags::Assert),
        9 => Ok(CircuitFlags::DoNotUpdateUnexpandedPC),
        10 => Ok(CircuitFlags::Advice),
        11 => Ok(CircuitFlags::IsCompressed),
        12 => Ok(CircuitFlags::IsFirstInSequence),
        13 => Ok(CircuitFlags::IsLastInSequence),
        _ => Err(JoltPackedWitnessError::UnknownCircuitFlag { index }),
    }
}

fn instruction_flag(index: usize) -> Result<InstructionFlags, JoltPackedWitnessError> {
    match index {
        0 => Ok(InstructionFlags::LeftOperandIsPC),
        1 => Ok(InstructionFlags::RightOperandIsImm),
        2 => Ok(InstructionFlags::LeftOperandIsRs1Value),
        3 => Ok(InstructionFlags::RightOperandIsRs2Value),
        4 => Ok(InstructionFlags::Branch),
        5 => Ok(InstructionFlags::IsNoop),
        _ => Err(JoltPackedWitnessError::UnknownInstructionFlag { index }),
    }
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
    use jolt_riscv::{CapturedState, NormalizedOperands, StoreState};

    fn trace_domain() -> PackedFactDomain {
        PackedFactDomain::TraceRows { log_t: 1 }
    }

    fn bytecode_domain() -> PackedFactDomain {
        PackedFactDomain::BytecodeRows { log_bytecode: 1 }
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
    fn packs_trace_ra_and_fused_increment_facts() {
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
                PackedFamilyId::IncByte { index: 0 },
                trace_domain(),
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::IncSign,
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
            get(&witness, PackedFamilyId::IncByte { index: 0 }, 0, 0, 7),
            AkitaField::one()
        );
        assert_eq!(
            get(&witness, PackedFamilyId::IncSign, 0, 0, 1),
            AkitaField::one()
        );
        assert_eq!(
            get(&witness, PackedFamilyId::IncByte { index: 0 }, 1, 0, 20),
            AkitaField::one()
        );
        assert!(get(&witness, PackedFamilyId::IncSign, 1, 0, 1).is_zero());
    }

    #[test]
    fn fused_increment_emits_canonical_zero_byte_symbols() {
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
                get(&witness, PackedFamilyId::IncByte { index }, 0, 0, 0),
                AkitaField::one()
            );
        }
        assert!(get(&witness, PackedFamilyId::IncSign, 0, 0, 1).is_zero());
    }

    #[test]
    fn fused_increment_ignores_rd_slots_without_rd_destination() {
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
                get(&witness, PackedFamilyId::IncByte { index }, 0, 0, 0),
                AkitaField::one()
            );
        }
        assert!(get(&witness, PackedFamilyId::IncSign, 0, 0, 1).is_zero());
    }

    #[test]
    fn fused_increment_rejects_store_with_rd_destination() {
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
            .expect_err("ambiguous fused increment source should reject");

        assert!(matches!(
            error,
            JoltPackedWitnessError::FusedIncrementSourceConflict { row: 0 }
        ));
    }

    #[test]
    fn negative_increment_emits_sign_and_zero_high_bytes() {
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
            get(&witness, PackedFamilyId::IncByte { index: 0 }, 0, 0, 7),
            AkitaField::one()
        );
        for index in 1..8 {
            assert_eq!(
                get(&witness, PackedFamilyId::IncByte { index }, 0, 0, 0),
                AkitaField::one()
            );
        }
        assert_eq!(
            get(&witness, PackedFamilyId::IncSign, 0, 0, 1),
            AkitaField::one()
        );
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
    fn packs_committed_bytecode_facts() {
        let layout = PackedWitnessLayout::new([
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeRegisterSelector {
                    chunk: 0,
                    selector: 2,
                },
                bytecode_domain(),
                1,
                PackedAlphabet::Fixed { size: 32 },
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeCircuitFlag {
                    chunk: 0,
                    flag: CircuitFlags::Store as usize,
                },
                bytecode_domain(),
                1,
                PackedAlphabet::Bit,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeLookupSelector { chunk: 0 },
                bytecode_domain(),
                1,
                PackedAlphabet::Fixed {
                    size: jolt_lookup_tables::LookupTableKind::<XLEN>::COUNT,
                },
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeRafFlag { chunk: 0 },
                bytecode_domain(),
                1,
                PackedAlphabet::Bit,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeUnexpandedPcBytes { chunk: 0 },
                bytecode_domain(),
                8,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeImmBytes { chunk: 0 },
                bytecode_domain(),
                AkitaField::NUM_BYTES,
                PackedAlphabet::Byte,
            ),
        ])
        .expect("layout should build");
        let bytecode = [
            instruction(
                JoltInstructionKind::SD,
                0x8000_0000,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(2),
                    rd: None,
                    imm: 8,
                },
            ),
            instruction(
                JoltInstructionKind::ADDI,
                0x8000_0004,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: None,
                    rd: Some(5),
                    imm: 7,
                },
            ),
        ];

        let mut builder = JoltPackedWitnessBuilder::new(layout);
        let _ = builder
            .pack_bytecode_rows(&bytecode)
            .expect("bytecode packing should succeed");
        let witness = builder.finish().expect("source should build");

        assert_eq!(
            get(
                &witness,
                PackedFamilyId::BytecodeCircuitFlag {
                    chunk: 0,
                    flag: CircuitFlags::Store as usize,
                },
                0,
                0,
                1,
            ),
            AkitaField::one()
        );
        assert_eq!(
            get(
                &witness,
                PackedFamilyId::BytecodeRegisterSelector {
                    chunk: 0,
                    selector: 2,
                },
                1,
                0,
                5,
            ),
            AkitaField::one()
        );
        assert_eq!(
            get(
                &witness,
                PackedFamilyId::BytecodeUnexpandedPcBytes { chunk: 0 },
                1,
                0,
                4,
            ),
            AkitaField::one()
        );
        assert_eq!(
            get(
                &witness,
                PackedFamilyId::BytecodeImmBytes { chunk: 0 },
                1,
                0,
                7,
            ),
            AkitaField::one()
        );
        assert_eq!(
            get(
                &witness,
                PackedFamilyId::BytecodeRafFlag { chunk: 0 },
                1,
                0,
                1,
            ),
            AkitaField::one()
        );
    }

    #[test]
    fn bytecode_packing_rejects_store_with_rd_destination() {
        let layout = PackedWitnessLayout::new([
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeRegisterSelector {
                    chunk: 0,
                    selector: 2,
                },
                bytecode_domain(),
                1,
                PackedAlphabet::Fixed { size: 32 },
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeCircuitFlag {
                    chunk: 0,
                    flag: CircuitFlags::Store as usize,
                },
                bytecode_domain(),
                1,
                PackedAlphabet::Bit,
            ),
        ])
        .expect("layout should build");
        let bytecode = [
            instruction(
                JoltInstructionKind::SD,
                0x8000_0000,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(2),
                    rd: Some(3),
                    imm: 8,
                },
            ),
            instruction(
                JoltInstructionKind::ADDI,
                0x8000_0004,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: None,
                    rd: Some(5),
                    imm: 7,
                },
            ),
        ];

        let mut builder = JoltPackedWitnessBuilder::new(layout);
        let error = builder
            .pack_bytecode_rows(&bytecode)
            .expect_err("ambiguous bytecode source row should reject");

        assert!(matches!(
            error,
            JoltPackedWitnessError::BytecodeSourceConflict { chunk: 0, row: 0 }
        ));
    }

    #[test]
    fn packs_program_image_words_as_little_endian_bytes() {
        let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
            PackedFamilyId::ProgramImageInit,
            PackedFactDomain::ProgramImageWords { log_words: 1 },
            8,
            PackedAlphabet::Byte,
        )])
        .expect("layout should build");

        let mut builder = JoltPackedWitnessBuilder::new(layout);
        let _ = builder
            .pack_program_image_words(&[0x0201, 0x0403])
            .expect("program image packing should succeed");
        let witness = builder.finish().expect("source should build");

        assert_eq!(
            get(&witness, PackedFamilyId::ProgramImageInit, 0, 0, 1),
            AkitaField::one()
        );
        assert_eq!(
            get(&witness, PackedFamilyId::ProgramImageInit, 0, 1, 2),
            AkitaField::one()
        );
        assert_eq!(
            get(&witness, PackedFamilyId::ProgramImageInit, 1, 0, 3),
            AkitaField::one()
        );
        assert_eq!(
            get(&witness, PackedFamilyId::ProgramImageInit, 1, 1, 4),
            AkitaField::one()
        );
    }

    #[test]
    fn trusted_advice_encoding_roundtrip() {
        assert_advice_encoding(PackedAdviceKind::Trusted, [0, 1, 2, 255]);
    }

    #[test]
    fn untrusted_advice_encoding_roundtrip() {
        assert_advice_encoding(PackedAdviceKind::Untrusted, [255, 0, 7, 8]);
    }

    fn assert_advice_encoding(kind: PackedAdviceKind, bytes: [u8; 4]) {
        let layout = advice_layout(kind);

        let mut builder = JoltPackedWitnessBuilder::new(layout);
        let _ = builder
            .pack_advice_bytes(kind, &bytes)
            .expect("advice packing should succeed");
        let witness = builder.finish().expect("source should build");

        for (row, byte) in bytes.iter().copied().enumerate() {
            assert_eq!(
                get(
                    &witness,
                    PackedFamilyId::AdviceBytes { kind, index: 0 },
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
                    PackedFamilyId::IncByte { index },
                    trace_domain(),
                    1,
                    PackedAlphabet::Byte,
                )
            })
            .collect::<Vec<_>>();
        specs.push(PackedFamilySpec::direct(
            PackedFamilyId::IncSign,
            trace_domain(),
            1,
            PackedAlphabet::Bit,
        ));
        PackedWitnessLayout::new(specs).expect("layout should build")
    }
}
