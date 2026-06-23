use jolt_akita::AkitaField;
use jolt_claims::protocols::jolt::unsigned_inc_lower_chunk_count;
use jolt_field::{CanonicalBytes, FromPrimitiveInt};
use jolt_openings::{
    PackingAdviceKind, PackingCellAddress, PackingFactDomain, PackingFamilyId, PackingLayoutError,
    PackingWitnessLayout, SparsePackingWitness,
};
use jolt_riscv::JoltTraceRow;
use thiserror::Error;

use crate::{
    stages::stage8::{
        lattice_protocol_config_for_packed_witness_layout,
        validate_lattice_packed_witness_layout_config,
    },
    VerifierError,
};

#[derive(Clone, Debug)]
pub struct AkitaPackingJoltWitnessInput<'a> {
    pub layout: PackingWitnessLayout,
    pub trace_rows: &'a [JoltTraceRow],
    pub log_k_chunk: usize,
    pub instruction_lookup_indices: &'a [u128],
    pub remapped_ram_addresses: Option<&'a [Option<u64>]>,
    pub untrusted_advice: Option<&'a [u8]>,
}

#[derive(Clone, Debug)]
pub struct JoltPackedWitnessBuilder {
    layout: PackingWitnessLayout,
    entries: Vec<(usize, AkitaField)>,
}

impl JoltPackedWitnessBuilder {
    pub fn new(layout: PackingWitnessLayout) -> Self {
        Self {
            layout,
            entries: Vec::new(),
        }
    }

    pub fn layout(&self) -> &PackingWitnessLayout {
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
            matches!(family, PackingFamilyId::InstructionRa { .. })
        });
        let bytecode_chunks = self
            .max_trace_family_index(|family| matches!(family, PackingFamilyId::BytecodeRa { .. }));
        let ram_chunks =
            self.max_trace_family_index(|family| matches!(family, PackingFamilyId::RamRa { .. }));

        for (row_index, row) in rows.iter().enumerate() {
            let lookup_index = lookup_index(row_index, row);
            for index in 0..instruction_chunks {
                let symbol = chunk(lookup_index, index, instruction_chunks, log_k_chunk)?;
                self.emit_one(
                    PackingFamilyId::InstructionRa { index },
                    row_index,
                    0,
                    symbol,
                )?;
            }

            let pc = row.pc() as u128;
            for index in 0..bytecode_chunks {
                let symbol = chunk(pc, index, bytecode_chunks, log_k_chunk)?;
                self.emit_one(PackingFamilyId::BytecodeRa { index }, row_index, 0, symbol)?;
            }

            if let Some(address) = ram_address(row_index, row) {
                for index in 0..ram_chunks {
                    let symbol = chunk(address as u128, index, ram_chunks, log_k_chunk)?;
                    self.emit_one(PackingFamilyId::RamRa { index }, row_index, 0, symbol)?;
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
        let kind = PackingAdviceKind::Untrusted;
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
            self.emit_byte(
                PackingFamilyId::AdviceBytes { kind, index: 0 },
                row,
                0,
                byte,
            )?;
        }
        Ok(self)
    }

    pub fn finish(self) -> Result<SparsePackingWitness<AkitaField>, JoltPackedWitnessError> {
        SparsePackingWitness::try_new(self.layout, self.entries).map_err(Into::into)
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
            .family(&PackingFamilyId::FieldRdIncByte { index: 0 })
            .is_some()
        {
            let rd_delta = if row.rd_index().is_some() {
                row.rd_write_value() as i128 - row.rd_pre_value() as i128
            } else {
                0
            };
            let encoded = AkitaField::from_i128(rd_delta).to_bytes_le_vec();
            for (index, byte) in encoded.into_iter().enumerate() {
                self.emit_byte(
                    PackingFamilyId::FieldRdIncByte { index },
                    row_index,
                    0,
                    byte,
                )?;
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
                PackingFamilyId::UnsignedIncChunk { index },
                row,
                0,
                little_endian_chunk(lower, index, log_k_chunk)?,
            )?;
        }
        if msb == 1 {
            self.emit_one(PackingFamilyId::UnsignedIncMsb, row, 0, 1)?;
        }
        Ok(())
    }

    fn emit_byte(
        &mut self,
        family: PackingFamilyId,
        row: usize,
        limb: usize,
        byte: u8,
    ) -> Result<(), JoltPackedWitnessError> {
        self.emit_one(family, row, limb, byte as usize)
    }

    fn emit_one(
        &mut self,
        family: PackingFamilyId,
        row: usize,
        limb: usize,
        symbol: usize,
    ) -> Result<(), JoltPackedWitnessError> {
        if self.layout.family(&family).is_none() {
            return Ok(());
        }
        let rank = self.layout.rank(&PackingCellAddress {
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
            if matches!(family.domain, PackingFactDomain::TraceRows { .. }) {
                merge_domain_rows(&mut rows, family.domain, "trace rows")?;
            }
        }
        Ok(rows)
    }

    fn advice_byte_count(
        &self,
        kind: PackingAdviceKind,
    ) -> Result<Option<usize>, JoltPackedWitnessError> {
        let mut rows = None;
        for family in &self.layout.families {
            if family.id == (PackingFamilyId::AdviceBytes { kind, index: 0 }) {
                merge_domain_rows(&mut rows, family.domain, "untrusted advice bytes")?;
            }
        }
        Ok(rows)
    }

    fn max_trace_family_index(&self, is_family: impl Fn(&PackingFamilyId) -> bool) -> usize {
        self.layout
            .families
            .iter()
            .filter(|family| matches!(family.domain, PackingFactDomain::TraceRows { .. }))
            .filter(|family| is_family(&family.id))
            .filter_map(|family| match family.id {
                PackingFamilyId::InstructionRa { index }
                | PackingFamilyId::BytecodeRa { index }
                | PackingFamilyId::RamRa { index } => Some(index + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }
}

pub fn build_akita_packing_jolt_witness(
    input: AkitaPackingJoltWitnessInput<'_>,
) -> Result<SparsePackingWitness<AkitaField>, VerifierError> {
    validate_akita_jolt_packed_witness_layout(&input.layout)?;
    let protocol = lattice_protocol_config_for_packed_witness_layout(&input.layout);
    validate_lattice_packed_witness_layout_config(&protocol, &input.layout)?;

    if input.instruction_lookup_indices.len() != input.trace_rows.len() {
        return Err(akita_witness_error(format!(
            "instruction lookup index count {} does not match trace row count {}",
            input.instruction_lookup_indices.len(),
            input.trace_rows.len()
        )));
    }
    if let Some(addresses) = input.remapped_ram_addresses {
        if addresses.len() != input.trace_rows.len() {
            return Err(akita_witness_error(format!(
                "remapped RAM address count {} does not match trace row count {}",
                addresses.len(),
                input.trace_rows.len()
            )));
        }
    }

    let mut builder = JoltPackedWitnessBuilder::new(input.layout.clone());
    builder
        .pack_trace_rows(
            input.trace_rows,
            input.log_k_chunk,
            |row, _| input.instruction_lookup_indices[row],
            |row_index, row| {
                input.remapped_ram_addresses.map_or_else(
                    || (row.is_load() || row.is_store()).then(|| row.ram_address()),
                    |addresses| addresses[row_index],
                )
            },
        )
        .map(|_| ())
        .map_err(akita_witness_error)?;

    pack_untrusted_advice_bytes(&mut builder, input.untrusted_advice)?;

    builder.finish().map_err(akita_witness_error)
}

fn validate_akita_jolt_packed_witness_layout(
    layout: &PackingWitnessLayout,
) -> Result<(), VerifierError> {
    for family in &layout.families {
        if jolt_packed_witness_family_is_precommitted(&family.id) {
            return Err(VerifierError::InvalidProtocolConfig {
                reason: format!(
                    "precommitted family {:?} cannot be included in the lattice packing witness layout",
                    family.id
                ),
            });
        }
    }
    Ok(())
}

fn jolt_packed_witness_family_is_precommitted(family: &PackingFamilyId) -> bool {
    matches!(
        family,
        PackingFamilyId::AdviceBytes {
            kind: PackingAdviceKind::Trusted,
            ..
        } | PackingFamilyId::BytecodeChunk { .. }
            | PackingFamilyId::BytecodeRegisterSelector { .. }
            | PackingFamilyId::BytecodeCircuitFlag { .. }
            | PackingFamilyId::BytecodeInstructionFlag { .. }
            | PackingFamilyId::BytecodeLookupSelector { .. }
            | PackingFamilyId::BytecodeRafFlag { .. }
            | PackingFamilyId::BytecodeUnexpandedPcBytes { .. }
            | PackingFamilyId::BytecodeImmBytes { .. }
            | PackingFamilyId::ProgramImageInit
    )
}

fn pack_untrusted_advice_bytes(
    builder: &mut JoltPackedWitnessBuilder,
    bytes: Option<&[u8]>,
) -> Result<(), VerifierError> {
    let expected = expected_rows_for_family(
        builder.layout(),
        |id| {
            matches!(
                id,
                PackingFamilyId::AdviceBytes {
                    kind: PackingAdviceKind::Untrusted,
                    index: 0,
                }
            )
        },
        "untrusted advice bytes",
    )?;
    let Some(expected) = expected else {
        if bytes.is_none_or(<[u8]>::is_empty) {
            return Ok(());
        }
        return Err(akita_witness_error(format!(
            "{} were supplied but the packed layout has no matching advice family",
            "untrusted advice bytes"
        )));
    };
    let padded = padded_slice(
        bytes.unwrap_or_default(),
        expected,
        "untrusted advice bytes",
    )?;
    builder
        .pack_untrusted_advice_bytes(&padded)
        .map(|_| ())
        .map_err(akita_witness_error)
}

fn expected_rows_for_family(
    layout: &PackingWitnessLayout,
    mut matches_family: impl FnMut(&PackingFamilyId) -> bool,
    domain: &'static str,
) -> Result<Option<usize>, VerifierError> {
    let mut rows = None;
    for family in &layout.families {
        if !matches_family(&family.id) {
            continue;
        }
        let got = domain_rows(family.domain).map_err(akita_witness_error)?;
        match rows {
            Some(expected) if expected != got => {
                return Err(akita_witness_error(format!(
                    "{domain} layout row count mismatch: expected {expected}, got {got}"
                )));
            }
            Some(_) => {}
            None => rows = Some(got),
        }
    }
    Ok(rows)
}

fn padded_slice<T: Clone + Default>(
    values: &[T],
    expected: usize,
    domain: &'static str,
) -> Result<Vec<T>, VerifierError> {
    if values.len() > expected {
        return Err(akita_witness_error(format!(
            "{domain} length {} exceeds packed layout size {expected}",
            values.len()
        )));
    }
    let mut padded = values.to_vec();
    padded.resize_with(expected, T::default);
    Ok(padded)
}

fn akita_witness_error(reason: impl ToString) -> VerifierError {
    VerifierError::LatticePackingCommitmentFailed {
        reason: format!(
            "lattice packing witness packing failed: {}",
            reason.to_string()
        ),
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
    Layout(#[from] PackingLayoutError),
}

fn merge_domain_rows(
    rows: &mut Option<usize>,
    domain: PackingFactDomain,
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

fn domain_rows(domain: PackingFactDomain) -> Result<usize, JoltPackedWitnessError> {
    let log_rows = match domain {
        PackingFactDomain::TraceRows { log_t } => log_t,
        PackingFactDomain::BytecodeRows { log_bytecode } => log_bytecode,
        PackingFactDomain::ProgramImageWords { log_words } => log_words,
        PackingFactDomain::AdviceBytes { log_bytes, .. } => log_bytes,
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
    use jolt_field::FixedByteSize;
    use jolt_openings::{PackingAlphabet, PackingFamilySpec, PackingWitnessSource};
    use jolt_riscv::{
        CapturedState, JoltInstructionKind, JoltInstructionRow, LoadState, NormalizedOperands,
        StoreState,
    };

    fn trace_domain() -> PackingFactDomain {
        PackingFactDomain::TraceRows { log_t: 1 }
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
        witness: &SparsePackingWitness<AkitaField>,
        family: PackingFamilyId,
        row: usize,
        limb: usize,
        symbol: usize,
    ) -> AkitaField {
        witness
            .eval_direct_fact(&PackingCellAddress {
                family,
                row,
                limb,
                symbol,
            })
            .expect("address should be in layout")
    }

    #[test]
    fn packs_trace_ra_and_unsigned_increment_facts() {
        let layout = PackingWitnessLayout::new([
            PackingFamilySpec::direct(
                PackingFamilyId::InstructionRa { index: 0 },
                trace_domain(),
                1,
                PackingAlphabet::Byte,
            ),
            PackingFamilySpec::direct(
                PackingFamilyId::BytecodeRa { index: 0 },
                trace_domain(),
                1,
                PackingAlphabet::Byte,
            ),
            PackingFamilySpec::direct(
                PackingFamilyId::RamRa { index: 0 },
                trace_domain(),
                1,
                PackingAlphabet::Byte,
            ),
            PackingFamilySpec::direct(
                PackingFamilyId::UnsignedIncChunk { index: 0 },
                trace_domain(),
                1,
                PackingAlphabet::Byte,
            ),
            PackingFamilySpec::direct(
                PackingFamilyId::UnsignedIncMsb,
                trace_domain(),
                1,
                PackingAlphabet::Bit,
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
                PackingFamilyId::InstructionRa { index: 0 },
                0,
                0,
                0x7f
            ),
            AkitaField::one()
        );
        assert_eq!(
            get(&witness, PackingFamilyId::BytecodeRa { index: 0 }, 1, 0, 11),
            AkitaField::one()
        );
        assert_eq!(
            get(&witness, PackingFamilyId::RamRa { index: 0 }, 1, 0, 0x42),
            AkitaField::one()
        );
        assert_eq!(
            get(
                &witness,
                PackingFamilyId::UnsignedIncChunk { index: 0 },
                0,
                0,
                249
            ),
            AkitaField::one()
        );
        assert!(get(&witness, PackingFamilyId::UnsignedIncMsb, 0, 0, 1).is_zero());
        assert_eq!(
            get(
                &witness,
                PackingFamilyId::UnsignedIncChunk { index: 0 },
                1,
                0,
                20
            ),
            AkitaField::one()
        );
        assert_eq!(
            get(&witness, PackingFamilyId::UnsignedIncMsb, 1, 0, 1),
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
                    PackingFamilyId::UnsignedIncChunk { index },
                    0,
                    0,
                    0
                ),
                AkitaField::one()
            );
        }
        assert_eq!(
            get(&witness, PackingFamilyId::UnsignedIncMsb, 0, 0, 1),
            AkitaField::one()
        );
    }

    #[test]
    fn store_zero_delta_uses_ram_increment_source() {
        let layout = increment_layout();
        let rows = [
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
                    rs2_value: 17,
                    ram_read_value: 17,
                    ram_address: 0x34,
                }),
                9,
            ),
            JoltTraceRow::no_op(),
        ];

        let mut builder = JoltPackedWitnessBuilder::new(layout);
        let _ = builder
            .pack_trace_rows(&rows, 8, |_, _| 0, |index, _| [Some(0x34), None][index])
            .expect("trace packing should succeed");
        let witness = builder.finish().expect("source should build");

        for index in 0..8 {
            assert_eq!(
                get(
                    &witness,
                    PackingFamilyId::UnsignedIncChunk { index },
                    0,
                    0,
                    0
                ),
                AkitaField::one()
            );
        }
        assert_eq!(
            get(&witness, PackingFamilyId::UnsignedIncMsb, 0, 0, 1),
            AkitaField::one()
        );
    }

    #[test]
    fn load_row_uses_rd_increment_source_not_ram_delta() {
        let layout = increment_layout();
        let rows = [
            trace_row(
                JoltInstructionKind::LD,
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: None,
                    rd: Some(3),
                    imm: 8,
                },
                CapturedState::Load(LoadState {
                    rs1_value: 1,
                    ram_address: 0x34,
                    rd_pre_value: 10,
                    rd_write_value: 13,
                }),
                9,
            ),
            JoltTraceRow::no_op(),
        ];

        let mut builder = JoltPackedWitnessBuilder::new(layout);
        let _ = builder
            .pack_trace_rows(&rows, 8, |_, _| 0, |index, _| [Some(0x34), None][index])
            .expect("trace packing should succeed");
        let witness = builder.finish().expect("source should build");

        assert_eq!(
            get(
                &witness,
                PackingFamilyId::UnsignedIncChunk { index: 0 },
                0,
                0,
                3
            ),
            AkitaField::one()
        );
        for index in 1..8 {
            assert_eq!(
                get(
                    &witness,
                    PackingFamilyId::UnsignedIncChunk { index },
                    0,
                    0,
                    0
                ),
                AkitaField::one()
            );
        }
        assert_eq!(
            get(&witness, PackingFamilyId::UnsignedIncMsb, 0, 0, 1),
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
                    PackingFamilyId::UnsignedIncChunk { index },
                    0,
                    0,
                    0
                ),
                AkitaField::one()
            );
        }
        assert_eq!(
            get(&witness, PackingFamilyId::UnsignedIncMsb, 0, 0, 1),
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
                PackingFamilyId::UnsignedIncChunk { index: 0 },
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
                    PackingFamilyId::UnsignedIncChunk { index },
                    0,
                    0,
                    255
                ),
                AkitaField::one()
            );
        }
        assert!(get(&witness, PackingFamilyId::UnsignedIncMsb, 0, 0, 1).is_zero());
    }

    #[test]
    fn unsigned_increment_supports_four_bit_lower_chunks() {
        let layout = increment_layout_with_chunk_bits(4);
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
                    rd_pre_value: 0,
                    rd_write_value: 0x1234,
                }),
                9,
            ),
            JoltTraceRow::no_op(),
        ];

        let mut builder = JoltPackedWitnessBuilder::new(layout);
        let _ = builder
            .pack_trace_rows(&rows, 4, |_, _| 0, |_, _| None)
            .expect("trace packing should succeed");
        let witness = builder.finish().expect("source should build");

        let expected = [4, 3, 2, 1];
        for (index, symbol) in expected.into_iter().enumerate() {
            assert_eq!(
                get(
                    &witness,
                    PackingFamilyId::UnsignedIncChunk { index },
                    0,
                    0,
                    symbol
                ),
                AkitaField::one()
            );
        }
        for index in expected.len()..16 {
            assert_eq!(
                get(
                    &witness,
                    PackingFamilyId::UnsignedIncChunk { index },
                    0,
                    0,
                    0
                ),
                AkitaField::one()
            );
        }
        assert_eq!(
            get(&witness, PackingFamilyId::UnsignedIncMsb, 0, 0, 1),
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
                    PackingFamilyId::FieldRdIncByte { index },
                    0,
                    0,
                    byte as usize
                ),
                AkitaField::one()
            );
        }
        assert!(witness
            .layout()
            .family(&PackingFamilyId::FieldRdIncSign)
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
                get(&witness, PackingFamilyId::FieldRdIncByte { index }, 0, 0, 0),
                AkitaField::one()
            );
        }
    }

    #[test]
    fn untrusted_advice_encoding_roundtrip() {
        let bytes = [255, 0, 7, 8];
        let layout = advice_layout(PackingAdviceKind::Untrusted);

        let mut builder = JoltPackedWitnessBuilder::new(layout);
        let _ = builder
            .pack_untrusted_advice_bytes(&bytes)
            .expect("advice packing should succeed");
        let witness = builder.finish().expect("source should build");

        for (row, byte) in bytes.iter().copied().enumerate() {
            assert_eq!(
                get(
                    &witness,
                    PackingFamilyId::AdviceBytes {
                        kind: PackingAdviceKind::Untrusted,
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

    fn advice_layout(kind: PackingAdviceKind) -> PackingWitnessLayout {
        PackingWitnessLayout::new([PackingFamilySpec::direct(
            PackingFamilyId::AdviceBytes { kind, index: 0 },
            PackingFactDomain::AdviceBytes { kind, log_bytes: 2 },
            1,
            PackingAlphabet::Byte,
        )])
        .expect("layout should build")
    }

    fn field_rd_inc_layout() -> PackingWitnessLayout {
        PackingWitnessLayout::new((0..AkitaField::NUM_BYTES).map(|index| {
            PackingFamilySpec::direct(
                PackingFamilyId::FieldRdIncByte { index },
                trace_domain(),
                1,
                PackingAlphabet::Byte,
            )
        }))
        .expect("layout should build")
    }

    fn increment_layout() -> PackingWitnessLayout {
        increment_layout_with_chunk_bits(8)
    }

    fn increment_layout_with_chunk_bits(log_k_chunk: usize) -> PackingWitnessLayout {
        let chunk_count =
            unsigned_inc_lower_chunk_count(log_k_chunk).expect("valid test chunk size");
        let alphabet = PackingAlphabet::Fixed {
            size: 1 << log_k_chunk,
        };
        let mut specs = (0..chunk_count)
            .map(|index| {
                PackingFamilySpec::direct(
                    PackingFamilyId::UnsignedIncChunk { index },
                    trace_domain(),
                    1,
                    alphabet,
                )
            })
            .collect::<Vec<_>>();
        specs.push(PackingFamilySpec::direct(
            PackingFamilyId::UnsignedIncMsb,
            trace_domain(),
            1,
            PackingAlphabet::Bit,
        ));
        PackingWitnessLayout::new(specs).expect("layout should build")
    }
}
