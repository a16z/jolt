use jolt_akita::AkitaField;
use jolt_claims::protocols::jolt::unsigned_inc_lower_chunk_count;
use jolt_field::{CanonicalBytes, FromPrimitiveInt};
use jolt_openings::{
    PackedAdviceKind, PackedCellAddress, PackedFactDomain, PackedFamilyId, PackedLayoutError,
    PackedWitnessLayout, SparsePackedWitness,
};
use jolt_riscv::JoltTraceRow;
use thiserror::Error;

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
#[path = "akita_witness_tests.rs"]
mod tests;
