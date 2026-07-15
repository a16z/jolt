//! Committed one-hot RA chunk witnesses: per-cycle hot addresses of one
//! chunk of an address decomposition. The chunk selector is the
//! `ExtractIndexed` index binding — which chunk of which decomposition is
//! bound at the use site.

use jolt_program::execution::TraceRow;

use super::{Extract, ExtractIndexed, LookupIndex, MappedPc, RemappedRamAddress, WitnessEnv};
use crate::{WitnessError, JOLT_VM_LABEL};

/// Selects one `chunk_bits`-wide chunk of a decomposed address, indexed from
/// the most significant chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaChunkSelector {
    shift: usize,
    mask: u128,
}

impl RaChunkSelector {
    pub fn new(index: usize, chunks: usize, chunk_bits: usize) -> Result<Self, WitnessError> {
        let remaining = chunks
            .checked_sub(index + 1)
            .ok_or(WitnessError::UnknownOracle {
                label: JOLT_VM_LABEL,
            })?;
        let shift =
            remaining
                .checked_mul(chunk_bits)
                .ok_or_else(|| WitnessError::InvalidDimensions {
                    label: JOLT_VM_LABEL,
                    reason: "RA chunk shift overflow".to_owned(),
                })?;
        if chunk_bits >= u128::BITS as usize {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: "RA chunk width overflow".to_owned(),
            });
        }
        Ok(Self {
            shift,
            mask: (1_u128 << chunk_bits) - 1,
        })
    }

    pub const fn chunk_usize(self, value: usize) -> usize {
        self.chunk_u128(value as u128)
    }

    pub const fn chunk_u128(self, value: u128) -> usize {
        ((value >> self.shift) & self.mask) as usize
    }
}

/// Hot address of one committed `InstructionRa` chunk: the selected chunk of
/// the instruction's lookup index. Every cycle is hot — no-op rows look up
/// index 0 — so, unlike its bytecode and RAM siblings, there is no cold case.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InstructionRaChunk(pub usize);

/// Hot address of one committed `BytecodeRa` chunk: the selected chunk of
/// the bytecode PC; cold when the row has no bytecode mapping.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BytecodeRaChunk(pub Option<usize>);

/// Hot address of one committed `RamRa` chunk: the selected chunk of the
/// remapped RAM word address; cold for no-ops and unremappable addresses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamRaChunk(pub Option<usize>);

// A chunk witness is its per-cycle hot address; `None` is a cold cycle.
// Instruction chunks are hot every cycle (no-op rows look up index 0).
impl From<InstructionRaChunk> for Option<usize> {
    fn from(chunk: InstructionRaChunk) -> Self {
        Some(chunk.0)
    }
}

impl From<BytecodeRaChunk> for Option<usize> {
    fn from(chunk: BytecodeRaChunk) -> Self {
        chunk.0
    }
}

impl From<RamRaChunk> for Option<usize> {
    fn from(chunk: RamRaChunk) -> Self {
        chunk.0
    }
}

impl ExtractIndexed<RaChunkSelector> for InstructionRaChunk {
    fn extract_indexed(
        selector: RaChunkSelector,
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let index = LookupIndex::extract(row, next, env)?.0;
        Ok(Self(selector.chunk_u128(index)))
    }
}

impl ExtractIndexed<RaChunkSelector> for BytecodeRaChunk {
    fn extract_indexed(
        selector: RaChunkSelector,
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(
            MappedPc::extract(row, next, env)?
                .0
                .map(|pc| selector.chunk_usize(pc)),
        ))
    }
}

impl ExtractIndexed<RaChunkSelector> for RamRaChunk {
    fn extract_indexed(
        selector: RaChunkSelector,
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(
            RemappedRamAddress::extract(row, next, env)?
                .0
                .map(|address| selector.chunk_usize(address as usize)),
        ))
    }
}
