//! Committed one-hot RA chunk witnesses: per-cycle hot addresses of one
//! chunk of an address decomposition. The chunk selector is the
//! `ExtractIndexed` index binding — which chunk of which decomposition is
//! bound at the use site.

use jolt_riscv::JoltTraceRow as TraceRow;

use super::{BytecodePc, Extract, ExtractIndexed, LookupIndex, RemappedRamAddress, WitnessEnv};
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

    pub const fn shift(self) -> usize {
        self.shift
    }
}

/// Writes every chunk of the `chunks.len()`-chunk decomposition of `value`,
/// one per byte, most significant first: `chunks[index]` is
/// `RaChunkSelector::new(index, chunks.len(), chunk_bits)?.chunk_u128(value)`.
/// `chunk_bits` is at most 8 so each chunk fits its byte.
pub fn write_ra_chunks(value: u128, chunk_bits: usize, chunks: &mut [u8]) {
    debug_assert!((1..=8).contains(&chunk_bits));
    if chunk_bits == 8 {
        let bytes = value.to_be_bytes();
        chunks.copy_from_slice(&bytes[bytes.len() - chunks.len()..]);
        return;
    }
    let mask = (1u128 << chunk_bits) - 1;
    for (remaining, chunk) in chunks.iter_mut().rev().enumerate() {
        *chunk = ((value >> (remaining * chunk_bits)) & mask) as u8;
    }
}

/// Hot address of one committed `InstructionRa` chunk: the selected chunk of
/// the instruction's lookup index. Every cycle is hot — no-op rows look up
/// index 0 — so there is no cold case.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InstructionRaChunk(pub usize);

/// Hot address of one committed `BytecodeRa` chunk: the selected chunk of
/// the bytecode PC. Every cycle is hot — [`BytecodePc`] is total, so no-op
/// rows land on the slot-0 chunk. `RamRaChunk` is the only cold one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BytecodeRaChunk(pub usize);

/// Hot address of one committed `RamRa` chunk: the selected chunk of the
/// remapped RAM word address; cold for no-ops and unremappable addresses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamRaChunk(pub Option<usize>);

// A chunk witness is its per-cycle hot address; `None` is a cold cycle.
// Only RAM has one — instruction and bytecode chunks are hot every cycle.
impl From<InstructionRaChunk> for Option<usize> {
    fn from(chunk: InstructionRaChunk) -> Self {
        Some(chunk.0)
    }
}

impl From<BytecodeRaChunk> for Option<usize> {
    fn from(chunk: BytecodeRaChunk) -> Self {
        Some(chunk.0)
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
        let pc = BytecodePc::extract(row, next, env)?.0;
        Ok(Self(selector.chunk_usize(pc)))
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

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use super::*;

    #[test]
    fn bulk_chunks_match_each_selector() {
        let values = [
            0u128,
            1,
            0x0123_4567_89ab_cdef,
            u128::MAX,
            1 << 127,
            0xfeed << 40,
        ];
        for (chunk_bits, chunk_counts) in [(8, [1, 3, 16]), (4, [1, 7, 32])] {
            for chunks in chunk_counts {
                for value in values {
                    let mut bulk = vec![0u8; chunks];
                    write_ra_chunks(value, chunk_bits, &mut bulk);
                    for (index, &chunk) in bulk.iter().enumerate() {
                        let selector = RaChunkSelector::new(index, chunks, chunk_bits).unwrap();
                        assert_eq!(usize::from(chunk), selector.chunk_u128(value));
                    }
                }
            }
        }
    }
}
