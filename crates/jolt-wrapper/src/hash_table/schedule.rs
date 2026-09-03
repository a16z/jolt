//! The Jolt verifier's transcript as a table segment, and the symbolic
//! schedule the verifier key is built from.
//!
//! [`JoltSchedule::new`] replays a recorded `jolt_verifier::verify` run
//! through the [`Chain`] (byte-exact, fail-closed) and classifies every item
//! structurally: the preamble is public input, labeled words are protocol
//! constants, raw 32-byte appends before the Dory segment are field elements
//! (relation wires), raw appends of the commitment and Dory segments are
//! group elements (limb-table operands), told apart by encoding length.
//!
//! The result is the [`SymbolicSchedule`]: per compression cell, the block
//! length, flags, the squeeze it serves and the [`ByteSource`] of each of its
//! 64 block bytes — a verifier-independent identity `(kind, index in the
//! schedule, byte)` that never mentions witness bytes except for protocol
//! constants, whose byte value *is* the identity. It is a deterministic
//! function of the wrapped profile (the verifier's schedule does not depend
//! on proof values), generated once with the verifier key; a proof's recorded
//! run only fills the witness and is checked against it.

use std::ops::Range;

use jolt_field::{CanonicalEncoding, Fr, Zero};
use jolt_transcript::{LabelWithCount, Transcript};

use super::blake3::{Block, ByteOrigin, Chain, BLOCK_BYTES, CHUNK_START, KEYED_HASH};
use super::recorder::{Decoder, Event, Recorded};
use super::wiring::{
    PublicInputs, VkColumns, CHALLENGE_POS, LOG_CELL, NEXT_FLAGS_POS, NEXT_LEN_POS,
};

/// A group element absorbed as bytes and committed in the limb table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ElementKind {
    /// A polynomial commitment: arkworks-uncompressed `Fq12` (12 × 32-byte
    /// little-endian coefficients, `c0` first), whole buffer reversed
    /// (`Bn254GT::append_to_transcript`).
    CommitmentGt,
    /// A Dory proof `GT`: arkworks-compressed `Fq12` (identical to
    /// uncompressed, not reversed), 384 bytes.
    DoryGt,
    /// A Dory proof `G1`: compressed x coordinate, 32 bytes little-endian,
    /// flags in the top two bits of the last byte.
    DoryG1,
    /// A Dory proof `G2`: compressed x ∈ `Fq2` (`c0`, `c1`), 64 bytes, flags
    /// in the top two bits of the last byte.
    DoryG2,
}

/// The external identity of one block byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ByteSource {
    /// Zero padding of a partial block.
    Padding,
    /// A protocol constant (domain-separation word byte).
    Constant(u8),
    /// Byte `offset` of the public preamble (hashed natively by the verifier).
    Public { offset: u32 },
    /// Byte `byte` of the `index`-th absorbed field element (32 bytes
    /// big-endian): a relation wire.
    Wire { index: u32, byte: u8 },
    Element {
        kind: ElementKind,
        index: u32,
        byte: u16,
    },
}

/// The squeeze a compression serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Squeeze {
    /// Index among the segment's squeezes.
    pub index: u32,
    pub decoder: Decoder,
}

/// One compression cell of the symbolic schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellPlan {
    pub block_len: u32,
    pub flags: u32,
    pub bytes: [ByteSource; BLOCK_BYTES],
    pub squeeze: Option<Squeeze>,
}

impl CellPlan {
    /// A padding cell: empty block, `CHUNK_START | KEYED_HASH`.
    pub const PADDING: Self = Self {
        block_len: 0,
        flags: KEYED_HASH | CHUNK_START,
        bytes: [ByteSource::Padding; BLOCK_BYTES],
        squeeze: None,
    };

    /// The constant value of half-word `half` (0..32) if both its bytes are
    /// protocol constants or padding.
    fn constant_half(&self, half: usize) -> Option<u16> {
        let value = |source: ByteSource| match source {
            ByteSource::Padding => Some(0u8),
            ByteSource::Constant(byte) => Some(byte),
            _ => None,
        };
        Some(u16::from_le_bytes([
            value(self.bytes[2 * half])?,
            value(self.bytes[2 * half + 1])?,
        ]))
    }
}

/// Table-local index of a compression cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CellIndex(pub usize);

impl CellIndex {
    pub fn first_row(self) -> usize {
        self.0 << LOG_CELL
    }
}

/// The verifier-key view of the table: one plan per cell (padding cells
/// included, so the padded row count is fixed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolicSchedule {
    pub cells: Vec<CellPlan>,
    pub log_rows: usize,
    /// The 22-or-so preamble bytes sharing the first block (public input).
    pub tail: Vec<u8>,
    /// Block length and flags the first compression uses (profile constants).
    pub first_block_len: u32,
    pub first_flags: u32,
    pub wires: u32,
    pub squeezes: u32,
    /// The cell of the stage-8 RLC γ squeeze (its chaining rows are
    /// `state_rlc`); the last cell with a squeeze holds `state_out`.
    pub rlc_cell: CellIndex,
    pub last_squeeze_cell: CellIndex,
}

impl SymbolicSchedule {
    pub fn rows(&self) -> usize {
        1 << self.log_rows
    }

    pub fn active_cells(&self) -> usize {
        self.cells
            .iter()
            .rposition(|cell| *cell != CellPlan::PADDING)
            .map_or(0, |i| i + 1)
    }

    /// The row of a squeeze's first challenge row, by squeeze index.
    pub fn challenge_rows(&self) -> Vec<(Squeeze, usize)> {
        self.cells
            .iter()
            .enumerate()
            .filter_map(|(i, cell)| {
                cell.squeeze
                    .map(|s| (s, CellIndex(i).first_row() + CHALLENGE_POS))
            })
            .collect()
    }

    /// The row and byte position of every non-padding block byte:
    /// `(source, row of the round-0 message word, byte within the word)`.
    pub fn byte_links(&self) -> impl Iterator<Item = (ByteSource, usize, u8)> + '_ {
        self.cells.iter().enumerate().flat_map(|(i, cell)| {
            cell.bytes
                .iter()
                .enumerate()
                .filter(|(_, source)| !matches!(source, ByteSource::Padding))
                .map(move |(byte, source)| {
                    (
                        *source,
                        CellIndex(i).first_row() + byte / 4,
                        (byte % 4) as u8,
                    )
                })
        })
    }

    /// The verifier-key columns: half-word pins of every constant / padding
    /// half-word of the round-0 message words, and of the next cell's block
    /// length and flags at positions 122 / 123. Public (preamble) bytes are
    /// not pinned here — the verifier adds them from the public inputs.
    pub fn vk_columns(&self) -> VkColumns {
        let mut vk = VkColumns::zero(self.rows());
        let mut pin = |row: usize, high: bool, value: u16| {
            let (is, constant) = if high {
                (&mut vk.hi_is_const, &mut vk.hi_const)
            } else {
                (&mut vk.lo_is_const, &mut vk.lo_const)
            };
            is[row] = 1;
            constant[row] = value;
        };
        for (i, cell) in self.cells.iter().enumerate() {
            let base = CellIndex(i).first_row();
            for half in 0..BLOCK_BYTES / 2 {
                if let Some(value) = cell.constant_half(half) {
                    pin(base + half / 2, half % 2 == 1, value);
                }
            }
            let next = self.cells.get(i + 1).unwrap_or(&CellPlan::PADDING);
            for (position, value) in [(NEXT_LEN_POS, next.block_len), (NEXT_FLAGS_POS, next.flags)]
            {
                pin(base + position, false, value as u16);
                pin(base + position, true, (value >> 16) as u16);
            }
        }
        vk
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ScheduleError {
    #[error("log does not start with Transcript::new")]
    MissingStart,
    #[error("no commitment absorb in the log")]
    MissingCommitments,
    #[error("no Dory segment in the log")]
    MissingDory,
    #[error("no squeeze after the Dory segment start")]
    MissingFinalSqueeze,
    #[error("raw append of {len} bytes at item {item} has no encoding")]
    UnknownEncoding { item: usize, len: usize },
    #[error("chain state after item {item} differs from the recorded transcript")]
    StateMismatch { item: usize },
    #[error("challenge of item {item} differs from the recorded transcript")]
    ChallengeMismatch { item: usize },
    #[error("{rows} rows do not fit 2^{log_rows}")]
    TooManyRows { rows: usize, log_rows: usize },
    #[error("the preamble tail sharing the first block has odd length {0}")]
    OddTail(usize),
}

/// A transcript that only collects the bytes appended to it, for encoding a
/// label word with the real `LabelWithCount` encoder.
#[derive(Default)]
struct Capture(Vec<u8>);

impl Transcript for Capture {
    type Challenge = Fr;

    fn new(_label: &'static [u8]) -> Self {
        Self::default()
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }

    fn challenge(&mut self) -> Fr {
        Fr::zero()
    }

    fn state(&self) -> [u8; 32] {
        [0; 32]
    }
}

/// The 32-byte word `LabelWithCount(label, count)` absorbs.
fn label_word(label: &'static [u8], count: u64) -> Vec<u8> {
    let mut capture = Capture::default();
    capture.append(&LabelWithCount(label, count));
    capture.0
}

/// Bytes of a logged item (empty for non-appends).
fn item_bytes(recorded: &Recorded) -> &[u8] {
    match &recorded.event {
        Event::Append { bytes, .. } => bytes,
        Event::Start { .. } | Event::Squeeze { .. } => &[],
    }
}

/// Structural class of a logged item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemClass {
    /// Preamble bytes hashed natively (public input); only the tail sharing
    /// a block with the first commitment absorb reaches the table.
    Public,
    /// A domain-separation word: fixed by the protocol.
    Constant,
    /// The `index`-th absorbed field element of the segment.
    Wire {
        index: u32,
    },
    Element {
        kind: ElementKind,
        index: u32,
    },
    /// The `index`-th squeeze of the segment.
    Squeeze(Squeeze),
    /// The start event or an append after the table's end.
    Outside,
}

/// A recorded verifier run replayed as a table: the chain (segment plus
/// padding cells), the item classes, and the symbolic schedule.
#[derive(Clone, Debug)]
pub struct JoltSchedule {
    pub chain: Chain,
    pub classes: Vec<ItemClass>,
    /// The chain blocks laid out as cells (`2^(log_rows − 7)` of them).
    pub blocks: Range<usize>,
    pub symbolic: SymbolicSchedule,
}

impl JoltSchedule {
    /// Replay a recorded `jolt_verifier::verify` run. The replayed chain is
    /// checked against the recording after every event (state) and squeeze
    /// (decoded challenge), so a table built from the result hashes exactly
    /// what the verifier's transcript hashed. `log_rows = None` picks the
    /// smallest power of two holding the segment's cells.
    pub fn new(log: &[Recorded], log_rows: Option<usize>) -> Result<Self, ScheduleError> {
        let Some(Recorded {
            event: Event::Start { label },
            ..
        }) = log.first()
        else {
            return Err(ScheduleError::MissingStart);
        };
        let mut padded = [0u8; 32];
        padded[..label.len()].copy_from_slice(label);
        let mut chain = Chain::new(blake3::hash(&padded).as_bytes());

        // Segment markers, by their labeled words: `jolt-verifier`'s
        // `b"commitment"` (count 384) and `jolt-dory`'s `b"dory_serde"`
        // (count = element length; only the label part is matched).
        let commitment_word = label_word(b"commitment", 384);
        let dory_prefix = label_word(b"dory_serde", 0)[..24].to_vec();
        let first_commitment_item = log
            .iter()
            .position(|r| item_bytes(r) == commitment_word.as_slice())
            .ok_or(ScheduleError::MissingCommitments)?;
        let dory_start_item = log
            .iter()
            .position(|r| item_bytes(r).get(..24) == Some(dory_prefix.as_slice()))
            .ok_or(ScheduleError::MissingDory)?;
        let last_squeeze_item = log
            .iter()
            .rposition(|r| matches!(r.event, Event::Squeeze { .. }))
            .filter(|&i| i > dory_start_item)
            .ok_or(ScheduleError::MissingFinalSqueeze)?;
        let rlc_item = log[..dory_start_item]
            .iter()
            .rposition(|r| matches!(r.event, Event::Squeeze { .. }))
            .ok_or(ScheduleError::MissingDory)?;

        let mut classes = Vec::with_capacity(log.len());
        let (mut wires, mut squeezes) = (0u32, 0u32);
        let mut elements = [0u32; 4];
        let mut element = |kind: ElementKind| {
            let slot = &mut elements[kind as usize];
            *slot += 1;
            ItemClass::Element {
                kind,
                index: *slot - 1,
            }
        };
        for (item, recorded) in log.iter().enumerate() {
            let class = match &recorded.event {
                Event::Start { .. } => ItemClass::Outside,
                Event::Append { bytes, labeled } => {
                    chain.absorb(item as u32, bytes);
                    if chain.state() != recorded.state {
                        return Err(ScheduleError::StateMismatch { item });
                    }
                    if item < first_commitment_item {
                        ItemClass::Public
                    } else if item > last_squeeze_item {
                        ItemClass::Outside
                    } else if *labeled {
                        ItemClass::Constant
                    } else if item >= dory_start_item {
                        match bytes.len() {
                            384 => element(ElementKind::DoryGt),
                            64 => element(ElementKind::DoryG2),
                            32 => element(ElementKind::DoryG1),
                            len => return Err(ScheduleError::UnknownEncoding { item, len }),
                        }
                    } else {
                        match bytes.len() {
                            384 => element(ElementKind::CommitmentGt),
                            32 => {
                                wires += 1;
                                ItemClass::Wire { index: wires - 1 }
                            }
                            0 => ItemClass::Constant,
                            len => return Err(ScheduleError::UnknownEncoding { item, len }),
                        }
                    }
                }
                Event::Squeeze { decoder, value } => {
                    let challenge = chain.squeeze(item as u32);
                    let decoded = match decoder {
                        Decoder::Challenge125 => Fr::from_challenge_bytes(&challenge),
                        Decoder::Scalar128 => Fr::from_scalar_challenge_bytes(&challenge),
                    };
                    if decoded != *value {
                        return Err(ScheduleError::ChallengeMismatch { item });
                    }
                    if chain.state() != recorded.state {
                        return Err(ScheduleError::StateMismatch { item });
                    }
                    if item > last_squeeze_item {
                        ItemClass::Outside
                    } else {
                        squeezes += 1;
                        ItemClass::Squeeze(Squeeze {
                            index: squeezes - 1,
                            decoder: *decoder,
                        })
                    }
                }
            };
            classes.push(class);
        }
        let squeeze_block = |chain: &Chain, item: usize| -> Result<usize, ScheduleError> {
            chain
                .blocks
                .iter()
                .position(|block| block.squeeze == Some(item as u32))
                .ok_or(ScheduleError::MissingFinalSqueeze)
        };
        let first_block = chain
            .blocks
            .iter()
            .position(|block| {
                block
                    .origins
                    .iter()
                    .flatten()
                    .any(|origin| origin.item as usize == first_commitment_item)
            })
            .ok_or(ScheduleError::MissingCommitments)?;
        let last_block = squeeze_block(&chain, last_squeeze_item)?;
        let rlc_block = squeeze_block(&chain, rlc_item)?;
        let active = last_block + 1 - first_block;
        let min_log_rows = (active << LOG_CELL).next_power_of_two().ilog2() as usize;
        let log_rows = log_rows.unwrap_or(min_log_rows);
        if log_rows < min_log_rows {
            return Err(ScheduleError::TooManyRows {
                rows: active << LOG_CELL,
                log_rows,
            });
        }
        let cells = 1usize << (log_rows - LOG_CELL);
        // Padding cells continue the chain past the final squeeze.
        chain.blocks.truncate(last_block + 1);
        for _ in active..cells {
            chain.pad();
        }
        let blocks = first_block..first_block + cells;

        // Preamble byte offsets of the public items.
        let mut preamble_offset = Vec::with_capacity(first_commitment_item);
        let mut offset = 0u32;
        for recorded in &log[..first_commitment_item] {
            preamble_offset.push(offset);
            offset += item_bytes(recorded).len() as u32;
        }
        let source = |origin: Option<ByteOrigin>| -> ByteSource {
            let Some(ByteOrigin { item, offset }) = origin else {
                return ByteSource::Padding;
            };
            match classes[item as usize] {
                ItemClass::Public => ByteSource::Public {
                    offset: preamble_offset[item as usize] + offset,
                },
                ItemClass::Constant => {
                    ByteSource::Constant(item_bytes(&log[item as usize])[offset as usize])
                }
                ItemClass::Wire { index } => ByteSource::Wire {
                    index,
                    byte: offset as u8,
                },
                ItemClass::Element { kind, index } => ByteSource::Element {
                    kind,
                    index,
                    byte: offset as u16,
                },
                ItemClass::Squeeze(_) | ItemClass::Outside => {
                    unreachable!("block bytes come from appends inside the segment")
                }
            }
        };
        let plans: Vec<CellPlan> = chain.blocks[blocks.clone()]
            .iter()
            .map(|block| CellPlan {
                block_len: block.compression.block_len,
                flags: block.compression.flags,
                bytes: std::array::from_fn(|i| source(block.origins[i])),
                squeeze: block.squeeze.and_then(|item| match classes[item as usize] {
                    ItemClass::Squeeze(squeeze) => Some(squeeze),
                    _ => None,
                }),
            })
            .collect();
        let first_words = &chain.blocks[first_block].compression.block;
        let tail: Vec<u8> = plans[0]
            .bytes
            .iter()
            .take_while(|source| matches!(source, ByteSource::Public { .. }))
            .enumerate()
            .map(|(i, _)| first_words[i / 4].to_le_bytes()[i % 4])
            .collect();
        if tail.len() % 2 == 1 {
            return Err(ScheduleError::OddTail(tail.len()));
        }
        let first = &chain.blocks[first_block].compression;
        let symbolic = SymbolicSchedule {
            first_block_len: first.block_len,
            first_flags: first.flags,
            cells: plans,
            log_rows,
            tail,
            wires,
            squeezes,
            rlc_cell: CellIndex(rlc_block - first_block),
            last_squeeze_cell: CellIndex(last_block - first_block),
        };
        Ok(Self {
            chain,
            classes,
            blocks,
            symbolic,
        })
    }

    /// The blocks laid out as cells, padding included.
    pub fn table_blocks(&self) -> &[Block] {
        &self.chain.blocks[self.blocks.clone()]
    }

    /// The public inputs of the table.
    pub fn public_inputs(&self) -> PublicInputs {
        PublicInputs {
            state_in: self.table_blocks()[0].compression.cv,
            block_len: self.symbolic.first_block_len,
            flags: self.symbolic.first_flags,
            tail: self.symbolic.tail.clone(),
        }
    }
}
