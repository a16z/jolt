//! The Jolt verifier's transcript as a table segment: replay of a recorded
//! run through the [`Chain`], the in-table block range (from the block holding
//! the first commitment absorb to the Dory `d` squeeze), and the external
//! source of every absorbed item.
//!
//! Items are classified structurally: the preamble is public input, labeled
//! words are constants, raw 32-byte appends before the Dory segment are field
//! elements (R1CS wires), raw appends of the commitment segment and of the
//! Dory segment are group elements (limb-table operands), told apart by
//! encoding length.

use std::ops::Range;

use jolt_field::{CanonicalEncoding, Fr, Zero};
use jolt_transcript::{LabelWithCount, Transcript};

use super::blake3::{Block, Chain};
use super::recorder::{Decoder, Event, Recorded};

/// A group element absorbed as bytes and committed in the limb table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemClass {
    /// Preamble bytes hashed natively (public input); only the tail sharing
    /// a block with the first commitment absorb reaches the table.
    Public,
    /// A domain-separation word: fixed by the protocol.
    Constant,
    /// The `index`-th absorbed field element of the segment (32 bytes
    /// big-endian): an R1CS wire.
    Wire {
        index: usize,
    },
    Element {
        kind: ElementKind,
        index: usize,
    },
    /// The `index`-th squeeze of the segment.
    Squeeze {
        index: usize,
        decoder: Decoder,
    },
    /// The start event or an append after the table's end.
    Outside,
}

#[derive(Clone, Debug)]
pub struct JoltSchedule {
    /// The replayed chain.
    pub chain: Chain,
    pub classes: Vec<ItemClass>,
    /// The in-table compressions.
    pub blocks: Range<usize>,
    /// The compression of the stage-8 RLC γ squeeze (its chaining rows hold
    /// `state_rlc`).
    pub rlc_block: usize,
    pub first_commitment_item: usize,
    pub dory_start_item: usize,
    pub wires: usize,
    pub squeezes: usize,
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

/// Bytes of item `item` in `log` (empty for non-appends).
fn item_bytes(recorded: &Recorded) -> &[u8] {
    match &recorded.event {
        Event::Append { bytes, .. } => bytes,
        Event::Start { .. } | Event::Squeeze { .. } => &[],
    }
}

impl JoltSchedule {
    /// Replay a recorded `jolt_verifier::verify` run. The replayed chain is
    /// checked against the recording after every event (state) and squeeze
    /// (decoded challenge), so a table built from the result hashes exactly
    /// what the verifier's transcript hashed.
    pub fn new(log: &[Recorded]) -> Result<Self, ScheduleError> {
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
        let (mut wires, mut squeezes) = (0, 0);
        let mut elements = [0usize; 4];
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
                        ItemClass::Squeeze {
                            index: squeezes - 1,
                            decoder: *decoder,
                        }
                    }
                }
            };
            classes.push(class);
        }
        let squeeze_block = |item: usize| -> Result<usize, ScheduleError> {
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
        let last_block = squeeze_block(last_squeeze_item)?;
        let rlc_block = squeeze_block(rlc_item)?;
        Ok(Self {
            chain,
            classes,
            blocks: first_block..last_block + 1,
            rlc_block,
            first_commitment_item,
            dory_start_item,
            wires,
            squeezes,
        })
    }

    pub fn table_blocks(&self) -> &[Block] {
        &self.chain.blocks[self.blocks.clone()]
    }
}
