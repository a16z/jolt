#[cfg(feature = "serialization")]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS, STACK_CANARY_SIZE};
use common::jolt_device::{JoltDevice, MemoryLayout, MemoryLayoutError};
use jolt_riscv::JoltInstructionRow;

use super::{public_io::PublicMemorySegment, PreprocessingError};

#[derive(Default, Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(
        CanonicalSerialize,
        CanonicalDeserialize,
        serde::Serialize,
        serde::Deserialize
    )
)]
pub struct RAMPreprocessing {
    pub min_bytecode_address: u64,
    pub bytecode_words: Vec<u64>,
    /// Canonical half-open intervals of protected RAM word indices.
    pub code_word_intervals: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RamDomainError {
    #[error("memory layout error: {0}")]
    MemoryLayout(MemoryLayoutError),
    #[error("memory layout heap end {heap_end} is below lowest mapped address {lowest_address}")]
    HeapBelowLowest { heap_end: u64, lowest_address: u64 },
    #[error("RAM domain size exceeds this platform's addressable usize range")]
    DomainTooLarge,
    #[error(
        "maximum padded RAM domain of {max_ram_k} words from {lowest_address} exceeds the 64-bit address space"
    )]
    AddressSpaceOverflow {
        lowest_address: u64,
        max_ram_k: usize,
    },
}

impl From<MemoryLayoutError> for RamDomainError {
    fn from(error: MemoryLayoutError) -> Self {
        Self::MemoryLayout(error)
    }
}

impl RAMPreprocessing {
    pub fn preprocess(mut memory_init: Vec<(u64, u8)>) -> Self {
        // `chunk_by` below groups only *consecutive* entries of the same
        // word; unsorted init data would split a word across chunks and let
        // a later chunk silently overwrite an earlier one byte-for-byte.
        // Sorting makes the grouping total regardless of the caller.
        memory_init.sort_by_key(|(address, _)| *address);
        let min_bytecode_address = memory_init
            .iter()
            .map(|(address, _)| *address)
            .min()
            .unwrap_or(0);

        let max_bytecode_address = memory_init
            .iter()
            .map(|(address, _)| *address)
            .max()
            .unwrap_or(0)
            + (BYTES_PER_INSTRUCTION as u64 - 1);

        let num_words = max_bytecode_address.div_ceil(8) - min_bytecode_address / 8 + 1;
        let mut bytecode_words = vec![0u64; num_words as usize];

        for chunk in
            memory_init.chunk_by(|(address_a, _), (address_b, _)| address_a / 8 == address_b / 8)
        {
            let Some(&(chunk_address, _)) = chunk.first() else {
                continue;
            };
            let mut word = [0u8; 8];
            for (address, byte) in chunk {
                #[expect(
                    clippy::indexing_slicing,
                    reason = "address % 8 < 8 indexes an 8-byte array"
                )]
                {
                    word[(address % 8) as usize] = *byte;
                }
            }
            let word = u64::from_le_bytes(word);
            let remapped_index = (chunk_address / 8 - min_bytecode_address / 8) as usize;
            #[expect(
                clippy::indexing_slicing,
                reason = "num_words covers every word between the min and max init addresses, so remapped_index < bytecode_words.len()"
            )]
            {
                bytecode_words[remapped_index] = word;
            }
        }

        Self {
            min_bytecode_address,
            bytecode_words,
            code_word_intervals: Vec::new(),
        }
    }
}

/// Round real source-instruction extents outward to RAM doublewords and merge
/// overlaps and neighbors. Expansion rows must be reduced to their source
/// anchors before calling; address-zero padding is not source code.
pub fn code_word_intervals(
    source_extents: impl IntoIterator<Item = (u64, bool)>,
    lowest_address: u64,
) -> Result<Vec<(usize, usize)>, PreprocessingError> {
    let intervals =
        source_word_intervals(source_extents, lowest_address)?.collect::<Result<Vec<_>, _>>()?;
    Ok(merge_word_intervals(intervals))
}

fn merge_word_intervals(mut intervals: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    intervals.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in intervals {
        if let Some((_, previous_end)) = merged.last_mut() {
            if start <= *previous_end {
                *previous_end = (*previous_end).max(end);
                continue;
            }
        }
        merged.push((start, end));
    }
    merged.shrink_to_fit();
    merged
}

/// The value-preserving RAM mask, restricted to the proof's address domain.
/// Layout regions and source extents round outward to doublewords; their union
/// is sorted, disjoint and merged, including intervals starting at index zero.
pub fn write_protected_word_intervals(
    code_intervals: &[(usize, usize)],
    memory_layout: &MemoryLayout,
    ram_k: usize,
) -> Result<Vec<(usize, usize)>, PreprocessingError> {
    validate_code_word_intervals(code_intervals, memory_layout)?;
    let lowest = memory_layout.get_lowest_address();
    if !lowest.is_multiple_of(8) {
        return Err(PreprocessingError::InvalidWriteProtectedIntervals(
            "lowest RAM address is not doubleword aligned",
        ));
    }
    let canary_end = memory_layout
        .stack_end
        .checked_add(STACK_CANARY_SIZE)
        .ok_or(PreprocessingError::InvalidWriteProtectedIntervals(
            "canary extent overflows",
        ))?;
    let mut intervals = code_intervals.to_vec();
    for (start, end) in [
        (memory_layout.input_start, memory_layout.input_end),
        (
            memory_layout.trusted_advice_start,
            memory_layout.trusted_advice_end,
        ),
        (
            memory_layout.untrusted_advice_start,
            memory_layout.untrusted_advice_end,
        ),
        (memory_layout.stack_end, canary_end),
    ] {
        if start > end {
            return Err(PreprocessingError::InvalidWriteProtectedIntervals(
                "region is reversed",
            ));
        }
        if start == end {
            continue;
        }
        let start =
            start
                .checked_sub(lowest)
                .ok_or(PreprocessingError::InvalidWriteProtectedIntervals(
                    "region is below lowest RAM address",
                ))?;
        let end =
            end.checked_sub(lowest)
                .ok_or(PreprocessingError::InvalidWriteProtectedIntervals(
                    "region is below lowest RAM address",
                ))?;
        let start = usize::try_from(start / 8).map_err(|_| {
            PreprocessingError::InvalidWriteProtectedIntervals("region exceeds usize range")
        })?;
        let end = usize::try_from(end.div_ceil(8)).map_err(|_| {
            PreprocessingError::InvalidWriteProtectedIntervals("region exceeds usize range")
        })?;
        intervals.push((start, end));
    }
    let mut intervals = merge_word_intervals(intervals);
    intervals.retain_mut(|(start, end)| {
        *end = (*end).min(ram_k);
        *start < *end
    });
    Ok(intervals)
}

/// The public RAM word masks of one proof's address domain, the `ram_K` words
/// from the lowest mapped address. The RAM read-write sumcheck weights one claim
/// by each mask.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RamWordMasks {
    /// Mask C: code, canary, input and advice words, whose value every access
    /// must preserve. See [`write_protected_word_intervals`].
    pub write_protected: Vec<(usize, usize)>,
    /// Mask N: words no access may touch. See [`unmapped_word_intervals`].
    pub unmapped: Vec<(usize, usize)>,
}

impl RamWordMasks {
    pub fn new(
        code_intervals: &[(usize, usize)],
        memory_layout: &MemoryLayout,
        ram_k: usize,
    ) -> Result<Self, PreprocessingError> {
        let masks = Self {
            write_protected: write_protected_word_intervals(code_intervals, memory_layout, ram_k)?,
            unmapped: unmapped_word_intervals(memory_layout, ram_k)?,
        };
        // The prover classifies a word for both masks with one search over
        // their merged intervals, which requires the masks to be disjoint.
        let overlaps = masks.write_protected.iter().any(|&(start, end)| {
            masks
                .unmapped
                .iter()
                .any(|&(other_start, other_end)| start < other_end && other_start < end)
        });
        if overlaps {
            return Err(PreprocessingError::InvalidUnmappedIntervals(
                "unmapped words overlap write-protected words",
            ));
        }
        Ok(masks)
    }
}

/// The words of the proof's domain that every tracer refuses to access: the
/// padding between `io_end` and `RAM_START_ADDRESS`, and the words from
/// `heap_end` to the end of the domain. The tracers fault on an access starting
/// at or above `io_end` below RAM, or at or above `heap_end` in RAM (see
/// `tracer::emulator::mmu::address_fault` and the x86 helpers' guest-load
/// bounds). Both bounds are doubleword aligned, so each tracer bound falls on a
/// word boundary of this mask. The panic and termination flag words lie below
/// `io_end` and are outside the mask.
pub fn unmapped_word_intervals(
    memory_layout: &MemoryLayout,
    ram_k: usize,
) -> Result<Vec<(usize, usize)>, PreprocessingError> {
    let lowest = memory_layout.get_lowest_address();
    let word = |address: u64| -> Result<usize, PreprocessingError> {
        let offset =
            address
                .checked_sub(lowest)
                .ok_or(PreprocessingError::InvalidUnmappedIntervals(
                    "bound is below lowest RAM address",
                ))?;
        if !offset.is_multiple_of(8) {
            return Err(PreprocessingError::InvalidUnmappedIntervals(
                "bound is not doubleword aligned",
            ));
        }
        usize::try_from(offset / 8)
            .map_err(|_| PreprocessingError::InvalidUnmappedIntervals("bound exceeds usize range"))
    };
    let io_gap = (word(memory_layout.io_end)?, word(RAM_START_ADDRESS)?);
    let heap_padding = (word(memory_layout.heap_end)?, ram_k);
    if io_gap.0 > io_gap.1 || io_gap.1 > heap_padding.0 {
        return Err(PreprocessingError::InvalidUnmappedIntervals(
            "regions are out of order",
        ));
    }
    let mut intervals = merge_word_intervals(vec![io_gap, heap_padding]);
    intervals.retain_mut(|(start, end)| {
        *end = (*end).min(ram_k);
        *start < *end
    });
    Ok(intervals)
}

fn source_word_intervals(
    source_extents: impl IntoIterator<Item = (u64, bool)>,
    lowest_address: u64,
) -> Result<impl Iterator<Item = Result<(usize, usize), PreprocessingError>>, PreprocessingError> {
    if !lowest_address.is_multiple_of(8) {
        return Err(PreprocessingError::InvalidCodeWordIntervals(
            "lowest RAM address is not doubleword aligned",
        ));
    }
    Ok(source_extents
        .into_iter()
        .filter(|&(address, _)| address != 0)
        .map(move |(address, is_compressed)| {
            let end = address
                .checked_add(if is_compressed { 2 } else { 4 })
                .ok_or(PreprocessingError::InvalidCodeWordIntervals(
                    "source instruction extent overflows",
                ))?;
            let offset = address.checked_sub(lowest_address).ok_or(
                PreprocessingError::InvalidCodeWordIntervals("source instruction is below RAM"),
            )?;
            if !address.is_multiple_of(2) {
                return Err(PreprocessingError::InvalidCodeWordIntervals(
                    "source instruction address is not halfword aligned",
                ));
            }
            let start = usize::try_from(offset / 8).map_err(|_| {
                PreprocessingError::InvalidCodeWordIntervals("source word exceeds usize range")
            })?;
            let end = usize::try_from((end - lowest_address).div_ceil(8)).map_err(|_| {
                PreprocessingError::InvalidCodeWordIntervals("source word exceeds usize range")
            })?;
            Ok((start, end))
        }))
}

/// Expansion metadata retains the source length on the final row only.
/// Taking each anchor avoids treating a compressed expansion as four bytes.
pub fn bytecode_code_word_intervals(
    bytecode: &[JoltInstructionRow],
    lowest_address: u64,
) -> Result<Vec<(usize, usize)>, PreprocessingError> {
    code_word_intervals(bytecode_source_extents(bytecode), lowest_address)
}

fn bytecode_source_extents(
    bytecode: &[JoltInstructionRow],
) -> impl Iterator<Item = (u64, bool)> + '_ {
    bytecode
        .iter()
        .filter(|row| row.virtual_sequence_remaining.unwrap_or(0) == 0)
        .map(|row| (row.address as u64, row.is_compressed))
}

pub(super) fn validate_bytecode_code_word_intervals(
    intervals: &[(usize, usize)],
    bytecode: &[JoltInstructionRow],
    memory_layout: &MemoryLayout,
) -> Result<(), PreprocessingError> {
    validate_code_word_intervals(intervals, memory_layout)?;
    let lowest_address = memory_layout.get_lowest_address();
    let extents = source_word_intervals(bytecode_source_extents(bytecode), lowest_address)?;
    let mut expected = intervals.iter().copied();
    let mut merged: Option<(usize, usize)> = None;
    let mut previous_start = None;
    let mut matches = true;
    for extent in extents {
        let (start, end) = extent?;
        // BytecodePCMapper::try_new checks expansion runs but does not enforce
        // address order. Only unordered word extents need the sorting fallback.
        if previous_start.is_some_and(|previous| start < previous) {
            return if intervals == bytecode_code_word_intervals(bytecode, lowest_address)? {
                Ok(())
            } else {
                Err(PreprocessingError::InvalidCodeWordIntervals(
                    "protected words do not match decoded source extents",
                ))
            };
        }
        previous_start = Some(start);
        match merged.as_mut() {
            Some((_, previous_end)) if start <= *previous_end => {
                *previous_end = (*previous_end).max(end);
            }
            _ => {
                if let Some(interval) = merged.replace((start, end)) {
                    matches &= expected.next() == Some(interval);
                }
            }
        }
    }
    if let Some(interval) = merged {
        matches &= expected.next() == Some(interval);
    }
    if !matches || expected.next().is_some() {
        return Err(PreprocessingError::InvalidCodeWordIntervals(
            "protected words do not match decoded source extents",
        ));
    }
    Ok(())
}

/// Membership in an admitted canonical interval list, without a RAM-sized mask.
pub fn is_code_word(intervals: &[(usize, usize)], word_index: usize) -> bool {
    let next = intervals.partition_point(|&(start, _)| start <= word_index);
    next.checked_sub(1)
        .and_then(|index| intervals.get(index))
        .is_some_and(|&(_, end)| word_index < end)
}

pub fn validate_code_word_intervals(
    intervals: &[(usize, usize)],
    memory_layout: &MemoryLayout,
) -> Result<(), PreprocessingError> {
    let max_ram_k = compute_max_ram_k(memory_layout)
        .map_err(|_| PreprocessingError::InvalidCodeWordIntervals("RAM domain is invalid"))?;
    let mut previous_end = None;
    for &(start, end) in intervals {
        if start >= end || end > max_ram_k {
            return Err(PreprocessingError::InvalidCodeWordIntervals(
                "protected interval is empty or outside RAM",
            ));
        }
        if previous_end.is_some_and(|previous| start <= previous) {
            return Err(PreprocessingError::InvalidCodeWordIntervals(
                "protected intervals are not sorted, disjoint and merged",
            ));
        }
        previous_end = Some(end);
    }
    Ok(())
}

pub fn compute_min_ram_k(
    min_bytecode_address: u64,
    program_image_len_words: usize,
    memory_layout: &MemoryLayout,
) -> Result<usize, RamDomainError> {
    let bytecode_start = match memory_layout.remap_word_address(min_bytecode_address)? {
        Some(address) => usize::try_from(address).map_err(|_| RamDomainError::DomainTooLarge)?,
        None => 0,
    };
    let bytecode_end = bytecode_start
        .checked_add(program_image_len_words)
        .ok_or(RamDomainError::DomainTooLarge)?;

    let io_end = usize::try_from(memory_layout.remapped_word_address(RAM_START_ADDRESS)?)
        .map_err(|_| RamDomainError::DomainTooLarge)?;

    bytecode_end
        .max(io_end)
        .checked_next_power_of_two()
        .ok_or(RamDomainError::DomainTooLarge)
}

pub fn compute_max_ram_k(memory_layout: &MemoryLayout) -> Result<usize, RamDomainError> {
    memory_layout.max_ram_k().map_err(|error| match error {
        MemoryLayoutError::HeapBelowLowest {
            heap_end,
            lowest_address,
        } => RamDomainError::HeapBelowLowest {
            heap_end,
            lowest_address,
        },
        MemoryLayoutError::RamDomainTooLarge => RamDomainError::DomainTooLarge,
        MemoryLayoutError::RamAddressSpaceOverflow {
            lowest_address,
            max_ram_k,
        } => RamDomainError::AddressSpaceOverflow {
            lowest_address,
            max_ram_k,
        },
        error @ (MemoryLayoutError::ZeroAddress
        | MemoryLayoutError::AddressBelowLowest { .. }
        | MemoryLayoutError::MissingProgramSize
        | MemoryLayoutError::SizeOverflow { .. }
        | MemoryLayoutError::InvalidTrustedAdviceSize { .. }
        | MemoryLayoutError::InvalidUntrustedAdviceSize { .. }
        | MemoryLayoutError::IoRegionTooLarge { .. }
        | MemoryLayoutError::HeapEndTooHigh { .. }
        | MemoryLayoutError::HeapEndUnaligned { .. }
        | MemoryLayoutError::IoTooLong { .. }) => RamDomainError::MemoryLayout(error),
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicInitialRam {
    pub segments: Vec<PublicMemorySegment>,
}

impl PublicInitialRam {
    pub fn new(ram: &RAMPreprocessing, public_io: &JoltDevice) -> Result<Self, MemoryLayoutError> {
        let layout = &public_io.memory_layout;
        let mut segments = Vec::new();

        if !ram.bytecode_words.is_empty() {
            segments.push(PublicMemorySegment {
                start_index: layout.remapped_word_address(ram.min_bytecode_address)? as u128,
                words: ram.bytecode_words.clone(),
            });
        }

        segments.extend(Self::inputs_only(public_io)?.segments);

        Ok(Self { segments })
    }

    /// Public initial RAM without the program image, used in committed program
    /// mode where the image contribution is a staged opening instead.
    pub fn inputs_only(public_io: &JoltDevice) -> Result<Self, MemoryLayoutError> {
        let layout = &public_io.memory_layout;
        let mut segments = Vec::new();

        if !public_io.inputs.is_empty() {
            segments.push(PublicMemorySegment {
                start_index: layout.remapped_word_address(layout.input_start)? as u128,
                words: public_io.input_words_le(),
            });
        }

        Ok(Self { segments })
    }
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "tests index fixture data and fail on construction errors"
)]
mod tests {
    use super::{
        compute_max_ram_k, compute_min_ram_k, PublicInitialRam, RAMPreprocessing, RamDomainError,
    };
    use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayoutError};

    fn bounded_device() -> JoltDevice {
        JoltDevice::new(&MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            max_input_size: 16,
            max_output_size: 16,
            stack_size: 8,
            heap_size: 8,
        })
    }

    #[test]
    fn source_code_extents_round_and_merge_without_freezing_gaps() {
        let lowest = 0x8000_0000;
        assert_eq!(
            super::code_word_intervals(
                [
                    (lowest + 38, false),
                    (lowest + 6, true),
                    (lowest + 8, false),
                    (lowest, false),
                    (0, false),
                ],
                lowest
            ),
            Ok(vec![(0, 2), (4, 6)])
        );
        let intervals = [(0, 2), (4, 6)];
        for word in 0..8 {
            assert_eq!(
                super::is_code_word(&intervals, word),
                matches!(word, 0 | 1 | 4 | 5)
            );
        }
    }

    #[test]
    fn protected_regions_merge_at_zero_and_clip_to_the_proof_domain() {
        let mut layout = bounded_device().memory_layout;
        let lowest = layout.get_lowest_address();
        layout.untrusted_advice_start = lowest;
        layout.untrusted_advice_end = lowest + 8;
        layout.trusted_advice_start = lowest + 8;
        layout.trusted_advice_end = lowest + 24;
        layout.input_start = lowest + 24;
        layout.input_end = lowest + 40;
        // Source words overlap input and touch the rounded canary extent.
        layout.stack_end = lowest + 62;
        let code = [(4, 8)];
        let intervals = super::write_protected_word_intervals(&code, &layout, 16).unwrap();
        assert_eq!(intervals, vec![(0, 16)]);
        assert_eq!(
            super::write_protected_word_intervals(&code, &layout, 32).unwrap(),
            vec![(0, 24)]
        );
        assert_eq!(
            super::write_protected_word_intervals(&code, &layout, 4).unwrap(),
            vec![(0, 4)]
        );
        layout.stack_end = lowest + 8 * 32;
        assert_eq!(
            super::write_protected_word_intervals(&code, &layout, 16).unwrap(),
            vec![(0, 8)]
        );
    }

    #[test]
    fn source_code_extents_reject_overflow_low_and_unaligned_addresses() {
        assert!(super::code_word_intervals([(u64::MAX - 1, true)], 0).is_err());
        assert!(super::code_word_intervals([(8, false)], 16).is_err());
        assert!(super::code_word_intervals([(17, true)], 16).is_err());
        assert!(super::code_word_intervals([(16, false)], 1).is_err());
    }

    #[test]
    fn preprocesses_memory_bytes_into_words() {
        let preprocessing = RAMPreprocessing::preprocess(vec![
            (0x8000_0000, 0x01),
            (0x8000_0001, 0x02),
            (0x8000_0008, 0x03),
        ]);

        assert_eq!(preprocessing.min_bytecode_address, 0x8000_0000);
        assert_eq!(preprocessing.bytecode_words[0], 0x0201);
        assert_eq!(preprocessing.bytecode_words[1], 0x03);
    }

    #[test]
    fn materializes_public_initial_ram_segments() {
        let mut device = JoltDevice::new(&MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            max_input_size: 16,
            ..Default::default()
        });
        device.inputs = vec![0x2a, 0, 0, 0, 0, 0, 0, 0, 0x07];
        let preprocessing = RAMPreprocessing {
            min_bytecode_address: 0x8000_0000,
            bytecode_words: vec![0x0201, 0x03],
            code_word_intervals: Vec::new(),
        };

        let initial = PublicInitialRam::new(&preprocessing, &device);
        assert!(
            initial.is_ok(),
            "initial RAM should materialize: {initial:?}"
        );
        let Ok(initial) = initial else {
            return;
        };

        assert_eq!(initial.segments.len(), 2);
        assert_eq!(initial.segments[0].words, vec![0x0201, 0x03]);
        assert_eq!(initial.segments[1].words, vec![0x2a, 0x07]);
    }

    #[test]
    fn computes_ram_domain_bounds_from_layout() {
        let device = JoltDevice::new(&MemoryConfig {
            program_size: Some(1024),
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            max_input_size: 16,
            max_output_size: 16,
            stack_size: 8,
            heap_size: 8,
        });
        assert_eq!(
            compute_min_ram_k(0x8000_0000, 7, &device.memory_layout),
            Ok(16)
        );
        assert_eq!(compute_max_ram_k(&device.memory_layout), Ok(256));
    }

    #[test]
    fn compute_max_ram_k_checks_the_padded_address_extent() {
        let mut layout = bounded_device().memory_layout;
        let lowest_address = layout.get_lowest_address();
        layout.heap_end = lowest_address + (1u64 << 63);
        assert_eq!(compute_max_ram_k(&layout), Ok(1usize << 60));
        layout.heap_end += 8;
        assert_eq!(
            compute_max_ram_k(&layout),
            Err(RamDomainError::AddressSpaceOverflow {
                lowest_address,
                max_ram_k: 1usize << 61
            })
        );

        layout.trusted_advice_start = 1u64 << 63;
        layout.untrusted_advice_start = 1u64 << 63;
        layout.heap_end = u64::MAX - 7;
        assert_eq!(compute_max_ram_k(&layout), Ok(1usize << 60));
        layout.trusted_advice_start += 8;
        layout.untrusted_advice_start += 8;
        assert_eq!(
            compute_max_ram_k(&layout),
            Err(RamDomainError::AddressSpaceOverflow {
                lowest_address: (1u64 << 63) + 8,
                max_ram_k: 1usize << 60
            })
        );
    }

    #[test]
    fn compute_max_ram_k_rechecks_deserialized_heap_headroom() {
        let mut layout = bounded_device().memory_layout;
        layout.heap_end = u64::MAX - 6;
        assert_eq!(
            compute_max_ram_k(&layout),
            Err(RamDomainError::MemoryLayout(
                MemoryLayoutError::HeapEndTooHigh {
                    heap_end: u64::MAX - 6
                }
            ))
        );
    }

    #[test]
    fn compute_max_ram_k_rejects_heap_end_below_lowest_address() {
        let mut layout = bounded_device().memory_layout;
        let lowest_address = layout.get_lowest_address();
        layout.heap_end = lowest_address - 8;
        assert_eq!(
            compute_max_ram_k(&layout),
            Err(RamDomainError::HeapBelowLowest {
                heap_end: lowest_address - 8,
                lowest_address,
            })
        );
    }

    #[test]
    fn compute_max_ram_k_rejects_an_unaligned_heap_end() {
        let mut layout = bounded_device().memory_layout;
        let lowest_address = layout.get_lowest_address();
        layout.heap_end = lowest_address + 8 * 1024;
        assert_eq!(compute_max_ram_k(&layout), Ok(1024));
        layout.heap_end = lowest_address + 8 * 1024 + 4;
        assert_eq!(
            compute_max_ram_k(&layout),
            Err(RamDomainError::MemoryLayout(
                MemoryLayoutError::HeapEndUnaligned {
                    heap_end: lowest_address + 8 * 1024 + 4
                }
            ))
        );
    }

    #[test]
    fn compute_max_ram_k_accepts_heap_end_equal_to_lowest_address() {
        let mut layout = bounded_device().memory_layout;
        layout.heap_end = layout.get_lowest_address();
        assert_eq!(compute_max_ram_k(&layout), Ok(1));
    }

    #[test]
    fn compute_min_ram_k_overflow_guards_return_domain_too_large() {
        let device = bounded_device();
        let layout = &device.memory_layout;
        let lowest_address = layout.get_lowest_address();
        assert_eq!(
            compute_min_ram_k(lowest_address + 8, usize::MAX, layout),
            Err(RamDomainError::DomainTooLarge)
        );
        assert_eq!(
            compute_min_ram_k(0, (usize::MAX >> 1) + 2, layout),
            Err(RamDomainError::DomainTooLarge)
        );
    }

    #[test]
    fn compute_min_ram_k_maps_zero_bytecode_address_to_word_zero() {
        let device = bounded_device();
        assert_eq!(compute_min_ram_k(0, 9, &device.memory_layout), Ok(16));
        assert_eq!(compute_min_ram_k(0, 0, &device.memory_layout), Ok(8));
    }

    #[test]
    fn compute_min_ram_k_propagates_below_lowest_address_error() {
        let device = bounded_device();
        let lowest_address = device.memory_layout.get_lowest_address();
        assert_eq!(
            compute_min_ram_k(lowest_address - 8, 1, &device.memory_layout),
            Err(RamDomainError::MemoryLayout(
                MemoryLayoutError::AddressBelowLowest {
                    address: lowest_address - 8,
                    lowest_address,
                }
            ))
        );
    }
}
