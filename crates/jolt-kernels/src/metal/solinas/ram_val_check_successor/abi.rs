//! Producer-owned row ABI and the two first-slice dispatch ABIs.

use core::mem::{align_of, size_of};

use jolt_field::Field;

pub const FIELD_BYTES: u128 = 16;
pub const ROW_BYTES: u128 = 16;
pub const MESSAGE_COLUMNS: usize = 3;
pub const SIMD_WIDTH: usize = 32;
pub const NO_RAM_ADDRESS: u32 = u32::MAX;
pub const STATUS_WORDS: usize = 1;
pub const STATUS_WORD_BYTES: u128 = 4;
pub const STATUS_UNSUPPORTED: u32 = 1 << 0;
pub const STATUS_INVALID_ROW: u32 = 1 << 1;

pub const FIRST_MESSAGE_PIPELINE: &str = "solinas_ram_val_check_successor_first_message";
pub const REDUCE_PIPELINE: &str = "solinas_ram_val_check_successor_reduce3";
pub const SPARSE_FIRST_MESSAGE_PIPELINE: &str = "solinas_ram_val_check_sparse_first_message";

const FLAG_INCREMENT_NONNEGATIVE: u32 = 1 << 0;
const FLAG_RAM_INCREMENT: u32 = 1 << 1;
const VALID_FLAGS: u32 = FLAG_INCREMENT_NONNEGATIVE | FLAG_RAM_INCREMENT;
const PAIR_LO_NEGATIVE: u32 = 1 << 0;
const PAIR_HI_NEGATIVE: u32 = 1 << 1;

/// The checked typed input to the common row producer.
///
/// `ram_increment` and `rd_increment` are both retained until the constructor
/// verifies the fused-column exclusivity invariant in release builds. A store
/// at raw address zero legitimately has `remapped_ram_address == None`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IncrementAccessSource {
    pub remapped_ram_address: Option<u64>,
    pub store: bool,
    pub ram_increment: i128,
    pub rd_increment: i128,
}

/// The producer-owned common base row for RAM value-check, increment-claim,
/// and Booleanity-family consumers.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IncrementAccessRow {
    increment_magnitude: u64,
    ram_address: u32,
    flags: u32,
}

/// One low-to-high cycle pair with at least one nonzero RAM increment.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamValActivePair {
    pair_index: u32,
    signs: u32,
    lo_magnitude: u64,
    hi_magnitude: u64,
}

const _: [(); 24] = [(); size_of::<RamValActivePair>()];
const _: [(); 8] = [(); align_of::<RamValActivePair>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamValSparseFirstMessageParams {
    pub active_pairs: u32,
    pub rows: u32,
    pub low_length: u32,
    pub address_domain: u32,
}

const _: [(); 16] = [(); size_of::<RamValSparseFirstMessageParams>()];
const _: [(); 4] = [(); align_of::<RamValSparseFirstMessageParams>()];

const _: [(); 16] = [(); size_of::<IncrementAccessRow>()];
const _: [(); 8] = [(); align_of::<IncrementAccessRow>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamValFirstMessageParams {
    pub rows: u32,
    pub high_blocks: u32,
    pub low_length: u32,
    pub address_domain: u32,
    pub threads: u32,
    pub no_address: u32,
    pub reserved: [u32; 2],
}

const _: [(); 32] = [(); size_of::<RamValFirstMessageParams>()];
const _: [(); 4] = [(); align_of::<RamValFirstMessageParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamValReductionParams {
    pub input_count: u32,
    pub output_count: u32,
    pub columns: u32,
    pub reserved: u32,
}

const _: [(); 16] = [(); size_of::<RamValReductionParams>()];
const _: [(); 4] = [(); align_of::<RamValReductionParams>()];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamValLaunch {
    pub threadgroups: u32,
    pub threads_per_threadgroup: u32,
    pub initial_status: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamValFirstMessageBufferLengths {
    pub rows: usize,
    pub eq_address: usize,
    pub lt_low: usize,
    pub lt_high: usize,
    pub eq_high: usize,
    pub partials: usize,
    pub status_words: usize,
}

/// A byte range within one host-tracked Metal allocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamValBufferRange {
    pub storage_id: u64,
    pub offset_bytes: u64,
    pub length_bytes: u64,
}

/// The three nonoverlapping ranges required by one reduction dispatch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamValReductionBuffers {
    pub input: RamValBufferRange,
    pub output: RamValBufferRange,
    pub status: RamValBufferRange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RamValSuccessorRowError {
    IncrementOutOfRange(i128),
    PairIndexOutOfRange(usize),
    RemappedAddressOutOfRange(u64),
    InvalidAddressDomain(u32),
    SentinelCollision,
    ReservedFlags(u32),
    NegativeZero,
    IncrementExclusivity {
        store: bool,
        ram_increment: i128,
        rd_increment: i128,
    },
    AddressOutOfDomain {
        address: u32,
        domain: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RamValSuccessorDispatchError {
    InvalidFirstMessageParams,
    InvalidReductionParams,
    WrongThreadgroups {
        expected: u32,
        got: u32,
    },
    WrongThreadsPerThreadgroup {
        expected: u32,
        got: u32,
    },
    StatusNotCleared(u32),
    BufferTooShort {
        name: &'static str,
        required: usize,
        got: usize,
    },
    MissingBufferIdentity {
        name: &'static str,
    },
    MisalignedBufferRange {
        name: &'static str,
        required_alignment: u64,
        offset_bytes: u64,
    },
    BufferRangeTooShort {
        name: &'static str,
        required_bytes: u64,
        got_bytes: u64,
    },
    BufferRangeOverflow {
        name: &'static str,
    },
    OverlappingBufferRanges {
        left: &'static str,
        right: &'static str,
    },
    ArithmeticOverflow,
}

impl IncrementAccessRow {
    /// Checks the fused-column invariant before discarding the unselected
    /// delta. These are ordinary release-mode checks, not debug assertions.
    pub fn from_source(
        source: IncrementAccessSource,
        address_domain: u32,
    ) -> Result<Self, RamValSuccessorRowError> {
        let selected = if source.store {
            if source.rd_increment != 0 {
                return Err(RamValSuccessorRowError::IncrementExclusivity {
                    store: source.store,
                    ram_increment: source.ram_increment,
                    rd_increment: source.rd_increment,
                });
            }
            source.ram_increment
        } else {
            if source.ram_increment != 0 {
                return Err(RamValSuccessorRowError::IncrementExclusivity {
                    store: source.store,
                    ram_increment: source.ram_increment,
                    rd_increment: source.rd_increment,
                });
            }
            source.rd_increment
        };
        if address_domain == 0
            || address_domain == NO_RAM_ADDRESS
            || !address_domain.is_power_of_two()
        {
            return Err(RamValSuccessorRowError::InvalidAddressDomain(
                address_domain,
            ));
        }
        let ram_address = match source.remapped_ram_address {
            None => NO_RAM_ADDRESS,
            Some(address) => {
                let address = u32::try_from(address)
                    .map_err(|_| RamValSuccessorRowError::RemappedAddressOutOfRange(address))?;
                if address == NO_RAM_ADDRESS {
                    return Err(RamValSuccessorRowError::SentinelCollision);
                }
                if address >= address_domain {
                    return Err(RamValSuccessorRowError::AddressOutOfDomain {
                        address,
                        domain: address_domain as usize,
                    });
                }
                address
            }
        };
        let magnitude = selected.unsigned_abs();
        if magnitude > u128::from(u64::MAX) {
            return Err(RamValSuccessorRowError::IncrementOutOfRange(selected));
        }
        let mut flags = 0;
        if selected >= 0 {
            flags |= FLAG_INCREMENT_NONNEGATIVE;
        }
        if source.store {
            flags |= FLAG_RAM_INCREMENT;
        }
        let row = Self {
            increment_magnitude: magnitude as u64,
            ram_address,
            flags,
        };
        row.validate()?;
        Ok(row)
    }

    #[cfg(test)]
    pub(crate) fn try_from_words(
        words: [u64; 2],
        address_domain: u32,
    ) -> Result<Self, RamValSuccessorRowError> {
        let row = Self {
            increment_magnitude: words[0],
            ram_address: words[1] as u32,
            flags: (words[1] >> 32) as u32,
        };
        row.validate_address_domain(address_domain)?;
        Ok(row)
    }

    #[cfg(test)]
    pub(crate) const fn words(self) -> [u64; 2] {
        [
            self.increment_magnitude,
            self.ram_address as u64 | ((self.flags as u64) << 32),
        ]
    }

    pub const fn magnitude(self) -> u64 {
        self.increment_magnitude
    }

    pub const fn ram_address(self) -> Option<u32> {
        if self.ram_address == NO_RAM_ADDRESS {
            None
        } else {
            Some(self.ram_address)
        }
    }

    pub const fn is_nonnegative(self) -> bool {
        self.flags & FLAG_INCREMENT_NONNEGATIVE != 0
    }

    pub const fn is_ram_increment(self) -> bool {
        self.flags & FLAG_RAM_INCREMENT != 0
    }

    pub const fn has_nonzero_ram_increment(self) -> bool {
        self.is_ram_increment() && self.increment_magnitude != 0
    }

    pub fn fused_value<F: Field>(self) -> F {
        let magnitude = F::from_u64(self.increment_magnitude);
        if self.is_nonnegative() {
            magnitude
        } else {
            F::zero() - magnitude
        }
    }

    pub fn ram_increment<F: Field>(self) -> F {
        if self.is_ram_increment() {
            self.fused_value()
        } else {
            F::zero()
        }
    }

    pub fn ram_ra<F: Field>(self, eq_address: &[F]) -> Result<F, RamValSuccessorRowError> {
        match self.ram_address() {
            None => Ok(F::zero()),
            Some(address) => eq_address.get(address as usize).copied().ok_or(
                RamValSuccessorRowError::AddressOutOfDomain {
                    address,
                    domain: eq_address.len(),
                },
            ),
        }
    }

    pub fn validate(self) -> Result<(), RamValSuccessorRowError> {
        let reserved = self.flags & !VALID_FLAGS;
        if reserved != 0 {
            return Err(RamValSuccessorRowError::ReservedFlags(reserved));
        }
        if self.increment_magnitude == 0 && !self.is_nonnegative() {
            return Err(RamValSuccessorRowError::NegativeZero);
        }
        Ok(())
    }

    pub fn validate_address_domain(
        self,
        address_domain: u32,
    ) -> Result<(), RamValSuccessorRowError> {
        if address_domain == 0
            || address_domain == NO_RAM_ADDRESS
            || !address_domain.is_power_of_two()
        {
            return Err(RamValSuccessorRowError::InvalidAddressDomain(
                address_domain,
            ));
        }
        self.validate()?;
        if let Some(address) = self.ram_address() {
            if address >= address_domain {
                return Err(RamValSuccessorRowError::AddressOutOfDomain {
                    address,
                    domain: address_domain as usize,
                });
            }
        }
        Ok(())
    }
}

impl RamValActivePair {
    pub fn new(
        pair_index: usize,
        lo_increment: i128,
        hi_increment: i128,
    ) -> Result<Self, RamValSuccessorRowError> {
        let pair_index = u32::try_from(pair_index)
            .map_err(|_| RamValSuccessorRowError::PairIndexOutOfRange(pair_index))?;
        let lo_magnitude = lo_increment.unsigned_abs();
        let hi_magnitude = hi_increment.unsigned_abs();
        if lo_magnitude > u128::from(u64::MAX) {
            return Err(RamValSuccessorRowError::IncrementOutOfRange(lo_increment));
        }
        if hi_magnitude > u128::from(u64::MAX) {
            return Err(RamValSuccessorRowError::IncrementOutOfRange(hi_increment));
        }
        let signs = (u32::from(lo_increment < 0) * PAIR_LO_NEGATIVE)
            | (u32::from(hi_increment < 0) * PAIR_HI_NEGATIVE);
        Ok(Self {
            pair_index,
            signs,
            lo_magnitude: lo_magnitude as u64,
            hi_magnitude: hi_magnitude as u64,
        })
    }

    pub const fn pair_index(self) -> usize {
        self.pair_index as usize
    }

    pub fn increments<F: Field>(self) -> [F; 2] {
        let mut lo = F::from_u64(self.lo_magnitude);
        let mut hi = F::from_u64(self.hi_magnitude);
        if self.signs & PAIR_LO_NEGATIVE != 0 {
            lo = -lo;
        }
        if self.signs & PAIR_HI_NEGATIVE != 0 {
            hi = -hi;
        }
        [lo, hi]
    }
}

impl RamValSparseFirstMessageParams {
    pub fn new(
        active_pairs: usize,
        rows: usize,
        low_length: usize,
        address_domain: usize,
    ) -> Result<Self, RamValSuccessorDispatchError> {
        if active_pairs == 0
            || rows < 2
            || !rows.is_power_of_two()
            || low_length < 2
            || !low_length.is_power_of_two()
            || !rows.is_multiple_of(low_length)
            || address_domain == 0
            || !address_domain.is_power_of_two()
            || address_domain >= NO_RAM_ADDRESS as usize
        {
            return Err(RamValSuccessorDispatchError::InvalidFirstMessageParams);
        }
        Ok(Self {
            active_pairs: u32::try_from(active_pairs)
                .map_err(|_| RamValSuccessorDispatchError::ArithmeticOverflow)?,
            rows: u32::try_from(rows)
                .map_err(|_| RamValSuccessorDispatchError::ArithmeticOverflow)?,
            low_length: u32::try_from(low_length)
                .map_err(|_| RamValSuccessorDispatchError::ArithmeticOverflow)?,
            address_domain: u32::try_from(address_domain)
                .map_err(|_| RamValSuccessorDispatchError::ArithmeticOverflow)?,
        })
    }
}

impl Default for IncrementAccessRow {
    fn default() -> Self {
        Self {
            increment_magnitude: 0,
            ram_address: NO_RAM_ADDRESS,
            flags: FLAG_INCREMENT_NONNEGATIVE,
        }
    }
}

impl RamValFirstMessageParams {
    pub fn validate_launch(
        self,
        launch: RamValLaunch,
        buffers: RamValFirstMessageBufferLengths,
    ) -> Result<(), RamValSuccessorDispatchError> {
        let shape_valid = self.rows >= 2
            && self.rows.is_power_of_two()
            && self.high_blocks != 0
            && self.high_blocks.is_power_of_two()
            && self.low_length >= 2
            && self.low_length.is_power_of_two()
            && self.address_domain != 0
            && self.address_domain.is_power_of_two()
            && self.no_address == NO_RAM_ADDRESS
            && self.threads == SIMD_WIDTH as u32
            && self.reserved == [0; 2]
            && u64::from(self.high_blocks) * u64::from(self.low_length) == u64::from(self.rows);
        if !shape_valid {
            return Err(RamValSuccessorDispatchError::InvalidFirstMessageParams);
        }
        validate_launch(launch, self.high_blocks, SIMD_WIDTH as u32)?;
        require_len("rows", buffers.rows, usize_from_u32(self.rows)?)?;
        require_len(
            "eq_address",
            buffers.eq_address,
            usize_from_u32(self.address_domain)?,
        )?;
        require_len("lt_low", buffers.lt_low, usize_from_u32(self.low_length)?)?;
        let high_blocks = usize_from_u32(self.high_blocks)?;
        require_len("lt_high", buffers.lt_high, high_blocks)?;
        require_len("eq_high", buffers.eq_high, high_blocks)?;
        let partials = high_blocks
            .checked_mul(MESSAGE_COLUMNS)
            .ok_or(RamValSuccessorDispatchError::ArithmeticOverflow)?;
        require_len("partials", buffers.partials, partials)?;
        require_len("status", buffers.status_words, STATUS_WORDS)?;
        Ok(())
    }
}

impl RamValReductionParams {
    pub fn validate_launch(
        self,
        launch: RamValLaunch,
        buffers: RamValReductionBuffers,
    ) -> Result<(), RamValSuccessorDispatchError> {
        let expected_output = u64::from(self.input_count).div_ceil(SIMD_WIDTH as u64);
        let params_valid = self.input_count != 0
            && u64::from(self.output_count) == expected_output
            && self.columns == MESSAGE_COLUMNS as u32
            && self.reserved == 0;
        if !params_valid {
            return Err(RamValSuccessorDispatchError::InvalidReductionParams);
        }
        validate_launch(launch, self.output_count, SIMD_WIDTH as u32)?;
        let input_elements = usize_from_u32(self.input_count)?
            .checked_mul(MESSAGE_COLUMNS)
            .ok_or(RamValSuccessorDispatchError::ArithmeticOverflow)?;
        let output_elements = usize_from_u32(self.output_count)?
            .checked_mul(MESSAGE_COLUMNS)
            .ok_or(RamValSuccessorDispatchError::ArithmeticOverflow)?;
        let input_bytes = field_bytes(input_elements)?;
        let output_bytes = field_bytes(output_elements)?;
        validate_buffer_range("input", buffers.input, input_bytes, FIELD_BYTES as u64)?;
        validate_buffer_range("output", buffers.output, output_bytes, FIELD_BYTES as u64)?;
        validate_buffer_range(
            "status",
            buffers.status,
            STATUS_WORD_BYTES as u64,
            STATUS_WORD_BYTES as u64,
        )?;
        reject_overlap("input", buffers.input, "output", buffers.output)?;
        reject_overlap("input", buffers.input, "status", buffers.status)?;
        reject_overlap("output", buffers.output, "status", buffers.status)?;
        Ok(())
    }
}

fn field_bytes(elements: usize) -> Result<u64, RamValSuccessorDispatchError> {
    u64::try_from(elements)
        .ok()
        .and_then(|value| value.checked_mul(FIELD_BYTES as u64))
        .ok_or(RamValSuccessorDispatchError::ArithmeticOverflow)
}

fn validate_buffer_range(
    name: &'static str,
    range: RamValBufferRange,
    required_bytes: u64,
    required_alignment: u64,
) -> Result<(), RamValSuccessorDispatchError> {
    if range.storage_id == 0 {
        return Err(RamValSuccessorDispatchError::MissingBufferIdentity { name });
    }
    if !range.offset_bytes.is_multiple_of(required_alignment) {
        return Err(RamValSuccessorDispatchError::MisalignedBufferRange {
            name,
            required_alignment,
            offset_bytes: range.offset_bytes,
        });
    }
    if range.length_bytes < required_bytes {
        return Err(RamValSuccessorDispatchError::BufferRangeTooShort {
            name,
            required_bytes,
            got_bytes: range.length_bytes,
        });
    }
    let _ = range
        .offset_bytes
        .checked_add(range.length_bytes)
        .ok_or(RamValSuccessorDispatchError::BufferRangeOverflow { name })?;
    Ok(())
}

fn reject_overlap(
    left_name: &'static str,
    left: RamValBufferRange,
    right_name: &'static str,
    right: RamValBufferRange,
) -> Result<(), RamValSuccessorDispatchError> {
    if left.storage_id != right.storage_id {
        return Ok(());
    }
    let left_end = left
        .offset_bytes
        .checked_add(left.length_bytes)
        .ok_or(RamValSuccessorDispatchError::BufferRangeOverflow { name: left_name })?;
    let right_end = right
        .offset_bytes
        .checked_add(right.length_bytes)
        .ok_or(RamValSuccessorDispatchError::BufferRangeOverflow { name: right_name })?;
    if left.offset_bytes < right_end && right.offset_bytes < left_end {
        return Err(RamValSuccessorDispatchError::OverlappingBufferRanges {
            left: left_name,
            right: right_name,
        });
    }
    Ok(())
}

fn validate_launch(
    launch: RamValLaunch,
    expected_threadgroups: u32,
    expected_threads: u32,
) -> Result<(), RamValSuccessorDispatchError> {
    if launch.initial_status != 0 {
        return Err(RamValSuccessorDispatchError::StatusNotCleared(
            launch.initial_status,
        ));
    }
    if launch.threadgroups != expected_threadgroups {
        return Err(RamValSuccessorDispatchError::WrongThreadgroups {
            expected: expected_threadgroups,
            got: launch.threadgroups,
        });
    }
    if launch.threads_per_threadgroup != expected_threads {
        return Err(RamValSuccessorDispatchError::WrongThreadsPerThreadgroup {
            expected: expected_threads,
            got: launch.threads_per_threadgroup,
        });
    }
    Ok(())
}

fn require_len(
    name: &'static str,
    got: usize,
    required: usize,
) -> Result<(), RamValSuccessorDispatchError> {
    if got < required {
        Err(RamValSuccessorDispatchError::BufferTooShort {
            name,
            required,
            got,
        })
    } else {
        Ok(())
    }
}

fn usize_from_u32(value: u32) -> Result<usize, RamValSuccessorDispatchError> {
    usize::try_from(value).map_err(|_| RamValSuccessorDispatchError::ArithmeticOverflow)
}
