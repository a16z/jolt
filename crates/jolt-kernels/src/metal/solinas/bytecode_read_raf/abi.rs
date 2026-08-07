use std::mem::{align_of, size_of};

use thiserror::Error;

pub const BYTECODE_ADDRESS_STAGES: usize = 9;
pub const BYTECODE_ADDRESS_BASE_STAGES: usize = 5;
pub const BYTECODE_ADDRESS_VALUE_TABLES: usize = 6;
pub const BYTECODE_ADDRESS_DOMAIN: usize = 1 << 13;
pub const BYTECODE_ADDRESS_ROUNDS: usize = 13;
pub const BYTECODE_ADDRESS_INNER_LOG2: usize = 15;
pub const BYTECODE_ADDRESS_INNER_LENGTH: usize = 1 << BYTECODE_ADDRESS_INNER_LOG2;
pub const BYTECODE_ADDRESS_SIMD_WIDTH: usize = 32;
pub const BYTECODE_ADDRESS_CSR_THREADS: usize = 1024;
pub const BYTECODE_ADDRESS_BINS_PER_THREAD: usize =
    BYTECODE_ADDRESS_DOMAIN / BYTECODE_ADDRESS_CSR_THREADS;
pub const BYTECODE_ADDRESS_SHORT_THRESHOLD: usize = 128;
pub const BYTECODE_ADDRESS_DEFAULT_TRACE_CUTOFF: usize = 1 << 20;
pub const BYTECODE_ADDRESS_ACCUMULATOR_WORDS: usize = 5;
pub const BYTECODE_ADDRESS_RUN_HISTOGRAM_BUCKETS: usize = 16;
pub const BYTECODE_ADDRESS_AKITA_OFFSET: u32 = 0xffff_a7f7;

pub const CSR_PIPELINE: &str = "solinas_bytecode_address_build_csr";
pub const WRITE_DISPATCH_PIPELINE: &str = "solinas_bytecode_address_write_dispatch";
pub const SHORT_U64_PIPELINE: &str = "solinas_bytecode_address_short_runs_u64";
pub const LONG_U64_PIPELINE: &str = "solinas_bytecode_address_long_runs_u64";
pub const SHORT_FULL_PIPELINE: &str = "solinas_bytecode_address_short_runs_full";
pub const LONG_FULL_PIPELINE: &str = "solinas_bytecode_address_long_runs_full";
pub const FINALIZE_PIPELINE: &str = "solinas_bytecode_address_finalize";

pub(super) const PACKED_PC_MASK: u64 = (1 << 56) - 1;
#[cfg(test)]
pub(super) const PACKED_TABLE_SHIFT: u32 = 56;
#[cfg(test)]
pub(super) const PACKED_TABLE_MASK: u64 = (1 << 6) - 1;
#[cfg(test)]
pub(super) const PACKED_RAF_SHIFT: u32 = 62;
pub(super) const PACKED_INC_SIGN_SHIFT: u32 = 63;
const FIELD_BYTES: usize = 16;

/// Metal-visible twin of the shared 40-byte Booleanity row.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BytecodeReadRafRowWords {
    pub lookup_lo: u64,
    pub lookup_hi: u64,
    pub ram_address_plus_one: u64,
    pub fused_inc_magnitude: u64,
    pub packed_pc_and_flags: u64,
}

const _: [(); 40] = [(); size_of::<BytecodeReadRafRowWords>()];
const _: [(); 8] = [(); align_of::<BytecodeReadRafRowWords>()];

impl BytecodeReadRafRowWords {
    pub fn new(
        lookup_index: u128,
        mapped_pc: Option<u64>,
        remapped_ram_address: Option<u64>,
        fused_inc: i128,
    ) -> Result<Self, BytecodeReadRafError> {
        let pc_plus_one = mapped_pc
            .map(|pc| pc.checked_add(1).ok_or(BytecodeReadRafError::InvalidRow))
            .transpose()?
            .unwrap_or(0);
        let ram_address_plus_one = remapped_ram_address
            .map(|address| {
                address
                    .checked_add(1)
                    .ok_or(BytecodeReadRafError::InvalidRow)
            })
            .transpose()?
            .unwrap_or(0);
        let magnitude = fused_inc.unsigned_abs();
        if pc_plus_one > PACKED_PC_MASK || magnitude > u64::MAX as u128 {
            return Err(BytecodeReadRafError::InvalidRow);
        }
        Ok(Self {
            lookup_lo: lookup_index as u64,
            lookup_hi: (lookup_index >> 64) as u64,
            ram_address_plus_one,
            fused_inc_magnitude: magnitude as u64,
            packed_pc_and_flags: pc_plus_one | (u64::from(fused_inc < 0) << PACKED_INC_SIGN_SHIFT),
        })
    }

    pub const fn from_words(words: [u64; 5]) -> Self {
        Self {
            lookup_lo: words[0],
            lookup_hi: words[1],
            ram_address_plus_one: words[2],
            fused_inc_magnitude: words[3],
            packed_pc_and_flags: words[4],
        }
    }

    pub const fn words(self) -> [u64; 5] {
        [
            self.lookup_lo,
            self.lookup_hi,
            self.ram_address_plus_one,
            self.fused_inc_magnitude,
            self.packed_pc_and_flags,
        ]
    }

    pub const fn mapped_pc(self) -> Option<u64> {
        let plus_one = self.packed_pc_and_flags & PACKED_PC_MASK;
        if plus_one == 0 {
            None
        } else {
            Some(plus_one - 1)
        }
    }

    pub const fn push_pc(self) -> u64 {
        match self.mapped_pc() {
            Some(pc) => pc,
            None => 0,
        }
    }

    pub const fn fused_inc(self) -> i128 {
        let magnitude = self.fused_inc_magnitude as i128;
        if self.packed_pc_and_flags >> PACKED_INC_SIGN_SHIFT != 0 {
            -magnitude
        } else {
            magnitude
        }
    }
}

/// One nonempty `(outer block, bytecode address)` occurrence range.
#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BytecodeReadRafRun {
    pub(super) start: u32,
    pub(super) count: u32,
    pub(super) outer: u32,
    pub(super) address: u32,
}

const _: [(); 16] = [(); size_of::<BytecodeReadRafRun>()];
const _: [(); 4] = [(); align_of::<BytecodeReadRafRun>()];

impl BytecodeReadRafRun {
    pub(super) fn new(
        start: usize,
        count: usize,
        outer: usize,
        address: usize,
    ) -> Result<Self, BytecodeReadRafError> {
        Ok(Self {
            start: shader_count("run start", start)?,
            count: shader_count("run count", count)?,
            outer: shader_count("run outer", outer)?,
            address: shader_count("run address", address)?,
        })
    }

    pub const fn start(self) -> u32 {
        self.start
    }

    pub const fn count(self) -> u32 {
        self.count
    }

    pub const fn outer(self) -> u32 {
        self.outer
    }

    pub const fn address(self) -> u32 {
        self.address
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BytecodeReadRafStatus {
    pub short_runs: u32,
    pub long_runs: u32,
    pub invalid_rows: u32,
    pub completed_groups: u32,
    pub occurrence_rows: u32,
    pub reserved: [u32; 3],
}

const _: [(); 32] = [(); size_of::<BytecodeReadRafStatus>()];
const _: [(); 16] = [(); align_of::<BytecodeReadRafStatus>()];

impl BytecodeReadRafStatus {
    pub fn validate(
        self,
        shape: BytecodeReadRafShape,
    ) -> Result<BytecodeReadRafRunCounts, BytecodeReadRafError> {
        if self.reserved != [0; 3] {
            return Err(BytecodeReadRafError::NonzeroReservedTelemetry);
        }
        if self.invalid_rows != 0 {
            return Err(BytecodeReadRafError::InvalidStatusRows(self.invalid_rows));
        }
        if self.completed_groups as usize != shape.outer_length {
            return Err(BytecodeReadRafError::IncompleteStatusGroups {
                expected: shape.outer_length,
                got: self.completed_groups as usize,
            });
        }
        if self.occurrence_rows as usize != shape.rows {
            return Err(BytecodeReadRafError::InvalidStatusOccurrences {
                expected: shape.rows,
                got: self.occurrence_rows as usize,
            });
        }
        let total = self
            .short_runs
            .checked_add(self.long_runs)
            .ok_or(BytecodeReadRafError::ArithmeticOverflow("status run count"))?;
        if (total as usize) < shape.outer_length || total as usize > shape.run_capacity {
            return Err(BytecodeReadRafError::InvalidRunCount {
                minimum: shape.outer_length,
                maximum: shape.run_capacity,
                got: total as usize,
            });
        }
        Ok(BytecodeReadRafRunCounts {
            short_runs: self.short_runs,
            long_runs: self.long_runs,
        })
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BytecodeReadRafDiagnostics {
    pub short_occurrences: u32,
    pub long_occurrences: u32,
    pub maximum_run: u32,
    pub reserved: u32,
    pub run_histogram: [u32; BYTECODE_ADDRESS_RUN_HISTOGRAM_BUCKETS],
}

const _: [(); 80] = [(); size_of::<BytecodeReadRafDiagnostics>()];
const _: [(); 16] = [(); align_of::<BytecodeReadRafDiagnostics>()];

impl BytecodeReadRafDiagnostics {
    pub fn validate(
        self,
        shape: BytecodeReadRafShape,
        status: BytecodeReadRafStatus,
        short_threshold: usize,
    ) -> Result<(), BytecodeReadRafError> {
        if short_threshold == 0 || short_threshold > shape.inner_length {
            return Err(BytecodeReadRafError::InvalidShortThreshold(short_threshold));
        }
        if self.reserved != 0 {
            return Err(BytecodeReadRafError::NonzeroReservedTelemetry);
        }
        let counts = status.validate(shape)?;
        let occurrences = self
            .short_occurrences
            .checked_add(self.long_occurrences)
            .ok_or(BytecodeReadRafError::ArithmeticOverflow(
                "diagnostic occurrence count",
            ))?;
        if occurrences as usize != shape.rows {
            return Err(BytecodeReadRafError::InvalidDiagnosticOccurrences {
                expected: shape.rows,
                got: occurrences as usize,
            });
        }
        let histogram_runs = self
            .run_histogram
            .into_iter()
            .try_fold(0u32, |sum, count| {
                sum.checked_add(count)
                    .ok_or(BytecodeReadRafError::ArithmeticOverflow(
                        "diagnostic histogram count",
                    ))
            })?;
        if histogram_runs != counts.total()? {
            return Err(BytecodeReadRafError::InvalidDiagnosticRunCount {
                expected: counts.total()? as usize,
                got: histogram_runs as usize,
            });
        }
        if self.maximum_run == 0 || self.maximum_run as usize > shape.inner_length {
            return Err(BytecodeReadRafError::InvalidDiagnosticMaximumRun(
                self.maximum_run as usize,
            ));
        }
        let maximum_bucket = self
            .run_histogram
            .iter()
            .rposition(|&count| count != 0)
            .ok_or(BytecodeReadRafError::InvalidDiagnosticRunCount {
                expected: counts.total()? as usize,
                got: 0,
            })?;
        if maximum_bucket != bytecode_address_run_histogram_bucket(self.maximum_run as usize)? {
            return Err(BytecodeReadRafError::InvalidDiagnosticMaximumRun(
                self.maximum_run as usize,
            ));
        }
        let minimum_long_length =
            short_threshold
                .checked_add(1)
                .ok_or(BytecodeReadRafError::ArithmeticOverflow(
                    "minimum long-run length",
                ))?;
        let minimum_long_occurrences = (counts.long_runs as usize)
            .checked_mul(minimum_long_length)
            .ok_or(BytecodeReadRafError::ArithmeticOverflow(
                "minimum long occurrences",
            ))?;
        let maximum_long_occurrences = (counts.long_runs as usize)
            .checked_mul(shape.inner_length)
            .ok_or(BytecodeReadRafError::ArithmeticOverflow(
                "maximum long occurrences",
            ))?;
        let minimum_short_occurrences = counts.short_runs as usize;
        let maximum_short_occurrences = (counts.short_runs as usize)
            .checked_mul(short_threshold)
            .ok_or(BytecodeReadRafError::ArithmeticOverflow(
            "maximum short occurrences",
        ))?;
        if (self.long_occurrences as usize) < minimum_long_occurrences
            || (self.long_occurrences as usize) > maximum_long_occurrences
            || (self.short_occurrences as usize) < minimum_short_occurrences
            || (self.short_occurrences as usize) > maximum_short_occurrences
            || (counts.long_runs == 0 && self.maximum_run as usize > short_threshold)
            || (counts.long_runs != 0 && self.maximum_run as usize <= short_threshold)
        {
            return Err(BytecodeReadRafError::InvalidDiagnosticPartition);
        }
        Ok(())
    }
}

pub fn bytecode_address_run_histogram_bucket(count: usize) -> Result<usize, BytecodeReadRafError> {
    if count == 0 || count > BYTECODE_ADDRESS_INNER_LENGTH {
        return Err(BytecodeReadRafError::InvalidDiagnosticMaximumRun(count));
    }
    Ok((usize::BITS - 1 - count.leading_zeros()) as usize)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafRunCounts {
    pub short_runs: u32,
    pub long_runs: u32,
}

impl BytecodeReadRafRunCounts {
    pub fn total(self) -> Result<u32, BytecodeReadRafError> {
        self.short_runs
            .checked_add(self.long_runs)
            .ok_or(BytecodeReadRafError::ArithmeticOverflow("run count"))
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BytecodeReadRafIndirectGrid {
    pub threadgroups_x: u32,
    pub threadgroups_y: u32,
    pub threadgroups_z: u32,
    pub reserved: u32,
}

const _: [(); 16] = [(); size_of::<BytecodeReadRafIndirectGrid>()];
const _: [(); 16] = [(); align_of::<BytecodeReadRafIndirectGrid>()];

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BytecodeReadRafDispatchArgs {
    pub short_runs: BytecodeReadRafIndirectGrid,
    pub long_runs: BytecodeReadRafIndirectGrid,
}

const _: [(); 32] = [(); size_of::<BytecodeReadRafDispatchArgs>()];
const _: [(); 16] = [(); align_of::<BytecodeReadRafDispatchArgs>()];

#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafCsrParams {
    pub rows: u32,
    pub addresses: u32,
    pub inner_length: u32,
    pub outer_length: u32,
    pub run_capacity: u32,
    pub short_threshold: u32,
    pub bins_per_thread: u32,
    pub reserved: u32,
}

const _: [(); 32] = [(); size_of::<BytecodeReadRafCsrParams>()];
const _: [(); 4] = [(); align_of::<BytecodeReadRafCsrParams>()];

#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafPushforwardParams {
    pub rows: u32,
    pub addresses: u32,
    pub inner_length: u32,
    pub outer_length: u32,
    pub run_capacity: u32,
    pub short_threshold: u32,
    pub short_threads: u32,
    pub long_threads: u32,
    pub stages: u32,
    pub base_stages: u32,
    pub accumulator_words: u32,
    pub reserved: u32,
}

const _: [(); 48] = [(); size_of::<BytecodeReadRafPushforwardParams>()];
const _: [(); 4] = [(); align_of::<BytecodeReadRafPushforwardParams>()];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafConfig {
    pub inner_log2: usize,
    pub short_threshold: usize,
    pub csr_threads: usize,
    pub short_threads: usize,
    pub long_threads: usize,
    pub trace_cutoff: usize,
}

impl Default for BytecodeReadRafConfig {
    fn default() -> Self {
        Self {
            inner_log2: BYTECODE_ADDRESS_INNER_LOG2,
            short_threshold: BYTECODE_ADDRESS_SHORT_THRESHOLD,
            csr_threads: BYTECODE_ADDRESS_CSR_THREADS,
            short_threads: 256,
            long_threads: 256,
            trace_cutoff: BYTECODE_ADDRESS_DEFAULT_TRACE_CUTOFF,
        }
    }
}

impl BytecodeReadRafConfig {
    pub fn validate(
        self,
        rows: usize,
        addresses: usize,
    ) -> Result<BytecodeReadRafShape, BytecodeReadRafError> {
        if self.inner_log2 != BYTECODE_ADDRESS_INNER_LOG2 {
            return Err(BytecodeReadRafError::UnsupportedInnerLog2 {
                got: self.inner_log2,
            });
        }
        if self.csr_threads != BYTECODE_ADDRESS_CSR_THREADS {
            return Err(BytecodeReadRafError::UnsupportedCsrThreads {
                got: self.csr_threads,
            });
        }
        if self.short_threshold == 0 || self.short_threshold > BYTECODE_ADDRESS_INNER_LENGTH {
            return Err(BytecodeReadRafError::InvalidShortThreshold(
                self.short_threshold,
            ));
        }
        validate_threads("short runs", self.short_threads)?;
        validate_threads("long runs", self.long_threads)?;
        if self.trace_cutoff < BYTECODE_ADDRESS_INNER_LENGTH || !self.trace_cutoff.is_power_of_two()
        {
            return Err(BytecodeReadRafError::InvalidTraceCutoff(self.trace_cutoff));
        }
        BytecodeReadRafShape::new(rows, addresses)
    }

    pub fn csr_params(
        self,
        shape: BytecodeReadRafShape,
    ) -> Result<BytecodeReadRafCsrParams, BytecodeReadRafError> {
        let _ = self.validate(shape.rows, shape.addresses)?;
        Ok(BytecodeReadRafCsrParams {
            rows: shader_count("rows", shape.rows)?,
            addresses: shader_count("addresses", shape.addresses)?,
            inner_length: shader_count("inner length", shape.inner_length)?,
            outer_length: shader_count("outer length", shape.outer_length)?,
            run_capacity: shader_count("run capacity", shape.run_capacity)?,
            short_threshold: shader_count("short threshold", self.short_threshold)?,
            bins_per_thread: shader_count("bins per thread", BYTECODE_ADDRESS_BINS_PER_THREAD)?,
            reserved: 0,
        })
    }

    pub fn pushforward_params(
        self,
        shape: BytecodeReadRafShape,
    ) -> Result<BytecodeReadRafPushforwardParams, BytecodeReadRafError> {
        let _ = self.validate(shape.rows, shape.addresses)?;
        Ok(BytecodeReadRafPushforwardParams {
            rows: shader_count("rows", shape.rows)?,
            addresses: shader_count("addresses", shape.addresses)?,
            inner_length: shader_count("inner length", shape.inner_length)?,
            outer_length: shader_count("outer length", shape.outer_length)?,
            run_capacity: shader_count("run capacity", shape.run_capacity)?,
            short_threshold: shader_count("short threshold", self.short_threshold)?,
            short_threads: shader_count("short threads", self.short_threads)?,
            long_threads: shader_count("long threads", self.long_threads)?,
            stages: BYTECODE_ADDRESS_STAGES as u32,
            base_stages: BYTECODE_ADDRESS_BASE_STAGES as u32,
            accumulator_words: BYTECODE_ADDRESS_ACCUMULATOR_WORDS as u32,
            reserved: 0,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafShape {
    pub(super) rows: usize,
    pub(super) addresses: usize,
    pub(super) inner_length: usize,
    pub(super) outer_length: usize,
    pub(super) run_capacity: usize,
}

impl BytecodeReadRafShape {
    pub fn new(rows: usize, addresses: usize) -> Result<Self, BytecodeReadRafError> {
        if rows < BYTECODE_ADDRESS_INNER_LENGTH
            || !rows.is_power_of_two()
            || !rows.is_multiple_of(BYTECODE_ADDRESS_INNER_LENGTH)
        {
            return Err(BytecodeReadRafError::InvalidRows(rows));
        }
        if addresses != BYTECODE_ADDRESS_DOMAIN {
            return Err(BytecodeReadRafError::UnsupportedAddressDomain {
                expected: BYTECODE_ADDRESS_DOMAIN,
                got: addresses,
            });
        }
        let outer_length = rows / BYTECODE_ADDRESS_INNER_LENGTH;
        let slots = checked_product("run slots", outer_length, addresses)?;
        let run_capacity = rows.min(slots);
        let _ = shader_count("rows", rows)?;
        let _ = shader_count("run capacity", run_capacity)?;
        Ok(Self {
            rows,
            addresses,
            inner_length: BYTECODE_ADDRESS_INNER_LENGTH,
            outer_length,
            run_capacity,
        })
    }

    pub const fn threadgroup_bytes(self) -> usize {
        self.addresses * size_of::<u32>()
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn addresses(self) -> usize {
        self.addresses
    }

    pub const fn inner_length(self) -> usize {
        self.inner_length
    }

    pub const fn outer_length(self) -> usize {
        self.outer_length
    }

    pub const fn run_capacity(self) -> usize {
        self.run_capacity
    }

    pub fn storage_plan(self) -> Result<BytecodeReadRafStoragePlan, BytecodeReadRafError> {
        let occurrence_bytes = checked_bytes("occurrences", self.rows, size_of::<u32>())?;
        let run_bytes = checked_bytes(
            "run arena",
            self.run_capacity,
            size_of::<BytecodeReadRafRun>(),
        )?;
        let e_lo_bytes = checked_bytes(
            "E_lo tables",
            checked_product("E_lo elements", BYTECODE_ADDRESS_STAGES, self.inner_length)?,
            FIELD_BYTES,
        )?;
        let e_hi_bytes = checked_bytes(
            "E_hi tables",
            checked_product("E_hi elements", BYTECODE_ADDRESS_STAGES, self.outer_length)?,
            FIELD_BYTES,
        )?;
        let output_fields =
            checked_product("output fields", BYTECODE_ADDRESS_STAGES, self.addresses)?;
        let deferred_output_bytes = checked_bytes(
            "deferred output",
            checked_product(
                "deferred output words",
                output_fields,
                BYTECODE_ADDRESS_ACCUMULATOR_WORDS,
            )?,
            size_of::<u32>(),
        )?;
        let output_bytes = checked_bytes("canonical output", output_fields, FIELD_BYTES)?;
        let status_bytes = size_of::<BytecodeReadRafStatus>();
        let diagnostics_bytes = size_of::<BytecodeReadRafDiagnostics>();
        let dispatch_bytes = size_of::<BytecodeReadRafDispatchArgs>();
        let owned_bytes = [
            occurrence_bytes,
            run_bytes,
            e_lo_bytes,
            e_hi_bytes,
            deferred_output_bytes,
            output_bytes,
            status_bytes,
            diagnostics_bytes,
            dispatch_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |sum, bytes| {
            sum.checked_add(bytes)
                .ok_or(BytecodeReadRafError::SizeOverflow("owned storage"))
        })?;
        Ok(BytecodeReadRafStoragePlan {
            occurrence_bytes,
            run_bytes,
            e_lo_bytes,
            e_hi_bytes,
            deferred_output_bytes,
            output_bytes,
            status_bytes,
            diagnostics_bytes,
            dispatch_bytes,
            owned_bytes,
            shared_row_bytes: checked_bytes(
                "shared rows",
                self.rows,
                size_of::<BytecodeReadRafRowWords>(),
            )?,
            threadgroup_bytes: self.threadgroup_bytes(),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafStoragePlan {
    pub occurrence_bytes: usize,
    pub run_bytes: usize,
    pub e_lo_bytes: usize,
    pub e_hi_bytes: usize,
    pub deferred_output_bytes: usize,
    pub output_bytes: usize,
    pub status_bytes: usize,
    pub diagnostics_bytes: usize,
    pub dispatch_bytes: usize,
    pub owned_bytes: usize,
    pub shared_row_bytes: usize,
    pub threadgroup_bytes: usize,
}

impl BytecodeReadRafStoragePlan {
    pub fn total_with_shared_rows(self) -> Result<usize, BytecodeReadRafError> {
        self.owned_bytes
            .checked_add(self.shared_row_bytes)
            .ok_or(BytecodeReadRafError::SizeOverflow("total storage"))
    }

    pub const fn maximum_owned_buffer_bytes(self) -> usize {
        let mut maximum = self.occurrence_bytes;
        if self.run_bytes > maximum {
            maximum = self.run_bytes;
        }
        if self.e_lo_bytes > maximum {
            maximum = self.e_lo_bytes;
        }
        if self.e_hi_bytes > maximum {
            maximum = self.e_hi_bytes;
        }
        if self.deferred_output_bytes > maximum {
            maximum = self.deferred_output_bytes;
        }
        if self.output_bytes > maximum {
            maximum = self.output_bytes;
        }
        if self.status_bytes > maximum {
            maximum = self.status_bytes;
        }
        if self.diagnostics_bytes > maximum {
            maximum = self.diagnostics_bytes;
        }
        if self.dispatch_bytes > maximum {
            maximum = self.dispatch_bytes;
        }
        maximum
    }
}

pub(super) fn shader_count(name: &'static str, value: usize) -> Result<u32, BytecodeReadRafError> {
    u32::try_from(value).map_err(|_| BytecodeReadRafError::ShaderCountOverflow { name, value })
}

fn validate_threads(phase: &'static str, threads: usize) -> Result<(), BytecodeReadRafError> {
    if threads == 0
        || threads > BYTECODE_ADDRESS_CSR_THREADS
        || !threads.is_multiple_of(BYTECODE_ADDRESS_SIMD_WIDTH)
    {
        Err(BytecodeReadRafError::InvalidThreadgroupWidth { phase, threads })
    } else {
        Ok(())
    }
}

fn checked_product(
    name: &'static str,
    left: usize,
    right: usize,
) -> Result<usize, BytecodeReadRafError> {
    left.checked_mul(right)
        .ok_or(BytecodeReadRafError::SizeOverflow(name))
}

fn checked_bytes(
    name: &'static str,
    elements: usize,
    element_bytes: usize,
) -> Result<usize, BytecodeReadRafError> {
    checked_product(name, elements, element_bytes)
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum BytecodeReadRafError {
    #[error("bytecode address rows must be a power of two and at least 2^15; got {0}")]
    InvalidRows(usize),
    #[error("bytecode address specialization requires {expected} addresses; got {got}")]
    UnsupportedAddressDomain { expected: usize, got: usize },
    #[error("bytecode address specialization requires inner_log2=15; got {got}")]
    UnsupportedInnerLog2 { got: usize },
    #[error("bytecode CSR specialization requires 1024 threads; got {got}")]
    UnsupportedCsrThreads { got: usize },
    #[error("invalid short-run threshold {0}")]
    InvalidShortThreshold(usize),
    #[error("invalid trace cutoff {0}")]
    InvalidTraceCutoff(usize),
    #[error("invalid {phase} threadgroup width {threads}")]
    InvalidThreadgroupWidth { phase: &'static str, threads: usize },
    #[error("invalid packed bytecode row")]
    InvalidRow,
    #[error("non-canonical Akita coefficient {0}")]
    NonCanonicalCoefficient(u128),
    #[error("row count mismatch: expected {expected}, got {got}")]
    RowCount { expected: usize, got: usize },
    #[error("row {row} mapped PC {pc} outside address domain {addresses}")]
    MappedPcOutsideDomain {
        row: usize,
        pc: usize,
        addresses: usize,
    },
    #[error("invalid bytecode stage count")]
    InvalidStageCount,
    #[error("stage {stage} point has length {got}; expected {expected}")]
    InvalidPointLength {
        stage: usize,
        expected: usize,
        got: usize,
    },
    #[error("address challenge count mismatch: expected {expected}, got {got}")]
    InvalidAddressChallengeCount { expected: usize, got: usize },
    #[error("invalid value-table count: expected {expected}, got {got}")]
    InvalidValueTableCount { expected: usize, got: usize },
    #[error("invalid address-table length {0}")]
    InvalidAddressTableLength(usize),
    #[error("{name} table {index} has length {got}; expected {expected}")]
    InvalidTableShape {
        name: &'static str,
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("stage value references missing table {0}")]
    InvalidStageValueSource(usize),
    #[error("bytecode CSR topology invariant failed")]
    TopologyInvariant,
    #[error("bytecode CSR reported {0} invalid rows or group invariants")]
    InvalidStatusRows(u32),
    #[error("bytecode CSR completed {got} groups; expected {expected}")]
    IncompleteStatusGroups { expected: usize, got: usize },
    #[error("bytecode CSR accounted for {got} rows; expected {expected}")]
    InvalidStatusOccurrences { expected: usize, got: usize },
    #[error("bytecode CSR diagnostics accounted for {got} rows; expected {expected}")]
    InvalidDiagnosticOccurrences { expected: usize, got: usize },
    #[error("bytecode CSR diagnostics reported {got} runs; expected {expected}")]
    InvalidDiagnosticRunCount { expected: usize, got: usize },
    #[error("bytecode CSR diagnostics reported invalid maximum run {0}")]
    InvalidDiagnosticMaximumRun(usize),
    #[error("bytecode CSR diagnostic short/long partition is inconsistent")]
    InvalidDiagnosticPartition,
    #[error("bytecode CSR telemetry has nonzero reserved words")]
    NonzeroReservedTelemetry,
    #[error("run count {got} outside [{minimum}, {maximum}]")]
    InvalidRunCount {
        minimum: usize,
        maximum: usize,
        got: usize,
    },
    #[error("long-run count {got} exceeds total run count {maximum}")]
    InvalidLongRunCount { maximum: u64, got: usize },
    #[error(
        "{runs} runs with {long_runs} long runs cannot cover {rows} rows at threshold {short_threshold}"
    )]
    InfeasibleRunPartition {
        rows: usize,
        runs: usize,
        long_runs: usize,
        short_threshold: usize,
    },
    #[error("invalid roof utilization {0}%")]
    InvalidUtilization(u64),
    #[error("zero {0} rate")]
    ZeroRate(&'static str),
    #[error("missing matched {0} rate")]
    MissingMatchedRate(&'static str),
    #[error("{name}={value} does not fit the shader ABI")]
    ShaderCountOverflow { name: &'static str, value: usize },
    #[error("size overflow while computing {0}")]
    SizeOverflow(&'static str),
    #[error("arithmetic overflow while computing {0}")]
    ArithmeticOverflow(&'static str),
}
