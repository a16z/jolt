//! Checked host/shader ABI for the Hamming-weight successor.

use std::mem::{align_of, size_of};

use thiserror::Error;

pub const HAMMING_WEIGHT_LOOKUP_SELECTORS: usize = 16;
pub const HAMMING_WEIGHT_BYTECODE_SELECTORS: usize = 2;
pub const HAMMING_WEIGHT_RAM_SELECTORS: usize = 2;
pub const HAMMING_WEIGHT_INC_CHUNK_SELECTORS: usize = 8;
pub const HAMMING_WEIGHT_INC_CARRY_SELECTORS: usize = 1;
pub const HAMMING_WEIGHT_SELECTORS: usize = HAMMING_WEIGHT_LOOKUP_SELECTORS
    + HAMMING_WEIGHT_BYTECODE_SELECTORS
    + HAMMING_WEIGHT_RAM_SELECTORS
    + HAMMING_WEIGHT_INC_CHUNK_SELECTORS
    + HAMMING_WEIGHT_INC_CARRY_SELECTORS;

pub const HAMMING_WEIGHT_BINS: usize = 256;
pub const HAMMING_WEIGHT_RETAINED_BINS: usize = HAMMING_WEIGHT_BINS - 1;
pub const HAMMING_WEIGHT_ADDRESS_ROUNDS: usize = 8;
pub const HAMMING_WEIGHT_SIMD_WIDTH: usize = 32;
pub const HAMMING_WEIGHT_THREADS: usize = HAMMING_WEIGHT_SELECTORS * HAMMING_WEIGHT_SIMD_WIDTH;
pub const HAMMING_WEIGHT_STAGE_ROWS: usize = 512;
pub const HAMMING_WEIGHT_ROW_BYTES: usize = 40;
pub const HAMMING_WEIGHT_FIELD_BYTES: usize = 16;
pub const HAMMING_WEIGHT_STAGE_HOT_BYTES: usize =
    HAMMING_WEIGHT_SELECTORS * HAMMING_WEIGHT_STAGE_ROWS;
pub const HAMMING_WEIGHT_STAGE_WEIGHT_BYTES: usize =
    HAMMING_WEIGHT_FIELD_BYTES * HAMMING_WEIGHT_STAGE_ROWS;
pub const HAMMING_WEIGHT_LOADER_SIMDGROUPS: usize =
    HAMMING_WEIGHT_STAGE_ROWS / HAMMING_WEIGHT_SIMD_WIDTH;
pub const HAMMING_WEIGHT_STAGE_AUDIT_BYTES: usize =
    3 * HAMMING_WEIGHT_LOADER_SIMDGROUPS * size_of::<u32>();
pub const HAMMING_WEIGHT_THREADGROUP_BYTES: usize = HAMMING_WEIGHT_STAGE_HOT_BYTES
    + HAMMING_WEIGHT_STAGE_WEIGHT_BYTES
    + HAMMING_WEIGHT_STAGE_AUDIT_BYTES;
pub const HAMMING_WEIGHT_AUDIT_ROW_BYTES: usize = 32;
pub const HAMMING_WEIGHT_STATUS_BYTES: usize = 16;

pub const HAMMING_WEIGHT_DEFAULT_INNER_LOG2: usize = 18;
pub const HAMMING_WEIGHT_MIN_OUTER_LOG2: usize = 6;
pub const HAMMING_WEIGHT_DEFAULT_TRACE_CUTOFF: usize = 1 << 18;
pub const HAMMING_WEIGHT_BALANCED_INC_BIAS: u64 = 0x8080_8080_8080_8080;

pub const HAMMING_WEIGHT_TARGET_LOG_T: usize = 26;
pub const HAMMING_WEIGHT_TARGET_ROWS: usize = 1 << HAMMING_WEIGHT_TARGET_LOG_T;
pub const HAMMING_WEIGHT_TARGET_CPU_NS: u64 = 548_702_500;
pub const HAMMING_WEIGHT_TARGET_FIVE_X_NS: u64 = 109_740_500;
pub const HAMMING_WEIGHT_TARGET_EIGHT_X_NS: u64 = 68_587_812;
pub const HAMMING_WEIGHT_TARGET_GPU_ACTIVE_NS: u64 = 40_000_000;

pub(crate) const HISTOGRAM_PIPELINE: &str = "solinas_hamming_weight_register_histogram";
pub(crate) const FINALIZE_PIPELINE: &str = "solinas_hamming_weight_register_finalize";

const _: () = assert!(HAMMING_WEIGHT_SELECTORS == 29);
const _: () = assert!(1 << HAMMING_WEIGHT_ADDRESS_ROUNDS == HAMMING_WEIGHT_BINS);
const _: () = assert!(HAMMING_WEIGHT_THREADS == 928);
const _: () = assert!(HAMMING_WEIGHT_THREADGROUP_BYTES == 23_232);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightProtocolTopology {
    pub lookup_selectors: usize,
    pub bytecode_selectors: usize,
    pub ram_selectors: usize,
    pub inc_chunk_selectors: usize,
    pub inc_carry_selectors: usize,
    pub bins: usize,
}

impl HammingWeightProtocolTopology {
    pub const PRODUCTION: Self = Self {
        lookup_selectors: HAMMING_WEIGHT_LOOKUP_SELECTORS,
        bytecode_selectors: HAMMING_WEIGHT_BYTECODE_SELECTORS,
        ram_selectors: HAMMING_WEIGHT_RAM_SELECTORS,
        inc_chunk_selectors: HAMMING_WEIGHT_INC_CHUNK_SELECTORS,
        inc_carry_selectors: HAMMING_WEIGHT_INC_CARRY_SELECTORS,
        bins: HAMMING_WEIGHT_BINS,
    };

    pub fn validate(self) -> Result<(), HammingWeightSuccessorError> {
        check_topology(
            "lookup selectors",
            HAMMING_WEIGHT_LOOKUP_SELECTORS,
            self.lookup_selectors,
        )?;
        check_topology(
            "bytecode selectors",
            HAMMING_WEIGHT_BYTECODE_SELECTORS,
            self.bytecode_selectors,
        )?;
        check_topology(
            "RAM selectors",
            HAMMING_WEIGHT_RAM_SELECTORS,
            self.ram_selectors,
        )?;
        check_topology(
            "increment chunk selectors",
            HAMMING_WEIGHT_INC_CHUNK_SELECTORS,
            self.inc_chunk_selectors,
        )?;
        check_topology(
            "increment carry selectors",
            HAMMING_WEIGHT_INC_CARRY_SELECTORS,
            self.inc_carry_selectors,
        )?;
        check_topology("bins", HAMMING_WEIGHT_BINS, self.bins)
    }

    pub fn selectors(self) -> Result<usize, HammingWeightSuccessorError> {
        self.validate()?;
        self.lookup_selectors
            .checked_add(self.bytecode_selectors)
            .and_then(|value| value.checked_add(self.ram_selectors))
            .and_then(|value| value.checked_add(self.inc_chunk_selectors))
            .and_then(|value| value.checked_add(self.inc_carry_selectors))
            .ok_or(HammingWeightSuccessorError::Overflow)
    }
}

/// The five-word row already produced by stage 5 and retained through stage 7.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HammingWeightResidentRow {
    lookup_lo: u64,
    lookup_hi: u64,
    ram_address_plus_one: u64,
    fused_inc_magnitude: u64,
    packed_pc_and_flags: u64,
}

impl HammingWeightResidentRow {
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
}

const _: [(); HAMMING_WEIGHT_ROW_BYTES] = [(); size_of::<HammingWeightResidentRow>()];
const _: [(); 8] = [(); align_of::<HammingWeightResidentRow>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HammingWeightHistogramParams {
    pub rows: u32,
    pub inner_length: u32,
    pub outer_length: u32,
    pub selectors: u32,
    pub bins: u32,
    pub stage_rows: u32,
    pub simd_width: u32,
    pub threads: u32,
    pub inc_bias: u64,
    pub reserved: [u32; 2],
}

const _: [(); 48] = [(); size_of::<HammingWeightHistogramParams>()];
const _: [(); 8] = [(); align_of::<HammingWeightHistogramParams>()];

/// Per-outer-block audit values. Sharding avoids the `u32` overflow that a
/// single retained-contribution counter would hit at `2^28` rows.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HammingWeightAuditRow {
    pub rows_seen: u32,
    pub pc_present: u32,
    pub ram_present: u32,
    pub retained_nonzero_contributions: u32,
    pub occupied_outer_bins: u32,
    pub reserved: [u32; 3],
}

const _: [(); HAMMING_WEIGHT_AUDIT_ROW_BYTES] = [(); size_of::<HammingWeightAuditRow>()];
const _: [(); 4] = [(); align_of::<HammingWeightAuditRow>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HammingWeightStatus {
    pub unsupported_dispatches: u32,
    pub reserved: [u32; 3],
}

const _: [(); HAMMING_WEIGHT_STATUS_BYTES] = [(); size_of::<HammingWeightStatus>()];
const _: [(); 4] = [(); align_of::<HammingWeightStatus>()];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightSuccessorConfig {
    pub inner_log2: usize,
    pub stage_rows: usize,
    pub threads_per_threadgroup: usize,
    pub trace_cutoff: usize,
}

impl Default for HammingWeightSuccessorConfig {
    fn default() -> Self {
        Self {
            inner_log2: HAMMING_WEIGHT_DEFAULT_INNER_LOG2,
            stage_rows: HAMMING_WEIGHT_STAGE_ROWS,
            threads_per_threadgroup: HAMMING_WEIGHT_THREADS,
            trace_cutoff: HAMMING_WEIGHT_DEFAULT_TRACE_CUTOFF,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightShape {
    rows: usize,
    inner_log2: usize,
    inner_length: usize,
    outer_length: usize,
}

impl HammingWeightShape {
    pub fn new(
        rows: usize,
        config: HammingWeightSuccessorConfig,
    ) -> Result<Self, HammingWeightSuccessorError> {
        config.validate()?;
        if rows == 0 || !rows.is_power_of_two() {
            return Err(HammingWeightSuccessorError::InvalidRows { rows });
        }
        if rows < config.trace_cutoff {
            return Err(HammingWeightSuccessorError::TraceBelowCutoff {
                rows,
                cutoff: config.trace_cutoff,
            });
        }
        let log_t = rows.ilog2() as usize;
        let inner_log2 = config
            .inner_log2
            .min(log_t.saturating_sub(HAMMING_WEIGHT_MIN_OUTER_LOG2));
        let inner_length = 1usize << inner_log2;
        if inner_length < config.stage_rows || !inner_length.is_multiple_of(config.stage_rows) {
            return Err(HammingWeightSuccessorError::InvalidInnerLength { inner_length });
        }
        if inner_length > u32::MAX as usize / HAMMING_WEIGHT_SELECTORS {
            return Err(HammingWeightSuccessorError::AuditShardOverflow { inner_length });
        }
        let outer_length = rows / inner_length;
        let _ = shader_count("rows", rows)?;
        let _ = shader_count("inner length", inner_length)?;
        let _ = shader_count("outer length", outer_length)?;
        Ok(Self {
            rows,
            inner_log2,
            inner_length,
            outer_length,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn inner_log2(self) -> usize {
        self.inner_log2
    }

    pub const fn inner_length(self) -> usize {
        self.inner_length
    }

    pub const fn outer_length(self) -> usize {
        self.outer_length
    }

    pub fn params(self) -> Result<HammingWeightHistogramParams, HammingWeightSuccessorError> {
        Ok(HammingWeightHistogramParams {
            rows: shader_count("rows", self.rows)?,
            inner_length: shader_count("inner length", self.inner_length)?,
            outer_length: shader_count("outer length", self.outer_length)?,
            selectors: HAMMING_WEIGHT_SELECTORS as u32,
            bins: HAMMING_WEIGHT_BINS as u32,
            stage_rows: HAMMING_WEIGHT_STAGE_ROWS as u32,
            simd_width: HAMMING_WEIGHT_SIMD_WIDTH as u32,
            threads: HAMMING_WEIGHT_THREADS as u32,
            inc_bias: HAMMING_WEIGHT_BALANCED_INC_BIAS,
            reserved: [0; 2],
        })
    }
}

impl HammingWeightSuccessorConfig {
    pub fn validate(self) -> Result<(), HammingWeightSuccessorError> {
        if self.stage_rows != HAMMING_WEIGHT_STAGE_ROWS {
            return Err(HammingWeightSuccessorError::UnsupportedStageRows {
                got: self.stage_rows,
            });
        }
        if self.threads_per_threadgroup != HAMMING_WEIGHT_THREADS {
            return Err(HammingWeightSuccessorError::UnsupportedThreads {
                got: self.threads_per_threadgroup,
            });
        }
        if self.inner_log2 < HAMMING_WEIGHT_STAGE_ROWS.ilog2() as usize
            || self.inner_log2 >= usize::BITS as usize
        {
            return Err(HammingWeightSuccessorError::InvalidInnerLog2 {
                got: self.inner_log2,
            });
        }
        if !self.trace_cutoff.is_power_of_two() || self.trace_cutoff < HAMMING_WEIGHT_STAGE_ROWS {
            return Err(HammingWeightSuccessorError::InvalidTraceCutoff {
                got: self.trace_cutoff,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum HammingWeightSuccessorError {
    #[error("Hamming-weight rows must be a nonzero power of two, got {rows}")]
    InvalidRows { rows: usize },
    #[error("Hamming-weight trace {rows} is below the Metal cutoff {cutoff}")]
    TraceBelowCutoff { rows: usize, cutoff: usize },
    #[error("Hamming-weight inner log2 is unsupported: {got}")]
    InvalidInnerLog2 { got: usize },
    #[error("Hamming-weight inner length must be staged exactly, got {inner_length}")]
    InvalidInnerLength { inner_length: usize },
    #[error("Hamming-weight inner length {inner_length} can overflow one audit shard")]
    AuditShardOverflow { inner_length: usize },
    #[error("Hamming-weight staging requires {HAMMING_WEIGHT_STAGE_ROWS} rows, got {got}")]
    UnsupportedStageRows { got: usize },
    #[error(
        "Hamming-weight production kernel requires {HAMMING_WEIGHT_THREADS} threads, got {got}"
    )]
    UnsupportedThreads { got: usize },
    #[error("Hamming-weight trace cutoff is invalid: {got}")]
    InvalidTraceCutoff { got: usize },
    #[error("Hamming-weight {name} does not fit the shader ABI: {value}")]
    ShaderCountOverflow { name: &'static str, value: usize },
    #[error("Hamming-weight {name} length mismatch: expected {expected}, got {got}")]
    StorageLength {
        name: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("Hamming-weight unfactored oracle needs {rows} rows, above its fixture cap {maximum}")]
    OracleFixtureTooLarge { rows: usize, maximum: usize },
    #[error("Hamming-weight topology {name} mismatch: expected {expected}, got {got}")]
    UnsupportedTopology {
        name: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("Hamming-weight audit counter {name} mismatch: expected {expected}, got {got}")]
    AuditMismatch {
        name: &'static str,
        expected: u64,
        got: u64,
    },
    #[error("Hamming-weight arithmetic overflow")]
    Overflow,
}

fn shader_count(name: &'static str, value: usize) -> Result<u32, HammingWeightSuccessorError> {
    u32::try_from(value)
        .map_err(|_| HammingWeightSuccessorError::ShaderCountOverflow { name, value })
}

fn check_topology(
    name: &'static str,
    expected: usize,
    got: usize,
) -> Result<(), HammingWeightSuccessorError> {
    if got == expected {
        Ok(())
    } else {
        Err(HammingWeightSuccessorError::UnsupportedTopology {
            name,
            expected,
            got,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::mem::offset_of;

    use super::*;

    #[test]
    fn host_layout_matches_the_metal_abi() {
        assert_eq!(offset_of!(HammingWeightResidentRow, lookup_lo), 0);
        assert_eq!(offset_of!(HammingWeightResidentRow, lookup_hi), 8);
        assert_eq!(
            offset_of!(HammingWeightResidentRow, ram_address_plus_one),
            16
        );
        assert_eq!(
            offset_of!(HammingWeightResidentRow, fused_inc_magnitude),
            24
        );
        assert_eq!(
            offset_of!(HammingWeightResidentRow, packed_pc_and_flags),
            32
        );
        assert_eq!(offset_of!(HammingWeightHistogramParams, inc_bias), 32);
        assert_eq!(offset_of!(HammingWeightHistogramParams, reserved), 40);
    }
}
