use core::mem::{align_of, size_of};

pub const HAMMING_RETAINED_SELECTORS: usize = 29;
pub const HAMMING_RETAINED_BINS: usize = 256;
pub const HAMMING_RETAINED_TILES: usize = 5;
pub const HAMMING_RETAINED_TILE_WIDTHS: [usize; HAMMING_RETAINED_TILES] = [6, 6, 6, 6, 5];
pub const HAMMING_RETAINED_DEFERRED_WORDS: usize = 5;
pub const HAMMING_RETAINED_SIMD_WIDTH: usize = 32;
pub const HAMMING_RETAINED_INNER_LOG2: usize = 15;
pub const HAMMING_RETAINED_INNER_LENGTH: usize = 1 << HAMMING_RETAINED_INNER_LOG2;
pub const HAMMING_RETAINED_ACCUMULATOR_THREADS: usize = 512;
pub const HAMMING_RETAINED_FINALIZE_THREADS: usize = 1024;
pub const HAMMING_RETAINED_MAX_THREADGROUP_BYTES: usize =
    6 * HAMMING_RETAINED_BINS * HAMMING_RETAINED_DEFERRED_WORDS * size_of::<u32>();
pub const HAMMING_RETAINED_FINALIZE_THREADGROUP_BYTES: usize =
    HAMMING_RETAINED_FINALIZE_THREADS * 16;

pub const HAMMING_RETAINED_TILE_PIPELINES: [&str; HAMMING_RETAINED_TILES] = [
    "solinas_hamming_retained_tile_0",
    "solinas_hamming_retained_tile_1",
    "solinas_hamming_retained_tile_2",
    "solinas_hamming_retained_tile_3",
    "solinas_hamming_retained_tile_4",
];
pub const HAMMING_RETAINED_FINALIZE_PIPELINE: &str = "solinas_hamming_retained_finalize";

pub const HAMMING_RETAINED_TILE_BUFFER_HOT: u64 = 0;
pub const HAMMING_RETAINED_TILE_BUFFER_E_IN: u64 = 1;
pub const HAMMING_RETAINED_TILE_BUFFER_E_OUT: u64 = 2;
pub const HAMMING_RETAINED_TILE_BUFFER_PARTIALS: u64 = 3;
pub const HAMMING_RETAINED_TILE_BUFFER_PARAMS: u64 = 4;

pub const HAMMING_RETAINED_FINALIZE_BUFFER_PARTIALS: u64 = 0;
pub const HAMMING_RETAINED_FINALIZE_BUFFER_OUTPUT: u64 = 1;
pub const HAMMING_RETAINED_FINALIZE_BUFFER_PARAMS: u64 = 2;

const FIELD_BYTES: u64 = 16;
const HOT_PLANES: u64 = HAMMING_RETAINED_SELECTORS as u64;

const _: () = assert!(HAMMING_RETAINED_TILE_WIDTHS[0] == 6);
const _: () = assert!(HAMMING_RETAINED_TILE_WIDTHS[4] == 5);
const _: () = assert!(
    HAMMING_RETAINED_TILE_WIDTHS[0]
        + HAMMING_RETAINED_TILE_WIDTHS[1]
        + HAMMING_RETAINED_TILE_WIDTHS[2]
        + HAMMING_RETAINED_TILE_WIDTHS[3]
        + HAMMING_RETAINED_TILE_WIDTHS[4]
        == HAMMING_RETAINED_SELECTORS
);
const _: () = assert!(HAMMING_RETAINED_MAX_THREADGROUP_BYTES == 30_720);
const _: () = assert!(HAMMING_RETAINED_FINALIZE_THREADGROUP_BYTES == 16_384);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightRetainedConfig {
    pub inner_log2: usize,
    pub accumulator_threads_per_threadgroup: usize,
    pub finalize_threads_per_threadgroup: usize,
}

impl Default for HammingWeightRetainedConfig {
    fn default() -> Self {
        Self {
            inner_log2: HAMMING_RETAINED_INNER_LOG2,
            accumulator_threads_per_threadgroup: HAMMING_RETAINED_ACCUMULATOR_THREADS,
            finalize_threads_per_threadgroup: HAMMING_RETAINED_FINALIZE_THREADS,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HammingWeightRetainedParams {
    pub rows: u32,
    pub e_in_length: u32,
    pub e_out_length: u32,
    pub selector_offset: u32,
    pub selectors_in_tile: u32,
    pub bins: u32,
    pub reserved: [u32; 2],
}

const _: [(); 32] = [(); size_of::<HammingWeightRetainedParams>()];
const _: [(); 4] = [(); align_of::<HammingWeightRetainedParams>()];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightRetainedGeometry {
    rows: usize,
    e_out_length: usize,
    accumulator_threads_per_threadgroup: usize,
    finalize_threads_per_threadgroup: usize,
}

impl HammingWeightRetainedGeometry {
    pub fn new(
        rows: usize,
        config: HammingWeightRetainedConfig,
    ) -> Result<Self, HammingWeightRetainedError> {
        if rows < HAMMING_RETAINED_INNER_LENGTH || !rows.is_power_of_two() {
            return Err(HammingWeightRetainedError::InvalidRows(rows));
        }
        if config.inner_log2 != HAMMING_RETAINED_INNER_LOG2 {
            return Err(HammingWeightRetainedError::InvalidInnerLog2(
                config.inner_log2,
            ));
        }
        validate_threads(
            "accumulator",
            config.accumulator_threads_per_threadgroup,
            HAMMING_RETAINED_ACCUMULATOR_THREADS,
        )?;
        validate_threads(
            "finalize",
            config.finalize_threads_per_threadgroup,
            HAMMING_RETAINED_FINALIZE_THREADS,
        )?;
        let _ = shader_u32("rows", rows)?;
        let e_out_length = rows / HAMMING_RETAINED_INNER_LENGTH;
        let _ = shader_u32("e_out length", e_out_length)?;
        Ok(Self {
            rows,
            e_out_length,
            accumulator_threads_per_threadgroup: config.accumulator_threads_per_threadgroup,
            finalize_threads_per_threadgroup: config.finalize_threads_per_threadgroup,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn e_in_length(self) -> usize {
        HAMMING_RETAINED_INNER_LENGTH
    }

    pub const fn e_out_length(self) -> usize {
        self.e_out_length
    }

    pub const fn accumulator_threads_per_threadgroup(self) -> usize {
        self.accumulator_threads_per_threadgroup
    }

    pub const fn finalize_threads_per_threadgroup(self) -> usize {
        self.finalize_threads_per_threadgroup
    }

    pub fn params(
        self,
        tile: usize,
    ) -> Result<HammingWeightRetainedParams, HammingWeightRetainedError> {
        let selectors_in_tile = *HAMMING_RETAINED_TILE_WIDTHS
            .get(tile)
            .ok_or(HammingWeightRetainedError::InvalidTile(tile))?;
        let selector_offset = HAMMING_RETAINED_TILE_WIDTHS[..tile].iter().sum::<usize>();
        Ok(HammingWeightRetainedParams {
            rows: shader_u32("rows", self.rows)?,
            e_in_length: HAMMING_RETAINED_INNER_LENGTH as u32,
            e_out_length: shader_u32("e_out length", self.e_out_length)?,
            selector_offset: selector_offset as u32,
            selectors_in_tile: selectors_in_tile as u32,
            bins: HAMMING_RETAINED_BINS as u32,
            reserved: [0; 2],
        })
    }

    pub fn buffer_lengths(
        self,
    ) -> Result<HammingWeightRetainedBufferLengths, HammingWeightRetainedError> {
        Ok(HammingWeightRetainedBufferLengths {
            hot_bytes: checked_mul(self.rows as u64, HOT_PLANES)?,
            e_in_fields: HAMMING_RETAINED_INNER_LENGTH as u64,
            e_out_fields: self.e_out_length as u64,
            partial_fields: checked_product(&[
                self.e_out_length as u64,
                6,
                HAMMING_RETAINED_BINS as u64,
            ])?,
            output_fields: checked_mul(
                HAMMING_RETAINED_SELECTORS as u64,
                HAMMING_RETAINED_BINS as u64,
            )?,
        })
    }

    pub fn dispatch_plan(self) -> HammingWeightRetainedDispatchPlan {
        HammingWeightRetainedDispatchPlan {
            command_buffers: 1,
            encoders: 2 * HAMMING_RETAINED_TILES,
            dispatches: 2 * HAMMING_RETAINED_TILES,
            completion_waits: 1,
            readbacks: 1,
            tile_threadgroups: self.e_out_length * HAMMING_RETAINED_TILES,
            finalize_threadgroups: HAMMING_RETAINED_SELECTORS,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HammingWeightRetainedBufferLengths {
    pub hot_bytes: u64,
    pub e_in_fields: u64,
    pub e_out_fields: u64,
    pub partial_fields: u64,
    pub output_fields: u64,
}

impl HammingWeightRetainedBufferLengths {
    pub fn owned_bytes(self) -> Result<u64, HammingWeightRetainedError> {
        [
            checked_mul(self.e_in_fields, FIELD_BYTES)?,
            checked_mul(self.e_out_fields, FIELD_BYTES)?,
            checked_mul(self.partial_fields, FIELD_BYTES)?,
            checked_mul(self.output_fields, FIELD_BYTES)?,
        ]
        .into_iter()
        .try_fold(0u64, |sum, bytes| {
            sum.checked_add(bytes)
                .ok_or(HammingWeightRetainedError::ArithmeticOverflow)
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightRetainedDispatchPlan {
    pub command_buffers: usize,
    pub encoders: usize,
    pub dispatches: usize,
    pub completion_waits: usize,
    pub readbacks: usize,
    pub tile_threadgroups: usize,
    pub finalize_threadgroups: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum HammingWeightRetainedError {
    #[error("retained-Hamming rows must be a power of two at least 2^15, got {0}")]
    InvalidRows(usize),
    #[error("retained-Hamming inner log2 is fixed at 15, got {0}")]
    InvalidInnerLog2(usize),
    #[error("retained-Hamming {name} threads are fixed at {expected}, got {got}")]
    InvalidThreads {
        name: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("retained-Hamming tile index {0} is out of range")]
    InvalidTile(usize),
    #[error("retained-Hamming {name} value {value} does not fit the shader ABI")]
    ShaderValueTooLong { name: &'static str, value: usize },
    #[error("retained-Hamming size arithmetic overflow")]
    ArithmeticOverflow,
}

fn validate_threads(
    name: &'static str,
    got: usize,
    expected: usize,
) -> Result<(), HammingWeightRetainedError> {
    if got != expected {
        return Err(HammingWeightRetainedError::InvalidThreads {
            name,
            expected,
            got,
        });
    }
    Ok(())
}

fn shader_u32(name: &'static str, value: usize) -> Result<u32, HammingWeightRetainedError> {
    u32::try_from(value).map_err(|_| HammingWeightRetainedError::ShaderValueTooLong { name, value })
}

fn checked_mul(lhs: u64, rhs: u64) -> Result<u64, HammingWeightRetainedError> {
    lhs.checked_mul(rhs)
        .ok_or(HammingWeightRetainedError::ArithmeticOverflow)
}

fn checked_product(values: &[u64]) -> Result<u64, HammingWeightRetainedError> {
    values
        .iter()
        .try_fold(1u64, |product, value| checked_mul(product, *value))
}
