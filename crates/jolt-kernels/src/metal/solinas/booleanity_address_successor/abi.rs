//! Checked host/shader ABI for the packed-hot successor slice.

use core::mem::{align_of, size_of};

pub const BOOLEANITY_ADDRESS_SUCCESSOR_BINS: usize = 256;
pub const BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS: usize = 29;
pub const BOOLEANITY_ADDRESS_SUCCESSOR_FIRST_TILE_SELECTORS: usize = 6;
pub const BOOLEANITY_ADDRESS_SUCCESSOR_PACKED_SELECTORS: usize =
    BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS - BOOLEANITY_ADDRESS_SUCCESSOR_FIRST_TILE_SELECTORS;
pub const BOOLEANITY_ADDRESS_SUCCESSOR_HOT_PLANES: usize = BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS;
pub const BOOLEANITY_ADDRESS_SUCCESSOR_VALIDITY_PLANES: usize = 1;
pub const BOOLEANITY_ADDRESS_SUCCESSOR_PACKED_PLANES: usize =
    BOOLEANITY_ADDRESS_SUCCESSOR_HOT_PLANES + BOOLEANITY_ADDRESS_SUCCESSOR_VALIDITY_PLANES;
pub const BOOLEANITY_ADDRESS_SUCCESSOR_REMAINING_TILES: usize = 4;
pub const BOOLEANITY_ADDRESS_SUCCESSOR_DEFERRED_WORDS: usize = 5;
pub const BOOLEANITY_ADDRESS_SUCCESSOR_SIMD_WIDTH: usize = 32;
pub const BOOLEANITY_ADDRESS_SUCCESSOR_INNER_LOG2: usize = 15;
pub const BOOLEANITY_ADDRESS_SUCCESSOR_INNER_LENGTH: usize =
    1 << BOOLEANITY_ADDRESS_SUCCESSOR_INNER_LOG2;
pub const BOOLEANITY_ADDRESS_SUCCESSOR_ACCUMULATOR_THREADS: usize = 512;
pub const BOOLEANITY_ADDRESS_SUCCESSOR_FINALIZE_THREADS: usize = 1024;
pub const BOOLEANITY_ADDRESS_SUCCESSOR_INC_BIAS: u64 = 0x8080_8080_8080_8080;

pub const BOOLEANITY_ADDRESS_SUCCESSOR_THREADGROUP_BYTES: usize =
    BOOLEANITY_ADDRESS_SUCCESSOR_FIRST_TILE_SELECTORS
        * BOOLEANITY_ADDRESS_SUCCESSOR_BINS
        * BOOLEANITY_ADDRESS_SUCCESSOR_DEFERRED_WORDS
        * size_of::<u32>();
pub const BOOLEANITY_ADDRESS_SUCCESSOR_FINALIZE_THREADGROUP_BYTES: usize =
    BOOLEANITY_ADDRESS_SUCCESSOR_FINALIZE_THREADS * 16;

pub const PACK_AND_FIRST_PIPELINE: &str = "solinas_booleanity_address_successor_pack_and_first";
pub const PACKED_TILES_PIPELINE: &str = "solinas_booleanity_address_successor_packed_tiles";
pub const FINALIZE_PIPELINE: &str = "solinas_booleanity_address_successor_finalize";

pub const PACK_AND_FIRST_BUFFER_ROWS: u64 = 0;
pub const PACK_AND_FIRST_BUFFER_E_IN: u64 = 1;
pub const PACK_AND_FIRST_BUFFER_E_OUT: u64 = 2;
pub const PACK_AND_FIRST_BUFFER_HOT: u64 = 3;
pub const PACK_AND_FIRST_BUFFER_VALIDITY: u64 = 4;
pub const PACK_AND_FIRST_BUFFER_PARTIALS: u64 = 5;
pub const PACK_AND_FIRST_BUFFER_PARAMS: u64 = 6;

pub const PACKED_TILES_BUFFER_HOT: u64 = 0;
pub const PACKED_TILES_BUFFER_VALIDITY: u64 = 1;
pub const PACKED_TILES_BUFFER_E_IN: u64 = 2;
pub const PACKED_TILES_BUFFER_E_OUT: u64 = 3;
pub const PACKED_TILES_BUFFER_PARTIALS: u64 = 4;
pub const PACKED_TILES_BUFFER_PARAMS: u64 = 5;

pub const FINALIZE_BUFFER_PARTIALS: u64 = 0;
pub const FINALIZE_BUFFER_OUTPUT: u64 = 1;
pub const FINALIZE_BUFFER_PARAMS: u64 = 2;

const FIELD_BYTES: u64 = 16;
const ROW_BYTES: u64 = 40;

const _: () = assert!(BOOLEANITY_ADDRESS_SUCCESSOR_PACKED_SELECTORS == 23);
const _: () = assert!(BOOLEANITY_ADDRESS_SUCCESSOR_HOT_PLANES == 29);
const _: () = assert!(BOOLEANITY_ADDRESS_SUCCESSOR_PACKED_PLANES == 30);
const _: () = assert!(BOOLEANITY_ADDRESS_SUCCESSOR_THREADGROUP_BYTES == 30_720);
const _: () = assert!(BOOLEANITY_ADDRESS_SUCCESSOR_FINALIZE_THREADGROUP_BYTES == 16_384);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BooleanityAddressSuccessorConfig {
    pub inner_log2: usize,
    pub accumulator_threads_per_threadgroup: usize,
    pub finalize_threads_per_threadgroup: usize,
}

impl Default for BooleanityAddressSuccessorConfig {
    fn default() -> Self {
        Self {
            inner_log2: BOOLEANITY_ADDRESS_SUCCESSOR_INNER_LOG2,
            accumulator_threads_per_threadgroup: BOOLEANITY_ADDRESS_SUCCESSOR_ACCUMULATOR_THREADS,
            finalize_threads_per_threadgroup: BOOLEANITY_ADDRESS_SUCCESSOR_FINALIZE_THREADS,
        }
    }
}

#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BooleanityAddressSuccessorParams {
    pub rows: u32,
    pub e_in_length: u32,
    pub e_out_length: u32,
    pub selector_count: u32,
    pub inc_bias: u64,
    pub packed_selector_base: u32,
    pub packed_planes: u32,
    pub remaining_tiles: u32,
    pub reserved: u32,
}

const _: [(); 40] = [(); size_of::<BooleanityAddressSuccessorParams>()];
const _: [(); 8] = [(); align_of::<BooleanityAddressSuccessorParams>()];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BooleanityAddressSuccessorGeometry {
    rows: usize,
    e_in_length: usize,
    e_out_length: usize,
    accumulator_threads_per_threadgroup: usize,
    finalize_threads_per_threadgroup: usize,
}

impl BooleanityAddressSuccessorGeometry {
    pub fn new(
        rows: usize,
        config: BooleanityAddressSuccessorConfig,
    ) -> Result<Self, BooleanityAddressSuccessorError> {
        if rows == 0 || !rows.is_power_of_two() || rows < BOOLEANITY_ADDRESS_SUCCESSOR_INNER_LENGTH
        {
            return Err(BooleanityAddressSuccessorError::InvalidRows(rows));
        }
        if config.inner_log2 != BOOLEANITY_ADDRESS_SUCCESSOR_INNER_LOG2 {
            return Err(BooleanityAddressSuccessorError::InvalidInnerLog2(
                config.inner_log2,
            ));
        }
        validate_threads(
            "accumulator",
            config.accumulator_threads_per_threadgroup,
            BOOLEANITY_ADDRESS_SUCCESSOR_ACCUMULATOR_THREADS,
        )?;
        validate_threads(
            "finalize",
            config.finalize_threads_per_threadgroup,
            BOOLEANITY_ADDRESS_SUCCESSOR_FINALIZE_THREADS,
        )?;
        let _ = shader_u32("rows", rows)?;
        let e_out_length = rows / BOOLEANITY_ADDRESS_SUCCESSOR_INNER_LENGTH;
        let _ = shader_u32("e_out length", e_out_length)?;
        Ok(Self {
            rows,
            e_in_length: BOOLEANITY_ADDRESS_SUCCESSOR_INNER_LENGTH,
            e_out_length,
            accumulator_threads_per_threadgroup: config.accumulator_threads_per_threadgroup,
            finalize_threads_per_threadgroup: config.finalize_threads_per_threadgroup,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn e_in_length(self) -> usize {
        self.e_in_length
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
    ) -> Result<BooleanityAddressSuccessorParams, BooleanityAddressSuccessorError> {
        Ok(BooleanityAddressSuccessorParams {
            rows: shader_u32("rows", self.rows)?,
            e_in_length: shader_u32("e_in length", self.e_in_length)?,
            e_out_length: shader_u32("e_out length", self.e_out_length)?,
            selector_count: BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS as u32,
            inc_bias: BOOLEANITY_ADDRESS_SUCCESSOR_INC_BIAS,
            packed_selector_base: BOOLEANITY_ADDRESS_SUCCESSOR_FIRST_TILE_SELECTORS as u32,
            packed_planes: BOOLEANITY_ADDRESS_SUCCESSOR_HOT_PLANES as u32,
            remaining_tiles: BOOLEANITY_ADDRESS_SUCCESSOR_REMAINING_TILES as u32,
            reserved: 0,
        })
    }

    pub fn buffer_lengths(
        self,
    ) -> Result<BooleanityAddressSuccessorBufferLengths, BooleanityAddressSuccessorError> {
        let rows = self.rows as u64;
        let partial_fields = checked_mul(
            checked_mul(
                BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS as u64,
                BOOLEANITY_ADDRESS_SUCCESSOR_BINS as u64,
            )?,
            self.e_out_length as u64,
        )?;
        let output_fields = checked_mul(
            BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS as u64,
            BOOLEANITY_ADDRESS_SUCCESSOR_BINS as u64,
        )?;
        Ok(BooleanityAddressSuccessorBufferLengths {
            resident_row_bytes: checked_mul(rows, ROW_BYTES)?,
            hot_bytes: checked_mul(rows, BOOLEANITY_ADDRESS_SUCCESSOR_HOT_PLANES as u64)?,
            validity_bytes: checked_mul(rows, BOOLEANITY_ADDRESS_SUCCESSOR_VALIDITY_PLANES as u64)?,
            e_in_fields: self.e_in_length as u64,
            e_out_fields: self.e_out_length as u64,
            partial_fields,
            output_fields,
        })
    }

    pub fn dispatch_plan(
        self,
    ) -> Result<BooleanityAddressSuccessorDispatchPlan, BooleanityAddressSuccessorError> {
        Ok(BooleanityAddressSuccessorDispatchPlan {
            command_buffers: 1,
            encoders: 3,
            dispatches: 3,
            completion_waits: 1,
            readbacks: 1,
            pack_and_first_threadgroups: shader_u32("pack threadgroups", self.e_out_length)?,
            packed_tile_threadgroups: shader_u32(
                "packed tile threadgroups",
                checked_usize_mul(
                    self.e_out_length,
                    BOOLEANITY_ADDRESS_SUCCESSOR_REMAINING_TILES,
                )?,
            )?,
            finalize_threadgroups: BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS as u32,
            row_lanes_per_simd: BOOLEANITY_ADDRESS_SUCCESSOR_SIMD_WIDTH as u32,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BooleanityAddressSuccessorBufferLengths {
    pub resident_row_bytes: u64,
    pub hot_bytes: u64,
    pub validity_bytes: u64,
    pub e_in_fields: u64,
    pub e_out_fields: u64,
    pub partial_fields: u64,
    pub output_fields: u64,
}

impl BooleanityAddressSuccessorBufferLengths {
    pub fn owned_bytes(self) -> Result<u64, BooleanityAddressSuccessorError> {
        [
            self.hot_bytes,
            self.validity_bytes,
            checked_mul(self.e_in_fields, FIELD_BYTES)?,
            checked_mul(self.e_out_fields, FIELD_BYTES)?,
            checked_mul(self.partial_fields, FIELD_BYTES)?,
            checked_mul(self.output_fields, FIELD_BYTES)?,
        ]
        .into_iter()
        .try_fold(0u64, |sum, value| {
            sum.checked_add(value)
                .ok_or(BooleanityAddressSuccessorError::ArithmeticOverflow)
        })
    }

    pub fn validate(self, got: Self) -> Result<(), BooleanityAddressSuccessorError> {
        for (name, expected, got) in [
            (
                "resident rows",
                self.resident_row_bytes,
                got.resident_row_bytes,
            ),
            ("hot rows", self.hot_bytes, got.hot_bytes),
            ("validity rows", self.validity_bytes, got.validity_bytes),
            ("e_in", self.e_in_fields, got.e_in_fields),
            ("e_out", self.e_out_fields, got.e_out_fields),
            ("partials", self.partial_fields, got.partial_fields),
            ("output", self.output_fields, got.output_fields),
        ] {
            if expected != got {
                return Err(BooleanityAddressSuccessorError::BufferLength {
                    name,
                    expected,
                    got,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BooleanityAddressSuccessorDispatchPlan {
    pub command_buffers: u32,
    pub encoders: u32,
    pub dispatches: u32,
    pub completion_waits: u32,
    pub readbacks: u32,
    pub pack_and_first_threadgroups: u32,
    pub packed_tile_threadgroups: u32,
    pub finalize_threadgroups: u32,
    /// Every SIMD lane owns a distinct cycle row; no lane owns a bucket.
    pub row_lanes_per_simd: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BooleanityAddressSuccessorError {
    InvalidRows(usize),
    InvalidInnerLog2(usize),
    InvalidThreads {
        phase: &'static str,
        expected: usize,
        got: usize,
    },
    ShaderIndexOverflow {
        name: &'static str,
        value: usize,
    },
    BufferLength {
        name: &'static str,
        expected: u64,
        got: u64,
    },
    WeightShape {
        rows: usize,
        e_in: usize,
        e_out: usize,
    },
    PackedStorageLength {
        expected: usize,
        got: usize,
    },
    ValidityStorageLength {
        expected: usize,
        got: usize,
    },
    InvalidPackedSelector(usize),
    RowOutOfBounds {
        rows: usize,
        row: usize,
    },
    InvalidCensus {
        name: &'static str,
        rows: u64,
        got: u64,
    },
    ArithmeticOverflow,
}

pub(crate) fn validate_weight_shape(
    rows: usize,
    e_in: usize,
    e_out: usize,
) -> Result<(), BooleanityAddressSuccessorError> {
    if e_in == 0
        || e_out == 0
        || e_in
            .checked_mul(e_out)
            .is_none_or(|covered| covered != rows)
    {
        return Err(BooleanityAddressSuccessorError::WeightShape { rows, e_in, e_out });
    }
    Ok(())
}

fn validate_threads(
    phase: &'static str,
    got: usize,
    expected: usize,
) -> Result<(), BooleanityAddressSuccessorError> {
    if got != expected {
        return Err(BooleanityAddressSuccessorError::InvalidThreads {
            phase,
            expected,
            got,
        });
    }
    Ok(())
}

fn shader_u32(name: &'static str, value: usize) -> Result<u32, BooleanityAddressSuccessorError> {
    u32::try_from(value)
        .map_err(|_| BooleanityAddressSuccessorError::ShaderIndexOverflow { name, value })
}

fn checked_mul(left: u64, right: u64) -> Result<u64, BooleanityAddressSuccessorError> {
    left.checked_mul(right)
        .ok_or(BooleanityAddressSuccessorError::ArithmeticOverflow)
}

fn checked_usize_mul(left: usize, right: usize) -> Result<usize, BooleanityAddressSuccessorError> {
    left.checked_mul(right)
        .ok_or(BooleanityAddressSuccessorError::ArithmeticOverflow)
}
