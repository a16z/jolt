//! Checked geometry and independent host oracle for the Spartan shift design.

use std::mem::{align_of, size_of};

use jolt_field::{AdditiveAccumulator, Field, RingAccumulator};
use jolt_poly::{EqPlusOnePrefixSuffix, EqPolynomial, Polynomial, UnivariatePoly};
use thiserror::Error;

mod runtime;

pub use runtime::{
    PendingSpartanShiftFold, PendingSpartanShiftPrefix, SpartanShiftFoldInvocation,
    SpartanShiftFoldObservation, SpartanShiftPrefixInvocation, SpartanShiftPrefixObservation,
    SpartanShiftResidentRows,
};

pub const SOURCE: &str = include_str!("shader.metal");

pub const SPARTAN_SHIFT_SIMD_WIDTH: usize = 32;
pub const SPARTAN_SHIFT_MAX_THREADS_PER_THREADGROUP: usize = 1024;
pub const SPARTAN_SHIFT_OUTPUT_COLUMNS: usize = 5;
pub const SPARTAN_SHIFT_PREFIX_PAIRS: usize = 4;
pub const SPARTAN_SHIFT_FLAG_ROWS_PER_WORD: usize = 32;
pub const SPARTAN_SHIFT_TARGET_LOG_T: usize = 26;

pub const BUILD_MIXED_PIPELINE: &str = "solinas_spartan_shift_build_mixed_partials";
pub const REDUCE_PREFIX_PIPELINE: &str = "solinas_spartan_shift_reduce_prefix";
pub const FOLD_NATIVE_PIPELINE: &str = "solinas_spartan_shift_fold_native";

/// Three current-cycle flag bitplanes for one block of 32 consecutive cycles.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SpartanShiftFlagWord {
    pub is_virtual: u32,
    pub is_first_in_sequence: u32,
    pub is_noop: u32,
}

const _: [(); 12] = [(); size_of::<SpartanShiftFlagWord>()];
const _: [(); 4] = [(); align_of::<SpartanShiftFlagWord>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SpartanShiftParams {
    pub prefix_elements: u32,
    pub suffix_elements: u32,
    pub high_tile_elements: u32,
    pub high_tiles: u32,
}

const _: [(); 16] = [(); size_of::<SpartanShiftParams>()];
const _: [(); 4] = [(); align_of::<SpartanShiftParams>()];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpartanShiftKernelConfig {
    pub build_threads_per_threadgroup: usize,
    pub high_tile_elements: usize,
    pub fold_threads_per_threadgroup: usize,
}

impl Default for SpartanShiftKernelConfig {
    fn default() -> Self {
        Self {
            build_threads_per_threadgroup: 64,
            high_tile_elements: 128,
            fold_threads_per_threadgroup: 32,
        }
    }
}

impl SpartanShiftKernelConfig {
    pub fn validate(self) -> Result<Self, SpartanShiftPlanError> {
        for (phase, width) in [
            ("build", self.build_threads_per_threadgroup),
            ("fold", self.fold_threads_per_threadgroup),
        ] {
            if width == 0
                || width > SPARTAN_SHIFT_MAX_THREADS_PER_THREADGROUP
                || !width.is_multiple_of(SPARTAN_SHIFT_SIMD_WIDTH)
            {
                return Err(SpartanShiftPlanError::InvalidThreadgroupWidth { phase, width });
            }
        }
        if self.high_tile_elements == 0 || !self.high_tile_elements.is_power_of_two() {
            return Err(SpartanShiftPlanError::InvalidHighTile(
                self.high_tile_elements,
            ));
        }
        Ok(self)
    }

    pub fn fold_threadgroup_bytes(self) -> Result<usize, SpartanShiftPlanError> {
        let config = self.validate()?;
        let simdgroups = config.fold_threads_per_threadgroup / SPARTAN_SHIFT_SIMD_WIDTH;
        checked_product(
            "fold threadgroup bytes",
            SPARTAN_SHIFT_OUTPUT_COLUMNS * simdgroups,
            16,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpartanShiftGeometry {
    rows: usize,
    log_t: usize,
    prefix_vars: usize,
    suffix_vars: usize,
    prefix_elements: usize,
    suffix_elements: usize,
    flag_words: usize,
}

impl SpartanShiftGeometry {
    pub fn new(rows: usize) -> Result<Self, SpartanShiftPlanError> {
        if rows < 2 || !rows.is_power_of_two() {
            return Err(SpartanShiftPlanError::InvalidRows(rows));
        }
        let _ = abi_count("rows", rows)?;
        let log_t = rows.trailing_zeros() as usize;
        let suffix_vars = log_t / 2;
        let prefix_vars = log_t - suffix_vars;
        let prefix_elements = checked_power_of_two("prefix elements", prefix_vars)?;
        let suffix_elements = checked_power_of_two("suffix elements", suffix_vars)?;
        let flag_words = rows.div_ceil(SPARTAN_SHIFT_FLAG_ROWS_PER_WORD);
        Ok(Self {
            rows,
            log_t,
            prefix_vars,
            suffix_vars,
            prefix_elements,
            suffix_elements,
            flag_words,
        })
    }

    pub const fn target() -> Self {
        Self {
            rows: 1 << 26,
            log_t: 26,
            prefix_vars: 13,
            suffix_vars: 13,
            prefix_elements: 1 << 13,
            suffix_elements: 1 << 13,
            flag_words: 1 << 21,
        }
    }

    copy_field_getters! { pub, {
        rows: usize,
        log_t: usize,
        prefix_vars: usize,
        suffix_vars: usize,
        prefix_elements: usize,
        suffix_elements: usize,
        flag_words: usize,
    }}

    pub fn row_index(self, x_hi: usize, x_lo: usize) -> Result<usize, SpartanShiftPlanError> {
        if x_hi >= self.suffix_elements || x_lo >= self.prefix_elements {
            return Err(SpartanShiftPlanError::CoordinateOutOfRange {
                x_hi,
                x_lo,
                suffix_elements: self.suffix_elements,
                prefix_elements: self.prefix_elements,
            });
        }
        Ok(x_hi * self.prefix_elements + x_lo)
    }

    pub fn params(
        self,
        config: SpartanShiftKernelConfig,
    ) -> Result<SpartanShiftParams, SpartanShiftPlanError> {
        let config = config.validate()?;
        if !self
            .suffix_elements
            .is_multiple_of(config.high_tile_elements)
        {
            return Err(SpartanShiftPlanError::NonIntegralHighTiles {
                suffix_elements: self.suffix_elements,
                high_tile_elements: config.high_tile_elements,
            });
        }
        Ok(SpartanShiftParams {
            prefix_elements: abi_count("prefix elements", self.prefix_elements)?,
            suffix_elements: abi_count("suffix elements", self.suffix_elements)?,
            high_tile_elements: abi_count("high tile elements", config.high_tile_elements)?,
            high_tiles: abi_count(
                "high tiles",
                self.suffix_elements / config.high_tile_elements,
            )?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpartanShiftPlan {
    pub geometry: SpartanShiftGeometry,
    pub config: SpartanShiftKernelConfig,
    pub params: SpartanShiftParams,
    pub storage: SpartanShiftStorage,
    pub cost: SpartanShiftCost,
}

impl SpartanShiftPlan {
    pub fn new(
        rows: usize,
        config: SpartanShiftKernelConfig,
    ) -> Result<Self, SpartanShiftPlanError> {
        let geometry = SpartanShiftGeometry::new(rows)?;
        let config = config.validate()?;
        let params = geometry.params(config)?;
        let storage = storage(geometry, config)?;
        let cost = cost(geometry, config)?;
        Ok(Self {
            geometry,
            config,
            params,
            storage,
            cost,
        })
    }

    pub fn build_threadgroups(self) -> usize {
        self.geometry
            .prefix_elements
            .div_ceil(self.config.build_threads_per_threadgroup)
            * self.params.high_tiles as usize
    }

    pub const fn fold_threadgroups(self) -> usize {
        self.geometry.suffix_elements
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpartanShiftStorage {
    pub native_value_bytes: usize,
    pub native_flag_bytes: usize,
    pub high_weight_bytes: usize,
    pub low_weight_bytes: usize,
    pub partial_bytes: usize,
    pub q_bytes: usize,
    pub dense_output_bytes: usize,
    pub total_resident_bytes: usize,
    pub maximum_buffer_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpartanShiftCost {
    pub high_tiles: usize,
    pub halo_rows: usize,
    pub halo_flag_words: usize,
    pub build_row_evaluations: usize,
    pub mixed_full_products: usize,
    pub mixed_half_products: usize,
    pub fold_half_products: usize,
    pub prefix_host_products: usize,
    pub suffix_host_products: usize,
    pub build_unique_bytes: usize,
    pub build_halo_value_bytes: usize,
    pub build_halo_flag_bytes: usize,
    pub build_coalesced_bytes_with_halo: usize,
    pub fold_unique_bytes: usize,
    pub readback_bytes: usize,
    pub command_buffers: usize,
    pub dispatches: usize,
}

fn storage(
    geometry: SpartanShiftGeometry,
    config: SpartanShiftKernelConfig,
) -> Result<SpartanShiftStorage, SpartanShiftPlanError> {
    let high_tiles = geometry.suffix_elements / config.high_tile_elements;
    let partials = checked_product("prefix partials", geometry.prefix_elements, high_tiles)?;
    let native_value_bytes = checked_bytes("native PC values", 2 * geometry.rows, 8)?;
    let native_flag_bytes = checked_bytes(
        "native flag words",
        geometry.flag_words,
        size_of::<SpartanShiftFlagWord>(),
    )?;
    let high_weight_bytes = checked_bytes("high weights", 2 * geometry.suffix_elements, 16)?;
    let low_weight_bytes = checked_bytes("low weights", geometry.prefix_elements, 16)?;
    let partial_bytes =
        checked_bytes("prefix partials", SPARTAN_SHIFT_PREFIX_PAIRS * partials, 16)?;
    let q_bytes = checked_bytes(
        "prefix Q",
        SPARTAN_SHIFT_PREFIX_PAIRS * geometry.prefix_elements,
        16,
    )?;
    let dense_output_bytes = checked_bytes(
        "dense outputs",
        SPARTAN_SHIFT_OUTPUT_COLUMNS * geometry.suffix_elements,
        16,
    )?;
    let total_resident_bytes = checked_sum(
        "resident storage",
        &[
            native_value_bytes,
            native_flag_bytes,
            high_weight_bytes,
            low_weight_bytes,
            partial_bytes,
            q_bytes,
            dense_output_bytes,
            size_of::<SpartanShiftParams>() + 4 * 16,
        ],
    )?;
    let maximum_buffer_bytes = [
        native_value_bytes / 2,
        native_flag_bytes,
        high_weight_bytes,
        low_weight_bytes,
        partial_bytes,
        q_bytes,
        dense_output_bytes,
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    Ok(SpartanShiftStorage {
        native_value_bytes,
        native_flag_bytes,
        high_weight_bytes,
        low_weight_bytes,
        partial_bytes,
        q_bytes,
        dense_output_bytes,
        total_resident_bytes,
        maximum_buffer_bytes,
    })
}

fn coalesced_halo_flag_words(
    geometry: SpartanShiftGeometry,
    config: SpartanShiftKernelConfig,
) -> Result<usize, SpartanShiftPlanError> {
    let high_tiles = geometry.suffix_elements / config.high_tile_elements;
    let mut words = 0usize;
    for tile in 1..high_tiles {
        let high = checked_product("halo high coordinate", tile, config.high_tile_elements)?;
        let first_row = checked_product("halo first row", high, geometry.prefix_elements)?;
        let last_row = first_row
            .checked_add(geometry.prefix_elements - 1)
            .ok_or(SpartanShiftPlanError::SizeOverflow)?;
        let first_word = first_row / SPARTAN_SHIFT_FLAG_ROWS_PER_WORD;
        let last_word = last_row / SPARTAN_SHIFT_FLAG_ROWS_PER_WORD;
        words = words
            .checked_add(last_word - first_word + 1)
            .ok_or(SpartanShiftPlanError::SizeOverflow)?;
    }
    Ok(words)
}

fn cost(
    geometry: SpartanShiftGeometry,
    config: SpartanShiftKernelConfig,
) -> Result<SpartanShiftCost, SpartanShiftPlanError> {
    let high_tiles = geometry.suffix_elements / config.high_tile_elements;
    let internal_halos = high_tiles - 1;
    let halo_rows = checked_product("build halo rows", geometry.prefix_elements, internal_halos)?;
    let halo_flag_words = coalesced_halo_flag_words(geometry, config)?;
    let build_row_evaluations = geometry
        .rows
        .checked_add(halo_rows)
        .ok_or(SpartanShiftPlanError::SizeOverflow)?;
    let successor_rows = geometry.rows - geometry.prefix_elements;
    let mixed_full_products = geometry
        .rows
        .checked_add(successor_rows)
        .ok_or(SpartanShiftPlanError::SizeOverflow)?;
    let mixed_half_products = build_row_evaluations;
    let fold_half_products = checked_product("fold half products", 2, geometry.rows)?;
    let prefix_host_products = geometry
        .prefix_elements
        .checked_mul(16)
        .and_then(|products| products.checked_sub(24))
        .ok_or(SpartanShiftPlanError::SizeOverflow)?;
    let suffix_host_products = geometry
        .suffix_elements
        .checked_mul(19)
        .and_then(|products| products.checked_sub(19))
        .ok_or(SpartanShiftPlanError::SizeOverflow)?;

    let storage = storage(geometry, config)?;
    let partial_read_write = checked_product("partial read/write", storage.partial_bytes, 2)?;
    let build_unique_bytes = checked_sum(
        "build unique traffic",
        &[
            storage.native_value_bytes,
            storage.native_flag_bytes,
            storage.high_weight_bytes,
            partial_read_write,
            storage.q_bytes,
        ],
    )?;
    let build_halo_value_bytes = checked_bytes("halo value traffic", halo_rows, 16)?;
    let build_halo_flag_bytes = checked_bytes(
        "halo flag traffic",
        halo_flag_words,
        size_of::<SpartanShiftFlagWord>(),
    )?;
    let build_coalesced_bytes_with_halo = checked_sum(
        "build coalesced traffic",
        &[
            build_unique_bytes,
            build_halo_value_bytes,
            build_halo_flag_bytes,
        ],
    )?;
    let fold_unique_bytes = checked_sum(
        "fold unique traffic",
        &[
            storage.native_value_bytes,
            storage.native_flag_bytes,
            storage.low_weight_bytes,
            storage.dense_output_bytes,
        ],
    )?;
    let readback_bytes = storage
        .q_bytes
        .checked_add(storage.dense_output_bytes)
        .ok_or(SpartanShiftPlanError::SizeOverflow)?;
    Ok(SpartanShiftCost {
        high_tiles,
        halo_rows,
        halo_flag_words,
        build_row_evaluations,
        mixed_full_products,
        mixed_half_products,
        fold_half_products,
        prefix_host_products,
        suffix_host_products,
        build_unique_bytes,
        build_halo_value_bytes,
        build_halo_flag_bytes,
        build_coalesced_bytes_with_halo,
        fold_unique_bytes,
        readback_bytes,
        command_buffers: 2,
        dispatches: 3,
    })
}

/// Non-owning facts that a future adapter must derive from an actual buffer.
///
/// `allocation_identity` must be stable for the buffer's borrowed lifetime; it
/// is not a caller-selected label.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentSpartanShiftBufferMetadata {
    pub allocation_identity: usize,
    pub byte_len: usize,
}

/// Checked description of the three buffers borrowed by both Metal commands.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentSpartanShiftMetadata {
    pub rows: usize,
    pub unexpanded_pc: ResidentSpartanShiftBufferMetadata,
    pub pc: ResidentSpartanShiftBufferMetadata,
    pub flags: ResidentSpartanShiftBufferMetadata,
    pub device_registry_id: u64,
    pub exact_current_flags: bool,
}

impl ResidentSpartanShiftMetadata {
    /// Checks metadata copied from the three resident buffers.
    ///
    /// The eventual Metal adapter must read the identities and lengths from its
    /// actual buffers before calling this method. This type does not own buffers.
    pub fn validate(
        self,
        geometry: SpartanShiftGeometry,
        expected_device_registry_id: u64,
    ) -> Result<Self, SpartanShiftPlanError> {
        if self.rows != geometry.rows {
            return Err(SpartanShiftPlanError::WrongLength {
                name: "resident rows",
                expected: geometry.rows,
                actual: self.rows,
            });
        }
        if expected_device_registry_id == 0 || self.device_registry_id == 0 {
            return Err(SpartanShiftPlanError::MissingDeviceRegistryIdentity);
        }
        if self.device_registry_id != expected_device_registry_id {
            return Err(SpartanShiftPlanError::DeviceRegistryMismatch {
                expected: expected_device_registry_id,
                actual: self.device_registry_id,
            });
        }
        let buffers = [self.unexpanded_pc, self.pc, self.flags];
        if buffers.iter().any(|buffer| buffer.allocation_identity == 0) {
            return Err(SpartanShiftPlanError::MissingAllocationIdentity);
        }
        if buffers[0].allocation_identity == buffers[1].allocation_identity
            || buffers[0].allocation_identity == buffers[2].allocation_identity
            || buffers[1].allocation_identity == buffers[2].allocation_identity
        {
            return Err(SpartanShiftPlanError::DuplicateAllocationIdentity);
        }
        let value_bytes = checked_bytes("resident PC bytes", geometry.rows, size_of::<u64>())?;
        let flag_bytes = checked_bytes(
            "resident flag bytes",
            geometry.flag_words,
            size_of::<SpartanShiftFlagWord>(),
        )?;
        for (name, buffer, expected) in [
            (
                "resident unexpanded PC bytes",
                self.unexpanded_pc,
                value_bytes,
            ),
            ("resident PC bytes", self.pc, value_bytes),
            ("resident flag bytes", self.flags, flag_bytes),
        ] {
            if buffer.byte_len != expected {
                return Err(SpartanShiftPlanError::WrongLength {
                    name,
                    expected,
                    actual: buffer.byte_len,
                });
            }
        }
        if !self.exact_current_flags {
            return Err(SpartanShiftPlanError::UncertifiedCurrentFlags);
        }
        Ok(self)
    }
}

/// Exact output and work counts for the disjoint 32-row producer partition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpartanShiftProducerPlan {
    pub row_extractions: usize,
    pub flag_chunks: usize,
    pub value_bytes_written: usize,
    pub flag_bytes_written: usize,
    pub total_bytes_written: usize,
}

impl SpartanShiftProducerPlan {
    pub fn new(geometry: SpartanShiftGeometry) -> Result<Self, SpartanShiftPlanError> {
        let value_bytes_written = checked_bytes("producer PC bytes", 2 * geometry.rows, 8)?;
        let flag_bytes_written = checked_bytes(
            "producer flag bytes",
            geometry.flag_words,
            size_of::<SpartanShiftFlagWord>(),
        )?;
        Ok(Self {
            row_extractions: geometry.rows,
            flag_chunks: geometry.flag_words,
            value_bytes_written,
            flag_bytes_written,
            total_bytes_written: value_bytes_written
                .checked_add(flag_bytes_written)
                .ok_or(SpartanShiftPlanError::SizeOverflow)?,
        })
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub struct SpartanShiftNativePlanes<'a> {
    geometry: SpartanShiftGeometry,
    unexpanded_pc: &'a [u64],
    pc: &'a [u64],
    flags: &'a [SpartanShiftFlagWord],
}

#[cfg(test)]
impl<'a> SpartanShiftNativePlanes<'a> {
    pub fn new(
        geometry: SpartanShiftGeometry,
        unexpanded_pc: &'a [u64],
        pc: &'a [u64],
        flags: &'a [SpartanShiftFlagWord],
    ) -> Result<Self, SpartanShiftPlanError> {
        for (name, actual) in [("unexpanded PC", unexpanded_pc.len()), ("PC", pc.len())] {
            if actual != geometry.rows {
                return Err(SpartanShiftPlanError::WrongLength {
                    name,
                    expected: geometry.rows,
                    actual,
                });
            }
        }
        if flags.len() != geometry.flag_words {
            return Err(SpartanShiftPlanError::WrongLength {
                name: "flag words",
                expected: geometry.flag_words,
                actual: flags.len(),
            });
        }
        Ok(Self {
            geometry,
            unexpanded_pc,
            pc,
            flags,
        })
    }

    fn row(self, index: usize) -> SpartanShiftNativeRow {
        let word = self.flags[index / SPARTAN_SHIFT_FLAG_ROWS_PER_WORD];
        let bit = 1u32 << (index % SPARTAN_SHIFT_FLAG_ROWS_PER_WORD);
        SpartanShiftNativeRow {
            unexpanded_pc: self.unexpanded_pc[index],
            pc: self.pc[index],
            is_virtual: word.is_virtual & bit != 0,
            is_first_in_sequence: word.is_first_in_sequence & bit != 0,
            is_noop: word.is_noop & bit != 0,
        }
    }

    copy_field_getters! { pub, { geometry: SpartanShiftGeometry }}
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpartanShiftNativeRow {
    pub unexpanded_pc: u64,
    pub pc: u64,
    pub is_virtual: bool,
    pub is_first_in_sequence: bool,
    pub is_noop: bool,
}

#[cfg(test)]
pub(crate) fn pack_flag_words(
    geometry: SpartanShiftGeometry,
    is_virtual: &[bool],
    is_first_in_sequence: &[bool],
    is_noop: &[bool],
) -> Result<Vec<SpartanShiftFlagWord>, SpartanShiftPlanError> {
    for (name, actual) in [
        ("is_virtual", is_virtual.len()),
        ("is_first_in_sequence", is_first_in_sequence.len()),
        ("is_noop", is_noop.len()),
    ] {
        if actual != geometry.rows {
            return Err(SpartanShiftPlanError::WrongLength {
                name,
                expected: geometry.rows,
                actual,
            });
        }
    }
    (0..geometry.flag_words)
        .map(|word| {
            let start = word * SPARTAN_SHIFT_FLAG_ROWS_PER_WORD;
            let end = (start + SPARTAN_SHIFT_FLAG_ROWS_PER_WORD).min(geometry.rows);
            pack_flag_word(
                &is_virtual[start..end],
                &is_first_in_sequence[start..end],
                &is_noop[start..end],
            )
        })
        .collect()
}

/// Packs one independently owned chunk of at most 32 rows.
///
/// A parallel producer assigns each chunk to one worker, which avoids atomic
/// updates to `SpartanShiftFlagWord` while the two value planes are filled.
#[cfg(test)]
fn pack_flag_word(
    is_virtual: &[bool],
    is_first_in_sequence: &[bool],
    is_noop: &[bool],
) -> Result<SpartanShiftFlagWord, SpartanShiftPlanError> {
    let rows = is_virtual.len();
    if rows == 0
        || rows > SPARTAN_SHIFT_FLAG_ROWS_PER_WORD
        || is_first_in_sequence.len() != rows
        || is_noop.len() != rows
    {
        return Err(SpartanShiftPlanError::InvalidFlagChunkLength {
            is_virtual: rows,
            is_first_in_sequence: is_first_in_sequence.len(),
            is_noop: is_noop.len(),
        });
    }
    let mut word = SpartanShiftFlagWord::default();
    for row in 0..rows {
        let bit = 1u32 << row;
        word.is_virtual |= u32::from(is_virtual[row]) * bit;
        word.is_first_in_sequence |= u32::from(is_first_in_sequence[row]) * bit;
        word.is_noop |= u32::from(is_noop[row]) * bit;
    }
    Ok(word)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpartanShiftPrefixTables<F> {
    pub p: [Vec<F>; SPARTAN_SHIFT_PREFIX_PAIRS],
    pub q: [Vec<F>; SPARTAN_SHIFT_PREFIX_PAIRS],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpartanShiftDenseState<F> {
    pub eq_plus_one_outer: Vec<F>,
    pub eq_plus_one_product: Vec<F>,
    pub unexpanded_pc: Vec<F>,
    pub pc: Vec<F>,
    pub is_virtual: Vec<F>,
    pub is_first_in_sequence: Vec<F>,
    pub is_noop: Vec<F>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpartanShiftOutputs<F> {
    pub unexpanded_pc: F,
    pub pc: F,
    pub is_virtual: F,
    pub is_first_in_sequence: F,
    pub is_noop: F,
}

/// Mixed shader multiplier order: `gamma`, `gamma^2`, `gamma^3`.
pub fn mixed_gamma_multipliers<F: Field>(gamma: F) -> [F; 3] {
    let powers = gamma_powers(gamma);
    [powers[1], powers[2], powers[3]]
}

/// Mixed shader weight order: `eq(r_outer_hi)` followed by
/// `gamma^4 * eq(r_product_hi)`.
pub fn mixed_high_weights<F: Field>(
    geometry: SpartanShiftGeometry,
    r_outer: &[F],
    r_product: &[F],
    gamma: F,
) -> Result<Vec<F>, SpartanShiftOracleError> {
    let (r_outer_hi, _) = split_point(geometry, r_outer)?;
    let (r_product_hi, _) = split_point(geometry, r_product)?;
    let mut weights = EqPolynomial::<F>::evals(r_outer_hi, None);
    let gamma_four = gamma_powers(gamma)[4];
    weights.extend(
        EqPolynomial::<F>::evals(r_product_hi, None)
            .into_iter()
            .map(|weight| gamma_four * weight),
    );
    Ok(weights)
}

#[expect(
    clippy::needless_range_loop,
    reason = "the low index addresses all four output tables"
)]
#[cfg(test)]
pub(crate) fn build_prefix_reference<F: Field>(
    geometry: SpartanShiftGeometry,
    planes: SpartanShiftNativePlanes<'_>,
    r_outer: &[F],
    r_product: &[F],
    gamma: F,
) -> Result<SpartanShiftPrefixTables<F>, SpartanShiftOracleError> {
    validate_oracle_inputs(geometry, planes, r_outer, r_product)?;
    let outer = EqPlusOnePrefixSuffix::new(r_outer);
    let product = EqPlusOnePrefixSuffix::new(r_product);
    let gamma_powers = gamma_powers(gamma);
    let mut q: [Vec<F>; SPARTAN_SHIFT_PREFIX_PAIRS] =
        std::array::from_fn(|_| vec![F::zero(); geometry.prefix_elements]);

    for x_hi in 0..geometry.suffix_elements {
        for x_lo in 0..geometry.prefix_elements {
            let row = planes.row(geometry.row_index(x_hi, x_lo)?);
            let outer_value = outer_value(row, gamma_powers);
            let product_value = product_value(row, gamma_powers[4]);
            q[0][x_lo] += outer.suffix_0[x_hi] * outer_value;
            q[1][x_lo] += outer.suffix_1[x_hi] * outer_value;
            q[2][x_lo] += product.suffix_0[x_hi] * product_value;
            q[3][x_lo] += product.suffix_1[x_hi] * product_value;
        }
    }

    Ok(SpartanShiftPrefixTables {
        p: [
            outer.prefix_0,
            outer.prefix_1,
            product.prefix_0,
            product.prefix_1,
        ],
        q,
    })
}

fn prefix_round_endpoints<F: Field>(
    tables: &SpartanShiftPrefixTables<F>,
) -> Result<[F; 2], SpartanShiftOracleError> {
    let length = tables.p[0].len();
    if length < 2 || !length.is_power_of_two() {
        return Err(SpartanShiftOracleError::InvalidRoundLength(length));
    }
    for table in tables.p.iter().chain(tables.q.iter()) {
        if table.len() != length {
            return Err(SpartanShiftOracleError::WrongTableLength {
                name: "prefix P/Q",
                expected: length,
                actual: table.len(),
            });
        }
    }
    let mut endpoints = [F::Accumulator::default(); 2];
    for pair in 0..SPARTAN_SHIFT_PREFIX_PAIRS {
        for y in 0..length / 2 {
            let p0 = tables.p[pair][2 * y];
            let p1 = tables.p[pair][2 * y + 1];
            let q0 = tables.q[pair][2 * y];
            let q1 = tables.q[pair][2 * y + 1];
            endpoints[0].fmadd(p0, q0);
            endpoints[1].fmadd(p1 + p1 - p0, q1 + q1 - q0);
        }
    }
    Ok(endpoints.map(F::Accumulator::reduce))
}

pub fn prefix_round<F: Field>(
    previous_claim: F,
    tables: &SpartanShiftPrefixTables<F>,
) -> Result<UnivariatePoly<F>, SpartanShiftOracleError> {
    let endpoints = prefix_round_endpoints(tables)?;
    Ok(UnivariatePoly::from_evals_and_hint(
        previous_claim,
        &endpoints,
    ))
}

pub fn bind_prefix_tables<F: Field>(
    tables: &mut SpartanShiftPrefixTables<F>,
    challenge: F,
) -> Result<(), SpartanShiftOracleError> {
    for table in tables.p.iter_mut().chain(tables.q.iter_mut()) {
        bind_table(table, challenge)?;
    }
    Ok(())
}

#[cfg(test)]
fn fold_native_prefix<F: Field>(
    geometry: SpartanShiftGeometry,
    planes: SpartanShiftNativePlanes<'_>,
    prefix_challenges: &[F],
) -> Result<SpartanShiftOutputs<Vec<F>>, SpartanShiftOracleError> {
    if planes.geometry != geometry {
        return Err(SpartanShiftOracleError::GeometryMismatch);
    }
    let weights = prefix_fold_weights(geometry, prefix_challenges)?;
    let mut outputs = SpartanShiftOutputs {
        unexpanded_pc: vec![F::zero(); geometry.suffix_elements],
        pc: vec![F::zero(); geometry.suffix_elements],
        is_virtual: vec![F::zero(); geometry.suffix_elements],
        is_first_in_sequence: vec![F::zero(); geometry.suffix_elements],
        is_noop: vec![F::zero(); geometry.suffix_elements],
    };
    for x_hi in 0..geometry.suffix_elements {
        for (x_lo, &weight) in weights.iter().enumerate() {
            let row = planes.row(geometry.row_index(x_hi, x_lo)?);
            outputs.unexpanded_pc[x_hi] += weight * F::from_u64(row.unexpanded_pc);
            outputs.pc[x_hi] += weight * F::from_u64(row.pc);
            if row.is_virtual {
                outputs.is_virtual[x_hi] += weight;
            }
            if row.is_first_in_sequence {
                outputs.is_first_in_sequence[x_hi] += weight;
            }
            if row.is_noop {
                outputs.is_noop[x_hi] += weight;
            }
        }
    }
    Ok(outputs)
}

pub fn build_dense_state<F: Field>(
    geometry: SpartanShiftGeometry,
    outputs: SpartanShiftOutputs<Vec<F>>,
    r_outer: &[F],
    r_product: &[F],
    prefix_challenges: &[F],
) -> Result<SpartanShiftDenseState<F>, SpartanShiftOracleError> {
    for (name, actual) in [
        ("unexpanded PC dense", outputs.unexpanded_pc.len()),
        ("PC dense", outputs.pc.len()),
        ("virtual dense", outputs.is_virtual.len()),
        ("first dense", outputs.is_first_in_sequence.len()),
        ("noop dense", outputs.is_noop.len()),
    ] {
        if actual != geometry.suffix_elements {
            return Err(SpartanShiftOracleError::WrongTableLength {
                name,
                expected: geometry.suffix_elements,
                actual,
            });
        }
    }
    Ok(SpartanShiftDenseState {
        eq_plus_one_outer: partially_bound_eq_plus_one(geometry, r_outer, prefix_challenges)?,
        eq_plus_one_product: partially_bound_eq_plus_one(geometry, r_product, prefix_challenges)?,
        unexpanded_pc: outputs.unexpanded_pc,
        pc: outputs.pc,
        is_virtual: outputs.is_virtual,
        is_first_in_sequence: outputs.is_first_in_sequence,
        is_noop: outputs.is_noop,
    })
}

fn dense_round_endpoints<F: Field>(
    state: &SpartanShiftDenseState<F>,
    gamma: F,
) -> Result<[F; 2], SpartanShiftOracleError> {
    let length = state.eq_plus_one_outer.len();
    if length < 2 || !length.is_power_of_two() {
        return Err(SpartanShiftOracleError::InvalidRoundLength(length));
    }
    validate_dense_lengths(state, length)?;
    let gamma_powers = gamma_powers(gamma);
    let mut endpoints = [F::Accumulator::default(); 2];
    for y in 0..length / 2 {
        for (node, t) in [F::zero(), F::from_u64(2)].into_iter().enumerate() {
            let eq_outer = extend_pair(&state.eq_plus_one_outer, y, t);
            let eq_product = extend_pair(&state.eq_plus_one_product, y, t);
            let unexpanded_pc = extend_pair(&state.unexpanded_pc, y, t);
            let pc = extend_pair(&state.pc, y, t);
            let is_virtual = extend_pair(&state.is_virtual, y, t);
            let is_first = extend_pair(&state.is_first_in_sequence, y, t);
            let is_noop = extend_pair(&state.is_noop, y, t);
            endpoints[node].fmadd(
                eq_outer,
                unexpanded_pc
                    + gamma_powers[1] * pc
                    + gamma_powers[2] * is_virtual
                    + gamma_powers[3] * is_first,
            );
            endpoints[node].fmadd(eq_product, gamma_powers[4] * (F::one() - is_noop));
        }
    }
    Ok(endpoints.map(F::Accumulator::reduce))
}

pub fn dense_round<F: Field>(
    previous_claim: F,
    state: &SpartanShiftDenseState<F>,
    gamma: F,
) -> Result<UnivariatePoly<F>, SpartanShiftOracleError> {
    let endpoints = dense_round_endpoints(state, gamma)?;
    Ok(UnivariatePoly::from_evals_and_hint(
        previous_claim,
        &endpoints,
    ))
}

pub fn bind_dense_state<F: Field>(
    state: &mut SpartanShiftDenseState<F>,
    challenge: F,
) -> Result<(), SpartanShiftOracleError> {
    for table in [
        &mut state.eq_plus_one_outer,
        &mut state.eq_plus_one_product,
        &mut state.unexpanded_pc,
        &mut state.pc,
        &mut state.is_virtual,
        &mut state.is_first_in_sequence,
        &mut state.is_noop,
    ] {
        bind_table(table, challenge)?;
    }
    Ok(())
}

pub fn final_outputs<F: Field>(
    state: &SpartanShiftDenseState<F>,
) -> Result<SpartanShiftOutputs<F>, SpartanShiftOracleError> {
    validate_dense_lengths(state, 1)?;
    Ok(SpartanShiftOutputs {
        unexpanded_pc: state.unexpanded_pc[0],
        pc: state.pc[0],
        is_virtual: state.is_virtual[0],
        is_first_in_sequence: state.is_first_in_sequence[0],
        is_noop: state.is_noop[0],
    })
}

#[cfg(test)]
fn validate_oracle_inputs<F: Field>(
    geometry: SpartanShiftGeometry,
    planes: SpartanShiftNativePlanes<'_>,
    r_outer: &[F],
    r_product: &[F],
) -> Result<(), SpartanShiftOracleError> {
    if planes.geometry != geometry {
        return Err(SpartanShiftOracleError::GeometryMismatch);
    }
    for (name, actual) in [("r_outer", r_outer.len()), ("r_product", r_product.len())] {
        if actual != geometry.log_t {
            return Err(SpartanShiftOracleError::WrongPointLength {
                name,
                expected: geometry.log_t,
                actual,
            });
        }
    }
    Ok(())
}

fn split_point<F: Field>(
    geometry: SpartanShiftGeometry,
    point: &[F],
) -> Result<(&[F], &[F]), SpartanShiftOracleError> {
    if point.len() != geometry.log_t {
        return Err(SpartanShiftOracleError::WrongPointLength {
            name: "eq+1 point",
            expected: geometry.log_t,
            actual: point.len(),
        });
    }
    Ok(point.split_at(geometry.suffix_vars))
}

pub fn prefix_fold_weights<F: Field>(
    geometry: SpartanShiftGeometry,
    challenges: &[F],
) -> Result<Vec<F>, SpartanShiftOracleError> {
    if challenges.len() != geometry.prefix_vars {
        return Err(SpartanShiftOracleError::WrongChallengeCount {
            expected: geometry.prefix_vars,
            actual: challenges.len(),
        });
    }
    let point = challenges.iter().rev().copied().collect::<Vec<_>>();
    Ok(EqPolynomial::<F>::evals(&point, None))
}

fn partially_bound_eq_plus_one<F: Field>(
    geometry: SpartanShiftGeometry,
    point: &[F],
    prefix_challenges: &[F],
) -> Result<Vec<F>, SpartanShiftOracleError> {
    let _ = split_point(geometry, point)?;
    let prefix_point = prefix_challenges.iter().rev().copied().collect::<Vec<_>>();
    if prefix_point.len() != geometry.prefix_vars {
        return Err(SpartanShiftOracleError::WrongChallengeCount {
            expected: geometry.prefix_vars,
            actual: prefix_point.len(),
        });
    }
    let split = EqPlusOnePrefixSuffix::new(point);
    let p0 = Polynomial::new(split.prefix_0).evaluate(&prefix_point);
    let p1 = Polynomial::new(split.prefix_1).evaluate(&prefix_point);
    Ok(split
        .suffix_0
        .into_iter()
        .zip(split.suffix_1)
        .map(|(s0, s1)| p0 * s0 + p1 * s1)
        .collect())
}

#[cfg(test)]
fn outer_value<F: Field>(row: SpartanShiftNativeRow, gamma_powers: [F; 5]) -> F {
    let mut value = F::from_u64(row.unexpanded_pc) + gamma_powers[1] * F::from_u64(row.pc);
    if row.is_virtual {
        value += gamma_powers[2];
    }
    if row.is_first_in_sequence {
        value += gamma_powers[3];
    }
    value
}

#[cfg(test)]
fn product_value<F: Field>(row: SpartanShiftNativeRow, gamma_four: F) -> F {
    if row.is_noop {
        F::zero()
    } else {
        gamma_four
    }
}

fn gamma_powers<F: Field>(gamma: F) -> [F; 5] {
    let mut powers = [F::one(); 5];
    for index in 1..5 {
        powers[index] = powers[index - 1] * gamma;
    }
    powers
}

fn extend_pair<F: Field>(table: &[F], pair: usize, t: F) -> F {
    let low = table[2 * pair];
    low + t * (table[2 * pair + 1] - low)
}

fn bind_table<F: Field>(table: &mut Vec<F>, challenge: F) -> Result<(), SpartanShiftOracleError> {
    if table.len() < 2 || !table.len().is_power_of_two() {
        return Err(SpartanShiftOracleError::InvalidRoundLength(table.len()));
    }
    let half = table.len() / 2;
    for y in 0..half {
        let low = table[2 * y];
        table[y] = low + challenge * (table[2 * y + 1] - low);
    }
    table.truncate(half);
    Ok(())
}

fn validate_dense_lengths<F: Field>(
    state: &SpartanShiftDenseState<F>,
    expected: usize,
) -> Result<(), SpartanShiftOracleError> {
    for (name, actual) in [
        ("eq+1 outer", state.eq_plus_one_outer.len()),
        ("eq+1 product", state.eq_plus_one_product.len()),
        ("unexpanded PC", state.unexpanded_pc.len()),
        ("PC", state.pc.len()),
        ("virtual", state.is_virtual.len()),
        ("first", state.is_first_in_sequence.len()),
        ("noop", state.is_noop.len()),
    ] {
        if actual != expected {
            return Err(SpartanShiftOracleError::WrongTableLength {
                name,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum SpartanShiftPlanError {
    #[error("Spartan shift needs a power-of-two row count of at least two, got {0}")]
    InvalidRows(usize),
    #[error("Spartan shift {phase} width {width} is not a nonzero multiple of 32 at most 1024")]
    InvalidThreadgroupWidth { phase: &'static str, width: usize },
    #[error("Spartan shift high tile must be a nonzero power of two, got {0}")]
    InvalidHighTile(usize),
    #[error(
        "Spartan shift suffix length {suffix_elements} is not divisible by high tile {high_tile_elements}"
    )]
    NonIntegralHighTiles {
        suffix_elements: usize,
        high_tile_elements: usize,
    },
    #[error(
        "Spartan shift coordinate ({x_hi}, {x_lo}) exceeds ({suffix_elements}, {prefix_elements})"
    )]
    CoordinateOutOfRange {
        x_hi: usize,
        x_lo: usize,
        suffix_elements: usize,
        prefix_elements: usize,
    },
    #[error("Spartan shift {name} length is {actual}, expected {expected}")]
    WrongLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error(
        "Spartan shift flag chunk lengths are invalid: virtual={is_virtual}, first={is_first_in_sequence}, noop={is_noop}"
    )]
    InvalidFlagChunkLength {
        is_virtual: usize,
        is_first_in_sequence: usize,
        is_noop: usize,
    },
    #[error("Spartan shift resident source has a missing device registry identity")]
    MissingDeviceRegistryIdentity,
    #[error("Spartan shift resident source is on device {actual}, expected {expected}")]
    DeviceRegistryMismatch { expected: u64, actual: u64 },
    #[error("Spartan shift resident source has a missing allocation identity")]
    MissingAllocationIdentity,
    #[error("Spartan shift resident source aliases two required allocations")]
    DuplicateAllocationIdentity,
    #[error("Spartan shift resident source does not certify exact current flags")]
    UncertifiedCurrentFlags,
    #[error("Spartan shift {name} exceeds the shader's 32-bit index space")]
    ShaderIndexOverflow { name: &'static str },
    #[error("Spartan shift size arithmetic overflowed")]
    SizeOverflow,
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum SpartanShiftOracleError {
    #[error(transparent)]
    Plan(#[from] SpartanShiftPlanError),
    #[error("Spartan shift native planes use a different geometry")]
    GeometryMismatch,
    #[error("Spartan shift {name} point has {actual} coordinates, expected {expected}")]
    WrongPointLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("Spartan shift has {actual} prefix challenges, expected {expected}")]
    WrongChallengeCount { expected: usize, actual: usize },
    #[error("Spartan shift {name} table has length {actual}, expected {expected}")]
    WrongTableLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("Spartan shift round table has invalid length {0}")]
    InvalidRoundLength(usize),
}

fn checked_power_of_two(
    name: &'static str,
    exponent: usize,
) -> Result<usize, SpartanShiftPlanError> {
    1usize
        .checked_shl(exponent as u32)
        .ok_or(SpartanShiftPlanError::ShaderIndexOverflow { name })
}

fn abi_count(name: &'static str, value: usize) -> Result<u32, SpartanShiftPlanError> {
    u32::try_from(value).map_err(|_| SpartanShiftPlanError::ShaderIndexOverflow { name })
}

fn checked_bytes(
    name: &'static str,
    elements: usize,
    element_bytes: usize,
) -> Result<usize, SpartanShiftPlanError> {
    checked_product(name, elements, element_bytes)
}

fn checked_product(
    _name: &'static str,
    lhs: usize,
    rhs: usize,
) -> Result<usize, SpartanShiftPlanError> {
    lhs.checked_mul(rhs)
        .ok_or(SpartanShiftPlanError::SizeOverflow)
}

fn checked_sum(_name: &'static str, values: &[usize]) -> Result<usize, SpartanShiftPlanError> {
    values.iter().try_fold(0usize, |sum, &value| {
        sum.checked_add(value)
            .ok_or(SpartanShiftPlanError::SizeOverflow)
    })
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_field::AkitaField;

    use super::*;

    #[test]
    fn metal_entry_points_compile() {
        let Ok(context) = super::super::SolinasMetal::for_akita() else {
            return;
        };
        for name in [
            BUILD_MIXED_PIPELINE,
            REDUCE_PREFIX_PIPELINE,
            FOLD_NATIVE_PIPELINE,
        ] {
            let pipeline = context.compile_named_pipeline(name).unwrap();
            let limits = super::super::SolinasMetal::limits(&pipeline);
            assert_eq!(limits.thread_execution_width, SPARTAN_SHIFT_SIMD_WIDTH);
            assert!(limits.max_total_threads_per_threadgroup >= 128);
        }
    }

    #[test]
    fn resident_upload_is_not_coupled_to_default_dispatch_geometry() {
        let Ok(context) = super::super::SolinasMetal::for_akita() else {
            return;
        };
        let rows = context
            .prepare_spartan_shift_rows(&[3, 5], &[7, 11], &[SpartanShiftFlagWord::default()], true)
            .unwrap();
        assert_eq!(rows.len(), 2);
    }

    fn point(len: usize, seed: u64) -> Vec<AkitaField> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                AkitaField::from_u64(state)
            })
            .collect()
    }

    fn resident_metadata(geometry: SpartanShiftGeometry) -> ResidentSpartanShiftMetadata {
        ResidentSpartanShiftMetadata {
            rows: geometry.rows,
            unexpanded_pc: ResidentSpartanShiftBufferMetadata {
                allocation_identity: 11,
                byte_len: geometry.rows * size_of::<u64>(),
            },
            pc: ResidentSpartanShiftBufferMetadata {
                allocation_identity: 13,
                byte_len: geometry.rows * size_of::<u64>(),
            },
            flags: ResidentSpartanShiftBufferMetadata {
                allocation_identity: 17,
                byte_len: geometry.flag_words * size_of::<SpartanShiftFlagWord>(),
            },
            device_registry_id: 19,
            exact_current_flags: true,
        }
    }

    #[test]
    fn target_plan_prices_packed_halos() {
        let geometry = SpartanShiftGeometry::target();
        let config = SpartanShiftKernelConfig::default();
        let plan = SpartanShiftPlan::new(geometry.rows, config).unwrap();

        assert_eq!(plan.cost.halo_rows, 516_096);
        assert_eq!(plan.cost.halo_flag_words, 16_128);
        assert_eq!(plan.cost.build_halo_value_bytes, 8_257_536);
        assert_eq!(plan.cost.build_halo_flag_bytes, 193_536);
        assert_eq!(plan.cost.build_unique_bytes, 1_166_802_944);
        assert_eq!(plan.cost.build_coalesced_bytes_with_halo, 1_175_254_016);

        let producer = SpartanShiftProducerPlan::new(geometry).unwrap();
        assert_eq!(producer.row_extractions, 67_108_864);
        assert_eq!(producer.flag_chunks, 2_097_152);
        assert_eq!(producer.value_bytes_written, 1_073_741_824);
        assert_eq!(producer.flag_bytes_written, 25_165_824);
        assert_eq!(producer.total_bytes_written, 1_098_907_648);
    }

    #[test]
    fn resident_metadata_checks_device_lengths_and_aliasing() {
        let geometry = SpartanShiftGeometry::new(1 << 10).unwrap();
        let metadata = resident_metadata(geometry);
        assert_eq!(metadata.validate(geometry, 19), Ok(metadata));

        assert_eq!(
            metadata.validate(geometry, 0),
            Err(SpartanShiftPlanError::MissingDeviceRegistryIdentity)
        );

        let mut missing_device = metadata;
        missing_device.device_registry_id = 0;
        assert_eq!(
            missing_device.validate(geometry, 19),
            Err(SpartanShiftPlanError::MissingDeviceRegistryIdentity)
        );

        assert_eq!(
            metadata.validate(geometry, 23),
            Err(SpartanShiftPlanError::DeviceRegistryMismatch {
                expected: 23,
                actual: 19,
            })
        );

        let mut wrong_length = metadata;
        wrong_length.flags.byte_len -= 1;
        assert!(matches!(
            wrong_length.validate(geometry, 19),
            Err(SpartanShiftPlanError::WrongLength {
                name: "resident flag bytes",
                ..
            })
        ));

        let mut aliased = metadata;
        aliased.flags.allocation_identity = aliased.pc.allocation_identity;
        assert_eq!(
            aliased.validate(geometry, 19),
            Err(SpartanShiftPlanError::DuplicateAllocationIdentity)
        );

        let mut missing_identity = metadata;
        missing_identity.unexpanded_pc.allocation_identity = 0;
        assert_eq!(
            missing_identity.validate(geometry, 19),
            Err(SpartanShiftPlanError::MissingAllocationIdentity)
        );

        let mut uncertified = metadata;
        uncertified.exact_current_flags = false;
        assert_eq!(
            uncertified.validate(geometry, 19),
            Err(SpartanShiftPlanError::UncertifiedCurrentFlags)
        );
    }

    #[test]
    fn flag_chunks_own_word_boundaries_and_clear_unused_bits() {
        let geometry = SpartanShiftGeometry::new(1 << 6).unwrap();
        let mut is_virtual = vec![false; geometry.rows];
        let mut is_first = vec![false; geometry.rows];
        let mut is_noop = vec![false; geometry.rows];
        is_virtual[31] = true;
        is_first[32] = true;
        is_noop[63] = true;

        let words = pack_flag_words(geometry, &is_virtual, &is_first, &is_noop).unwrap();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].is_virtual, 1u32 << 31);
        assert_eq!(words[0].is_first_in_sequence, 0);
        assert_eq!(words[1].is_first_in_sequence, 1);
        assert_eq!(words[1].is_noop, 1u32 << 31);

        let tail = pack_flag_word(&[true, false, true], &[false; 3], &[false; 3]).unwrap();
        assert_eq!(tail.is_virtual, 0b101);
        assert_eq!(tail.is_first_in_sequence, 0);
        assert_eq!(tail.is_noop, 0);
    }

    #[test]
    fn metal_runtime_matches_prefix_and_fold_oracles() {
        let Ok(context) = super::super::SolinasMetal::for_akita() else {
            return;
        };
        let geometry = SpartanShiftGeometry::new(1 << 16).unwrap();
        let mut unexpanded_pc = vec![0u64; geometry.rows];
        let mut pc = vec![0u64; geometry.rows];
        let mut is_virtual = vec![false; geometry.rows];
        let mut is_first = vec![false; geometry.rows];
        let mut is_noop = vec![false; geometry.rows];
        for row in 0..geometry.rows {
            unexpanded_pc[row] = (row as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .rotate_left((row & 63) as u32);
            pc[row] = u64::MAX.wrapping_sub(
                (row as u64)
                    .wrapping_mul(0xD134_2543_DE82_EF95)
                    .rotate_right((row & 31) as u32),
            );
            is_virtual[row] = row % 5 == 1;
            is_first[row] = row % 17 == 3;
            is_noop[row] = row % 7 == 0;
        }
        let flags = pack_flag_words(geometry, &is_virtual, &is_first, &is_noop).unwrap();
        let planes = SpartanShiftNativePlanes::new(geometry, &unexpanded_pc, &pc, &flags).unwrap();
        let rows = context
            .prepare_spartan_shift_rows(&unexpanded_pc, &pc, &flags, true)
            .unwrap();
        let source_allocations = rows.allocation_identities();
        let r_outer = point(geometry.log_t, 0xA11C_E001);
        let r_product = point(geometry.log_t, 0xB22D_F002);
        let gamma = AkitaField::from_u64(0xC33E_1003);
        let expected_prefix =
            build_prefix_reference(geometry, planes, &r_outer, &r_product, gamma).unwrap();

        let invocation = context
            .prepare_spartan_shift_prefix(
                &rows,
                &r_outer,
                &r_product,
                gamma,
                SpartanShiftKernelConfig::default(),
            )
            .unwrap();
        assert_eq!(
            invocation.source_allocation_identities(),
            source_allocations
        );
        assert_eq!(invocation.execute_device_buffer_allocations(), 0);
        let pending = invocation.submit().unwrap();
        let (invocation, observation) = pending.join().unwrap();
        assert_eq!(
            invocation.source_allocation_identities(),
            source_allocations
        );
        assert_eq!(observation.q, expected_prefix.q);
        assert!(!observation.gpu_active.is_zero());

        let prefix_challenges = point(geometry.prefix_vars, 0xD44F_2004);
        let expected_fold = fold_native_prefix(geometry, planes, &prefix_challenges).unwrap();
        let fold = context
            .prepare_spartan_shift_fold(
                &rows,
                &prefix_challenges,
                SpartanShiftKernelConfig::default(),
            )
            .unwrap();
        assert_eq!(fold.source_allocation_identities(), source_allocations);
        assert_eq!(fold.execute_device_buffer_allocations(), 0);
        let pending = fold.submit().unwrap();
        let (fold, observation) = pending.join().unwrap();
        assert_eq!(fold.source_allocation_identities(), source_allocations);
        assert_eq!(observation.outputs, expected_fold);
        assert!(!observation.gpu_active.is_zero());
    }
}
