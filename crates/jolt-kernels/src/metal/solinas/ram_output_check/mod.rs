//! Deferred-prefix RAM output-check plan, oracle, and Metal fold.

use std::mem::size_of;

use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use thiserror::Error;

mod runtime;

pub use runtime::{RamOutputCheckFold, ResidentRamFinalValues};

pub(super) const SOURCE: &str = include_str!("shader.metal");

pub const RAM_OUTPUT_CHECK_FOLD_PIPELINE: &str = "solinas_ram_output_check_fold_partials";
pub const RAM_OUTPUT_CHECK_REDUCE_PIPELINE: &str = "solinas_ram_output_check_fold_reduce";
pub const RAM_OUTPUT_CHECK_TARGET_LOG_K: usize = 13;
pub const RAM_OUTPUT_CHECK_TARGET_ADDRESSES: usize = 1 << RAM_OUTPUT_CHECK_TARGET_LOG_K;
pub const RAM_OUTPUT_CHECK_TARGET_MASK_START: usize = 1 << 10;
pub const RAM_OUTPUT_CHECK_TARGET_MASK_END: usize = 1 << 12;
pub const RAM_OUTPUT_CHECK_FIELD_BYTES: usize = 16;
pub const RAM_OUTPUT_CHECK_SOURCE_BYTES: usize = 8;
pub const RAM_OUTPUT_CHECK_SIMD_WIDTH: usize = 32;
pub const RAM_OUTPUT_CHECK_TARGET_CPU_NS: u64 = 1_734_583;
pub const RAM_OUTPUT_CHECK_FIVE_X_CAP_NS: u64 = 346_916;
pub const RAM_OUTPUT_CHECK_RELATION_CPU_NS: u64 = 1_454_541;
pub const RAM_OUTPUT_CHECK_RELATION_FIVE_X_CAP_NS: u64 = 290_908;
pub const RAM_OUTPUT_CHECK_COMPONENT_GATE_NS: u64 = 335_000;

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum RamOutputCheckPlanError {
    #[error("RAM output check needs a nonzero power-of-two address count, got {0}")]
    InvalidAddresses(usize),
    #[error("RAM output mask [{start}, {end}) is invalid for {addresses} addresses")]
    InvalidMask {
        addresses: usize,
        start: usize,
        end: usize,
    },
    #[error("RAM output fold width {width} must be a power-of-two multiple of 32")]
    InvalidThreadgroupWidth { width: usize },
    #[error("RAM output fold block {block_elements} is not divisible by width {width}")]
    NonIntegralChunks { block_elements: usize, width: usize },
    #[error("RAM output {name} length is {got}, expected {expected}")]
    Length {
        name: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("RAM output segment [{start}, {end}) exceeds {addresses} addresses")]
    SegmentOutOfRange {
        addresses: usize,
        start: usize,
        end: usize,
    },
    #[error("RAM output prefix challenge count is {got}, expected {expected}")]
    ChallengeCount { expected: usize, got: usize },
    #[error("RAM output resident source has {got} bytes, expected {expected}")]
    ResidentBytes { expected: usize, got: usize },
    #[error("RAM output resident source does not certify public-I/O construction")]
    UncertifiedPublicIo,
    #[error("RAM output deferred round {round} was not the zero polynomial")]
    NonZeroDeferredRound { round: usize },
    #[error("RAM output round {round} claim check failed")]
    RoundClaim { round: usize },
    #[error("RAM output size arithmetic overflowed")]
    SizeOverflow,
    #[error("RAM output {name} exceeds the shader's 32-bit index space")]
    ShaderIndexOverflow { name: &'static str },
}

/// The single-command GPU prefix and serial CPU-tail geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamOutputCheckHybridPlan {
    addresses: usize,
    log_k: usize,
    mask_start: usize,
    mask_end: usize,
    zero_rounds: usize,
    block_elements: usize,
    tail_elements: usize,
    threads_per_threadgroup: usize,
    chunks_per_block: usize,
}

impl RamOutputCheckHybridPlan {
    pub fn new(
        addresses: usize,
        mask_start: usize,
        mask_end: usize,
        threads_per_threadgroup: usize,
    ) -> Result<Self, RamOutputCheckPlanError> {
        if addresses == 0 || !addresses.is_power_of_two() {
            return Err(RamOutputCheckPlanError::InvalidAddresses(addresses));
        }
        if mask_start >= mask_end || mask_end > addresses {
            return Err(RamOutputCheckPlanError::InvalidMask {
                addresses,
                start: mask_start,
                end: mask_end,
            });
        }
        if threads_per_threadgroup < RAM_OUTPUT_CHECK_SIMD_WIDTH
            || !threads_per_threadgroup.is_power_of_two()
            || !threads_per_threadgroup.is_multiple_of(RAM_OUTPUT_CHECK_SIMD_WIDTH)
        {
            return Err(RamOutputCheckPlanError::InvalidThreadgroupWidth {
                width: threads_per_threadgroup,
            });
        }

        let log_k = addresses.trailing_zeros() as usize;
        let zero_rounds =
            (mask_start.trailing_zeros().min(mask_end.trailing_zeros()) as usize).min(log_k);
        let block_elements = 1usize << zero_rounds;
        if !block_elements.is_multiple_of(threads_per_threadgroup) {
            return Err(RamOutputCheckPlanError::NonIntegralChunks {
                block_elements,
                width: threads_per_threadgroup,
            });
        }
        let tail_elements = addresses / block_elements;
        let chunks_per_block = block_elements / threads_per_threadgroup;

        Ok(Self {
            addresses,
            log_k,
            mask_start,
            mask_end,
            zero_rounds,
            block_elements,
            tail_elements,
            threads_per_threadgroup,
            chunks_per_block,
        })
    }

    pub const fn target() -> Self {
        Self {
            addresses: RAM_OUTPUT_CHECK_TARGET_ADDRESSES,
            log_k: RAM_OUTPUT_CHECK_TARGET_LOG_K,
            mask_start: RAM_OUTPUT_CHECK_TARGET_MASK_START,
            mask_end: RAM_OUTPUT_CHECK_TARGET_MASK_END,
            zero_rounds: 10,
            block_elements: 1024,
            tail_elements: 8,
            threads_per_threadgroup: 128,
            chunks_per_block: 8,
        }
    }

    pub const fn addresses(self) -> usize {
        self.addresses
    }

    pub const fn log_k(self) -> usize {
        self.log_k
    }

    pub const fn mask_start(self) -> usize {
        self.mask_start
    }

    pub const fn mask_end(self) -> usize {
        self.mask_end
    }

    pub const fn zero_rounds(self) -> usize {
        self.zero_rounds
    }

    pub const fn block_elements(self) -> usize {
        self.block_elements
    }

    pub const fn tail_elements(self) -> usize {
        self.tail_elements
    }

    pub const fn tail_rounds(self) -> usize {
        self.log_k - self.zero_rounds
    }

    pub const fn threads_per_threadgroup(self) -> usize {
        self.threads_per_threadgroup
    }

    pub const fn chunks_per_block(self) -> usize {
        self.chunks_per_block
    }

    pub const fn partial_count(self) -> usize {
        self.tail_elements * self.chunks_per_block
    }

    pub fn low_weights<F: Field>(
        self,
        challenges: &[F],
    ) -> Result<Vec<F>, RamOutputCheckPlanError> {
        if challenges.len() != self.zero_rounds {
            return Err(RamOutputCheckPlanError::ChallengeCount {
                expected: self.zero_rounds,
                got: challenges.len(),
            });
        }
        Ok(low_binding_weights(challenges))
    }

    pub fn shader_params(self) -> Result<RamOutputCheckFoldParams, RamOutputCheckPlanError> {
        Ok(RamOutputCheckFoldParams {
            block_elements: shader_count("block elements", self.block_elements)?,
            blocks: shader_count("blocks", self.tail_elements)?,
            chunks_per_block: shader_count("chunks per block", self.chunks_per_block)?,
            chunk_elements: shader_count("chunk elements", self.threads_per_threadgroup)?,
        })
    }

    pub fn storage(self) -> Result<RamOutputCheckStorage, RamOutputCheckPlanError> {
        let borrowed_input_bytes = source_bytes(self.addresses)?;
        let weight_bytes = field_bytes(self.block_elements)?;
        let partial_bytes = field_bytes(self.partial_count())?;
        let output_bytes = field_bytes(self.tail_elements)?;
        let private_bytes = weight_bytes
            .checked_add(partial_bytes)
            .and_then(|bytes| bytes.checked_add(output_bytes))
            .ok_or(RamOutputCheckPlanError::SizeOverflow)?;
        Ok(RamOutputCheckStorage {
            borrowed_input_bytes,
            weight_bytes,
            partial_bytes,
            output_bytes,
            private_bytes,
            resident_bytes: borrowed_input_bytes
                .checked_add(private_bytes)
                .ok_or(RamOutputCheckPlanError::SizeOverflow)?,
            maximum_buffer_bytes: borrowed_input_bytes
                .max(weight_bytes)
                .max(partial_bytes)
                .max(output_bytes),
        })
    }

    pub fn cost(self) -> Result<RamOutputCheckCost, RamOutputCheckPlanError> {
        let partials = self.partial_count();
        let partial_traffic_fields = partials
            .checked_mul(2)
            .ok_or(RamOutputCheckPlanError::SizeOverflow)?;
        let field_traffic = self
            .block_elements
            .checked_add(partial_traffic_fields)
            .and_then(|fields| fields.checked_add(self.tail_elements))
            .ok_or(RamOutputCheckPlanError::SizeOverflow)?;
        let issued_fields_without_weight_cache = self
            .addresses
            .checked_add(partial_traffic_fields)
            .and_then(|fields| fields.checked_add(self.tail_elements))
            .ok_or(RamOutputCheckPlanError::SizeOverflow)?;
        let simdgroups = partials
            .checked_mul(self.threads_per_threadgroup / RAM_OUTPUT_CHECK_SIMD_WIDTH)
            .ok_or(RamOutputCheckPlanError::SizeOverflow)?;
        Ok(RamOutputCheckCost {
            device_products: self.addresses,
            host_weight_products: self.block_elements - 1,
            command_buffers: 1,
            dispatches: 2,
            partial_threadgroups: partials,
            simdgroups,
            compulsory_bytes: source_bytes(self.addresses)?
                .checked_add(field_bytes(field_traffic)?)
                .ok_or(RamOutputCheckPlanError::SizeOverflow)?,
            issued_bytes_without_weight_cache: source_bytes(self.addresses)?
                .checked_add(field_bytes(issued_fields_without_weight_cache)?)
                .ok_or(RamOutputCheckPlanError::SizeOverflow)?,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamOutputCheckFoldParams {
    pub block_elements: u32,
    pub blocks: u32,
    pub chunks_per_block: u32,
    pub chunk_elements: u32,
}

const _: [(); 16] = [(); size_of::<RamOutputCheckFoldParams>()];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamOutputCheckStorage {
    pub borrowed_input_bytes: usize,
    pub weight_bytes: usize,
    pub partial_bytes: usize,
    pub output_bytes: usize,
    pub private_bytes: usize,
    pub resident_bytes: usize,
    pub maximum_buffer_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamOutputCheckCost {
    pub device_products: usize,
    pub host_weight_products: usize,
    pub command_buffers: usize,
    pub dispatches: usize,
    pub partial_threadgroups: usize,
    pub simdgroups: usize,
    pub compulsory_bytes: usize,
    pub issued_bytes_without_weight_cache: usize,
}

/// Metadata carried by the stage-2 producer alongside its native `u64`
/// allocation. The integration layer must additionally compare the actual
/// Metal buffer's storage identity with `allocation_identity`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentRamFinalMetadata {
    pub elements: usize,
    pub bytes: usize,
    pub device_registry_id: u64,
    pub allocation_identity: usize,
    pub public_io_certified: bool,
}

impl ResidentRamFinalMetadata {
    pub fn validate(self, plan: RamOutputCheckHybridPlan) -> Result<Self, RamOutputCheckPlanError> {
        if self.elements != plan.addresses {
            return Err(RamOutputCheckPlanError::Length {
                name: "resident native RamValFinal",
                expected: plan.addresses,
                got: self.elements,
            });
        }
        let expected = source_bytes(plan.addresses)?;
        if self.bytes != expected {
            return Err(RamOutputCheckPlanError::ResidentBytes {
                expected,
                got: self.bytes,
            });
        }
        if !self.public_io_certified {
            return Err(RamOutputCheckPlanError::UncertifiedPublicIo);
        }
        Ok(self)
    }
}

/// Builds weights for low-to-high binding. Challenge `j` selects address bit
/// `j`, so adding a challenge appends the new high half rather than interleaving
/// it with the existing low-bit table.
pub fn low_binding_weights<F: Field>(challenges: &[F]) -> Vec<F> {
    let mut weights = vec![F::one()];
    for &challenge in challenges {
        let old_len = weights.len();
        let mut next = vec![F::zero(); 2 * old_len];
        for (index, &weight) in weights.iter().enumerate() {
            let high = weight * challenge;
            next[index] = weight - high;
            next[index + old_len] = high;
        }
        weights = next;
    }
    weights
}

/// Independent host oracle for the shader's block dot products.
pub fn fold_low_prefix<F: Field>(
    values: &[F],
    weights: &[F],
) -> Result<Vec<F>, RamOutputCheckPlanError> {
    if weights.is_empty() || !weights.len().is_power_of_two() {
        return Err(RamOutputCheckPlanError::InvalidAddresses(weights.len()));
    }
    if !values.len().is_multiple_of(weights.len()) {
        return Err(RamOutputCheckPlanError::Length {
            name: "fold source",
            expected: values.len().next_multiple_of(weights.len()),
            got: values.len(),
        });
    }
    Ok(values
        .chunks_exact(weights.len())
        .map(|block| {
            block
                .iter()
                .zip(weights)
                .fold(F::zero(), |sum, (&value, &weight)| sum + value * weight)
        })
        .collect())
}

/// Matches the shader's native `u64` source without materializing field cells.
pub fn fold_u64_low_prefix<F: Field>(
    values: &[u64],
    weights: &[F],
) -> Result<Vec<F>, RamOutputCheckPlanError> {
    if weights.is_empty() || !weights.len().is_power_of_two() {
        return Err(RamOutputCheckPlanError::InvalidAddresses(weights.len()));
    }
    if !values.len().is_multiple_of(weights.len()) {
        return Err(RamOutputCheckPlanError::Length {
            name: "native fold source",
            expected: values.len().next_multiple_of(weights.len()),
            got: values.len(),
        });
    }
    Ok(values
        .chunks_exact(weights.len())
        .map(|block| {
            block
                .iter()
                .zip(weights)
                .fold(F::zero(), |sum, (&value, &weight)| {
                    sum + F::from_u64(value) * weight
                })
        })
        .collect())
}

pub fn folded_range_mask<F: Field>(
    plan: RamOutputCheckHybridPlan,
) -> Result<Vec<F>, RamOutputCheckPlanError> {
    if !plan.mask_start.is_multiple_of(plan.block_elements)
        || !plan.mask_end.is_multiple_of(plan.block_elements)
    {
        return Err(RamOutputCheckPlanError::InvalidMask {
            addresses: plan.addresses,
            start: plan.mask_start,
            end: plan.mask_end,
        });
    }
    Ok((0..plan.tail_elements)
        .map(|block| {
            let address = block * plan.block_elements;
            if address >= plan.mask_start && address < plan.mask_end {
                F::one()
            } else {
                F::zero()
            }
        })
        .collect())
}

#[derive(Clone, Copy, Debug)]
pub struct RamOutputPublicSegment<'a> {
    pub start: usize,
    pub words: &'a [u64],
}

/// Folds the sparse public-I/O table without materializing a dense `K` table.
pub fn fold_public_segments<F: Field>(
    plan: RamOutputCheckHybridPlan,
    weights: &[F],
    segments: &[RamOutputPublicSegment<'_>],
) -> Result<Vec<F>, RamOutputCheckPlanError> {
    if weights.len() != plan.block_elements {
        return Err(RamOutputCheckPlanError::Length {
            name: "low weights",
            expected: plan.block_elements,
            got: weights.len(),
        });
    }
    let mut result = vec![F::zero(); plan.tail_elements];
    for segment in segments {
        let end = segment
            .start
            .checked_add(segment.words.len())
            .ok_or(RamOutputCheckPlanError::SizeOverflow)?;
        if end > plan.addresses {
            return Err(RamOutputCheckPlanError::SegmentOutOfRange {
                addresses: plan.addresses,
                start: segment.start,
                end,
            });
        }
        for (offset, &word) in segment.words.iter().enumerate() {
            if word == 0 {
                continue;
            }
            let address = segment.start + offset;
            let block = address / plan.block_elements;
            let low = address % plan.block_elements;
            result[block] += F::from_u64(word) * weights[low];
        }
    }
    Ok(result)
}

/// Dense, representation-independent oracle for round and deferral parity.
pub struct DenseRamOutputOracle<F: Field> {
    eq_address: Vec<F>,
    io_mask: Vec<F>,
    val_io: Vec<F>,
    val_final: Vec<F>,
    round: usize,
}

impl<F: Field> DenseRamOutputOracle<F> {
    pub fn new(
        eq_address: Vec<F>,
        io_mask: Vec<F>,
        val_io: Vec<F>,
        val_final: Vec<F>,
    ) -> Result<Self, RamOutputCheckPlanError> {
        let addresses = eq_address.len();
        if addresses == 0 || !addresses.is_power_of_two() {
            return Err(RamOutputCheckPlanError::InvalidAddresses(addresses));
        }
        for (name, got) in [
            ("io mask", io_mask.len()),
            ("public IO", val_io.len()),
            ("val final", val_final.len()),
        ] {
            if got != addresses {
                return Err(RamOutputCheckPlanError::Length {
                    name,
                    expected: addresses,
                    got,
                });
            }
        }
        Ok(Self {
            eq_address,
            io_mask,
            val_io,
            val_final,
            round: 0,
        })
    }

    pub fn message_evals(&self) -> [F; 4] {
        std::array::from_fn(|node| {
            let t = F::from_u64(node as u64);
            (0..self.eq_address.len() / 2).fold(F::zero(), |sum, pair| {
                let eq = extend_pair(&self.eq_address, pair, t);
                let mask = extend_pair(&self.io_mask, pair, t);
                let val_io = extend_pair(&self.val_io, pair, t);
                let val_final = extend_pair(&self.val_final, pair, t);
                sum + eq * mask * (val_final - val_io)
            })
        })
    }

    pub fn message(&self, previous_claim: F) -> Result<UnivariatePoly<F>, RamOutputCheckPlanError> {
        let evals = self.message_evals();
        if evals[0] + evals[1] != previous_claim {
            return Err(RamOutputCheckPlanError::RoundClaim { round: self.round });
        }
        Ok(UnivariatePoly::from_evals(&evals))
    }

    pub fn bind(&mut self, challenge: F) {
        for table in [
            &mut self.eq_address,
            &mut self.io_mask,
            &mut self.val_io,
            &mut self.val_final,
        ] {
            bind_low(table, challenge);
        }
        self.round += 1;
    }

    pub fn defer_zero_prefix(&mut self, challenges: &[F]) -> Result<(), RamOutputCheckPlanError> {
        for &challenge in challenges {
            if self.message_evals().iter().any(|value| *value != F::zero()) {
                return Err(RamOutputCheckPlanError::NonZeroDeferredRound { round: self.round });
            }
            self.bind(challenge);
        }
        Ok(())
    }

    pub fn val_final(&self) -> &[F] {
        &self.val_final
    }
}

fn extend_pair<F: Field>(table: &[F], pair: usize, t: F) -> F {
    let low = table[2 * pair];
    low + t * (table[2 * pair + 1] - low)
}

fn bind_low<F: Field>(table: &mut Vec<F>, challenge: F) {
    let bound_len = table.len() / 2;
    for pair in 0..bound_len {
        table[pair] = extend_pair(table, pair, challenge);
    }
    table.truncate(bound_len);
}

fn shader_count(name: &'static str, value: usize) -> Result<u32, RamOutputCheckPlanError> {
    value
        .try_into()
        .map_err(|_| RamOutputCheckPlanError::ShaderIndexOverflow { name })
}

fn field_bytes(fields: usize) -> Result<usize, RamOutputCheckPlanError> {
    fields
        .checked_mul(RAM_OUTPUT_CHECK_FIELD_BYTES)
        .ok_or(RamOutputCheckPlanError::SizeOverflow)
}

fn source_bytes(elements: usize) -> Result<usize, RamOutputCheckPlanError> {
    elements
        .checked_mul(RAM_OUTPUT_CHECK_SOURCE_BYTES)
        .ok_or(RamOutputCheckPlanError::SizeOverflow)
}
