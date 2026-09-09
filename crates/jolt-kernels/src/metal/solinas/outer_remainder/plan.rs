use std::mem::size_of;

use super::super::{Fp128, MetalError, PipelineLimits, SolinasMetal};
use super::api::{
    OuterRemainderSequenceConfig, DEVICE_BUFFERS, OUTER_REMAINDER_A_LOOKUP_FIELDS,
    OUTER_REMAINDER_MAX_OUTPUTS, OUTER_REMAINDER_OPENINGS,
};
use super::registers_claim::carrier_geometry;

pub(super) const SIMD_WIDTH: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct OpeningLayout {
    pub(super) tile_rows: usize,
    pub(super) source_row_words: usize,
    pub(super) row_stride_words: usize,
    pub(super) shard_sums: bool,
}

pub(super) const fn opening_layout() -> OpeningLayout {
    OpeningLayout {
        tile_rows: 64,
        source_row_words: 20,
        row_stride_words: 20,
        shard_sums: true,
    }
}

#[derive(Clone, Copy)]
pub(super) struct StorageGeometry {
    pub(super) current_elements: usize,
    pub(super) weight_capacity: usize,
    pub(super) max_threadgroups: usize,
    pub(super) element_counts: [usize; DEVICE_BUFFERS],
    pub(super) owned_bytes: u64,
}

pub(crate) fn outer_remainder_sequence_storage_bytes_with_config(
    rows: usize,
    config: OuterRemainderSequenceConfig,
) -> Result<u64, MetalError> {
    Ok(storage_geometry(rows, config)?.owned_bytes)
}

pub(crate) fn outer_remainder_sequence_max_buffer_bytes_with_config(
    rows: usize,
    config: OuterRemainderSequenceConfig,
) -> Result<u64, MetalError> {
    let base = storage_geometry(rows, config)?
        .element_counts
        .into_iter()
        .try_fold(0, |maximum, elements| {
            Ok::<u64, MetalError>(maximum.max(field_bytes(elements)?))
        })?;
    if config.registers_claim_carrier {
        Ok(base.max(carrier_geometry(rows)?.max_buffer_bytes))
    } else {
        Ok(base)
    }
}

pub(super) fn validate_opening_threadgroup_memory(
    context: &SolinasMetal,
    limits: PipelineLimits,
    threads: usize,
    product_uniskip_carrier: bool,
) -> Result<(), MetalError> {
    let dynamic = opening_threadgroup_memory_lengths(threads, product_uniskip_carrier)?
        .into_iter()
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .ok_or(MetalError::InvalidOuterRemainderConfig(
            "opening threadgroup byte count overflowed",
        ))?;
    let requested = limits
        .static_threadgroup_memory_length
        .checked_add(dynamic)
        .ok_or(MetalError::InvalidOuterRemainderConfig(
            "opening threadgroup byte count overflowed",
        ))?;
    let maximum = context.device.max_threadgroup_memory_length();
    if requested > maximum {
        return Err(MetalError::OuterRemainderThreadgroupMemory { requested, maximum });
    }
    Ok(())
}

pub(super) fn opening_threadgroup_memory_lengths(
    threads: usize,
    product_uniskip_carrier: bool,
) -> Result<[u64; 3], MetalError> {
    let layout = opening_layout();
    let row_words = layout
        .tile_rows
        .checked_mul(layout.row_stride_words)
        .ok_or(MetalError::InvalidOuterRemainderConfig(
            "opening threadgroup byte count overflowed",
        ))?;
    let row_bytes = row_words.checked_mul(size_of::<u64>());
    let weight_bytes = layout.tile_rows.checked_mul(size_of::<Fp128>());
    let outputs = opening_output_count(product_uniskip_carrier);
    let shard_bytes = if layout.shard_sums {
        outputs
            .checked_mul(threads / outputs)
            .and_then(|elements| elements.checked_mul(size_of::<Fp128>()))
    } else {
        Some(0)
    };
    let [Some(row_bytes), Some(weight_bytes), Some(shard_bytes)] =
        [row_bytes, weight_bytes, shard_bytes]
    else {
        return Err(MetalError::InvalidOuterRemainderConfig(
            "opening threadgroup byte count overflowed",
        ));
    };
    Ok([row_bytes as u64, weight_bytes as u64, shard_bytes as u64])
}

pub(super) const fn opening_output_count(product_uniskip_carrier: bool) -> usize {
    if product_uniskip_carrier {
        OUTER_REMAINDER_MAX_OUTPUTS
    } else {
        OUTER_REMAINDER_OPENINGS
    }
}

pub(super) fn storage_geometry(
    cycles: usize,
    config: OuterRemainderSequenceConfig,
) -> Result<StorageGeometry, MetalError> {
    if cycles < 4 || !cycles.is_power_of_two() {
        return Err(MetalError::InvalidOuterRemainderRows(cycles));
    }
    if config.max_threadgroups == 0 {
        return Err(MetalError::InvalidOuterRemainderConfig(
            "max_threadgroups must be nonzero",
        ));
    }
    if config.cpu_tail_elements < 2 || !config.cpu_tail_elements.is_power_of_two() {
        return Err(MetalError::InvalidOuterRemainderConfig(
            "cpu_tail_elements must be a power of two of at least two",
        ));
    }
    let current_elements = cycles
        .checked_mul(2)
        .ok_or(MetalError::InputTooLong(cycles))?;
    validate_u32(current_elements)?;
    let weight_bits = (cycles.ilog2() as usize).div_ceil(2);
    let weight_capacity = 1usize
        .checked_shl(weight_bits as u32)
        .ok_or(MetalError::InputTooLong(cycles))?;
    let max_threadgroups = config.max_threadgroups.min(weight_capacity);
    let message_partials = 2usize
        .checked_mul(max_threadgroups)
        .ok_or(MetalError::InputTooLong(max_threadgroups))?;
    let opening_outputs = opening_output_count(config.product_uniskip_carrier);
    let opening_partials = opening_outputs
        .checked_mul(max_threadgroups)
        .ok_or(MetalError::InputTooLong(max_threadgroups))?;
    let element_counts = [
        current_elements,
        current_elements / 2,
        weight_capacity,
        weight_capacity,
        OUTER_REMAINDER_A_LOOKUP_FIELDS,
        message_partials,
        2,
        opening_partials,
        opening_outputs,
    ];
    let mut owned_bytes = element_counts.iter().try_fold(0u64, |total, &elements| {
        total
            .checked_add(field_bytes(elements)?)
            .ok_or(MetalError::InputTooLong(cycles))
    })?;
    if config.registers_claim_carrier {
        owned_bytes = owned_bytes
            .checked_add(carrier_geometry(cycles)?.owned_bytes)
            .ok_or(MetalError::InputTooLong(cycles))?;
    }
    Ok(StorageGeometry {
        current_elements,
        weight_capacity,
        max_threadgroups,
        element_counts,
        owned_bytes,
    })
}

pub(super) fn field_bytes(elements: usize) -> Result<u64, MetalError> {
    let bytes = elements
        .checked_mul(size_of::<Fp128>())
        .ok_or(MetalError::InputTooLong(elements))?;
    u64::try_from(bytes).map_err(|_| MetalError::InputTooLong(elements))
}

fn validate_u32(elements: usize) -> Result<(), MetalError> {
    let _ = to_u32(elements)?;
    Ok(())
}

pub(super) fn to_u32(elements: usize) -> Result<u32, MetalError> {
    u32::try_from(elements).map_err(|_| MetalError::InputTooLong(elements))
}

pub(super) fn message_threadgroup_bytes(threads: usize) -> u64 {
    (2 * (threads / SIMD_WIDTH) * size_of::<Fp128>()) as u64
}
