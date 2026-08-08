use std::mem::size_of;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use jolt_field::AkitaField;
use jolt_poly::EqPolynomial;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::{
    instruction_read_raf_stage1_claim_bytes, instruction_read_raf_stage1_row_bytes,
    InstructionReadRafCountOrder, InstructionReadRafStage1Lease, InstructionReadRafStage1Receipt,
    INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS, INSTRUCTION_READ_RAF_SEGMENTS,
    INSTRUCTION_READ_RAF_TABLES,
};
use crate::metal::solinas::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, SolinasMetal,
};

const PIPELINE: &str = "solinas_instruction_read_raf_compatibility_scatter";
const SIMD_WIDTH: usize = 32;
const STATUS_BYTES: u64 = size_of::<u32>() as u64;
const LOOKUP_BYTES_PER_ROW: u64 = 2 * size_of::<u64>() as u64;
const PACKED_BYTES_PER_ROW: u64 = size_of::<u8>() as u64;
const INVERSE_BYTES_PER_ROW: u64 = size_of::<u32>() as u64;
const WEIGHT_BYTES_PER_ROW: u64 = size_of::<Fp128>() as u64;

type ScatterLayout = (
    Vec<u32>,
    [u32; INSTRUCTION_READ_RAF_SEGMENTS + 1],
    [Range<usize>; INSTRUCTION_READ_RAF_SEGMENTS],
);

static NEXT_SCATTER_COMPLETION_SERIAL: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InstructionReadRafCompatibilityScatterConfig {
    pub(crate) threads_per_threadgroup: usize,
}

impl Default for InstructionReadRafCompatibilityScatterConfig {
    fn default() -> Self {
        Self {
            threads_per_threadgroup: 256,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InstructionReadRafProducerExecution {
    pub(crate) preparation_wall: Duration,
    pub(crate) command_wall: Duration,
    pub(crate) gpu_active: Duration,
    pub(crate) status_readback_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct InstructionReadRafDenseGroupedReceipt {
    source: InstructionReadRafStage1Receipt,
    rows: usize,
    segment_ranges: [Range<usize>; INSTRUCTION_READ_RAF_SEGMENTS],
    packed_rows_bytes: u64,
    lookups_bytes: u64,
    inverse_bytes: u64,
    weights_bytes: u64,
    packed_rows_identity: usize,
    lookups_identity: usize,
    inverse_identity: usize,
    weights_identity: usize,
    device_registry_id: u64,
    completion_serial: u64,
    e_in_length: usize,
    e_out_length: usize,
    additional_allocation_bytes: u64,
    threadgroups: usize,
    threads_per_threadgroup: usize,
    dynamic_threadgroup_bytes: u64,
    static_threadgroup_bytes: u64,
    command_buffers: usize,
    waits: usize,
    encoders: usize,
    dispatches: usize,
    source_copy_bytes: u64,
    full_plane_readback_bytes: u64,
    complete_overwrite: bool,
}

impl InstructionReadRafDenseGroupedReceipt {
    pub(crate) const fn source(&self) -> InstructionReadRafStage1Receipt {
        self.source
    }

    pub(crate) const fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn segment_ranges(&self) -> &[Range<usize>; INSTRUCTION_READ_RAF_SEGMENTS] {
        &self.segment_ranges
    }

    pub(crate) const fn packed_rows_bytes(&self) -> u64 {
        self.packed_rows_bytes
    }

    pub(crate) const fn lookups_bytes(&self) -> u64 {
        self.lookups_bytes
    }

    pub(crate) const fn inverse_bytes(&self) -> u64 {
        self.inverse_bytes
    }

    pub(crate) const fn weights_bytes(&self) -> u64 {
        self.weights_bytes
    }

    pub(crate) const fn allocation_identities(&self) -> [usize; 4] {
        [
            self.packed_rows_identity,
            self.lookups_identity,
            self.inverse_identity,
            self.weights_identity,
        ]
    }

    pub(crate) const fn device_registry_id(&self) -> u64 {
        self.device_registry_id
    }

    pub(crate) const fn completion_serial(&self) -> u64 {
        self.completion_serial
    }

    pub(crate) const fn e_in_length(&self) -> usize {
        self.e_in_length
    }

    pub(crate) const fn e_out_length(&self) -> usize {
        self.e_out_length
    }

    pub(crate) const fn additional_allocation_bytes(&self) -> u64 {
        self.additional_allocation_bytes
    }

    pub(crate) const fn threadgroups(&self) -> usize {
        self.threadgroups
    }

    pub(crate) const fn threads_per_threadgroup(&self) -> usize {
        self.threads_per_threadgroup
    }

    pub(crate) const fn dynamic_threadgroup_bytes(&self) -> u64 {
        self.dynamic_threadgroup_bytes
    }

    pub(crate) const fn static_threadgroup_bytes(&self) -> u64 {
        self.static_threadgroup_bytes
    }

    pub(crate) const fn command_buffers(&self) -> usize {
        self.command_buffers
    }

    pub(crate) const fn waits(&self) -> usize {
        self.waits
    }

    pub(crate) const fn encoders(&self) -> usize {
        self.encoders
    }

    pub(crate) const fn dispatches(&self) -> usize {
        self.dispatches
    }

    pub(crate) const fn source_copy_bytes(&self) -> u64 {
        self.source_copy_bytes
    }

    pub(crate) const fn full_plane_readback_bytes(&self) -> u64 {
        self.full_plane_readback_bytes
    }

    pub(crate) const fn complete_overwrite(&self) -> bool {
        self.complete_overwrite
    }
}

pub(crate) struct InstructionReadRafDenseGroupedPlanes {
    packed_rows: Buffer,
    lookups: Buffer,
    inverse: Buffer,
    weights: Buffer,
    receipt: InstructionReadRafDenseGroupedReceipt,
    execution: InstructionReadRafProducerExecution,
}

pub(crate) struct InstructionReadRafDenseGroupedParts {
    pub(crate) packed_rows: Buffer,
    pub(crate) lookups: Buffer,
    pub(crate) inverse: Buffer,
    pub(crate) weights: Buffer,
    pub(crate) receipt: InstructionReadRafDenseGroupedReceipt,
}

impl InstructionReadRafDenseGroupedPlanes {
    pub(crate) fn receipt(&self) -> &InstructionReadRafDenseGroupedReceipt {
        &self.receipt
    }

    pub(crate) const fn execution(&self) -> InstructionReadRafProducerExecution {
        self.execution
    }

    pub(crate) fn buffers(&self) -> [&Buffer; 4] {
        [
            &self.packed_rows,
            &self.lookups,
            &self.inverse,
            &self.weights,
        ]
    }

    pub(crate) fn into_parts(self) -> InstructionReadRafDenseGroupedParts {
        InstructionReadRafDenseGroupedParts {
            packed_rows: self.packed_rows,
            lookups: self.lookups,
            inverse: self.inverse,
            weights: self.weights,
            receipt: self.receipt,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScatterParams {
    rows: u32,
    chunks: u32,
    chunk_rows: u32,
    segments: u32,
    e_in_length: u32,
    e_out_length: u32,
    packed_rows_elements: u32,
    lookup_elements: u32,
    inverse_elements: u32,
    weight_elements: u32,
    status_elements: u32,
    e_in_log2: u32,
}

const _: () = assert!(size_of::<ScatterParams>() == 48);
const _: () = assert!(size_of::<Fp128>() == 16);

impl SolinasMetal {
    pub(crate) fn prepare_instruction_read_raf_compatibility_scatter(
        &self,
        source: InstructionReadRafStage1Lease,
        r_reduction: &[AkitaField],
        config: InstructionReadRafCompatibilityScatterConfig,
    ) -> Result<InstructionReadRafDenseGroupedPlanes, MetalError> {
        let preparation_start = Instant::now();
        let source_receipt = source.receipt();
        let rows = source_receipt.rows();
        let log_rows = rows.ilog2() as usize;
        if r_reduction.len() != log_rows {
            return Err(invalid_scatter(
                "reduction point length disagrees with the Stage-1 row domain",
            ));
        }
        if source_receipt.count_order() != InstructionReadRafCountOrder::TableMajorThenNoneV1 {
            return Err(invalid_scatter("Stage-1 counts use an unsupported order"));
        }
        if source.row_buffer().length() != instruction_read_raf_stage1_row_bytes(rows)?
            || source.claim_buffer().length() != instruction_read_raf_stage1_claim_bytes(rows)?
            || source.counts().len() != rows.div_ceil(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS)
        {
            return Err(invalid_scatter(
                "Stage-1 source shape changed before scatter",
            ));
        }
        if source_receipt.device_registry_id() != self.device.registry_id() {
            return Err(invalid_scatter(
                "Stage-1 source belongs to another Metal device",
            ));
        }

        let pipeline = self.compile_named_pipeline(PIPELINE)?;
        let limits = Self::limits(&pipeline);
        if limits.thread_execution_width != SIMD_WIDTH {
            return Err(MetalError::UnsupportedAddressRafExecutionWidth {
                pipeline: PIPELINE,
                expected: SIMD_WIDTH,
                got: limits.thread_execution_width,
            });
        }
        let threads =
            Self::resolve_threadgroup_width(Some(config.threads_per_threadgroup), limits)?;
        if threads < 128 {
            return Err(invalid_scatter(
                "compatibility scatter requires at least 128 threads",
            ));
        }
        let threadgroup_bytes = (INSTRUCTION_READ_RAF_SEGMENTS * size_of::<u32>()) as u64;
        let total_threadgroup_bytes = threadgroup_bytes
            .checked_add(limits.static_threadgroup_memory_length)
            .ok_or(MetalError::InputTooLong(INSTRUCTION_READ_RAF_SEGMENTS))?;
        let maximum_threadgroup_bytes = self.device.max_threadgroup_memory_length();
        if total_threadgroup_bytes > maximum_threadgroup_bytes {
            return Err(MetalError::AddressRafDirectThreadgroupMemory {
                requested: total_threadgroup_bytes,
                maximum: maximum_threadgroup_bytes,
            });
        }

        let chunks = source.counts().len();
        let (chunk_bases, segment_offsets, segment_ranges) = scatter_layout(source.counts(), rows)?;
        let out_log = log_rows / 2;
        let (out_point, in_point) = r_reduction.split_at(out_log);
        let e_in = EqPolynomial::<AkitaField>::evals(in_point, None)
            .iter()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>();
        let e_out = EqPolynomial::<AkitaField>::evals(out_point, None)
            .iter()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>();
        let e_in_length = e_in.len();
        let e_out_length = e_out.len();

        let packed_rows_bytes = checked_row_bytes(rows, PACKED_BYTES_PER_ROW)?;
        let lookups_bytes = checked_row_bytes(rows, LOOKUP_BYTES_PER_ROW)?;
        let inverse_bytes = checked_row_bytes(rows, INVERSE_BYTES_PER_ROW)?;
        let weights_bytes = checked_row_bytes(rows, WEIGHT_BYTES_PER_ROW)?;
        let buffer_lengths = [
            packed_rows_bytes,
            lookups_bytes,
            inverse_bytes,
            weights_bytes,
            checked_buffer_bytes::<u32>(chunk_bases.len(), rows)?,
            checked_buffer_bytes::<u32>(segment_offsets.len(), rows)?,
            checked_buffer_bytes::<Fp128>(e_in_length, rows)?,
            checked_buffer_bytes::<Fp128>(e_out_length, rows)?,
            STATUS_BYTES,
            size_of::<ScatterParams>() as u64,
        ];
        for requested in buffer_lengths {
            self.validate_buffer_length(requested)?;
        }
        let additional = buffer_lengths
            .iter()
            .try_fold(0u64, |sum, value| sum.checked_add(*value))
            .ok_or(MetalError::InputTooLong(rows))?;
        self.validate_additional_working_set(additional)?;

        let packed_rows = self
            .device
            .new_buffer(packed_rows_bytes, MTLResourceOptions::StorageModeShared);
        let lookups = self
            .device
            .new_buffer(lookups_bytes, MTLResourceOptions::StorageModeShared);
        let inverse = self
            .device
            .new_buffer(inverse_bytes, MTLResourceOptions::StorageModeShared);
        let weights = self
            .device
            .new_buffer(weights_bytes, MTLResourceOptions::StorageModeShared);
        let chunk_bases = buffer_from_slice(&self.device, &chunk_bases);
        let segment_offsets = buffer_from_slice(&self.device, &segment_offsets);
        let e_in = buffer_from_slice(&self.device, &e_in);
        let e_out = buffer_from_slice(&self.device, &e_out);
        let status = buffer_from_slice(&self.device, &[0u32]);
        let rows_u32 = u32::try_from(rows).map_err(|_| MetalError::InputTooLong(rows))?;
        let params = ScatterParams {
            rows: rows_u32,
            chunks: u32::try_from(chunks).map_err(|_| MetalError::InputTooLong(chunks))?,
            chunk_rows: INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS as u32,
            segments: INSTRUCTION_READ_RAF_SEGMENTS as u32,
            e_in_length: u32::try_from(e_in_length).map_err(|_| MetalError::InputTooLong(rows))?,
            e_out_length: u32::try_from(e_out_length)
                .map_err(|_| MetalError::InputTooLong(rows))?,
            packed_rows_elements: rows_u32,
            lookup_elements: rows_u32,
            inverse_elements: rows_u32,
            weight_elements: rows_u32,
            status_elements: 1,
            e_in_log2: e_in_length.ilog2(),
        };
        let params = buffer_from_slice(&self.device, std::slice::from_ref(&params));
        let preparation_wall = preparation_start.elapsed();

        let command_start = Instant::now();
        let command_buffer = self.queue.new_command_buffer();
        autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(source.row_buffer()), 0);
            encoder.set_buffer(1, Some(source.claim_buffer()), 0);
            encoder.set_buffer(2, Some(&chunk_bases), 0);
            encoder.set_buffer(3, Some(&segment_offsets), 0);
            encoder.set_buffer(4, Some(&e_in), 0);
            encoder.set_buffer(5, Some(&e_out), 0);
            encoder.set_buffer(6, Some(&lookups), 0);
            encoder.set_buffer(7, Some(&packed_rows), 0);
            encoder.set_buffer(8, Some(&inverse), 0);
            encoder.set_buffer(9, Some(&weights), 0);
            encoder.set_buffer(10, Some(&status), 0);
            encoder.set_buffer(11, Some(&params), 0);
            encoder.set_threadgroup_memory_length(0, threadgroup_bytes);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: chunks as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
        });
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let command_wall = command_start.elapsed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(MetalError::CommandFailed(command_buffer.status()));
        }
        let gpu_start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
        let gpu_end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
        if !gpu_start.is_finite() || !gpu_end.is_finite() || gpu_end < gpu_start {
            return Err(MetalError::InvalidGpuTimestamps {
                start: gpu_start,
                end: gpu_end,
            });
        }
        let gpu_active = Duration::from_secs_f64(gpu_end - gpu_start);
        let status_value = read_status(&status);
        if status_value != 0 {
            return Err(invalid_scatter(
                "compatibility scatter reported invalid output",
            ));
        }

        let allocation_identities = [
            packed_rows.as_ptr() as usize,
            lookups.as_ptr() as usize,
            inverse.as_ptr() as usize,
            weights.as_ptr() as usize,
        ];
        if allocation_identities.contains(&0)
            || allocation_identities
                .iter()
                .enumerate()
                .any(|(index, identity)| allocation_identities[..index].contains(identity))
        {
            return Err(invalid_scatter("compatibility output allocations alias"));
        }
        let completion_serial = next_completion_serial()?;
        let receipt = InstructionReadRafDenseGroupedReceipt {
            source: source_receipt,
            rows,
            segment_ranges,
            packed_rows_bytes,
            lookups_bytes,
            inverse_bytes,
            weights_bytes,
            packed_rows_identity: allocation_identities[0],
            lookups_identity: allocation_identities[1],
            inverse_identity: allocation_identities[2],
            weights_identity: allocation_identities[3],
            device_registry_id: self.device.registry_id(),
            completion_serial,
            e_in_length,
            e_out_length,
            additional_allocation_bytes: additional,
            threadgroups: chunks,
            threads_per_threadgroup: threads,
            dynamic_threadgroup_bytes: threadgroup_bytes,
            static_threadgroup_bytes: limits.static_threadgroup_memory_length,
            command_buffers: 1,
            waits: 1,
            encoders: 1,
            dispatches: 1,
            source_copy_bytes: 0,
            full_plane_readback_bytes: 0,
            complete_overwrite: true,
        };
        Ok(InstructionReadRafDenseGroupedPlanes {
            packed_rows,
            lookups,
            inverse,
            weights,
            receipt,
            execution: InstructionReadRafProducerExecution {
                preparation_wall,
                command_wall,
                gpu_active,
                status_readback_bytes: STATUS_BYTES,
            },
        })
    }
}

fn scatter_layout(
    counts: &[[u32; INSTRUCTION_READ_RAF_SEGMENTS]],
    rows: usize,
) -> Result<ScatterLayout, MetalError> {
    let mut totals = [0usize; INSTRUCTION_READ_RAF_SEGMENTS];
    for chunk in counts {
        for (total, count) in totals.iter_mut().zip(chunk) {
            *total = total
                .checked_add(*count as usize)
                .ok_or(MetalError::InputTooLong(rows))?;
        }
    }
    let covered_rows = totals
        .iter()
        .try_fold(0usize, |sum, count| sum.checked_add(*count))
        .ok_or(MetalError::InputTooLong(rows))?;
    if covered_rows != rows {
        return Err(invalid_scatter(
            "Stage-1 counts do not cover the row domain",
        ));
    }
    let mut offsets = [0u32; INSTRUCTION_READ_RAF_SEGMENTS + 1];
    for (index, count) in totals.iter().enumerate() {
        offsets[index + 1] = u32::try_from(offsets[index] as usize + count)
            .map_err(|_| MetalError::InputTooLong(rows))?;
    }
    let mut running: [u32; INSTRUCTION_READ_RAF_SEGMENTS] =
        std::array::from_fn(|rank| offsets[rank]);
    let mut chunk_bases = vec![0u32; counts.len() * INSTRUCTION_READ_RAF_SEGMENTS];
    for (chunk, counts) in counts.iter().enumerate() {
        for rank in 0..INSTRUCTION_READ_RAF_SEGMENTS {
            chunk_bases[chunk * INSTRUCTION_READ_RAF_SEGMENTS + rank] = running[rank];
            running[rank] = running[rank]
                .checked_add(counts[rank])
                .ok_or(MetalError::InputTooLong(rows))?;
        }
    }
    if running
        .iter()
        .enumerate()
        .any(|(rank, value)| *value != offsets[rank + 1])
    {
        return Err(invalid_scatter(
            "chunk prefixes disagree with segment totals",
        ));
    }
    let ranges = std::array::from_fn(|logical| {
        let physical = if logical < 2 {
            logical + 2 * INSTRUCTION_READ_RAF_TABLES
        } else {
            logical - 2
        };
        offsets[physical] as usize..offsets[physical + 1] as usize
    });
    Ok((chunk_bases, offsets, ranges))
}

fn checked_row_bytes(rows: usize, bytes_per_row: u64) -> Result<u64, MetalError> {
    u64::try_from(rows)
        .ok()
        .and_then(|rows| rows.checked_mul(bytes_per_row))
        .ok_or(MetalError::InputTooLong(rows))
}

fn checked_buffer_bytes<T>(elements: usize, domain_rows: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(domain_rows))
}

fn read_status(buffer: &Buffer) -> u32 {
    // SAFETY: the status buffer contains one initialized u32 and remains
    // alive for the duration of this read.
    unsafe { *buffer.contents().cast::<u32>() }
}

fn next_completion_serial() -> Result<u64, MetalError> {
    NEXT_SCATTER_COMPLETION_SERIAL
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| invalid_scatter("scatter completion counter is exhausted"))
}

fn invalid_scatter(message: &'static str) -> MetalError {
    MetalError::InvalidInstructionReadRafGrouped(message.to_owned())
}
