use std::mem::size_of;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_poly::EqPolynomial;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, MTLResourceOptions, MTLSize,
};

use super::{
    instruction_read_raf_stage1_claim_bytes, instruction_read_raf_stage1_row_bytes,
    InstructionReadRafCountOrder, InstructionReadRafStage1Lease, InstructionReadRafStage1Receipt,
    INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS, INSTRUCTION_READ_RAF_SEGMENTS,
    INSTRUCTION_READ_RAF_TABLES,
};
use crate::metal::solinas::{
    buffer_from_slice,
    bytecode_read_raf_address::{
        BytecodeAddressFusedScatterRequest, BytecodeAddressSparseStage1Carrier,
        BytecodeAddressStage1TopologyReceipt,
    },
    Fp128, MetalError, SolinasMetal,
};

const PIPELINE: &str = "solinas_instruction_read_raf_compatibility_scatter";
const SIMD_WIDTH: usize = 32;
const STATUS_BYTES: u64 = size_of::<u32>() as u64;
const LOOKUP_BYTES_PER_ROW: u64 = 2 * size_of::<u64>() as u64;
const PACKED_BYTES_PER_ROW: u64 = size_of::<u8>() as u64;
const INVERSE_BYTES_PER_ROW: u64 = size_of::<u32>() as u64;
const WEIGHT_BYTES_PER_ROW: u64 = size_of::<Fp128>() as u64;
const BYTECODE_DESCRIPTOR_BYTES: usize = 8;
const BYTECODE_PIVOT_BYTES: usize = size_of::<u16>();
const BYTECODE_OCCURRENCE_BYTES_PER_ROW: u64 = size_of::<u16>() as u64;
const BYTECODE_MAGNITUDE_BYTES_PER_ROW: u64 = size_of::<u64>() as u64;
const BYTECODE_INNER_LOG2: u32 = 15;
const BYTECODE_MAX_DESCRIPTORS_PER_CHUNK: usize = INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
const BYTECODE_MAX_PIVOTS_PER_CHUNK: usize = 15;
const THREADGROUP_ALLOCATION_ALIGNMENT: u64 = 16;

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
    dispatches: usize,
    source_copy_bytes: u64,
    full_plane_readback_bytes: u64,
    complete_overwrite: bool,
    bytecode: Option<InstructionReadRafFusedBytecodeReceipt>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InstructionReadRafFusedBytecodeReceipt {
    physical_rows: usize,
    work_items: usize,
    descriptor_elements: usize,
    descriptor_bytes: usize,
    descriptor_identity: usize,
    pivot_elements: usize,
    pivot_bytes: usize,
    pivot_identity: usize,
    chunk_offset_elements: usize,
    chunk_offset_bytes: usize,
    chunk_offset_identity: usize,
    work_item_bytes: usize,
    work_item_identity: usize,
    address_offset_elements: usize,
    address_offset_bytes: usize,
    address_offset_identity: usize,
    occurrence_bytes: usize,
    occurrence_identity: usize,
    magnitude_bytes: usize,
    magnitude_identity: usize,
    max_descriptors_per_chunk: usize,
    max_pivots_per_chunk: usize,
    dynamic_threadgroup_bytes: u64,
    max_admitted_descriptors_per_chunk: usize,
    max_admitted_pivots_per_chunk: usize,
    threadgroup_memory_limit_bytes: u64,
    shared_source_row_scans: usize,
    additional_source_row_scans: usize,
    member_upload_bytes: usize,
}

impl InstructionReadRafDenseGroupedReceipt {
    copy_field_getters! { pub(crate), {
        source: InstructionReadRafStage1Receipt,
        rows: usize,
        packed_rows_bytes: u64,
        lookups_bytes: u64,
        inverse_bytes: u64,
        weights_bytes: u64,
        device_registry_id: u64,
        completion_serial: u64,
        e_in_length: usize,
        e_out_length: usize,
        dispatches: usize,
        source_copy_bytes: u64,
        full_plane_readback_bytes: u64,
        complete_overwrite: bool,
        bytecode: Option<InstructionReadRafFusedBytecodeReceipt>,
    } }

    pub(crate) fn segment_ranges(&self) -> &[Range<usize>; INSTRUCTION_READ_RAF_SEGMENTS] {
        &self.segment_ranges
    }

    pub(crate) const fn allocation_identities(&self) -> [usize; 4] {
        [
            self.packed_rows_identity,
            self.lookups_identity,
            self.inverse_identity,
            self.weights_identity,
        ]
    }
}

impl InstructionReadRafFusedBytecodeReceipt {
    copy_field_getters! { pub(crate), {
        physical_rows: usize,
        work_items: usize,
        descriptor_elements: usize,
        descriptor_bytes: usize,
        descriptor_identity: usize,
        pivot_elements: usize,
        pivot_bytes: usize,
        pivot_identity: usize,
        chunk_offset_elements: usize,
        chunk_offset_bytes: usize,
        chunk_offset_identity: usize,
        work_item_bytes: usize,
        work_item_identity: usize,
        address_offset_elements: usize,
        address_offset_bytes: usize,
        address_offset_identity: usize,
        occurrence_bytes: usize,
        occurrence_identity: usize,
        magnitude_bytes: usize,
        magnitude_identity: usize,
        max_descriptors_per_chunk: usize,
        max_pivots_per_chunk: usize,
        dynamic_threadgroup_bytes: u64,
        max_admitted_descriptors_per_chunk: usize,
        max_admitted_pivots_per_chunk: usize,
        threadgroup_memory_limit_bytes: u64,
        shared_source_row_scans: usize,
        additional_source_row_scans: usize,
        member_upload_bytes: usize,
    } }
}

pub(crate) struct InstructionReadRafDenseGroupedPlanes {
    packed_rows: Buffer,
    lookups: Buffer,
    inverse: Buffer,
    weights: Buffer,
    receipt: InstructionReadRafDenseGroupedReceipt,
    bytecode_carrier: Option<BytecodeAddressSparseStage1Carrier>,
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

    pub(crate) fn into_parts(self) -> InstructionReadRafDenseGroupedParts {
        InstructionReadRafDenseGroupedParts {
            packed_rows: self.packed_rows,
            lookups: self.lookups,
            inverse: self.inverse,
            weights: self.weights,
            receipt: self.receipt,
        }
    }

    pub(crate) fn take_bytecode_carrier(&mut self) -> Option<BytecodeAddressSparseStage1Carrier> {
        self.bytecode_carrier.take()
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
    bytecode_enabled: u32,
    bytecode_physical_rows: u32,
    bytecode_descriptor_elements: u32,
    bytecode_pivot_elements: u32,
    bytecode_chunk_offset_elements: u32,
    bytecode_occurrence_elements: u32,
    bytecode_magnitude_elements: u32,
    bytecode_inner_log2: u32,
    bytecode_max_descriptors_per_chunk: u32,
    bytecode_max_pivots_per_chunk: u32,
}

const _: () = assert!(size_of::<ScatterParams>() == 88);
const _: () = assert!(size_of::<Fp128>() == 16);

impl SolinasMetal {
    pub(crate) fn prepare_instruction_read_raf_compatibility_scatter(
        &self,
        source: InstructionReadRafStage1Lease,
        r_reduction: &[AkitaField],
        config: InstructionReadRafCompatibilityScatterConfig,
        bytecode: Option<BytecodeAddressFusedScatterRequest>,
    ) -> Result<InstructionReadRafDenseGroupedPlanes, MetalError> {
        let prepare_started = Instant::now();
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
        if let Some(request) = bytecode.as_ref() {
            validate_bytecode_request(request, source_receipt, rows, self.device.registry_id())?;
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
        let count_threadgroup_bytes = (INSTRUCTION_READ_RAF_SEGMENTS * size_of::<u32>()) as u64;
        let (descriptor_threadgroup_bytes, pivot_threadgroup_bytes) = bytecode
            .as_ref()
            .map(|request| bytecode_threadgroup_bytes(&request.receipt()))
            .transpose()?
            .unwrap_or((0, 0));
        let threadgroup_bytes = count_threadgroup_bytes
            .checked_add(descriptor_threadgroup_bytes)
            .and_then(|bytes| bytes.checked_add(pivot_threadgroup_bytes))
            .ok_or(MetalError::InputTooLong(rows))?;
        let total_threadgroup_bytes = threadgroup_bytes
            .checked_add(limits.static_threadgroup_memory_length)
            .ok_or(MetalError::InputTooLong(INSTRUCTION_READ_RAF_SEGMENTS))?;
        let maximum_threadgroup_bytes = self.device.max_threadgroup_memory_length();
        let dynamic_threadgroup_memory_limit =
            maximum_threadgroup_bytes.saturating_sub(limits.static_threadgroup_memory_length);
        let bytecode_max_admitted_descriptors_per_chunk = if let Some(request) = bytecode.as_ref() {
            let observed = request.receipt().max_descriptors_per_chunk();
            let admitted = bytecode_descriptor_capacity(
                dynamic_threadgroup_memory_limit,
                request.receipt().max_pivots_per_chunk(),
            );
            if observed > admitted {
                return Err(invalid_scatter(format!(
                    "fused bytecode topology admission failed: clause=max_descriptors_per_chunk observed={observed} allowed=1..={admitted}"
                )));
            }
            admitted
        } else {
            0
        };
        if total_threadgroup_bytes > maximum_threadgroup_bytes {
            return Err(MetalError::AddressRafDirectThreadgroupMemory {
                requested: total_threadgroup_bytes,
                maximum: maximum_threadgroup_bytes,
            });
        }

        let chunks = source.counts().len();
        let layout_started = Instant::now();
        let (chunk_bases, segment_offsets, segment_ranges) = scatter_layout(source.counts(), rows)?;
        tracing::info!(
            target: "jolt::metal",
            rows,
            chunks,
            wall_ns = duration_nanos(layout_started.elapsed()),
            "prepared Instruction Read-RAF compatibility-scatter layout"
        );
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
        let (bytecode_occurrence_bytes, bytecode_magnitude_bytes) = bytecode
            .as_ref()
            .map(|request| {
                let physical_rows = request.receipt().physical_rows();
                Ok::<_, MetalError>((
                    checked_row_bytes(physical_rows, BYTECODE_OCCURRENCE_BYTES_PER_ROW)?,
                    checked_row_bytes(physical_rows, BYTECODE_MAGNITUDE_BYTES_PER_ROW)?,
                ))
            })
            .transpose()?
            .unwrap_or((0, 0));
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
            bytecode_occurrence_bytes,
            bytecode_magnitude_bytes,
        ];
        for &requested in &buffer_lengths {
            if requested != 0 {
                self.validate_buffer_length(requested)?;
            }
        }
        let additional = buffer_lengths
            .iter()
            .try_fold(0u64, |sum, value| sum.checked_add(*value))
            .ok_or(MetalError::InputTooLong(rows))?;
        self.validate_additional_working_set(additional)?;

        let allocation_started = Instant::now();
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
        let bytecode_output_buffers = bytecode.as_ref().map(|_| {
            (
                self.device.new_buffer(
                    bytecode_occurrence_bytes,
                    MTLResourceOptions::StorageModePrivate,
                ),
                self.device.new_buffer(
                    bytecode_magnitude_bytes,
                    MTLResourceOptions::StorageModePrivate,
                ),
            )
        });
        let chunk_bases = buffer_from_slice(&self.device, &chunk_bases);
        let segment_offsets = buffer_from_slice(&self.device, &segment_offsets);
        let e_in = buffer_from_slice(&self.device, &e_in);
        let e_out = buffer_from_slice(&self.device, &e_out);
        let status = buffer_from_slice(&self.device, &[0u32]);
        let rows_u32 = u32::try_from(rows).map_err(|_| MetalError::InputTooLong(rows))?;
        let bytecode_receipt = bytecode
            .as_ref()
            .map(BytecodeAddressFusedScatterRequest::receipt);
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
            bytecode_enabled: u32::from(bytecode_receipt.is_some()),
            bytecode_physical_rows: optional_shader_count(
                bytecode_receipt.map(|receipt| receipt.physical_rows()),
                rows,
            )?,
            bytecode_descriptor_elements: optional_shader_count(
                bytecode_receipt.map(|receipt| receipt.descriptor_elements()),
                rows,
            )?,
            bytecode_pivot_elements: optional_shader_count(
                bytecode_receipt.map(|receipt| receipt.pivot_elements()),
                rows,
            )?,
            bytecode_chunk_offset_elements: optional_shader_count(
                bytecode_receipt.map(|receipt| receipt.chunk_offset_elements()),
                rows,
            )?,
            bytecode_occurrence_elements: optional_shader_count(
                bytecode_receipt.map(|receipt| receipt.physical_rows()),
                rows,
            )?,
            bytecode_magnitude_elements: optional_shader_count(
                bytecode_receipt.map(|receipt| receipt.physical_rows()),
                rows,
            )?,
            bytecode_inner_log2: bytecode_receipt.map_or(0, |_| BYTECODE_INNER_LOG2),
            bytecode_max_descriptors_per_chunk: optional_shader_count(
                bytecode_receipt.map(|receipt| receipt.max_descriptors_per_chunk()),
                rows,
            )?,
            bytecode_max_pivots_per_chunk: optional_shader_count(
                bytecode_receipt.map(|receipt| receipt.max_pivots_per_chunk()),
                rows,
            )?,
        };
        tracing::info!(
            target: "jolt::metal",
            rows,
            output_bytes = packed_rows_bytes + lookups_bytes + inverse_bytes + weights_bytes,
            wall_ns = duration_nanos(allocation_started.elapsed()),
            storage_mode = "shared",
            "allocated Instruction Read-RAF compatibility-scatter outputs"
        );
        let dispatch_started = Instant::now();
        let params = buffer_from_slice(&self.device, std::slice::from_ref(&params));
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
            if let (Some(request), Some((occurrences, magnitudes))) =
                (bytecode.as_ref(), bytecode_output_buffers.as_ref())
            {
                encoder.set_buffer(12, Some(request.descriptors_buffer()), 0);
                encoder.set_buffer(13, Some(request.pivots_buffer()), 0);
                encoder.set_buffer(14, Some(request.chunk_offsets_buffer()), 0);
                encoder.set_buffer(15, Some(occurrences), 0);
                encoder.set_buffer(16, Some(magnitudes), 0);
            } else {
                encoder.set_buffer(12, Some(source.row_buffer()), 0);
                encoder.set_buffer(13, Some(source.row_buffer()), 0);
                encoder.set_buffer(14, Some(source.row_buffer()), 0);
                encoder.set_buffer(15, Some(source.row_buffer()), 0);
                encoder.set_buffer(16, Some(source.row_buffer()), 0);
            }
            encoder.set_threadgroup_memory_length(0, count_threadgroup_bytes);
            if descriptor_threadgroup_bytes != 0 {
                encoder.set_threadgroup_memory_length(1, descriptor_threadgroup_bytes);
            }
            if pivot_threadgroup_bytes != 0 {
                encoder.set_threadgroup_memory_length(2, pivot_threadgroup_bytes);
            }
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
        tracing::info!(
            target: "jolt::metal",
            rows,
            bytecode_descriptors = bytecode_receipt.map_or(0, |receipt| receipt.descriptors()),
            bytecode_max_descriptors_per_chunk = bytecode_receipt
                .map_or(0, |receipt| receipt.max_descriptors_per_chunk()),
            bytecode_pivots = bytecode_receipt.map_or(0, |receipt| receipt.pivots()),
            bytecode_max_pivots_per_chunk = bytecode_receipt
                .map_or(0, |receipt| receipt.max_pivots_per_chunk()),
            dynamic_threadgroup_bytes = threadgroup_bytes,
            static_threadgroup_bytes = limits.static_threadgroup_memory_length,
            threadgroup_memory_limit_bytes = maximum_threadgroup_bytes,
            wall_ns = duration_nanos(dispatch_started.elapsed()),
            storage_mode = "shared",
            "completed Instruction Read-RAF compatibility scatter"
        );
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
        let (bytecode_receipt, bytecode_carrier) = match (bytecode, bytecode_output_buffers) {
            (Some(request), Some((occurrences, magnitudes))) => {
                let topology = request.receipt();
                let fused = InstructionReadRafFusedBytecodeReceipt {
                    physical_rows: topology.physical_rows(),
                    work_items: topology.work_items(),
                    descriptor_elements: topology.descriptor_elements(),
                    descriptor_bytes: topology.descriptor_bytes(),
                    descriptor_identity: topology.descriptor_allocation_identity(),
                    pivot_elements: topology.pivot_elements(),
                    pivot_bytes: topology.pivot_bytes(),
                    pivot_identity: topology.pivot_allocation_identity(),
                    chunk_offset_elements: topology.chunk_offset_elements(),
                    chunk_offset_bytes: topology.chunk_offset_bytes(),
                    chunk_offset_identity: topology.chunk_offset_allocation_identity(),
                    work_item_bytes: topology.work_item_bytes(),
                    work_item_identity: topology.work_item_allocation_identity(),
                    address_offset_elements: topology.address_offset_elements(),
                    address_offset_bytes: topology.address_offset_bytes(),
                    address_offset_identity: topology.address_offset_allocation_identity(),
                    occurrence_bytes: usize::try_from(bytecode_occurrence_bytes)
                        .map_err(|_| MetalError::InputTooLong(topology.physical_rows()))?,
                    occurrence_identity: occurrences.as_ptr() as usize,
                    magnitude_bytes: usize::try_from(bytecode_magnitude_bytes)
                        .map_err(|_| MetalError::InputTooLong(topology.physical_rows()))?,
                    magnitude_identity: magnitudes.as_ptr() as usize,
                    max_descriptors_per_chunk: topology.max_descriptors_per_chunk(),
                    max_pivots_per_chunk: topology.max_pivots_per_chunk(),
                    dynamic_threadgroup_bytes: threadgroup_bytes,
                    max_admitted_descriptors_per_chunk: bytecode_max_admitted_descriptors_per_chunk,
                    max_admitted_pivots_per_chunk: BYTECODE_MAX_PIVOTS_PER_CHUNK,
                    threadgroup_memory_limit_bytes: dynamic_threadgroup_memory_limit,
                    shared_source_row_scans: topology.shared_source_row_scans(),
                    additional_source_row_scans: topology.additional_source_row_scans(),
                    member_upload_bytes: topology.member_upload_bytes(),
                };
                let carrier = request.publish(source_receipt, occurrences, magnitudes)?;
                (Some(fused), Some(carrier))
            }
            (None, None) => (None, None),
            _ => {
                return Err(invalid_scatter(
                    "fused bytecode request lost its output allocations",
                ));
            }
        };
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
            dispatches: 1,
            source_copy_bytes: 0,
            full_plane_readback_bytes: 0,
            complete_overwrite: true,
            bytecode: bytecode_receipt,
        };
        let planes = InstructionReadRafDenseGroupedPlanes {
            packed_rows,
            lookups,
            inverse,
            weights,
            receipt,
            bytecode_carrier,
        };
        tracing::info!(
            target: "jolt::metal",
            rows,
            wall_ns = duration_nanos(prepare_started.elapsed()),
            storage_mode = "shared",
            "prepared Instruction Read-RAF compatibility-scatter planes"
        );
        Ok(planes)
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

pub(crate) fn validate_bytecode_topology_admission(
    max_descriptors_per_chunk: usize,
    max_pivots_per_chunk: usize,
) -> Result<(), MetalError> {
    if !(1..=BYTECODE_MAX_DESCRIPTORS_PER_CHUNK).contains(&max_descriptors_per_chunk) {
        return Err(invalid_scatter(format!(
            "fused bytecode topology admission failed: clause=max_descriptors_per_chunk observed={max_descriptors_per_chunk} allowed=1..={BYTECODE_MAX_DESCRIPTORS_PER_CHUNK}"
        )));
    }
    if max_pivots_per_chunk > BYTECODE_MAX_PIVOTS_PER_CHUNK {
        return Err(invalid_scatter(format!(
            "fused bytecode topology admission failed: clause=max_pivots_per_chunk observed={max_pivots_per_chunk} allowed=0..={BYTECODE_MAX_PIVOTS_PER_CHUNK}"
        )));
    }
    Ok(())
}

fn bytecode_descriptor_capacity(
    dynamic_threadgroup_memory_limit: u64,
    max_pivots_per_chunk: usize,
) -> usize {
    let count_bytes =
        aligned_threadgroup_bytes((INSTRUCTION_READ_RAF_SEGMENTS * size_of::<u32>()) as u64);
    let pivot_bytes =
        aligned_threadgroup_bytes((max_pivots_per_chunk.max(1) * BYTECODE_PIVOT_BYTES) as u64);
    let descriptor_budget = dynamic_threadgroup_memory_limit
        .saturating_sub(count_bytes)
        .saturating_sub(pivot_bytes);
    let descriptor_elements = descriptor_budget / THREADGROUP_ALLOCATION_ALIGNMENT
        * (THREADGROUP_ALLOCATION_ALIGNMENT / BYTECODE_DESCRIPTOR_BYTES as u64);
    usize::try_from(descriptor_elements.saturating_sub(1))
        .unwrap_or(usize::MAX)
        .min(BYTECODE_MAX_DESCRIPTORS_PER_CHUNK)
}

const fn aligned_threadgroup_bytes(bytes: u64) -> u64 {
    bytes.div_ceil(THREADGROUP_ALLOCATION_ALIGNMENT) * THREADGROUP_ALLOCATION_ALIGNMENT
}

fn validate_bytecode_request(
    request: &BytecodeAddressFusedScatterRequest,
    source: InstructionReadRafStage1Receipt,
    padded_rows: usize,
    device_registry_id: u64,
) -> Result<(), MetalError> {
    let receipt = request.receipt();
    let physical_rows = receipt.physical_rows();
    let chunks = physical_rows.div_ceil(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS);
    let expected_address_offsets = (1usize << 13) + 1;
    let source_matches = request.source_receipt() == source
        && receipt.source_receipt() == source
        && receipt.source_generation() == source.source_generation()
        && receipt.source_completion_serial() == source.completion_serial()
        && receipt.source_rows_storage_id() == source.row_allocation_identity()
        && receipt.source_claim_storage_id() == source.claim_allocation_identity()
        && receipt.source_windows() == source.rows();
    let topology_shape = receipt.padded_rows() == padded_rows
        && receipt.shape().rows().is_ok_and(|rows| rows == padded_rows)
        && physical_rows != 0
        && physical_rows <= padded_rows
        && receipt.chunks() == chunks
        && receipt.descriptors() != 0
        && receipt.descriptor_elements() == receipt.descriptors() + chunks
        && receipt.pivot_elements() == receipt.pivots() + 1
        && receipt.chunk_offset_elements() == 2 * chunks
        && receipt.work_items() != 0
        && receipt.address_offset_elements() == expected_address_offsets;
    let buffers = [
        (
            request.descriptors_buffer(),
            receipt.descriptor_bytes(),
            receipt.descriptor_allocation_identity(),
        ),
        (
            request.pivots_buffer(),
            receipt.pivot_bytes(),
            receipt.pivot_allocation_identity(),
        ),
        (
            request.chunk_offsets_buffer(),
            receipt.chunk_offset_bytes(),
            receipt.chunk_offset_allocation_identity(),
        ),
        (
            request.work_items_buffer(),
            receipt.work_item_bytes(),
            receipt.work_item_allocation_identity(),
        ),
        (
            request.address_offsets_buffer(),
            receipt.address_offset_bytes(),
            receipt.address_offset_allocation_identity(),
        ),
    ];
    let buffers_match = buffers.iter().all(|(buffer, bytes, identity)| {
        *identity != 0
            && buffer.as_ptr() as usize == *identity
            && buffer.length() == *bytes as u64
            && buffer.device().registry_id() == device_registry_id
    });
    let byte_ledgers = receipt.descriptor_bytes()
        == receipt.descriptor_elements() * BYTECODE_DESCRIPTOR_BYTES
        && receipt.pivot_bytes() == receipt.pivot_elements() * BYTECODE_PIVOT_BYTES
        && receipt.chunk_offset_bytes() == receipt.chunk_offset_elements() * size_of::<u32>()
        && receipt.work_item_bytes() == receipt.work_items() * 8
        && receipt.address_offset_bytes() == receipt.address_offset_elements() * size_of::<u32>();
    let publication = receipt.device_registry_id() == device_registry_id
        && receipt.completion_serial() != 0
        && receipt.complete_overwrite()
        && receipt.covered_rows() == physical_rows
        && receipt.shared_source_row_scans() == 1
        && receipt.additional_source_row_scans() == 0
        && receipt.member_upload_bytes() == 0;
    if !source_matches {
        return Err(invalid_scatter(
            "fused bytecode topology source receipt does not match Stage-1",
        ));
    }
    if !topology_shape {
        return Err(invalid_scatter(format!(
            "fused bytecode topology shape is invalid: padded_rows={} physical_rows={} chunks={}/{} descriptors={} max_descriptors={} max_pivots={}",
            receipt.padded_rows(),
            physical_rows,
            receipt.chunks(),
            chunks,
            receipt.descriptors(),
            receipt.max_descriptors_per_chunk(),
            receipt.max_pivots_per_chunk(),
        )));
    }
    validate_bytecode_topology_admission(
        receipt.max_descriptors_per_chunk(),
        receipt.max_pivots_per_chunk(),
    )?;
    if !buffers_match {
        return Err(invalid_scatter(
            "fused bytecode topology buffer provenance does not match its receipt",
        ));
    }
    if !byte_ledgers {
        return Err(invalid_scatter(
            "fused bytecode topology byte ledgers do not match their element counts",
        ));
    }
    if !publication {
        return Err(invalid_scatter(
            "fused bytecode topology publication receipt is incomplete",
        ));
    }
    Ok(())
}

fn bytecode_threadgroup_bytes(
    receipt: &BytecodeAddressStage1TopologyReceipt,
) -> Result<(u64, u64), MetalError> {
    let descriptor_bytes = receipt
        .max_descriptors_per_chunk()
        .checked_add(1)
        .and_then(|elements| elements.checked_mul(BYTECODE_DESCRIPTOR_BYTES))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(receipt.physical_rows()))?;
    let pivot_bytes = receipt
        .max_pivots_per_chunk()
        .max(1)
        .checked_mul(BYTECODE_PIVOT_BYTES)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(receipt.physical_rows()))?;
    Ok((descriptor_bytes, pivot_bytes))
}

fn optional_shader_count(value: Option<usize>, rows: usize) -> Result<u32, MetalError> {
    value.map_or(Ok(0), |value| {
        u32::try_from(value).map_err(|_| MetalError::InputTooLong(rows))
    })
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

fn invalid_scatter(message: impl Into<String>) -> MetalError {
    MetalError::InvalidInstructionReadRafGrouped(message.into())
}

fn duration_nanos(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}
