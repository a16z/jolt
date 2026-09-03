use std::{cell::Cell, mem::size_of, slice, time::Duration};

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLResourceOptions, MTLSize, NSRange,
};

use super::{
    carrier::{AddressMajorShape, INNER_LOG2},
    worklist::{
        BytecodeAddressWorkItem, BYTECODE_ADDRESS_BASE_STAGES, BYTECODE_ADDRESS_PUSHFORWARD_STAGES,
    },
    worklist_runtime::{
        field_bytes, flatten_tables, padding_base_terms, shader_count, to_u64, validate_pipeline,
        validate_table_shape, BytecodeAddressSparseParams, BytecodeAddressSparseRuntimeError,
        REDUCE_PIPELINE, REDUCE_THREADS, WORKER_ITEMS_PER_THREADGROUP, WORKER_PIPELINE,
        WORKER_THREADS,
    },
};
use crate::metal::solinas::{
    buffer_from_slice, completed_command_gpu_time, set_inline_bytes, BooleanityRows, Fp128,
    SolinasMetal,
};

const COUNT_PIPELINE: &str = "solinas_bytecode_address_resident_count";
const SUMMARIZE_PIPELINE: &str = "solinas_bytecode_address_resident_summarize";
const LAYOUT_PIPELINE: &str = "solinas_bytecode_address_resident_layout";
const SCATTER_PIPELINE: &str = "solinas_bytecode_address_resident_scatter";
const COUNT_THREADS: usize = 256;
const SUMMARIZE_THREADS: usize = 256;
const LAYOUT_THREADS: usize = 64;
const SCATTER_THREADS: usize = 256;
const MAX_MEMBER_OWNED_BYTES: usize = 6 << 30;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct BytecodeAddressResidentStatus {
    invalid_rows: u32,
    first_push_pc: u32,
}

const _: [(); 8] = [(); size_of::<BytecodeAddressResidentStatus>()];

struct BytecodeAddressResidentBuffers {
    counts: Buffer,
    status: Buffer,
    occurrences: Buffer,
    magnitudes: Buffer,
    work_items: Buffer,
    item_cursors: Buffer,
    address_offsets: Buffer,
    e_lo: Buffer,
    e_hi: Buffer,
    padding: Buffer,
    partials: Buffer,
    output: Buffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BytecodeAddressResidentStats {
    work_items: usize,
    member_owned_bytes: usize,
    count_gpu_time: Duration,
    reduce_gpu_time: Duration,
}

impl BytecodeAddressResidentStats {
    copy_field_getters! { pub(crate), {
        work_items: usize,
        member_owned_bytes: usize,
        count_gpu_time: Duration,
        reduce_gpu_time: Duration,
    } }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BytecodeAddressResidentObservation {
    pub(crate) output: Vec<AkitaField>,
    pub(crate) first_push_pc: usize,
    pub(crate) stats: BytecodeAddressResidentStats,
}

pub(crate) struct BytecodeAddressResidentInvocation {
    context: SolinasMetal,
    resident_rows: BooleanityRows,
    layout_pipeline: ComputePipelineState,
    scatter_pipeline: ComputePipelineState,
    worker_pipeline: ComputePipelineState,
    reduce_pipeline: ComputePipelineState,
    buffers: BytecodeAddressResidentBuffers,
    params: BytecodeAddressSparseParams,
    address_offsets: Vec<u32>,
    first_push_pc: usize,
    member_owned_bytes: usize,
    count_gpu_time: Duration,
    completed: Cell<bool>,
}

impl SolinasMetal {
    pub(crate) fn prepare_bytecode_address_resident(
        &self,
        resident_rows: BooleanityRows,
        log_addresses: u32,
        e_lo: &[Vec<AkitaField>],
        e_hi: &[Vec<AkitaField>],
    ) -> Result<BytecodeAddressResidentInvocation, BytecodeAddressSparseRuntimeError> {
        self.validate_booleanity_rows(&resident_rows)?;
        let rows = resident_rows.len();
        if rows < 1usize << INNER_LOG2 || !rows.is_power_of_two() {
            return Err(BytecodeAddressSparseRuntimeError::InvalidRows {
                physical: rows,
                padded: rows.next_power_of_two(),
            });
        }
        let shape = AddressMajorShape::new(rows.ilog2(), log_addresses, INNER_LOG2)?;
        let addresses = shape.addresses()?;
        let inner_length = shape.inner_length()?;
        let outer_length = shape.outer_length()?;
        validate_table_shape("resident E_lo", e_lo, inner_length)?;
        validate_table_shape("resident E_hi", e_hi, outer_length)?;

        let count_cells = addresses.checked_mul(outer_length).ok_or(
            BytecodeAddressSparseRuntimeError::SizeOverflow("resident count cells"),
        )?;
        let count_bytes = count_cells.checked_mul(size_of::<u32>()).ok_or(
            BytecodeAddressSparseRuntimeError::SizeOverflow("resident count bytes"),
        )?;
        let summary_elements =
            addresses
                .checked_mul(2)
                .ok_or(BytecodeAddressSparseRuntimeError::SizeOverflow(
                    "resident summary elements",
                ))?;
        let summary_bytes = summary_elements.checked_mul(size_of::<u32>()).ok_or(
            BytecodeAddressSparseRuntimeError::SizeOverflow("resident summary bytes"),
        )?;
        let status_bytes = size_of::<BytecodeAddressResidentStatus>();
        let initial_bytes = checked_sum(&[count_bytes, summary_bytes, status_bytes])?;
        enforce_member_cap(initial_bytes)?;
        self.validate_additional_working_set(to_u64(
            "resident initial working set",
            initial_bytes,
        )?)?;
        for bytes in [count_bytes, summary_bytes, status_bytes] {
            self.validate_buffer_length(to_u64("resident buffer", bytes)?)?;
        }

        let count_pipeline = self.compile_named_pipeline(COUNT_PIPELINE)?;
        let summarize_pipeline = self.compile_named_pipeline(SUMMARIZE_PIPELINE)?;
        let layout_pipeline = self.compile_named_pipeline(LAYOUT_PIPELINE)?;
        let scatter_pipeline = self.compile_named_pipeline(SCATTER_PIPELINE)?;
        let worker_pipeline = self.compile_named_pipeline(WORKER_PIPELINE)?;
        let reduce_pipeline = self.compile_named_pipeline(REDUCE_PIPELINE)?;
        for (name, pipeline, threads) in [
            (COUNT_PIPELINE, &count_pipeline, COUNT_THREADS),
            (SUMMARIZE_PIPELINE, &summarize_pipeline, SUMMARIZE_THREADS),
            (LAYOUT_PIPELINE, &layout_pipeline, LAYOUT_THREADS),
            (SCATTER_PIPELINE, &scatter_pipeline, SCATTER_THREADS),
            (WORKER_PIPELINE, &worker_pipeline, WORKER_THREADS),
            (REDUCE_PIPELINE, &reduce_pipeline, REDUCE_THREADS),
        ] {
            validate_pipeline(name, SolinasMetal::limits(pipeline), threads)?;
        }

        let counts = self
            .device
            .new_buffer(count_bytes as u64, MTLResourceOptions::StorageModePrivate);
        let summary = buffer_from_slice(&self.device, &vec![0u32; summary_elements]);
        let status = buffer_from_slice(&self.device, &[BytecodeAddressResidentStatus::default()]);
        let mut params = BytecodeAddressSparseParams {
            physical_rows: shader_count("resident physical rows", rows)?,
            addresses: shader_count("resident addresses", addresses)?,
            inner_length: shader_count("resident inner length", inner_length)?,
            outer_length: shader_count("resident outer length", outer_length)?,
            work_items: 0,
            stages: BYTECODE_ADDRESS_PUSHFORWARD_STAGES as u32,
            base_stages: BYTECODE_ADDRESS_BASE_STAGES as u32,
            reserved: 0,
        };

        let count_gpu_time = autoreleasepool(|| {
            let command = self.queue.new_command_buffer();
            let blit = command.new_blit_command_encoder();
            blit.fill_buffer(&counts, NSRange::new(0, counts.length()), 0);
            blit.end_encoding();

            let count = command.new_compute_command_encoder();
            count.set_compute_pipeline_state(&count_pipeline);
            count.set_buffer(0, Some(resident_rows.buffer()), 0);
            count.set_buffer(1, Some(&counts), 0);
            count.set_buffer(2, Some(&status), 0);
            set_inline_bytes(count, 3, &params);
            count.dispatch_thread_groups(
                one_dimensional_groups(rows, COUNT_THREADS),
                one_dimensional_threads(COUNT_THREADS),
            );
            count.end_encoding();

            let summarize = command.new_compute_command_encoder();
            summarize.set_compute_pipeline_state(&summarize_pipeline);
            summarize.set_buffer(0, Some(&counts), 0);
            summarize.set_buffer(1, Some(&summary), 0);
            set_inline_bytes(summarize, 2, &params);
            summarize.dispatch_thread_groups(
                one_dimensional_groups(addresses, SUMMARIZE_THREADS),
                one_dimensional_threads(SUMMARIZE_THREADS),
            );
            summarize.end_encoding();

            command.commit();
            command.wait_until_completed();
            completed_command_gpu_time(command)
        })?;

        let status_value = read_status(&status)?;
        if status_value.invalid_rows != 0 || status_value.first_push_pc as usize >= addresses {
            return Err(BytecodeAddressSparseRuntimeError::InvalidResidentState {
                invalid_rows: status_value.invalid_rows,
            });
        }
        let summary_values = read_u32s(&summary, summary_elements)?;
        let (item_counts, populations) = summary_values.split_at(addresses);
        let population = populations.iter().try_fold(0usize, |sum, count| {
            sum.checked_add(*count as usize)
                .ok_or(BytecodeAddressSparseRuntimeError::SizeOverflow(
                    "resident population",
                ))
        })?;
        if population != rows {
            return Err(BytecodeAddressSparseRuntimeError::InvalidResidentSummary {
                expected_rows: rows,
                summarized_rows: population,
            });
        }
        let address_offsets = prefix_offsets(item_counts)?;
        let work_items = *address_offsets
            .last()
            .ok_or(BytecodeAddressSparseRuntimeError::InvalidState)?
            as usize;
        if work_items == 0 {
            return Err(BytecodeAddressSparseRuntimeError::InvalidState);
        }
        params.work_items = shader_count("resident work items", work_items)?;

        let occurrence_bytes = rows.checked_mul(size_of::<u16>()).ok_or(
            BytecodeAddressSparseRuntimeError::SizeOverflow("resident occurrence bytes"),
        )?;
        let magnitude_bytes = rows.checked_mul(size_of::<u64>()).ok_or(
            BytecodeAddressSparseRuntimeError::SizeOverflow("resident magnitude bytes"),
        )?;
        let work_item_bytes = work_items
            .checked_mul(size_of::<BytecodeAddressWorkItem>())
            .ok_or(BytecodeAddressSparseRuntimeError::SizeOverflow(
                "resident work-item bytes",
            ))?;
        let cursor_bytes = addresses.checked_mul(size_of::<u32>()).ok_or(
            BytecodeAddressSparseRuntimeError::SizeOverflow("resident cursor bytes"),
        )?;
        let address_offset_bytes = address_offsets.len().checked_mul(size_of::<u32>()).ok_or(
            BytecodeAddressSparseRuntimeError::SizeOverflow("resident address-offset bytes"),
        )?;
        let flat_e_lo = flatten_tables(e_lo);
        let flat_e_hi = flatten_tables(e_hi);
        let padding = padding_base_terms(rows, e_lo, e_hi)?;
        self.validate_inputs("resident bytecode E_lo", &flat_e_lo)?;
        self.validate_inputs("resident bytecode E_hi", &flat_e_hi)?;
        self.validate_inputs("resident bytecode padding", &padding)?;
        let equality_bytes = field_bytes(flat_e_lo.len().checked_add(flat_e_hi.len()).ok_or(
            BytecodeAddressSparseRuntimeError::SizeOverflow("resident equality fields"),
        )?)?;
        let padding_bytes = field_bytes(padding.len())?;
        let partial_bytes = field_bytes(
            BYTECODE_ADDRESS_PUSHFORWARD_STAGES
                .checked_mul(work_items)
                .ok_or(BytecodeAddressSparseRuntimeError::SizeOverflow(
                    "resident partial fields",
                ))?,
        )?;
        let output_bytes = field_bytes(
            BYTECODE_ADDRESS_PUSHFORWARD_STAGES
                .checked_mul(addresses)
                .ok_or(BytecodeAddressSparseRuntimeError::SizeOverflow(
                    "resident output fields",
                ))?,
        )?;
        let remaining_bytes = checked_sum(&[
            occurrence_bytes,
            magnitude_bytes,
            work_item_bytes,
            cursor_bytes,
            address_offset_bytes,
            equality_bytes,
            padding_bytes,
            partial_bytes,
            output_bytes,
        ])?;
        let member_owned_bytes = initial_bytes.checked_add(remaining_bytes).ok_or(
            BytecodeAddressSparseRuntimeError::SizeOverflow("resident member-owned bytes"),
        )?;
        enforce_member_cap(member_owned_bytes)?;
        self.validate_additional_working_set(to_u64(
            "resident remaining working set",
            remaining_bytes,
        )?)?;
        for bytes in [
            occurrence_bytes,
            magnitude_bytes,
            work_item_bytes,
            cursor_bytes,
            address_offset_bytes,
            equality_bytes,
            padding_bytes,
            partial_bytes,
            output_bytes,
        ] {
            self.validate_buffer_length(to_u64("resident buffer", bytes)?)?;
        }

        let item_cursors = buffer_from_slice(&self.device, &address_offsets[..addresses]);
        Ok(BytecodeAddressResidentInvocation {
            context: self.clone(),
            resident_rows,
            layout_pipeline,
            scatter_pipeline,
            worker_pipeline,
            reduce_pipeline,
            buffers: BytecodeAddressResidentBuffers {
                counts,
                status,
                occurrences: self.device.new_buffer(
                    occurrence_bytes as u64,
                    MTLResourceOptions::StorageModePrivate,
                ),
                magnitudes: self.device.new_buffer(
                    magnitude_bytes as u64,
                    MTLResourceOptions::StorageModePrivate,
                ),
                work_items: self.device.new_buffer(
                    work_item_bytes as u64,
                    MTLResourceOptions::StorageModePrivate,
                ),
                item_cursors,
                address_offsets: buffer_from_slice(&self.device, &address_offsets),
                e_lo: buffer_from_slice(&self.device, &flat_e_lo),
                e_hi: buffer_from_slice(&self.device, &flat_e_hi),
                padding: buffer_from_slice(&self.device, &padding),
                partials: self
                    .device
                    .new_buffer(partial_bytes as u64, MTLResourceOptions::StorageModePrivate),
                output: self
                    .device
                    .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared),
            },
            params,
            address_offsets,
            first_push_pc: status_value.first_push_pc as usize,
            member_owned_bytes,
            count_gpu_time,
            completed: Cell::new(false),
        })
    }
}

impl BytecodeAddressResidentInvocation {
    pub(crate) fn execute_timed(
        &self,
    ) -> Result<BytecodeAddressResidentObservation, BytecodeAddressSparseRuntimeError> {
        if self.completed.replace(true) {
            return Err(BytecodeAddressSparseRuntimeError::InvalidState);
        }
        let reduce_gpu_time = autoreleasepool(|| {
            let command = self.context.queue.new_command_buffer();

            let layout = command.new_compute_command_encoder();
            layout.set_compute_pipeline_state(&self.layout_pipeline);
            layout.set_buffer(0, Some(&self.buffers.counts), 0);
            layout.set_buffer(1, Some(&self.buffers.work_items), 0);
            layout.set_buffer(2, Some(&self.buffers.item_cursors), 0);
            layout.set_buffer(3, Some(&self.buffers.status), 0);
            set_inline_bytes(layout, 4, &self.params);
            layout.dispatch_thread_groups(
                one_dimensional_groups(self.params.outer_length as usize, LAYOUT_THREADS),
                one_dimensional_threads(LAYOUT_THREADS),
            );
            layout.end_encoding();

            let scatter = command.new_compute_command_encoder();
            scatter.set_compute_pipeline_state(&self.scatter_pipeline);
            scatter.set_buffer(0, Some(self.resident_rows.buffer()), 0);
            scatter.set_buffer(1, Some(&self.buffers.counts), 0);
            scatter.set_buffer(2, Some(&self.buffers.occurrences), 0);
            scatter.set_buffer(3, Some(&self.buffers.magnitudes), 0);
            scatter.set_buffer(4, Some(&self.buffers.status), 0);
            set_inline_bytes(scatter, 5, &self.params);
            scatter.dispatch_thread_groups(
                one_dimensional_groups(self.params.physical_rows as usize, SCATTER_THREADS),
                one_dimensional_threads(SCATTER_THREADS),
            );
            scatter.end_encoding();

            let worker = command.new_compute_command_encoder();
            worker.set_compute_pipeline_state(&self.worker_pipeline);
            worker.set_buffer(0, Some(&self.buffers.occurrences), 0);
            worker.set_buffer(1, Some(&self.buffers.magnitudes), 0);
            worker.set_buffer(2, Some(&self.buffers.work_items), 0);
            worker.set_buffer(3, Some(&self.buffers.e_lo), 0);
            worker.set_buffer(4, Some(&self.buffers.e_hi), 0);
            worker.set_buffer(5, Some(&self.buffers.partials), 0);
            set_inline_bytes(worker, 6, &self.params);
            worker.dispatch_thread_groups(
                MTLSize {
                    width: u64::from(self.params.work_items)
                        .div_ceil(WORKER_ITEMS_PER_THREADGROUP as u64),
                    height: 1,
                    depth: 1,
                },
                one_dimensional_threads(WORKER_THREADS),
            );
            worker.end_encoding();

            let reduce = command.new_compute_command_encoder();
            reduce.set_compute_pipeline_state(&self.reduce_pipeline);
            reduce.set_buffer(0, Some(&self.buffers.partials), 0);
            reduce.set_buffer(1, Some(&self.buffers.address_offsets), 0);
            reduce.set_buffer(2, Some(&self.buffers.padding), 0);
            reduce.set_buffer(3, Some(&self.buffers.output), 0);
            set_inline_bytes(reduce, 4, &self.params);
            let output_fields = self.params.stages as usize * self.params.addresses as usize;
            reduce.dispatch_thread_groups(
                one_dimensional_groups(output_fields, REDUCE_THREADS),
                one_dimensional_threads(REDUCE_THREADS),
            );
            reduce.end_encoding();

            command.commit();
            command.wait_until_completed();
            completed_command_gpu_time(command)
        })?;

        let status = read_status(&self.buffers.status)?;
        if status.invalid_rows != 0 || status.first_push_pc as usize != self.first_push_pc {
            return Err(BytecodeAddressSparseRuntimeError::InvalidResidentState {
                invalid_rows: status.invalid_rows,
            });
        }
        let cursors = read_u32s(&self.buffers.item_cursors, self.params.addresses as usize)?;
        if cursors
            .iter()
            .zip(&self.address_offsets[1..])
            .any(|(actual, expected)| actual != expected)
        {
            return Err(BytecodeAddressSparseRuntimeError::InvalidState);
        }
        let fields = self.params.stages as usize * self.params.addresses as usize;
        let output = read_fp128s(&self.buffers.output, fields)?;
        self.context
            .validate_inputs("resident bytecode output", output)?;
        Ok(BytecodeAddressResidentObservation {
            output: output
                .iter()
                .map(|value| (*value).into_jolt_field())
                .collect(),
            first_push_pc: self.first_push_pc,
            stats: BytecodeAddressResidentStats {
                work_items: self.params.work_items as usize,
                member_owned_bytes: self.member_owned_bytes,
                count_gpu_time: self.count_gpu_time,
                reduce_gpu_time,
            },
        })
    }
}

fn prefix_offsets(counts: &[u32]) -> Result<Vec<u32>, BytecodeAddressSparseRuntimeError> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    offsets.push(0u32);
    for count in counts {
        let next = offsets
            .last()
            .copied()
            .ok_or(BytecodeAddressSparseRuntimeError::InvalidState)?
            .checked_add(*count)
            .ok_or(BytecodeAddressSparseRuntimeError::SizeOverflow(
                "resident work-item prefix",
            ))?;
        offsets.push(next);
    }
    Ok(offsets)
}

fn checked_sum(values: &[usize]) -> Result<usize, BytecodeAddressSparseRuntimeError> {
    values.iter().try_fold(0usize, |sum, value| {
        sum.checked_add(*value)
            .ok_or(BytecodeAddressSparseRuntimeError::SizeOverflow(
                "resident allocation sum",
            ))
    })
}

fn enforce_member_cap(bytes: usize) -> Result<(), BytecodeAddressSparseRuntimeError> {
    if bytes > MAX_MEMBER_OWNED_BYTES {
        return Err(BytecodeAddressSparseRuntimeError::ResidentStorageTooLarge {
            bytes,
            maximum: MAX_MEMBER_OWNED_BYTES,
        });
    }
    Ok(())
}

fn one_dimensional_groups(elements: usize, threads: usize) -> MTLSize {
    MTLSize {
        width: elements.div_ceil(threads) as u64,
        height: 1,
        depth: 1,
    }
}

const fn one_dimensional_threads(threads: usize) -> MTLSize {
    MTLSize {
        width: threads as u64,
        height: 1,
        depth: 1,
    }
}

fn read_status(
    buffer: &Buffer,
) -> Result<BytecodeAddressResidentStatus, BytecodeAddressSparseRuntimeError> {
    if buffer.length() != size_of::<BytecodeAddressResidentStatus>() as u64 {
        return Err(BytecodeAddressSparseRuntimeError::InvalidState);
    }
    // SAFETY: the shared buffer has exactly one initialized status value and
    // all commands touching it have completed before host access.
    Ok(unsafe { *buffer.contents().cast::<BytecodeAddressResidentStatus>() })
}

fn read_u32s(
    buffer: &Buffer,
    elements: usize,
) -> Result<&[u32], BytecodeAddressSparseRuntimeError> {
    if buffer.length() != (elements * size_of::<u32>()) as u64 {
        return Err(BytecodeAddressSparseRuntimeError::InvalidState);
    }
    // SAFETY: the shared buffer holds exactly `elements` u32 values and the
    // producing command has completed before host access.
    Ok(unsafe { slice::from_raw_parts(buffer.contents().cast::<u32>(), elements) })
}

fn read_fp128s(
    buffer: &Buffer,
    elements: usize,
) -> Result<&[Fp128], BytecodeAddressSparseRuntimeError> {
    if buffer.length() != (elements * size_of::<Fp128>()) as u64 {
        return Err(BytecodeAddressSparseRuntimeError::InvalidState);
    }
    // SAFETY: the shared buffer holds exactly `elements` Fp128 values and the
    // producing command has completed before host access.
    Ok(unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), elements) })
}
