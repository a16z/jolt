//! Opt-in grouped Read-RAF phase probe.
//!
//! Production keeps the established sequence until a producer-owned layout
//! meets the full 16-phase wall-time gate; host repacking is not admissible.

use std::{mem::size_of, ops::Range, slice, time::Duration};

use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::{byte_length, set_inline_bytes, AddressPhaseSums};
use crate::metal::solinas::instruction_read_raf_v3::{
    AddressJob, AtomPhaseParams, SuffixPlan, ATOM_PHASE_PIPELINE, FINALIZE_RAF_PIPELINE,
    FINALIZE_SUFFIX_PIPELINE, JOB_FIELDS, PHASE_THREADGROUP_BYTES, SEGMENTS, SEGMENT_OFFSETS,
    SIMD_WIDTH, TABLES, TOTAL_SUFFIXES,
};

use super::super::{
    buffer_from_slice, command_buffer_timestamp, AddressRafSums, AddressSuffixFullSums, Fp128,
    MetalError, SolinasMetal, ADDRESS_RAF_BINS, ADDRESS_RAF_LANES,
};

struct GroupedPhaseBuffers {
    jobs: Buffer,
    job_offsets: Buffer,
    suffix_kinds: Buffer,
    suffix_counts: Buffer,
    suffix_descriptors: Buffer,
    suffix_output_lanes: Buffer,
    partials: Buffer,
    raf_output: Buffer,
    suffix_output: Buffer,
}

pub(super) struct GroupedAddressPhase {
    context: SolinasMetal,
    phase_pipeline: ComputePipelineState,
    raf_finalize_pipeline: ComputePipelineState,
    suffix_finalize_pipeline: ComputePipelineState,
    buffers: GroupedPhaseBuffers,
    job_count: usize,
    phase_threads: usize,
    raf_finalize_threads: usize,
    suffix_finalize_threads: usize,
}

impl GroupedAddressPhase {
    pub(super) fn prepare(
        context: &SolinasMetal,
        rows: usize,
        segment_ranges: &[Range<usize>; SEGMENTS],
        rows_per_job: usize,
        requested_threads: Option<usize>,
    ) -> Result<Self, MetalError> {
        validate_ranges(rows, segment_ranges)?;
        let mut jobs = Vec::with_capacity(rows.div_ceil(rows_per_job) + SEGMENTS);
        let mut job_offsets = [0u32; SEGMENT_OFFSETS];
        for (segment, range) in segment_ranges.iter().enumerate() {
            for start in (range.start..range.end).step_by(rows_per_job) {
                let end = (start + rows_per_job).min(range.end);
                jobs.push(AddressJob {
                    start: shader_u32(start)?,
                    end: shader_u32(end)?,
                    segment: shader_u32(segment)?,
                    reserved: 0,
                });
            }
            job_offsets[segment + 1] = shader_u32(jobs.len())?;
        }
        if jobs.is_empty() {
            return Err(MetalError::EmptyInput);
        }

        let suffix_plan = SuffixPlan::production()
            .map_err(|error| MetalError::InvalidInstructionReadRafGrouped(error.to_string()))?;
        let phase_pipeline = context.compile_named_pipeline(ATOM_PHASE_PIPELINE)?;
        let raf_finalize_pipeline = context.compile_named_pipeline(FINALIZE_RAF_PIPELINE)?;
        let suffix_finalize_pipeline = context.compile_named_pipeline(FINALIZE_SUFFIX_PIPELINE)?;
        for (pipeline, state) in [
            (ATOM_PHASE_PIPELINE, &phase_pipeline),
            (FINALIZE_RAF_PIPELINE, &raf_finalize_pipeline),
            (FINALIZE_SUFFIX_PIPELINE, &suffix_finalize_pipeline),
        ] {
            let limits = SolinasMetal::limits(state);
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedAddressRafExecutionWidth {
                    pipeline,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let phase_threads = SolinasMetal::resolve_threadgroup_width(
            requested_threads,
            SolinasMetal::limits(&phase_pipeline),
        )?;
        let raf_finalize_threads = SolinasMetal::resolve_threadgroup_width(
            Some(ADDRESS_RAF_BINS),
            SolinasMetal::limits(&raf_finalize_pipeline),
        )?;
        let suffix_finalize_threads = SolinasMetal::resolve_threadgroup_width(
            Some(4 * ADDRESS_RAF_BINS),
            SolinasMetal::limits(&suffix_finalize_pipeline),
        )?;
        let maximum_threadgroup_bytes = context.device.max_threadgroup_memory_length();
        if PHASE_THREADGROUP_BYTES as u64 > maximum_threadgroup_bytes {
            return Err(MetalError::AddressRafDirectThreadgroupMemory {
                requested: PHASE_THREADGROUP_BYTES as u64,
                maximum: maximum_threadgroup_bytes,
            });
        }

        let partial_elements = jobs
            .len()
            .checked_mul(JOB_FIELDS)
            .ok_or(MetalError::InputTooLong(jobs.len()))?;
        let lengths = [
            byte_length::<AddressJob>(jobs.len())?,
            byte_length::<u32>(job_offsets.len())?,
            byte_length::<u8>(suffix_plan.explicit_kinds().len())?,
            byte_length::<u8>(suffix_plan.explicit_counts().len())?,
            byte_length::<u8>(suffix_plan.output_lanes().len())?,
            byte_length::<Fp128>(partial_elements)?,
            byte_length::<Fp128>(ADDRESS_RAF_LANES * ADDRESS_RAF_BINS)?,
            byte_length::<Fp128>(TOTAL_SUFFIXES * ADDRESS_RAF_BINS)?,
        ];
        for requested in lengths {
            let maximum = context.device.max_buffer_length();
            if requested > maximum {
                return Err(MetalError::BufferTooLong { requested, maximum });
            }
        }

        Ok(Self {
            context: context.clone(),
            phase_pipeline,
            raf_finalize_pipeline,
            suffix_finalize_pipeline,
            buffers: GroupedPhaseBuffers {
                jobs: buffer_from_slice(&context.device, &jobs),
                job_offsets: buffer_from_slice(&context.device, &job_offsets),
                suffix_kinds: buffer_from_slice(&context.device, suffix_plan.explicit_kinds()),
                suffix_counts: buffer_from_slice(&context.device, suffix_plan.explicit_counts()),
                suffix_descriptors: buffer_from_slice(&context.device, suffix_plan.descriptors()),
                suffix_output_lanes: buffer_from_slice(&context.device, suffix_plan.output_lanes()),
                partials: context.device.new_buffer(
                    byte_length::<Fp128>(partial_elements)?,
                    MTLResourceOptions::StorageModeShared,
                ),
                raf_output: context.device.new_buffer(
                    byte_length::<Fp128>(ADDRESS_RAF_LANES * ADDRESS_RAF_BINS)?,
                    MTLResourceOptions::StorageModeShared,
                ),
                suffix_output: context.device.new_buffer(
                    byte_length::<Fp128>(TOTAL_SUFFIXES * ADDRESS_RAF_BINS)?,
                    MTLResourceOptions::StorageModeShared,
                ),
            },
            job_count: jobs.len(),
            phase_threads,
            raf_finalize_threads,
            suffix_finalize_threads,
        })
    }

    pub(super) fn execute(
        &self,
        lookups: &Buffer,
        weights: &Buffer,
        previous_phase_table: &Buffer,
        suffix_len: u32,
        table_offsets: &[usize],
    ) -> Result<AddressPhaseSums, MetalError> {
        let params = AtomPhaseParams::grouped(suffix_len as usize, self.job_count)
            .map_err(|error| MetalError::InvalidInstructionReadRafGrouped(error.to_string()))?;
        let command_buffer = self.context.queue.new_command_buffer();
        autoreleasepool(|| {
            let phase = command_buffer.new_compute_command_encoder();
            phase.set_compute_pipeline_state(&self.phase_pipeline);
            phase.set_buffer(0, Some(lookups), 0);
            phase.set_buffer(1, Some(weights), 0);
            phase.set_buffer(2, Some(previous_phase_table), 0);
            phase.set_buffer(3, Some(&self.buffers.jobs), 0);
            phase.set_buffer(4, Some(&self.buffers.suffix_kinds), 0);
            phase.set_buffer(5, Some(&self.buffers.suffix_counts), 0);
            phase.set_buffer(6, Some(&self.buffers.partials), 0);
            set_inline_bytes(phase, 7, &params);
            phase.set_threadgroup_memory_length(0, PHASE_THREADGROUP_BYTES as u64);
            phase.dispatch_thread_groups(
                MTLSize {
                    width: self.job_count as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.phase_threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            phase.end_encoding();

            let raf = command_buffer.new_compute_command_encoder();
            raf.set_compute_pipeline_state(&self.raf_finalize_pipeline);
            raf.set_buffer(0, Some(&self.buffers.partials), 0);
            raf.set_buffer(1, Some(&self.buffers.job_offsets), 0);
            raf.set_buffer(2, Some(&self.buffers.raf_output), 0);
            raf.dispatch_thread_groups(
                MTLSize {
                    width: ADDRESS_RAF_LANES as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.raf_finalize_threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            raf.end_encoding();

            let suffix = command_buffer.new_compute_command_encoder();
            suffix.set_compute_pipeline_state(&self.suffix_finalize_pipeline);
            suffix.set_buffer(0, Some(&self.buffers.partials), 0);
            suffix.set_buffer(1, Some(&self.buffers.job_offsets), 0);
            suffix.set_buffer(2, Some(&self.buffers.suffix_descriptors), 0);
            suffix.set_buffer(3, Some(&self.buffers.suffix_output_lanes), 0);
            suffix.set_buffer(4, Some(&self.buffers.suffix_output), 0);
            suffix.dispatch_thread_groups(
                MTLSize {
                    width: TABLES as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.suffix_finalize_threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            suffix.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(MetalError::CommandFailed(command_buffer.status()));
        }
        let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
        let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
        if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
            return Err(MetalError::InvalidGpuTimestamps { start, end });
        }
        let gpu_active_time = Duration::from_secs_f64(end - start);
        let raf_elements = ADDRESS_RAF_LANES * ADDRESS_RAF_BINS;
        let suffix_elements = TOTAL_SUFFIXES * ADDRESS_RAF_BINS;
        // SAFETY: the shared outputs have these fixed capacities and the
        // command completed before either slice is observed.
        let raf = unsafe {
            slice::from_raw_parts(
                self.buffers.raf_output.contents().cast::<Fp128>(),
                raf_elements,
            )
        };
        // SAFETY: see the synchronization and capacity argument above.
        let suffix = unsafe {
            slice::from_raw_parts(
                self.buffers.suffix_output.contents().cast::<Fp128>(),
                suffix_elements,
            )
        };
        self.context
            .validate_inputs("grouped InstructionReadRaf RAF output", raf)?;
        self.context
            .validate_inputs("grouped InstructionReadRaf suffix output", suffix)?;
        Ok(AddressPhaseSums::from_parts(
            AddressRafSums::from_values(raf.to_vec()),
            AddressSuffixFullSums::from_values(suffix.to_vec(), table_offsets.to_vec()),
            gpu_active_time,
        ))
    }
}

fn validate_ranges(rows: usize, ranges: &[Range<usize>; SEGMENTS]) -> Result<(), MetalError> {
    if ranges
        .iter()
        .any(|range| range.end < range.start || range.end > rows)
    {
        return Err(MetalError::AddressPhaseLayoutLength {
            expected: rows,
            got: 0,
        });
    }
    let mut ordered: Vec<_> = ranges
        .iter()
        .filter(|range| !range.is_empty())
        .cloned()
        .collect();
    ordered.sort_unstable_by_key(|range| range.start);
    let mut cursor = 0usize;
    for range in ordered {
        if range.start != cursor {
            return Err(MetalError::AddressPhaseLayoutLength {
                expected: rows,
                got: cursor,
            });
        }
        cursor = range.end;
    }
    if cursor != rows {
        return Err(MetalError::AddressPhaseLayoutLength {
            expected: rows,
            got: cursor,
        });
    }
    Ok(())
}

fn shader_u32(value: usize) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| MetalError::InputTooLong(value))
}

const _: () = assert!(size_of::<AddressJob>() == 16);

#[cfg(test)]
mod tests {
    use super::validate_ranges;

    #[test]
    fn empty_segments_do_not_break_range_validation() {
        let mut ranges = std::array::from_fn(|_| 0..0);
        ranges[17] = 0..4;
        ranges[81] = 4..9;

        assert!(validate_ranges(9, &ranges).is_ok());
    }
}
