use std::{mem::size_of, slice, time::Duration};

use jolt_field::AkitaField;
use metal::{
    objc::rc::autoreleasepool, Buffer, MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

use super::{
    compile_probe::{HammingWeightProbeContext, HammingWeightProbePipelines},
    model::HammingWeightCensus,
    HammingWeightAuditRow, HammingWeightCompileReport, HammingWeightHistogramParams,
    HammingWeightProtocolTopology, HammingWeightResidentRow, HammingWeightSlicePlan,
    HammingWeightStatus, HammingWeightSuccessorConfig, FINALIZE_PIPELINE, HAMMING_WEIGHT_BINS,
    HAMMING_WEIGHT_SIMD_WIDTH, HAMMING_WEIGHT_THREADS, HISTOGRAM_PIPELINE,
};
use crate::metal::solinas::{
    buffer_from_slice, command_buffer_timestamp, validate_working_set, Fp128, MetalError,
    PipelineLimits, AKITA_OFFSET_FFFFA7F7,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HammingWeightFixtureExecution {
    pub compile: HammingWeightCompileReport,
    pub gpu_active: Duration,
    pub masses: Vec<AkitaField>,
    pub audit_rows: Vec<HammingWeightAuditRow>,
    pub status: HammingWeightStatus,
    pub census: HammingWeightCensus,
}

pub fn execute_hamming_weight_claim_reduction_fixture(
    rows: &[HammingWeightResidentRow],
    e_in: &[AkitaField],
    e_out: &[AkitaField],
    config: HammingWeightSuccessorConfig,
) -> Result<HammingWeightFixtureExecution, MetalError> {
    let plan = HammingWeightSlicePlan::new(
        rows.len(),
        config,
        HammingWeightProtocolTopology::PRODUCTION,
    )?;
    let requirements = plan.requirements();
    requirements.validate(super::HammingWeightBufferLengths {
        resident_rows: rows.len(),
        e_in: e_in.len(),
        e_out: e_out.len(),
        partials: requirements.partials,
        output: requirements.output,
        audit_rows: requirements.audit_rows,
        status: requirements.status,
    })?;

    let context = HammingWeightProbeContext::new()?;
    let pipelines = context.compile_pipelines()?;
    validate_pipeline_admission(&pipelines.report)?;
    let e_in = encode_fields("Hamming-weight inner equality", e_in)?;
    let e_out = encode_fields("Hamming-weight outer equality", e_out)?;

    let resident_bytes = byte_length::<HammingWeightResidentRow>(requirements.resident_rows)?;
    let e_in_bytes = byte_length::<Fp128>(requirements.e_in)?;
    let e_out_bytes = byte_length::<Fp128>(requirements.e_out)?;
    let partial_bytes = byte_length::<Fp128>(requirements.partials)?;
    let output_bytes = byte_length::<Fp128>(requirements.output)?;
    let audit_bytes = byte_length::<HammingWeightAuditRow>(requirements.audit_rows)?;
    let status_bytes = byte_length::<HammingWeightStatus>(requirements.status)?;
    let allocation_bytes = [
        resident_bytes,
        e_in_bytes,
        e_out_bytes,
        partial_bytes,
        output_bytes,
        audit_bytes,
        status_bytes,
    ];
    for bytes in allocation_bytes {
        validate_buffer_length(&context, bytes)?;
    }
    let additional = allocation_bytes
        .into_iter()
        .try_fold(0u64, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or(MetalError::InputTooLong(rows.len()))
        })?;
    validate_working_set(
        context.device.current_allocated_size(),
        additional,
        context.device.recommended_max_working_set_size(),
    )?;

    let resident_buffer = buffer_from_slice(&context.device, rows);
    let e_in_buffer = buffer_from_slice(&context.device, &e_in);
    let e_out_buffer = buffer_from_slice(&context.device, &e_out);
    let partial_buffer = context
        .device
        .new_buffer(partial_bytes, MTLResourceOptions::StorageModeShared);
    let output_buffer = context
        .device
        .new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);
    let audit_buffer = context
        .device
        .new_buffer(audit_bytes, MTLResourceOptions::StorageModeShared);
    let status_buffer = buffer_from_slice(&context.device, &[HammingWeightStatus::default()]);
    let params = plan.params();

    let gpu_active = dispatch(
        &context,
        &pipelines,
        plan,
        &params,
        &resident_buffer,
        &e_in_buffer,
        &e_out_buffer,
        &partial_buffer,
        &output_buffer,
        &audit_buffer,
        &status_buffer,
    )?;
    let status = read_one::<HammingWeightStatus>(&status_buffer);
    let audit_rows = read_values::<HammingWeightAuditRow>(&audit_buffer, requirements.audit_rows);
    let census = HammingWeightCensus::from_audit_rows(&audit_rows, status, plan.shape())?;
    let encoded_masses = read_values::<Fp128>(&output_buffer, requirements.output);
    let masses = encoded_masses
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            if value.is_canonical(AKITA_OFFSET_FFFFA7F7) {
                Ok(value.into_jolt_field())
            } else {
                Err(MetalError::NonCanonicalOutput {
                    index,
                    offset: AKITA_OFFSET_FFFFA7F7,
                })
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(HammingWeightFixtureExecution {
        compile: pipelines.report,
        gpu_active,
        masses,
        audit_rows,
        status,
        census,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "the arguments mirror the shader ABI"
)]
fn dispatch(
    context: &HammingWeightProbeContext,
    pipelines: &HammingWeightProbePipelines,
    plan: HammingWeightSlicePlan,
    params: &HammingWeightHistogramParams,
    rows: &Buffer,
    e_in: &Buffer,
    e_out: &Buffer,
    partials: &Buffer,
    output: &Buffer,
    audits: &Buffer,
    status: &Buffer,
) -> Result<Duration, MetalError> {
    autoreleasepool(|| {
        let command = context.queue.new_command_buffer();
        let histogram = command.new_compute_command_encoder();
        histogram.set_compute_pipeline_state(&pipelines.histogram);
        histogram.set_buffer(0, Some(rows), 0);
        histogram.set_buffer(1, Some(e_in), 0);
        histogram.set_buffer(2, Some(e_out), 0);
        histogram.set_buffer(3, Some(partials), 0);
        histogram.set_buffer(4, Some(audits), 0);
        histogram.set_buffer(5, Some(status), 0);
        set_inline_bytes(histogram, 6, params);
        histogram.set_threadgroup_memory_length(
            0,
            plan.histogram_launch().threadgroup_memory_bytes as u64,
        );
        histogram.dispatch_thread_groups(
            MTLSize {
                width: plan.histogram_launch().threadgroups as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: plan.histogram_launch().threads_per_threadgroup as u64,
                height: 1,
                depth: 1,
            },
        );
        histogram.end_encoding();

        let finalize = command.new_compute_command_encoder();
        finalize.set_compute_pipeline_state(&pipelines.finalize);
        finalize.set_buffer(0, Some(partials), 0);
        finalize.set_buffer(1, Some(output), 0);
        finalize.set_buffer(2, Some(status), 0);
        set_inline_bytes(finalize, 3, params);
        finalize.dispatch_thread_groups(
            MTLSize {
                width: plan.finalize_launch().threadgroups as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: plan.finalize_launch().threads_per_threadgroup as u64,
                height: 1,
                depth: 1,
            },
        );
        finalize.end_encoding();

        command.commit();
        command.wait_until_completed();
        let status = command.status();
        if status != MTLCommandBufferStatus::Completed {
            return Err(MetalError::CommandFailed(status));
        }
        let start = command_buffer_timestamp(command, "GPUStartTime")?;
        let end = command_buffer_timestamp(command, "GPUEndTime")?;
        if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
            return Err(MetalError::InvalidGpuTimestamps { start, end });
        }
        Ok(Duration::from_secs_f64(end - start))
    })
}

fn validate_pipeline_admission(report: &HammingWeightCompileReport) -> Result<(), MetalError> {
    validate_pipeline(
        HISTOGRAM_PIPELINE,
        report.histogram,
        HAMMING_WEIGHT_THREADS,
        report.dynamic_threadgroup_memory_bytes,
        report.device.max_threadgroup_memory_length,
    )?;
    validate_pipeline(
        FINALIZE_PIPELINE,
        report.finalize,
        HAMMING_WEIGHT_BINS,
        0,
        report.device.max_threadgroup_memory_length,
    )
}

fn validate_pipeline(
    name: &'static str,
    limits: PipelineLimits,
    requested_threads: usize,
    dynamic_memory: u64,
    maximum_memory: u64,
) -> Result<(), MetalError> {
    if limits.thread_execution_width != HAMMING_WEIGHT_SIMD_WIDTH {
        return Err(MetalError::UnsupportedHammingWeightExecutionWidth {
            pipeline: name,
            expected: HAMMING_WEIGHT_SIMD_WIDTH,
            got: limits.thread_execution_width,
        });
    }
    if limits.max_total_threads_per_threadgroup < requested_threads {
        return Err(MetalError::HammingWeightThreadgroupLimit {
            pipeline: name,
            requested: requested_threads,
            maximum: limits.max_total_threads_per_threadgroup,
        });
    }
    let requested_memory = limits
        .static_threadgroup_memory_length
        .checked_add(dynamic_memory)
        .ok_or(MetalError::InputTooLong(dynamic_memory as usize))?;
    if requested_memory > maximum_memory {
        return Err(MetalError::HammingWeightThreadgroupMemory {
            pipeline: name,
            requested: requested_memory,
            maximum: maximum_memory,
        });
    }
    Ok(())
}

fn encode_fields(side: &'static str, values: &[AkitaField]) -> Result<Vec<Fp128>, MetalError> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let encoded = Fp128::from_jolt_field(value);
            if encoded.is_canonical(AKITA_OFFSET_FFFFA7F7) {
                Ok(encoded)
            } else {
                Err(MetalError::NonCanonicalInput {
                    side,
                    index,
                    offset: AKITA_OFFSET_FFFFA7F7,
                })
            }
        })
        .collect()
}

fn validate_buffer_length(
    context: &HammingWeightProbeContext,
    requested: u64,
) -> Result<(), MetalError> {
    let maximum = context.device.max_buffer_length();
    if requested > maximum {
        Err(MetalError::BufferTooLong { requested, maximum })
    } else {
        Ok(())
    }
}

fn byte_length<T>(elements: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}

fn read_one<T: Copy>(buffer: &Buffer) -> T {
    // SAFETY: every read follows command completion and the buffer was allocated for `T`.
    unsafe { *buffer.contents().cast::<T>() }
}

fn read_values<T: Copy>(buffer: &Buffer, elements: usize) -> Vec<T> {
    // SAFETY: every read follows command completion and the buffer contains `elements` values.
    unsafe { slice::from_raw_parts(buffer.contents().cast::<T>(), elements).to_vec() }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "the Metal parity fixture uses checked constants"
)]
mod tests {
    use jolt_poly::EqPolynomial;

    use super::super::oracle::{recentered_pushforwards, unfactored_recentered_pushforwards};
    use super::*;

    #[test]
    fn fixed_29_kernel_matches_the_unfactored_two_stage_fixture() {
        let config = HammingWeightSuccessorConfig {
            inner_log2: 10,
            stage_rows: super::super::HAMMING_WEIGHT_STAGE_ROWS,
            threads_per_threadgroup: HAMMING_WEIGHT_THREADS,
            trace_cutoff: 1 << 15,
        };
        let plan =
            HammingWeightSlicePlan::new(1 << 16, config, HammingWeightProtocolTopology::PRODUCTION)
                .unwrap();
        let rows = fixture_rows(plan.shape().rows(), plan.shape().inner_length());
        let point = (0..16)
            .map(|coordinate| AkitaField::from_u64((coordinate * 17 + 3) as u64))
            .collect::<Vec<_>>();
        let (outer_point, inner_point) = point.split_at(6);
        let e_out = EqPolynomial::evals(outer_point, None);
        let e_in = EqPolynomial::evals(inner_point, None);
        let direct = unfactored_recentered_pushforwards(&rows, &point, plan.shape()).unwrap();
        let split = recentered_pushforwards(&rows, &e_in, &e_out, plan.shape()).unwrap();
        let execution =
            execute_hamming_weight_claim_reduction_fixture(&rows, &e_in, &e_out, config).unwrap();

        assert!(execution.compile.admitted());
        assert_eq!(execution.status, HammingWeightStatus::default());
        assert_eq!(execution.masses, direct);
        assert_eq!(execution.masses, split.masses);
        assert_eq!(execution.audit_rows, split.audit_rows);
        assert_eq!(execution.census, split.census);
    }

    fn fixture_rows(rows: usize, inner: usize) -> Vec<HammingWeightResidentRow> {
        let mut fixture = (0..rows)
            .map(|index| {
                let lookup_lo = (index as u64).wrapping_mul(0x0102_0304_0506_0708);
                let lookup_hi = (!(index as u64)).rotate_left(17);
                let ram = if index.is_multiple_of(3) {
                    0
                } else {
                    (index & 0xffff) as u64 + 1
                };
                let magnitude = (index as u64).wrapping_mul(0x1_0001);
                let pc = if index.is_multiple_of(5) {
                    0
                } else {
                    ((index * 7) & 0xffff) as u64 + 1
                };
                let negative = u64::from(index.is_multiple_of(7)) << 63;
                HammingWeightResidentRow::from_words([
                    lookup_lo,
                    lookup_hi,
                    ram,
                    magnitude,
                    pc | negative,
                ])
            })
            .collect::<Vec<_>>();

        fixture[0] = HammingWeightResidentRow::default();
        fixture[511] = HammingWeightResidentRow::from_words([
            0x0011_2233_4455_6677,
            0x8899_aabb_ccdd_eeff,
            1,
            super::super::HAMMING_WEIGHT_BALANCED_INC_BIAS + 1,
            1 << 63,
        ]);
        fixture[512] = HammingWeightResidentRow::from_words([
            0xffee_ddcc_bbaa_9988,
            0x7766_5544_3322_1100,
            0x1_0000,
            u64::MAX - super::super::HAMMING_WEIGHT_BALANCED_INC_BIAS + 1,
            0x1_0000,
        ]);
        fixture[inner - 1] = HammingWeightResidentRow::default();
        fixture[inner + 1] = fixture[1];
        fixture[rows - inner] = fixture[511];
        fixture[rows - 1] = fixture[512];
        fixture
    }
}
