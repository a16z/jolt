use std::{cell::Cell, slice, time::Duration};

use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::{
    checked_probe_shape, HalfWidthOperand, HalfWidthProbe, HalfWidthProbeShape,
    HALF_WIDTH_AKITA_OFFSET,
};
use crate::metal::solinas::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};

struct HalfWidthProbeBuffers {
    coefficients: Buffer,
    operands: Buffer,
    output: Buffer,
    params: Buffer,
}

pub struct HalfWidthProbeInvocation<'a> {
    context: &'a SolinasMetal,
    probe: HalfWidthProbe,
    pipeline: ComputePipelineState,
    buffers: HalfWidthProbeBuffers,
    shape: HalfWidthProbeShape,
    limits: PipelineLimits,
    threads_per_threadgroup: usize,
    elements: usize,
    completed: Cell<bool>,
}

impl SolinasMetal {
    pub fn prepare_half_width_probe(
        &self,
        probe: HalfWidthProbe,
        coefficients: &[Fp128],
        operands: &[HalfWidthOperand],
        iterations: u32,
        threads_per_threadgroup: Option<usize>,
    ) -> Result<HalfWidthProbeInvocation<'_>, MetalError> {
        let device = self.device_info();
        if device.offset != HALF_WIDTH_AKITA_OFFSET {
            return Err(MetalError::UnexpectedSolinasOffset {
                expected: HALF_WIDTH_AKITA_OFFSET,
                got: device.offset,
            });
        }
        let shape = checked_probe_shape(
            probe,
            coefficients,
            operands,
            iterations,
            device.offset,
            device.max_buffer_length,
        )?;
        self.validate_additional_working_set(shape.allocated_bytes())?;
        let pipeline = self.compile_named_pipeline(probe.name())?;
        let limits = Self::limits(&pipeline);
        let threads_per_threadgroup =
            Self::resolve_threadgroup_width(threads_per_threadgroup, limits)?;
        let buffers = HalfWidthProbeBuffers {
            coefficients: buffer_from_slice(&self.device, coefficients),
            operands: buffer_from_slice(&self.device, operands),
            output: self.device.new_buffer(
                shape.field_buffer_bytes(),
                MTLResourceOptions::StorageModeShared,
            ),
            params: buffer_from_slice(&self.device, slice::from_ref(&shape.params())),
        };
        Ok(HalfWidthProbeInvocation {
            context: self,
            probe,
            pipeline,
            buffers,
            shape,
            limits,
            threads_per_threadgroup,
            elements: coefficients.len(),
            completed: Cell::new(false),
        })
    }
}

impl HalfWidthProbeInvocation<'_> {
    pub const fn probe(&self) -> HalfWidthProbe {
        self.probe
    }

    pub const fn shape(&self) -> HalfWidthProbeShape {
        self.shape
    }

    pub const fn pipeline_limits(&self) -> PipelineLimits {
        self.limits
    }

    pub const fn threads_per_threadgroup(&self) -> usize {
        self.threads_per_threadgroup
    }

    pub const fn execute_device_buffer_allocations(&self) -> usize {
        0
    }

    pub fn execute(&self) -> Result<(), MetalError> {
        self.execute_timed().map(|_| ())
    }

    pub fn execute_timed(&self) -> Result<Duration, MetalError> {
        self.completed.set(false);
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);
            encoder.set_buffer(0, Some(&self.buffers.coefficients), 0);
            encoder.set_buffer(1, Some(&self.buffers.operands), 0);
            encoder.set_buffer(2, Some(&self.buffers.output), 0);
            encoder.set_buffer(3, Some(&self.buffers.params), 0);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self
                        .shape
                        .grid_threads()
                        .div_ceil(self.threads_per_threadgroup) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads_per_threadgroup as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            if command_buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(MetalError::CommandFailed(command_buffer.status()));
            }
            let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
            let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
            if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
                return Err(MetalError::InvalidGpuTimestamps { start, end });
            }
            self.completed.set(true);
            Ok(Duration::from_secs_f64(end - start))
        })
    }

    pub fn read_output(&self) -> Result<Vec<Fp128>, MetalError> {
        if !self.completed.get() {
            return Err(MetalError::NotExecuted);
        }
        // SAFETY: output owns exactly `elements` Fp128 values and execution
        // completes before this shared-storage read.
        let output = unsafe {
            slice::from_raw_parts(
                self.buffers.output.contents().cast::<Fp128>(),
                self.elements,
            )
            .to_vec()
        };
        if let Some((index, _)) = output
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_canonical(HALF_WIDTH_AKITA_OFFSET))
        {
            return Err(MetalError::NonCanonicalOutput {
                index,
                offset: HALF_WIDTH_AKITA_OFFSET,
            });
        }
        Ok(output)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "fixed Metal half-width fixtures")]
mod tests {
    use super::*;
    use crate::metal::solinas::half_width_probe::{reference_outputs, HalfWidthDomain};

    #[test]
    fn every_probe_matches_the_independent_limb_oracle() {
        let context = SolinasMetal::for_akita().unwrap();
        let coefficients = (0..32)
            .map(|index| {
                Fp128::from_u128(
                    ((index as u128 + 1) << 96) | (0xfeed_beefu128 * (index as u128 + 3)),
                )
            })
            .collect::<Vec<_>>();
        for probe in HalfWidthProbe::ALL {
            let operands = (0..coefficients.len())
                .map(|index| match probe.domain() {
                    HalfWidthDomain::Unsigned => {
                        HalfWidthOperand::unsigned(u64::MAX - index as u64)
                    }
                    HalfWidthDomain::SignedMagnitude => {
                        HalfWidthOperand::signed_magnitude(u64::MAX - index as u64, index % 2 != 0)
                    }
                    HalfWidthDomain::UnsignedDelta if index % 2 == 0 => {
                        HalfWidthOperand::delta(u64::MAX - index as u64, index as u64)
                    }
                    HalfWidthDomain::UnsignedDelta => {
                        HalfWidthOperand::delta(index as u64, u64::MAX - index as u64)
                    }
                })
                .collect::<Vec<_>>();
            let iterations = if probe.is_chain() { 3 } else { 1 };
            let expected = reference_outputs(probe, &coefficients, &operands, iterations).unwrap();
            let invocation = context
                .prepare_half_width_probe(probe, &coefficients, &operands, iterations, Some(128))
                .unwrap();
            assert_eq!(invocation.execute_device_buffer_allocations(), 0);
            invocation.execute().unwrap();
            assert_eq!(invocation.read_output().unwrap(), expected, "{probe:?}");
        }
    }
}
