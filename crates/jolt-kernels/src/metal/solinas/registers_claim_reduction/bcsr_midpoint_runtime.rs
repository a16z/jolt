#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the BCSR midpoint runtime remains hidden until its resident producer is wired"
    )
)]

use std::{mem::size_of, slice, time::Duration, time::Instant};

use jolt_field::AkitaField;
use jolt_poly::EqPolynomial;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

use super::super::registers_read_write_v3::{
    RegisterBcsr256, RegisterBcsrGeometry, RegisterBcsrLayout, REGISTER_BCSR_POSITION_SLOTS,
    REGISTER_CSR_COLUMNS,
};
use super::super::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};
use super::bcsr_runtime::RegistersClaimBcsrRuntimeError;
use super::{
    RegistersClaimBcsrMidpointParams, RegistersClaimGeometry, RegistersClaimPlanError,
    BCSR_MIDPOINT_EQ_PREFIX_SLOT, BCSR_MIDPOINT_OUTPUT_SLOT, BCSR_MIDPOINT_PARAMS_SLOT,
    BCSR_MIDPOINT_PIPELINE, BCSR_MIDPOINT_RD_OFFSETS_SLOT, BCSR_MIDPOINT_RD_POSITIONS_SLOT,
    BCSR_MIDPOINT_RD_POST_VALUES_SLOT, BCSR_MIDPOINT_THREADGROUP_BYTES,
    BCSR_MIDPOINT_THREADS_PER_THREADGROUP, REGISTERS_CLAIM_AKITA_OFFSET,
    REGISTERS_CLAIM_SIMD_WIDTH,
};

#[cfg(feature = "test-utils")]
use super::registers_claim_q_checksum;
#[cfg(feature = "test-utils")]
use super::RegistersClaimBcsrBenchmarkError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RegistersClaimBcsrMidpointPlan {
    geometry: RegistersClaimGeometry,
    layout: RegisterBcsrLayout,
    blocks: usize,
    low_blocks: usize,
    source_bytes: usize,
    eq_prefix_bytes: usize,
    output_bytes: usize,
    rd_events: u64,
    params: RegistersClaimBcsrMidpointParams,
}

impl RegistersClaimBcsrMidpointPlan {
    fn new(
        bcsr: &RegisterBcsr256,
        max_buffer_length: usize,
    ) -> Result<Self, RegistersClaimBcsrRuntimeError> {
        bcsr.validate()?;
        Self::for_geometry(
            bcsr.geometry(),
            bcsr.event_counts().rd() as u64,
            max_buffer_length,
        )
    }

    fn for_geometry(
        bcsr_geometry: RegisterBcsrGeometry,
        rd_events: u64,
        max_buffer_length: usize,
    ) -> Result<Self, RegistersClaimBcsrRuntimeError> {
        let geometry = RegistersClaimGeometry::new(bcsr_geometry.cycles())?;
        if geometry.prefix_elements() < REGISTER_BCSR_POSITION_SLOTS
            || !geometry
                .prefix_elements()
                .is_multiple_of(REGISTER_BCSR_POSITION_SLOTS)
        {
            return Err(RegistersClaimBcsrRuntimeError::UnsupportedPrefix {
                prefix_elements: geometry.prefix_elements(),
            });
        }
        let blocks = bcsr_geometry.blocks();
        let low_blocks = geometry.prefix_elements() / REGISTER_BCSR_POSITION_SLOTS;
        if blocks != geometry.suffix_elements() * low_blocks {
            return Err(RegistersClaimBcsrRuntimeError::InvalidState(
                "BCSR blocks do not tile the midpoint domain",
            ));
        }
        let layout = RegisterBcsrLayout::new(bcsr_geometry)?;
        let source_bytes = checked_sum(&[
            layout.offsets().bytes(),
            layout.positions().bytes(),
            layout.rd_post_values().bytes(),
        ])?;
        let eq_prefix_bytes = checked_bytes(
            "BCSR midpoint equality prefix",
            geometry.prefix_elements(),
            size_of::<Fp128>(),
        )?;
        let output_bytes = checked_bytes(
            "BCSR midpoint output",
            geometry.suffix_elements(),
            size_of::<Fp128>(),
        )?;
        for (name, bytes) in [
            ("BCSR midpoint offsets", layout.offsets().bytes()),
            ("BCSR midpoint positions", layout.positions().bytes()),
            (
                "BCSR midpoint rd post values",
                layout.rd_post_values().bytes(),
            ),
            ("BCSR midpoint equality prefix", eq_prefix_bytes),
            ("BCSR midpoint output", output_bytes),
        ] {
            if bytes > max_buffer_length {
                return Err(RegistersClaimPlanError::BufferTooLarge {
                    name,
                    bytes,
                    max_buffer_length,
                }
                .into());
            }
        }
        let params = RegistersClaimBcsrMidpointParams {
            blocks: abi_count("BCSR midpoint blocks", blocks)?,
            prefix_elements: abi_count(
                "BCSR midpoint prefix elements",
                geometry.prefix_elements(),
            )?,
            suffix_elements: abi_count(
                "BCSR midpoint suffix elements",
                geometry.suffix_elements(),
            )?,
            low_blocks: abi_count("BCSR midpoint low blocks", low_blocks)?,
            columns: REGISTER_CSR_COLUMNS as u32,
            offset_stride: (REGISTER_CSR_COLUMNS + 1) as u32,
            position_stride: REGISTER_BCSR_POSITION_SLOTS as u32,
            reserved: 0,
        };
        Ok(Self {
            geometry,
            layout,
            blocks,
            low_blocks,
            source_bytes,
            eq_prefix_bytes,
            output_bytes,
            rd_events,
            params,
        })
    }
}

struct RegistersClaimBcsrMidpointSources {
    offsets: Buffer,
    positions: Buffer,
    post_values: Buffer,
}

impl RegistersClaimBcsrMidpointSources {
    fn validate(
        &self,
        context: &SolinasMetal,
        plan: RegistersClaimBcsrMidpointPlan,
    ) -> Result<(), RegistersClaimBcsrRuntimeError> {
        let expected_device = context.device_registry_id();
        let buffers = [
            (
                "BCSR midpoint rd offsets",
                &self.offsets,
                plan.layout.offsets().bytes(),
            ),
            (
                "BCSR midpoint rd positions",
                &self.positions,
                plan.layout.positions().bytes(),
            ),
            (
                "BCSR midpoint rd post values",
                &self.post_values,
                plan.layout.rd_post_values().bytes(),
            ),
        ];
        let mut identities = [0usize; 3];
        for (index, (name, buffer, bytes)) in buffers.into_iter().enumerate() {
            let got_device = buffer.device().registry_id();
            if got_device != expected_device {
                return Err(RegistersClaimBcsrRuntimeError::BufferDevice {
                    name,
                    expected: expected_device,
                    got: got_device,
                });
            }
            let expected = to_u64(bytes)?;
            if buffer.length() != expected {
                return Err(RegistersClaimBcsrRuntimeError::BufferLength {
                    name,
                    expected,
                    actual: buffer.length(),
                });
            }
            identities[index] = buffer.as_ptr() as usize;
        }
        if identities[0] == identities[1]
            || identities[0] == identities[2]
            || identities[1] == identities[2]
        {
            return Err(RegistersClaimBcsrRuntimeError::AliasedBuffers);
        }
        Ok(())
    }
}

struct RegistersClaimBcsrMidpointWorking {
    eq_prefix: Buffer,
    output: Buffer,
}

pub(crate) struct RegistersClaimBcsrMidpointInvocation {
    context: SolinasMetal,
    pipeline: ComputePipelineState,
    limits: PipelineLimits,
    sources: RegistersClaimBcsrMidpointSources,
    working: RegistersClaimBcsrMidpointWorking,
    plan: RegistersClaimBcsrMidpointPlan,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RegistersClaimBcsrMidpointObservation {
    pub rd_write_value: Vec<AkitaField>,
    pub useful_half_width_terms: u64,
    pub dispatches: usize,
    pub source_bytes: usize,
    pub gpu_active: Duration,
    pub resident_wall: Duration,
}

#[cfg(feature = "test-utils")]
pub struct RegistersClaimBcsrMidpointBenchmarkInvocation {
    inner: RegistersClaimBcsrMidpointInvocation,
    setup_wall: Duration,
}

#[cfg(feature = "test-utils")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimBcsrMidpointBenchmarkObservation {
    pub checksum: u64,
    pub useful_half_width_terms: u64,
    pub dispatches: usize,
    pub source_bytes: usize,
    pub setup_wall: Duration,
    pub gpu_active: Duration,
    pub resident_wall: Duration,
}

impl SolinasMetal {
    pub(crate) fn prepare_registers_claim_bcsr_midpoint(
        &self,
        bcsr: &RegisterBcsr256,
        prefix_challenges: &[AkitaField],
    ) -> Result<RegistersClaimBcsrMidpointInvocation, RegistersClaimBcsrRuntimeError> {
        if self.offset != REGISTERS_CLAIM_AKITA_OFFSET {
            return Err(RegistersClaimBcsrRuntimeError::UnsupportedOffset {
                expected: REGISTERS_CLAIM_AKITA_OFFSET,
                got: self.offset,
            });
        }
        let max_buffer_length = usize::try_from(self.device.max_buffer_length())
            .map_err(|_| MetalError::InputTooLong(bcsr.geometry().cycles()))?;
        let plan = RegistersClaimBcsrMidpointPlan::new(bcsr, max_buffer_length)?;
        let parts = bcsr.parts();
        let sources = RegistersClaimBcsrMidpointSources {
            offsets: buffer_from_slice(&self.device, &parts.rd_offsets),
            positions: buffer_from_slice(&self.device, &parts.rd_positions),
            post_values: buffer_from_slice(&self.device, &parts.rd_post_values),
        };
        self.prepare_registers_claim_bcsr_midpoint_sources(sources, plan, prefix_challenges)
    }

    #[cfg(feature = "test-utils")]
    pub fn prepare_registers_claim_bcsr_midpoint_benchmark(
        &self,
        cycles: usize,
        prefix_challenges: &[AkitaField],
    ) -> Result<RegistersClaimBcsrMidpointBenchmarkInvocation, RegistersClaimBcsrBenchmarkError>
    {
        use super::bcsr_runtime::{
            fill_synthetic_rd_plane, shared_slice_mut, SYNTHETIC_RD_EVENTS_PER_BLOCK,
        };

        let setup_started = Instant::now();
        if self.offset != REGISTERS_CLAIM_AKITA_OFFSET {
            return Err(RegistersClaimBcsrRuntimeError::UnsupportedOffset {
                expected: REGISTERS_CLAIM_AKITA_OFFSET,
                got: self.offset,
            }
            .into());
        }
        let max_buffer_length = usize::try_from(self.device.max_buffer_length())
            .map_err(|_| MetalError::InputTooLong(cycles))?;
        let bcsr_geometry =
            RegisterBcsrGeometry::new(cycles).map_err(RegistersClaimBcsrRuntimeError::from)?;
        let rd_events = u64::try_from(bcsr_geometry.blocks())
            .map_err(|_| MetalError::InputTooLong(bcsr_geometry.blocks()))?
            * SYNTHETIC_RD_EVENTS_PER_BLOCK as u64;
        let plan = RegistersClaimBcsrMidpointPlan::for_geometry(
            bcsr_geometry,
            rd_events,
            max_buffer_length,
        )?;
        self.validate_registers_claim_bcsr_midpoint_working_set(plan)?;
        let shared = MTLResourceOptions::StorageModeShared;
        let mut sources = RegistersClaimBcsrMidpointSources {
            offsets: self
                .device
                .new_buffer(to_u64(plan.layout.offsets().bytes())?, shared),
            positions: self
                .device
                .new_buffer(to_u64(plan.layout.positions().bytes())?, shared),
            post_values: self
                .device
                .new_buffer(to_u64(plan.layout.rd_post_values().bytes())?, shared),
        };
        sources.validate(self, plan)?;
        // SAFETY: the shared buffers have the exact checked layout and are not
        // submitted until the synthetic fill returns.
        unsafe {
            fill_synthetic_rd_plane(
                shared_slice_mut(&mut sources.offsets, plan.layout.offsets().elements()),
                shared_slice_mut(&mut sources.positions, plan.layout.positions().elements()),
                shared_slice_mut(
                    &mut sources.post_values,
                    plan.layout.rd_post_values().elements(),
                ),
                plan.blocks,
            );
        }
        let inner =
            self.prepare_registers_claim_bcsr_midpoint_sources(sources, plan, prefix_challenges)?;
        Ok(RegistersClaimBcsrMidpointBenchmarkInvocation {
            inner,
            setup_wall: setup_started.elapsed(),
        })
    }

    fn validate_registers_claim_bcsr_midpoint_working_set(
        &self,
        plan: RegistersClaimBcsrMidpointPlan,
    ) -> Result<(), RegistersClaimBcsrRuntimeError> {
        let additional = plan
            .source_bytes
            .checked_add(plan.eq_prefix_bytes)
            .and_then(|bytes| bytes.checked_add(plan.output_bytes))
            .ok_or(MetalError::InputTooLong(plan.geometry.rows()))?;
        self.validate_additional_working_set(to_u64(additional)?)?;
        Ok(())
    }

    fn prepare_registers_claim_bcsr_midpoint_sources(
        &self,
        sources: RegistersClaimBcsrMidpointSources,
        plan: RegistersClaimBcsrMidpointPlan,
        prefix_challenges: &[AkitaField],
    ) -> Result<RegistersClaimBcsrMidpointInvocation, RegistersClaimBcsrRuntimeError> {
        sources.validate(self, plan)?;
        if prefix_challenges.len() != plan.geometry.prefix_vars() {
            return Err(RegistersClaimBcsrRuntimeError::WrongPointLength {
                expected: plan.geometry.prefix_vars(),
                actual: prefix_challenges.len(),
            });
        }
        self.validate_registers_claim_bcsr_midpoint_working_set(plan)?;
        let prefix_point = prefix_challenges.iter().rev().copied().collect::<Vec<_>>();
        let eq_prefix = EqPolynomial::<AkitaField>::evals(&prefix_point, None)
            .iter()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>();
        self.validate_inputs("registers claim BCSR midpoint equality", &eq_prefix)?;

        let pipeline = self.compile_named_pipeline(BCSR_MIDPOINT_PIPELINE)?;
        let limits = Self::limits(&pipeline);
        if limits.thread_execution_width != REGISTERS_CLAIM_SIMD_WIDTH {
            return Err(RegistersClaimBcsrRuntimeError::InvalidState(
                "BCSR midpoint pipeline has an unsupported execution width",
            ));
        }
        let width = Self::resolve_threadgroup_width(
            Some(BCSR_MIDPOINT_THREADS_PER_THREADGROUP as usize),
            limits,
        )?;
        if width != BCSR_MIDPOINT_THREADS_PER_THREADGROUP as usize {
            return Err(RegistersClaimBcsrRuntimeError::InvalidState(
                "resolved midpoint width differs from the BCSR ABI",
            ));
        }
        let requested = BCSR_MIDPOINT_THREADGROUP_BYTES
            .checked_add(limits.static_threadgroup_memory_length)
            .ok_or(MetalError::InputTooLong(plan.geometry.rows()))?;
        let maximum = self.device.max_threadgroup_memory_length();
        if requested > maximum {
            return Err(RegistersClaimBcsrRuntimeError::ThreadgroupMemory { requested, maximum });
        }
        let working = RegistersClaimBcsrMidpointWorking {
            eq_prefix: buffer_from_slice(&self.device, &eq_prefix),
            output: self.device.new_buffer(
                to_u64(plan.output_bytes)?,
                MTLResourceOptions::StorageModeShared,
            ),
        };
        Ok(RegistersClaimBcsrMidpointInvocation {
            context: self.clone(),
            pipeline,
            limits,
            sources,
            working,
            plan,
        })
    }
}

impl RegistersClaimBcsrMidpointInvocation {
    pub(crate) fn execute_timed(
        &self,
    ) -> Result<RegistersClaimBcsrMidpointObservation, RegistersClaimBcsrRuntimeError> {
        let wall_started = Instant::now();
        self.validate_state()?;
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);
            encoder.set_buffer(
                BCSR_MIDPOINT_RD_OFFSETS_SLOT,
                Some(&self.sources.offsets),
                0,
            );
            encoder.set_buffer(
                BCSR_MIDPOINT_RD_POSITIONS_SLOT,
                Some(&self.sources.positions),
                0,
            );
            encoder.set_buffer(
                BCSR_MIDPOINT_RD_POST_VALUES_SLOT,
                Some(&self.sources.post_values),
                0,
            );
            encoder.set_buffer(
                BCSR_MIDPOINT_EQ_PREFIX_SLOT,
                Some(&self.working.eq_prefix),
                0,
            );
            encoder.set_buffer(BCSR_MIDPOINT_OUTPUT_SLOT, Some(&self.working.output), 0);
            set_inline_bytes(encoder, BCSR_MIDPOINT_PARAMS_SLOT, &self.plan.params);
            encoder.set_threadgroup_memory_length(0, BCSR_MIDPOINT_THREADGROUP_BYTES);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.plan.geometry.suffix_elements() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: BCSR_MIDPOINT_THREADS_PER_THREADGROUP,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
            if command_buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(MetalError::CommandFailed(command_buffer.status()).into());
            }
            let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
            let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
            if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
                return Err(MetalError::InvalidGpuTimestamps { start, end }.into());
            }
            Ok(RegistersClaimBcsrMidpointObservation {
                rd_write_value: self.read_output()?,
                useful_half_width_terms: self.plan.rd_events,
                dispatches: 1,
                source_bytes: self.plan.source_bytes,
                gpu_active: Duration::from_secs_f64(end - start),
                resident_wall: wall_started.elapsed(),
            })
        })
    }

    fn validate_state(&self) -> Result<(), RegistersClaimBcsrRuntimeError> {
        self.sources.validate(&self.context, self.plan)?;
        if self.context.offset != REGISTERS_CLAIM_AKITA_OFFSET
            || self.plan.params.reserved != 0
            || self.plan.params.blocks as usize != self.plan.blocks
            || self.plan.params.low_blocks as usize != self.plan.low_blocks
            || self.limits.thread_execution_width != REGISTERS_CLAIM_SIMD_WIDTH
            || self.limits.max_total_threads_per_threadgroup
                < BCSR_MIDPOINT_THREADS_PER_THREADGROUP as usize
        {
            return Err(RegistersClaimBcsrRuntimeError::InvalidState(
                "prepared BCSR midpoint metadata changed",
            ));
        }
        Ok(())
    }

    fn read_output(&self) -> Result<Vec<AkitaField>, RegistersClaimBcsrRuntimeError> {
        let fields = self.plan.geometry.suffix_elements();
        // SAFETY: the shared output owns exactly `fields` values and command
        // completion precedes this read.
        let values = unsafe {
            slice::from_raw_parts(self.working.output.contents().cast::<Fp128>(), fields)
        };
        self.context
            .validate_inputs("registers claim BCSR midpoint output", values)?;
        Ok(values
            .iter()
            .map(|&value| value.into_jolt_field())
            .collect())
    }
}

#[cfg(feature = "test-utils")]
impl RegistersClaimBcsrMidpointBenchmarkInvocation {
    pub fn execute_timed(
        &self,
    ) -> Result<RegistersClaimBcsrMidpointBenchmarkObservation, RegistersClaimBcsrBenchmarkError>
    {
        let observation = self.inner.execute_timed()?;
        Ok(RegistersClaimBcsrMidpointBenchmarkObservation {
            checksum: registers_claim_q_checksum(&observation.rd_write_value),
            useful_half_width_terms: observation.useful_half_width_terms,
            dispatches: observation.dispatches,
            source_bytes: observation.source_bytes,
            setup_wall: self.setup_wall,
            gpu_active: observation.gpu_active,
            resident_wall: observation.resident_wall,
        })
    }
}

fn checked_bytes(
    name: &'static str,
    elements: usize,
    element_bytes: usize,
) -> Result<usize, RegistersClaimPlanError> {
    elements
        .checked_mul(element_bytes)
        .ok_or(RegistersClaimPlanError::SizeOverflow { name })
}

fn checked_sum(values: &[usize]) -> Result<usize, RegistersClaimPlanError> {
    values.iter().try_fold(0usize, |total, &value| {
        total
            .checked_add(value)
            .ok_or(RegistersClaimPlanError::SizeOverflow {
                name: "BCSR midpoint bytes",
            })
    })
}

fn abi_count(name: &'static str, value: usize) -> Result<u32, RegistersClaimPlanError> {
    u32::try_from(value).map_err(|_| RegistersClaimPlanError::AbiCountOverflow { name, value })
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}

fn to_u64(value: usize) -> Result<u64, MetalError> {
    u64::try_from(value).map_err(|_| MetalError::InputTooLong(value))
}
