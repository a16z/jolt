#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the BCSR runtime remains hidden until the resident producer lease is wired"
    )
)]

use std::{
    mem::size_of,
    slice,
    time::{Duration, Instant},
};

use jolt_field::AkitaField;
use jolt_poly::EqPolynomial;
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
use thiserror::Error;

use super::super::registers_read_write_v3::{
    RegisterBcsr256, RegisterBcsrLayout, RegistersRwV3Error, REGISTER_BCSR_OFFSET_ENTRIES,
    REGISTER_BCSR_POSITION_SLOTS, REGISTER_CSR_COLUMNS,
};
use super::super::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};
use super::{
    RegistersClaimBcsrComponentParams, RegistersClaimBcsrKernelConfig,
    RegistersClaimBcsrReduceParams, RegistersClaimGeometry, RegistersClaimLinearComponents,
    RegistersClaimPlanError, BCSR_COMPONENT_EQ_SUFFIX_SLOT, BCSR_COMPONENT_PARAMS_SLOT,
    BCSR_COMPONENT_PARTIALS_SLOT, BCSR_COMPONENT_PIPELINE,
    BCSR_COMPONENT_RD_OFFSETS_SLOT, BCSR_COMPONENT_RD_POSITIONS_SLOT,
    BCSR_COMPONENT_RD_POST_VALUES_SLOT, BCSR_COMPONENT_REDUCE_INPUT_SLOT,
    BCSR_COMPONENT_REDUCE_OUTPUT_SLOT, BCSR_COMPONENT_REDUCE_PARAMS_SLOT,
    BCSR_COMPONENT_REDUCE_PIPELINE, BCSR_COMPONENT_RS1_OFFSETS_SLOT,
    BCSR_COMPONENT_RS1_POSITIONS_SLOT, BCSR_COMPONENT_RS2_OFFSETS_SLOT,
    BCSR_COMPONENT_RS2_POSITIONS_SLOT, BCSR_COMPONENT_START_VALUES_SLOT,
    BCSR_COMPONENT_THREADGROUP_BYTES, BCSR_COMPONENT_THREADGROUP_SLOT,
    BCSR_COMPONENT_THREADS_PER_THREADGROUP, BCSR_COMPONENT_REDUCE_THREADS_PER_THREADGROUP,
    REGISTERS_CLAIM_AKITA_OFFSET, REGISTERS_CLAIM_OUTPUT_COLUMNS,
};

#[derive(Debug, Error)]
pub(crate) enum RegistersClaimBcsrRuntimeError {
    #[error(transparent)]
    Plan(#[from] RegistersClaimPlanError),
    #[error(transparent)]
    Bcsr(#[from] RegistersRwV3Error),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("registers claim BCSR requires Akita offset {expected:#x}, got {got:#x}")]
    UnsupportedOffset { expected: u32, got: u32 },
    #[error("registers claim BCSR point has length {actual}, expected {expected}")]
    WrongPointLength { expected: usize, actual: usize },
    #[error(
        "registers claim BCSR needs a prefix domain divisible by 256, got {prefix_elements}"
    )]
    UnsupportedPrefix { prefix_elements: usize },
    #[error("invalid registers claim BCSR state: {0}")]
    InvalidState(&'static str),
    #[error(
        "registers claim BCSR needs {requested} bytes of threadgroup memory, device maximum is {maximum}"
    )]
    ThreadgroupMemory { requested: u64, maximum: u64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegistersClaimBcsrExecutionPlan {
    geometry: RegistersClaimGeometry,
    blocks: usize,
    partial_blocks: usize,
    low_blocks: usize,
    suffixes_per_partial: usize,
    partial_bytes: usize,
    component_bytes: usize,
    source_bytes: usize,
}

impl RegistersClaimBcsrExecutionPlan {
    fn new(
        bcsr: &RegisterBcsr256,
        max_buffer_length: usize,
        config: RegistersClaimBcsrKernelConfig,
    ) -> Result<Self, RegistersClaimBcsrRuntimeError> {
        bcsr.validate()?;
        let cycles = bcsr.geometry().cycles();
        let geometry = RegistersClaimGeometry::new(cycles)?;
        if geometry.prefix_elements() < REGISTER_BCSR_POSITION_SLOTS
            || !geometry
                .prefix_elements()
                .is_multiple_of(REGISTER_BCSR_POSITION_SLOTS)
        {
            return Err(RegistersClaimBcsrRuntimeError::UnsupportedPrefix {
                prefix_elements: geometry.prefix_elements(),
            });
        }
        let blocks = bcsr.geometry().blocks();
        let low_blocks = geometry.prefix_elements() / REGISTER_BCSR_POSITION_SLOTS;
        if blocks != geometry.suffix_elements() * low_blocks {
            return Err(RegistersClaimBcsrRuntimeError::InvalidState(
                "BCSR blocks do not tile the factorized row domain",
            ));
        }
        let partial_blocks = config.partial_blocks;
        if partial_blocks == 0
            || partial_blocks > geometry.suffix_elements()
            || !partial_blocks.is_power_of_two()
            || !geometry.suffix_elements().is_multiple_of(partial_blocks)
        {
            return Err(RegistersClaimBcsrRuntimeError::InvalidState(
                "partial blocks must be a power-of-two divisor of the suffix domain",
            ));
        }
        let suffixes_per_partial = geometry.suffix_elements() / partial_blocks;
        let partial_bytes = checked_bytes(
            "BCSR component partials",
            REGISTERS_CLAIM_OUTPUT_COLUMNS * partial_blocks * geometry.prefix_elements(),
            size_of::<Fp128>(),
        )?;
        let component_bytes = checked_bytes(
            "BCSR component output",
            REGISTERS_CLAIM_OUTPUT_COLUMNS * geometry.prefix_elements(),
            size_of::<Fp128>(),
        )?;
        let layout = RegisterBcsrLayout::new(bcsr.geometry())?;
        let source_bytes = layout.topology_bytes()?;
        for (name, bytes) in [
            ("BCSR start values", layout.start_values().bytes()),
            ("BCSR offsets", layout.offsets().bytes()),
            ("BCSR positions", layout.positions().bytes()),
            ("BCSR rd post values", layout.rd_post_values().bytes()),
            (
                "BCSR equality suffix",
                checked_bytes(
                    "BCSR equality suffix",
                    geometry.suffix_elements(),
                    size_of::<Fp128>(),
                )?,
            ),
            ("BCSR component partials", partial_bytes),
            ("BCSR component output", component_bytes),
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

        Ok(Self {
            geometry,
            blocks,
            partial_blocks,
            low_blocks,
            suffixes_per_partial,
            partial_bytes,
            component_bytes,
            source_bytes,
        })
    }

    fn component_params(self) -> Result<RegistersClaimBcsrComponentParams, RegistersClaimPlanError> {
        Ok(RegistersClaimBcsrComponentParams {
            cycles: abi_count("BCSR cycles", self.geometry.rows())?,
            blocks: abi_count("BCSR blocks", self.blocks)?,
            prefix_elements: abi_count("BCSR prefix elements", self.geometry.prefix_elements())?,
            suffix_elements: abi_count("BCSR suffix elements", self.geometry.suffix_elements())?,
            partial_blocks: abi_count("BCSR partial blocks", self.partial_blocks)?,
            low_blocks: abi_count("BCSR low blocks", self.low_blocks)?,
            suffixes_per_partial: abi_count(
                "BCSR suffixes per partial",
                self.suffixes_per_partial,
            )?,
            columns: REGISTER_CSR_COLUMNS as u32,
        })
    }

    fn reduce_params(self) -> Result<RegistersClaimBcsrReduceParams, RegistersClaimPlanError> {
        Ok(RegistersClaimBcsrReduceParams {
            partial_blocks: abi_count("BCSR partial blocks", self.partial_blocks)?,
            prefix_elements: abi_count("BCSR prefix elements", self.geometry.prefix_elements())?,
            columns: REGISTERS_CLAIM_OUTPUT_COLUMNS as u32,
            reserved: 0,
        })
    }

    const fn component_threadgroups(self) -> usize {
        self.partial_blocks * self.low_blocks
    }

    const fn reduce_threadgroups(self) -> usize {
        REGISTERS_CLAIM_OUTPUT_COLUMNS * self.low_blocks
    }
}

struct RegistersClaimBcsrSourceBuffers {
    start_values: Buffer,
    rs1_offsets: Buffer,
    rs1_positions: Buffer,
    rs2_offsets: Buffer,
    rs2_positions: Buffer,
    rd_offsets: Buffer,
    rd_positions: Buffer,
    rd_post_values: Buffer,
}

struct RegistersClaimBcsrWorkingBuffers {
    eq_suffix: Buffer,
    partials: Buffer,
    components: Buffer,
}

pub(crate) struct RegistersClaimBcsrComponentInvocation {
    context: SolinasMetal,
    component_pipeline: ComputePipelineState,
    reduce_pipeline: ComputePipelineState,
    component_limits: PipelineLimits,
    reduce_limits: PipelineLimits,
    sources: RegistersClaimBcsrSourceBuffers,
    working: RegistersClaimBcsrWorkingBuffers,
    plan: RegistersClaimBcsrExecutionPlan,
    component_params: RegistersClaimBcsrComponentParams,
    reduce_params: RegistersClaimBcsrReduceParams,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RegistersClaimBcsrComponentObservation {
    pub components: RegistersClaimLinearComponents<AkitaField>,
    pub dispatches: usize,
    pub gpu_active: Duration,
    pub resident_wall: Duration,
}

impl SolinasMetal {
    pub(crate) fn prepare_registers_claim_bcsr_components(
        &self,
        bcsr: &RegisterBcsr256,
        tau: &[AkitaField],
        config: RegistersClaimBcsrKernelConfig,
    ) -> Result<RegistersClaimBcsrComponentInvocation, RegistersClaimBcsrRuntimeError> {
        if self.offset != REGISTERS_CLAIM_AKITA_OFFSET {
            return Err(RegistersClaimBcsrRuntimeError::UnsupportedOffset {
                expected: REGISTERS_CLAIM_AKITA_OFFSET,
                got: self.offset,
            });
        }
        let max_buffer_length = usize::try_from(self.device.max_buffer_length())
            .map_err(|_| MetalError::InputTooLong(bcsr.geometry().cycles()))?;
        let plan = RegistersClaimBcsrExecutionPlan::new(bcsr, max_buffer_length, config)?;
        if tau.len() != plan.geometry.log_t() {
            return Err(RegistersClaimBcsrRuntimeError::WrongPointLength {
                expected: plan.geometry.log_t(),
                actual: tau.len(),
            });
        }
        let eq_suffix = EqPolynomial::<AkitaField>::evals(
            &tau[..plan.geometry.suffix_vars()],
            None,
        )
        .iter()
        .map(Fp128::from_jolt_field)
        .collect::<Vec<_>>();
        self.validate_inputs("registers claim BCSR equality suffix", &eq_suffix)?;

        let component_pipeline = self.compile_named_pipeline(BCSR_COMPONENT_PIPELINE)?;
        let reduce_pipeline = self.compile_named_pipeline(BCSR_COMPONENT_REDUCE_PIPELINE)?;
        let component_limits = Self::limits(&component_pipeline);
        let reduce_limits = Self::limits(&reduce_pipeline);
        let component_width = Self::resolve_threadgroup_width(
            Some(BCSR_COMPONENT_THREADS_PER_THREADGROUP as usize),
            component_limits,
        )?;
        let reduce_width = Self::resolve_threadgroup_width(
            Some(BCSR_COMPONENT_REDUCE_THREADS_PER_THREADGROUP as usize),
            reduce_limits,
        )?;
        if component_width != BCSR_COMPONENT_THREADS_PER_THREADGROUP as usize
            || reduce_width != BCSR_COMPONENT_REDUCE_THREADS_PER_THREADGROUP as usize
        {
            return Err(RegistersClaimBcsrRuntimeError::InvalidState(
                "resolved threadgroup widths differ from the BCSR ABI",
            ));
        }
        let requested_threadgroup_bytes = BCSR_COMPONENT_THREADGROUP_BYTES
            .checked_add(component_limits.static_threadgroup_memory_length)
            .ok_or(MetalError::InputTooLong(plan.geometry.rows()))?;
        let maximum = self.device.max_threadgroup_memory_length();
        if requested_threadgroup_bytes > maximum {
            return Err(RegistersClaimBcsrRuntimeError::ThreadgroupMemory {
                requested: requested_threadgroup_bytes,
                maximum,
            });
        }

        let eq_bytes = eq_suffix.len() * size_of::<Fp128>();
        let additional = plan
            .source_bytes
            .checked_add(eq_bytes)
            .and_then(|bytes| bytes.checked_add(plan.partial_bytes))
            .and_then(|bytes| bytes.checked_add(plan.component_bytes))
            .ok_or(MetalError::InputTooLong(plan.geometry.rows()))?;
        self.validate_additional_working_set(to_u64(additional)?)?;

        let parts = bcsr.parts();
        let sources = RegistersClaimBcsrSourceBuffers {
            start_values: buffer_from_slice(&self.device, &parts.start_values),
            rs1_offsets: buffer_from_slice(&self.device, &parts.rs1_offsets),
            rs1_positions: buffer_from_slice(&self.device, &parts.rs1_positions),
            rs2_offsets: buffer_from_slice(&self.device, &parts.rs2_offsets),
            rs2_positions: buffer_from_slice(&self.device, &parts.rs2_positions),
            rd_offsets: buffer_from_slice(&self.device, &parts.rd_offsets),
            rd_positions: buffer_from_slice(&self.device, &parts.rd_positions),
            rd_post_values: buffer_from_slice(&self.device, &parts.rd_post_values),
        };
        let working = RegistersClaimBcsrWorkingBuffers {
            eq_suffix: buffer_from_slice(&self.device, &eq_suffix),
            partials: self.device.new_buffer(
                to_u64(plan.partial_bytes)?,
                MTLResourceOptions::StorageModePrivate,
            ),
            components: self.device.new_buffer(
                to_u64(plan.component_bytes)?,
                MTLResourceOptions::StorageModeShared,
            ),
        };

        Ok(RegistersClaimBcsrComponentInvocation {
            context: self.clone(),
            component_pipeline,
            reduce_pipeline,
            component_limits,
            reduce_limits,
            sources,
            working,
            plan,
            component_params: plan.component_params()?,
            reduce_params: plan.reduce_params()?,
        })
    }
}

impl RegistersClaimBcsrComponentInvocation {
    pub(crate) fn execute_timed(
        &self,
    ) -> Result<RegistersClaimBcsrComponentObservation, RegistersClaimBcsrRuntimeError> {
        let wall_started = Instant::now();
        self.validate_state()?;
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let component_encoder = command_buffer.new_compute_command_encoder();
            component_encoder.set_compute_pipeline_state(&self.component_pipeline);
            component_encoder.set_buffer(
                BCSR_COMPONENT_START_VALUES_SLOT,
                Some(&self.sources.start_values),
                0,
            );
            component_encoder.set_buffer(
                BCSR_COMPONENT_RS1_OFFSETS_SLOT,
                Some(&self.sources.rs1_offsets),
                0,
            );
            component_encoder.set_buffer(
                BCSR_COMPONENT_RS1_POSITIONS_SLOT,
                Some(&self.sources.rs1_positions),
                0,
            );
            component_encoder.set_buffer(
                BCSR_COMPONENT_RS2_OFFSETS_SLOT,
                Some(&self.sources.rs2_offsets),
                0,
            );
            component_encoder.set_buffer(
                BCSR_COMPONENT_RS2_POSITIONS_SLOT,
                Some(&self.sources.rs2_positions),
                0,
            );
            component_encoder.set_buffer(
                BCSR_COMPONENT_RD_OFFSETS_SLOT,
                Some(&self.sources.rd_offsets),
                0,
            );
            component_encoder.set_buffer(
                BCSR_COMPONENT_RD_POSITIONS_SLOT,
                Some(&self.sources.rd_positions),
                0,
            );
            component_encoder.set_buffer(
                BCSR_COMPONENT_RD_POST_VALUES_SLOT,
                Some(&self.sources.rd_post_values),
                0,
            );
            component_encoder.set_buffer(
                BCSR_COMPONENT_EQ_SUFFIX_SLOT,
                Some(&self.working.eq_suffix),
                0,
            );
            component_encoder.set_buffer(
                BCSR_COMPONENT_PARTIALS_SLOT,
                Some(&self.working.partials),
                0,
            );
            set_inline_bytes(
                component_encoder,
                BCSR_COMPONENT_PARAMS_SLOT,
                &self.component_params,
            );
            component_encoder.set_threadgroup_memory_length(
                BCSR_COMPONENT_THREADGROUP_SLOT,
                BCSR_COMPONENT_THREADGROUP_BYTES,
            );
            component_encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.plan.component_threadgroups() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: BCSR_COMPONENT_THREADS_PER_THREADGROUP,
                    height: 1,
                    depth: 1,
                },
            );
            component_encoder.end_encoding();

            let reduce_encoder = command_buffer.new_compute_command_encoder();
            reduce_encoder.set_compute_pipeline_state(&self.reduce_pipeline);
            reduce_encoder.set_buffer(
                BCSR_COMPONENT_REDUCE_INPUT_SLOT,
                Some(&self.working.partials),
                0,
            );
            reduce_encoder.set_buffer(
                BCSR_COMPONENT_REDUCE_OUTPUT_SLOT,
                Some(&self.working.components),
                0,
            );
            set_inline_bytes(
                reduce_encoder,
                BCSR_COMPONENT_REDUCE_PARAMS_SLOT,
                &self.reduce_params,
            );
            reduce_encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.plan.reduce_threadgroups() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: BCSR_COMPONENT_REDUCE_THREADS_PER_THREADGROUP,
                    height: 1,
                    depth: 1,
                },
            );
            reduce_encoder.end_encoding();

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
            Ok(RegistersClaimBcsrComponentObservation {
                components: self.read_components()?,
                dispatches: 2,
                gpu_active: Duration::from_secs_f64(end - start),
                resident_wall: wall_started.elapsed(),
            })
        })
    }

    fn validate_state(&self) -> Result<(), RegistersClaimBcsrRuntimeError> {
        if self.context.offset != REGISTERS_CLAIM_AKITA_OFFSET
            || self.component_params != self.plan.component_params()?
            || self.reduce_params != self.plan.reduce_params()?
            || self.component_limits.max_total_threads_per_threadgroup
                < BCSR_COMPONENT_THREADS_PER_THREADGROUP as usize
            || self.reduce_limits.max_total_threads_per_threadgroup
                < BCSR_COMPONENT_REDUCE_THREADS_PER_THREADGROUP as usize
        {
            return Err(RegistersClaimBcsrRuntimeError::InvalidState(
                "prepared BCSR dispatch metadata changed",
            ));
        }
        Ok(())
    }

    fn read_components(
        &self,
    ) -> Result<RegistersClaimLinearComponents<AkitaField>, RegistersClaimBcsrRuntimeError> {
        let fields = REGISTERS_CLAIM_OUTPUT_COLUMNS * self.plan.geometry.prefix_elements();
        // SAFETY: the shared buffer owns exactly `fields` Fp128 values and the
        // command has completed before this read.
        let values = unsafe {
            slice::from_raw_parts(self.working.components.contents().cast::<Fp128>(), fields)
        };
        self.context
            .validate_inputs("registers claim BCSR components", values)?;
        let mut columns = values.chunks_exact(self.plan.geometry.prefix_elements());
        let decode = |values: &[Fp128]| {
            values
                .iter()
                .map(|&value| value.into_jolt_field())
                .collect::<Vec<_>>()
        };
        let rd_write_value = columns
            .next()
            .ok_or(RegistersClaimBcsrRuntimeError::InvalidState(
                "missing BCSR rd component",
            ))?;
        let rs1_value = columns
            .next()
            .ok_or(RegistersClaimBcsrRuntimeError::InvalidState(
                "missing BCSR rs1 component",
            ))?;
        let rs2_value = columns
            .next()
            .ok_or(RegistersClaimBcsrRuntimeError::InvalidState(
                "missing BCSR rs2 component",
            ))?;
        if !columns.remainder().is_empty() {
            return Err(RegistersClaimBcsrRuntimeError::InvalidState(
                "BCSR component buffer has a partial column",
            ));
        }
        Ok(RegistersClaimLinearComponents {
            rd_write_value: decode(rd_write_value),
            rs1_value: decode(rs1_value),
            rs2_value: decode(rs2_value),
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

const _: () = assert!(REGISTER_BCSR_OFFSET_ENTRIES == 129);
