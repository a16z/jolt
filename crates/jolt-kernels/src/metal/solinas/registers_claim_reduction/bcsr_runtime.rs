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
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
use thiserror::Error;

use super::super::registers_read_write_v3::{
    RegisterBcsr256, RegisterBcsrGeometry, RegisterBcsrLayout, RegistersRwV3Error,
    REGISTER_BCSR_OFFSET_ENTRIES, REGISTER_BCSR_POSITION_SLOTS, REGISTER_CSR_COLUMNS,
};
use super::super::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};
use super::{
    RegistersClaimBcsrComponentParams, RegistersClaimBcsrKernelConfig,
    RegistersClaimBcsrReduceParams, RegistersClaimBcsrReplayStrategy, RegistersClaimGeometry,
    RegistersClaimLinearComponents, RegistersClaimPlanError, BCSR_COMPONENT_EQ_SUFFIX_SLOT,
    BCSR_COMPONENT_PARAMS_SLOT, BCSR_COMPONENT_PARTIALS_SLOT, BCSR_COMPONENT_RD_OFFSETS_SLOT,
    BCSR_COMPONENT_RD_POSITIONS_SLOT, BCSR_COMPONENT_RD_POST_VALUES_SLOT,
    BCSR_COMPONENT_REDUCE_INPUT_SLOT, BCSR_COMPONENT_REDUCE_OUTPUT_SLOT,
    BCSR_COMPONENT_REDUCE_PARAMS_SLOT, BCSR_COMPONENT_REDUCE_PIPELINE,
    BCSR_COMPONENT_REDUCE_THREADS_PER_THREADGROUP, BCSR_COMPONENT_RS1_OFFSETS_SLOT,
    BCSR_COMPONENT_RS1_POSITIONS_SLOT, BCSR_COMPONENT_RS2_OFFSETS_SLOT,
    BCSR_COMPONENT_RS2_POSITIONS_SLOT, BCSR_COMPONENT_START_VALUES_SLOT,
    BCSR_COMPONENT_THREADGROUP_SLOT, BCSR_COMPONENT_THREADS_PER_THREADGROUP,
    BCSR_INDEXED_EQ_SUFFIX_SLOT, BCSR_INDEXED_PARAMS_SLOT, BCSR_INDEXED_PARTIALS_SLOT,
    BCSR_INDEXED_RD_OFFSETS_SLOT, BCSR_INDEXED_RD_POSITIONS_SLOT, BCSR_INDEXED_RD_POST_VALUES_SLOT,
    BCSR_INDEXED_RS1_INDEX_SLOT, BCSR_INDEXED_RS2_INDEX_SLOT, BCSR_INDEXED_START_VALUES_SLOT,
    REGISTERS_CLAIM_AKITA_OFFSET, REGISTERS_CLAIM_OUTPUT_COLUMNS,
};

#[cfg(feature = "test-utils")]
use super::registers_claim_q_checksum;

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
    #[error("registers claim BCSR needs a prefix domain divisible by 256, got {prefix_elements}")]
    UnsupportedPrefix { prefix_elements: usize },
    #[error("invalid registers claim BCSR state: {0}")]
    InvalidState(&'static str),
    #[error("registers claim BCSR source buffers alias")]
    AliasedBuffers,
    #[error("{name} buffer belongs to Metal device {got}, expected {expected}")]
    BufferDevice {
        name: &'static str,
        expected: u64,
        got: u64,
    },
    #[error("{name} buffer has {actual} bytes, expected {expected}")]
    BufferLength {
        name: &'static str,
        expected: u64,
        actual: u64,
    },
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
    layout: RegisterBcsrLayout,
    replay: RegistersClaimBcsrReplayStrategy,
}

impl RegistersClaimBcsrExecutionPlan {
    fn new(
        bcsr: &RegisterBcsr256,
        max_buffer_length: usize,
        config: RegistersClaimBcsrKernelConfig,
    ) -> Result<Self, RegistersClaimBcsrRuntimeError> {
        bcsr.validate()?;
        Self::for_geometry(bcsr.geometry(), max_buffer_length, config)
    }

    fn for_cycles(
        cycles: usize,
        max_buffer_length: usize,
        config: RegistersClaimBcsrKernelConfig,
    ) -> Result<Self, RegistersClaimBcsrRuntimeError> {
        Self::for_geometry(
            RegisterBcsrGeometry::new(cycles)?,
            max_buffer_length,
            config,
        )
    }

    fn for_geometry(
        bcsr_geometry: RegisterBcsrGeometry,
        max_buffer_length: usize,
        config: RegistersClaimBcsrKernelConfig,
    ) -> Result<Self, RegistersClaimBcsrRuntimeError> {
        let cycles = bcsr_geometry.cycles();
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
        let blocks = bcsr_geometry.blocks();
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
        let layout = RegisterBcsrLayout::new(bcsr_geometry)?;
        let source_bytes = match config.replay {
            RegistersClaimBcsrReplayStrategy::ColumnReplay => layout.topology_bytes()?,
            RegistersClaimBcsrReplayStrategy::IndexedPredecessor => checked_sum(&[
                layout.start_values().bytes(),
                layout.offsets().bytes(),
                layout.positions().bytes(),
                layout.rd_post_values().bytes(),
                checked_bytes("BCSR read-index planes", cycles, 2)?,
            ])?,
        };
        for (name, bytes) in [
            ("BCSR start values", layout.start_values().bytes()),
            ("BCSR offsets", layout.offsets().bytes()),
            ("BCSR positions", layout.positions().bytes()),
            ("BCSR rd post values", layout.rd_post_values().bytes()),
            (
                "BCSR read index",
                checked_bytes("BCSR read index", cycles, 1)?,
            ),
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
            layout,
            replay: config.replay,
        })
    }

    fn component_params(
        self,
    ) -> Result<RegistersClaimBcsrComponentParams, RegistersClaimPlanError> {
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

struct RegistersClaimBcsrColumnSourceBuffers {
    start_values: Buffer,
    rs1_offsets: Buffer,
    rs1_positions: Buffer,
    rs2_offsets: Buffer,
    rs2_positions: Buffer,
    rd_offsets: Buffer,
    rd_positions: Buffer,
    rd_post_values: Buffer,
}

impl RegistersClaimBcsrColumnSourceBuffers {
    fn validate(
        &self,
        context: &SolinasMetal,
        plan: RegistersClaimBcsrExecutionPlan,
    ) -> Result<(), RegistersClaimBcsrRuntimeError> {
        let layout = plan.layout;
        let buffers = [
            (
                "BCSR start values",
                &self.start_values,
                layout.start_values().bytes(),
            ),
            (
                "BCSR rs1 offsets",
                &self.rs1_offsets,
                layout.offsets().bytes(),
            ),
            (
                "BCSR rs1 positions",
                &self.rs1_positions,
                layout.positions().bytes(),
            ),
            (
                "BCSR rs2 offsets",
                &self.rs2_offsets,
                layout.offsets().bytes(),
            ),
            (
                "BCSR rs2 positions",
                &self.rs2_positions,
                layout.positions().bytes(),
            ),
            (
                "BCSR rd offsets",
                &self.rd_offsets,
                layout.offsets().bytes(),
            ),
            (
                "BCSR rd positions",
                &self.rd_positions,
                layout.positions().bytes(),
            ),
            (
                "BCSR rd post values",
                &self.rd_post_values,
                layout.rd_post_values().bytes(),
            ),
        ];
        validate_source_buffers(context, &buffers)
    }
}

struct RegistersClaimBcsrIndexedSourceBuffers {
    start_values: Buffer,
    rd_offsets: Buffer,
    rd_positions: Buffer,
    rd_post_values: Buffer,
    rs1_index: Buffer,
    rs2_index: Buffer,
}

impl RegistersClaimBcsrIndexedSourceBuffers {
    fn validate(
        &self,
        context: &SolinasMetal,
        plan: RegistersClaimBcsrExecutionPlan,
    ) -> Result<(), RegistersClaimBcsrRuntimeError> {
        let layout = plan.layout;
        let buffers = [
            (
                "BCSR start values",
                &self.start_values,
                layout.start_values().bytes(),
            ),
            (
                "BCSR rd offsets",
                &self.rd_offsets,
                layout.offsets().bytes(),
            ),
            (
                "BCSR rd positions",
                &self.rd_positions,
                layout.positions().bytes(),
            ),
            (
                "BCSR rd post values",
                &self.rd_post_values,
                layout.rd_post_values().bytes(),
            ),
            ("BCSR rs1 index", &self.rs1_index, plan.geometry.rows()),
            ("BCSR rs2 index", &self.rs2_index, plan.geometry.rows()),
        ];
        validate_source_buffers(context, &buffers)
    }
}

enum RegistersClaimBcsrSourceBuffers {
    Column(RegistersClaimBcsrColumnSourceBuffers),
    Indexed(RegistersClaimBcsrIndexedSourceBuffers),
}

impl RegistersClaimBcsrSourceBuffers {
    fn replay(&self) -> RegistersClaimBcsrReplayStrategy {
        match self {
            Self::Column(_) => RegistersClaimBcsrReplayStrategy::ColumnReplay,
            Self::Indexed(_) => RegistersClaimBcsrReplayStrategy::IndexedPredecessor,
        }
    }

    fn validate(
        &self,
        context: &SolinasMetal,
        plan: RegistersClaimBcsrExecutionPlan,
    ) -> Result<(), RegistersClaimBcsrRuntimeError> {
        if self.replay() != plan.replay {
            return Err(RegistersClaimBcsrRuntimeError::InvalidState(
                "BCSR source topology does not match the replay strategy",
            ));
        }
        match self {
            Self::Column(sources) => sources.validate(context, plan),
            Self::Indexed(sources) => sources.validate(context, plan),
        }
    }
}

fn validate_source_buffers(
    context: &SolinasMetal,
    buffers: &[(&'static str, &Buffer, usize)],
) -> Result<(), RegistersClaimBcsrRuntimeError> {
    let expected_device = context.device_registry_id();
    let mut identities = Vec::with_capacity(buffers.len());
    for &(name, buffer, bytes) in buffers {
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
        identities.push(buffer.as_ptr() as usize);
    }
    for left in 0..identities.len() {
        for right in left + 1..identities.len() {
            if identities[left] == identities[right] {
                return Err(RegistersClaimBcsrRuntimeError::AliasedBuffers);
            }
        }
    }
    Ok(())
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

#[cfg(feature = "test-utils")]
#[derive(Debug, Error)]
pub enum RegistersClaimBcsrBenchmarkError {
    #[error(transparent)]
    Plan(#[from] RegistersClaimPlanError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("registers claim BCSR benchmark setup failed: {0}")]
    InvalidState(String),
}

#[cfg(feature = "test-utils")]
impl From<RegistersClaimBcsrRuntimeError> for RegistersClaimBcsrBenchmarkError {
    fn from(error: RegistersClaimBcsrRuntimeError) -> Self {
        match error {
            RegistersClaimBcsrRuntimeError::Plan(error) => Self::Plan(error),
            RegistersClaimBcsrRuntimeError::Metal(error) => Self::Metal(error),
            error => Self::InvalidState(error.to_string()),
        }
    }
}

#[cfg(feature = "test-utils")]
pub struct RegistersClaimBcsrBenchmarkInvocation {
    inner: RegistersClaimBcsrComponentInvocation,
    event_counts: [u64; 3],
    setup_wall: Duration,
}

#[cfg(feature = "test-utils")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimBcsrBenchmarkObservation {
    pub checksum: u64,
    pub event_counts: [u64; 3],
    pub dispatches: usize,
    pub partial_blocks: usize,
    pub replay: RegistersClaimBcsrReplayStrategy,
    pub component_threadgroups: usize,
    pub source_bytes: usize,
    pub partial_bytes: usize,
    pub setup_wall: Duration,
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
        if config.replay != RegistersClaimBcsrReplayStrategy::ColumnReplay {
            return Err(RegistersClaimBcsrRuntimeError::InvalidState(
                "column BCSR preparation requires the column-replay strategy",
            ));
        }
        if self.offset != REGISTERS_CLAIM_AKITA_OFFSET {
            return Err(RegistersClaimBcsrRuntimeError::UnsupportedOffset {
                expected: REGISTERS_CLAIM_AKITA_OFFSET,
                got: self.offset,
            });
        }
        let max_buffer_length = usize::try_from(self.device.max_buffer_length())
            .map_err(|_| MetalError::InputTooLong(bcsr.geometry().cycles()))?;
        let plan = RegistersClaimBcsrExecutionPlan::new(bcsr, max_buffer_length, config)?;
        self.validate_registers_claim_bcsr_working_set(plan)?;
        let parts = bcsr.parts();
        let sources =
            RegistersClaimBcsrSourceBuffers::Column(RegistersClaimBcsrColumnSourceBuffers {
                start_values: buffer_from_slice(&self.device, &parts.start_values),
                rs1_offsets: buffer_from_slice(&self.device, &parts.rs1_offsets),
                rs1_positions: buffer_from_slice(&self.device, &parts.rs1_positions),
                rs2_offsets: buffer_from_slice(&self.device, &parts.rs2_offsets),
                rs2_positions: buffer_from_slice(&self.device, &parts.rs2_positions),
                rd_offsets: buffer_from_slice(&self.device, &parts.rd_offsets),
                rd_positions: buffer_from_slice(&self.device, &parts.rd_positions),
                rd_post_values: buffer_from_slice(&self.device, &parts.rd_post_values),
            });
        self.prepare_registers_claim_bcsr_sources(sources, plan, tau)
    }

    pub(crate) fn prepare_registers_claim_bcsr_indexed_components(
        &self,
        bcsr: &RegisterBcsr256,
        rs1_index: &[u8],
        rs2_index: &[u8],
        tau: &[AkitaField],
        config: RegistersClaimBcsrKernelConfig,
    ) -> Result<RegistersClaimBcsrComponentInvocation, RegistersClaimBcsrRuntimeError> {
        if config.replay != RegistersClaimBcsrReplayStrategy::IndexedPredecessor {
            return Err(RegistersClaimBcsrRuntimeError::InvalidState(
                "indexed BCSR preparation requires the indexed-predecessor strategy",
            ));
        }
        if self.offset != REGISTERS_CLAIM_AKITA_OFFSET {
            return Err(RegistersClaimBcsrRuntimeError::UnsupportedOffset {
                expected: REGISTERS_CLAIM_AKITA_OFFSET,
                got: self.offset,
            });
        }
        let max_buffer_length = usize::try_from(self.device.max_buffer_length())
            .map_err(|_| MetalError::InputTooLong(bcsr.geometry().cycles()))?;
        let plan = RegistersClaimBcsrExecutionPlan::new(bcsr, max_buffer_length, config)?;
        validate_register_index_plane("rs1", rs1_index, plan.geometry.rows())?;
        validate_register_index_plane("rs2", rs2_index, plan.geometry.rows())?;
        self.validate_registers_claim_bcsr_working_set(plan)?;
        let parts = bcsr.parts();
        let sources =
            RegistersClaimBcsrSourceBuffers::Indexed(RegistersClaimBcsrIndexedSourceBuffers {
                start_values: buffer_from_slice(&self.device, &parts.start_values),
                rd_offsets: buffer_from_slice(&self.device, &parts.rd_offsets),
                rd_positions: buffer_from_slice(&self.device, &parts.rd_positions),
                rd_post_values: buffer_from_slice(&self.device, &parts.rd_post_values),
                rs1_index: buffer_from_slice(&self.device, rs1_index),
                rs2_index: buffer_from_slice(&self.device, rs2_index),
            });
        self.prepare_registers_claim_bcsr_sources(sources, plan, tau)
    }

    #[cfg(feature = "test-utils")]
    pub fn prepare_registers_claim_bcsr_benchmark(
        &self,
        cycles: usize,
        tau: &[AkitaField],
        config: RegistersClaimBcsrKernelConfig,
    ) -> Result<RegistersClaimBcsrBenchmarkInvocation, RegistersClaimBcsrBenchmarkError> {
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
        let plan = RegistersClaimBcsrExecutionPlan::for_cycles(cycles, max_buffer_length, config)?;
        self.validate_registers_claim_bcsr_working_set(plan)?;
        let sources = self.allocate_synthetic_registers_claim_bcsr_sources(plan)?;
        let inner = self.prepare_registers_claim_bcsr_sources(sources, plan, tau)?;
        let blocks =
            u64::try_from(plan.blocks).map_err(|_| MetalError::InputTooLong(plan.blocks))?;
        Ok(RegistersClaimBcsrBenchmarkInvocation {
            inner,
            event_counts: [
                blocks * SYNTHETIC_RS1_EVENTS_PER_BLOCK as u64,
                blocks * SYNTHETIC_RS2_EVENTS_PER_BLOCK as u64,
                blocks * SYNTHETIC_RD_EVENTS_PER_BLOCK as u64,
            ],
            setup_wall: setup_started.elapsed(),
        })
    }

    #[cfg(feature = "test-utils")]
    fn allocate_synthetic_registers_claim_bcsr_sources(
        &self,
        plan: RegistersClaimBcsrExecutionPlan,
    ) -> Result<RegistersClaimBcsrSourceBuffers, RegistersClaimBcsrRuntimeError> {
        let layout = plan.layout;
        let shared = MTLResourceOptions::StorageModeShared;
        match plan.replay {
            RegistersClaimBcsrReplayStrategy::ColumnReplay => {
                let mut sources = RegistersClaimBcsrColumnSourceBuffers {
                    start_values: self
                        .device
                        .new_buffer(to_u64(layout.start_values().bytes())?, shared),
                    rs1_offsets: self
                        .device
                        .new_buffer(to_u64(layout.offsets().bytes())?, shared),
                    rs1_positions: self
                        .device
                        .new_buffer(to_u64(layout.positions().bytes())?, shared),
                    rs2_offsets: self
                        .device
                        .new_buffer(to_u64(layout.offsets().bytes())?, shared),
                    rs2_positions: self
                        .device
                        .new_buffer(to_u64(layout.positions().bytes())?, shared),
                    rd_offsets: self
                        .device
                        .new_buffer(to_u64(layout.offsets().bytes())?, shared),
                    rd_positions: self
                        .device
                        .new_buffer(to_u64(layout.positions().bytes())?, shared),
                    rd_post_values: self
                        .device
                        .new_buffer(to_u64(layout.rd_post_values().bytes())?, shared),
                };
                sources.validate(self, plan)?;

                // SAFETY: every shared buffer has the exact element count recorded by
                // `layout`, and no command observes it until all fills return.
                unsafe {
                    fill_synthetic_start_values(shared_slice_mut(
                        &mut sources.start_values,
                        layout.start_values().elements(),
                    ));
                    fill_synthetic_position_plane(
                        shared_slice_mut(&mut sources.rs1_offsets, layout.offsets().elements()),
                        shared_slice_mut(&mut sources.rs1_positions, layout.positions().elements()),
                        plan.blocks,
                        SYNTHETIC_RS1_EVENTS_PER_BLOCK,
                        3,
                    );
                    fill_synthetic_position_plane(
                        shared_slice_mut(&mut sources.rs2_offsets, layout.offsets().elements()),
                        shared_slice_mut(&mut sources.rs2_positions, layout.positions().elements()),
                        plan.blocks,
                        SYNTHETIC_RS2_EVENTS_PER_BLOCK,
                        5,
                    );
                    fill_synthetic_rd_plane(
                        shared_slice_mut(&mut sources.rd_offsets, layout.offsets().elements()),
                        shared_slice_mut(&mut sources.rd_positions, layout.positions().elements()),
                        shared_slice_mut(
                            &mut sources.rd_post_values,
                            layout.rd_post_values().elements(),
                        ),
                        plan.blocks,
                    );
                }
                Ok(RegistersClaimBcsrSourceBuffers::Column(sources))
            }
            RegistersClaimBcsrReplayStrategy::IndexedPredecessor => {
                let mut sources = RegistersClaimBcsrIndexedSourceBuffers {
                    start_values: self
                        .device
                        .new_buffer(to_u64(layout.start_values().bytes())?, shared),
                    rd_offsets: self
                        .device
                        .new_buffer(to_u64(layout.offsets().bytes())?, shared),
                    rd_positions: self
                        .device
                        .new_buffer(to_u64(layout.positions().bytes())?, shared),
                    rd_post_values: self
                        .device
                        .new_buffer(to_u64(layout.rd_post_values().bytes())?, shared),
                    rs1_index: self
                        .device
                        .new_buffer(to_u64(plan.geometry.rows())?, shared),
                    rs2_index: self
                        .device
                        .new_buffer(to_u64(plan.geometry.rows())?, shared),
                };
                sources.validate(self, plan)?;

                // SAFETY: every shared buffer has the checked element count, and the
                // benchmark invocation is not visible until all fills return.
                unsafe {
                    fill_synthetic_start_values(shared_slice_mut(
                        &mut sources.start_values,
                        layout.start_values().elements(),
                    ));
                    fill_synthetic_rd_plane(
                        shared_slice_mut(&mut sources.rd_offsets, layout.offsets().elements()),
                        shared_slice_mut(&mut sources.rd_positions, layout.positions().elements()),
                        shared_slice_mut(
                            &mut sources.rd_post_values,
                            layout.rd_post_values().elements(),
                        ),
                        plan.blocks,
                    );
                    fill_synthetic_index_plane(
                        shared_slice_mut(&mut sources.rs1_index, plan.geometry.rows()),
                        plan.blocks,
                        SYNTHETIC_RS1_EVENTS_PER_BLOCK,
                        3,
                    );
                    fill_synthetic_index_plane(
                        shared_slice_mut(&mut sources.rs2_index, plan.geometry.rows()),
                        plan.blocks,
                        SYNTHETIC_RS2_EVENTS_PER_BLOCK,
                        5,
                    );
                }
                Ok(RegistersClaimBcsrSourceBuffers::Indexed(sources))
            }
        }
    }

    fn validate_registers_claim_bcsr_working_set(
        &self,
        plan: RegistersClaimBcsrExecutionPlan,
    ) -> Result<(), RegistersClaimBcsrRuntimeError> {
        let eq_bytes = checked_bytes(
            "BCSR equality suffix",
            plan.geometry.suffix_elements(),
            size_of::<Fp128>(),
        )?;
        let additional = plan
            .source_bytes
            .checked_add(eq_bytes)
            .and_then(|bytes| bytes.checked_add(plan.partial_bytes))
            .and_then(|bytes| bytes.checked_add(plan.component_bytes))
            .ok_or(MetalError::InputTooLong(plan.geometry.rows()))?;
        self.validate_additional_working_set(to_u64(additional)?)?;
        Ok(())
    }

    fn prepare_registers_claim_bcsr_sources(
        &self,
        sources: RegistersClaimBcsrSourceBuffers,
        plan: RegistersClaimBcsrExecutionPlan,
        tau: &[AkitaField],
    ) -> Result<RegistersClaimBcsrComponentInvocation, RegistersClaimBcsrRuntimeError> {
        sources.validate(self, plan)?;
        if tau.len() != plan.geometry.log_t() {
            return Err(RegistersClaimBcsrRuntimeError::WrongPointLength {
                expected: plan.geometry.log_t(),
                actual: tau.len(),
            });
        }
        let eq_suffix =
            EqPolynomial::<AkitaField>::evals(&tau[..plan.geometry.suffix_vars()], None)
                .iter()
                .map(Fp128::from_jolt_field)
                .collect::<Vec<_>>();
        self.validate_inputs("registers claim BCSR equality suffix", &eq_suffix)?;

        let component_pipeline = self.compile_named_pipeline(plan.replay.component_pipeline())?;
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
        let requested_threadgroup_bytes = plan
            .replay
            .threadgroup_bytes()
            .checked_add(component_limits.static_threadgroup_memory_length)
            .ok_or(MetalError::InputTooLong(plan.geometry.rows()))?;
        let maximum = self.device.max_threadgroup_memory_length();
        if requested_threadgroup_bytes > maximum {
            return Err(RegistersClaimBcsrRuntimeError::ThreadgroupMemory {
                requested: requested_threadgroup_bytes,
                maximum,
            });
        }

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
            match &self.sources {
                RegistersClaimBcsrSourceBuffers::Column(sources) => {
                    component_encoder.set_buffer(
                        BCSR_COMPONENT_START_VALUES_SLOT,
                        Some(&sources.start_values),
                        0,
                    );
                    component_encoder.set_buffer(
                        BCSR_COMPONENT_RS1_OFFSETS_SLOT,
                        Some(&sources.rs1_offsets),
                        0,
                    );
                    component_encoder.set_buffer(
                        BCSR_COMPONENT_RS1_POSITIONS_SLOT,
                        Some(&sources.rs1_positions),
                        0,
                    );
                    component_encoder.set_buffer(
                        BCSR_COMPONENT_RS2_OFFSETS_SLOT,
                        Some(&sources.rs2_offsets),
                        0,
                    );
                    component_encoder.set_buffer(
                        BCSR_COMPONENT_RS2_POSITIONS_SLOT,
                        Some(&sources.rs2_positions),
                        0,
                    );
                    component_encoder.set_buffer(
                        BCSR_COMPONENT_RD_OFFSETS_SLOT,
                        Some(&sources.rd_offsets),
                        0,
                    );
                    component_encoder.set_buffer(
                        BCSR_COMPONENT_RD_POSITIONS_SLOT,
                        Some(&sources.rd_positions),
                        0,
                    );
                    component_encoder.set_buffer(
                        BCSR_COMPONENT_RD_POST_VALUES_SLOT,
                        Some(&sources.rd_post_values),
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
                }
                RegistersClaimBcsrSourceBuffers::Indexed(sources) => {
                    component_encoder.set_buffer(
                        BCSR_INDEXED_START_VALUES_SLOT,
                        Some(&sources.start_values),
                        0,
                    );
                    component_encoder.set_buffer(
                        BCSR_INDEXED_RD_OFFSETS_SLOT,
                        Some(&sources.rd_offsets),
                        0,
                    );
                    component_encoder.set_buffer(
                        BCSR_INDEXED_RD_POSITIONS_SLOT,
                        Some(&sources.rd_positions),
                        0,
                    );
                    component_encoder.set_buffer(
                        BCSR_INDEXED_RD_POST_VALUES_SLOT,
                        Some(&sources.rd_post_values),
                        0,
                    );
                    component_encoder.set_buffer(
                        BCSR_INDEXED_RS1_INDEX_SLOT,
                        Some(&sources.rs1_index),
                        0,
                    );
                    component_encoder.set_buffer(
                        BCSR_INDEXED_RS2_INDEX_SLOT,
                        Some(&sources.rs2_index),
                        0,
                    );
                    component_encoder.set_buffer(
                        BCSR_INDEXED_EQ_SUFFIX_SLOT,
                        Some(&self.working.eq_suffix),
                        0,
                    );
                    component_encoder.set_buffer(
                        BCSR_INDEXED_PARTIALS_SLOT,
                        Some(&self.working.partials),
                        0,
                    );
                    set_inline_bytes(
                        component_encoder,
                        BCSR_INDEXED_PARAMS_SLOT,
                        &self.component_params,
                    );
                }
            }
            component_encoder.set_threadgroup_memory_length(
                BCSR_COMPONENT_THREADGROUP_SLOT,
                self.plan.replay.threadgroup_bytes(),
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
        self.sources.validate(&self.context, self.plan)?;
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

#[cfg(feature = "test-utils")]
impl RegistersClaimBcsrBenchmarkInvocation {
    pub fn execute_timed(
        &self,
    ) -> Result<RegistersClaimBcsrBenchmarkObservation, RegistersClaimBcsrBenchmarkError> {
        let observation = self.inner.execute_timed()?;
        let checksum = registers_claim_q_checksum(&observation.components.rd_write_value)
            ^ registers_claim_q_checksum(&observation.components.rs1_value).rotate_left(21)
            ^ registers_claim_q_checksum(&observation.components.rs2_value).rotate_left(42);
        Ok(RegistersClaimBcsrBenchmarkObservation {
            checksum,
            event_counts: self.event_counts,
            dispatches: observation.dispatches,
            partial_blocks: self.inner.plan.partial_blocks,
            replay: self.inner.plan.replay,
            component_threadgroups: self.inner.plan.component_threadgroups(),
            source_bytes: self.inner.plan.source_bytes,
            partial_bytes: self.inner.plan.partial_bytes,
            setup_wall: self.setup_wall,
            gpu_active: observation.gpu_active,
            resident_wall: observation.resident_wall,
        })
    }
}

#[cfg(feature = "test-utils")]
const SYNTHETIC_RS1_EVENTS_PER_BLOCK: usize = 228;
#[cfg(feature = "test-utils")]
const SYNTHETIC_RS2_EVENTS_PER_BLOCK: usize = 213;
#[cfg(feature = "test-utils")]
pub(super) const SYNTHETIC_RD_EVENTS_PER_BLOCK: usize = 192;

#[cfg(feature = "test-utils")]
fn fill_synthetic_start_values(values: &mut [u64]) {
    for (index, value) in values.iter_mut().enumerate() {
        *value = (index as u64)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(0xd1b5_4a32_d192_ed03)
            | 1;
    }
}

#[cfg(feature = "test-utils")]
fn fill_synthetic_position_plane(
    offsets: &mut [u16],
    positions: &mut [u8],
    blocks: usize,
    events_per_block: usize,
    register_multiplier: usize,
) {
    debug_assert_eq!(offsets.len(), blocks * REGISTER_BCSR_OFFSET_ENTRIES);
    debug_assert_eq!(positions.len(), blocks * REGISTER_BCSR_POSITION_SLOTS);
    for block in 0..blocks {
        let offset_block = &mut offsets
            [block * REGISTER_BCSR_OFFSET_ENTRIES..(block + 1) * REGISTER_BCSR_OFFSET_ENTRIES];
        let position_block = &mut positions
            [block * REGISTER_BCSR_POSITION_SLOTS..(block + 1) * REGISTER_BCSR_POSITION_SLOTS];
        position_block.fill(0);
        let mut counts = [0u16; REGISTER_CSR_COLUMNS];
        for event in 0..events_per_block {
            let position = event * REGISTER_BCSR_POSITION_SLOTS / events_per_block;
            counts[synthetic_register(event, position, block, register_multiplier)] += 1;
        }
        offset_block[0] = 0;
        for register in 0..REGISTER_CSR_COLUMNS {
            offset_block[register + 1] = offset_block[register] + counts[register];
        }
        let mut cursors = [0u16; REGISTER_CSR_COLUMNS];
        cursors.copy_from_slice(&offset_block[..REGISTER_CSR_COLUMNS]);
        for event in 0..events_per_block {
            let position = event * REGISTER_BCSR_POSITION_SLOTS / events_per_block;
            let register = synthetic_register(event, position, block, register_multiplier);
            let destination = usize::from(cursors[register]);
            position_block[destination] = position as u8;
            cursors[register] += 1;
        }
    }
}

#[cfg(feature = "test-utils")]
pub(super) fn fill_synthetic_rd_plane(
    offsets: &mut [u16],
    positions: &mut [u8],
    post_values: &mut [u64],
    blocks: usize,
) {
    debug_assert_eq!(post_values.len(), blocks * REGISTER_BCSR_POSITION_SLOTS);
    fill_synthetic_position_plane(offsets, positions, blocks, SYNTHETIC_RD_EVENTS_PER_BLOCK, 7);
    for block in 0..blocks {
        let offset_block = &offsets
            [block * REGISTER_BCSR_OFFSET_ENTRIES..(block + 1) * REGISTER_BCSR_OFFSET_ENTRIES];
        let position_block = &positions
            [block * REGISTER_BCSR_POSITION_SLOTS..(block + 1) * REGISTER_BCSR_POSITION_SLOTS];
        let post_block = &mut post_values
            [block * REGISTER_BCSR_POSITION_SLOTS..(block + 1) * REGISTER_BCSR_POSITION_SLOTS];
        post_block.fill(0);
        let events = usize::from(offset_block[REGISTER_CSR_COLUMNS]);
        for event in 0..events {
            post_block[event] = (block as u64)
                .wrapping_mul(0x94d0_49bb_1331_11eb)
                .wrapping_add(u64::from(position_block[event]) + 1)
                | 1;
        }
    }
}

#[cfg(feature = "test-utils")]
fn fill_synthetic_index_plane(
    values: &mut [u8],
    blocks: usize,
    events_per_block: usize,
    register_multiplier: usize,
) {
    debug_assert_eq!(values.len(), blocks * REGISTER_BCSR_POSITION_SLOTS);
    for block in 0..blocks {
        let block_values = &mut values
            [block * REGISTER_BCSR_POSITION_SLOTS..(block + 1) * REGISTER_BCSR_POSITION_SLOTS];
        block_values.fill(u8::MAX);
        for event in 0..events_per_block {
            let position = event * REGISTER_BCSR_POSITION_SLOTS / events_per_block;
            block_values[position] =
                synthetic_register(event, position, block, register_multiplier) as u8;
        }
    }
}

#[cfg(feature = "test-utils")]
fn synthetic_register(event: usize, position: usize, block: usize, multiplier: usize) -> usize {
    if event.is_multiple_of(8) {
        0
    } else {
        1 + (multiplier * position + block) % (REGISTER_CSR_COLUMNS - 1)
    }
}

#[cfg(feature = "test-utils")]
pub(super) unsafe fn shared_slice_mut<T>(buffer: &mut Buffer, elements: usize) -> &mut [T] {
    // SAFETY: callers validate that the shared allocation contains exactly
    // `elements` values of `T` and hold exclusive setup access.
    unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<T>(), elements) }
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
                name: "BCSR source bytes",
            })
    })
}

fn validate_register_index_plane(
    name: &'static str,
    values: &[u8],
    expected: usize,
) -> Result<(), RegistersClaimBcsrRuntimeError> {
    if values.len() != expected {
        return Err(RegistersClaimBcsrRuntimeError::InvalidState(match name {
            "rs1" => "rs1 index length does not match the BCSR cycle count",
            _ => "rs2 index length does not match the BCSR cycle count",
        }));
    }
    if values
        .iter()
        .any(|&value| value != u8::MAX && usize::from(value) >= REGISTER_CSR_COLUMNS)
    {
        return Err(RegistersClaimBcsrRuntimeError::InvalidState(match name {
            "rs1" => "rs1 index contains an invalid register",
            _ => "rs2 index contains an invalid register",
        }));
    }
    Ok(())
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
