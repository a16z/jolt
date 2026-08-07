use std::{
    slice,
    time::{Duration, Instant},
};

use jolt_field::AkitaField;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLCommandBufferStatus, MTLResourceOptions, NSRange,
};

use super::super::{command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal};
use super::{
    api::{
        OuterRemainderSequenceConfig, OuterRemainderStorageInitialization,
        OuterRemainderStorageInitializationStats, DEVICE_BUFFERS, OUTER_REMAINDER_STREAM_ROWS,
    },
    plan::{
        field_bytes, opening_output_count, outer_remainder_sequence_storage_bytes_with_config,
        storage_geometry, validate_opening_threadgroup_memory, SIMD_WIDTH,
    },
    shader::pipeline_names,
};

pub(super) struct Pipelines {
    pub(super) materialize: ComputePipelineState,
    pub(super) stream_bind: ComputePipelineState,
    pub(super) transition: ComputePipelineState,
    pub(super) opening: ComputePipelineState,
    pub(super) reduction: ComputePipelineState,
}

pub(super) struct PipelineSetLimits {
    pub(super) materialize: PipelineLimits,
    pub(super) stream_bind: PipelineLimits,
    pub(super) transition: PipelineLimits,
    pub(super) opening: PipelineLimits,
    pub(super) reduction: PipelineLimits,
}

pub(super) struct Threads {
    pub(super) materialize: usize,
    pub(super) stream_bind: usize,
    pub(super) transition: usize,
    pub(super) opening: usize,
    pub(super) reduction: usize,
}

pub(super) struct DenseBuffers {
    pub(super) state_a: Buffer,
    pub(super) state_b: Buffer,
}

pub(super) struct Buffers {
    pub(super) dense: Option<DenseBuffers>,
    pub(super) e_in: Buffer,
    pub(super) e_out: Buffer,
    pub(super) lagrange: Buffer,
    pub(super) message_partials: Buffer,
    pub(super) message_output: Buffer,
    pub(super) opening_partials: Buffer,
    pub(super) opening_output: Buffer,
}

impl Buffers {
    pub(super) fn identities(&self) -> [usize; DEVICE_BUFFERS] {
        let dense = self.dense.as_ref().map_or([0, 0], |dense| {
            [
                dense.state_a.as_ptr() as usize,
                dense.state_b.as_ptr() as usize,
            ]
        });
        [
            dense[0],
            dense[1],
            self.e_in.as_ptr() as usize,
            self.e_out.as_ptr() as usize,
            self.lagrange.as_ptr() as usize,
            self.message_partials.as_ptr() as usize,
            self.message_output.as_ptr() as usize,
            self.opening_partials.as_ptr() as usize,
            self.opening_output.as_ptr() as usize,
        ]
    }

    fn all(&self) -> Result<[&Buffer; DEVICE_BUFFERS], MetalError> {
        let dense = self
            .dense
            .as_ref()
            .ok_or(MetalError::InvalidOuterRemainderState {
                expected: "allocated dense storage",
                got: "released dense storage",
            })?;
        Ok([
            &dense.state_a,
            &dense.state_b,
            &self.e_in,
            &self.e_out,
            &self.lagrange,
            &self.message_partials,
            &self.message_output,
            &self.opening_partials,
            &self.opening_output,
        ])
    }
}

pub(super) struct Storage {
    pub(super) context: SolinasMetal,
    pub(super) pipelines: Pipelines,
    pub(super) limits: PipelineSetLimits,
    pub(super) threads: Threads,
    pub(super) buffers: Buffers,
    pub(super) weight_capacity: usize,
    pub(super) max_threadgroups: usize,
    pub(super) dense_bytes: u64,
    pub(super) owned_bytes: u64,
    pub(super) initialization: OuterRemainderStorageInitializationStats,
    #[cfg(feature = "test-utils")]
    pub(super) pipeline_compile_wall: Duration,
}

pub(crate) struct OuterRemainderSequenceStorage {
    pub(super) storage: Storage,
    pub(super) cycles: usize,
    pub(super) config: OuterRemainderSequenceConfig,
    pub(super) current_elements: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for OuterRemainderSequenceStorage {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("device_storage"),
            self.storage.owned_bytes as usize,
        );
        visitor.exit();
    }
}

impl SolinasMetal {
    pub(crate) fn prepare_outer_remainder_sequence_storage(
        &self,
        cycles: usize,
        config: OuterRemainderSequenceConfig,
    ) -> Result<OuterRemainderSequenceStorage, MetalError> {
        if cycles < 4 || !cycles.is_power_of_two() {
            return Err(MetalError::InvalidOuterRemainderRows(cycles));
        }
        let geometry = storage_geometry(cycles, config)?;
        let current_elements = geometry.current_elements;
        let weight_capacity = geometry.weight_capacity;
        let max_threadgroups = geometry.max_threadgroups;

        let names = pipeline_names(config.binding_plan);
        #[cfg(feature = "test-utils")]
        let pipeline_compile_started = Instant::now();
        let pipelines = Pipelines {
            materialize: self.compile_named_pipeline(names.materialize)?,
            stream_bind: self.compile_named_pipeline(names.stream_bind)?,
            transition: self.compile_named_pipeline(names.transition)?,
            opening: self.compile_named_pipeline(names.opening)?,
            reduction: self.compile_named_pipeline(names.reduction)?,
        };
        #[cfg(feature = "test-utils")]
        let pipeline_compile_wall = pipeline_compile_started.elapsed();
        let limits = PipelineSetLimits {
            materialize: Self::limits(&pipelines.materialize),
            stream_bind: Self::limits(&pipelines.stream_bind),
            transition: Self::limits(&pipelines.transition),
            opening: Self::limits(&pipelines.opening),
            reduction: Self::limits(&pipelines.reduction),
        };
        for (pipeline, pipeline_limits) in [
            (names.materialize, limits.materialize),
            (names.stream_bind, limits.stream_bind),
            (names.transition, limits.transition),
            (names.opening, limits.opening),
            (names.reduction, limits.reduction),
        ] {
            if pipeline_limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedOuterRemainderExecutionWidth {
                    pipeline,
                    expected: SIMD_WIDTH,
                    got: pipeline_limits.thread_execution_width,
                });
            }
        }

        let threads = Threads {
            materialize: Self::resolve_threadgroup_width(
                config.materialize_threads_per_threadgroup,
                limits.materialize,
            )?,
            stream_bind: Self::resolve_threadgroup_width(
                config.stream_bind_threads_per_threadgroup,
                limits.stream_bind,
            )?,
            transition: Self::resolve_threadgroup_width(
                config.transition_threads_per_threadgroup,
                limits.transition,
            )?,
            opening: Self::resolve_threadgroup_width(
                config.opening_threads_per_threadgroup,
                limits.opening,
            )?,
            reduction: Self::resolve_threadgroup_width(None, limits.reduction)?,
        };
        if threads.opening < 128 {
            return Err(MetalError::InvalidOuterRemainderConfig(
                "opening threadgroup needs at least 128 threads",
            ));
        }
        validate_opening_threadgroup_memory(
            self,
            limits.opening,
            config.binding_plan,
            threads.opening,
            config.product_uniskip_carrier,
        )?;

        for elements in geometry.element_counts {
            let bytes = field_bytes(elements)?;
            self.validate_buffer_length(bytes)?;
        }
        self.validate_additional_working_set(outer_remainder_sequence_storage_bytes_with_config(
            cycles, config,
        )?)?;
        let dense_bytes = field_bytes(current_elements)?
            .checked_mul(2)
            .ok_or(MetalError::InputTooLong(current_elements))?;
        let opening_outputs = opening_output_count(config.product_uniskip_carrier);
        let buffers = Buffers {
            dense: Some(DenseBuffers {
                state_a: new_field_buffer(self, current_elements)?,
                state_b: new_field_buffer(self, current_elements)?,
            }),
            e_in: new_field_buffer(self, weight_capacity)?,
            e_out: new_field_buffer(self, weight_capacity)?,
            lagrange: new_field_buffer(self, OUTER_REMAINDER_STREAM_ROWS)?,
            message_partials: new_field_buffer(self, 2 * max_threadgroups)?,
            message_output: new_field_buffer(self, 2)?,
            opening_partials: new_field_buffer(self, opening_outputs * max_threadgroups)?,
            opening_output: new_field_buffer(self, opening_outputs)?,
        };
        let initialization = initialize_storage(self, &buffers, config.storage_initialization)?;

        Ok(OuterRemainderSequenceStorage {
            storage: Storage {
                context: self.clone(),
                pipelines,
                limits,
                threads,
                buffers,
                weight_capacity,
                max_threadgroups,
                dense_bytes,
                owned_bytes: geometry.owned_bytes,
                initialization,
                #[cfg(feature = "test-utils")]
                pipeline_compile_wall,
            },
            cycles,
            config,
            current_elements,
        })
    }
}

impl OuterRemainderSequenceStorage {
    pub(crate) fn matches(
        &self,
        context: &SolinasMetal,
        cycles: usize,
        config: OuterRemainderSequenceConfig,
    ) -> bool {
        self.storage.context.device_registry_id() == context.device_registry_id()
            && self.cycles == cycles
            && self.config == config
    }

    pub(crate) const fn owned_bytes(&self) -> u64 {
        self.storage.owned_bytes
    }

    pub(crate) const fn initialization(&self) -> OuterRemainderStorageInitializationStats {
        self.storage.initialization
    }

    #[cfg(feature = "test-utils")]
    pub(crate) const fn pipeline_compile_wall(&self) -> Duration {
        self.storage.pipeline_compile_wall
    }
}

fn initialize_storage(
    context: &SolinasMetal,
    buffers: &Buffers,
    mode: OuterRemainderStorageInitialization,
) -> Result<OuterRemainderStorageInitializationStats, MetalError> {
    let buffer_identities = buffers.identities();
    let buffers = buffers.all()?;
    let bytes = match mode {
        OuterRemainderStorageInitialization::Lazy => 0,
        OuterRemainderStorageInitialization::Full => {
            buffers.iter().try_fold(0u64, |total, buffer| {
                total
                    .checked_add(buffer.length())
                    .ok_or(MetalError::InvalidOuterRemainderConfig(
                        "storage initialization byte count overflowed",
                    ))
            })?
        }
    };
    let device_buffers =
        usize::from(mode == OuterRemainderStorageInitialization::Full) * DEVICE_BUFFERS;
    let span = tracing::info_span!(
        "MetalOuterRemainder::storage_initialize",
        mode = mode.as_str(),
        device_buffers,
        bytes,
        protocol_dispatches = 0u64,
        buffer_0 = buffer_identities[0],
        buffer_1 = buffer_identities[1],
        buffer_2 = buffer_identities[2],
        buffer_3 = buffer_identities[3],
        buffer_4 = buffer_identities[4],
        buffer_5 = buffer_identities[5],
        buffer_6 = buffer_identities[6],
        buffer_7 = buffer_identities[7],
        buffer_8 = buffer_identities[8],
    );
    let _entered = span.enter();
    let started = Instant::now();
    let gpu_active = match mode {
        OuterRemainderStorageInitialization::Lazy => Duration::ZERO,
        OuterRemainderStorageInitialization::Full => {
            let command_buffer = context.queue.new_command_buffer();
            autoreleasepool(|| {
                let encoder = command_buffer.new_blit_command_encoder();
                for buffer in buffers {
                    encoder.fill_buffer(buffer, NSRange::new(0, buffer.length()), 0);
                }
                encoder.end_encoding();
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
            Duration::from_secs_f64(end - start)
        }
    };
    let wall = started.elapsed();
    let gpu_active_ns = u64::try_from(gpu_active.as_nanos()).unwrap_or(u64::MAX);
    let wall_ns = u64::try_from(wall.as_nanos()).unwrap_or(u64::MAX);
    let completion = tracing::info_span!(
        "MetalOuterRemainder::storage_initialize_complete",
        mode = mode.as_str(),
        command_completed = true,
        bytes,
        wall_ns,
        gpu_active_ns,
    );
    let _completed = completion.enter();
    Ok(OuterRemainderStorageInitializationStats {
        mode,
        device_buffers,
        bytes,
        wall,
        gpu_active,
        buffer_identities,
    })
}

pub(super) fn write_fields(
    context: &SolinasMetal,
    buffer: &Buffer,
    capacity: usize,
    values: &[AkitaField],
) -> Result<(), MetalError> {
    if values.len() > capacity {
        return Err(MetalError::OuterRemainderStorageLength {
            name: "field input",
            capacity,
            got: values.len(),
        });
    }
    // SAFETY: no command is active while the host overwrites this prefix and
    // the shared allocation has `capacity` fields.
    let destination =
        unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<Fp128>(), capacity) };
    for (destination, value) in destination.iter_mut().zip(values) {
        *destination = Fp128::from_jolt_field(value);
    }
    context.validate_inputs("outer host input", &destination[..values.len()])?;
    Ok(())
}

fn new_field_buffer(context: &SolinasMetal, elements: usize) -> Result<Buffer, MetalError> {
    let bytes = field_bytes(elements)?;
    context.validate_buffer_length(bytes)?;
    Ok(context
        .device
        .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
}
