use std::{slice, time::Duration};

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLResourceOptions, NSRange,
};

use super::super::{completed_command_gpu_time, Fp128, MetalError, PipelineLimits, SolinasMetal};
use super::{
    api::{
        OuterRemainderSequenceConfig, OuterRemainderStorageInitialization, DEVICE_BUFFERS,
        OUTER_REMAINDER_A_LOOKUP_FIELDS,
    },
    plan::{
        field_bytes, opening_output_count, storage_geometry, validate_opening_threadgroup_memory,
        SIMD_WIDTH,
    },
    registers_claim::{carrier_geometry, RegistersClaimCarrierGeometry},
    shader::{
        pipeline_names, OPENING_PIPELINE, REGISTERS_CLAIM_BUILD_PIPELINE,
        REGISTERS_CLAIM_DOT_PIPELINE, REGISTERS_CLAIM_REDUCE_PIPELINE,
    },
};

pub(super) struct Pipelines {
    pub(super) materialize: ComputePipelineState,
    pub(super) stream_bind: ComputePipelineState,
    pub(super) transition: ComputePipelineState,
    pub(super) opening: ComputePipelineState,
    pub(super) reduction: ComputePipelineState,
    pub(super) registers_claim_build: Option<ComputePipelineState>,
    pub(super) registers_claim_reduce: Option<ComputePipelineState>,
    pub(super) registers_claim_dot: Option<ComputePipelineState>,
}

pub(super) struct PipelineSetLimits {
    pub(super) materialize: PipelineLimits,
    pub(super) stream_bind: PipelineLimits,
    pub(super) transition: PipelineLimits,
    pub(super) opening: PipelineLimits,
    pub(super) reduction: PipelineLimits,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(feature = "test-utils")]
pub(crate) struct OuterRemainderStorageEvalStats {
    pub(crate) borrowed_state_b: bool,
    pub(crate) initialized_bytes: u64,
    pub(crate) initialization_gpu_active: Duration,
    pub(crate) initialization_device_buffers: usize,
    pub(crate) materialize_limits: PipelineLimits,
    pub(crate) stream_bind_limits: PipelineLimits,
    pub(crate) transition_limits: PipelineLimits,
    pub(crate) opening_limits: PipelineLimits,
    pub(crate) reduction_limits: PipelineLimits,
    pub(crate) registers_claim_build_limits: Option<PipelineLimits>,
    pub(crate) registers_claim_reduce_limits: Option<PipelineLimits>,
    pub(crate) registers_claim_dot_limits: Option<PipelineLimits>,
    pub(crate) materialize_threads: usize,
    pub(crate) stream_bind_threads: usize,
    pub(crate) transition_threads: usize,
    pub(crate) opening_threads: usize,
    pub(crate) reduction_threads: usize,
    pub(crate) registers_claim_build_threads: usize,
    pub(crate) registers_claim_reduce_threads: usize,
    pub(crate) registers_claim_dot_threads: usize,
    pub(crate) opening_dynamic_threadgroup_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct StorageInitializationStats {
    bytes: u64,
    gpu_active: Duration,
    device_buffers: usize,
}

pub(super) struct Threads {
    pub(super) materialize: usize,
    pub(super) stream_bind: usize,
    pub(super) transition: usize,
    pub(super) opening: usize,
    pub(super) reduction: usize,
    pub(super) registers_claim_build: usize,
    pub(super) registers_claim_reduce: usize,
    pub(super) registers_claim_dot: usize,
}

pub(super) struct DenseBuffers {
    pub(super) state_a: Buffer,
    pub(super) state_b: Option<Buffer>,
}

pub(super) struct Buffers {
    pub(super) dense: Option<DenseBuffers>,
    pub(super) e_in: Buffer,
    pub(super) e_out: Buffer,
    pub(super) a_lookup: Buffer,
    pub(super) message_partials: Buffer,
    pub(super) message_output: Buffer,
    pub(super) opening_partials: Buffer,
    pub(super) opening_output: Buffer,
    pub(super) registers_claim: Option<RegistersClaimBuffers>,
}

pub(super) struct RegistersClaimBuffers {
    pub(super) partials: Buffer,
    pub(super) components: Buffer,
    pub(super) rd_write_value: Buffer,
    pub(super) geometry: RegistersClaimCarrierGeometry,
}

impl Buffers {
    pub(super) fn identities(&self) -> [usize; DEVICE_BUFFERS] {
        let dense = self.dense.as_ref().map_or([0, 0], |dense| {
            [
                dense.state_a.as_ptr() as usize,
                dense
                    .state_b
                    .as_ref()
                    .map_or(0, |state_b| state_b.as_ptr() as usize),
            ]
        });
        [
            dense[0],
            dense[1],
            self.e_in.as_ptr() as usize,
            self.e_out.as_ptr() as usize,
            self.a_lookup.as_ptr() as usize,
            self.message_partials.as_ptr() as usize,
            self.message_output.as_ptr() as usize,
            self.opening_partials.as_ptr() as usize,
            self.opening_output.as_ptr() as usize,
        ]
    }
}

enum StateBSource {
    Owned,
    #[cfg_attr(
        not(feature = "test-utils"),
        expect(
            dead_code,
            reason = "borrowed storage is exposed by the evaluation API"
        )
    )]
    Borrowed(Buffer),
    Deferred,
}

impl StateBSource {
    const fn is_borrowed(&self) -> bool {
        !matches!(self, Self::Owned)
    }
}

pub(super) struct Storage {
    pub(super) context: SolinasMetal,
    pub(super) pipelines: Pipelines,
    pub(super) threads: Threads,
    pub(super) buffers: Buffers,
    pub(super) weight_capacity: usize,
    pub(super) max_threadgroups: usize,
    pub(super) dense_bytes: u64,
    pub(super) owned_bytes: u64,
    borrowed_state_b: bool,
    #[cfg_attr(
        not(feature = "test-utils"),
        expect(
            dead_code,
            reason = "initialization telemetry is exposed by the evaluation API"
        )
    )]
    initialization: StorageInitializationStats,
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
        self.prepare_outer_remainder_sequence_storage_inner(cycles, config, StateBSource::Owned)
    }

    pub(crate) fn prepare_outer_remainder_sequence_storage_deferring_state_b(
        &self,
        cycles: usize,
        config: OuterRemainderSequenceConfig,
    ) -> Result<OuterRemainderSequenceStorage, MetalError> {
        self.prepare_outer_remainder_sequence_storage_inner(cycles, config, StateBSource::Deferred)
    }

    #[cfg(feature = "test-utils")]
    pub(crate) fn prepare_outer_remainder_sequence_storage_borrowing_state_b(
        &self,
        cycles: usize,
        config: OuterRemainderSequenceConfig,
        state_b: Buffer,
    ) -> Result<OuterRemainderSequenceStorage, MetalError> {
        self.prepare_outer_remainder_sequence_storage_inner(
            cycles,
            config,
            StateBSource::Borrowed(state_b),
        )
    }

    fn prepare_outer_remainder_sequence_storage_inner(
        &self,
        cycles: usize,
        config: OuterRemainderSequenceConfig,
        state_b_source: StateBSource,
    ) -> Result<OuterRemainderSequenceStorage, MetalError> {
        if cycles < 4 || !cycles.is_power_of_two() {
            return Err(MetalError::InvalidOuterRemainderRows(cycles));
        }
        let geometry = storage_geometry(cycles, config)?;
        let current_elements = geometry.current_elements;
        let weight_capacity = geometry.weight_capacity;
        let max_threadgroups = geometry.max_threadgroups;

        let names = pipeline_names();
        let opening_name = OPENING_PIPELINE;
        let pipelines = Pipelines {
            materialize: self.compile_named_pipeline(names.materialize)?,
            stream_bind: self.compile_named_pipeline(names.stream_bind)?,
            transition: self.compile_named_pipeline(names.transition)?,
            opening: self.compile_named_pipeline(opening_name)?,
            reduction: self.compile_named_pipeline(names.reduction)?,
            registers_claim_build: config
                .registers_claim_carrier
                .then(|| self.compile_named_pipeline(REGISTERS_CLAIM_BUILD_PIPELINE))
                .transpose()?,
            registers_claim_reduce: config
                .registers_claim_carrier
                .then(|| self.compile_named_pipeline(REGISTERS_CLAIM_REDUCE_PIPELINE))
                .transpose()?,
            registers_claim_dot: config
                .registers_claim_carrier
                .then(|| self.compile_named_pipeline(REGISTERS_CLAIM_DOT_PIPELINE))
                .transpose()?,
        };
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
            (opening_name, limits.opening),
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
        for (pipeline, state) in [
            (
                REGISTERS_CLAIM_BUILD_PIPELINE,
                &pipelines.registers_claim_build,
            ),
            (
                REGISTERS_CLAIM_REDUCE_PIPELINE,
                &pipelines.registers_claim_reduce,
            ),
            (REGISTERS_CLAIM_DOT_PIPELINE, &pipelines.registers_claim_dot),
        ] {
            if let Some(state) = state {
                let limits = Self::limits(state);
                if limits.thread_execution_width != SIMD_WIDTH {
                    return Err(MetalError::UnsupportedOuterRemainderExecutionWidth {
                        pipeline,
                        expected: SIMD_WIDTH,
                        got: limits.thread_execution_width,
                    });
                }
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
            registers_claim_build: if let Some(pipeline) = &pipelines.registers_claim_build {
                Self::resolve_threadgroup_width(Some(256), Self::limits(pipeline))?
            } else {
                0
            },
            registers_claim_reduce: if let Some(pipeline) = &pipelines.registers_claim_reduce {
                Self::resolve_threadgroup_width(Some(128), Self::limits(pipeline))?
            } else {
                0
            },
            registers_claim_dot: if let Some(pipeline) = &pipelines.registers_claim_dot {
                Self::resolve_threadgroup_width(Some(256), Self::limits(pipeline))?
            } else {
                0
            },
        };
        if threads.opening < 128 {
            return Err(MetalError::InvalidOuterRemainderConfig(
                "opening threadgroup needs at least 128 threads",
            ));
        }
        validate_opening_threadgroup_memory(
            self,
            limits.opening,
            threads.opening,
            config.product_uniskip_carrier,
        )?;

        for elements in geometry.element_counts {
            let bytes = field_bytes(elements)?;
            self.validate_buffer_length(bytes)?;
        }
        let state_b_bytes = field_bytes(current_elements / 2)?;
        if matches!(
            &state_b_source,
            StateBSource::Borrowed(state_b) if state_b.length() != state_b_bytes
        ) {
            return Err(MetalError::InvalidOuterRemainderConfig(
                "borrowed state B has the wrong length",
            ));
        }
        let borrowed_state_b = state_b_source.is_borrowed();
        let owned_bytes = geometry
            .owned_bytes
            .checked_sub(u64::from(borrowed_state_b) * state_b_bytes)
            .ok_or(MetalError::InputTooLong(current_elements))?;
        self.validate_additional_working_set(owned_bytes)?;
        let dense_bytes = field_bytes(current_elements)?
            .checked_add(u64::from(!borrowed_state_b) * state_b_bytes)
            .ok_or(MetalError::InputTooLong(current_elements))?;
        let opening_outputs = opening_output_count(config.product_uniskip_carrier);
        let registers_claim = if config.registers_claim_carrier {
            let geometry = carrier_geometry(cycles)?;
            for bytes in [
                geometry.partial_bytes,
                geometry.component_bytes,
                geometry.rd_bytes,
            ] {
                self.validate_buffer_length(bytes)?;
            }
            Some(RegistersClaimBuffers {
                partials: new_private_buffer(self, geometry.partial_bytes)?,
                components: new_buffer(self, geometry.component_bytes)?,
                rd_write_value: new_private_buffer(self, geometry.rd_bytes)?,
                geometry,
            })
        } else {
            None
        };
        let state_b = match state_b_source {
            StateBSource::Owned => Some(new_field_buffer(self, current_elements / 2)?),
            StateBSource::Borrowed(state_b) => Some(state_b),
            StateBSource::Deferred => None,
        };
        let buffers = Buffers {
            dense: Some(DenseBuffers {
                state_a: new_field_buffer(self, current_elements)?,
                state_b,
            }),
            e_in: new_field_buffer(self, weight_capacity)?,
            e_out: new_field_buffer(self, weight_capacity)?,
            a_lookup: new_field_buffer(self, OUTER_REMAINDER_A_LOOKUP_FIELDS)?,
            message_partials: new_field_buffer(self, 2 * max_threadgroups)?,
            message_output: new_field_buffer(self, 2)?,
            opening_partials: new_field_buffer(self, opening_outputs * max_threadgroups)?,
            opening_output: new_field_buffer(self, opening_outputs)?,
            registers_claim,
        };
        let initialization = initialize_storage(
            self,
            &buffers,
            config.storage_initialization,
            borrowed_state_b,
        )?;

        Ok(OuterRemainderSequenceStorage {
            storage: Storage {
                context: self.clone(),
                pipelines,
                threads,
                buffers,
                weight_capacity,
                max_threadgroups,
                dense_bytes,
                owned_bytes,
                borrowed_state_b,
                initialization,
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
        let Some((expected_owned_bytes, expected_state_b_bytes)) =
            storage_geometry(cycles, config).ok().and_then(|geometry| {
                field_bytes(geometry.current_elements / 2)
                    .ok()
                    .and_then(|state_b_bytes| {
                        geometry
                            .owned_bytes
                            .checked_sub(u64::from(self.storage.borrowed_state_b) * state_b_bytes)
                            .map(|owned_bytes| (owned_bytes, state_b_bytes))
                    })
            })
        else {
            return false;
        };
        self.storage.context.device_registry_id() == context.device_registry_id()
            && self.cycles == cycles
            && self.config == config
            && self.storage.owned_bytes == expected_owned_bytes
            && self
                .storage
                .buffers
                .dense
                .as_ref()
                .and_then(|dense| dense.state_b.as_ref())
                .is_some_and(|state_b| state_b.length() == expected_state_b_bytes)
    }

    pub(crate) const fn owned_bytes(&self) -> u64 {
        self.storage.owned_bytes
    }

    #[cfg(feature = "test-utils")]
    pub(crate) fn eval_stats(&self) -> Result<OuterRemainderStorageEvalStats, MetalError> {
        let pipelines = &self.storage.pipelines;
        let threads = &self.storage.threads;
        let pipeline_limits = |pipeline: &ComputePipelineState| SolinasMetal::limits(pipeline);
        let opening_dynamic_threadgroup_bytes = opening_threadgroup_memory_lengths(
            threads.opening,
            self.config.product_uniskip_carrier,
        )?
        .into_iter()
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .ok_or(MetalError::InvalidOuterRemainderConfig(
            "opening threadgroup byte count overflowed",
        ))?;
        Ok(OuterRemainderStorageEvalStats {
            borrowed_state_b: self.storage.borrowed_state_b,
            initialized_bytes: self.storage.initialization.bytes,
            initialization_gpu_active: self.storage.initialization.gpu_active,
            initialization_device_buffers: self.storage.initialization.device_buffers,
            materialize_limits: pipeline_limits(&pipelines.materialize),
            stream_bind_limits: pipeline_limits(&pipelines.stream_bind),
            transition_limits: pipeline_limits(&pipelines.transition),
            opening_limits: pipeline_limits(&pipelines.opening),
            reduction_limits: pipeline_limits(&pipelines.reduction),
            registers_claim_build_limits: pipelines
                .registers_claim_build
                .as_ref()
                .map(pipeline_limits),
            registers_claim_reduce_limits: pipelines
                .registers_claim_reduce
                .as_ref()
                .map(pipeline_limits),
            registers_claim_dot_limits: pipelines.registers_claim_dot.as_ref().map(pipeline_limits),
            materialize_threads: threads.materialize,
            stream_bind_threads: threads.stream_bind,
            transition_threads: threads.transition,
            opening_threads: threads.opening,
            reduction_threads: threads.reduction,
            registers_claim_build_threads: threads.registers_claim_build,
            registers_claim_reduce_threads: threads.registers_claim_reduce,
            registers_claim_dot_threads: threads.registers_claim_dot,
            opening_dynamic_threadgroup_bytes,
        })
    }

    pub(crate) fn buffer_identities(&self) -> [usize; DEVICE_BUFFERS] {
        self.storage.buffers.identities()
    }

    pub(crate) fn awaits_product_state_b(&self) -> bool {
        self.storage.borrowed_state_b
            && self
                .storage
                .buffers
                .dense
                .as_ref()
                .is_some_and(|dense| dense.state_b.is_none())
    }

    pub(crate) fn attach_product_state_b(&mut self, state_b: Buffer) -> Result<(), MetalError> {
        if !self.awaits_product_state_b() {
            return Err(MetalError::InvalidOuterRemainderState {
                expected: "deferred Product-owned state B",
                got: "owned or already attached state B",
            });
        }
        let expected_bytes = field_bytes(self.current_elements / 2)?;
        if state_b.length() != expected_bytes
            || state_b.device().registry_id() != self.storage.context.device_registry_id()
        {
            return Err(MetalError::InvalidOuterRemainderConfig(
                "Product-owned state B has the wrong shape or device",
            ));
        }
        let dense =
            self.storage
                .buffers
                .dense
                .as_mut()
                .ok_or(MetalError::InvalidOuterRemainderState {
                    expected: "allocated dense storage for Product state-B attachment",
                    got: "released dense storage",
                })?;
        if dense.state_a.as_ptr() == state_b.as_ptr() {
            return Err(MetalError::InvalidOuterRemainderConfig(
                "Outer dense state buffers alias",
            ));
        }
        dense.state_b = Some(state_b);
        Ok(())
    }

    pub(crate) fn share_product_state_a(&self) -> Result<Buffer, MetalError> {
        self.storage
            .buffers
            .dense
            .as_ref()
            .map(|dense| dense.state_a.clone())
            .ok_or(MetalError::InvalidOuterRemainderState {
                expected: "allocated dense storage for sequential Product reuse",
                got: "released dense storage",
            })
    }
}

fn initialize_storage(
    context: &SolinasMetal,
    buffers: &Buffers,
    mode: OuterRemainderStorageInitialization,
    borrowed_state_b: bool,
) -> Result<StorageInitializationStats, MetalError> {
    let dense = buffers
        .dense
        .as_ref()
        .ok_or(MetalError::InvalidOuterRemainderState {
            expected: "allocated dense storage",
            got: "released dense storage",
        })?;
    let mut initialized = Vec::with_capacity(DEVICE_BUFFERS);
    initialized.push(&dense.state_a);
    if !borrowed_state_b {
        initialized.push(
            dense
                .state_b
                .as_ref()
                .ok_or(MetalError::InvalidOuterRemainderState {
                    expected: "owned dense state B",
                    got: "missing dense state B",
                })?,
        );
    }
    initialized.extend([
        &buffers.e_in,
        &buffers.e_out,
        &buffers.a_lookup,
        &buffers.message_partials,
        &buffers.message_output,
        &buffers.opening_partials,
        &buffers.opening_output,
    ]);
    let bytes = match mode {
        OuterRemainderStorageInitialization::Lazy => 0,
        OuterRemainderStorageInitialization::Full => {
            initialized.iter().try_fold(0u64, |total, buffer| {
                total
                    .checked_add(buffer.length())
                    .ok_or(MetalError::InvalidOuterRemainderConfig(
                        "storage initialization byte count overflowed",
                    ))
            })?
        }
    };
    let device_buffers = usize::from(mode == OuterRemainderStorageInitialization::Full)
        * (DEVICE_BUFFERS - usize::from(borrowed_state_b));
    let span = tracing::info_span!(
        "MetalOuterRemainder::storage_initialize",
        mode = mode.as_str(),
        device_buffers,
        bytes,
        gpu_active_ns = tracing::field::Empty,
    );
    let _entered = span.enter();
    let gpu_active = if mode == OuterRemainderStorageInitialization::Full {
        let command_buffer = context.queue.new_command_buffer();
        autoreleasepool(|| {
            let encoder = command_buffer.new_blit_command_encoder();
            for buffer in initialized {
                encoder.fill_buffer(buffer, NSRange::new(0, buffer.length()), 0);
            }
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        completed_command_gpu_time(command_buffer)?
    } else {
        Duration::ZERO
    };
    let gpu_active_ns = u64::try_from(gpu_active.as_nanos()).unwrap_or(u64::MAX);
    let _ = span.record("gpu_active_ns", gpu_active_ns);
    Ok(StorageInitializationStats {
        bytes,
        gpu_active,
        device_buffers,
    })
}

#[cfg(feature = "test-utils")]
impl SolinasMetal {
    pub(crate) fn prepare_eval_outer_state_b(&self, cycles: usize) -> Result<Buffer, MetalError> {
        let state_b = new_field_buffer(self, cycles)?;
        let command_buffer = self.queue.new_command_buffer();
        autoreleasepool(|| {
            let encoder = command_buffer.new_blit_command_encoder();
            encoder.fill_buffer(&state_b, NSRange::new(0, state_b.length()), 0);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        let _ = completed_command_gpu_time(command_buffer)?;
        Ok(state_b)
    }
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
    new_buffer(context, bytes)
}

fn new_buffer(context: &SolinasMetal, bytes: u64) -> Result<Buffer, MetalError> {
    context.validate_buffer_length(bytes)?;
    Ok(context
        .device
        .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
}

fn new_private_buffer(context: &SolinasMetal, bytes: u64) -> Result<Buffer, MetalError> {
    context.validate_buffer_length(bytes)?;
    Ok(context
        .device
        .new_buffer(bytes, MTLResourceOptions::StorageModePrivate))
}
