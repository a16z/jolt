use core::mem::size_of;
use std::{slice, time::Duration};

use jolt_field::AkitaField;
#[cfg(feature = "test-utils")]
use jolt_field::FromPrimitiveInt;
use jolt_poly::EqPolynomial;
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::super::{
    command_buffer_timestamp, AddressPhaseSums, AddressRafSums, AddressSuffixFullSums, Fp128,
    MetalError, PipelineLimits, ResidentLookupIndexPlane, SolinasMetal, AKITA_OFFSET_FFFFA7F7,
};
use super::shader_abi::{
    AddressLookup, AtomMassFinalizeParams, AtomMassPhaseParams, AtomPhaseParams, SuffixPlan,
    JOB_FIELDS, PHASE_THREADGROUP_BYTES, SIMD_WIDTH, TABLES, TOTAL_SUFFIXES,
};
use super::topology::{AddressAtomTopology, AddressAtomTopologyCensus};
use super::{
    InstructionReadRafV3Error, ADDRESS_BINS, ADDRESS_PHASES, ATOM_MASS_FINALIZE_PIPELINE,
    ATOM_MASS_PHASE_PIPELINE, ATOM_PHASE_PIPELINE, FINALIZE_RAF_PIPELINE, FINALIZE_SUFFIX_PIPELINE,
};

const PHASE_CHALLENGES: usize = 8;
const RAF_OUTPUT_LANES: usize = 6;

#[cfg(feature = "test-utils")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddressAtomProbeResult {
    pub rows: usize,
    pub atoms: usize,
    pub phases: usize,
    pub all_exact: bool,
    pub finished: bool,
    pub gpu_active_ns: u128,
}

#[cfg(feature = "test-utils")]
pub fn run_address_atom_probe(
    context: &SolinasMetal,
    log_rows: usize,
) -> Result<AddressAtomProbeResult, String> {
    use super::oracle::{aggregate_address_atoms, atom_address_phase, InstructionReadRafRow};
    use super::topology::AddressAtomTopologyConfig;

    if !(3..=16).contains(&log_rows) {
        return Err("probe log_rows must be in 3..=16".to_owned());
    }
    let row_count = 1usize << log_rows;
    let rows = (0..row_count)
        .map(|index| {
            if index < row_count / 2 {
                return InstructionReadRafRow::new(17, Some(0), true);
            }
            let pattern = index % 32;
            let lookup = match pattern % 8 {
                0 => 0,
                1 => 5,
                2 => u64::MAX as u128,
                3 => u128::MAX,
                4 => 1u128 << 127,
                5 => (1u128 << 96) | 0x2a,
                6 => (pattern as u128) << 56,
                _ => 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
            };
            let table = (pattern % 5 != 0).then_some(pattern % TABLES);
            InstructionReadRafRow::new(lookup, table, pattern % 3 == 0)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    let reduction_point = (0..log_rows)
        .map(|coordinate| AkitaField::from_u64((coordinate + 2) as u64))
        .collect::<Vec<_>>();
    let challenge_batches: [[AkitaField; PHASE_CHALLENGES]; ADDRESS_PHASES] =
        std::array::from_fn(|phase| {
            std::array::from_fn(|coordinate| {
                AkitaField::from_u64((phase * PHASE_CHALLENGES + coordinate + 3) as u64)
            })
        });

    let topology =
        AddressAtomTopology::from_rows_reference(&rows, AddressAtomTopologyConfig::default())
            .map_err(|error| error.to_string())?;
    let mut atoms =
        aggregate_address_atoms(&rows, &reduction_point).map_err(|error| error.to_string())?;
    let mut sequence = context
        .prepare_instruction_read_raf_v3_address(
            topology,
            &reduction_point,
            AddressAtomRuntimeConfig::default(),
        )
        .map_err(|error| error.to_string())?;
    let mut phase_tables = Vec::<Vec<AkitaField>>::with_capacity(ADDRESS_PHASES);
    let mut all_exact = true;
    for phase in 0..ADDRESS_PHASES {
        let output = if phase == 0 {
            sequence.first_phase()
        } else {
            sequence.next_phase(challenge_batches[phase - 1])
        }
        .map_err(|error| error.to_string())?;
        let previous = phase
            .checked_sub(1)
            .and_then(|index| phase_tables.get(index))
            .map(Vec::as_slice);
        let expected = atom_address_phase(&mut atoms, 120 - phase * PHASE_CHALLENGES, previous)
            .map_err(|error| error.to_string())?;
        let actual_raf = output
            .raf()
            .as_flat_slice()
            .iter()
            .map(|value| value.into_jolt_field::<AkitaField>())
            .collect::<Vec<_>>();
        let actual_suffix = output
            .suffix()
            .as_flat_slice()
            .iter()
            .map(|value| value.into_jolt_field::<AkitaField>())
            .collect::<Vec<_>>();
        all_exact &= output.phase() == phase
            && output.suffix_len() == 120 - phase * PHASE_CHALLENGES
            && output.gpu_active() > Duration::ZERO
            && actual_raf == expected.raf
            && actual_suffix == expected.suffixes;
        let table: Vec<AkitaField> = EqPolynomial::evals(&challenge_batches[phase], None);
        phase_tables.push(table);
    }
    sequence
        .finish_address(challenge_batches[ADDRESS_PHASES - 1])
        .map_err(|error| error.to_string())?;
    let census = sequence.census().map_err(|error| error.to_string())?;
    all_exact &= sequence.phases_executed() == ADDRESS_PHASES
        && sequence.phase_challenges() == challenge_batches
        && sequence.is_finished();
    Ok(AddressAtomProbeResult {
        rows: row_count,
        atoms: census.atoms,
        phases: sequence.phases_executed(),
        all_exact,
        finished: sequence.is_finished(),
        gpu_active_ns: sequence.gpu_active().as_nanos(),
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AddressAtomRuntimeConfig {
    pub(crate) phase: usize,
    pub(crate) mass_finalize: usize,
    pub(crate) raf_finalize: usize,
    pub(crate) suffix_finalize: usize,
}

impl Default for AddressAtomRuntimeConfig {
    fn default() -> Self {
        Self {
            phase: 1024,
            mass_finalize: 256,
            raf_finalize: ADDRESS_BINS,
            suffix_finalize: 4 * ADDRESS_BINS,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum AddressAtomRuntimeError {
    #[error(transparent)]
    Plan(#[from] InstructionReadRafV3Error),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("InstructionReadRaf reduction point has {got} coordinates, expected {expected}")]
    ReductionPointLength { expected: usize, got: usize },
    #[error("InstructionReadRaf pipeline `{pipeline}` requires SIMD width {expected}, got {got}")]
    ExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "InstructionReadRaf pipeline `{pipeline}` needs {requested} threads, maximum is {maximum}"
    )]
    ThreadLimit {
        pipeline: &'static str,
        requested: usize,
        maximum: usize,
    },
    #[error(
        "InstructionReadRaf pipeline `{pipeline}` needs {requested} threadgroup bytes, maximum is {maximum}"
    )]
    ThreadgroupMemory {
        pipeline: &'static str,
        requested: u64,
        maximum: u64,
    },
    #[error("InstructionReadRaf address runtime state is invalid: {0}")]
    InvalidState(&'static str),
}

struct Pipelines {
    mass_phase: ComputePipelineState,
    mass_finalize: ComputePipelineState,
    atom_phase: ComputePipelineState,
    raf_finalize: ComputePipelineState,
    suffix_finalize: ComputePipelineState,
}

struct Buffers {
    atom_lookups: Buffer,
    cycle_indices: Buffer,
    mass_jobs: Buffer,
    mass_groups: Buffer,
    phase_zero_group_offsets: Buffer,
    split_atoms: Buffer,
    phase_jobs: Buffer,
    phase_job_offsets: Buffer,
    suffix_kinds: Buffer,
    suffix_counts: Buffer,
    output_lanes: Buffer,
    table_descriptors: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    atom_masses: Buffer,
    mass_partials: Buffer,
    phase_tables: Buffer,
    address_partials: Buffer,
    raf_output: Buffer,
    suffix_output: Buffer,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AddressAtomPhaseOutput {
    phase: usize,
    suffix_len: usize,
    raf: AddressRafSums,
    suffix: AddressSuffixFullSums,
    gpu_active: Duration,
}

impl AddressAtomPhaseOutput {
    pub(crate) const fn phase(&self) -> usize {
        self.phase
    }

    pub(crate) const fn suffix_len(&self) -> usize {
        self.suffix_len
    }

    pub(crate) const fn raf(&self) -> &AddressRafSums {
        &self.raf
    }

    pub(crate) const fn suffix(&self) -> &AddressSuffixFullSums {
        &self.suffix
    }

    pub(crate) const fn gpu_active(&self) -> Duration {
        self.gpu_active
    }

    pub(crate) fn into_phase_sums(self) -> AddressPhaseSums {
        AddressPhaseSums::from_parts(self.raf, self.suffix, self.gpu_active)
    }
}

pub(crate) struct AddressAtomSequence {
    context: SolinasMetal,
    pipelines: Pipelines,
    buffers: Buffers,
    topology: AddressAtomTopology,
    config: AddressAtomRuntimeConfig,
    e_in_length: usize,
    e_out_length: usize,
    table_offsets: Vec<usize>,
    phases_executed: usize,
    challenge_batches: Vec<[AkitaField; PHASE_CHALLENGES]>,
    finished: bool,
    gpu_active: Duration,
}

impl SolinasMetal {
    pub(crate) fn prepare_instruction_read_raf_v3_lookup_plane(
        &self,
        topology: &AddressAtomTopology,
    ) -> Result<ResidentLookupIndexPlane, AddressAtomRuntimeError> {
        let rows = topology.rows();
        let lookup_bytes = checked_bytes::<AddressLookup>(rows, "resident lookup plane")?;
        let inverse_bytes = checked_bytes::<u32>(rows, "resident lookup inverse")?;
        let maximum = self.device.max_buffer_length();
        for requested in [lookup_bytes, inverse_bytes] {
            if requested > maximum {
                return Err(MetalError::BufferTooLong { requested, maximum }.into());
            }
        }
        let lookups = self
            .device
            .new_buffer(lookup_bytes, MTLResourceOptions::StorageModeShared);
        let inverse = self
            .device
            .new_buffer(inverse_bytes, MTLResourceOptions::StorageModeShared);
        // SAFETY: both buffers are fresh, shared allocations with exactly
        // `rows` elements and remain CPU-exclusive until this function returns.
        let lookup_values =
            unsafe { slice::from_raw_parts_mut(lookups.contents().cast::<AddressLookup>(), rows) };
        // SAFETY: the same allocation and exclusivity argument applies.
        let inverse_values =
            unsafe { slice::from_raw_parts_mut(inverse.contents().cast::<u32>(), rows) };
        for (atom, lookup) in topology.atom_lookups().iter().copied().enumerate() {
            let start = topology.atom_cycle_offsets()[atom] as usize;
            let end = topology.atom_cycle_offsets()[atom + 1] as usize;
            for (local_position, lookup_value) in lookup_values[start..end].iter_mut().enumerate() {
                let position = start + local_position;
                let cycle = topology.cycle_indices()[position] as usize;
                *lookup_value = lookup;
                inverse_values[cycle] = u32::try_from(position).map_err(|_| {
                    InstructionReadRafV3Error::SizeOverflow("resident lookup position")
                })?;
            }
        }
        Ok(ResidentLookupIndexPlane::from_buffers(
            lookups,
            inverse,
            rows,
            self.device.registry_id(),
        ))
    }

    pub(crate) fn prepare_instruction_read_raf_v3_address(
        &self,
        topology: AddressAtomTopology,
        reduction_point: &[AkitaField],
        config: AddressAtomRuntimeConfig,
    ) -> Result<AddressAtomSequence, AddressAtomRuntimeError> {
        if self.offset != AKITA_OFFSET_FFFFA7F7 {
            return Err(MetalError::UnexpectedSolinasOffset {
                expected: AKITA_OFFSET_FFFFA7F7,
                got: self.offset,
            }
            .into());
        }
        let expected_point = topology.rows().ilog2() as usize;
        if reduction_point.len() != expected_point {
            return Err(AddressAtomRuntimeError::ReductionPointLength {
                expected: expected_point,
                got: reduction_point.len(),
            });
        }
        let out_log = expected_point / 2;
        let (out_point, in_point) = reduction_point.split_at(out_log);
        let e_in: Vec<AkitaField> = EqPolynomial::evals(in_point, None);
        let e_out: Vec<AkitaField> = EqPolynomial::evals(out_point, None);
        let e_in_values = e_in.iter().map(Fp128::from_jolt_field).collect::<Vec<_>>();
        let e_out_values = e_out.iter().map(Fp128::from_jolt_field).collect::<Vec<_>>();

        let pipelines = Pipelines {
            mass_phase: self.compile_named_pipeline(ATOM_MASS_PHASE_PIPELINE)?,
            mass_finalize: self.compile_named_pipeline(ATOM_MASS_FINALIZE_PIPELINE)?,
            atom_phase: self.compile_named_pipeline(ATOM_PHASE_PIPELINE)?,
            raf_finalize: self.compile_named_pipeline(FINALIZE_RAF_PIPELINE)?,
            suffix_finalize: self.compile_named_pipeline(FINALIZE_SUFFIX_PIPELINE)?,
        };
        validate_pipeline(
            ATOM_MASS_PHASE_PIPELINE,
            Self::limits(&pipelines.mass_phase),
            config.phase,
            PHASE_THREADGROUP_BYTES as u64,
            self.device.max_threadgroup_memory_length(),
        )?;
        validate_pipeline(
            ATOM_PHASE_PIPELINE,
            Self::limits(&pipelines.atom_phase),
            config.phase,
            PHASE_THREADGROUP_BYTES as u64,
            self.device.max_threadgroup_memory_length(),
        )?;
        for (pipeline, state, threads) in [
            (
                ATOM_MASS_FINALIZE_PIPELINE,
                &pipelines.mass_finalize,
                config.mass_finalize,
            ),
            (
                FINALIZE_RAF_PIPELINE,
                &pipelines.raf_finalize,
                config.raf_finalize,
            ),
            (
                FINALIZE_SUFFIX_PIPELINE,
                &pipelines.suffix_finalize,
                config.suffix_finalize,
            ),
        ] {
            validate_pipeline(
                pipeline,
                Self::limits(state),
                threads,
                0,
                self.device.max_threadgroup_memory_length(),
            )?;
        }
        if config.raf_finalize != ADDRESS_BINS || config.suffix_finalize != 4 * ADDRESS_BINS {
            return Err(AddressAtomRuntimeError::InvalidState(
                "finalize threadgroup widths differ from the fixed shader layout",
            ));
        }

        let suffix_plan = SuffixPlan::production()?;
        let table_offsets = suffix_plan
            .descriptors()
            .iter()
            .map(|descriptor| descriptor.output_start as usize)
            .chain(std::iter::once(TOTAL_SUFFIXES))
            .collect::<Vec<_>>();
        let census = topology.census()?;
        let partial_jobs = census.mass_groups.max(census.later_phase_jobs);
        let partial_fields = checked_product(partial_jobs, JOB_FIELDS, "address partial fields")?;
        let field_bytes = |fields, name| checked_bytes::<Fp128>(fields, name);
        let allocation_lengths = [
            checked_bytes::<super::shader_abi::AddressLookup>(census.atoms, "atom lookups")?,
            checked_bytes::<u32>(census.rows, "cycle permutation")?,
            checked_bytes::<super::shader_abi::AtomMassJob>(census.mass_jobs, "mass jobs")?,
            checked_bytes::<super::shader_abi::AtomMassGroup>(census.mass_groups, "mass groups")?,
            checked_bytes::<u32>(super::shader_abi::SEGMENT_OFFSETS, "mass group offsets")?,
            checked_bytes::<super::shader_abi::SplitAtom>(
                census.split_atoms.max(1),
                "split atoms",
            )?,
            checked_bytes::<super::shader_abi::AddressJob>(census.later_phase_jobs, "phase jobs")?,
            checked_bytes::<u32>(super::shader_abi::SEGMENT_OFFSETS, "phase job offsets")?,
            checked_bytes::<u8>(suffix_plan.explicit_kinds().len(), "suffix kinds")?,
            checked_bytes::<u8>(suffix_plan.explicit_counts().len(), "suffix counts")?,
            checked_bytes::<u8>(suffix_plan.output_lanes().len(), "suffix output lanes")?,
            checked_bytes::<super::shader_abi::TableDescriptor>(TABLES, "table descriptors")?,
            field_bytes(e_in_values.len(), "E_in")?,
            field_bytes(e_out_values.len(), "E_out")?,
            field_bytes(census.atoms, "atom masses")?,
            field_bytes(census.mass_partials.max(1), "mass partials")?,
            field_bytes(ADDRESS_PHASES * ADDRESS_BINS, "address phase tables")?,
            field_bytes(partial_fields, "address partials")?,
            field_bytes(RAF_OUTPUT_LANES * ADDRESS_BINS, "RAF output")?,
            field_bytes(TOTAL_SUFFIXES * ADDRESS_BINS, "suffix output")?,
        ];
        for &bytes in &allocation_lengths {
            self.validate_buffer_length(bytes)?;
        }
        let additional = allocation_lengths.iter().try_fold(0u64, |sum, &bytes| {
            sum.checked_add(bytes)
                .ok_or(InstructionReadRafV3Error::SizeOverflow(
                    "address runtime working set",
                ))
        })?;
        self.validate_additional_working_set(additional)?;

        let buffers = Buffers {
            atom_lookups: super::super::buffer_from_slice(&self.device, topology.atom_lookups()),
            cycle_indices: super::super::buffer_from_slice(&self.device, topology.cycle_indices()),
            mass_jobs: super::super::buffer_from_slice(&self.device, topology.mass_jobs()),
            mass_groups: super::super::buffer_from_slice(&self.device, topology.mass_groups()),
            phase_zero_group_offsets: super::super::buffer_from_slice(
                &self.device,
                topology.phase_zero_group_offsets(),
            ),
            split_atoms: buffer_from_slice_or_placeholder(&self.device, topology.split_atoms()),
            phase_jobs: super::super::buffer_from_slice(&self.device, topology.phase_jobs()),
            phase_job_offsets: super::super::buffer_from_slice(
                &self.device,
                topology.phase_job_offsets(),
            ),
            suffix_kinds: super::super::buffer_from_slice(
                &self.device,
                suffix_plan.explicit_kinds(),
            ),
            suffix_counts: super::super::buffer_from_slice(
                &self.device,
                suffix_plan.explicit_counts(),
            ),
            output_lanes: super::super::buffer_from_slice(&self.device, suffix_plan.output_lanes()),
            table_descriptors: super::super::buffer_from_slice(
                &self.device,
                suffix_plan.descriptors(),
            ),
            e_in: super::super::buffer_from_slice(&self.device, &e_in_values),
            e_out: super::super::buffer_from_slice(&self.device, &e_out_values),
            atom_masses: self.device.new_buffer(
                field_bytes(census.atoms, "atom masses")?,
                MTLResourceOptions::StorageModePrivate,
            ),
            mass_partials: self.device.new_buffer(
                field_bytes(census.mass_partials.max(1), "mass partials")?,
                MTLResourceOptions::StorageModePrivate,
            ),
            phase_tables: self.device.new_buffer(
                field_bytes(ADDRESS_PHASES * ADDRESS_BINS, "address phase tables")?,
                MTLResourceOptions::StorageModeShared,
            ),
            address_partials: self.device.new_buffer(
                field_bytes(partial_fields, "address partials")?,
                MTLResourceOptions::StorageModePrivate,
            ),
            raf_output: self.device.new_buffer(
                field_bytes(RAF_OUTPUT_LANES * ADDRESS_BINS, "RAF output")?,
                MTLResourceOptions::StorageModeShared,
            ),
            suffix_output: self.device.new_buffer(
                field_bytes(TOTAL_SUFFIXES * ADDRESS_BINS, "suffix output")?,
                MTLResourceOptions::StorageModeShared,
            ),
        };

        Ok(AddressAtomSequence {
            context: self.clone(),
            pipelines,
            buffers,
            topology,
            config,
            e_in_length: e_in_values.len(),
            e_out_length: e_out_values.len(),
            table_offsets,
            phases_executed: 0,
            challenge_batches: Vec::with_capacity(ADDRESS_PHASES),
            finished: false,
            gpu_active: Duration::ZERO,
        })
    }
}

impl AddressAtomSequence {
    pub(crate) fn first_phase(
        &mut self,
    ) -> Result<AddressAtomPhaseOutput, AddressAtomRuntimeError> {
        if self.phases_executed != 0 || self.finished {
            return Err(AddressAtomRuntimeError::InvalidState(
                "the first phase was requested more than once",
            ));
        }
        self.execute_phase()
    }

    pub(crate) fn next_phase(
        &mut self,
        previous_phase_challenges: [AkitaField; PHASE_CHALLENGES],
    ) -> Result<AddressAtomPhaseOutput, AddressAtomRuntimeError> {
        if !(1..ADDRESS_PHASES).contains(&self.phases_executed) || self.finished {
            return Err(AddressAtomRuntimeError::InvalidState(
                "a later phase was requested outside phases 1..15",
            ));
        }
        self.install_challenge_table(previous_phase_challenges)?;
        self.execute_phase()
    }

    pub(crate) fn finish_address(
        &mut self,
        final_phase_challenges: [AkitaField; PHASE_CHALLENGES],
    ) -> Result<(), AddressAtomRuntimeError> {
        if self.phases_executed != ADDRESS_PHASES || self.finished {
            return Err(AddressAtomRuntimeError::InvalidState(
                "address finish requires exactly 16 completed phases",
            ));
        }
        self.install_challenge_table(final_phase_challenges)?;
        self.finished = true;
        Ok(())
    }

    pub(crate) fn census(&self) -> Result<AddressAtomTopologyCensus, AddressAtomRuntimeError> {
        Ok(self.topology.census()?)
    }

    pub(crate) fn atom_count(&self) -> Result<usize, AddressAtomRuntimeError> {
        Ok(self.census()?.atoms)
    }

    pub(crate) const fn phases_executed(&self) -> usize {
        self.phases_executed
    }

    pub(crate) const fn is_finished(&self) -> bool {
        self.finished
    }

    pub(crate) const fn gpu_active(&self) -> Duration {
        self.gpu_active
    }

    pub(crate) fn phase_challenges(&self) -> &[[AkitaField; PHASE_CHALLENGES]] {
        &self.challenge_batches
    }

    fn install_challenge_table(
        &mut self,
        challenges: [AkitaField; PHASE_CHALLENGES],
    ) -> Result<(), AddressAtomRuntimeError> {
        let table: Vec<AkitaField> = EqPolynomial::evals(&challenges, None);
        if table.len() != ADDRESS_BINS || self.challenge_batches.len() + 1 != self.phases_executed {
            return Err(AddressAtomRuntimeError::InvalidState(
                "challenge table does not match the preceding phase",
            ));
        }
        let offset = self.challenge_batches.len() * ADDRESS_BINS;
        // SAFETY: the shared buffer owns exactly 16 phase tables. Commands are
        // completed before this method can install the next table.
        let output = unsafe {
            slice::from_raw_parts_mut(
                self.buffers.phase_tables.contents().cast::<Fp128>(),
                ADDRESS_PHASES * ADDRESS_BINS,
            )
        };
        for (output, value) in output[offset..offset + ADDRESS_BINS].iter_mut().zip(&table) {
            *output = Fp128::from_jolt_field(value);
        }
        self.challenge_batches.push(challenges);
        Ok(())
    }

    fn execute_phase(&mut self) -> Result<AddressAtomPhaseOutput, AddressAtomRuntimeError> {
        let phase = self.phases_executed;
        let suffix_len = 120usize.checked_sub(phase * PHASE_CHALLENGES).ok_or(
            AddressAtomRuntimeError::InvalidState("address suffix length underflowed"),
        )?;
        let command_buffer = self.context.queue.new_command_buffer();
        autoreleasepool(|| {
            if phase == 0 {
                self.encode_mass_phase(command_buffer, suffix_len)?;
            } else {
                self.encode_atom_phase(command_buffer, suffix_len)?;
            }
            self.encode_finalizers(command_buffer, phase == 0);
            command_buffer.commit();
            command_buffer.wait_until_completed();
            Ok::<(), AddressAtomRuntimeError>(())
        })?;
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(MetalError::CommandFailed(command_buffer.status()).into());
        }
        let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
        let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
        if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
            return Err(MetalError::InvalidGpuTimestamps { start, end }.into());
        }
        let gpu_active = Duration::from_secs_f64(end - start);
        self.gpu_active += gpu_active;
        self.phases_executed += 1;

        let raf_values = read_fields(&self.buffers.raf_output, RAF_OUTPUT_LANES * ADDRESS_BINS);
        let suffix_values = read_fields(&self.buffers.suffix_output, TOTAL_SUFFIXES * ADDRESS_BINS);
        self.context
            .validate_inputs("InstructionReadRaf v3 RAF output", raf_values)?;
        self.context
            .validate_inputs("InstructionReadRaf v3 suffix output", suffix_values)?;
        Ok(AddressAtomPhaseOutput {
            phase,
            suffix_len,
            raf: AddressRafSums::from_values(raf_values.to_vec()),
            suffix: AddressSuffixFullSums::from_values(
                suffix_values.to_vec(),
                self.table_offsets.clone(),
            ),
            gpu_active,
        })
    }

    fn encode_mass_phase(
        &self,
        command_buffer: &metal::CommandBufferRef,
        suffix_len: usize,
    ) -> Result<(), AddressAtomRuntimeError> {
        let census = self.topology.census()?;
        let params = AtomMassPhaseParams::new(
            census.rows,
            census.atoms,
            census.mass_jobs,
            census.mass_groups,
            self.e_in_length,
            self.e_out_length,
        )?;
        if params.suffix_len as usize != suffix_len {
            return Err(AddressAtomRuntimeError::InvalidState(
                "mass phase suffix length differs from phase zero",
            ));
        }
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines.mass_phase);
        encoder.set_buffer(0, Some(&self.buffers.atom_lookups), 0);
        encoder.set_buffer(1, Some(&self.buffers.mass_jobs), 0);
        encoder.set_buffer(2, Some(&self.buffers.mass_groups), 0);
        encoder.set_buffer(3, Some(&self.buffers.cycle_indices), 0);
        encoder.set_buffer(4, Some(&self.buffers.e_in), 0);
        encoder.set_buffer(5, Some(&self.buffers.e_out), 0);
        encoder.set_buffer(6, Some(&self.buffers.atom_masses), 0);
        encoder.set_buffer(7, Some(&self.buffers.mass_partials), 0);
        encoder.set_buffer(8, Some(&self.buffers.suffix_kinds), 0);
        encoder.set_buffer(9, Some(&self.buffers.suffix_counts), 0);
        encoder.set_buffer(10, Some(&self.buffers.address_partials), 0);
        set_inline_bytes(encoder, 11, &params);
        encoder.set_threadgroup_memory_length(0, PHASE_THREADGROUP_BYTES as u64);
        encoder.dispatch_thread_groups(grid(census.mass_groups), grid(self.config.phase));
        encoder.end_encoding();

        if census.split_atoms != 0 {
            let params = AtomMassFinalizeParams::new(
                census.atoms,
                census.split_atoms,
                census.mass_partials,
            )?;
            let finalize = command_buffer.new_compute_command_encoder();
            finalize.set_compute_pipeline_state(&self.pipelines.mass_finalize);
            finalize.set_buffer(0, Some(&self.buffers.split_atoms), 0);
            finalize.set_buffer(1, Some(&self.buffers.mass_partials), 0);
            finalize.set_buffer(2, Some(&self.buffers.atom_masses), 0);
            set_inline_bytes(finalize, 3, &params);
            finalize.dispatch_thread_groups(
                grid(census.split_atoms.div_ceil(self.config.mass_finalize)),
                grid(self.config.mass_finalize),
            );
            finalize.end_encoding();
        }
        Ok(())
    }

    fn encode_atom_phase(
        &self,
        command_buffer: &metal::CommandBufferRef,
        suffix_len: usize,
    ) -> Result<(), AddressAtomRuntimeError> {
        let census = self.topology.census()?;
        let params = AtomPhaseParams::new(suffix_len, census.later_phase_jobs)?;
        let previous_table_offset = (self.phases_executed - 1)
            .checked_mul(ADDRESS_BINS * size_of::<Fp128>())
            .ok_or(InstructionReadRafV3Error::SizeOverflow(
                "previous phase table offset",
            ))? as u64;
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines.atom_phase);
        encoder.set_buffer(0, Some(&self.buffers.atom_lookups), 0);
        encoder.set_buffer(1, Some(&self.buffers.atom_masses), 0);
        encoder.set_buffer(2, Some(&self.buffers.phase_tables), previous_table_offset);
        encoder.set_buffer(3, Some(&self.buffers.phase_jobs), 0);
        encoder.set_buffer(4, Some(&self.buffers.suffix_kinds), 0);
        encoder.set_buffer(5, Some(&self.buffers.suffix_counts), 0);
        encoder.set_buffer(6, Some(&self.buffers.address_partials), 0);
        set_inline_bytes(encoder, 7, &params);
        encoder.set_threadgroup_memory_length(0, PHASE_THREADGROUP_BYTES as u64);
        encoder.dispatch_thread_groups(grid(census.later_phase_jobs), grid(self.config.phase));
        encoder.end_encoding();
        Ok(())
    }

    fn encode_finalizers(&self, command_buffer: &metal::CommandBufferRef, phase_zero: bool) {
        let offsets = if phase_zero {
            &self.buffers.phase_zero_group_offsets
        } else {
            &self.buffers.phase_job_offsets
        };
        let raf = command_buffer.new_compute_command_encoder();
        raf.set_compute_pipeline_state(&self.pipelines.raf_finalize);
        raf.set_buffer(0, Some(&self.buffers.address_partials), 0);
        raf.set_buffer(1, Some(offsets), 0);
        raf.set_buffer(2, Some(&self.buffers.raf_output), 0);
        raf.dispatch_thread_groups(grid(RAF_OUTPUT_LANES), grid(self.config.raf_finalize));
        raf.end_encoding();

        let suffix = command_buffer.new_compute_command_encoder();
        suffix.set_compute_pipeline_state(&self.pipelines.suffix_finalize);
        suffix.set_buffer(0, Some(&self.buffers.address_partials), 0);
        suffix.set_buffer(1, Some(offsets), 0);
        suffix.set_buffer(2, Some(&self.buffers.table_descriptors), 0);
        suffix.set_buffer(3, Some(&self.buffers.output_lanes), 0);
        suffix.set_buffer(4, Some(&self.buffers.suffix_output), 0);
        suffix.dispatch_thread_groups(grid(TABLES), grid(self.config.suffix_finalize));
        suffix.end_encoding();
    }
}

fn validate_pipeline(
    pipeline: &'static str,
    limits: PipelineLimits,
    threads: usize,
    dynamic_threadgroup_bytes: u64,
    maximum_threadgroup_bytes: u64,
) -> Result<(), AddressAtomRuntimeError> {
    if limits.thread_execution_width != SIMD_WIDTH {
        return Err(AddressAtomRuntimeError::ExecutionWidth {
            pipeline,
            expected: SIMD_WIDTH,
            got: limits.thread_execution_width,
        });
    }
    if threads == 0
        || threads > limits.max_total_threads_per_threadgroup
        || !threads.is_multiple_of(SIMD_WIDTH)
    {
        return Err(AddressAtomRuntimeError::ThreadLimit {
            pipeline,
            requested: threads,
            maximum: limits.max_total_threads_per_threadgroup,
        });
    }
    let requested = limits
        .static_threadgroup_memory_length
        .checked_add(dynamic_threadgroup_bytes)
        .ok_or(InstructionReadRafV3Error::SizeOverflow(
            "pipeline threadgroup memory",
        ))?;
    if requested > maximum_threadgroup_bytes {
        return Err(AddressAtomRuntimeError::ThreadgroupMemory {
            pipeline,
            requested,
            maximum: maximum_threadgroup_bytes,
        });
    }
    Ok(())
}

fn buffer_from_slice_or_placeholder<T>(device: &metal::Device, values: &[T]) -> Buffer {
    if values.is_empty() {
        device.new_buffer(
            size_of::<T>() as u64,
            MTLResourceOptions::StorageModePrivate,
        )
    } else {
        super::super::buffer_from_slice(device, values)
    }
}

fn read_fields(buffer: &Buffer, fields: usize) -> &[Fp128] {
    // SAFETY: callers allocate shared output buffers for exactly this many
    // fields and read only after the writing command completes.
    unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), fields) }
}

fn checked_product(
    left: usize,
    right: usize,
    name: &'static str,
) -> Result<usize, InstructionReadRafV3Error> {
    left.checked_mul(right)
        .ok_or(InstructionReadRafV3Error::SizeOverflow(name))
}

fn checked_bytes<T>(elements: usize, name: &'static str) -> Result<u64, InstructionReadRafV3Error> {
    checked_product(elements, size_of::<T>(), name).and_then(|bytes| {
        u64::try_from(bytes).map_err(|_| InstructionReadRafV3Error::SizeOverflow(name))
    })
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}

const fn grid(width: usize) -> MTLSize {
    MTLSize {
        width: width as u64,
        height: 1,
        depth: 1,
    }
}
