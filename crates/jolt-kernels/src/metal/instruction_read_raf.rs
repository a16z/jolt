use std::{mem::size_of, sync::Arc};

use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_field::AkitaField;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::SumcheckInputClaims;
use jolt_verifier::stages::stage5::InstructionReadRaf;
use jolt_witness::{JoltWitnessPlane, PolynomialEncoding};

use super::backend::MetalBackend;
use super::solinas::instruction_read_raf_v3::{AddressAtomSequence, ADDRESS_PHASE_BITS};
use super::solinas::{
    AddressPhaseSequence, AddressPhaseSequenceConfig, BooleanityRows,
    InstructionReadRafCompatibilityScatterConfig, InstructionReadRafDenseGroupedPlanes,
    InstructionReadRafStage1Owner, Product5Sequence, Product5SequenceConfig,
    ResidentLookupIndexPlane, SolinasMetal, PRODUCT5_FACTORS,
};
use crate::optimized::instruction_read_raf::{
    prepare_metal_instruction_read_raf, OptimizedInstructionReadRafKernel,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Dispatch and crossover settings for the stage-5 dense cycle tail.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionReadRafMetalConfig {
    /// First trace length whose address phases run on Metal.
    pub address_cutoff_elements: usize,
    /// Address implementation selected before the first round is absorbed.
    pub address_implementation: InstructionReadRafAddressImplementation,
    /// Maximum exact-key atom count admitted by the compressed address path.
    pub address_atom_max_unique: usize,
    /// Threadgroup width for the Stage-1 compatibility scatter.
    pub stage1_scatter_threads_per_threadgroup: usize,
    /// Dispatch geometry for the resident address sequence.
    pub address_dispatch: AddressPhaseSequenceConfig,
    /// First table length whose next round runs on the CPU.
    pub cutoff_elements: usize,
    /// Threadgroup widths for the initial message and fused transitions.
    pub dispatch: Product5SequenceConfig,
}

impl Default for InstructionReadRafMetalConfig {
    fn default() -> Self {
        Self {
            address_cutoff_elements: 1 << 25,
            address_implementation: InstructionReadRafAddressImplementation::Stage1Grouped,
            address_atom_max_unique: 1 << 16,
            stage1_scatter_threads_per_threadgroup: 256,
            address_dispatch: AddressPhaseSequenceConfig::default(),
            cutoff_elements: 1 << 16,
            dispatch: Product5SequenceConfig::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InstructionReadRafAddressImplementation {
    GroupedRows,
    AtomV3,
    Stage1Grouped,
}

impl PrepareKernel<AkitaField, InstructionReadRaf<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, InstructionReadRaf<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = InstructionReadRaf<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let dimensions = inputs.relation.dimensions();
        let trace_elements = 1usize << dimensions.log_t();
        let use_metal_address =
            trace_elements >= self.config.instruction_read_raf.address_cutoff_elements;
        let collect_bytecode_support = self.config.bytecode_read_raf_address.implementation
            == super::bytecode_read_raf::BytecodeReadRafAddressImplementation::AddressMajorShadow
            && trace_elements >= self.config.bytecode_read_raf_address.dispatch.trace_cutoff;
        let retain_lookup_plane = trace_elements
            >= self
                .config
                .instruction_ra_virtualization
                .trace_cutoff_elements;
        let stage1_owner = (use_metal_address
            && self.config.instruction_read_raf.address_implementation
                == InstructionReadRafAddressImplementation::Stage1Grouped
            && dimensions.num_virtual_ra_polys() + 1 == PRODUCT5_FACTORS
            && !collect_bytecode_support)
            .then(|| session.take::<InstructionReadRafStage1Owner>())
            .flatten();
        let (cpu, resident_grouped_planes) = if let Some(owner) = stage1_owner.as_ref() {
            let device_registry_id = self.context.device_registry_id();
            let scatter_source = owner
                .lease(trace_elements, device_registry_id)
                .map_err(metal_prepare_error)?;
            let retained_claims = owner
                .lease(trace_elements, device_registry_id)
                .map_err(metal_prepare_error)?;
            let cpu = OptimizedInstructionReadRafKernel::new_metal_resident(
                dimensions,
                &inputs.points.lookup_output,
                retained_claims,
                inputs.challenges.gamma,
            )?;
            let planes = self
                .context
                .prepare_instruction_read_raf_compatibility_scatter(
                    scatter_source,
                    &inputs.points.lookup_output,
                    InstructionReadRafCompatibilityScatterConfig {
                        threads_per_threadgroup: self
                            .config
                            .instruction_read_raf
                            .stage1_scatter_threads_per_threadgroup,
                    },
                )
                .map_err(metal_prepare_error)?;
            let receipt = planes.receipt();
            let execution = planes.execution();
            let identities = receipt.allocation_identities();
            let _span = tracing::info_span!(
                "MetalInstructionReadRaf::stage1_grouped_scatter",
                rows = receipt.rows(),
                preparation_wall_ns = duration_ns(execution.preparation_wall),
                command_wall_ns = duration_ns(execution.command_wall),
                gpu_active_ns = duration_ns(execution.gpu_active),
                status_readback_bytes = execution.status_readback_bytes,
                packed_rows_bytes = receipt.packed_rows_bytes(),
                lookups_bytes = receipt.lookups_bytes(),
                inverse_bytes = receipt.inverse_bytes(),
                weights_bytes = receipt.weights_bytes(),
                packed_rows_identity = identities[0],
                lookups_identity = identities[1],
                inverse_identity = identities[2],
                weights_identity = identities[3],
                source_generation = receipt.source().source_generation(),
                source_completion_serial = receipt.source().completion_serial(),
                source_row_allocation_identity = receipt.source().row_allocation_identity(),
                source_claim_allocation_identity = receipt.source().claim_allocation_identity(),
                source_count_allocation_identity = receipt.source().count_allocation_identity(),
                source_count_chunks = receipt.source().count_chunks(),
                source_count_bytes = receipt.source().count_bytes(),
                source_device_registry_id = receipt.source().device_registry_id(),
                source_count_order = "table_major_then_none_v1",
                scatter_completion_serial = receipt.completion_serial(),
                e_in_length = receipt.e_in_length(),
                e_out_length = receipt.e_out_length(),
                additional_allocation_bytes = receipt.additional_allocation_bytes(),
                command_buffers = receipt.command_buffers(),
                waits = receipt.waits(),
                encoders = receipt.encoders(),
                threadgroups = receipt.threadgroups(),
                threads_per_threadgroup = receipt.threads_per_threadgroup(),
                dynamic_threadgroup_bytes = receipt.dynamic_threadgroup_bytes(),
                static_threadgroup_bytes = receipt.static_threadgroup_bytes(),
                dispatches = receipt.dispatches(),
                source_copy_bytes = receipt.source_copy_bytes(),
                full_plane_readback_bytes = receipt.full_plane_readback_bytes(),
                legacy_row_collection_rows = 0u64,
                legacy_bucket_scan_rows = 0u64,
                legacy_host_repack_bytes = 0u64,
                legacy_booleanity_upload_bytes = 0u64,
                complete_overwrite = receipt.complete_overwrite(),
            )
            .entered();
            (cpu, Some(planes))
        } else {
            (
                prepare_metal_instruction_read_raf(
                    session,
                    witness,
                    inputs,
                    use_metal_address,
                    collect_bytecode_support,
                )?,
                None,
            )
        };
        let hamming_log_k_chunk = committed_hamming_log_k_chunk(witness, dimensions.log_t());
        let hamming_rows_requested = hamming_log_k_chunk.is_some_and(|log_k_chunk| {
            self.config.hamming_weight_claim_reduction.admits(
                trace_elements,
                dimensions.log_t(),
                log_k_chunk,
            )
        });
        let resident_rows_requested = trace_elements
            >= self.config.booleanity_address.trace_cutoff_elements
            || trace_elements >= self.config.bytecode_read_raf_address.dispatch.trace_cutoff
            || trace_elements >= self.config.booleanity_cycle.trace_cutoff_elements
            || trace_elements >= self.config.bytecode_read_raf_cycle.trace_cutoff_elements
            || hamming_rows_requested;
        if resident_rows_requested && session.state::<BooleanityRows>().is_none() {
            let prepared_rows = if let Some(owner) = stage1_owner.as_ref() {
                let receipt = owner.receipt();
                Ok((
                    owner.booleanity_rows(),
                    0u64,
                    0u64,
                    "stage1_owner_v1",
                    receipt.source_generation(),
                    receipt.completion_serial(),
                    receipt.claim_allocation_identity(),
                ))
            } else {
                cpu.metal_prepare_booleanity_rows(&self.context)
                    .map(|rows| {
                        let upload_bytes =
                            (rows.len() * size_of::<super::solinas::BooleanityRow>()) as u64;
                        (rows, upload_bytes, 1, "member_upload_v1", 0, 0, 0)
                    })
            };
            match prepared_rows {
                Ok((
                    rows,
                    row_upload_bytes,
                    row_allocations,
                    source_kind,
                    source_generation,
                    source_completion_serial,
                    source_claim_allocation_identity,
                )) => {
                    let lifecycle_span = tracing::info_span!(
                        "MetalBooleanityRows::stage5_prepare",
                        resident_rows_storage_id = rows.allocation_identity(),
                        resident_rows = rows.len(),
                        resident_row_bytes = size_of::<super::solinas::BooleanityRow>(),
                        device_registry_id = rows.device_registry_id(),
                        row_allocations,
                        row_upload_bytes,
                        source_kind,
                        source_generation,
                        source_completion_serial,
                        source_claim_allocation_identity,
                    )
                    .entered();
                    drop(lifecycle_span);
                    session.park(rows);
                }
                Err(error) if error.is_capacity_error() => {
                    tracing::warn!(
                        target: "jolt::metal",
                        error = %error,
                        "Booleanity resident rows were not admitted"
                    );
                }
                Err(error) => return Err(backend_error(error.to_string()).into()),
            }
        }
        Ok(Box::new(MetalInstructionReadRafKernel::new(
            cpu,
            Arc::clone(&self.context),
            self.config.instruction_read_raf,
            use_metal_address,
            retain_lookup_plane,
            resident_grouped_planes,
        )?))
    }
}

fn committed_hamming_log_k_chunk(
    witness: &dyn JoltWitnessPlane<AkitaField>,
    log_t: usize,
) -> Option<usize> {
    witness
        .shape(JoltCommittedPolynomial::InstructionRa(0).into())
        .ok()
        .filter(|shape| shape.encoding == PolynomialEncoding::OneHot)
        .and_then(|shape| shape.log_rows.checked_sub(log_t))
}

pub(crate) struct MetalInstructionReadRafKernel {
    cpu: OptimizedInstructionReadRafKernel<AkitaField>,
    context: Arc<SolinasMetal>,
    config: InstructionReadRafMetalConfig,
    address_sequence: Option<ResidentAddressSequence>,
    address_challenge_batch: Vec<AkitaField>,
    resident_lookup_plane: Option<ResidentLookupIndexPlane>,
    sequence: Option<Product5Sequence>,
    host_tail: Option<[Vec<AkitaField>; PRODUCT5_FACTORS]>,
    metal_rounds: usize,
    metal_address_phases: usize,
}

enum ResidentAddressSequence {
    Grouped(Box<AddressPhaseSequence>),
    Atom(Box<AddressAtomSequence>),
}

impl MetalInstructionReadRafKernel {
    pub(crate) fn new(
        cpu: OptimizedInstructionReadRafKernel<AkitaField>,
        context: Arc<SolinasMetal>,
        config: InstructionReadRafMetalConfig,
        use_metal_address: bool,
        retain_lookup_plane: bool,
        resident_grouped_planes: Option<InstructionReadRafDenseGroupedPlanes>,
    ) -> Result<Self, SumcheckError<AkitaField>> {
        let mut kernel = Self {
            cpu,
            context,
            config,
            address_sequence: None,
            address_challenge_batch: Vec::with_capacity(ADDRESS_PHASE_BITS),
            resident_lookup_plane: None,
            sequence: None,
            host_tail: Some(std::array::from_fn(|_| {
                vec![AkitaField::zero(); config.cutoff_elements]
            })),
            metal_rounds: 0,
            metal_address_phases: 0,
        };
        if use_metal_address {
            if let Some(planes) = resident_grouped_planes {
                if config.address_implementation
                    != InstructionReadRafAddressImplementation::Stage1Grouped
                {
                    return Err(backend_error(
                        "Stage-1 grouped planes reached another address implementation",
                    ));
                }
                let mut sequence = {
                    let _span = tracing::info_span!(
                        "MetalInstructionReadRaf::stage1_grouped_sequence_prepare"
                    )
                    .entered();
                    kernel
                        .context
                        .prepare_address_phase_sequence_from_resident_grouped(
                            planes,
                            config.address_dispatch,
                        )
                        .map_err(|error| backend_error(error.to_string()))?
                };
                if retain_lookup_plane {
                    kernel.resident_lookup_plane = Some(sequence.resident_lookup_index_plane());
                }
                let (suffix_len, previous) = kernel.cpu.metal_address_phase_request()?;
                let sums = sequence
                    .phase(suffix_len, previous.as_ref())
                    .map_err(|error| backend_error(error.to_string()))?;
                kernel.cpu.metal_install_address_phase(sums)?;
                kernel.metal_address_phases = 1;
                kernel.address_sequence =
                    Some(ResidentAddressSequence::Grouped(Box::new(sequence)));
                return Ok(kernel);
            }
            let atom = if config.address_implementation
                == InstructionReadRafAddressImplementation::AtomV3
            {
                let _span =
                    tracing::info_span!("MetalInstructionReadRaf::atom_v3_sequence_prepare")
                        .entered();
                kernel.cpu.metal_prepare_atom_address_sequence(
                    &kernel.context,
                    config.address_atom_max_unique,
                    retain_lookup_plane,
                )?
            } else {
                None
            };
            if let Some((mut sequence, lookup_plane)) = atom {
                let (suffix_len, _) = kernel.cpu.metal_address_phase_request()?;
                let address_atoms = sequence
                    .atom_count()
                    .map_err(|error| backend_error(error.to_string()))?;
                let output = {
                    let _span = tracing::info_span!(
                        "MetalInstructionReadRaf::atom_v3_initial_address_phase",
                        address_atoms = address_atoms as u64,
                    )
                    .entered();
                    sequence
                        .first_phase()
                        .map_err(|error| backend_error(error.to_string()))?
                };
                if output.suffix_len() != suffix_len as usize || output.phase() != 0 {
                    return Err(backend_error("atom address phase zero has the wrong shape"));
                }
                kernel
                    .cpu
                    .metal_install_address_phase(output.into_phase_sums())?;
                kernel.resident_lookup_plane = lookup_plane;
                kernel.metal_address_phases = 1;
                kernel.address_sequence = Some(ResidentAddressSequence::Atom(Box::new(sequence)));
            } else {
                let mut sequence = {
                    let _span =
                        tracing::info_span!("MetalInstructionReadRaf::sequence_prepare").entered();
                    kernel
                        .cpu
                        .metal_prepare_address_sequence(&kernel.context, config.address_dispatch)?
                };
                if retain_lookup_plane {
                    kernel.resident_lookup_plane = Some(sequence.resident_lookup_index_plane());
                }
                let (suffix_len, previous) = kernel.cpu.metal_address_phase_request()?;
                let sums = {
                    let _span =
                        tracing::info_span!("MetalInstructionReadRaf::initial_address_phase")
                            .entered();
                    sequence
                        .phase(suffix_len, previous.as_ref())
                        .map_err(|error| backend_error(error.to_string()))?
                };
                kernel.cpu.metal_install_address_phase(sums)?;
                kernel.metal_address_phases = 1;
                kernel.address_sequence =
                    Some(ResidentAddressSequence::Grouped(Box::new(sequence)));
            }
        }
        Ok(kernel)
    }

    #[cfg(test)]
    pub(crate) const fn metal_rounds(&self) -> usize {
        self.metal_rounds
    }

    #[cfg(test)]
    pub(crate) const fn metal_address_phases(&self) -> usize {
        self.metal_address_phases
    }

    fn install_next_address_phase(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let (suffix_len, previous) = self.cpu.metal_address_phase_request()?;
        let atom_batch = matches!(
            self.address_sequence,
            Some(ResidentAddressSequence::Atom(_))
        )
        .then(|| self.take_atom_challenge_batch())
        .transpose()?;
        let sequence = self
            .address_sequence
            .as_mut()
            .ok_or_else(|| backend_error("resident address sequence disappeared"))?;
        let sums = match sequence {
            ResidentAddressSequence::Grouped(sequence) => sequence
                .phase(suffix_len, previous.as_ref())
                .map_err(|error| backend_error(error.to_string()))?,
            ResidentAddressSequence::Atom(sequence) => {
                let _span =
                    tracing::info_span!("MetalInstructionReadRaf::atom_v3_address_phase").entered();
                let output = sequence
                    .next_phase(atom_batch.ok_or_else(|| {
                        backend_error("atom address phase is missing its challenge batch")
                    })?)
                    .map_err(|error| backend_error(error.to_string()))?;
                if output.suffix_len() != suffix_len as usize
                    || output.phase() != self.metal_address_phases
                {
                    return Err(backend_error("atom address phase has the wrong shape"));
                }
                output.into_phase_sums()
            }
        };
        self.cpu.metal_install_address_phase(sums)?;
        self.metal_address_phases += 1;
        Ok(())
    }

    fn take_atom_challenge_batch(
        &mut self,
    ) -> Result<[AkitaField; ADDRESS_PHASE_BITS], SumcheckError<AkitaField>> {
        let challenges = std::mem::take(&mut self.address_challenge_batch);
        challenges.try_into().map_err(|challenges: Vec<_>| {
            backend_error(format!(
                "atom address phase received {} challenges, expected {ADDRESS_PHASE_BITS}",
                challenges.len()
            ))
        })
    }

    fn finish_atom_address_sequence(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let challenges = self.take_atom_challenge_batch()?;
        let sequence = self
            .address_sequence
            .take()
            .ok_or_else(|| backend_error("atom address sequence disappeared before final bind"))?;
        let ResidentAddressSequence::Atom(mut sequence) = sequence else {
            return Err(backend_error(
                "grouped address sequence reached the atom finalization path",
            ));
        };
        sequence
            .finish_address(challenges)
            .map_err(|error| backend_error(error.to_string()))?;
        if !sequence.is_finished() || sequence.phases_executed() != self.metal_address_phases {
            return Err(backend_error(
                "atom address sequence did not finish every installed phase",
            ));
        }
        Ok(())
    }

    fn restore_cpu_tail(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let _span = tracing::info_span!("MetalInstructionReadRaf::readback").entered();
        let sequence = self
            .sequence
            .take()
            .ok_or_else(|| backend_error("device sequence is absent during readback"))?;
        let mut tables = self
            .host_tail
            .take()
            .ok_or_else(|| backend_error("CPU tail buffers were already consumed"))?;
        sequence
            .read_current_factor_tables(&mut tables)
            .map_err(|error| backend_error(error.to_string()))?;
        self.cpu.metal_restore_dense(tables)
    }
}

impl ProveRounds<AkitaField> for MetalInstructionReadRafKernel {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<jolt_poly::UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        let mut bind = bind;
        if self.address_sequence.is_some() && self.cpu.metal_address_active() {
            let _span = tracing::info_span!("MetalInstructionReadRaf::address_round").entered();
            if let Some(challenge) = bind.take() {
                let atom_address = matches!(
                    self.address_sequence,
                    Some(ResidentAddressSequence::Atom(_))
                );
                self.cpu.metal_bind_address(challenge)?;
                if atom_address {
                    self.address_challenge_batch.push(challenge);
                    if self.address_challenge_batch.len() > ADDRESS_PHASE_BITS {
                        return Err(backend_error(
                            "atom address phase accumulated too many challenges",
                        ));
                    }
                }
                if self.cpu.metal_address_phase_pending() {
                    self.install_next_address_phase()?;
                } else if atom_address && !self.cpu.metal_address_active() {
                    self.finish_atom_address_sequence()?;
                }
            }
            if self.cpu.metal_address_active() {
                return self.cpu.metal_address_message(previous_claim);
            }
        }

        if self.address_sequence.is_some() && !self.cpu.metal_resident_cycle_available() {
            self.address_sequence = None;
        }

        if matches!(
            self.address_sequence,
            Some(ResidentAddressSequence::Grouped(_))
        ) {
            if let Some(challenge) = bind.take() {
                let _span =
                    tracing::info_span!("MetalInstructionReadRaf::resident_handoff").entered();
                let address_sequence = self
                    .address_sequence
                    .take()
                    .ok_or_else(|| backend_error("resident address sequence disappeared"))?;
                let ResidentAddressSequence::Grouped(address_sequence) = address_sequence else {
                    return Err(backend_error(
                        "atom address sequence reached the grouped cycle handoff",
                    ));
                };
                let (sequence, q_evals) = self.cpu.metal_offload_resident_bind(
                    challenge,
                    *address_sequence,
                    self.config.dispatch,
                )?;
                let poly = self.cpu.metal_cycle_message(&q_evals, previous_claim)?;
                self.sequence = Some(sequence);
                self.metal_rounds += 1;
                return Ok(poly);
            }
            let _span =
                tracing::info_span!("MetalInstructionReadRaf::resident_first_message").entered();
            let (cpu, address_sequence) = (&self.cpu, self.address_sequence.as_mut());
            let address_sequence = address_sequence
                .ok_or_else(|| backend_error("resident address sequence disappeared"))?;
            let ResidentAddressSequence::Grouped(address_sequence) = address_sequence else {
                return Err(backend_error(
                    "atom address sequence reached the grouped cycle message",
                ));
            };
            let poly = cpu.metal_resident_cycle_message(address_sequence, previous_claim)?;
            self.metal_rounds += 1;
            return Ok(poly);
        }

        if self
            .sequence
            .as_ref()
            .is_some_and(|sequence| sequence.current_elements() <= self.config.cutoff_elements)
        {
            self.restore_cpu_tail()?;
            return self.cpu.prove_round(bind, round, previous_claim);
        }

        if self.sequence.is_some() {
            let _span = tracing::info_span!("MetalInstructionReadRaf::resident_round").entered();
            let challenge = bind.ok_or_else(|| {
                backend_error("device-resident cycle round did not receive its prior challenge")
            })?;
            self.cpu.metal_bind_offloaded(challenge)?;
            let (cpu, sequence) = (&self.cpu, self.sequence.as_mut());
            let sequence = sequence
                .ok_or_else(|| backend_error("device sequence disappeared before dispatch"))?;
            let (e_in, e_out) = cpu.metal_cycle_weights()?;
            let q_evals = sequence
                .bind_and_message(challenge, e_in, e_out)
                .map_err(|error| backend_error(error.to_string()))?;
            self.metal_rounds += 1;
            return cpu.metal_cycle_message(&q_evals, previous_claim);
        }

        if let Some(challenge) = bind {
            if self
                .cpu
                .metal_handoff_available(self.config.cutoff_elements)
            {
                let _span = tracing::info_span!("MetalInstructionReadRaf::handoff").entered();
                let mut sequence = self.cpu.metal_offload_pending_bind(
                    challenge,
                    &self.context,
                    self.config.dispatch,
                )?;
                let (e_in, e_out) = self.cpu.metal_cycle_weights()?;
                let q_evals = sequence
                    .message(e_in, e_out)
                    .map_err(|error| backend_error(error.to_string()))?;
                let poly = self.cpu.metal_cycle_message(&q_evals, previous_claim)?;
                self.metal_rounds += 1;
                self.sequence = Some(sequence);
                return Ok(poly);
            }
        }

        self.cpu.prove_round(bind, round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.sequence.is_some() {
            self.restore_cpu_tail()?;
        }
        self.cpu.finish_rounds(bind)
    }
}

impl SumcheckKernel<AkitaField> for MetalInstructionReadRafKernel {
    type Relation = InstructionReadRaf<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<
        jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafOutputClaims<
            AkitaField,
        >,
        SumcheckKernelError<AkitaField>,
    > {
        self.cpu.output_claims(inputs)
    }

    fn park_residue(mut self: Box<Self>, session: &mut ProofSession) {
        if let Some(plane) = self.resident_lookup_plane.take() {
            session.park(plane);
        }
    }
}

fn backend_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}

fn metal_prepare_error(error: super::solinas::MetalError) -> KernelError<AkitaField> {
    backend_error(error.to_string()).into()
}

fn duration_ns(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use jolt_witness::testing::with_sample_backend_at_log_t;

    use super::committed_hamming_log_k_chunk;

    #[test]
    fn derives_committed_chunk_width_from_the_witness_grid() {
        with_sample_backend_at_log_t(3, 8, |backend| {
            assert_eq!(committed_hamming_log_k_chunk(backend, 3), Some(8));
        });
        with_sample_backend_at_log_t(3, 4, |backend| {
            assert_eq!(committed_hamming_log_k_chunk(backend, 3), Some(4));
        });
    }
}
