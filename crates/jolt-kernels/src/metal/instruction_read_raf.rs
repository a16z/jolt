#[cfg(feature = "parallel")]
use std::sync::OnceLock;
use std::sync::{mpsc, Arc};
use std::thread::{Builder, JoinHandle};

use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_field::Zero as _;
use jolt_sumcheck::{ProveRounds, RoundExecutionDomain, SumcheckError};
use jolt_verifier::stages::relations::SumcheckInputClaims;
use jolt_verifier::stages::stage5::InstructionReadRaf;
use jolt_witness::{JoltWitnessPlane, PolynomialEncoding};
#[cfg(feature = "parallel")]
use rayon::{ThreadPool, ThreadPoolBuilder};
use tracing::field::Empty;

use super::backend::MetalBackend;
use super::solinas::bytecode_read_raf_address::{
    BytecodeAddressFusedScatterRequest, BytecodeAddressSparseStage1Carrier,
    BytecodeAddressStage1TopologyOwner,
};
use super::solinas::{
    instruction_read_raf_stage5_main_plane_bytes, AddressPhaseSequence, AddressPhaseSequenceConfig,
    AddressPhaseSums, BooleanityRows, InstructionReadRafCompatibilityScatterConfig,
    InstructionReadRafDenseGroupedPlanes, InstructionReadRafDenseGroupedReceipt,
    InstructionReadRafFusedBytecodeReceipt, InstructionReadRafStage1Owner,
    InstructionReadRafStage1Receipt, PendingInstructionReadRafSourcePrimer, Product5Sequence,
    Product5SequenceConfig, RegistersValInstructionSourceLease, ResidentLookupIndexPlane,
    SolinasMetal, PRODUCT5_FACTORS,
};
use crate::optimized::instruction_read_raf::{
    prepare_metal_instruction_read_raf, OptimizedInstructionReadRafKernel,
};
use crate::optimized::spartan_outer::{
    prepare_metal_instruction_read_raf_stage1_owner, MetalSpartanDenseRowsError,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, Stage5InstructionReadRafPrefetch,
    SumcheckKernel, SumcheckKernelError,
};

/// Dispatch and crossover settings for the stage-5 dense cycle tail.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionReadRafMetalConfig {
    /// First trace length whose address phases run on Metal.
    pub address_cutoff_elements: usize,
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
            address_cutoff_elements: 1 << 24,
            stage1_scatter_threads_per_threadgroup: 256,
            address_dispatch: AddressPhaseSequenceConfig::default(),
            cutoff_elements: 1 << 16,
            dispatch: Product5SequenceConfig::default(),
        }
    }
}

const SOURCE_PRIMER_CUTOFF_ELEMENTS: usize = 1 << 28;
const INITIAL_ADDRESS_SUFFIX_BITS: u32 = 120;
/// The resident cycle route binds its first two cycle challenges before its
/// handoff consumes them in the third cycle round (`prove_round`), so it runs
/// only on traces of eight rows or more; `MetalBackend::validate_config`
/// holds the address cutoff to this.
pub(super) const MIN_ADDRESS_CUTOFF_ELEMENTS: usize = 8;

struct PrefetchedInstructionReadRafScatter {
    sequence: Box<AddressPhaseSequence>,
    initial_sums: AddressPhaseSums,
    receipt: InstructionReadRafDenseGroupedReceipt,
    bytecode_carrier: Option<BytecodeAddressSparseStage1Carrier>,
    registers_val_lease: Option<RegistersValInstructionSourceLease>,
}

struct PendingInstructionReadRafScatter {
    rows: usize,
    lookup_output_point: Vec<AkitaField>,
    start_sender: mpsc::Sender<()>,
    handle: Option<JoinHandle<Result<PrefetchedInstructionReadRafScatter, String>>>,
}

struct InstructionReadRafScatterStart {
    sender: mpsc::Sender<()>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for InstructionReadRafScatterStart {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.enter_self_sized::<Self>().exit();
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for PendingInstructionReadRafScatter {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.enter_self_sized::<Self>().exit();
    }
}

impl Drop for PendingInstructionReadRafScatter {
    fn drop(&mut self) {
        let _ = self.start_sender.send(());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl PendingInstructionReadRafScatter {
    fn join(mut self) -> Result<PrefetchedInstructionReadRafScatter, KernelError<AkitaField>> {
        let _ = self.start_sender.send(());
        let handle = self.handle.take().ok_or(KernelError::InvariantViolation {
            reason: "Instruction Read-RAF scatter prefetch was already consumed",
        })?;
        let result = handle.join().map_err(|_| KernelError::InvariantViolation {
            reason: "Instruction Read-RAF scatter prefetch worker panicked",
        })?;
        result.map_err(metal_prepare_error)
    }
}

pub(super) fn start_instruction_read_raf_scatter(
    session: &mut ProofSession,
) -> Result<(), KernelError<AkitaField>> {
    let Some(start) = session.take::<InstructionReadRafScatterStart>() else {
        return Ok(());
    };
    start
        .sender
        .send(())
        .map_err(|_| KernelError::InvariantViolation {
            reason: "Instruction Read-RAF scatter prefetch worker stopped before release",
        })
}

/// A Stage-5 owner admitted when Stage 4 began, waiting for Stage 4's prepare
/// to pass its footprint peak ([`MetalBackend::start_stage5_owner_build`]).
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct Stage5OwnerRequest {
    rows: usize,
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct PendingStage5Owner {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    handle: Option<JoinHandle<Result<InstructionReadRafStage1Owner, KernelError<AkitaField>>>>,
}

impl Drop for PendingStage5Owner {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl PendingStage5Owner {
    fn join(
        mut self,
        rows: usize,
    ) -> Result<InstructionReadRafStage1Owner, KernelError<AkitaField>> {
        let span = tracing::info_span!(
            "MetalInstructionReadRaf::stage5_owner_join",
            rows,
            completed_before_join = Empty,
        );
        let _entered = span.enter();
        let handle = self.handle.take().ok_or(KernelError::InvariantViolation {
            reason: "Stage-5 owner build was already consumed",
        })?;
        let _ = span.record("completed_before_join", handle.is_finished());
        let owner = handle
            .join()
            .map_err(|_| KernelError::InvariantViolation {
                reason: "Stage-5 owner build worker panicked",
            })??;
        if owner.receipt().rows() != rows {
            return Err(KernelError::InvariantViolation {
                reason: "Stage-5 owner build has stale rows",
            });
        }
        Ok(owner)
    }
}

/// The owner build runs on its own pool so the rayon jobs of the Stage-4
/// rounds it overlaps do not queue behind its chunks; two cores stay with the
/// stage's submitting thread and host lane.
#[cfg(feature = "parallel")]
#[expect(
    clippy::expect_used,
    reason = "a pool that cannot spawn threads is an unrecoverable environment failure"
)]
fn on_stage5_owner_pool<R: Send>(build: impl FnOnce() -> R + Send) -> R {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::thread::available_parallelism()
            .map_or(1, |threads| threads.get().saturating_sub(2).max(1));
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("jolt-stage5-owner-{index}"))
            .build()
            .expect("the Stage-5 owner pool must build")
    })
    .install(build)
}

#[cfg(not(feature = "parallel"))]
fn on_stage5_owner_pool<R: Send>(build: impl FnOnce() -> R + Send) -> R {
    build()
}

impl MetalBackend {
    /// Starts the GPU page-mapping of the Stage-1 instruction read-RAF rows well
    /// before Stage 4 touches them. The registers read-write cycle sequence reads
    /// that plane in its first message; without a primer the first command pays
    /// the whole first-GPU-touch cost on the critical path (639 ms at T=2^28).
    pub(super) fn prime_instruction_read_raf_source(
        &self,
        session: &mut ProofSession,
    ) -> Result<(), KernelError<AkitaField>> {
        if session
            .state::<PendingInstructionReadRafScatter>()
            .is_some()
            || session
                .state::<PendingInstructionReadRafSourcePrimer>()
                .is_some()
        {
            return Ok(());
        }
        let Some(owner) = session
            .state::<InstructionReadRafStage1Owner>()
            .filter(|owner| owner.receipt().rows() >= SOURCE_PRIMER_CUTOFF_ELEMENTS)
            .cloned()
        else {
            return Ok(());
        };
        self.submit_source_primer(session, &owner)
    }

    /// Whether Stage 5 runs the resident owner route, the only consumer of a
    /// Stage-5 owner.
    fn stage5_owner_geometry(&self, dimensions: InstructionReadRafDimensions) -> bool {
        1usize << dimensions.log_t() >= self.config.instruction_read_raf.address_cutoff_elements
            && dimensions.num_virtual_ra_polys() + 1 == PRODUCT5_FACTORS
    }

    /// Builds the Stage-1 owner at Stage 5 when Stage 0 could not co-produce
    /// it: only for a random-access witness, and only when the Stage-5 main
    /// planes fit the device working set. Otherwise Stage 5 keeps its
    /// CPU-built route and RegistersVal its CPU kernel.
    fn prepare_stage5_owner(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        rows: usize,
        registers_val_lease: bool,
    ) -> Result<(), KernelError<AkitaField>> {
        let Some(access) = witness.owned_rows().filter(|owned| rows <= owned.cycles()) else {
            return Ok(());
        };
        if !self.admit_stage5_owner(rows, registers_val_lease)? {
            return Ok(());
        }
        let owner = prepare_metal_instruction_read_raf_stage1_owner(&self.context, &access, rows)
            .map_err(MetalSpartanDenseRowsError::into_kernel_error)?;
        session.park(owner);
        Ok(())
    }

    fn admit_stage5_owner(
        &self,
        rows: usize,
        registers_val_lease: bool,
    ) -> Result<bool, KernelError<AkitaField>> {
        let main_plane_bytes =
            instruction_read_raf_stage5_main_plane_bytes(rows, registers_val_lease)
                .map_err(metal_prepare_error)?;
        let span = tracing::info_span!(
            "MetalInstructionReadRaf::stage5_owner_admission",
            rows,
            main_plane_bytes,
            registers_val_lease,
            admitted = Empty,
        );
        let _entered = span.enter();
        if let Err(error) = self.validate_additional_working_set(main_plane_bytes) {
            if !error.is_capacity_error() {
                return Err(metal_prepare_error(error));
            }
            let _ = span.record("admitted", false);
            tracing::warn!(
                target: "jolt::metal",
                %error,
                "Stage-5 InstructionReadRAF owner was not admitted; using the CPU-built route"
            );
            return Ok(false);
        }
        let _ = span.record("admitted", true);
        Ok(true)
    }

    /// Builds an admitted Stage-5 owner on CPU workers once Stage 4's prepare
    /// has passed the stage's footprint peak; Stage 4's device rounds leave
    /// the host cores idle while Stage 5 would otherwise build it serially.
    pub(super) fn start_stage5_owner_build(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<(), KernelError<AkitaField>> {
        let Some(Stage5OwnerRequest { rows }) = session.take::<Stage5OwnerRequest>() else {
            return Ok(());
        };
        let Some(access) = witness.owned_rows().filter(|owned| rows <= owned.cycles()) else {
            return Ok(());
        };
        let context = Arc::clone(&self.context);
        let handle = Builder::new()
            .name("jolt-stage5-owner".to_owned())
            .spawn(move || {
                on_stage5_owner_pool(|| {
                    prepare_metal_instruction_read_raf_stage1_owner(&context, &access, rows)
                })
                .map_err(MetalSpartanDenseRowsError::into_kernel_error)
            })
            .map_err(|_| KernelError::InvariantViolation {
                reason: "Stage-5 owner build worker could not start",
            })?;
        session.park(PendingStage5Owner {
            handle: Some(handle),
        });
        Ok(())
    }

    fn submit_source_primer(
        &self,
        session: &mut ProofSession,
        owner: &InstructionReadRafStage1Owner,
    ) -> Result<(), KernelError<AkitaField>> {
        let pending = self
            .context
            .submit_instruction_read_raf_source_primer(owner)
            .map_err(metal_prepare_error)?;
        let span = tracing::info_span!("MetalInstructionReadRaf::source_primer_submit",);
        let _entered = span.enter();
        session.park(pending);
        Ok(())
    }
}

impl PrepareKernel<AkitaField, InstructionReadRaf<AkitaField>> for MetalBackend {
    fn prefetch(&self, session: &mut ProofSession) -> Result<(), KernelError<AkitaField>> {
        if session
            .state::<PendingInstructionReadRafScatter>()
            .is_some()
        {
            return Err(KernelError::InvariantViolation {
                reason: "Instruction Read-RAF prefetch was submitted more than once",
            });
        }
        // The source primer may already be in flight from the Stage-3 hook
        // (`prime_instruction_read_raf_source`); the scatter prefetch runs on top
        // of it and the primer-only branch below must not submit a second one.
        let primer_in_flight = session
            .state::<PendingInstructionReadRafSourcePrimer>()
            .is_some();
        let Some(owner) = session
            .state::<InstructionReadRafStage1Owner>()
            .filter(|owner| owner.receipt().rows() >= SOURCE_PRIMER_CUTOFF_ELEMENTS)
            .cloned()
        else {
            return Ok(());
        };

        let rows = owner.receipt().rows();
        let fuse_bytecode_carrier = self.config.bytecode_read_raf_address.implementation
            == crate::metal::BytecodeReadRafAddressImplementation::AddressMajor
            && rows >= self.config.bytecode_read_raf_address.trace_cutoff_elements
            && session
                .state::<BytecodeAddressStage1TopologyOwner>()
                .is_some();
        let can_prefetch_scatter = session
            .state::<Stage5InstructionReadRafPrefetch<AkitaField>>()
            .is_some()
            && (!fuse_bytecode_carrier
                || session
                    .state::<BytecodeAddressStage1TopologyOwner>()
                    .is_some());
        if can_prefetch_scatter {
            let point = session
                .take::<Stage5InstructionReadRafPrefetch<AkitaField>>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "Instruction Read-RAF prefetch point disappeared",
                })?
                .lookup_output_point;
            let topology_owner = if fuse_bytecode_carrier {
                Some(session.take::<BytecodeAddressStage1TopologyOwner>().ok_or(
                    KernelError::InvariantViolation {
                        reason: "fused bytecode topology disappeared before prefetch",
                    },
                )?)
            } else {
                None
            };
            let share_registers_val = self
                .config
                .registers_val_evaluation
                .shares_instruction_source(rows);
            let context = Arc::clone(&self.context);
            let config = self.config.instruction_read_raf;
            let worker_point = point.clone();
            let (start_sender, start_receiver) = mpsc::channel();
            let handle = Builder::new()
                .name("jolt-instruction-read-raf-prefetch".to_owned())
                .spawn(move || {
                    start_receiver.recv().map_err(|_| {
                        "Instruction Read-RAF scatter release was dropped".to_owned()
                    })?;
                    let source = owner
                        .lease(rows, context.device_registry_id())
                        .map_err(|error| error.to_string())?;
                    let bytecode_request = topology_owner
                        .map(|topology_owner| {
                            let topology_source = owner
                                .lease(rows, context.device_registry_id())
                                .map_err(|error| error.to_string())?;
                            let topology = topology_owner
                                .lease(topology_source)
                                .map_err(|error| error.to_string())?;
                            BytecodeAddressFusedScatterRequest::new(topology)
                                .map_err(|error| error.to_string())
                        })
                        .transpose()?;
                    let registers_val_lease = share_registers_val
                        .then(|| {
                            let register_source = owner
                                .lease(rows, context.device_registry_id())
                                .map_err(|error| error.to_string())?;
                            RegistersValInstructionSourceLease::new(&context, register_source)
                                .map_err(|error| error.to_string())
                        })
                        .transpose()?;
                    let mut planes = context
                        .prepare_instruction_read_raf_compatibility_scatter(
                            source,
                            &worker_point,
                            InstructionReadRafCompatibilityScatterConfig {
                                threads_per_threadgroup: config
                                    .stage1_scatter_threads_per_threadgroup,
                            },
                            bytecode_request,
                        )
                        .map_err(|error| error.to_string())?;
                    let receipt = planes.receipt().clone();
                    let bytecode_carrier = planes.take_bytecode_carrier();
                    let span = tracing::info_span!(
                        "MetalInstructionReadRaf::address_prefetch",
                        rows,
                        suffix_bits = INITIAL_ADDRESS_SUFFIX_BITS,
                        complete = Empty,
                    );
                    let _entered = span.enter();
                    let mut sequence = context
                        .prepare_address_phase_sequence_from_resident_grouped(
                            planes,
                            config.address_dispatch,
                        )
                        .map_err(|error| error.to_string())?;
                    let initial_sums = sequence
                        .phase(INITIAL_ADDRESS_SUFFIX_BITS, None)
                        .map_err(|error| error.to_string())?;
                    let _ = span.record("complete", true);
                    Ok(PrefetchedInstructionReadRafScatter {
                        sequence: Box::new(sequence),
                        initial_sums,
                        receipt,
                        bytecode_carrier,
                        registers_val_lease,
                    })
                })
                .map_err(metal_prepare_error)?;
            session.park(InstructionReadRafScatterStart {
                sender: start_sender.clone(),
            });
            session.park(PendingInstructionReadRafScatter {
                rows,
                lookup_output_point: point,
                start_sender,
                handle: Some(handle),
            });
            return Ok(());
        }

        if primer_in_flight {
            return Ok(());
        }
        self.submit_source_primer(session, &owner)
    }

    /// Admits the owner Stage 0 deferred before Stage 4 allocates anything,
    /// so Stage 4 can build it under its device rounds.
    fn prefetch_relation(
        &self,
        session: &mut ProofSession,
        relation: &InstructionReadRaf<AkitaField>,
    ) -> Result<(), KernelError<AkitaField>> {
        let dimensions = relation.dimensions();
        let rows = 1usize << dimensions.log_t();
        if self.stage5_owner_geometry(dimensions)
            && session.state::<InstructionReadRafStage1Owner>().is_none()
            && session
                .state::<PendingInstructionReadRafScatter>()
                .is_none()
            && self.admit_stage5_owner(
                rows,
                self.config
                    .registers_val_evaluation
                    .shares_instruction_source(rows),
            )?
        {
            session.park(Stage5OwnerRequest { rows });
        }
        Ok(())
    }

    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, InstructionReadRaf<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = InstructionReadRaf<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        if let Some(primer) = session.take::<PendingInstructionReadRafSourcePrimer>() {
            let span = tracing::info_span!("MetalInstructionReadRaf::source_primer_join");
            let _entered = span.enter();
            primer.join().map_err(metal_prepare_error)?;
        }
        let dimensions = inputs.relation.dimensions();
        let trace_elements = 1usize << dimensions.log_t();
        let _ = session.take::<Stage5OwnerRequest>();
        if let Some(pending) = session.take::<PendingStage5Owner>() {
            let owner = pending.join(trace_elements)?;
            session.park(owner);
        }
        let prefetched_scatter =
            if let Some(pending) = session.take::<PendingInstructionReadRafScatter>() {
                if pending.rows != trace_elements
                    || pending.lookup_output_point != inputs.points.lookup_output
                {
                    return Err(KernelError::InvariantViolation {
                        reason: "Instruction Read-RAF scatter prefetch has stale inputs",
                    });
                }
                let span = tracing::info_span!(
                    "MetalInstructionReadRaf::scatter_prefetch_join",
                    rows = trace_elements,
                );
                let _entered = span.enter();
                Some(pending.join()?)
            } else {
                None
            };
        let use_metal_address =
            trace_elements >= self.config.instruction_read_raf.address_cutoff_elements;
        let fuse_bytecode_carrier = session
            .state::<BytecodeAddressStage1TopologyOwner>()
            .is_some()
            || prefetched_scatter
                .as_ref()
                .is_some_and(|prefetched| prefetched.bytecode_carrier.is_some());
        let retain_lookup_plane = trace_elements
            >= self
                .config
                .instruction_ra_virtualization
                .trace_cutoff_elements;
        let resident_owner_geometry = self.stage5_owner_geometry(dimensions);
        let share_registers_val = self
            .config
            .registers_val_evaluation
            .shares_instruction_source(trace_elements);
        if resident_owner_geometry
            && prefetched_scatter.is_none()
            && session.state::<InstructionReadRafStage1Owner>().is_none()
        {
            self.prepare_stage5_owner(session, witness, trace_elements, share_registers_val)?;
        }
        let stage1_owner = resident_owner_geometry
            .then(|| session.take::<InstructionReadRafStage1Owner>())
            .flatten();
        let share_registers_val_source = prefetched_scatter
            .as_ref()
            .is_some_and(|prefetched| prefetched.registers_val_lease.is_some())
            || (prefetched_scatter.is_none() && share_registers_val && stage1_owner.is_some());
        if prefetched_scatter.is_some() && stage1_owner.is_none() {
            return Err(KernelError::InvariantViolation {
                reason: "prefetched Instruction Read-RAF scatter requires its Stage-1 owner",
            });
        }
        if (fuse_bytecode_carrier || share_registers_val_source) && stage1_owner.is_none() {
            return Err(KernelError::InvariantViolation {
                reason: "fused Stage-1 consumers require the InstructionReadRAF Stage-1 owner",
            });
        }
        let (cpu, resident_grouped_input) = if let Some(owner) = stage1_owner.as_ref() {
            let device_registry_id = self.context.device_registry_id();
            let retained_claims = owner
                .lease(trace_elements, device_registry_id)
                .map_err(metal_prepare_error)?;
            let cpu = OptimizedInstructionReadRafKernel::new_metal_resident(
                dimensions,
                &inputs.points.lookup_output,
                retained_claims,
                inputs.challenges.gamma,
            )?;
            let (resident_input, receipt, bytecode_carrier, registers_val_lease) =
                if let Some(prefetched) = prefetched_scatter {
                    let PrefetchedInstructionReadRafScatter {
                        sequence,
                        initial_sums,
                        receipt,
                        bytecode_carrier,
                        registers_val_lease,
                    } = prefetched;
                    (
                        ResidentGroupedInput::Prefetched {
                            sequence,
                            initial_sums,
                        },
                        receipt,
                        bytecode_carrier,
                        registers_val_lease,
                    )
                } else {
                    let scatter_source = owner
                        .lease(trace_elements, device_registry_id)
                        .map_err(metal_prepare_error)?;
                    let bytecode_request = if fuse_bytecode_carrier {
                        let topology_owner = session
                            .take::<BytecodeAddressStage1TopologyOwner>()
                            .ok_or(KernelError::InvariantViolation {
                            reason: "fused bytecode address carrier topology is missing",
                        })?;
                        let topology_source = owner
                            .lease(trace_elements, device_registry_id)
                            .map_err(metal_prepare_error)?;
                        let topology = topology_owner
                            .lease(topology_source)
                            .map_err(metal_prepare_error)?;
                        Some(
                            BytecodeAddressFusedScatterRequest::new(topology)
                                .map_err(metal_prepare_error)?,
                        )
                    } else {
                        None
                    };
                    let registers_val_lease = if share_registers_val_source {
                        let source = owner
                            .lease(trace_elements, device_registry_id)
                            .map_err(metal_prepare_error)?;
                        Some(
                            RegistersValInstructionSourceLease::new(&self.context, source)
                                .map_err(metal_prepare_error)?,
                        )
                    } else {
                        None
                    };
                    let mut planes = self
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
                            bytecode_request,
                        )
                        .map_err(metal_prepare_error)?;
                    let receipt = planes.receipt().clone();
                    let bytecode_carrier = planes.take_bytecode_carrier();
                    (
                        ResidentGroupedInput::Planes(Box::new(planes)),
                        receipt,
                        bytecode_carrier,
                        registers_val_lease,
                    )
                };
            let fused = receipt.bytecode();
            match (fuse_bytecode_carrier, fused, bytecode_carrier) {
                (true, Some(fused), Some(carrier)) => {
                    validate_fused_bytecode_carrier(receipt.source(), fused, &carrier)?;
                    if session
                        .state::<BytecodeAddressSparseStage1Carrier>()
                        .is_some()
                    {
                        return Err(KernelError::InvariantViolation {
                            reason:
                                "fused bytecode address carrier would replace an existing carrier",
                        });
                    }
                    session.park(carrier);
                }
                (true, _, _) => {
                    return Err(KernelError::InvariantViolation {
                        reason: "fused bytecode address scatter did not publish its carrier",
                    });
                }
                (false, None, None) => {}
                (false, _, _) => {
                    return Err(KernelError::InvariantViolation {
                        reason: "non-fused InstructionReadRAF scatter published bytecode state",
                    });
                }
            }
            match (share_registers_val_source, registers_val_lease) {
                (true, Some(lease)) => {
                    if session
                        .state::<RegistersValInstructionSourceLease>()
                        .is_some()
                    {
                        return Err(KernelError::InvariantViolation {
                            reason:
                                "RegistersVal instruction source would replace an existing lease",
                        });
                    }
                    let source_receipt = lease.receipt();
                    let _span = tracing::info_span!(
                        "MetalRegistersValEvaluation::instruction_source_publish",
                        cycles = source_receipt.cycles(),
                        source = "instruction_read_raf_stage1_rows_v1",
                        row_layout = "column_major_packed_u64_v3",
                        source_generation = source_receipt.generation(),
                        source_device_registry_id = source_receipt.device_registry_id(),
                        source_ready_serial = source_receipt.completion_serial(),
                        instruction_rows_storage_id = source_receipt.instruction_rows_storage_id(),
                        instruction_rows_bytes = source_receipt.instruction_rows_bytes(),
                        producer_plane_allocations = 0usize,
                        producer_device_bytes = 0u64,
                        additional_command_buffers = 0usize,
                        additional_waits = 0usize,
                        additional_dispatches = 0usize,
                        shared_source_row_scans = 1usize,
                        additional_source_row_scans = 0usize,
                        member_upload_bytes = 0u64,
                        complete_publication = true,
                    )
                    .entered();
                    session.park(lease);
                }
                (true, None) => {
                    return Err(KernelError::InvariantViolation {
                        reason: "RegistersVal instruction source did not publish its lease",
                    });
                }
                (false, None) => {}
                (false, Some(_)) => {
                    return Err(KernelError::InvariantViolation {
                        reason: "unrequested RegistersVal instruction source was published",
                    });
                }
            }
            (cpu, Some(resident_input))
        } else {
            (
                prepare_metal_instruction_read_raf(session, witness, inputs, use_metal_address)?,
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
            || trace_elements >= self.config.bytecode_read_raf_address.trace_cutoff_elements
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
                            (rows.len() * super::solinas::BOOLEANITY_SOURCE_ROW_BYTES) as u64;
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
                        resident_row_bytes = super::solinas::BOOLEANITY_SOURCE_ROW_BYTES,
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
            resident_grouped_input,
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
    address_sequence: Option<Box<AddressPhaseSequence>>,
    resident_lookup_plane: Option<ResidentLookupIndexPlane>,
    sequence: Option<Product5Sequence>,
    /// The cycle challenge bound on the fly by the second resident cycle
    /// round, until the handoff binds the next one into product5 tables.
    bound_cycle_challenge: Option<AkitaField>,
    host_tail: Option<[Vec<AkitaField>; PRODUCT5_FACTORS]>,
    metal_rounds: usize,
    metal_address_phases: usize,
}

pub(crate) enum ResidentGroupedInput {
    Planes(Box<InstructionReadRafDenseGroupedPlanes>),
    Prefetched {
        sequence: Box<AddressPhaseSequence>,
        initial_sums: AddressPhaseSums,
    },
}

impl MetalInstructionReadRafKernel {
    pub(crate) fn new(
        cpu: OptimizedInstructionReadRafKernel<AkitaField>,
        context: Arc<SolinasMetal>,
        config: InstructionReadRafMetalConfig,
        use_metal_address: bool,
        retain_lookup_plane: bool,
        resident_grouped_input: Option<ResidentGroupedInput>,
    ) -> Result<Self, SumcheckError<AkitaField>> {
        let mut kernel = Self {
            cpu,
            context,
            config,
            address_sequence: None,
            resident_lookup_plane: None,
            sequence: None,
            bound_cycle_challenge: None,
            host_tail: Some(std::array::from_fn(|_| {
                vec![AkitaField::zero(); config.cutoff_elements]
            })),
            metal_rounds: 0,
            metal_address_phases: 0,
        };
        if use_metal_address {
            if let Some(input) = resident_grouped_input {
                let (mut sequence, initial_sums) = match input {
                    ResidentGroupedInput::Planes(planes) => {
                        let _span = tracing::info_span!(
                            "MetalInstructionReadRaf::stage1_grouped_sequence_prepare"
                        )
                        .entered();
                        (
                            Box::new(
                                kernel
                                    .context
                                    .prepare_address_phase_sequence_from_resident_grouped(
                                        *planes,
                                        config.address_dispatch,
                                    )
                                    .map_err(|error| backend_error(error.to_string()))?,
                            ),
                            None,
                        )
                    }
                    ResidentGroupedInput::Prefetched {
                        sequence,
                        initial_sums,
                    } => (sequence, Some(initial_sums)),
                };
                if retain_lookup_plane {
                    kernel.resident_lookup_plane = Some(sequence.resident_lookup_index_plane());
                }
                let (suffix_len, previous) = kernel.cpu.metal_address_phase_request()?;
                let sums = if let Some(initial_sums) = initial_sums {
                    if suffix_len != INITIAL_ADDRESS_SUFFIX_BITS || previous.is_some() {
                        return Err(backend_error(
                            "prefetched Instruction Read-RAF address phase has stale geometry",
                        ));
                    }
                    initial_sums
                } else {
                    sequence
                        .phase(suffix_len, previous.as_ref())
                        .map_err(|error| backend_error(error.to_string()))?
                };
                kernel.cpu.metal_install_address_phase(sums)?;
                kernel.metal_address_phases = 1;
                kernel.address_sequence = Some(sequence);
                return Ok(kernel);
            }
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
                    tracing::info_span!("MetalInstructionReadRaf::initial_address_phase").entered();
                sequence
                    .phase(suffix_len, previous.as_ref())
                    .map_err(|error| backend_error(error.to_string()))?
            };
            kernel.cpu.metal_install_address_phase(sums)?;
            kernel.metal_address_phases = 1;
            kernel.address_sequence = Some(Box::new(sequence));
        }
        Ok(kernel)
    }

    fn install_next_address_phase(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let (suffix_len, previous) = self.cpu.metal_address_phase_request()?;
        let sequence = self
            .address_sequence
            .as_mut()
            .ok_or_else(|| backend_error("resident address sequence disappeared"))?;
        let sums = sequence
            .phase(suffix_len, previous.as_ref())
            .map_err(|error| backend_error(error.to_string()))?;
        self.cpu.metal_install_address_phase(sums)?;
        self.metal_address_phases += 1;
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

    /// The bound cycle message and the handoff run beside RegistersVal's
    /// native and first dense transitions; the device passes touch disjoint
    /// buffers.
    fn execution_domain(&self) -> RoundExecutionDomain {
        if self.address_sequence.is_some() && !self.cpu.metal_address_active() {
            RoundExecutionDomain::Accelerator
        } else {
            RoundExecutionDomain::Host
        }
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
                self.cpu.metal_bind_address(challenge)?;
                if self.cpu.metal_address_phase_pending() {
                    self.install_next_address_phase()?;
                }
            }
            if self.cpu.metal_address_active() {
                return self.cpu.metal_address_message(previous_claim);
            }
        }

        if self.address_sequence.is_some() && !self.cpu.metal_resident_cycle_available() {
            self.address_sequence = None;
        }

        if self.address_sequence.is_some() {
            if let Some(challenge) = bind.take() {
                let Some(first_challenge) = self.bound_cycle_challenge.take() else {
                    let _span =
                        tracing::info_span!("MetalInstructionReadRaf::resident_bound_message")
                            .entered();
                    let (cpu, address_sequence) = (&mut self.cpu, self.address_sequence.as_mut());
                    let address_sequence = address_sequence
                        .ok_or_else(|| backend_error("resident address sequence disappeared"))?;
                    let poly = cpu.metal_resident_bound_message(
                        challenge,
                        address_sequence,
                        previous_claim,
                    )?;
                    self.bound_cycle_challenge = Some(challenge);
                    self.metal_rounds += 1;
                    return Ok(poly);
                };
                let _span =
                    tracing::info_span!("MetalInstructionReadRaf::resident_handoff").entered();
                let address_sequence = self
                    .address_sequence
                    .take()
                    .ok_or_else(|| backend_error("resident address sequence disappeared"))?;
                let (sequence, q_evals) = self.cpu.metal_offload_resident_bind(
                    first_challenge,
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

fn validate_fused_bytecode_carrier(
    source: InstructionReadRafStage1Receipt,
    fused: InstructionReadRafFusedBytecodeReceipt,
    carrier: &BytecodeAddressSparseStage1Carrier,
) -> Result<(), KernelError<AkitaField>> {
    let receipt = carrier.receipt();
    let topology = carrier
        .fused_topology_receipt()
        .ok_or(KernelError::InvariantViolation {
            reason: "fused bytecode address carrier lost its topology receipt",
        })?;
    let topology_ids = [
        topology.descriptor_allocation_identity(),
        topology.pivot_allocation_identity(),
        topology.chunk_offset_allocation_identity(),
    ];
    let carrier_ids = [
        receipt.occurrence_storage_id(),
        receipt.magnitude_storage_id(),
        receipt.work_item_storage_id(),
        receipt.address_offset_storage_id(),
    ];
    if topology.source_receipt() != source
        || topology.source_generation() != source.source_generation()
        || topology.source_completion_serial() != source.completion_serial()
        || topology.source_rows_storage_id() != source.row_allocation_identity()
        || topology.source_claim_storage_id() != source.claim_allocation_identity()
        || topology.source_windows() != source.rows()
        || topology.physical_rows() != fused.physical_rows()
        || topology.work_items() != fused.work_items()
        || topology.descriptor_elements() != fused.descriptor_elements()
        || topology.descriptor_bytes() != fused.descriptor_bytes()
        || topology.descriptor_allocation_identity() != fused.descriptor_identity()
        || topology.pivot_elements() != fused.pivot_elements()
        || topology.pivot_bytes() != fused.pivot_bytes()
        || topology.pivot_allocation_identity() != fused.pivot_identity()
        || topology.chunk_offset_elements() != fused.chunk_offset_elements()
        || topology.chunk_offset_bytes() != fused.chunk_offset_bytes()
        || topology.chunk_offset_allocation_identity() != fused.chunk_offset_identity()
        || topology.work_item_bytes() != fused.work_item_bytes()
        || topology.work_item_allocation_identity() != fused.work_item_identity()
        || topology.address_offset_elements() != fused.address_offset_elements()
        || topology.address_offset_bytes() != fused.address_offset_bytes()
        || topology.address_offset_allocation_identity() != fused.address_offset_identity()
        || topology.max_descriptors_per_chunk() != fused.max_descriptors_per_chunk()
        || topology.max_pivots_per_chunk() != fused.max_pivots_per_chunk()
        || topology.max_descriptors_per_chunk() > fused.max_admitted_descriptors_per_chunk()
        || topology.max_pivots_per_chunk() > fused.max_admitted_pivots_per_chunk()
        || fused.dynamic_threadgroup_bytes() > fused.threadgroup_memory_limit_bytes()
        || fused.shared_source_row_scans() != 1
        || fused.additional_source_row_scans() != 0
        || fused.member_upload_bytes() != 0
        || receipt.physical_rows() != topology.physical_rows()
        || receipt.work_items() != topology.work_items()
        || receipt.source_generation() != source.source_generation()
        || receipt.source_completion_serial() != source.completion_serial()
        || receipt.source_rows_storage_id() != source.row_allocation_identity()
        || receipt.source_claim_storage_id() != source.claim_allocation_identity()
        || receipt.source_windows() != source.rows()
        || receipt.device_registry_id() != source.device_registry_id()
        || receipt.occurrence_storage_id() != fused.occurrence_identity()
        || receipt.occurrence_bytes() != fused.occurrence_bytes()
        || receipt.magnitude_storage_id() != fused.magnitude_identity()
        || receipt.magnitude_bytes() != fused.magnitude_bytes()
        || receipt.work_item_storage_id() != fused.work_item_identity()
        || receipt.work_item_bytes() != fused.work_item_bytes()
        || receipt.address_offset_storage_id() != fused.address_offset_identity()
        || receipt.address_offset_bytes() != fused.address_offset_bytes()
        || !receipt.complete_overwrite()
        || receipt.covered_rows() != receipt.physical_rows()
        || receipt.additional_source_scans() != 0
        || receipt.member_upload_bytes() != 0
        || !topology.complete_overwrite()
        || topology.covered_rows() != topology.physical_rows()
        || topology.shared_source_row_scans() != 1
        || topology.additional_source_row_scans() != 0
        || topology.member_upload_bytes() != 0
        || topology_ids.contains(&0)
        || carrier_ids.contains(&0)
        || carrier_ids
            .iter()
            .enumerate()
            .any(|(index, identity)| carrier_ids[..index].contains(identity))
        || carrier_ids
            .iter()
            .any(|identity| topology_ids.contains(identity))
    {
        return Err(KernelError::InvariantViolation {
            reason: "fused bytecode address carrier provenance is malformed",
        });
    }
    Ok(())
}

fn backend_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}

fn metal_prepare_error(error: impl ToString) -> KernelError<AkitaField> {
    backend_error(error.to_string()).into()
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Metal Stage-5 owner fixtures")]
mod tests {
    use std::num::NonZeroUsize;

    use jolt_claims::protocols::jolt::geometry::dimensions::{
        TraceDimensions, REGISTER_ADDRESS_BITS,
    };
    use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
    use jolt_claims::protocols::jolt::relations::instruction::{
        InstructionReadRafChallenges, InstructionReadRafInputClaims,
    };
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_claims::NoChallenges;
    use jolt_field::FromPrimitiveInt as _;
    use jolt_lookup_tables::XLEN as RISCV_XLEN;
    use jolt_poly::Polynomial;
    use jolt_verifier::stages::relations::{ConcreteSumcheck, SumcheckOutputClaims};
    use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
    use jolt_verifier::stages::stage5::registers_val_evaluation::{
        RegistersValEvaluation, RegistersValEvaluationInputClaims,
    };
    use jolt_witness::testing::with_sample_backend_at_log_t;
    use jolt_witness::JoltWitnessOracle;

    use super::*;
    use crate::metal::solinas::{MetalError, INSTRUCTION_READ_RAF_SEGMENTS};
    use crate::metal::{
        MetalConfig, RegistersValEvaluationMetalConfig, RegistersValEvaluationSource,
    };
    use crate::optimized::instruction_read_raf::OptimizedInstructionReadRaf;
    use crate::optimized::parity::run_lockstep;
    use crate::optimized::registers_val_evaluation::OptimizedRegistersValEvaluation;
    use crate::optimized::testing::{with_ram_fixture_backend, FixtureShape, RamOp};
    use crate::uniskip::UniskipKernel;

    const LOG_T: usize = 15;

    fn point(len: usize, seed: u64) -> Vec<AkitaField> {
        (0..len as u64)
            .map(|index| AkitaField::from_u64(seed + 37 * index + 5))
            .collect()
    }

    fn owner_contents(
        owner: &InstructionReadRafStage1Owner,
        device_registry_id: u64,
    ) -> (
        Vec<u64>,
        Vec<u8>,
        Vec<[u32; INSTRUCTION_READ_RAF_SEGMENTS]>,
        bool,
    ) {
        let rows = owner.receipt().rows();
        let lease = owner.lease(rows, device_registry_id).unwrap();
        // SAFETY: a sealed owner's row buffer holds its four complete u64
        // columns, and the lease keeps the buffer alive while it is copied.
        let columns = unsafe {
            std::slice::from_raw_parts(lease.row_buffer().contents().cast::<u64>(), 4 * rows)
        }
        .to_vec();
        (
            columns,
            lease.claim_slice().to_vec(),
            lease.counts().to_vec(),
            owner.ram_remap_compatible(),
        )
    }

    #[test]
    fn stage5_owner_matches_the_stage0_owner() {
        // Loads of falling values give rd a negative increment in the first
        // two producer chunks; stores, no-ops and the padded tail cover the
        // RAM and padding rows.
        let falling_loads = [
            RamOp::Write { word: 3, post: 9 },
            RamOp::Read { word: 3 },
            RamOp::Write { word: 4, post: 2 },
            RamOp::Read { word: 4 },
        ];
        let mut ops = falling_loads.to_vec();
        ops.extend(std::iter::repeat_n(RamOp::None, 4_096));
        ops.extend(falling_loads);
        let shape = FixtureShape {
            log_t: LOG_T,
            ram_k: 1 << 8,
        };
        with_ram_fixture_backend(shape, ops, |witness| {
            let backend = MetalBackend::new(MetalConfig {
                instruction_read_raf: InstructionReadRafMetalConfig {
                    address_cutoff_elements: 1 << LOG_T,
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap();
            let mut session = ProofSession::default();
            <MetalBackend as UniskipKernel<AkitaField, OuterRemainder<AkitaField>>>::prepare_witness(
                &backend,
                &mut session,
                LOG_T,
                witness,
            )
            .unwrap();
            let stage0 = session.take::<InstructionReadRafStage1Owner>().unwrap();
            let stage5 = prepare_metal_instruction_read_raf_stage1_owner(
                &backend.context,
                &(witness as &dyn JoltWitnessPlane<AkitaField>)
                    .owned_rows()
                    .unwrap(),
                1 << LOG_T,
            )
            .unwrap();

            let registry_id = backend.context.device_registry_id();
            assert_eq!(stage5.receipt().row_bytes(), stage0.receipt().row_bytes());
            assert_eq!(
                owner_contents(&stage5, registry_id),
                owner_contents(&stage0, registry_id)
            );
        });
    }

    fn stage5_backend() -> MetalBackend {
        MetalBackend::new(MetalConfig {
            instruction_read_raf: InstructionReadRafMetalConfig {
                address_cutoff_elements: 1 << LOG_T,
                cutoff_elements: 8,
                ..Default::default()
            },
            registers_val_evaluation: RegistersValEvaluationMetalConfig {
                source: RegistersValEvaluationSource::Stage1Resident,
                trace_cutoff_elements: 1 << LOG_T,
                cutoff_elements: 16,
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap()
    }

    type Stage5Outputs = (
        SumcheckOutputClaims<AkitaField, InstructionReadRaf<AkitaField>>,
        SumcheckOutputClaims<AkitaField, RegistersValEvaluation<AkitaField>>,
    );

    /// Prepares InstructionReadRAF then RegistersVal through `backend` in
    /// Stage-5 declaration order, proves both in lockstep with the optimized
    /// CPU kernels, and returns whether a RegistersVal lease was published
    /// together with both output claims. `stage4_owner_build` first runs the
    /// Stage-4 admission and release of the owner build.
    fn stage5_lockstep(
        backend: &MetalBackend,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        log_t: usize,
        stage4_owner_build: bool,
    ) -> (bool, Stage5Outputs) {
        let instruction_relation = InstructionReadRaf::new(InstructionReadRafDimensions::new(
            log_t,
            2 * RISCV_XLEN,
            NonZeroUsize::new(4).unwrap(),
        ));
        let reduction_point = point(log_t, 151);
        let opening = |polynomial| {
            let table = JoltWitnessOracle::<AkitaField>::oracle_table(
                witness,
                JoltPolynomialId::Virtual(polynomial),
            )
            .unwrap();
            Polynomial::new(table).evaluate(&reduction_point)
        };
        let instruction_claims = InstructionReadRafInputClaims {
            lookup_output: opening(JoltVirtualPolynomial::LookupOutput),
            left_lookup_operand: opening(JoltVirtualPolynomial::LeftLookupOperand),
            right_lookup_operand: opening(JoltVirtualPolynomial::RightLookupOperand),
        };
        let instruction_points = InstructionReadRafInputClaims {
            lookup_output: reduction_point.clone(),
            left_lookup_operand: reduction_point.clone(),
            right_lookup_operand: reduction_point.clone(),
        };
        let instruction_challenges = InstructionReadRafChallenges {
            gamma: AkitaField::from_u64(167),
        };
        let instruction_inputs = || ProverInputs {
            relation: &instruction_relation,
            claims: &instruction_claims,
            points: &instruction_points,
            challenges: &instruction_challenges,
        };

        let registers_relation = RegistersValEvaluation::new(TraceDimensions::new(log_t));
        let registers_point = point(REGISTER_ADDRESS_BITS + log_t, 19);
        let registers_table = JoltWitnessOracle::<AkitaField>::oracle_table(
            witness,
            JoltPolynomialId::Virtual(JoltVirtualPolynomial::RegistersVal),
        )
        .unwrap();
        let registers_claims = RegistersValEvaluationInputClaims {
            registers_val: Polynomial::new(registers_table).evaluate(&registers_point),
        };
        let registers_points = RegistersValEvaluationInputClaims {
            registers_val: registers_point,
        };
        let no_challenges = NoChallenges::default();
        let registers_inputs = || ProverInputs {
            relation: &registers_relation,
            claims: &registers_claims,
            points: &registers_points,
            challenges: &no_challenges,
        };

        let mut session = ProofSession::default();
        if stage4_owner_build {
            <MetalBackend as PrepareKernel<AkitaField, InstructionReadRaf<AkitaField>>>::prefetch_relation(
                backend,
                &mut session,
                &instruction_relation,
            )
            .unwrap();
            backend
                .start_stage5_owner_build(&mut session, witness)
                .unwrap();
            assert!(session.state::<PendingStage5Owner>().is_some());
        }
        let mut instruction_actual = <MetalBackend as PrepareKernel<
            AkitaField,
            InstructionReadRaf<AkitaField>,
        >>::prepare(
            backend, &mut session, witness, instruction_inputs()
        )
        .unwrap();
        assert!(session.state::<PendingStage5Owner>().is_none());
        assert!(session.state::<InstructionReadRafStage1Owner>().is_none());
        let leased = session
            .state::<RegistersValInstructionSourceLease>()
            .is_some();
        let mut registers_actual = <MetalBackend as PrepareKernel<
            AkitaField,
            RegistersValEvaluation<AkitaField>,
        >>::prepare(
            backend, &mut session, witness, registers_inputs()
        )
        .unwrap();
        assert!(session
            .state::<RegistersValInstructionSourceLease>()
            .is_none());

        let mut instruction_expected = OptimizedInstructionReadRaf
            .prepare(&mut ProofSession::default(), witness, instruction_inputs())
            .unwrap();
        let claim = instruction_relation
            .input_claim(&instruction_claims, &instruction_challenges)
            .unwrap();
        let challenges = point(instruction_expected.num_rounds(), 211);
        run_lockstep(
            instruction_expected.as_mut(),
            instruction_actual.as_mut(),
            claim,
            &challenges,
        );
        let instruction_outputs = instruction_actual
            .output_claims(&instruction_claims)
            .unwrap();
        assert_eq!(
            instruction_outputs,
            instruction_expected
                .output_claims(&instruction_claims)
                .unwrap()
        );

        let mut registers_expected = OptimizedRegistersValEvaluation
            .prepare(&mut ProofSession::default(), witness, registers_inputs())
            .unwrap();
        let challenges = point(log_t, 223);
        run_lockstep(
            registers_expected.as_mut(),
            registers_actual.as_mut(),
            registers_claims.registers_val,
            &challenges,
        );
        let registers_outputs = registers_actual.output_claims(&registers_claims).unwrap();
        assert_eq!(
            registers_outputs,
            registers_expected.output_claims(&registers_claims).unwrap()
        );
        (leased, (instruction_outputs, registers_outputs))
    }

    #[test]
    fn stage5_owner_is_leased_to_registers_val_before_its_prepare() {
        with_sample_backend_at_log_t(LOG_T, 8, |witness| {
            let backend = stage5_backend();
            let (leased, _) = stage5_lockstep(&backend, witness, LOG_T, false);
            assert!(leased);
            assert_eq!(backend.registers_val_sequences(), 1);
        });
    }

    #[test]
    fn stage4_built_owner_is_joined_and_leased_by_stage5() {
        with_sample_backend_at_log_t(LOG_T, 8, |witness| {
            let backend = stage5_backend();
            let (leased, _) = stage5_lockstep(&backend, witness, LOG_T, true);
            assert!(leased);
            assert_eq!(backend.registers_val_sequences(), 1);
        });
    }

    #[test]
    fn declined_stage5_owner_keeps_both_cpu_routes() {
        with_sample_backend_at_log_t(LOG_T, 8, |witness| {
            let (_, admitted) = stage5_lockstep(&stage5_backend(), witness, LOG_T, false);
            let mut backend = stage5_backend();
            backend.working_set_limit = Some(0);
            let (leased, declined) = stage5_lockstep(&backend, witness, LOG_T, false);
            assert!(!leased);
            assert_eq!(backend.registers_val_sequences(), 0);
            assert_eq!(declined, admitted);
        });
    }

    fn resident_cycle_backend(address_cutoff_elements: usize) -> Result<MetalBackend, MetalError> {
        MetalBackend::new(MetalConfig {
            instruction_read_raf: InstructionReadRafMetalConfig {
                address_cutoff_elements,
                cutoff_elements: 2,
                ..Default::default()
            },
            ..Default::default()
        })
    }

    #[test]
    fn resident_cycle_route_starts_at_eight_rows() {
        assert!(matches!(
            resident_cycle_backend(4),
            Err(MetalError::InvalidHybridCutoff(4))
        ));
        let backend = resident_cycle_backend(8).unwrap();
        with_sample_backend_at_log_t(3, 8, |witness| {
            let _ = stage5_lockstep(&backend, witness, 3, false);
        });
    }

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
