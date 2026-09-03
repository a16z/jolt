use std::{
    mem::size_of,
    sync::Arc,
    thread::JoinHandle,
    time::{Duration, Instant},
};

use jolt_claims::protocols::jolt::{
    JoltDerivedId, JoltPolynomialId, JoltVirtualPolynomial, RamRafEvaluationPublic,
};
use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::ram_raf_evaluation::{
    RamRafEvaluation, RamRafEvaluationOutputClaims,
};
use jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteChecking;
use jolt_witness::JoltWitnessPlane;

use super::backend::MetalBackend;
use super::ram_cycle_family::shared_ram_cycle_family_owner;
use super::solinas::{
    MetalError, PendingRamRafSequence, RamRafAddressPlane, RamRafAffineTail, RamRafConfig,
    RamRafSegmentedAddressPlane, RamRafTailOutput, RAM_RAF_ADDRESS_DOMAIN,
};
use crate::metal::ram_records::RamAccessColumns;
use crate::optimized::OptimizedBackend;
use crate::ram_access::RamAccessTape;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[cfg(feature = "test-utils")]
mod evaluation;
#[cfg(feature = "test-utils")]
pub use evaluation::{
    RamRafEvaluationCpuEvalSample, RamRafEvaluationCpuMetalEvalFixture, RamRafEvaluationEvalError,
    RamRafEvaluationEvalResult, RamRafEvaluationMetalEvalSample, RamRafEvaluationRoundTiming,
    RamRafEvaluationShapeSnapshot,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRafEvaluationMetalConfig {
    pub dispatch: RamRafConfig,
    pub cpu_prefetch_cutoff_elements: usize,
}

impl Default for RamRafEvaluationMetalConfig {
    fn default() -> Self {
        Self {
            dispatch: RamRafConfig::default(),
            cpu_prefetch_cutoff_elements: 1 << 28,
        }
    }
}

struct MetalRamRafEvaluationKernel {
    pending: Option<PendingRamRafSequence>,
    pending_cpu: Option<PendingRamRafCpuPrefetch>,
    tail: Option<RamRafAffineTail<AkitaField>>,
    output: Option<RamRafTailOutput<AkitaField>>,
    lowest_address: u64,
    rounds: usize,
    next_round: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalRamRafEvaluationKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Some(pending) = &self.pending {
            visitor.visit_field(allocative::Key::new("pending"), pending);
        }
        if let Some(pending) = &self.pending_cpu {
            visitor.visit_field(allocative::Key::new("pending_cpu"), pending);
        }
        if let Some(tail) = &self.tail {
            visitor.visit_simple(allocative::Key::new("host_tail"), tail.heap_bytes());
        }
        visitor.exit();
    }
}

struct RamRafCpuPrefetchOutput {
    masses: Result<Vec<AkitaField>, String>,
    worker_wall: Duration,
}

pub(super) struct PendingRamRafCpuPrefetch {
    rows: usize,
    addresses: usize,
    accesses: usize,
    source_bytes: usize,
    source_storage_id: usize,
    started: Instant,
    handle: Option<JoinHandle<RamRafCpuPrefetchOutput>>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for PendingRamRafCpuPrefetch {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.enter_self_sized::<Self>().exit();
    }
}

impl Drop for PendingRamRafCpuPrefetch {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl PendingRamRafCpuPrefetch {
    fn join(mut self) -> Result<RamRafCpuPrefetchOutput, SumcheckError<AkitaField>> {
        let handle = self
            .handle
            .take()
            .ok_or_else(|| metal_error("segmented CPU RAF prefetch was already consumed"))?;
        handle
            .join()
            .map_err(|_| metal_error("segmented CPU RAF prefetch worker panicked"))
    }
}

impl MetalBackend {
    pub(super) fn start_ram_raf_cpu_prefetch(
        session: &mut ProofSession,
        source: RamRafSegmentedAddressPlane,
        cycle_point: &[AkitaField],
    ) -> Result<(), KernelError<AkitaField>> {
        if session.state::<PendingRamRafCpuPrefetch>().is_some()
            || session.state::<PendingRamRafSequence>().is_some()
        {
            return Err(KernelError::InvariantViolation {
                reason: "RAM RAF pushforward prefetch was submitted twice",
            });
        }
        if cycle_point.len() != source.rows().ilog2() as usize {
            return Err(KernelError::InvariantViolation {
                reason: "segmented CPU RAF prefetch has the wrong cycle point",
            });
        }
        let rows = source.rows();
        let addresses = source.addresses();
        let accesses = source.accesses();
        let source_bytes = source.borrowed_bytes();
        let source_storage_id = source.storage_id();
        let cycle_point = cycle_point.to_vec();
        let started = Instant::now();
        let handle = std::thread::Builder::new()
            .name("jolt-ram-raf-cpu-prefetch".to_owned())
            .spawn(move || {
                let worker_started = Instant::now();
                let masses = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    source.cpu_pushforward(&cycle_point)
                }))
                .map_err(|_| "segmented CPU RAF pushforward panicked".to_owned())
                .and_then(|masses| masses.map_err(|error| error.to_string()));
                RamRafCpuPrefetchOutput {
                    masses,
                    worker_wall: worker_started.elapsed(),
                }
            })
            .map_err(|_| KernelError::InvariantViolation {
                reason: "segmented CPU RAF prefetch worker could not start",
            })?;
        tracing::info!(
            target: "jolt::metal",
            rows,
            addresses,
            accesses,
            source_bytes,
            source_storage_id,
            "started segmented CPU RAM RAF prefetch"
        );
        session.park(PendingRamRafCpuPrefetch {
            rows,
            addresses,
            accesses,
            source_bytes,
            source_storage_id,
            started,
            handle: Some(handle),
        });
        Ok(())
    }

    pub(super) fn ram_raf_witness_requested(
        &self,
        log_t: usize,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<bool, KernelError<AkitaField>> {
        let cycles = 1usize << log_t;
        if cycles < self.config.ram_raf_evaluation.dispatch.trace_cutoff {
            return Ok(false);
        }
        let ram_ra_shape =
            witness.shape(JoltPolynomialId::Virtual(JoltVirtualPolynomial::RamRa))?;
        Ok(ram_ra_shape.log_rows == log_t + RAM_RAF_ADDRESS_DOMAIN.ilog2() as usize)
    }

    pub(super) fn prepare_ram_raf_witness(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<(), KernelError<AkitaField>> {
        let config = self.config.ram_raf_evaluation.dispatch;
        let cycles = 1usize << log_t;
        if !self.ram_raf_witness_requested(log_t, witness)? {
            return Ok(());
        }
        let log_k = RAM_RAF_ADDRESS_DOMAIN.ilog2() as usize;
        let source_collection_performed = session.state::<Arc<RamAccessColumns>>().is_none();
        let witness_span = tracing::info_span!(
            "MetalRamCycleFamily::witness_prepare",
            schema_version = super::solinas::ram_cycle_family::RAM_CYCLE_FAMILY_SCHEMA_VERSION,
            requested = tracing::field::Empty,
            selected = tracing::field::Empty,
            fallback_reason = tracing::field::Empty,
            log_t,
            log_k,
            cycles,
            address_domain = RAM_RAF_ADDRESS_DOMAIN,
            source_generation = tracing::field::Empty,
            source_fingerprint = tracing::field::Empty,
            source_collection_performed,
            witness_source_scans = usize::from(source_collection_performed),
            additional_witness_source_scans = 0,
            address_validation_passes = tracing::field::Empty,
            address_rows = cycles,
            address_plane_storage_id = tracing::field::Empty,
            address_plane_device_registry_id = tracing::field::Empty,
            address_plane_bytes = tracing::field::Empty,
            address_plane_upload_bytes = tracing::field::Empty,
            address_plane_allocations = tracing::field::Empty,
            owner_published = tracing::field::Empty,
            address_plane_published = tracing::field::Empty,
            complete_publication = tracing::field::Empty,
        );
        let _witness_guard = witness_span.enter();
        let columns = RamAccessColumns::shared(session, witness, log_t)?;
        let (access_count, increment_compatible, ram_ra_compatible, hamming_exact) = {
            let tape = session
                .state::<RamAccessTape>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "RAM access collection did not publish its sparse tape",
                })?;
            (
                tape.access_count(),
                tape.increment_compatible(),
                tape.ram_ra_compatible(),
                tape.hamming_exact(),
            )
        };
        let read_write_config = self.config.ram_read_write;
        let high_activity = cycles >= read_write_config.trace_cutoff_elements
            && access_count >= read_write_config.minimum_accesses
            && increment_compatible
            && ram_ra_compatible
            && hamming_exact;
        let requested = if high_activity {
            "metal_address_segmented_v1"
        } else {
            "retained_address_segmented_v1"
        };
        let _ = witness_span.record("requested", requested);
        if high_activity {
            let _ = witness_span.record("selected", "metal_address_segmented_v1");
            let _ = witness_span.record("fallback_reason", "none");
            let _ = witness_span.record("owner_published", false);
            let _ = witness_span.record("address_plane_bytes", 0);
            let _ = witness_span.record("address_plane_upload_bytes", 0);
            let _ = witness_span.record("address_plane_allocations", 0);
            let _ = witness_span.record("address_validation_passes", 0);
            let _ = witness_span.record("address_plane_published", false);
            let _ = witness_span.record("complete_publication", false);
            return Ok(());
        }
        let owner = shared_ram_cycle_family_owner(session, witness, log_t, log_k)?;
        if let Some(owner) = &owner {
            let _ = witness_span.record("source_generation", owner.receipt().source_generation());
            let _ = witness_span.record("source_fingerprint", owner.receipt().fingerprint());
            let _ = witness_span.record("owner_published", true);
        } else {
            let _ = witness_span.record("owner_published", false);
        }
        if let Some(owner) = &owner {
            let records = owner.access_records();
            if !records.is_empty() {
                let cycle_ids = records
                    .iter()
                    .map(|record| record.cycle())
                    .collect::<Vec<_>>();
                let addresses = records
                    .iter()
                    .map(|record| record.address())
                    .collect::<Vec<_>>();
                let plane = match self.context.prepare_ram_raf_segmented_accesses(
                    cycles,
                    RAM_RAF_ADDRESS_DOMAIN,
                    &cycle_ids,
                    &addresses,
                ) {
                    Ok(plane) => plane,
                    Err(error) if error.is_capacity_error() => return Ok(()),
                    Err(error) => return Err(metal_prepare_error(error)),
                };
                let _ = witness_span.record("selected", "retained_address_segmented_v1");
                let _ = witness_span.record("fallback_reason", "none");
                let _ = witness_span.record("address_plane_storage_id", plane.storage_id());
                let _ = witness_span.record(
                    "address_plane_device_registry_id",
                    plane.device_registry_id(),
                );
                let _ = witness_span.record("address_plane_bytes", plane.borrowed_bytes());
                let _ = witness_span.record("address_plane_upload_bytes", plane.borrowed_bytes());
                let _ = witness_span.record("address_plane_allocations", 5);
                let _ = witness_span.record("address_validation_passes", 0);
                let _ = witness_span.record("address_plane_published", true);
                let _ = witness_span.record("complete_publication", true);
                session.park(plane);
                return Ok(());
            }
        }
        if let Some(plane) = session.state::<RamRafAddressPlane>() {
            let _ = witness_span.record("selected", requested);
            let _ = witness_span.record("fallback_reason", "none");
            let _ = witness_span.record("address_plane_storage_id", plane.storage_id());
            let _ = witness_span.record(
                "address_plane_device_registry_id",
                plane.device_registry_id(),
            );
            let _ = witness_span.record("address_plane_bytes", plane.resident_bytes());
            let _ = witness_span.record("address_plane_upload_bytes", 0);
            let _ = witness_span.record("address_plane_allocations", 0);
            let _ = witness_span.record("address_validation_passes", 0);
            let _ = witness_span.record("address_plane_published", true);
            let _ = witness_span.record("complete_publication", owner.is_some());
            return Ok(());
        }
        let addresses = columns.validated_addresses::<AkitaField>(RAM_RAF_ADDRESS_DOMAIN)?;
        tracing::info!(
            target: "jolt::metal",
            access_count,
            increment_compatible,
            ram_ra_compatible,
            "prepared resident RAM address source"
        );
        let plane = match self
            .context
            .prepare_ram_raf_certified_addresses(addresses, config)
        {
            Ok(plane) => plane,
            Err(error) if error.is_capacity_error() => {
                tracing::warn!(
                    target: "jolt::metal",
                    error = %error,
                    cycles,
                    "Metal RAM address plane was not admitted"
                );
                return Ok(());
            }
            Err(error) => return Err(metal_prepare_error(error)),
        };
        let _ = witness_span.record("selected", requested);
        let _ = witness_span.record("fallback_reason", "none");
        let _ = witness_span.record("address_plane_storage_id", plane.storage_id());
        let _ = witness_span.record(
            "address_plane_device_registry_id",
            plane.device_registry_id(),
        );
        let _ = witness_span.record("address_plane_bytes", plane.resident_bytes());
        let _ = witness_span.record("address_plane_upload_bytes", plane.resident_bytes());
        let _ = witness_span.record("address_plane_allocations", 1);
        let _ = witness_span.record("address_validation_passes", 0);
        let _ = witness_span.record("address_plane_published", true);
        let _ = witness_span.record("complete_publication", owner.is_some());
        session.park(plane);
        Ok(())
    }

    pub(super) fn submit_ram_raf(
        &self,
        session: &mut ProofSession,
        relation: &RamReadWriteChecking<AkitaField>,
    ) -> Result<(), KernelError<AkitaField>> {
        let dimensions = relation.dimensions();
        let cycles = 1usize << dimensions.log_t();
        let config = self.config.ram_raf_evaluation.dispatch;
        if cycles < config.trace_cutoff
            || dimensions.raf_evaluation_rounds() != relation.ram_log_k()
        {
            return Ok(());
        }
        if session.state::<PendingRamRafCpuPrefetch>().is_some() {
            return Ok(());
        }
        if session.state::<PendingRamRafSequence>().is_some() {
            return Err(KernelError::InvariantViolation {
                reason: "RAM RAF pushforward was submitted twice",
            });
        }
        if let Some(source) = session.state::<RamRafSegmentedAddressPlane>().cloned() {
            return self.submit_segmented_ram_raf(session, relation, source);
        }
        if relation.ram_log_k() != RAM_RAF_ADDRESS_DOMAIN.ilog2() as usize {
            return Ok(());
        }
        let Some(addresses) = session.state::<RamRafAddressPlane>().cloned() else {
            return Ok(());
        };
        if addresses.rows() != cycles || addresses.address_domain() != RAM_RAF_ADDRESS_DOMAIN {
            return Err(KernelError::InvariantViolation {
                reason: "resident Metal RAM address plane has the wrong geometry",
            });
        }
        let sequence = match self.context.prepare_ram_raf_sequence(
            addresses,
            relation.product_tau_low(),
            config,
        ) {
            Ok(sequence) => sequence,
            Err(error) if error.is_capacity_error() => return Ok(()),
            Err(error) => return Err(metal_prepare_error(error)),
        };
        let address_storage_id = sequence.address_storage_id();
        let pending = {
            let _span = tracing::info_span!(
                "MetalRamRafEvaluation::submit",
                cycles,
                resident_address_bytes = cycles * size_of::<u32>(),
                address_storage_id,
            )
            .entered();
            sequence.submit()
        };
        session.park(pending);
        Ok(())
    }

    pub(super) fn submit_segmented_ram_raf(
        &self,
        session: &mut ProofSession,
        relation: &RamReadWriteChecking<AkitaField>,
        source: RamRafSegmentedAddressPlane,
    ) -> Result<(), KernelError<AkitaField>> {
        let dimensions = relation.dimensions();
        let cycles = 1usize << dimensions.log_t();
        if source.rows() != cycles
            || source.addresses() != 1usize << relation.ram_log_k()
            || dimensions.raf_evaluation_rounds() != relation.ram_log_k()
        {
            return Err(KernelError::InvariantViolation {
                reason: "segmented RAM RAF source has the wrong geometry",
            });
        }
        if session.state::<PendingRamRafSequence>().is_some() {
            return Err(KernelError::InvariantViolation {
                reason: "segmented RAM RAF pushforward was submitted twice",
            });
        }
        let source_already_parked = match session.state::<RamRafSegmentedAddressPlane>() {
            Some(existing)
                if existing.storage_id() == source.storage_id()
                    && existing.rows() == source.rows()
                    && existing.addresses() == source.addresses() =>
            {
                true
            }
            Some(_) => {
                return Err(KernelError::InvariantViolation {
                    reason: "segmented RAM RAF source has stale resident provenance",
                });
            }
            None => false,
        };
        let sequence = match self
            .context
            .prepare_ram_raf_segmented_sequence(source.clone(), relation.product_tau_low())
        {
            Ok(sequence) => sequence,
            Err(MetalError::WorkingSetTooLarge {
                current,
                additional,
                maximum,
            }) => {
                tracing::warn!(
                    target: "jolt::metal",
                    current_allocated_bytes = current,
                    additional_bytes = additional,
                    recommended_max_bytes = maximum,
                    overage_bytes = current.saturating_add(additional).saturating_sub(maximum),
                    "segmented RAM RAF route exceeded the recommended Metal working set"
                );
                return Ok(());
            }
            Err(error) if error.is_capacity_error() => {
                tracing::warn!(
                    target: "jolt::metal",
                    %error,
                    "segmented RAM RAF route was not admitted"
                );
                return Ok(());
            }
            Err(error) => return Err(metal_prepare_error(error)),
        };
        let address_storage_id = sequence.address_storage_id();
        let pending = {
            let _span = tracing::info_span!(
                "MetalRamRafEvaluation::submit_segmented",
                cycles,
                addresses = source.addresses(),
                accesses = source.accesses(),
                borrowed_bytes = source.borrowed_bytes(),
                address_storage_id,
            )
            .entered();
            sequence.submit()
        };
        if !source_already_parked {
            session.park(source);
        }
        session.park(pending);
        Ok(())
    }
}

impl PrepareKernel<AkitaField, RamRafEvaluation<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, RamRafEvaluation<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = RamRafEvaluation<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let relation = inputs.relation;
        let log_t = relation.read_write_dimensions().log_t();
        let cycles = 1usize << log_t;
        let config = self.config.ram_raf_evaluation.dispatch;
        if cycles < config.trace_cutoff
            || relation.read_write_dimensions().raf_evaluation_rounds() != relation.ram_log_k()
        {
            return OptimizedBackend.prepare(session, witness, inputs);
        }
        let address_domain = 1usize << relation.ram_log_k();
        if let Some(pending_cpu) = session.take::<PendingRamRafCpuPrefetch>() {
            let address_storage_id = session
                .state::<RamRafSegmentedAddressPlane>()
                .map(RamRafSegmentedAddressPlane::storage_id);
            if pending_cpu.rows != cycles
                || pending_cpu.addresses != address_domain
                || Some(pending_cpu.source_storage_id) != address_storage_id
            {
                return Err(KernelError::InvariantViolation {
                    reason: "pending segmented CPU RAM RAF has stale resident provenance",
                });
            }
            let _ = session.take::<RamRafSegmentedAddressPlane>();
            return Ok(Box::new(MetalRamRafEvaluationKernel {
                pending: None,
                pending_cpu: Some(pending_cpu),
                tail: None,
                output: None,
                lowest_address: relation.lowest_address(),
                rounds: relation.ram_log_k(),
                next_round: 0,
            }));
        }
        let Some(pending) = session.take::<PendingRamRafSequence>() else {
            return OptimizedBackend.prepare(session, witness, inputs);
        };
        let address_storage_id = if pending.is_segmented() {
            session
                .state::<RamRafSegmentedAddressPlane>()
                .map(RamRafSegmentedAddressPlane::storage_id)
        } else {
            session
                .state::<RamRafAddressPlane>()
                .map(RamRafAddressPlane::storage_id)
        };
        if pending.rows() != Some(cycles)
            || pending.address_domain() != Some(address_domain)
            || pending.address_storage_id() != address_storage_id
        {
            return Err(KernelError::InvariantViolation {
                reason: "pending RAM RAF pushforward has stale resident provenance",
            });
        }
        if pending.is_segmented() {
            let _ = session.take::<RamRafSegmentedAddressPlane>();
        }
        Ok(Box::new(MetalRamRafEvaluationKernel {
            pending: Some(pending),
            pending_cpu: None,
            tail: None,
            output: None,
            lowest_address: relation.lowest_address(),
            rounds: relation.ram_log_k(),
            next_round: 0,
        }))
    }
}

impl MetalRamRafEvaluationKernel {
    fn join_pushforward(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        if self.tail.is_some() {
            return Ok(());
        }
        let masses = if let Some(pending_cpu) = self.pending_cpu.take() {
            let source_bytes = pending_cpu.source_bytes;
            let accesses = pending_cpu.accesses;
            let total_started = pending_cpu.started;
            let join_started = Instant::now();
            let output = pending_cpu.join()?;
            let join_wall = join_started.elapsed();
            let total_wall = total_started.elapsed();
            let span = tracing::info_span!(
                "MetalRamRafEvaluation::cpu_prefetch_join",
                accesses,
                source_bytes,
                output_bytes = output
                    .masses
                    .as_ref()
                    .map_or(0, |masses| masses.len() * size_of::<AkitaField>()),
                worker_wall_ns = u64::try_from(output.worker_wall.as_nanos()).unwrap_or(u64::MAX),
                join_wall_ns = u64::try_from(join_wall.as_nanos()).unwrap_or(u64::MAX),
                total_wall_ns = u64::try_from(total_wall.as_nanos()).unwrap_or(u64::MAX),
                metal_commands = 0u64,
            );
            let _entered = span.enter();
            output
                .masses
                .map_err(|message| SumcheckError::ComputeBackend {
                    backend: "cpu",
                    message,
                })?
        } else {
            let pending = self
                .pending
                .take()
                .ok_or_else(|| metal_error("RAM RAF pushforward is missing"))?;
            let span = tracing::info_span!(
                "MetalRamRafEvaluation::join",
                gpu_active_ns = tracing::field::Empty,
            );
            let _entered = span.enter();
            let observation = pending.join().map_err(metal_tail_error)?;
            let _ = span.record(
                "gpu_active_ns",
                u64::try_from(observation.gpu_active.as_nanos()).unwrap_or(u64::MAX),
            );
            observation.masses
        };
        let tail = RamRafAffineTail::new(masses, self.lowest_address).map_err(metal_tail_error)?;
        if tail.remaining_rounds() != self.rounds {
            return Err(metal_error("Metal RAM RAF tail has the wrong round count"));
        }
        self.tail = Some(tail);
        Ok(())
    }
}

impl ProveRounds<AkitaField> for MetalRamRafEvaluationKernel {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if round != self.next_round || round >= self.rounds || (round == 0) != bind.is_none() {
            return Err(metal_error("RAM RAF received an out-of-order round"));
        }
        self.join_pushforward()?;
        let tail = self
            .tail
            .as_mut()
            .ok_or_else(|| metal_error("RAM RAF round requested after finish"))?;
        if let Some(challenge) = bind {
            tail.bind(challenge).map_err(metal_tail_error)?;
        }
        let coefficients = tail
            .message(previous_claim)
            .map_err(metal_tail_error)?
            .coefficients();
        self.next_round += 1;
        Ok(UnivariatePoly::new(coefficients.to_vec()))
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.next_round != self.rounds || self.output.is_some() {
            return Err(metal_error("RAM RAF reached finish in an invalid state"));
        }
        let mut tail = self
            .tail
            .take()
            .ok_or_else(|| metal_error("RAM RAF finish has no host tail"))?;
        tail.bind(bind).map_err(metal_tail_error)?;
        self.output = Some(tail.output().map_err(metal_tail_error)?);
        Ok(())
    }
}

impl SumcheckKernel<AkitaField> for MetalRamRafEvaluationKernel {
    type Relation = RamRafEvaluation<AkitaField>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<RamRafEvaluationOutputClaims<AkitaField>, SumcheckKernelError<AkitaField>> {
        let output = self.output.ok_or(SumcheckKernelError::NotFullyBound {
            remaining: self.rounds.saturating_sub(self.next_round),
        })?;
        Ok(RamRafEvaluationOutputClaims {
            ram_ra: output.ram_ra,
        })
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        let output = self.output.ok_or(SumcheckKernelError::NotFullyBound {
            remaining: self.rounds.saturating_sub(self.next_round),
        })?;
        let id = JoltDerivedId::from(RamRafEvaluationPublic::UnmapAddress);
        let expected = relation.derive_output_term(&id, input_points, output_points, challenges)?;
        if output.unmap_address != expected {
            return Err(SumcheckKernelError::DerivedTableDrift {
                id,
                expected,
                got: output.unmap_address,
            });
        }
        Ok(())
    }
}

fn metal_prepare_error(error: MetalError) -> KernelError<AkitaField> {
    metal_error(error.to_string()).into()
}

fn metal_tail_error(error: impl ToString) -> SumcheckError<AkitaField> {
    metal_error(error.to_string())
}

fn metal_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Metal parity test setup")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::ReadWriteDimensions;
    use jolt_claims::protocols::jolt::geometry::ram::RamRafEvaluationDimensions;
    use jolt_claims::protocols::jolt::relations::ram::{
        RamRafEvaluationInputClaims, RamReadWriteChallenges, RamReadWriteInputClaims,
    };
    use jolt_claims::NoChallenges;
    use jolt_field::{Ring as _, Zero as _};
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;

    use super::*;
    use crate::metal::MetalConfig;
    use crate::optimized::parity::{probe_input_claim, run_lockstep};
    use crate::optimized::testing::{
        fixture_lowest_address, with_ram_fixture_backend, FixtureShape, RamOp,
    };
    use crate::uniskip::UniskipKernel;

    fn point(seed: u64, len: usize) -> Vec<AkitaField> {
        (0..len as u64)
            .map(|index| AkitaField::from_u64(seed + 37 * index + 5))
            .collect()
    }

    #[test]
    fn prepared_retained_kernel_matches_optimized_cpu() {
        let shape = FixtureShape {
            log_t: 15,
            ram_k: RAM_RAF_ADDRESS_DOMAIN,
        };
        let mut ops = Vec::with_capacity(1 << 14);
        for (word, count) in [(3, 128), (5, 129), (7, 4096), (9, 4097)] {
            ops.push(RamOp::Write {
                word,
                post: word + 17,
            });
            ops.extend(std::iter::repeat_n(RamOp::Read { word }, count - 1));
        }
        ops.resize(1 << 14, RamOp::None);
        with_ram_fixture_backend(shape, ops, |witness| {
            let read_write_dimensions =
                ReadWriteDimensions::new(shape.log_t, shape.log_k(), shape.log_t, shape.log_k());
            let tau_low = point(83, shape.log_t);
            let relation = RamRafEvaluation::new(
                read_write_dimensions,
                RamRafEvaluationDimensions::try_from(read_write_dimensions).unwrap(),
                shape.log_k(),
                fixture_lowest_address(),
                tau_low.clone(),
            );
            let claims = RamRafEvaluationInputClaims {
                ram_address: AkitaField::zero(),
            };
            let points = RamRafEvaluationInputClaims::<Vec<AkitaField>>::default();
            let challenges = NoChallenges::default();
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

            let mut expected = OptimizedBackend
                .prepare(&mut ProofSession::default(), witness, inputs())
                .unwrap();
            let mut config = MetalConfig::default();
            config.ram_raf_evaluation.dispatch.trace_cutoff = 1 << shape.log_t;
            let metal = MetalBackend::new(config).unwrap();
            let mut session = ProofSession::default();
            <MetalBackend as UniskipKernel<AkitaField, OuterRemainder<AkitaField>>>::prepare_witness(
                &metal,
                &mut session,
                shape.log_t,
                witness,
            )
            .unwrap();
            assert!(session.state::<RamRafAddressPlane>().is_none());
            let source = session.state::<RamRafSegmentedAddressPlane>().unwrap();
            assert_eq!(source.addresses(), shape.ram_k);
            assert!(source.bounded_address_count() >= 2);
            assert!(source.hot_address_count() >= 1);
            assert!(source.hot_message_chunk_count() >= 2);
            let cpu_masses = source.cpu_pushforward(&tau_low).unwrap();
            let gpu_masses = metal
                .context
                .prepare_ram_raf_segmented_sequence(source.clone(), &tau_low)
                .unwrap()
                .execute_timed()
                .unwrap()
                .masses;
            assert_eq!(cpu_masses, gpu_masses);
            let mut cpu_session = ProofSession::default();
            cpu_session.park(source.clone());
            MetalBackend::start_ram_raf_cpu_prefetch(&mut cpu_session, source.clone(), &tau_low)
                .unwrap();
            let mut cpu_actual =
                PrepareKernel::prepare(&metal, &mut cpu_session, witness, inputs()).unwrap();
            let mut cpu_expected = OptimizedBackend
                .prepare(&mut ProofSession::default(), witness, inputs())
                .unwrap();
            let cpu_input_claim = probe_input_claim(cpu_expected.as_mut());
            run_lockstep(
                cpu_expected.as_mut(),
                cpu_actual.as_mut(),
                cpu_input_claim,
                &point(173, shape.log_k()),
            );
            assert_eq!(
                cpu_actual.output_claims(&claims).unwrap(),
                cpu_expected.output_claims(&claims).unwrap()
            );
            assert!(session.state::<RamAccessTape>().is_some());
            let read_write =
                RamReadWriteChecking::new(read_write_dimensions, shape.log_k(), tau_low.clone());
            let read_write_claims = RamReadWriteInputClaims::<AkitaField>::default();
            let read_write_points = RamReadWriteInputClaims::<Vec<AkitaField>>::default();
            let read_write_challenges = RamReadWriteChallenges {
                gamma: AkitaField::from_u64(17),
            };
            let _read_write_kernel = <MetalBackend as PrepareKernel<
                AkitaField,
                RamReadWriteChecking<AkitaField>,
            >>::prepare(
                &metal,
                &mut session,
                witness,
                ProverInputs {
                    relation: &read_write,
                    claims: &read_write_claims,
                    points: &read_write_points,
                    challenges: &read_write_challenges,
                },
            )
            .unwrap();
            assert!(session.state::<RamAccessTape>().is_none());
            let pending = session.state::<PendingRamRafSequence>().unwrap();
            assert!(pending.is_segmented());
            let mut actual =
                PrepareKernel::prepare(&metal, &mut session, witness, inputs()).unwrap();
            assert!(session.state::<PendingRamRafSequence>().is_none());

            let input_claim = probe_input_claim(expected.as_mut());
            let round_challenges = point(211, shape.log_k());
            run_lockstep(
                expected.as_mut(),
                actual.as_mut(),
                input_claim,
                &round_challenges,
            );
            assert_eq!(
                actual.output_claims(&claims).unwrap(),
                expected.output_claims(&claims).unwrap()
            );
            let output_points = relation
                .derive_opening_points(&round_challenges, &points)
                .unwrap();
            actual
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
        });
    }

    #[test]
    fn segmented_source_survives_read_write_and_matches_optimized_cpu() {
        let shape = FixtureShape {
            log_t: 15,
            ram_k: 1 << 14,
        };
        let mut ops = Vec::with_capacity(1 << 14);
        for (word, count) in [(3, 128), (5, 129), (7, 4096), (9, 4097)] {
            ops.push(RamOp::Write {
                word,
                post: word + 17,
            });
            ops.extend(std::iter::repeat_n(RamOp::Read { word }, count - 1));
        }
        ops.resize(1 << 14, RamOp::None);

        with_ram_fixture_backend(shape, ops, |witness| {
            let dimensions =
                ReadWriteDimensions::new(shape.log_t, shape.log_k(), shape.log_t, shape.log_k());
            let tau_low = point(83, shape.log_t);
            let read_write = RamReadWriteChecking::new(dimensions, shape.log_k(), tau_low.clone());
            let read_write_claims = RamReadWriteInputClaims::<AkitaField>::default();
            let read_write_points = RamReadWriteInputClaims::<Vec<AkitaField>>::default();
            let read_write_challenges = RamReadWriteChallenges {
                gamma: AkitaField::from_u64(17),
            };
            let read_write_inputs = || ProverInputs {
                relation: &read_write,
                claims: &read_write_claims,
                points: &read_write_points,
                challenges: &read_write_challenges,
            };

            let mut expected_read_write = OptimizedBackend
                .prepare(&mut ProofSession::default(), witness, read_write_inputs())
                .unwrap();
            let mut config = MetalConfig::default();
            config.ram_read_write.trace_cutoff_elements = 2;
            config.ram_read_write.minimum_accesses = 1;
            config.ram_raf_evaluation.dispatch.trace_cutoff = 1 << shape.log_t;
            let metal = MetalBackend::new(config).unwrap();
            let mut session = ProofSession::default();
            let mut actual_read_write =
                PrepareKernel::prepare(&metal, &mut session, witness, read_write_inputs()).unwrap();

            let pending = session.state::<PendingRamRafSequence>().unwrap();
            assert!(pending.is_segmented());
            assert_eq!(pending.address_domain(), Some(shape.ram_k));
            let source = session.state::<RamRafSegmentedAddressPlane>().unwrap();
            assert!(source.bounded_address_count() >= 2);
            assert!(source.hot_address_count() >= 1);
            assert!(source.hot_message_chunk_count() >= 2);

            let read_write_round_challenges = point(211, shape.log_t + shape.log_k());
            let mut read_write_claim = AkitaField::zero();
            for (round, &challenge) in read_write_round_challenges.iter().enumerate() {
                let bind = round
                    .checked_sub(1)
                    .map(|previous| read_write_round_challenges[previous]);
                let expected_poly = expected_read_write
                    .prove_round(bind, round, read_write_claim)
                    .unwrap();
                let actual_poly = actual_read_write
                    .prove_round(bind, round, read_write_claim)
                    .unwrap();
                assert_eq!(expected_poly.coefficients(), actual_poly.coefficients());
                read_write_claim = expected_poly.evaluate(challenge);
            }
            let last_read_write_challenge = *read_write_round_challenges.last().unwrap();
            expected_read_write
                .finish_rounds(last_read_write_challenge)
                .unwrap();
            actual_read_write
                .finish_rounds(last_read_write_challenge)
                .unwrap();
            assert_eq!(
                actual_read_write.output_claims(&read_write_claims).unwrap(),
                expected_read_write
                    .output_claims(&read_write_claims)
                    .unwrap()
            );

            let relation = RamRafEvaluation::new(
                dimensions,
                RamRafEvaluationDimensions::try_from(dimensions).unwrap(),
                shape.log_k(),
                fixture_lowest_address(),
                tau_low,
            );
            let claims = RamRafEvaluationInputClaims {
                ram_address: AkitaField::zero(),
            };
            let points = RamRafEvaluationInputClaims::<Vec<AkitaField>>::default();
            let challenges = NoChallenges::default();
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };
            let mut expected = OptimizedBackend
                .prepare(&mut ProofSession::default(), witness, inputs())
                .unwrap();
            let mut actual =
                PrepareKernel::prepare(&metal, &mut session, witness, inputs()).unwrap();
            assert!(session.state::<PendingRamRafSequence>().is_none());
            assert!(session.state::<RamRafSegmentedAddressPlane>().is_none());

            let input_claim = probe_input_claim(expected.as_mut());
            let round_challenges = point(401, shape.log_k());
            run_lockstep(
                expected.as_mut(),
                actual.as_mut(),
                input_claim,
                &round_challenges,
            );
            assert_eq!(
                actual.output_claims(&claims).unwrap(),
                expected.output_claims(&claims).unwrap()
            );
            let output_points = relation
                .derive_opening_points(&round_challenges, &points)
                .unwrap();
            actual
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
        });
    }
}
