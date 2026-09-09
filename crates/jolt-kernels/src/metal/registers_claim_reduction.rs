use std::{
    mem,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread::JoinHandle,
};

use super::backend::MetalBackend;
use super::solinas::registers_claim_reduction::{
    RegistersClaimAliasSnapshot, RegistersClaimDenseOutputs, RegistersClaimGeometry,
    RegistersClaimKernelConfig, RegistersClaimPartialQHandoff, RegistersClaimResidentRdPlane,
};
#[cfg(feature = "allocative")]
use super::solinas::OuterRegistersClaimCarrierSubmission;
use super::solinas::{
    OuterRegistersClaimCarrier, OuterRegistersClaimCarrierReceipt,
    PendingOuterRegistersClaimCarrier, RegistersReadWriteStage1Source, SolinasMetal,
};
use crate::optimized::registers_claim_reduction::OptimizedRegistersClaimReduction;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use jolt_claims::protocols::jolt::{JoltDerivedId, RegistersClaimReductionPublic};
use jolt_field::{Accumulator, Prime128OffsetA7F7 as AkitaField, WithAccumulator};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage3::registers_claim_reduction::{
    RegistersClaimReduction, RegistersClaimReductionOutputClaims,
};
use jolt_witness::JoltWitnessPlane;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RegistersClaimReductionImplementation {
    #[default]
    Cpu,
    OuterCarrierAliasHybrid,
}

impl RegistersClaimReductionImplementation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::OuterCarrierAliasHybrid => "outer_carrier_alias_hybrid",
        }
    }
}

static NEXT_ALIAS_GENERATION: AtomicU64 = AtomicU64::new(1);

pub(super) struct MetalRegistersClaimOuterSource<'a> {
    pub(super) context: &'a SolinasMetal,
    pub(super) product_tau_low: &'a [AkitaField],
    pub(super) rows: usize,
    pub(super) compact_storage_id: usize,
    pub(super) residual_storage_id: usize,
    pub(super) device_registry_id: u64,
}

pub(super) struct MetalRegistersClaimStage1Carry {
    receipt: OuterRegistersClaimCarrierReceipt,
    partial_q: RegistersClaimPartialQHandoff<AkitaField>,
    rd: RegistersClaimResidentRdPlane,
}

pub(super) struct MetalRegistersClaimPendingStage1Carry {
    pending: PendingOuterRegistersClaimCarrier,
    #[cfg(feature = "allocative")]
    submission: OuterRegistersClaimCarrierSubmission,
    context: SolinasMetal,
    product_tau_low: Vec<AkitaField>,
    rows: usize,
    compact_storage_id: usize,
    residual_storage_id: usize,
    device_registry_id: u64,
}

pub(super) struct MetalRegistersClaimAsyncStage1Carry {
    #[cfg(feature = "allocative")]
    submission: OuterRegistersClaimCarrierSubmission,
    handle: Option<JoinHandle<Result<MetalRegistersClaimStage1Carry, String>>>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalRegistersClaimAsyncStage1Carry {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("device_storage"),
            (self.submission.partial_bytes
                + self.submission.component_bytes
                + self.submission.rd_bytes) as usize,
        );
        visitor.exit();
    }
}

impl Drop for MetalRegistersClaimAsyncStage1Carry {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl MetalRegistersClaimAsyncStage1Carry {
    fn join(mut self) -> Result<MetalRegistersClaimStage1Carry, KernelError<AkitaField>> {
        let handle = self.handle.take().ok_or(KernelError::InvariantViolation {
            reason: "registers claim-reduction carrier worker was already consumed",
        })?;
        handle
            .join()
            .map_err(|_| KernelError::InvariantViolation {
                reason: "registers claim-reduction carrier worker panicked",
            })?
            .map_err(metal_prepare_error)
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalRegistersClaimPendingStage1Carry {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(allocative::Key::new("pending"), &self.pending);
        visitor.visit_simple(
            allocative::Key::new("product_tau_low"),
            crate::backend::vec_heap_bytes(&self.product_tau_low),
        );
        visitor.exit();
    }
}

impl MetalRegistersClaimPendingStage1Carry {
    pub(super) fn from_outer(
        pending: PendingOuterRegistersClaimCarrier,
        source: MetalRegistersClaimOuterSource<'_>,
    ) -> Result<Self, super::solinas::MetalError> {
        let submission = pending.submission()?;
        if submission.rows != source.rows
            || submission.explicit_rows > submission.rows
            || submission.device_registry_id != source.device_registry_id
            || submission.source_compact_storage_id != source.compact_storage_id
            || submission.source_residual_storage_id != source.residual_storage_id
        {
            return Err(super::solinas::MetalError::InvalidOuterRemainderConfig(
                "pending registers-claim carrier provenance is inconsistent",
            ));
        }
        Ok(Self {
            pending,
            #[cfg(feature = "allocative")]
            submission,
            context: source.context.clone(),
            product_tau_low: source.product_tau_low.to_vec(),
            rows: source.rows,
            compact_storage_id: source.compact_storage_id,
            residual_storage_id: source.residual_storage_id,
            device_registry_id: source.device_registry_id,
        })
    }

    pub(super) fn start(self) -> MetalRegistersClaimAsyncStage1Carry {
        #[cfg(feature = "allocative")]
        let submission = self.submission;
        let handle = std::thread::spawn(move || self.join().map_err(|error| error.to_string()));
        MetalRegistersClaimAsyncStage1Carry {
            #[cfg(feature = "allocative")]
            submission,
            handle: Some(handle),
        }
    }

    fn join(self) -> Result<MetalRegistersClaimStage1Carry, KernelError<AkitaField>> {
        let carrier = self.pending.join().map_err(metal_prepare_error)?;
        let carry = MetalRegistersClaimStage1Carry::from_outer(
            carrier,
            MetalRegistersClaimOuterSource {
                context: &self.context,
                product_tau_low: &self.product_tau_low,
                rows: self.rows,
                compact_storage_id: self.compact_storage_id,
                residual_storage_id: self.residual_storage_id,
                device_registry_id: self.device_registry_id,
            },
        )
        .map_err(metal_prepare_error)?;
        Ok(carry)
    }
}

impl MetalRegistersClaimStage1Carry {
    pub(super) fn from_outer(
        carrier: OuterRegistersClaimCarrier,
        source: MetalRegistersClaimOuterSource<'_>,
    ) -> Result<Self, super::solinas::registers_claim_reduction::RegistersClaimError> {
        let (receipt, components, rd_buffer) = carrier.into_parts();
        let geometry = RegistersClaimGeometry::new(source.rows)?;
        let identities = [
            receipt.source_compact_storage_id,
            receipt.source_residual_storage_id,
            receipt.partial_storage_id,
            receipt.component_storage_id,
            receipt.rd_storage_id,
        ];
        if receipt.rows != source.rows
            || receipt.explicit_rows > receipt.rows
            || receipt.prefix_elements != geometry.prefix_elements()
            || receipt.suffix_elements != geometry.suffix_elements()
            || receipt.device_registry_id != source.device_registry_id
            || receipt.source_compact_storage_id != source.compact_storage_id
            || receipt.source_residual_storage_id != source.residual_storage_id
            || identities.contains(&0)
            || identities
                .iter()
                .enumerate()
                .any(|(index, identity)| identities[..index].contains(identity))
        {
            return Err(
                super::solinas::registers_claim_reduction::RegistersClaimError::InvalidState(
                    "Outer registers-claim carrier provenance is inconsistent",
                ),
            );
        }
        let partial_q = RegistersClaimPartialQHandoff::new(
            geometry,
            receipt.source_generation,
            source.product_tau_low.to_vec(),
            components,
        )
        .map_err(|_| {
            super::solinas::registers_claim_reduction::RegistersClaimError::InvalidState(
                "Outer registers-claim component handoff is invalid",
            )
        })?;
        let rd = source.context.attach_registers_claim_resident_rd_plane(
            rd_buffer,
            source.rows,
            receipt.source_generation,
            receipt.completion_serial,
        )?;
        Ok(Self {
            receipt,
            partial_q,
            rd,
        })
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalRegistersClaimStage1Carry {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("rd_device_bytes"),
            self.rd.resident_bytes() as usize,
        );
        let components = self.partial_q.components();
        visitor.visit_simple(
            allocative::Key::new("partial_q"),
            crate::backend::vec_heap_bytes(&components.rd_write_value)
                + crate::backend::vec_heap_bytes(&components.rs1_value)
                + crate::backend::vec_heap_bytes(&components.rs2_value),
        );
        visitor.exit();
    }
}

struct RegistersClaimAliasBridge {
    generation: u64,
    rows: usize,
    source_compact_storage_id: usize,
    state: Mutex<RegistersClaimAliasState>,
}

enum RegistersClaimAliasState {
    Empty,
    Published(RegistersClaimAliasSnapshot<AkitaField>),
    Consumed,
}

pub(super) struct RegistersClaimAliasPublisher(Arc<RegistersClaimAliasBridge>);
pub(super) struct RegistersClaimAliasReceiver(Arc<RegistersClaimAliasBridge>);

#[cfg(feature = "allocative")]
impl allocative::Allocative for RegistersClaimAliasReceiver {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Ok(state) = self.0.state.lock() {
            if let RegistersClaimAliasState::Published(snapshot) = &*state {
                visitor.visit_simple(
                    allocative::Key::new("published_snapshot"),
                    crate::backend::vec_heap_bytes(&snapshot.prefix_challenges)
                        + crate::backend::vec_heap_bytes(&snapshot.rs1_value)
                        + crate::backend::vec_heap_bytes(&snapshot.rs2_value),
                );
            }
        }
        visitor.exit();
    }
}

pub(super) fn registers_claim_alias_pair(
    rows: usize,
    source_compact_storage_id: usize,
) -> Result<(RegistersClaimAliasPublisher, RegistersClaimAliasReceiver), KernelError<AkitaField>> {
    if rows < 2 || !rows.is_power_of_two() || source_compact_storage_id == 0 {
        return Err(KernelError::InvariantViolation {
            reason: "registers claim alias bridge geometry is invalid",
        });
    }
    let generation = NEXT_ALIAS_GENERATION
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| KernelError::InvariantViolation {
            reason: "registers claim alias generation exhausted",
        })?;
    let bridge = Arc::new(RegistersClaimAliasBridge {
        generation,
        rows,
        source_compact_storage_id,
        state: Mutex::new(RegistersClaimAliasState::Empty),
    });
    Ok((
        RegistersClaimAliasPublisher(bridge.clone()),
        RegistersClaimAliasReceiver(bridge),
    ))
}

impl RegistersClaimAliasPublisher {
    pub(super) fn generation(&self) -> u64 {
        self.0.generation
    }

    pub(super) fn rows(&self) -> usize {
        self.0.rows
    }

    pub(super) fn source_compact_storage_id(&self) -> usize {
        self.0.source_compact_storage_id
    }

    pub(super) fn publish(
        &self,
        prefix_challenges: Vec<AkitaField>,
        rs1_value: Vec<AkitaField>,
        rs2_value: Vec<AkitaField>,
    ) -> Result<(), SumcheckError<AkitaField>> {
        let geometry = RegistersClaimGeometry::new(self.0.rows).map_err(metal_round_error)?;
        let snapshot =
            RegistersClaimAliasSnapshot::new(geometry, prefix_challenges, rs1_value, rs2_value)
                .map_err(metal_round_error)?;
        let mut state =
            self.0.state.lock().map_err(|_| {
                round_state_error("registers claim alias bridge mutex was poisoned")
            })?;
        if !matches!(*state, RegistersClaimAliasState::Empty) {
            return Err(round_state_error(
                "registers claim aliases were published more than once",
            ));
        }
        *state = RegistersClaimAliasState::Published(snapshot);
        Ok(())
    }
}

impl RegistersClaimAliasReceiver {
    fn take(
        &self,
        expected_rows: usize,
        expected_source_compact_storage_id: usize,
        expected_prefix_challenges: &[AkitaField],
    ) -> Result<RegistersClaimAliasSnapshot<AkitaField>, SumcheckError<AkitaField>> {
        if self.0.generation == 0
            || self.0.rows != expected_rows
            || self.0.source_compact_storage_id != expected_source_compact_storage_id
        {
            return Err(round_state_error(
                "registers claim alias bridge provenance is inconsistent",
            ));
        }
        let mut state =
            self.0.state.lock().map_err(|_| {
                round_state_error("registers claim alias bridge mutex was poisoned")
            })?;
        let RegistersClaimAliasState::Published(snapshot) =
            mem::replace(&mut *state, RegistersClaimAliasState::Consumed)
        else {
            return Err(round_state_error(
                "registers claim aliases were unavailable at the midpoint",
            ));
        };
        snapshot
            .validate_identity(expected_prefix_challenges)
            .map_err(metal_round_error)?;
        Ok(snapshot)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimReductionMetalConfig {
    pub implementation: RegistersClaimReductionImplementation,
    pub trace_cutoff_elements: usize,
    pub dispatch: RegistersClaimKernelConfig,
}

impl Default for RegistersClaimReductionMetalConfig {
    fn default() -> Self {
        Self {
            implementation: RegistersClaimReductionImplementation::Cpu,
            trace_cutoff_elements: 1 << 25,
            dispatch: RegistersClaimKernelConfig::default(),
        }
    }
}

impl PrepareKernel<AkitaField, RegistersClaimReduction<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, RegistersClaimReduction<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = RegistersClaimReduction<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let config = self.config.registers_claim_reduction;
        let log_t = inputs.relation.rounds();
        let cycles = 1usize
            .checked_shl(log_t as u32)
            .ok_or(KernelError::InvariantViolation {
                reason: "registers claim-reduction trace domain overflows usize",
            })?;
        let route_span = tracing::info_span!(
            "MetalRegistersClaimReduction::route",
            cycles,
            requested = config.implementation.as_str(),
            realized_route = tracing::field::Empty,
            fallback_reason = tracing::field::Empty,
        );
        let _route_guard = route_span.enter();
        let fallback_reason = if config.implementation == RegistersClaimReductionImplementation::Cpu
        {
            Some("cpu_config")
        } else if cycles < config.trace_cutoff_elements {
            Some("member_cutoff")
        } else if config.implementation
            == RegistersClaimReductionImplementation::OuterCarrierAliasHybrid
            && cycles < self.config.spartan_outer_remainder.trace_cutoff_elements
        {
            Some("outer_cutoff")
        } else if config.implementation
            == RegistersClaimReductionImplementation::OuterCarrierAliasHybrid
            && cycles < self.config.instruction_input.trace_cutoff_elements
        {
            Some("instruction_input_trace_cutoff")
        } else if config.implementation
            == RegistersClaimReductionImplementation::OuterCarrierAliasHybrid
            && cycles <= self.config.instruction_input.cutoff_elements
        {
            Some("instruction_input_cpu_tail")
        } else {
            None
        };
        if let Some(reason) = fallback_reason {
            let _ = route_span.record("realized_route", "optimized_cpu");
            let _ = route_span.record("fallback_reason", reason);
            if config.implementation
                == RegistersClaimReductionImplementation::OuterCarrierAliasHybrid
            {
                drop(session.take::<MetalRegistersClaimStage1Carry>());
                drop(session.take::<MetalRegistersClaimAsyncStage1Carry>());
                drop(session.take::<RegistersClaimAliasReceiver>());
            }
            return OptimizedRegistersClaimReduction.prepare(session, witness, inputs);
        }
        let tau = inputs.relation.product_uniskip_tau_low();
        if log_t == 0 || tau.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "registers claim-reduction relation has invalid geometry",
            });
        }

        let geometry = RegistersClaimGeometry::new(cycles).map_err(metal_prepare_error)?;
        let prepare_span = tracing::info_span!("MetalRegistersClaimReduction::prepare", cycles);
        let _entered = prepare_span.enter();
        let (midpoint_source, prefix) = match config.implementation {
            RegistersClaimReductionImplementation::Cpu => {
                unreachable!("CPU registers claim-reduction returns before Metal preparation")
            }
            RegistersClaimReductionImplementation::OuterCarrierAliasHybrid => {
                let carry = session.take::<MetalRegistersClaimStage1Carry>();
                let pending = session.take::<MetalRegistersClaimAsyncStage1Carry>();
                let aliases = session.take::<RegistersClaimAliasReceiver>();
                let carry = match (carry, pending) {
                    (Some(carry), None) => Some(carry),
                    (None, Some(pending)) => {
                        let join_span =
                            tracing::info_span!("MetalRegistersClaimReduction::carrier_join");
                        let _join_guard = join_span.enter();
                        Some(pending.join()?)
                    }
                    (None, None) => None,
                    (Some(_), Some(_)) => {
                        return Err(KernelError::InvariantViolation {
                            reason: "registers claim-reduction found duplicate stage-1 carries",
                        });
                    }
                };
                let (Some(carry), Some(aliases)) = (carry, aliases) else {
                    let _ = route_span.record("realized_route", "optimized_cpu");
                    let _ = route_span.record("fallback_reason", "missing_carrier");
                    tracing::warn!(
                        target: "jolt::metal",
                        "registers claim-reduction carriers were unavailable; using optimized CPU"
                    );
                    return OptimizedRegistersClaimReduction.prepare(session, witness, inputs);
                };
                if carry.receipt.rows != cycles
                    || carry.receipt.source_generation == 0
                    || carry.rd.geometry() != geometry
                    || carry.rd.source_generation() != carry.receipt.source_generation
                    || carry.rd.device_registry_id() != self.context.device_registry_id()
                    || carry.rd.allocation_identity() != carry.receipt.rd_storage_id
                {
                    return Err(KernelError::InvariantViolation {
                        reason: "registers claim-reduction stage-1 carry changed provenance",
                    });
                }
                if session.state::<RegistersReadWriteStage1Source>().is_some() {
                    if session.state::<RegistersClaimResidentRdPlane>().is_some() {
                        return Err(KernelError::InvariantViolation {
                            reason: "resident register rd-post plane was already parked",
                        });
                    }
                    let rd_post = carry.rd.clone();
                    session.park(rd_post);
                }
                let prefix = carry
                    .partial_q
                    .stage3_prefix_tables(
                        geometry,
                        carry.receipt.source_generation,
                        tau,
                        inputs.challenges.gamma,
                    )
                    .map_err(metal_prepare_error)?;
                let _ = route_span.record("realized_route", "outer_carrier_alias_hybrid");
                let _ = route_span.record("fallback_reason", "none");
                (
                    RegistersClaimMidpointSource::OuterCarrier {
                        rd: carry.rd,
                        aliases,
                        source_compact_storage_id: carry.receipt.source_compact_storage_id,
                    },
                    prefix,
                )
            }
        };
        Ok(Box::new(MetalRegistersClaimReductionKernel {
            context: self.context.clone(),
            midpoint_source: Some(midpoint_source),
            geometry,
            config: config.dispatch,
            gamma: inputs.challenges.gamma,
            gamma_sq: inputs.challenges.gamma * inputs.challenges.gamma,
            tau: tau.to_vec(),
            bound_challenges: Vec::with_capacity(log_t),
            phase: RegistersClaimPhase::Prefix {
                p: prefix.p,
                q: prefix.q,
            },
            next_round: 0,
            finished: false,
            #[cfg(any(test, feature = "test-utils"))]
            test_counters: self.test_counters.clone(),
        }))
    }
}

enum RegistersClaimPhase {
    Prefix {
        p: Vec<AkitaField>,
        q: Vec<AkitaField>,
    },
    Dense {
        eq: Vec<AkitaField>,
        rd_write_value: Vec<AkitaField>,
        rs1_value: Vec<AkitaField>,
        rs2_value: Vec<AkitaField>,
    },
    Poisoned,
}

type DenseTables<'a> = (
    &'a [AkitaField],
    &'a [AkitaField],
    &'a [AkitaField],
    &'a [AkitaField],
);

enum RegistersClaimMidpointSource {
    OuterCarrier {
        rd: RegistersClaimResidentRdPlane,
        aliases: RegistersClaimAliasReceiver,
        source_compact_storage_id: usize,
    },
}

struct MetalRegistersClaimReductionKernel {
    context: std::sync::Arc<super::solinas::SolinasMetal>,
    midpoint_source: Option<RegistersClaimMidpointSource>,
    geometry: RegistersClaimGeometry,
    config: RegistersClaimKernelConfig,
    gamma: AkitaField,
    gamma_sq: AkitaField,
    tau: Vec<AkitaField>,
    bound_challenges: Vec<AkitaField>,
    phase: RegistersClaimPhase,
    next_round: usize,
    finished: bool,
    #[cfg(any(test, feature = "test-utils"))]
    test_counters: Arc<super::backend::MetalTestCounters>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalRegistersClaimReductionKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        use crate::backend::vec_heap_bytes;

        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(allocative::Key::new("tau"), vec_heap_bytes(&self.tau));
        visitor.visit_simple(
            allocative::Key::new("bound_challenges"),
            vec_heap_bytes(&self.bound_challenges),
        );
        if let Some(source) = &self.midpoint_source {
            let bytes = match source {
                RegistersClaimMidpointSource::OuterCarrier { rd, .. } => rd.resident_bytes(),
            };
            visitor.visit_simple(allocative::Key::new("device_rows"), bytes as usize);
        }
        let host_phase = match &self.phase {
            RegistersClaimPhase::Prefix { p, q } => vec_heap_bytes(p) + vec_heap_bytes(q),
            RegistersClaimPhase::Dense {
                eq,
                rd_write_value,
                rs1_value,
                rs2_value,
            } => {
                vec_heap_bytes(eq)
                    + vec_heap_bytes(rd_write_value)
                    + vec_heap_bytes(rs1_value)
                    + vec_heap_bytes(rs2_value)
            }
            RegistersClaimPhase::Poisoned => 0,
        };
        visitor.visit_simple(allocative::Key::new("host_phase"), host_phase);
        visitor.exit();
    }
}

impl MetalRegistersClaimReductionKernel {
    fn bind(&mut self, challenge: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        self.bound_challenges.push(challenge);
        if matches!(&self.phase, RegistersClaimPhase::Prefix { p, .. } if p.len() == 2) {
            return self.transition_to_dense();
        }
        match &mut self.phase {
            RegistersClaimPhase::Prefix { p, q } => {
                bind_table(p, challenge)?;
                bind_table(q, challenge)
            }
            RegistersClaimPhase::Dense {
                eq,
                rd_write_value,
                rs1_value,
                rs2_value,
            } => {
                for table in [eq, rd_write_value, rs1_value, rs2_value] {
                    bind_table(table, challenge)?;
                }
                Ok(())
            }
            RegistersClaimPhase::Poisoned => Err(round_state_error(
                "registers claim-reduction bind found poisoned state",
            )),
        }
    }

    fn transition_to_dense(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        if self.bound_challenges.len() != self.geometry.prefix_vars() {
            return Err(round_state_error(
                "registers claim-reduction midpoint has the wrong bind count",
            ));
        }
        let phase = mem::replace(&mut self.phase, RegistersClaimPhase::Poisoned);
        if !matches!(phase, RegistersClaimPhase::Prefix { .. }) {
            return Err(round_state_error(
                "registers claim-reduction midpoint requires prefix tables",
            ));
        }
        let source = self.midpoint_source.take().ok_or_else(|| {
            round_state_error("registers claim-reduction lost its midpoint source")
        })?;
        let outputs = match source {
            RegistersClaimMidpointSource::OuterCarrier {
                rd,
                aliases,
                source_compact_storage_id,
            } => {
                let phase = tracing::info_span!(
                    "MetalRegistersClaimReduction::midpoint_projection",
                    round = self.geometry.prefix_vars(),
                    rows = self.geometry.rows(),
                    gpu_active_ns = tracing::field::Empty,
                );
                let _phase_guard = phase.enter();
                let aliases = aliases.take(
                    self.geometry.rows(),
                    source_compact_storage_id,
                    &self.bound_challenges,
                )?;
                let invocation = self
                    .context
                    .prepare_registers_claim_alias_fold(&rd, &self.bound_challenges, self.config)
                    .map_err(metal_round_error)?;
                let observation = invocation.execute_timed().map_err(metal_round_error)?;
                let _ = phase.record("gpu_active_ns", duration_nanos(observation.gpu_active));
                #[cfg(any(test, feature = "test-utils"))]
                let _ = self
                    .test_counters
                    .registers_claim_alias_sequences
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                RegistersClaimDenseOutputs {
                    rd_write_value: observation.rd_write_value,
                    rs1_value: aliases.rs1_value,
                    rs2_value: aliases.rs2_value,
                }
            }
        };
        self.install_dense(outputs)
    }

    fn install_dense(
        &mut self,
        outputs: RegistersClaimDenseOutputs<AkitaField>,
    ) -> Result<(), SumcheckError<AkitaField>> {
        let (tau_hi, tau_lo) = self.tau.split_at(self.geometry.suffix_vars());
        let prefix_point = self
            .bound_challenges
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        let scale = EqPolynomial::<AkitaField>::mle(&prefix_point, tau_lo);
        self.phase = RegistersClaimPhase::Dense {
            eq: EqPolynomial::<AkitaField>::evals(tau_hi, Some(scale)),
            rd_write_value: outputs.rd_write_value,
            rs1_value: outputs.rs1_value,
            rs2_value: outputs.rs2_value,
        };
        Ok(())
    }

    fn require_dense(&self) -> Result<DenseTables<'_>, SumcheckKernelError<AkitaField>> {
        let remaining = self.geometry.log_t() - self.bound_challenges.len();
        if !self.finished || remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let RegistersClaimPhase::Dense {
            eq,
            rd_write_value,
            rs1_value,
            rs2_value,
        } = &self.phase
        else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "registers claim-reduction finished without dense tables",
            });
        };
        Ok((eq, rd_write_value, rs1_value, rs2_value))
    }
}

impl ProveRounds<AkitaField> for MetalRegistersClaimReductionKernel {
    fn num_rounds(&self) -> usize {
        self.geometry.log_t()
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if self.finished || round != self.next_round || round >= self.geometry.log_t() {
            return Err(round_state_error(
                "registers claim-reduction round calls are out of order",
            ));
        }
        if bind.is_some() != (round != 0) {
            return Err(round_state_error(
                "registers claim-reduction round has the wrong bind argument",
            ));
        }
        if let Some(challenge) = bind {
            self.bind(challenge)?;
        }
        self.next_round += 1;

        let endpoints = match &self.phase {
            RegistersClaimPhase::Prefix { p, q } => product_endpoints(p, q)?,
            RegistersClaimPhase::Dense {
                eq,
                rd_write_value,
                rs1_value,
                rs2_value,
            } => dense_endpoints(
                eq,
                rd_write_value,
                rs1_value,
                rs2_value,
                self.gamma,
                self.gamma_sq,
            )?,
            RegistersClaimPhase::Poisoned => {
                return Err(round_state_error(
                    "registers claim-reduction round found poisoned state",
                ));
            }
        };
        Ok(UnivariatePoly::from_evals_and_hint(
            previous_claim,
            &endpoints,
        ))
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.finished || self.next_round != self.geometry.log_t() {
            return Err(round_state_error(
                "registers claim-reduction cannot finish before every round",
            ));
        }
        self.bind(bind)?;
        self.finished = true;
        Ok(())
    }
}

impl SumcheckKernel<AkitaField> for MetalRegistersClaimReductionKernel {
    type Relation = RegistersClaimReduction<AkitaField>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<RegistersClaimReductionOutputClaims<AkitaField>, SumcheckKernelError<AkitaField>>
    {
        let (_, rd_write_value, rs1_value, rs2_value) = self.require_dense()?;
        Ok(RegistersClaimReductionOutputClaims {
            rd_write_value: rd_write_value[0],
            rs1_value: rs1_value[0],
            rs2_value: rs2_value[0],
        })
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        let (eq, ..) = self.require_dense()?;
        let id = JoltDerivedId::from(RegistersClaimReductionPublic::EqSpartan);
        let expected = relation.derive_output_term(&id, input_points, output_points, challenges)?;
        if eq[0] != expected {
            return Err(SumcheckKernelError::DerivedTableDrift {
                id,
                expected,
                got: eq[0],
            });
        }
        Ok(())
    }
}

fn bind_table(
    table: &mut Vec<AkitaField>,
    challenge: AkitaField,
) -> Result<(), SumcheckError<AkitaField>> {
    if table.len() < 2 || !table.len().is_power_of_two() {
        return Err(round_state_error(
            "registers claim-reduction table has invalid bind geometry",
        ));
    }
    let half = table.len() / 2;
    for index in 0..half {
        let lo = table[2 * index];
        table[index] = lo + challenge * (table[2 * index + 1] - lo);
    }
    table.truncate(half);
    Ok(())
}

fn product_endpoints(
    left: &[AkitaField],
    right: &[AkitaField],
) -> Result<[AkitaField; 2], SumcheckError<AkitaField>> {
    if left.len() != right.len() || left.len() < 2 || !left.len().is_power_of_two() {
        return Err(round_state_error(
            "registers claim-reduction prefix tables disagree",
        ));
    }
    let mut accumulators = [<AkitaField as WithAccumulator>::Accumulator::default(); 2];
    for index in 0..left.len() / 2 {
        let (left_0, left_1) = (left[2 * index], left[2 * index + 1]);
        let (right_0, right_1) = (right[2 * index], right[2 * index + 1]);
        accumulators[0].fmadd(left_0, right_0);
        accumulators[1].fmadd(left_1 + left_1 - left_0, right_1 + right_1 - right_0);
    }
    Ok(accumulators.map(<AkitaField as WithAccumulator>::Accumulator::reduce))
}

fn dense_endpoints(
    eq: &[AkitaField],
    rd_write_value: &[AkitaField],
    rs1_value: &[AkitaField],
    rs2_value: &[AkitaField],
    gamma: AkitaField,
    gamma_sq: AkitaField,
) -> Result<[AkitaField; 2], SumcheckError<AkitaField>> {
    if [
        eq.len(),
        rd_write_value.len(),
        rs1_value.len(),
        rs2_value.len(),
    ]
    .iter()
    .any(|&length| length != eq.len())
        || eq.len() < 2
        || !eq.len().is_power_of_two()
    {
        return Err(round_state_error(
            "registers claim-reduction dense tables disagree",
        ));
    }
    let mut accumulators = [<AkitaField as WithAccumulator>::Accumulator::default(); 2];
    for index in 0..eq.len() / 2 {
        let pair = |table: &[AkitaField]| (table[2 * index], table[2 * index + 1]);
        let (eq_0, eq_1) = pair(eq);
        let (rd_0, rd_1) = pair(rd_write_value);
        let (rs1_0, rs1_1) = pair(rs1_value);
        let (rs2_0, rs2_1) = pair(rs2_value);
        accumulators[0].fmadd(eq_0, rd_0 + gamma * rs1_0 + gamma_sq * rs2_0);
        accumulators[1].fmadd(
            eq_1 + eq_1 - eq_0,
            (rd_1 + rd_1 - rd_0)
                + gamma * (rs1_1 + rs1_1 - rs1_0)
                + gamma_sq * (rs2_1 + rs2_1 - rs2_0),
        );
    }
    Ok(accumulators.map(<AkitaField as WithAccumulator>::Accumulator::reduce))
}

fn metal_prepare_error(error: impl ToString) -> KernelError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
    .into()
}

fn metal_round_error(error: impl ToString) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
}

fn round_state_error(reason: &'static str) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: reason.to_owned(),
    }
}

fn duration_nanos(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}
