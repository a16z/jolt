use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use jolt_claims::protocols::jolt::geometry::dimensions::PRODUCT_UNISKIP_DOMAIN_SIZE;
use jolt_claims::protocols::jolt::geometry::spartan::{
    branch_flag_product, jump_flag_product, left_instruction_input_product, lookup_output_product,
    next_is_noop_product, right_instruction_input_product, virtual_instruction_product,
    write_lookup_output_to_rd_product,
};
use jolt_claims::protocols::jolt::{
    JoltDerivedId, JoltOpeningId, SpartanProductVirtualizationPublic,
};
use jolt_claims::{InputClaims as _, OutputClaims as _};
use jolt_field::AkitaField;
use jolt_poly::lagrange::{
    centered_lagrange_evals, centered_lagrange_kernel, interpolate_to_coeffs, poly_mul,
};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
use jolt_verifier::VerifierError;
use jolt_witness::JoltWitnessPlane;

use super::backend::MetalBackend;
use super::solinas::{
    MetalError, OuterRemainderSequenceStorage, PendingProductInstructionInitialMessage,
    PendingProductRemainderInitialMessage, ProductInstructionInitialMessageStats,
    ProductInstructionRoundService, ProductInstructionRoundStats, ProductRemainderRow,
    ProductRemainderRows, ProductRemainderSequence, ProductRemainderSequenceConfig,
    ProductRemainderWorkspacePrimerStats, SpartanOuterUniskipRows,
};
#[cfg(test)]
use crate::optimized::spartan_product::SpartanProductRow;
use crate::optimized::spartan_product::{OptimizedProductRemainder, OptimizedProductUniskip};
use crate::uniskip::UniskipKernel;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, Stage2ProductInstructionPrefetch,
    SumcheckKernel, SumcheckKernelError,
};
#[cfg(test)]
use jolt_witness::collect_bundles;

const DOMAIN: usize = PRODUCT_UNISKIP_DOMAIN_SIZE;
const EXTENDED_SIZE: usize = 2 * DOMAIN - 1;
const DOMAIN_START: i64 = -((DOMAIN as i64 - 1) / 2);
const EXTENDED_START: i64 = -((EXTENDED_SIZE as i64 - 1) / 2);

pub(super) struct MetalInstructionClaimResidentRows {
    pub(super) log_t: usize,
    pub(super) product: ProductRemainderRows,
    pub(super) device_registry_id: u64,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalInstructionClaimResidentRows {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(allocative::Key::new("product"), &self.product);
        visitor.exit();
    }
}

#[derive(Clone, Copy)]
pub(super) struct MetalInstructionClaimAliases {
    pub(super) lookup_output: AkitaField,
    pub(super) left_instruction_input: AkitaField,
    pub(super) right_instruction_input: AkitaField,
}

pub(super) struct MetalInstructionClaimAliasOutput {
    pub(super) row_storage_id: usize,
    pub(super) challenges: Vec<AkitaField>,
    pub(super) values: MetalInstructionClaimAliases,
}

pub(super) type MetalInstructionClaimAliasSlot =
    Arc<Mutex<Option<MetalInstructionClaimAliasOutput>>>;

pub(super) struct MetalInstructionClaimHandoff {
    pub(super) rows: MetalInstructionClaimResidentRows,
    pub(super) aliases: MetalInstructionClaimAliasSlot,
    pub(super) prefetched_initial: Option<MetalInstructionClaimPrefetchedInitial>,
}

pub(super) struct MetalInstructionClaimPrefetchedInitial {
    pub(super) service: Arc<Mutex<ProductInstructionRoundService>>,
    pub(super) endpoints: [AkitaField; 2],
    pub(super) stats: ProductInstructionInitialMessageStats,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalInstructionClaimHandoff {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(allocative::Key::new("rows"), &self.rows);
        visitor.exit();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpartanProductRemainderMetalConfig {
    pub trace_cutoff_elements: usize,
    pub cpu_tail_elements: usize,
    pub reuse_outer_state_a: bool,
    pub dispatch: ProductRemainderSequenceConfig,
}

impl Default for SpartanProductRemainderMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 18,
            cpu_tail_elements: 1 << 12,
            reuse_outer_state_a: false,
            dispatch: ProductRemainderSequenceConfig::default(),
        }
    }
}

impl MetalBackend {
    pub(super) fn prepare_product_remainder_witness(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        _witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<(), KernelError<AkitaField>> {
        let cycles = 1usize
            .checked_shl(log_t as u32)
            .ok_or(KernelError::InvariantViolation {
                reason: "Spartan product trace length overflows usize",
            })?;
        if cycles < self.config.spartan_product_remainder.trace_cutoff_elements {
            return Ok(());
        }

        let span = tracing::info_span!(
            "MetalProductRemainder::witness_prepare",
            cycles,
            row_bytes = cycles.saturating_mul(std::mem::size_of::<ProductRemainderRow>()),
            lookup_companion_bytes = 0u64,
            residual_witness_scan_rows = tracing::field::Empty,
            additional_source_row_scans = tracing::field::Empty,
            source_upload_bytes = tracing::field::Empty,
            source_allocation_count = tracing::field::Empty,
            source_compact_storage_id = tracing::field::Empty,
            source_residual_storage_id = tracing::field::Empty,
            collect_wall_ns = tracing::field::Empty,
            upload_wall_ns = tracing::field::Empty,
            lookup_upload_wall_ns = tracing::field::Empty,
            sequence_prepare_wall_ns = tracing::field::Empty,
            workspace_bytes = tracing::field::Empty,
            primer_mode = "state_arenas_blit_v1",
            primer_dispatches = 0u64,
            primer_command_buffers = 1u64,
            primer_waits = 1u64,
            primer_readback_bytes = 0u64,
            primer_bytes = tracing::field::Empty,
            primer_buffers = tracing::field::Empty,
            primer_wall_ns = tracing::field::Empty,
            primer_gpu_active_ns = tracing::field::Empty,
            primer_state_a_storage_id = tracing::field::Empty,
            primer_state_b_storage_id = tracing::field::Empty,
            resident_rows_storage_id = tracing::field::Empty,
            owner_generation = tracing::field::Empty,
            source_generation = tracing::field::Empty,
            row_source = tracing::field::Empty,
            admitted = tracing::field::Empty,
            fallback_reason = tracing::field::Empty,
        );
        let _entered = span.enter();
        let Some(rows) = session
            .state::<SpartanOuterUniskipRows>()
            .filter(|rows| {
                rows.len() == cycles
                    && rows.device_registry_id() == self.context.device_registry_id()
            })
            .map(SpartanOuterUniskipRows::share_product_remainder_rows)
            .transpose()
            .map_err(metal_prepare_error)?
        else {
            let _ = span.record("admitted", false);
            let _ = span.record("fallback_reason", "stage1_source_missing");
            return Ok(());
        };
        let identities = rows.allocation_identities();
        let _ = span.record("residual_witness_scan_rows", 0u64);
        let _ = span.record("additional_source_row_scans", 0u64);
        let _ = span.record("source_upload_bytes", 0u64);
        let _ = span.record("source_allocation_count", 0u64);
        let _ = span.record("collect_wall_ns", 0u64);
        let _ = span.record("upload_wall_ns", 0u64);
        let _ = span.record("lookup_upload_wall_ns", 0u64);
        let _ = span.record("source_compact_storage_id", identities[0] as u64);
        let _ = span.record("source_residual_storage_id", identities[1] as u64);
        let _ = span.record("row_source", rows.source_kind().as_str());
        let row_storage_id = rows.allocation_identity();
        if let Some(source_generation) = rows.source_generation() {
            let _ = span.record("source_generation", source_generation);
        }
        let _ = span.record("resident_rows_storage_id", row_storage_id as u64);

        let e_in_capacity = 1usize << (log_t / 2);
        let e_out_capacity = cycles / e_in_capacity;
        let instruction_product_rows = rows.clone();
        let started = Instant::now();
        let state_a = self
            .config
            .spartan_product_remainder
            .reuse_outer_state_a
            .then(|| {
                session
                    .state::<OuterRemainderSequenceStorage>()
                    .ok_or(KernelError::InvariantViolation {
                        reason: "Product state reuse requires resident Outer storage",
                    })?
                    .share_product_state_a()
                    .map_err(metal_prepare_error)
            })
            .transpose()?;
        let sequence = self
            .context
            .prepare_product_remainder_sequence_with_rows_and_state_a(
                rows,
                [AkitaField::zero(); DOMAIN],
                e_in_capacity,
                e_out_capacity,
                self.config.spartan_product_remainder.dispatch,
                state_a,
            );
        let _ = span.record(
            "sequence_prepare_wall_ns",
            duration_nanos(started.elapsed()),
        );
        let sequence = match sequence {
            Ok(sequence) => sequence,
            Err(error) if error.is_capacity_error() => {
                let _ = span.record("admitted", false);
                let _ = span.record("fallback_reason", "capacity");
                tracing::warn!(
                    target: "jolt::metal",
                    error = %error,
                    "product-remainder workspace was not admitted; using optimized CPU"
                );
                return Ok(());
            }
            Err(error) => return Err(metal_prepare_error(error)),
        };
        let primer: ProductRemainderWorkspacePrimerStats =
            sequence.prime_workspace().map_err(metal_prepare_error)?;
        let _ = span.record("primer_bytes", primer.bytes);
        let _ = span.record("primer_buffers", primer.buffers);
        let _ = span.record("primer_wall_ns", duration_nanos(primer.wall));
        let _ = span.record("primer_gpu_active_ns", duration_nanos(primer.gpu_active));
        let _ = span.record("primer_state_a_storage_id", primer.state_a_identity as u64);
        let _ = span.record("primer_state_b_storage_id", primer.state_b_identity as u64);
        let _ = span.record(
            "workspace_bytes",
            sequence.storage_layout().workspace_bytes(),
        );
        if !sequence.is_ready() || sequence.row_allocation_identity() != row_storage_id {
            return Err(KernelError::InvariantViolation {
                reason: "product-remainder witness preparation ended in the wrong state",
            });
        }
        let _ = span.record("admitted", true);
        let _ = span.record("fallback_reason", "none");
        session.park(MetalInstructionClaimResidentRows {
            log_t,
            product: instruction_product_rows,
            device_registry_id: self.context.device_registry_id(),
        });
        session.park(sequence);
        Ok(())
    }
}

struct MetalProductUniskipCarry {
    log_t: usize,
    tau_low: Vec<AkitaField>,
    endpoints: [AkitaField; 2],
    row_storage_id: usize,
    device_registry_id: u64,
}

enum MetalProductRemainderPrefetchCommand {
    Product(Box<PendingProductRemainderInitialMessage>),
    ProductInstruction(Box<PendingProductInstructionInitialMessage>),
}

struct MetalProductRemainderPrefetch {
    command: MetalProductRemainderPrefetchCommand,
    instruction_rows: Option<MetalInstructionClaimResidentRows>,
    log_t: usize,
    tau_low: Vec<AkitaField>,
    uniskip_challenge: AkitaField,
    tau_high: AkitaField,
    row_storage_id: usize,
    device_registry_id: u64,
}

pub(super) struct MetalProductUniskipEndpointCarrier {
    pub(super) log_t: usize,
    pub(super) tau_low: Vec<AkitaField>,
    pub(super) endpoints: [AkitaField; 2],
    pub(super) source_rows: usize,
    pub(super) source_row_storage_id: usize,
    pub(super) device_registry_id: u64,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalProductUniskipEndpointCarrier {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("tau_low"),
            crate::backend::vec_heap_bytes(&self.tau_low),
        );
        visitor.exit();
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalProductUniskipCarry {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        use crate::backend::vec_heap_bytes;
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("tau_low"),
            vec_heap_bytes(&self.tau_low),
        );
        visitor.exit();
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalProductRemainderPrefetch {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        use crate::backend::vec_heap_bytes;
        let mut visitor = visitor.enter_self_sized::<Self>();
        match &self.command {
            MetalProductRemainderPrefetchCommand::Product(pending) => {
                visitor.visit_field(allocative::Key::new("product"), pending.as_ref());
            }
            MetalProductRemainderPrefetchCommand::ProductInstruction(pending) => {
                visitor.visit_field(
                    allocative::Key::new("product_instruction"),
                    pending.as_ref(),
                );
            }
        }
        if let Some(rows) = &self.instruction_rows {
            visitor.visit_field(allocative::Key::new("instruction_rows"), rows);
        }
        visitor.visit_simple(
            allocative::Key::new("tau_low"),
            vec_heap_bytes(&self.tau_low),
        );
        visitor.exit();
    }
}

impl UniskipKernel<AkitaField, ProductRemainder<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        tau_low: &[AkitaField],
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<(), KernelError<AkitaField>> {
        if tau_low.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan product tau_low must carry log_t challenges",
            });
        }
        let cycles = 1usize
            .checked_shl(log_t as u32)
            .ok_or(KernelError::InvariantViolation {
                reason: "Spartan product trace length overflows usize",
            })?;
        let endpoint_carrier = session.take::<MetalProductUniskipEndpointCarrier>();
        let use_metal = cycles >= self.config.spartan_product_remainder.trace_cutoff_elements
            && session.state::<ProductRemainderSequence>().is_some();
        if !use_metal {
            drop(session.take::<ProductRemainderSequence>());
            drop(session.take::<MetalProductUniskipCarry>());
            drop(session.take::<MetalInstructionClaimResidentRows>());
            drop(session.take::<MetalInstructionClaimHandoff>());
            return OptimizedProductUniskip.prepare(session, log_t, tau_low, witness);
        }

        let mut sequence =
            session
                .take::<ProductRemainderSequence>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "Metal product uni-skip lost its resident sequence",
                })?;
        if !sequence.is_ready()
            || sequence.storage_layout().rows() != cycles
            || sequence.device_registry_id() != self.context.device_registry_id()
        {
            return Err(KernelError::InvariantViolation {
                reason: "Metal product uni-skip sequence has the wrong state, shape, or device",
            });
        }
        let row_storage_id = sequence.row_allocation_identity();
        let endpoints = if let Some(carrier) = endpoint_carrier {
            if carrier.log_t != log_t
                || carrier.tau_low != tau_low
                || carrier.source_rows != cycles
                || carrier.source_row_storage_id == 0
                || carrier.device_registry_id != self.context.device_registry_id()
            {
                return Err(KernelError::InvariantViolation {
                    reason: "stage-1 product uni-skip endpoint carrier has stale provenance",
                });
            }
            let _span = tracing::info_span!(
                "MetalProductUniskip::outer_opening_carrier",
                cycles,
                source_rows_storage_id = carrier.source_row_storage_id as u64,
                product_rows_storage_id = row_storage_id as u64,
                row_upload_bytes = 0u64,
                dispatches = 0u64,
                command_buffers = 0u64,
                readback_bytes = 0u64,
            )
            .entered();
            #[cfg(any(test, feature = "test-utils"))]
            let _ = self
                .test_counters
                .product_uniskip_carrier_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            carrier.endpoints
        } else {
            let split = tau_low.len().div_ceil(2);
            let (out_point, in_point) = tau_low.split_at(split);
            let e_in = EqPolynomial::evals(in_point, None);
            let e_out = EqPolynomial::evals(out_point, None);
            let span = tracing::info_span!(
                "MetalProductUniskip::prepare",
                cycles,
                resident_rows_storage_id = row_storage_id as u64,
                row_upload_bytes = 0u64,
                round_device_buffer_allocations = 0u64,
                dispatch_wall_ns = tracing::field::Empty,
                gpu_active_ns = tracing::field::Empty,
            );
            let _entered = span.enter();
            let started = Instant::now();
            let endpoints = sequence.uniskip_message_timed(&e_in, &e_out);
            let dispatch_wall = started.elapsed();
            let (endpoints, gpu_active) = match endpoints {
                Ok(result) => result,
                Err(error) if product_prepare_fallback_reason(&error).is_some() => {
                    tracing::warn!(
                        target: "jolt::metal",
                        error = %error,
                        "product uni-skip dispatch failed before Fiat-Shamir; using optimized CPU"
                    );
                    return OptimizedProductUniskip.prepare(session, log_t, tau_low, witness);
                }
                Err(error) => return Err(metal_prepare_error(error)),
            };
            let _ = span.record("dispatch_wall_ns", duration_nanos(dispatch_wall));
            let _ = span.record("gpu_active_ns", duration_nanos(gpu_active));
            #[cfg(any(test, feature = "test-utils"))]
            let _ = self
                .test_counters
                .product_uniskip_dispatches
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            endpoints.as_array()
        };
        if sequence.row_allocation_identity() != row_storage_id || !sequence.is_ready() {
            return Err(KernelError::InvariantViolation {
                reason: "product uni-skip changed the resident sequence allocation or phase",
            });
        }
        session.park(sequence);
        session.park(MetalProductUniskipCarry {
            log_t,
            tau_low: tau_low.to_vec(),
            endpoints,
            row_storage_id,
            device_registry_id: self.context.device_registry_id(),
        });
        Ok(())
    }

    fn first_round_poly(
        &self,
        session: &mut ProofSession,
        late_tau: &[AkitaField],
        known_values: &[AkitaField],
    ) -> Result<UnivariatePoly<AkitaField>, KernelError<AkitaField>> {
        let Some(carry) = session.state::<MetalProductUniskipCarry>() else {
            return OptimizedProductUniskip.first_round_poly(session, late_tau, known_values);
        };
        let &[tau_high] = late_tau else {
            return Err(KernelError::InvariantViolation {
                reason:
                    "the product uni-skip first-round polynomial expects exactly one late challenge",
            });
        };
        let &[product, should_branch, should_jump] = known_values else {
            return Err(KernelError::InvariantViolation {
                reason: "the product uni-skip first-round polynomial expects three known nodes",
            });
        };
        let sequence =
            session
                .state::<ProductRemainderSequence>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "Metal product uni-skip lost its sequence before interpolation",
                })?;
        if sequence.row_allocation_identity() != carry.row_storage_id
            || sequence.device_registry_id() != carry.device_registry_id
            || !sequence.is_ready()
        {
            return Err(KernelError::InvariantViolation {
                reason: "Metal product uni-skip carry disagrees with its resident sequence",
            });
        }
        let kernel_values = centered_lagrange_evals::<AkitaField>(DOMAIN, tau_high)?;
        let kernel_coefficients = interpolate_to_coeffs(DOMAIN_START, &kernel_values);
        let t1_values = [
            carry.endpoints[0],
            product,
            should_branch,
            should_jump,
            carry.endpoints[1],
        ];
        let t1_coefficients = interpolate_to_coeffs(EXTENDED_START, &t1_values);
        Ok(UnivariatePoly::new(poly_mul(
            &kernel_coefficients,
            &t1_coefficients,
        )))
    }
}

impl PrepareKernel<AkitaField, ProductRemainder<AkitaField>> for MetalBackend {
    fn prefetch_relation(
        &self,
        session: &mut ProofSession,
        relation: &ProductRemainder<AkitaField>,
    ) -> Result<(), KernelError<AkitaField>> {
        let rounds = relation.rounds();
        let cycles = 1usize
            .checked_shl(rounds as u32)
            .ok_or(KernelError::InvariantViolation {
                reason: "Spartan product trace length overflows usize",
            })?;
        if cycles < self.config.spartan_product_remainder.trace_cutoff_elements
            || session.state::<MetalProductUniskipCarry>().is_none()
            || session.state::<ProductRemainderSequence>().is_none()
        {
            return Ok(());
        }
        if session.state::<MetalProductRemainderPrefetch>().is_some() {
            return Err(KernelError::InvariantViolation {
                reason: "product-remainder prefetch was submitted twice",
            });
        }
        let (carry_log_t, tau_low, row_storage_id, device_registry_id) = {
            let carry = session.state::<MetalProductUniskipCarry>().ok_or(
                KernelError::InvariantViolation {
                    reason: "product-remainder prefetch lost its uni-skip carry",
                },
            )?;
            (
                carry.log_t,
                carry.tau_low.clone(),
                carry.row_storage_id,
                carry.device_registry_id,
            )
        };
        if carry_log_t != rounds || tau_low.len() != rounds {
            return Err(KernelError::InvariantViolation {
                reason: "product-remainder prefetch disagrees with the uni-skip carry",
            });
        }
        let host = MetalProductRemainderHost::new(
            &tau_low,
            relation.uniskip_challenge(),
            relation.tau_high(),
        )?;
        let mut sequence =
            session
                .take::<ProductRemainderSequence>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "product-remainder prefetch lost its resident sequence",
                })?;
        if !sequence.is_ready()
            || sequence.storage_layout().rows() != cycles
            || sequence.device_registry_id() != self.context.device_registry_id()
            || sequence.device_registry_id() != device_registry_id
            || sequence.row_allocation_identity() != row_storage_id
        {
            return Err(KernelError::InvariantViolation {
                reason: "product-remainder prefetch has the wrong state, shape, device, or rows",
            });
        }
        sequence
            .set_lagrange_weights(host.lagrange_weights)
            .map_err(metal_prepare_error)?;
        let (e_in, e_out) = host.current_weights()?;
        let instruction_gamma = session
            .state::<Stage2ProductInstructionPrefetch<AkitaField>>()
            .map(|prefetch| prefetch.instruction_gamma);
        let instruction_rows = if instruction_gamma.is_some()
            && cycles
                >= self
                    .config
                    .instruction_claim_reduction
                    .trace_cutoff_elements
        {
            session.take::<MetalInstructionClaimResidentRows>()
        } else {
            None
        };
        let joint = match (instruction_gamma, instruction_rows) {
            (Some(gamma), Some(rows)) => {
                if rows.log_t != rounds
                    || rows.product.allocation_identity() != row_storage_id
                    || rows.product.source_kind()
                        != super::solinas::ProductRemainderSourceKind::SpartanStage1
                    || rows.device_registry_id != self.context.device_registry_id()
                {
                    return Err(KernelError::InvariantViolation {
                        reason: "joint Product/Instruction prefetch received mismatched rows",
                    });
                }
                match self
                    .context
                    .prepare_instruction_claim_sequence_with_stage1_rows(
                        rows.product.clone(),
                        gamma,
                        self.config.instruction_claim_reduction.dispatch,
                    ) {
                    Ok(instruction) => Some((instruction, rows)),
                    Err(error) if error.is_capacity_error() => {
                        session.park(rows);
                        None
                    }
                    Err(error) => return Err(metal_prepare_error(error)),
                }
            }
            (None, Some(rows)) => {
                session.park(rows);
                None
            }
            (_, None) => None,
        };
        let span = tracing::info_span!(
            "MetalProductRemainder::prefetch_submit",
            cycles,
            rounds,
            resident_rows_storage_id = row_storage_id as u64,
            row_upload_bytes = 0u64,
            joint_product_instruction = joint.is_some(),
            command_committed = true,
            protocol_state_advanced = false,
            submit_wall_ns = tracing::field::Empty,
        );
        let _entered = span.enter();
        let started = Instant::now();
        let (command, instruction_rows) = if let Some((instruction, rows)) = joint {
            let pending = self
                .context
                .submit_product_instruction_initial_message(sequence, instruction, &e_in, &e_out)
                .map_err(metal_prepare_error)?;
            (
                MetalProductRemainderPrefetchCommand::ProductInstruction(Box::new(pending)),
                Some(rows),
            )
        } else {
            let pending = sequence
                .submit_initial_message(&e_in, &e_out)
                .map_err(metal_prepare_error)?;
            (
                MetalProductRemainderPrefetchCommand::Product(Box::new(pending)),
                None,
            )
        };
        let _ = span.record("submit_wall_ns", duration_nanos(started.elapsed()));
        session.park(MetalProductRemainderPrefetch {
            command,
            instruction_rows,
            log_t: rounds,
            tau_low,
            uniskip_challenge: relation.uniskip_challenge(),
            tau_high: relation.tau_high(),
            row_storage_id,
            device_registry_id,
        });
        Ok(())
    }

    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, ProductRemainder<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = ProductRemainder<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let rounds = inputs.relation.rounds();
        let cycles = 1usize
            .checked_shl(rounds as u32)
            .ok_or(KernelError::InvariantViolation {
                reason: "Spartan product trace length overflows usize",
            })?;
        let has_metal_carry = session.state::<MetalProductUniskipCarry>().is_some();
        if !has_metal_carry {
            drop(session.take::<ProductRemainderSequence>());
            drop(session.take::<MetalProductRemainderPrefetch>());
            drop(session.take::<MetalInstructionClaimResidentRows>());
            drop(session.take::<MetalInstructionClaimHandoff>());
            return OptimizedProductRemainder.prepare(session, witness, inputs);
        }
        let has_metal_remainder = session.state::<ProductRemainderSequence>().is_some()
            || session.state::<MetalProductRemainderPrefetch>().is_some();
        if cycles < self.config.spartan_product_remainder.trace_cutoff_elements
            || !has_metal_remainder
        {
            return Err(KernelError::InvariantViolation {
                reason: "Metal product uni-skip cannot hand off without its resident remainder",
            });
        }
        let (carry_log_t, tau_low, carry_row_storage_id, carry_device_registry_id) = {
            let carry = session.state::<MetalProductUniskipCarry>().ok_or(
                KernelError::InvariantViolation {
                    reason: "Metal product uni-skip carry disappeared before handoff",
                },
            )?;
            (
                carry.log_t,
                carry.tau_low.clone(),
                carry.row_storage_id,
                carry.device_registry_id,
            )
        };
        if carry_log_t != rounds || tau_low.len() != rounds {
            return Err(KernelError::InvariantViolation {
                reason: "product uni-skip carry disagrees with the remainder relation",
            });
        }
        let host = MetalProductRemainderHost::new(
            &tau_low,
            inputs.relation.uniskip_challenge(),
            inputs.relation.tau_high(),
        )?;
        let prefetched = session.take::<MetalProductRemainderPrefetch>();
        if prefetched.is_some() && session.state::<ProductRemainderSequence>().is_some() {
            return Err(KernelError::InvariantViolation {
                reason: "product-remainder prefetch duplicated its resident sequence",
            });
        }
        let (sequence, first_message, prefetched_instruction_parts, prefetched_instruction_rows) =
            if let Some(prefetched) = prefetched {
                if prefetched.log_t != rounds
                    || prefetched.tau_low != tau_low
                    || prefetched.uniskip_challenge != inputs.relation.uniskip_challenge()
                    || prefetched.tau_high != inputs.relation.tau_high()
                    || prefetched.row_storage_id != carry_row_storage_id
                    || prefetched.device_registry_id != carry_device_registry_id
                {
                    return Err(KernelError::InvariantViolation {
                        reason: "prefetched product remainder disagrees with its relation or carry",
                    });
                }
                let prepare_span = tracing::info_span!(
                    "MetalProductRemainder::prepare",
                    cycles,
                    rounds,
                    resident_rows_storage_id = carry_row_storage_id as u64,
                    row_upload_bytes = 0u64,
                    round_device_buffer_allocations = 0u64,
                    sequence_prepare_wall_ns = 0u64,
                    prefetched = true,
                    materialize_wall_ns = tracing::field::Empty,
                    materialize_submit_wall_ns = tracing::field::Empty,
                    materialize_overlap_wall_ns = tracing::field::Empty,
                    materialize_join_wall_ns = tracing::field::Empty,
                    materialize_gpu_active_ns = tracing::field::Empty,
                    completed_before_join = tracing::field::Empty,
                    joint_product_instruction = matches!(
                        &prefetched.command,
                        MetalProductRemainderPrefetchCommand::ProductInstruction(_)
                    ),
                    joint_threads_per_threadgroup = tracing::field::Empty,
                    joint_threadgroup_bytes = tracing::field::Empty,
                );
                let _entered = prepare_span.enter();
                match prefetched.command {
                    MetalProductRemainderPrefetchCommand::Product(pending) => {
                        let (sequence, first_message, stats) =
                            (*pending).join().map_err(metal_prepare_error)?;
                        let _ = prepare_span
                            .record("materialize_wall_ns", duration_nanos(stats.lifecycle_wall));
                        let _ = prepare_span.record(
                            "materialize_submit_wall_ns",
                            duration_nanos(stats.submit_wall),
                        );
                        let _ = prepare_span.record(
                            "materialize_overlap_wall_ns",
                            duration_nanos(stats.overlap_wall),
                        );
                        let _ = prepare_span
                            .record("materialize_join_wall_ns", duration_nanos(stats.join_wall));
                        let _ = prepare_span.record(
                            "materialize_gpu_active_ns",
                            duration_nanos(stats.gpu_active),
                        );
                        let _ = prepare_span
                            .record("completed_before_join", stats.completed_before_join);
                        (sequence, first_message, None, None)
                    }
                    MetalProductRemainderPrefetchCommand::ProductInstruction(pending) => {
                        let (sequence, first_message, instruction, endpoints, stats) =
                            (*pending).join().map_err(metal_prepare_error)?;
                        let _ =
                            prepare_span.record("materialize_wall_ns", duration_nanos(stats.wall));
                        let _ = prepare_span.record(
                            "materialize_submit_wall_ns",
                            duration_nanos(stats.submit_wall),
                        );
                        let _ = prepare_span.record(
                            "materialize_overlap_wall_ns",
                            duration_nanos(stats.overlap_wall),
                        );
                        let _ = prepare_span
                            .record("materialize_join_wall_ns", duration_nanos(stats.join_wall));
                        let _ = prepare_span.record(
                            "materialize_gpu_active_ns",
                            duration_nanos(stats.gpu_active),
                        );
                        let _ = prepare_span
                            .record("completed_before_join", stats.completed_before_join);
                        let _ = prepare_span.record(
                            "joint_threads_per_threadgroup",
                            stats.threads_per_threadgroup as u64,
                        );
                        let _ = prepare_span
                            .record("joint_threadgroup_bytes", stats.threadgroup_bytes as u64);
                        (
                            sequence,
                            first_message,
                            Some((instruction, endpoints, stats)),
                            prefetched.instruction_rows,
                        )
                    }
                }
            } else {
                let mut sequence = session.take::<ProductRemainderSequence>().ok_or(
                    KernelError::InvariantViolation {
                        reason: "Metal product remainder lost its preinitialized sequence",
                    },
                )?;
                if !sequence.is_ready()
                    || sequence.storage_layout().rows() != cycles
                    || sequence.device_registry_id() != self.context.device_registry_id()
                    || sequence.device_registry_id() != carry_device_registry_id
                    || sequence.row_allocation_identity() != carry_row_storage_id
                {
                    return Err(KernelError::InvariantViolation {
                    reason:
                        "Metal product-remainder sequence has the wrong state, shape, device, or rows",
                });
                }
                let row_storage_id = sequence.row_allocation_identity();
                let (e_in, e_out) = host.current_weights()?;
                let prepare_span = tracing::info_span!(
                    "MetalProductRemainder::prepare",
                    cycles,
                    rounds,
                    resident_rows_storage_id = row_storage_id as u64,
                    row_upload_bytes = 0u64,
                    round_device_buffer_allocations = 0u64,
                    primed_device_bytes = sequence.storage_layout().workspace_bytes(),
                    sequence_prepare_wall_ns = 0u64,
                    prefetched = false,
                    materialize_wall_ns = tracing::field::Empty,
                    materialize_gpu_active_ns = tracing::field::Empty,
                );
                let _entered = prepare_span.enter();
                sequence
                    .set_lagrange_weights(host.lagrange_weights)
                    .map_err(metal_prepare_error)?;
                let started = Instant::now();
                let (first_message, materialize_gpu_active) = sequence
                    .restart_message_timed(&e_in, &e_out)
                    .map_err(metal_prepare_error)?;
                let _ =
                    prepare_span.record("materialize_wall_ns", duration_nanos(started.elapsed()));
                let _ = prepare_span.record(
                    "materialize_gpu_active_ns",
                    duration_nanos(materialize_gpu_active),
                );
                (sequence, first_message, None, None)
            };
        if sequence.storage_layout().rows() != cycles
            || sequence.device_registry_id() != self.context.device_registry_id()
            || sequence.device_registry_id() != carry_device_registry_id
            || sequence.row_allocation_identity() != carry_row_storage_id
        {
            return Err(KernelError::InvariantViolation {
                reason: "Metal product-remainder handoff changed shape, device, or rows",
            });
        }
        let row_storage_id = sequence.row_allocation_identity();
        let (state, prefetched_instruction) =
            if let Some((instruction, endpoints, stats)) = prefetched_instruction_parts {
                let service = Arc::new(Mutex::new(
                    ProductInstructionRoundService::new(sequence, instruction, &tau_low)
                        .map_err(metal_prepare_error)?,
                ));
                (
                    MetalProductRemainderState::Joint(Arc::clone(&service)),
                    Some(MetalInstructionClaimPrefetchedInitial {
                        service,
                        endpoints,
                        stats,
                    }),
                )
            } else {
                (
                    MetalProductRemainderState::Standalone(Box::new(sequence)),
                    None,
                )
            };
        let carry =
            session
                .take::<MetalProductUniskipCarry>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "Metal product uni-skip carry disappeared during handoff",
                })?;
        if carry.log_t != carry_log_t
            || carry.tau_low != tau_low
            || carry.row_storage_id != row_storage_id
            || carry.device_registry_id != self.context.device_registry_id()
        {
            return Err(KernelError::InvariantViolation {
                reason: "Metal product uni-skip carry changed during handoff",
            });
        }
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .test_counters
            .product_remainder_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if prefetched_instruction.is_some() != prefetched_instruction_rows.is_some() {
            return Err(KernelError::InvariantViolation {
                reason: "joint prefetch lost one side of its instruction handoff",
            });
        }
        let instruction_rows = if prefetched_instruction_rows.is_some() {
            if session
                .state::<MetalInstructionClaimResidentRows>()
                .is_some()
            {
                return Err(KernelError::InvariantViolation {
                    reason: "joint prefetch duplicated instruction resident rows",
                });
            }
            prefetched_instruction_rows
        } else {
            session.take::<MetalInstructionClaimResidentRows>()
        };
        let instruction_aliases = match instruction_rows {
            Some(rows)
                if rows.log_t == rounds
                    && rows.product.allocation_identity() == row_storage_id
                    && rows.device_registry_id == self.context.device_registry_id() =>
            {
                let aliases = Arc::new(Mutex::new(None));
                session.park(MetalInstructionClaimHandoff {
                    rows,
                    aliases: Arc::clone(&aliases),
                    prefetched_initial: prefetched_instruction,
                });
                Some(aliases)
            }
            Some(_) => {
                return Err(KernelError::InvariantViolation {
                    reason: "instruction claim resident rows disagree with the product sequence",
                });
            }
            None => None,
        };

        Ok(Box::new(MetalProductRemainderKernel {
            host,
            state,
            pending_endpoints: Some(first_message),
            row_storage_id,
            instruction_aliases,
            cpu_tail: None,
            cpu_tail_elements: self.config.spartan_product_remainder.cpu_tail_elements,
        }))
    }
}

fn product_prepare_fallback_reason(error: &MetalError) -> Option<&'static str> {
    if error.is_capacity_error() {
        return Some("capacity");
    }
    match error {
        MetalError::CommandFailed(_) => Some("command_failed"),
        MetalError::GpuTimestampLookup { .. } => Some("gpu_timestamp_lookup"),
        MetalError::InvalidGpuTimestamps { .. } => Some("invalid_gpu_timestamps"),
        _ => None,
    }
}

fn metal_prepare_error(error: MetalError) -> KernelError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
    .into()
}

fn metal_round_error(error: MetalError) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
}

fn metal_output_error(error: MetalError) -> SumcheckKernelError<AkitaField> {
    SumcheckKernelError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
}

fn duration_nanos(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

struct MetalProductRemainderHost {
    rounds: usize,
    tau_low: Vec<AkitaField>,
    split_eq: GruenSplitEqPolynomial<AkitaField>,
    challenges: Vec<AkitaField>,
    lagrange_weights: [AkitaField; DOMAIN],
}

impl MetalProductRemainderHost {
    fn new(
        tau_low: &[AkitaField],
        uniskip_challenge: AkitaField,
        tau_high: AkitaField,
    ) -> Result<Self, KernelError<AkitaField>> {
        let lagrange_weights = centered_lagrange_evals(DOMAIN, uniskip_challenge)?
            .try_into()
            .map_err(|_| KernelError::InvariantViolation {
                reason: "product-remainder Lagrange vector has the wrong length",
            })?;
        let scale = centered_lagrange_kernel(DOMAIN, tau_high, uniskip_challenge)?;
        Ok(Self {
            rounds: tau_low.len(),
            tau_low: tau_low.to_vec(),
            split_eq: GruenSplitEqPolynomial::new_with_scaling(
                tau_low,
                BindingOrder::LowToHigh,
                Some(scale),
            ),
            challenges: Vec::with_capacity(tau_low.len()),
            lagrange_weights,
        })
    }

    fn current_weights(
        &self,
    ) -> Result<(Vec<AkitaField>, Vec<AkitaField>), SumcheckError<AkitaField>> {
        let remaining = self.rounds.saturating_sub(self.challenges.len());
        let head_len = remaining
            .checked_sub(1)
            .ok_or_else(|| SumcheckError::ComputeBackend {
                backend: "metal",
                message: "product-remainder weights requested after the final bind".to_string(),
            })?;
        let head = &self.tau_low[..head_len];
        let split = head.len().div_ceil(2);
        let (out_point, in_point) = head.split_at(split);
        Ok((
            EqPolynomial::evals(in_point, None),
            EqPolynomial::evals(out_point, None),
        ))
    }

    fn bind(&mut self, challenge: AkitaField) {
        self.split_eq.bind(challenge);
        self.challenges.push(challenge);
    }

    fn polynomial(
        &self,
        endpoints: [AkitaField; 2],
        previous_claim: AkitaField,
    ) -> UnivariatePoly<AkitaField> {
        self.split_eq
            .gruen_poly_deg_3(endpoints[0], endpoints[1], previous_claim)
    }

    fn opening_weights(&self) -> (Vec<AkitaField>, Vec<AkitaField>) {
        let point = self.challenges.iter().rev().copied().collect::<Vec<_>>();
        let split = point.len().div_ceil(2);
        let (out_point, in_point) = point.split_at(split);
        (
            EqPolynomial::evals(in_point, None),
            EqPolynomial::evals(out_point, None),
        )
    }
}

struct ProductRemainderCpuTail {
    left: Vec<AkitaField>,
    right: Vec<AkitaField>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for ProductRemainderCpuTail {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        use crate::backend::vec_heap_bytes;
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(allocative::Key::new("left"), vec_heap_bytes(&self.left));
        visitor.visit_simple(allocative::Key::new("right"), vec_heap_bytes(&self.right));
        visitor.exit();
    }
}

impl ProductRemainderCpuTail {
    fn new(
        left: Vec<AkitaField>,
        right: Vec<AkitaField>,
    ) -> Result<Self, SumcheckError<AkitaField>> {
        if left.len() != right.len() || left.len() < 2 || !left.len().is_power_of_two() {
            return Err(SumcheckError::ComputeBackend {
                backend: "metal",
                message: "product-remainder CPU tail received malformed resident state".to_string(),
            });
        }
        Ok(Self { left, right })
    }

    fn current_elements(&self) -> usize {
        self.left.len()
    }

    fn bind_and_message(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; 2], SumcheckError<AkitaField>> {
        let destination_elements = self.current_elements() / 2;
        for index in 0..destination_elements {
            let source = 2 * index;
            let left_low = self.left[source];
            let right_low = self.right[source];
            self.left[index] = left_low + challenge * (self.left[source + 1] - left_low);
            self.right[index] = right_low + challenge * (self.right[source + 1] - right_low);
        }
        self.left.truncate(destination_elements);
        self.right.truncate(destination_elements);
        self.message(e_in, e_out)
    }

    fn message(
        &self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; 2], SumcheckError<AkitaField>> {
        if 2 * e_in.len() * e_out.len() != self.current_elements() {
            return Err(SumcheckError::ComputeBackend {
                backend: "metal",
                message: "product-remainder CPU tail equality shape drifted".to_string(),
            });
        }
        let mut endpoints = [AkitaField::zero(); 2];
        for (x_out, &outer_weight) in e_out.iter().enumerate() {
            let mut inner = [AkitaField::zero(); 2];
            for (x_in, &inner_weight) in e_in.iter().enumerate() {
                let low = 2 * (x_out * e_in.len() + x_in);
                let high = low + 1;
                let left_low = self.left[low];
                let left_high = self.left[high];
                let right_low = self.right[low];
                let right_high = self.right[high];
                inner[0] += inner_weight * (left_low * right_low);
                inner[1] += inner_weight * ((left_high - left_low) * (right_high - right_low));
            }
            endpoints[0] += outer_weight * inner[0];
            endpoints[1] += outer_weight * inner[1];
        }
        Ok(endpoints)
    }
}

enum MetalProductRemainderState {
    Standalone(Box<ProductRemainderSequence>),
    Joint(Arc<Mutex<ProductInstructionRoundService>>),
}

impl MetalProductRemainderState {
    const fn is_joint(&self) -> bool {
        matches!(self, Self::Joint(_))
    }

    fn current_elements(&self) -> Result<usize, SumcheckError<AkitaField>> {
        match self {
            Self::Standalone(sequence) => Ok(sequence.current_elements()),
            Self::Joint(service) => service
                .lock()
                .map(|service| service.product_current_elements())
                .map_err(|_| SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "joint Product/Instruction service lock is poisoned".to_string(),
                }),
        }
    }

    fn read_current_state(
        &self,
    ) -> Result<(Vec<AkitaField>, Vec<AkitaField>), SumcheckError<AkitaField>> {
        match self {
            Self::Standalone(sequence) => sequence.read_current_state().map_err(metal_round_error),
            Self::Joint(service) => service
                .lock()
                .map_err(|_| SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "joint Product/Instruction service lock is poisoned".to_string(),
                })?
                .read_product_current_state()
                .map_err(metal_round_error),
        }
    }

    fn bind_and_message(
        &mut self,
        round: usize,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<([AkitaField; 2], ProductInstructionRoundStats), SumcheckError<AkitaField>> {
        match self {
            Self::Standalone(sequence) => {
                let started = Instant::now();
                let (message, gpu_active) = sequence
                    .bind_and_message_timed(challenge, e_in, e_out)
                    .map_err(metal_round_error)?;
                Ok((
                    message,
                    ProductInstructionRoundStats {
                        wall: started.elapsed(),
                        gpu_active,
                        joint: false,
                    },
                ))
            }
            Self::Joint(service) => service
                .lock()
                .map_err(|_| SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "joint Product/Instruction service lock is poisoned".to_string(),
                })?
                .product_bind_and_message(round, challenge, e_in, e_out)
                .map_err(metal_round_error),
        }
    }

    fn openings(
        &mut self,
        after_cpu_tail: bool,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<([AkitaField; 8], Duration), MetalError> {
        match self {
            Self::Standalone(sequence) if after_cpu_tail => {
                sequence.openings_after_cpu_tail_timed(e_in, e_out)
            }
            Self::Standalone(sequence) => sequence.openings_timed(e_in, e_out),
            Self::Joint(service) => service
                .lock()
                .map_err(|_| {
                    MetalError::InvalidProductRemainderState(
                        "joint Product/Instruction service lock is poisoned",
                    )
                })?
                .product_openings(after_cpu_tail, e_in, e_out),
        }
    }
}

struct MetalProductRemainderKernel {
    host: MetalProductRemainderHost,
    state: MetalProductRemainderState,
    pending_endpoints: Option<[AkitaField; 2]>,
    row_storage_id: usize,
    instruction_aliases: Option<MetalInstructionClaimAliasSlot>,
    cpu_tail: Option<ProductRemainderCpuTail>,
    cpu_tail_elements: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalProductRemainderKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        match &self.state {
            MetalProductRemainderState::Standalone(sequence) => {
                visitor.visit_field(allocative::Key::new("sequence"), sequence);
            }
            MetalProductRemainderState::Joint(service) => {
                if let Ok(service) = service.lock() {
                    visitor.visit_field(allocative::Key::new("joint_service"), &*service);
                }
            }
        }
        if let Some(cpu_tail) = &self.cpu_tail {
            visitor.visit_field(allocative::Key::new("cpu_tail"), cpu_tail);
        }
        visitor.exit();
    }
}

impl ProveRounds<AkitaField> for MetalProductRemainderKernel {
    fn num_rounds(&self) -> usize {
        self.host.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        let endpoints = if let Some(challenge) = bind {
            let source_elements = if let Some(tail) = &self.cpu_tail {
                tail.current_elements()
            } else {
                self.state.current_elements()?
            };
            if self.cpu_tail.is_none()
                && source_elements <= self.cpu_tail_elements
                && source_elements > 2
            {
                let span = tracing::info_span!(
                    "MetalProductRemainder::cpu_tail_handoff",
                    round,
                    source_elements,
                    readback_bytes = source_elements
                        .saturating_mul(2)
                        .saturating_mul(std::mem::size_of::<AkitaField>()),
                    wall_ns = tracing::field::Empty,
                );
                let _entered = span.enter();
                let started = Instant::now();
                let (left, right) = self.state.read_current_state()?;
                self.cpu_tail = Some(ProductRemainderCpuTail::new(left, right)?);
                let _ = span.record("wall_ns", duration_nanos(started.elapsed()));
            }
            self.host.bind(challenge);
            if self.host.challenges.len() != round {
                return Err(SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "product-remainder round order drifted".to_string(),
                });
            }
            let (e_in, e_out) = self.host.current_weights()?;
            if let Some(cpu_tail) = &mut self.cpu_tail {
                let span = tracing::info_span!(
                    "MetalProductRemainder::cpu_tail_round",
                    round,
                    source_elements,
                    wall_ns = tracing::field::Empty,
                );
                let _entered = span.enter();
                let started = Instant::now();
                let message = cpu_tail.bind_and_message(challenge, &e_in, &e_out)?;
                let _ = span.record("wall_ns", duration_nanos(started.elapsed()));
                message
            } else {
                let span = tracing::info_span!(
                    "MetalProductRemainder::bind_and_message",
                    round,
                    source_elements,
                    joint_product_instruction = self.state.is_joint(),
                    resident_rows_storage_id = self.row_storage_id as u64,
                    dispatch_wall_ns = tracing::field::Empty,
                    gpu_active_ns = tracing::field::Empty,
                );
                let _entered = span.enter();
                let (message, stats) = self
                    .state
                    .bind_and_message(round, challenge, &e_in, &e_out)?;
                let _ = span.record("dispatch_wall_ns", duration_nanos(stats.wall));
                let _ = span.record("gpu_active_ns", duration_nanos(stats.gpu_active));
                message
            }
        } else {
            if round != 0 || !self.host.challenges.is_empty() {
                return Err(SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "product-remainder first message was requested out of order"
                        .to_string(),
                });
            }
            self.pending_endpoints
                .take()
                .ok_or_else(|| SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "product-remainder first message was already consumed".to_string(),
                })?
        };
        Ok(self.host.polynomial(endpoints, previous_claim))
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        self.host.bind(bind);
        let terminal_elements = if let Some(tail) = &self.cpu_tail {
            tail.current_elements()
        } else {
            self.state.current_elements()?
        };
        if self.host.challenges.len() != self.host.rounds || terminal_elements != 2 {
            return Err(SumcheckError::ComputeBackend {
                backend: "metal",
                message: "product-remainder sequence did not reach its terminal state".to_string(),
            });
        }
        Ok(())
    }
}

impl SumcheckKernel<AkitaField> for MetalProductRemainderKernel {
    type Relation = ProductRemainder<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<AkitaField, Self::Relation>, SumcheckKernelError<AkitaField>>
    {
        let remaining = self.host.rounds.saturating_sub(self.host.challenges.len());
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let (e_in, e_out) = self.host.opening_weights();
        let span = tracing::info_span!(
            "MetalProductRemainder::output_claims",
            resident_rows_storage_id = self.row_storage_id as u64,
            row_upload_bytes = 0u64,
            dispatch_wall_ns = tracing::field::Empty,
            gpu_active_ns = tracing::field::Empty,
        );
        let _entered = span.enter();
        let started = Instant::now();
        let (values, gpu_active) = self
            .state
            .openings(self.cpu_tail.is_some(), &e_in, &e_out)
            .map_err(metal_output_error)?;
        let _ = span.record("dispatch_wall_ns", duration_nanos(started.elapsed()));
        let _ = span.record("gpu_active_ns", duration_nanos(gpu_active));
        if let Some(slot) = &self.instruction_aliases {
            let mut slot = slot
                .lock()
                .map_err(|_| SumcheckKernelError::ComputeBackend {
                    backend: "metal",
                    message: "instruction claim alias slot lock is poisoned".to_string(),
                })?;
            if slot.is_some() {
                return Err(SumcheckKernelError::ComputeBackend {
                    backend: "metal",
                    message: "instruction claim aliases were already published".to_string(),
                });
            }
            *slot = Some(MetalInstructionClaimAliasOutput {
                row_storage_id: self.row_storage_id,
                challenges: self.host.challenges.clone(),
                values: MetalInstructionClaimAliases {
                    lookup_output: values[4],
                    left_instruction_input: values[0],
                    right_instruction_input: values[1],
                },
            });
        }
        let ids = [
            left_instruction_input_product(),
            right_instruction_input_product(),
            jump_flag_product(),
            write_lookup_output_to_rd_product(),
            lookup_output_product(),
            branch_flag_product(),
            next_is_noop_product(),
            virtual_instruction_product(),
        ];
        let claims: BTreeMap<JoltOpeningId, AkitaField> = ids.into_iter().zip(values).collect();
        SumcheckOutputClaims::<AkitaField, Self::Relation>::from_opening_values(|id| {
            claims.get(id).copied().or_else(|| inputs.resolve_input(id))
        })
        .map_err(SumcheckKernelError::from)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        let remaining = self.host.rounds.saturating_sub(self.host.challenges.len());
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let ids = std::iter::once(SpartanProductVirtualizationPublic::TauKernel)
            .chain((0..DOMAIN).map(SpartanProductVirtualizationPublic::LagrangeWeight));
        for public_id in ids {
            let id = JoltDerivedId::from(public_id);
            let expected =
                match relation.derive_output_term(&id, input_points, output_points, challenges) {
                    Ok(value) => value,
                    Err(VerifierError::MissingStageClaimDerived { .. }) => continue,
                    Err(error) => return Err(error.into()),
                };
            let got = match public_id {
                SpartanProductVirtualizationPublic::TauKernel => {
                    self.host.split_eq.current_scalar()
                }
                SpartanProductVirtualizationPublic::LagrangeWeight(index) => {
                    self.host.lagrange_weights[index]
                }
                SpartanProductVirtualizationPublic::UniskipLagrangeWeight(_) => continue,
            };
            if got != expected {
                return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::spartan::SpartanProductDimensions;
    use jolt_claims::NoChallenges;
    use jolt_verifier::stages::stage2::product_remainder::{
        product_remainder_input_values_from_uniskip_output, ProductRemainderInputClaims,
    };
    use jolt_witness::testing::with_sample_backend_at_log_t;

    use super::*;
    use crate::optimized::spartan_product::OptimizedProductUniskip;
    use crate::uniskip::UniskipKernel;

    fn true_input_claim(
        rows: &[ProductRemainderRow],
        tau_low: &[AkitaField],
        tau_high: AkitaField,
        uniskip_challenge: AkitaField,
    ) -> AkitaField {
        let eq = EqPolynomial::<AkitaField>::evals(tau_low, None);
        let weights: [AkitaField; DOMAIN] = centered_lagrange_evals(DOMAIN, uniskip_challenge)
            .unwrap()
            .try_into()
            .unwrap();
        let scale = centered_lagrange_kernel(DOMAIN, tau_high, uniskip_challenge).unwrap();
        scale
            * rows
                .iter()
                .zip(eq)
                .map(|(&row, eq)| {
                    let (left, right) = row.relation_values(&weights);
                    eq * left * right
                })
                .sum::<AkitaField>()
    }

    fn known_t1_values(
        rows: &[ProductRemainderRow],
        tau_low: &[AkitaField],
    ) -> [AkitaField; DOMAIN] {
        let eq = EqPolynomial::<AkitaField>::evals(tau_low, None);
        let mut values = [AkitaField::zero(); DOMAIN];
        for (&row, weight) in rows.iter().zip(eq) {
            for node in 0..DOMAIN {
                let mut basis = [AkitaField::zero(); DOMAIN];
                basis[node] = AkitaField::one();
                let (left, right) = row.relation_values(&basis);
                values[node] += weight * left * right;
            }
        }
        values
    }

    #[test]
    fn resident_product_remainder_matches_optimized_cpu() {
        for log_t in [4usize, 5] {
            with_sample_backend_at_log_t(log_t, 4, |witness| {
                let tau_low = (0..log_t)
                    .map(|index| AkitaField::from_u64(19 + 7 * index as u64))
                    .collect::<Vec<_>>();
                let tau_high = AkitaField::from_u64(313);
                let uniskip_challenge = AkitaField::from_u64(911);
                let rows = collect_bundles::<SpartanProductRow>(witness, 1 << log_t)
                    .unwrap()
                    .iter()
                    .map(ProductRemainderRow::from)
                    .collect::<Vec<_>>();
                let known_values = known_t1_values(&rows, &tau_low);
                let input_claim = true_input_claim(&rows, &tau_low, tau_high, uniskip_challenge);
                let relation = ProductRemainder::new(
                    SpartanProductDimensions::new(log_t),
                    uniskip_challenge,
                    tau_high,
                    tau_low.clone(),
                );
                let claims = product_remainder_input_values_from_uniskip_output(input_claim);
                let points = ProductRemainderInputClaims::<Vec<AkitaField>>::default();
                let no_challenges = NoChallenges::<AkitaField>::default();

                let mut optimized_session = ProofSession::default();
                OptimizedProductUniskip
                    .prepare(&mut optimized_session, log_t, &tau_low, witness)
                    .unwrap();
                let expected_uniskip = OptimizedProductUniskip
                    .first_round_poly(&mut optimized_session, &[tau_high], &known_values)
                    .unwrap();
                let mut optimized = OptimizedProductRemainder
                    .prepare(
                        &mut optimized_session,
                        witness,
                        ProverInputs {
                            relation: &relation,
                            claims: &claims,
                            points: &points,
                            challenges: &no_challenges,
                        },
                    )
                    .unwrap();

                let mut config = super::super::MetalConfig::default();
                config.spartan_product_remainder.trace_cutoff_elements = 2;
                let metal = MetalBackend::new(config).unwrap();
                let mut metal_session = ProofSession::default();
                let stage1 =
                    crate::optimized::spartan_outer::prepare_metal_spartan_outer_witness_rows(
                        &metal.context,
                        witness,
                        1 << log_t,
                    )
                    .unwrap();
                metal_session.park(stage1);
                metal
                    .prepare_product_remainder_witness(&mut metal_session, log_t, witness)
                    .unwrap();
                <MetalBackend as UniskipKernel<AkitaField, ProductRemainder<AkitaField>>>::prepare(
                    &metal,
                    &mut metal_session,
                    log_t,
                    &tau_low,
                    witness,
                )
                .unwrap();
                let actual_uniskip = <MetalBackend as UniskipKernel<
                    AkitaField,
                    ProductRemainder<AkitaField>,
                >>::first_round_poly(
                    &metal, &mut metal_session, &[tau_high], &known_values
                )
                .unwrap();
                assert_eq!(actual_uniskip, expected_uniskip);
                assert_eq!(metal.product_uniskip_dispatches(), 1);
                let resident_row_id = metal_session
                    .state::<ProductRemainderSequence>()
                    .unwrap()
                    .row_allocation_identity();
                assert_eq!(
                    metal_session
                        .state::<MetalProductUniskipCarry>()
                        .unwrap()
                        .row_storage_id,
                    resident_row_id
                );
                <MetalBackend as PrepareKernel<
                    AkitaField,
                    ProductRemainder<AkitaField>,
                >>::prefetch_relation(&metal, &mut metal_session, &relation)
                .unwrap();
                assert!(metal_session
                    .state::<MetalProductRemainderPrefetch>()
                    .is_some());
                assert!(metal_session.state::<ProductRemainderSequence>().is_none());
                let mut actual = <MetalBackend as PrepareKernel<
                    AkitaField,
                    ProductRemainder<AkitaField>,
                >>::prepare(
                    &metal,
                    &mut metal_session,
                    witness,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &no_challenges,
                    },
                )
                .unwrap();
                assert_eq!(metal.product_remainder_sequences(), 1);

                let challenges = (0..log_t)
                    .map(|index| AkitaField::from_u64(1201 + 43 * index as u64))
                    .collect::<Vec<_>>();
                let mut bind = None;
                let mut previous_claim = input_claim;
                for (round, &challenge) in challenges.iter().enumerate() {
                    let expected = optimized.prove_round(bind, round, previous_claim).unwrap();
                    let got = actual.prove_round(bind, round, previous_claim).unwrap();
                    assert_eq!(got, expected, "round {round}");
                    previous_claim = expected.evaluate(challenge);
                    bind = Some(challenge);
                }
                let final_challenge = *challenges.last().unwrap();
                optimized.finish_rounds(final_challenge).unwrap();
                actual.finish_rounds(final_challenge).unwrap();
                assert_eq!(
                    actual.output_claims(&claims).unwrap(),
                    optimized.output_claims(&claims).unwrap()
                );

                let output_points = relation
                    .derive_opening_points(&challenges, &points)
                    .unwrap();
                optimized
                    .validate_derived_tables(&relation, &points, &output_points, &no_challenges)
                    .unwrap();
                actual
                    .validate_derived_tables(&relation, &points, &output_points, &no_challenges)
                    .unwrap();
            });
        }
    }
}
