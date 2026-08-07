use std::{cell::RefCell, collections::BTreeMap, rc::Rc, time::Instant};

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
use jolt_riscv::{CircuitFlags, InstructionFlags};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
use jolt_verifier::VerifierError;
use jolt_witness::witnesses::{
    InstructionFlag, LeftInstructionInput, LeftLookupOperand, LookupOutput, NextIsNoop, OpFlag,
    RightInstructionInput, RightLookupOperand,
};
use jolt_witness::{collect_bundles, JoltWitnessPlane, WitnessBundle};

use super::backend::MetalBackend;
use super::solinas::{
    instruction_claim_reduction::{InstructionClaimLookupOperandRow, InstructionClaimLookupRows},
    MetalError, ProductRemainderRow, ProductRemainderRows, ProductRemainderSequence,
    ProductRemainderSequenceConfig,
};
#[cfg(test)]
use crate::optimized::spartan_product::SpartanProductRow;
use crate::optimized::spartan_product::{OptimizedProductRemainder, OptimizedProductUniskip};
use crate::uniskip::UniskipKernel;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const DOMAIN: usize = PRODUCT_UNISKIP_DOMAIN_SIZE;
const EXTENDED_SIZE: usize = 2 * DOMAIN - 1;
const DOMAIN_START: i64 = -((DOMAIN as i64 - 1) / 2);
const EXTENDED_START: i64 = -((EXTENDED_SIZE as i64 - 1) / 2);

#[derive(Clone, Copy, Debug, WitnessBundle)]
struct Stage2ProductInstructionRow {
    #[opening(LeftInstructionInput)]
    left_instruction_input: LeftInstructionInput,
    #[opening(RightInstructionInput)]
    right_instruction_input: RightInstructionInput,
    #[opening(OpFlags(CircuitFlags::Jump))]
    jump_flag: OpFlag,
    #[opening(OpFlags(CircuitFlags::WriteLookupOutputToRD))]
    write_lookup_output_to_rd: OpFlag,
    #[opening(LookupOutput)]
    lookup_output: LookupOutput,
    #[opening(InstructionFlags(InstructionFlags::Branch))]
    branch_flag: InstructionFlag,
    #[opening(NextIsNoop)]
    next_is_noop: NextIsNoop,
    #[opening(OpFlags(CircuitFlags::VirtualInstruction))]
    virtual_instruction: OpFlag,
    #[opening(LeftLookupOperand)]
    left_lookup_operand: LeftLookupOperand,
    #[opening(RightLookupOperand)]
    right_lookup_operand: RightLookupOperand,
}

impl Stage2ProductInstructionRow {
    fn product(self) -> ProductRemainderRow {
        ProductRemainderRow::new(
            self.left_instruction_input.0,
            self.right_instruction_input.0,
            self.jump_flag.0,
            self.write_lookup_output_to_rd.0,
            self.lookup_output.0,
            self.branch_flag.0,
            self.next_is_noop.0,
            self.virtual_instruction.0,
        )
    }

    fn lookup(self) -> InstructionClaimLookupOperandRow {
        InstructionClaimLookupOperandRow::new(
            self.left_lookup_operand.0,
            self.right_lookup_operand.0,
        )
    }
}

pub(super) struct MetalInstructionClaimResidentRows {
    pub(super) log_t: usize,
    pub(super) product: ProductRemainderRows,
    pub(super) lookup: InstructionClaimLookupRows,
    pub(super) device_registry_id: u64,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalInstructionClaimResidentRows {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(allocative::Key::new("product"), &self.product);
        visitor.visit_field(allocative::Key::new("lookup"), &self.lookup);
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
    Rc<RefCell<Option<MetalInstructionClaimAliasOutput>>>;

pub(super) struct MetalInstructionClaimHandoff {
    pub(super) rows: MetalInstructionClaimResidentRows,
    pub(super) aliases: MetalInstructionClaimAliasSlot,
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
    pub dispatch: ProductRemainderSequenceConfig,
}

impl Default for SpartanProductRemainderMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 18,
            dispatch: ProductRemainderSequenceConfig::default(),
        }
    }
}

impl MetalBackend {
    pub(super) fn prepare_product_remainder_witness(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        witness: &dyn JoltWitnessPlane<AkitaField>,
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
            lookup_companion_bytes =
                cycles.saturating_mul(std::mem::size_of::<InstructionClaimLookupOperandRow>()),
            collect_wall_ns = tracing::field::Empty,
            upload_wall_ns = tracing::field::Empty,
            lookup_upload_wall_ns = tracing::field::Empty,
            sequence_prepare_wall_ns = tracing::field::Empty,
            workspace_bytes = tracing::field::Empty,
            primer_materialize_wall_ns = tracing::field::Empty,
            primer_materialize_gpu_active_ns = tracing::field::Empty,
            primer_transition_wall_ns = tracing::field::Empty,
            primer_transition_gpu_active_ns = tracing::field::Empty,
            resident_rows_storage_id = tracing::field::Empty,
            admitted = tracing::field::Empty,
            fallback_reason = tracing::field::Empty,
        );
        let _entered = span.enter();
        let started = Instant::now();
        let rows: Vec<Stage2ProductInstructionRow> = collect_bundles(witness, cycles)?;
        let packed = rows
            .iter()
            .copied()
            .map(Stage2ProductInstructionRow::product)
            .collect::<Vec<_>>();
        let lookup = rows
            .iter()
            .copied()
            .map(Stage2ProductInstructionRow::lookup)
            .collect::<Vec<_>>();
        drop(rows);
        let _ = span.record("collect_wall_ns", duration_nanos(started.elapsed()));

        let started = Instant::now();
        let rows = match self.context.prepare_product_remainder_rows(&packed) {
            Ok(rows) => rows,
            Err(error) if error.is_capacity_error() => {
                let _ = span.record("upload_wall_ns", duration_nanos(started.elapsed()));
                let _ = span.record("admitted", false);
                let _ = span.record("fallback_reason", "capacity");
                tracing::warn!(
                    target: "jolt::metal",
                    error = %error,
                    "product-remainder resident rows were not admitted; using optimized CPU"
                );
                return Ok(());
            }
            Err(error) => return Err(metal_prepare_error(error)),
        };
        let _ = span.record("upload_wall_ns", duration_nanos(started.elapsed()));
        let row_storage_id = rows.allocation_identity();
        let _ = span.record("resident_rows_storage_id", row_storage_id as u64);
        drop(packed);

        let started = Instant::now();
        let lookup = match self.context.prepare_instruction_claim_lookup_rows(&lookup) {
            Ok(rows) => Some(rows),
            Err(error) if error.is_capacity_error() => {
                tracing::warn!(
                    target: "jolt::metal",
                    error = %error,
                    "instruction claim companion rows were not admitted; using optimized CPU for that member"
                );
                None
            }
            Err(error) => return Err(metal_prepare_error(error)),
        };
        let _ = span.record("lookup_upload_wall_ns", duration_nanos(started.elapsed()));

        let e_in_capacity = 1usize << (log_t / 2);
        let e_out_capacity = cycles / e_in_capacity;
        let started = Instant::now();
        let instruction_product_rows = rows.clone();
        let sequence = self.context.prepare_product_remainder_sequence_with_rows(
            rows,
            [AkitaField::zero(); DOMAIN],
            e_in_capacity,
            e_out_capacity,
            self.config.spartan_product_remainder.dispatch,
        );
        let _ = span.record(
            "sequence_prepare_wall_ns",
            duration_nanos(started.elapsed()),
        );
        let mut sequence = match sequence {
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
        let _ = span.record(
            "workspace_bytes",
            sequence.storage_layout().workspace_bytes(),
        );
        let primer = match sequence.prime() {
            Ok(primer) => primer,
            Err(error) if product_prepare_fallback_reason(&error).is_some() => {
                let _ = span.record("admitted", false);
                let _ = span.record("fallback_reason", "pipeline_primer");
                tracing::warn!(
                    target: "jolt::metal",
                    error = %error,
                    "product-remainder pipeline primer failed; using optimized CPU"
                );
                return Ok(());
            }
            Err(error) => return Err(metal_prepare_error(error)),
        };
        let _ = span.record(
            "primer_materialize_wall_ns",
            duration_nanos(primer.materialize_wall()),
        );
        let _ = span.record(
            "primer_materialize_gpu_active_ns",
            duration_nanos(primer.materialize_gpu_active()),
        );
        let _ = span.record(
            "primer_transition_wall_ns",
            duration_nanos(primer.transition_wall()),
        );
        let _ = span.record(
            "primer_transition_gpu_active_ns",
            duration_nanos(primer.transition_gpu_active()),
        );
        if !sequence.is_primed() || sequence.row_allocation_identity() != row_storage_id {
            return Err(KernelError::InvariantViolation {
                reason: "product-remainder pipeline primer ended in the wrong state",
            });
        }
        let _ = span.record("admitted", true);
        let _ = span.record("fallback_reason", "none");
        if let Some(lookup) = lookup {
            session.park(MetalInstructionClaimResidentRows {
                log_t,
                product: instruction_product_rows,
                lookup,
                device_registry_id: self.context.device_registry_id(),
            });
        }
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
        if !sequence.is_primed()
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
                .product_uniskip_dispatches
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            endpoints.as_array()
        };
        if sequence.row_allocation_identity() != row_storage_id || !sequence.is_primed() {
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
            || !sequence.is_primed()
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
            drop(session.take::<MetalInstructionClaimResidentRows>());
            drop(session.take::<MetalInstructionClaimHandoff>());
            return OptimizedProductRemainder.prepare(session, witness, inputs);
        }
        if cycles < self.config.spartan_product_remainder.trace_cutoff_elements
            || session.state::<ProductRemainderSequence>().is_none()
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
        let mut sequence =
            session
                .take::<ProductRemainderSequence>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "Metal product remainder lost its preinitialized sequence",
                })?;
        if sequence.storage_layout().rows() != cycles
            || sequence.device_registry_id() != self.context.device_registry_id()
            || sequence.device_registry_id() != carry_device_registry_id
            || sequence.row_allocation_identity() != carry_row_storage_id
        {
            return Err(KernelError::InvariantViolation {
                reason: "Metal product-remainder sequence has the wrong shape, device, or rows",
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
            materialize_wall_ns = tracing::field::Empty,
            materialize_gpu_active_ns = tracing::field::Empty,
        );
        let _entered = prepare_span.enter();
        sequence
            .set_lagrange_weights(host.lagrange_weights)
            .map_err(metal_prepare_error)?;
        let started = Instant::now();
        let first_message = sequence.restart_message_timed(&e_in, &e_out);
        let materialize_wall = started.elapsed();
        let (first_message, materialize_gpu_active) = first_message.map_err(metal_prepare_error)?;
        let _ = prepare_span.record("materialize_wall_ns", duration_nanos(materialize_wall));
        let _ = prepare_span.record(
            "materialize_gpu_active_ns",
            duration_nanos(materialize_gpu_active),
        );
        if sequence.row_allocation_identity() != row_storage_id {
            return Err(KernelError::InvariantViolation {
                reason: "product-remainder sequence changed the resident row allocation",
            });
        }
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
            .product_remainder_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let instruction_aliases = match session.take::<MetalInstructionClaimResidentRows>() {
            Some(rows)
                if rows.log_t == rounds
                    && rows.product.allocation_identity() == row_storage_id
                    && rows.device_registry_id == self.context.device_registry_id() =>
            {
                let aliases = Rc::new(RefCell::new(None));
                session.park(MetalInstructionClaimHandoff {
                    rows,
                    aliases: Rc::clone(&aliases),
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
            sequence,
            pending_endpoints: Some(first_message),
            row_storage_id,
            instruction_aliases,
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

struct MetalProductRemainderKernel {
    host: MetalProductRemainderHost,
    sequence: ProductRemainderSequence,
    pending_endpoints: Option<[AkitaField; 2]>,
    row_storage_id: usize,
    instruction_aliases: Option<MetalInstructionClaimAliasSlot>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalProductRemainderKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(allocative::Key::new("sequence"), &self.sequence);
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
            self.host.bind(challenge);
            if self.host.challenges.len() != round {
                return Err(SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "product-remainder round order drifted".to_string(),
                });
            }
            let source_elements = self.sequence.current_elements();
            let (e_in, e_out) = self.host.current_weights()?;
            let span = tracing::info_span!(
                "MetalProductRemainder::bind_and_message",
                round,
                source_elements,
                resident_rows_storage_id = self.row_storage_id as u64,
                dispatch_wall_ns = tracing::field::Empty,
                gpu_active_ns = tracing::field::Empty,
            );
            let _entered = span.enter();
            let started = Instant::now();
            let (message, gpu_active) = self
                .sequence
                .bind_and_message_timed(challenge, &e_in, &e_out)
                .map_err(metal_round_error)?;
            let _ = span.record("dispatch_wall_ns", duration_nanos(started.elapsed()));
            let _ = span.record("gpu_active_ns", duration_nanos(gpu_active));
            message
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
        if self.host.challenges.len() != self.host.rounds || self.sequence.current_elements() != 2 {
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
            .sequence
            .openings_timed(&e_in, &e_out)
            .map_err(metal_output_error)?;
        let _ = span.record("dispatch_wall_ns", duration_nanos(started.elapsed()));
        let _ = span.record("gpu_active_ns", duration_nanos(gpu_active));
        if let Some(slot) = &self.instruction_aliases {
            let mut slot =
                slot.try_borrow_mut()
                    .map_err(|_| SumcheckKernelError::ComputeBackend {
                        backend: "metal",
                        message: "instruction claim alias slot is already borrowed".to_string(),
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
