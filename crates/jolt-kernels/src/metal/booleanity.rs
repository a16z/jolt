use std::mem::size_of;

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_field::Zero as _;
use jolt_sumcheck::{ProveRounds, RoundExecutionDomain, SumcheckError};
use jolt_verifier::stages::relations::{
    SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage6a::booleanity::BooleanityAddressPhase;
use jolt_verifier::stages::stage6b::booleanity::{Booleanity, BooleanityCyclePhaseChallenges};
use jolt_witness::JoltWitnessPlane;

use super::backend::MetalBackend;
use super::solinas::{
    BooleanityAddressPushforwardConfig, BooleanityRows, BooleanitySequence,
    BooleanitySequenceConfig, MetalError, BOOLEANITY_SOURCE_ROW_BYTES,
};
use crate::optimized::booleanity::{
    prepare_metal_booleanity_cycle, prepare_optimized_booleanity_cycle, BooleanityAddressMetalPlan,
    OptimizedBooleanityAddress, OptimizedBooleanityCycleKernel,
};
use crate::optimized::instruction_read_raf::InstructionCycleRow;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BooleanityAddressMetalConfig {
    pub trace_cutoff_elements: usize,
    pub dispatch: BooleanityAddressPushforwardConfig,
}

impl Default for BooleanityAddressMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 18,
            dispatch: BooleanityAddressPushforwardConfig {
                inner_log2: 15,
                selectors_per_tile: 6,
                tile_threads_per_threadgroup: Some(512),
                finalize_threads_per_threadgroup: Some(1024),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BooleanityMetalConfig {
    pub trace_cutoff_elements: usize,
    pub cutoff_elements: usize,
    pub dispatch: BooleanitySequenceConfig,
}

impl Default for BooleanityMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 18,
            cutoff_elements: 1 << 10,
            dispatch: BooleanitySequenceConfig {
                threads_per_threadgroup: Some(256),
                dense_threads_per_threadgroup: Some(128),
                materialize_width: 8,
            },
        }
    }
}

impl PrepareKernel<AkitaField, BooleanityAddressPhase<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, BooleanityAddressPhase<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = BooleanityAddressPhase<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let cpu_inputs = || ProverInputs {
            relation: inputs.relation,
            claims: inputs.claims,
            points: inputs.points,
            challenges: inputs.challenges,
        };
        let cpu = |session: &mut ProofSession| {
            OptimizedBooleanityAddress.prepare(session, witness, cpu_inputs())
        };
        let dimensions = inputs.relation.dimensions();
        let trace_elements = 1usize << dimensions.log_t;
        let config = self.config.booleanity_address;
        if trace_elements < config.trace_cutoff_elements
            || dimensions.log_k_chunk != 8
            || config.dispatch.inner_log2 > dimensions.log_t
        {
            return cpu(session);
        }
        let resident_rows = match session.state::<BooleanityRows>().cloned() {
            Some(rows)
                if rows.len() == trace_elements
                    && rows.device_registry_id() == self.context.device_registry_id() =>
            {
                rows
            }
            _ => return cpu(session),
        };
        let _span = tracing::info_span!("MetalBooleanityAddressPhase::prepare").entered();
        let plan = BooleanityAddressMetalPlan::new(witness, inputs.relation, inputs.challenges)?;
        prepare_accepted_booleanity_address(self, session, resident_rows, plan, config, cpu)
    }
}

fn prepare_accepted_booleanity_address(
    backend: &MetalBackend,
    session: &mut ProofSession,
    resident_rows: BooleanityRows,
    plan: BooleanityAddressMetalPlan,
    config: BooleanityAddressMetalConfig,
    cpu: impl FnOnce(
        &mut ProofSession,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = BooleanityAddressPhase<AkitaField>>>,
        KernelError<AkitaField>,
    >,
) -> Result<
    Box<dyn SumcheckKernel<AkitaField, Relation = BooleanityAddressPhase<AkitaField>>>,
    KernelError<AkitaField>,
> {
    let resident_row_identity = resident_rows.allocation_identity();
    let trace_elements = resident_rows.len();
    let resident_row_bytes = BOOLEANITY_SOURCE_ROW_BYTES;
    let e_in_elements = 1usize << config.dispatch.inner_log2;
    let e_out_elements = trace_elements / e_in_elements;
    let selector_bytes = plan.selectors().len() * size_of::<[u32; 2]>();
    let e_in_bytes = e_in_elements * size_of::<AkitaField>();
    let e_out_bytes = e_out_elements * size_of::<AkitaField>();
    let partial_bytes =
        e_out_elements * config.dispatch.selectors_per_tile * 256 * size_of::<AkitaField>();
    let output_bytes = plan.selectors().len() * 256 * size_of::<AkitaField>();
    let planned_device_bytes =
        selector_bytes + e_in_bytes + e_out_bytes + partial_bytes + output_bytes;
    let device = backend.context.device_info();
    let requested_tile_threads = config.dispatch.tile_threads_per_threadgroup.unwrap_or(0);
    let requested_finalize_threads = config
        .dispatch
        .finalize_threads_per_threadgroup
        .unwrap_or(0);
    let sequence_span = tracing::info_span!(
        "MetalBooleanityAddressPhase::sequence_prepare",
        resident_rows_storage_id = resident_row_identity,
        resident_rows = trace_elements,
        resident_row_bytes,
        row_upload_bytes = 0u64,
        polys = plan.selectors().len(),
        k = 256usize,
        e_in_elements,
        e_out_elements,
        requested_inner_log2 = config.dispatch.inner_log2,
        effective_inner_log2 = config.dispatch.inner_log2,
        requested_selectors_per_tile = config.dispatch.selectors_per_tile,
        effective_selectors_per_tile = tracing::field::Empty,
        requested_tile_threads,
        effective_tile_threads = tracing::field::Empty,
        requested_finalize_threads,
        effective_finalize_threads = tracing::field::Empty,
        selector_tiles = tracing::field::Empty,
        production_specialized = tracing::field::Empty,
    );
    let sequence_guard = sequence_span.enter();
    let allocation_span = tracing::info_span!(
        "MetalBooleanityAddressPhase::allocation_plan",
        device_buffers = 5u64,
        planned_device_bytes,
        current_device_bytes = device.current_allocated_size,
        recommended_device_bytes = device.recommended_max_working_set_size,
    );
    let allocation_guard = allocation_span.enter();
    let invocation = match backend.context.prepare_booleanity_address_pushforward(
        resident_rows,
        plan.selectors(),
        plan.reference_cycle(),
        config.dispatch,
    ) {
        Ok(invocation) => invocation,
        Err(error) if booleanity_address_can_fallback(&error) => {
            tracing::warn!(error = %error, "Booleanity address Metal preparation fell back to CPU");
            return cpu(session);
        }
        Err(error) => return Err(metal_error(error.to_string()).into()),
    };
    drop(allocation_guard);
    let _ = sequence_span.record(
        "effective_selectors_per_tile",
        invocation.selectors_per_tile(),
    );
    let _ = sequence_span.record(
        "effective_tile_threads",
        invocation.tile_threads_per_threadgroup(),
    );
    let _ = sequence_span.record(
        "effective_finalize_threads",
        invocation.finalize_threads_per_threadgroup(),
    );
    let _ = sequence_span.record("selector_tiles", invocation.selector_tiles());
    let _ = sequence_span.record(
        "production_specialized",
        invocation.uses_production_specialization(),
    );
    drop(sequence_guard);

    let dispatch_span = tracing::info_span!(
        "MetalBooleanityAddressPhase::dispatch",
        command_buffers = 1u64,
        tile_dispatches = invocation.selector_tiles(),
        finalize_dispatches = invocation.selector_tiles(),
        command_completed = tracing::field::Empty,
        gpu_active_ns = tracing::field::Empty,
        resident_rows_storage_id = resident_row_identity,
    );
    let dispatch_guard = dispatch_span.enter();
    let gpu_active = invocation
        .execute_timed()
        .map_err(|error| metal_error(error.to_string()))?;
    let gpu_active_ns = u64::try_from(gpu_active.as_nanos()).unwrap_or(u64::MAX);
    let _ = dispatch_span.record("command_completed", true);
    let _ = dispatch_span.record("gpu_active_ns", gpu_active_ns);
    drop(dispatch_guard);

    let readback_span = tracing::info_span!(
        "MetalBooleanityAddressPhase::readback",
        elements = invocation.output_elements(),
        bytes = invocation.output_elements() * size_of::<AkitaField>(),
        readbacks = 1u64,
    );
    let readback_guard = readback_span.enter();
    let masses = invocation
        .read_masses()
        .map_err(|error| metal_error(error.to_string()))?;
    drop(readback_guard);
    plan.finish(masses)
}

impl PrepareKernel<AkitaField, Booleanity<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, Booleanity<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = Booleanity<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let dimensions = inputs.relation.dimensions();
        let trace_elements = 1usize << dimensions.log_t;
        let hamming_config = self.config.hamming_weight_claim_reduction;
        let retain_for_hamming =
            hamming_config.admits(trace_elements, dimensions.log_t, dimensions.log_k_chunk);
        let has_resident_rows = trace_elements
            >= self.config.booleanity_cycle.trace_cutoff_elements
            && session.state::<BooleanityRows>().is_some_and(|rows| {
                rows.len() == trace_elements && self.context.validate_booleanity_rows(rows).is_ok()
            });
        let mut cpu = if has_resident_rows {
            prepare_metal_booleanity_cycle(witness, inputs)?
        } else {
            prepare_optimized_booleanity_cycle(session, witness, inputs)?
        };
        if trace_elements < self.config.booleanity_cycle.trace_cutoff_elements {
            let retained_rows = session.state::<BooleanityRows>().cloned();
            match retained_rows {
                Some(rows)
                    if retain_for_hamming
                        && rows.len() == trace_elements
                        && self.context.validate_booleanity_rows(&rows).is_ok() => {}
                Some(_) => {
                    let _ = session.take::<BooleanityRows>();
                }
                None => {}
            }
            return Ok(Box::new(cpu));
        }
        let resident_rows = match session.state::<BooleanityRows>().cloned() {
            Some(rows)
                if rows.len() == trace_elements
                    && self.context.validate_booleanity_rows(&rows).is_ok() =>
            {
                rows
            }
            _ => {
                let _ = session.take::<BooleanityRows>();
                let source = cpu.metal_row_source()?;
                match self
                    .context
                    .prepare_booleanity_rows(InstructionCycleRow::metal_booleanity_rows(source))
                {
                    Ok(rows) => rows,
                    Err(error) if error.is_capacity_error() => {
                        tracing::warn!(
                            target: "jolt::metal",
                            error = %error,
                            "Booleanity cycle resident rows were not admitted"
                        );
                        return Ok(Box::new(cpu));
                    }
                    Err(error) => return Err(metal_error(error.to_string()).into()),
                }
            }
        };
        if retain_for_hamming {
            session.park(resident_rows.clone());
        } else {
            let _ = session.take::<BooleanityRows>();
        }
        let mut dispatch = self.config.booleanity_cycle.dispatch;
        if trace_elements >= 1 << 28 {
            dispatch.materialize_width = 32;
        }
        let sequence = cpu.metal_offload(&self.context, resident_rows, dispatch)?;
        Ok(Box::new(MetalBooleanityKernel::new(
            cpu,
            sequence,
            self.config.booleanity_cycle.cutoff_elements,
        )?))
    }
}

pub(crate) struct MetalBooleanityKernel {
    cpu: OptimizedBooleanityCycleKernel<AkitaField>,
    sequence: Option<BooleanitySequence>,
    host_tail: Vec<AkitaField>,
    cutoff_elements: usize,
    metal_rounds: usize,
}

impl MetalBooleanityKernel {
    fn new(
        cpu: OptimizedBooleanityCycleKernel<AkitaField>,
        sequence: BooleanitySequence,
        cutoff_elements: usize,
    ) -> Result<Self, SumcheckError<AkitaField>> {
        let host_elements = cpu
            .metal_polys()
            .checked_mul(cutoff_elements)
            .ok_or_else(|| metal_error("Booleanity host-tail capacity overflow"))?;
        Ok(Self {
            cpu,
            sequence: Some(sequence),
            host_tail: vec![AkitaField::zero(); host_elements],
            cutoff_elements,
            metal_rounds: 0,
        })
    }

    fn restore_cpu_tail(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let sequence = self
            .sequence
            .take()
            .ok_or_else(|| metal_error("Booleanity sequence disappeared before readback"))?;
        if !sequence.is_dense() {
            return Err(metal_error(
                "Booleanity CPU handoff requires resident dense tables",
            ));
        }
        let elements = sequence.current_elements();
        let output_len = self
            .cpu
            .metal_polys()
            .checked_mul(elements)
            .ok_or_else(|| metal_error("Booleanity readback length overflow"))?;
        if output_len > self.host_tail.len() {
            return Err(metal_error(
                "Booleanity readback exceeds the preallocated CPU tail",
            ));
        }
        sequence
            .read_current_tables(&mut self.host_tail[..output_len])
            .map_err(|error| metal_error(error.to_string()))?;
        self.cpu
            .metal_restore_dense(&self.host_tail[..output_len], elements)
    }
}

impl ProveRounds<AkitaField> for MetalBooleanityKernel {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
    }

    fn execution_domain(&self) -> RoundExecutionDomain {
        if self.sequence.as_ref().is_some_and(|sequence| {
            !sequence.is_dense() || sequence.current_elements() > self.cutoff_elements
        }) {
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
        if self.sequence.as_ref().is_some_and(|sequence| {
            sequence.is_dense() && sequence.current_elements() <= self.cutoff_elements
        }) {
            self.restore_cpu_tail()?;
            return self.cpu.prove_round(bind, round, previous_claim);
        }

        if self.sequence.is_some() {
            let message = if let Some(challenge) = bind {
                self.cpu.metal_bind_offloaded(challenge)?;
                let (e_in, e_out) = self.cpu.metal_weights()?;
                self.sequence
                    .as_mut()
                    .ok_or_else(|| metal_error("Booleanity sequence disappeared before bind"))?
                    .bind_and_message(challenge, e_in, e_out)
                    .map_err(|error| metal_error(error.to_string()))?
            } else {
                let (e_in, e_out) = self.cpu.metal_weights()?;
                self.sequence
                    .as_mut()
                    .ok_or_else(|| metal_error("Booleanity sequence disappeared before message"))?
                    .message(e_in, e_out)
                    .map_err(|error| metal_error(error.to_string()))?
            };
            self.metal_rounds += 1;
            return self.cpu.metal_message(message, previous_claim);
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

impl SumcheckKernel<AkitaField> for MetalBooleanityKernel {
    type Relation = Booleanity<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<AkitaField, Self::Relation>, SumcheckKernelError<AkitaField>>
    {
        self.cpu.output_claims(inputs)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &BooleanityCyclePhaseChallenges<AkitaField>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        self.cpu
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }
}

fn metal_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}

pub(super) fn booleanity_address_can_fallback(error: &MetalError) -> bool {
    matches!(
        error,
        MetalError::InputTooLong(_)
            | MetalError::BufferTooLong { .. }
            | MetalError::WorkingSetTooLarge { .. }
            | MetalError::FunctionLookup { .. }
            | MetalError::PipelineCompilation { .. }
            | MetalError::UnsupportedBooleanityExecutionWidth { .. }
            | MetalError::InvalidThreadgroupWidth { .. }
            | MetalError::InvalidBooleanityAddressFinalizeWidth(_)
            | MetalError::BooleanityAddressThreadgroupMemory { .. }
    )
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Metal parity test setup")]
mod tests {
    use jolt_claims::protocols::jolt::lattice::relations::booleanity::LatticeBooleanityDimensions;
    use jolt_field::Ring as _;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage6a::booleanity::BooleanityAddressPhaseChallenges;
    use jolt_verifier::stages::stage6b::booleanity::BooleanityInputClaims;

    use super::*;
    use crate::optimized::booleanity::{
        testing::with_booleanity_backend, OptimizedBooleanityAddress, OptimizedBooleanityCycle,
    };
    use crate::optimized::instruction_read_raf::{
        collect_instruction_cycle_rows, InstructionCycleRow,
    };

    fn point(seed: u64, len: usize) -> Vec<AkitaField> {
        (0..len as u64)
            .map(|index| AkitaField::from_u64(seed + 37 * index + 5))
            .collect()
    }

    #[test]
    fn address_prepare_matches_optimized_cpu_and_preserves_resident_rows() {
        let log_t = 10;
        with_booleanity_backend(log_t, 8, |witness, dimensions| {
            let relation = BooleanityAddressPhase::new(
                dimensions,
                point(900, dimensions.log_k_chunk),
                point(400, log_t),
            );
            let claims = Default::default();
            let points = Default::default();
            let challenges = BooleanityAddressPhaseChallenges {
                reference_address: point(700, dimensions.log_k_chunk),
                gamma: AkitaField::from_u64(31),
            };
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

            let mut expected = OptimizedBooleanityAddress
                .prepare(&mut ProofSession::default(), witness, inputs())
                .unwrap();
            let metal = MetalBackend::new(super::super::MetalConfig {
                booleanity_address: BooleanityAddressMetalConfig {
                    trace_cutoff_elements: 2,
                    dispatch: BooleanityAddressPushforwardConfig {
                        inner_log2: 8,
                        selectors_per_tile: 6,
                        tile_threads_per_threadgroup: Some(256),
                        finalize_threads_per_threadgroup: Some(256),
                    },
                },
                ..Default::default()
            })
            .unwrap();
            let packed = collect_instruction_cycle_rows::<AkitaField>(witness, 1 << log_t).unwrap();
            let resident = metal
                .context
                .prepare_booleanity_rows(InstructionCycleRow::metal_booleanity_rows(&packed))
                .unwrap();
            let retained = resident.clone();
            let mut session = ProofSession::default();
            session.park(resident);
            let mut actual = metal.prepare(&mut session, witness, inputs()).unwrap();
            let parked = session.state::<BooleanityRows>().unwrap();
            assert!(retained.shares_allocation(parked));

            let mut claim = AkitaField::zero();
            let mut bind = None;
            let mut round_challenges = Vec::new();
            for round in 0..expected.num_rounds() {
                let expected_poly = expected.prove_round(bind, round, claim).unwrap();
                let actual_poly = actual.prove_round(bind, round, claim).unwrap();
                assert_eq!(actual_poly, expected_poly, "round {round}");
                let challenge = AkitaField::from_u64(0x1234_5678 + 1000 * round as u64 + 7);
                claim = expected_poly.evaluate(challenge);
                round_challenges.push(challenge);
                bind = Some(challenge);
            }
            let final_bind = *round_challenges.last().unwrap();
            expected.finish_rounds(final_bind).unwrap();
            actual.finish_rounds(final_bind).unwrap();
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
    fn prepare_kernel_matches_optimized_cpu_with_resident_rows() {
        let log_t = 10;
        with_booleanity_backend(log_t, 8, |witness, dimensions| {
            let r_address = point(110, dimensions.log_k_chunk);
            let reference_address = point(700, dimensions.log_k_chunk);
            let reference_cycle = point(400, log_t);
            let relation = Booleanity::new(
                LatticeBooleanityDimensions::new(dimensions).unwrap(),
                r_address.clone(),
                reference_address,
                reference_cycle,
            );
            let claims = BooleanityInputClaims {
                address_phase: AkitaField::from_u64(17),
            };
            let points = BooleanityInputClaims {
                address_phase: r_address,
            };
            let challenges = BooleanityCyclePhaseChallenges {
                gamma: AkitaField::from_u64(31),
            };
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

            let mut expected = OptimizedBooleanityCycle
                .prepare(&mut ProofSession::default(), witness, inputs())
                .unwrap();
            let metal = MetalBackend::new(super::super::MetalConfig {
                booleanity_cycle: BooleanityMetalConfig {
                    trace_cutoff_elements: 2,
                    cutoff_elements: 4,
                    dispatch: BooleanitySequenceConfig {
                        threads_per_threadgroup: Some(256),
                        dense_threads_per_threadgroup: Some(128),
                        materialize_width: 2,
                    },
                },
                ..Default::default()
            })
            .unwrap();
            let packed = collect_instruction_cycle_rows::<AkitaField>(witness, 1 << log_t).unwrap();
            let resident = metal
                .context
                .prepare_booleanity_rows(InstructionCycleRow::metal_booleanity_rows(&packed))
                .unwrap();
            let retained = resident.clone();
            let mut session = ProofSession::default();
            session.park(resident);
            let bytecode_rows = session.state::<BooleanityRows>().cloned().unwrap();
            assert!(retained.shares_allocation(&bytecode_rows));
            let mut actual = metal.prepare(&mut session, witness, inputs()).unwrap();
            assert!(session.state::<BooleanityRows>().is_none());
            assert_eq!(bytecode_rows.len(), 1 << log_t);
            assert_eq!(
                bytecode_rows.device_registry_id(),
                metal.context.device_registry_id()
            );
            assert!(retained.shares_allocation(&bytecode_rows));

            let mut claim = claims.address_phase;
            let mut bind = None;
            let mut round_challenges = Vec::new();
            for round in 0..expected.num_rounds() {
                let expected_poly = expected.prove_round(bind, round, claim).unwrap();
                let actual_poly = actual.prove_round(bind, round, claim).unwrap();
                assert_eq!(actual_poly, expected_poly, "round {round}");
                let challenge = AkitaField::from_u64(0x1234_5678 + 1000 * round as u64 + 7);
                claim = expected_poly.evaluate(challenge);
                round_challenges.push(challenge);
                bind = Some(challenge);
            }
            let final_bind = *round_challenges.last().unwrap();
            expected.finish_rounds(final_bind).unwrap();
            actual.finish_rounds(final_bind).unwrap();
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
    fn cycle_retains_resident_rows_for_hamming() {
        let log_t = 10;
        with_booleanity_backend(log_t, 8, |witness, dimensions| {
            let r_address = point(110, dimensions.log_k_chunk);
            let relation = Booleanity::new(
                LatticeBooleanityDimensions::new(dimensions).unwrap(),
                r_address.clone(),
                point(700, dimensions.log_k_chunk),
                point(400, log_t),
            );
            let claims = BooleanityInputClaims {
                address_phase: AkitaField::from_u64(17),
            };
            let points = BooleanityInputClaims {
                address_phase: r_address,
            };
            let challenges = BooleanityCyclePhaseChallenges {
                gamma: AkitaField::from_u64(31),
            };
            let metal = MetalBackend::new(super::super::MetalConfig {
                booleanity_cycle: BooleanityMetalConfig {
                    trace_cutoff_elements: 2,
                    cutoff_elements: 4,
                    dispatch: BooleanitySequenceConfig {
                        threads_per_threadgroup: Some(256),
                        dense_threads_per_threadgroup: Some(128),
                        materialize_width: 2,
                    },
                },
                hamming_weight_claim_reduction: super::super::HammingWeightMetalConfig {
                    trace_cutoff_elements: 2,
                    dispatch: BooleanityAddressPushforwardConfig {
                        inner_log2: 8,
                        selectors_per_tile: 6,
                        tile_threads_per_threadgroup: Some(256),
                        finalize_threads_per_threadgroup: Some(256),
                    },
                },
                ..Default::default()
            })
            .unwrap();
            let packed = collect_instruction_cycle_rows::<AkitaField>(witness, 1 << log_t).unwrap();
            let resident = metal
                .context
                .prepare_booleanity_rows(InstructionCycleRow::metal_booleanity_rows(&packed))
                .unwrap();
            let retained = resident.clone();
            let mut session = ProofSession::default();
            session.park(resident);

            let _kernel = metal
                .prepare(
                    &mut session,
                    witness,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .unwrap();

            assert!(retained.shares_allocation(session.state::<BooleanityRows>().unwrap()));
        });
    }

    #[test]
    fn cpu_cycle_retains_resident_rows_for_hamming() {
        let log_t = 10;
        with_booleanity_backend(log_t, 8, |witness, dimensions| {
            let r_address = point(110, dimensions.log_k_chunk);
            let relation = Booleanity::new(
                LatticeBooleanityDimensions::new(dimensions).unwrap(),
                r_address.clone(),
                point(700, dimensions.log_k_chunk),
                point(400, log_t),
            );
            let claims = BooleanityInputClaims {
                address_phase: AkitaField::from_u64(17),
            };
            let points = BooleanityInputClaims {
                address_phase: r_address,
            };
            let challenges = BooleanityCyclePhaseChallenges {
                gamma: AkitaField::from_u64(31),
            };
            let metal = MetalBackend::new(super::super::MetalConfig {
                booleanity_cycle: BooleanityMetalConfig {
                    trace_cutoff_elements: 1 << 12,
                    cutoff_elements: 4,
                    dispatch: BooleanitySequenceConfig {
                        threads_per_threadgroup: Some(256),
                        dense_threads_per_threadgroup: Some(128),
                        materialize_width: 2,
                    },
                },
                hamming_weight_claim_reduction: super::super::HammingWeightMetalConfig {
                    trace_cutoff_elements: 2,
                    dispatch: BooleanityAddressPushforwardConfig {
                        inner_log2: 8,
                        selectors_per_tile: 6,
                        tile_threads_per_threadgroup: Some(256),
                        finalize_threads_per_threadgroup: Some(256),
                    },
                },
                ..Default::default()
            })
            .unwrap();
            let packed = collect_instruction_cycle_rows::<AkitaField>(witness, 1 << log_t).unwrap();
            let resident = metal
                .context
                .prepare_booleanity_rows(InstructionCycleRow::metal_booleanity_rows(&packed))
                .unwrap();
            let retained = resident.clone();
            let mut session = ProofSession::default();
            session.park(resident);

            let _kernel = metal
                .prepare(
                    &mut session,
                    witness,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .unwrap();

            assert!(retained.shares_allocation(session.state::<BooleanityRows>().unwrap()));
        });
    }

    #[test]
    fn k16_cycle_releases_rows_that_hamming_cannot_use() {
        let log_t = 10;
        with_booleanity_backend(log_t, 4, |witness, dimensions| {
            let r_address = point(110, dimensions.log_k_chunk);
            let relation = Booleanity::new(
                LatticeBooleanityDimensions::new(dimensions).unwrap(),
                r_address.clone(),
                point(700, dimensions.log_k_chunk),
                point(400, log_t),
            );
            let claims = BooleanityInputClaims {
                address_phase: AkitaField::from_u64(17),
            };
            let points = BooleanityInputClaims {
                address_phase: r_address,
            };
            let challenges = BooleanityCyclePhaseChallenges {
                gamma: AkitaField::from_u64(31),
            };
            let metal = MetalBackend::new(super::super::MetalConfig {
                booleanity_cycle: BooleanityMetalConfig {
                    trace_cutoff_elements: 1 << 12,
                    cutoff_elements: 4,
                    dispatch: BooleanitySequenceConfig {
                        threads_per_threadgroup: Some(256),
                        dense_threads_per_threadgroup: Some(128),
                        materialize_width: 2,
                    },
                },
                hamming_weight_claim_reduction: super::super::HammingWeightMetalConfig {
                    trace_cutoff_elements: 2,
                    dispatch: BooleanityAddressPushforwardConfig {
                        inner_log2: 8,
                        selectors_per_tile: 6,
                        tile_threads_per_threadgroup: Some(256),
                        finalize_threads_per_threadgroup: Some(256),
                    },
                },
                ..Default::default()
            })
            .unwrap();
            let packed = collect_instruction_cycle_rows::<AkitaField>(witness, 1 << log_t).unwrap();
            let resident = metal
                .context
                .prepare_booleanity_rows(InstructionCycleRow::metal_booleanity_rows(&packed))
                .unwrap();
            let mut session = ProofSession::default();
            session.park(resident);

            let _kernel = metal
                .prepare(
                    &mut session,
                    witness,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .unwrap();

            assert!(session.state::<BooleanityRows>().is_none());
        });
    }
}
