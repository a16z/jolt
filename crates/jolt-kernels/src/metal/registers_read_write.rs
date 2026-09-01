use std::{cmp::Ordering, sync::Arc};

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, RoundExecutionDomain, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage4::registers_read_write_checking::{
    RegistersReadWriteChecking, RegistersReadWriteOutputClaims,
};
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::backend::MetalBackend;
use super::registers_val_evaluation::RegistersValEvaluationSource;
use super::solinas::registers_claim_reduction::RegistersClaimResidentRdPlane;
use super::solinas::registers_read_write::{
    RegistersReadWriteCycleFinish, RegistersReadWriteCycleSequence,
};
use super::solinas::{
    MetalError, PendingRegistersReadWriteStage1Pipelines, RegistersReadWriteStage1Source,
};
use crate::optimized::registers_read_write::{
    AlignedPackedRegisterRows, AlignedPackedRegisterRowsError, OptimizedRegistersReadWrite,
    PackedRegisterCycleRow, SharedRdIndices,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[cfg(feature = "test-utils")]
mod evaluation;
#[cfg(feature = "test-utils")]
pub use evaluation::{
    RegistersReadWriteCpuEvalSample, RegistersReadWriteCpuMetalEvalFixture,
    RegistersReadWriteEvalError, RegistersReadWriteEvalResult, RegistersReadWriteMetalEvalSample,
    RegistersReadWriteRoundTiming, RegistersReadWriteShapeSnapshot,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersReadWriteMetalConfig {
    pub trace_cutoff_elements: usize,
}

impl Default for RegistersReadWriteMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 25,
        }
    }
}

type RegistersReadWriteKernelBox =
    Box<dyn SumcheckKernel<AkitaField, Relation = RegistersReadWriteChecking<AkitaField>>>;

const CONCURRENT_RAM_VAL_ROUND: usize = 8;

struct MetalRegistersReadWriteKernel {
    sequence: RegistersReadWriteCycleSequence,
    source_owner: Option<Arc<AlignedPackedRegisterRows>>,
    continuation: Option<RegistersReadWriteKernelBox>,
    gruen: GruenSplitEqPolynomial<AkitaField>,
    r_cycle: Vec<AkitaField>,
    cycle_challenges: Vec<AkitaField>,
    bound_challenges: Vec<AkitaField>,
    gamma: AkitaField,
    log_t: usize,
    log_k: usize,
    use_metal_operand_claims: bool,
    next_round: usize,
    finished: bool,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalRegistersReadWriteKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        use crate::backend::{gruen_heap_bytes, vec_heap_bytes};
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("sequence"),
            self.sequence.resident_bytes(),
        );
        visitor.visit_simple(allocative::Key::new("gruen"), gruen_heap_bytes(&self.gruen));
        visitor.visit_simple(
            allocative::Key::new("challenges"),
            vec_heap_bytes(&self.r_cycle)
                + vec_heap_bytes(&self.cycle_challenges)
                + vec_heap_bytes(&self.bound_challenges),
        );
        visitor.exit();
    }
}

impl MetalRegistersReadWriteKernel {
    fn validate_round(
        &self,
        bind: Option<&AkitaField>,
        round: usize,
    ) -> Result<(), SumcheckError<AkitaField>> {
        if self.finished || round != self.next_round || round >= self.num_rounds() {
            return Err(metal_error(
                "registers read-write received an out-of-order round",
            ));
        }
        if (round == 0) != bind.is_none() {
            return Err(metal_error(
                "registers read-write round has the wrong bind shape",
            ));
        }
        Ok(())
    }

    fn cycle_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        let observation = match bind {
            Some(challenge) => {
                self.gruen.bind(challenge);
                self.cycle_challenges.push(challenge);
                self.bound_challenges.push(challenge);
                self.sequence
                    .bind_and_message(
                        challenge,
                        self.gruen.e_in_current(),
                        self.gruen.e_out_current(),
                    )
                    .map_err(metal_runtime_error)?
            }
            None => self
                .sequence
                .message(
                    self.gruen.e_in_current(),
                    self.gruen.e_out_current(),
                    self.gamma,
                )
                .map_err(metal_runtime_error)?,
        };
        tracing::info!(
            target: "jolt::metal",
            round,
            wall_ns = duration_nanos(observation.wall),
            gpu_active_ns = duration_nanos(observation.gpu_active),
            prefill_gpu_active_ns = duration_nanos(observation.prefill_gpu_active),
            allocation_ns = duration_nanos(observation.allocation),
            resident_bytes = observation.resident_bytes,
            peak_transition_bytes = observation.peak_transition_bytes,
            retired_source_bytes = observation.retired_source_bytes,
            "completed registers read-write Metal cycle round"
        );
        Ok(self.gruen.gruen_poly_deg_3(
            observation.quadratic[0],
            observation.quadratic[1],
            previous_claim,
        ))
    }

    fn enter_address_phase(
        &mut self,
        challenge: AkitaField,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if self.continuation.is_some() || self.cycle_challenges.len() + 1 != self.log_t {
            return Err(metal_error(
                "registers read-write reached the address phase in an invalid state",
            ));
        }
        self.cycle_challenges.push(challenge);
        self.bound_challenges.push(challenge);
        let RegistersReadWriteCycleFinish {
            roots,
            increment,
            wall,
            gpu_active,
            allocation,
            resident_bytes,
            peak_transition_bytes,
        } = self
            .sequence
            .finish(challenge)
            .map_err(metal_runtime_error)?;
        tracing::info!(
            target: "jolt::metal",
            wall_ns = duration_nanos(wall),
            gpu_active_ns = duration_nanos(gpu_active),
            allocation_ns = duration_nanos(allocation),
            resident_bytes,
            peak_transition_bytes,
            "completed registers read-write Metal cycle handoff"
        );
        let operand_rows = if self.use_metal_operand_claims {
            None
        } else {
            Some(
                self.source_owner
                    .as_ref()
                    .ok_or_else(|| metal_error("registers read-write lost its CPU operand rows"))?
                    .as_slice(),
            )
        };
        let mut continuation = OptimizedRegistersReadWrite::prepare_after_cycle_phase(
            self.log_t,
            self.log_k,
            &self.r_cycle,
            operand_rows,
            &self.cycle_challenges,
            roots,
            increment,
        )
        .map_err(kernel_runtime_error)?;
        let polynomial = continuation.prove_round(None, round, previous_claim)?;
        self.continuation = Some(continuation);
        Ok(polynomial)
    }

    fn continuation_mut(
        &mut self,
    ) -> Result<&mut RegistersReadWriteKernelBox, SumcheckError<AkitaField>> {
        self.continuation
            .as_mut()
            .ok_or_else(|| metal_error("registers read-write lost its CPU address continuation"))
    }
}

impl ProveRounds<AkitaField> for MetalRegistersReadWriteKernel {
    fn num_rounds(&self) -> usize {
        self.log_t + self.log_k
    }

    fn execution_domain(&self) -> RoundExecutionDomain {
        if self.log_t >= 28 && self.next_round == CONCURRENT_RAM_VAL_ROUND {
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
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        self.validate_round(bind.as_ref(), round)?;
        let polynomial = match round.cmp(&self.log_t) {
            Ordering::Less => self.cycle_round(bind, round, previous_claim)?,
            Ordering::Equal => self.enter_address_phase(
                bind.ok_or_else(|| {
                    metal_error("registers read-write address phase missed the final cycle bind")
                })?,
                round,
                previous_claim,
            )?,
            Ordering::Greater => {
                let challenge = bind.ok_or_else(|| {
                    metal_error("registers read-write address round missed its bind")
                })?;
                self.bound_challenges.push(challenge);
                self.continuation_mut()?
                    .prove_round(Some(challenge), round, previous_claim)?
            }
        };
        self.next_round += 1;
        Ok(polynomial)
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.finished || self.next_round != self.num_rounds() {
            return Err(metal_error(
                "registers read-write reached finish in an invalid state",
            ));
        }
        self.bound_challenges.push(bind);
        self.continuation_mut()?.finish_rounds(bind)?;
        self.finished = true;
        Ok(())
    }
}

impl SumcheckKernel<AkitaField> for MetalRegistersReadWriteKernel {
    type Relation = RegistersReadWriteChecking<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<RegistersReadWriteOutputClaims<AkitaField>, SumcheckKernelError<AkitaField>> {
        if !self.finished || self.bound_challenges.len() != self.num_rounds() {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "registers read-write output requested before finish",
            });
        }
        let mut outputs = self
            .continuation
            .as_mut()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "registers read-write output lost its address continuation",
            })?
            .output_claims(inputs)?;
        if self.use_metal_operand_claims {
            let cycle_point = self.bound_challenges[..self.log_t]
                .iter()
                .rev()
                .copied()
                .collect::<Vec<_>>();
            let address_point = self.bound_challenges[self.log_t..]
                .iter()
                .rev()
                .copied()
                .collect::<Vec<_>>();
            let observation = self
                .sequence
                .operand_claims(&cycle_point, &address_point)
                .map_err(|error| SumcheckKernelError::ComputeBackend {
                    backend: "metal",
                    message: error.to_string(),
                })?;
            outputs.rs1_ra = observation.claims[0];
            outputs.rs2_ra = observation.claims[1];
            tracing::info!(
                target: "jolt::metal",
                prepare_ns = duration_nanos(observation.prepare),
                wall_ns = duration_nanos(observation.wall),
                gpu_active_ns = duration_nanos(observation.gpu_active),
                "completed registers read-write Metal operand claims"
            );
        }
        Ok(outputs)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        if !self.finished {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "registers read-write derived table requested before finish",
            });
        }
        self.continuation
            .as_ref()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "registers read-write validation lost its address continuation",
            })?
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }
}

impl PrepareKernel<AkitaField, RegistersReadWriteChecking<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, RegistersReadWriteChecking<AkitaField>>,
    ) -> Result<RegistersReadWriteKernelBox, KernelError<AkitaField>> {
        let dimensions = inputs.relation.register_dimensions();
        let log_t = dimensions.log_t();
        let log_k = dimensions.log_k();
        if dimensions.phase1_num_rounds() != log_t
            || log_t == 0
            || log_t >= 32
            || inputs.points.rd_write_value.len() != log_t
        {
            return Err(KernelError::Unsupported {
                reason: "Metal registers read-write requires the default cycle phase",
            });
        }
        let cycles = 1usize << log_t;
        if cycles < self.config.registers_read_write.trace_cutoff_elements {
            record_route(
                cycles,
                "metal_cycle_sequence_v1",
                "optimized_cpu",
                "trace_cutoff",
                0,
                "none",
            );
            let prepared = OptimizedRegistersReadWrite.prepare(session, witness, inputs)?;
            super::instruction_read_raf::start_instruction_read_raf_scatter(session)?;
            return Ok(prepared);
        }
        let needs_rd_carry = self.config.registers_val_evaluation.source
            != RegistersValEvaluationSource::Stage1Resident
            || cycles < self.config.registers_val_evaluation.trace_cutoff_elements;
        if log_t == 28 && !needs_rd_carry {
            let resident_source = session.state::<RegistersReadWriteStage1Source>().is_some();
            let resident_rd_post = session.state::<RegistersClaimResidentRdPlane>().is_some();
            let pending_pipelines = session
                .state::<PendingRegistersReadWriteStage1Pipelines>()
                .is_some();
            if resident_source != resident_rd_post || resident_source != pending_pipelines {
                return Err(KernelError::InvariantViolation {
                    reason: "registers read-write resident source carries are incomplete",
                });
            }
            if resident_source {
                let pipeline_join = session
                    .take::<PendingRegistersReadWriteStage1Pipelines>()
                    .ok_or(KernelError::InvariantViolation {
                        reason: "registers read-write pipeline warm-up disappeared",
                    })?
                    .join()
                    .map_err(metal_prepare_error)?;
                tracing::info!(
                    target: "jolt::metal",
                    wall_ns = duration_nanos(pipeline_join),
                    "joined registers read-write Stage-1 pipeline warm-up"
                );
                let source_owner = session.take::<RegistersReadWriteStage1Source>().ok_or(
                    KernelError::InvariantViolation {
                        reason: "registers read-write resident source disappeared",
                    },
                )?;
                let rd_post = session.take::<RegistersClaimResidentRdPlane>().ok_or(
                    KernelError::InvariantViolation {
                        reason: "registers read-write resident rd-post plane disappeared",
                    },
                )?;
                let source = source_owner.device_view();
                let physical_rows = source.physical_rows;
                let _span = tracing::info_span!(
                    "MetalRegistersReadWrite::source_prepare",
                    source_kind = "stage1_fused_delta_carried_rs1_v1",
                    physical_rows,
                    witness_row_extractions = 0usize,
                    packed_owner_bytes = 0usize,
                )
                .entered();
                let sequence = self
                    .context
                    .prepare_registers_read_write_cycle_sequence_from_stage1(
                        source_owner,
                        rd_post,
                        log_t,
                        inputs.challenges.gamma,
                    )
                    .map_err(metal_prepare_error)?;
                let source_bytes = sequence.source_bytes();
                record_route(
                    cycles,
                    "metal_cycle_sequence_v1",
                    "metal_cycle_sequence_v1",
                    "none",
                    source_bytes,
                    "stage1_fused_delta_carried_rs1_v1",
                );
                #[cfg(any(test, feature = "test-utils"))]
                let _ = self
                    .test_counters
                    .registers_read_write_metal_sequences
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let kernel = Box::new(MetalRegistersReadWriteKernel {
                    sequence,
                    source_owner: None,
                    continuation: None,
                    gruen: GruenSplitEqPolynomial::new(
                        &inputs.points.rd_write_value,
                        BindingOrder::LowToHigh,
                    ),
                    r_cycle: inputs.points.rd_write_value.clone(),
                    cycle_challenges: Vec::with_capacity(log_t),
                    bound_challenges: Vec::with_capacity(log_t + log_k),
                    gamma: inputs.challenges.gamma,
                    log_t,
                    log_k,
                    use_metal_operand_claims: true,
                    next_round: 0,
                    finished: false,
                });
                return Ok(kernel);
            }
        }
        let Some(access) = witness.random_access() else {
            record_route(
                cycles,
                "metal_cycle_sequence_v1",
                "optimized_cpu",
                "random_access_unavailable",
                0,
                "none",
            );
            let prepared = OptimizedRegistersReadWrite.prepare(session, witness, inputs)?;
            super::instruction_read_raf::start_instruction_read_raf_scatter(session)?;
            return Ok(prepared);
        };
        if access.cycles() < cycles {
            return Err(KernelError::InvariantViolation {
                reason: "registers read-write random-access witness is shorter than its domain",
            });
        }
        let physical_rows = access.physical_rows().min(cycles);
        let _span = tracing::info_span!(
            "MetalRegistersReadWrite::source_prepare",
            source_kind = "direct_witness",
            physical_rows,
            witness_row_extractions = physical_rows,
            packed_owner_bytes = 0,
        )
        .entered();
        let rows = AlignedPackedRegisterRows::collect(&access, physical_rows, log_t >= 28)
            .map_err(packed_source_error)?;
        let source_owner = Arc::new(rows);
        let source_kind = "direct_witness";
        if needs_rd_carry {
            #[cfg(feature = "parallel")]
            let mut rd_indices = source_owner
                .as_slice()
                .par_iter()
                .copied()
                .map(PackedRegisterCycleRow::rd_index)
                .collect::<Vec<_>>();
            #[cfg(not(feature = "parallel"))]
            let mut rd_indices = source_owner
                .as_slice()
                .iter()
                .copied()
                .map(PackedRegisterCycleRow::rd_index)
                .collect::<Vec<_>>();
            rd_indices.resize(cycles, None);
            session.park(SharedRdIndices(rd_indices));
        }
        let source_bytes = source_owner.allocation_bytes();
        let continuation_source = (log_t < 25).then(|| Arc::clone(&source_owner));
        let sequence = self
            .context
            .prepare_registers_read_write_cycle_sequence(
                source_owner,
                log_t,
                inputs.challenges.gamma,
            )
            .map_err(metal_prepare_error)?;
        record_route(
            cycles,
            "metal_cycle_sequence_v1",
            "metal_cycle_sequence_v1",
            "none",
            source_bytes,
            source_kind,
        );
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .test_counters
            .registers_read_write_metal_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let kernel = Box::new(MetalRegistersReadWriteKernel {
            sequence,
            source_owner: continuation_source,
            continuation: None,
            gruen: GruenSplitEqPolynomial::new(
                &inputs.points.rd_write_value,
                BindingOrder::LowToHigh,
            ),
            r_cycle: inputs.points.rd_write_value.clone(),
            cycle_challenges: Vec::with_capacity(log_t),
            bound_challenges: Vec::with_capacity(log_t + log_k),
            gamma: inputs.challenges.gamma,
            log_t,
            log_k,
            use_metal_operand_claims: log_t >= 25,
            next_round: 0,
            finished: false,
        });
        super::instruction_read_raf::start_instruction_read_raf_scatter(session)?;
        Ok(kernel)
    }
}

fn record_route(
    cycles: usize,
    requested: &'static str,
    selected: &'static str,
    fallback_reason: &'static str,
    source_bytes: usize,
    source_kind: &'static str,
) {
    let _span = tracing::info_span!(
        "MetalRegistersReadWrite::route",
        cycles,
        requested,
        selected,
        fallback_reason,
        source_bytes,
        source_kind,
    )
    .entered();
}

fn packed_source_error(error: AlignedPackedRegisterRowsError) -> KernelError<AkitaField> {
    match error {
        AlignedPackedRegisterRowsError::Witness(error) => KernelError::Witness(error),
        AlignedPackedRegisterRowsError::Storage(_) => KernelError::InvariantViolation {
            reason: "registers read-write packed source allocation failed",
        },
    }
}

fn metal_prepare_error(error: MetalError) -> KernelError<AkitaField> {
    metal_runtime_error(error).into()
}

fn kernel_runtime_error(error: KernelError<AkitaField>) -> SumcheckError<AkitaField> {
    metal_error(error.to_string())
}

fn metal_runtime_error(error: MetalError) -> SumcheckError<AkitaField> {
    metal_error(error.to_string())
}

fn metal_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}

fn duration_nanos(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "Metal registers read-write parity setup"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::{
        ReadWriteDimensions, REGISTER_ADDRESS_BITS,
    };
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_field::Ring as _;
    use jolt_poly::Polynomial;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage4::registers_read_write_checking::{
        RegistersReadWriteChallenges, RegistersReadWriteInputClaims,
    };
    use jolt_witness::JoltWitnessOracle;

    use super::*;
    use crate::metal::MetalConfig;
    use crate::optimized::harness::run_lockstep;
    use crate::optimized::registers_read_write::test_support::structured_fixture;

    fn point(len: usize, seed: u64) -> Vec<AkitaField> {
        (0..len as u64)
            .map(|index| AkitaField::from_u64(seed + 37 * index + 5))
            .collect()
    }

    #[test]
    fn adapter_selects_metal_and_matches_optimized_cpu() {
        let log_t = 8;
        structured_fixture(1 << log_t).with_plane(log_t, |witness| {
            let relation = RegistersReadWriteChecking::new(ReadWriteDimensions::new(
                log_t,
                REGISTER_ADDRESS_BITS,
                log_t,
                0,
            ));
            let r_cycle = point(log_t, 19);
            let evaluate = |polynomial: JoltVirtualPolynomial| {
                let table = JoltWitnessOracle::<AkitaField>::oracle_table(
                    witness,
                    JoltPolynomialId::Virtual(polynomial),
                )
                .unwrap();
                Polynomial::new(table).evaluate(&r_cycle)
            };
            let claims = RegistersReadWriteInputClaims {
                rd_write_value: evaluate(JoltVirtualPolynomial::RdWriteValue),
                rs1_value: evaluate(JoltVirtualPolynomial::Rs1Value),
                rs2_value: evaluate(JoltVirtualPolynomial::Rs2Value),
            };
            let points = RegistersReadWriteInputClaims {
                rd_write_value: r_cycle.clone(),
                rs1_value: r_cycle.clone(),
                rs2_value: r_cycle,
            };
            let challenges = RegistersReadWriteChallenges {
                gamma: AkitaField::from_u64(31),
            };
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };
            let input_claim = claims.rd_write_value
                + challenges.gamma * claims.rs1_value
                + challenges.gamma * challenges.gamma * claims.rs2_value;
            let round_challenges = point(log_t + REGISTER_ADDRESS_BITS, 101);
            let mut expected = OptimizedRegistersReadWrite
                .prepare(&mut ProofSession::default(), witness, inputs())
                .unwrap();
            let metal = MetalBackend::new(MetalConfig {
                registers_read_write: RegistersReadWriteMetalConfig {
                    trace_cutoff_elements: 2,
                },
                ..Default::default()
            })
            .unwrap();
            let mut session = ProofSession::default();
            let mut actual = metal.prepare(&mut session, witness, inputs()).unwrap();
            assert_eq!(metal.registers_read_write_metal_sequences(), 1);
            assert!(session.state::<SharedRdIndices>().is_some());
            run_lockstep(
                expected.as_mut(),
                actual.as_mut(),
                input_claim,
                &round_challenges,
            );
            assert_eq!(
                expected.output_claims(&claims).unwrap(),
                actual.output_claims(&claims).unwrap()
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
