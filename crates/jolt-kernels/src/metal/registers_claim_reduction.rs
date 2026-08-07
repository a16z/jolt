use std::mem;

use jolt_claims::protocols::jolt::{JoltDerivedId, RegistersClaimReductionPublic};
use jolt_field::{AdditiveAccumulator, AkitaField, RingAccumulator, WithAccumulator};
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
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::backend::MetalBackend;
use super::solinas::registers_claim_reduction::{
    RegistersClaimDenseOutputs, RegistersClaimGeometry, RegistersClaimKernelConfig,
    RegistersClaimResidentPlanes,
};
use crate::optimized::registers_claim_reduction::{
    OptimizedRegistersClaimReduction, RegisterValuesRow,
};
use crate::optimized::support::collect_rows;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RegistersClaimReductionImplementation {
    #[default]
    Cpu,
    DirectHybrid,
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
        if config.implementation == RegistersClaimReductionImplementation::Cpu
            || cycles < config.trace_cutoff_elements
        {
            return OptimizedRegistersClaimReduction.prepare(session, witness, inputs);
        }
        let tau = inputs.relation.product_uniskip_tau_low();
        if log_t == 0 || tau.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "registers claim-reduction relation has invalid geometry",
            });
        }

        let prepare_span = tracing::info_span!(
            "MetalRegistersClaimReduction::prepare",
            cycles,
            resident_native_bytes = 3 * cycles * mem::size_of::<u64>(),
        );
        let prepared = {
            let _entered = prepare_span.enter();
            self.prepare_direct_hybrid(witness, cycles, tau, inputs.challenges.gamma, config)
        };
        let (resident, q) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                tracing::warn!(
                    target: "jolt::metal",
                    error = %error,
                    "registers claim-reduction Metal preparation failed; using optimized CPU"
                );
                return OptimizedRegistersClaimReduction.prepare(session, witness, inputs);
            }
        };

        let (_, tau_lo) = tau.split_at(log_t / 2);
        Ok(Box::new(MetalRegistersClaimReductionKernel {
            context: self.context.clone(),
            resident: Some(resident),
            geometry: RegistersClaimGeometry::new(cycles).map_err(metal_prepare_error)?,
            config: config.dispatch,
            gamma: inputs.challenges.gamma,
            gamma_sq: inputs.challenges.gamma * inputs.challenges.gamma,
            tau: tau.to_vec(),
            bound_challenges: Vec::with_capacity(log_t),
            phase: RegistersClaimPhase::Prefix {
                p: EqPolynomial::<AkitaField>::evals(tau_lo, None),
                q,
            },
            next_round: 0,
            finished: false,
        }))
    }
}

impl MetalBackend {
    fn prepare_direct_hybrid(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        cycles: usize,
        tau: &[AkitaField],
        gamma: AkitaField,
        config: RegistersClaimReductionMetalConfig,
    ) -> Result<
        (RegistersClaimResidentPlanes, Vec<AkitaField>),
        super::solinas::registers_claim_reduction::RegistersClaimLinearQError,
    > {
        let values: Vec<RegisterValuesRow> = collect_rows(witness, cycles).map_err(|_| {
            super::solinas::registers_claim_reduction::RegistersClaimLinearQError::InvalidState(
                "register witness extraction failed",
            )
        })?;
        let resident = self
            .context
            .prepare_registers_claim_resident_planes_with_fill(
                cycles,
                |rd_write_value, rs1_value, rs2_value| {
                    #[cfg(feature = "parallel")]
                    rd_write_value
                        .par_iter_mut()
                        .zip(rs1_value.par_iter_mut())
                        .zip(rs2_value.par_iter_mut())
                        .zip(values.par_iter())
                        .for_each(|(((rd, rs1), rs2), row)| {
                            *rd = row.0[0];
                            *rs1 = row.0[1];
                            *rs2 = row.0[2];
                        });
                    #[cfg(not(feature = "parallel"))]
                    for (((rd, rs1), rs2), row) in rd_write_value
                        .iter_mut()
                        .zip(rs1_value)
                        .zip(rs2_value)
                        .zip(&values)
                    {
                        *rd = row.0[0];
                        *rs1 = row.0[1];
                        *rs2 = row.0[2];
                    }
                },
            )?;
        let invocation = self.context.prepare_registers_claim_linear_q(
            &resident,
            tau,
            gamma,
            config.dispatch,
        )?;
        let observation = invocation.execute_timed()?;
        tracing::info!(
            target: "jolt::metal",
            gpu_active_ns = duration_nanos(observation.gpu_active),
            resident_wall_ns = duration_nanos(observation.resident_wall),
            useful_half_width_terms = observation.useful_half_width_terms,
            "completed registers claim-reduction q projection"
        );
        Ok((resident, observation.q))
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

struct MetalRegistersClaimReductionKernel {
    context: std::sync::Arc<super::solinas::SolinasMetal>,
    resident: Option<RegistersClaimResidentPlanes>,
    geometry: RegistersClaimGeometry,
    config: RegistersClaimKernelConfig,
    gamma: AkitaField,
    gamma_sq: AkitaField,
    tau: Vec<AkitaField>,
    bound_challenges: Vec<AkitaField>,
    phase: RegistersClaimPhase,
    next_round: usize,
    finished: bool,
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
        if let Some(resident) = &self.resident {
            visitor.visit_simple(
                allocative::Key::new("device_rows"),
                resident.resident_bytes() as usize,
            );
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
        let phase = mem::replace(&mut self.phase, RegistersClaimPhase::Poisoned);
        if !matches!(phase, RegistersClaimPhase::Prefix { .. }) {
            return Err(round_state_error(
                "registers claim-reduction midpoint requires prefix tables",
            ));
        }
        let resident = self.resident.take().ok_or_else(|| {
            round_state_error("registers claim-reduction lost its resident native planes")
        })?;
        let invocation = self
            .context
            .prepare_registers_claim_direct_fold(&resident, &self.bound_challenges, self.config)
            .map_err(metal_round_error)?;
        let observation = invocation.execute_timed().map_err(metal_round_error)?;
        tracing::info!(
            target: "jolt::metal",
            gpu_active_ns = duration_nanos(observation.gpu_active),
            resident_wall_ns = duration_nanos(observation.resident_wall),
            useful_half_width_terms = observation.useful_half_width_terms,
            "completed registers claim-reduction midpoint projection"
        );
        self.install_dense(observation.outputs)
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

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_field::AkitaField;
    use jolt_poly::Polynomial;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage3::registers_claim_reduction::{
        RegistersClaimReduction, RegistersClaimReductionChallenges,
        RegistersClaimReductionInputClaims,
    };
    use jolt_witness::JoltWitnessOracle;

    use super::{
        MetalBackend, OptimizedRegistersClaimReduction, RegistersClaimReductionImplementation,
        RegistersClaimReductionMetalConfig,
    };
    use crate::metal::solinas::MetalError;
    use crate::metal::MetalConfig;
    use crate::optimized::registers_read_write::test_support::structured_fixture;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    fn point(len: usize, seed: u64) -> Vec<AkitaField> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                AkitaField::from_u64(state | 1)
            })
            .collect()
    }

    #[test]
    fn direct_hybrid_matches_the_optimized_complete_sumcheck() {
        let metal = match MetalBackend::new(MetalConfig {
            registers_claim_reduction: RegistersClaimReductionMetalConfig {
                implementation: RegistersClaimReductionImplementation::DirectHybrid,
                trace_cutoff_elements: 2,
                ..RegistersClaimReductionMetalConfig::default()
            },
            ..MetalConfig::default()
        }) {
            Ok(metal) => metal,
            Err(MetalError::DeviceUnavailable) => return,
            Err(error) => panic!("Akita Metal backend failed to compile: {error:?}"),
        };
        for log_t in [11, 12] {
            structured_fixture(1 << log_t).with_plane(log_t, |witness| {
                let tau = point(log_t, 0x7e7e);
                let relation = RegistersClaimReduction::<AkitaField>::new(
                    jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions::new(log_t),
                    tau.clone(),
                );
                let evaluate = |polynomial: JoltVirtualPolynomial| {
                    let table = JoltWitnessOracle::<AkitaField>::oracle_table(
                        witness,
                        JoltPolynomialId::Virtual(polynomial),
                    )
                    .unwrap();
                    Polynomial::new(table).evaluate(&tau)
                };
                let gamma = AkitaField::from_u64(0x0ddb_a11c_0ffe_e123);
                let claims = RegistersClaimReductionInputClaims {
                    rd_write_value: evaluate(JoltVirtualPolynomial::RdWriteValue),
                    rs1_value: evaluate(JoltVirtualPolynomial::Rs1Value),
                    rs2_value: evaluate(JoltVirtualPolynomial::Rs2Value),
                };
                let points = RegistersClaimReductionInputClaims::default();
                let relation_challenges = RegistersClaimReductionChallenges { gamma };
                let inputs = || ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &relation_challenges,
                };

                let mut cpu = OptimizedRegistersClaimReduction
                    .prepare(&mut ProofSession::default(), witness, inputs())
                    .unwrap();
                let mut gpu = <MetalBackend as PrepareKernel<
                    AkitaField,
                    RegistersClaimReduction<AkitaField>,
                >>::prepare(
                    &metal, &mut ProofSession::default(), witness, inputs()
                )
                .unwrap();

                let input_claim = claims.rd_write_value
                    + gamma * claims.rs1_value
                    + gamma * gamma * claims.rs2_value;
                let round_challenges = point(log_t, 0x5151);
                let mut claim = input_claim;
                for round in 0..log_t {
                    let bind = round
                        .checked_sub(1)
                        .map(|previous| round_challenges[previous]);
                    let expected = cpu.prove_round(bind, round, claim).unwrap();
                    let actual = gpu.prove_round(bind, round, claim).unwrap();
                    assert_eq!(actual, expected, "round {round}");
                    claim = expected.evaluate(round_challenges[round]);
                }
                let final_bind = round_challenges[log_t - 1];
                cpu.finish_rounds(final_bind).unwrap();
                gpu.finish_rounds(final_bind).unwrap();

                let output_points = relation
                    .derive_opening_points(&round_challenges, &points)
                    .unwrap();
                cpu.validate_derived_tables(
                    &relation,
                    &points,
                    &output_points,
                    &relation_challenges,
                )
                .unwrap();
                gpu.validate_derived_tables(
                    &relation,
                    &points,
                    &output_points,
                    &relation_challenges,
                )
                .unwrap();
                assert_eq!(
                    gpu.output_claims(&claims).unwrap(),
                    cpu.output_claims(&claims).unwrap()
                );
            });
        }
    }
}
