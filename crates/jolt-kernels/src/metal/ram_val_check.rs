use std::sync::Arc;

use jolt_claims::protocols::jolt::{JoltDerivedId, RamValCheckPublic};
use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage4::ram_val_check::{RamValCheck, RamValCheckOutputClaims};
use jolt_witness::JoltWitnessPlane;

use super::backend::MetalBackend;
use super::ram_cycle_family::shared_ram_cycle_family_owner;
use super::solinas::ram_cycle_family::HostSparseRamValCheck;
use super::solinas::{RamRafAddressPlane, RamValSequence};
use crate::optimized::ram_trace::{RamAccessColumns, RamIncrementActivity};
use crate::optimized::ram_val_check::prepare_optimized_ram_val_check;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[cfg(feature = "test-utils")]
mod evaluation;
#[cfg(feature = "test-utils")]
pub use evaluation::{
    RamValCheckCpuEvalSample, RamValCheckCpuMetalEvalFixture, RamValCheckEvalError,
    RamValCheckEvalResult, RamValCheckRoundTiming, RamValCheckShapeSnapshot,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamValCheckMetalConfig {
    pub trace_cutoff_elements: usize,
}

impl Default for RamValCheckMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 25,
        }
    }
}

struct HostSparseRamValCheckKernel {
    sequence: HostSparseRamValCheck<AkitaField>,
    next_round: usize,
}

struct MetalRamValCheckKernel {
    sequence: RamValSequence,
    rounds: usize,
    next_round: usize,
    terminal: Option<[AkitaField; 3]>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for HostSparseRamValCheckKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("sparse_sequence"),
            std::mem::size_of_val(&self.sequence),
        );
        visitor.exit();
    }
}

impl ProveRounds<AkitaField> for HostSparseRamValCheckKernel {
    fn num_rounds(&self) -> usize {
        self.sequence.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if round != self.next_round || (round == 0) != bind.is_none() {
            return Err(metal_error(
                "sparse RAM value-check received an out-of-order round",
            ));
        }
        if let Some(challenge) = bind {
            self.sequence
                .bind(challenge)
                .map_err(|error| host_sparse_error(error.to_string()))?;
        }
        let message = self
            .sequence
            .message()
            .map_err(|error| host_sparse_error(error.to_string()))?;
        self.next_round += 1;
        Ok(UnivariatePoly::from_evals_and_hint(
            previous_claim,
            &message.sampled_evaluations(),
        ))
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        self.sequence
            .bind(bind)
            .map_err(|error| host_sparse_error(error.to_string()))
    }
}

impl SumcheckKernel<AkitaField> for HostSparseRamValCheckKernel {
    type Relation = RamValCheck<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<RamValCheckOutputClaims<AkitaField>, SumcheckKernelError<AkitaField>> {
        let terminal = self.sequence.terminal_factors().map_err(|error| {
            SumcheckKernelError::ComputeBackend {
                backend: "host-sparse",
                message: error.to_string(),
            }
        })?;
        Ok(RamValCheckOutputClaims {
            untrusted_advice: inputs.untrusted_advice,
            trusted_advice: inputs.trusted_advice,
            program_image: inputs.program_image,
            ram_ra: terminal.ram_ra(),
            ram_inc: terminal.ram_increment(),
        })
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        let id = JoltDerivedId::from(RamValCheckPublic::LtCyclePlusGamma);
        let expected = relation.derive_output_term(&id, input_points, output_points, challenges)?;
        let got = self
            .sequence
            .terminal_factors()
            .map_err(|error| SumcheckKernelError::ComputeBackend {
                backend: "host-sparse",
                message: error.to_string(),
            })?
            .lt_cycle_plus_gamma();
        if got != expected {
            return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
        }
        Ok(())
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalRamValCheckKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(allocative::Key::new("sequence"), &self.sequence);
        visitor.exit();
    }
}

impl ProveRounds<AkitaField> for MetalRamValCheckKernel {
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
            return Err(metal_error(
                "RAM value-check Metal sequence received an out-of-order round",
            ));
        }
        let evaluations = match bind {
            Some(challenge) => self
                .sequence
                .bind_and_message(challenge)
                .map_err(|error| metal_error(error.to_string()))?,
            None => self
                .sequence
                .message()
                .map_err(|error| metal_error(error.to_string()))?,
        };
        self.next_round += 1;
        Ok(UnivariatePoly::from_evals_and_hint(
            previous_claim,
            &evaluations,
        ))
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.next_round != self.rounds || self.terminal.is_some() {
            return Err(metal_error(
                "RAM value-check Metal sequence cannot finish before all rounds",
            ));
        }
        self.terminal = Some(
            self.sequence
                .finish_bind(bind)
                .map_err(|error| metal_error(error.to_string()))?,
        );
        Ok(())
    }
}

impl MetalRamValCheckKernel {
    fn terminal(&self) -> Result<[AkitaField; 3], SumcheckKernelError<AkitaField>> {
        self.terminal
            .ok_or(SumcheckKernelError::NotFullyBound { remaining: 1 })
    }
}

impl SumcheckKernel<AkitaField> for MetalRamValCheckKernel {
    type Relation = RamValCheck<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<RamValCheckOutputClaims<AkitaField>, SumcheckKernelError<AkitaField>> {
        let terminal = self.terminal()?;
        Ok(RamValCheckOutputClaims {
            untrusted_advice: inputs.untrusted_advice,
            trusted_advice: inputs.trusted_advice,
            program_image: inputs.program_image,
            ram_ra: terminal[1],
            ram_inc: terminal[0],
        })
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        let id = JoltDerivedId::from(RamValCheckPublic::LtCyclePlusGamma);
        let expected = relation.derive_output_term(&id, input_points, output_points, challenges)?;
        let got = self.terminal()?[2];
        if got != expected {
            return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
        }
        Ok(())
    }
}

impl PrepareKernel<AkitaField, RamValCheck<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, RamValCheck<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = RamValCheck<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let relation = inputs.relation;
        let points = inputs.points;
        let gamma = inputs.challenges.gamma;
        let log_t = relation.trace_dimensions().log_t();
        let cycles = 1usize << log_t;
        let columns = RamAccessColumns::shared(session, witness, log_t)?;
        columns.validate_addresses(1usize << relation.ram_log_k())?;
        if cycles >= self.config.ram_val_check.trace_cutoff_elements {
            if let Some(sequence) = prepare_host_sparse_ram_val_check(
                session,
                witness,
                relation,
                &points.ram_val,
                gamma,
            )? {
                let _ = session.take::<RamRafAddressPlane>();
                let _ = session.take::<Arc<RamIncrementActivity>>();
                #[cfg(any(test, feature = "test-utils"))]
                let _ = self
                    .test_counters
                    .ram_val_sparse_sequences
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(Box::new(sequence));
            }
            if session
                .state::<Arc<RamIncrementActivity>>()
                .is_some_and(|activity| activity.len() != 0)
            {
                let activity = Arc::clone(session.state::<Arc<RamIncrementActivity>>().ok_or(
                    KernelError::InvariantViolation {
                        reason: "RAM value-check lost its sparse increment stream",
                    },
                )?);
                let _ = session.take::<RamRafAddressPlane>();
                let (r_address, r_cycle) = points.ram_val.split_at(relation.ram_log_k());
                let route = tracing::info_span!(
                    "MetalRamValCheck::route",
                    cycles,
                    log_t,
                    log_k = relation.ram_log_k(),
                    requested = "sparse_increment_width32_v1",
                    selected = "sparse_increment_width32_v1",
                    fallback_reason = "none",
                    increment_records = activity.len(),
                    additional_source_row_scans = 0,
                    member_upload_bytes = 0,
                    complete_sequence = true,
                );
                let _route_guard = route.enter();
                let sequence = self
                    .context
                    .prepare_ram_val_sequence(
                        Arc::clone(&columns),
                        activity,
                        r_address,
                        r_cycle,
                        gamma,
                    )
                    .map_err(|error| KernelError::Sumcheck(metal_error(error.to_string())))?;
                #[cfg(any(test, feature = "test-utils"))]
                let _ = self
                    .test_counters
                    .ram_val_sparse_sequences
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(Box::new(MetalRamValCheckKernel {
                    sequence,
                    rounds: log_t,
                    next_round: 0,
                    terminal: None,
                }));
            }
        }
        Ok(Box::new(prepare_optimized_ram_val_check(
            session, witness, inputs,
        )?))
    }
}

fn prepare_host_sparse_ram_val_check(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    relation: &RamValCheck<AkitaField>,
    ram_val_point: &[AkitaField],
    gamma: AkitaField,
) -> Result<Option<HostSparseRamValCheckKernel>, KernelError<AkitaField>> {
    let log_t = relation.trace_dimensions().log_t();
    let log_k = relation.ram_log_k();
    if ram_val_point.len() != log_k + log_t {
        return Err(KernelError::InvariantViolation {
            reason: "RAM value-check input point has the wrong variable count",
        });
    }
    let Some(owner) = shared_ram_cycle_family_owner(session, witness, log_t, log_k)? else {
        return Ok(None);
    };
    let route = tracing::info_span!(
        "MetalRamValCheck::route",
        cycles = 1usize << log_t,
        log_t,
        log_k,
        requested = "host_sparse_v1",
        selected = "host_sparse_v1",
        fallback_reason = "none",
        source_generation = owner.receipt().source_generation(),
        source_fingerprint = owner.receipt().fingerprint(),
        access_records = owner.receipt().access_count(),
        increment_records = owner.receipt().increment_count(),
        additional_source_row_scans = 0,
        member_upload_bytes = 0,
        complete_sequence = true,
    );
    let _route_guard = route.enter();
    let (r_address, r_cycle) = ram_val_point.split_at(log_k);
    let sequence = HostSparseRamValCheck::new(owner, r_address, r_cycle, gamma)
        .map_err(|error| host_sparse_prepare_error(error.to_string()))?;
    Ok(Some(HostSparseRamValCheckKernel {
        sequence,
        next_round: 0,
    }))
}

fn host_sparse_prepare_error(message: impl Into<String>) -> KernelError<AkitaField> {
    KernelError::Sumcheck(host_sparse_error(message))
}

fn host_sparse_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "host-sparse",
        message: message.into(),
    }
}

fn metal_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "Metal parity test setup"
)]
mod tests {
    use jolt_field::{Ring as _, Zero as _};
    use std::sync::Arc;

    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::geometry::ram::{
        ram_inc_val_check, ram_ra_val_check, RamValCheckInit,
    };
    use jolt_claims::protocols::jolt::relations::ram::{
        RamValCheckChallenges, RamValCheckInputClaims,
    };
    use jolt_poly::LtPolynomial;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
    use jolt_witness::JoltWitnessOracle;

    use super::*;
    use crate::metal::solinas::ram_cycle_family::RamCycleFamilyOwner;
    use crate::metal::solinas::RAM_RAF_ADDRESS_DOMAIN;
    use crate::metal::MetalConfig;
    use crate::optimized::harness::run_lockstep;
    use crate::optimized::testing::{with_ram_fixture_backend, FixtureShape, RamOp};
    use crate::reference::views::address_fold;
    use crate::uniskip::UniskipKernel;
    use crate::OptimizedBackend;

    fn point(seed: u64, len: usize) -> Vec<AkitaField> {
        (0..len as u64)
            .map(|index| AkitaField::from_u64(seed + 37 * index + 5))
            .collect()
    }

    #[test]
    fn host_and_device_sequences_match_optimized_cpu() {
        let shape = FixtureShape {
            log_t: 15,
            ram_k: RAM_RAF_ADDRESS_DOMAIN,
        };
        let ops = vec![
            RamOp::Write { word: 7, post: 19 },
            RamOp::Read { word: 7 },
            RamOp::Write { word: 31, post: 41 },
            RamOp::Read { word: 31 },
            RamOp::Write { word: 7, post: 5 },
        ];
        with_ram_fixture_backend(shape, ops, |witness| {
            let relation = RamValCheck::<AkitaField>::new(
                TraceDimensions::new(shape.log_t),
                shape.log_k(),
                RamValCheckInit::full(AkitaField::zero()),
            );
            let claims = RamValCheckInputClaims {
                ram_val: AkitaField::zero(),
                ram_val_final: AkitaField::zero(),
                untrusted_advice: None,
                trusted_advice: None,
                program_image: None,
            };
            let points = RamValCheckInputClaims::<Vec<AkitaField>> {
                ram_val: [point(83, shape.log_k()), point(107, shape.log_t)].concat(),
                ram_val_final: point(131, shape.log_k()),
                untrusted_advice: None,
                trusted_advice: None,
                program_image: None,
            };
            let challenges = RamValCheckChallenges {
                gamma: AkitaField::from_u64(17),
            };
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
            config.ram_val_check.trace_cutoff_elements = 1 << shape.log_t;
            let metal = MetalBackend::new(config).unwrap();
            let mut session = ProofSession::default();
            <MetalBackend as UniskipKernel<AkitaField, OuterRemainder<AkitaField>>>::prepare_witness(
                &metal,
                &mut session,
                shape.log_t,
                witness,
            )
            .unwrap();
            assert!(session.state::<Arc<RamIncrementActivity>>().is_some());
            let shared_owner = Arc::clone(
                session
                    .state::<Arc<RamCycleFamilyOwner>>()
                    .expect("RAM owner prepared with the witness"),
            );

            let mut actual =
                PrepareKernel::prepare(&metal, &mut session, witness, inputs()).unwrap();
            assert_eq!(metal.ram_val_sparse_sequences(), 1);
            assert!(session.state::<RamRafAddressPlane>().is_none());
            assert!(session.state::<Arc<RamIncrementActivity>>().is_none());
            assert!(Arc::ptr_eq(
                &shared_owner,
                session
                    .state::<Arc<RamCycleFamilyOwner>>()
                    .expect("RAM value-check retains the shared owner")
            ));
            let (r_address, r_cycle) = points.ram_val.split_at(shape.log_k());
            let ra_folded =
                address_fold::<AkitaField>(witness, ram_ra_val_check(), shape.log_t, r_address)
                    .unwrap();
            let inc: Vec<AkitaField> = witness
                .oracle_table(ram_inc_val_check().polynomial_id())
                .unwrap();
            let lt = LtPolynomial::evaluations(r_cycle);
            let input_claim = (0..1usize << shape.log_t)
                .map(|j| inc[j] * ra_folded[j] * (lt[j] + challenges.gamma))
                .sum();
            let round_challenges = point(211, shape.log_t);
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

            let mut direct_session = ProofSession::default();
            let columns =
                RamAccessColumns::shared::<AkitaField>(&mut direct_session, witness, shape.log_t)
                    .unwrap();
            let increments = direct_session
                .take::<Arc<RamIncrementActivity>>()
                .expect("RAM collection publishes sparse increments");
            let direct_sequence = metal
                .context
                .prepare_ram_val_sequence(columns, increments, r_address, r_cycle, challenges.gamma)
                .unwrap();
            let mut direct = MetalRamValCheckKernel {
                sequence: direct_sequence,
                rounds: shape.log_t,
                next_round: 0,
                terminal: None,
            };
            let mut direct_expected = OptimizedBackend
                .prepare(&mut ProofSession::default(), witness, inputs())
                .unwrap();
            run_lockstep(
                direct_expected.as_mut(),
                &mut direct,
                input_claim,
                &round_challenges,
            );
            assert_eq!(
                direct.output_claims(&claims).unwrap(),
                direct_expected.output_claims(&claims).unwrap()
            );
            direct
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
        });
    }
}
