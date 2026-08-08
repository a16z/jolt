use jolt_claims::protocols::jolt::{JoltDerivedId, RamValCheckPublic};
use jolt_field::AkitaField;
use jolt_poly::{EqPolynomial, LtPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage4::ram_val_check::{RamValCheck, RamValCheckOutputClaims};
use jolt_witness::JoltWitnessPlane;

use super::backend::MetalBackend;
use super::ram_cycle_family::shared_ram_cycle_family_owner;
use super::solinas::ram_cycle_family_v3::HostSparseRamValCheck;
use super::solinas::{
    MetalError, PendingRamValSparseFirstMessage, RamRafAddressPlane, RamValActivePair,
};
use crate::optimized::ram_trace::{RamAccessColumns, RamIncrementActivity};
use crate::optimized::ram_val_check::{prepare_optimized_ram_val_check, RamValCheckKernel};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
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

struct MetalRamValCheckShadow {
    cpu: RamValCheckKernel<AkitaField>,
    pending: Option<PendingRamValSparseFirstMessage>,
    address_storage_id: usize,
    next_round: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalRamValCheckShadow {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(allocative::Key::new("cpu"), &self.cpu);
        if let Some(pending) = &self.pending {
            visitor.visit_field(allocative::Key::new("pending"), pending);
        }
        visitor.exit();
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
                let _ = session.take::<RamIncrementActivity>();
                #[cfg(any(test, feature = "test-utils"))]
                let _ = self
                    .ram_val_sparse_sequences
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(Box::new(sequence));
            }
        }
        let cpu = prepare_optimized_ram_val_check(session, witness, inputs)?;
        if cycles < self.config.ram_val_check.trace_cutoff_elements {
            return Ok(Box::new(cpu));
        }
        let Some(addresses) = session.state::<RamRafAddressPlane>().cloned() else {
            return Ok(Box::new(cpu));
        };
        let Some(activity) = session.state::<RamIncrementActivity>() else {
            return Ok(Box::new(cpu));
        };
        if activity.is_empty() {
            return Ok(Box::new(cpu));
        }
        if addresses.rows() != cycles
            || addresses.address_domain() != 1usize << relation.ram_log_k()
        {
            return Err(KernelError::InvariantViolation {
                reason: "RAM value-check resident address plane has stale geometry",
            });
        }
        let active_pairs = collect_active_pairs(activity, cycles)?;
        if active_pairs.is_empty() {
            return Ok(Box::new(cpu));
        }

        let point = &points.ram_val;
        let (r_address, r_cycle) = point.split_at(relation.ram_log_k());
        let split = r_cycle.len() / 2;
        let (r_high, r_low) = r_cycle.split_at(r_cycle.len() - split);
        let eq_address = EqPolynomial::<AkitaField>::evals(r_address, None);
        let lt_low = LtPolynomial::evaluations(r_low);
        let lt_high = LtPolynomial::evaluations(r_high)
            .into_iter()
            .map(|value| value + gamma)
            .collect::<Vec<_>>();
        let eq_high = EqPolynomial::<AkitaField>::evals(r_high, None);
        let address_storage_id = addresses.storage_id();
        let invocation = self.context.prepare_ram_val_sparse_first_message(
            &active_pairs,
            addresses,
            &eq_address,
            &lt_low,
            &lt_high,
            &eq_high,
        );
        let invocation = match invocation {
            Ok(invocation) => invocation,
            Err(error) if error.is_capacity_error() => return Ok(Box::new(cpu)),
            Err(error) => return Err(metal_prepare_error(error)),
        };
        let pending = {
            let _span = tracing::info_span!(
                "MetalRamValCheck::shadow_submit",
                cycles,
                active_increments = activity.len(),
                active_pairs = active_pairs.len(),
                address_storage_id,
                incremental_upload_bytes =
                    active_pairs.len() * std::mem::size_of::<RamValActivePair>(),
            )
            .entered();
            invocation.submit()
        };
        Ok(Box::new(MetalRamValCheckShadow {
            cpu,
            pending: Some(pending),
            address_storage_id,
            next_round: 0,
        }))
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
    let (r_address, r_cycle) = ram_val_point.split_at(log_k);
    let sequence = HostSparseRamValCheck::new(owner, r_address, r_cycle, gamma)
        .map_err(|error| host_sparse_prepare_error(error.to_string()))?;
    Ok(Some(HostSparseRamValCheckKernel {
        sequence,
        next_round: 0,
    }))
}

fn collect_active_pairs(
    activity: &RamIncrementActivity,
    cycles: usize,
) -> Result<Vec<RamValActivePair>, KernelError<AkitaField>> {
    let mut output = Vec::with_capacity(activity.len());
    let mut current_pair = None;
    let mut increments = [0i128; 2];
    let flush = |output: &mut Vec<RamValActivePair>, pair: usize, increments: [i128; 2]| {
        RamValActivePair::new(pair, increments[0], increments[1])
            .map(|row| output.push(row))
            .map_err(|_| KernelError::InvariantViolation {
                reason: "RAM increment activity does not fit the sparse pair ABI",
            })
    };
    for (cycle, increment) in activity.records() {
        if cycle >= cycles {
            return Err(KernelError::InvariantViolation {
                reason: "RAM increment activity exceeds the cycle domain",
            });
        }
        let pair = cycle / 2;
        if let Some(current) = current_pair.filter(|&current| current != pair) {
            flush(&mut output, current, increments)?;
            increments = [0; 2];
        }
        current_pair = Some(pair);
        let endpoint = cycle % 2;
        if increments[endpoint] != 0 {
            return Err(KernelError::InvariantViolation {
                reason: "RAM increment activity contains a duplicate cycle",
            });
        }
        increments[endpoint] = increment;
    }
    if let Some(pair) = current_pair {
        flush(&mut output, pair, increments)?;
    }
    Ok(output)
}

impl ProveRounds<AkitaField> for MetalRamValCheckShadow {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if round != self.next_round || (round == 0) != bind.is_none() {
            return Err(metal_error(
                "RAM value-check shadow received an out-of-order round",
            ));
        }
        let polynomial = self.cpu.prove_round(bind, round, previous_claim)?;
        if round == 0 {
            let pending = self
                .pending
                .take()
                .ok_or_else(|| metal_error("RAM value-check shadow result is missing"))?;
            let (message, stats) = {
                let _span = tracing::info_span!("MetalRamValCheck::shadow_join").entered();
                pending
                    .join()
                    .map_err(|error| metal_error(error.to_string()))?
            };
            let expected = [
                polynomial.evaluate(AkitaField::zero()),
                polynomial.evaluate(AkitaField::from_u64(2)),
                polynomial.evaluate(AkitaField::from_u64(3)),
            ];
            if message != expected || stats.address_storage_id != self.address_storage_id {
                return Err(metal_error(
                    "RAM value-check sparse message disagrees with CPU",
                ));
            }
            tracing::info!(
                target: "jolt::metal",
                submit_wall_ns = stats.submit_wall.as_nanos() as u64,
                overlap_wall_ns = stats.overlap_wall.as_nanos() as u64,
                join_wall_ns = stats.join_wall.as_nanos() as u64,
                lifecycle_wall_ns = stats.lifecycle_wall.as_nanos() as u64,
                gpu_active_ns = stats.gpu_active.as_nanos() as u64,
                completed_before_join = stats.completed_before_join,
                active_pairs = stats.active_pairs,
                address_storage_id = stats.address_storage_id,
                "Metal RAM value-check round-0 shadow joined"
            );
        }
        self.next_round += 1;
        Ok(polynomial)
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        self.cpu.finish_rounds(bind)
    }
}

impl SumcheckKernel<AkitaField> for MetalRamValCheckShadow {
    type Relation = RamValCheck<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<RamValCheckOutputClaims<AkitaField>, SumcheckKernelError<AkitaField>> {
        self.cpu.output_claims(inputs)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        self.cpu
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }
}

fn metal_prepare_error(error: MetalError) -> KernelError<AkitaField> {
    KernelError::Sumcheck(SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    })
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
    use std::sync::Arc;

    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::geometry::ram::{
        ram_inc_val_check, ram_ra_val_check, RamValCheckInit,
    };
    use jolt_claims::protocols::jolt::relations::ram::{
        RamValCheckChallenges, RamValCheckInputClaims,
    };
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
    use jolt_witness::JoltWitnessOracle;

    use super::*;
    use crate::metal::solinas::ram_cycle_family_v3::RamCycleFamilyOwner;
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
    fn host_sparse_sequence_matches_optimized_cpu() {
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
            assert!(session.state::<RamRafAddressPlane>().is_some());
            assert!(session.state::<RamIncrementActivity>().is_some());
            let shared_owner = Arc::clone(
                session
                    .state::<Arc<RamCycleFamilyOwner>>()
                    .expect("RAM owner prepared with the witness"),
            );

            let mut actual =
                PrepareKernel::prepare(&metal, &mut session, witness, inputs()).unwrap();
            assert_eq!(metal.ram_val_sparse_sequences(), 1);
            assert!(session.state::<RamRafAddressPlane>().is_none());
            assert!(session.state::<RamIncrementActivity>().is_none());
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
        });
    }
}
