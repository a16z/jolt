use std::mem::size_of;

use jolt_claims::protocols::jolt::{
    JoltDerivedId, JoltPolynomialId, JoltVirtualPolynomial, RamRafEvaluationPublic,
};
use jolt_field::AkitaField;
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
use super::solinas::{
    MetalError, PendingRamRafSequence, RamRafAddressPlane, RamRafAffineTail, RamRafConfig,
    RamRafTailOutput, RAM_RAF_ADDRESS_DOMAIN,
};
use crate::optimized::ram_trace::RamAccessColumns;
use crate::optimized::OptimizedBackend;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RamRafEvaluationMetalConfig {
    pub dispatch: RamRafConfig,
}

struct MetalRamRafEvaluationKernel {
    pending: Option<PendingRamRafSequence>,
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
        if let Some(tail) = &self.tail {
            visitor.visit_simple(allocative::Key::new("host_tail"), tail.heap_bytes());
        }
        visitor.exit();
    }
}

impl MetalBackend {
    pub(super) fn prepare_ram_raf_witness(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<(), KernelError<AkitaField>> {
        let config = self.config.ram_raf_evaluation.dispatch;
        let cycles = 1usize << log_t;
        if cycles < config.trace_cutoff || session.state::<RamRafAddressPlane>().is_some() {
            return Ok(());
        }
        let ram_ra_shape =
            witness.shape(JoltPolynomialId::Virtual(JoltVirtualPolynomial::RamRa))?;
        if ram_ra_shape.log_rows != log_t + RAM_RAF_ADDRESS_DOMAIN.ilog2() as usize {
            return Ok(());
        }
        let columns = RamAccessColumns::shared(session, witness, log_t)?;
        columns.validate_addresses(RAM_RAF_ADDRESS_DOMAIN)?;
        let plane = match self
            .context
            .prepare_ram_raf_addresses(&columns.addresses, config)
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
        session.park(plane);
        Ok(())
    }

    fn submit_ram_raf(
        &self,
        session: &mut ProofSession,
        relation: &RamReadWriteChecking<AkitaField>,
    ) -> Result<(), KernelError<AkitaField>> {
        let dimensions = relation.dimensions();
        let cycles = 1usize << dimensions.log_t();
        let config = self.config.ram_raf_evaluation.dispatch;
        if cycles < config.trace_cutoff
            || relation.ram_log_k() != RAM_RAF_ADDRESS_DOMAIN.ilog2() as usize
            || dimensions.raf_evaluation_rounds() != relation.ram_log_k()
        {
            return Ok(());
        }
        if session.state::<PendingRamRafSequence>().is_some() {
            return Err(KernelError::InvariantViolation {
                reason: "RAM RAF pushforward was submitted twice",
            });
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
}

impl PrepareKernel<AkitaField, RamReadWriteChecking<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, RamReadWriteChecking<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = RamReadWriteChecking<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        self.submit_ram_raf(session, inputs.relation)?;
        OptimizedBackend.prepare(session, witness, inputs)
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
            || relation.ram_log_k() != RAM_RAF_ADDRESS_DOMAIN.ilog2() as usize
            || relation.read_write_dimensions().raf_evaluation_rounds() != relation.ram_log_k()
        {
            return OptimizedBackend.prepare(session, witness, inputs);
        }
        let Some(pending) = session.take::<PendingRamRafSequence>() else {
            return OptimizedBackend.prepare(session, witness, inputs);
        };
        let address_storage_id = session
            .state::<RamRafAddressPlane>()
            .map(RamRafAddressPlane::storage_id);
        if pending.rows() != Some(cycles)
            || pending.address_domain() != Some(RAM_RAF_ADDRESS_DOMAIN)
            || pending.address_storage_id() != address_storage_id
        {
            return Err(KernelError::InvariantViolation {
                reason: "pending RAM RAF pushforward has stale resident provenance",
            });
        }
        Ok(Box::new(MetalRamRafEvaluationKernel {
            pending: Some(pending),
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
        let pending = self
            .pending
            .take()
            .ok_or_else(|| metal_error("RAM RAF pushforward is missing"))?;
        let (observation, stats) = {
            let _span = tracing::info_span!("MetalRamRafEvaluation::join").entered();
            pending.join().map_err(metal_tail_error)?
        };
        tracing::info!(
            target: "jolt::metal",
            submit_wall_ns = stats.submit_wall.as_nanos() as u64,
            overlap_wall_ns = stats.overlap_wall.as_nanos() as u64,
            join_wall_ns = stats.join_wall.as_nanos() as u64,
            lifecycle_wall_ns = stats.lifecycle_wall.as_nanos() as u64,
            gpu_active_ns = stats.gpu_active.as_nanos() as u64,
            completed_before_join = stats.completed_before_join,
            accessed_rows = observation.counters.accessed_rows,
            live_subtotals = observation.counters.nonzero_subtotals,
            "Metal RAM RAF pushforward joined"
        );
        let tail = RamRafAffineTail::new(observation.masses, self.lowest_address)
            .map_err(metal_tail_error)?;
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
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;

    use super::*;
    use crate::metal::MetalConfig;
    use crate::optimized::harness::{probe_input_claim, run_lockstep};
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
    fn prepared_resident_kernel_matches_optimized_cpu() {
        let shape = FixtureShape {
            log_t: 15,
            ram_k: RAM_RAF_ADDRESS_DOMAIN,
        };
        let ops = vec![
            RamOp::Write { word: 7, post: 19 },
            RamOp::Read { word: 7 },
            RamOp::None,
            RamOp::Write { word: 31, post: 41 },
            RamOp::Read { word: 31 },
        ];
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
            assert!(session.state::<RamRafAddressPlane>().is_some());
            let read_write =
                RamReadWriteChecking::new(read_write_dimensions, shape.log_k(), tau_low);
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
            assert!(session.state::<PendingRamRafSequence>().is_some());
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
}
