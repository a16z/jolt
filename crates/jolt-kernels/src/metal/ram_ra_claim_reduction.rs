use jolt_claims::protocols::jolt::{JoltDerivedId, RamRaClaimReductionPublic};
use jolt_field::AkitaField;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage5::ram_ra_claim_reduction::{
    RamRaClaimReduction, RamRaClaimReductionOutputClaims,
};
use jolt_witness::JoltWitnessPlane;

use super::backend::MetalBackend;
use super::ram_cycle_family::shared_ram_cycle_family_owner;
use super::solinas::ram_cycle_family::{
    estimated_ram_ra_claim_products, HostSparseRamRaClaimReduction,
};
use crate::optimized::OptimizedBackend;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const MAX_SPARSE_PRODUCTS: u128 = 1_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaClaimReductionMetalConfig {
    pub trace_cutoff_elements: usize,
}

impl Default for RamRaClaimReductionMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 25,
        }
    }
}

struct HostSparseRamRaClaimKernel {
    sequence: HostSparseRamRaClaimReduction<AkitaField>,
    next_round: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for HostSparseRamRaClaimKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("sparse_sequence"),
            self.sequence.owned_heap_bytes(),
        );
        visitor.exit();
    }
}

impl ProveRounds<AkitaField> for HostSparseRamRaClaimKernel {
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
            return Err(sparse_error(
                "sparse RAM RA claim reduction received an out-of-order round",
            ));
        }
        if let Some(challenge) = bind {
            self.sequence
                .bind(challenge)
                .map_err(|error| sparse_error(error.to_string()))?;
        }
        let message = self
            .sequence
            .message()
            .map_err(|error| sparse_error(error.to_string()))?;
        self.next_round += 1;
        Ok(UnivariatePoly::from_evals_and_hint(
            previous_claim,
            &message.sampled_evaluations(),
        ))
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.next_round != self.sequence.num_rounds() {
            return Err(sparse_error(
                "sparse RAM RA claim reduction finished before its final message",
            ));
        }
        self.sequence
            .bind(bind)
            .map_err(|error| sparse_error(error.to_string()))
    }
}

impl SumcheckKernel<AkitaField> for HostSparseRamRaClaimKernel {
    type Relation = RamRaClaimReduction<AkitaField>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<RamRaClaimReductionOutputClaims<AkitaField>, SumcheckKernelError<AkitaField>> {
        let terminal = self.sequence.terminal().map_err(kernel_error)?;
        Ok(RamRaClaimReductionOutputClaims {
            ram_ra: terminal.ram_ra(),
        })
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        let terminal = self.sequence.terminal().map_err(kernel_error)?;
        let ids = [
            RamRaClaimReductionPublic::EqCycleRaf,
            RamRaClaimReductionPublic::EqCycleReadWrite,
            RamRaClaimReductionPublic::EqCycleValCheck,
        ];
        for (id, got) in ids.into_iter().zip(terminal.eq_cycles()) {
            let id = JoltDerivedId::from(id);
            let expected =
                relation.derive_output_term(&id, input_points, output_points, challenges)?;
            if got != expected {
                return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
            }
        }
        Ok(())
    }
}

impl PrepareKernel<AkitaField, RamRaClaimReduction<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, RamRaClaimReduction<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = RamRaClaimReduction<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let relation = inputs.relation;
        let log_t = relation.trace_dimensions().log_t();
        let cycles = 1usize
            .checked_shl(u32::try_from(log_t).map_err(|_| KernelError::Unsupported {
                reason: "RAM RA claim-reduction cycle domain is too large",
            })?)
            .ok_or(KernelError::Unsupported {
                reason: "RAM RA claim-reduction cycle domain is too large",
            })?;
        if cycles < self.config.ram_ra_claim_reduction.trace_cutoff_elements {
            return OptimizedBackend.prepare(session, witness, inputs);
        }
        let log_k = relation.ram_log_k();
        let expected_len = log_k + log_t;
        for point in [
            inputs.points.raf(),
            inputs.points.read_write(),
            inputs.points.val_check(),
        ] {
            if point.len() != expected_len {
                return Err(KernelError::InvariantViolation {
                    reason: "RAM RA claim-reduction input point has the wrong variable count",
                });
            }
        }
        let address_prefix = &inputs.points.read_write()[..log_k];
        if &inputs.points.raf()[..log_k] != address_prefix
            || &inputs.points.val_check()[..log_k] != address_prefix
        {
            return Err(KernelError::InvariantViolation {
                reason: "RAM RA claim-reduction input points disagree on the address prefix",
            });
        }
        let r_address = address_prefix;
        let cycle_points = [
            &inputs.points.raf()[log_k..],
            &inputs.points.read_write()[log_k..],
            &inputs.points.val_check()[log_k..],
        ];
        let Some(owner) = shared_ram_cycle_family_owner(session, witness, log_t, log_k)? else {
            return OptimizedBackend.prepare(session, witness, inputs);
        };
        let predicted = estimated_ram_ra_claim_products(&owner)
            .map_err(|error| prepare_error(error.to_string()))?;
        if predicted > MAX_SPARSE_PRODUCTS {
            return OptimizedBackend.prepare(session, witness, inputs);
        }
        let route = tracing::info_span!(
            "MetalRamRaClaimReduction::route",
            cycles,
            log_t,
            log_k,
            requested = "host_sparse_v1",
            selected = "host_sparse_v1",
            fallback_reason = "none",
            source_generation = owner.receipt().source_generation(),
            source_fingerprint = owner.receipt().fingerprint(),
            access_records = owner.receipt().access_count(),
            increment_records = owner.receipt().increment_count(),
            estimated_products = predicted,
            product_cap = MAX_SPARSE_PRODUCTS,
            additional_source_row_scans = 0,
            member_upload_bytes = 0,
            complete_sequence = true,
        );
        let _route_guard = route.enter();
        let sequence = HostSparseRamRaClaimReduction::new(
            owner,
            r_address,
            cycle_points,
            inputs.challenges.gamma,
        )
        .map_err(|error| prepare_error(error.to_string()))?;
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .ram_ra_claim_sparse_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        tracing::info!(
            target: "jolt::metal",
            predicted_products = predicted,
            "prepared sparse RAM RA claim-reduction sequence"
        );
        Ok(Box::new(HostSparseRamRaClaimKernel {
            sequence,
            next_round: 0,
        }))
    }
}

fn sparse_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "host-sparse",
        message: message.into(),
    }
}

fn prepare_error(message: impl Into<String>) -> KernelError<AkitaField> {
    KernelError::Sumcheck(sparse_error(message))
}

fn kernel_error(error: impl ToString) -> SumcheckKernelError<AkitaField> {
    SumcheckKernelError::ComputeBackend {
        backend: "host-sparse",
        message: error.to_string(),
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Metal parity test setup")]
mod tests {
    use std::sync::Arc;

    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::geometry::ram::ram_ra_claim_reduction;
    use jolt_claims::protocols::jolt::relations::ram::{
        RamRaClaimReductionChallenges, RamRaClaimReductionInputClaims,
    };
    use jolt_poly::EqPolynomial;

    use super::*;
    use crate::metal::solinas::ram_cycle_family::RamCycleFamilyOwner;
    use crate::metal::MetalConfig;
    use crate::optimized::harness::run_lockstep;
    use crate::optimized::testing::{with_ram_fixture_backend, FixtureShape, RamOp};
    use crate::reference::views::address_fold;

    fn point(seed: u64, len: usize) -> Vec<AkitaField> {
        (0..len as u64)
            .map(|index| AkitaField::from_u64(seed + 37 * index + 5))
            .collect()
    }

    #[test]
    fn topology_sparse_sequence_matches_optimized_cpu() {
        let shape = FixtureShape {
            log_t: 5,
            ram_k: 16,
        };
        let ops = vec![
            RamOp::Write { word: 3, post: 5 },
            RamOp::Read { word: 3 },
            RamOp::Write { word: 7, post: 9 },
            RamOp::None,
            RamOp::Read { word: 7 },
        ];
        with_ram_fixture_backend(shape, ops, |witness| {
            let r_address = point(11, shape.log_k());
            let r_cycle_raf = point(37, shape.log_t);
            let r_cycle_rw = point(71, shape.log_t);
            let r_cycle_val = point(103, shape.log_t);
            let relation = RamRaClaimReduction::<AkitaField>::new(
                TraceDimensions::new(shape.log_t),
                shape.log_k(),
            );
            let claims = RamRaClaimReductionInputClaims::<AkitaField>::default();
            let points = RamRaClaimReductionInputClaims::<Vec<AkitaField>> {
                raf: [r_address.clone(), r_cycle_raf.clone()].concat(),
                read_write: [r_address.clone(), r_cycle_rw.clone()].concat(),
                val_check: [r_address.clone(), r_cycle_val.clone()].concat(),
            };
            let challenges = RamRaClaimReductionChallenges {
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
            config.ram_ra_claim_reduction.trace_cutoff_elements = 1 << shape.log_t;
            let metal = MetalBackend::new(config).unwrap();
            let mut session = ProofSession::default();
            let mut actual =
                PrepareKernel::prepare(&metal, &mut session, witness, inputs()).unwrap();
            assert_eq!(metal.ram_ra_claim_sparse_sequences(), 1);
            assert!(session.state::<Arc<RamCycleFamilyOwner>>().is_some());

            let h = address_fold::<AkitaField>(
                witness,
                ram_ra_claim_reduction(),
                shape.log_t,
                &r_address,
            )
            .unwrap();
            let eq_raf = EqPolynomial::new(r_cycle_raf).evaluations();
            let eq_rw = EqPolynomial::new(r_cycle_rw).evaluations();
            let eq_val = EqPolynomial::new(r_cycle_val).evaluations();
            let gamma_squared = challenges.gamma * challenges.gamma;
            let input_claim = h
                .iter()
                .enumerate()
                .map(|(cycle, &value)| {
                    value
                        * (eq_raf[cycle]
                            + challenges.gamma * eq_rw[cycle]
                            + gamma_squared * eq_val[cycle])
                })
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
