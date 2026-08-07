use std::sync::Arc;

use jolt_claims::protocols::jolt::{JoltDerivedId, RamHammingBooleanityPublic};
use jolt_field::AkitaField;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage6b::ram_hamming_booleanity::{
    RamHammingBooleanity, RamHammingBooleanityOutputClaims,
};
use jolt_verifier::VerifierError;
use jolt_witness::JoltWitnessPlane;

use super::backend::MetalBackend;
use super::solinas::ram_cycle_family_v3::{
    estimated_ram_hamming_products, HostSparseRamHammingBooleanity, RamCycleFamilyOwner,
};
use crate::optimized::ram_hamming_booleanity::OptimizedRamHammingBooleanity;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamHammingBooleanityMetalConfig {
    pub trace_cutoff_elements: usize,
    pub max_sparse_products: u128,
}

impl Default for RamHammingBooleanityMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 25,
            max_sparse_products: 1_000_000,
        }
    }
}

struct HostSparseRamHammingKernel {
    sequence: HostSparseRamHammingBooleanity<AkitaField>,
    next_round: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for HostSparseRamHammingKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("sparse_sequence"),
            self.sequence.owned_heap_bytes(),
        );
        visitor.exit();
    }
}

impl ProveRounds<AkitaField> for HostSparseRamHammingKernel {
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
                "sparse RAM Hamming booleanity received an out-of-order round",
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
        let polynomial = UnivariatePoly::new(message.coefficients().to_vec());
        let actual =
            polynomial.evaluate(AkitaField::zero()) + polynomial.evaluate(AkitaField::one());
        if actual != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual,
            });
        }
        self.next_round += 1;
        Ok(polynomial)
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.next_round != self.sequence.num_rounds() {
            return Err(sparse_error(
                "sparse RAM Hamming booleanity finished before its final message",
            ));
        }
        self.sequence
            .bind(bind)
            .map_err(|error| sparse_error(error.to_string()))
    }
}

impl SumcheckKernel<AkitaField> for HostSparseRamHammingKernel {
    type Relation = RamHammingBooleanity<AkitaField>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<RamHammingBooleanityOutputClaims<AkitaField>, SumcheckKernelError<AkitaField>> {
        let terminal = self.sequence.terminal().map_err(kernel_error)?;
        Ok(RamHammingBooleanityOutputClaims {
            ram_hamming_weight: terminal.ram_hamming_weight(),
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
        let id = JoltDerivedId::from(RamHammingBooleanityPublic::EqCycle);
        let expected =
            match relation.derive_output_term(&id, input_points, output_points, challenges) {
                Ok(value) => value,
                Err(VerifierError::MissingStageClaimDerived { .. }) => return Ok(()),
                Err(error) => return Err(error.into()),
            };
        let got = terminal.eq_cycle();
        if got != expected {
            return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
        }
        Ok(())
    }
}

impl PrepareKernel<AkitaField, RamHammingBooleanity<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, RamHammingBooleanity<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = RamHammingBooleanity<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let relation = inputs.relation;
        let log_t = relation.trace_dimensions().log_t();
        let cycles = 1usize
            .checked_shl(u32::try_from(log_t).map_err(|_| KernelError::Unsupported {
                reason: "RAM Hamming cycle domain is too large",
            })?)
            .ok_or(KernelError::Unsupported {
                reason: "RAM Hamming cycle domain is too large",
            })?;
        if cycles < self.config.ram_hamming_booleanity.trace_cutoff_elements {
            return OptimizedRamHammingBooleanity.prepare(session, witness, inputs);
        }
        let stage1_cycle_binding = relation.stage1_cycle_binding();
        if stage1_cycle_binding.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "stage-1 cycle binding has the wrong variable count",
            });
        }

        let Some(owner) = session.state::<Arc<RamCycleFamilyOwner>>().cloned() else {
            return OptimizedRamHammingBooleanity.prepare(session, witness, inputs);
        };
        if owner.receipt().log_t() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "RAM cycle-family owner has stale geometry",
            });
        }
        owner
            .verify_integrity()
            .map_err(|error| prepare_error(error.to_string()))?;
        let predicted = estimated_ram_hamming_products(&owner)
            .map_err(|error| prepare_error(error.to_string()))?;
        if predicted > self.config.ram_hamming_booleanity.max_sparse_products {
            return OptimizedRamHammingBooleanity.prepare(session, witness, inputs);
        }
        let sequence =
            HostSparseRamHammingBooleanity::new_from_verified_owner(owner, stage1_cycle_binding)
                .map_err(|error| prepare_error(error.to_string()))?;
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .ram_hamming_sparse_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        tracing::info!(
            target: "jolt::metal",
            predicted_products = predicted,
            "prepared sparse RAM Hamming booleanity sequence"
        );
        Ok(Box::new(HostSparseRamHammingKernel {
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
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::NoChallenges;
    use jolt_verifier::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanityInputClaims;

    use super::*;
    use crate::metal::ram_cycle_family::shared_ram_cycle_family_owner;
    use crate::metal::MetalConfig;
    use crate::optimized::testing::{with_ram_fixture_backend, FixtureShape, RamOp};

    fn point(seed: u64, len: usize) -> Vec<AkitaField> {
        (0..len as u64)
            .map(|index| AkitaField::from_u64(seed + 37 * index + 5))
            .collect()
    }

    #[test]
    fn topology_sparse_sequence_matches_optimized_cpu() {
        let shape = FixtureShape {
            log_t: 5,
            ram_k: 64,
        };
        let ops = vec![
            RamOp::Write { word: 3, post: 5 },
            RamOp::Read { word: 3 },
            RamOp::Write { word: 57, post: 9 },
            RamOp::None,
            RamOp::Read { word: 57 },
        ];
        with_ram_fixture_backend(shape, ops, |witness| {
            let stage1_cycle_binding = point(71, shape.log_t);
            let relation =
                RamHammingBooleanity::new(TraceDimensions::new(shape.log_t), stage1_cycle_binding);
            let claims = RamHammingBooleanityInputClaims::default();
            let points = RamHammingBooleanityInputClaims::default();
            let challenges = NoChallenges::default();
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

            let mut expected = OptimizedRamHammingBooleanity
                .prepare(&mut ProofSession::default(), witness, inputs())
                .unwrap();
            let mut config = MetalConfig::default();
            config.ram_hamming_booleanity.trace_cutoff_elements = 1 << shape.log_t;
            let metal = MetalBackend::new(config).unwrap();
            let mut session = ProofSession::default();
            assert!(shared_ram_cycle_family_owner(
                &mut session,
                witness,
                shape.log_t,
                shape.log_k(),
            )
            .unwrap()
            .is_some());
            let mut actual =
                PrepareKernel::prepare(&metal, &mut session, witness, inputs()).unwrap();
            assert_eq!(metal.ram_hamming_sparse_sequences(), 1);

            let round_challenges = point(211, shape.log_t);
            let mut claim = AkitaField::zero();
            let mut bind = None;
            for (round, challenge) in round_challenges.iter().copied().enumerate() {
                let expected_poly = expected.prove_round(bind, round, claim).unwrap();
                let actual_poly = actual.prove_round(bind, round, claim).unwrap();
                assert_eq!(
                    actual_poly, expected_poly,
                    "round {round} polynomial mismatch"
                );
                claim = expected_poly.evaluate(challenge);
                bind = Some(challenge);
            }
            let last = *round_challenges.last().unwrap();
            expected.finish_rounds(last).unwrap();
            actual.finish_rounds(last).unwrap();
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
