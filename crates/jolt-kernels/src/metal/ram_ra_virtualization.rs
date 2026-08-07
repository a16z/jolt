use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_claims::protocols::jolt::{JoltDerivedId, RamRaVirtualizationPublic};
use jolt_field::AkitaField;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage6b::ram_ra_virtualization::{
    RamRaVirtualization, RamRaVirtualizationOutputClaims,
};
use jolt_witness::JoltWitnessPlane;

use super::backend::MetalBackend;
use super::ram_cycle_family::shared_ram_cycle_family_owner;
use super::solinas::ram_cycle_family_v3::{
    estimated_ram_ra_virtualization_products, HostSparseRamRaVirtualization,
};
use crate::optimized::ram_trace::RamAccessColumns;
use crate::optimized::OptimizedBackend;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const MAX_SPARSE_PRODUCTS: u128 = 1_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaVirtualizationMetalConfig {
    pub trace_cutoff_elements: usize,
}

impl Default for RamRaVirtualizationMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 25,
        }
    }
}

struct HostSparseRamRaVirtualizationKernel {
    sequence: HostSparseRamRaVirtualization<AkitaField>,
    next_round: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for HostSparseRamRaVirtualizationKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("sparse_sequence"),
            self.sequence.owned_heap_bytes(),
        );
        visitor.exit();
    }
}

impl ProveRounds<AkitaField> for HostSparseRamRaVirtualizationKernel {
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
                "sparse RAM RA virtualization received an out-of-order round",
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
        let evaluations = message.evaluations();
        let actual = evaluations[0] + evaluations[1];
        if actual != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual,
            });
        }
        self.next_round += 1;
        Ok(UnivariatePoly::from_evals(evaluations))
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.next_round != self.sequence.num_rounds() {
            return Err(sparse_error(
                "sparse RAM RA virtualization finished before its final message",
            ));
        }
        self.sequence
            .bind(bind)
            .map_err(|error| sparse_error(error.to_string()))
    }
}

impl SumcheckKernel<AkitaField> for HostSparseRamRaVirtualizationKernel {
    type Relation = RamRaVirtualization<AkitaField>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<RamRaVirtualizationOutputClaims<AkitaField>, SumcheckKernelError<AkitaField>> {
        let terminal = self.sequence.terminal().map_err(kernel_error)?;
        Ok(RamRaVirtualizationOutputClaims {
            ram_ra: terminal.ram_ra().to_vec(),
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
        let id = JoltDerivedId::from(RamRaVirtualizationPublic::EqCycle);
        let expected = relation.derive_output_term(&id, input_points, output_points, challenges)?;
        let got = terminal.eq_cycle();
        if got != expected {
            return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
        }
        Ok(())
    }
}

impl PrepareKernel<AkitaField, RamRaVirtualization<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, RamRaVirtualization<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = RamRaVirtualization<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let log_t = dimensions.log_t();
        let cycles = 1usize
            .checked_shl(u32::try_from(log_t).map_err(|_| KernelError::Unsupported {
                reason: "RAM RA virtualization cycle domain is too large",
            })?)
            .ok_or(KernelError::Unsupported {
                reason: "RAM RA virtualization cycle domain is too large",
            })?;
        if cycles < self.config.ram_ra_virtualization.trace_cutoff_elements {
            return OptimizedBackend.prepare(session, witness, inputs);
        }

        let r_address = relation.ram_reduced_address();
        let r_cycle = relation.ram_reduced_cycle();
        if r_cycle.len() != log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "RAM RA reduced cycle point".to_owned(),
                expected: log_t,
                got: r_cycle.len(),
            });
        }
        let chunk_bits = relation.committed_chunk_bits();
        if chunk_bits == 0 || chunk_bits > u32::BITS as usize {
            return Err(KernelError::Unsupported {
                reason: "committed RAM RA chunk width outside the supported one-hot range",
            });
        }
        if committed_address_chunks(r_address, chunk_bits).len()
            != dimensions.num_committed_ra_polys()
        {
            return Err(KernelError::InvariantViolation {
                reason: "RAM address chunk count disagrees with the committed RA count",
            });
        }

        let log_k = r_address.len();
        let Some(owner) = shared_ram_cycle_family_owner(session, witness, log_t, log_k)? else {
            return OptimizedBackend.prepare(session, witness, inputs);
        };
        let predicted = estimated_ram_ra_virtualization_products(&owner, chunk_bits)
            .map_err(|error| prepare_error(error.to_string()))?;
        if predicted > MAX_SPARSE_PRODUCTS {
            return OptimizedBackend.prepare(session, witness, inputs);
        }
        let sequence = HostSparseRamRaVirtualization::new_from_verified_owner(
            owner, r_address, chunk_bits, r_cycle,
        )
        .map_err(|error| prepare_error(error.to_string()))?;
        let _ = session.take::<Arc<RamAccessColumns>>();
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .ram_ra_virtualization_sparse_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        tracing::info!(
            target: "jolt::metal",
            predicted_products = predicted,
            "prepared sparse RAM RA virtualization sequence"
        );
        Ok(Box::new(HostSparseRamRaVirtualizationKernel {
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
    use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
    use jolt_claims::protocols::jolt::geometry::ram::{
        committed_ram_ra, RamRaVirtualizationDimensions,
    };
    use jolt_claims::protocols::jolt::relations::ram::RamRaVirtualizationInputClaims;
    use jolt_claims::NoChallenges;
    use jolt_poly::EqPolynomial;

    use super::*;
    use crate::metal::solinas::ram_cycle_family_v3::RamCycleFamilyOwner;
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
            let chunk_bits = 4;
            let r_address = point(11, shape.log_k());
            let r_cycle = point(71, shape.log_t);
            let chunks = committed_address_chunks(&r_address, chunk_bits);
            let relation = RamRaVirtualization::<AkitaField>::new(
                RamRaVirtualizationDimensions::new(shape.log_t, chunks.len()),
                r_address.clone(),
                r_cycle.clone(),
                chunk_bits,
            );
            let eq_cycle = EqPolynomial::new(r_cycle).evaluations();
            let folded = chunks
                .iter()
                .enumerate()
                .map(|(index, chunk)| {
                    address_fold::<AkitaField>(witness, committed_ram_ra(index), shape.log_t, chunk)
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let input_claim = (0..1usize << shape.log_t)
                .map(|cycle| {
                    folded
                        .iter()
                        .fold(eq_cycle[cycle], |product, values| product * values[cycle])
                })
                .sum();
            let claims = RamRaVirtualizationInputClaims {
                ram_ra_reduced: input_claim,
            };
            let points = RamRaVirtualizationInputClaims::<Vec<AkitaField>>::default();
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
            config.ram_ra_virtualization.trace_cutoff_elements = 1 << shape.log_t;
            let metal = MetalBackend::new(config).unwrap();
            let mut session = ProofSession::default();
            let mut actual =
                PrepareKernel::prepare(&metal, &mut session, witness, inputs()).unwrap();
            assert_eq!(metal.ram_ra_virtualization_sparse_sequences(), 1);
            assert!(session.state::<Arc<RamCycleFamilyOwner>>().is_some());
            assert!(session.state::<Arc<RamAccessColumns>>().is_none());

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
