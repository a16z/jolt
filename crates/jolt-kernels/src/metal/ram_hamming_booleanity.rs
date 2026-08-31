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
use super::solinas::ram_cycle_family::{
    HostSparseRamHammingBooleanity, RamCycleFamilyOwner, RamHammingSparsePlan,
};
use super::solinas::RamHammingSequence;
use crate::optimized::ram_hamming_booleanity::OptimizedRamHammingBooleanity;
use crate::optimized::ram_trace::RamAccessColumns;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const METAL_RAM_HAMMING_MIN_ELEMENTS: usize = 64;

#[cfg(feature = "test-utils")]
mod evaluation;
#[cfg(feature = "test-utils")]
pub use evaluation::{
    RamHammingBooleanityCpuEvalFixture, RamHammingBooleanityEvalError,
    RamHammingBooleanityEvalResult, RamHammingBooleanityEvalSample,
    RamHammingBooleanityRoundTiming, RamHammingBooleanityShapeSnapshot,
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

struct MetalRamHammingKernel {
    sequence: RamHammingSequence,
    rounds: usize,
    next_round: usize,
    terminal: Option<[AkitaField; 2]>,
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

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalRamHammingKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(allocative::Key::new("sequence"), &self.sequence);
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

impl ProveRounds<AkitaField> for MetalRamHammingKernel {
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
                "RAM Hamming Metal sequence received an out-of-order round",
            ));
        }
        let polynomial = match bind {
            Some(challenge) => self
                .sequence
                .bind_and_message(challenge, previous_claim)
                .map_err(|error| metal_error(error.to_string()))?,
            None => self
                .sequence
                .message(previous_claim)
                .map_err(|error| metal_error(error.to_string()))?,
        };
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
        if self.next_round != self.rounds || self.terminal.is_some() {
            return Err(metal_error(
                "RAM Hamming Metal sequence cannot finish before all rounds",
            ));
        }
        let terminal = self
            .sequence
            .finish_bind(bind)
            .map_err(|error| metal_error(error.to_string()))?;
        self.terminal = Some([terminal.hamming(), terminal.eq_cycle()]);
        Ok(())
    }
}

impl MetalRamHammingKernel {
    fn terminal(&self) -> Result<[AkitaField; 2], SumcheckKernelError<AkitaField>> {
        self.terminal
            .ok_or(SumcheckKernelError::NotFullyBound { remaining: 1 })
    }
}

impl SumcheckKernel<AkitaField> for MetalRamHammingKernel {
    type Relation = RamHammingBooleanity<AkitaField>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<RamHammingBooleanityOutputClaims<AkitaField>, SumcheckKernelError<AkitaField>> {
        Ok(RamHammingBooleanityOutputClaims {
            ram_hamming_weight: self.terminal()?[0],
        })
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        let id = JoltDerivedId::from(RamHammingBooleanityPublic::EqCycle);
        let expected =
            match relation.derive_output_term(&id, input_points, output_points, challenges) {
                Ok(value) => value,
                Err(VerifierError::MissingStageClaimDerived { .. }) => return Ok(()),
                Err(error) => return Err(error.into()),
            };
        let got = self.terminal()?[1];
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
            record_route(cycles, "optimized_cpu", "below_cutoff", 0, 0);
            return OptimizedRamHammingBooleanity.prepare(session, witness, inputs);
        }
        let stage1_cycle_binding = relation.stage1_cycle_binding();
        if stage1_cycle_binding.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "stage-1 cycle binding has the wrong variable count",
            });
        }
        if cycles < METAL_RAM_HAMMING_MIN_ELEMENTS {
            record_route(cycles, "optimized_cpu", "below_device_minimum", 0, 0);
            return OptimizedRamHammingBooleanity.prepare(session, witness, inputs);
        }

        let Some(owner) = session.state::<Arc<RamCycleFamilyOwner>>().cloned() else {
            return prepare_metal_ram_hamming(
                self,
                session,
                witness,
                stage1_cycle_binding,
                log_t,
                cycles,
                "missing_owner",
                0,
                0,
            );
        };
        if owner.receipt().log_t() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "RAM cycle-family owner has stale geometry",
            });
        }
        if minimum_sparse_products(owner.receipt().access_count())
            > self.config.ram_hamming_booleanity.max_sparse_products
        {
            return prepare_metal_ram_hamming(
                self,
                session,
                witness,
                stage1_cycle_binding,
                log_t,
                cycles,
                "sparse_lower_bound",
                owner.receipt().source_generation(),
                owner.receipt().fingerprint(),
            );
        }
        let prepare_span = tracing::info_span!(
            "MetalRamHammingBooleanity::sparse_prepare",
            selected = tracing::field::Empty,
            fallback_reason = tracing::field::Empty,
            source_generation = owner.receipt().source_generation(),
            source_fingerprint = owner.receipt().fingerprint(),
            log_t,
            access_leaves = tracing::field::Empty,
            parent_nodes = tracing::field::Empty,
            middle_nodes = tracing::field::Empty,
            rounds = log_t,
            estimated_products = tracing::field::Empty,
            product_cap = tracing::field::Empty,
            topology_builds = tracing::field::Empty,
            topology_bytes = tracing::field::Empty,
            member_heap_bytes_including_topology = tracing::field::Empty,
            non_topology_heap_bytes = tracing::field::Empty,
            additional_source_row_scans = 0,
            dense_h_elements = 0,
            member_upload_bytes = 0,
            gpu_dispatches = 0,
            command_buffers = 0,
            waits = 0,
            readbacks = 0,
            complete_plan = tracing::field::Empty,
        );
        let _prepare_guard = prepare_span.enter();
        let plan =
            RamHammingSparsePlan::new(&owner).map_err(|error| prepare_error(error.to_string()))?;
        let predicted = plan.estimated_products();
        let estimated_products = u64::try_from(predicted).map_err(|_| {
            prepare_error("RAM Hamming sparse product estimate does not fit telemetry")
        })?;
        let product_cap = u64::try_from(self.config.ram_hamming_booleanity.max_sparse_products)
            .map_err(|_| prepare_error("RAM Hamming sparse product cap does not fit telemetry"))?;
        if predicted > self.config.ram_hamming_booleanity.max_sparse_products {
            let _ = prepare_span.record("selected", "packed_access_width32_v1");
            let _ = prepare_span.record("fallback_reason", "product_cap");
            let _ = prepare_span.record("access_leaves", plan.access_leaves());
            let _ = prepare_span.record("parent_nodes", plan.parent_nodes());
            let _ = prepare_span.record("middle_nodes", plan.middle_nodes());
            let _ = prepare_span.record("estimated_products", estimated_products);
            let _ = prepare_span.record("product_cap", product_cap);
            let _ = prepare_span.record("topology_builds", 1);
            let _ = prepare_span.record("topology_bytes", plan.topology_bytes());
            let _ = prepare_span.record("complete_plan", true);
            let source_generation = owner.receipt().source_generation();
            let source_fingerprint = owner.receipt().fingerprint();
            drop(_prepare_guard);
            return prepare_metal_ram_hamming(
                self,
                session,
                witness,
                stage1_cycle_binding,
                log_t,
                cycles,
                "product_cap",
                source_generation,
                source_fingerprint,
            );
        }
        let access_leaves = plan.access_leaves();
        let parent_nodes = plan.parent_nodes();
        let middle_nodes = plan.middle_nodes();
        let topology_bytes = plan.topology_bytes();
        let source_generation = owner.receipt().source_generation();
        let source_fingerprint = owner.receipt().fingerprint();
        let sequence =
            HostSparseRamHammingBooleanity::new_from_plan(owner, stage1_cycle_binding, plan)
                .map_err(|error| prepare_error(error.to_string()))?;
        let member_heap_bytes_including_topology = sequence.owned_heap_bytes();
        let non_topology_heap_bytes = member_heap_bytes_including_topology
            .checked_sub(topology_bytes)
            .ok_or(KernelError::InvariantViolation {
                reason: "RAM Hamming member heap ledger underflowed",
            })?;
        let _ = prepare_span.record("selected", "host_sparse_v1");
        let _ = prepare_span.record("fallback_reason", "none");
        let _ = prepare_span.record("access_leaves", access_leaves);
        let _ = prepare_span.record("parent_nodes", parent_nodes);
        let _ = prepare_span.record("middle_nodes", middle_nodes);
        let _ = prepare_span.record("estimated_products", estimated_products);
        let _ = prepare_span.record("product_cap", product_cap);
        let _ = prepare_span.record("topology_builds", 1);
        let _ = prepare_span.record("topology_bytes", topology_bytes);
        let _ = prepare_span.record(
            "member_heap_bytes_including_topology",
            member_heap_bytes_including_topology,
        );
        let _ = prepare_span.record("non_topology_heap_bytes", non_topology_heap_bytes);
        let _ = prepare_span.record("complete_plan", true);
        record_route(
            cycles,
            "host_sparse_v1",
            "none",
            source_generation,
            source_fingerprint,
        );
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .test_counters
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

#[expect(
    clippy::too_many_arguments,
    reason = "the production route receipt keeps geometry and source identity explicit"
)]
fn prepare_metal_ram_hamming(
    backend: &MetalBackend,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    stage1_cycle_binding: &[AkitaField],
    log_t: usize,
    cycles: usize,
    fallback_reason: &'static str,
    source_generation: u64,
    source_fingerprint: u64,
) -> Result<
    Box<dyn SumcheckKernel<AkitaField, Relation = RamHammingBooleanity<AkitaField>>>,
    KernelError<AkitaField>,
> {
    let route = tracing::info_span!(
        "MetalRamHammingBooleanity::device_prepare",
        cycles,
        log_t,
        requested = "packed_access_width32_v1",
        selected = "packed_access_width32_v1",
        fallback_reason,
        source_generation,
        source_fingerprint,
        additional_source_row_scans = 0,
        member_upload_bytes = 0,
        complete_sequence = true,
    );
    let _route_guard = route.enter();
    let columns = RamAccessColumns::shared(session, witness, log_t)?;
    let sequence = backend
        .context
        .prepare_ram_hamming_sequence(columns, stage1_cycle_binding)
        .map_err(|error| KernelError::Sumcheck(metal_error(error.to_string())))?;
    record_route(
        cycles,
        "packed_access_width32_v1",
        fallback_reason,
        source_generation,
        source_fingerprint,
    );
    #[cfg(any(test, feature = "test-utils"))]
    let _ = backend
        .test_counters
        .ram_hamming_metal_sequences
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(Box::new(MetalRamHammingKernel {
        sequence,
        rounds: log_t,
        next_round: 0,
        terminal: None,
    }))
}

fn minimum_sparse_products(access_leaves: usize) -> u128 {
    (access_leaves.div_ceil(2) as u128).saturating_mul(7)
}

fn record_route(
    cycles: usize,
    selected: &'static str,
    fallback_reason: &'static str,
    source_generation: u64,
    source_fingerprint: u64,
) {
    let _span = tracing::info_span!(
        "MetalRamHammingBooleanity::route",
        cycles,
        requested = "hybrid_sparse_or_packed_v1",
        selected,
        fallback_reason,
        source_generation,
        source_fingerprint,
    )
    .entered();
}

fn sparse_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
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
            log_t: 6,
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

            let mut direct_expected = OptimizedRamHammingBooleanity
                .prepare(&mut ProofSession::default(), witness, inputs())
                .unwrap();
            let mut direct_config = MetalConfig::default();
            direct_config.ram_hamming_booleanity.trace_cutoff_elements = 1 << shape.log_t;
            direct_config.ram_hamming_booleanity.max_sparse_products = 0;
            let direct_metal = MetalBackend::new(direct_config).unwrap();
            let mut direct_session = ProofSession::default();
            assert!(shared_ram_cycle_family_owner(
                &mut direct_session,
                witness,
                shape.log_t,
                shape.log_k(),
            )
            .unwrap()
            .is_some());
            let mut direct =
                PrepareKernel::prepare(&direct_metal, &mut direct_session, witness, inputs())
                    .unwrap();
            assert_eq!(direct_metal.ram_hamming_sparse_sequences(), 0);
            assert_eq!(direct_metal.ram_hamming_metal_sequences(), 1);

            let mut direct_claim = AkitaField::zero();
            let mut direct_bind = None;
            for (round, challenge) in round_challenges.iter().copied().enumerate() {
                let expected_poly = direct_expected
                    .prove_round(direct_bind, round, direct_claim)
                    .unwrap();
                let actual_poly = direct
                    .prove_round(direct_bind, round, direct_claim)
                    .unwrap();
                assert_eq!(actual_poly, expected_poly, "direct round {round} mismatch");
                direct_claim = expected_poly.evaluate(challenge);
                direct_bind = Some(challenge);
            }
            direct_expected.finish_rounds(last).unwrap();
            direct.finish_rounds(last).unwrap();
            assert_eq!(
                direct.output_claims(&claims).unwrap(),
                direct_expected.output_claims(&claims).unwrap()
            );
            direct
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
        });
    }

    #[test]
    fn sparse_product_cap_is_inclusive_and_selects_device_before_round_zero() {
        let shape = FixtureShape {
            log_t: 6,
            ram_k: 64,
        };
        let ops = vec![
            RamOp::Write { word: 3, post: 5 },
            RamOp::Read { word: 3 },
            RamOp::Write { word: 57, post: 9 },
        ];
        with_ram_fixture_backend(shape, ops, |witness| {
            let relation = RamHammingBooleanity::new(
                TraceDimensions::new(shape.log_t),
                point(71, shape.log_t),
            );
            let claims = RamHammingBooleanityInputClaims::default();
            let points = RamHammingBooleanityInputClaims::default();
            let challenges = NoChallenges::default();
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

            let mut session = ProofSession::default();
            let owner =
                shared_ram_cycle_family_owner(&mut session, witness, shape.log_t, shape.log_k())
                    .unwrap()
                    .unwrap();
            let predicted = RamHammingSparsePlan::new(&owner)
                .unwrap()
                .estimated_products();
            assert!(predicted > 0);
            let mut config = MetalConfig::default();
            config.ram_hamming_booleanity.trace_cutoff_elements = 1 << shape.log_t;
            config.ram_hamming_booleanity.max_sparse_products = predicted;
            let metal = MetalBackend::new(config).unwrap();
            let _kernel = PrepareKernel::prepare(&metal, &mut session, witness, inputs()).unwrap();
            assert_eq!(metal.ram_hamming_sparse_sequences(), 1);

            let mut fallback_session = ProofSession::default();
            assert!(shared_ram_cycle_family_owner(
                &mut fallback_session,
                witness,
                shape.log_t,
                shape.log_k(),
            )
            .unwrap()
            .is_some());
            let mut fallback_config = MetalConfig::default();
            fallback_config.ram_hamming_booleanity.trace_cutoff_elements = 1 << shape.log_t;
            fallback_config.ram_hamming_booleanity.max_sparse_products = predicted - 1;
            let fallback = MetalBackend::new(fallback_config).unwrap();
            let _fallback_kernel =
                PrepareKernel::prepare(&fallback, &mut fallback_session, witness, inputs())
                    .unwrap();
            assert_eq!(fallback.ram_hamming_sparse_sequences(), 0);
            assert_eq!(fallback.ram_hamming_metal_sequences(), 1);
        });
    }
}
