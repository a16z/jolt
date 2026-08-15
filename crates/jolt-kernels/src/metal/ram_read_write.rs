use jolt_claims::protocols::jolt::{JoltDerivedId, RamReadWritePublic};
use jolt_field::{AkitaField, FromPrimitiveInt};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::ram_read_write_checking::{
    RamReadWriteChecking, RamReadWriteOutputClaims,
};
use jolt_verifier::VerifierError;
use jolt_witness::JoltWitnessPlane;

use super::backend::MetalBackend;
use super::ram_cycle_family::shared_ram_cycle_family_owner;
use crate::optimized::ram_trace::RamAccessValues;
use crate::optimized::rw_matrix::{AddressMajorMatrix, CycleMajorEntry, CycleMajorMatrix};
use crate::optimized::OptimizedBackend;
use crate::ram_access::RamAccessTape;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[derive(Clone, Copy)]
struct SparseCycleEntry {
    block: usize,
    value: AkitaField,
}

struct SparseCyclePolynomial {
    entries: Vec<SparseCycleEntry>,
    rounds_bound: usize,
    rounds: usize,
}

impl SparseCyclePolynomial {
    fn pair(&self, parent: usize) -> [AkitaField; 2] {
        let low_block = 2 * parent;
        let high_block = low_block + 1;
        let low = self.value(low_block);
        [low, self.value(high_block) - low]
    }

    fn value(&self, block: usize) -> AkitaField {
        self.entries
            .binary_search_by_key(&block, |entry| entry.block)
            .ok()
            .map_or(AkitaField::zero(), |index| self.entries[index].value)
    }

    fn bind(&mut self, challenge: AkitaField) {
        let mut bound = Vec::with_capacity(self.entries.len());
        let mut index = 0;
        while index < self.entries.len() {
            let parent = self.entries[index].block / 2;
            let mut low = AkitaField::zero();
            let mut high = AkitaField::zero();
            while self
                .entries
                .get(index)
                .is_some_and(|entry| entry.block / 2 == parent)
            {
                let entry = self.entries[index];
                if entry.block.is_multiple_of(2) {
                    low = entry.value;
                } else {
                    high = entry.value;
                }
                index += 1;
            }
            let value = low + challenge * (high - low);
            if value != AkitaField::zero() {
                bound.push(SparseCycleEntry {
                    block: parent,
                    value,
                });
            }
        }
        self.entries = bound;
        self.rounds_bound += 1;
    }

    fn final_value(&self) -> Result<AkitaField, SumcheckKernelError<AkitaField>> {
        let remaining = self.rounds - self.rounds_bound;
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        match self.entries.as_slice() {
            [] => Ok(AkitaField::zero()),
            [entry] if entry.block == 0 => Ok(entry.value),
            _ => Err(SumcheckKernelError::InvariantViolation {
                reason: "sparse RAM increment frontier has an invalid terminal state",
            }),
        }
    }

    #[cfg(feature = "allocative")]
    fn heap_bytes(&self) -> usize {
        self.entries.capacity() * std::mem::size_of::<SparseCycleEntry>()
    }
}

enum Phase {
    Cycle {
        matrix: CycleMajorMatrix<AkitaField>,
        gruen: GruenSplitEqPolynomial<AkitaField>,
    },
    Address {
        matrix: AddressMajorMatrix<AkitaField>,
        merged_eq: Polynomial<AkitaField>,
    },
    Done {
        merged_eq: Polynomial<AkitaField>,
        final_ra: AkitaField,
        final_val: AkitaField,
    },
}

struct HostSparseRamReadWriteKernel {
    phase: Option<Phase>,
    inc: SparseCyclePolynomial,
    val_init: Polynomial<AkitaField>,
    gamma: AkitaField,
    log_t: usize,
    log_k: usize,
    source_generation: u64,
    source_fingerprint: u64,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for HostSparseRamReadWriteKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        use crate::backend::{gruen_heap_bytes, poly_heap_bytes, vec_heap_bytes};
        let mut visitor = visitor.enter_self_sized::<Self>();
        let phase_bytes = self.phase.as_ref().map_or(0, |phase| match phase {
            Phase::Cycle { matrix, gruen } => {
                vec_heap_bytes(&matrix.entries) + gruen_heap_bytes(gruen)
            }
            Phase::Address { matrix, merged_eq } => {
                vec_heap_bytes(&matrix.entries) + poly_heap_bytes(merged_eq)
            }
            Phase::Done { merged_eq, .. } => poly_heap_bytes(merged_eq),
        });
        visitor.visit_simple(allocative::Key::new("phase"), phase_bytes);
        visitor.visit_simple(allocative::Key::new("inc"), self.inc.heap_bytes());
        visitor.visit_simple(
            allocative::Key::new("val_init"),
            poly_heap_bytes(&self.val_init),
        );
        visitor.exit();
    }
}

impl HostSparseRamReadWriteKernel {
    fn ingest(
        &mut self,
        challenge: AkitaField,
        round: usize,
    ) -> Result<(), SumcheckError<AkitaField>> {
        if round < self.log_t {
            let Some(Phase::Cycle { matrix, gruen }) = &mut self.phase else {
                return Err(phase_error());
            };
            matrix.bind(challenge);
            gruen.bind(challenge);
            self.inc.bind(challenge);
            if round == self.log_t - 1 {
                let Some(Phase::Cycle { matrix, gruen }) = self.phase.take() else {
                    return Err(phase_error());
                };
                self.phase = Some(Phase::Address {
                    matrix: matrix.into_address_major(),
                    merged_eq: gruen.merge(),
                });
                if self.log_k == 0 {
                    self.finalize()?;
                }
            }
        } else {
            let Some(Phase::Address { matrix, .. }) = &mut self.phase else {
                return Err(phase_error());
            };
            matrix.bind(challenge, &mut self.val_init);
            if round == self.log_t + self.log_k - 1 {
                self.finalize()?;
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let Some(Phase::Address { matrix, merged_eq }) = self.phase.take() else {
            return Err(phase_error());
        };
        let (final_ra, final_val) = matrix.final_values(&self.val_init);
        self.phase = Some(Phase::Done {
            merged_eq,
            final_ra,
            final_val,
        });
        Ok(())
    }

    fn cycle_round_message(
        &self,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        let Some(Phase::Cycle { matrix, gruen }) = &self.phase else {
            return Err(phase_error());
        };
        let e_in = gruen.e_in_current();
        let e_out = gruen.e_out_current();
        let in_bits = e_in.len().trailing_zeros() as usize;
        let in_mask = e_in.len() - 1;
        let [q_0, q_infinity] = matrix.quadratic_coefficients_with(
            |pair| e_out[pair >> in_bits] * e_in[pair & in_mask],
            |pair| self.inc.pair(pair),
            self.gamma,
        );
        Ok(gruen.gruen_poly_deg_3(q_0, q_infinity, previous_claim))
    }

    fn address_round_message(
        &self,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        let Some(Phase::Address { matrix, merged_eq }) = &self.phase else {
            return Err(phase_error());
        };
        let inc = self
            .inc
            .final_value()
            .map_err(|error| sparse_error(error.to_string()))?;
        let evals = matrix.address_round_evals_scalars(
            &self.val_init,
            inc,
            merged_eq.evals()[0],
            self.gamma,
        );
        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }
}

impl ProveRounds<AkitaField> for HostSparseRamReadWriteKernel {
    fn num_rounds(&self) -> usize {
        self.log_t + self.log_k
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if let Some(challenge) = bind {
            self.ingest(challenge, round - 1)?;
        }
        if round < self.log_t {
            self.cycle_round_message(previous_claim)
        } else {
            self.address_round_message(previous_claim)
        }
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        self.ingest(bind, self.num_rounds() - 1)
    }
}

impl SumcheckKernel<AkitaField> for HostSparseRamReadWriteKernel {
    type Relation = RamReadWriteChecking<AkitaField>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<AkitaField, Self::Relation>, SumcheckKernelError<AkitaField>>
    {
        let Some(Phase::Done {
            final_ra,
            final_val,
            ..
        }) = &self.phase
        else {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.num_rounds(),
            });
        };
        let claims = RamReadWriteOutputClaims {
            val: *final_val,
            ra: *final_ra,
            inc: self.inc.final_value()?,
        };
        let _span = tracing::info_span!(
            "MetalRamReadWrite::sparse_complete",
            selected = "host_sparse_v1",
            source_generation = self.source_generation,
            source_fingerprint = self.source_fingerprint,
            output_claims_valid = true,
        )
        .entered();
        Ok(claims)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        let Some(Phase::Done { merged_eq, .. }) = &self.phase else {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.num_rounds(),
            });
        };
        let id = JoltDerivedId::from(RamReadWritePublic::EqCycle);
        let expected =
            match relation.derive_output_term(&id, input_points, output_points, challenges) {
                Ok(value) => value,
                Err(VerifierError::MissingStageClaimDerived { .. }) => return Ok(()),
                Err(error) => return Err(error.into()),
            };
        let got = merged_eq.evals()[0];
        if got != expected {
            return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
        }
        let _span = tracing::info_span!(
            "MetalRamReadWrite::sparse_derived_validate",
            source_generation = self.source_generation,
            source_fingerprint = self.source_fingerprint,
            derived_claim_valid = true,
        )
        .entered();
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
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let log_t = dimensions.log_t();
        let log_k = relation.ram_log_k();
        let tau_low = relation.product_tau_low();
        if dimensions.phase1_num_rounds() != log_t {
            record_route(log_t, log_k, "optimized_cpu", "unsupported_phase", 0, 0);
            return OptimizedBackend.prepare(session, witness, inputs);
        }
        if log_t == 0 || dimensions.log_k() != log_k || tau_low.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "RAM read-write checking geometry is inconsistent",
            });
        }
        let Some(owner) = shared_ram_cycle_family_owner(session, witness, log_t, log_k)? else {
            record_route(log_t, log_k, "optimized_cpu", "missing_owner", 0, 0);
            return OptimizedBackend.prepare(session, witness, inputs);
        };
        let source_generation = owner.receipt().source_generation();
        let source_fingerprint = owner.receipt().fingerprint();
        let sparse_prepare = tracing::info_span!(
            "MetalRamReadWrite::sparse_prepare",
            selected = "host_sparse_v1",
            source_generation,
            source_fingerprint,
            log_t,
            log_k,
            rounds = log_t + log_k,
            access_records = owner.receipt().access_count(),
            increment_records = owner.receipt().increment_count(),
            owner_bytes = owner.owned_heap_bytes(),
            cycle_cutoff = 0,
            additional_source_row_scans = 0,
            member_upload_bytes = 0,
            gpu_dispatches = 0,
            command_buffers = 0,
            waits = 0,
            readbacks = 0,
        );
        let _sparse_prepare_guard = sparse_prepare.enter();
        let _ = session
            .take::<RamAccessValues>()
            .ok_or(KernelError::InvariantViolation {
                reason: "RAM sparse read-write lost the shared value columns",
            })?;

        let mut val_init = owner
            .final_memory()
            .iter()
            .copied()
            .map(AkitaField::from_u64)
            .collect::<Vec<_>>();
        let mut seen = vec![false; val_init.len()];
        let entries = owner
            .access_records()
            .iter()
            .map(|record| {
                let address = record.address() as usize;
                if !seen[address] {
                    seen[address] = true;
                    val_init[address] = AkitaField::from_u64(record.pre_value());
                }
                CycleMajorEntry {
                    row: record.cycle() as usize,
                    col: address,
                    prev_val: record.pre_value(),
                    next_val: record.post_value(),
                    val: AkitaField::from_u64(record.pre_value()),
                    ra: AkitaField::one(),
                }
            })
            .collect::<Vec<_>>();
        let increments = owner
            .increment_records()
            .map(|record| SparseCycleEntry {
                block: record.cycle() as usize,
                value: AkitaField::from_i128(record.increment()),
            })
            .collect::<Vec<_>>();
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .test_counters
            .ram_read_write_sparse_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        tracing::info!(
            target: "jolt::metal",
            generation = owner.receipt().source_generation(),
            accesses = entries.len(),
            increments = increments.len(),
            "prepared sparse RAM read-write sequence"
        );
        record_route(
            log_t,
            log_k,
            "host_sparse_v1",
            "none",
            source_generation,
            source_fingerprint,
        );
        let _ = session
            .take::<RamAccessTape>()
            .ok_or(KernelError::InvariantViolation {
                reason: "RAM sparse read-write lost the retained access tape",
            })?;
        Ok(Box::new(HostSparseRamReadWriteKernel {
            phase: Some(Phase::Cycle {
                matrix: CycleMajorMatrix { entries },
                gruen: GruenSplitEqPolynomial::new(tau_low, BindingOrder::LowToHigh),
            }),
            inc: SparseCyclePolynomial {
                entries: increments,
                rounds_bound: 0,
                rounds: log_t,
            },
            val_init: Polynomial::new(val_init),
            gamma: inputs.challenges.gamma,
            log_t,
            log_k,
            source_generation,
            source_fingerprint,
        }))
    }
}

fn record_route(
    log_t: usize,
    log_k: usize,
    selected: &'static str,
    fallback_reason: &'static str,
    source_generation: u64,
    source_fingerprint: u64,
) {
    let cycles = 1usize
        .checked_shl(u32::try_from(log_t).unwrap_or(u32::MAX))
        .unwrap_or(0);
    let _span = tracing::info_span!(
        "MetalRamReadWrite::route",
        cycles,
        log_t,
        log_k,
        requested = "host_sparse_v1",
        selected,
        fallback_reason,
        source_generation,
        source_fingerprint,
    )
    .entered();
}

fn phase_error() -> SumcheckError<AkitaField> {
    SumcheckError::MissingEvaluationSource {
        kind: "RAM read-write sparse phase state",
    }
}

fn sparse_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "host-sparse",
        message: message.into(),
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Metal parity test setup")]
mod tests {
    use std::sync::Arc;

    use jolt_claims::protocols::jolt::geometry::dimensions::ReadWriteDimensions;
    use jolt_claims::protocols::jolt::geometry::ram::{ram_inc, ram_ra, ram_val};
    use jolt_poly::EqPolynomial;
    use jolt_verifier::stages::stage2::ram_read_write_checking::{
        RamReadWriteChallenges, RamReadWriteInputClaims,
    };

    use super::*;
    use crate::metal::solinas::ram_cycle_family::RamCycleFamilyOwner;
    use crate::metal::MetalConfig;
    use crate::optimized::harness::run_lockstep;
    use crate::optimized::testing::{with_ram_fixture_backend, FixtureShape, RamOp};

    fn point(seed: u64, len: usize) -> Vec<AkitaField> {
        (0..len as u64)
            .map(|index| AkitaField::from_u64(seed + 37 * index + 5))
            .collect()
    }

    fn dense_input_claim(
        witness: &dyn JoltWitnessPlane<AkitaField>,
        tau_low: &[AkitaField],
        gamma: AkitaField,
        ram_k: usize,
    ) -> AkitaField {
        let cycles = 1usize << tau_low.len();
        let eq = EqPolynomial::new(tau_low.to_vec()).evaluations();
        let ra: Vec<AkitaField> = witness.oracle_table(ram_ra().polynomial_id()).unwrap();
        let val: Vec<AkitaField> = witness.oracle_table(ram_val().polynomial_id()).unwrap();
        let inc: Vec<AkitaField> = witness.oracle_table(ram_inc().polynomial_id()).unwrap();
        let mut claim = AkitaField::zero();
        for address in 0..ram_k {
            for cycle in 0..cycles {
                let index = address * cycles + cycle;
                claim += eq[cycle] * ra[index] * (val[index] + gamma * (val[index] + inc[cycle]));
            }
        }
        claim
    }

    #[test]
    fn host_sparse_sequence_matches_optimized_cpu() {
        let shape = FixtureShape {
            log_t: 5,
            ram_k: 16,
        };
        let ops = vec![
            RamOp::Write { word: 3, post: 5 },
            RamOp::Read { word: 3 },
            RamOp::Write { word: 3, post: 9 },
            RamOp::Read { word: 7 },
            RamOp::None,
            RamOp::Write { word: 4, post: 2 },
            RamOp::Read { word: 3 },
            RamOp::Write { word: 7, post: 6 },
        ];
        with_ram_fixture_backend(shape, ops, |witness| {
            let tau_low = point(17, shape.log_t);
            let relation = RamReadWriteChecking::<AkitaField>::new(
                ReadWriteDimensions::new(shape.log_t, shape.log_k(), shape.log_t, shape.log_k()),
                shape.log_k(),
                tau_low.clone(),
            );
            let claims = RamReadWriteInputClaims::<AkitaField>::default();
            let points = RamReadWriteInputClaims::<Vec<AkitaField>>::default();
            let challenges = RamReadWriteChallenges {
                gamma: AkitaField::from_u64(23),
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
            let metal = MetalBackend::new(MetalConfig::default()).unwrap();
            let mut session = ProofSession::default();
            let mut actual =
                PrepareKernel::prepare(&metal, &mut session, witness, inputs()).unwrap();
            assert_eq!(metal.ram_read_write_sparse_sequences(), 1);
            assert!(session.state::<RamAccessValues>().is_none());
            assert!(session
                .state::<crate::ram_access::RamAccessTape>()
                .is_none());
            assert!(session.state::<Arc<RamCycleFamilyOwner>>().is_some());

            let input_claim = dense_input_claim(witness, &tau_low, challenges.gamma, shape.ram_k);
            let round_challenges = point(211, shape.log_t + shape.log_k());
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
