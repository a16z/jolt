use jolt_claims::protocols::jolt::geometry::ram::ram_val_final;
use jolt_claims::protocols::jolt::{JoltDerivedId, RamReadWritePublic};
use jolt_field::{AkitaField, CanonicalU64, FromPrimitiveInt};
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
use crate::optimized::ram_trace::{RamAccessColumns, RamAccessValues};
use crate::optimized::rw_matrix::{
    AddressMajorEntry, AddressMajorMatrix, CycleMajorEntry, CycleMajorMatrix,
};
use crate::optimized::OptimizedBackend;
use crate::ram_access::RamAccessTape;
use crate::reference::views::dense_view;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

use super::solinas::{MetalError, RamReadWriteFinish, RamReadWriteSequence, SparseCycleProduct};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamReadWriteMetalConfig {
    pub trace_cutoff_elements: usize,
    pub minimum_accesses: usize,
}

impl Default for RamReadWriteMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 25,
            minimum_accesses: crate::ram_access::MAX_RETAINED_RAM_ACCESSES + 1,
        }
    }
}

type RamReadWriteKernelBox =
    Box<dyn SumcheckKernel<AkitaField, Relation = RamReadWriteChecking<AkitaField>>>;

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

enum MetalPhase {
    Cycle {
        sequence: Box<RamReadWriteSequence>,
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

struct MetalRamReadWriteKernel {
    phase: Option<MetalPhase>,
    cycle_tail: Option<SparseCycleProduct>,
    val_init: Polynomial<AkitaField>,
    gamma: AkitaField,
    log_t: usize,
    log_k: usize,
    access_count: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalRamReadWriteKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        use crate::backend::{poly_heap_bytes, vec_heap_bytes};
        let mut visitor = visitor.enter_self_sized::<Self>();
        let phase_bytes = self.phase.as_ref().map_or(0, |phase| match phase {
            MetalPhase::Cycle { sequence, .. } => sequence.resident_bytes(),
            MetalPhase::Address { matrix, merged_eq } => {
                vec_heap_bytes(&matrix.entries) + poly_heap_bytes(merged_eq)
            }
            MetalPhase::Done { merged_eq, .. } => poly_heap_bytes(merged_eq),
        });
        visitor.visit_simple(allocative::Key::new("phase"), phase_bytes);
        visitor.visit_simple(
            allocative::Key::new("val_init"),
            poly_heap_bytes(&self.val_init),
        );
        visitor.exit();
    }
}

impl MetalRamReadWriteKernel {
    fn cycle_round_message(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        let Some(MetalPhase::Cycle { sequence, gruen }) = &mut self.phase else {
            return Err(phase_error());
        };
        let observation = if let Some(challenge) = bind {
            gruen.bind(challenge);
            if let Some(tail) = &mut self.cycle_tail {
                tail.bind(challenge);
            }
            sequence
                .bind_and_message(challenge, gruen.e_in_current(), gruen.e_out_current())
                .map_err(metal_round_error)?
        } else {
            sequence
                .message(gruen.e_in_current(), gruen.e_out_current())
                .map_err(metal_round_error)?
        };
        if let Some(roots) = observation.cycle_roots {
            if self.cycle_tail.is_some() {
                return Err(metal_error(
                    "RAM read-write cycle frontier was handed off twice",
                ));
            }
            self.cycle_tail = Some(SparseCycleProduct::from_roots(roots, round, self.log_t));
        }
        let cycle_quadratic = match observation.cycle_quadratic {
            Some(quadratic) => quadratic,
            None => self
                .cycle_tail
                .as_ref()
                .ok_or_else(|| metal_error("RAM read-write lost its cycle frontier"))?
                .quadratic_coefficients(gruen.e_in_current(), gruen.e_out_current()),
        };
        let one_plus_gamma = AkitaField::one() + self.gamma;
        let quadratic = [
            one_plus_gamma * observation.address_quadratic[0] + self.gamma * cycle_quadratic[0],
            one_plus_gamma * observation.address_quadratic[1] + self.gamma * cycle_quadratic[1],
        ];
        tracing::info!(
            target: "jolt::metal",
            round,
            address_live_entries = observation.address_live_entries,
            cycle_live_entries = observation.cycle_live_entries,
            wall_ns = duration_nanos(observation.wall),
            gpu_active_ns = duration_nanos(observation.gpu_active),
            "completed high-activity RAM read-write cycle round"
        );
        Ok(gruen.gruen_poly_deg_3(quadratic[0], quadratic[1], previous_claim))
    }

    fn enter_address_phase(
        &mut self,
        challenge: AkitaField,
    ) -> Result<(), SumcheckError<AkitaField>> {
        let Some(MetalPhase::Cycle {
            mut sequence,
            mut gruen,
        }) = self.phase.take()
        else {
            return Err(phase_error());
        };
        gruen.bind(challenge);
        if let Some(tail) = &mut self.cycle_tail {
            tail.bind(challenge);
        }
        let RamReadWriteFinish {
            address_roots,
            cycle_roots,
            gpu_active,
        } = sequence.finish(challenge).map_err(metal_round_error)?;
        if let Some(roots) = cycle_roots {
            if self.cycle_tail.is_some() {
                return Err(metal_error(
                    "RAM read-write finish produced a duplicate cycle frontier",
                ));
            }
            self.cycle_tail = Some(SparseCycleProduct::from_roots(
                roots, self.log_t, self.log_t,
            ));
        }
        if self
            .cycle_tail
            .as_ref()
            .and_then(SparseCycleProduct::final_increment)
            .is_none()
        {
            return Err(metal_error(
                "RAM read-write cycle frontier has no terminal increment",
            ));
        }
        let entries: Vec<AddressMajorEntry<AkitaField>> = address_roots
            .into_iter()
            .map(|root| AddressMajorEntry {
                row: 0,
                col: root.address,
                prev_val: AkitaField::from_u64(root.previous),
                next_val: AkitaField::from_u64(root.next),
                val: root.value,
                ra: root.ra,
            })
            .collect();
        tracing::info!(
            target: "jolt::metal",
            roots = entries.len(),
            gpu_active_ns = duration_nanos(gpu_active),
            "completed high-activity RAM read-write cycle handoff"
        );
        self.phase = Some(MetalPhase::Address {
            matrix: AddressMajorMatrix { entries },
            merged_eq: gruen.merge(),
        });
        if self.log_k == 0 {
            self.finalize()?;
        }
        Ok(())
    }

    fn bind_address(
        &mut self,
        challenge: AkitaField,
        round: usize,
    ) -> Result<(), SumcheckError<AkitaField>> {
        let Some(MetalPhase::Address { matrix, .. }) = &mut self.phase else {
            return Err(phase_error());
        };
        matrix.bind(challenge, &mut self.val_init);
        if round == self.log_t + self.log_k - 1 {
            self.finalize()?;
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let Some(MetalPhase::Address { matrix, merged_eq }) = self.phase.take() else {
            return Err(phase_error());
        };
        let (final_ra, final_val) = matrix.final_values(&self.val_init);
        self.phase = Some(MetalPhase::Done {
            merged_eq,
            final_ra,
            final_val,
        });
        Ok(())
    }

    fn address_round_message(
        &self,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        let Some(MetalPhase::Address { matrix, merged_eq }) = &self.phase else {
            return Err(phase_error());
        };
        let increment = self
            .cycle_tail
            .as_ref()
            .and_then(SparseCycleProduct::final_increment)
            .ok_or_else(|| metal_error("RAM read-write lost its final increment"))?;
        let evals = matrix.address_round_evals_scalars(
            &self.val_init,
            increment,
            merged_eq.evals()[0],
            self.gamma,
        );
        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }
}

impl ProveRounds<AkitaField> for MetalRamReadWriteKernel {
    fn num_rounds(&self) -> usize {
        self.log_t + self.log_k
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if round < self.log_t {
            return self.cycle_round_message(bind, round, previous_claim);
        }
        let challenge = bind.ok_or_else(|| metal_error("RAM address phase missed a bind"))?;
        if round == self.log_t {
            self.enter_address_phase(challenge)?;
        } else {
            self.bind_address(challenge, round - 1)?;
        }
        self.address_round_message(previous_claim)
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.log_k == 0 {
            self.enter_address_phase(bind)
        } else {
            self.bind_address(bind, self.num_rounds() - 1)
        }
    }
}

impl SumcheckKernel<AkitaField> for MetalRamReadWriteKernel {
    type Relation = RamReadWriteChecking<AkitaField>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<AkitaField, Self::Relation>, SumcheckKernelError<AkitaField>>
    {
        let Some(MetalPhase::Done {
            final_ra,
            final_val,
            ..
        }) = &self.phase
        else {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.num_rounds(),
            });
        };
        let increment = self
            .cycle_tail
            .as_ref()
            .and_then(SparseCycleProduct::final_increment)
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "RAM read-write high-activity increment is not fully bound",
            })?;
        let _span = tracing::info_span!(
            "MetalRamReadWrite::complete",
            selected = "metal_address_segmented_v1",
            access_records = self.access_count,
            output_claims_valid = true,
        )
        .entered();
        Ok(RamReadWriteOutputClaims {
            val: *final_val,
            ra: *final_ra,
            inc: increment,
        })
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        let Some(MetalPhase::Done { merged_eq, .. }) = &self.phase else {
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
        Ok(())
    }
}

impl MetalBackend {
    fn prepare_high_activity_ram_read_write(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: &ProverInputs<'_, AkitaField, RamReadWriteChecking<AkitaField>>,
        log_t: usize,
        log_k: usize,
    ) -> Result<Option<RamReadWriteKernelBox>, KernelError<AkitaField>> {
        let cycles = 1usize << log_t;
        let config = self.config.ram_read_write;
        if cycles < config.trace_cutoff_elements {
            return Ok(None);
        }
        let address_count = 1usize << log_k;
        let columns = RamAccessColumns::shared(session, witness, log_t)?;
        columns.validate_addresses::<AkitaField>(address_count)?;
        let (access_count, qualified) = {
            let tape = session
                .state::<RamAccessTape>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "RAM access collection did not publish its certificate",
                })?;
            tape.validate(log_t, address_count)
                .map_err(|error| KernelError::Unsupported {
                    reason: match error {
                        crate::ram_access::RamAccessTapeError::WrongCycleDomain => {
                            "RAM high-activity route has the wrong cycle domain"
                        }
                        crate::ram_access::RamAccessTapeError::UnremappableAccess => {
                            "RAM high-activity route has an unremappable access"
                        }
                        _ => "RAM high-activity route has an invalid access certificate",
                    },
                })?;
            (
                tape.access_count(),
                tape.access_count() >= config.minimum_accesses
                    && tape.increment_compatible()
                    && tape.ram_ra_compatible()
                    && tape.hamming_exact(),
            )
        };
        if !qualified {
            return Ok(None);
        }
        let values = session
            .state::<RamAccessValues>()
            .ok_or(KernelError::InvariantViolation {
                reason: "RAM high-activity route lost the shared value columns",
            })?;
        let started = std::time::Instant::now();
        let sequence = match self.context.prepare_ram_read_write_sequence(
            &columns.addresses,
            &values.pre_values,
            &values.post_values,
            log_t,
            address_count,
        ) {
            Ok(sequence) => sequence,
            Err(error) if error.is_capacity_error() => {
                tracing::warn!(
                    target: "jolt::metal",
                    %error,
                    cycles,
                    access_count,
                    "high-activity RAM read-write route was not admitted"
                );
                return Ok(None);
            }
            Err(error) => return Err(metal_prepare_error(error)),
        };
        let mut initial_memory = dense_view::<AkitaField>(witness, ram_val_final())?
            .into_iter()
            .map(|value| {
                value
                    .to_canonical_u64_checked()
                    .ok_or(KernelError::InvariantViolation {
                        reason: "RAM final memory is not canonically representable as u64",
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if initial_memory.len() != address_count {
            return Err(KernelError::TableSizeMismatch {
                table: format!("{:?}", ram_val_final()),
                expected: address_count,
                got: initial_memory.len(),
            });
        }
        sequence
            .apply_initial_memory(&mut initial_memory)
            .map_err(metal_prepare_error)?;
        let val_init = Polynomial::new(
            initial_memory
                .into_iter()
                .map(AkitaField::from_u64)
                .collect(),
        );
        let stats = sequence.bucket_stats();
        let prepare_wall = started.elapsed();
        tracing::info!(
            target: "jolt::metal",
            selected = "metal_address_segmented_v1",
            cycles,
            log_t,
            log_k,
            access_count,
            active_addresses = stats.active_addresses,
            maximum_segment = stats.maximum_segment,
            p50_segment = stats.p50_segment,
            p95_segment = stats.p95_segment,
            p99_segment = stats.p99_segment,
            hot_addresses = stats.hot_addresses,
            hot_message_chunks = stats.hot_message_chunks,
            hot_state_entries = stats.hot_state_entries,
            hot_compaction_threads = stats.hot_compaction_threads,
            hot_compaction_threadgroup_bytes = stats.hot_compaction_threadgroup_bytes,
            hot_auxiliary_bytes = stats.hot_auxiliary_bytes,
            address_bytes = stats.address_bytes,
            cycle_bytes = stats.cycle_bytes,
            resident_bytes = sequence.resident_bytes(),
            prepare_wall_ns = duration_nanos(prepare_wall),
            "prepared high-activity RAM read-write sequence"
        );
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .test_counters
            .ram_read_write_metal_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        #[cfg(any(test, feature = "test-utils"))]
        if stats.hot_addresses != 0 {
            let _ = self
                .test_counters
                .ram_read_write_multigroup_hot_sequences
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(Some(Box::new(MetalRamReadWriteKernel {
            phase: Some(MetalPhase::Cycle {
                sequence: Box::new(sequence),
                gruen: GruenSplitEqPolynomial::new(
                    inputs.relation.product_tau_low(),
                    BindingOrder::LowToHigh,
                ),
            }),
            cycle_tail: None,
            val_init,
            gamma: inputs.challenges.gamma,
            log_t,
            log_k,
            access_count,
        })))
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
        if let Some(kernel) =
            self.prepare_high_activity_ram_read_write(session, witness, &inputs, log_t, log_k)?
        {
            return Ok(kernel);
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

fn metal_prepare_error(error: MetalError) -> KernelError<AkitaField> {
    metal_round_error(error).into()
}

fn metal_round_error(error: MetalError) -> SumcheckError<AkitaField> {
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

    #[test]
    fn address_segmented_sequence_matches_optimized_cpu() {
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
            let mut config = MetalConfig::default();
            config.ram_read_write.trace_cutoff_elements = 2;
            config.ram_read_write.minimum_accesses = 1;
            let metal = MetalBackend::new(config).unwrap();
            let mut session = ProofSession::default();
            let mut actual =
                PrepareKernel::prepare(&metal, &mut session, witness, inputs()).unwrap();
            assert_eq!(metal.ram_read_write_metal_sequences(), 1);
            assert_eq!(metal.ram_read_write_sparse_sequences(), 0);
            assert!(session.state::<RamAccessValues>().is_some());
            assert!(session.state::<RamAccessTape>().is_some());
            assert!(session.state::<Arc<RamCycleFamilyOwner>>().is_none());

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

    #[test]
    fn address_segmented_cycle_handoff_matches_optimized_cpu() {
        let shape = FixtureShape {
            log_t: 13,
            ram_k: 16,
        };
        let mut ops = vec![RamOp::Read { word: 3 }; (1 << shape.log_t) - 1];
        ops[0] = RamOp::Write { word: 3, post: 11 };
        with_ram_fixture_backend(shape, ops, |witness| {
            let tau_low = point(29, shape.log_t);
            let relation = RamReadWriteChecking::<AkitaField>::new(
                ReadWriteDimensions::new(shape.log_t, shape.log_k(), shape.log_t, shape.log_k()),
                shape.log_k(),
                tau_low.clone(),
            );
            let claims = RamReadWriteInputClaims::<AkitaField>::default();
            let points = RamReadWriteInputClaims::<Vec<AkitaField>>::default();
            let challenges = RamReadWriteChallenges {
                gamma: AkitaField::from_u64(31),
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
            config.ram_read_write.trace_cutoff_elements = 2;
            config.ram_read_write.minimum_accesses = 1;
            let metal = MetalBackend::new(config).unwrap();
            let mut actual =
                PrepareKernel::prepare(&metal, &mut ProofSession::default(), witness, inputs())
                    .unwrap();

            let input_claim = dense_input_claim(witness, &tau_low, challenges.gamma, shape.ram_k);
            let round_challenges = point(401, shape.log_t + shape.log_k());
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
        });
    }

    #[test]
    fn address_segmented_hot_chunk_boundary_matches_optimized_cpu() {
        let shape = FixtureShape {
            log_t: 15,
            ram_k: 1 << 19,
        };
        let mut ops = vec![RamOp::None; 1 << 14];
        for cycle in (0..ops.len()).step_by(2) {
            ops[cycle] = RamOp::Read { word: 3 };
        }
        ops[511] = RamOp::Read { word: 3 };
        ops[8189] = RamOp::Read { word: 3 };
        ops[12287] = RamOp::Read { word: 3 };
        let termination_cycle = ops.len();
        with_ram_fixture_backend(shape, ops, |witness| {
            let tau_low = point(43, shape.log_t);
            let relation = RamReadWriteChecking::<AkitaField>::new(
                ReadWriteDimensions::new(shape.log_t, shape.log_k(), shape.log_t, shape.log_k()),
                shape.log_k(),
                tau_low.clone(),
            );
            let claims = RamReadWriteInputClaims::<AkitaField>::default();
            let points = RamReadWriteInputClaims::<Vec<AkitaField>>::default();
            let challenges = RamReadWriteChallenges {
                gamma: AkitaField::from_u64(47),
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
            config.ram_read_write.trace_cutoff_elements = 2;
            config.ram_read_write.minimum_accesses = 1;
            let metal = MetalBackend::new(config).unwrap();
            let mut actual =
                PrepareKernel::prepare(&metal, &mut ProofSession::default(), witness, inputs())
                    .unwrap();
            assert_eq!(metal.ram_read_write_multigroup_hot_sequences(), 1);

            let input_claim = EqPolynomial::new(tau_low.clone()).evaluations()[termination_cycle]
                * challenges.gamma;
            let round_challenges = point(503, shape.log_t + shape.log_k());
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
        });
    }
}
