use std::sync::Arc;

use jolt_claims::protocols::jolt::{JoltDerivedId, RamRaClaimReductionPublic};
use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_field::{One as _, Zero as _};
use jolt_poly::{EqPolynomial, UnivariatePoly};
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
use super::solinas::RamRaClaimReductionSequence;
use crate::optimized::ram_trace::RamAccessColumns;
use crate::optimized::OptimizedBackend;
use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[cfg(feature = "test-utils")]
mod evaluation;
#[cfg(feature = "test-utils")]
pub use evaluation::{
    RamRaClaimReductionCpuMetalEvalFixture, RamRaClaimReductionEvalError,
    RamRaClaimReductionEvalResult, RamRaClaimReductionEvalSample,
    RamRaClaimReductionMetalEvalSample, RamRaClaimReductionRoundTiming,
    RamRaClaimReductionShapeSnapshot,
};

const MAX_SPARSE_PRODUCTS: u128 = 1_000_000;
const TERMS: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaClaimReductionMetalConfig {
    pub trace_cutoff_elements: usize,
    pub q_slices: usize,
}

impl Default for RamRaClaimReductionMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 25,
            q_slices: 32,
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

enum MetalDensePhase {
    Prefix {
        p: [Vec<AkitaField>; TERMS],
        q: [Vec<AkitaField>; TERMS],
        eq_hi: [Vec<AkitaField>; TERMS],
        r_cycle_lo: [Vec<AkitaField>; TERMS],
        challenges: Vec<AkitaField>,
    },
    Suffix {
        h: Vec<AkitaField>,
        eq_hi: [Vec<AkitaField>; TERMS],
        scales: [AkitaField; TERMS],
    },
}

struct MetalDenseRamRaClaimKernel {
    sequence: Option<RamRaClaimReductionSequence>,
    phase: MetalDensePhase,
    gamma_powers: [AkitaField; TERMS],
    rounds: usize,
    rounds_bound: usize,
    next_round: usize,
    #[cfg(any(test, feature = "test-utils"))]
    test_counters: Arc<super::backend::MetalTestCounters>,
}

impl MetalDenseRamRaClaimKernel {
    fn bind(&mut self, challenge: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        self.rounds_bound += 1;
        match &mut self.phase {
            MetalDensePhase::Prefix {
                p, q, challenges, ..
            } => {
                for table in p.iter_mut().chain(q.iter_mut()) {
                    bind_pairs(table, challenge);
                }
                challenges.push(challenge);
                if p[0].len() == 1 {
                    self.transition_to_suffix()?;
                }
            }
            MetalDensePhase::Suffix { h, eq_hi, .. } => {
                bind_pairs(h, challenge);
                for table in eq_hi {
                    bind_pairs(table, challenge);
                }
            }
        }
        Ok(())
    }

    fn transition_to_suffix(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let placeholder = MetalDensePhase::Suffix {
            h: Vec::new(),
            eq_hi: [Vec::new(), Vec::new(), Vec::new()],
            scales: [AkitaField::one(); TERMS],
        };
        let MetalDensePhase::Prefix {
            eq_hi,
            r_cycle_lo,
            challenges,
            ..
        } = core::mem::replace(&mut self.phase, placeholder)
        else {
            return Err(metal_error(
                "RAM RA claim-reduction transitioned outside the prefix phase",
            ));
        };
        let r_prefix = challenges.iter().rev().copied().collect::<Vec<_>>();
        let eq_prefix = eq_table(&r_prefix);
        let sequence = self.sequence.take().ok_or_else(|| {
            metal_error("RAM RA claim-reduction lost its resident source before H-prime")
        })?;
        let gather_started = std::time::Instant::now();
        let observation = sequence
            .gather_h(&eq_prefix)
            .map_err(|error| metal_error(error.to_string()))?;
        let h_wall = gather_started.elapsed();
        #[cfg(any(test, feature = "test-utils"))]
        {
            self.test_counters.ram_ra_claim_h_wall_ns.store(
                usize::try_from(h_wall.as_nanos()).unwrap_or(usize::MAX),
                std::sync::atomic::Ordering::Relaxed,
            );
            self.test_counters.ram_ra_claim_h_gpu_ns.store(
                usize::try_from(observation.gpu_active.as_nanos()).unwrap_or(usize::MAX),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        tracing::info!(
            target: "jolt::metal",
            wall_ns = u64::try_from(h_wall.as_nanos()).unwrap_or(u64::MAX),
            gpu_active_ns = u64::try_from(observation.gpu_active.as_nanos()).unwrap_or(u64::MAX),
            output_elements = observation.h_prime.len(),
            "completed RAM RA claim H-prime scan"
        );
        let scales = core::array::from_fn(|term| {
            EqPolynomial::<AkitaField>::mle(&r_cycle_lo[term], &r_prefix)
        });
        self.phase = MetalDensePhase::Suffix {
            h: observation.h_prime,
            eq_hi,
            scales,
        };
        Ok(())
    }

    fn message_evals(&self) -> [AkitaField; 2] {
        match &self.phase {
            MetalDensePhase::Prefix { p, q, .. } => {
                let mut evals = [AkitaField::zero(); 2];
                for term in 0..TERMS {
                    let mut sum = [AkitaField::zero(); 2];
                    for index in 0..p[term].len() / 2 {
                        let p_zero = p[term][2 * index];
                        let p_one = p[term][2 * index + 1];
                        let q_zero = q[term][2 * index];
                        let q_one = q[term][2 * index + 1];
                        sum[0] += p_zero * q_zero;
                        sum[1] += (p_one + p_one - p_zero) * (q_one + q_one - q_zero);
                    }
                    evals[0] += self.gamma_powers[term] * sum[0];
                    evals[1] += self.gamma_powers[term] * sum[1];
                }
                evals
            }
            MetalDensePhase::Suffix { h, eq_hi, scales } => {
                let coefficients: [AkitaField; TERMS] =
                    core::array::from_fn(|term| self.gamma_powers[term] * scales[term]);
                let mut evals = [AkitaField::zero(); 2];
                for index in 0..h.len() / 2 {
                    let h_zero = h[2 * index];
                    let h_one = h[2 * index + 1];
                    let mut eq_zero = AkitaField::zero();
                    let mut eq_two = AkitaField::zero();
                    for term in 0..TERMS {
                        let e_zero = eq_hi[term][2 * index];
                        let e_one = eq_hi[term][2 * index + 1];
                        eq_zero += coefficients[term] * e_zero;
                        eq_two += coefficients[term] * (e_one + e_one - e_zero);
                    }
                    evals[0] += h_zero * eq_zero;
                    evals[1] += (h_one + h_one - h_zero) * eq_two;
                }
                evals
            }
        }
    }

    fn require_fully_bound(&self) -> Result<(), SumcheckKernelError<AkitaField>> {
        let remaining = self.rounds - self.rounds_bound;
        if remaining == 0 {
            Ok(())
        } else {
            Err(SumcheckKernelError::NotFullyBound { remaining })
        }
    }
}

impl ProveRounds<AkitaField> for MetalDenseRamRaClaimKernel {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if round != self.next_round || (round == 0) != bind.is_none() {
            return Err(metal_error(
                "RAM RA claim-reduction received an out-of-order round",
            ));
        }
        if let Some(challenge) = bind {
            self.bind(challenge)?;
        }
        self.next_round += 1;
        Ok(UnivariatePoly::from_evals_and_hint(
            previous_claim,
            &self.message_evals(),
        ))
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.next_round != self.rounds {
            return Err(metal_error(
                "RAM RA claim-reduction finished before its final message",
            ));
        }
        self.bind(bind)
    }
}

impl SumcheckKernel<AkitaField> for MetalDenseRamRaClaimKernel {
    type Relation = RamRaClaimReduction<AkitaField>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<RamRaClaimReductionOutputClaims<AkitaField>, SumcheckKernelError<AkitaField>> {
        self.require_fully_bound()?;
        let MetalDensePhase::Suffix { h, .. } = &self.phase else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "RAM RA claim-reduction fully bound in the prefix phase",
            });
        };
        let ram_ra = h
            .first()
            .copied()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "RAM RA claim-reduction has no terminal H value",
            })?;
        Ok(RamRaClaimReductionOutputClaims { ram_ra })
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        self.require_fully_bound()?;
        let MetalDensePhase::Suffix { eq_hi, scales, .. } = &self.phase else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "RAM RA claim-reduction fully bound in the prefix phase",
            });
        };
        let ids = [
            RamRaClaimReductionPublic::EqCycleRaf,
            RamRaClaimReductionPublic::EqCycleReadWrite,
            RamRaClaimReductionPublic::EqCycleValCheck,
        ];
        for (term, public_id) in ids.into_iter().enumerate() {
            let id = JoltDerivedId::from(public_id);
            let expected =
                relation.derive_output_term(&id, input_points, output_points, challenges)?;
            let got = scales[term] * eq_hi[term][0];
            if got != expected {
                return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
            }
        }
        Ok(())
    }
}

fn bind_pairs(table: &mut Vec<AkitaField>, challenge: AkitaField) {
    let half = table.len() / 2;
    for index in 0..half {
        let even = table[2 * index];
        table[index] = even + challenge * (table[2 * index + 1] - even);
    }
    table.truncate(half);
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
        let columns = RamAccessColumns::shared(session, witness, log_t)?;
        let address_count = 1usize
            .checked_shl(u32::try_from(log_k).map_err(|_| KernelError::Unsupported {
                reason: "RAM RA claim-reduction address domain is too large",
            })?)
            .ok_or(KernelError::Unsupported {
                reason: "RAM RA claim-reduction address domain is too large",
            })?;
        columns.validate_addresses::<AkitaField>(address_count)?;
        if let Some(owner) = shared_ram_cycle_family_owner(session, witness, log_t, log_k)? {
            let predicted = estimated_ram_ra_claim_products(&owner)
                .map_err(|error| prepare_error(error.to_string()))?;
            if predicted <= MAX_SPARSE_PRODUCTS {
                let route = tracing::info_span!(
                    "MetalRamRaClaimReduction::route",
                    cycles,
                    log_t,
                    log_k,
                    requested = "hybrid_v1",
                    selected = "host_sparse_v1",
                    fallback_reason = "sparse_product_budget",
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
                    .test_counters
                    .ram_ra_claim_sparse_sequences
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                tracing::info!(
                    target: "jolt::metal",
                    predicted_products = predicted,
                    "prepared sparse RAM RA claim-reduction sequence"
                );
                return Ok(Box::new(HostSparseRamRaClaimKernel {
                    sequence,
                    next_round: 0,
                }));
            }
        }

        let prefix_bits = log_t / 2;
        let suffix_bits = log_t - prefix_bits;
        let eq_address = eq_table(r_address);
        let eq_hi = cycle_points.map(|point| eq_table(&point[..suffix_bits]));
        let p = cycle_points.map(|point| eq_table(&point[suffix_bits..]));
        let r_cycle_lo = cycle_points.map(|point| point[suffix_bits..].to_vec());
        let route = tracing::info_span!(
            "MetalRamRaClaimReduction::route",
            cycles,
            log_t,
            log_k,
            requested = "hybrid_v1",
            selected = "no_copy_q_hybrid_v1",
            fallback_reason = "none",
            source_bytes = cycles * std::mem::size_of::<u32>(),
            source_copy_bytes = 0,
            additional_source_row_scans = 0,
            active_cycle_bound = columns.active_cycle_bound(),
            member_upload_bytes = 0,
            address_alias_reused = tracing::field::Empty,
            compact_source = tracing::field::Empty,
            complete_sequence = true,
        );
        let _route_guard = route.enter();
        let sequence = self
            .context
            .prepare_ram_ra_claim_reduction(
                Arc::clone(&columns),
                address_count,
                prefix_bits,
                &eq_address,
                &eq_hi,
                self.config.ram_ra_claim_reduction.q_slices,
            )
            .map_err(prepare_metal_error)?;
        let _ = route.record("address_alias_reused", sequence.address_alias_reused());
        let _ = route.record("compact_source", sequence.compact_source());
        tracing::debug!(
            target: "jolt::metal",
            active_high_elements = sequence.active_high_elements(),
            active_q_slices = sequence.active_q_slices(),
            suffix_elements = 1usize << suffix_bits,
            "certified RAM RA claim scan bound"
        );
        let source_copy_bytes = RamRaClaimReductionSequence::source_copy_bytes();
        let readback_bytes = sequence.readback_bytes();
        #[cfg(any(test, feature = "test-utils"))]
        self.test_counters.ram_ra_claim_address_alias_reuses.store(
            usize::from(sequence.address_alias_reused()),
            std::sync::atomic::Ordering::Relaxed,
        );
        let q_started = std::time::Instant::now();
        let observation = sequence.build_q().map_err(prepare_metal_error)?;
        let q_wall = q_started.elapsed();
        #[cfg(any(test, feature = "test-utils"))]
        {
            self.test_counters.ram_ra_claim_q_wall_ns.store(
                usize::try_from(q_wall.as_nanos()).unwrap_or(usize::MAX),
                std::sync::atomic::Ordering::Relaxed,
            );
            self.test_counters.ram_ra_claim_q_gpu_ns.store(
                usize::try_from(observation.gpu_active.as_nanos()).unwrap_or(usize::MAX),
                std::sync::atomic::Ordering::Relaxed,
            );
            self.test_counters.ram_ra_claim_q_wait_wall_ns.store(
                usize::try_from(observation.wait_wall.as_nanos()).unwrap_or(usize::MAX),
                std::sync::atomic::Ordering::Relaxed,
            );
            self.test_counters.ram_ra_claim_q_readback_wall_ns.store(
                usize::try_from(observation.readback_wall.as_nanos()).unwrap_or(usize::MAX),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        tracing::info!(
            target: "jolt::metal",
            wall_ns = u64::try_from(q_wall.as_nanos()).unwrap_or(u64::MAX),
            gpu_active_ns = u64::try_from(observation.gpu_active.as_nanos()).unwrap_or(u64::MAX),
            source_copy_bytes,
            readback_bytes,
            output_elements = observation.q.iter().map(Vec::len).sum::<usize>(),
            "completed RAM RA claim Q scan"
        );
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .test_counters
            .ram_ra_claim_metal_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let gamma = inputs.challenges.gamma;
        Ok(Box::new(MetalDenseRamRaClaimKernel {
            sequence: Some(sequence),
            phase: MetalDensePhase::Prefix {
                p,
                q: observation.q,
                eq_hi,
                r_cycle_lo,
                challenges: Vec::with_capacity(prefix_bits),
            },
            gamma_powers: [AkitaField::one(), gamma, gamma * gamma],
            rounds: log_t,
            rounds_bound: 0,
            next_round: 0,
            #[cfg(any(test, feature = "test-utils"))]
            test_counters: Arc::clone(&self.test_counters),
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

fn prepare_metal_error(error: impl ToString) -> KernelError<AkitaField> {
    KernelError::Sumcheck(metal_error(error.to_string()))
}

fn metal_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
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
    use jolt_field::Ring as _;
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
    use crate::optimized::parity::run_lockstep;
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
