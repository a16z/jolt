use std::{slice, sync::Arc, thread::JoinHandle};

use jolt_claims::protocols::jolt::geometry::ram::ram_val_final;
use jolt_claims::protocols::jolt::{JoltDerivedId, RamReadWritePublic};
use jolt_field::{CanonicalEncoding, Prime128OffsetA7F7 as AkitaField, Ring};
use jolt_field::{One as _, Zero as _};
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
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::backend::MetalBackend;
use super::ram_cycle_family::shared_ram_cycle_family_owner;
use crate::metal::ram_records::{
    RamAccessColumns, RamAccessValues, RamIncrementActivity, RamReadWriteRecordChunks, NO_ACCESS,
};
use crate::optimized::rw_matrix::{
    AddressMajorEntry, AddressMajorMatrix, CycleMajorEntry, CycleMajorMatrix,
};
use crate::optimized::OptimizedBackend;
use crate::ram_access::RamAccessTape;
use crate::reference::views::dense_view;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

use super::solinas::{
    InstructionInputRow, InstructionReadRafStage1Owner, MetalError, ProductRemainderRows,
    ProductRemainderSourceKind, RamRafSegmentedAddressPlane, RamReadWriteDispatchTiming,
    RamReadWriteFinish, RamReadWriteSequence, SparseCycleProduct, SpartanOuterUniskipSuccessorRow,
    RAM_READ_WRITE_CYCLE_TILE_LOG2,
};

#[cfg(feature = "test-utils")]
mod evaluation;
#[cfg(feature = "test-utils")]
pub use evaluation::{
    RamReadWriteBucketSnapshot, RamReadWriteCpuEvalSample, RamReadWriteCpuMetalEvalFixture,
    RamReadWriteDispatchSnapshot, RamReadWriteEvalError, RamReadWriteEvalResult,
    RamReadWriteMetalEvalSample, RamReadWritePreparationSnapshot, RamReadWriteRoundTiming,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamReadWriteMetalConfig {
    pub trace_cutoff_elements: usize,
    pub minimum_accesses: usize,
    pub gpu_record_scatter_cutoff_elements: usize,
}

impl Default for RamReadWriteMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 25,
            minimum_accesses: crate::ram_access::MAX_RETAINED_RAM_ACCESSES + 1,
            gpu_record_scatter_cutoff_elements: 1 << 29,
        }
    }
}

pub(super) const RAM_READ_WRITE_STAGE1_SOURCE_CUTOFF_ELEMENTS: usize = 1 << 28;

pub(super) struct RamReadWriteStage1Source {
    instruction: InstructionReadRafStage1Owner,
    product: ProductRemainderRows,
}

struct RamReadWriteDirectWorker {
    access_count: usize,
    active_cycle_bound: usize,
    required_address_domain: usize,
}

struct RamReadWriteDirectProjection {
    columns: Arc<RamAccessColumns>,
    tape: RamAccessTape,
}

impl RamReadWriteDirectProjection {
    fn publish(self, session: &mut ProofSession) -> Result<(), KernelError<AkitaField>> {
        if session.state::<Arc<RamAccessColumns>>().is_some()
            || session.state::<RamAccessValues>().is_some()
            || session.state::<Arc<RamIncrementActivity>>().is_some()
            || session.state::<RamAccessTape>().is_some()
        {
            return Err(KernelError::InvariantViolation {
                reason: "direct Stage-1 RAM projection would replace resident state",
            });
        }
        session.park(self.columns);
        session.park(self.tape);
        Ok(())
    }
}

struct RamReadWriteStage1Values<'a> {
    compact: &'a [InstructionInputRow],
    residual: &'a [SpartanOuterUniskipSuccessorRow],
}

impl RamReadWriteStage1Values<'_> {
    fn value_at(&self, row: usize) -> Result<(u64, u64), MetalError> {
        let (load, store, rs2) = self.compact[row].stage1_ram_source();
        if load == store {
            return Err(MetalError::InvalidRamReadWriteState(
                "Stage-1 RAM metadata does not identify one access kind",
            ));
        }
        let pre_value = self.residual[row].stage1_ram_pre_value();
        Ok((pre_value, if load { pre_value } else { rs2 }))
    }
}

impl RamReadWriteStage1Source {
    pub(super) fn new(
        instruction: InstructionReadRafStage1Owner,
        product: ProductRemainderRows,
    ) -> Result<Self, KernelError<AkitaField>> {
        let receipt = instruction.receipt();
        if product.source_kind() != ProductRemainderSourceKind::SpartanStage1
            || receipt.rows() != product.len()
            || receipt.device_registry_id() != product.device_registry_id()
        {
            return Err(KernelError::InvariantViolation {
                reason: "Stage-1 RAM source owners disagree on provenance or geometry",
            });
        }
        Ok(Self {
            instruction,
            product,
        })
    }

    fn rows(&self) -> usize {
        self.product.len()
    }

    fn source_bytes(&self) -> u64 {
        let product_bytes = self
            .product
            .stage1_buffers()
            .map_or(0, |(compact, residual)| {
                compact.length() + residual.length()
            });
        self.instruction.receipt().row_bytes() + product_bytes
    }

    fn ram_remap_compatible(&self) -> bool {
        self.instruction.ram_remap_compatible()
    }

    fn values(&self) -> Result<RamReadWriteStage1Values<'_>, KernelError<AkitaField>> {
        let rows = self.rows();
        let (compact, residual) =
            self.product
                .stage1_buffers()
                .ok_or(KernelError::InvariantViolation {
                    reason: "Stage-1 RAM source lost the Spartan row buffers",
                })?;
        // SAFETY: ProductRemainderRows validates both immutable Stage-1
        // allocations against these element ABIs and keeps them alive.
        let compact = unsafe {
            slice::from_raw_parts(compact.contents().cast::<InstructionInputRow>(), rows)
        };
        // SAFETY: ProductRemainderRows validates this allocation against the
        // successor-row ABI and keeps it alive for the returned borrow.
        let residual = unsafe {
            slice::from_raw_parts(
                residual
                    .contents()
                    .cast::<SpartanOuterUniskipSuccessorRow>(),
                rows,
            )
        };
        Ok(RamReadWriteStage1Values { compact, residual })
    }

    fn collect_direct_addresses(
        &self,
    ) -> Result<RamReadWriteDirectProjection, KernelError<AkitaField>> {
        let rows = self.rows();
        if !rows.is_power_of_two() {
            return Err(KernelError::InvariantViolation {
                reason: "Stage-1 RAM source does not cover a power-of-two cycle domain",
            });
        }
        let packed_metadata = self.instruction.packed_metadata();
        if packed_metadata.len() != rows {
            return Err(KernelError::InvariantViolation {
                reason: "Stage-1 RAM metadata has the wrong cycle domain",
            });
        }

        #[cfg(feature = "parallel")]
        let worker_count = rayon::current_num_threads().min(rows);
        #[cfg(not(feature = "parallel"))]
        let worker_count = 1;
        let chunk_rows = rows.div_ceil(worker_count);
        let mut addresses = vec![NO_ACCESS; rows];
        let collect_worker = |worker: usize,
                              addresses: &mut [u32]|
         -> Result<RamReadWriteDirectWorker, KernelError<AkitaField>> {
            let base = worker * chunk_rows;
            let mut access_count = 0usize;
            let mut active_cycle_bound = 0usize;
            let mut required_address_domain = 0usize;
            for (offset, address) in addresses.iter_mut().enumerate() {
                let row = base + offset;
                let address_plus_one = packed_metadata[row] & u64::from(u32::MAX);
                let Some(remapped_address) = address_plus_one.checked_sub(1) else {
                    continue;
                };
                let remapped_address = u32::try_from(remapped_address).map_err(|_| {
                    KernelError::InvariantViolation {
                        reason: "Stage-1 RAM address exceeds its packed u32 ABI",
                    }
                })?;
                *address = remapped_address;
                access_count += 1;
                active_cycle_bound = row + 1;
                required_address_domain =
                    required_address_domain.max(remapped_address as usize + 1);
            }
            Ok(RamReadWriteDirectWorker {
                access_count,
                active_cycle_bound,
                required_address_domain,
            })
        };
        #[cfg(feature = "parallel")]
        let workers = addresses
            .par_chunks_mut(chunk_rows)
            .enumerate()
            .map(|(worker, addresses)| collect_worker(worker, addresses))
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(not(feature = "parallel"))]
        let workers = vec![collect_worker(0, &mut addresses)?];

        let access_count = workers.iter().map(|worker| worker.access_count).sum();
        let active_cycle_bound = workers
            .iter()
            .map(|worker| worker.active_cycle_bound)
            .max()
            .unwrap_or(0);
        let required_address_domain = workers
            .iter()
            .map(|worker| worker.required_address_domain)
            .max()
            .unwrap_or(0);
        Ok(RamReadWriteDirectProjection {
            columns: Arc::new(RamAccessColumns::from_direct_addresses(
                addresses,
                active_cycle_bound,
                required_address_domain,
            )),
            tape: RamAccessTape::new(rows.ilog2() as usize, access_count, None, true, true, true),
        })
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RamReadWriteStage1Source {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.enter_self_sized::<Self>().exit();
    }
}

type RamReadWriteKernelBox =
    Box<dyn SumcheckKernel<AkitaField, Relation = RamReadWriteChecking<AkitaField>>>;
struct PrefetchedRamReadWriteSequence {
    source: RamReadWriteRecordChunks,
    prepared: Result<RamReadWriteSequence, String>,
}

pub(super) struct PendingRamReadWriteSequence {
    rows: usize,
    address_count: usize,
    tile_log: usize,
    started: std::time::Instant,
    handle: Option<JoinHandle<PrefetchedRamReadWriteSequence>>,
    ready: Option<PrefetchedRamReadWriteSequence>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for PendingRamReadWriteSequence {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.enter_self_sized::<Self>().exit();
    }
}

impl Drop for PendingRamReadWriteSequence {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl PendingRamReadWriteSequence {
    fn ensure_ready(&mut self) -> Result<&PrefetchedRamReadWriteSequence, KernelError<AkitaField>> {
        if self.ready.is_none() {
            let handle = self.handle.take().ok_or(KernelError::InvariantViolation {
                reason: "RAM read-write sequence prefetch was already consumed",
            })?;
            self.ready = Some(handle.join().map_err(|_| KernelError::InvariantViolation {
                reason: "RAM read-write sequence prefetch worker panicked",
            })?);
        }
        self.ready.as_ref().ok_or(KernelError::InvariantViolation {
            reason: "RAM read-write sequence prefetch lost its completed result",
        })
    }

    fn join(mut self) -> Result<PrefetchedRamReadWriteSequence, KernelError<AkitaField>> {
        let _ = self.ensure_ready()?;
        self.ready.take().ok_or(KernelError::InvariantViolation {
            reason: "RAM read-write sequence prefetch lost its completed result",
        })
    }
}

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
    cycle_sequence_wall: std::time::Duration,
    cycle_sequence_gpu_active: std::time::Duration,
    cycle_dispatch_timing: RamReadWriteDispatchTiming,
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
        self.cycle_sequence_wall += observation.wall;
        self.cycle_sequence_gpu_active += observation.gpu_active;
        if let Some(dispatch_timing) = observation.dispatch_timing {
            self.cycle_dispatch_timing += dispatch_timing;
        }
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
            wall,
            gpu_active,
            dispatch_timing,
        } = sequence.finish(challenge).map_err(metal_round_error)?;
        self.cycle_sequence_wall += wall;
        self.cycle_sequence_gpu_active += gpu_active;
        if let Some(dispatch_timing) = dispatch_timing {
            self.cycle_dispatch_timing += dispatch_timing;
        }
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
                col: root.address as u32,
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
    pub(super) fn start_prefetched_ram_raf_cpu(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        cycle_point: &[AkitaField],
    ) -> Result<(), KernelError<AkitaField>> {
        let cycles = 1usize
            .checked_shl(log_t as u32)
            .ok_or(KernelError::InvariantViolation {
                reason: "segmented CPU RAM RAF cycle domain overflows usize",
            })?;
        if cycles < self.config.ram_raf_evaluation.cpu_prefetch_cutoff_elements
            || cycles < self.config.ram_raf_evaluation.dispatch.trace_cutoff
        {
            return Ok(());
        }
        if cycle_point.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "segmented CPU RAM RAF received the wrong cycle point",
            });
        }
        let Some(mut pending) = session.take::<PendingRamReadWriteSequence>() else {
            return Ok(());
        };
        let geometry_matches = pending.rows == cycles
            && pending.address_count.is_power_of_two()
            && pending.tile_log == log_t.min(RAM_READ_WRITE_CYCLE_TILE_LOG2);
        let prefetch_started = pending.started;
        let join_started = std::time::Instant::now();
        let source = {
            let prefetched = pending.ensure_ready()?;
            match &prefetched.prepared {
                Ok(sequence) if geometry_matches => {
                    Some(sequence.ram_raf_segmented_address_plane())
                }
                _ => None,
            }
        };
        let join_wall = join_started.elapsed();
        let total_wall = prefetch_started.elapsed();
        tracing::info!(
            target: "jolt::metal",
            cycles,
            geometry_matches,
            join_wall_ns = duration_nanos(join_wall),
            total_wall_ns = duration_nanos(total_wall),
            "joined RAM sequence for segmented CPU RAF prefetch"
        );
        session.park(pending);
        if let Some(source) = source {
            Self::start_ram_raf_cpu_prefetch(session, source, cycle_point)?;
        }
        Ok(())
    }

    pub(super) fn start_ram_read_write_sequence_prefetch(
        &self,
        session: &mut ProofSession,
        log_t: usize,
    ) -> Result<(), KernelError<AkitaField>> {
        let cycles = 1usize
            .checked_shl(log_t as u32)
            .ok_or(KernelError::InvariantViolation {
                reason: "RAM read-write prefetch cycle domain overflows usize",
            })?;
        if cycles < self.config.ram_read_write.trace_cutoff_elements
            || session.state::<RamReadWriteRecordChunks>().is_none()
        {
            return Ok(());
        }
        if session.state::<PendingRamReadWriteSequence>().is_some() {
            return Err(KernelError::InvariantViolation {
                reason: "RAM read-write sequence prefetch was started twice",
            });
        }
        let access_count = {
            let tape = session
                .state::<RamAccessTape>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "RAM read-write prefetch lost its access certificate",
                })?;
            if tape.access_count() < self.config.ram_read_write.minimum_accesses
                || !tape.increment_compatible()
                || !tape.ram_ra_compatible()
                || !tape.hamming_exact()
            {
                return Ok(());
            }
            tape.access_count()
        };
        let source =
            session
                .take::<RamReadWriteRecordChunks>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "RAM read-write prefetch source disappeared",
                })?;
        if source.rows() != cycles || source.access_count() != access_count {
            return Err(KernelError::InvariantViolation {
                reason: "RAM read-write prefetch source has inconsistent geometry",
            });
        }
        let address_count = source.address_count();
        let tile_log = source.tile_log();
        let gpu_record_scatter = cycles
            >= self
                .config
                .ram_read_write
                .gpu_record_scatter_cutoff_elements;
        let context = Arc::clone(&self.context);
        let started = std::time::Instant::now();
        let handle = std::thread::Builder::new()
            .name("jolt-ram-read-write-prefetch".to_owned())
            .spawn(move || {
                let prepared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    context.prepare_ram_read_write_record_sequence(
                        &source,
                        log_t,
                        address_count,
                        gpu_record_scatter,
                    )
                }))
                .map_err(|_| "RAM read-write sequence prefetch panicked".to_owned())
                .and_then(|prepared| prepared.map_err(|error| error.to_string()));
                PrefetchedRamReadWriteSequence { source, prepared }
            })
            .map_err(|_| KernelError::InvariantViolation {
                reason: "RAM read-write sequence prefetch worker could not start",
            })?;
        tracing::info!(
            target: "jolt::metal",
            cycles,
            address_count,
            tile_log,
            accesses = access_count,
            gpu_record_scatter,
            "started RAM read-write sequence prefetch"
        );
        session.park(PendingRamReadWriteSequence {
            rows: cycles,
            address_count,
            tile_log,
            started,
            handle: Some(handle),
            ready: None,
        });
        Ok(())
    }

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
        let mut prefetched_sequence = None;
        let mut prefetch_attempted = false;
        let mut prefetch_used = false;
        let mut prefetch_join_wall = std::time::Duration::ZERO;
        let mut prefetch_total_wall = std::time::Duration::ZERO;
        if let Some(pending) = session.take::<PendingRamReadWriteSequence>() {
            prefetch_attempted = true;
            let pending_rows = pending.rows;
            let pending_address_count = pending.address_count;
            let pending_tile_log = pending.tile_log;
            let pending_started = pending.started;
            let join_started = std::time::Instant::now();
            let prefetched = {
                let _span = tracing::info_span!(
                    "MetalRamReadWrite::sequence_prefetch_join",
                    rows = pending_rows,
                    address_count = pending_address_count,
                )
                .entered();
                pending.join()?
            };
            prefetch_join_wall = join_started.elapsed();
            prefetch_total_wall = pending_started.elapsed();
            session.park(prefetched.source);
            let geometry_matches = pending_rows == cycles
                && pending_address_count == address_count
                && pending_tile_log == log_t.min(RAM_READ_WRITE_CYCLE_TILE_LOG2);
            match prefetched.prepared {
                Ok(prepared) if geometry_matches => {
                    prefetched_sequence = Some(prepared);
                }
                Ok(_) => tracing::warn!(
                    target: "jolt::metal",
                    pending_rows,
                    cycles,
                    pending_address_count,
                    address_count,
                    pending_tile_log,
                    expected_tile_log = log_t.min(RAM_READ_WRITE_CYCLE_TILE_LOG2),
                    "RAM read-write sequence prefetch geometry changed; rebuilding synchronously"
                ),
                Err(error) => tracing::warn!(
                    target: "jolt::metal",
                    %error,
                    "RAM read-write sequence prefetch failed; rebuilding synchronously"
                ),
            }
        }
        let source_collection_performed = session
            .state::<std::sync::Arc<RamAccessColumns>>()
            .is_none();
        let source_collection_started = std::time::Instant::now();
        let mut source_kind = if source_collection_performed {
            "witness_random_access"
        } else {
            "shared_columns"
        };
        let mut source_alias_bytes = 0u64;
        let mut source_compaction_wall_ns = 0u64;
        let mut source_worker_arenas = 0usize;
        let mut source_census_address_count = 0usize;
        let mut source_remap_compatible = false;
        let mut direct_stage1_source = false;
        let coproduced_records = session.state::<RamReadWriteRecordChunks>().is_some();
        let mut witness_row_extractions = usize::from(source_collection_performed) * cycles;
        if coproduced_records {
            let records = session.state::<RamReadWriteRecordChunks>().ok_or(
                KernelError::InvariantViolation {
                    reason: "RAM co-produced record source disappeared",
                },
            )?;
            if records.rows() != cycles {
                return Err(KernelError::InvariantViolation {
                    reason: "RAM co-produced records have the wrong cycle domain",
                });
            }
            source_kind = "stage1_coproduced_records_v1";
            source_alias_bytes = records.record_bytes() as u64;
            source_compaction_wall_ns = records.compaction_wall_ns();
            source_worker_arenas = records.chunks().len();
            source_census_address_count = records.address_count();
            source_remap_compatible = true;
            witness_row_extractions = 0;
        } else if source_collection_performed
            && (cycles >= RAM_READ_WRITE_STAGE1_SOURCE_CUTOFF_ELEMENTS || cfg!(test))
        {
            if let Some(source) = session.state::<RamReadWriteStage1Source>() {
                source_alias_bytes = source.source_bytes();
                source_remap_compatible = source.ram_remap_compatible();
                if source_remap_compatible {
                    let projection = source.collect_direct_addresses()?;
                    projection.publish(session)?;
                    direct_stage1_source = true;
                    source_kind = "stage1_direct_scatter_v1";
                    witness_row_extractions = 0;
                }
            }
        }
        if coproduced_records || !direct_stage1_source {
            drop(session.take::<RamReadWriteStage1Source>());
        }
        let columns = RamAccessColumns::shared(session, witness, log_t)?;
        let source_collection_wall = source_collection_started.elapsed();
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
        if let Some(records) = session.state::<RamReadWriteRecordChunks>() {
            if records.access_count() != access_count {
                return Err(KernelError::InvariantViolation {
                    reason: "RAM co-produced records disagree with the access certificate",
                });
            }
        }
        if !qualified {
            if direct_stage1_source {
                let _ = session.take::<Arc<RamAccessColumns>>();
                let _ = session.take::<RamAccessTape>();
                drop(session.take::<RamReadWriteStage1Source>());
            }
            drop(session.take::<RamReadWriteRecordChunks>());
            return Ok(None);
        }
        let started = std::time::Instant::now();
        let mut direct_activity = None;
        let gpu_record_scatter = cycles >= config.gpu_record_scatter_cutoff_elements;
        let sequence_result = if coproduced_records {
            if let Some(sequence) = prefetched_sequence.take() {
                prefetch_used = true;
                Ok(sequence)
            } else {
                let records = session.state::<RamReadWriteRecordChunks>().ok_or(
                    KernelError::InvariantViolation {
                        reason: "RAM co-produced record source disappeared before sequence setup",
                    },
                )?;
                self.context.prepare_ram_read_write_record_sequence(
                    records,
                    log_t,
                    address_count,
                    gpu_record_scatter,
                )
            }
        } else if direct_stage1_source {
            let source = session.state::<RamReadWriteStage1Source>().ok_or(
                KernelError::InvariantViolation {
                    reason: "RAM direct scatter lost its Stage-1 source",
                },
            )?;
            let values = source.values()?;
            let value_at = |cycle| values.value_at(cycle);
            self.context
                .prepare_ram_read_write_direct_sequence(
                    &columns.addresses,
                    log_t,
                    address_count,
                    &value_at,
                )
                .map(|(sequence, activity)| {
                    direct_activity = Some(activity);
                    sequence
                })
        } else {
            let values =
                session
                    .state::<RamAccessValues>()
                    .ok_or(KernelError::InvariantViolation {
                        reason: "RAM high-activity route lost the shared value columns",
                    })?;
            self.context.prepare_ram_read_write_sequence(
                &columns.addresses,
                &values.pre_values,
                &values.post_values,
                log_t,
                address_count,
            )
        };
        let sequence = match sequence_result {
            Ok(sequence) => sequence,
            Err(error) if error.is_capacity_error() => {
                tracing::warn!(
                    target: "jolt::metal",
                    %error,
                    cycles,
                    access_count,
                    "high-activity RAM read-write route was not admitted"
                );
                if direct_stage1_source {
                    let _ = session.take::<Arc<RamAccessColumns>>();
                    let _ = session.take::<RamAccessTape>();
                    drop(session.take::<RamReadWriteStage1Source>());
                }
                drop(session.take::<RamReadWriteRecordChunks>());
                return Ok(None);
            }
            Err(error) => return Err(metal_prepare_error(error)),
        };
        let mut record_source = if coproduced_records {
            Some(session.take::<RamReadWriteRecordChunks>().ok_or(
                KernelError::InvariantViolation {
                    reason: "RAM record source disappeared after sequence construction",
                },
            )?)
        } else {
            None
        };
        if let Some(records) = &mut record_source {
            direct_activity = Some(records.take_increment_chunks());
        }
        if let Some(activity) = direct_activity {
            let increment_count = activity
                .iter()
                .map(|(_, increments)| increments.len())
                .sum();
            let mut increment_cycles = Vec::with_capacity(increment_count);
            let mut increments = Vec::with_capacity(increment_count);
            for (mut worker_cycles, mut worker_increments) in activity {
                increment_cycles.append(&mut worker_cycles);
                increments.append(&mut worker_increments);
            }
            session.park(Arc::new(RamIncrementActivity::from_sorted_parts(
                increment_cycles,
                increments,
            )));
        }
        if direct_stage1_source {
            drop(session.take::<RamReadWriteStage1Source>());
        }
        let mut initial_memory = dense_view::<AkitaField>(witness, ram_val_final())?
            .into_iter()
            .map(|value| {
                value
                    .to_u64_checked()
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
        if let Some(records) = &record_source {
            records
                .apply_initial_memory(&mut initial_memory)
                .map_err(|error| error.into_kernel_error())?;
        } else {
            sequence
                .apply_initial_memory(&mut initial_memory)
                .map_err(metal_prepare_error)?;
        }
        drop(record_source);
        let val_init = Polynomial::new(
            initial_memory
                .into_iter()
                .map(AkitaField::from_u64)
                .collect(),
        );
        let stats = sequence.bucket_stats();
        let preparation = sequence.preparation_timing();
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
            source_collection_performed,
            source_kind,
            source_alias_bytes,
            source_compaction_wall_ns,
            source_worker_arenas,
            source_census_address_count,
            source_remap_compatible,
            gpu_record_scatter,
            prefetch_attempted,
            prefetch_used,
            prefetch_join_wall_ns = duration_nanos(prefetch_join_wall),
            prefetch_total_wall_ns = duration_nanos(prefetch_total_wall),
            intermediate_record_bytes = usize::from(coproduced_records) * source_alias_bytes as usize,
            dense_value_bytes = usize::from(
                source_kind != "stage1_direct_scatter_v1"
                    && source_kind != "stage1_coproduced_records_v1"
            )
                * 2
                * cycles
                * std::mem::size_of::<u64>(),
            witness_row_extractions,
            source_collection_wall_ns = duration_nanos(source_collection_wall),
            bucket_plan_wall_ns = duration_nanos(preparation.bucket_plan),
            allocation_wall_ns = duration_nanos(preparation.allocation),
            initialization_and_scatter_wall_ns =
                duration_nanos(preparation.initialization_and_scatter),
            pipeline_setup_wall_ns = duration_nanos(preparation.pipeline_setup),
            gpu_scatter_wall_ns = duration_nanos(preparation.gpu_scatter_wall),
            gpu_scatter_active_ns = duration_nanos(preparation.gpu_scatter_active),
            sequence_prepare_wall_ns = duration_nanos(preparation.total),
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
        if session.state::<RamRafSegmentedAddressPlane>().is_some() {
            return Err(KernelError::InvariantViolation {
                reason: "RAM read-write tried to replace a resident RAM-RAF source",
            });
        }
        session.park(sequence.ram_raf_segmented_address_plane());
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
            cycle_sequence_wall: std::time::Duration::ZERO,
            cycle_sequence_gpu_active: std::time::Duration::ZERO,
            cycle_dispatch_timing: RamReadWriteDispatchTiming::default(),
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
            self.submit_ram_raf(session, relation)?;
            return Ok(kernel);
        }
        self.submit_ram_raf(session, relation)?;
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
                    row: record.cycle(),
                    col: address as u32,
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
    use crate::metal::solinas::SolinasMetal;
    use crate::metal::MetalConfig;
    use crate::optimized::parity::run_lockstep;
    use crate::optimized::spartan_outer::prepare_metal_spartan_outer_stage1_owner_witness_rows;
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
    fn stage1_resident_ram_source_matches_direct_projection() {
        let shape = FixtureShape {
            log_t: 5,
            ram_k: 16,
        };
        let ops = vec![
            RamOp::Write { word: 3, post: 5 },
            RamOp::Read { word: 3 },
            RamOp::Write { word: 3, post: 9 },
            RamOp::None,
            RamOp::Write { word: 4, post: 0 },
            RamOp::Read { word: 7 },
        ];
        with_ram_fixture_backend(shape, ops, |witness| {
            let context = SolinasMetal::for_akita_production().unwrap();
            let (outer_rows, ready) = prepare_metal_spartan_outer_stage1_owner_witness_rows(
                &context,
                witness,
                1 << shape.log_t,
                false,
                false,
                false,
                false,
                false,
            )
            .unwrap();
            let product_rows = outer_rows.share_product_remainder_rows().unwrap();
            let source = RamReadWriteStage1Source::new(ready.owner, product_rows).unwrap();
            assert!(source.ram_remap_compatible());
            let projection = source.collect_direct_addresses().unwrap();
            let mut actual_session = ProofSession::default();
            projection.publish(&mut actual_session).unwrap();

            let mut expected_session = ProofSession::default();
            let expected_columns =
                RamAccessColumns::shared::<AkitaField>(&mut expected_session, witness, shape.log_t)
                    .unwrap();
            let actual_columns = actual_session.state::<Arc<RamAccessColumns>>().unwrap();
            assert_eq!(actual_columns.addresses, expected_columns.addresses);
            assert_eq!(
                actual_columns.active_cycle_bound(),
                expected_columns.active_cycle_bound()
            );

            assert!(actual_session.state::<RamAccessValues>().is_none());
            let expected_values = expected_session.state::<RamAccessValues>().unwrap();
            let values = source.values().unwrap();
            let actual_records = actual_columns
                .addresses
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, address)| *address != NO_ACCESS)
                .map(|(cycle, address)| {
                    let (pre_value, post_value) = values.value_at(cycle).unwrap();
                    (cycle as u32, address, pre_value, post_value)
                })
                .collect::<Vec<_>>();
            let expected_records = expected_session
                .state::<RamAccessTape>()
                .unwrap()
                .records()
                .unwrap()
                .iter()
                .map(|record| {
                    (
                        record.cycle,
                        record.address,
                        record.pre_value,
                        record.post_value,
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(actual_records, expected_records);
            for &(cycle, _, pre_value, post_value) in &actual_records {
                assert_eq!(pre_value, expected_values.pre_values[cycle as usize]);
                assert_eq!(post_value, expected_values.post_values[cycle as usize]);
            }

            assert!(actual_session
                .state::<Arc<crate::metal::ram_records::RamIncrementActivity>>()
                .is_none());

            let actual_tape = actual_session.state::<RamAccessTape>().unwrap();
            let expected_tape = expected_session.state::<RamAccessTape>().unwrap();
            assert_eq!(actual_tape.access_count(), expected_tape.access_count());
            assert_eq!(
                actual_tape.increment_compatible(),
                expected_tape.increment_compatible()
            );
            assert_eq!(
                actual_tape.ram_ra_compatible(),
                expected_tape.ram_ra_compatible()
            );
            assert_eq!(actual_tape.hamming_exact(), expected_tape.hamming_exact());
            assert!(actual_tape.records().is_none());
            assert!(actual_columns.ram_ra_sparse_layout().is_none());
        });
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
            config.ram_read_write.gpu_record_scatter_cutoff_elements = 2;
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
    fn stage1_coproduced_record_sequence_matches_optimized_cpu() {
        let shape = FixtureShape {
            log_t: 5,
            ram_k: 16,
        };
        let ops = vec![
            RamOp::Write { word: 3, post: 5 },
            RamOp::Read { word: 3 },
            RamOp::Write { word: 3, post: 9 },
            RamOp::Read { word: 15 },
            RamOp::None,
            RamOp::Write { word: 4, post: 2 },
            RamOp::Read { word: 3 },
            RamOp::Write { word: 15, post: 6 },
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
            config.ram_read_write.gpu_record_scatter_cutoff_elements = 2;
            let metal = MetalBackend::new(config).unwrap();
            let (_outer_rows, mut ready) = prepare_metal_spartan_outer_stage1_owner_witness_rows(
                &metal.context,
                witness,
                1 << shape.log_t,
                false,
                false,
                false,
                false,
                true,
            )
            .unwrap();
            let mut session = ProofSession::default();
            ready
                .ram_read_write_records
                .take()
                .unwrap()
                .publish::<AkitaField>(&mut session)
                .unwrap();
            metal
                .start_ram_read_write_sequence_prefetch(&mut session, shape.log_t)
                .unwrap();
            assert!(session.state::<PendingRamReadWriteSequence>().is_some());
            assert!(session.state::<RamReadWriteRecordChunks>().is_none());
            let mut actual =
                PrepareKernel::prepare(&metal, &mut session, witness, inputs()).unwrap();
            assert_eq!(metal.ram_read_write_metal_sequences(), 1);
            assert_eq!(metal.ram_read_write_sparse_sequences(), 0);
            assert!(session.state::<RamAccessValues>().is_none());
            assert!(session.state::<RamAccessTape>().is_some());
            assert!(session.state::<Arc<RamAccessColumns>>().is_some());
            assert!(session.state::<Arc<RamIncrementActivity>>().is_some());
            assert!(session.state::<RamReadWriteStage1Source>().is_none());
            assert!(session.state::<RamReadWriteRecordChunks>().is_none());
            assert!(session.state::<PendingRamReadWriteSequence>().is_none());

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
    fn stage1_coproduced_record_prefix_matches_optimized_cpu() {
        let shape = FixtureShape {
            log_t: 13,
            ram_k: 4,
        };
        let mut ops = vec![RamOp::Read { word: 3 }; (1 << shape.log_t) - 1];
        ops[0] = RamOp::Write { word: 3, post: 11 };
        ops[4097] = RamOp::Write { word: 3, post: 29 };
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
            let (_outer_rows, mut ready) = prepare_metal_spartan_outer_stage1_owner_witness_rows(
                &metal.context,
                witness,
                1 << shape.log_t,
                false,
                false,
                false,
                false,
                true,
            )
            .unwrap();
            let mut session = ProofSession::default();
            ready
                .ram_read_write_records
                .take()
                .unwrap()
                .publish::<AkitaField>(&mut session)
                .unwrap();
            metal
                .start_ram_read_write_sequence_prefetch(&mut session, shape.log_t)
                .unwrap();
            let mut actual =
                PrepareKernel::prepare(&metal, &mut session, witness, inputs()).unwrap();

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
