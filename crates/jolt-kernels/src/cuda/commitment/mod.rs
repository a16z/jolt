use std::sync::{mpsc, Arc};

use ark_bn254::{Fq, G1Affine, G1Projective};
use ark_ec::CurveGroup;
use ark_ff::BigInt;
use cudarc::driver::CudaSlice;
use jolt_claims::protocols::jolt::{
    JoltCommittedPolynomial, JoltPolynomialId, TracePolynomialOrder,
};
use jolt_crypto::Bn254G1;
use jolt_dory::DoryScheme;
use jolt_field::Field;
use jolt_openings::{CommitmentScheme, StreamingCommitment};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use jolt_witness::backend::cuda::HotSource;
use jolt_witness::{stream_witnesses, JoltWitnessOracle, JoltWitnessPlane, StreamConsumer};

use super::common::context::CudaKernelContext;
use super::common::device::require_fr_slice;
use super::common::device_columns::{park_device_column, DeviceColumn};
use super::common::error::CudaError;
use super::common::msm::{AffineLimbs, DeviceG1Bases, JacobianLimbs, FQ_LIMBS};
use super::common::pack::COLD;
use super::{require_context, CudaBackend};
use crate::commitment::{
    finish_streamed, CommitWitness, CommitmentGrid, ModeStreamingCommitment, WitnessCommitment,
};
use crate::cuda::inc_claim_reduction::witness::IncClaimReductionWitness;
use crate::cuda::witness::session_device_trace;
#[cfg(feature = "parallel")]
use crate::optimized::rows::{collect_range_into, RandomAccessRows};
use crate::reference::commitment::{column_kinds, ColumnKind, MaterializedColumn};
use crate::{KernelError, ProofSession};

pub type FinishedColumn<PCS> = (
    <PCS as jolt_crypto::Commitment>::Output,
    <PCS as CommitmentScheme>::OpeningHint,
);

pub trait DeviceTier1Commitment: CommitmentScheme + ModeStreamingCommitment {
    const BATCHES_TIER2: bool = false;

    fn tier1_bases(setup: &Self::ProverSetup, count: usize) -> Result<Vec<AffineLimbs>, CudaError>;

    fn partial_from_rows(
        setup: &Self::ProverSetup,
        rows: &[JacobianLimbs],
    ) -> Result<Self::PartialCommitment, CudaError>;

    fn tier2_columns(
        setup: &Self::ProverSetup,
        columns: &[Vec<JacobianLimbs>],
    ) -> Result<Vec<FinishedColumn<Self>>, CudaError> {
        columns
            .iter()
            .map(|rows| {
                let partial = Self::partial_from_rows(setup, rows)?;
                Ok(finish_streamed::<Self>(partial, setup))
            })
            .collect()
    }
}

fn fq_from_limbs(limbs: [u64; FQ_LIMBS]) -> Fq {
    Fq::new_unchecked(BigInt(limbs))
}

fn fq_limbs(value: Fq) -> [u64; FQ_LIMBS] {
    value.0 .0
}

fn affine_limbs(point: G1Affine) -> AffineLimbs {
    if point.infinity {
        return AffineLimbs::IDENTITY;
    }
    AffineLimbs {
        x: fq_limbs(point.x),
        y: fq_limbs(point.y),
        infinity: false,
    }
}

fn jolt_g1(point: JacobianLimbs) -> Bn254G1 {
    Bn254G1::from(G1Projective::new_unchecked(
        fq_from_limbs(point.x),
        fq_from_limbs(point.y),
        fq_from_limbs(point.z),
    ))
}

impl DeviceTier1Commitment for DoryScheme {
    fn tier1_bases(setup: &Self::ProverSetup, count: usize) -> Result<Vec<AffineLimbs>, CudaError> {
        if setup.0.g1_vec.len() < count {
            return Err(CudaError::LengthMismatch {
                expected: count,
                got: setup.0.g1_vec.len(),
            });
        }
        Ok(setup.0.g1_vec[..count]
            .iter()
            .map(|base| affine_limbs(base.0.into_affine()))
            .collect())
    }

    fn partial_from_rows(
        setup: &Self::ProverSetup,
        rows: &[JacobianLimbs],
    ) -> Result<Self::PartialCommitment, CudaError> {
        let mut partial = <Self as StreamingCommitment>::begin(setup);
        partial
            .row_commitments
            .extend(rows.iter().copied().map(jolt_g1));
        Ok(partial)
    }
}

struct IncrementColumns {
    kinds: Vec<ColumnKind>,
    increments: Vec<Vec<i128>>,
}

impl IncrementColumns {
    fn begin(kinds: &[ColumnKind], cycles: usize) -> Self {
        let increments = kinds
            .iter()
            .map(|kind| {
                if kind.is_one_hot() {
                    Vec::new()
                } else {
                    Vec::with_capacity(cycles)
                }
            })
            .collect();
        Self {
            kinds: kinds.to_vec(),
            increments,
        }
    }
}

fn project_increment(kind: ColumnKind, chunk: &[IncClaimReductionWitness], out: &mut Vec<i128>) {
    match kind {
        ColumnKind::RdInc => out.extend(chunk.iter().map(|row| row.rd.0)),
        ColumnKind::RamInc => out.extend(chunk.iter().map(|row| row.ram.0)),
        ColumnKind::InstructionRa(_) | ColumnKind::BytecodeRa(_) | ColumnKind::RamRa(_) => {}
    }
}

impl StreamConsumer for IncrementColumns {
    type Witness = IncClaimReductionWitness;

    fn consume(&mut self, chunk: &[IncClaimReductionWitness]) {
        #[cfg(feature = "parallel")]
        self.increments
            .par_iter_mut()
            .zip(self.kinds.par_iter())
            .for_each(|(out, &kind)| project_increment(kind, chunk, out));
        #[cfg(not(feature = "parallel"))]
        for (out, &kind) in self.increments.iter_mut().zip(self.kinds.iter()) {
            project_increment(kind, chunk, out);
        }
    }
}

#[cfg(feature = "parallel")]
const SUPERCHUNK_CYCLES_MAX: usize = 1 << 21;

#[cfg(feature = "parallel")]
fn superchunk_cycles() -> usize {
    (rayon::current_num_threads() << 15)
        .next_power_of_two()
        .clamp(1 << 17, SUPERCHUNK_CYCLES_MAX)
}

#[cfg(feature = "parallel")]
fn collect_columns_pipelined<F, C>(
    access: &RandomAccessRows<'_>,
    collected: &mut C,
    cycles: usize,
    superchunk: usize,
) -> Result<(), KernelError<F>>
where
    F: Field,
    C: StreamConsumer,
    C::Witness: Copy + Sync,
{
    let mut front: Vec<C::Witness> = Vec::new();
    let mut back: Vec<C::Witness> = Vec::new();
    let mut end = superchunk.min(cycles);
    collect_range_into(access, 0..end, &mut front)?;
    loop {
        let next_end = (end + superchunk).min(cycles);
        let (fill, ()) = rayon::join(
            || {
                if end < next_end {
                    collect_range_into(access, end..next_end, &mut back).map(|()| true)
                } else {
                    Ok(false)
                }
            },
            || collected.consume(&front),
        );
        if !fill? {
            break;
        }
        core::mem::swap(&mut front, &mut back);
        end = next_end;
    }
    Ok(())
}

fn collect_increments<F: Field>(
    source: &dyn JoltWitnessPlane<F>,
    kinds: &[ColumnKind],
    cycles: usize,
    row_width: usize,
    superchunk: usize,
) -> Result<IncrementColumns, KernelError<F>> {
    let mut collected = IncrementColumns::begin(kinds, cycles);
    #[cfg(feature = "parallel")]
    if let Some(access) = RandomAccessRows::new(source, cycles)? {
        collect_columns_pipelined::<F, _>(&access, &mut collected, cycles, superchunk)?;
        return Ok(collected);
    }
    let _ = superchunk;
    let mut consumers = (collected,);
    stream_witnesses(source, 0..cycles, row_width, &mut consumers)?;
    Ok(consumers.0)
}

fn device_hot_columns<F: Field>(
    context: &'static CudaKernelContext,
    session: &mut ProofSession,
    source: &dyn JoltWitnessPlane<F>,
    kinds: &[ColumnKind],
    ids: &[JoltCommittedPolynomial],
    cycles: usize,
    one_hot_k: usize,
) -> Result<RetainedHotColumns, KernelError<F>> {
    let trace = session_device_trace(context, session, source, cycles)?;
    let wanted = |family: fn(&ColumnKind) -> bool| kinds.iter().any(family);
    let lookup = wanted(|kind| matches!(kind, ColumnKind::InstructionRa(_)))
        .then(|| trace.lookup_index_limbs())
        .transpose()?;
    let pc = wanted(|kind| matches!(kind, ColumnKind::BytecodeRa(_)))
        .then(|| trace.mapped_pc_words())
        .transpose()?;
    let ram = wanted(|kind| matches!(kind, ColumnKind::RamRa(_)))
        .then(|| trace.remapped_ram_words(COLD as usize))
        .transpose()?
        .map(|(column, _)| column);

    let missing = || KernelError::InvariantViolation {
        reason: "a committed one-hot family has no device source column",
    };
    let mut requests = Vec::with_capacity(ids.len());
    for &kind in kinds {
        let request = match kind {
            ColumnKind::InstructionRa(selector) => (
                HotSource::Interleaved(lookup.as_ref().ok_or_else(missing)?),
                selector,
            ),
            ColumnKind::BytecodeRa(selector) => {
                (HotSource::Word(pc.as_ref().ok_or_else(missing)?), selector)
            }
            ColumnKind::RamRa(selector) => {
                (HotSource::Word(ram.as_ref().ok_or_else(missing)?), selector)
            }
            ColumnKind::RdInc | ColumnKind::RamInc => continue,
        };
        requests.push(request);
    }
    let mut chunked = trace.hot_chunk_columns(&requests, one_hot_k)?.into_iter();

    let mut retained = Vec::with_capacity(ids.len());
    for (index, &id) in ids.iter().enumerate() {
        if !kinds[index].is_one_hot() {
            retained.push(None);
            continue;
        }
        let (column, span) = chunked.next().ok_or(KernelError::InvariantViolation {
            reason: "the device chunk batch is shorter than the one-hot column count",
        })?;
        let column = Arc::new(column);
        park_device_column(
            session,
            source,
            DeviceColumn::CommittedHot(id),
            cycles,
            span,
            Arc::clone(&column),
        );
        retained.push(Some(column));
    }
    Ok(retained)
}

type RetainedHotColumns = Vec<Option<Arc<CudaSlice<u32>>>>;

fn tier1_order(hot: &RetainedHotColumns) -> impl Iterator<Item = usize> + '_ {
    let one_hot = hot
        .iter()
        .enumerate()
        .filter_map(|(index, column)| column.is_some().then_some(index));
    let dense = hot
        .iter()
        .enumerate()
        .filter_map(|(index, column)| column.is_none().then_some(index));
    one_hot.chain(dense)
}

struct ResidentBases {
    count: usize,
    bases: DeviceG1Bases,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for ResidentBases {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(allocative::Key::new("bases"), self.bases.device_bytes());
        visitor.exit();
    }
}

fn device_bases<'a, F, PCS>(
    session: &'a mut ProofSession,
    context: &'static CudaKernelContext,
    setup: &PCS::ProverSetup,
    count: usize,
) -> Result<&'a DeviceG1Bases, KernelError<F>>
where
    F: Field,
    PCS: DeviceTier1Commitment,
{
    let stale = session
        .state::<ResidentBases>()
        .is_none_or(|resident| resident.count < count);
    if stale {
        let bases = PCS::tier1_bases(setup, count)?;
        let bases = context.upload_g1_bases(&bases)?;
        session.park(ResidentBases { count, bases });
    }
    session
        .state::<ResidentBases>()
        .map(|resident| &resident.bases)
        .ok_or(KernelError::InvariantViolation {
            reason: "the resident tier-1 base table was just parked",
        })
}

fn finish<F, PCS>(
    setup: &PCS::ProverSetup,
    rows: &[JacobianLimbs],
    id: JoltCommittedPolynomial,
) -> Result<WitnessCommitment<PCS>, KernelError<F>>
where
    F: Field,
    PCS: DeviceTier1Commitment,
{
    let partial = PCS::partial_from_rows(setup, rows)?;
    let (commitment, hint) = finish_streamed::<PCS>(partial, setup);
    Ok(WitnessCommitment {
        id,
        commitment,
        hint,
    })
}

impl<F, PCS> CommitWitness<F, PCS> for CudaBackend
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + DeviceTier1Commitment,
{
    fn commit_witness(
        &self,
        session: &mut ProofSession,
        source: &dyn JoltWitnessPlane<F>,
        ids: &[JoltCommittedPolynomial],
        grid: CommitmentGrid,
        setup: &PCS::ProverSetup,
    ) -> Result<Vec<WitnessCommitment<PCS>>, KernelError<F>> {
        let context = require_context::<F>()?;
        let kinds = column_kinds::<F>(ids, grid)?;
        let cycles = 1usize << grid.log_t;
        let row_width = grid.num_columns();

        if grid.order == TracePolynomialOrder::CycleMajor && row_width <= cycles {
            let one_hot_k = 1usize << grid.log_k_chunk;
            #[cfg(feature = "parallel")]
            let superchunk = superchunk_cycles().max(row_width);
            #[cfg(not(feature = "parallel"))]
            let superchunk = row_width;
            let hot = tracing::info_span!("cuda_commit_park_hot", columns = kinds.len()).in_scope(
                || {
                    device_hot_columns::<F>(
                        context, session, source, &kinds, ids, cycles, one_hot_k,
                    )
                },
            )?;
            let collected =
                tracing::info_span!("cuda_commit_collect_columns", cycles).in_scope(|| {
                    collect_increments::<F>(source, &kinds, cycles, row_width, superchunk)
                })?;
            let bases = tracing::info_span!("cuda_commit_bases", bases = row_width)
                .in_scope(|| device_bases::<F, PCS>(session, context, setup, row_width))?;
            if PCS::BATCHES_TIER2 {
                let mut columns: Vec<Option<Vec<JacobianLimbs>>> =
                    (0..ids.len()).map(|_| None).collect();
                for index in tier1_order(&hot) {
                    let rows = match &hot[index] {
                        Some(column) => {
                            tracing::info_span!("cuda_commit_tier1_one_hot").in_scope(|| {
                                context.one_hot_rows_device(
                                    bases, column, cycles, one_hot_k, row_width,
                                )
                            })
                        }
                        None => tracing::info_span!("cuda_commit_tier1_dense").in_scope(|| {
                            context.msm_rows_i128(bases, &collected.increments[index], row_width)
                        }),
                    }?;
                    let slot = columns
                        .get_mut(index)
                        .ok_or(KernelError::InvariantViolation {
                            reason: "the commit pipeline produced a column outside the id list",
                        })?;
                    *slot = Some(rows);
                }
                let columns = columns.into_iter().collect::<Option<Vec<_>>>().ok_or(
                    KernelError::InvariantViolation {
                        reason: "the commit pipeline finished fewer columns than it was given",
                    },
                )?;
                let finished = tracing::info_span!("cuda_commit_tier2", columns = columns.len())
                    .in_scope(|| PCS::tier2_columns(setup, &columns))?;
                return finished
                    .into_iter()
                    .zip(ids)
                    .map(|((commitment, hint), &id)| {
                        Ok(WitnessCommitment {
                            id,
                            commitment,
                            hint,
                        })
                    })
                    .collect();
            }

            let parent = tracing::Span::current();
            let (tx, rx) = mpsc::channel::<(usize, Vec<JacobianLimbs>)>();
            let mut tier1 = Ok(());
            let tier2 = std::thread::scope(|scope| {
                let consumer = scope.spawn(move || {
                    let mut finished: Vec<Option<WitnessCommitment<PCS>>> =
                        (0..ids.len()).map(|_| None).collect();
                    for (index, rows) in rx {
                        let id = *ids.get(index).ok_or(KernelError::InvariantViolation {
                            reason: "the commit pipeline produced a column outside the id list",
                        })?;
                        let slot =
                            finished
                                .get_mut(index)
                                .ok_or(KernelError::InvariantViolation {
                                    reason:
                                        "the commit pipeline produced a column outside the id list",
                                })?;
                        *slot = Some(
                            tracing::info_span!(parent: &parent, "cuda_commit_tier2")
                                .in_scope(|| finish::<F, PCS>(setup, &rows, id))?,
                        );
                    }
                    finished.into_iter().collect::<Option<Vec<_>>>().ok_or(
                        KernelError::InvariantViolation {
                            reason: "the commit pipeline finished fewer columns than it was given",
                        },
                    )
                });
                for index in tier1_order(&hot) {
                    let rows = match &hot[index] {
                        Some(column) => {
                            tracing::info_span!("cuda_commit_tier1_one_hot").in_scope(|| {
                                context.one_hot_rows_device(
                                    bases, column, cycles, one_hot_k, row_width,
                                )
                            })
                        }
                        None => tracing::info_span!("cuda_commit_tier1_dense").in_scope(|| {
                            context.msm_rows_i128(bases, &collected.increments[index], row_width)
                        }),
                    };
                    match rows {
                        Ok(rows) => {
                            if tx.send((index, rows)).is_err() {
                                break;
                            }
                        }
                        Err(error) => {
                            tier1 = Err(error);
                            break;
                        }
                    }
                }
                drop(tx);
                consumer.join()
            });
            tier1?;
            return match tier2 {
                Ok(finished) => finished,
                Err(payload) => std::panic::resume_unwind(payload),
            };
        }

        let bases = tracing::info_span!("cuda_commit_bases", bases = row_width)
            .in_scope(|| device_bases::<F, PCS>(session, context, setup, row_width))?;
        kinds
            .iter()
            .zip(ids)
            .map(|(&kind, &id)| {
                let mut consumers = (MaterializedColumn::<F>::begin(kind, grid),);
                stream_witnesses(source, 0..cycles, row_width, &mut consumers)?;
                let table = consumers.0.table;
                let rows = tracing::info_span!("cuda_commit_tier1_dense")
                    .in_scope(|| device_rows::<F>(context, bases, &table, row_width))?;
                tracing::info_span!("cuda_commit_tier2")
                    .in_scope(|| finish::<F, PCS>(setup, &rows, id))
            })
            .collect()
    }

    fn commit_advice(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessOracle<F>,
        id: JoltCommittedPolynomial,
        grid: CommitmentGrid,
        setup: &PCS::ProverSetup,
    ) -> Result<WitnessCommitment<PCS>, KernelError<F>> {
        let context = require_context::<F>()?;
        let row_width = grid.num_columns();
        let bases = tracing::info_span!("cuda_commit_bases", bases = row_width)
            .in_scope(|| device_bases::<F, PCS>(session, context, setup, row_width))?;
        let values = witness.oracle_table(JoltPolynomialId::Committed(id))?;
        let rows = device_rows::<F>(context, bases, &values, row_width)?;
        finish::<F, PCS>(setup, &rows, id)
    }
}

fn device_rows<F: Field>(
    context: &'static CudaKernelContext,
    bases: &DeviceG1Bases,
    table: &[F],
    row_width: usize,
) -> Result<Vec<JacobianLimbs>, KernelError<F>> {
    if table.len() <= row_width {
        let scalars = context.upload(require_fr_slice(table)?)?;
        return Ok(context.msm_rows_fr(bases, &scalars, table.len())?);
    }
    if !table.len().is_multiple_of(row_width) {
        return Err(KernelError::InvalidGeometry {
            reason: format!(
                "materialized table of {} coefficients is not a multiple of the {row_width}-wide                  grid row",
                table.len()
            ),
        });
    }
    let scalars = context.upload(require_fr_slice(table)?)?;
    Ok(context.msm_rows_fr(bases, &scalars, row_width)?)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    #[cfg(not(feature = "zk"))]
    use jolt_claims::protocols::jolt::JoltAdviceKind;
    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltOneHotConfig, TracePolynomialOrder,
    };
    #[cfg(not(feature = "zk"))]
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    #[cfg(not(feature = "zk"))]
    use jolt_openings::CommitmentScheme;
    use jolt_witness::{JoltWitnessOracle, JoltWitnessPlane};

    #[cfg(not(feature = "zk"))]
    use super::CudaBackend;
    #[cfg(not(feature = "zk"))]
    use crate::commitment::{CommitWitness, WitnessCommitment};
    use crate::commitment::{CommitmentGrid, CommittedColumnsWitness};
    use crate::cuda::common::context::shared_context;
    #[cfg(not(feature = "zk"))]
    use crate::cuda::common::testing::advice_plane;
    use crate::cuda::common::testing::with_r1cs_witness;
    #[cfg(not(feature = "zk"))]
    use crate::reference::ReferenceBackend;
    use crate::ProofSession;

    const RAM_K: usize = 1 << 10;

    const LOG_K_CHUNK: usize = 8;

    const TOTAL_VARS: usize = 16;

    #[cfg(not(feature = "zk"))]
    const ADVICE_BYTES: usize = 4096;

    const fn one_hot() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: LOG_K_CHUNK as u8,
            lookups_ra_virtual_log_k_chunk: 32,
        }
    }

    const fn grid_at(order: TracePolynomialOrder, log_t: usize) -> CommitmentGrid {
        CommitmentGrid {
            total_vars: TOTAL_VARS,
            log_t,
            log_k_chunk: LOG_K_CHUNK,
            order,
        }
    }

    const CONFIGS: [(TracePolynomialOrder, usize); 3] = [
        (TracePolynomialOrder::CycleMajor, 8),
        (TracePolynomialOrder::AddressMajor, 8),
        (TracePolynomialOrder::AddressMajor, 6),
    ];

    #[cfg(not(feature = "zk"))]
    const ADVICE_KINDS: [(JoltAdviceKind, JoltCommittedPolynomial); 2] = [
        (
            JoltAdviceKind::Trusted,
            JoltCommittedPolynomial::TrustedAdvice,
        ),
        (
            JoltAdviceKind::Untrusted,
            JoltCommittedPolynomial::UntrustedAdvice,
        ),
    ];

    #[cfg(not(feature = "zk"))]
    fn commit_columns(
        backend: &dyn CommitWitness<Fr, DoryScheme>,
        witness: &impl JoltWitnessPlane<Fr>,
        grid: CommitmentGrid,
    ) -> Vec<WitnessCommitment<DoryScheme>> {
        let ids = trace_ids(witness);
        let setup = DoryScheme::setup_prover(grid.total_vars);
        backend
            .commit_witness(&mut ProofSession::default(), witness, &ids, grid, &setup)
            .expect("commit_witness")
    }

    #[cfg(not(feature = "zk"))]
    fn commit_advice_column(
        backend: &dyn CommitWitness<Fr, DoryScheme>,
        witness: &dyn JoltWitnessOracle<Fr>,
        id: JoltCommittedPolynomial,
        grid: CommitmentGrid,
        setup: &<DoryScheme as CommitmentScheme>::ProverSetup,
    ) -> WitnessCommitment<DoryScheme> {
        backend
            .commit_advice(&mut ProofSession::default(), witness, id, grid, setup)
            .expect("commit_advice")
    }

    #[cfg(not(feature = "zk"))]
    fn advice_grid(words: usize) -> CommitmentGrid {
        CommitmentGrid {
            total_vars: words.ilog2() as usize,
            log_t: 0,
            log_k_chunk: 0,
            order: TracePolynomialOrder::CycleMajor,
        }
    }

    fn trace_ids(witness: &dyn JoltWitnessOracle<Fr>) -> Vec<JoltCommittedPolynomial> {
        witness
            .committed_order()
            .expect("committed order")
            .into_iter()
            .filter(|id| {
                !matches!(
                    id,
                    JoltCommittedPolynomial::TrustedAdvice
                        | JoltCommittedPolynomial::UntrustedAdvice
                )
            })
            .collect()
    }

    #[cfg(not(feature = "zk"))]
    fn assert_commitments_match(
        expected: &[WitnessCommitment<DoryScheme>],
        got: &[WitnessCommitment<DoryScheme>],
        label: &str,
    ) {
        assert_eq!(
            got.len(),
            expected.len(),
            "{label}: committed column count diverged",
        );
        for (expected, got) in expected.iter().zip(got) {
            assert_eq!(got.id, expected.id, "{label}: committed order diverged");
            assert_eq!(
                got.commitment, expected.commitment,
                "{label}: commitment for {:?} diverged",
                expected.id,
            );
            assert_eq!(
                got.hint, expected.hint,
                "{label}: opening hint for {:?} diverged",
                expected.id,
            );
        }
    }

    #[test]
    fn fixture_commit_configs_cover_every_reference_placement_branch() {
        let mut fused = 0usize;
        let mut materialized = 0usize;
        let mut widened = 0usize;
        for (order, log_t) in CONFIGS {
            let grid = grid_at(order, log_t);
            let cycles = 1usize << grid.log_t;
            if order == TracePolynomialOrder::CycleMajor && grid.num_columns() <= cycles {
                fused += 1;
            } else {
                materialized += 1;
            }
            if grid.one_hot_stride() > 1 {
                widened += 1;
            }
            assert!(
                grid.total_vars >= grid.log_t + grid.log_k_chunk,
                "log_T {log_t}: the grid must be at least as wide as the main one-hot matrix",
            );
        }
        assert!(
            fused > 0,
            "no config reaches the fused streaming commit, which is the production \
             cycle-major path",
        );
        assert!(
            materialized > 0,
            "no config reaches the materializing commit, so the address-major strides are \
             untested",
        );
        assert!(
            widened > 0,
            "no config widens the grid past the main matrix, so a kernel that ignored \
             `one_hot_stride` would pass",
        );
    }

    #[test]
    fn fixture_committed_columns_discriminate() {
        with_r1cs_witness(8, RAM_K, one_hot(), 11, |witness| {
            let ids = trace_ids(witness);
            let dense = ids
                .iter()
                .filter(|id| {
                    matches!(
                        id,
                        JoltCommittedPolynomial::RdInc | JoltCommittedPolynomial::RamInc
                    )
                })
                .count();
            let one_hot_columns = ids.len() - dense;
            assert!(
                dense > 0,
                "no dense increment column, so the scalar feed path is untested",
            );
            assert!(
                one_hot_columns > 0,
                "no one-hot column, so the column-major one-hot path is untested",
            );
            let instruction_chunks = ids
                .iter()
                .filter(|id| matches!(id, JoltCommittedPolynomial::InstructionRa(_)))
                .count();
            assert!(
                instruction_chunks > 1,
                "only {instruction_chunks} instruction chunk, so a kernel that ignored the \
                 chunk selector would pass",
            );
        });
    }

    #[cfg(not(feature = "zk"))]
    #[test]
    fn commit_witness_matches_reference() {
        let Some(_) = shared_context() else {
            return;
        };
        let expected: Vec<Vec<WitnessCommitment<DoryScheme>>> = CONFIGS
            .iter()
            .map(|&(order, log_t)| {
                with_r1cs_witness(log_t, RAM_K, one_hot(), 11, |witness| {
                    commit_columns(&ReferenceBackend, witness, grid_at(order, log_t))
                })
            })
            .collect();
        for (&(order, log_t), expected) in CONFIGS.iter().zip(&expected) {
            with_r1cs_witness(log_t, RAM_K, one_hot(), 11, |witness| {
                let got = commit_columns(&CudaBackend, witness, grid_at(order, log_t));
                assert_commitments_match(expected, &got, &format!("{order:?} log_T {log_t}"));
            });
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn pipelined_increment_collection_matches_the_chunk_walk() {
        let log_t = 8;
        let cycles = 1usize << log_t;
        let grid = grid_at(TracePolynomialOrder::CycleMajor, log_t);
        let row_width = grid.num_columns();
        with_r1cs_witness(log_t, RAM_K, one_hot(), 17, |witness| {
            let ids = trace_ids(witness);
            let kinds = super::column_kinds::<Fr>(&ids, grid).expect("column kinds");
            let plane = witness as &dyn JoltWitnessPlane<Fr>;
            assert!(
                super::RandomAccessRows::new::<Fr>(plane, cycles)
                    .expect("random-access view")
                    .is_some(),
                "the fixture is not slice-backed, so collect_increments takes the chunk walk on \
                 both sides and this compares the walk to itself",
            );
            let walked = {
                let mut consumers = (super::IncrementColumns::begin(&kinds, cycles),);
                jolt_witness::stream_witnesses(plane, 0..cycles, row_width, &mut consumers)
                    .expect("chunk walk");
                consumers.0
            };
            assert!(
                walked
                    .increments
                    .iter()
                    .any(|column| column.len() == cycles),
                "the fixture commits no increment column, so the pipelined path is untested",
            );
            for superchunk in [row_width, 2 * row_width, cycles / 2, cycles, 2 * cycles] {
                let got =
                    super::collect_increments::<Fr>(plane, &kinds, cycles, row_width, superchunk)
                        .expect("pipelined collection");
                assert_eq!(
                    got.increments, walked.increments,
                    "superchunk {superchunk}: the increment columns diverge from the chunk walk",
                );
            }
        });
    }

    #[test]
    fn device_hot_columns_match_the_host_encoder() {
        let Some(context) = shared_context() else {
            return;
        };
        let log_t = 8;
        let cycles = 1usize << log_t;
        let grid = grid_at(TracePolynomialOrder::CycleMajor, log_t);
        let one_hot_k = 1usize << grid.log_k_chunk;
        with_r1cs_witness(log_t, RAM_K, one_hot(), 23, |witness| {
            let ids = trace_ids(witness);
            let kinds = super::column_kinds::<Fr>(&ids, grid).expect("column kinds");
            let plane = witness as &dyn JoltWitnessPlane<Fr>;
            let rows = crate::optimized::support::collect_rows::<Fr, CommittedColumnsWitness>(
                plane, cycles,
            )
            .expect("reference rows");

            let mut session = ProofSession::default();
            let hot = super::device_hot_columns::<Fr>(
                context,
                &mut session,
                plane,
                &kinds,
                &ids,
                cycles,
                one_hot_k,
            )
            .expect("device hot columns");

            let mut one_hot_columns = 0usize;
            let mut cold_seen = false;
            for (index, (column, &kind)) in hot.iter().zip(&kinds).enumerate() {
                let Some(column) = column else {
                    assert!(
                        !kind.is_one_hot(),
                        "{:?} is one-hot but the device plan has no column",
                        ids[index],
                    );
                    continue;
                };
                one_hot_columns += 1;
                let expected: Vec<u32> = rows
                    .iter()
                    .map(|row| {
                        kind.hot_address(row)
                            .map_or(super::COLD, |address| address as u32)
                    })
                    .collect();
                cold_seen |= expected.contains(&super::COLD);
                assert!(
                    expected.iter().any(|&address| address != expected[0]),
                    "{:?}: every cycle shares one address, so a chunk kernel that ignored the \
                     shift would pass",
                    ids[index],
                );
                assert_eq!(
                    context.download_u32(column).expect("download"),
                    expected,
                    "{:?}: the device chunk column diverges from the host encoder",
                    ids[index],
                );
            }
            assert!(
                one_hot_columns > 2,
                "only {one_hot_columns} one-hot columns, so the per-family chunk shifts are \
                 barely exercised",
            );
            assert!(
                cold_seen,
                "no committed column has a cold cycle, so the COLD sentinel is untested",
            );
        });
    }

    #[cfg(not(feature = "zk"))]
    #[test]
    fn commit_advice_matches_reference() {
        let Some(_) = shared_context() else {
            return;
        };
        let fixture = advice_plane(ADVICE_BYTES, 23);
        let grid = advice_grid(fixture.trusted.len());
        let setup = DoryScheme::setup_prover(grid.total_vars);
        let witness = &fixture.plane as &dyn JoltWitnessOracle<Fr>;
        let expected: Vec<WitnessCommitment<DoryScheme>> = ADVICE_KINDS
            .iter()
            .map(|&(_, id)| commit_advice_column(&ReferenceBackend, witness, id, grid, &setup))
            .collect();
        for (&(kind, id), expected) in ADVICE_KINDS.iter().zip(&expected) {
            let got = commit_advice_column(&CudaBackend, witness, id, grid, &setup);
            assert_commitments_match(
                std::slice::from_ref(expected),
                std::slice::from_ref(&got),
                &format!("{kind:?} advice"),
            );
        }
    }
}
