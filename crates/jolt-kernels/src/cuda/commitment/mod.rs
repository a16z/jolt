//! Packed witness columns for CUDA row commitments.
//!
//! CUDA subtracts packed pre/post values on device and keeps each difference
//! as a `u64` magnitude plus a sign byte. Optimized constructs signed `i128`
//! increments from host witness bundles for its streaming commitment calls.
//! CUDA also extracts one-hot chunk indices in batches and retains them in the
//! proof-session cache; optimized streams one-hot addresses from host bundles.
//! Partitioned rows are stitched and grouped for Dory finalization. The default
//! adapter reconstructs the streaming PCS state and finishes it on the host.

use std::sync::Arc;

#[cfg(not(feature = "zk"))]
use ark_bn254::G1Projective;
use ark_bn254::{Fq, G1Affine};
use ark_ff::BigInt;
use cudarc::driver::CudaSlice;
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
#[cfg(not(feature = "zk"))]
use jolt_crypto::Bn254G1;
#[cfg(not(feature = "zk"))]
use jolt_dory::DoryScheme;
use jolt_field::JoltField;
#[cfg(not(feature = "zk"))]
use jolt_openings::{CommitmentScheme, StreamingCommitment};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_witness::backend::cuda::{DeviceTrace, HotSource};
use jolt_witness::RowSource;

use super::common::context::CudaKernelContext;
use super::common::device_columns::{park_device_column, whole_domain, DeviceColumn};
use super::common::devices::CycleWindow;
#[cfg(not(feature = "zk"))]
use super::common::error::CudaError;
#[cfg(not(feature = "zk"))]
use super::common::msm::JacobianLimbs;
use super::common::msm::{AffineLimbs, IncrementKind, FQ_LIMBS};
use super::common::pack::COLD;
#[cfg(not(feature = "zk"))]
use crate::commitment::{finish_streamed, ModeStreamingCommitment};
use crate::cuda::witness::session_row_trace_window;
use crate::reference::commitment::{column_kinds, ColumnKind};
use crate::{KernelError, ProofSession};

#[cfg(not(feature = "zk"))]
pub type FinishedColumn<PCS> = (
    <PCS as jolt_crypto::Commitment>::Output,
    <PCS as CommitmentScheme>::OpeningHint,
);

/// Adapts device-produced G1 row commitments to a streaming PCS.
///
/// Bases and returned rows must use the PCS's row order. The default tier-two
/// path reconstructs a partial commitment and applies mode-specific host
/// finalization; implementations may batch selected columns instead.
#[cfg(not(feature = "zk"))]
pub trait DeviceTier1Commitment: CommitmentScheme + ModeStreamingCommitment {
    fn partial_from_rows(
        setup: &Self::ProverSetup,
        rows: &[JacobianLimbs],
    ) -> Result<Self::PartialCommitment, CudaError>;

    fn tier2_selected(
        _context: &CudaKernelContext,
        setup: &Self::ProverSetup,
        columns: &[Vec<JacobianLimbs>],
        selection: &[usize],
    ) -> Result<Vec<(usize, FinishedColumn<Self>)>, CudaError> {
        selection
            .iter()
            .map(|&index| {
                let rows = columns.get(index).ok_or(CudaError::InvariantViolation {
                    reason: "a tier-2 selection named a column outside the batch",
                })?;
                let partial = Self::partial_from_rows(setup, rows)?;
                Ok((index, finish_streamed::<Self>(partial, setup)))
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

#[cfg(not(feature = "zk"))]
fn jolt_g1(point: JacobianLimbs) -> Bn254G1 {
    Bn254G1::from(G1Projective::new_unchecked(
        fq_from_limbs(point.x),
        fq_from_limbs(point.y),
        fq_from_limbs(point.z),
    ))
}

#[cfg(not(feature = "zk"))]
impl DeviceTier1Commitment for DoryScheme {
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

mod partition;

pub(super) const fn increment_kind(kind: ColumnKind) -> Option<IncrementKind> {
    match kind {
        ColumnKind::RdInc => Some(IncrementKind::Rd),
        ColumnKind::RamInc => Some(IncrementKind::Ram),
        ColumnKind::InstructionRa(_) | ColumnKind::BytecodeRa(_) | ColumnKind::RamRa(_) => None,
    }
}

type HotColumnBuild = Vec<Option<(CudaSlice<u32>, usize)>>;

fn hot_columns_from_trace<F: JoltField>(
    trace: &DeviceTrace,
    kinds: &[ColumnKind],
    one_hot_k: usize,
    window: &CycleWindow,
) -> Result<HotColumnBuild, KernelError<F>> {
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
    let words = window.start..window.end();
    let limbs = 2 * window.start..2 * window.end();
    let mut requests = Vec::with_capacity(kinds.len());
    for &kind in kinds {
        let request = match kind {
            ColumnKind::InstructionRa(selector) => (
                HotSource::Interleaved(lookup.as_ref().ok_or_else(missing)?.slice(limbs.clone())),
                selector,
            ),
            ColumnKind::BytecodeRa(selector) => (
                HotSource::Word(pc.as_ref().ok_or_else(missing)?.slice(words.clone())),
                selector,
            ),
            ColumnKind::RamRa(selector) => (
                HotSource::Word(ram.as_ref().ok_or_else(missing)?.slice(words.clone())),
                selector,
            ),
            ColumnKind::RdInc | ColumnKind::RamInc => continue,
        };
        requests.push(request);
    }
    let mut chunked = trace
        .hot_chunk_columns(&requests, one_hot_k, window.len)?
        .into_iter();

    let mut retained = Vec::with_capacity(kinds.len());
    for &kind in kinds {
        if !kind.is_one_hot() {
            retained.push(None);
            continue;
        }
        let entry = chunked.next().ok_or(KernelError::InvariantViolation {
            reason: "the device chunk batch is shorter than the one-hot column count",
        })?;
        retained.push(Some(entry));
    }
    Ok(retained)
}

fn park_hot_columns(
    session: &mut ProofSession,
    source: &dyn RowSource,
    ordinal: usize,
    ids: &[JoltCommittedPolynomial],
    window: &CycleWindow,
    built: WindowHotColumns,
) -> RetainedHotColumns {
    let mut retained = Vec::with_capacity(ids.len());
    for (&id, entry) in ids.iter().zip(built) {
        match entry {
            None => retained.push(None),
            Some((column, span)) => {
                park_device_column(
                    session,
                    source,
                    ordinal,
                    DeviceColumn::CommittedHot(id),
                    window,
                    span,
                    Arc::clone(&column),
                );
                retained.push(Some(column));
            }
        }
    }
    retained
}

struct HotPlan<'a> {
    kinds: &'a [ColumnKind],
    ids: &'a [JoltCommittedPolynomial],
    one_hot_k: usize,
    window: &'a CycleWindow,
}

fn device_hot_columns<F: JoltField>(
    context: &'static CudaKernelContext,
    session: &mut ProofSession,
    source: &dyn RowSource,
    cycles: usize,
    plan: &HotPlan<'_>,
    preprocessing: &JoltProgramPreprocessing,
) -> Result<(RetainedHotColumns, Arc<DeviceTrace>), KernelError<F>> {
    let trace = session_row_trace_window::<F>(
        context,
        session,
        source,
        preprocessing,
        cycles,
        &plan.window.residency(cycles),
    )?;
    let built = retain_hot_columns(hot_columns_from_trace::<F>(
        &trace,
        plan.kinds,
        plan.one_hot_k,
        plan.window,
    )?);
    Ok((
        park_hot_columns(
            session,
            source,
            context.ordinal(),
            plan.ids,
            plan.window,
            built,
        ),
        trace,
    ))
}

pub(super) type WindowHotColumns = Vec<Option<(Arc<CudaSlice<u32>>, usize)>>;

pub(super) fn retain_hot_columns(built: HotColumnBuild) -> WindowHotColumns {
    built
        .into_iter()
        .map(|entry| entry.map(|(column, span)| (Arc::new(column), span)))
        .collect()
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

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use super::{HotPlan, COLD};
    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltOneHotConfig, TracePolynomialOrder,
    };
    use jolt_field::Fr;
    use jolt_witness::backend::cuda::{
        EXTRA_RAM_READ, EXTRA_RAM_WRITE, EXTRA_RD_POST, EXTRA_RD_PRE, EXTRA_REGISTERS, EXTRA_WORDS,
        REGISTER_ABSENT,
    };
    use jolt_witness::{JoltWitnessOracle, JoltWitnessPlane};

    use crate::commitment::{CommitmentGrid, CommittedColumnsWitness};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::with_r1cs_witness;
    use crate::ProofSession;

    const RAM_K: usize = 1 << 10;
    const LOG_K_CHUNK: usize = 8;
    const TOTAL_VARS: usize = 16;

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

    #[test]
    fn the_increment_kernel_source_agrees_on_the_packed_layout() {
        let source = include_str!("../kernels/commit_increments.cu");
        for (name, value) in [
            ("CI_EXTRA_WORDS", EXTRA_WORDS),
            ("CI_EXTRA_RD_POST", EXTRA_RD_POST),
            ("CI_EXTRA_RAM_READ", EXTRA_RAM_READ),
            ("CI_EXTRA_RAM_WRITE", EXTRA_RAM_WRITE),
            ("CI_EXTRA_REGISTERS", EXTRA_REGISTERS),
            ("CI_EXTRA_RD_PRE", EXTRA_RD_PRE),
            ("CI_REGISTER_ABSENT", REGISTER_ABSENT as usize),
        ] {
            let expected = format!("#define {name} {value}");
            assert!(
                source.contains(&expected),
                "the CUDA source must declare `{expected}`",
            );
        }
    }

    #[test]
    fn device_increment_columns_match_the_host_encoder() {
        let Some(context) = shared_context() else {
            return;
        };
        let log_t = 8;
        let cycles = 1usize << log_t;
        let grid = grid_at(TracePolynomialOrder::CycleMajor, log_t);
        with_r1cs_witness(log_t, RAM_K, one_hot(), 23, |witness| {
            let ids = trace_ids(witness);
            let kinds = super::column_kinds::<Fr>(&ids, grid).expect("column kinds");
            let plane = witness as &dyn JoltWitnessPlane<Fr>;
            let rows =
                crate::optimized::support::collect_rows::<CommittedColumnsWitness>(plane, cycles)
                    .expect("reference rows");

            let mut session = ProofSession::default();
            let trace = crate::cuda::witness::session_device_trace_window(
                context,
                &mut session,
                plane,
                cycles,
                &super::whole_domain(cycles),
            )
            .expect("device residency");

            let mut dense_columns = 0usize;
            for (&id, &kind) in ids.iter().zip(&kinds) {
                let Some(increment) = super::increment_kind(kind) else {
                    continue;
                };
                dense_columns += 1;
                let expected: Vec<i128> = rows.iter().map(|row| kind.increment(row)).collect();
                assert!(
                    expected.iter().any(|&value| value > 0),
                    "{id:?}: no positive increment, so a kernel that dropped the value would pass",
                );
                assert!(
                    expected.contains(&0),
                    "{id:?}: no zero increment, so the absent-operand case is untested",
                );
                let expected = context.signed_column(&expected).expect("host column");

                let got = context
                    .increment_column(trace.extras(), increment, cycles)
                    .expect("device increment column");

                assert_eq!(
                    context.download_u64(got.magnitudes()).expect("magnitudes"),
                    context
                        .download_u64(expected.magnitudes())
                        .expect("magnitudes"),
                    "{id:?}: the device increment magnitudes diverge from the host encoder",
                );
                assert_eq!(
                    context.download_u8(got.signs()).expect("signs"),
                    context.download_u8(expected.signs()).expect("signs"),
                    "{id:?}: the device increment signs diverge from the host encoder",
                );
            }
            assert_eq!(
                dense_columns, 2,
                "the fixture must commit both increment columns",
            );
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
            let rows =
                crate::optimized::support::collect_rows::<CommittedColumnsWitness>(plane, cycles)
                    .expect("reference rows");

            let mut session = ProofSession::default();
            let mut preprocessing = None;
            plane
                .visit_chunks(0..1, 1, &mut |_, _, env| {
                    preprocessing = Some(env.cuda_preprocessing().clone());
                    Ok(())
                })
                .expect("preprocessing");
            let (hot, _) = super::device_hot_columns::<Fr>(
                context,
                &mut session,
                plane,
                cycles,
                &HotPlan {
                    kinds: &kinds,
                    ids: &ids,
                    one_hot_k,
                    window: &super::whole_domain(cycles),
                },
                &preprocessing.expect("witness preprocessing"),
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
                    .map(|row| kind.hot_address(row).map_or(COLD, |address| address as u32))
                    .collect();
                cold_seen |= expected.contains(&COLD);
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
}
