use ark_bn254::{Fq, G1Affine, G1Projective};
use ark_ec::CurveGroup;
use ark_ff::BigInt;
use jolt_claims::protocols::jolt::{
    JoltCommittedPolynomial, JoltPolynomialId, TracePolynomialOrder,
};
use jolt_crypto::Bn254G1;
use jolt_dory::DoryScheme;
use jolt_field::Field;
use jolt_openings::{CommitmentScheme, StreamingCommitment};
use jolt_witness::{stream_witnesses, JoltWitnessOracle, RowSource, StreamConsumer};

use super::common::context::CudaKernelContext;
use super::common::device::require_fr_slice;
use super::common::error::CudaError;
use super::common::msm::{AffineLimbs, DeviceG1Bases, JacobianLimbs, FQ_LIMBS};
use super::common::trace_columns::store_columns;
use super::{require_context, CudaBackend};
use crate::commitment::{
    finish_streamed, CommitWitness, CommitmentGrid, CommittedColumnsWitness,
    ModeStreamingCommitment, WitnessCommitment,
};
use crate::reference::commitment::{column_kinds, ColumnKind, MaterializedColumn};
use crate::{KernelError, ProofSession};

pub trait DeviceTier1Commitment: CommitmentScheme + ModeStreamingCommitment {
    fn tier1_bases(setup: &Self::ProverSetup, count: usize) -> Result<Vec<AffineLimbs>, CudaError>;

    fn partial_from_rows(
        setup: &Self::ProverSetup,
        rows: &[JacobianLimbs],
    ) -> Result<Self::PartialCommitment, CudaError>;
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

struct CollectedColumns {
    kinds: Vec<ColumnKind>,
    increments: Vec<Vec<i128>>,
    hot: Vec<Vec<Option<usize>>>,
    rows: Vec<CommittedColumnsWitness>,
}

impl CollectedColumns {
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
        let hot = kinds
            .iter()
            .map(|kind| {
                if kind.is_one_hot() {
                    Vec::with_capacity(cycles)
                } else {
                    Vec::new()
                }
            })
            .collect();
        Self {
            kinds: kinds.to_vec(),
            increments,
            hot,
            rows: Vec::with_capacity(cycles),
        }
    }
}

impl StreamConsumer for CollectedColumns {
    type Witness = CommittedColumnsWitness;

    fn consume(&mut self, chunk: &[CommittedColumnsWitness]) {
        self.rows.extend_from_slice(chunk);
        for (index, kind) in self.kinds.iter().copied().enumerate() {
            if kind.is_one_hot() {
                self.hot[index].extend(chunk.iter().map(|row| kind.hot_address(row)));
            } else {
                self.increments[index].extend(chunk.iter().map(|row| kind.increment(row)));
            }
        }
    }
}

struct ResidentBases {
    count: usize,
    bases: DeviceG1Bases,
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
        source: &dyn RowSource,
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
            let mut consumers = (CollectedColumns::begin(&kinds, cycles),);
            stream_witnesses(source, 0..cycles, row_width, &mut consumers)?;
            let mut collected = consumers.0;
            let rows = std::mem::take(&mut collected.rows);
            tracing::info_span!("cuda_commit_store_columns", cycles)
                .in_scope(|| store_columns(session, source, cycles, &rows));
            drop(rows);
            let bases = tracing::info_span!("cuda_commit_bases", bases = row_width)
                .in_scope(|| device_bases::<F, PCS>(session, context, setup, row_width))?;
            return kinds
                .iter()
                .zip(ids)
                .enumerate()
                .map(|(index, (kind, &id))| {
                    let rows = if kind.is_one_hot() {
                        tracing::info_span!("cuda_commit_tier1_one_hot").in_scope(|| {
                            context.one_hot_rows(bases, &collected.hot[index], one_hot_k, row_width)
                        })?
                    } else {
                        tracing::info_span!("cuda_commit_tier1_dense").in_scope(|| {
                            context.msm_rows_i128(bases, &collected.increments[index], row_width)
                        })?
                    };
                    tracing::info_span!("cuda_commit_tier2")
                        .in_scope(|| finish::<F, PCS>(setup, &rows, id))
                })
                .collect();
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
    use jolt_claims::protocols::jolt::{
        JoltAdviceKind, JoltCommittedPolynomial, JoltOneHotConfig, TracePolynomialOrder,
    };
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_openings::CommitmentScheme;
    use jolt_witness::{JoltWitnessOracle, RowSource};

    use super::CudaBackend;
    use crate::commitment::{CommitWitness, CommitmentGrid, WitnessCommitment};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{advice_plane, with_r1cs_witness};
    use crate::reference::ReferenceBackend;
    use crate::ProofSession;

    const RAM_K: usize = 1 << 10;

    const LOG_K_CHUNK: usize = 8;

    const TOTAL_VARS: usize = 16;

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

    fn commit_columns(
        backend: &dyn CommitWitness<Fr, DoryScheme>,
        witness: &(impl JoltWitnessOracle<Fr> + RowSource),
        grid: CommitmentGrid,
    ) -> Vec<WitnessCommitment<DoryScheme>> {
        let ids = trace_ids(witness);
        let setup = DoryScheme::setup_prover(grid.total_vars);
        backend
            .commit_witness(
                &mut ProofSession::default(),
                witness as &dyn RowSource,
                &ids,
                grid,
                &setup,
            )
            .expect("commit_witness")
    }

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
