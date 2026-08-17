use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::claim_reductions::advice::ram_val_check_advice_opening;
use jolt_claims::protocols::jolt::geometry::committed_openings::final_opening_id;
use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltCommittedPolynomial, TracePolynomialOrder};
use jolt_field::Field;
use jolt_poly::MultilinearPoly;
use jolt_witness::JoltWitnessOracle;

use super::common::context::CudaKernelContext;
use super::common::device::{fr_into, fr_vec_into, require_fr_slice, DeviceFrVec};
use super::common::error::CudaError;
use super::common::trace_columns::{cached_columns, witness_identity, CachedBundle};
use super::{require_context, CudaBackend};
use crate::commitment::{CommitmentGrid, CommittedColumnsWitness};
use crate::opening::{AdviceOpeningEvaluation, JointOpeningPolynomials};
use crate::reference::commitment::column_kinds;
use crate::reference::views::dense_view;
use crate::{KernelError, ProofSession};

pub(crate) const NO_HOT: u32 = u32::MAX;

enum EmbedPlan {
    ZeroExtend,
    Block { block_vars: usize },
    AddressMajorDense,
    AddressMajorOneHot { cycles: usize },
}

enum JointSource<F> {
    Dense { table: Vec<F>, plan: EmbedPlan },
    SparseOneHot { hot: Vec<u32> },
}

struct JointPolynomial<F> {
    context: &'static CudaKernelContext,
    source: JointSource<F>,
    grid: CommitmentGrid,
}

impl<F: Field> JointPolynomial<F> {
    fn embed(&self) -> Result<DeviceFrVec, CudaError> {
        let _span =
            tracing::info_span!("cuda_opening_embed", vars = self.grid.total_vars).entered();
        let domain = 1usize << self.grid.total_vars;
        let (table, plan) = match &self.source {
            JointSource::SparseOneHot { hot } => return self.context.one_hot_embed(hot, domain),
            JointSource::Dense { table, plan } => (table, plan),
        };
        let source = self.context.upload(require_fr_slice(table)?)?;
        match *plan {
            EmbedPlan::ZeroExtend if source.len() == domain => Ok(source),
            EmbedPlan::ZeroExtend => self.context.zero_extend(&source, domain),
            EmbedPlan::Block { block_vars } => {
                self.context
                    .block_embed(&source, block_vars, self.grid.total_vars)
            }
            EmbedPlan::AddressMajorDense => {
                self.context
                    .scatter_strided(&source, self.grid.cycle_stride(), domain)
            }
            EmbedPlan::AddressMajorOneHot { cycles } => self.context.scatter_one_hot(
                &source,
                cycles,
                self.grid.cycle_stride(),
                self.grid.one_hot_stride(),
                domain,
            ),
        }
    }

    fn evaluate_device(&self, point: &[F]) -> Result<F, CudaError> {
        let embedded = self.embed()?;
        let eq = self.context.eq_evals(require_fr_slice(point)?)?;
        let products = self.context.mul(&embedded, &eq)?;
        let value = self.context.sum(&products)?;
        fr_into(value).ok_or(CudaError::NotImplemented {
            kernel: "CUDA kernels support only the BN254 scalar field",
        })
    }

    fn fold_rows_device(&self, left: &[F], sigma: usize) -> Result<Vec<F>, CudaError> {
        if let JointSource::SparseOneHot { hot } = &self.source {
            if sigma <= self.grid.log_t {
                let folded =
                    tracing::info_span!("cuda_opening_sparse_fold", sigma, cycles = hot.len())
                        .in_scope(|| {
                            self.context
                                .one_hot_fold(hot, require_fr_slice(left)?, sigma)
                        })?;
                return fr_vec_into(folded).ok_or(CudaError::NotImplemented {
                    kernel: "CUDA kernels support only the BN254 scalar field",
                });
            }
        }
        let embedded = self.embed()?;
        let folded = tracing::info_span!("cuda_opening_fold", sigma).in_scope(|| {
            self.context
                .fold_rows(&embedded, require_fr_slice(left)?, sigma)
        })?;
        fr_vec_into(folded).ok_or(CudaError::NotImplemented {
            kernel: "CUDA kernels support only the BN254 scalar field",
        })
    }

    fn dense_device(&self) -> Result<Vec<F>, CudaError> {
        let embedded = self.embed()?;
        fr_vec_into(embedded.to_host()?).ok_or(CudaError::NotImplemented {
            kernel: "CUDA kernels support only the BN254 scalar field",
        })
    }
}

impl<F: Field> MultilinearPoly<F> for JointPolynomial<F> {
    fn num_vars(&self) -> usize {
        self.grid.total_vars
    }

    fn evaluate(&self, point: &[F]) -> F {
        self.evaluate_device(point).unwrap_or_default()
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[F])) {
        let Ok(table) = self.dense_device() else {
            return;
        };
        for (index, row) in table.chunks(1usize << sigma).enumerate() {
            f(index, row);
        }
    }

    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F> {
        self.fold_rows_device(left, sigma).unwrap_or_default()
    }
}

fn embed_plan<F: Field>(
    polynomial: JoltCommittedPolynomial,
    table: &[F],
    grid: CommitmentGrid,
) -> Result<EmbedPlan, KernelError<F>> {
    let domain = 1usize << grid.total_vars;
    if table.len() > domain {
        return Err(KernelError::TableSizeMismatch {
            table: format!("{polynomial:?}"),
            expected: domain,
            got: table.len(),
        });
    }
    let cycles = 1usize << grid.log_t;
    match polynomial {
        JoltCommittedPolynomial::TrustedAdvice
        | JoltCommittedPolynomial::UntrustedAdvice
        | JoltCommittedPolynomial::BytecodeChunk(_)
        | JoltCommittedPolynomial::ProgramImageInit => {
            if !table.len().is_power_of_two() {
                return Err(KernelError::TableSizeMismatch {
                    table: format!("{polynomial:?}"),
                    expected: table.len().next_power_of_two(),
                    got: table.len(),
                });
            }
            Ok(EmbedPlan::Block {
                block_vars: table.len().ilog2() as usize,
            })
        }
        _ if grid.order != TracePolynomialOrder::AddressMajor => Ok(EmbedPlan::ZeroExtend),
        JoltCommittedPolynomial::RdInc | JoltCommittedPolynomial::RamInc => {
            if table.len() > cycles {
                return Err(KernelError::TableSizeMismatch {
                    table: format!("{polynomial:?}"),
                    expected: cycles,
                    got: table.len(),
                });
            }
            Ok(EmbedPlan::AddressMajorDense)
        }
        JoltCommittedPolynomial::InstructionRa(_)
        | JoltCommittedPolynomial::BytecodeRa(_)
        | JoltCommittedPolynomial::RamRa(_) => {
            let max_k = 1usize << grid.log_k_chunk;
            if !table.len().is_multiple_of(cycles) || table.len() / cycles > max_k {
                return Err(KernelError::InvalidGeometry {
                    reason: format!(
                        "one-hot table for {polynomial:?} ({} entries) is not a (K × {cycles}) \
                         grid with K at most {max_k}",
                        table.len()
                    ),
                });
            }
            Ok(EmbedPlan::AddressMajorOneHot { cycles })
        }
        _ => Err(KernelError::InvariantViolation {
            reason: "only trace polynomials embed address-major",
        }),
    }
}

fn is_trace_one_hot(polynomial: JoltCommittedPolynomial) -> bool {
    matches!(
        polynomial,
        JoltCommittedPolynomial::InstructionRa(_)
            | JoltCommittedPolynomial::BytecodeRa(_)
            | JoltCommittedPolynomial::RamRa(_)
    )
}

fn is_trace_derived(polynomial: JoltCommittedPolynomial) -> bool {
    is_trace_one_hot(polynomial)
        || matches!(
            polynomial,
            JoltCommittedPolynomial::RdInc | JoltCommittedPolynomial::RamInc
        )
}

pub(crate) fn sparse_hot_columns<F: Field>(
    session: &ProofSession,
    identity: usize,
    polynomials: &[JoltCommittedPolynomial],
    grid: CommitmentGrid,
) -> Result<BTreeMap<JoltCommittedPolynomial, Vec<u32>>, KernelError<F>> {
    let cycles = 1usize << grid.log_t;
    if grid.order != TracePolynomialOrder::CycleMajor {
        return Ok(BTreeMap::new());
    }
    let one_hot_k = 1usize << grid.log_k_chunk;
    if one_hot_k
        .checked_mul(cycles)
        .is_none_or(|span| span > 1usize << grid.total_vars)
        || one_hot_k > NO_HOT as usize
    {
        return Ok(BTreeMap::new());
    }
    let Some(columns) = cached_columns(session, identity, cycles) else {
        return Ok(BTreeMap::new());
    };
    let Some(rows) = CommittedColumnsWitness::restore(columns, cycles) else {
        return Ok(BTreeMap::new());
    };
    let trace_ids: Vec<JoltCommittedPolynomial> = polynomials
        .iter()
        .copied()
        .filter(|&id| is_trace_derived(id))
        .collect();
    let kinds = column_kinds::<F>(&trace_ids, grid)?;
    let mut sparse = BTreeMap::new();
    for (&id, &kind) in trace_ids.iter().zip(&kinds) {
        if !kind.is_one_hot() {
            continue;
        }
        let mut hot = Vec::with_capacity(cycles);
        for row in &rows {
            hot.push(
                kind.hot_address(row)
                    .map_or(NO_HOT, |address| address as u32),
            );
        }
        let _ = sparse.insert(id, hot);
    }
    Ok(sparse)
}

impl<F: Field> JointOpeningPolynomials<F> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessOracle<F>,
        polynomials: &[JoltCommittedPolynomial],
        precommitted_tables: &BTreeMap<JoltCommittedPolynomial, Vec<F>>,
        grid: CommitmentGrid,
    ) -> Result<Vec<Box<dyn MultilinearPoly<F>>>, KernelError<F>> {
        let context = require_context::<F>()?;
        let identity = witness_identity(witness);
        let mut sparse = tracing::info_span!("cuda_opening_sparse_plan")
            .in_scope(|| sparse_hot_columns::<F>(session, identity, polynomials, grid))?;
        polynomials
            .iter()
            .map(|&polynomial| {
                let source = if let Some(hot) = sparse.remove(&polynomial) {
                    JointSource::SparseOneHot { hot }
                } else {
                    let table = match precommitted_tables.get(&polynomial) {
                        Some(table) => table.clone(),
                        None => dense_view(witness, final_opening_id(polynomial))?,
                    };
                    let plan = embed_plan(polynomial, &table, grid)?;
                    JointSource::Dense { table, plan }
                };
                Ok(Box::new(JointPolynomial {
                    context,
                    source,
                    grid,
                }) as Box<dyn MultilinearPoly<F>>)
            })
            .collect()
    }
}

impl<F: Field> AdviceOpeningEvaluation<F> for CudaBackend {
    fn evaluate(
        &self,
        _session: &mut ProofSession,
        kind: JoltAdviceKind,
        point: &[F],
        witness: &dyn JoltWitnessOracle<F>,
    ) -> Result<F, KernelError<F>> {
        let context = require_context::<F>()?;
        let table = dense_view(witness, ram_val_check_advice_opening(kind))?;
        if table.len() != 1usize << point.len() {
            return Err(KernelError::TableSizeMismatch {
                table: format!("{kind:?} advice"),
                expected: 1usize << point.len(),
                got: table.len(),
            });
        }
        let device = context.upload(require_fr_slice(&table)?)?;
        let eq = context.eq_evals(require_fr_slice(point)?)?;
        let products = context.mul(&device, &eq)?;
        let value = context.sum(&products)?;
        fr_into(value).ok_or(KernelError::Unsupported {
            reason: "CUDA kernels support only the BN254 scalar field",
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use std::collections::BTreeMap;

    use jolt_claims::protocols::jolt::{
        JoltAdviceKind, JoltCommittedPolynomial, JoltOneHotConfig, TracePolynomialOrder,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::MultilinearPoly;
    use jolt_program::execution::OwnedTrace;
    use jolt_witness::{JoltWitnessOracle, TraceBackend};
    use proptest::prelude::*;

    use super::{CudaBackend, NO_HOT};
    use crate::commitment::{CommitmentGrid, CommittedColumnsWitness};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{advice_plane, fr, with_r1cs_witness};
    use crate::cuda::common::trace_columns::cached_bundles;
    use crate::opening::{AdviceOpeningEvaluation, JointOpeningPolynomials};
    use crate::reference::ReferenceBackend;
    use crate::ProofSession;

    const RAM_K: usize = 1 << 10;

    const LOG_K_CHUNK: usize = 8;

    const TOTAL_VARS: usize = 16;

    const ADVICE_BYTES: usize = 4096;

    const CONFIGS: [(TracePolynomialOrder, usize); 3] = [
        (TracePolynomialOrder::CycleMajor, 8),
        (TracePolynomialOrder::AddressMajor, 8),
        (TracePolynomialOrder::AddressMajor, 6),
    ];

    const PRECOMMITTED_VARS: [(JoltCommittedPolynomial, usize); 4] = [
        (JoltCommittedPolynomial::TrustedAdvice, 9),
        (JoltCommittedPolynomial::UntrustedAdvice, 10),
        (JoltCommittedPolynomial::BytecodeChunk(0), 12),
        (JoltCommittedPolynomial::ProgramImageInit, 11),
    ];

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

    const ADVICE_KINDS: [JoltAdviceKind; 2] = [JoltAdviceKind::Trusted, JoltAdviceKind::Untrusted];

    struct OpeningProbe {
        grid: CommitmentGrid,
        sigma: usize,
        left: Vec<Fr>,
        point: Vec<Fr>,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct PolynomialFacts {
        id: JoltCommittedPolynomial,
        num_vars: usize,
        dense: Vec<Fr>,
        folded: Vec<Fr>,
        evaluation: Fr,
    }

    fn probe(order: TracePolynomialOrder, log_t: usize, seed: u64) -> OpeningProbe {
        let grid = grid_at(order, log_t);
        let sigma = grid.total_vars.div_ceil(2);
        OpeningProbe {
            grid,
            sigma,
            left: (0..1usize << (grid.total_vars - sigma))
                .map(|index| fr(seed.wrapping_add(index as u64 + 7)))
                .collect(),
            point: (0..grid.total_vars)
                .map(|index| fr(seed ^ (index as u64).wrapping_mul(31).wrapping_add(3)))
                .collect(),
        }
    }

    fn polynomial_facts(
        backend: &dyn JointOpeningPolynomials<Fr>,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessOracle<Fr>,
        tables: &BTreeMap<JoltCommittedPolynomial, Vec<Fr>>,
        probe: &OpeningProbe,
    ) -> Vec<PolynomialFacts> {
        let order_ids = batch_order(witness, tables);
        let polynomials = backend
            .prepare(session, witness, &order_ids, tables, probe.grid)
            .expect("joint opening polynomials");
        polynomials
            .iter()
            .zip(&order_ids)
            .map(|(polynomial, &id)| PolynomialFacts {
                id,
                num_vars: polynomial.num_vars(),
                dense: polynomial.to_dense().to_vec(),
                folded: polynomial.fold_rows(&probe.left, probe.sigma),
                evaluation: polynomial.evaluate(&probe.point),
            })
            .collect()
    }

    fn warm_session(witness: &TraceBackend<'_, OwnedTrace>, log_t: usize) -> ProofSession {
        let mut session = ProofSession::default();
        let _ =
            cached_bundles::<CommittedColumnsWitness, _>(&mut session, witness, 1usize << log_t)
                .expect("the fixture serves the committed columns bundle");
        session
    }

    fn one_hot_ids(
        witness: &dyn JoltWitnessOracle<Fr>,
        tables: &BTreeMap<JoltCommittedPolynomial, Vec<Fr>>,
    ) -> Vec<JoltCommittedPolynomial> {
        batch_order(witness, tables)
            .into_iter()
            .filter(|&id| super::is_trace_one_hot(id))
            .collect()
    }

    const FOLD_GEOMETRIES: [(usize, usize, usize); 4] =
        [(8, 16, 8), (10, 18, 8), (12, 18, 6), (6, 16, 4)];

    fn synthetic_hot(cycles: usize, one_hot_k: usize, seed: u64) -> Vec<u32> {
        (0..cycles)
            .map(|cycle| {
                let mixed = (cycle as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(seed);
                if mixed.is_multiple_of(7) {
                    NO_HOT
                } else {
                    (mixed >> 13) as usize as u32 % one_hot_k as u32
                }
            })
            .collect()
    }

    fn dense_one_hot_grid(hot: &[u32], cycles: usize, domain: usize) -> Vec<Fr> {
        let mut table = vec![Fr::from_u64(0); domain];
        for (cycle, &address) in hot.iter().enumerate() {
            if address != NO_HOT {
                table[address as usize * cycles + cycle] = Fr::from_u64(1);
            }
        }
        table
    }

    #[test]
    fn one_hot_fold_matches_the_dense_fold() {
        let Some(context) = shared_context() else {
            return;
        };
        for (index, &(log_t, total_vars, log_k_chunk)) in FOLD_GEOMETRIES.iter().enumerate() {
            let cycles = 1usize << log_t;
            let one_hot_k = 1usize << log_k_chunk;
            let domain = 1usize << total_vars;
            assert!(
                one_hot_k * cycles <= domain,
                "geometry {index}: the one-hot span exceeds the grid",
            );
            let sigma = total_vars.div_ceil(2);
            let rows = domain >> sigma;
            let hot = synthetic_hot(cycles, one_hot_k, 17 + index as u64);
            assert!(
                hot.contains(&NO_HOT) && hot.iter().any(|&address| address != NO_HOT),
                "geometry {index}: the synthetic column is all hot or all cold",
            );
            let left: Vec<Fr> = (0..rows)
                .map(|row| fr((index as u64 + 1) * 1_000 + row as u64 + 3))
                .collect();

            let table = dense_one_hot_grid(&hot, cycles, domain);
            let device_table = context.upload(&table).expect("upload dense grid");
            let expected = context
                .fold_rows(&device_table, &left, sigma)
                .expect("dense fold");
            if sigma > log_t {
                assert!(
                    context.one_hot_fold(&hot, &left, sigma).is_err(),
                    "geometry {index} (sigma {sigma} > log_T {log_t}): the sparse fold accepted a \
                     geometry whose column index depends on the address, where its \
                     thread-per-column scheme does not hold",
                );
            } else {
                let got = context
                    .one_hot_fold(&hot, &left, sigma)
                    .expect("sparse fold");
                assert_eq!(
                    got, expected,
                    "geometry {index} (log_T {log_t}, total_vars {total_vars}, log_k \
                     {log_k_chunk}, sigma {sigma}): the sparse fold diverged from the dense fold",
                );
            }

            let embedded = context
                .one_hot_embed(&hot, domain)
                .expect("sparse embed")
                .to_host()
                .expect("download embedded grid");
            assert_eq!(
                embedded, table,
                "geometry {index}: the sparse embed diverged from the dense grid",
            );
        }
    }

    #[test]
    fn fixture_fold_geometries_cover_the_production_regime() {
        let strict = FOLD_GEOMETRIES
            .iter()
            .filter(|&&(log_t, total_vars, _)| total_vars.div_ceil(2) < log_t)
            .count();
        assert!(
            strict >= 2,
            "fewer than two geometries have sigma < log_T, so the multi-iteration fold loop that \
             production runs (sigma 13, log_T 22) is barely covered",
        );
        assert!(
            FOLD_GEOMETRIES
                .iter()
                .any(|&(log_t, total_vars, _)| total_vars.div_ceil(2) == log_t),
            "no geometry has sigma == log_T, the single-iteration boundary case",
        );
        assert!(
            FOLD_GEOMETRIES
                .iter()
                .any(|&(log_t, total_vars, _)| total_vars.div_ceil(2) > log_t),
            "no geometry has sigma > log_T, where the fold must fall back to the dense path",
        );
    }

    #[test]
    fn fixture_cycle_major_one_hot_takes_the_sparse_plan() {
        for &(order, log_t) in &CONFIGS {
            let grid = grid_at(order, log_t);
            let tables = precommitted_tables(3);
            with_r1cs_witness(log_t, RAM_K, one_hot(), 3, |witness| {
                let order_ids = batch_order(witness, &tables);
                let expected_one_hot = one_hot_ids(witness, &tables);
                assert!(
                    !expected_one_hot.is_empty(),
                    "{order:?} log_T {log_t}: the fixture commits no one-hot polynomial, so the \
                     sparse plan is untestable",
                );

                let cold = ProofSession::default();
                let identity = super::witness_identity(witness as &dyn JoltWitnessOracle<Fr>);
                let sparse_cold =
                    super::sparse_hot_columns::<Fr>(&cold, identity, &order_ids, grid)
                        .expect("cold sparse plan");
                assert!(
                    sparse_cold.is_empty(),
                    "{order:?} log_T {log_t}: a cold session produced a sparse plan, so the plan \
                     does not depend on residency at all",
                );

                let warm = warm_session(witness, log_t);
                let sparse_warm =
                    super::sparse_hot_columns::<Fr>(&warm, identity, &order_ids, grid)
                        .expect("warm sparse plan");
                if order == TracePolynomialOrder::CycleMajor {
                    let mut covered = sparse_warm.keys().copied().collect::<Vec<_>>();
                    let mut wanted = expected_one_hot.clone();
                    covered.sort_unstable();
                    wanted.sort_unstable();
                    assert_eq!(
                        covered, wanted,
                        "{order:?} log_T {log_t}: the sparse plan does not cover exactly the \
                         one-hot polynomials",
                    );
                    for (id, hot) in &sparse_warm {
                        assert_eq!(
                            hot.len(),
                            1usize << log_t,
                            "{id:?}: the hot column is not one entry per cycle",
                        );
                        assert!(
                            hot.iter().any(|&address| address != NO_HOT),
                            "{id:?}: every cycle is cold, so a fold that ignored the column \
                             would pass",
                        );
                        assert!(
                            hot.iter().any(|&address| address != hot[0]),
                            "{id:?}: every cycle shares one address, so a fold that used a \
                             constant address would pass",
                        );
                    }
                    assert!(
                        sparse_warm.values().any(|hot| hot.contains(&NO_HOT)),
                        "no one-hot column has a cold cycle, so the fold's cold-cycle handling \
                         is untested by this fixture",
                    );
                } else {
                    assert!(
                        sparse_warm.is_empty(),
                        "{order:?} log_T {log_t}: address-major took the sparse plan, whose \
                         index arithmetic only holds for cycle-major grids",
                    );
                }
            });
        }
    }

    #[test]
    fn warm_session_identity_matches_across_the_witness_traits() {
        let log_t = 8;
        with_r1cs_witness(log_t, RAM_K, one_hot(), 5, |witness| {
            let as_oracle = super::witness_identity(witness as &dyn JoltWitnessOracle<Fr>);
            let as_source = super::witness_identity(witness);
            assert_eq!(
                as_oracle, as_source,
                "the oracle and concrete views of one witness have different identities, so \
                 stage 8 could never read what commit stored",
            );
        });
    }

    fn advice_opening(
        backend: &dyn AdviceOpeningEvaluation<Fr>,
        witness: &dyn JoltWitnessOracle<Fr>,
        kind: JoltAdviceKind,
        point: &[Fr],
    ) -> Fr {
        backend
            .evaluate(&mut ProofSession::default(), kind, point, witness)
            .expect("advice opening")
    }

    fn precommitted_tables(seed: u64) -> BTreeMap<JoltCommittedPolynomial, Vec<Fr>> {
        PRECOMMITTED_VARS
            .iter()
            .map(|&(id, vars)| {
                let salt = seed ^ (vars as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let table = (0..1usize << vars)
                    .map(|index| fr(salt.wrapping_add(index as u64 + 1)))
                    .collect();
                (id, table)
            })
            .collect()
    }

    fn batch_order(
        witness: &dyn JoltWitnessOracle<Fr>,
        tables: &BTreeMap<JoltCommittedPolynomial, Vec<Fr>>,
    ) -> Vec<JoltCommittedPolynomial> {
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
            .chain(tables.keys().copied())
            .collect()
    }

    #[test]
    fn fixture_joint_opening_tables_discriminate() {
        let grid = grid_at(TracePolynomialOrder::AddressMajor, 8);
        let domain = 1usize << grid.total_vars;
        let tables = precommitted_tables(5);
        assert!(
            tables.values().all(|table| table.len() < domain),
            "a precommitted table fills the whole grid, so the block embedding is a no-op",
        );
        assert!(
            PRECOMMITTED_VARS.iter().any(|&(_, vars)| vars % 2 == 1),
            "no precommitted table has an odd variable count, so a wrong column-split \
             rounding would pass",
        );
        assert!(
            PRECOMMITTED_VARS.iter().any(|&(_, vars)| vars % 2 == 0),
            "no precommitted table has an even variable count",
        );
        for (id, table) in &tables {
            let varying = table.windows(2).filter(|pair| pair[0] != pair[1]).count();
            assert!(
                varying > table.len() / 2,
                "{id:?}: only {varying} adjacent coefficients differ, so a wrong placement \
                 could pass",
            );
        }
        assert!(
            CONFIGS
                .iter()
                .any(|&(order, _)| order == TracePolynomialOrder::CycleMajor),
            "no cycle-major config, so the zero-extending trace embedding is untested",
        );
        assert!(
            CONFIGS.iter().any(
                |&(order, log_t)| order == TracePolynomialOrder::AddressMajor
                    && grid_at(order, log_t).one_hot_stride() > 1
            ),
            "no widened address-major config, so a kernel that ignored `one_hot_stride` \
             would pass",
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2))]

        #[test]
        fn joint_opening_matches_reference(seed in any::<u64>()) {
            let Some(_) = shared_context() else { return Ok(()); };
            let tables = precommitted_tables(seed);
            let expected: Vec<Vec<PolynomialFacts>> = CONFIGS
                .iter()
                .map(|&(order, log_t)| {
                    let probe = probe(order, log_t, seed);
                    with_r1cs_witness(log_t, RAM_K, one_hot(), seed, |witness| {
                        polynomial_facts(
                            &ReferenceBackend,
                            &mut ProofSession::default(),
                            witness,
                            &tables,
                            &probe,
                        )
                    })
                })
                .collect();
            for (&(order, log_t), expected) in CONFIGS.iter().zip(&expected) {
                let probe = probe(order, log_t, seed);
                let got = with_r1cs_witness(log_t, RAM_K, one_hot(), seed, |witness| {
                    let mut session = warm_session(witness, log_t);
                    polynomial_facts(&CudaBackend, &mut session, witness, &tables, &probe)
                });
                prop_assert_eq!(
                    got.len(),
                    expected.len(),
                    "{:?} log_T {}: polynomial count diverged",
                    order,
                    log_t
                );
                for (expected, got) in expected.iter().zip(&got) {
                    prop_assert_eq!(
                        got,
                        expected,
                        "{:?} log_T {}: {:?} diverged",
                        order,
                        log_t,
                        expected.id
                    );
                }
            }
        }

        #[test]
        fn advice_opening_matches_reference(seed in any::<u64>()) {
            let Some(_) = shared_context() else { return Ok(()); };
            let fixture = advice_plane(ADVICE_BYTES, seed);
            let vars = fixture.trusted.len().ilog2() as usize;
            let witness = &fixture.plane as &dyn JoltWitnessOracle<Fr>;
            let point: Vec<Fr> = (0..vars)
                .map(|index| fr(seed ^ (index as u64).wrapping_mul(17).wrapping_add(5)))
                .collect();
            let expected: Vec<Fr> = ADVICE_KINDS
                .iter()
                .map(|&kind| advice_opening(&ReferenceBackend, witness, kind, &point))
                .collect();
            for (&kind, &expected) in ADVICE_KINDS.iter().zip(&expected) {
                let got = advice_opening(&CudaBackend, witness, kind, &point);
                prop_assert_eq!(got, expected, "{:?} advice opening diverged", kind);
            }
        }
    }

    #[test]
    fn fixture_advice_columns_discriminate() {
        let fixture = advice_plane(ADVICE_BYTES, 41);
        assert_eq!(
            fixture.trusted.len(),
            fixture.untrusted.len(),
            "the two advice columns must share a width for one point to serve both",
        );
        assert_ne!(
            fixture.trusted, fixture.untrusted,
            "the two advice columns are identical, so a kernel that ignored the kind would pass",
        );
        assert!(
            fixture
                .trusted
                .iter()
                .any(|value| *value != Fr::from_u64(0)),
            "the trusted advice column is all zeros",
        );
        assert!(
            fixture.trusted.len().is_power_of_two() && fixture.trusted.len() > 1,
            "the advice column must span at least one variable",
        );
    }
}
