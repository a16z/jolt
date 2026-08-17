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
use super::{require_context, CudaBackend};
use crate::commitment::CommitmentGrid;
use crate::opening::{AdviceOpeningEvaluation, JointOpeningPolynomials};
use crate::reference::views::dense_view;
use crate::{KernelError, ProofSession};

enum EmbedPlan {
    ZeroExtend,
    Block { block_vars: usize },
    AddressMajorDense,
    AddressMajorOneHot { cycles: usize },
}

struct JointPolynomial<F> {
    context: &'static CudaKernelContext,
    source: Vec<F>,
    plan: EmbedPlan,
    grid: CommitmentGrid,
}

impl<F: Field> JointPolynomial<F> {
    fn embed(&self) -> Result<DeviceFrVec, CudaError> {
        let _span =
            tracing::info_span!("cuda_opening_embed", vars = self.grid.total_vars).entered();
        let domain = 1usize << self.grid.total_vars;
        let source = self.context.upload(require_fr_slice(&self.source)?)?;
        match self.plan {
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

impl<F: Field> JointOpeningPolynomials<F> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessOracle<F>,
        polynomials: &[JoltCommittedPolynomial],
        precommitted_tables: &BTreeMap<JoltCommittedPolynomial, Vec<F>>,
        grid: CommitmentGrid,
    ) -> Result<Vec<Box<dyn MultilinearPoly<F>>>, KernelError<F>> {
        let context = require_context::<F>()?;
        polynomials
            .iter()
            .map(|&polynomial| {
                let source = match precommitted_tables.get(&polynomial) {
                    Some(table) => table.clone(),
                    None => dense_view(witness, final_opening_id(polynomial))?,
                };
                let plan = embed_plan(polynomial, &source, grid)?;
                Ok(Box::new(JointPolynomial {
                    context,
                    source,
                    plan,
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
    use jolt_witness::JoltWitnessOracle;
    use proptest::prelude::*;

    use super::CudaBackend;
    use crate::commitment::CommitmentGrid;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{advice_plane, fr, with_r1cs_witness};
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
        witness: &dyn JoltWitnessOracle<Fr>,
        tables: &BTreeMap<JoltCommittedPolynomial, Vec<Fr>>,
        probe: &OpeningProbe,
    ) -> Vec<PolynomialFacts> {
        let order_ids = batch_order(witness, tables);
        let polynomials = backend
            .prepare(
                &mut ProofSession::default(),
                witness,
                &order_ids,
                tables,
                probe.grid,
            )
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
                        polynomial_facts(&ReferenceBackend, witness, &tables, &probe)
                    })
                })
                .collect();
            for (&(order, log_t), expected) in CONFIGS.iter().zip(&expected) {
                let probe = probe(order, log_t, seed);
                let got = with_r1cs_witness(log_t, RAM_K, one_hot(), seed, |witness| {
                    polynomial_facts(&CudaBackend, witness, &tables, &probe)
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
