//! Streaming registers value evaluation (stage 5): the default kernel's
//! summand, with the increment and write-address columns read from the
//! witness's resident rows instead of the `rd` indices stage 4 parks. While
//! the two dense tables exceed [`DENSE_TABLE_CAP`], round `j` contracts each
//! column over the low and high halves of its `2^(j+1)`-cycle pairs with
//! `eq(r_1..r_j, ·)` (LowToHigh binding gives
//! `f^(j)(y) = Σ_b eq(r, b)·f(2^j·y + b)`), with
//! `wa(t) = eq(r_address)[rd_t]` served from the `K`-entry eq table. The
//! first round whose tables fit fills them in the pass that yields its
//! evaluations, and later rounds bind them in place. `LT` and the
//! `{0, 2, 3}` sampling are the default kernel's.
//!
//! The passes and the tables cover the trace's prefix, as the default
//! kernel's do: rows from the witness's constant suffix on repeat one row
//! that writes no register, so their summands vanish, a pass visits only the
//! pairs that hold a prefix row, and the tables keep the suffix as one tail
//! value ([`PrefixColumn`]).

use std::ops::Range;

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::{JoltDerivedId, JoltPolynomialId, RegistersValEvaluationPublic};
use jolt_field::{Accumulator, JoltField};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage5::registers_val_evaluation::{
    RegistersValEvaluation, RegistersValEvaluationOutputClaims,
};
use jolt_witness::__private::TraceRow;
use jolt_witness::witnesses::{Extract, RdInc, ToField, WitnessEnv};
use jolt_witness::{JoltWitnessPlane, WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::optimized::prefix_product::PrefixColumn;
use crate::optimized::registers_read_write::RegisterCycleRow;
use crate::optimized::streaming::{
    dense_crossover, split_at_tasks, task_ranges, PairHalves, DENSE_TABLE_CAP, VISIT_ROWS,
};
use crate::optimized::support::{
    accumulate_triple_product, pin_derived_term, triple_product_round_evals, BundleStore,
    RoundChallenges, SplitLt,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

pub struct StreamingRegistersValEvaluation;

/// The round, of `rounds`, whose pass fills the two dense tables over
/// `prefix_len` cycles: the first whose tables fit `cap` bytes, at least
/// round 1 when there are two or more rounds, so each table never holds more
/// than every second prefix cycle, and at most the last round, so openings
/// are always table entries. Sizes are compared in cycles against the cycles
/// `cap` holds, which is the same test as bytes against `cap` and cannot
/// overflow where the byte count of the prefix would.
fn crossover_round<F: JoltField>(rounds: usize, prefix_len: usize, cap: usize) -> usize {
    let cap_cycles = cap / (2 * size_of::<F>());
    dense_crossover(prefix_len, cap_cycles, rounds)
        .max(1)
        .min(rounds.saturating_sub(1))
}

impl<F: JoltField> PrepareKernel<F, RegistersValEvaluation<F>> for StreamingRegistersValEvaluation {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersValEvaluation<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersValEvaluation<F>>>, KernelError<F>>
    {
        let rows = BundleStore::resident(
            witness,
            inputs.relation.trace_dimensions().log_t(),
            ROWS_LABEL,
        )?;
        Ok(Box::new(StreamingValEvaluationKernel::new(
            rows,
            &inputs,
            DENSE_TABLE_CAP,
        )?))
    }
}

const ROWS_LABEL: &str = "registers value-evaluation trace rows";

/// One cycle's rd write: the canonical committed increment and the written
/// register, range-checked with the cycle's other operands by
/// [`RegisterCycleRow`].
#[derive(Clone, Copy, Debug, PartialEq)]
struct RdWriteRow {
    rd_inc: RdInc,
    rd: Option<u8>,
}

impl WitnessBundle for RdWriteRow {
    fn from_row(
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self {
            rd_inc: RdInc::extract(row, next, env)?,
            rd: RegisterCycleRow::from_row(row, next, env)?
                .rd
                .map(|(register, ..)| register),
        })
    }

    fn annotated_ids() -> Vec<JoltPolynomialId> {
        Vec::new()
    }
}

/// The increment and write-address columns over the cycle domain.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum ValueTables<F: JoltField> {
    /// Rounds before the crossover contract both columns from the first
    /// `prefix_len` rows; every later row has increment `tail` and writes no
    /// register.
    Streamed { prefix_len: usize, tail: F },
    /// Filled by the crossover round's pass, bound in place afterwards. Both
    /// hold the prefix's entries; the write-address tail is zero.
    Dense {
        inc: PrefixColumn<F>,
        wa: PrefixColumn<F>,
    },
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct StreamingValEvaluationKernel<F: JoltField> {
    tables: ValueTables<F>,
    /// [`crossover_round`] at the kernel's cap.
    dense_round: usize,
    challenges: RoundChallenges<F>,
    rows: BundleStore<RdWriteRow>,
    eq_address: Vec<F>,
    lt: SplitLt<F>,
}

fn rows_unavailable<F: JoltField>(_: WitnessError) -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource { kind: ROWS_LABEL }
}

impl<F: JoltField> StreamingValEvaluationKernel<F> {
    /// `dense_cap` bounds the bytes of the two dense tables together;
    /// production passes [`DENSE_TABLE_CAP`].
    fn new(
        rows: BundleStore<RdWriteRow>,
        inputs: &ProverInputs<'_, F, RegistersValEvaluation<F>>,
        dense_cap: usize,
    ) -> Result<Self, KernelError<F>> {
        let log_t = inputs.relation.trace_dimensions().log_t();
        if log_t == 0 {
            return Err(KernelError::Unsupported {
                reason: "optimized registers val-evaluation requires at least one cycle round",
            });
        }
        let registers_val_point: &[F] = &inputs.points.registers_val;
        if registers_val_point.len() != REGISTER_ADDRESS_BITS + log_t {
            return Err(KernelError::InvariantViolation {
                reason: "registers value-evaluation input point has the wrong variable count",
            });
        }
        let (r_address, r_cycle) = registers_val_point.split_at(REGISTER_ADDRESS_BITS);
        // `log_t < usize::BITS`: `rows` span `2^log_t` cycles.
        let cycles = 1usize << log_t;
        let access = rows.access();
        let suffix_start = access.constant_suffix_start(cycles)?;
        let suffix = (suffix_start < cycles)
            .then(|| access.row(suffix_start))
            .transpose()?;
        // A constant suffix that writes a register is not skipped: its
        // summands do not vanish. The witness's padding rows write none.
        let (prefix_len, tail) = match suffix {
            Some(RdWriteRow { rd_inc, rd: None }) => (suffix_start, rd_inc.to_field::<F>()),
            _ => (cycles, F::zero()),
        };
        Ok(Self {
            tables: ValueTables::Streamed { prefix_len, tail },
            dense_round: crossover_round::<F>(log_t, prefix_len, dense_cap),
            challenges: RoundChallenges::new(log_t),
            rows,
            eq_address: EqPolynomial::<F>::evals(r_address, None),
            lt: SplitLt::new(r_cycle),
        })
    }

    /// The crossover round's pass, filling the dense tables: each holds the
    /// `⌈prefix_len / 2^round⌉` entries with a prefix row, allocated once at
    /// its final capacity and written in disjoint task slices.
    fn fill_tables(&mut self, prefix_len: usize, tail: F) -> Result<[F; 3], WitnessError> {
        let len = prefix_len.div_ceil(1 << self.challenges.bound());
        let mut inc: Vec<F> = unsafe_allocate_zero_vec(len);
        let mut wa: Vec<F> = unsafe_allocate_zero_vec(len);
        let evals = self.round_pass(prefix_len, Some([&mut inc[..], &mut wa[..]]))?;
        self.tables = ValueTables::Dense {
            inc: PrefixColumn {
                values: inc,
                tail: [tail; 2],
            },
            wa: PrefixColumn {
                values: wa,
                tail: [F::zero(); 2],
            },
        };
        Ok(evals)
    }

    /// The `s(0), s(2), s(3)` samples of the current round `j` from the rows:
    /// the `2^(j+1)`-cycle pairs that hold one of the first `prefix_len` rows
    /// split into [`task_ranges`], each walked once ([`Self::walk_pairs`])
    /// under the weights `eq(r_1..r_j, c)` of the cycle offsets `c` within a
    /// pair half (offset bit `i` pairing `r_{i+1}`). The last pair's rows
    /// past the prefix are read like any other. A filling pass splits
    /// `tables` (pair `z` at entries `2z, 2z + 1`, the last pair's second
    /// entry absent when its half lies past the prefix) at the task
    /// boundaries, so each task writes only its own slices.
    fn round_pass(
        &self,
        prefix_len: usize,
        tables: Option<[&mut [F]; 2]>,
    ) -> Result<[F; 3], WitnessError> {
        let bound = self.challenges.bound();
        let cycles = 1usize << self.challenges.total();
        // The prefix rounded up to whole pairs, which the domain holds.
        let end = prefix_len
            .checked_next_multiple_of(2 << bound)
            .map_or(cycles, |end| end.min(cycles));
        let ranges = task_ranges(end, 2 << bound);
        let slices: Vec<Option<[&mut [F]; 2]>> = match tables {
            Some(tables) => {
                let len = tables[0].len();
                split_at_tasks(tables, &ranges, |cycle| (cycle >> bound).min(len))
                    .into_iter()
                    .map(Some)
                    .collect()
            }
            None => ranges.iter().map(|_| None).collect(),
        };
        let point: Vec<F> = self.challenges.as_slice().iter().rev().copied().collect();
        let weights = EqPolynomial::<F>::evals(&point, None);
        let add = |left: [F; 3], right: [F; 3]| std::array::from_fn(|i| left[i] + right[i]);
        #[cfg(feature = "parallel")]
        return ranges
            .par_iter()
            .zip(slices)
            .map(|(range, slices)| self.walk_pairs(range.clone(), &weights, slices))
            .try_reduce(|| [F::zero(); 3], |left, right| Ok(add(left, right)));
        #[cfg(not(feature = "parallel"))]
        ranges
            .iter()
            .zip(slices)
            .try_fold([F::zero(); 3], |sum, (range, slices)| {
                Ok(add(sum, self.walk_pairs(range.clone(), &weights, slices)?))
            })
    }

    /// One task's serial visit of `range` in the current round `j`: each
    /// row's increment and write-address values, weighted for its offset
    /// within its half, add to the halves of its `2^(j+1)`-cycle pair
    /// ([`PairHalves`]); each completed pair folds its triple product with
    /// `LT` and, in a filling pass, writes into `slices` the halves they
    /// have entries for.
    fn walk_pairs(
        &self,
        range: Range<usize>,
        weights: &[F],
        mut slices: Option<[&mut [F]; 2]>,
    ) -> Result<[F; 3], WitnessError> {
        let bound = self.challenges.bound();
        let first_pair = range.start >> (bound + 1);
        let mut halves = PairHalves::new(bound);
        let mut evals = [F::Accumulator::default(); 3];
        self.rows.visit(range, VISIT_ROWS, |start, rows| {
            for (t, row) in (start..).zip(rows) {
                let weight = weights[halves.offset(t)];
                let values = [
                    weight.mul_i128(row.rd_inc.0),
                    row.rd.map_or(F::zero(), |register| {
                        weight * self.eq_address[usize::from(register)]
                    }),
                ];
                let Some((pair, [inc, wa])) = halves.add(t, values) else {
                    continue;
                };
                if let Some(tables) = &mut slices {
                    let entry = 2 * (pair - first_pair);
                    for (table, halves) in tables.iter_mut().zip([inc, wa]) {
                        for (slot, half) in table[entry..].iter_mut().zip(halves) {
                            *slot = half;
                        }
                    }
                }
                accumulate_triple_product(
                    &mut evals,
                    (inc[0], inc[1]),
                    (wa[0], wa[1]),
                    self.lt.pair(pair),
                );
            }
            Ok(())
        })?;
        Ok(evals.map(F::Accumulator::reduce))
    }

    fn bind(&mut self, challenge: F) {
        if let ValueTables::Dense { inc, wa } = &mut self.tables {
            let _ = inc.bind(challenge);
            let _ = wa.bind(challenge);
        }
        self.lt.bind(challenge);
        self.challenges.push(challenge);
    }
}

impl<F: JoltField> ProveRounds<F> for StreamingValEvaluationKernel<F> {
    fn num_rounds(&self) -> usize {
        self.challenges.total()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let evals = match self.tables {
            ValueTables::Dense { ref inc, ref wa } => triple_product_round_evals(
                wa.values.len().div_ceil(2),
                |y| inc.pair(y),
                |y| wa.pair(y),
                |y| self.lt.pair(y),
            ),
            ValueTables::Streamed { prefix_len, tail } => {
                if self.challenges.bound() == self.dense_round {
                    self.fill_tables(prefix_len, tail)
                } else {
                    self.round_pass(prefix_len, None)
                }
                .map_err(rows_unavailable)?
            }
        };
        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for StreamingValEvaluationKernel<F> {
    type Relation = RegistersValEvaluation<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RegistersValEvaluationOutputClaims<F>, SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
        match &self.tables {
            ValueTables::Dense { inc, wa } => Ok(RegistersValEvaluationOutputClaims {
                rd_inc: inc.pair(0).0,
                rd_wa: wa.pair(0).0,
            }),
            ValueTables::Streamed { .. } => Err(SumcheckKernelError::InvariantViolation {
                reason: "streamed registers value-evaluation rounds ended before the crossover",
            }),
        }
    }

    /// Pin the split-LT tables to the verifier's scalar path: the fully bound
    /// LT value must equal `derive_output_term(LtCycle)`.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
        pin_derived_term(
            relation,
            JoltDerivedId::from(RegistersValEvaluationPublic::LtCycle),
            input_points,
            output_points,
            challenges,
            self.lt.final_value(),
        )
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests;
