//! The optimized registers value-evaluation (stage 5) kernel.
//!
//! Byte-parity contract: identical round polynomials and output claims to the
//! reference kernel (`reference/registers_val_evaluation.rs`), which folds the
//! `(2^7 × T)` one-hot `rd_wa` grid into a dense cycle table at prepare time
//! and binds three dense `T`-sized tables per round.
//!
//! - **Streamed prefix**: every pass reads the increment and the written
//!   register through the witness's row handle, split into tasks of whole
//!   pairs, at most `TASKS_PER_THREAD` per pool thread (`task_ranges`), each
//!   visiting its rows once, serially. While the dense increment and
//!   write-address tables together exceed `REGISTER_VALUE_DENSE_CAP`, round
//!   `j` contracts each column's rows separately under `eq(r_1..r_j, ·)`
//!   (LowToHigh binding gives `f^(j)(y) = Σ_b eq(r, b)·f(2^j·y + b)`), with
//!   `wa(t) = eq(r_address)[rd_t]` served from the K-sized eq table, so the
//!   `K × T` grid never exists. The first round whose tables fit, at the
//!   earliest round 1, fills them in the pass that yields its evaluations;
//!   later rounds bind them in place. Tables that fit no round never exist:
//!   one more walk yields the openings.
//! - **Split LT** ([`SplitLt`], legacy `LtPolynomial`): `LT(j, r_cycle)` is
//!   served from three ~√T tables instead of a dense `T`-sized one.
//! - **Eval-at-{0,2,3} sampling** with the engine hint supplying s(1)
//!   (legacy samples {1,2,∞}; both interpolate the same cubic exactly).
//! - **Deferred-reduction accumulation** of the triple products.

use std::ops::Range;

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::{JoltDerivedId, JoltPolynomialId, RegistersValEvaluationPublic};
use jolt_field::{Accumulator, JoltField, WithAccumulator};
use jolt_poly::{EqPolynomial, Polynomial, TensorEqTable, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage5::registers_val_evaluation::{
    RegistersValEvaluation, RegistersValEvaluationOutputClaims,
};
use jolt_witness::__private::TraceRow;
use jolt_witness::witnesses::{Extract, RdInc, WitnessEnv};
use jolt_witness::{JoltWitnessPlane, WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::registers_read_write::RegisterCycleRow;
use super::support::{
    dense_crossover, pair, pin_derived_term, split_at_tasks, task_ranges,
    triple_product_round_evals, try_par_sum_vecs, KernelRows, PairHalves, RoundChallenges, SplitLt,
    VISIT_ROWS,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Byte cap on the dense increment and write-address tables together: rounds
/// contract both columns from the rows until the current round's tables fit
/// ([`dense_crossover`]).
const REGISTER_VALUE_DENSE_CAP: usize = 1 << 30;

const ROWS_LABEL: &str = "registers value-evaluation trace rows";

pub struct OptimizedRegistersValEvaluation;

impl<F: JoltField> PrepareKernel<F, RegistersValEvaluation<F>> for OptimizedRegistersValEvaluation {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersValEvaluation<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersValEvaluation<F>>>, KernelError<F>>
    {
        let log_t = inputs.relation.trace_dimensions().log_t();
        let rows = KernelRows::from_witness(witness, log_t, ROWS_LABEL)?;
        Ok(Box::new(ValEvaluationKernel::new(
            rows,
            &inputs,
            REGISTER_VALUE_DENSE_CAP,
        )?))
    }
}

/// One cycle's rd write: the canonical committed increment and the written
/// register, range-checked with the cycle's other operands by
/// [`RegisterCycleRow`].
#[derive(Clone, Copy, Debug)]
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

/// The increment and write-address tables over the cycle domain.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum ValueTables<F: JoltField> {
    /// Rounds before the crossover contract both columns from the rows.
    Streamed,
    /// Filled by the crossover round's pass, bound in place afterwards.
    Dense {
        inc: Polynomial<F>,
        wa: Polynomial<F>,
    },
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct ValEvaluationKernel<F: JoltField> {
    tables: ValueTables<F>,
    /// The round whose pass fills `tables`, at least 1 so the tables never
    /// hold `T` entries; the round count when they never fit the cap.
    dense_round: usize,
    challenges: RoundChallenges<F>,
    /// The witness plane owns these rows; it reports them itself.
    #[cfg_attr(feature = "allocative", allocative(skip))]
    rows: KernelRows<RdWriteRow>,
    eq_address: Vec<F>,
    lt: SplitLt<F>,
}

fn rows_unavailable<F: JoltField>(_: WitnessError) -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource { kind: ROWS_LABEL }
}

impl<F: JoltField> ValEvaluationKernel<F> {
    /// `dense_cap` bounds the bytes of the two dense tables together;
    /// production passes [`REGISTER_VALUE_DENSE_CAP`].
    fn new(
        rows: KernelRows<RdWriteRow>,
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
        let table_bytes = (2 * size_of::<F>()) << log_t;
        Ok(Self {
            tables: ValueTables::Streamed,
            dense_round: dense_crossover(table_bytes, dense_cap, log_t).max(1),
            challenges: RoundChallenges::new(log_t),
            rows,
            eq_address: EqPolynomial::<F>::evals(r_address, None),
            lt: SplitLt::new(r_cycle),
        })
    }

    fn cycles(&self) -> usize {
        1 << self.challenges.total()
    }

    /// The current round's pass, filling the dense tables: each of length
    /// `T / 2^round`, allocated once at its final capacity and written in
    /// disjoint task slices.
    fn fill_tables(&mut self) -> Result<[F; 3], WitnessError> {
        let len = self.cycles() >> self.challenges.bound();
        let mut inc: Vec<F> = unsafe_allocate_zero_vec(len);
        let mut wa: Vec<F> = unsafe_allocate_zero_vec(len);
        let evals = self.round_pass(Some([&mut inc[..], &mut wa[..]]))?;
        self.tables = ValueTables::Dense {
            inc: Polynomial::new(inc),
            wa: Polynomial::new(wa),
        };
        Ok(evals)
    }

    /// Weights of the current round `j`'s pass, one per cycle offset `c`
    /// within a pair half: `eq(r_1..r_j, c)`, offset bit `i` pairing
    /// `r_{i+1}`. Before the crossover, `2^j < max(2, table_bytes / cap)`.
    fn round_weights(&self) -> Vec<F> {
        let point: Vec<F> = self.challenges.as_slice().iter().rev().copied().collect();
        EqPolynomial::<F>::evals(&point, None)
    }

    /// The `s(0), s(2), s(3)` samples of the current round `j` from the rows:
    /// the domain splits into [`task_ranges`] of whole `2^(j+1)`-cycle pairs,
    /// each walked once ([`Self::walk_pairs`]). A filling pass splits
    /// `tables` (pair `z` at entries `2z, 2z + 1`) at the task boundaries, so
    /// each task writes only its own slices. Task results are three field
    /// elements, so they reduce.
    fn round_pass(&self, tables: Option<[&mut [F]; 2]>) -> Result<[F; 3], WitnessError> {
        let bound = self.challenges.bound();
        let ranges = task_ranges(self.cycles(), 2 << bound);
        let slices: Vec<Option<[&mut [F]; 2]>> = match tables {
            Some(tables) => split_at_tasks(tables, &ranges, |cycle| 2 * (cycle >> (bound + 1)))
                .into_iter()
                .map(Some)
                .collect(),
            None => ranges.iter().map(|_| None).collect(),
        };
        let weights = self.round_weights();
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
    /// `LT` and, in a filling pass, writes its halves into `slices`.
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
                let Some((z, [inc, wa])) = halves.add(t, values) else {
                    continue;
                };
                if let Some([inc_table, wa_table]) = &mut slices {
                    let entry = 2 * (z - first_pair);
                    inc_table[entry..entry + 2].copy_from_slice(&inc);
                    wa_table[entry..entry + 2].copy_from_slice(&wa);
                }
                let (lt_0, lt_1) = self.lt.pair(z);
                let (inc_m, wa_m, lt_m) = (inc[1] - inc[0], wa[1] - wa[0], lt_1 - lt_0);
                let (inc_2, wa_2, lt_2) = (inc[1] + inc_m, wa[1] + wa_m, lt_1 + lt_m);
                evals[0].fmadd(inc[0] * wa[0], lt_0);
                evals[1].fmadd(inc_2 * wa_2, lt_2);
                evals[2].fmadd((inc_2 + inc_m) * (wa_2 + wa_m), lt_2 + lt_m);
            }
            Ok(())
        })?;
        Ok(evals.map(F::Accumulator::reduce))
    }

    /// The two openings at the bound cycle point of tables that never
    /// materialized: one walk over the rows weighted by split
    /// `eq(r_cycle, ·)` tables, the bound challenges reversed into
    /// big-endian order.
    fn streamed_openings(&self) -> Result<[F; 2], WitnessError> {
        let reversed: Vec<F> = self.challenges.as_slice().iter().rev().copied().collect();
        let weights = TensorEqTable::new(&reversed);
        let (e_out, e_in) = (weights.e_out(), weights.e_in());
        let group = e_in.len();
        let ranges = task_ranges(weights.len(), group);
        let openings = try_par_sum_vecs(ranges.len(), 2, |task| -> Result<_, WitnessError> {
            let mut openings = vec![F::zero(); 2];
            // A callback of at most `group` aligned rows lies in one `e_out`
            // block, and its accumulators take at most VISIT_ROWS terms.
            self.rows.visit(
                ranges[task].clone(),
                VISIT_ROWS.min(group),
                |start, rows| {
                    let mut inc = <F as WithAccumulator>::SignedProductAccumulator::default();
                    let mut wa = F::Accumulator::default();
                    for (row, &weight) in rows.iter().zip(&e_in[start % group..]) {
                        inc.fmadd_i128(weight, row.rd_inc.0);
                        if let Some(register) = row.rd {
                            wa.fmadd(weight, self.eq_address[usize::from(register)]);
                        }
                    }
                    let scale = e_out[start / group];
                    openings[0] += scale * inc.reduce();
                    openings[1] += scale * wa.reduce();
                    Ok(())
                },
            )?;
            Ok(openings)
        })?;
        Ok([openings[0], openings[1]])
    }

    fn bind(&mut self, challenge: F) {
        if let ValueTables::Dense { inc, wa } = &mut self.tables {
            let _ = inc.bind_low_to_high_in_place(challenge);
            let _ = wa.bind_low_to_high_in_place(challenge);
        }
        self.lt.bind(challenge);
        self.challenges.push(challenge);
    }
}

impl<F: JoltField> ProveRounds<F> for ValEvaluationKernel<F> {
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

        // Evaluate at 0, 2, 3; the engine supplies s(1).
        let evals = if let ValueTables::Dense { inc, wa } = &self.tables {
            triple_product_round_evals(
                inc.len() / 2,
                |y| pair(inc, y),
                |y| pair(wa, y),
                |y| self.lt.pair(y),
            )
        } else if self.challenges.bound() == self.dense_round {
            self.fill_tables().map_err(rows_unavailable)?
        } else {
            self.round_pass(None).map_err(rows_unavailable)?
        };

        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for ValEvaluationKernel<F> {
    type Relation = RegistersValEvaluation<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RegistersValEvaluationOutputClaims<F>, SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
        let [rd_inc, rd_wa] = match &self.tables {
            ValueTables::Dense { inc, wa } => [inc.evals()[0], wa.evals()[0]],
            ValueTables::Streamed => {
                self.streamed_openings()
                    .map_err(|_| SumcheckKernelError::InvariantViolation {
                        reason: "register value opening walk re-extraction failed after the rounds",
                    })?
            }
        };
        Ok(RegistersValEvaluationOutputClaims { rd_inc, rd_wa })
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
        let id = JoltDerivedId::from(RegistersValEvaluationPublic::LtCycle);
        pin_derived_term(
            relation,
            id,
            input_points,
            output_points,
            challenges,
            self.lt.final_value(),
        )
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltVirtualPolynomial};
    use jolt_claims::NoChallenges;
    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::{Fr, Zero};
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_verifier::stages::relations::ConcreteSumcheck as _;
    use jolt_verifier::stages::stage5::registers_val_evaluation::RegistersValEvaluationInputClaims;
    use jolt_witness::witnesses::ToField;
    use jolt_witness::{
        ChunkVisitor, FixedBackend, JoltVmWitnessInputs, JoltWitnessOracle, PolynomialEncoding,
        ProgramSource, RowSource, Shape, TraceBackend,
    };

    use super::super::registers_read_write::test_support::{
        assert_kernel_parity, assert_nontrivial, challenge_sequence, structured_fixture,
        TraceFixture,
    };
    use super::*;
    use crate::optimized::parity::probe_input_claim;
    use crate::optimized::testing::trimmed;
    use crate::ReferenceBackend;

    fn run_parity(fixture: TraceFixture, log_t: usize, seed: u64) {
        fixture.with_plane(log_t, |backend| {
            let relation = RegistersValEvaluation::<Fr>::new(TraceDimensions::new(log_t));
            let point = challenge_sequence(REGISTER_ADDRESS_BITS + log_t, seed ^ 0x3C3C);
            let grid = JoltWitnessOracle::<Fr>::oracle_table(
                backend,
                JoltPolynomialId::Virtual(JoltVirtualPolynomial::RegistersVal),
            )
            .unwrap();
            let input_claim = Polynomial::new(grid).evaluate(&point);
            assert_nontrivial(input_claim);

            let claims = RegistersValEvaluationInputClaims {
                registers_val: input_claim,
            };
            let points = RegistersValEvaluationInputClaims {
                registers_val: point,
            };
            let round_challenges = challenge_sequence(log_t, seed);
            assert_kernel_parity(
                &OptimizedRegistersValEvaluation,
                backend,
                &relation,
                &claims,
                &points,
                &NoChallenges::default(),
                input_claim,
                &round_challenges,
            );
        });
    }

    #[test]
    fn parity_structured_odd_log_t() {
        run_parity(structured_fixture(8), 3, 53);
    }

    #[test]
    fn parity_structured_even_log_t() {
        run_parity(structured_fixture(16), 4, 59);
    }

    #[test]
    fn parity_minimal_padded_trace() {
        let mut fixture = TraceFixture::new();
        fixture.op(Some(4), Some(1), None);
        fixture.op(Some(4), Some(4), Some(4));
        fixture.op(None, Some(4), None);
        run_parity(fixture, 2, 61);
    }

    #[test]
    fn parity_single_cycle_round() {
        let mut fixture = TraceFixture::new();
        fixture.op(Some(11), Some(11), None);
        fixture.op(Some(11), None, Some(11));
        run_parity(fixture, 1, 67);
    }

    /// The reference tier's witness over synthetic `RdInc` and one-hot
    /// `RdWa` columns; the reference kernel reads only these two tables.
    struct SyntheticTables<F>(FixedBackend<F>);

    impl<F: JoltField> SyntheticTables<F> {
        fn new(rows: &[RdWriteRow]) -> Self {
            let log_t = rows.len().ilog2() as usize;
            let mut wa = vec![F::zero(); rows.len() << REGISTER_ADDRESS_BITS];
            for (cycle, row) in rows.iter().enumerate() {
                if let Some(register) = row.rd {
                    wa[(usize::from(register) << log_t) | cycle] = F::one();
                }
            }
            let mut tables = FixedBackend::new();
            tables
                .insert(
                    JoltPolynomialId::Committed(JoltCommittedPolynomial::RdInc),
                    Shape::new(log_t, PolynomialEncoding::Dense),
                    rows.iter().map(|row| row.rd_inc.to_field()).collect(),
                )
                .unwrap();
            tables
                .insert(
                    JoltPolynomialId::Virtual(JoltVirtualPolynomial::RdWa),
                    Shape::new(REGISTER_ADDRESS_BITS + log_t, PolynomialEncoding::Dense),
                    wa,
                )
                .unwrap();
            Self(tables)
        }
    }

    impl<F: JoltField> JoltWitnessOracle<F> for SyntheticTables<F> {
        fn shape(&self, id: JoltPolynomialId) -> Result<Shape, WitnessError> {
            self.0.shape(id)
        }

        fn oracle_table(&self, id: JoltPolynomialId) -> Result<Vec<F>, WitnessError> {
            self.0.oracle_table(id)
        }

        fn committed_order(&self) -> Result<Vec<JoltCommittedPolynomial>, WitnessError> {
            self.0.committed_order()
        }
    }

    impl<F> RowSource for SyntheticTables<F> {
        fn visit_chunks(
            &self,
            _range: Range<usize>,
            _chunk_size: usize,
            _visitor: &mut ChunkVisitor<'_>,
        ) -> Result<(), WitnessError> {
            Err(WitnessError::UnavailableView {
                label: "synthetic register value rows",
            })
        }
    }

    impl<F> ProgramSource for SyntheticTables<F> {
        fn program_preprocessing(&self) -> &JoltProgramPreprocessing {
            unreachable!("the reference value-evaluation kernel reads only oracle tables")
        }
    }

    /// Rows over the whole signed increment range — both `±(2^64 − 1)`
    /// extremes of a u64 write delta and the `i128` limits — and both ends of
    /// the register domain, with canonical absence: a cycle without rd
    /// carries increment zero.
    fn synthetic_rows(log_t: usize, seed: u64) -> Vec<RdWriteRow> {
        const EXTREMES: [i128; 4] = [u64::MAX as i128, -(u64::MAX as i128), i128::MAX, i128::MIN];
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        (0..1usize << log_t)
            .map(|cycle| match cycle % 4 {
                0 => RdWriteRow {
                    rd_inc: RdInc(EXTREMES[cycle / 4 % 4]),
                    rd: Some(if cycle & 4 == 0 { 0 } else { 127 }),
                },
                2 => RdWriteRow {
                    rd_inc: RdInc(0),
                    rd: None,
                },
                _ => {
                    let bits = next();
                    let magnitude = i128::from(next());
                    RdWriteRow {
                        rd_inc: RdInc(if bits & 1 == 0 { magnitude } else { -magnitude }),
                        rd: Some((bits >> 1) as u8 & 0x7F),
                    }
                }
            })
            .collect()
    }

    /// The reference kernel over `witness` and the optimized kernel over
    /// `rows` with its tables capped at `dense_cap` bytes, driven through
    /// every round on the reference's messages: equal round polynomials, the
    /// dense tables first present at round `materialized` with
    /// `T / 2^materialized` entries each (never when it is the round count),
    /// equal output claims and both LT pins. Returns the input claim.
    fn assert_capped_parity<F: JoltField>(
        witness: &dyn JoltWitnessPlane<F>,
        rows: KernelRows<RdWriteRow>,
        point: Vec<F>,
        round_challenges: &[F],
        dense_cap: usize,
        materialized: usize,
    ) -> F {
        let log_t = round_challenges.len();
        let relation = RegistersValEvaluation::<F>::new(TraceDimensions::new(log_t));
        let claims = RegistersValEvaluationInputClaims {
            registers_val: F::zero(),
        };
        let points = RegistersValEvaluationInputClaims {
            registers_val: point,
        };
        let challenges = NoChallenges::default();
        let inputs = || ProverInputs {
            relation: &relation,
            claims: &claims,
            points: &points,
            challenges: &challenges,
        };
        let mut reference = ReferenceBackend
            .prepare(&mut ProofSession::default(), witness, inputs())
            .unwrap();
        let input_claim = probe_input_claim(reference.as_mut());
        let mut optimized = ValEvaluationKernel::new(rows, &inputs(), dense_cap).unwrap();

        let dense_len = |kernel: &ValEvaluationKernel<F>| match &kernel.tables {
            ValueTables::Dense { inc, wa } => {
                assert_eq!(inc.len(), wa.len());
                Some(inc.len())
            }
            ValueTables::Streamed => None,
        };
        let mut first_dense = None;
        let mut claim = input_claim;
        for (round, &challenge) in round_challenges.iter().enumerate() {
            let bind = round
                .checked_sub(1)
                .map(|previous| round_challenges[previous]);
            let reference_round = reference.prove_round(bind, round, claim).unwrap();
            let optimized_round = optimized.prove_round(bind, round, claim).unwrap();
            assert_eq!(
                trimmed(&optimized_round),
                trimmed(&reference_round),
                "round {round}, log_t {log_t}, cap {dense_cap}"
            );
            first_dense = first_dense.or_else(|| dense_len(&optimized).map(|len| (round, len)));
            claim = reference_round.evaluate(challenge);
        }
        let last = round_challenges[log_t - 1];
        reference.finish_rounds(last).unwrap();
        optimized.finish_rounds(last).unwrap();
        assert_eq!(
            first_dense,
            (materialized < log_t).then(|| (materialized, (1 << log_t) >> materialized)),
            "dense tables, log_t {log_t}, cap {dense_cap}"
        );
        assert_eq!(
            optimized.output_claims(&claims).unwrap(),
            reference.output_claims(&claims).unwrap(),
            "output claims, log_t {log_t}, cap {dense_cap}"
        );
        let output_points = relation
            .derive_opening_points(round_challenges, &points)
            .unwrap();
        reference
            .validate_derived_tables(&relation, &points, &output_points, &challenges)
            .unwrap();
        optimized
            .validate_derived_tables(&relation, &points, &output_points, &challenges)
            .unwrap();
        input_claim
    }

    /// The cap of the two tables at their size after `depth` binds, so they
    /// first fit there; at depth 0 they would fit unbound, which the kernel
    /// still defers to round 1.
    fn cap_at_depth<F: JoltField>(log_t: usize, depth: usize) -> usize {
        ((2 * size_of::<F>()) << log_t) >> depth
    }

    /// Full-range synthetic rows at every crossover the private cap selects:
    /// round 1 under a cap the unbound tables meet, every later round, and
    /// never.
    #[test]
    fn synthetic_parity_at_every_crossover() {
        fn case<F: JoltField>(log_t: usize, seed: u64) {
            let rows = synthetic_rows(log_t, seed);
            let reference = SyntheticTables::<F>::new(&rows);
            let point = challenge_sequence(REGISTER_ADDRESS_BITS + log_t, seed);
            let round_challenges = challenge_sequence(log_t, seed ^ 0x5A5A);
            for depth in 0..=log_t {
                let claim = assert_capped_parity(
                    &reference,
                    KernelRows::Synthetic(rows.clone()),
                    point.clone(),
                    &round_challenges,
                    cap_at_depth::<F>(log_t, depth),
                    depth.max(1),
                );
                assert_nontrivial(claim);
            }
        }
        for (log_t, seed) in [(1, 11), (2, 12), (3, 13), (4, 14), (6, 16)] {
            case::<Fr>(log_t, seed);
            #[cfg(feature = "akita")]
            case::<Prime128OffsetA7F7>(log_t, seed);
        }
    }

    /// Zero and one as round challenges and as input-point coordinates:
    /// vanishing contraction weights, a one-hot `eq(r_address, ·)` and LT
    /// tables with zero entries, down to an identically zero sum.
    #[test]
    fn exceptional_challenges_and_points_match_reference() {
        fn case<F: JoltField>() {
            let log_t = 4;
            let rows = synthetic_rows(log_t, 41);
            let reference = SyntheticTables::<F>::new(&rows);
            let random: Vec<F> = challenge_sequence(REGISTER_ADDRESS_BITS + log_t, 43);
            let (zero, one) = (F::zero(), F::one());
            let mixed: Vec<F> = random
                .iter()
                .enumerate()
                .map(|(i, &value)| [zero, one, value][i % 3])
                .collect();
            let points = [
                vec![zero; REGISTER_ADDRESS_BITS + log_t],
                vec![one; REGISTER_ADDRESS_BITS + log_t],
                mixed,
            ];
            let challenge_sets = [
                vec![zero, one, zero, one],
                vec![one; log_t],
                vec![random[0], zero, random[1], one],
            ];
            for point in &points {
                for round_challenges in &challenge_sets {
                    for depth in [1, 2, log_t] {
                        let _ = assert_capped_parity(
                            &reference,
                            KernelRows::Synthetic(rows.clone()),
                            point.clone(),
                            round_challenges,
                            cap_at_depth::<F>(log_t, depth),
                            depth,
                        );
                    }
                }
            }
        }
        case::<Fr>();
        #[cfg(feature = "akita")]
        case::<Prime128OffsetA7F7>();
    }

    /// The witness's trace rows, with task boundaries and a padded proof
    /// suffix inside the domain, at every crossover in both fields. At
    /// log_t 13 with no materialization, the last round's one pair spans two
    /// visit callbacks.
    #[test]
    fn streamed_rounds_over_trace_rows_match_reference() {
        fn case<F: JoltField>(backend: &TraceBackend, log_t: usize, depth: usize) {
            let _ = assert_capped_parity(
                backend,
                KernelRows::from_witness::<F>(backend, log_t, ROWS_LABEL).unwrap(),
                challenge_sequence(REGISTER_ADDRESS_BITS + log_t, 97),
                &challenge_sequence::<F>(log_t, 101),
                cap_at_depth::<F>(log_t, depth),
                depth,
            );
        }
        for log_t in [1, 3, 6] {
            structured_fixture((1 << log_t) - 1).with_plane(log_t, |backend| {
                for depth in 1..=log_t {
                    case::<Fr>(backend, log_t, depth);
                    #[cfg(feature = "akita")]
                    case::<Prime128OffsetA7F7>(backend, log_t, depth);
                }
            });
        }
        structured_fixture((1 << 13) - 3).with_plane(13, |backend| {
            for depth in [1, 13] {
                case::<Fr>(backend, 13, depth);
            }
        });
    }

    /// A witness without a row handle or over another cycle domain fails
    /// typed instead of collecting rows; an operand outside the register
    /// domain fails the first streamed round.
    #[test]
    fn missing_rows_and_row_failures_are_typed() {
        let prepare = |witness: &TraceBackend, log_t: usize| {
            let relation = RegistersValEvaluation::<Fr>::new(TraceDimensions::new(log_t));
            let claims = RegistersValEvaluationInputClaims {
                registers_val: Fr::zero(),
            };
            let points = RegistersValEvaluationInputClaims {
                registers_val: challenge_sequence(REGISTER_ADDRESS_BITS + log_t, 7),
            };
            OptimizedRegistersValEvaluation.prepare(
                &mut ProofSession::default(),
                witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &NoChallenges::default(),
                },
            )
        };
        let first_round_fails = |witness: &TraceBackend| {
            matches!(
                prepare(witness, 3)
                    .unwrap()
                    .prove_round(None, 0, Fr::zero()),
                Err(SumcheckError::MissingEvaluationSource { .. })
            )
        };
        structured_fixture(7).with_plane(3, |resident| {
            // A configured domain shorter than the trace leaves no valid handle.
            let mut shortened = TraceBackend::new(
                resident.config.clone(),
                JoltVmWitnessInputs::new(
                    &resident.program,
                    &resident.preprocessing,
                    resident.trace.clone(),
                ),
            );
            shortened.config.log_t = 1;
            assert!(matches!(
                prepare(&shortened, 3),
                Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
            ));
            assert!(matches!(
                prepare(resident, 4),
                Err(KernelError::Witness(WitnessError::InvalidDimensions { .. }))
            ));
        });
        let mut fixture = structured_fixture(5);
        fixture.unchecked_read(200);
        fixture.with_plane(3, |backend| assert!(first_round_fails(backend)));
    }
}
