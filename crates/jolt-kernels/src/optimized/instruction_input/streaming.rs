//! Streaming instruction input virtualization: the eager kernel's rounds
//! after a crossover, preceded by rounds that read the witness's resident
//! rows in task-partitioned passes instead of materializing the ten tables
//! at `T/2`. Round 0 evaluates native integer row pairs. Round `j ≥ 1`
//! contracts each column over the low and high halves of its
//! `2^(j+1)`-cycle pairs with `eq(r_1..r_j, ·)` (LowToHigh binding gives
//! `f^(j)(y) = Σ_b eq(r, b)·f(2^j·y + b)`). The first round whose tables fit
//! [`DENSE_TABLE_CAP`] fills them in the pass that yields its message, and the
//! eager kernel binds them from there.

use jolt_claims::protocols::jolt::relations::instruction::InstructionInputOutputClaims;
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage3::outputs::InstructionInput;
use jolt_witness::{JoltWitnessPlane, WitnessError};

use super::{
    operand_product, row_extraction_error, InputState, InstructionInputRow, NativePairSums,
    OptimizedInstructionInputKernel, NUM_TABLES,
};
use crate::optimized::streaming::{
    dense_crossover, gruen_pair_pass, task_ranges, PairHalves, DENSE_TABLE_CAP, VISIT_ROWS,
};
use crate::optimized::support::{
    try_par_sum_vecs, BundleStore, GruenRoundMessage, RoundChallenges,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const TERMS: usize = 3;

/// The quadratic's constant and leading coefficients, then its value at one.
#[inline]
fn pair_q_terms<F: JoltField>(
    values: [F; NUM_TABLES],
    steps: [F; NUM_TABLES],
    gamma: F,
) -> [F; TERMS] {
    let high = core::array::from_fn(|i| values[i] + steps[i]);
    [
        operand_product(&values, gamma),
        operand_product(&steps, gamma),
        operand_product(&high, gamma),
    ]
}

pub struct StreamingInstructionInput;

/// The round whose pass fills the dense tables over `2^rounds` cycles: the
/// first whose tables fit `cap`, at least round 1 when there are two or more
/// rounds, and at most the last round, so openings are always table entries.
fn crossover_round<F: JoltField>(rounds: usize, cap: usize) -> usize {
    let table_bytes = (NUM_TABLES * size_of::<F>()) << rounds;
    dense_crossover(table_bytes, cap, rounds)
        .max(1)
        .min(rounds.saturating_sub(1))
}

impl<F: JoltField> PrepareKernel<F, InstructionInput<F>> for StreamingInstructionInput {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionInput<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionInput<F>>>, KernelError<F>> {
        let r_product = inputs.relation.product_remainder_opening_point();
        let rows = BundleStore::resident(witness, r_product.len(), "instruction input trace rows")?;
        Ok(Box::new(StreamingInstructionInputKernel::new(
            r_product,
            rows,
            inputs.challenges.gamma,
            DENSE_TABLE_CAP,
        )))
    }
}

/// The rounds before the crossover.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct StreamedRounds<F: JoltField> {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gamma: F,
    /// The round whose pass fills the tables ([`crossover_round`]).
    dense_round: usize,
    gruen: GruenSplitEqPolynomial<F>,
    challenges: RoundChallenges<F>,
    rows: BundleStore<InstructionInputRow>,
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum Phase<F: JoltField> {
    Streamed(StreamedRounds<F>),
    Dense(OptimizedInstructionInputKernel<F>),
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub struct StreamingInstructionInputKernel<F: JoltField> {
    phase: Phase<F>,
}

impl InstructionInputRow {
    /// The ten column values of this cycle times `weight`, in table order.
    #[inline]
    fn weighted_values<F: JoltField>(&self, weight: F) -> [F; NUM_TABLES] {
        let flag = |set: bool| if set { weight } else { F::zero() };
        [
            flag(self.is_rs1.0),
            weight.mul_u64(self.rs1_value.0),
            flag(self.is_pc.0),
            weight.mul_u64(self.unexpanded_pc.0),
            flag(self.is_rs2.0),
            weight.mul_u64(self.rs2_value.0),
            flag(self.is_imm.0),
            weight.mul_i128(self.imm.0),
            flag(self.is_rd.0),
            weight.mul_u64(self.rd_write_value.0),
        ]
    }
}

impl<F: JoltField> StreamedRounds<F> {
    /// Round 0's quadratic terms over native integer row pairs. A callback of
    /// at most `group` aligned rows lies in one `e_out` block and holds whole
    /// even/odd pairs ([`task_ranges`] starts every task on a group boundary).
    fn native_q_terms(&self) -> Result<[F; TERMS], WitnessError> {
        let e_out = self.gruen.e_out_current();
        let e_in = self.gruen.e_in_current();
        let group = 2 * e_in.len();
        let ranges = task_ranges(e_out.len() * group, group);
        try_par_sum_vecs(ranges.len(), TERMS, |task| {
            let mut q_terms = vec![F::zero(); TERMS];
            self.rows.visit(
                ranges[task].clone(),
                VISIT_ROWS.min(group),
                |start, bundles| {
                    let mut sums = NativePairSums::<F, TERMS>::new();
                    for (pair, &weight) in bundles.chunks_exact(2).zip(&e_in[start % group / 2..]) {
                        sums.add(&pair[0], &pair[1], weight);
                    }
                    let block = sums.finish(e_out[start / group], self.gamma);
                    for (q, value) in q_terms.iter_mut().zip(block) {
                        *q += value;
                    }
                    Ok(())
                },
            )?;
            Ok(q_terms)
        })
        .map(|q_terms| core::array::from_fn(|t| q_terms[t]))
    }

    /// One pass of the current round `j` over its pairs of `2^(j+1)` cycles
    /// ([`gruen_pair_pass`]), filling `tables` when given: each row's ten
    /// column values, weighted by `eq(r_1..r_j, c)` for its offset `c` within
    /// its half (offset bit `i` pairing `r_{i+1}`), add to the low or high
    /// half of its pair ([`PairHalves`]), and every completed pair contributes
    /// its quadratic terms in `pair_q_terms` order.
    fn round_pass(
        &self,
        tables: Option<[&mut [F]; NUM_TABLES]>,
    ) -> Result<[F; TERMS], WitnessError> {
        let point: Vec<F> = self.challenges.as_slice().iter().rev().copied().collect();
        let weights = EqPolynomial::<F>::evals(&point, None);
        let log_half = self.challenges.bound();
        let gamma = self.gamma;
        gruen_pair_pass(
            &self.gruen,
            log_half + 1,
            tables,
            |halves: [[F; 2]; NUM_TABLES]| {
                pair_q_terms(
                    halves.map(|[low, _]| low),
                    halves.map(|[low, high]| high - low),
                    gamma,
                )
            },
            |range, chunk, sink| {
                let mut halves = PairHalves::new(log_half);
                self.rows.visit(range, chunk, |start, bundles| {
                    for (t, row) in (start..).zip(bundles) {
                        let values = row.weighted_values(weights[halves.offset(t)]);
                        if let Some((pair, pair_halves)) = halves.add(t, values) {
                            sink.emit(pair, pair_halves);
                        }
                    }
                    Ok(())
                })
            },
        )
    }

    /// The crossover round's pass, filling the ten tables of `T / 2^j`
    /// entries each, allocated once at their final capacity and written in
    /// disjoint task slices.
    fn fill_tables(&self) -> Result<(Vec<Vec<F>>, [F; TERMS]), WitnessError> {
        let len = (1usize << self.challenges.total()) >> self.challenges.bound();
        let mut tables: [Vec<F>; NUM_TABLES] =
            core::array::from_fn(|_| unsafe_allocate_zero_vec(len));
        let q_terms = self.round_pass(Some(tables.each_mut().map(|table| table.as_mut_slice())))?;
        Ok((tables.into_iter().collect(), q_terms))
    }

    /// The error of reading opening values before the crossover, which every
    /// round count of at least one reaches.
    fn not_dense(&self) -> SumcheckKernelError<F> {
        match self.challenges.require_complete() {
            Err(error) => error,
            Ok(()) => SumcheckKernelError::InvariantViolation {
                reason: "streamed instruction input rounds ended before the crossover",
            },
        }
    }
}

impl<F: JoltField> StreamingInstructionInputKernel<F> {
    /// `dense_cap` bounds the bytes of the ten dense tables; production
    /// passes [`DENSE_TABLE_CAP`].
    fn new(
        r_product: &[F],
        rows: BundleStore<InstructionInputRow>,
        gamma: F,
        dense_cap: usize,
    ) -> Self {
        let rounds = r_product.len();
        Self {
            phase: Phase::Streamed(StreamedRounds {
                gamma,
                dense_round: crossover_round::<F>(rounds, dense_cap),
                gruen: GruenSplitEqPolynomial::new(r_product, BindingOrder::LowToHigh),
                challenges: RoundChallenges::new(rounds),
                rows,
            }),
        }
    }

    /// The current round's message. The crossover round hands its tables,
    /// split-eq state and progress to the eager kernel.
    fn message(
        &mut self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let streamed = match &mut self.phase {
            Phase::Dense(kernel) => return kernel.message(round, previous_claim),
            Phase::Streamed(streamed) => streamed,
        };
        let bound = streamed.challenges.bound();
        let (q_terms, tables) = if bound == streamed.dense_round {
            let (tables, q_terms) = streamed.fill_tables().map_err(row_extraction_error)?;
            (q_terms, Some(tables))
        } else if bound == 0 {
            let q_terms = streamed.native_q_terms().map_err(row_extraction_error)?;
            (q_terms, None)
        } else {
            let q_terms = streamed.round_pass(None).map_err(row_extraction_error)?;
            (q_terms, None)
        };
        let poly =
            streamed
                .gruen
                .checked_cubic(q_terms[0], q_terms[1], previous_claim, round, || q_terms[2])?;
        if let Some(tables) = tables {
            let prefix_len = tables[0].len();
            let tail = core::array::from_fn(|i| tables[i][prefix_len - 1]);
            self.phase = Phase::Dense(OptimizedInstructionInputKernel {
                progress: streamed.challenges.progress(),
                gamma: streamed.gamma,
                state: InputState::Dense(tables),
                prefix_len,
                tail,
                gruen: streamed.gruen.clone(),
            });
        }
        Ok(poly)
    }

    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        match &mut self.phase {
            Phase::Streamed(streamed) => {
                streamed.gruen.bind(challenge);
                streamed.challenges.push(challenge);
                Ok(())
            }
            Phase::Dense(kernel) => kernel.bind(challenge),
        }
    }
}

impl<F: JoltField> ProveRounds<F> for StreamingInstructionInputKernel<F> {
    fn num_rounds(&self) -> usize {
        match &self.phase {
            Phase::Streamed(streamed) => streamed.challenges.total(),
            Phase::Dense(kernel) => kernel.num_rounds(),
        }
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge)?;
        }
        self.message(round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: JoltField> SumcheckKernel<F> for StreamingInstructionInputKernel<F> {
    type Relation = InstructionInput<F>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<InstructionInputOutputClaims<F>, SumcheckKernelError<F>> {
        match &mut self.phase {
            Phase::Dense(kernel) => kernel.output_claims(inputs),
            Phase::Streamed(streamed) => Err(streamed.not_dense()),
        }
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        match &self.phase {
            Phase::Dense(kernel) => {
                kernel.validate_derived_tables(relation, input_points, output_points, challenges)
            }
            Phase::Streamed(streamed) => Err(streamed.not_dense()),
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use std::collections::BTreeMap;

    #[cfg(feature = "akita")]
    use jolt_akita::AkitaField;
    use jolt_claims::protocols::jolt::geometry::instruction::{
        imm, left_operand_is_pc, left_operand_is_rs1, rd_write_value, right_operand_is_imm,
        right_operand_is_rd_write_value, right_operand_is_rs2, rs1_value, rs2_value, unexpanded_pc,
    };
    use jolt_claims::protocols::jolt::relations::instruction::{
        InstructionInputChallenges, InstructionInputInputClaims,
    };
    use jolt_claims::protocols::jolt::{InstructionInputPublic, JoltDerivedId, TraceDimensions};
    use jolt_field::{Fr, Ring};
    use jolt_poly::Polynomial;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_witness::witnesses::{
        Imm, InstructionFlag, RdWriteValue, Rs1Value, Rs2Value, UnexpandedPc,
    };
    use jolt_witness::{JoltVmWitnessInputs, TraceBackend};

    use super::*;
    use crate::optimized::parity::{probe_input_claim, run_lockstep, ExceptionalEq};
    use crate::optimized::registers_read_write::test_support::structured_fixture;
    use crate::reference::views::eq_table;
    use crate::{NaiveSumcheckProver, ReferenceBackend};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn challenge(round: usize) -> Fr {
        fr(0x1357_9BDF_2468_ACE0 ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x55)
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Rows no trace produces: random flag combinations, full-range words
    /// and immediates of both signs. Every fifth row sets every flag with
    /// `u64::MAX` words and an `i128::MIN` immediate; the row after it
    /// carries an `i128::MAX` immediate.
    fn synthetic_rows(log_t: usize, seed: u64) -> Vec<InstructionInputRow> {
        let mut state = seed;
        (0..1usize << log_t)
            .map(|index| {
                let flags = splitmix(&mut state);
                let wide = ((splitmix(&mut state) as i128) << 64) | splitmix(&mut state) as i128;
                let row = InstructionInputRow {
                    is_rs1: InstructionFlag(flags & 1 != 0),
                    rs1_value: Rs1Value(splitmix(&mut state)),
                    is_pc: InstructionFlag(flags & 2 != 0),
                    unexpanded_pc: UnexpandedPc(splitmix(&mut state)),
                    is_rs2: InstructionFlag(flags & 4 != 0),
                    rs2_value: Rs2Value(splitmix(&mut state)),
                    is_imm: InstructionFlag(flags & 8 != 0),
                    imm: Imm(wide),
                    is_rd: InstructionFlag(flags & 16 != 0),
                    rd_write_value: RdWriteValue(splitmix(&mut state)),
                };
                match index % 5 {
                    0 => InstructionInputRow {
                        is_rs1: InstructionFlag(true),
                        rs1_value: Rs1Value(u64::MAX),
                        is_pc: InstructionFlag(true),
                        unexpanded_pc: UnexpandedPc(u64::MAX),
                        is_rs2: InstructionFlag(true),
                        rs2_value: Rs2Value(u64::MAX),
                        is_imm: InstructionFlag(true),
                        imm: Imm(i128::MIN),
                        is_rd: InstructionFlag(true),
                        rd_write_value: RdWriteValue(u64::MAX),
                    },
                    1 => InstructionInputRow {
                        imm: Imm(i128::MAX),
                        ..row
                    },
                    _ => row,
                }
            })
            .collect()
    }

    /// The dense tables' length once the eager kernel holds them.
    fn dense_len(kernel: &StreamingInstructionInputKernel<Fr>) -> Option<usize> {
        match &kernel.phase {
            Phase::Dense(OptimizedInstructionInputKernel {
                state: InputState::Dense(tables),
                ..
            }) => Some(tables[0].len()),
            _ => None,
        }
    }

    /// Drives the reference prover over the rows' dense columns and the
    /// streaming kernel, its tables capped at exactly their size at round
    /// `depth`, through every round on `challenges`: equal round polynomials
    /// and output claims, the pinned eq scalar, and tables filled with
    /// `T / 2^c` entries at round `c = min(depth, log_t − 1)`.
    fn assert_parity(
        rows: Vec<InstructionInputRow>,
        r_product: &[Fr],
        challenges: &[Fr],
        depth: usize,
    ) {
        let log_t = r_product.len();
        let tables: Vec<Vec<Fr>> = (0..NUM_TABLES)
            .map(|table| {
                rows.iter()
                    .map(|row| row.field_values::<Fr>()[table])
                    .collect()
            })
            .collect();
        let gamma = fr(0xDADA_CAFE);
        let relation = InstructionInput::<Fr>::new(TraceDimensions::new(log_t), r_product.to_vec());
        let ids = [
            left_operand_is_rs1(),
            rs1_value(),
            left_operand_is_pc(),
            unexpanded_pc(),
            right_operand_is_rs2(),
            rs2_value(),
            right_operand_is_imm(),
            imm(),
            right_operand_is_rd_write_value(),
            rd_write_value(),
        ];
        let opening_tables: BTreeMap<_, _> = ids
            .into_iter()
            .zip(&tables)
            .map(|(id, values)| (id, Polynomial::new(values.clone())))
            .collect();
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(InstructionInputPublic::EqProduct),
            Polynomial::new(eq_table(r_product)),
        )]);
        let input_claims = InstructionInputInputClaims {
            right_instruction_input: fr(0),
            left_instruction_input: fr(0),
        };
        let input_points = InstructionInputInputClaims {
            right_instruction_input: Vec::new(),
            left_instruction_input: Vec::new(),
        };
        let gamma_challenge = InstructionInputChallenges { gamma };
        let inputs = ProverInputs {
            relation: &relation,
            claims: &input_claims,
            points: &input_points,
            challenges: &gamma_challenge,
        };
        let mut reference = NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )
        .unwrap();
        let cap = ((NUM_TABLES * size_of::<Fr>()) << log_t) >> depth;
        let mut streaming = StreamingInstructionInputKernel::new(
            r_product,
            BundleStore::Retained(rows),
            gamma,
            cap,
        );

        let mut claim = probe_input_claim(&mut reference);
        let mut filled = None;
        let mut bind = None;
        for (round, &challenge) in challenges.iter().enumerate() {
            let reference_poly = reference.prove_round(bind, round, claim).unwrap();
            let streaming_poly = streaming.prove_round(bind, round, claim).unwrap();
            assert_eq!(
                reference_poly.coefficients(),
                streaming_poly.coefficients(),
                "round {round} polynomial, log_t {log_t}, depth {depth}"
            );
            filled = filled.or_else(|| dense_len(&streaming).map(|len| (round, len)));
            claim = reference_poly.evaluate(challenge);
            bind = Some(challenge);
        }
        let last = bind.unwrap();
        reference.finish_rounds(last).unwrap();
        streaming.finish_rounds(last).unwrap();
        let crossover = depth.min(log_t - 1);
        assert_eq!(
            filled,
            Some((crossover, (1 << log_t) >> crossover)),
            "dense tables, log_t {log_t}, depth {depth}"
        );
        assert_eq!(
            streaming.output_claims(&input_claims).unwrap(),
            reference.output_claims(&input_claims).unwrap(),
            "output claims, log_t {log_t}, depth {depth}"
        );
        let output_points = relation
            .derive_opening_points(challenges, &input_points)
            .unwrap();
        streaming
            .validate_derived_tables(&relation, &input_points, &output_points, &gamma_challenge)
            .unwrap();
    }

    fn product_point(log_t: usize) -> Vec<Fr> {
        (0..log_t).map(|i| fr(900 + 53 * i as u64)).collect()
    }

    /// Wide synthetic rows at every crossover round. At 2^13 cycles the
    /// last-round crossover's pairs span two visit callbacks; one round fills
    /// at round 0.
    #[test]
    fn synthetic_parity_at_every_crossover() {
        for (log_t, seed) in [
            (1usize, 4242u64),
            (2, 1717),
            (3, 8080),
            (4, 21),
            (5, 55),
            (8, 88),
            (13, 1313),
        ] {
            let challenges: Vec<Fr> = (0..log_t).map(challenge).collect();
            let depths = match log_t {
                1 => 1..=1,
                13 => 12..=12,
                _ => 1..=log_t - 1,
            };
            for depth in depths {
                assert_parity(
                    synthetic_rows(log_t, seed),
                    &product_point(log_t),
                    &challenges,
                    depth,
                );
            }
        }
    }

    /// Exceptional eq points (coordinates 0 or 1, a prefix that vanishes at
    /// the first bind) and zero/one challenges, through round-1 fills,
    /// streamed rounds before a later fill and the last-round fill.
    #[test]
    fn exceptional_eq_and_zero_one_challenges_match_reference() {
        let log_t = 4;
        let challenges: Vec<Fr> = (0..log_t).map(challenge).collect();
        let zero_one: Vec<Fr> = (0..log_t).map(|round| fr(round as u64 % 2)).collect();
        for depth in [1, 2, log_t - 1] {
            for case in ExceptionalEq::ALL {
                assert_parity(
                    synthetic_rows(log_t, 379),
                    &case.point(log_t, challenges[0]),
                    &challenges,
                    depth,
                );
            }
            assert_parity(
                synthetic_rows(log_t, 379),
                &product_point(log_t),
                &zero_one,
                depth,
            );
        }
    }

    /// One [`trace_rows_match_reference_at_every_crossover`] case: the tables
    /// capped to first fit at round `depth`, or the production cap through
    /// the trait path when `None`.
    fn trace_case<F: JoltField>(backend: &TraceBackend, log_t: usize, depth: Option<usize>) {
        let r_product: Vec<F> = (0..log_t)
            .map(|i| F::from_u64(37 + 23 * i as u64))
            .collect();
        let relation = InstructionInput::new(TraceDimensions::new(log_t), r_product.clone());
        let claims = InstructionInputInputClaims {
            right_instruction_input: F::zero(),
            left_instruction_input: F::zero(),
        };
        let points = InstructionInputInputClaims {
            right_instruction_input: Vec::new(),
            left_instruction_input: Vec::new(),
        };
        let challenges = InstructionInputChallenges {
            gamma: F::from_u64(7019),
        };
        let inputs = || ProverInputs {
            relation: &relation,
            claims: &claims,
            points: &points,
            challenges: &challenges,
        };
        let mut reference = ReferenceBackend
            .prepare(&mut ProofSession::default(), backend, inputs())
            .unwrap();
        let mut streaming: Box<dyn SumcheckKernel<F, Relation = InstructionInput<F>>> = match depth
        {
            Some(depth) => Box::new(StreamingInstructionInputKernel::new(
                &r_product,
                BundleStore::resident::<F>(backend, log_t, "trace rows").unwrap(),
                challenges.gamma,
                ((NUM_TABLES * size_of::<F>()) << log_t) >> depth,
            )),
            None => StreamingInstructionInput
                .prepare(&mut ProofSession::default(), backend, inputs())
                .unwrap(),
        };
        let input_claim = probe_input_claim(reference.as_mut());
        let round_challenges: Vec<F> = (0..log_t)
            .map(|i| F::from_u64(883 + 29 * i as u64))
            .collect();
        run_lockstep(
            reference.as_mut(),
            streaming.as_mut(),
            input_claim,
            &round_challenges,
        );
        assert_eq!(
            streaming.output_claims(&claims).unwrap(),
            reference.output_claims(&claims).unwrap(),
            "output claims, log_t {log_t}, depth {depth:?}"
        );
    }

    /// Every round and the output claims over the witness's resident rows
    /// (task boundaries and a padded proof suffix inside the domain) at every
    /// crossover round and at the production cap, against the reference, in
    /// both fields.
    #[test]
    fn trace_rows_match_reference_at_every_crossover() {
        for log_t in [1, 3, 6] {
            structured_fixture((1 << log_t) - 1).with_plane(log_t, |backend| {
                for depth in (1..log_t).map(Some).chain([None]) {
                    trace_case::<Fr>(backend, log_t, depth);
                    #[cfg(feature = "akita")]
                    trace_case::<AkitaField>(backend, log_t, depth);
                }
            });
        }
    }

    /// A witness without resident rows, or with another cycle domain, fails
    /// typed instead of collecting rows.
    #[test]
    fn missing_or_mismatched_rows_are_typed() {
        let log_t = 3;
        let point = |len: usize| (0..len).map(|i| fr(5 + i as u64)).collect::<Vec<_>>();
        let claims = InstructionInputInputClaims {
            right_instruction_input: fr(0),
            left_instruction_input: fr(0),
        };
        let points = InstructionInputInputClaims {
            right_instruction_input: Vec::new(),
            left_instruction_input: Vec::new(),
        };
        let challenges = InstructionInputChallenges { gamma: fr(3) };
        let prepare = |witness: &TraceBackend, log_t: usize| {
            let relation = InstructionInput::new(TraceDimensions::new(log_t), point(log_t));
            StreamingInstructionInput
                .prepare(
                    &mut ProofSession::default(),
                    witness,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .map(|_| ())
        };
        structured_fixture(7).with_plane(log_t, |resident| {
            // A configured domain shorter than the trace leaves no rows.
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
                prepare(&shortened, log_t),
                Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
            ));
            assert!(matches!(
                prepare(resident, log_t + 1),
                Err(KernelError::Witness(WitnessError::InvalidDimensions { .. }))
            ));
        });
    }
}
