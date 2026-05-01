//! Stage 1 outer-sumcheck kernel ABI.

use std::error::Error;
use std::fmt::{self, Display, Formatter};

use jolt_field::{Field, FieldAccumulator};
use jolt_poly::lagrange::{interpolate_to_coeffs, lagrange_evals, lagrange_kernel_eval};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_r1cs::{constraints::rv64, R1csKey, R1csRowDotSlice, R1csRowDotTable};
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::{Label, LabelWithCount, Transcript};
use rayon::prelude::*;

mod rv64_typed;
pub use rv64_typed::{Stage1OuterRv64Data, Stage1Rv64Cycle};

const OUTER_UNISKIP_DOMAIN_SIZE: usize = 10;
const OUTER_UNISKIP_DEGREE: usize = 9;
const OUTER_UNISKIP_EXTENDED_SIZE: usize = 19;
const OUTER_UNISKIP_NUM_COEFFS: usize = 28;
const OUTER_UNISKIP_DEGREE_BOUND: usize = OUTER_UNISKIP_NUM_COEFFS - 1;
const OUTER_UNISKIP_EXTENDED_START: i64 = -(OUTER_UNISKIP_DEGREE as i64);
const OUTER_UNISKIP_BASE_START: i64 = -((OUTER_UNISKIP_DOMAIN_SIZE as i64 - 1) / 2);
const OUTER_REMAINING_DEGREE_BOUND: usize = 3;
const DENSE_BIND_PAR_THRESHOLD: usize = 1024;
const OUTER_FIRST_GROUP_ROWS: [usize; 10] = [1, 2, 3, 4, 5, 6, 11, 14, 17, 18];
const OUTER_SECOND_GROUP_ROWS: [usize; 9] = [0, 7, 8, 9, 10, 12, 13, 15, 16];
const OUTER_EQ_CONSTRAINT_ROWS: usize =
    OUTER_FIRST_GROUP_ROWS.len() + OUTER_SECOND_GROUP_ROWS.len();
const OUTER_UNISKIP_TARGET_COEFFS: [[i64; OUTER_UNISKIP_DOMAIN_SIZE]; OUTER_UNISKIP_DEGREE] = [
    [10, -45, 120, -210, 252, -210, 120, -45, 10, -1],
    [-1, 10, -45, 120, -210, 252, -210, 120, -45, 10],
    [55, -330, 990, -1848, 2310, -1980, 1155, -440, 99, -10],
    [-10, 99, -440, 1155, -1980, 2310, -1848, 990, -330, 55],
    [
        220, -1485, 4752, -9240, 11880, -10395, 6160, -2376, 540, -55,
    ],
    [
        -55, 540, -2376, 6160, -10395, 11880, -9240, 4752, -1485, 220,
    ],
    [
        715, -5148, 17160, -34320, 45045, -40040, 24024, -9360, 2145, -220,
    ],
    [
        -220, 2145, -9360, 24024, -40040, 45045, -34320, 17160, -5148, 715,
    ],
    [
        2002, -15015, 51480, -105_105, 140_140, -126_126, 76440, -30030, 6930, -715,
    ],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage1ExecutionMode {
    Prover,
    Verifier,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1Params {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1KernelPlan {
    pub symbol: &'static str,
    pub relation: &'static str,
    pub kind: &'static str,
    pub backend: &'static str,
    pub abi: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1SumcheckClaimPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub domain: &'static str,
    pub num_rounds: usize,
    pub degree: usize,
    pub claim: &'static str,
    pub kernel: &'static str,
    pub claim_value: &'static str,
    pub input_openings: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1SumcheckBatchPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [&'static str],
    pub claim_operands: &'static [&'static str],
    pub claim_label: &'static str,
    pub round_label: &'static str,
    pub round_schedule: &'static [usize],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1SumcheckDriverPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub kernel: &'static str,
    pub batch: &'static str,
    pub policy: &'static str,
    pub round_schedule: &'static [usize],
    pub claim_label: &'static str,
    pub round_label: &'static str,
    pub num_rounds: usize,
    pub degree: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1SumcheckInstanceResultPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub claim: &'static str,
    pub relation: &'static str,
    pub index: usize,
    pub point_arity: usize,
    pub num_rounds: usize,
    pub round_offset: usize,
    pub point_order: &'static str,
    pub degree: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1SumcheckEvalPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub name: &'static str,
    pub index: usize,
    pub oracle: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1OpeningClaimPlan {
    pub symbol: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
    pub point_source: &'static str,
    pub eval_source: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1OpeningBatchPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [&'static str],
    pub claim_operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1TranscriptSqueezePlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub kind: &'static str,
    pub count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1CpuProgramPlan {
    pub params: Stage1Params,
    pub transcript_squeezes: &'static [Stage1TranscriptSqueezePlan],
    pub kernels: &'static [Stage1KernelPlan],
    pub claims: &'static [Stage1SumcheckClaimPlan],
    pub batches: &'static [Stage1SumcheckBatchPlan],
    pub drivers: &'static [Stage1SumcheckDriverPlan],
    pub instance_results: &'static [Stage1SumcheckInstanceResultPlan],
    pub evals: &'static [Stage1SumcheckEvalPlan],
    pub opening_claims: &'static [Stage1OpeningClaimPlan],
    pub opening_batches: &'static [Stage1OpeningBatchPlan],
}

impl Stage1CpuProgramPlan {
    pub fn evals_for_driver<'a>(
        &'a self,
        driver: &'a str,
    ) -> impl Iterator<Item = &'a Stage1SumcheckEvalPlan> + 'a {
        self.evals.iter().filter(move |eval| eval.source == driver)
    }

    pub fn instance_results_for_driver<'a>(
        &'a self,
        driver: &'a str,
    ) -> impl Iterator<Item = &'a Stage1SumcheckInstanceResultPlan> + 'a {
        self.instance_results
            .iter()
            .filter(move |instance| instance.source == driver)
    }
}

#[derive(Clone, Debug)]
pub struct Stage1NamedEval<F: Field> {
    pub name: &'static str,
    pub oracle: &'static str,
    pub value: F,
}

#[derive(Clone, Debug)]
pub struct Stage1SumcheckOutput<F: Field> {
    pub driver: &'static str,
    pub point: Vec<F>,
    pub evals: Vec<Stage1NamedEval<F>>,
    pub proof: SumcheckProof<F>,
}

#[derive(Clone, Debug)]
pub struct Stage1ChallengeVector<F: Field> {
    pub symbol: &'static str,
    pub values: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct Stage1OpeningValue<F: Field> {
    pub symbol: &'static str,
    pub oracle: &'static str,
    pub point: Vec<F>,
    pub eval: F,
}

#[derive(Clone, Debug)]
pub struct Stage1ExecutionArtifacts<F: Field> {
    pub challenge_vectors: Vec<Stage1ChallengeVector<F>>,
    pub sumchecks: Vec<Stage1SumcheckOutput<F>>,
    pub opening_values: Vec<Stage1OpeningValue<F>>,
    pub opening_batches: Vec<&'static Stage1OpeningBatchPlan>,
}

impl<F: Field> Default for Stage1ExecutionArtifacts<F> {
    fn default() -> Self {
        Self {
            challenge_vectors: Vec::new(),
            sumchecks: Vec::new(),
            opening_values: Vec::new(),
            opening_batches: Vec::new(),
        }
    }
}

impl<F: Field> Stage1ExecutionArtifacts<F> {
    pub fn opening_value(&self, symbol: &str) -> Option<&Stage1OpeningValue<F>> {
        self.opening_values
            .iter()
            .find(|opening| opening.symbol == symbol)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stage1OracleData<'a, F: Field> {
    pub name: &'static str,
    pub evaluations: &'a [F],
}

#[derive(Clone, Copy, Debug)]
pub struct Stage1OuterRemainingContext<'a, F: Field> {
    pub tau: &'a [F],
    pub r0: F,
}

pub type Stage1RemainingRoundProof<F> = Result<(Vec<F>, Vec<UnivariatePoly<F>>), Stage1KernelError>;

pub trait Stage1OuterRemainingEvaluator<F: Field>: Sync {
    fn evaluate(&self, context: Stage1OuterRemainingContext<'_, F>, point: &[F]) -> F;

    fn uniskip_extended_evals(&self, _tau: &[F]) -> Option<Vec<F>> {
        None
    }

    fn evaluate_virtual_oracle(
        &self,
        _context: Stage1OuterRemainingContext<'_, F>,
        _oracle: &str,
        _point: &[F],
    ) -> Option<F> {
        None
    }

    fn evaluate_virtual_oracles(
        &self,
        context: Stage1OuterRemainingContext<'_, F>,
        oracles: &[&str],
        point: &[F],
    ) -> Option<Vec<F>> {
        oracles
            .iter()
            .map(|oracle| self.evaluate_virtual_oracle(context, oracle, point))
            .collect()
    }

    fn prove_remaining_rounds(
        &self,
        _context: Stage1OuterRemainingContext<'_, F>,
        _num_rounds: usize,
        _batching_coeff: F,
        _initial_claim: F,
        _observe_round: &mut dyn FnMut(&UnivariatePoly<F>) -> F,
    ) -> Option<Stage1RemainingRoundProof<F>> {
        None
    }
}

#[derive(Clone, Copy)]
pub struct Stage1ProverInputs<'a, F: Field> {
    pub trace_num_vars: usize,
    pub virtual_oracles: &'a [Stage1OracleData<'a, F>],
    pub uniskip_extended_evals: Option<&'a [F]>,
    pub outer_remaining_evaluator: Option<&'a dyn Stage1OuterRemainingEvaluator<F>>,
}

impl<'a, F: Field> Stage1ProverInputs<'a, F> {
    pub fn new(trace_num_vars: usize, virtual_oracles: &'a [Stage1OracleData<'a, F>]) -> Self {
        Self {
            trace_num_vars,
            virtual_oracles,
            uniskip_extended_evals: None,
            outer_remaining_evaluator: None,
        }
    }

    pub fn empty(trace_num_vars: usize) -> Self {
        Self {
            trace_num_vars,
            virtual_oracles: &[],
            uniskip_extended_evals: None,
            outer_remaining_evaluator: None,
        }
    }

    pub fn with_uniskip_extended_evals(mut self, evaluations: &'a [F]) -> Self {
        self.uniskip_extended_evals = Some(evaluations);
        self
    }

    pub fn with_outer_remaining_evaluator(
        mut self,
        evaluator: &'a dyn Stage1OuterRemainingEvaluator<F>,
    ) -> Self {
        self.outer_remaining_evaluator = Some(evaluator);
        self
    }

    pub fn oracle(&self, name: &str) -> Option<&'a [F]> {
        self.virtual_oracles
            .iter()
            .find(|oracle| oracle.name == name)
            .map(|oracle| oracle.evaluations)
    }
}

#[derive(Debug)]
pub struct Stage1OuterR1csData<'a, F: Field> {
    pub key: &'a R1csKey<F>,
    pub witness: &'a [F],
    row_dots: R1csRowDotTable<F>,
}

impl<'a, F: Field> Stage1OuterR1csData<'a, F> {
    #[tracing::instrument(skip_all, name = "Stage1OuterR1csData::new")]
    pub fn new(key: &'a R1csKey<F>, witness: &'a [F]) -> Result<Self, Stage1KernelError> {
        let expected = key.num_cycles * key.num_vars_padded;
        if witness.len() != expected {
            return Err(Stage1KernelError::InvalidInputLength {
                input: "r1cs_witness",
                expected,
                actual: witness.len(),
            });
        }
        Ok(Self {
            key,
            witness,
            row_dots: R1csRowDotTable::compute_ab_prefix(key, witness, OUTER_EQ_CONSTRAINT_ROWS),
        })
    }

    fn witness_evals(&self, cycle_point: &[F]) -> Vec<F> {
        assert_eq!(
            cycle_point.len(),
            self.key.num_cycle_vars(),
            "stage1 cycle point dimension mismatch"
        );
        if let Some(cycle) = boolean_index(cycle_point) {
            let base = cycle * self.key.num_vars_padded;
            return self.witness[base..base + self.key.matrices.num_vars].to_vec();
        }
        (0..self.key.matrices.num_vars)
            .map(|variable| self.witness_eval(variable, cycle_point))
            .collect()
    }

    fn witness_eval(&self, variable: usize, cycle_point: &[F]) -> F {
        if let Some(cycle) = boolean_index(cycle_point) {
            return self.witness[cycle * self.key.num_vars_padded + variable];
        }
        let eq = EqPolynomial::new(cycle_point.to_vec()).evaluations();
        eq.iter()
            .take(self.key.num_cycles)
            .enumerate()
            .fold(F::zero(), |acc, (cycle, &weight)| {
                acc + weight * self.witness[cycle * self.key.num_vars_padded + variable]
            })
    }

    fn witness_evals_for_variables(&self, variables: &[usize], cycle_point: &[F]) -> Vec<F> {
        if let Some(cycle) = boolean_index(cycle_point) {
            let base = cycle * self.key.num_vars_padded;
            return variables
                .iter()
                .map(|&variable| self.witness[base + variable])
                .collect();
        }

        let eq = EqPolynomial::new(cycle_point.to_vec()).evaluations();
        let accumulators = eq
            .par_iter()
            .take(self.key.num_cycles)
            .enumerate()
            .fold(
                || vec![F::Accumulator::default(); variables.len()],
                |mut local, (cycle, &weight)| {
                    let base = cycle * self.key.num_vars_padded;
                    for (accumulator, &variable) in local.iter_mut().zip(variables) {
                        accumulator.fmadd(weight, self.witness[base + variable]);
                    }
                    local
                },
            )
            .reduce(
                || vec![F::Accumulator::default(); variables.len()],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        left.merge(right);
                    }
                    left
                },
            );
        accumulators
            .into_iter()
            .map(FieldAccumulator::reduce)
            .collect()
    }

    fn inner_sum_product(&self, r_stream: F, r0: F, witness_evals: &[F]) -> F {
        let weights = lagrange_evals(OUTER_UNISKIP_BASE_START, OUTER_UNISKIP_DOMAIN_SIZE, r0);
        let (az_g0, bz_g0) = self.group_matvecs(&OUTER_FIRST_GROUP_ROWS, &weights, witness_evals);
        let (az_g1, bz_g1) = self.group_matvecs(&OUTER_SECOND_GROUP_ROWS, &weights, witness_evals);
        let az = az_g0 + r_stream * (az_g1 - az_g0);
        let bz = bz_g0 + r_stream * (bz_g1 - bz_g0);
        az * bz
    }

    fn group_matvecs(&self, rows: &[usize], weights: &[F], witness_evals: &[F]) -> (F, F) {
        let mut az = F::zero();
        let mut bz = F::zero();
        for (&row, &weight) in rows.iter().zip(weights.iter()) {
            az += weight * Self::row_dot(&self.key.matrices.a[row], witness_evals);
            bz += weight * Self::row_dot(&self.key.matrices.b[row], witness_evals);
        }
        (az, bz)
    }

    fn group_matvecs_from_dots(
        rows: &[usize],
        weights: &[F],
        dots: R1csRowDotSlice<'_, F>,
    ) -> (F, F) {
        let mut az = F::zero();
        let mut bz = F::zero();
        for (&row, &weight) in rows.iter().zip(weights.iter()) {
            az += weight * dots.a[row];
            bz += weight * dots.b[row];
        }
        (az, bz)
    }

    #[cfg(test)]
    fn group_matvecs_from_integer_coeffs(
        rows: &[usize],
        coefficients: &[i64],
        coefficient_fields: &[F],
        dots: R1csRowDotSlice<'_, F>,
    ) -> (F, F) {
        let mut az = F::zero();
        let mut bz = F::zero();
        for ((&row, &coefficient), &coefficient_field) in rows
            .iter()
            .zip(coefficients.iter())
            .zip(coefficient_fields.iter())
        {
            if coefficient == 0 {
                continue;
            }
            let a = dots.a[row];
            if a == F::one() {
                az += coefficient_field;
            } else if a != F::zero() {
                az += a.mul_i64(coefficient);
            }

            let b = dots.b[row];
            if b != F::zero() {
                bz += b.mul_i64(coefficient);
            }
        }
        (az, bz)
    }

    fn group_matvecs_all_uniskip_targets(
        rows: &[usize],
        target_coeff_fields: &[[F; OUTER_UNISKIP_DOMAIN_SIZE]; OUTER_UNISKIP_DEGREE],
        dots: R1csRowDotSlice<'_, F>,
    ) -> [(F, F); OUTER_UNISKIP_DEGREE] {
        let mut az = [F::zero(); OUTER_UNISKIP_DEGREE];
        let mut bz = [F::zero(); OUTER_UNISKIP_DEGREE];
        for (position, &row) in rows.iter().enumerate() {
            let a = dots.a[row];
            let b = dots.b[row];
            if a == F::one() {
                for target in 0..OUTER_UNISKIP_DEGREE {
                    az[target] += target_coeff_fields[target][position];
                }
            } else if a != F::zero() {
                for target in 0..OUTER_UNISKIP_DEGREE {
                    az[target] += a.mul_i64(OUTER_UNISKIP_TARGET_COEFFS[target][position]);
                }
            }
            if b != F::zero() {
                for target in 0..OUTER_UNISKIP_DEGREE {
                    bz[target] += b.mul_i64(OUTER_UNISKIP_TARGET_COEFFS[target][position]);
                }
            }
        }
        core::array::from_fn(|target| (az[target], bz[target]))
    }

    fn row_dot(row: &[(usize, F)], witness_evals: &[F]) -> F {
        let mut acc = F::zero();
        for &(variable, coefficient) in row {
            acc += coefficient * witness_evals[variable];
        }
        acc
    }

    fn remaining_cycle_point(point: &[F]) -> Vec<F> {
        point[1..].iter().rev().copied().collect()
    }

    #[tracing::instrument(skip_all, name = "Stage1OuterR1csData::dense_outer_state")]
    fn dense_outer_state(
        &self,
        context: Stage1OuterRemainingContext<'_, F>,
        num_rounds: usize,
        batching_coeff: F,
    ) -> DenseOuterState<F> {
        let tau_high = context.tau[context.tau.len() - 1];
        let tau_low = &context.tau[..context.tau.len() - 1];
        let lagrange_tau_r0 = lagrange_kernel_eval(
            OUTER_UNISKIP_BASE_START,
            OUTER_UNISKIP_DOMAIN_SIZE,
            tau_high,
            context.r0,
        );
        let weights = lagrange_evals(
            OUTER_UNISKIP_BASE_START,
            OUTER_UNISKIP_DOMAIN_SIZE,
            context.r0,
        );
        let len = 1usize << num_rounds;
        let scale = lagrange_tau_r0 * batching_coeff;
        let eq_evals = EqPolynomial::new(tau_low.to_vec()).evaluations();
        let mut eq = vec![F::zero(); len];
        let mut az = vec![F::zero(); len];
        let mut bz = vec![F::zero(); len];
        eq.par_chunks_mut(2)
            .zip(az.par_chunks_mut(2))
            .zip(bz.par_chunks_mut(2))
            .enumerate()
            .for_each(|(cycle, ((eq_pair, az_pair), bz_pair))| {
                let index = cycle << 1;
                let dots = self.row_dots.cycle(cycle);
                let (az_g0, bz_g0) =
                    Self::group_matvecs_from_dots(&OUTER_FIRST_GROUP_ROWS, &weights, dots);
                let (az_g1, bz_g1) =
                    Self::group_matvecs_from_dots(&OUTER_SECOND_GROUP_ROWS, &weights, dots);
                eq_pair[0] = eq_evals[index] * scale;
                az_pair[0] = az_g0;
                bz_pair[0] = bz_g0;
                eq_pair[1] = eq_evals[index + 1] * scale;
                az_pair[1] = az_g1;
                bz_pair[1] = bz_g1;
            });
        DenseOuterState {
            eq,
            az,
            bz,
            eq_scratch: Vec::with_capacity(len / 2),
            az_scratch: Vec::with_capacity(len / 2),
            bz_scratch: Vec::with_capacity(len / 2),
        }
    }
}

impl<F: Field> Stage1OuterRemainingEvaluator<F> for Stage1OuterR1csData<'_, F> {
    fn evaluate(&self, context: Stage1OuterRemainingContext<'_, F>, point: &[F]) -> F {
        assert_eq!(
            point.len(),
            self.key.num_cycle_vars() + 1,
            "stage1 outer remaining point dimension mismatch"
        );
        assert_eq!(
            context.tau.len(),
            self.key.num_cycle_vars() + 2,
            "stage1 tau dimension mismatch"
        );
        let tau_high = context.tau[context.tau.len() - 1];
        let tau_low = &context.tau[..context.tau.len() - 1];
        let mut point_reversed = point.to_vec();
        point_reversed.reverse();
        let tau_weight = EqPolynomial::<F>::mle(tau_low, &point_reversed);
        let lagrange_tau_r0 = lagrange_kernel_eval(
            OUTER_UNISKIP_BASE_START,
            OUTER_UNISKIP_DOMAIN_SIZE,
            tau_high,
            context.r0,
        );
        let cycle_point = Self::remaining_cycle_point(point);
        let witness_evals = self.witness_evals(&cycle_point);
        lagrange_tau_r0 * tau_weight * self.inner_sum_product(point[0], context.r0, &witness_evals)
    }

    #[tracing::instrument(skip_all, name = "Stage1OuterR1csData::uniskip_extended_evals")]
    fn uniskip_extended_evals(&self, tau: &[F]) -> Option<Vec<F>> {
        if tau.len() != self.key.num_cycle_vars() + 2 {
            return None;
        }
        let tau_low = &tau[..tau.len() - 1];
        let num_rounds = self.key.num_cycle_vars() + 1;
        let eq_evals = EqPolynomial::new(tau_low.to_vec()).evaluations();
        let num_cycles = 1usize << (num_rounds - 1);
        assert_eq!(self.row_dots.cycle_count(), num_cycles);
        let target_coeff_fields =
            OUTER_UNISKIP_TARGET_COEFFS.map(|coefficients| coefficients.map(F::from_i64));
        let accumulators = (0..num_cycles)
            .into_par_iter()
            .fold(
                || [F::Accumulator::default(); OUTER_UNISKIP_DEGREE],
                |mut local, cycle| {
                    let dots = self.row_dots.cycle(cycle);
                    let index = cycle << 1;
                    let first_group = Self::group_matvecs_all_uniskip_targets(
                        &OUTER_FIRST_GROUP_ROWS,
                        &target_coeff_fields,
                        dots,
                    );
                    let second_group = Self::group_matvecs_all_uniskip_targets(
                        &OUTER_SECOND_GROUP_ROWS,
                        &target_coeff_fields,
                        dots,
                    );
                    for target in 0..OUTER_UNISKIP_DEGREE {
                        let (az_g0, bz_g0) = first_group[target];
                        let (az_g1, bz_g1) = second_group[target];
                        local[target].fmadd(eq_evals[index], az_g0 * bz_g0);
                        local[target].fmadd(eq_evals[index + 1], az_g1 * bz_g1);
                    }
                    local
                },
            )
            .reduce(
                || [F::Accumulator::default(); OUTER_UNISKIP_DEGREE],
                |mut left, right| {
                    for target in 0..OUTER_UNISKIP_DEGREE {
                        left[target].merge(right[target]);
                    }
                    left
                },
            );
        let extended_evals = accumulators.map(FieldAccumulator::reduce).to_vec();
        Some(extended_evals)
    }

    fn evaluate_virtual_oracle(
        &self,
        _context: Stage1OuterRemainingContext<'_, F>,
        oracle: &str,
        point: &[F],
    ) -> Option<F> {
        let variable = r1cs_oracle_variable(oracle)?;
        let cycle_point = Self::remaining_cycle_point(point);
        Some(self.witness_eval(variable, &cycle_point))
    }

    #[tracing::instrument(skip_all, name = "Stage1OuterR1csData::evaluate_virtual_oracles")]
    fn evaluate_virtual_oracles(
        &self,
        _context: Stage1OuterRemainingContext<'_, F>,
        oracles: &[&str],
        point: &[F],
    ) -> Option<Vec<F>> {
        let variables = oracles
            .iter()
            .map(|oracle| r1cs_oracle_variable(oracle))
            .collect::<Option<Vec<_>>>()?;
        let cycle_point = Self::remaining_cycle_point(point);
        Some(self.witness_evals_for_variables(&variables, &cycle_point))
    }

    #[tracing::instrument(skip_all, name = "Stage1OuterR1csData::prove_remaining_rounds")]
    fn prove_remaining_rounds(
        &self,
        context: Stage1OuterRemainingContext<'_, F>,
        num_rounds: usize,
        batching_coeff: F,
        initial_claim: F,
        observe_round: &mut dyn FnMut(&UnivariatePoly<F>) -> F,
    ) -> Option<Stage1RemainingRoundProof<F>> {
        let mut state = self.dense_outer_state(context, num_rounds, batching_coeff);
        let mut running_sum = initial_claim * batching_coeff;
        let mut point = Vec::with_capacity(num_rounds);
        let mut round_polynomials = Vec::with_capacity(num_rounds);

        for _round in 0..num_rounds {
            let poly = state.round_poly();
            if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != running_sum {
                return Some(Err(Stage1KernelError::InvalidProof {
                    driver: "stage1.outer.remaining",
                    reason: "dense outer remaining claim mismatch",
                }));
            }
            let challenge = observe_round(&poly);
            running_sum = poly.evaluate(challenge);
            state.bind(challenge);
            point.push(challenge);
            round_polynomials.push(poly);
        }
        Some(Ok((point, round_polynomials)))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stage1Proof<F: Field> {
    pub sumchecks: Vec<Stage1SumcheckOutput<F>>,
}

impl<F: Field> From<Stage1ExecutionArtifacts<F>> for Stage1Proof<F> {
    fn from(artifacts: Stage1ExecutionArtifacts<F>) -> Self {
        Self {
            sumchecks: artifacts.sumchecks,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stage1KernelContext<'a> {
    pub mode: Stage1ExecutionMode,
    pub program: &'a Stage1CpuProgramPlan,
    pub kernel: &'a Stage1KernelPlan,
    pub batch: &'a Stage1SumcheckBatchPlan,
    pub driver: &'a Stage1SumcheckDriverPlan,
}

pub trait Stage1KernelExecutor<F: Field> {
    fn observe_challenge_vector(
        &mut self,
        _plan: &'static Stage1TranscriptSqueezePlan,
        _values: &[F],
    ) -> Result<(), Stage1KernelError> {
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        _output: &Stage1SumcheckOutput<F>,
    ) -> Result<(), Stage1KernelError> {
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage1KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage1SumcheckOutput<F>, Stage1KernelError>
    where
        T: Transcript<Challenge = F>;

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage1KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage1SumcheckOutput<F>, Stage1KernelError>
    where
        T: Transcript<Challenge = F>;
}

#[derive(Clone, Debug, Default)]
pub struct UnsupportedStage1KernelExecutor;

impl<F: Field> Stage1KernelExecutor<F> for UnsupportedStage1KernelExecutor {
    fn prove_sumcheck<T>(
        &mut self,
        context: Stage1KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage1SumcheckOutput<F>, Stage1KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage1KernelError::KernelNotImplemented {
            abi: context.kernel.abi,
        })
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage1KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage1SumcheckOutput<F>, Stage1KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage1KernelError::KernelNotImplemented {
            abi: context.kernel.abi,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stage1ShapeKernelExecutor;

impl<F: Field> Stage1KernelExecutor<F> for Stage1ShapeKernelExecutor {
    fn prove_sumcheck<T>(
        &mut self,
        context: Stage1KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage1SumcheckOutput<F>, Stage1KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        run_shape_kernel(context, transcript)
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage1KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage1SumcheckOutput<F>, Stage1KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        run_shape_kernel(context, transcript)
    }
}

#[derive(Clone)]
pub struct Stage1ProverKernelExecutor<'a, F: Field> {
    pub inputs: Stage1ProverInputs<'a, F>,
    challenge_vectors: Vec<Stage1ChallengeVector<F>>,
    completed_sumchecks: Vec<Stage1SumcheckOutput<F>>,
}

impl<'a, F: Field> Stage1ProverKernelExecutor<'a, F> {
    pub fn new(inputs: Stage1ProverInputs<'a, F>) -> Self {
        Self {
            inputs,
            challenge_vectors: Vec::new(),
            completed_sumchecks: Vec::new(),
        }
    }
}

impl<F: Field> Stage1KernelExecutor<F> for Stage1ProverKernelExecutor<'_, F> {
    fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage1TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage1KernelError> {
        self.challenge_vectors.push(Stage1ChallengeVector {
            symbol: plan.symbol,
            values: values.to_vec(),
        });
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        output: &Stage1SumcheckOutput<F>,
    ) -> Result<(), Stage1KernelError> {
        self.completed_sumchecks.push(output.clone());
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage1KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage1SumcheckOutput<F>, Stage1KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        prove_stage1_kernel(
            context,
            &self.inputs,
            &self.challenge_vectors,
            &self.completed_sumchecks,
            transcript,
        )
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage1KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage1SumcheckOutput<F>, Stage1KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage1KernelError::WrongExecutorMode {
            driver: context.driver.symbol,
            expected: Stage1ExecutionMode::Prover,
            actual: Stage1ExecutionMode::Verifier,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Stage1VerifierKernelExecutor<'a, F: Field> {
    pub proof: &'a Stage1Proof<F>,
    pub cursor: usize,
    challenge_vectors: Vec<Stage1ChallengeVector<F>>,
    completed_sumchecks: Vec<Stage1SumcheckOutput<F>>,
}

impl<'a, F: Field> Stage1VerifierKernelExecutor<'a, F> {
    pub fn new(proof: &'a Stage1Proof<F>) -> Self {
        Self {
            proof,
            cursor: 0,
            challenge_vectors: Vec::new(),
            completed_sumchecks: Vec::new(),
        }
    }
}

impl<F: Field> Stage1KernelExecutor<F> for Stage1VerifierKernelExecutor<'_, F> {
    fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage1TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage1KernelError> {
        self.challenge_vectors.push(Stage1ChallengeVector {
            symbol: plan.symbol,
            values: values.to_vec(),
        });
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        output: &Stage1SumcheckOutput<F>,
    ) -> Result<(), Stage1KernelError> {
        self.completed_sumchecks.push(output.clone());
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage1KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage1SumcheckOutput<F>, Stage1KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage1KernelError::WrongExecutorMode {
            driver: context.driver.symbol,
            expected: Stage1ExecutionMode::Verifier,
            actual: Stage1ExecutionMode::Prover,
        })
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage1KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage1SumcheckOutput<F>, Stage1KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        let proof = self.proof.sumchecks.get(self.cursor);
        self.cursor += usize::from(proof.is_some());
        verify_stage1_kernel(
            context,
            proof,
            &self.challenge_vectors,
            &self.completed_sumchecks,
            transcript,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage1KernelError {
    MissingKernel {
        driver: &'static str,
        kernel: &'static str,
    },
    MissingBatch {
        driver: &'static str,
        batch: &'static str,
    },
    PlanCountMismatch {
        artifact: &'static str,
        expected: usize,
        actual: usize,
    },
    KernelNotImplemented {
        abi: &'static str,
    },
    WrongExecutorMode {
        driver: &'static str,
        expected: Stage1ExecutionMode,
        actual: Stage1ExecutionMode,
    },
    MissingProof {
        driver: &'static str,
    },
    MissingKernelInput {
        kernel: &'static str,
        input: &'static str,
    },
    InvalidInputLength {
        input: &'static str,
        expected: usize,
        actual: usize,
    },
    UnsupportedPointOrder {
        symbol: &'static str,
        point_order: &'static str,
    },
    InvalidProof {
        driver: &'static str,
        reason: &'static str,
    },
}

impl Display for Stage1KernelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingKernel { driver, kernel } => {
                write!(
                    formatter,
                    "stage1 driver @{driver} references missing kernel @{kernel}"
                )
            }
            Self::MissingBatch { driver, batch } => {
                write!(
                    formatter,
                    "stage1 driver @{driver} references missing batch @{batch}"
                )
            }
            Self::PlanCountMismatch {
                artifact,
                expected,
                actual,
            } => write!(
                formatter,
                "stage1 plan @{artifact} count mismatch: expected {expected}, got {actual}"
            ),
            Self::KernelNotImplemented { abi } => {
                write!(formatter, "stage1 kernel ABI `{abi}` is not implemented")
            }
            Self::WrongExecutorMode {
                driver,
                expected,
                actual,
            } => write!(
                formatter,
                "stage1 driver @{driver} ran with {actual:?} executor path, expected {expected:?}"
            ),
            Self::MissingProof { driver } => {
                write!(
                    formatter,
                    "stage1 verifier missing proof for driver @{driver}"
                )
            }
            Self::MissingKernelInput { kernel, input } => {
                write!(
                    formatter,
                    "stage1 kernel `{kernel}` missing input `{input}`"
                )
            }
            Self::InvalidInputLength {
                input,
                expected,
                actual,
            } => write!(
                formatter,
                "stage1 input `{input}` length mismatch: expected {expected}, got {actual}"
            ),
            Self::UnsupportedPointOrder {
                symbol,
                point_order,
            } => write!(
                formatter,
                "stage1 instance @{symbol} uses unsupported point order `{point_order}`"
            ),
            Self::InvalidProof { driver, reason } => {
                write!(
                    formatter,
                    "stage1 proof for driver @{driver} is invalid: {reason}"
                )
            }
        }
    }
}

impl Error for Stage1KernelError {}

pub fn execute_stage1_program<F, E, T>(
    program: &'static Stage1CpuProgramPlan,
    mode: Stage1ExecutionMode,
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage1ExecutionArtifacts<F>, Stage1KernelError>
where
    F: Field,
    E: Stage1KernelExecutor<F>,
    T: Transcript<Challenge = F>,
{
    verify_static_program_shape(program)?;
    let mut artifacts = Stage1ExecutionArtifacts::default();
    for squeeze in program.transcript_squeezes {
        let values = transcript.challenge_vector(squeeze.count);
        executor.observe_challenge_vector(squeeze, &values)?;
        artifacts.challenge_vectors.push(Stage1ChallengeVector {
            symbol: squeeze.symbol,
            values,
        });
    }
    for driver in program.drivers {
        let kernel =
            find_kernel(program, driver.kernel).ok_or(Stage1KernelError::MissingKernel {
                driver: driver.symbol,
                kernel: driver.kernel,
            })?;
        let batch = find_batch(program, driver.batch).ok_or(Stage1KernelError::MissingBatch {
            driver: driver.symbol,
            batch: driver.batch,
        })?;
        let context = Stage1KernelContext {
            mode,
            program,
            kernel,
            batch,
            driver,
        };
        let output = match mode {
            Stage1ExecutionMode::Prover => executor.prove_sumcheck(context, transcript)?,
            Stage1ExecutionMode::Verifier => executor.verify_sumcheck(context, transcript)?,
        };
        executor.observe_sumcheck_output(&output)?;
        artifacts.sumchecks.push(output);
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    artifacts.opening_values = stage1_opening_values(program, &artifacts.sumchecks)?;
    Ok(artifacts)
}

fn stage1_opening_values<F: Field>(
    program: &'static Stage1CpuProgramPlan,
    sumchecks: &[Stage1SumcheckOutput<F>],
) -> Result<Vec<Stage1OpeningValue<F>>, Stage1KernelError> {
    let mut points = Vec::<Stage1PointValue<F>>::new();
    let mut scalars = Vec::<Stage1ScalarValue<F>>::new();
    for output in sumchecks {
        points.push(Stage1PointValue {
            symbol: output.driver,
            point: output.point.clone(),
        });
        for instance in program.instance_results_for_driver(output.driver) {
            points.push(Stage1PointValue {
                symbol: instance.symbol,
                point: stage1_instance_point(instance, &output.point)?,
            });
        }
        for eval in program.evals_for_driver(output.driver) {
            let value = output
                .evals
                .iter()
                .find(|value| value.name == eval.name)
                .or_else(|| output.evals.get(eval.index))
                .ok_or(Stage1KernelError::MissingKernelInput {
                    kernel: output.driver,
                    input: eval.symbol,
                })?
                .value;
            scalars.push(Stage1ScalarValue {
                symbol: eval.symbol,
                value,
            });
            scalars.push(Stage1ScalarValue {
                symbol: eval.name,
                value,
            });
        }
    }
    program
        .opening_claims
        .iter()
        .map(|claim| {
            let point = points
                .iter()
                .find(|point| point.symbol == claim.point_source)
                .ok_or(Stage1KernelError::MissingKernelInput {
                    kernel: claim.symbol,
                    input: claim.point_source,
                })?
                .point
                .clone();
            let eval = scalars
                .iter()
                .find(|scalar| scalar.symbol == claim.eval_source)
                .ok_or(Stage1KernelError::MissingKernelInput {
                    kernel: claim.symbol,
                    input: claim.eval_source,
                })?
                .value;
            Ok(Stage1OpeningValue {
                symbol: claim.symbol,
                oracle: claim.oracle,
                point,
                eval,
            })
        })
        .collect()
}

fn stage1_instance_point<F: Field>(
    instance: &Stage1SumcheckInstanceResultPlan,
    point: &[F],
) -> Result<Vec<F>, Stage1KernelError> {
    let end = instance.round_offset + instance.point_arity;
    let mut point = point
        .get(instance.round_offset..end)
        .ok_or(Stage1KernelError::InvalidInputLength {
            input: instance.symbol,
            expected: end,
            actual: point.len(),
        })?
        .to_vec();
    match instance.point_order {
        "as_is" => Ok(point),
        "reverse" => {
            point.reverse();
            Ok(point)
        }
        point_order => Err(Stage1KernelError::UnsupportedPointOrder {
            symbol: instance.symbol,
            point_order,
        }),
    }
}

#[derive(Clone, Debug)]
struct Stage1PointValue<F: Field> {
    symbol: &'static str,
    point: Vec<F>,
}

#[derive(Clone, Copy, Debug)]
struct Stage1ScalarValue<F: Field> {
    symbol: &'static str,
    value: F,
}

fn verify_static_program_shape(
    program: &'static Stage1CpuProgramPlan,
) -> Result<(), Stage1KernelError> {
    for batch in program.batches {
        verify_count(batch.symbol, batch.count, batch.ordered_claims.len())?;
        verify_count(batch.symbol, batch.count, batch.claim_operands.len())?;
    }
    for batch in program.opening_batches {
        verify_count(batch.symbol, batch.count, batch.ordered_claims.len())?;
        verify_count(batch.symbol, batch.count, batch.claim_operands.len())?;
    }
    Ok(())
}

fn verify_count(
    artifact: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), Stage1KernelError> {
    if expected == actual {
        Ok(())
    } else {
        Err(Stage1KernelError::PlanCountMismatch {
            artifact,
            expected,
            actual,
        })
    }
}

fn find_kernel(
    program: &'static Stage1CpuProgramPlan,
    symbol: &str,
) -> Option<&'static Stage1KernelPlan> {
    program
        .kernels
        .iter()
        .find(|kernel| kernel.symbol == symbol)
}

fn find_batch(
    program: &'static Stage1CpuProgramPlan,
    symbol: &str,
) -> Option<&'static Stage1SumcheckBatchPlan> {
    program.batches.iter().find(|batch| batch.symbol == symbol)
}

fn prove_stage1_kernel<F, T>(
    context: Stage1KernelContext<'_>,
    inputs: &Stage1ProverInputs<'_, F>,
    challenge_vectors: &[Stage1ChallengeVector<F>],
    completed_sumchecks: &[Stage1SumcheckOutput<F>],
    transcript: &mut T,
) -> Result<Stage1SumcheckOutput<F>, Stage1KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    match context.kernel.abi {
        "jolt_stage1_outer_uniskip" => {
            prove_outer_uniskip(context, inputs, challenge_vectors, transcript)
        }
        "jolt_stage1_outer_remaining" => prove_outer_remaining(
            context,
            inputs,
            challenge_vectors,
            completed_sumchecks,
            transcript,
        ),
        abi => Err(Stage1KernelError::KernelNotImplemented { abi }),
    }
}

fn verify_stage1_kernel<F, T>(
    context: Stage1KernelContext<'_>,
    proof: Option<&Stage1SumcheckOutput<F>>,
    challenge_vectors: &[Stage1ChallengeVector<F>],
    completed_sumchecks: &[Stage1SumcheckOutput<F>],
    transcript: &mut T,
) -> Result<Stage1SumcheckOutput<F>, Stage1KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    match context.kernel.abi {
        "jolt_stage1_outer_uniskip" => {
            verify_outer_uniskip(context, proof, challenge_vectors, transcript)
        }
        "jolt_stage1_outer_remaining" => {
            verify_outer_remaining(context, proof, completed_sumchecks, transcript)
        }
        abi => Err(Stage1KernelError::KernelNotImplemented { abi }),
    }
}

#[tracing::instrument(skip_all, name = "prove_outer_uniskip")]
fn prove_outer_uniskip<F, T>(
    context: Stage1KernelContext<'_>,
    inputs: &Stage1ProverInputs<'_, F>,
    challenge_vectors: &[Stage1ChallengeVector<F>],
    transcript: &mut T,
) -> Result<Stage1SumcheckOutput<F>, Stage1KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let tau = find_challenge_vector(challenge_vectors, "stage1.tau").ok_or(
        Stage1KernelError::MissingKernelInput {
            kernel: context.kernel.abi,
            input: "stage1.tau",
        },
    )?;
    let tau_high = tau
        .last()
        .copied()
        .ok_or(Stage1KernelError::InvalidInputLength {
            input: "stage1.tau",
            expected: 1,
            actual: 0,
        })?;
    let owned_extended_evals;
    let extended_evals = if let Some(extended_evals) = inputs.uniskip_extended_evals {
        extended_evals
    } else {
        let evaluator =
            inputs
                .outer_remaining_evaluator
                .ok_or(Stage1KernelError::MissingKernelInput {
                    kernel: context.kernel.abi,
                    input: "uniskip_extended_evals",
                })?;
        owned_extended_evals =
            evaluator
                .uniskip_extended_evals(tau)
                .ok_or(Stage1KernelError::MissingKernelInput {
                    kernel: context.kernel.abi,
                    input: "uniskip_extended_evals",
                })?;
        owned_extended_evals.as_slice()
    };
    let poly = build_outer_uniskip_poly(extended_evals, tau_high)?;
    append_univariate_poly(transcript, context.driver.round_label, &poly);
    let r0 = transcript.challenge();
    let eval = poly.evaluate(r0);
    append_labeled_scalar(transcript, "opening_claim", &eval);
    Ok(Stage1SumcheckOutput {
        driver: context.driver.symbol,
        point: vec![r0],
        evals: driver_evals(context, eval),
        proof: SumcheckProof {
            round_polynomials: vec![poly],
        },
    })
}

#[tracing::instrument(skip_all, name = "prove_outer_remaining")]
fn prove_outer_remaining<F, T>(
    context: Stage1KernelContext<'_>,
    inputs: &Stage1ProverInputs<'_, F>,
    challenge_vectors: &[Stage1ChallengeVector<F>],
    completed_sumchecks: &[Stage1SumcheckOutput<F>],
    transcript: &mut T,
) -> Result<Stage1SumcheckOutput<F>, Stage1KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let evaluator =
        inputs
            .outer_remaining_evaluator
            .ok_or(Stage1KernelError::MissingKernelInput {
                kernel: context.kernel.abi,
                input: "outer_remaining_evaluator",
            })?;
    let tau = find_challenge_vector(challenge_vectors, "stage1.tau").ok_or(
        Stage1KernelError::MissingKernelInput {
            kernel: context.kernel.abi,
            input: "stage1.tau",
        },
    )?;
    let (r0, input_claim) = uniskip_point_and_claim(completed_sumchecks).ok_or(
        Stage1KernelError::MissingKernelInput {
            kernel: context.kernel.abi,
            input: "stage1.uniskip.eval",
        },
    )?;
    let remaining_context = Stage1OuterRemainingContext { tau, r0 };
    append_labeled_scalar(transcript, context.driver.claim_label, &input_claim);
    let batching_coeff = transcript.challenge();
    let fast_path = evaluator.prove_remaining_rounds(
        remaining_context,
        context.driver.num_rounds,
        batching_coeff,
        input_claim,
        &mut |poly| {
            append_compressed_univariate_poly(transcript, context.driver.round_label, poly);
            transcript.challenge()
        },
    );
    let (point, round_polynomials) = if let Some(result) = fast_path {
        result?
    } else {
        prove_outer_remaining_fallback(
            context,
            evaluator,
            remaining_context,
            batching_coeff,
            input_claim,
            transcript,
        )?
    };

    let evals = remaining_driver_evals(context, evaluator, remaining_context, &point)?;
    append_opening_claims(transcript, &evals);
    Ok(Stage1SumcheckOutput {
        driver: context.driver.symbol,
        point,
        evals,
        proof: SumcheckProof { round_polynomials },
    })
}

fn verify_outer_uniskip<F, T>(
    context: Stage1KernelContext<'_>,
    proof: Option<&Stage1SumcheckOutput<F>>,
    _challenge_vectors: &[Stage1ChallengeVector<F>],
    transcript: &mut T,
) -> Result<Stage1SumcheckOutput<F>, Stage1KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let Some(proof) = proof else {
        return Err(Stage1KernelError::MissingProof {
            driver: context.driver.symbol,
        });
    };
    if proof.driver != context.driver.symbol {
        return Err(Stage1KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "driver symbol mismatch",
        });
    }
    let Some(poly) = proof.proof.round_polynomials.first() else {
        return Err(Stage1KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "missing uniskip round polynomial",
        });
    };
    if proof.proof.round_polynomials.len() != 1 {
        return Err(Stage1KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "unexpected uniskip round count",
        });
    }
    if polynomial_degree(poly) > OUTER_UNISKIP_DEGREE_BOUND {
        return Err(Stage1KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "uniskip polynomial exceeds degree bound",
        });
    }
    append_univariate_poly(transcript, context.driver.round_label, poly);
    let r0 = transcript.challenge();
    if !uniskip_sum_matches(poly, F::zero()) {
        return Err(Stage1KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "uniskip polynomial sum check failed",
        });
    }
    let eval = poly.evaluate(r0);
    append_labeled_scalar(transcript, "opening_claim", &eval);
    Ok(Stage1SumcheckOutput {
        driver: context.driver.symbol,
        point: vec![r0],
        evals: driver_evals(context, eval),
        proof: proof.proof.clone(),
    })
}

fn verify_outer_remaining<F, T>(
    context: Stage1KernelContext<'_>,
    proof: Option<&Stage1SumcheckOutput<F>>,
    completed_sumchecks: &[Stage1SumcheckOutput<F>],
    transcript: &mut T,
) -> Result<Stage1SumcheckOutput<F>, Stage1KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let Some(proof) = proof else {
        return Err(Stage1KernelError::MissingProof {
            driver: context.driver.symbol,
        });
    };
    if proof.driver != context.driver.symbol {
        return Err(Stage1KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "driver symbol mismatch",
        });
    }
    if proof.proof.round_polynomials.len() != context.driver.num_rounds {
        return Err(Stage1KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "unexpected outer remaining round count",
        });
    }
    let input_claim =
        uniskip_output_claim(completed_sumchecks).ok_or(Stage1KernelError::MissingKernelInput {
            kernel: context.kernel.abi,
            input: "stage1.uniskip.eval",
        })?;
    append_labeled_scalar(transcript, context.driver.claim_label, &input_claim);
    let batching_coeff = transcript.challenge();
    let mut running_sum = input_claim * batching_coeff;
    let mut point = Vec::with_capacity(context.driver.num_rounds);

    for poly in &proof.proof.round_polynomials {
        if polynomial_degree(poly) > OUTER_REMAINING_DEGREE_BOUND {
            return Err(Stage1KernelError::InvalidProof {
                driver: context.driver.symbol,
                reason: "outer remaining polynomial exceeds degree bound",
            });
        }
        if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != running_sum {
            return Err(Stage1KernelError::InvalidProof {
                driver: context.driver.symbol,
                reason: "outer remaining round check failed",
            });
        }
        append_compressed_univariate_poly(transcript, context.driver.round_label, poly);
        let challenge = transcript.challenge();
        running_sum = poly.evaluate(challenge);
        point.push(challenge);
    }
    if !proof.point.is_empty() && proof.point != point {
        return Err(Stage1KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "outer remaining point mismatch",
        });
    }
    append_opening_claims(transcript, &proof.evals);
    Ok(Stage1SumcheckOutput {
        driver: context.driver.symbol,
        point,
        evals: proof.evals.clone(),
        proof: proof.proof.clone(),
    })
}

fn prove_outer_remaining_fallback<F, T>(
    context: Stage1KernelContext<'_>,
    evaluator: &dyn Stage1OuterRemainingEvaluator<F>,
    remaining_context: Stage1OuterRemainingContext<'_, F>,
    batching_coeff: F,
    input_claim: F,
    transcript: &mut T,
) -> Result<(Vec<F>, Vec<UnivariatePoly<F>>), Stage1KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let mut running_sum = input_claim * batching_coeff;
    let mut point = Vec::with_capacity(context.driver.num_rounds);
    let mut round_polynomials = Vec::with_capacity(context.driver.num_rounds);
    for _round in 0..context.driver.num_rounds {
        let poly = outer_remaining_round_poly(
            evaluator,
            remaining_context,
            context.driver.num_rounds,
            &point,
        );
        let scaled_poly = scale_poly(&poly, batching_coeff);
        if scaled_poly.evaluate(F::zero()) + scaled_poly.evaluate(F::one()) != running_sum {
            return Err(Stage1KernelError::InvalidProof {
                driver: context.driver.symbol,
                reason: "outer remaining prover claim mismatch",
            });
        }
        append_compressed_univariate_poly(transcript, context.driver.round_label, &scaled_poly);
        let challenge = transcript.challenge();
        running_sum = scaled_poly.evaluate(challenge);
        point.push(challenge);
        round_polynomials.push(scaled_poly);
    }
    Ok((point, round_polynomials))
}

fn find_challenge_vector<'a, F: Field>(
    challenges: &'a [Stage1ChallengeVector<F>],
    symbol: &str,
) -> Option<&'a [F]> {
    challenges
        .iter()
        .find(|challenge| challenge.symbol == symbol)
        .map(|challenge| challenge.values.as_slice())
}

fn build_outer_uniskip_poly<F: Field>(
    extended_evals: &[F],
    tau_high: F,
) -> Result<UnivariatePoly<F>, Stage1KernelError> {
    if extended_evals.len() != OUTER_UNISKIP_DEGREE {
        return Err(Stage1KernelError::InvalidInputLength {
            input: "uniskip_extended_evals",
            expected: OUTER_UNISKIP_DEGREE,
            actual: extended_evals.len(),
        });
    }

    let mut t1_values = vec![F::zero(); OUTER_UNISKIP_EXTENDED_SIZE];
    for (value, target) in extended_evals.iter().zip(uniskip_targets()) {
        let index = (target - OUTER_UNISKIP_EXTENDED_START) as usize;
        t1_values[index] = *value;
    }

    let t1_coeffs = interpolate_to_coeffs(OUTER_UNISKIP_EXTENDED_START, &t1_values);
    let lagrange_values = lagrange_evals(
        OUTER_UNISKIP_BASE_START,
        OUTER_UNISKIP_DOMAIN_SIZE,
        tau_high,
    );
    let lagrange_coeffs = interpolate_to_coeffs(OUTER_UNISKIP_BASE_START, &lagrange_values);

    let mut coefficients = vec![F::zero(); OUTER_UNISKIP_NUM_COEFFS];
    for (i, &lagrange_coeff) in lagrange_coeffs.iter().enumerate() {
        for (j, &t1_coeff) in t1_coeffs.iter().enumerate() {
            coefficients[i + j] += lagrange_coeff * t1_coeff;
        }
    }
    Ok(UnivariatePoly::new(coefficients))
}

fn uniskip_targets() -> [i64; OUTER_UNISKIP_DEGREE] {
    let ext_left = OUTER_UNISKIP_EXTENDED_START;
    let ext_right = OUTER_UNISKIP_DEGREE as i64;
    let base_left = OUTER_UNISKIP_BASE_START;
    let base_right = base_left + OUTER_UNISKIP_DOMAIN_SIZE as i64 - 1;
    let mut targets = [0i64; OUTER_UNISKIP_DEGREE];
    let mut index = 0;
    let mut negative = base_left - 1;
    let mut positive = base_right + 1;
    while negative >= ext_left && positive <= ext_right && index < OUTER_UNISKIP_DEGREE {
        targets[index] = negative;
        index += 1;
        if index >= OUTER_UNISKIP_DEGREE {
            break;
        }
        targets[index] = positive;
        index += 1;
        negative -= 1;
        positive += 1;
    }
    while index < OUTER_UNISKIP_DEGREE && negative >= ext_left {
        targets[index] = negative;
        index += 1;
        negative -= 1;
    }
    while index < OUTER_UNISKIP_DEGREE && positive <= ext_right {
        targets[index] = positive;
        index += 1;
        positive += 1;
    }
    targets
}

fn boolean_index<F: Field>(point: &[F]) -> Option<usize> {
    let mut index = 0usize;
    for value in point {
        index <<= 1;
        if *value == F::one() {
            index |= 1;
        } else if *value != F::zero() {
            return None;
        }
    }
    Some(index)
}

fn r1cs_oracle_variable(oracle: &str) -> Option<usize> {
    match oracle {
        "LeftInstructionInput" => Some(rv64::V_LEFT_INSTRUCTION_INPUT),
        "RightInstructionInput" => Some(rv64::V_RIGHT_INSTRUCTION_INPUT),
        "Product" => Some(rv64::V_PRODUCT),
        "ShouldBranch" => Some(rv64::V_SHOULD_BRANCH),
        "PC" => Some(rv64::V_PC),
        "UnexpandedPC" => Some(rv64::V_UNEXPANDED_PC),
        "Imm" => Some(rv64::V_IMM),
        "RamAddress" => Some(rv64::V_RAM_ADDRESS),
        "Rs1Value" => Some(rv64::V_RS1_VALUE),
        "Rs2Value" => Some(rv64::V_RS2_VALUE),
        "RdWriteValue" => Some(rv64::V_RD_WRITE_VALUE),
        "RamReadValue" => Some(rv64::V_RAM_READ_VALUE),
        "RamWriteValue" => Some(rv64::V_RAM_WRITE_VALUE),
        "LeftLookupOperand" => Some(rv64::V_LEFT_LOOKUP_OPERAND),
        "RightLookupOperand" => Some(rv64::V_RIGHT_LOOKUP_OPERAND),
        "NextUnexpandedPC" => Some(rv64::V_NEXT_UNEXPANDED_PC),
        "NextPC" => Some(rv64::V_NEXT_PC),
        "NextIsVirtual" => Some(rv64::V_NEXT_IS_VIRTUAL),
        "NextIsFirstInSequence" => Some(rv64::V_NEXT_IS_FIRST_IN_SEQUENCE),
        "LookupOutput" => Some(rv64::V_LOOKUP_OUTPUT),
        "ShouldJump" => Some(rv64::V_SHOULD_JUMP),
        "OpFlagAddOperands" => Some(rv64::V_FLAG_ADD_OPERANDS),
        "OpFlagSubtractOperands" => Some(rv64::V_FLAG_SUBTRACT_OPERANDS),
        "OpFlagMultiplyOperands" => Some(rv64::V_FLAG_MULTIPLY_OPERANDS),
        "OpFlagLoad" => Some(rv64::V_FLAG_LOAD),
        "OpFlagStore" => Some(rv64::V_FLAG_STORE),
        "OpFlagJump" => Some(rv64::V_FLAG_JUMP),
        "OpFlagWriteLookupOutputToRD" => Some(rv64::V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD),
        "OpFlagVirtualInstruction" => Some(rv64::V_FLAG_VIRTUAL_INSTRUCTION),
        "OpFlagAssert" => Some(rv64::V_FLAG_ASSERT),
        "OpFlagDoNotUpdateUnexpandedPC" => Some(rv64::V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC),
        "OpFlagAdvice" => Some(rv64::V_FLAG_ADVICE),
        "OpFlagIsCompressed" => Some(rv64::V_FLAG_IS_COMPRESSED),
        "OpFlagIsFirstInSequence" => Some(rv64::V_FLAG_IS_FIRST_IN_SEQUENCE),
        "OpFlagIsLastInSequence" => Some(rv64::V_FLAG_IS_LAST_IN_SEQUENCE),
        _ => None,
    }
}

fn append_univariate_poly<F, T>(transcript: &mut T, label: &'static str, poly: &UnivariatePoly<F>)
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append(&LabelWithCount(
        label.as_bytes(),
        poly.coefficients().len() as u64,
    ));
    for coefficient in poly.coefficients() {
        transcript.append(coefficient);
    }
}

fn append_compressed_univariate_poly<F, T>(
    transcript: &mut T,
    label: &'static str,
    poly: &UnivariatePoly<F>,
) where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let compressed = poly.compress();
    transcript.append(&LabelWithCount(
        label.as_bytes(),
        compressed.coeffs_except_linear_term().len() as u64,
    ));
    for coefficient in compressed.coeffs_except_linear_term() {
        transcript.append(coefficient);
    }
}

fn append_labeled_scalar<F, T>(transcript: &mut T, label: &'static str, scalar: &F)
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append(&Label(label.as_bytes()));
    transcript.append(scalar);
}

fn append_opening_claims<F, T>(transcript: &mut T, evals: &[Stage1NamedEval<F>])
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    for eval in evals {
        append_labeled_scalar(transcript, "opening_claim", &eval.value);
    }
}

struct DenseOuterState<F: Field> {
    eq: Vec<F>,
    az: Vec<F>,
    bz: Vec<F>,
    eq_scratch: Vec<F>,
    az_scratch: Vec<F>,
    bz_scratch: Vec<F>,
}

impl<F: Field> DenseOuterState<F> {
    #[tracing::instrument(skip_all, name = "DenseOuterState::round_poly")]
    fn round_poly(&self) -> UnivariatePoly<F> {
        let pair_count = self.eq.len() / 2;
        let accumulators = if pair_count >= DENSE_BIND_PAR_THRESHOLD {
            self.eq
                .par_chunks_exact(2)
                .zip(self.az.par_chunks_exact(2))
                .zip(self.bz.par_chunks_exact(2))
                .map(|((eq_pair, az_pair), bz_pair)| {
                    let mut local = [F::Accumulator::default(); OUTER_REMAINING_DEGREE_BOUND + 1];
                    let eq0 = eq_pair[0];
                    let eq_delta = eq_pair[1] - eq_pair[0];
                    let az0 = az_pair[0];
                    let az_delta = az_pair[1] - az_pair[0];
                    let bz0 = bz_pair[0];
                    let bz_delta = bz_pair[1] - bz_pair[0];
                    accumulate_cubic_product_coefficients(
                        &mut local, eq0, eq_delta, az0, az_delta, bz0, bz_delta,
                    );
                    local
                })
                .reduce(
                    || [F::Accumulator::default(); OUTER_REMAINING_DEGREE_BOUND + 1],
                    |mut left, right| {
                        for i in 0..left.len() {
                            left[i].merge(right[i]);
                        }
                        left
                    },
                )
        } else {
            self.eq
                .chunks_exact(2)
                .zip(self.az.chunks_exact(2))
                .zip(self.bz.chunks_exact(2))
                .fold(
                    [F::Accumulator::default(); OUTER_REMAINING_DEGREE_BOUND + 1],
                    |mut local, ((eq_pair, az_pair), bz_pair)| {
                        let eq0 = eq_pair[0];
                        let eq_delta = eq_pair[1] - eq_pair[0];
                        let az0 = az_pair[0];
                        let az_delta = az_pair[1] - az_pair[0];
                        let bz0 = bz_pair[0];
                        let bz_delta = bz_pair[1] - bz_pair[0];
                        accumulate_cubic_product_coefficients(
                            &mut local, eq0, eq_delta, az0, az_delta, bz0, bz_delta,
                        );
                        local
                    },
                )
        };
        UnivariatePoly::new(accumulators.map(FieldAccumulator::reduce).to_vec())
    }

    #[tracing::instrument(skip_all, name = "DenseOuterState::bind")]
    fn bind(&mut self, challenge: F) {
        bind_dense_evals_reuse(&mut self.eq, &mut self.eq_scratch, challenge);
        bind_dense_evals_reuse(&mut self.az, &mut self.az_scratch, challenge);
        bind_dense_evals_reuse(&mut self.bz, &mut self.bz_scratch, challenge);
    }
}

#[inline]
fn accumulate_cubic_product_coefficients<F: Field>(
    coefficients: &mut [F::Accumulator; OUTER_REMAINING_DEGREE_BOUND + 1],
    eq0: F,
    eq_delta: F,
    az0: F,
    az_delta: F,
    bz0: F,
    bz_delta: F,
) {
    let az0_bz0 = az0 * bz0;
    let az_delta_bz0 = az_delta * bz0;
    let az0_bz_delta = az0 * bz_delta;
    let az_delta_bz_delta = az_delta * bz_delta;

    coefficients[0].fmadd(eq0, az0_bz0);
    coefficients[1].fmadd(eq_delta, az0_bz0);
    coefficients[1].fmadd(eq0, az_delta_bz0);
    coefficients[1].fmadd(eq0, az0_bz_delta);
    coefficients[2].fmadd(eq_delta, az_delta_bz0);
    coefficients[2].fmadd(eq_delta, az0_bz_delta);
    coefficients[2].fmadd(eq0, az_delta_bz_delta);
    coefficients[3].fmadd(eq_delta, az_delta_bz_delta);
}

fn bind_dense_evals_reuse<F: Field>(values: &mut Vec<F>, scratch: &mut Vec<F>, challenge: F) {
    let half = values.len() / 2;
    scratch.resize(half, F::zero());
    if half >= DENSE_BIND_PAR_THRESHOLD {
        scratch
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, output)| {
                let low = values[index << 1];
                let high = values[(index << 1) + 1];
                *output = low + (high - low) * challenge;
            });
    } else {
        for (index, output) in scratch.iter_mut().enumerate() {
            let low = values[index << 1];
            let high = values[(index << 1) + 1];
            *output = low + (high - low) * challenge;
        }
    }
    std::mem::swap(values, scratch);
    scratch.clear();
}

fn outer_remaining_round_poly<F: Field>(
    evaluator: &dyn Stage1OuterRemainingEvaluator<F>,
    context: Stage1OuterRemainingContext<'_, F>,
    num_rounds: usize,
    prefix: &[F],
) -> UnivariatePoly<F> {
    let suffix_rounds = num_rounds - prefix.len() - 1;
    let mut evals = Vec::with_capacity(OUTER_REMAINING_DEGREE_BOUND + 1);
    for x in 0..=OUTER_REMAINING_DEGREE_BOUND {
        let mut sum = F::zero();
        for suffix in 0..(1usize << suffix_rounds) {
            let mut point = Vec::with_capacity(num_rounds);
            point.extend_from_slice(prefix);
            point.push(F::from_u64(x as u64));
            for bit in 0..suffix_rounds {
                point.push(F::from_u64(((suffix >> bit) & 1) as u64));
            }
            sum += evaluator.evaluate(context, &point);
        }
        evals.push(sum);
    }
    UnivariatePoly::new(interpolate_to_coeffs(0, &evals))
}

fn scale_poly<F: Field>(poly: &UnivariatePoly<F>, scalar: F) -> UnivariatePoly<F> {
    UnivariatePoly::new(
        poly.coefficients()
            .iter()
            .map(|coefficient| *coefficient * scalar)
            .collect(),
    )
}

fn remaining_driver_evals<F: Field>(
    context: Stage1KernelContext<'_>,
    evaluator: &dyn Stage1OuterRemainingEvaluator<F>,
    remaining_context: Stage1OuterRemainingContext<'_, F>,
    point: &[F],
) -> Result<Vec<Stage1NamedEval<F>>, Stage1KernelError> {
    let plans = context
        .program
        .evals
        .iter()
        .filter(|eval| eval.source == context.driver.symbol)
        .collect::<Vec<_>>();
    let oracles = plans.iter().map(|eval| eval.oracle).collect::<Vec<_>>();
    let values = evaluator
        .evaluate_virtual_oracles(remaining_context, &oracles, point)
        .ok_or(Stage1KernelError::MissingKernelInput {
            kernel: context.kernel.abi,
            input: "remaining_driver_evals",
        })?;
    if values.len() != plans.len() {
        return Err(Stage1KernelError::InvalidInputLength {
            input: "remaining_driver_evals",
            expected: plans.len(),
            actual: values.len(),
        });
    }
    Ok(plans
        .into_iter()
        .zip(values)
        .map(|(eval, value)| Stage1NamedEval {
            name: eval.name,
            oracle: eval.oracle,
            value,
        })
        .collect())
}

fn uniskip_output_claim<F: Field>(sumchecks: &[Stage1SumcheckOutput<F>]) -> Option<F> {
    uniskip_point_and_claim(sumchecks).map(|(_, claim)| claim)
}

fn uniskip_point_and_claim<F: Field>(sumchecks: &[Stage1SumcheckOutput<F>]) -> Option<(F, F)> {
    sumchecks.iter().find_map(|sumcheck| {
        let point = *sumcheck.point.first()?;
        let claim = sumcheck
            .evals
            .iter()
            .find(|eval| eval.oracle == "UnivariateSkip")
            .map(|eval| eval.value)?;
        Some((point, claim))
    })
}

fn uniskip_sum_matches<F: Field>(poly: &UnivariatePoly<F>, claim: F) -> bool {
    let power_sums = integer_domain_power_sums(
        OUTER_UNISKIP_BASE_START,
        OUTER_UNISKIP_DOMAIN_SIZE,
        poly.coefficients().len(),
    );
    let sum = poly
        .coefficients()
        .iter()
        .zip(power_sums)
        .fold(F::zero(), |acc, (coefficient, power_sum)| {
            acc + coefficient.mul_i128(power_sum)
        });
    sum == claim
}

fn integer_domain_power_sums(domain_start: i64, domain_size: usize, count: usize) -> Vec<i128> {
    let mut sums = vec![0i128; count];
    for offset in 0..domain_size {
        let point = i128::from(domain_start + offset as i64);
        let mut power = 1i128;
        for sum in &mut sums {
            *sum += power;
            power *= point;
        }
    }
    sums
}

fn polynomial_degree<F: Field>(poly: &UnivariatePoly<F>) -> usize {
    poly.coefficients().len().saturating_sub(1)
}

fn driver_evals<F: Field>(context: Stage1KernelContext<'_>, value: F) -> Vec<Stage1NamedEval<F>> {
    context
        .program
        .evals
        .iter()
        .filter(|eval| eval.source == context.driver.symbol)
        .map(|eval| Stage1NamedEval {
            name: eval.name,
            oracle: eval.oracle,
            value,
        })
        .collect()
}

fn run_shape_kernel<F, T>(
    context: Stage1KernelContext<'_>,
    transcript: &mut T,
) -> Result<Stage1SumcheckOutput<F>, Stage1KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    match context.kernel.abi {
        "jolt_stage1_outer_uniskip" | "jolt_stage1_outer_remaining" => {
            Ok(shape_sumcheck_output(context, transcript))
        }
        abi => Err(Stage1KernelError::KernelNotImplemented { abi }),
    }
}

fn shape_sumcheck_output<F, T>(
    context: Stage1KernelContext<'_>,
    transcript: &mut T,
) -> Stage1SumcheckOutput<F>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let point = (0..context.driver.num_rounds)
        .map(|_| transcript.challenge())
        .collect();
    let evals = context
        .program
        .evals
        .iter()
        .filter(|eval| eval.source == context.driver.symbol)
        .map(|eval| Stage1NamedEval {
            name: eval.name,
            oracle: eval.oracle,
            value: F::from_u64(eval.index as u64),
        })
        .collect();
    Stage1SumcheckOutput {
        driver: context.driver.symbol,
        point,
        evals,
        proof: shape_sumcheck_proof(context.driver),
    }
}

fn shape_sumcheck_proof<F: Field>(driver: &Stage1SumcheckDriverPlan) -> SumcheckProof<F> {
    let coefficients = vec![F::zero(); driver.degree + 1];
    SumcheckProof {
        round_polynomials: (0..driver.num_rounds)
            .map(|_| UnivariatePoly::new(coefficients.clone()))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use jolt_field::{Field, Fr};
    use jolt_transcript::{MockTranscript, Transcript};

    use super::*;

    static KERNELS: &[Stage1KernelPlan] = &[Stage1KernelPlan {
        symbol: "kernel",
        relation: "relation",
        kind: "sumcheck",
        backend: "cpu",
        abi: "jolt_stage1_outer_uniskip",
    }];
    static FULL_KERNELS: &[Stage1KernelPlan] = &[
        Stage1KernelPlan {
            symbol: "uniskip_kernel",
            relation: "jolt.stage1.outer.uniskip",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage1_outer_uniskip",
        },
        Stage1KernelPlan {
            symbol: "remaining_kernel",
            relation: "jolt.stage1.outer.remaining",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage1_outer_remaining",
        },
    ];
    static CLAIMS: &[Stage1SumcheckClaimPlan] = &[Stage1SumcheckClaimPlan {
        symbol: "claim",
        stage: "stage",
        domain: "domain",
        num_rounds: 1,
        degree: OUTER_UNISKIP_DEGREE_BOUND,
        claim: "zero",
        kernel: "kernel",
        claim_value: "zero",
        input_openings: &[],
    }];
    static BATCHES: &[Stage1SumcheckBatchPlan] = &[Stage1SumcheckBatchPlan {
        symbol: "batch",
        stage: "stage",
        proof_slot: "proof",
        policy: "single",
        count: 1,
        ordered_claims: &["claim"],
        claim_operands: &["claim"],
        claim_label: "claim_label",
        round_label: "round_label",
        round_schedule: &[1],
    }];
    static BAD_BATCHES: &[Stage1SumcheckBatchPlan] = &[Stage1SumcheckBatchPlan {
        symbol: "batch",
        stage: "stage",
        proof_slot: "proof",
        policy: "single",
        count: 2,
        ordered_claims: &["claim"],
        claim_operands: &["claim"],
        claim_label: "claim_label",
        round_label: "round_label",
        round_schedule: &[1],
    }];

    #[test]
    fn cubic_product_coefficients_match_interpolation() {
        let eq0 = Fr::from_u64(3);
        let eq_delta = Fr::from_u64(5);
        let az0 = Fr::from_u64(7);
        let az_delta = Fr::from_u64(11);
        let bz0 = Fr::from_u64(13);
        let bz_delta = Fr::from_u64(17);

        let mut accumulators =
            [<Fr as Field>::Accumulator::default(); OUTER_REMAINING_DEGREE_BOUND + 1];
        accumulate_cubic_product_coefficients(
            &mut accumulators,
            eq0,
            eq_delta,
            az0,
            az_delta,
            bz0,
            bz_delta,
        );
        let direct = accumulators.map(FieldAccumulator::reduce).to_vec();

        let evals = (0..=OUTER_REMAINING_DEGREE_BOUND)
            .map(|x| {
                let point = Fr::from_u64(x as u64);
                (eq0 + eq_delta * point) * (az0 + az_delta * point) * (bz0 + bz_delta * point)
            })
            .collect::<Vec<_>>();
        assert_eq!(direct, interpolate_to_coeffs(0, &evals));
    }

    #[test]
    fn uniskip_integer_coefficients_match_lagrange_weights() {
        for (target, coefficients) in uniskip_targets()
            .into_iter()
            .zip(OUTER_UNISKIP_TARGET_COEFFS)
        {
            let weights = lagrange_evals(
                OUTER_UNISKIP_BASE_START,
                OUTER_UNISKIP_DOMAIN_SIZE,
                Fr::from_i64(target),
            );
            let expected = coefficients.map(Fr::from_i64).to_vec();
            assert_eq!(weights, expected);
        }
    }

    #[test]
    fn integer_coeff_group_matvec_matches_field_weights() {
        let mut a_values = [Fr::from_u64(0); OUTER_EQ_CONSTRAINT_ROWS];
        let mut b_values = [Fr::from_u64(0); OUTER_EQ_CONSTRAINT_ROWS];
        for i in 0..OUTER_EQ_CONSTRAINT_ROWS {
            a_values[i] = match i % 3 {
                0 => Fr::from_u64(0),
                1 => Fr::from_u64(1),
                _ => Fr::from_u64((i + 2) as u64),
            };
            b_values[i] = Fr::from_i64(i as i64 - 7);
        }
        let dots = R1csRowDotSlice {
            a: &a_values,
            b: &b_values,
        };
        let coefficients = &OUTER_UNISKIP_TARGET_COEFFS[3];
        let coefficient_fields = coefficients.map(Fr::from_i64);
        let weights = coefficient_fields.to_vec();

        let integer = Stage1OuterR1csData::group_matvecs_from_integer_coeffs(
            &OUTER_FIRST_GROUP_ROWS,
            coefficients,
            &coefficient_fields,
            dots,
        );
        let field =
            Stage1OuterR1csData::group_matvecs_from_dots(&OUTER_FIRST_GROUP_ROWS, &weights, dots);
        assert_eq!(integer, field);
    }

    static FULL_CLAIMS: &[Stage1SumcheckClaimPlan] = &[
        Stage1SumcheckClaimPlan {
            symbol: "uniskip_claim",
            stage: "stage",
            domain: "uniskip_domain",
            num_rounds: 1,
            degree: OUTER_UNISKIP_DEGREE_BOUND,
            claim: "zero",
            kernel: "uniskip_kernel",
            claim_value: "zero",
            input_openings: &[],
        },
        Stage1SumcheckClaimPlan {
            symbol: "remaining_claim",
            stage: "stage",
            domain: "trace_domain",
            num_rounds: 2,
            degree: OUTER_REMAINING_DEGREE_BOUND,
            claim: "stage1.uniskip.opening",
            kernel: "remaining_kernel",
            claim_value: "stage1.uniskip.eval",
            input_openings: &["stage1.uniskip.opening"],
        },
    ];
    static FULL_BATCHES: &[Stage1SumcheckBatchPlan] = &[
        Stage1SumcheckBatchPlan {
            symbol: "uniskip_batch",
            stage: "stage",
            proof_slot: "uniskip_proof",
            policy: "single",
            count: 1,
            ordered_claims: &["uniskip_claim"],
            claim_operands: &["uniskip_claim"],
            claim_label: "uniskip_claim",
            round_label: "uniskip_poly",
            round_schedule: &[1],
        },
        Stage1SumcheckBatchPlan {
            symbol: "remaining_batch",
            stage: "stage",
            proof_slot: "remaining_proof",
            policy: "jolt_core_front_loaded",
            count: 1,
            ordered_claims: &["remaining_claim"],
            claim_operands: &["remaining_claim"],
            claim_label: "sumcheck_claim",
            round_label: "sumcheck_poly",
            round_schedule: &[2],
        },
    ];
    static DRIVERS: &[Stage1SumcheckDriverPlan] = &[Stage1SumcheckDriverPlan {
        symbol: "driver",
        stage: "stage",
        proof_slot: "proof",
        kernel: "kernel",
        batch: "batch",
        policy: "single",
        round_schedule: &[1],
        claim_label: "claim_label",
        round_label: "round_label",
        num_rounds: 1,
        degree: OUTER_UNISKIP_DEGREE_BOUND,
    }];
    static FULL_DRIVERS: &[Stage1SumcheckDriverPlan] = &[
        Stage1SumcheckDriverPlan {
            symbol: "stage1.uniskip.sumcheck",
            stage: "stage",
            proof_slot: "uniskip_proof",
            kernel: "uniskip_kernel",
            batch: "uniskip_batch",
            policy: "univariate_skip",
            round_schedule: &[1],
            claim_label: "uniskip_claim",
            round_label: "uniskip_poly",
            num_rounds: 1,
            degree: OUTER_UNISKIP_DEGREE_BOUND,
        },
        Stage1SumcheckDriverPlan {
            symbol: "stage1.outer_remaining.sumcheck",
            stage: "stage",
            proof_slot: "remaining_proof",
            kernel: "remaining_kernel",
            batch: "remaining_batch",
            policy: "jolt_core_front_loaded",
            round_schedule: &[2],
            claim_label: "sumcheck_claim",
            round_label: "sumcheck_poly",
            num_rounds: 2,
            degree: OUTER_REMAINING_DEGREE_BOUND,
        },
    ];
    static FULL_EVALS: &[Stage1SumcheckEvalPlan] = &[
        Stage1SumcheckEvalPlan {
            symbol: "uniskip_eval",
            source: "stage1.uniskip.sumcheck",
            name: "stage1.uniskip.eval",
            index: 0,
            oracle: "UnivariateSkip",
        },
        Stage1SumcheckEvalPlan {
            symbol: "remaining_eval",
            source: "stage1.outer_remaining.sumcheck",
            name: "stage1.outer_remaining.eval.Synthetic",
            index: 0,
            oracle: "Synthetic",
        },
    ];
    static SQUEEZES: &[Stage1TranscriptSqueezePlan] = &[Stage1TranscriptSqueezePlan {
        symbol: "stage1.tau",
        label: "outer_tau",
        kind: "challenge_vector",
        count: 2,
    }];
    static REAL_SQUEEZES: &[Stage1TranscriptSqueezePlan] = &[Stage1TranscriptSqueezePlan {
        symbol: "stage1.tau",
        label: "outer_tau",
        kind: "challenge_vector",
        count: 3,
    }];
    static PROGRAM: Stage1CpuProgramPlan = Stage1CpuProgramPlan {
        params: Stage1Params {
            field: "bn254_fr",
            pcs: "dory",
            transcript: "blake2b_transcript",
        },
        transcript_squeezes: SQUEEZES,
        kernels: KERNELS,
        claims: CLAIMS,
        batches: BATCHES,
        drivers: DRIVERS,
        instance_results: &[],
        evals: &[],
        opening_claims: &[],
        opening_batches: &[],
    };
    static BAD_PROGRAM: Stage1CpuProgramPlan = Stage1CpuProgramPlan {
        params: Stage1Params {
            field: "bn254_fr",
            pcs: "dory",
            transcript: "blake2b_transcript",
        },
        transcript_squeezes: SQUEEZES,
        kernels: KERNELS,
        claims: CLAIMS,
        batches: BAD_BATCHES,
        drivers: DRIVERS,
        instance_results: &[],
        evals: &[],
        opening_claims: &[],
        opening_batches: &[],
    };
    static FULL_PROGRAM: Stage1CpuProgramPlan = Stage1CpuProgramPlan {
        params: Stage1Params {
            field: "bn254_fr",
            pcs: "dory",
            transcript: "blake2b_transcript",
        },
        transcript_squeezes: SQUEEZES,
        kernels: FULL_KERNELS,
        claims: FULL_CLAIMS,
        batches: FULL_BATCHES,
        drivers: FULL_DRIVERS,
        instance_results: &[],
        evals: FULL_EVALS,
        opening_claims: &[],
        opening_batches: &[],
    };
    static REAL_EVALS: &[Stage1SumcheckEvalPlan] = &[
        Stage1SumcheckEvalPlan {
            symbol: "uniskip_eval",
            source: "stage1.uniskip.sumcheck",
            name: "stage1.uniskip.eval",
            index: 0,
            oracle: "UnivariateSkip",
        },
        Stage1SumcheckEvalPlan {
            symbol: "remaining_eval",
            source: "stage1.outer_remaining.sumcheck",
            name: "stage1.outer_remaining.eval.PC",
            index: 0,
            oracle: "PC",
        },
    ];
    static REAL_PROGRAM: Stage1CpuProgramPlan = Stage1CpuProgramPlan {
        params: Stage1Params {
            field: "bn254_fr",
            pcs: "dory",
            transcript: "blake2b_transcript",
        },
        transcript_squeezes: REAL_SQUEEZES,
        kernels: FULL_KERNELS,
        claims: FULL_CLAIMS,
        batches: FULL_BATCHES,
        drivers: FULL_DRIVERS,
        instance_results: &[],
        evals: REAL_EVALS,
        opening_claims: &[],
        opening_batches: &[],
    };

    struct SumZeroRemainingEvaluator;

    impl Stage1OuterRemainingEvaluator<Fr> for SumZeroRemainingEvaluator {
        fn evaluate(&self, _context: Stage1OuterRemainingContext<'_, Fr>, point: &[Fr]) -> Fr {
            point[0] + point[0] - Fr::from_u64(1)
        }

        fn evaluate_virtual_oracle(
            &self,
            _context: Stage1OuterRemainingContext<'_, Fr>,
            oracle: &str,
            point: &[Fr],
        ) -> Option<Fr> {
            (oracle == "Synthetic").then(|| point.iter().copied().sum())
        }
    }

    #[derive(Default)]
    struct RecordingExecutor {
        modes: Vec<Stage1ExecutionMode>,
        drivers: Vec<&'static str>,
    }

    impl RecordingExecutor {
        fn output<F: Field>(
            &mut self,
            context: Stage1KernelContext<'_>,
            point: F,
        ) -> Stage1SumcheckOutput<F> {
            self.modes.push(context.mode);
            self.drivers.push(context.driver.symbol);
            Stage1SumcheckOutput {
                driver: context.driver.symbol,
                point: vec![point],
                evals: Vec::new(),
                proof: SumcheckProof::default(),
            }
        }
    }

    impl Stage1KernelExecutor<Fr> for RecordingExecutor {
        fn prove_sumcheck<T>(
            &mut self,
            context: Stage1KernelContext<'_>,
            transcript: &mut T,
        ) -> Result<Stage1SumcheckOutput<Fr>, Stage1KernelError>
        where
            T: Transcript<Challenge = Fr>,
        {
            Ok(self.output(context, transcript.challenge()))
        }

        fn verify_sumcheck<T>(
            &mut self,
            context: Stage1KernelContext<'_>,
            transcript: &mut T,
        ) -> Result<Stage1SumcheckOutput<Fr>, Stage1KernelError>
        where
            T: Transcript<Challenge = Fr>,
        {
            Ok(self.output(context, transcript.challenge()))
        }
    }

    fn noop_r1cs_key_and_witness(num_cycles: usize) -> (R1csKey<Fr>, Vec<Fr>) {
        let matrices = rv64::rv64_constraints::<Fr>();
        let key = R1csKey::new(matrices, num_cycles);
        let mut witness = vec![Fr::from_u64(0); key.num_cycles * key.num_vars_padded];
        for cycle in 0..key.num_cycles {
            let base = cycle * key.num_vars_padded;
            witness[base + rv64::V_CONST] = Fr::from_u64(1);
            witness[base + rv64::V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = Fr::from_u64(1);
            key.matrices
                .check_witness(&witness[base..base + rv64::NUM_VARS_PER_CYCLE])
                .expect("noop cycle satisfies RV64 constraints");
        }
        (key, witness)
    }

    #[test]
    fn execute_stage1_program_dispatches_driver_to_executor() {
        let mut executor = RecordingExecutor::default();
        let mut transcript = MockTranscript::<Fr>::new(b"stage1");
        let artifacts = execute_stage1_program(
            &PROGRAM,
            Stage1ExecutionMode::Prover,
            &mut executor,
            &mut transcript,
        )
        .expect("dispatch succeeds");

        assert_eq!(executor.modes, vec![Stage1ExecutionMode::Prover]);
        assert_eq!(executor.drivers, vec!["driver"]);
        assert_eq!(artifacts.sumchecks.len(), 1);
        assert_eq!(artifacts.challenge_vectors.len(), 1);
        assert_eq!(artifacts.sumchecks[0].driver, "driver");
        assert_eq!(artifacts.sumchecks[0].point.len(), 1);
    }

    #[test]
    fn execute_stage1_program_rejects_static_count_mismatch() {
        let mut executor = RecordingExecutor::default();
        let mut transcript = MockTranscript::<Fr>::new(b"stage1");
        let error = execute_stage1_program(
            &BAD_PROGRAM,
            Stage1ExecutionMode::Verifier,
            &mut executor,
            &mut transcript,
        )
        .expect_err("bad static plan rejected");

        assert_eq!(
            error,
            Stage1KernelError::PlanCountMismatch {
                artifact: "batch",
                expected: 2,
                actual: 1,
            }
        );
        assert!(executor.drivers.is_empty());
    }

    #[test]
    fn shape_kernel_executor_runs_known_stage1_kernel_abis() {
        let mut executor = Stage1ShapeKernelExecutor;
        let mut transcript = MockTranscript::<Fr>::new(b"stage1");
        let artifacts = execute_stage1_program(
            &PROGRAM,
            Stage1ExecutionMode::Verifier,
            &mut executor,
            &mut transcript,
        )
        .expect("shape kernel dispatch succeeds");

        assert_eq!(artifacts.sumchecks.len(), 1);
        assert_eq!(artifacts.sumchecks[0].driver, "driver");
        assert_eq!(artifacts.sumchecks[0].point.len(), 1);
        assert_eq!(artifacts.sumchecks[0].proof.round_polynomials.len(), 1);
        assert_eq!(
            artifacts.sumchecks[0].proof.round_polynomials[0]
                .coefficients()
                .len(),
            OUTER_UNISKIP_DEGREE_BOUND + 1
        );
    }

    #[test]
    fn prover_kernel_executor_requires_uniskip_extended_evals() {
        let inputs = Stage1ProverInputs::<Fr>::empty(1);
        let mut executor = Stage1ProverKernelExecutor::new(inputs);
        let mut transcript = MockTranscript::<Fr>::new(b"stage1");
        let error = execute_stage1_program(
            &PROGRAM,
            Stage1ExecutionMode::Prover,
            &mut executor,
            &mut transcript,
        )
        .expect_err("real prover requires extended evaluations");

        assert_eq!(
            error,
            Stage1KernelError::MissingKernelInput {
                kernel: "jolt_stage1_outer_uniskip",
                input: "uniskip_extended_evals",
            }
        );
    }

    #[test]
    fn uniskip_kernel_prover_verifier_self_parity() {
        let extended_evals = (1..=OUTER_UNISKIP_DEGREE)
            .map(|index| Fr::from_u64(index as u64))
            .collect::<Vec<_>>();
        let inputs = Stage1ProverInputs::empty(1).with_uniskip_extended_evals(&extended_evals);
        let mut prover_executor = Stage1ProverKernelExecutor::new(inputs);
        let mut prover_transcript = MockTranscript::<Fr>::new(b"stage1");
        let prover_artifacts = execute_stage1_program(
            &PROGRAM,
            Stage1ExecutionMode::Prover,
            &mut prover_executor,
            &mut prover_transcript,
        )
        .expect("uniskip prover succeeds");

        let proof = Stage1Proof::from(prover_artifacts.clone());
        let mut verifier_executor = Stage1VerifierKernelExecutor::new(&proof);
        let mut verifier_transcript = MockTranscript::<Fr>::new(b"stage1");
        let verifier_artifacts = execute_stage1_program(
            &PROGRAM,
            Stage1ExecutionMode::Verifier,
            &mut verifier_executor,
            &mut verifier_transcript,
        )
        .expect("uniskip verifier accepts prover proof");

        assert_eq!(prover_transcript.state(), verifier_transcript.state());
        assert_eq!(prover_artifacts.sumchecks.len(), 1);
        assert_eq!(verifier_artifacts.sumchecks.len(), 1);
        assert_eq!(
            prover_artifacts.sumchecks[0].point,
            verifier_artifacts.sumchecks[0].point
        );
        assert_eq!(
            prover_artifacts.sumchecks[0].proof.round_polynomials[0].coefficients(),
            verifier_artifacts.sumchecks[0].proof.round_polynomials[0].coefficients()
        );
    }

    #[test]
    fn full_stage1_uniskip_and_remaining_self_parity() {
        let extended_evals = vec![Fr::from_u64(0); OUTER_UNISKIP_DEGREE];
        let evaluator = SumZeroRemainingEvaluator;
        let inputs = Stage1ProverInputs::empty(1)
            .with_uniskip_extended_evals(&extended_evals)
            .with_outer_remaining_evaluator(&evaluator);
        let mut prover_executor = Stage1ProverKernelExecutor::new(inputs);
        let mut prover_transcript = MockTranscript::<Fr>::new(b"stage1");
        let prover_artifacts = execute_stage1_program(
            &FULL_PROGRAM,
            Stage1ExecutionMode::Prover,
            &mut prover_executor,
            &mut prover_transcript,
        )
        .expect("full stage1 prover succeeds");

        let proof = Stage1Proof::from(prover_artifacts.clone());
        let mut verifier_executor = Stage1VerifierKernelExecutor::new(&proof);
        let mut verifier_transcript = MockTranscript::<Fr>::new(b"stage1");
        let verifier_artifacts = execute_stage1_program(
            &FULL_PROGRAM,
            Stage1ExecutionMode::Verifier,
            &mut verifier_executor,
            &mut verifier_transcript,
        )
        .expect("full stage1 verifier accepts prover proof");

        assert_eq!(prover_transcript.state(), verifier_transcript.state());
        assert_eq!(prover_artifacts.sumchecks.len(), 2);
        assert_eq!(verifier_artifacts.sumchecks.len(), 2);
        assert_eq!(
            prover_artifacts.sumchecks[1].point,
            verifier_artifacts.sumchecks[1].point
        );
        assert_eq!(prover_artifacts.sumchecks[1].evals.len(), 1);
        assert_eq!(
            prover_artifacts.sumchecks[1].evals[0].value,
            verifier_artifacts.sumchecks[1].evals[0].value
        );
    }

    #[test]
    fn full_stage1_r1cs_data_self_parity() {
        let (key, witness) = noop_r1cs_key_and_witness(2);
        let data = Stage1OuterR1csData::new(&key, &witness).expect("valid R1CS witness shape");
        let inputs =
            Stage1ProverInputs::empty(key.num_cycle_vars()).with_outer_remaining_evaluator(&data);
        let mut prover_executor = Stage1ProverKernelExecutor::new(inputs);
        let mut prover_transcript = MockTranscript::<Fr>::new(b"stage1");
        let prover_artifacts = execute_stage1_program(
            &REAL_PROGRAM,
            Stage1ExecutionMode::Prover,
            &mut prover_executor,
            &mut prover_transcript,
        )
        .expect("real R1CS-backed stage1 prover succeeds");

        let proof = Stage1Proof::from(prover_artifacts.clone());
        let mut verifier_executor = Stage1VerifierKernelExecutor::new(&proof);
        let mut verifier_transcript = MockTranscript::<Fr>::new(b"stage1");
        let verifier_artifacts = execute_stage1_program(
            &REAL_PROGRAM,
            Stage1ExecutionMode::Verifier,
            &mut verifier_executor,
            &mut verifier_transcript,
        )
        .expect("real R1CS-backed stage1 verifier accepts prover proof");

        assert_eq!(prover_transcript.state(), verifier_transcript.state());
        assert_eq!(prover_artifacts.sumchecks.len(), 2);
        assert_eq!(verifier_artifacts.sumchecks.len(), 2);
        assert_eq!(
            prover_artifacts.sumchecks[1].evals[0].value,
            verifier_artifacts.sumchecks[1].evals[0].value
        );
    }

    #[test]
    fn full_stage1_r1cs_data_verifier_rejects_tampered_remaining_round() {
        let (key, witness) = noop_r1cs_key_and_witness(2);
        let data = Stage1OuterR1csData::new(&key, &witness).expect("valid R1CS witness shape");
        let inputs =
            Stage1ProverInputs::empty(key.num_cycle_vars()).with_outer_remaining_evaluator(&data);
        let mut prover_executor = Stage1ProverKernelExecutor::new(inputs);
        let mut prover_transcript = MockTranscript::<Fr>::new(b"stage1");
        let prover_artifacts = execute_stage1_program(
            &REAL_PROGRAM,
            Stage1ExecutionMode::Prover,
            &mut prover_executor,
            &mut prover_transcript,
        )
        .expect("real R1CS-backed stage1 prover succeeds");

        let mut proof = Stage1Proof::from(prover_artifacts);
        let remaining = &mut proof.sumchecks[1].proof.round_polynomials[0];
        let mut coefficients = remaining.clone().into_coefficients();
        coefficients[0] += Fr::from_u64(1);
        *remaining = UnivariatePoly::new(coefficients);

        let mut verifier_executor = Stage1VerifierKernelExecutor::new(&proof);
        let mut verifier_transcript = MockTranscript::<Fr>::new(b"stage1");
        let error = execute_stage1_program(
            &REAL_PROGRAM,
            Stage1ExecutionMode::Verifier,
            &mut verifier_executor,
            &mut verifier_transcript,
        )
        .expect_err("tampered remaining round is rejected");

        assert_eq!(
            error,
            Stage1KernelError::InvalidProof {
                driver: "stage1.outer_remaining.sumcheck",
                reason: "outer remaining round check failed",
            }
        );
    }

    #[test]
    fn verifier_kernel_executor_rejects_invalid_uniskip_proof() {
        let proof = Stage1Proof {
            sumchecks: vec![Stage1SumcheckOutput {
                driver: "driver",
                point: Vec::new(),
                evals: Vec::new(),
                proof: SumcheckProof::default(),
            }],
        };
        let mut executor = Stage1VerifierKernelExecutor::new(&proof);
        let mut transcript = MockTranscript::<Fr>::new(b"stage1");
        let error = execute_stage1_program(
            &PROGRAM,
            Stage1ExecutionMode::Verifier,
            &mut executor,
            &mut transcript,
        )
        .expect_err("empty verifier proof is invalid");

        assert_eq!(
            error,
            Stage1KernelError::InvalidProof {
                driver: "driver",
                reason: "missing uniskip round polynomial",
            }
        );
    }
}
