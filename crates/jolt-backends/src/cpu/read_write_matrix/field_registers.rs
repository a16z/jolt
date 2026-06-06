use jolt_field::{Field, OptimizedMul, RingAccumulator, WithAccumulator};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use rayon::prelude::*;

use crate::{
    BackendError, SumcheckFieldRegisterRead, SumcheckFieldRegisterWrite,
    SumcheckFieldRegistersIncClaimReductionOutput,
    SumcheckFieldRegistersIncClaimReductionStateRequest, SumcheckFieldRegistersReadWriteRow,
    SumcheckFieldRegistersReadWriteStateRequest, SumcheckFieldRegistersValEvaluationOutput,
    SumcheckFieldRegistersValEvaluationStateRequest, SumcheckRegistersReadWriteOutput,
};

use super::{
    AddressMajorBindableEntry, AddressMajorMatrixEntry, AddressMajorMessageEntry,
    AddressMajorMessageInputs, CycleMajorMatrixEntry, CycleMajorMessageEntry,
    CycleMajorToAddressMajor, OneHotCoeff, OneHotCoeffTable, ReadWriteMatrixAddressMajor,
    ReadWriteMatrixCycleMajor,
};

const DEGREE_BOUND: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FieldRegistersReadWriteParams<F: Field> {
    log_t: usize,
    log_k: usize,
    phase1_num_rounds: usize,
    phase2_num_rounds: usize,
    gamma: F,
}

impl<F: Field> FieldRegistersReadWriteParams<F> {
    const fn rounds(self) -> usize {
        self.log_t + self.log_k
    }

    const fn phase3_cycle_rounds(self) -> usize {
        self.log_t - self.phase1_num_rounds
    }

    const fn register_count(self) -> usize {
        1usize << self.log_k
    }
}

#[derive(Debug, Default)]
enum FieldRegistersSparseMatrix<F: Field> {
    #[default]
    None,
    CycleMajor(ReadWriteMatrixCycleMajor<F, FieldRegistersCycleMajorEntry<F, F>>),
    AddressMajor(ReadWriteMatrixAddressMajor<F, FieldRegistersAddressMajorEntry<F>>),
}

impl<F: Field> FieldRegistersSparseMatrix<F> {
    fn bind(&mut self, challenge: F) {
        match self {
            Self::None => unreachable!("cannot bind empty field-register sparse matrix"),
            Self::CycleMajor(matrix) => matrix.bind(challenge),
            Self::AddressMajor(matrix) => matrix.bind(challenge),
        }
    }

    fn materialize(
        self,
        register_count: usize,
        cycles: usize,
    ) -> (Polynomial<F>, Polynomial<F>, Polynomial<F>) {
        match self {
            Self::None => unreachable!("cannot materialize empty field-register sparse matrix"),
            Self::CycleMajor(matrix) => materialize_cycle_major(matrix, register_count, cycles),
            Self::AddressMajor(matrix) => materialize_address_major(matrix, register_count, cycles),
        }
    }
}

pub struct FieldRegistersReadWriteState<F: Field> {
    sparse_matrix: FieldRegistersSparseMatrix<F>,
    gruen_eq: Option<GruenSplitEqPolynomial<F>>,
    inc: Polynomial<F>,
    ra: Option<Polynomial<F>>,
    wa: Option<Polynomial<F>>,
    val: Option<Polynomial<F>>,
    merged_eq: Option<Polynomial<F>>,
    input_claim: F,
    params: FieldRegistersReadWriteParams<F>,
    rs2_registers: Vec<Option<u8>>,
    round: usize,
}

impl<F> FieldRegistersReadWriteState<F>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    pub fn new(
        backend: &'static str,
        task: &'static str,
        request: &SumcheckFieldRegistersReadWriteStateRequest<F>,
    ) -> Result<Self, BackendError> {
        validate_request(backend, task, request)?;
        let params = FieldRegistersReadWriteParams {
            log_t: request.log_t,
            log_k: request.log_k,
            phase1_num_rounds: request.phase1_num_rounds,
            phase2_num_rounds: request.phase2_num_rounds,
            gamma: request.gamma,
        };
        let rs2_registers = request
            .rows
            .iter()
            .map(|row| row.rs2.map(|read| read.register))
            .collect();
        let inc = Polynomial::new(request.rows.iter().map(|row| row.rd_increment).collect());
        let (gruen_eq, merged_eq) = if params.phase1_num_rounds > 0 {
            (
                Some(GruenSplitEqPolynomial::new(
                    &request.r_cycle,
                    BindingOrder::LowToHigh,
                )),
                None,
            )
        } else {
            (
                None,
                Some(Polynomial::new(jolt_poly::EqPolynomial::<F>::evals(
                    &request.r_cycle,
                    None,
                ))),
            )
        };

        let sparse_matrix = field_register_cycle_major(&request.rows, request.gamma);
        let (sparse_matrix, ra, wa, val) = if params.phase1_num_rounds > 0 {
            (
                FieldRegistersSparseMatrix::CycleMajor(sparse_matrix),
                None,
                None,
                None,
            )
        } else if params.phase2_num_rounds > 0 {
            (
                FieldRegistersSparseMatrix::AddressMajor(
                    field_registers_address_major_from_cycle_major(
                        sparse_matrix,
                        params.register_count(),
                    ),
                ),
                None,
                None,
                None,
            )
        } else {
            let (ra, wa, val) = materialize_cycle_major(
                sparse_matrix,
                params.register_count(),
                1usize << params.log_t,
            );
            (
                FieldRegistersSparseMatrix::None,
                Some(ra),
                Some(wa),
                Some(val),
            )
        };

        Ok(Self {
            sparse_matrix,
            gruen_eq,
            inc,
            ra,
            wa,
            val,
            merged_eq,
            input_claim: request.input_claim,
            params,
            rs2_registers,
            round: 0,
        })
    }

    pub fn input_claim(&self) -> F {
        self.input_claim
    }

    pub fn num_rounds(&self) -> usize {
        self.params.rounds()
    }

    pub fn evaluate_round(
        &self,
        backend: &'static str,
        task: &'static str,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError> {
        if self.round >= self.params.rounds() {
            return invalid(
                backend,
                task,
                format!(
                    "field-register read-write round {} is outside {} rounds",
                    self.round,
                    self.params.rounds()
                ),
            );
        }
        if self.round < self.params.phase1_num_rounds {
            self.phase1_compute_message(previous_claim)
        } else if self.round < self.params.phase1_num_rounds + self.params.phase2_num_rounds {
            self.phase2_compute_message(previous_claim)
        } else {
            self.phase3_compute_message(previous_claim)
        }
    }

    pub fn bind(
        &mut self,
        backend: &'static str,
        task: &'static str,
        challenge: F,
    ) -> Result<(), BackendError> {
        if self.round >= self.params.rounds() {
            return invalid(
                backend,
                task,
                format!(
                    "field-register read-write bind round {} is outside {} rounds",
                    self.round,
                    self.params.rounds()
                ),
            );
        }
        if self.round < self.params.phase1_num_rounds {
            self.phase1_bind(challenge);
        } else if self.round < self.params.phase1_num_rounds + self.params.phase2_num_rounds {
            self.phase2_bind(challenge);
        } else {
            self.phase3_bind(challenge);
        }
        self.round += 1;
        Ok(())
    }

    pub fn output_claims(
        &self,
        opening_point: &[F],
    ) -> Result<SumcheckRegistersReadWriteOutput<F>, BackendError> {
        let Some(val) = final_claim(self.val.as_ref()) else {
            return invalid(
                "cpu",
                "field-register read-write output claims",
                "missing value state",
            );
        };
        let Some(combined_ra) = final_claim(self.ra.as_ref()) else {
            return invalid(
                "cpu",
                "field-register read-write output claims",
                "missing RA state",
            );
        };
        let Some(rd_wa) = final_claim(self.wa.as_ref()) else {
            return invalid(
                "cpu",
                "field-register read-write output claims",
                "missing WA state",
            );
        };
        let field_rd_inc = self.inc.evaluations().first().copied().unwrap_or(F::zero());
        let (r_address, r_cycle) = opening_point.split_at(self.params.log_k);
        let field_rs2_ra = compute_rs2_ra_claim(&self.rs2_registers, r_address, r_cycle);
        let gamma_inv = self.gamma_inverse()?;
        let field_rs1_ra =
            (combined_ra - self.params.gamma * self.params.gamma * field_rs2_ra) * gamma_inv;
        Ok(SumcheckRegistersReadWriteOutput {
            registers_val: val,
            rs1_ra: field_rs1_ra,
            rs2_ra: field_rs2_ra,
            rd_wa,
            rd_inc: field_rd_inc,
        })
    }

    fn gamma_inverse(&self) -> Result<F, BackendError> {
        self.params
            .gamma
            .inverse()
            .ok_or_else(|| BackendError::InvalidRequest {
                backend: "cpu",
                task: "field-register read-write output claims",
                reason: "field-registers read-write gamma is not invertible".to_owned(),
            })
    }

    fn phase1_compute_message(&self, previous_claim: F) -> Result<UnivariatePoly<F>, BackendError> {
        match &self.sparse_matrix {
            FieldRegistersSparseMatrix::CycleMajor(matrix) => {
                self.phase1_compute_message_for_matrix(matrix, previous_claim)
            }
            _ => invalid(
                "cpu",
                "field-register read-write phase1",
                "missing cycle-major matrix",
            ),
        }
    }

    fn phase1_compute_message_for_matrix<C>(
        &self,
        matrix: &ReadWriteMatrixCycleMajor<F, FieldRegistersCycleMajorEntry<F, C>>,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>
    where
        C: OneHotCoeff<F>,
    {
        let gruen_eq = self
            .gruen_eq
            .as_ref()
            .ok_or_else(|| BackendError::InvalidRequest {
                backend: "cpu",
                task: "field-register read-write phase1",
                reason: "missing phase1 equality state".to_owned(),
            })?;
        let inc = self.inc.evaluations();
        let e_in = gruen_eq.e_in_current();
        let e_in_len = e_in.len();
        let num_x_in_bits = e_in_len.max(1).ilog2() as usize;
        let x_bitmask = (1usize << num_x_in_bits) - 1;

        let quadratic_coeffs = matrix
            .entries
            .par_chunk_by(|a, b| {
                ((a.row() / 2) >> num_x_in_bits) == ((b.row() / 2) >> num_x_in_bits)
            })
            .map(|entries| {
                let x_out = (entries[0].row() / 2) >> num_x_in_bits;
                let e_out_eval = gruen_eq.e_out_current()[x_out];
                let outer_sum = entries
                    .par_chunk_by(|a, b| a.row() / 2 == b.row() / 2)
                    .map(|entries| {
                        let odd_row_start =
                            entries.partition_point(|entry| entry.row().is_multiple_of(2));
                        let (even_row, odd_row) = entries.split_at(odd_row_start);
                        let row = 2 * (entries[0].row() / 2);
                        let e_in_eval = if e_in_len <= 1 {
                            F::one()
                        } else {
                            e_in[(row / 2) & x_bitmask]
                        };
                        let inc_evals = [inc[row], inc[row + 1] - inc[row]];
                        let inner = matrix.prover_message_contribution(
                            even_row,
                            odd_row,
                            inc_evals,
                            self.params.gamma,
                        );
                        [e_in_eval * inner[0], e_in_eval * inner[1]]
                    })
                    .reduce(|| [F::zero(); 2], sum_arrays::<F, 2>);
                [e_out_eval * outer_sum[0], e_out_eval * outer_sum[1]]
            })
            .reduce(|| [F::zero(); 2], sum_arrays::<F, 2>);

        Ok(gruen_eq.gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim))
    }

    fn phase2_compute_message(&self, previous_claim: F) -> Result<UnivariatePoly<F>, BackendError> {
        let merged_eq = self
            .merged_eq
            .as_ref()
            .ok_or_else(|| BackendError::InvalidRequest {
                backend: "cpu",
                task: "field-register read-write phase2",
                reason: "missing phase2 equality state".to_owned(),
            })?;
        let matrix = match &self.sparse_matrix {
            FieldRegistersSparseMatrix::AddressMajor(matrix) => matrix,
            _ => {
                return invalid(
                    "cpu",
                    "field-register read-write phase2",
                    "missing address-major matrix",
                );
            }
        };
        let inc = self.inc.evaluations();
        let eq = merged_eq.evaluations();
        let evals = matrix
            .entries
            .par_chunk_by(|a, b| a.column() / 2 == b.column() / 2)
            .map(|entries| {
                let odd_col_start =
                    entries.partition_point(|entry| entry.column().is_multiple_of(2));
                let (even_col, odd_col) = entries.split_at(odd_col_start);
                let even_col_idx = 2 * (entries[0].column() / 2);
                let odd_col_idx = even_col_idx + 1;
                ReadWriteMatrixAddressMajor::prover_message_contribution(
                    even_col,
                    odd_col,
                    matrix.val_init[even_col_idx],
                    matrix.val_init[odd_col_idx],
                    inc,
                    eq,
                    self.params.gamma,
                )
            })
            .reduce(|| [F::zero(); 2], sum_arrays::<F, 2>);
        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    fn phase3_compute_message(&self, previous_claim: F) -> Result<UnivariatePoly<F>, BackendError> {
        let merged_eq = require_poly(
            self.merged_eq.as_ref(),
            "field-register read-write phase3",
            "eq",
        )?;
        let ra = require_poly(self.ra.as_ref(), "field-register read-write phase3", "RA")?;
        let wa = require_poly(self.wa.as_ref(), "field-register read-write phase3", "WA")?;
        let val = require_poly(
            self.val.as_ref(),
            "field-register read-write phase3",
            "value",
        )?;
        if self.inc.len() > 1 {
            let k_prime = self.params.register_count() >> self.params.phase2_num_rounds;
            let t_prime = self.inc.len();
            let evals = (0..self.inc.len() / 2)
                .into_par_iter()
                .map(|row| {
                    let inc_evals = sumcheck_evals_array::<F, DEGREE_BOUND>(
                        &self.inc,
                        row,
                        BindingOrder::LowToHigh,
                    );
                    let eq_evals = sumcheck_evals_array::<F, DEGREE_BOUND>(
                        merged_eq,
                        row,
                        BindingOrder::LowToHigh,
                    );
                    let inner = (0..k_prime)
                        .into_par_iter()
                        .map(|register| {
                            let base = register * t_prime / 2 + row;
                            let ra_evals = sumcheck_evals_array::<F, DEGREE_BOUND>(
                                ra,
                                base,
                                BindingOrder::LowToHigh,
                            );
                            let wa_evals = sumcheck_evals_array::<F, DEGREE_BOUND>(
                                wa,
                                base,
                                BindingOrder::LowToHigh,
                            );
                            let val_evals = sumcheck_evals_array::<F, DEGREE_BOUND>(
                                val,
                                base,
                                BindingOrder::LowToHigh,
                            );
                            std::array::from_fn(|i| {
                                ra_evals[i] * val_evals[i]
                                    + wa_evals[i] * (val_evals[i] + inc_evals[i])
                            })
                        })
                        .reduce(|| [F::zero(); DEGREE_BOUND], sum_arrays::<F, DEGREE_BOUND>);
                    std::array::from_fn(|i| eq_evals[i] * inner[i])
                })
                .reduce(|| [F::zero(); DEGREE_BOUND], sum_arrays::<F, DEGREE_BOUND>);
            Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
        } else {
            let inc_eval = self.inc.evaluations()[0];
            let eq_eval = merged_eq.evaluations()[0];
            let evals = (0..ra.len() / 2)
                .into_par_iter()
                .map(|register| {
                    let ra_evals =
                        sumcheck_evals_array::<F, 2>(ra, register, BindingOrder::LowToHigh);
                    let wa_evals =
                        sumcheck_evals_array::<F, 2>(wa, register, BindingOrder::LowToHigh);
                    let val_evals =
                        sumcheck_evals_array::<F, 2>(val, register, BindingOrder::LowToHigh);
                    [
                        ra_evals[0] * val_evals[0] + wa_evals[0] * (val_evals[0] + inc_eval),
                        ra_evals[1] * val_evals[1] + wa_evals[1] * (val_evals[1] + inc_eval),
                    ]
                })
                .reduce(|| [F::zero(); 2], sum_arrays::<F, 2>);
            Ok(UnivariatePoly::from_evals_and_hint(
                previous_claim,
                &[eq_eval * evals[0], eq_eval * evals[1]],
            ))
        }
    }

    fn phase1_bind(&mut self, challenge: F) {
        if let Some(eq) = self.gruen_eq.as_mut() {
            eq.bind(challenge);
        }
        self.inc.bind_with_order(challenge, BindingOrder::LowToHigh);
        self.sparse_matrix.bind(challenge);
        if self.round == self.params.phase1_num_rounds - 1 {
            if let Some(eq) = self.gruen_eq.as_ref() {
                self.merged_eq = Some(eq.merge());
            }
            let matrix = std::mem::take(&mut self.sparse_matrix);
            if self.params.phase2_num_rounds > 0 {
                self.sparse_matrix = match matrix {
                    FieldRegistersSparseMatrix::CycleMajor(matrix) => {
                        FieldRegistersSparseMatrix::AddressMajor(
                            field_registers_address_major_from_cycle_major(
                                matrix,
                                self.params.register_count(),
                            ),
                        )
                    }
                    _ => unreachable!("field-register phase1 output must be cycle-major"),
                };
            } else {
                let cycles = 1usize << self.params.phase3_cycle_rounds();
                let (ra, wa, val) = matrix.materialize(self.params.register_count(), cycles);
                self.ra = Some(ra);
                self.wa = Some(wa);
                self.val = Some(val);
            }
        }
    }

    fn phase2_bind(&mut self, challenge: F) {
        self.sparse_matrix.bind(challenge);
        if self.round == self.params.phase1_num_rounds + self.params.phase2_num_rounds - 1 {
            let matrix = std::mem::take(&mut self.sparse_matrix);
            let (ra, wa, val) = matrix.materialize(
                self.params.register_count() >> self.params.phase2_num_rounds,
                1usize << self.params.phase3_cycle_rounds(),
            );
            self.ra = Some(ra);
            self.wa = Some(wa);
            self.val = Some(val);
        }
    }

    fn phase3_bind(&mut self, challenge: F) {
        if self.inc.len() > 1 {
            self.inc.bind_with_order(challenge, BindingOrder::LowToHigh);
            if let Some(eq) = self.merged_eq.as_mut() {
                eq.bind_with_order(challenge, BindingOrder::LowToHigh);
            }
        }
        if let Some(ra) = self.ra.as_mut() {
            ra.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        if let Some(wa) = self.wa.as_mut() {
            wa.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        if let Some(val) = self.val.as_mut() {
            val.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
    }
}

pub struct FieldRegistersValEvaluationState<F: Field> {
    inc: Polynomial<F>,
    wa: Polynomial<F>,
    lt: jolt_poly::LtPolynomial<F>,
    input_claim: F,
    log_t: usize,
    round: usize,
}

impl<F: Field> FieldRegistersValEvaluationState<F> {
    pub fn new(
        backend: &'static str,
        task: &'static str,
        request: &SumcheckFieldRegistersValEvaluationStateRequest<F>,
    ) -> Result<Self, BackendError> {
        validate_val_evaluation_request(backend, task, request)?;
        let register_count = 1usize << request.log_k;
        let eq_register = jolt_poly::EqPolynomial::new(request.r_address.clone()).evaluations();
        let inc = request
            .rows
            .iter()
            .map(|row| row.rd_increment)
            .collect::<Vec<_>>();
        let wa = request
            .rows
            .iter()
            .map(|row| {
                let Some(write) = row.rd else {
                    return Ok(F::zero());
                };
                let register = usize::from(write.register);
                if register >= register_count {
                    return invalid(
                        backend,
                        task,
                        format!(
                            "field-register value-evaluation write address {register} is outside {register_count} registers"
                        ),
                    );
                }
                Ok(eq_register[register])
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
        let lt = jolt_poly::LtPolynomial::new(&request.r_cycle);

        Ok(Self {
            inc: Polynomial::new(inc),
            wa: Polynomial::new(wa),
            lt,
            input_claim: request.input_claim,
            log_t: request.log_t,
            round: 0,
        })
    }

    pub fn input_claim(&self) -> F {
        self.input_claim
    }

    pub fn num_rounds(&self) -> usize {
        self.log_t
    }

    pub fn evaluate_round(
        &self,
        backend: &'static str,
        task: &'static str,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError> {
        if self.round >= self.num_rounds() {
            return invalid(
                backend,
                task,
                format!(
                    "field-register value-evaluation round {} is outside {} rounds",
                    self.round,
                    self.num_rounds()
                ),
            );
        }
        let evals = (0..self.inc.len() / 2)
            .into_par_iter()
            .map(|index| {
                let inc = sumcheck_evals_array::<F, DEGREE_BOUND>(
                    &self.inc,
                    index,
                    BindingOrder::LowToHigh,
                );
                let wa = sumcheck_evals_array::<F, DEGREE_BOUND>(
                    &self.wa,
                    index,
                    BindingOrder::LowToHigh,
                );
                let lt = lt_sumcheck_evals_array::<F, DEGREE_BOUND>(
                    &self.lt,
                    index,
                    BindingOrder::LowToHigh,
                );
                std::array::from_fn(|i| inc[i] * wa[i] * lt[i])
            })
            .reduce(|| [F::zero(); DEGREE_BOUND], sum_arrays::<F, DEGREE_BOUND>);
        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    pub fn bind(
        &mut self,
        backend: &'static str,
        task: &'static str,
        challenge: F,
    ) -> Result<(), BackendError> {
        if self.round >= self.num_rounds() {
            return invalid(
                backend,
                task,
                format!(
                    "field-register value-evaluation bind round {} is outside {} rounds",
                    self.round,
                    self.num_rounds()
                ),
            );
        }
        self.inc.bind_with_order(challenge, BindingOrder::LowToHigh);
        self.wa.bind_with_order(challenge, BindingOrder::LowToHigh);
        self.lt.bind_with_order(challenge, BindingOrder::LowToHigh);
        self.round += 1;
        Ok(())
    }

    pub fn output_claims(
        &self,
    ) -> Result<SumcheckFieldRegistersValEvaluationOutput<F>, BackendError> {
        let Some(&field_rd_inc) = self.inc.evaluations().first() else {
            return invalid(
                "cpu",
                "field-register value-evaluation output claims",
                "empty field-register increment state",
            );
        };
        let Some(&field_rd_wa) = self.wa.evaluations().first() else {
            return invalid(
                "cpu",
                "field-register value-evaluation output claims",
                "empty field-register write-address state",
            );
        };
        Ok(SumcheckFieldRegistersValEvaluationOutput {
            field_rd_inc,
            field_rd_wa,
        })
    }
}

pub struct FieldRegistersIncClaimReductionState<F: Field> {
    phase: FieldRegistersIncClaimReductionPhase<F>,
    input_claim: F,
    log_t: usize,
    round: usize,
}

enum FieldRegistersIncClaimReductionPhase<F: Field> {
    Prefix(FieldRegistersIncClaimReductionPrefixState<F>),
    Suffix(FieldRegistersIncClaimReductionSuffixState<F>),
    Taken,
}

struct FieldRegistersIncClaimReductionPrefixState<F: Field> {
    log_t: usize,
    prefix_vars: usize,
    gamma: F,
    field_rd_inc_by_reversed_cycle: Vec<F>,
    r_cycle_read_write: Vec<F>,
    r_cycle_val_evaluation: Vec<F>,
    p_read_write: Polynomial<F>,
    p_val_evaluation: Polynomial<F>,
    q_read_write: Polynomial<F>,
    q_val_evaluation: Polynomial<F>,
    challenges: Vec<F>,
}

struct FieldRegistersIncClaimReductionSuffixState<F: Field> {
    field_rd_inc: Polynomial<F>,
    coeff: Polynomial<F>,
}

impl<F: Field> FieldRegistersIncClaimReductionState<F> {
    pub fn new(
        backend: &'static str,
        task: &'static str,
        request: &SumcheckFieldRegistersIncClaimReductionStateRequest<F>,
    ) -> Result<Self, BackendError> {
        validate_inc_claim_reduction_request(backend, task, request)?;
        let prefix_vars = request.log_t / 2;
        let field_rd_inc_by_reversed_cycle = reverse_cycle_table(
            request
                .rows
                .iter()
                .map(|row| row.rd_increment)
                .collect::<Vec<_>>(),
            request.log_t,
        );
        let phase = if prefix_vars == 0 {
            FieldRegistersIncClaimReductionPhase::Suffix(
                FieldRegistersIncClaimReductionSuffixState::from_prefix_bound(
                    request.log_t,
                    prefix_vars,
                    &[],
                    &field_rd_inc_by_reversed_cycle,
                    &request.r_cycle_read_write,
                    &request.r_cycle_val_evaluation,
                    request.gamma,
                    F::one(),
                    F::one(),
                ),
            )
        } else {
            FieldRegistersIncClaimReductionPhase::Prefix(
                FieldRegistersIncClaimReductionPrefixState::new(
                    request,
                    field_rd_inc_by_reversed_cycle,
                ),
            )
        };

        Ok(Self {
            phase,
            input_claim: request.input_claim,
            log_t: request.log_t,
            round: 0,
        })
    }

    pub fn input_claim(&self) -> F {
        self.input_claim
    }

    pub fn num_rounds(&self) -> usize {
        self.log_t
    }

    pub fn evaluate_round(
        &self,
        backend: &'static str,
        task: &'static str,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError> {
        if self.round >= self.num_rounds() {
            return invalid(
                backend,
                task,
                format!(
                    "field-register increment claim-reduction round {} is outside {} rounds",
                    self.round,
                    self.num_rounds()
                ),
            );
        }
        match &self.phase {
            FieldRegistersIncClaimReductionPhase::Prefix(state) => {
                Ok(state.evaluate_round(previous_claim))
            }
            FieldRegistersIncClaimReductionPhase::Suffix(state) => {
                Ok(state.evaluate_round(previous_claim))
            }
            FieldRegistersIncClaimReductionPhase::Taken => invalid(
                backend,
                task,
                "field-register increment claim-reduction state was temporarily moved",
            ),
        }
    }

    pub fn bind(
        &mut self,
        backend: &'static str,
        task: &'static str,
        challenge: F,
    ) -> Result<(), BackendError> {
        if self.round >= self.num_rounds() {
            return invalid(
                backend,
                task,
                format!(
                    "field-register increment claim-reduction bind round {} is outside {} rounds",
                    self.round,
                    self.num_rounds()
                ),
            );
        }
        let should_transition = match &mut self.phase {
            FieldRegistersIncClaimReductionPhase::Prefix(state) => {
                state.bind(challenge);
                state.is_complete()
            }
            FieldRegistersIncClaimReductionPhase::Suffix(state) => {
                state.bind(challenge);
                false
            }
            FieldRegistersIncClaimReductionPhase::Taken => {
                unreachable!("field-register increment state is never left taken")
            }
        };
        if should_transition {
            let FieldRegistersIncClaimReductionPhase::Prefix(state) =
                std::mem::replace(&mut self.phase, FieldRegistersIncClaimReductionPhase::Taken)
            else {
                unreachable!("field-register increment transition requires prefix phase");
            };
            self.phase = FieldRegistersIncClaimReductionPhase::Suffix(state.into_suffix());
        }
        self.round += 1;
        Ok(())
    }

    pub fn output_claims(
        &self,
    ) -> Result<SumcheckFieldRegistersIncClaimReductionOutput<F>, BackendError> {
        let FieldRegistersIncClaimReductionPhase::Suffix(state) = &self.phase else {
            return invalid(
                "cpu",
                "field-register increment claim-reduction output claims",
                "field-register increment output requested before suffix phase completed",
            );
        };
        state.output_claims()
    }
}

impl<F: Field> FieldRegistersIncClaimReductionPrefixState<F> {
    fn new(
        request: &SumcheckFieldRegistersIncClaimReductionStateRequest<F>,
        field_rd_inc_by_reversed_cycle: Vec<F>,
    ) -> Self {
        let prefix_vars = request.log_t / 2;
        let suffix_vars = request.log_t - prefix_vars;
        let (read_write_prefix, read_write_suffix) =
            request.r_cycle_read_write.split_at(prefix_vars);
        let (val_evaluation_prefix, val_evaluation_suffix) =
            request.r_cycle_val_evaluation.split_at(prefix_vars);
        Self {
            log_t: request.log_t,
            prefix_vars,
            gamma: request.gamma,
            q_read_write: Polynomial::new(prefix_suffix_field_rd_inc_q(
                &field_rd_inc_by_reversed_cycle,
                prefix_vars,
                suffix_vars,
                read_write_suffix,
            )),
            q_val_evaluation: Polynomial::new(prefix_suffix_field_rd_inc_q(
                &field_rd_inc_by_reversed_cycle,
                prefix_vars,
                suffix_vars,
                val_evaluation_suffix,
            )),
            field_rd_inc_by_reversed_cycle,
            r_cycle_read_write: request.r_cycle_read_write.clone(),
            r_cycle_val_evaluation: request.r_cycle_val_evaluation.clone(),
            p_read_write: Polynomial::new(jolt_poly::EqPolynomial::<F>::evals(
                read_write_prefix,
                None,
            )),
            p_val_evaluation: Polynomial::new(jolt_poly::EqPolynomial::<F>::evals(
                val_evaluation_prefix,
                None,
            )),
            challenges: Vec::with_capacity(prefix_vars),
        }
    }

    fn evaluate_round(&self, previous_claim: F) -> UnivariatePoly<F> {
        let evals = (0..self.p_read_write.len() / 2)
            .into_par_iter()
            .map(|index| {
                let p_read_write = sumcheck_evals_array::<F, 2>(
                    &self.p_read_write,
                    index,
                    BindingOrder::HighToLow,
                );
                let p_val_evaluation = sumcheck_evals_array::<F, 2>(
                    &self.p_val_evaluation,
                    index,
                    BindingOrder::HighToLow,
                );
                let q_read_write = sumcheck_evals_array::<F, 2>(
                    &self.q_read_write,
                    index,
                    BindingOrder::HighToLow,
                );
                let q_val_evaluation = sumcheck_evals_array::<F, 2>(
                    &self.q_val_evaluation,
                    index,
                    BindingOrder::HighToLow,
                );
                std::array::from_fn(|point| {
                    p_read_write[point] * q_read_write[point]
                        + self.gamma * p_val_evaluation[point] * q_val_evaluation[point]
                })
            })
            .reduce(|| [F::zero(); 2], sum_arrays::<F, 2>);
        UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, challenge: F) {
        self.p_read_write
            .bind_with_order(challenge, BindingOrder::HighToLow);
        self.p_val_evaluation
            .bind_with_order(challenge, BindingOrder::HighToLow);
        self.q_read_write
            .bind_with_order(challenge, BindingOrder::HighToLow);
        self.q_val_evaluation
            .bind_with_order(challenge, BindingOrder::HighToLow);
        self.challenges.push(challenge);
    }

    fn is_complete(&self) -> bool {
        self.challenges.len() == self.prefix_vars
    }

    fn into_suffix(self) -> FieldRegistersIncClaimReductionSuffixState<F> {
        FieldRegistersIncClaimReductionSuffixState::from_prefix_bound(
            self.log_t,
            self.prefix_vars,
            &self.challenges,
            &self.field_rd_inc_by_reversed_cycle,
            &self.r_cycle_read_write,
            &self.r_cycle_val_evaluation,
            self.gamma,
            self.p_read_write.evaluations()[0],
            self.p_val_evaluation.evaluations()[0],
        )
    }
}

impl<F: Field> FieldRegistersIncClaimReductionSuffixState<F> {
    #[expect(
        clippy::too_many_arguments,
        reason = "The two verifier cycle points are the field-register increment relation inputs."
    )]
    fn from_prefix_bound(
        log_t: usize,
        prefix_vars: usize,
        prefix_challenges: &[F],
        field_rd_inc_by_reversed_cycle: &[F],
        r_cycle_read_write: &[F],
        r_cycle_val_evaluation: &[F],
        gamma: F,
        read_write_prefix_scale: F,
        val_evaluation_prefix_scale: F,
    ) -> Self {
        let suffix_vars = log_t - prefix_vars;
        let suffix_len = 1usize << suffix_vars;
        let prefix_len = 1usize << prefix_vars;
        let prefix_eq = jolt_poly::EqPolynomial::<F>::evals(prefix_challenges, None);
        let mut field_rd_inc = jolt_poly::thread::unsafe_allocate_zero_vec(suffix_len);
        field_rd_inc
            .par_iter_mut()
            .enumerate()
            .for_each(|(suffix_index, output)| {
                let mut acc = F::zero();
                for (prefix_index, &prefix_weight) in prefix_eq.iter().enumerate().take(prefix_len)
                {
                    let index = (prefix_index << suffix_vars) | suffix_index;
                    acc += prefix_weight * field_rd_inc_by_reversed_cycle[index];
                }
                *output = acc;
            });

        let (_, read_write_suffix) = r_cycle_read_write.split_at(prefix_vars);
        let (_, val_evaluation_suffix) = r_cycle_val_evaluation.split_at(prefix_vars);
        let eq_read_write =
            jolt_poly::EqPolynomial::<F>::evals(read_write_suffix, Some(read_write_prefix_scale));
        let eq_val_evaluation = jolt_poly::EqPolynomial::<F>::evals(
            val_evaluation_suffix,
            Some(val_evaluation_prefix_scale),
        );
        let coeff = eq_read_write
            .into_iter()
            .zip(eq_val_evaluation)
            .map(|(read_write, val_evaluation)| read_write + gamma * val_evaluation)
            .collect::<Vec<_>>();

        Self {
            field_rd_inc: Polynomial::new(field_rd_inc),
            coeff: Polynomial::new(coeff),
        }
    }

    fn evaluate_round(&self, previous_claim: F) -> UnivariatePoly<F> {
        let evals = (0..self.field_rd_inc.len() / 2)
            .into_par_iter()
            .map(|index| {
                let field_rd_inc = sumcheck_evals_array::<F, 2>(
                    &self.field_rd_inc,
                    index,
                    BindingOrder::HighToLow,
                );
                let coeff =
                    sumcheck_evals_array::<F, 2>(&self.coeff, index, BindingOrder::HighToLow);
                std::array::from_fn(|point| field_rd_inc[point] * coeff[point])
            })
            .reduce(|| [F::zero(); 2], sum_arrays::<F, 2>);
        UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, challenge: F) {
        self.field_rd_inc
            .bind_with_order(challenge, BindingOrder::HighToLow);
        self.coeff
            .bind_with_order(challenge, BindingOrder::HighToLow);
    }

    fn output_claims(
        &self,
    ) -> Result<SumcheckFieldRegistersIncClaimReductionOutput<F>, BackendError> {
        let Some(field_rd_inc) = final_claim(Some(&self.field_rd_inc)) else {
            return invalid(
                "cpu",
                "field-register increment claim-reduction output claims",
                "field-register increment polynomial is not fully bound",
            );
        };
        Ok(SumcheckFieldRegistersIncClaimReductionOutput { field_rd_inc })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FieldRegistersCycleMajorEntry<F: Field, C: OneHotCoeff<F>> {
    val_coeff: F,
    prev_val: F,
    next_val: F,
    row: usize,
    col: u8,
    ra_coeff: C,
    wa_coeff: C,
}

impl<F: Field, C: OneHotCoeff<F> + Default> Default for FieldRegistersCycleMajorEntry<F, C> {
    fn default() -> Self {
        Self {
            val_coeff: F::zero(),
            prev_val: F::zero(),
            next_val: F::zero(),
            row: 0,
            col: 0,
            ra_coeff: C::default(),
            wa_coeff: C::default(),
        }
    }
}

impl<F: Field, C: OneHotCoeff<F>> CycleMajorMatrixEntry<F> for FieldRegistersCycleMajorEntry<F, C> {
    fn row(&self) -> usize {
        self.row
    }

    fn column(&self) -> usize {
        usize::from(self.col)
    }

    fn bind_entries(
        even: Option<&Self>,
        odd: Option<&Self>,
        challenge: F,
        ra_lookup_table: Option<&OneHotCoeffTable<F>>,
        wa_lookup_table: Option<&OneHotCoeffTable<F>>,
    ) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => Self {
                row: even.row / 2,
                col: even.col,
                ra_coeff: OneHotCoeff::bind(
                    Some(&even.ra_coeff),
                    Some(&odd.ra_coeff),
                    challenge,
                    ra_lookup_table,
                ),
                wa_coeff: OneHotCoeff::bind(
                    Some(&even.wa_coeff),
                    Some(&odd.wa_coeff),
                    challenge,
                    wa_lookup_table,
                ),
                val_coeff: even.val_coeff
                    + challenge.mul_0_optimized(odd.val_coeff - even.val_coeff),
                prev_val: even.prev_val,
                next_val: odd.next_val,
            },
            (Some(even), None) => Self {
                row: even.row / 2,
                col: even.col,
                ra_coeff: OneHotCoeff::bind(Some(&even.ra_coeff), None, challenge, ra_lookup_table),
                wa_coeff: OneHotCoeff::bind(Some(&even.wa_coeff), None, challenge, wa_lookup_table),
                val_coeff: even.val_coeff
                    + challenge.mul_0_optimized(even.next_val - even.val_coeff),
                prev_val: even.prev_val,
                next_val: even.next_val,
            },
            (None, Some(odd)) => Self {
                row: odd.row / 2,
                col: odd.col,
                ra_coeff: OneHotCoeff::bind(None, Some(&odd.ra_coeff), challenge, ra_lookup_table),
                wa_coeff: OneHotCoeff::bind(None, Some(&odd.wa_coeff), challenge, wa_lookup_table),
                val_coeff: odd.prev_val + challenge.mul_0_optimized(odd.val_coeff - odd.prev_val),
                prev_val: odd.prev_val,
                next_val: odd.next_val,
            },
            (None, None) => unreachable!("field-register bind requires at least one entry"),
        }
    }
}

impl<F, C> CycleMajorMessageEntry<F> for FieldRegistersCycleMajorEntry<F, C>
where
    F: Field,
    C: OneHotCoeff<F>,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    fn accumulate_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        inc_evals: [F; 2],
        _gamma: F,
        accumulators: &mut [<F as WithAccumulator>::Accumulator; 2],
        ra_lookup_table: Option<&OneHotCoeffTable<F>>,
        wa_lookup_table: Option<&OneHotCoeffTable<F>>,
    ) {
        let (ra_evals, wa_evals, val_evals) = match (even, odd) {
            (Some(even), Some(odd)) => (
                OneHotCoeff::evals(Some(&even.ra_coeff), Some(&odd.ra_coeff), ra_lookup_table),
                OneHotCoeff::evals(Some(&even.wa_coeff), Some(&odd.wa_coeff), wa_lookup_table),
                [even.val_coeff, odd.val_coeff - even.val_coeff],
            ),
            (Some(even), None) => (
                OneHotCoeff::evals(Some(&even.ra_coeff), None, ra_lookup_table),
                OneHotCoeff::evals(Some(&even.wa_coeff), None, wa_lookup_table),
                [even.val_coeff, even.next_val - even.val_coeff],
            ),
            (None, Some(odd)) => (
                OneHotCoeff::evals(None, Some(&odd.ra_coeff), ra_lookup_table),
                OneHotCoeff::evals(None, Some(&odd.wa_coeff), wa_lookup_table),
                [odd.prev_val, odd.val_coeff - odd.prev_val],
            ),
            (None, None) => unreachable!("field-register message requires at least one entry"),
        };
        for index in 0..2 {
            accumulators[index].fmadd(ra_evals[index], val_evals[index]);
            accumulators[index].fmadd(wa_evals[index], val_evals[index] + inc_evals[index]);
        }
    }
}

impl<F: Field, C: OneHotCoeff<F>> CycleMajorToAddressMajor<F>
    for FieldRegistersCycleMajorEntry<F, C>
{
    type AddressMajor = FieldRegistersAddressMajorEntry<F>;

    fn to_address_major(
        self,
        ra_lookup_table: Option<&OneHotCoeffTable<F>>,
        wa_lookup_table: Option<&OneHotCoeffTable<F>>,
    ) -> Self::AddressMajor {
        FieldRegistersAddressMajorEntry {
            prev_val: self.prev_val,
            next_val: self.next_val,
            val_coeff: self.val_coeff,
            ra_coeff: self.ra_coeff.to_field(ra_lookup_table),
            wa_coeff: self.wa_coeff.to_field(wa_lookup_table),
            row: self.row,
            col: self.col,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FieldRegistersAddressMajorEntry<F: Field> {
    prev_val: F,
    next_val: F,
    val_coeff: F,
    ra_coeff: F,
    wa_coeff: F,
    row: usize,
    col: u8,
}

impl<F: Field> AddressMajorMatrixEntry<F> for FieldRegistersAddressMajorEntry<F> {
    fn row(&self) -> usize {
        self.row
    }

    fn column(&self) -> usize {
        usize::from(self.col)
    }
}

impl<F: Field> AddressMajorBindableEntry<F> for FieldRegistersAddressMajorEntry<F> {
    fn prev_val(&self) -> F {
        self.prev_val
    }

    fn next_val(&self) -> F {
        self.next_val
    }

    fn bind_entries(
        even: Option<&Self>,
        odd: Option<&Self>,
        even_checkpoint: F,
        odd_checkpoint: F,
        challenge: F,
    ) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => Self {
                row: even.row,
                col: even.col / 2,
                ra_coeff: even.ra_coeff + challenge.mul_01_optimized(odd.ra_coeff - even.ra_coeff),
                wa_coeff: even.wa_coeff + challenge.mul_01_optimized(odd.wa_coeff - even.wa_coeff),
                val_coeff: even.val_coeff
                    + challenge.mul_0_optimized(odd.val_coeff - even.val_coeff),
                prev_val: even.prev_val + challenge.mul_0_optimized(odd.prev_val - even.prev_val),
                next_val: even.next_val + challenge.mul_0_optimized(odd.next_val - even.next_val),
            },
            (Some(even), None) => Self {
                row: even.row,
                col: even.col / 2,
                ra_coeff: (F::one() - challenge).mul_01_optimized(even.ra_coeff),
                wa_coeff: (F::one() - challenge).mul_01_optimized(even.wa_coeff),
                val_coeff: even.val_coeff
                    + challenge.mul_0_optimized(odd_checkpoint - even.val_coeff),
                prev_val: even.prev_val + challenge.mul_0_optimized(odd_checkpoint - even.prev_val),
                next_val: even.next_val + challenge.mul_0_optimized(odd_checkpoint - even.next_val),
            },
            (None, Some(odd)) => Self {
                row: odd.row,
                col: odd.col / 2,
                ra_coeff: challenge.mul_01_optimized(odd.ra_coeff),
                wa_coeff: challenge.mul_01_optimized(odd.wa_coeff),
                val_coeff: even_checkpoint
                    + challenge.mul_0_optimized(odd.val_coeff - even_checkpoint),
                prev_val: even_checkpoint
                    + challenge.mul_0_optimized(odd.prev_val - even_checkpoint),
                next_val: even_checkpoint
                    + challenge.mul_0_optimized(odd.next_val - even_checkpoint),
            },
            (None, None) => unreachable!("field-register address bind requires at least one entry"),
        }
    }
}

impl<F> AddressMajorMessageEntry<F> for FieldRegistersAddressMajorEntry<F>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    fn accumulate_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        inputs: AddressMajorMessageInputs<F>,
        accumulators: &mut [<F as WithAccumulator>::Accumulator; 2],
    ) {
        let (ra_evals, wa_evals, val_evals) = match (even, odd) {
            (Some(even), Some(odd)) => (
                [even.ra_coeff, odd.ra_coeff + odd.ra_coeff - even.ra_coeff],
                [even.wa_coeff, odd.wa_coeff + odd.wa_coeff - even.wa_coeff],
                [
                    even.val_coeff,
                    odd.val_coeff + odd.val_coeff - even.val_coeff,
                ],
            ),
            (Some(even), None) => (
                [even.ra_coeff, -even.ra_coeff],
                [even.wa_coeff, -even.wa_coeff],
                [
                    even.val_coeff,
                    inputs.odd_checkpoint + inputs.odd_checkpoint - even.val_coeff,
                ],
            ),
            (None, Some(odd)) => (
                [F::zero(), odd.ra_coeff + odd.ra_coeff],
                [F::zero(), odd.wa_coeff + odd.wa_coeff],
                [
                    inputs.even_checkpoint,
                    odd.val_coeff + odd.val_coeff - inputs.even_checkpoint,
                ],
            ),
            (None, None) => {
                unreachable!("field-register address message requires at least one entry")
            }
        };
        for index in 0..2 {
            accumulators[index].fmadd(
                inputs.eq_eval,
                ra_evals[index] * val_evals[index]
                    + wa_evals[index] * (val_evals[index] + inputs.inc_eval),
            );
        }
    }
}

fn field_register_cycle_major<F: Field>(
    rows: &[SumcheckFieldRegistersReadWriteRow<F>],
    gamma: F,
) -> ReadWriteMatrixCycleMajor<F, FieldRegistersCycleMajorEntry<F, F>> {
    let entry_count = rows
        .iter()
        .map(field_register_entry_count)
        .map(usize::from)
        .sum();
    let mut entries = Vec::with_capacity(entry_count);
    let gamma2 = gamma * gamma;
    for (row, data) in rows.iter().copied().enumerate() {
        let mut row_entries = [FieldRegistersCycleMajorEntry::default(); 3];
        let count = fill_field_register_entries(row, data, gamma, gamma2, &mut row_entries);
        entries.extend_from_slice(&row_entries[..count]);
    }
    ReadWriteMatrixCycleMajor {
        entries,
        ra_lookup_table: None,
        wa_lookup_table: None,
    }
}

fn field_register_entry_count<F: Field>(row: &SumcheckFieldRegistersReadWriteRow<F>) -> u8 {
    let mut registers = [None; 3];
    let mut len = 0usize;
    if let Some(read) = row.rs1 {
        registers[len] = Some(read.register);
        len += 1;
    }
    if let Some(read) = row.rs2 {
        if !registers[..len].contains(&Some(read.register)) {
            registers[len] = Some(read.register);
            len += 1;
        }
    }
    if let Some(write) = row.rd {
        if !registers[..len].contains(&Some(write.register)) {
            len += 1;
        }
    }
    len as u8
}

fn fill_field_register_entries<F: Field>(
    row: usize,
    data: SumcheckFieldRegistersReadWriteRow<F>,
    gamma: F,
    gamma2: F,
    out: &mut [FieldRegistersCycleMajorEntry<F, F>],
) -> usize {
    let mut len = 0usize;
    if let Some(read) = data.rs1 {
        out[len] = read_entry(row, read, gamma);
        len += 1;
    }
    if let Some(read) = data.rs2 {
        if let Some(entry) = out[..len]
            .iter_mut()
            .find(|entry| entry.col == read.register)
        {
            entry.ra_coeff += gamma2;
        } else {
            out[len] = read_entry(row, read, gamma2);
            len += 1;
        }
    }
    if let Some(write) = data.rd {
        if let Some(entry) = out[..len]
            .iter_mut()
            .find(|entry| entry.col == write.register)
        {
            entry.wa_coeff = F::one();
            entry.next_val = write.post_value;
        } else {
            out[len] = write_entry(row, write);
            len += 1;
        }
    }
    out[..len].sort_by_key(|entry| entry.col);
    len
}

fn read_entry<F: Field>(
    row: usize,
    read: SumcheckFieldRegisterRead<F>,
    ra_coeff: F,
) -> FieldRegistersCycleMajorEntry<F, F> {
    FieldRegistersCycleMajorEntry {
        val_coeff: read.value,
        prev_val: read.value,
        next_val: read.value,
        row,
        col: read.register,
        ra_coeff,
        wa_coeff: F::zero(),
    }
}

fn write_entry<F: Field>(
    row: usize,
    write: SumcheckFieldRegisterWrite<F>,
) -> FieldRegistersCycleMajorEntry<F, F> {
    FieldRegistersCycleMajorEntry {
        val_coeff: write.pre_value,
        prev_val: write.pre_value,
        next_val: write.post_value,
        row,
        col: write.register,
        ra_coeff: F::zero(),
        wa_coeff: F::one(),
    }
}

fn field_registers_address_major_from_cycle_major<F: Field, C: OneHotCoeff<F>>(
    mut cycle_major: ReadWriteMatrixCycleMajor<F, FieldRegistersCycleMajorEntry<F, C>>,
    register_count: usize,
) -> ReadWriteMatrixAddressMajor<F, FieldRegistersAddressMajorEntry<F>> {
    let mut entries = std::mem::take(&mut cycle_major.entries);
    entries.par_sort_by(|a, b| {
        a.column()
            .cmp(&b.column())
            .then_with(|| a.row().cmp(&b.row()))
    });
    let entries = entries
        .into_par_iter()
        .map(|entry| {
            entry.to_address_major(
                cycle_major.ra_lookup_table.as_ref(),
                cycle_major.wa_lookup_table.as_ref(),
            )
        })
        .collect();
    ReadWriteMatrixAddressMajor::new_with_val_init(entries, vec![F::zero(); register_count])
}

fn materialize_cycle_major<F: Field, C: OneHotCoeff<F>>(
    matrix: ReadWriteMatrixCycleMajor<F, FieldRegistersCycleMajorEntry<F, C>>,
    register_count: usize,
    cycles: usize,
) -> (Polynomial<F>, Polynomial<F>, Polynomial<F>) {
    let len = register_count * cycles;
    let mut ra = vec![F::zero(); len];
    let mut wa = vec![F::zero(); len];
    let mut val = vec![F::zero(); len];
    for entry in matrix.entries {
        let index = entry.column() * cycles + entry.row();
        ra[index] = entry.ra_coeff.to_field(matrix.ra_lookup_table.as_ref());
        wa[index] = entry.wa_coeff.to_field(matrix.wa_lookup_table.as_ref());
        val[index] = entry.val_coeff;
    }
    (
        Polynomial::new(ra),
        Polynomial::new(wa),
        Polynomial::new(val),
    )
}

fn materialize_address_major<F: Field>(
    matrix: ReadWriteMatrixAddressMajor<F, FieldRegistersAddressMajorEntry<F>>,
    register_count: usize,
    cycles: usize,
) -> (Polynomial<F>, Polynomial<F>, Polynomial<F>) {
    let len = register_count * cycles;
    let mut ra = vec![F::zero(); len];
    let mut wa = vec![F::zero(); len];
    let mut val = vec![F::zero(); len];
    for column in matrix.entries.chunk_by(|a, b| a.column() == b.column()) {
        let register = column[0].column();
        let mut current = matrix.val_init[register];
        let mut entries = column.iter().peekable();
        for cycle in 0..cycles {
            let index = register * cycles + cycle;
            if let Some(entry) = entries.next_if(|entry| entry.row == cycle) {
                ra[index] = entry.ra_coeff;
                wa[index] = entry.wa_coeff;
                val[index] = entry.val_coeff;
                current = entry.next_val;
            } else {
                val[index] = current;
            }
        }
    }
    (
        Polynomial::new(ra),
        Polynomial::new(wa),
        Polynomial::new(val),
    )
}

fn compute_rs2_ra_claim<F: Field>(
    rs2_registers: &[Option<u8>],
    r_address: &[F],
    r_cycle: &[F],
) -> F {
    let eq_address = jolt_poly::EqPolynomial::new(r_address.to_vec()).evaluations();
    let eq_cycle = jolt_poly::EqPolynomial::new(r_cycle.to_vec()).evaluations();
    rs2_registers
        .par_iter()
        .zip(eq_cycle.par_iter())
        .filter_map(|(register, &eq)| {
            register.map(|register| eq * eq_address[usize::from(register)])
        })
        .sum()
}

fn final_claim<F: Field>(polynomial: Option<&Polynomial<F>>) -> Option<F> {
    polynomial.and_then(|poly| (poly.len() == 1).then(|| poly.evaluations()[0]))
}

fn require_poly<'a, F: Field>(
    polynomial: Option<&'a Polynomial<F>>,
    task: &'static str,
    name: &'static str,
) -> Result<&'a Polynomial<F>, BackendError> {
    polynomial.ok_or_else(|| BackendError::InvalidRequest {
        backend: "cpu",
        task,
        reason: format!("missing {name} state"),
    })
}

fn sumcheck_evals_array<F: Field, const DEGREE: usize>(
    polynomial: &Polynomial<F>,
    index: usize,
    order: BindingOrder,
) -> [F; DEGREE] {
    let (lo, hi) = polynomial.sumcheck_eval_pair(index, order);
    let mut evals = [F::zero(); DEGREE];
    evals[0] = lo;
    if DEGREE == 1 {
        return evals;
    }
    let step = hi - lo;
    let mut value = hi;
    for eval in evals.iter_mut().skip(1) {
        value += step;
        *eval = value;
    }
    evals
}

fn lt_sumcheck_evals_array<F: Field, const DEGREE: usize>(
    polynomial: &jolt_poly::LtPolynomial<F>,
    index: usize,
    order: BindingOrder,
) -> [F; DEGREE] {
    let (lo, hi) = polynomial.sumcheck_eval_pair_with_order(index, order);
    integer_domain_evals(lo, hi)
}

fn integer_domain_evals<F: Field, const DEGREE: usize>(lo: F, hi: F) -> [F; DEGREE] {
    let mut evals = [F::zero(); DEGREE];
    evals[0] = lo;
    if DEGREE == 1 {
        return evals;
    }
    let step = hi - lo;
    let mut value = hi;
    for eval in evals.iter_mut().skip(1) {
        value += step;
        *eval = value;
    }
    evals
}

fn sum_arrays<F: Field, const N: usize>(left: [F; N], right: [F; N]) -> [F; N] {
    std::array::from_fn(|i| left[i] + right[i])
}

fn reverse_cycle_table<F: Field>(values: Vec<F>, log_t: usize) -> Vec<F> {
    let mut reversed = jolt_poly::thread::unsafe_allocate_zero_vec(values.len());
    for (cycle, value) in values.into_iter().enumerate() {
        reversed[cycle.reverse_bits() >> (usize::BITS as usize - log_t)] = value;
    }
    reversed
}

fn prefix_suffix_field_rd_inc_q<F: Field>(
    field_rd_inc_by_reversed_cycle: &[F],
    prefix_vars: usize,
    suffix_vars: usize,
    suffix_point: &[F],
) -> Vec<F> {
    let prefix_len = 1usize << prefix_vars;
    let suffix_len = 1usize << suffix_vars;
    let suffix_eq = jolt_poly::EqPolynomial::<F>::evals(suffix_point, None);
    let mut q = jolt_poly::thread::unsafe_allocate_zero_vec(prefix_len);
    q.par_iter_mut()
        .enumerate()
        .for_each(|(prefix_index, output)| {
            let mut acc = F::zero();
            for (suffix_index, &suffix_weight) in suffix_eq.iter().enumerate().take(suffix_len) {
                let index = (prefix_index << suffix_vars) | suffix_index;
                acc += suffix_weight * field_rd_inc_by_reversed_cycle[index];
            }
            *output = acc;
        });
    q
}

fn validate_request<F: Field>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckFieldRegistersReadWriteStateRequest<F>,
) -> Result<(), BackendError> {
    let rows =
        1usize
            .checked_shl(request.log_t as u32)
            .ok_or_else(|| BackendError::InvalidRequest {
                backend,
                task,
                reason: format!(
                    "field-register read-write requires 2^{} rows, which does not fit in usize",
                    request.log_t
                ),
            })?;
    if request.rows.len() != rows {
        return invalid(
            backend,
            task,
            format!(
                "field-register read-write has {} rows, expected {rows}",
                request.rows.len()
            ),
        );
    }
    if request.r_cycle.len() != request.log_t {
        return invalid(
            backend,
            task,
            format!(
                "field-register read-write cycle point has {} variables, expected {}",
                request.r_cycle.len(),
                request.log_t
            ),
        );
    }
    if request.phase1_num_rounds > request.log_t || request.phase2_num_rounds > request.log_k {
        return invalid(
            backend,
            task,
            format!(
                "invalid field-register read-write phase split p1={} p2={} log_t={} log_k={}",
                request.phase1_num_rounds, request.phase2_num_rounds, request.log_t, request.log_k
            ),
        );
    }
    Ok(())
}

fn validate_val_evaluation_request<F: Field>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckFieldRegistersValEvaluationStateRequest<F>,
) -> Result<(), BackendError> {
    let rows =
        1usize
            .checked_shl(request.log_t as u32)
            .ok_or_else(|| BackendError::InvalidRequest {
                backend,
                task,
                reason: format!(
                "field-register value-evaluation requires 2^{} rows, which does not fit in usize",
                request.log_t
            ),
            })?;
    if request.rows.len() != rows {
        return invalid(
            backend,
            task,
            format!(
                "field-register value-evaluation has {} rows, expected {rows}",
                request.rows.len()
            ),
        );
    }
    if request.r_address.len() != request.log_k {
        return invalid(
            backend,
            task,
            format!(
                "field-register value-evaluation address point has {} variables, expected {}",
                request.r_address.len(),
                request.log_k
            ),
        );
    }
    if request.r_cycle.len() != request.log_t {
        return invalid(
            backend,
            task,
            format!(
                "field-register value-evaluation cycle point has {} variables, expected {}",
                request.r_cycle.len(),
                request.log_t
            ),
        );
    }
    Ok(())
}

fn validate_inc_claim_reduction_request<F: Field>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckFieldRegistersIncClaimReductionStateRequest<F>,
) -> Result<(), BackendError> {
    let rows =
        1usize
            .checked_shl(request.log_t as u32)
            .ok_or_else(|| BackendError::InvalidRequest {
                backend,
                task,
                reason: format!(
                    "field-register increment claim-reduction requires 2^{} rows, which does not fit in usize",
                    request.log_t
                ),
            })?;
    if request.rows.len() != rows {
        return invalid(
            backend,
            task,
            format!(
                "field-register increment claim-reduction has {} rows, expected {rows}",
                request.rows.len()
            ),
        );
    }
    if request.r_cycle_read_write.len() != request.log_t {
        return invalid(
            backend,
            task,
            format!(
                "field-register read-write cycle point has {} variables, expected {}",
                request.r_cycle_read_write.len(),
                request.log_t
            ),
        );
    }
    if request.r_cycle_val_evaluation.len() != request.log_t {
        return invalid(
            backend,
            task,
            format!(
                "field-register value-evaluation cycle point has {} variables, expected {}",
                request.r_cycle_val_evaluation.len(),
                request.log_t
            ),
        );
    }
    Ok(())
}

fn invalid<T>(
    backend: &'static str,
    task: &'static str,
    reason: impl Into<String>,
) -> Result<T, BackendError> {
    Err(BackendError::InvalidRequest {
        backend,
        task,
        reason: reason.into(),
    })
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::EqPolynomial;

    use super::*;

    #[test]
    fn field_registers_read_write_matches_dense_reference() {
        let log_t = 3;
        let log_k = 2;
        let phase1 = 2;
        let phase2 = 1;
        let gamma = Fr::from_u64(7);
        let fixed_cycle = [11, 13, 17].map(Fr::from_u64).to_vec();
        let rows = vec![
            row(None, None, Some(write(0, 0, 10))),
            row(Some(read(0, 10)), None, Some(write(1, 0, 15))),
            row(Some(read(1, 15)), Some(read(0, 10)), Some(write(1, 15, 25))),
            row(None, None, None),
            row(None, Some(read(1, 25)), None),
            row(Some(read(1, 25)), None, Some(write(2, 0, 8))),
            row(Some(read(2, 8)), Some(read(1, 25)), None),
            row(None, None, Some(write(0, 10, 12))),
        ];
        let input_claim = dense_input_claim(&rows, &fixed_cycle, gamma);
        let request = SumcheckFieldRegistersReadWriteStateRequest::new(
            "test.field_registers_read_write",
            rows.clone(),
            fixed_cycle.clone(),
            gamma,
            input_claim,
            log_t,
            log_k,
            phase1,
            phase2,
        );
        let mut state = FieldRegistersReadWriteState::new("cpu", "test", &request).unwrap();
        let challenges = [19, 23, 29, 31, 37].map(Fr::from_u64);
        let mut claim = input_claim;
        for challenge in challenges {
            let round = state.evaluate_round("cpu", "test", claim).unwrap();
            assert_eq!(
                round.evaluate(Fr::from_u64(0)) + round.evaluate(Fr::from_u64(1)),
                claim
            );
            claim = round.evaluate(challenge);
            state.bind("cpu", "test", challenge).unwrap();
        }

        let (r_address, r_cycle) = opening_point(&challenges, log_t, log_k, phase1, phase2);
        let opening_point = [r_address.as_slice(), r_cycle.as_slice()].concat();
        let output = state.output_claims(&opening_point).unwrap();
        let reference = dense_output_claims(&rows, &r_address, &r_cycle);
        assert_eq!(output, reference);

        let eq_cycle = eq_eval(&fixed_cycle, &r_cycle);
        let expected_final = eq_cycle
            * (reference.rd_wa * (reference.rd_inc + reference.registers_val)
                + gamma * reference.rs1_ra * reference.registers_val
                + gamma * gamma * reference.rs2_ra * reference.registers_val);
        assert_eq!(claim, expected_final);
    }

    #[test]
    fn field_registers_val_evaluation_matches_dense_reference() {
        let log_t = 3;
        let log_k = 2;
        let fixed_address = [5, 7].map(Fr::from_u64).to_vec();
        let fixed_cycle = [11, 13, 17].map(Fr::from_u64).to_vec();
        let rows = vec![
            row(None, None, Some(write(0, 0, 10))),
            row(Some(read(0, 10)), None, Some(write(1, 0, 15))),
            row(Some(read(1, 15)), Some(read(0, 10)), Some(write(1, 15, 25))),
            row(None, None, None),
            row(None, Some(read(1, 25)), None),
            row(Some(read(1, 25)), None, Some(write(2, 0, 8))),
            row(Some(read(2, 8)), Some(read(1, 25)), None),
            row(None, None, Some(write(0, 10, 12))),
        ];
        let input_claim = dense_output_claims(&rows, &fixed_address, &fixed_cycle).registers_val;
        let request = SumcheckFieldRegistersValEvaluationStateRequest::new(
            "test.field_registers_val_evaluation",
            rows.clone(),
            fixed_address.clone(),
            fixed_cycle.clone(),
            input_claim,
            log_t,
            log_k,
        );
        let mut state = FieldRegistersValEvaluationState::new("cpu", "test", &request).unwrap();
        let challenges = [19, 23, 29].map(Fr::from_u64);
        let mut claim = input_claim;
        for challenge in challenges {
            let round = state.evaluate_round("cpu", "test", claim).unwrap();
            assert_eq!(
                round.evaluate(Fr::from_u64(0)) + round.evaluate(Fr::from_u64(1)),
                claim
            );
            claim = round.evaluate(challenge);
            state.bind("cpu", "test", challenge).unwrap();
        }

        let r_cycle = challenges.iter().rev().copied().collect::<Vec<_>>();
        let opening_claims = state.output_claims().unwrap();
        let reference = dense_output_claims(&rows, &fixed_address, &r_cycle);
        assert_eq!(opening_claims.field_rd_inc, reference.rd_inc);
        assert_eq!(opening_claims.field_rd_wa, reference.rd_wa);

        let expected_final = jolt_poly::LtPolynomial::evaluate(&r_cycle, &fixed_cycle)
            * reference.rd_inc
            * reference.rd_wa;
        assert_eq!(claim, expected_final);
    }

    #[test]
    fn field_registers_inc_claim_reduction_matches_dense_reference() {
        let log_t = 3;
        let read_write_cycle = [11, 13, 17].map(Fr::from_u64).to_vec();
        let val_evaluation_cycle = [19, 23, 29].map(Fr::from_u64).to_vec();
        let gamma = Fr::from_u64(31);
        let rows = vec![
            row(None, None, Some(write(0, 0, 10))),
            row(Some(read(0, 10)), None, Some(write(1, 0, 15))),
            row(Some(read(1, 15)), Some(read(0, 10)), Some(write(1, 15, 25))),
            row(None, None, None),
            row(None, Some(read(1, 25)), None),
            row(Some(read(1, 25)), None, Some(write(2, 0, 8))),
            row(Some(read(2, 8)), Some(read(1, 25)), None),
            row(None, None, Some(write(0, 10, 12))),
        ];
        let read_write_request_cycle = read_write_cycle.iter().rev().copied().collect::<Vec<_>>();
        let val_evaluation_request_cycle = val_evaluation_cycle
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        let input_claim = dense_inc_reduction_input_claim(
            &rows,
            &read_write_request_cycle,
            &val_evaluation_request_cycle,
            gamma,
            log_t,
        );
        let request = SumcheckFieldRegistersIncClaimReductionStateRequest::new(
            "test.field_registers_inc_claim_reduction",
            rows.clone(),
            read_write_request_cycle,
            val_evaluation_request_cycle,
            gamma,
            input_claim,
            log_t,
        );
        let mut state = FieldRegistersIncClaimReductionState::new("cpu", "test", &request).unwrap();
        let challenges = [37, 41, 43].map(Fr::from_u64);
        let mut claim = input_claim;
        for challenge in challenges {
            let round = state.evaluate_round("cpu", "test", claim).unwrap();
            assert_eq!(
                round.evaluate(Fr::from_u64(0)) + round.evaluate(Fr::from_u64(1)),
                claim
            );
            claim = round.evaluate(challenge);
            state.bind("cpu", "test", challenge).unwrap();
        }

        let r_cycle = challenges.iter().rev().copied().collect::<Vec<_>>();
        let opening_claims = state.output_claims().unwrap();
        let field_rd_inc = dense_field_rd_inc(&rows, &r_cycle);
        assert_eq!(opening_claims.field_rd_inc, field_rd_inc);

        let expected_final = (eq_eval(&r_cycle, &read_write_cycle)
            + gamma * eq_eval(&r_cycle, &val_evaluation_cycle))
            * field_rd_inc;
        assert_eq!(claim, expected_final);
    }

    fn read(register: u8, value: u64) -> SumcheckFieldRegisterRead<Fr> {
        SumcheckFieldRegisterRead {
            register,
            value: Fr::from_u64(value),
        }
    }

    fn write(register: u8, pre_value: u64, post_value: u64) -> SumcheckFieldRegisterWrite<Fr> {
        SumcheckFieldRegisterWrite {
            register,
            pre_value: Fr::from_u64(pre_value),
            post_value: Fr::from_u64(post_value),
        }
    }

    fn row(
        rs1: Option<SumcheckFieldRegisterRead<Fr>>,
        rs2: Option<SumcheckFieldRegisterRead<Fr>>,
        rd: Option<SumcheckFieldRegisterWrite<Fr>>,
    ) -> SumcheckFieldRegistersReadWriteRow<Fr> {
        let rd_increment = rd.map_or_else(
            || Fr::from_u64(0),
            |write| write.post_value - write.pre_value,
        );
        SumcheckFieldRegistersReadWriteRow {
            rs1,
            rs2,
            rd,
            rd_increment,
        }
    }

    fn dense_input_claim(
        rows: &[SumcheckFieldRegistersReadWriteRow<Fr>],
        fixed_cycle: &[Fr],
        gamma: Fr,
    ) -> Fr {
        let eq_cycle = EqPolynomial::<Fr>::evals(fixed_cycle, None);
        rows.iter()
            .zip(eq_cycle)
            .map(|(row, eq)| {
                let rd = row
                    .rd
                    .map_or_else(|| Fr::from_u64(0), |write| write.post_value);
                let rs1 = row.rs1.map_or_else(|| Fr::from_u64(0), |read| read.value);
                let rs2 = row.rs2.map_or_else(|| Fr::from_u64(0), |read| read.value);
                eq * (rd + gamma * rs1 + gamma * gamma * rs2)
            })
            .sum()
    }

    fn dense_output_claims(
        rows: &[SumcheckFieldRegistersReadWriteRow<Fr>],
        r_address: &[Fr],
        r_cycle: &[Fr],
    ) -> SumcheckRegistersReadWriteOutput<Fr> {
        let eq_address = EqPolynomial::<Fr>::evals(r_address, None);
        let eq_cycle = EqPolynomial::<Fr>::evals(r_cycle, None);
        let mut state = vec![Fr::from_u64(0); eq_address.len()];
        let mut registers_val = Fr::from_u64(0);
        let mut rs1_ra = Fr::from_u64(0);
        let mut rs2_ra = Fr::from_u64(0);
        let mut rd_wa = Fr::from_u64(0);
        let mut rd_inc = Fr::from_u64(0);

        for (cycle, row) in rows.iter().enumerate() {
            for (register, value) in state.iter().enumerate() {
                registers_val += eq_address[register] * eq_cycle[cycle] * *value;
            }
            if let Some(read) = row.rs1 {
                rs1_ra += eq_address[usize::from(read.register)] * eq_cycle[cycle];
            }
            if let Some(read) = row.rs2 {
                rs2_ra += eq_address[usize::from(read.register)] * eq_cycle[cycle];
            }
            if let Some(write) = row.rd {
                rd_wa += eq_address[usize::from(write.register)] * eq_cycle[cycle];
                state[usize::from(write.register)] = write.post_value;
            }
            rd_inc += eq_cycle[cycle] * row.rd_increment;
        }

        SumcheckRegistersReadWriteOutput {
            registers_val,
            rs1_ra,
            rs2_ra,
            rd_wa,
            rd_inc,
        }
    }

    fn dense_inc_reduction_input_claim(
        rows: &[SumcheckFieldRegistersReadWriteRow<Fr>],
        read_write_cycle: &[Fr],
        val_evaluation_cycle: &[Fr],
        gamma: Fr,
        log_t: usize,
    ) -> Fr {
        let inc = reverse_cycle_table(
            rows.iter().map(|row| row.rd_increment).collect::<Vec<_>>(),
            log_t,
        );
        let eq_read_write = EqPolynomial::<Fr>::evals(read_write_cycle, None);
        let eq_val_evaluation = EqPolynomial::<Fr>::evals(val_evaluation_cycle, None);
        inc.into_iter()
            .zip(eq_read_write.into_iter().zip(eq_val_evaluation))
            .map(|(inc, (read_write, val_evaluation))| inc * (read_write + gamma * val_evaluation))
            .sum()
    }

    fn dense_field_rd_inc(rows: &[SumcheckFieldRegistersReadWriteRow<Fr>], r_cycle: &[Fr]) -> Fr {
        let eq_cycle = EqPolynomial::<Fr>::evals(r_cycle, None);
        rows.iter()
            .zip(eq_cycle)
            .map(|(row, eq)| row.rd_increment * eq)
            .sum()
    }

    fn opening_point(
        challenges: &[Fr],
        log_t: usize,
        log_k: usize,
        phase1: usize,
        phase2: usize,
    ) -> (Vec<Fr>, Vec<Fr>) {
        let (phase1_challenges, rest) = challenges.split_at(phase1);
        let (phase2_challenges, rest) = rest.split_at(phase2);
        let (phase3_cycle, phase3_address) = rest.split_at(log_t - phase1);
        assert_eq!(phase3_address.len(), log_k - phase2);
        let r_cycle = phase3_cycle
            .iter()
            .rev()
            .copied()
            .chain(phase1_challenges.iter().rev().copied())
            .collect::<Vec<_>>();
        let r_address = phase3_address
            .iter()
            .rev()
            .copied()
            .chain(phase2_challenges.iter().rev().copied())
            .collect::<Vec<_>>();
        (r_address, r_cycle)
    }

    fn eq_eval(left: &[Fr], right: &[Fr]) -> Fr {
        left.iter()
            .zip(right)
            .map(|(&a, &b)| (Fr::from_u64(1) - a) * (Fr::from_u64(1) - b) + a * b)
            .product()
    }
}
