use jolt_field::{Field, OptimizedMul, RingAccumulator, WithAccumulator};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use rayon::prelude::*;

use crate::{
    BackendError, SumcheckRegisterRead, SumcheckRegisterWrite, SumcheckRegistersReadWriteOutput,
    SumcheckRegistersReadWriteRow, SumcheckRegistersReadWriteStateRequest,
    SumcheckRegistersValEvaluationOutput, SumcheckRegistersValEvaluationStateRequest,
};

use super::{
    AddressMajorBindableEntry, AddressMajorMatrixEntry, AddressMajorMessageEntry,
    AddressMajorMessageInputs, CycleMajorMatrixEntry, CycleMajorMessageEntry,
    CycleMajorToAddressMajor, OneHotCoeff, OneHotCoeffIndex, OneHotCoeffTable,
    ReadWriteMatrixAddressMajor, ReadWriteMatrixCycleMajor,
};

const DEGREE_BOUND: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RegistersReadWriteParams<F: Field> {
    log_t: usize,
    log_k: usize,
    phase1_num_rounds: usize,
    phase2_num_rounds: usize,
    gamma: F,
}

impl<F: Field> RegistersReadWriteParams<F> {
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
enum RegistersSparseMatrix<F: Field> {
    #[default]
    None,
    CycleMajorWithLookups(
        ReadWriteMatrixCycleMajor<F, RegistersCycleMajorEntry<F, OneHotCoeffIndex>>,
    ),
    CycleMajor(ReadWriteMatrixCycleMajor<F, RegistersCycleMajorEntry<F, F>>),
    AddressMajor(ReadWriteMatrixAddressMajor<F, RegistersAddressMajorEntry<F>>),
}

impl<F: Field> RegistersSparseMatrix<F> {
    fn bind(&mut self, challenge: F) {
        match self {
            Self::None => unreachable!("cannot bind empty register sparse matrix"),
            Self::CycleMajorWithLookups(matrix) => matrix.bind(challenge),
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
            Self::None => unreachable!("cannot materialize empty register sparse matrix"),
            Self::CycleMajorWithLookups(matrix) => {
                materialize_cycle_major(matrix, register_count, cycles)
            }
            Self::CycleMajor(matrix) => materialize_cycle_major(matrix, register_count, cycles),
            Self::AddressMajor(matrix) => materialize_address_major(matrix, register_count, cycles),
        }
    }
}

pub struct RegistersReadWriteState<F: Field> {
    sparse_matrix: RegistersSparseMatrix<F>,
    gruen_eq: Option<GruenSplitEqPolynomial<F>>,
    inc: Polynomial<F>,
    ra: Option<Polynomial<F>>,
    wa: Option<Polynomial<F>>,
    val: Option<Polynomial<F>>,
    merged_eq: Option<Polynomial<F>>,
    input_claim: F,
    params: RegistersReadWriteParams<F>,
    rs2_registers: Vec<Option<u8>>,
    round: usize,
}

impl<F> RegistersReadWriteState<F>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    pub fn new(
        backend: &'static str,
        task: &'static str,
        request: &SumcheckRegistersReadWriteStateRequest<F>,
    ) -> Result<Self, BackendError> {
        validate_request(backend, task, request)?;
        let params = RegistersReadWriteParams {
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
        let inc = Polynomial::new(
            request
                .rows
                .iter()
                .map(|row| F::from_i128(row.rd_increment))
                .collect(),
        );
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

        let sparse_matrix = register_cycle_major(&request.rows, request.gamma);
        let (sparse_matrix, ra, wa, val) = if params.phase1_num_rounds > 0 {
            (
                RegistersSparseMatrix::CycleMajorWithLookups(sparse_matrix),
                None,
                None,
                None,
            )
        } else if params.phase2_num_rounds > 0 {
            (
                RegistersSparseMatrix::AddressMajor(registers_address_major_from_cycle_major(
                    sparse_matrix.deref_coeffs(),
                    params.register_count(),
                )),
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
            (RegistersSparseMatrix::None, Some(ra), Some(wa), Some(val))
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
                    "register read-write round {} is outside {} rounds",
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
                    "register read-write bind round {} is outside {} rounds",
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
        registers_opening_point: &[F],
    ) -> Result<SumcheckRegistersReadWriteOutput<F>, BackendError> {
        let Some(val) = final_claim(self.val.as_ref()) else {
            return invalid(
                "cpu",
                "register read-write output claims",
                "missing value state",
            );
        };
        let Some(combined_ra) = final_claim(self.ra.as_ref()) else {
            return invalid(
                "cpu",
                "register read-write output claims",
                "missing RA state",
            );
        };
        let Some(rd_wa) = final_claim(self.wa.as_ref()) else {
            return invalid(
                "cpu",
                "register read-write output claims",
                "missing WA state",
            );
        };
        let rd_inc = self.inc.evaluations().first().copied().unwrap_or(F::zero());
        let (r_address, r_cycle) = registers_opening_point.split_at(self.params.log_k);
        let rs2_ra = compute_rs2_ra_claim(&self.rs2_registers, r_address, r_cycle);
        let gamma_inv = self.gamma_inverse()?;
        let rs1_ra = (combined_ra - self.params.gamma * self.params.gamma * rs2_ra) * gamma_inv;
        Ok(SumcheckRegistersReadWriteOutput {
            registers_val: val,
            rs1_ra,
            rs2_ra,
            rd_wa,
            rd_inc,
        })
    }

    fn gamma_inverse(&self) -> Result<F, BackendError> {
        self.params
            .gamma
            .inverse()
            .ok_or_else(|| BackendError::InvalidRequest {
                backend: "cpu",
                task: "register read-write output claims",
                reason: "registers read-write gamma is not invertible".to_owned(),
            })
    }

    fn phase1_compute_message(&self, previous_claim: F) -> Result<UnivariatePoly<F>, BackendError> {
        match &self.sparse_matrix {
            RegistersSparseMatrix::CycleMajorWithLookups(matrix) => {
                self.phase1_compute_message_for_matrix(matrix, previous_claim)
            }
            RegistersSparseMatrix::CycleMajor(matrix) => {
                self.phase1_compute_message_for_matrix(matrix, previous_claim)
            }
            _ => invalid(
                "cpu",
                "register read-write phase1",
                "missing cycle-major matrix",
            ),
        }
    }

    fn phase1_compute_message_for_matrix<C>(
        &self,
        matrix: &ReadWriteMatrixCycleMajor<F, RegistersCycleMajorEntry<F, C>>,
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
                task: "register read-write phase1",
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
                task: "register read-write phase2",
                reason: "missing phase2 equality state".to_owned(),
            })?;
        let matrix = match &self.sparse_matrix {
            RegistersSparseMatrix::AddressMajor(matrix) => matrix,
            _ => {
                return invalid(
                    "cpu",
                    "register read-write phase2",
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
        let merged_eq = require_poly(self.merged_eq.as_ref(), "register read-write phase3", "eq")?;
        let ra = require_poly(self.ra.as_ref(), "register read-write phase3", "RA")?;
        let wa = require_poly(self.wa.as_ref(), "register read-write phase3", "WA")?;
        let val = require_poly(self.val.as_ref(), "register read-write phase3", "value")?;
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
        if let RegistersSparseMatrix::CycleMajorWithLookups(matrix) = &mut self.sparse_matrix {
            let saturated = matrix
                .ra_lookup_table
                .as_ref()
                .is_some_and(OneHotCoeffTable::is_saturated)
                || matrix
                    .wa_lookup_table
                    .as_ref()
                    .is_some_and(OneHotCoeffTable::is_saturated);
            if saturated {
                let matrix = std::mem::take(matrix);
                self.sparse_matrix = RegistersSparseMatrix::CycleMajor(matrix.deref_coeffs());
            }
        }
        self.sparse_matrix.bind(challenge);
        if self.round == self.params.phase1_num_rounds - 1 {
            if let Some(eq) = self.gruen_eq.as_ref() {
                self.merged_eq = Some(eq.merge());
            }
            let matrix = std::mem::take(&mut self.sparse_matrix);
            if self.params.phase2_num_rounds > 0 {
                self.sparse_matrix = match matrix {
                    RegistersSparseMatrix::CycleMajorWithLookups(matrix) => {
                        RegistersSparseMatrix::AddressMajor(
                            registers_address_major_from_cycle_major(
                                matrix.deref_coeffs(),
                                self.params.register_count(),
                            ),
                        )
                    }
                    RegistersSparseMatrix::CycleMajor(matrix) => {
                        RegistersSparseMatrix::AddressMajor(
                            registers_address_major_from_cycle_major(
                                matrix,
                                self.params.register_count(),
                            ),
                        )
                    }
                    _ => unreachable!("phase1 output must be cycle-major"),
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

pub struct RegistersValEvaluationState<F: Field> {
    inc: Polynomial<F>,
    wa: Polynomial<F>,
    lt: Polynomial<F>,
    input_claim: F,
    log_t: usize,
    round: usize,
}

impl<F: Field> RegistersValEvaluationState<F> {
    pub fn new(
        backend: &'static str,
        task: &'static str,
        request: &SumcheckRegistersValEvaluationStateRequest<F>,
    ) -> Result<Self, BackendError> {
        validate_registers_val_evaluation_request(backend, task, request)?;
        let register_count = 1usize << request.log_k;
        let eq_register = jolt_poly::EqPolynomial::new(request.r_address.clone()).evaluations();
        let inc = request
            .rows
            .iter()
            .map(|row| F::from_i128(row.rd_increment))
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
                            "registers value-evaluation write address {register} is outside {register_count} registers"
                        ),
                    );
                }
                Ok(eq_register[register])
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
        let lt = jolt_poly::LtPolynomial::evaluations(&request.r_cycle);

        Ok(Self {
            inc: Polynomial::new(inc),
            wa: Polynomial::new(wa),
            lt: Polynomial::new(lt),
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
                    "registers value-evaluation round {} is outside {} rounds",
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
                let lt = sumcheck_evals_array::<F, DEGREE_BOUND>(
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
                    "registers value-evaluation bind round {} is outside {} rounds",
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

    pub fn output_claims(&self) -> Result<SumcheckRegistersValEvaluationOutput<F>, BackendError> {
        let Some(&rd_inc) = self.inc.evaluations().first() else {
            return invalid(
                "cpu",
                "registers value-evaluation output claims",
                "empty register increment state",
            );
        };
        let Some(&rd_wa) = self.wa.evaluations().first() else {
            return invalid(
                "cpu",
                "registers value-evaluation output claims",
                "empty register write-address state",
            );
        };
        Ok(SumcheckRegistersValEvaluationOutput { rd_inc, rd_wa })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RegistersCycleMajorEntry<F: Field, C: OneHotCoeff<F>> {
    val_coeff: F,
    prev_val: u64,
    next_val: u64,
    row: usize,
    col: u8,
    ra_coeff: C,
    wa_coeff: C,
}

impl<F: Field, C: OneHotCoeff<F> + Default> Default for RegistersCycleMajorEntry<F, C> {
    fn default() -> Self {
        Self {
            val_coeff: F::zero(),
            prev_val: 0,
            next_val: 0,
            row: 0,
            col: 0,
            ra_coeff: C::default(),
            wa_coeff: C::default(),
        }
    }
}

impl<F: Field> ReadWriteMatrixCycleMajor<F, RegistersCycleMajorEntry<F, OneHotCoeffIndex>> {
    fn deref_coeffs(self) -> ReadWriteMatrixCycleMajor<F, RegistersCycleMajorEntry<F, F>> {
        let Some(ra_lookup_table) = self.ra_lookup_table.as_ref() else {
            unreachable!("register RA lookup table must exist");
        };
        let Some(wa_lookup_table) = self.wa_lookup_table.as_ref() else {
            unreachable!("register WA lookup table must exist");
        };
        let entries = self
            .entries
            .into_par_iter()
            .map(|entry| RegistersCycleMajorEntry {
                val_coeff: entry.val_coeff,
                prev_val: entry.prev_val,
                next_val: entry.next_val,
                row: entry.row,
                col: entry.col,
                ra_coeff: entry.ra_coeff.to_field(Some(ra_lookup_table)),
                wa_coeff: entry.wa_coeff.to_field(Some(wa_lookup_table)),
            })
            .collect();
        ReadWriteMatrixCycleMajor {
            entries,
            ra_lookup_table: None,
            wa_lookup_table: None,
        }
    }
}

impl<F: Field, C: OneHotCoeff<F>> CycleMajorMatrixEntry<F> for RegistersCycleMajorEntry<F, C> {
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
            (Some(even), None) => {
                let odd_val = F::from_u64(even.next_val);
                Self {
                    row: even.row / 2,
                    col: even.col,
                    ra_coeff: OneHotCoeff::bind(
                        Some(&even.ra_coeff),
                        None,
                        challenge,
                        ra_lookup_table,
                    ),
                    wa_coeff: OneHotCoeff::bind(
                        Some(&even.wa_coeff),
                        None,
                        challenge,
                        wa_lookup_table,
                    ),
                    val_coeff: even.val_coeff + challenge.mul_0_optimized(odd_val - even.val_coeff),
                    prev_val: even.prev_val,
                    next_val: even.next_val,
                }
            }
            (None, Some(odd)) => {
                let even_val = F::from_u64(odd.prev_val);
                Self {
                    row: odd.row / 2,
                    col: odd.col,
                    ra_coeff: OneHotCoeff::bind(
                        None,
                        Some(&odd.ra_coeff),
                        challenge,
                        ra_lookup_table,
                    ),
                    wa_coeff: OneHotCoeff::bind(
                        None,
                        Some(&odd.wa_coeff),
                        challenge,
                        wa_lookup_table,
                    ),
                    val_coeff: even_val + challenge.mul_0_optimized(odd.val_coeff - even_val),
                    prev_val: odd.prev_val,
                    next_val: odd.next_val,
                }
            }
            (None, None) => unreachable!("register bind requires at least one entry"),
        }
    }
}

impl<F, C> CycleMajorMessageEntry<F> for RegistersCycleMajorEntry<F, C>
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
            (Some(even), None) => {
                let odd_val = F::from_u64(even.next_val);
                (
                    OneHotCoeff::evals(Some(&even.ra_coeff), None, ra_lookup_table),
                    OneHotCoeff::evals(Some(&even.wa_coeff), None, wa_lookup_table),
                    [even.val_coeff, odd_val - even.val_coeff],
                )
            }
            (None, Some(odd)) => {
                let even_val = F::from_u64(odd.prev_val);
                (
                    OneHotCoeff::evals(None, Some(&odd.ra_coeff), ra_lookup_table),
                    OneHotCoeff::evals(None, Some(&odd.wa_coeff), wa_lookup_table),
                    [even_val, odd.val_coeff - even_val],
                )
            }
            (None, None) => unreachable!("register message requires at least one entry"),
        };
        for index in 0..2 {
            accumulators[index].fmadd(ra_evals[index], val_evals[index]);
            accumulators[index].fmadd(wa_evals[index], val_evals[index] + inc_evals[index]);
        }
    }
}

impl<F: Field, C: OneHotCoeff<F>> CycleMajorToAddressMajor<F> for RegistersCycleMajorEntry<F, C> {
    type AddressMajor = RegistersAddressMajorEntry<F>;

    fn to_address_major(
        self,
        ra_lookup_table: Option<&OneHotCoeffTable<F>>,
        wa_lookup_table: Option<&OneHotCoeffTable<F>>,
    ) -> Self::AddressMajor {
        RegistersAddressMajorEntry {
            prev_val: F::from_u64(self.prev_val),
            next_val: F::from_u64(self.next_val),
            val_coeff: self.val_coeff,
            ra_coeff: self.ra_coeff.to_field(ra_lookup_table),
            wa_coeff: self.wa_coeff.to_field(wa_lookup_table),
            row: self.row,
            col: self.col,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RegistersAddressMajorEntry<F: Field> {
    prev_val: F,
    next_val: F,
    val_coeff: F,
    ra_coeff: F,
    wa_coeff: F,
    row: usize,
    col: u8,
}

impl<F: Field> AddressMajorMatrixEntry<F> for RegistersAddressMajorEntry<F> {
    fn row(&self) -> usize {
        self.row
    }

    fn column(&self) -> usize {
        usize::from(self.col)
    }
}

impl<F: Field> AddressMajorBindableEntry<F> for RegistersAddressMajorEntry<F> {
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
            (None, None) => unreachable!("register address bind requires at least one entry"),
        }
    }
}

impl<F> AddressMajorMessageEntry<F> for RegistersAddressMajorEntry<F>
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
            (None, None) => unreachable!("register address message requires at least one entry"),
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

fn register_cycle_major<F: Field>(
    rows: &[SumcheckRegistersReadWriteRow],
    gamma: F,
) -> ReadWriteMatrixCycleMajor<F, RegistersCycleMajorEntry<F, OneHotCoeffIndex>> {
    let entry_count = rows.iter().map(register_entry_count).map(usize::from).sum();
    let mut entries = Vec::with_capacity(entry_count);
    for (row, data) in rows.iter().copied().enumerate() {
        let mut row_entries = [RegistersCycleMajorEntry::default(); 3];
        let count = fill_register_entries(row, data, &mut row_entries);
        entries.extend_from_slice(&row_entries[..count]);
    }
    ReadWriteMatrixCycleMajor {
        entries,
        ra_lookup_table: Some(OneHotCoeffTable::new(vec![
            F::zero(),
            gamma,
            gamma * gamma,
            gamma + gamma * gamma,
        ])),
        wa_lookup_table: Some(OneHotCoeffTable::new(vec![F::zero(), F::one()])),
    }
}

fn register_entry_count(row: &SumcheckRegistersReadWriteRow) -> u8 {
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

fn fill_register_entries<F: Field>(
    row: usize,
    data: SumcheckRegistersReadWriteRow,
    out: &mut [RegistersCycleMajorEntry<F, OneHotCoeffIndex>],
) -> usize {
    let mut len = 0usize;
    if let Some(read) = data.rs1 {
        out[len] = read_entry(row, read, OneHotCoeffIndex(1));
        len += 1;
    }
    if let Some(read) = data.rs2 {
        if let Some(entry) = out[..len]
            .iter_mut()
            .find(|entry| entry.col == read.register)
        {
            entry.ra_coeff = OneHotCoeffIndex(3);
        } else {
            out[len] = read_entry(row, read, OneHotCoeffIndex(2));
            len += 1;
        }
    }
    if let Some(write) = data.rd {
        if let Some(entry) = out[..len]
            .iter_mut()
            .find(|entry| entry.col == write.register)
        {
            entry.wa_coeff = OneHotCoeffIndex(1);
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
    read: SumcheckRegisterRead,
    ra_coeff: OneHotCoeffIndex,
) -> RegistersCycleMajorEntry<F, OneHotCoeffIndex> {
    RegistersCycleMajorEntry {
        val_coeff: F::from_u64(read.value),
        prev_val: read.value,
        next_val: read.value,
        row,
        col: read.register,
        ra_coeff,
        wa_coeff: OneHotCoeffIndex(0),
    }
}

fn write_entry<F: Field>(
    row: usize,
    write: SumcheckRegisterWrite,
) -> RegistersCycleMajorEntry<F, OneHotCoeffIndex> {
    RegistersCycleMajorEntry {
        val_coeff: F::from_u64(write.pre_value),
        prev_val: write.pre_value,
        next_val: write.post_value,
        row,
        col: write.register,
        ra_coeff: OneHotCoeffIndex(0),
        wa_coeff: OneHotCoeffIndex(1),
    }
}

fn registers_address_major_from_cycle_major<F: Field, C: OneHotCoeff<F>>(
    mut cycle_major: ReadWriteMatrixCycleMajor<F, RegistersCycleMajorEntry<F, C>>,
    register_count: usize,
) -> ReadWriteMatrixAddressMajor<F, RegistersAddressMajorEntry<F>> {
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
    matrix: ReadWriteMatrixCycleMajor<F, RegistersCycleMajorEntry<F, C>>,
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
    matrix: ReadWriteMatrixAddressMajor<F, RegistersAddressMajorEntry<F>>,
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

fn sum_arrays<F: Field, const N: usize>(left: [F; N], right: [F; N]) -> [F; N] {
    std::array::from_fn(|i| left[i] + right[i])
}

fn validate_registers_val_evaluation_request<F: Field>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckRegistersValEvaluationStateRequest<F>,
) -> Result<(), BackendError> {
    let rows =
        1usize
            .checked_shl(request.log_t as u32)
            .ok_or_else(|| BackendError::InvalidRequest {
                backend,
                task,
                reason: format!(
                    "registers value-evaluation requires 2^{} rows, which does not fit in usize",
                    request.log_t
                ),
            })?;
    if request.rows.len() != rows {
        return invalid(
            backend,
            task,
            format!(
                "registers value-evaluation has {} rows, expected {rows}",
                request.rows.len()
            ),
        );
    }
    if request.r_address.len() != request.log_k {
        return invalid(
            backend,
            task,
            format!(
                "registers value-evaluation address point has {} variables, expected {}",
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
                "registers value-evaluation cycle point has {} variables, expected {}",
                request.r_cycle.len(),
                request.log_t
            ),
        );
    }
    Ok(())
}

fn validate_request<F: Field>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckRegistersReadWriteStateRequest<F>,
) -> Result<(), BackendError> {
    let rows =
        1usize
            .checked_shl(request.log_t as u32)
            .ok_or_else(|| BackendError::InvalidRequest {
                backend,
                task,
                reason: format!(
                    "register read-write requires 2^{} rows, which does not fit in usize",
                    request.log_t
                ),
            })?;
    if request.rows.len() != rows {
        return invalid(
            backend,
            task,
            format!(
                "register read-write has {} rows, expected {rows}",
                request.rows.len()
            ),
        );
    }
    if request.r_cycle.len() != request.log_t {
        return invalid(
            backend,
            task,
            format!(
                "register read-write cycle point has {} variables, expected {}",
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
                "invalid register read-write phase split p1={} p2={} log_t={} log_k={}",
                request.phase1_num_rounds, request.phase2_num_rounds, request.log_t, request.log_k
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
