use jolt_field::{Field, OptimizedMul, RingAccumulator, WithAccumulator};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use rayon::prelude::*;

use crate::{
    BackendError, SumcheckRamOutputCheckStateRequest, SumcheckRamRaClaimReductionOutput,
    SumcheckRamRaClaimReductionStateRequest, SumcheckRamRafStateRequest, SumcheckRamReadWriteRow,
    SumcheckRamReadWriteStateRequest, SumcheckRamValCheckOutput, SumcheckRamValCheckStateRequest,
};

use super::{
    AddressMajorBindableEntry, AddressMajorMatrixEntry, AddressMajorMessageEntry,
    AddressMajorMessageInputs, CycleMajorMatrixEntry, CycleMajorMessageEntry,
    CycleMajorToAddressMajor, OneHotCoeffTable, ReadWriteMatrixAddressMajor,
    ReadWriteMatrixCycleMajor,
};

const DEGREE_BOUND: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RamReadWriteParams<F: Field> {
    log_t: usize,
    log_k: usize,
    phase1_num_rounds: usize,
    phase2_num_rounds: usize,
    gamma: F,
}

impl<F: Field> RamReadWriteParams<F> {
    const fn rounds(self) -> usize {
        self.log_t + self.log_k
    }

    const fn phase3_cycle_rounds(self) -> usize {
        self.log_t - self.phase1_num_rounds
    }

    const fn address_count(self) -> usize {
        1usize << self.log_k
    }
}

pub struct RamReadWriteState<F: Field> {
    sparse_matrix_phase1: ReadWriteMatrixCycleMajor<F, RamCycleMajorEntry<F>>,
    sparse_matrix_phase2: ReadWriteMatrixAddressMajor<F, RamAddressMajorEntry<F>>,
    gruen_eq: Option<GruenSplitEqPolynomial<F>>,
    inc: Polynomial<F>,
    ra: Option<Polynomial<F>>,
    val: Option<Polynomial<F>>,
    merged_eq: Option<Polynomial<F>>,
    val_init: Vec<F>,
    params: RamReadWriteParams<F>,
    round: usize,
}

impl<F> RamReadWriteState<F>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    pub fn new(
        backend: &'static str,
        task: &'static str,
        request: &SumcheckRamReadWriteStateRequest<F>,
    ) -> Result<Self, BackendError> {
        validate_request(backend, task, request)?;

        let params = RamReadWriteParams {
            log_t: request.log_t,
            log_k: request.log_k,
            phase1_num_rounds: request.phase1_num_rounds,
            phase2_num_rounds: request.phase2_num_rounds,
            gamma: request.gamma,
        };
        let val_init = request
            .initial_ram_state
            .par_iter()
            .map(|&value| F::from_u64(value))
            .collect::<Vec<_>>();
        let entries = request
            .rows
            .par_iter()
            .enumerate()
            .filter_map(|(row, entry)| RamCycleMajorEntry::from_row(row, *entry))
            .collect::<Vec<_>>();
        let sparse_matrix = ReadWriteMatrixCycleMajor::new(entries);
        let inc = Polynomial::new(
            request
                .rows
                .iter()
                .map(|row| F::from_i128(row.ram_increment))
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

        let (sparse_matrix_phase1, sparse_matrix_phase2, ra, val) = if params.phase1_num_rounds > 0
        {
            (
                sparse_matrix,
                ReadWriteMatrixAddressMajor::default(),
                None,
                None,
            )
        } else if params.phase2_num_rounds > 0 {
            (
                ReadWriteMatrixCycleMajor::default(),
                ram_address_major_from_cycle_major(sparse_matrix, val_init.clone()),
                None,
                None,
            )
        } else {
            let (ra, val) = materialize_cycle_major(
                sparse_matrix,
                params.address_count(),
                1usize << params.log_t,
                &val_init,
            );
            (
                ReadWriteMatrixCycleMajor::default(),
                ReadWriteMatrixAddressMajor::default(),
                Some(ra),
                Some(val),
            )
        };

        Ok(Self {
            sparse_matrix_phase1,
            sparse_matrix_phase2,
            gruen_eq,
            inc,
            ra,
            val,
            merged_eq,
            val_init,
            params,
            round: 0,
        })
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
                    "RAM read-write round {} is outside {} rounds",
                    self.round,
                    self.params.rounds()
                ),
            );
        }

        Ok(if self.round < self.params.phase1_num_rounds {
            self.phase1_compute_message(previous_claim)?
        } else if self.round < self.params.phase1_num_rounds + self.params.phase2_num_rounds {
            self.phase2_compute_message(previous_claim)?
        } else {
            self.phase3_compute_message(previous_claim)?
        })
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
                    "RAM read-write bind round {} is outside {} rounds",
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

    fn phase1_compute_message(&self, previous_claim: F) -> Result<UnivariatePoly<F>, BackendError> {
        let gruen_eq = self
            .gruen_eq
            .as_ref()
            .ok_or_else(|| BackendError::InvalidRequest {
                backend: "cpu",
                task: "RAM read-write phase1",
                reason: "missing phase1 equality state".to_owned(),
            })?;
        let sparse_matrix = &self.sparse_matrix_phase1;
        let e_in = gruen_eq.e_in_current();
        let e_in_len = e_in.len();
        let num_x_in_bits = e_in_len.max(1).ilog2() as usize;
        let x_bitmask = (1usize << num_x_in_bits) - 1;
        let inc = self.inc.evaluations();

        let quadratic_coeffs = sparse_matrix
            .entries
            .par_chunk_by(|a, b| ((a.row / 2) >> num_x_in_bits) == ((b.row / 2) >> num_x_in_bits))
            .map(|entries| {
                let x_out = (entries[0].row / 2) >> num_x_in_bits;
                let e_out_eval = gruen_eq.e_out_current()[x_out];
                let outer_sum = entries
                    .par_chunk_by(|a, b| a.row / 2 == b.row / 2)
                    .map(|entries| {
                        let odd_row_start_index =
                            entries.partition_point(|entry| entry.row.is_multiple_of(2));
                        let (even_row, odd_row) = entries.split_at(odd_row_start_index);
                        let j_prime = 2 * (entries[0].row / 2);
                        let e_in_eval = if e_in_len <= 1 {
                            F::one()
                        } else {
                            let x_in = (j_prime / 2) & x_bitmask;
                            e_in[x_in]
                        };
                        let inc_0 = inc[j_prime];
                        let inc_1 = inc[j_prime + 1];
                        let inner = sparse_matrix.prover_message_contribution(
                            even_row,
                            odd_row,
                            [inc_0, inc_1 - inc_0],
                            self.params.gamma,
                        );
                        [e_in_eval * inner[0], e_in_eval * inner[1]]
                    })
                    .reduce(|| [F::zero(); DEGREE_BOUND - 1], sum_arrays::<F, 2>);
                [e_out_eval * outer_sum[0], e_out_eval * outer_sum[1]]
            })
            .reduce(|| [F::zero(); DEGREE_BOUND - 1], sum_arrays::<F, 2>);

        Ok(gruen_eq.gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim))
    }

    fn phase2_compute_message(&self, previous_claim: F) -> Result<UnivariatePoly<F>, BackendError> {
        let merged_eq = self
            .merged_eq
            .as_ref()
            .ok_or_else(|| BackendError::InvalidRequest {
                backend: "cpu",
                task: "RAM read-write phase2",
                reason: "missing phase2 equality state".to_owned(),
            })?;
        let sparse_matrix = &self.sparse_matrix_phase2;
        let inc = self.inc.evaluations();
        let eq = merged_eq.evaluations();

        let evals = sparse_matrix
            .entries
            .par_chunk_by(|x, y| x.column() / 2 == y.column() / 2)
            .map(|entries| {
                let odd_col_start_index =
                    entries.partition_point(|entry| entry.column().is_multiple_of(2));
                let (even_col, odd_col) = entries.split_at(odd_col_start_index);
                let even_col_idx = 2 * (entries[0].column() / 2);
                let odd_col_idx = even_col_idx + 1;
                ReadWriteMatrixAddressMajor::prover_message_contribution(
                    even_col,
                    odd_col,
                    sparse_matrix.val_init[even_col_idx],
                    sparse_matrix.val_init[odd_col_idx],
                    inc,
                    eq,
                    self.params.gamma,
                )
            })
            .reduce(|| [F::zero(); DEGREE_BOUND - 1], sum_arrays::<F, 2>);

        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    fn phase3_compute_message(&self, previous_claim: F) -> Result<UnivariatePoly<F>, BackendError> {
        let merged_eq = self
            .merged_eq
            .as_ref()
            .ok_or_else(|| BackendError::InvalidRequest {
                backend: "cpu",
                task: "RAM read-write phase3",
                reason: "missing phase3 equality state".to_owned(),
            })?;
        let ra = self
            .ra
            .as_ref()
            .ok_or_else(|| BackendError::InvalidRequest {
                backend: "cpu",
                task: "RAM read-write phase3",
                reason: "missing phase3 ra state".to_owned(),
            })?;
        let val = self
            .val
            .as_ref()
            .ok_or_else(|| BackendError::InvalidRequest {
                backend: "cpu",
                task: "RAM read-write phase3",
                reason: "missing phase3 val state".to_owned(),
            })?;

        if self.inc.len() > 1 {
            let k_prime = self.params.address_count() >> self.params.phase2_num_rounds;
            let t_prime = self.inc.len();
            let inc = &self.inc;
            let evals = (0..inc.len() / 2)
                .into_par_iter()
                .map(|j| {
                    let inc_evals =
                        sumcheck_evals_array::<F, DEGREE_BOUND>(inc, j, BindingOrder::LowToHigh);
                    let eq_evals = sumcheck_evals_array::<F, DEGREE_BOUND>(
                        merged_eq,
                        j,
                        BindingOrder::LowToHigh,
                    );
                    let inner = (0..k_prime)
                        .into_par_iter()
                        .map(|k| {
                            let base = k * t_prime / 2 + j;
                            let ra_evals = sumcheck_evals_array::<F, DEGREE_BOUND>(
                                ra,
                                base,
                                BindingOrder::LowToHigh,
                            );
                            let val_evals = sumcheck_evals_array::<F, DEGREE_BOUND>(
                                val,
                                base,
                                BindingOrder::LowToHigh,
                            );
                            std::array::from_fn(|index| {
                                ra_evals[index]
                                    * (val_evals[index]
                                        + self.params.gamma * (val_evals[index] + inc_evals[index]))
                            })
                        })
                        .reduce(|| [F::zero(); DEGREE_BOUND], sum_arrays::<F, DEGREE_BOUND>);
                    std::array::from_fn(|index| eq_evals[index] * inner[index])
                })
                .reduce(|| [F::zero(); DEGREE_BOUND], sum_arrays::<F, DEGREE_BOUND>);

            Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
        } else {
            let inc_eval = self.inc.evaluations()[0];
            let eq_eval = merged_eq.evaluations()[0];
            let evals = (0..ra.len() / 2)
                .into_par_iter()
                .map(|k| {
                    let ra_evals = sumcheck_evals_array::<F, 2>(ra, k, BindingOrder::LowToHigh);
                    let val_evals = sumcheck_evals_array::<F, 2>(val, k, BindingOrder::LowToHigh);
                    std::array::from_fn(|index| {
                        ra_evals[index]
                            * (val_evals[index] + self.params.gamma * (val_evals[index] + inc_eval))
                    })
                })
                .reduce(|| [F::zero(); 2], sum_arrays::<F, 2>);
            let evals = evals.map(|eval| eq_eval * eval);
            Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
        }
    }

    #[expect(clippy::expect_used)]
    fn phase1_bind(&mut self, challenge: F) {
        self.sparse_matrix_phase1.bind(challenge);
        self.gruen_eq
            .as_mut()
            .expect("phase1 equality state must exist")
            .bind(challenge);
        self.inc.bind_with_order(challenge, BindingOrder::LowToHigh);

        if self.round == self.params.phase1_num_rounds - 1 {
            let gruen_eq = self
                .gruen_eq
                .as_ref()
                .expect("phase1 equality state must exist");
            self.merged_eq = Some(gruen_eq.merge());
            let sparse_matrix = std::mem::take(&mut self.sparse_matrix_phase1);
            if self.params.phase2_num_rounds > 0 {
                self.sparse_matrix_phase2 =
                    ram_address_major_from_cycle_major(sparse_matrix, self.val_init.clone());
            } else {
                let t_prime = 1usize << self.params.phase3_cycle_rounds();
                let (ra, val) = materialize_cycle_major(
                    sparse_matrix,
                    self.params.address_count(),
                    t_prime,
                    &self.val_init,
                );
                self.ra = Some(ra);
                self.val = Some(val);
            }
        }
    }

    fn phase2_bind(&mut self, challenge: F) {
        self.sparse_matrix_phase2.bind(challenge);
        if self.round == self.params.phase1_num_rounds + self.params.phase2_num_rounds - 1 {
            let sparse_matrix = std::mem::take(&mut self.sparse_matrix_phase2);
            let (ra, val) = materialize_address_major(
                sparse_matrix,
                self.params.address_count() >> self.params.phase2_num_rounds,
                1usize << self.params.phase3_cycle_rounds(),
            );
            self.ra = Some(ra);
            self.val = Some(val);
        }
    }

    #[expect(clippy::expect_used)]
    fn phase3_bind(&mut self, challenge: F) {
        if self.inc.len() > 1 {
            self.inc.bind_with_order(challenge, BindingOrder::LowToHigh);
            self.merged_eq
                .as_mut()
                .expect("phase3 equality state must exist")
                .bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        self.ra
            .as_mut()
            .expect("phase3 ra state must exist")
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.val
            .as_mut()
            .expect("phase3 val state must exist")
            .bind_with_order(challenge, BindingOrder::LowToHigh);
    }

    pub fn output_claims(
        &self,
        backend: &'static str,
        task: &'static str,
    ) -> Result<[F; 3], BackendError> {
        if self.round != self.params.rounds() {
            return invalid(
                backend,
                task,
                format!(
                    "RAM read-write output claims requested after {} of {} rounds",
                    self.round,
                    self.params.rounds()
                ),
            );
        }
        let val = final_polynomial_value(backend, task, "RAM read-write val", self.val.as_ref())?;
        let ra = final_polynomial_value(backend, task, "RAM read-write ra", self.ra.as_ref())?;
        let inc = final_required_polynomial_value(backend, task, "RAM read-write inc", &self.inc)?;
        Ok([val, ra, inc])
    }
}

pub struct RamRafState<F: Field> {
    ra: Polynomial<F>,
    unmap: Polynomial<F>,
    params: RamReadWriteParams<F>,
    input_claim: F,
    round: usize,
}

impl<F: Field> RamRafState<F> {
    pub fn new(
        backend: &'static str,
        task: &'static str,
        request: &SumcheckRamRafStateRequest<F>,
    ) -> Result<Self, BackendError> {
        validate_stage2_tail_shape(
            backend,
            task,
            request.rows.len(),
            request.r_cycle.len(),
            request.log_t,
            request.log_t,
            request.log_t,
            request.log_k,
            request.phase1_num_rounds,
            request.phase2_num_rounds,
        )?;
        let address_count = 1usize << request.log_k;
        let mut ra_by_address = vec![F::zero(); address_count];
        let cycle_eq = jolt_poly::TensorEqTable::<F>::new(&request.r_cycle);
        for (cycle, row) in request.rows.iter().enumerate() {
            if let Some(address) = row.remapped_ram_address {
                ra_by_address[address] += cycle_eq.evaluate_index(cycle);
            }
        }
        let unmap = (0..address_count)
            .map(|address| F::from_u64(address as u64 * 8 + request.start_address))
            .collect::<Vec<_>>();
        Ok(Self {
            ra: Polynomial::new(ra_by_address),
            unmap: Polynomial::new(unmap),
            params: RamReadWriteParams {
                log_t: request.log_t,
                log_k: request.log_k,
                phase1_num_rounds: request.phase1_num_rounds,
                phase2_num_rounds: request.phase2_num_rounds,
                gamma: F::zero(),
            },
            input_claim: request.input_claim,
            round: 0,
        })
    }

    pub fn input_claim(&self) -> F {
        self.input_claim
    }

    pub fn num_rounds(&self) -> usize {
        self.params.log_t + self.params.log_k - self.params.phase1_num_rounds
    }

    pub fn round_offset(&self) -> usize {
        self.params.phase1_num_rounds
    }

    pub fn evaluate_round(&self, previous_claim: F) -> Result<UnivariatePoly<F>, BackendError> {
        if self.is_internal_cycle_gap_round() {
            return Ok(half_claim_poly(previous_claim));
        }
        let evals = (0..self.ra.len() / 2)
            .into_par_iter()
            .map(|index| {
                let ra_evals =
                    sumcheck_evals_array::<F, 2>(&self.ra, index, BindingOrder::LowToHigh);
                let unmap_evals =
                    sumcheck_evals_array::<F, 2>(&self.unmap, index, BindingOrder::LowToHigh);
                [ra_evals[0] * unmap_evals[0], ra_evals[1] * unmap_evals[1]]
            })
            .reduce(|| [F::zero(); 2], sum_arrays::<F, 2>);
        let gap = if self.round < self.params.phase2_num_rounds {
            self.params.phase3_cycle_rounds()
        } else {
            0
        };
        let evals = if gap > 0 {
            evals.map(|eval| eval.mul_pow_2(gap))
        } else {
            evals
        };
        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    pub fn bind(&mut self, challenge: F) {
        if !self.is_internal_cycle_gap_round() {
            self.ra.bind_with_order(challenge, BindingOrder::LowToHigh);
            self.unmap
                .bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        self.round += 1;
    }

    fn is_internal_cycle_gap_round(&self) -> bool {
        let start = self.params.phase2_num_rounds;
        let end = start + self.params.phase3_cycle_rounds();
        self.round >= start && self.round < end
    }

    pub fn output_claim(
        &self,
        backend: &'static str,
        task: &'static str,
    ) -> Result<F, BackendError> {
        if self.round != self.num_rounds() {
            return invalid(
                backend,
                task,
                format!(
                    "RAM RAF output claim requested after {} of {} rounds",
                    self.round,
                    self.num_rounds()
                ),
            );
        }
        final_required_polynomial_value(backend, task, "RAM RAF ra", &self.ra)
    }
}

pub struct RamOutputCheckState<F: Field> {
    eq: GruenSplitEqPolynomial<F>,
    io_mask: Polynomial<F>,
    val_final: Polynomial<F>,
    val_io: Polynomial<F>,
    num_zero_address_vars: usize,
    params: RamReadWriteParams<F>,
    round: usize,
}

impl<F: Field> RamOutputCheckState<F> {
    pub fn new(
        backend: &'static str,
        task: &'static str,
        request: &SumcheckRamOutputCheckStateRequest<F>,
    ) -> Result<Self, BackendError> {
        validate_stage2_tail_shape(
            backend,
            task,
            request.final_ram_state.len(),
            request.r_address.len(),
            request.log_k,
            request.log_k,
            request.log_t,
            request.log_k,
            request.phase1_num_rounds,
            request.phase2_num_rounds,
        )?;
        let address_count = 1usize << request.log_k;
        if request.public_io_state.len() != address_count {
            return invalid(
                backend,
                task,
                format!(
                    "RAM output-check public IO state has {} addresses, expected {address_count}",
                    request.public_io_state.len()
                ),
            );
        }
        if request.io_start > request.io_end || request.io_end > address_count {
            return invalid(
                backend,
                task,
                format!(
                    "RAM output-check IO range {}..{} is outside {address_count} addresses",
                    request.io_start, request.io_end
                ),
            );
        }

        let val_final = request
            .final_ram_state
            .iter()
            .map(|&value| F::from_u64(value))
            .collect::<Vec<_>>();
        let mut val_io = vec![F::zero(); address_count];
        val_io[request.io_start..request.io_end]
            .iter_mut()
            .zip(&request.public_io_state[request.io_start..request.io_end])
            .for_each(|(dest, &value)| *dest = F::from_u64(value));
        let mut io_mask = vec![F::zero(); address_count];
        io_mask[request.io_start..request.io_end].fill(F::one());
        let eq = GruenSplitEqPolynomial::new(&request.r_address, BindingOrder::LowToHigh);
        let num_zero_address_vars = (request
            .io_start
            .trailing_zeros()
            .min(request.io_end.trailing_zeros()) as usize)
            .min(request.log_k);

        Ok(Self {
            eq,
            io_mask: Polynomial::new(io_mask),
            val_final: Polynomial::new(val_final),
            val_io: Polynomial::new(val_io),
            num_zero_address_vars,
            params: RamReadWriteParams {
                log_t: request.log_t,
                log_k: request.log_k,
                phase1_num_rounds: request.phase1_num_rounds,
                phase2_num_rounds: request.phase2_num_rounds,
                gamma: F::zero(),
            },
            round: 0,
        })
    }

    pub fn input_claim(&self) -> F {
        F::zero()
    }

    pub fn num_rounds(&self) -> usize {
        self.params.log_t + self.params.log_k - self.params.phase1_num_rounds
    }

    pub fn round_offset(&self) -> usize {
        self.params.phase1_num_rounds
    }

    pub fn evaluate_round(&self, previous_claim: F) -> UnivariatePoly<F> {
        if self.is_internal_cycle_gap_round() || self.round < self.num_zero_address_vars {
            return half_claim_poly(previous_claim);
        }
        let [q_constant, q_quadratic] = self.eq.par_fold_out_in(
            || [F::zero(); 2],
            |inner, row, _x_in, e_in| {
                let (io0, io1) = self
                    .io_mask
                    .sumcheck_eval_pair(row, BindingOrder::LowToHigh);
                let (vf0, vf1) = self
                    .val_final
                    .sumcheck_eval_pair(row, BindingOrder::LowToHigh);
                let (vio0, vio1) = self.val_io.sumcheck_eval_pair(row, BindingOrder::LowToHigh);
                let v0 = vf0 - vio0;
                let v1 = vf1 - vio1;
                inner[0] += e_in * io0 * v0;
                inner[1] += e_in * (io1 - io0) * (v1 - v0);
            },
            |_x_out, e_out, inner| [e_out * inner[0], e_out * inner[1]],
            sum_arrays::<F, 2>,
        );
        let gap = if self.round < self.params.phase2_num_rounds {
            self.params.phase3_cycle_rounds()
        } else {
            0
        };
        if gap > 0 {
            self.eq.gruen_poly_deg_3(
                q_constant.mul_pow_2(gap),
                q_quadratic.mul_pow_2(gap),
                previous_claim,
            )
        } else {
            self.eq
                .gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
        }
    }

    pub fn bind(&mut self, challenge: F) {
        if !self.is_internal_cycle_gap_round() {
            self.eq.bind(challenge);
            self.io_mask
                .bind_with_order(challenge, BindingOrder::LowToHigh);
            self.val_final
                .bind_with_order(challenge, BindingOrder::LowToHigh);
            self.val_io
                .bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        self.round += 1;
    }

    fn is_internal_cycle_gap_round(&self) -> bool {
        let start = self.params.phase2_num_rounds;
        let end = start + self.params.phase3_cycle_rounds();
        self.round >= start && self.round < end
    }

    pub fn output_claim(
        &self,
        backend: &'static str,
        task: &'static str,
    ) -> Result<F, BackendError> {
        if self.round != self.num_rounds() {
            return invalid(
                backend,
                task,
                format!(
                    "RAM output-check output claim requested after {} of {} rounds",
                    self.round,
                    self.num_rounds()
                ),
            );
        }
        final_required_polynomial_value(
            backend,
            task,
            "RAM output-check final value",
            &self.val_final,
        )
    }
}

pub struct RamRaClaimReductionState<F: Field> {
    phase: RamRaClaimReductionPhase<F>,
    round: usize,
    log_t: usize,
}

#[expect(
    clippy::large_enum_variant,
    reason = "The phase enum is state-machine local and avoids heap allocation in the hot clear path."
)]
enum RamRaClaimReductionPhase<F: Field> {
    Phase1(RamRaClaimReductionPhase1<F>),
    Phase2(RamRaClaimReductionPhase2<F>),
}

struct RamRaClaimReductionPhase1<F: Field> {
    p_raf: Polynomial<F>,
    p_read_write: Polynomial<F>,
    p_val_check: Polynomial<F>,
    q_raf: Polynomial<F>,
    q_read_write: Polynomial<F>,
    q_val_check: Polynomial<F>,
    addresses: Vec<Option<usize>>,
    address_eq: Vec<F>,
    r_cycle_raf_hi: Vec<F>,
    r_cycle_raf_lo: Vec<F>,
    r_cycle_read_write_hi: Vec<F>,
    r_cycle_read_write_lo: Vec<F>,
    r_cycle_val_check_hi: Vec<F>,
    r_cycle_val_check_lo: Vec<F>,
    prefix_challenges: Vec<F>,
    prefix_vars: usize,
    suffix_vars: usize,
    gamma: F,
    gamma2: F,
}

struct RamRaClaimReductionPhase2<F: Field> {
    ram_ra: Polynomial<F>,
    eq_raf_hi: Polynomial<F>,
    eq_read_write_hi: Polynomial<F>,
    eq_val_check_hi: Polynomial<F>,
    coeff_raf: F,
    coeff_read_write: F,
    coeff_val_check: F,
}

impl<F: Field> RamRaClaimReductionState<F> {
    pub fn new(
        backend: &'static str,
        task: &'static str,
        request: &SumcheckRamRaClaimReductionStateRequest<F>,
    ) -> Result<Self, BackendError> {
        validate_ram_ra_claim_reduction_request(backend, task, request)?;
        let log_t = request.log_t;
        let prefix_vars = log_t / 2;
        let suffix_vars = log_t - prefix_vars;
        let address_count = 1usize << request.log_k;
        let addresses = request
            .rows
            .iter()
            .map(|row| {
                if let Some(address) = row.remapped_ram_address {
                    if address >= address_count {
                        return invalid(
                            backend,
                            task,
                            format!(
                                "RAM RA claim-reduction address {address} is outside {address_count} addresses"
                            ),
                        );
                    }
                }
                Ok(row.remapped_ram_address)
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
        let address_eq = jolt_poly::EqPolynomial::new(request.r_address.clone()).evaluations();
        let gamma2 = request.gamma * request.gamma;

        let phase = if prefix_vars == 0 {
            let (r_cycle_raf_hi, r_cycle_raf_lo) = request.r_cycle_raf.split_at(suffix_vars);
            let (r_cycle_read_write_hi, r_cycle_read_write_lo) =
                request.r_cycle_read_write.split_at(suffix_vars);
            let (r_cycle_val_check_hi, r_cycle_val_check_lo) =
                request.r_cycle_val_check.split_at(suffix_vars);
            RamRaClaimReductionPhase::Phase2(RamRaClaimReductionPhase2::new(
                &addresses,
                &address_eq,
                &[],
                r_cycle_raf_hi,
                r_cycle_raf_lo,
                r_cycle_read_write_hi,
                r_cycle_read_write_lo,
                r_cycle_val_check_hi,
                r_cycle_val_check_lo,
                request.gamma,
                prefix_vars,
                suffix_vars,
            ))
        } else {
            RamRaClaimReductionPhase::Phase1(RamRaClaimReductionPhase1::new(
                addresses,
                address_eq,
                request,
                prefix_vars,
                suffix_vars,
                gamma2,
            ))
        };

        Ok(Self {
            phase,
            round: 0,
            log_t,
        })
    }

    pub fn evaluate_round(
        &self,
        backend: &'static str,
        task: &'static str,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError> {
        if self.round >= self.log_t {
            return invalid(
                backend,
                task,
                format!(
                    "RAM RA claim-reduction round {} is outside {} rounds",
                    self.round, self.log_t
                ),
            );
        }
        Ok(match &self.phase {
            RamRaClaimReductionPhase::Phase1(state) => state.compute_message(previous_claim),
            RamRaClaimReductionPhase::Phase2(state) => state.compute_message(previous_claim),
        })
    }

    pub fn bind(
        &mut self,
        backend: &'static str,
        task: &'static str,
        challenge: F,
    ) -> Result<(), BackendError> {
        if self.round >= self.log_t {
            return invalid(
                backend,
                task,
                format!(
                    "RAM RA claim-reduction bind round {} is outside {} rounds",
                    self.round, self.log_t
                ),
            );
        }
        match &mut self.phase {
            RamRaClaimReductionPhase::Phase1(state) => {
                state.bind(challenge);
                if state.p_raf.len() == 1 {
                    self.phase = RamRaClaimReductionPhase::Phase2(state.to_phase2());
                }
            }
            RamRaClaimReductionPhase::Phase2(state) => state.bind(challenge),
        }
        self.round += 1;
        Ok(())
    }

    pub fn output_claims(&self) -> Result<SumcheckRamRaClaimReductionOutput<F>, BackendError> {
        let RamRaClaimReductionPhase::Phase2(state) = &self.phase else {
            return invalid(
                "cpu",
                "RAM RA claim-reduction output claims",
                "missing reduced RAM RA state",
            );
        };
        let Some(&ram_ra) = state.ram_ra.evaluations().first() else {
            return invalid(
                "cpu",
                "RAM RA claim-reduction output claims",
                "empty reduced RAM RA state",
            );
        };
        Ok(SumcheckRamRaClaimReductionOutput { ram_ra })
    }
}

impl<F: Field> RamRaClaimReductionPhase1<F> {
    fn new(
        addresses: Vec<Option<usize>>,
        address_eq: Vec<F>,
        request: &SumcheckRamRaClaimReductionStateRequest<F>,
        prefix_vars: usize,
        suffix_vars: usize,
        gamma2: F,
    ) -> Self {
        let (r_cycle_raf_hi, r_cycle_raf_lo) = request.r_cycle_raf.split_at(suffix_vars);
        let (r_cycle_read_write_hi, r_cycle_read_write_lo) =
            request.r_cycle_read_write.split_at(suffix_vars);
        let (r_cycle_val_check_hi, r_cycle_val_check_lo) =
            request.r_cycle_val_check.split_at(suffix_vars);

        let p_raf =
            Polynomial::new(jolt_poly::EqPolynomial::new(r_cycle_raf_lo.to_vec()).evaluations());
        let p_read_write = Polynomial::new(
            jolt_poly::EqPolynomial::new(r_cycle_read_write_lo.to_vec()).evaluations(),
        );
        let p_val_check = Polynomial::new(
            jolt_poly::EqPolynomial::new(r_cycle_val_check_lo.to_vec()).evaluations(),
        );

        let eq_raf_hi = jolt_poly::EqPolynomial::new(r_cycle_raf_hi.to_vec()).evaluations();
        let eq_read_write_hi =
            jolt_poly::EqPolynomial::new(r_cycle_read_write_hi.to_vec()).evaluations();
        let eq_val_check_hi =
            jolt_poly::EqPolynomial::new(r_cycle_val_check_hi.to_vec()).evaluations();

        let prefix_size = 1usize << prefix_vars;
        let (q_raf, q_read_write, q_val_check) = compute_ram_ra_q_arrays(
            &addresses,
            &address_eq,
            &eq_raf_hi,
            &eq_read_write_hi,
            &eq_val_check_hi,
            prefix_size,
            prefix_vars,
        );

        Self {
            p_raf,
            p_read_write,
            p_val_check,
            q_raf: Polynomial::new(q_raf),
            q_read_write: Polynomial::new(q_read_write),
            q_val_check: Polynomial::new(q_val_check),
            addresses,
            address_eq,
            r_cycle_raf_hi: r_cycle_raf_hi.to_vec(),
            r_cycle_raf_lo: r_cycle_raf_lo.to_vec(),
            r_cycle_read_write_hi: r_cycle_read_write_hi.to_vec(),
            r_cycle_read_write_lo: r_cycle_read_write_lo.to_vec(),
            r_cycle_val_check_hi: r_cycle_val_check_hi.to_vec(),
            r_cycle_val_check_lo: r_cycle_val_check_lo.to_vec(),
            prefix_challenges: Vec::with_capacity(prefix_vars),
            prefix_vars,
            suffix_vars,
            gamma: request.gamma,
            gamma2,
        }
    }

    fn compute_message(&self, previous_claim: F) -> UnivariatePoly<F> {
        let terms = self.p_raf.len() / 2;
        let evals = (0..terms)
            .into_par_iter()
            .map(|index| {
                let p_raf =
                    sumcheck_evals_array::<F, 2>(&self.p_raf, index, BindingOrder::LowToHigh);
                let q_raf =
                    sumcheck_evals_array::<F, 2>(&self.q_raf, index, BindingOrder::LowToHigh);
                let p_read_write = sumcheck_evals_array::<F, 2>(
                    &self.p_read_write,
                    index,
                    BindingOrder::LowToHigh,
                );
                let q_read_write = sumcheck_evals_array::<F, 2>(
                    &self.q_read_write,
                    index,
                    BindingOrder::LowToHigh,
                );
                let p_val_check =
                    sumcheck_evals_array::<F, 2>(&self.p_val_check, index, BindingOrder::LowToHigh);
                let q_val_check =
                    sumcheck_evals_array::<F, 2>(&self.q_val_check, index, BindingOrder::LowToHigh);
                [
                    p_raf[0] * q_raf[0]
                        + self.gamma * p_read_write[0] * q_read_write[0]
                        + self.gamma2 * p_val_check[0] * q_val_check[0],
                    p_raf[1] * q_raf[1]
                        + self.gamma * p_read_write[1] * q_read_write[1]
                        + self.gamma2 * p_val_check[1] * q_val_check[1],
                ]
            })
            .reduce(|| [F::zero(); 2], sum_arrays::<F, 2>);
        UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, challenge: F) {
        self.prefix_challenges.push(challenge);
        self.p_raf
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.p_read_write
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.p_val_check
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.q_raf
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.q_read_write
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.q_val_check
            .bind_with_order(challenge, BindingOrder::LowToHigh);
    }

    fn to_phase2(&self) -> RamRaClaimReductionPhase2<F> {
        RamRaClaimReductionPhase2::new(
            &self.addresses,
            &self.address_eq,
            &self.prefix_challenges,
            &self.r_cycle_raf_hi,
            &self.r_cycle_raf_lo,
            &self.r_cycle_read_write_hi,
            &self.r_cycle_read_write_lo,
            &self.r_cycle_val_check_hi,
            &self.r_cycle_val_check_lo,
            self.gamma,
            self.prefix_vars,
            self.suffix_vars,
        )
    }
}

impl<F: Field> RamRaClaimReductionPhase2<F> {
    #[expect(
        clippy::too_many_arguments,
        reason = "Phase 2 initialization takes the verifier-derived points separately to preserve auditability."
    )]
    fn new(
        addresses: &[Option<usize>],
        address_eq: &[F],
        prefix_challenges: &[F],
        r_cycle_raf_hi: &[F],
        r_cycle_raf_lo: &[F],
        r_cycle_read_write_hi: &[F],
        r_cycle_read_write_lo: &[F],
        r_cycle_val_check_hi: &[F],
        r_cycle_val_check_lo: &[F],
        gamma: F,
        prefix_vars: usize,
        suffix_vars: usize,
    ) -> Self {
        let r_cycle_prefix = prefix_challenges.iter().rev().copied().collect::<Vec<_>>();
        let eq_prefix = jolt_poly::EqPolynomial::new(r_cycle_prefix.clone()).evaluations();
        let ram_ra =
            compute_ram_ra_h_prime(addresses, address_eq, &eq_prefix, prefix_vars, suffix_vars);

        let scale_raf =
            jolt_poly::try_eq_mle(r_cycle_raf_lo, &r_cycle_prefix).unwrap_or_else(|_| F::zero());
        let scale_read_write = jolt_poly::try_eq_mle(r_cycle_read_write_lo, &r_cycle_prefix)
            .unwrap_or_else(|_| F::zero());
        let scale_val_check = jolt_poly::try_eq_mle(r_cycle_val_check_lo, &r_cycle_prefix)
            .unwrap_or_else(|_| F::zero());
        let gamma2 = gamma * gamma;

        Self {
            ram_ra: Polynomial::new(ram_ra),
            eq_raf_hi: Polynomial::new(
                jolt_poly::EqPolynomial::new(r_cycle_raf_hi.to_vec()).evaluations(),
            ),
            eq_read_write_hi: Polynomial::new(
                jolt_poly::EqPolynomial::new(r_cycle_read_write_hi.to_vec()).evaluations(),
            ),
            eq_val_check_hi: Polynomial::new(
                jolt_poly::EqPolynomial::new(r_cycle_val_check_hi.to_vec()).evaluations(),
            ),
            coeff_raf: scale_raf,
            coeff_read_write: gamma * scale_read_write,
            coeff_val_check: gamma2 * scale_val_check,
        }
    }

    fn compute_message(&self, previous_claim: F) -> UnivariatePoly<F> {
        let terms = self.ram_ra.len() / 2;
        let evals = (0..terms)
            .into_par_iter()
            .map(|index| {
                let ram_ra =
                    sumcheck_evals_array::<F, 2>(&self.ram_ra, index, BindingOrder::LowToHigh);
                let eq_raf =
                    sumcheck_evals_array::<F, 2>(&self.eq_raf_hi, index, BindingOrder::LowToHigh);
                let eq_read_write = sumcheck_evals_array::<F, 2>(
                    &self.eq_read_write_hi,
                    index,
                    BindingOrder::LowToHigh,
                );
                let eq_val_check = sumcheck_evals_array::<F, 2>(
                    &self.eq_val_check_hi,
                    index,
                    BindingOrder::LowToHigh,
                );
                [
                    ram_ra[0]
                        * (self.coeff_raf * eq_raf[0]
                            + self.coeff_read_write * eq_read_write[0]
                            + self.coeff_val_check * eq_val_check[0]),
                    ram_ra[1]
                        * (self.coeff_raf * eq_raf[1]
                            + self.coeff_read_write * eq_read_write[1]
                            + self.coeff_val_check * eq_val_check[1]),
                ]
            })
            .reduce(|| [F::zero(); 2], sum_arrays::<F, 2>);
        UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, challenge: F) {
        self.ram_ra
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.eq_raf_hi
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.eq_read_write_hi
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.eq_val_check_hi
            .bind_with_order(challenge, BindingOrder::LowToHigh);
    }
}

pub struct RamValCheckState<F: Field> {
    inc: Polynomial<F>,
    wa: Polynomial<F>,
    lt: Polynomial<F>,
    input_claim: F,
    log_t: usize,
    gamma: F,
    round: usize,
}

impl<F: Field> RamValCheckState<F> {
    pub fn new(
        backend: &'static str,
        task: &'static str,
        request: &SumcheckRamValCheckStateRequest<F>,
    ) -> Result<Self, BackendError> {
        let cycles = 1usize.checked_shl(request.log_t as u32).ok_or_else(|| {
            BackendError::InvalidRequest {
                backend,
                task,
                reason: format!(
                    "RAM value-check requires 2^{} rows, which does not fit in usize",
                    request.log_t
                ),
            }
        })?;
        if request.rows.len() != cycles {
            return invalid(
                backend,
                task,
                format!(
                    "RAM value-check has {} rows, expected {cycles}",
                    request.rows.len()
                ),
            );
        }
        if request.r_cycle.len() != request.log_t {
            return invalid(
                backend,
                task,
                format!(
                    "RAM value-check cycle point has {} variables, expected {}",
                    request.r_cycle.len(),
                    request.log_t
                ),
            );
        }
        let address_count = 1usize
            .checked_shl(request.r_address.len() as u32)
            .ok_or_else(|| BackendError::InvalidRequest {
                backend,
                task,
                reason: format!(
                    "RAM value-check address point has {} variables, which overflows usize",
                    request.r_address.len()
                ),
            })?;
        let eq_address = jolt_poly::EqPolynomial::new(request.r_address.clone()).evaluations();
        let inc = request
            .rows
            .iter()
            .map(|row| F::from_i128(row.ram_increment))
            .collect::<Vec<_>>();
        let wa = request
            .rows
            .iter()
            .map(|row| {
                let Some(address) = row.remapped_ram_address else {
                    return Ok(F::zero());
                };
                if address >= address_count {
                    return invalid(
                        backend,
                        task,
                        format!(
                            "RAM value-check address {address} is outside {address_count} addresses"
                        ),
                    );
                }
                Ok(eq_address[address])
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
        let lt = jolt_poly::LtPolynomial::evaluations(&request.r_cycle);

        Ok(Self {
            inc: Polynomial::new(inc),
            wa: Polynomial::new(wa),
            lt: Polynomial::new(lt),
            input_claim: request.input_claim,
            log_t: request.log_t,
            gamma: request.gamma,
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
                    "RAM value-check round {} is outside {} rounds",
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
                std::array::from_fn(|i| inc[i] * wa[i] * (lt[i] + self.gamma))
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
                    "RAM value-check bind round {} is outside {} rounds",
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

    pub fn output_claims(&self) -> Result<SumcheckRamValCheckOutput<F>, BackendError> {
        if self.inc.len() != 1 {
            return invalid(
                "cpu",
                "RAM value-check output claims",
                "missing RAM increment state",
            );
        }
        if self.wa.len() != 1 {
            return invalid(
                "cpu",
                "RAM value-check output claims",
                "missing RAM RA state",
            );
        }
        let ram_inc = self.inc.evaluations()[0];
        let ram_ra = self.wa.evaluations()[0];
        Ok(SumcheckRamValCheckOutput { ram_ra, ram_inc })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamCycleMajorEntry<F: Field> {
    row: usize,
    col: usize,
    prev_val: u64,
    next_val: u64,
    val_coeff: F,
    ra_coeff: F,
}

impl<F: Field> Default for RamCycleMajorEntry<F> {
    fn default() -> Self {
        Self {
            row: 0,
            col: 0,
            prev_val: 0,
            next_val: 0,
            val_coeff: F::zero(),
            ra_coeff: F::zero(),
        }
    }
}

impl<F: Field> RamCycleMajorEntry<F> {
    fn from_row(row: usize, entry: SumcheckRamReadWriteRow) -> Option<Self> {
        let col = entry.remapped_ram_address?;
        Some(Self {
            row,
            col,
            prev_val: entry.ram_read_value,
            next_val: entry.ram_write_value,
            val_coeff: F::from_u64(entry.ram_read_value),
            ra_coeff: F::one(),
        })
    }
}

impl<F: Field> CycleMajorMatrixEntry<F> for RamCycleMajorEntry<F> {
    fn row(&self) -> usize {
        self.row
    }

    fn column(&self) -> usize {
        self.col
    }

    fn bind_entries(
        even: Option<&Self>,
        odd: Option<&Self>,
        r: F,
        _: Option<&OneHotCoeffTable<F>>,
        _: Option<&OneHotCoeffTable<F>>,
    ) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => Self {
                row: even.row / 2,
                col: even.col,
                ra_coeff: even.ra_coeff + r.mul_0_optimized(odd.ra_coeff - even.ra_coeff),
                val_coeff: even.val_coeff + r.mul_0_optimized(odd.val_coeff - even.val_coeff),
                prev_val: even.prev_val,
                next_val: odd.next_val,
            },
            (Some(even), None) => {
                let odd_val_coeff = F::from_u64(even.next_val);
                Self {
                    row: even.row / 2,
                    col: even.col,
                    ra_coeff: (F::one() - r).mul_1_optimized(even.ra_coeff),
                    val_coeff: even.val_coeff + r.mul_0_optimized(odd_val_coeff - even.val_coeff),
                    prev_val: even.prev_val,
                    next_val: even.next_val,
                }
            }
            (None, Some(odd)) => {
                let even_val_coeff = F::from_u64(odd.prev_val);
                Self {
                    row: odd.row / 2,
                    col: odd.col,
                    ra_coeff: r.mul_1_optimized(odd.ra_coeff),
                    val_coeff: even_val_coeff + r.mul_0_optimized(odd.val_coeff - even_val_coeff),
                    prev_val: odd.prev_val,
                    next_val: odd.next_val,
                }
            }
            (None, None) => unreachable!("RAM cycle-major bind requires at least one entry"),
        }
    }
}

impl<F> CycleMajorMessageEntry<F> for RamCycleMajorEntry<F>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    fn accumulate_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        inc_evals: [F; 2],
        gamma: F,
        accumulators: &mut [<F as WithAccumulator>::Accumulator; 2],
        _: Option<&OneHotCoeffTable<F>>,
        _: Option<&OneHotCoeffTable<F>>,
    ) {
        let (ra_evals, val_evals) = match (even, odd) {
            (Some(even), Some(odd)) => (
                [even.ra_coeff, odd.ra_coeff - even.ra_coeff],
                [even.val_coeff, odd.val_coeff - even.val_coeff],
            ),
            (Some(even), None) => {
                let odd_val_coeff = F::from_u64(even.next_val);
                (
                    [even.ra_coeff, -even.ra_coeff],
                    [even.val_coeff, odd_val_coeff - even.val_coeff],
                )
            }
            (None, Some(odd)) => {
                let even_val_coeff = F::from_u64(odd.prev_val);
                (
                    [F::zero(), odd.ra_coeff],
                    [even_val_coeff, odd.val_coeff - even_val_coeff],
                )
            }
            (None, None) => unreachable!("RAM cycle-major message requires at least one entry"),
        };
        for index in 0..2 {
            accumulators[index].fmadd(
                ra_evals[index],
                val_evals[index] + gamma * (inc_evals[index] + val_evals[index]),
            );
        }
    }
}

impl<F: Field> CycleMajorToAddressMajor<F> for RamCycleMajorEntry<F> {
    type AddressMajor = RamAddressMajorEntry<F>;

    fn to_address_major(
        self,
        _: Option<&OneHotCoeffTable<F>>,
        _: Option<&OneHotCoeffTable<F>>,
    ) -> Self::AddressMajor {
        RamAddressMajorEntry {
            row: self.row,
            col: self.col,
            prev_val: F::from_u64(self.prev_val),
            next_val: F::from_u64(self.next_val),
            val_coeff: self.val_coeff,
            ra_coeff: self.ra_coeff,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamAddressMajorEntry<F: Field> {
    row: usize,
    col: usize,
    prev_val: F,
    next_val: F,
    val_coeff: F,
    ra_coeff: F,
}

impl<F: Field> Default for RamAddressMajorEntry<F> {
    fn default() -> Self {
        Self {
            row: 0,
            col: 0,
            prev_val: F::zero(),
            next_val: F::zero(),
            val_coeff: F::zero(),
            ra_coeff: F::zero(),
        }
    }
}

impl<F: Field> AddressMajorMatrixEntry<F> for RamAddressMajorEntry<F> {
    fn row(&self) -> usize {
        self.row
    }

    fn column(&self) -> usize {
        self.col
    }
}

impl<F: Field> AddressMajorBindableEntry<F> for RamAddressMajorEntry<F> {
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
        r: F,
    ) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => Self {
                row: even.row,
                col: even.col / 2,
                ra_coeff: even.ra_coeff + r.mul_0_optimized(odd.ra_coeff - even.ra_coeff),
                val_coeff: even.val_coeff + r.mul_0_optimized(odd.val_coeff - even.val_coeff),
                prev_val: even.prev_val + r.mul_0_optimized(odd.prev_val - even.prev_val),
                next_val: even.next_val + r.mul_0_optimized(odd.next_val - even.next_val),
            },
            (Some(even), None) => Self {
                row: even.row,
                col: even.col / 2,
                ra_coeff: (F::one() - r).mul_1_optimized(even.ra_coeff),
                val_coeff: even.val_coeff + r.mul_0_optimized(odd_checkpoint - even.val_coeff),
                prev_val: even.prev_val + r.mul_0_optimized(odd_checkpoint - even.prev_val),
                next_val: even.next_val + r.mul_0_optimized(odd_checkpoint - even.next_val),
            },
            (None, Some(odd)) => Self {
                row: odd.row,
                col: odd.col / 2,
                ra_coeff: r.mul_1_optimized(odd.ra_coeff),
                val_coeff: even_checkpoint + r.mul_0_optimized(odd.val_coeff - even_checkpoint),
                prev_val: even_checkpoint + r.mul_0_optimized(odd.prev_val - even_checkpoint),
                next_val: even_checkpoint + r.mul_0_optimized(odd.next_val - even_checkpoint),
            },
            (None, None) => unreachable!("RAM address-major bind requires at least one entry"),
        }
    }
}

impl<F> AddressMajorMessageEntry<F> for RamAddressMajorEntry<F>
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
        let (ra_evals, val_evals) = match (even, odd) {
            (Some(even), Some(odd)) => (
                [even.ra_coeff, odd.ra_coeff + odd.ra_coeff - even.ra_coeff],
                [
                    even.val_coeff,
                    odd.val_coeff + odd.val_coeff - even.val_coeff,
                ],
            ),
            (Some(even), None) => (
                [even.ra_coeff, -even.ra_coeff],
                [
                    even.val_coeff,
                    inputs.odd_checkpoint + inputs.odd_checkpoint - even.val_coeff,
                ],
            ),
            (None, Some(odd)) => (
                [F::zero(), odd.ra_coeff + odd.ra_coeff],
                [
                    inputs.even_checkpoint,
                    odd.val_coeff + odd.val_coeff - inputs.even_checkpoint,
                ],
            ),
            (None, None) => unreachable!("RAM address-major message requires at least one entry"),
        };
        for index in 0..2 {
            accumulators[index].fmadd(
                inputs.eq_eval,
                ra_evals[index]
                    * (val_evals[index] + inputs.gamma * (inputs.inc_eval + val_evals[index])),
            );
        }
    }
}

fn ram_address_major_from_cycle_major<F: Field>(
    mut cycle_major: ReadWriteMatrixCycleMajor<F, RamCycleMajorEntry<F>>,
    val_init: Vec<F>,
) -> ReadWriteMatrixAddressMajor<F, RamAddressMajorEntry<F>> {
    let mut entries = std::mem::take(&mut cycle_major.entries);
    entries.par_sort_by(|a, b| {
        a.column()
            .cmp(&b.column())
            .then_with(|| a.row().cmp(&b.row()))
    });
    let entries = entries
        .into_par_iter()
        .map(|entry| entry.to_address_major(None, None))
        .collect();
    ReadWriteMatrixAddressMajor::new_with_val_init(entries, val_init)
}

fn materialize_cycle_major<F: Field>(
    matrix: ReadWriteMatrixCycleMajor<F, RamCycleMajorEntry<F>>,
    k_prime: usize,
    t_prime: usize,
    val_init: &[F],
) -> (Polynomial<F>, Polynomial<F>) {
    let len = k_prime * t_prime;
    let mut ra = vec![F::zero(); len];
    let mut val = vec![F::zero(); len];
    val.par_chunks_mut(t_prime)
        .zip(val_init.par_iter())
        .for_each(|(chunk, &value)| chunk.fill(value));

    for entry in matrix.entries {
        let index = entry.col * t_prime + entry.row;
        ra[index] = entry.ra_coeff;
        val[index] = entry.val_coeff;
    }
    (Polynomial::new(ra), Polynomial::new(val))
}

fn materialize_address_major<F: Field>(
    matrix: ReadWriteMatrixAddressMajor<F, RamAddressMajorEntry<F>>,
    k_prime: usize,
    t_prime: usize,
) -> (Polynomial<F>, Polynomial<F>) {
    let len = k_prime * t_prime;
    let mut ra = vec![F::zero(); len];
    let mut val = vec![F::zero(); len];
    val.par_chunks_mut(t_prime)
        .zip(matrix.val_init.par_iter())
        .for_each(|(chunk, &value)| chunk.fill(value));

    for column in matrix.entries.chunk_by(|a, b| a.column() == b.column()) {
        let k = column[0].column();
        let mut current_val = matrix.val_init[k];
        let mut entries = column.iter().peekable();
        for row in 0..t_prime {
            let index = k * t_prime + row;
            if let Some(entry) = entries.next_if(|entry| entry.row == row) {
                ra[index] = entry.ra_coeff;
                val[index] = entry.val_coeff;
                current_val = entry.next_val;
            } else {
                val[index] = current_val;
            }
        }
    }
    (Polynomial::new(ra), Polynomial::new(val))
}

fn sumcheck_evals_array<F: Field, const DEGREE: usize>(
    polynomial: &Polynomial<F>,
    index: usize,
    order: BindingOrder,
) -> [F; DEGREE] {
    debug_assert!(DEGREE > 0);
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

fn compute_ram_ra_q_arrays<F: Field>(
    addresses: &[Option<usize>],
    address_eq: &[F],
    eq_raf_hi: &[F],
    eq_read_write_hi: &[F],
    eq_val_check_hi: &[F],
    prefix_size: usize,
    prefix_vars: usize,
) -> (Vec<F>, Vec<F>, Vec<F>) {
    let chunk_size = 1 << 14;
    addresses
        .par_chunks(chunk_size)
        .enumerate()
        .fold(
            || {
                (
                    jolt_poly::thread::unsafe_allocate_zero_vec(prefix_size),
                    jolt_poly::thread::unsafe_allocate_zero_vec(prefix_size),
                    jolt_poly::thread::unsafe_allocate_zero_vec(prefix_size),
                )
            },
            |(mut q_raf, mut q_read_write, mut q_val_check), (chunk_index, chunk)| {
                let base_cycle = chunk_index * chunk_size;
                for (offset, address) in chunk.iter().enumerate() {
                    if let Some(address) = address {
                        let cycle = base_cycle + offset;
                        let cycle_lo = cycle & (prefix_size - 1);
                        let cycle_hi = cycle >> prefix_vars;
                        let value = address_eq[*address];
                        q_raf[cycle_lo] += value * eq_raf_hi[cycle_hi];
                        q_read_write[cycle_lo] += value * eq_read_write_hi[cycle_hi];
                        q_val_check[cycle_lo] += value * eq_val_check_hi[cycle_hi];
                    }
                }
                (q_raf, q_read_write, q_val_check)
            },
        )
        .reduce(
            || {
                (
                    jolt_poly::thread::unsafe_allocate_zero_vec(prefix_size),
                    jolt_poly::thread::unsafe_allocate_zero_vec(prefix_size),
                    jolt_poly::thread::unsafe_allocate_zero_vec(prefix_size),
                )
            },
            |(mut raf_acc, mut read_write_acc, mut val_check_acc), (raf, read_write, val_check)| {
                for (acc, value) in raf_acc.iter_mut().zip(raf) {
                    *acc += value;
                }
                for (acc, value) in read_write_acc.iter_mut().zip(read_write) {
                    *acc += value;
                }
                for (acc, value) in val_check_acc.iter_mut().zip(val_check) {
                    *acc += value;
                }
                (raf_acc, read_write_acc, val_check_acc)
            },
        )
}

fn compute_ram_ra_h_prime<F: Field>(
    addresses: &[Option<usize>],
    address_eq: &[F],
    eq_prefix: &[F],
    prefix_vars: usize,
    suffix_vars: usize,
) -> Vec<F> {
    let prefix_size = 1usize << prefix_vars;
    let suffix_size = 1usize << suffix_vars;
    let chunk_size = 1 << 14;
    addresses
        .par_chunks(chunk_size)
        .enumerate()
        .fold(
            || jolt_poly::thread::unsafe_allocate_zero_vec(suffix_size),
            |mut ram_ra, (chunk_index, chunk)| {
                let base_cycle = chunk_index * chunk_size;
                for (offset, address) in chunk.iter().enumerate() {
                    if let Some(address) = address {
                        let cycle = base_cycle + offset;
                        let cycle_lo = cycle & (prefix_size - 1);
                        let cycle_hi = cycle >> prefix_vars;
                        ram_ra[cycle_hi] += address_eq[*address] * eq_prefix[cycle_lo];
                    }
                }
                ram_ra
            },
        )
        .reduce(
            || jolt_poly::thread::unsafe_allocate_zero_vec(suffix_size),
            |mut acc, values| {
                for (acc, value) in acc.iter_mut().zip(values) {
                    *acc += value;
                }
                acc
            },
        )
}

fn validate_ram_ra_claim_reduction_request<F: Field>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckRamRaClaimReductionStateRequest<F>,
) -> Result<(), BackendError> {
    let expected_rows = 1usize << request.log_t;
    if request.rows.len() != expected_rows {
        return invalid(
            backend,
            task,
            format!(
                "RAM RA claim-reduction row count {} does not match log_t {}",
                request.rows.len(),
                request.log_t
            ),
        );
    }
    for (label, point) in [
        ("address", &request.r_address),
        ("RAF cycle", &request.r_cycle_raf),
        ("read-write cycle", &request.r_cycle_read_write),
        ("value-check cycle", &request.r_cycle_val_check),
    ] {
        let expected = if label == "address" {
            request.log_k
        } else {
            request.log_t
        };
        if point.len() != expected {
            return invalid(
                backend,
                task,
                format!(
                    "RAM RA claim-reduction {label} point has {} variables, expected {expected}",
                    point.len()
                ),
            );
        }
    }
    Ok(())
}

fn validate_request<F: Field>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckRamReadWriteStateRequest<F>,
) -> Result<(), BackendError> {
    let expected_rows = 1usize << request.log_t;
    if request.rows.len() != expected_rows {
        return invalid(
            backend,
            task,
            format!(
                "RAM read-write row count {} does not match log_t {}",
                request.rows.len(),
                request.log_t
            ),
        );
    }
    let expected_addresses = 1usize << request.log_k;
    if request.initial_ram_state.len() != expected_addresses {
        return invalid(
            backend,
            task,
            format!(
                "RAM read-write initial state has {} addresses, expected {expected_addresses}",
                request.initial_ram_state.len()
            ),
        );
    }
    if request.r_cycle.len() != request.log_t {
        return invalid(
            backend,
            task,
            format!(
                "RAM read-write r_cycle has {} variables, expected {}",
                request.r_cycle.len(),
                request.log_t
            ),
        );
    }
    if request.phase1_num_rounds > request.log_t {
        return invalid(
            backend,
            task,
            format!(
                "RAM read-write phase1 rounds {} exceed log_t {}",
                request.phase1_num_rounds, request.log_t
            ),
        );
    }
    if request.phase2_num_rounds > request.log_k {
        return invalid(
            backend,
            task,
            format!(
                "RAM read-write phase2 rounds {} exceed log_k {}",
                request.phase2_num_rounds, request.log_k
            ),
        );
    }
    Ok(())
}

#[expect(clippy::too_many_arguments)]
fn validate_stage2_tail_shape(
    backend: &'static str,
    task: &'static str,
    row_count: usize,
    point_len: usize,
    row_log: usize,
    point_log: usize,
    log_t: usize,
    log_k: usize,
    phase1_num_rounds: usize,
    phase2_num_rounds: usize,
) -> Result<(), BackendError> {
    let expected_rows = 1usize << row_log;
    if row_count != expected_rows {
        return invalid(
            backend,
            task,
            format!("Stage 2 RAM state has {row_count} rows, expected {expected_rows}"),
        );
    }
    if point_len != point_log {
        return invalid(
            backend,
            task,
            format!("Stage 2 RAM state point has {point_len} variables, expected {point_log}"),
        );
    }
    if phase1_num_rounds > log_t {
        return invalid(
            backend,
            task,
            format!("Stage 2 RAM phase1 rounds {phase1_num_rounds} exceed log_t {log_t}"),
        );
    }
    if phase2_num_rounds > log_k {
        return invalid(
            backend,
            task,
            format!("Stage 2 RAM phase2 rounds {phase2_num_rounds} exceed log_k {log_k}"),
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

fn final_polynomial_value<F: Field>(
    backend: &'static str,
    task: &'static str,
    label: &'static str,
    polynomial: Option<&Polynomial<F>>,
) -> Result<F, BackendError> {
    let Some(polynomial) = polynomial else {
        return invalid(backend, task, format!("{label} is not materialized"));
    };
    final_required_polynomial_value(backend, task, label, polynomial)
}

fn final_required_polynomial_value<F: Field>(
    backend: &'static str,
    task: &'static str,
    label: &'static str,
    polynomial: &Polynomial<F>,
) -> Result<F, BackendError> {
    let [value] = polynomial.evaluations() else {
        return invalid(
            backend,
            task,
            format!("{label} has {} evaluations, expected 1", polynomial.len()),
        );
    };
    Ok(*value)
}

#[expect(clippy::expect_used)]
fn half_claim_poly<F: Field>(claim: F) -> UnivariatePoly<F> {
    let two_inv = F::from_u64(2).inverse().expect("2 is invertible");
    UnivariatePoly::new(vec![claim * two_inv])
}

fn sum_arrays<F: Field, const N: usize>(left: [F; N], right: [F; N]) -> [F; N] {
    std::array::from_fn(|index| left[index] + right[index])
}
