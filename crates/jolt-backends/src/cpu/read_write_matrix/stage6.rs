use std::{array, ops::Index, sync::Arc};

use jolt_field::{AdditiveAccumulator, Field, RingAccumulator, WithAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use rayon::prelude::*;

use crate::{
    BackendError, SumcheckBooleanityOutput, SumcheckBooleanityStateRequest,
    SumcheckBytecodeReadRafExtraStageValues, SumcheckBytecodeReadRafOutput,
    SumcheckBytecodeReadRafStateRequest, SumcheckIncClaimReductionOutput,
    SumcheckIncClaimReductionStateRequest, SumcheckInstructionRaVirtualizationOutput,
    SumcheckInstructionRaVirtualizationStateRequest, SumcheckRamHammingBooleanityOutput,
    SumcheckRamHammingBooleanityStateRequest, SumcheckRamRaVirtualizationOutput,
    SumcheckRamRaVirtualizationStateRequest, SumcheckStage6IncRow, SumcheckStage6RaRow,
};

use crate::cpu::{
    field,
    ra::{pushforward_indices, RaCycleIndices, RaFamilyLayout, RaPolynomial, SharedRaPolynomials},
};

const BYTECODE_STAGES: usize = 5;
const HAMMING_DEGREE: usize = 3;
const MAX_STACK_STAGE6_EVALS: usize = 16;
const MAX_STACK_BYTECODE_EXTRA_STAGES: usize = 8;

pub struct BytecodeReadRafState<F: Field> {
    round: usize,
    log_t: usize,
    log_k: usize,
    chunk_bits: usize,
    pc_indices: Arc<Vec<usize>>,
    stage_f: [Polynomial<F>; BYTECODE_STAGES],
    stage_val: [Polynomial<F>; BYTECODE_STAGES],
    extra_stage_values: Vec<BytecodeExtraStageState<F>>,
    entry_trace: Polynomial<F>,
    entry_expected: Polynomial<F>,
    gamma_powers: Vec<F>,
    r_cycles: [Vec<F>; BYTECODE_STAGES],
    r_address: Vec<F>,
    cycle_value: Option<BytecodeCycleValueState<F>>,
    ra_polys: Vec<RaPolynomial<u8, F>>,
}

struct BytecodeCycleValueState<F: Field> {
    stage_eq: [GruenSplitEqPolynomial<F>; BYTECODE_STAGES],
    bound_stage_val: [F; BYTECODE_STAGES],
    extra_stage_values: Vec<BytecodeExtraStageCycleState<F>>,
    entry_eq_zero: GruenSplitEqPolynomial<F>,
    bound_entry: F,
}

struct BytecodeExtraStageState<F: Field> {
    stage: usize,
    f: Polynomial<F>,
    val: Polynomial<F>,
    r_cycle: Vec<F>,
}

struct BytecodeExtraStageCycleState<F: Field> {
    stage: usize,
    eq: Polynomial<F>,
    bound_val: F,
}

impl<F> BytecodeReadRafState<F>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    pub fn new(
        backend: &'static str,
        task: &'static str,
        request: &SumcheckBytecodeReadRafStateRequest<F>,
    ) -> Result<Self, BackendError> {
        validate_bytecode_request(backend, task, request)?;

        let k = 1usize << request.log_k;
        let f_values = bytecode_pc_pushforwards(
            &request.r_cycles,
            &request.pc_indices,
            request.log_t,
            request.log_k,
        );

        let mut stage_val = (0..BYTECODE_STAGES)
            .map(|stage| {
                let mut values = jolt_poly::thread::unsafe_allocate_zero_vec(k);
                for (row, row_values) in request.bytecode_stage_values.iter().enumerate() {
                    values[reverse_bits(row, request.log_k)] = row_values[stage];
                }
                values
            })
            .collect::<Vec<_>>();
        let extra_stage_values = request
            .extra_stage_values
            .iter()
            .map(|extra| materialize_extra_bytecode_stage(backend, task, extra, request))
            .collect::<Result<Vec<_>, BackendError>>()?;
        for (index, values) in stage_val.iter_mut().enumerate() {
            if index == 0 {
                for (row, value) in values.iter_mut().enumerate() {
                    let bytecode_row = reverse_bits(row, request.log_k);
                    *value += request.gamma_powers[5] * F::from_u64(bytecode_row as u64);
                }
            } else if index == 2 {
                for (row, value) in values.iter_mut().enumerate() {
                    let bytecode_row = reverse_bits(row, request.log_k);
                    *value += request.gamma_powers[4] * F::from_u64(bytecode_row as u64);
                }
            }
        }

        let entry_trace =
            request
                .pc_indices
                .first()
                .copied()
                .ok_or_else(|| BackendError::InvalidRequest {
                    backend,
                    task,
                    reason: "bytecode read-RAF has no trace rows".to_owned(),
                })?;
        let mut entry_trace_values = jolt_poly::thread::unsafe_allocate_zero_vec(k);
        entry_trace_values[reverse_bits(entry_trace, request.log_k)] = F::one();
        let mut entry_expected_values = jolt_poly::thread::unsafe_allocate_zero_vec(k);
        entry_expected_values[reverse_bits(request.entry_bytecode_index, request.log_k)] = F::one();

        let state = Self {
            round: 0,
            log_t: request.log_t,
            log_k: request.log_k,
            chunk_bits: request.chunk_bits,
            pc_indices: Arc::new(request.pc_indices.clone()),
            stage_f: f_values.map(Polynomial::new),
            stage_val: std::array::from_fn(|stage| Polynomial::new(stage_val[stage].clone())),
            extra_stage_values,
            entry_trace: Polynomial::new(entry_trace_values),
            entry_expected: Polynomial::new(entry_expected_values),
            gamma_powers: request.gamma_powers.clone(),
            r_cycles: request.r_cycles.clone(),
            r_address: Vec::with_capacity(request.log_k),
            cycle_value: None,
            ra_polys: Vec::new(),
        };

        let input_sum = state.address_relation_sum();
        if input_sum != request.input_claim {
            return invalid(
                backend,
                task,
                format!(
                    "bytecode read-RAF input claim mismatch: expected {}, got {}",
                    request.input_claim, input_sum
                ),
            );
        }
        Ok(state)
    }

    pub fn evaluate_round(
        &self,
        backend: &'static str,
        task: &'static str,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError> {
        if self.round < self.log_k {
            return Ok(self.address_round(previous_claim));
        }
        let Some(cycle_value) = &self.cycle_value else {
            return invalid(backend, task, "bytecode read-RAF cycle state is missing");
        };
        Ok(self.cycle_round(previous_claim, cycle_value))
    }

    pub fn bind(
        &mut self,
        backend: &'static str,
        task: &'static str,
        challenge: F,
    ) -> Result<(), BackendError> {
        if self.round < self.log_k {
            let stage_f = &mut self.stage_f;
            let stage_val = &mut self.stage_val;
            let extra_stage_values = &mut self.extra_stage_values;
            let entry_trace = &mut self.entry_trace;
            let entry_expected = &mut self.entry_expected;
            rayon::scope(|scope| {
                scope.spawn(|_| {
                    stage_f
                        .par_iter_mut()
                        .for_each(|poly| poly.bind_with_order(challenge, BindingOrder::HighToLow));
                });
                scope.spawn(|_| {
                    stage_val
                        .par_iter_mut()
                        .for_each(|poly| poly.bind_with_order(challenge, BindingOrder::HighToLow));
                });
                scope.spawn(|_| {
                    extra_stage_values.par_iter_mut().for_each(|extra| {
                        extra.f.bind_with_order(challenge, BindingOrder::HighToLow);
                        extra
                            .val
                            .bind_with_order(challenge, BindingOrder::HighToLow);
                    });
                });
                scope.spawn(|_| {
                    entry_trace.bind_with_order(challenge, BindingOrder::HighToLow);
                });
                scope.spawn(|_| {
                    entry_expected.bind_with_order(challenge, BindingOrder::HighToLow);
                });
            });
            self.r_address.push(challenge);
            self.round += 1;
            if self.round == self.log_k {
                self.init_cycle_state(backend, task)?;
            }
            return Ok(());
        }

        let Some(cycle_value) = &mut self.cycle_value else {
            return invalid(backend, task, "bytecode read-RAF cycle state is missing");
        };
        let stage_eq = &mut cycle_value.stage_eq;
        let extra_stage_values = &mut cycle_value.extra_stage_values;
        let entry_eq_zero = &mut cycle_value.entry_eq_zero;
        let ra_polys = &mut self.ra_polys;
        rayon::scope(|scope| {
            scope.spawn(|_| {
                for poly in stage_eq {
                    poly.bind(challenge);
                }
            });
            scope.spawn(|_| {
                extra_stage_values
                    .par_iter_mut()
                    .for_each(|extra| extra.eq.bind_with_order(challenge, BindingOrder::HighToLow));
            });
            scope.spawn(|_| {
                entry_eq_zero.bind(challenge);
            });
            scope.spawn(|_| {
                ra_polys
                    .par_iter_mut()
                    .for_each(|poly| poly.bind_parallel(challenge, BindingOrder::HighToLow));
            });
        });
        self.round += 1;
        Ok(())
    }

    pub fn output_claims(&self) -> Result<SumcheckBytecodeReadRafOutput<F>, BackendError> {
        Ok(SumcheckBytecodeReadRafOutput {
            bytecode_ra: self
                .ra_polys
                .iter()
                .map(|poly| {
                    poly.final_sumcheck_claim()
                        .ok_or_else(|| BackendError::InvalidRequest {
                            backend: "cpu",
                            task: "bytecode read-RAF output claims",
                            reason: "bytecode RA polynomial is not fully bound".to_owned(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    fn address_round(&self, previous_claim: F) -> UnivariatePoly<F> {
        let len = self.stage_f[0].len() / 2;
        let accumulators = (0..len)
            .into_par_iter()
            .map(|row| {
                let mut evals = [<F as WithAccumulator>::Accumulator::default(); 2];
                for stage in 0..BYTECODE_STAGES {
                    let [f0, f2] =
                        poly_evals_at_0_and_2(&self.stage_f[stage], row, BindingOrder::HighToLow);
                    let [v0, v2] =
                        poly_evals_at_0_and_2(&self.stage_val[stage], row, BindingOrder::HighToLow);
                    evals[0].fmadd(self.gamma_powers[stage] * f0, v0);
                    evals[1].fmadd(self.gamma_powers[stage] * f2, v2);
                }
                for extra in &self.extra_stage_values {
                    let [f0, f2] = poly_evals_at_0_and_2(&extra.f, row, BindingOrder::HighToLow);
                    let [v0, v2] = poly_evals_at_0_and_2(&extra.val, row, BindingOrder::HighToLow);
                    evals[0].fmadd(self.gamma_powers[extra.stage] * f0, v0);
                    evals[1].fmadd(self.gamma_powers[extra.stage] * f2, v2);
                }
                let [trace0, trace2] =
                    poly_evals_at_0_and_2(&self.entry_trace, row, BindingOrder::HighToLow);
                let [entry0, entry2] =
                    poly_evals_at_0_and_2(&self.entry_expected, row, BindingOrder::HighToLow);
                evals[0].fmadd(self.gamma_powers[7] * trace0, entry0);
                evals[1].fmadd(self.gamma_powers[7] * trace2, entry2);
                evals
            })
            .reduce(
                || [<F as WithAccumulator>::Accumulator::default(); 2],
                merge_accumulator_arrays::<F, 2>,
            );
        let [at_zero, at_two] = accumulators.map(AdditiveAccumulator::reduce);
        UnivariatePoly::from_evals_and_hint(previous_claim, &[at_zero, at_two])
    }

    fn address_relation_sum(&self) -> F {
        let rows = self.stage_f[0].len();
        (0..rows)
            .into_par_iter()
            .map(|row| {
                let mut sum = F::zero();
                for stage in 0..BYTECODE_STAGES {
                    sum += self.gamma_powers[stage]
                        * self.stage_f[stage].evaluations()[row]
                        * self.stage_val[stage].evaluations()[row];
                }
                for extra in &self.extra_stage_values {
                    sum += self.gamma_powers[extra.stage]
                        * extra.f.evaluations()[row]
                        * extra.val.evaluations()[row];
                }
                sum + self.gamma_powers[7]
                    * self.entry_trace.evaluations()[row]
                    * self.entry_expected.evaluations()[row]
            })
            .sum()
    }

    fn cycle_round(
        &self,
        previous_claim: F,
        cycle_value: &BytecodeCycleValueState<F>,
    ) -> UnivariatePoly<F> {
        let degree = self.ra_polys.len() + 1;
        if degree > MAX_STACK_STAGE6_EVALS
            || self.ra_polys.len() > MAX_STACK_STAGE6_EVALS
            || cycle_value.extra_stage_values.len() > MAX_STACK_BYTECODE_EXTRA_STAGES
        {
            return self.cycle_round_heap(cycle_value);
        }

        let points = stack_hint_points::<F>(degree);
        let stage_coefficients: [F; BYTECODE_STAGES] =
            array::from_fn(|stage| self.gamma_powers[stage] * cycle_value.bound_stage_val[stage]);
        let stage_linear_evals: [[F; MAX_STACK_STAGE6_EVALS]; BYTECODE_STAGES] =
            array::from_fn(|stage| {
                let (lo, hi) = cycle_value.stage_eq[stage].current_linear_evals();
                array::from_fn(|index| linear_eval(lo, hi, points[index]))
            });
        let extra_coefficients =
            stack_extra_coefficients(&cycle_value.extra_stage_values, &self.gamma_powers);
        let (entry_lo, entry_hi) = cycle_value.entry_eq_zero.current_linear_evals();
        let entry_linear_evals: [F; MAX_STACK_STAGE6_EVALS] =
            array::from_fn(|index| linear_eval(entry_lo, entry_hi, points[index]));
        let entry_coefficient = self.gamma_powers[7] * cycle_value.bound_entry;
        let eq_layout = &cycle_value.stage_eq[0];
        let out_bits = eq_layout.e_out_current_len().trailing_zeros() as usize;
        let accumulators = (0..eq_layout.e_out_current_len())
            .into_par_iter()
            .map(|x_out| {
                let mut values =
                    [<F as WithAccumulator>::Accumulator::default(); MAX_STACK_STAGE6_EVALS];
                for x_in in 0..eq_layout.e_in_current_len() {
                    let row = (x_in << out_bits) | x_out;
                    let stage_eq_heads: [F; BYTECODE_STAGES] = array::from_fn(|stage| {
                        cycle_value.stage_eq[stage].e_out_current()[x_out]
                            * cycle_value.stage_eq[stage].e_in_current()[x_in]
                    });
                    let entry_eq_head = cycle_value.entry_eq_zero.e_out_current()[x_out]
                        * cycle_value.entry_eq_zero.e_in_current()[x_in];
                    let mut extra_pairs =
                        [([F::zero(); 2], F::zero()); MAX_STACK_BYTECODE_EXTRA_STAGES];
                    for (index, extra) in cycle_value.extra_stage_values.iter().enumerate() {
                        extra_pairs[index] = (
                            poly_eval_pair(&extra.eq, row, BindingOrder::HighToLow),
                            extra_coefficients[index],
                        );
                    }
                    let mut ra_pairs = [(F::zero(), F::zero()); MAX_STACK_STAGE6_EVALS];
                    for (index, poly) in self.ra_polys.iter().enumerate() {
                        ra_pairs[index] = ra_high_to_low_pair(poly, row);
                    }

                    for (point_index, &point) in points[..degree].iter().enumerate() {
                        let mut coeff = F::zero();
                        for stage in 0..BYTECODE_STAGES {
                            coeff += stage_coefficients[stage]
                                * stage_eq_heads[stage]
                                * stage_linear_evals[stage][point_index];
                        }
                        for &(pair, coefficient) in
                            &extra_pairs[..cycle_value.extra_stage_values.len()]
                        {
                            coeff += coefficient * linear_eval(pair[0], pair[1], point);
                        }
                        coeff +=
                            entry_coefficient * entry_eq_head * entry_linear_evals[point_index];
                        let product = ra_pairs[..self.ra_polys.len()]
                            .iter()
                            .fold(F::one(), |acc, &(lo, hi)| acc * linear_eval(lo, hi, point));
                        values[point_index].fmadd(coeff, product);
                    }
                }
                values
            })
            .reduce(
                || [<F as WithAccumulator>::Accumulator::default(); MAX_STACK_STAGE6_EVALS],
                merge_accumulator_arrays::<F, MAX_STACK_STAGE6_EVALS>,
            );
        let evals = accumulators.map(AdditiveAccumulator::reduce);
        UnivariatePoly::from_evals_and_hint(previous_claim, &evals[..degree])
    }

    fn cycle_round_heap(&self, cycle_value: &BytecodeCycleValueState<F>) -> UnivariatePoly<F> {
        let degree = self.ra_polys.len() + 1;
        let rows = self.ra_polys.first().map_or(0, |poly| poly.len() / 2);
        let stage_eq = cycle_value
            .stage_eq
            .each_ref()
            .map(GruenSplitEqPolynomial::merge);
        let entry_eq_zero = cycle_value.entry_eq_zero.merge();
        let evals = (0..rows)
            .into_par_iter()
            .map(|row| {
                let mut values = vec![F::zero(); degree + 1];
                let ra_evals = self
                    .ra_polys
                    .iter()
                    .map(|poly| ra_evals_over_integer_domain(poly, row, degree))
                    .collect::<Vec<_>>();
                for point in 0..=degree {
                    let mut coeff = F::zero();
                    for (stage, stage_eq) in stage_eq.iter().enumerate() {
                        let [eq0, eq1] = poly_eval_pair(stage_eq, row, BindingOrder::HighToLow);
                        coeff += self.gamma_powers[stage]
                            * linear_eval(eq0, eq1, F::from_u64(point as u64))
                            * cycle_value.bound_stage_val[stage];
                    }
                    for extra in &cycle_value.extra_stage_values {
                        let [eq0, eq1] = poly_eval_pair(&extra.eq, row, BindingOrder::HighToLow);
                        coeff += self.gamma_powers[extra.stage]
                            * linear_eval(eq0, eq1, F::from_u64(point as u64))
                            * extra.bound_val;
                    }
                    let [entry0, entry1] =
                        poly_eval_pair(&entry_eq_zero, row, BindingOrder::HighToLow);
                    coeff += self.gamma_powers[7]
                        * cycle_value.bound_entry
                        * linear_eval(entry0, entry1, F::from_u64(point as u64));
                    let product = ra_evals
                        .iter()
                        .fold(F::one(), |acc, evals| acc * evals[point]);
                    values[point] = coeff * product;
                }
                values
            })
            .reduce(
                || vec![F::zero(); degree + 1],
                |mut left, right| {
                    merge_vec(&mut left, right);
                    left
                },
            );
        UnivariatePoly::from_evals(&evals)
    }

    fn init_cycle_state(
        &mut self,
        backend: &'static str,
        task: &'static str,
    ) -> Result<(), BackendError> {
        let chunks = checked_chunks(backend, task, self.log_k, self.chunk_bits)?;
        let opening_address = self.r_address.iter().rev().copied().collect::<Vec<_>>();
        let r_address_chunks = challenge_chunks(&opening_address, chunks, self.chunk_bits);
        self.ra_polys = r_address_chunks
            .into_iter()
            .enumerate()
            .map(|(chunk, point)| {
                let selector = RaChunkSelector::new(chunk, chunks, self.chunk_bits);
                let address_eq = EqPolynomial::<F>::evals(&point, None);
                let mut indices = vec![None; self.pc_indices.len()];
                for (cycle, &pc) in self.pc_indices.iter().enumerate() {
                    indices[reverse_bits(cycle, self.log_t)] = Some(selector.chunk_usize(pc) as u8);
                }
                RaPolynomial::new(Arc::new(indices), address_eq)
            })
            .collect();
        let bound_stage_val = self
            .stage_val
            .each_ref()
            .map(|poly| final_poly_claim(poly).unwrap_or_else(F::zero));
        let extra_stage_values = self
            .extra_stage_values
            .iter()
            .map(|extra| BytecodeExtraStageCycleState {
                stage: extra.stage,
                eq: Polynomial::new(reverse_index_table(
                    EqPolynomial::<F>::evals(&extra.r_cycle, None),
                    self.log_t,
                )),
                bound_val: final_poly_claim(&extra.val).unwrap_or_else(F::zero),
            })
            .collect::<Vec<_>>();
        let bound_entry = final_poly_claim(&self.entry_expected).unwrap_or_else(F::zero);
        let zero = vec![F::zero(); self.log_t];
        self.cycle_value = Some(BytecodeCycleValueState {
            stage_eq: self.r_cycles.each_ref().map(|r_cycle| {
                let point = r_cycle.iter().rev().copied().collect::<Vec<_>>();
                GruenSplitEqPolynomial::new(&point, BindingOrder::HighToLow)
            }),
            bound_stage_val,
            extra_stage_values,
            entry_eq_zero: GruenSplitEqPolynomial::new(&zero, BindingOrder::HighToLow),
            bound_entry,
        });
        self.stage_f = std::array::from_fn(|_| Polynomial::new(vec![F::zero()]));
        self.stage_val = std::array::from_fn(|_| Polynomial::new(vec![F::zero()]));
        self.extra_stage_values.clear();
        self.entry_trace = Polynomial::new(vec![F::zero()]);
        self.entry_expected = Polynomial::new(vec![F::zero()]);
        Ok(())
    }
}

pub struct BooleanityState<F: Field> {
    round: usize,
    chunk_bits: usize,
    layout: RaFamilyLayout,
    indices: Vec<RaCycleIndices>,
    b: GruenSplitEqPolynomial<F>,
    d: GruenSplitEqPolynomial<F>,
    g: Vec<Vec<F>>,
    f: ExpandingTable<F>,
    h: Option<SharedRaPolynomials<F>>,
    eq_r_r: F,
    gamma_powers: Vec<F>,
    gamma_powers_inv: Vec<F>,
    gamma_powers_square: Vec<F>,
}

impl<F> BooleanityState<F>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    pub fn new(
        backend: &'static str,
        task: &'static str,
        request: &SumcheckBooleanityStateRequest<F>,
    ) -> Result<Self, BackendError> {
        validate_booleanity_request(backend, task, request)?;
        let layout = RaFamilyLayout::new(
            1usize << request.chunk_bits,
            request.instruction_chunks,
            request.bytecode_chunks,
            request.ram_chunks,
        );
        let indices = ra_cycle_indices(&request.rows, request.chunk_bits, layout, None);
        let g = pushforward_indices(&indices, layout, &request.r_cycle)
            .into_iter()
            .collect::<Vec<_>>();
        if request.input_claim != F::zero() {
            return invalid(
                backend,
                task,
                format!(
                    "booleanity input claim mismatch: expected 0, got {}",
                    request.input_claim
                ),
            );
        }
        let mut f = ExpandingTable::new(1usize << request.chunk_bits, BindingOrder::LowToHigh);
        f.reset(F::one());
        let mut gamma_powers = Vec::with_capacity(layout.num_polys());
        let mut gamma_powers_inv = Vec::with_capacity(layout.num_polys());
        let mut gamma_power = F::one();
        for _ in 0..layout.num_polys() {
            gamma_powers.push(gamma_power);
            gamma_powers_inv.push(gamma_power.inverse().ok_or_else(|| {
                BackendError::InvalidRequest {
                    backend,
                    task,
                    reason: "booleanity gamma power is not invertible".to_owned(),
                }
            })?);
            gamma_power *= request.gamma;
        }
        let gamma_squared = request.gamma * request.gamma;
        let mut gamma_powers_square = Vec::with_capacity(layout.num_polys());
        let mut gamma = F::one();
        for _ in 0..layout.num_polys() {
            gamma_powers_square.push(gamma);
            gamma *= gamma_squared;
        }
        Ok(Self {
            round: 0,
            chunk_bits: request.chunk_bits,
            layout,
            indices,
            b: GruenSplitEqPolynomial::new(&request.r_address, BindingOrder::LowToHigh),
            d: GruenSplitEqPolynomial::new(&request.r_cycle, BindingOrder::LowToHigh),
            g,
            f,
            h: None,
            eq_r_r: F::zero(),
            gamma_powers,
            gamma_powers_inv,
            gamma_powers_square,
        })
    }

    pub fn evaluate_round(&self, previous_claim: F) -> Result<UnivariatePoly<F>, BackendError> {
        if self.round < self.chunk_bits {
            return Ok(self.phase1_round(previous_claim));
        }
        self.phase2_round(previous_claim)
    }

    pub fn bind(&mut self, challenge: F) -> Result<(), BackendError> {
        if self.round < self.chunk_bits {
            self.b.bind(challenge);
            self.f.update(challenge);
            self.round += 1;
            if self.round == self.chunk_bits {
                self.eq_r_r = self.b.current_scalar();
                let base_eq = self.f.clone_values();
                let tables = (0..self.layout.num_polys())
                    .into_par_iter()
                    .map(|index| {
                        let rho = self.gamma_powers[index];
                        base_eq.iter().map(|value| rho * *value).collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let indices = std::mem::take(&mut self.indices);
                self.h = Some(SharedRaPolynomials::new(tables, indices, self.layout));
                self.g.clear();
            }
        } else {
            self.d.bind(challenge);
            if let Some(h) = &mut self.h {
                h.bind_in_place(challenge, BindingOrder::LowToHigh);
            }
            self.round += 1;
        }
        Ok(())
    }

    pub fn output_claims(&self) -> Result<SumcheckBooleanityOutput<F>, BackendError> {
        let h = self
            .h
            .as_ref()
            .ok_or_else(|| BackendError::InvalidRequest {
                backend: "cpu",
                task: "booleanity output claims",
                reason: "booleanity RA state is missing".to_owned(),
            })?;
        let mut claims = (0..self.layout.num_polys())
            .map(|index| {
                h.final_sumcheck_claim(index)
                    .map(|claim| claim * self.gamma_powers_inv[index])
                    .ok_or_else(|| BackendError::InvalidRequest {
                        backend: "cpu",
                        task: "booleanity output claims",
                        reason: "booleanity RA polynomial is not fully bound".to_owned(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ram_ra = claims.split_off(self.layout.instruction_chunks + self.layout.bytecode_chunks);
        let bytecode_ra = claims.split_off(self.layout.instruction_chunks);
        let instruction_ra = claims;
        Ok(SumcheckBooleanityOutput {
            instruction_ra,
            bytecode_ra,
            ram_ra,
        })
    }

    fn phase1_round(&self, previous_claim: F) -> UnivariatePoly<F> {
        let m = self.round + 1;
        let num_polys = self.layout.num_polys();
        let quadratic_coeffs = gruen_fold2(&self.b, |k_prime| {
            (0..num_polys)
                .into_par_iter()
                .map(|index| {
                    let g = &self.g[index];
                    let inner_sum = g[k_prime << m..(k_prime + 1) << m]
                        .par_iter()
                        .enumerate()
                        .map(|(k, &g_k)| {
                            let k_m = k >> (m - 1);
                            let f_k = self.f[k & ((1usize << (m - 1)) - 1)];
                            let g_times_f = g_k * f_k;
                            let mut evals = [<F as WithAccumulator>::Accumulator::default(); 2];
                            evals[1].fmadd(g_times_f, f_k);
                            if k_m == 0 {
                                evals[0].fmadd(g_times_f, f_k);
                                evals[0].add(-g_times_f);
                            }
                            evals
                        })
                        .reduce(
                            || [<F as WithAccumulator>::Accumulator::default(); 2],
                            merge_accumulator_arrays::<F, 2>,
                        )
                        .map(AdditiveAccumulator::reduce);
                    let gamma_2i = self.gamma_powers_square[index];
                    [gamma_2i * inner_sum[0], gamma_2i * inner_sum[1]]
                })
                .reduce(|| [F::zero(); 2], sum_arrays::<F, 2>)
        });
        self.b
            .gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
    }

    fn phase2_round(&self, previous_claim: F) -> Result<UnivariatePoly<F>, BackendError> {
        let h = self
            .h
            .as_ref()
            .ok_or_else(|| BackendError::InvalidRequest {
                backend: "cpu",
                task: "booleanity phase2 round",
                reason: "booleanity RA state is missing".to_owned(),
            })?;
        let quadratic_coeffs = gruen_fold2(&self.d, |j_prime| {
            (0..h.num_polys())
                .fold(
                    [<F as WithAccumulator>::Accumulator::default(); 2],
                    |mut acc, index| {
                        let h_0 = h.get_bound_coeff(index, 2 * j_prime);
                        let h_1 = h.get_bound_coeff(index, 2 * j_prime + 1);
                        let slope = h_1 - h_0;
                        let rho = self.gamma_powers[index];
                        acc[0].fmadd(h_0, h_0 - rho);
                        acc[1].fmadd(slope, slope);
                        acc
                    },
                )
                .map(AdditiveAccumulator::reduce)
        });
        let adjusted_claim = previous_claim
            * self
                .eq_r_r
                .inverse()
                .ok_or_else(|| BackendError::InvalidRequest {
                    backend: "cpu",
                    task: "booleanity phase2 round",
                    reason: "booleanity address equality scalar is not invertible".to_owned(),
                })?;
        Ok(self
            .d
            .gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], adjusted_claim)
            * self.eq_r_r)
    }
}

pub struct RamHammingBooleanityState<F: Field> {
    eq_cycle: Polynomial<F>,
    hamming_weight: Polynomial<F>,
}

impl<F> RamHammingBooleanityState<F>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    pub fn new(
        backend: &'static str,
        task: &'static str,
        request: &SumcheckRamHammingBooleanityStateRequest<F>,
    ) -> Result<Self, BackendError> {
        validate_hamming_request(backend, task, request)?;
        Ok(Self {
            eq_cycle: Polynomial::new(EqPolynomial::<F>::evals(&request.r_cycle, None)),
            hamming_weight: Polynomial::new(reverse_cycle_table(
                request
                    .hamming_weight
                    .iter()
                    .map(|&value| F::from_bool(value))
                    .collect(),
                request.log_t,
            )),
        })
    }

    pub fn evaluate_round(&self, previous_claim: F) -> UnivariatePoly<F> {
        let rows = self.eq_cycle.len() / 2;
        let accumulators = (0..rows)
            .into_par_iter()
            .map(|row| {
                let mut out = [<F as WithAccumulator>::Accumulator::default(); HAMMING_DEGREE + 1];
                let [eq0, eq1] = poly_eval_pair(&self.eq_cycle, row, BindingOrder::HighToLow);
                let [h0, h1] = poly_eval_pair(&self.hamming_weight, row, BindingOrder::HighToLow);
                for (point, value) in out.iter_mut().enumerate().take(HAMMING_DEGREE + 1) {
                    let point_f = F::from_u64(point as u64);
                    let eq = linear_eval(eq0, eq1, point_f);
                    let h = linear_eval(h0, h1, point_f);
                    value.fmadd(eq, h * h - h);
                }
                out
            })
            .reduce(
                || [<F as WithAccumulator>::Accumulator::default(); HAMMING_DEGREE + 1],
                merge_accumulator_arrays::<F, { HAMMING_DEGREE + 1 }>,
            );
        let evals = accumulators.map(AdditiveAccumulator::reduce);
        UnivariatePoly::from_evals_and_hint(previous_claim, &[evals[0], evals[2], evals[3]])
    }

    pub fn bind(&mut self, challenge: F) {
        self.eq_cycle
            .bind_with_order(challenge, BindingOrder::HighToLow);
        self.hamming_weight
            .bind_with_order(challenge, BindingOrder::HighToLow);
    }

    pub fn output_claims(&self) -> Result<SumcheckRamHammingBooleanityOutput<F>, BackendError> {
        Ok(SumcheckRamHammingBooleanityOutput {
            ram_hamming_weight: final_poly_claim(&self.hamming_weight).ok_or_else(|| {
                BackendError::InvalidRequest {
                    backend: "cpu",
                    task: "RAM hamming booleanity output claims",
                    reason: "RAM hamming polynomial is not fully bound".to_owned(),
                }
            })?,
        })
    }
}

pub struct RamRaVirtualizationState<F: Field> {
    eq_cycle: GruenSplitEqPolynomial<F>,
    ra_polys: Vec<RaPolynomial<u8, F>>,
}

impl<F> RamRaVirtualizationState<F>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    pub fn new(
        backend: &'static str,
        task: &'static str,
        request: &SumcheckRamRaVirtualizationStateRequest<F>,
    ) -> Result<Self, BackendError> {
        validate_ram_ra_virtual_request(backend, task, request)?;
        Ok(Self {
            eq_cycle: GruenSplitEqPolynomial::new(&request.r_cycle, BindingOrder::LowToHigh),
            ra_polys: request
                .r_address_chunks
                .iter()
                .enumerate()
                .map(|(chunk, point)| {
                    let selector = RaChunkSelector::new(
                        chunk,
                        request.r_address_chunks.len(),
                        request.chunk_bits,
                    );
                    let address_eq = EqPolynomial::<F>::evals(point, None);
                    let mut indices = vec![None; request.rows.len()];
                    for (cycle, row) in request.rows.iter().enumerate() {
                        indices[cycle] = row
                            .ram_address
                            .map(|address| selector.chunk_usize(address) as u8);
                    }
                    RaPolynomial::new(Arc::new(indices), address_eq)
                })
                .collect(),
        })
    }

    pub fn evaluate_round(&self, previous_claim: F) -> UnivariatePoly<F> {
        ra_product_sum_round(
            &self.eq_cycle,
            &[&self.ra_polys],
            &[F::one()],
            previous_claim,
        )
    }

    pub fn bind(&mut self, challenge: F) {
        self.eq_cycle.bind(challenge);
        self.ra_polys
            .par_iter_mut()
            .for_each(|poly| poly.bind_parallel(challenge, BindingOrder::LowToHigh));
    }

    pub fn output_claims(&self) -> Result<SumcheckRamRaVirtualizationOutput<F>, BackendError> {
        Ok(SumcheckRamRaVirtualizationOutput {
            ram_ra: self
                .ra_polys
                .iter()
                .map(|poly| {
                    poly.final_sumcheck_claim()
                        .ok_or_else(|| BackendError::InvalidRequest {
                            backend: "cpu",
                            task: "RAM RA virtualization output claims",
                            reason: "RAM RA polynomial is not fully bound".to_owned(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

pub struct InstructionRaVirtualizationState<F: Field> {
    eq_cycle: GruenSplitEqPolynomial<F>,
    ra_polys: Vec<RaPolynomial<u8, F>>,
    virtual_polys: usize,
    committed_per_virtual: usize,
    gamma_powers: Vec<F>,
}

impl<F> InstructionRaVirtualizationState<F>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    pub fn new(
        backend: &'static str,
        task: &'static str,
        request: &SumcheckInstructionRaVirtualizationStateRequest<F>,
    ) -> Result<Self, BackendError> {
        validate_instruction_ra_virtual_request(backend, task, request)?;
        Ok(Self {
            eq_cycle: GruenSplitEqPolynomial::new(&request.r_cycle, BindingOrder::LowToHigh),
            ra_polys: request
                .r_address_chunks
                .iter()
                .enumerate()
                .map(|(chunk, point)| {
                    let selector = RaChunkSelector::new(
                        chunk,
                        request.r_address_chunks.len(),
                        request.chunk_bits,
                    );
                    let batch = chunk / request.committed_per_virtual;
                    let scaling_factor = (chunk % request.committed_per_virtual == 0)
                        .then(|| request.gamma_powers[batch])
                        .filter(|gamma| *gamma != F::one());
                    let address_eq = EqPolynomial::<F>::evals(point, scaling_factor);
                    let mut indices = vec![None; request.rows.len()];
                    for (cycle, row) in request.rows.iter().enumerate() {
                        indices[cycle] =
                            Some(selector.chunk_u128(row.instruction_lookup_index) as u8);
                    }
                    RaPolynomial::new(Arc::new(indices), address_eq)
                })
                .collect(),
            virtual_polys: request.virtual_polys,
            committed_per_virtual: request.committed_per_virtual,
            gamma_powers: request.gamma_powers.clone(),
        })
    }

    pub fn evaluate_round(&self, previous_claim: F) -> UnivariatePoly<F> {
        instruction_ra_product_sum_round(
            &self.eq_cycle,
            &self.ra_polys,
            self.virtual_polys,
            self.committed_per_virtual,
            previous_claim,
        )
    }

    pub fn bind(&mut self, challenge: F) {
        self.eq_cycle.bind(challenge);
        self.ra_polys
            .par_iter_mut()
            .for_each(|poly| poly.bind_parallel(challenge, BindingOrder::LowToHigh));
    }

    pub fn output_claims(
        &self,
    ) -> Result<SumcheckInstructionRaVirtualizationOutput<F>, BackendError> {
        let expected = self.virtual_polys * self.committed_per_virtual;
        if self.ra_polys.len() != expected {
            return Err(BackendError::InvalidRequest {
                backend: "cpu",
                task: "instruction RA virtualization output claims",
                reason: format!(
                    "instruction RA state has {} polynomials, expected {expected}",
                    self.ra_polys.len()
                ),
            });
        }
        Ok(SumcheckInstructionRaVirtualizationOutput {
            instruction_ra: self
                .ra_polys
                .iter()
                .enumerate()
                .map(|(index, poly)| {
                    let mut claim = poly.final_sumcheck_claim().ok_or_else(|| {
                        BackendError::InvalidRequest {
                            backend: "cpu",
                            task: "instruction RA virtualization output claims",
                            reason: "instruction RA polynomial is not fully bound".to_owned(),
                        }
                    })?;
                    if index % self.committed_per_virtual == 0 {
                        let gamma = self.gamma_powers[index / self.committed_per_virtual];
                        if gamma != F::one() {
                            claim *=
                                gamma
                                    .inverse()
                                    .ok_or_else(|| BackendError::InvalidRequest {
                                        backend: "cpu",
                                        task: "instruction RA virtualization output claims",
                                        reason: "instruction RA gamma is not invertible".to_owned(),
                                    })?;
                        }
                    }
                    Ok::<F, BackendError>(claim)
                })
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

pub struct IncClaimReductionState<F: Field> {
    phase: IncClaimReductionPhase<F>,
}

#[expect(
    clippy::large_enum_variant,
    reason = "phase state is moved linearly; boxing would add indirection to the hot path"
)]
enum IncClaimReductionPhase<F: Field> {
    Prefix(IncClaimReductionPrefixState<F>),
    Suffix(IncClaimReductionSuffixState<F>),
    Taken,
}

struct IncClaimReductionPrefixState<F: Field> {
    log_t: usize,
    prefix_vars: usize,
    gamma: F,
    gamma_squared: F,
    gamma_cubed: F,
    rows: Vec<SumcheckStage6IncRow>,
    r_cycle_stage2: Vec<F>,
    r_cycle_stage4: Vec<F>,
    s_cycle_stage4: Vec<F>,
    s_cycle_stage5: Vec<F>,
    p_ram: [Polynomial<F>; 2],
    p_rd: [Polynomial<F>; 2],
    q_ram: [Polynomial<F>; 2],
    q_rd: [Polynomial<F>; 2],
    challenges: Vec<F>,
}

struct IncClaimReductionSuffixState<F: Field> {
    gamma_squared: F,
    ram_inc: Polynomial<F>,
    rd_inc: Polynomial<F>,
    ram_coeff: Polynomial<F>,
    rd_coeff: Polynomial<F>,
}

impl<F: Field> IncClaimReductionState<F> {
    pub fn new(
        backend: &'static str,
        task: &'static str,
        request: &SumcheckIncClaimReductionStateRequest<F>,
    ) -> Result<Self, BackendError> {
        validate_inc_request(backend, task, request)?;
        let prefix_vars = request.log_t / 2;
        let phase = if prefix_vars == 0 {
            IncClaimReductionPhase::Suffix(IncClaimReductionSuffixState::from_prefix_bound(
                request.log_t,
                prefix_vars,
                &[],
                &request.rows,
                &request.r_cycle_stage2,
                &request.r_cycle_stage4,
                &request.s_cycle_stage4,
                &request.s_cycle_stage5,
                request.gamma,
                F::one(),
                F::one(),
                F::one(),
                F::one(),
            ))
        } else {
            IncClaimReductionPhase::Prefix(IncClaimReductionPrefixState::new(request))
        };
        Ok(Self { phase })
    }

    pub fn evaluate_round(&self, previous_claim: F) -> Result<UnivariatePoly<F>, BackendError> {
        match &self.phase {
            IncClaimReductionPhase::Prefix(state) => Ok(state.evaluate_round(previous_claim)),
            IncClaimReductionPhase::Suffix(state) => Ok(state.evaluate_round(previous_claim)),
            IncClaimReductionPhase::Taken => invalid(
                "cpu",
                "increment claim-reduction round",
                "increment claim-reduction state was temporarily moved",
            ),
        }
    }

    pub fn bind(&mut self, challenge: F) {
        let should_transition = match &mut self.phase {
            IncClaimReductionPhase::Prefix(state) => {
                state.bind(challenge);
                state.is_complete()
            }
            IncClaimReductionPhase::Suffix(state) => {
                state.bind(challenge);
                false
            }
            IncClaimReductionPhase::Taken => unreachable!("increment state is never left taken"),
        };
        if should_transition {
            let IncClaimReductionPhase::Prefix(state) =
                std::mem::replace(&mut self.phase, IncClaimReductionPhase::Taken)
            else {
                unreachable!("increment state transition requires prefix phase");
            };
            self.phase = IncClaimReductionPhase::Suffix(state.into_suffix());
        }
    }

    pub fn output_claims(&self) -> Result<SumcheckIncClaimReductionOutput<F>, BackendError> {
        let IncClaimReductionPhase::Suffix(state) = &self.phase else {
            return invalid(
                "cpu",
                "increment claim-reduction output claims",
                "increment claim-reduction output requested before suffix phase completed",
            );
        };
        state.output_claims()
    }
}

impl<F: Field> IncClaimReductionPrefixState<F> {
    fn new(request: &SumcheckIncClaimReductionStateRequest<F>) -> Self {
        let prefix_vars = request.log_t / 2;
        let suffix_vars = request.log_t - prefix_vars;
        let (r2_prefix, r2_suffix) = request.r_cycle_stage2.split_at(prefix_vars);
        let (r4_prefix, r4_suffix) = request.r_cycle_stage4.split_at(prefix_vars);
        let (s4_prefix, s4_suffix) = request.s_cycle_stage4.split_at(prefix_vars);
        let (s5_prefix, s5_suffix) = request.s_cycle_stage5.split_at(prefix_vars);
        let p_ram = [
            Polynomial::new(EqPolynomial::<F>::evals(r2_prefix, None)),
            Polynomial::new(EqPolynomial::<F>::evals(r4_prefix, None)),
        ];
        let p_rd = [
            Polynomial::new(EqPolynomial::<F>::evals(s4_prefix, None)),
            Polynomial::new(EqPolynomial::<F>::evals(s5_prefix, None)),
        ];
        let q_ram = [
            Polynomial::new(prefix_suffix_inc_q(
                &request.rows,
                request.log_t,
                prefix_vars,
                suffix_vars,
                r2_suffix,
                |row| F::from_i128(row.ram_increment),
            )),
            Polynomial::new(prefix_suffix_inc_q(
                &request.rows,
                request.log_t,
                prefix_vars,
                suffix_vars,
                r4_suffix,
                |row| F::from_i128(row.ram_increment),
            )),
        ];
        let q_rd = [
            Polynomial::new(prefix_suffix_inc_q(
                &request.rows,
                request.log_t,
                prefix_vars,
                suffix_vars,
                s4_suffix,
                |row| F::from_i128(row.rd_increment),
            )),
            Polynomial::new(prefix_suffix_inc_q(
                &request.rows,
                request.log_t,
                prefix_vars,
                suffix_vars,
                s5_suffix,
                |row| F::from_i128(row.rd_increment),
            )),
        ];
        let gamma_squared = request.gamma * request.gamma;
        Self {
            log_t: request.log_t,
            prefix_vars,
            gamma: request.gamma,
            gamma_squared,
            gamma_cubed: gamma_squared * request.gamma,
            rows: request.rows.clone(),
            r_cycle_stage2: request.r_cycle_stage2.clone(),
            r_cycle_stage4: request.r_cycle_stage4.clone(),
            s_cycle_stage4: request.s_cycle_stage4.clone(),
            s_cycle_stage5: request.s_cycle_stage5.clone(),
            p_ram,
            p_rd,
            q_ram,
            q_rd,
            challenges: Vec::with_capacity(prefix_vars),
        }
    }

    fn evaluate_round(&self, previous_claim: F) -> UnivariatePoly<F> {
        let rows = self.p_ram[0].len() / 2;
        let evals = (0..rows)
            .into_par_iter()
            .map(|row| {
                let p_ram_0 = poly_evals_at_0_and_2(&self.p_ram[0], row, BindingOrder::HighToLow);
                let p_ram_1 = poly_evals_at_0_and_2(&self.p_ram[1], row, BindingOrder::HighToLow);
                let p_rd_0 = poly_evals_at_0_and_2(&self.p_rd[0], row, BindingOrder::HighToLow);
                let p_rd_1 = poly_evals_at_0_and_2(&self.p_rd[1], row, BindingOrder::HighToLow);
                let q_ram_0 = poly_evals_at_0_and_2(&self.q_ram[0], row, BindingOrder::HighToLow);
                let q_ram_1 = poly_evals_at_0_and_2(&self.q_ram[1], row, BindingOrder::HighToLow);
                let q_rd_0 = poly_evals_at_0_and_2(&self.q_rd[0], row, BindingOrder::HighToLow);
                let q_rd_1 = poly_evals_at_0_and_2(&self.q_rd[1], row, BindingOrder::HighToLow);
                std::array::from_fn(|point| {
                    p_ram_0[point] * q_ram_0[point]
                        + self.gamma * p_ram_1[point] * q_ram_1[point]
                        + self.gamma_squared * p_rd_0[point] * q_rd_0[point]
                        + self.gamma_cubed * p_rd_1[point] * q_rd_1[point]
                })
            })
            .reduce(|| [F::zero(); 2], sum_arrays::<F, 2>);
        UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, challenge: F) {
        for polynomial in &mut self.p_ram {
            polynomial.bind_with_order(challenge, BindingOrder::HighToLow);
        }
        for polynomial in &mut self.p_rd {
            polynomial.bind_with_order(challenge, BindingOrder::HighToLow);
        }
        for polynomial in &mut self.q_ram {
            polynomial.bind_with_order(challenge, BindingOrder::HighToLow);
        }
        for polynomial in &mut self.q_rd {
            polynomial.bind_with_order(challenge, BindingOrder::HighToLow);
        }
        self.challenges.push(challenge);
    }

    fn is_complete(&self) -> bool {
        self.challenges.len() == self.prefix_vars
    }

    fn into_suffix(self) -> IncClaimReductionSuffixState<F> {
        let [r2_prefix_scale, r4_prefix_scale] = self.p_ram.map(|poly| poly.evaluations()[0]);
        let [s4_prefix_scale, s5_prefix_scale] = self.p_rd.map(|poly| poly.evaluations()[0]);
        IncClaimReductionSuffixState::from_prefix_bound(
            self.log_t,
            self.prefix_vars,
            &self.challenges,
            &self.rows,
            &self.r_cycle_stage2,
            &self.r_cycle_stage4,
            &self.s_cycle_stage4,
            &self.s_cycle_stage5,
            self.gamma,
            r2_prefix_scale,
            r4_prefix_scale,
            s4_prefix_scale,
            s5_prefix_scale,
        )
    }
}

impl<F: Field> IncClaimReductionSuffixState<F> {
    #[expect(
        clippy::too_many_arguments,
        reason = "The four verifier opening points are the Stage 6 increment relation inputs."
    )]
    fn from_prefix_bound(
        log_t: usize,
        prefix_vars: usize,
        prefix_challenges: &[F],
        rows: &[SumcheckStage6IncRow],
        r_cycle_stage2: &[F],
        r_cycle_stage4: &[F],
        s_cycle_stage4: &[F],
        s_cycle_stage5: &[F],
        gamma: F,
        r2_prefix_scale: F,
        r4_prefix_scale: F,
        s4_prefix_scale: F,
        s5_prefix_scale: F,
    ) -> Self {
        let suffix_vars = log_t - prefix_vars;
        let suffix_len = 1usize << suffix_vars;
        let prefix_len = 1usize << prefix_vars;
        let prefix_eq = EqPolynomial::<F>::evals(prefix_challenges, None);
        let mut ram_inc = jolt_poly::thread::unsafe_allocate_zero_vec(suffix_len);
        let mut rd_inc = jolt_poly::thread::unsafe_allocate_zero_vec(suffix_len);
        ram_inc
            .par_iter_mut()
            .zip(rd_inc.par_iter_mut())
            .enumerate()
            .for_each(|(suffix_index, (ram_out, rd_out))| {
                let mut ram_acc = F::zero();
                let mut rd_acc = F::zero();
                for (prefix_index, &prefix_weight) in prefix_eq.iter().enumerate().take(prefix_len)
                {
                    let index = (prefix_index << suffix_vars) | suffix_index;
                    let row = &rows[cycle_for_reversed_index(index, log_t)];
                    ram_acc += prefix_weight * F::from_i128(row.ram_increment);
                    rd_acc += prefix_weight * F::from_i128(row.rd_increment);
                }
                *ram_out = ram_acc;
                *rd_out = rd_acc;
            });

        let (_, r2_suffix) = r_cycle_stage2.split_at(prefix_vars);
        let (_, r4_suffix) = r_cycle_stage4.split_at(prefix_vars);
        let (_, s4_suffix) = s_cycle_stage4.split_at(prefix_vars);
        let (_, s5_suffix) = s_cycle_stage5.split_at(prefix_vars);
        let eq_r2 = EqPolynomial::<F>::evals(r2_suffix, Some(r2_prefix_scale));
        let eq_r4 = EqPolynomial::<F>::evals(r4_suffix, Some(r4_prefix_scale));
        let eq_s4 = EqPolynomial::<F>::evals(s4_suffix, Some(s4_prefix_scale));
        let eq_s5 = EqPolynomial::<F>::evals(s5_suffix, Some(s5_prefix_scale));
        let ram_coeff = eq_r2
            .into_iter()
            .zip(eq_r4)
            .map(|(left, right)| left + gamma * right)
            .collect::<Vec<_>>();
        let rd_coeff = eq_s4
            .into_iter()
            .zip(eq_s5)
            .map(|(left, right)| left + gamma * right)
            .collect::<Vec<_>>();

        Self {
            gamma_squared: gamma * gamma,
            ram_inc: Polynomial::new(ram_inc),
            rd_inc: Polynomial::new(rd_inc),
            ram_coeff: Polynomial::new(ram_coeff),
            rd_coeff: Polynomial::new(rd_coeff),
        }
    }

    fn evaluate_round(&self, previous_claim: F) -> UnivariatePoly<F> {
        let rows = self.ram_inc.len() / 2;
        let evals = (0..rows)
            .into_par_iter()
            .map(|row| {
                let ram_inc = poly_evals_at_0_and_2(&self.ram_inc, row, BindingOrder::HighToLow);
                let rd_inc = poly_evals_at_0_and_2(&self.rd_inc, row, BindingOrder::HighToLow);
                let ram_coeff =
                    poly_evals_at_0_and_2(&self.ram_coeff, row, BindingOrder::HighToLow);
                let rd_coeff = poly_evals_at_0_and_2(&self.rd_coeff, row, BindingOrder::HighToLow);
                std::array::from_fn(|point| {
                    ram_inc[point] * ram_coeff[point]
                        + self.gamma_squared * rd_inc[point] * rd_coeff[point]
                })
            })
            .reduce(|| [F::zero(); 2], sum_arrays::<F, 2>);
        UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, challenge: F) {
        self.ram_inc
            .bind_with_order(challenge, BindingOrder::HighToLow);
        self.rd_inc
            .bind_with_order(challenge, BindingOrder::HighToLow);
        self.ram_coeff
            .bind_with_order(challenge, BindingOrder::HighToLow);
        self.rd_coeff
            .bind_with_order(challenge, BindingOrder::HighToLow);
    }

    fn output_claims(&self) -> Result<SumcheckIncClaimReductionOutput<F>, BackendError> {
        Ok(SumcheckIncClaimReductionOutput {
            ram_inc: final_poly_claim(&self.ram_inc).ok_or_else(|| {
                BackendError::InvalidRequest {
                    backend: "cpu",
                    task: "increment claim-reduction output claims",
                    reason: "RAM increment polynomial is not fully bound".to_owned(),
                }
            })?,
            rd_inc: final_poly_claim(&self.rd_inc).ok_or_else(|| BackendError::InvalidRequest {
                backend: "cpu",
                task: "increment claim-reduction output claims",
                reason: "register increment polynomial is not fully bound".to_owned(),
            })?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct RaChunkSelector {
    shift: usize,
    mask: u128,
}

impl RaChunkSelector {
    fn new(index: usize, chunks: usize, chunk_bits: usize) -> Self {
        let remaining = chunks - index - 1;
        let shift = remaining * chunk_bits;
        let mask = (1u128 << chunk_bits) - 1;
        Self { shift, mask }
    }

    const fn chunk_usize(self, value: usize) -> usize {
        self.chunk_u128(value as u128)
    }

    const fn chunk_u128(self, value: u128) -> usize {
        ((value >> self.shift) & self.mask) as usize
    }
}

fn ra_product_sum_round<F: Field>(
    eq: &GruenSplitEqPolynomial<F>,
    groups: &[&[RaPolynomial<u8, F>]],
    coefficients: &[F],
    previous_claim: F,
) -> UnivariatePoly<F>
where
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    let product_degree = groups.first().map_or(0, |group| group.len());
    struct RaProductInner<F: Field> {
        values: Vec<<F as WithAccumulator>::Accumulator>,
        group_evals: Vec<F>,
    }
    let q_evals = eq.par_fold_out_in(
        || RaProductInner {
            values: vec![<F as WithAccumulator>::Accumulator::default(); product_degree],
            group_evals: vec![F::zero(); product_degree],
        },
        |inner, row, _x_in, e_in| {
            for (group, coefficient) in groups.iter().zip(coefficients) {
                evaluate_ra_product_on_gruen_domain(group, row, &mut inner.group_evals);
                let weighted_eq = e_in * *coefficient;
                for (inner_eval, product_eval) in inner.values.iter_mut().zip(&inner.group_evals) {
                    inner_eval.fmadd(weighted_eq, *product_eval);
                }
            }
        },
        |_x_out, e_out, mut inner| {
            for value in &mut inner.values {
                let inner_value = std::mem::take(value).reduce();
                value.fmadd(e_out, inner_value);
            }
            inner.values
        },
        |mut left, right| {
            left.iter_mut()
                .zip(right)
                .for_each(|(left, right)| left.merge(right));
            left
        },
    );
    let q_evals = q_evals
        .into_iter()
        .map(AdditiveAccumulator::reduce)
        .collect::<Vec<_>>();
    eq.gruen_poly_from_evals(&q_evals, previous_claim)
}

fn instruction_ra_product_sum_round<F: Field>(
    eq: &GruenSplitEqPolynomial<F>,
    ra_polys: &[RaPolynomial<u8, F>],
    virtual_polys: usize,
    committed_per_virtual: usize,
    previous_claim: F,
) -> UnivariatePoly<F>
where
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    match committed_per_virtual {
        4 => instruction_ra_product_sum_round_fixed::<F, 4>(
            eq,
            ra_polys,
            virtual_polys,
            previous_claim,
            field::eval_linear_product_d4_assign::<F>,
        ),
        8 => instruction_ra_product_sum_round_fixed::<F, 8>(
            eq,
            ra_polys,
            virtual_polys,
            previous_claim,
            field::eval_linear_product_d8_assign::<F>,
        ),
        16 => instruction_ra_product_sum_round_fixed::<F, 16>(
            eq,
            ra_polys,
            virtual_polys,
            previous_claim,
            field::eval_linear_product_d16_assign::<F>,
        ),
        _ => {
            let groups = ra_polys.chunks(committed_per_virtual).collect::<Vec<_>>();
            let coefficients = vec![F::one(); virtual_polys];
            ra_product_sum_round(eq, &groups, &coefficients, previous_claim)
        }
    }
}

fn instruction_ra_product_sum_round_fixed<F: Field, const D: usize>(
    eq: &GruenSplitEqPolynomial<F>,
    ra_polys: &[RaPolynomial<u8, F>],
    virtual_polys: usize,
    previous_claim: F,
    eval_product: fn(&[(F, F); D], &mut [F]),
) -> UnivariatePoly<F>
where
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    debug_assert_eq!(ra_polys.len(), virtual_polys * D);
    let q_evals = eq.par_fold_out_in(
        || array::from_fn(|_| <F as WithAccumulator>::Accumulator::default()),
        |inner, row, _x_in, e_in| {
            let mut row_sums = [F::zero(); D];
            for product in 0..virtual_polys {
                let base = product * D;
                let pairs =
                    array::from_fn(|offset| ra_low_to_high_pair(&ra_polys[base + offset], row));
                let mut product_evals = [F::zero(); D];
                eval_product(&pairs, &mut product_evals);
                for (sum, value) in row_sums.iter_mut().zip(product_evals) {
                    *sum += value;
                }
            }
            for (inner_eval, row_sum) in inner.iter_mut().zip(row_sums) {
                inner_eval.fmadd(e_in, row_sum);
            }
        },
        |_x_out, e_out, mut inner| {
            for value in &mut inner {
                let inner_value = std::mem::take(value).reduce();
                value.fmadd(e_out, inner_value);
            }
            inner
        },
        merge_accumulator_arrays::<F, D>,
    );
    let q_evals = q_evals.map(AdditiveAccumulator::reduce);
    eq.gruen_poly_from_evals(&q_evals, previous_claim)
}

fn ra_evals_over_integer_domain<F: Field>(
    polynomial: &RaPolynomial<u8, F>,
    row: usize,
    degree: usize,
) -> Vec<F> {
    ra_evals_over_integer_domain_with_order(polynomial, row, degree, BindingOrder::HighToLow)
}

fn ra_evals_over_integer_domain_with_order<F: Field>(
    polynomial: &RaPolynomial<u8, F>,
    row: usize,
    degree: usize,
    order: BindingOrder,
) -> Vec<F> {
    let (lo, hi) = match order {
        BindingOrder::HighToLow => {
            let mid = polynomial.len() / 2;
            (
                polynomial.get_bound_coeff(row),
                polynomial.get_bound_coeff(row + mid),
            )
        }
        BindingOrder::LowToHigh => (
            polynomial.get_bound_coeff(2 * row),
            polynomial.get_bound_coeff(2 * row + 1),
        ),
    };
    (0..=degree)
        .map(|point| linear_eval(lo, hi, F::from_u64(point as u64)))
        .collect()
}

fn stack_hint_points<F: Field>(degree: usize) -> [F; MAX_STACK_STAGE6_EVALS] {
    array::from_fn(|index| {
        if index == 0 {
            F::zero()
        } else if index < degree {
            F::from_u64((index + 1) as u64)
        } else {
            F::zero()
        }
    })
}

fn stack_extra_coefficients<F: Field>(
    extras: &[BytecodeExtraStageCycleState<F>],
    gamma_powers: &[F],
) -> [F; MAX_STACK_BYTECODE_EXTRA_STAGES] {
    let mut coefficients = [F::zero(); MAX_STACK_BYTECODE_EXTRA_STAGES];
    for (index, extra) in extras.iter().enumerate() {
        coefficients[index] = gamma_powers[extra.stage] * extra.bound_val;
    }
    coefficients
}

fn evaluate_ra_product_on_gruen_domain<F: Field>(
    group: &[RaPolynomial<u8, F>],
    row: usize,
    output: &mut [F],
) {
    debug_assert_eq!(group.len(), output.len());
    match group.len() {
        2 => {
            let pairs = std::array::from_fn(|index| ra_low_to_high_pair(&group[index], row));
            field::eval_linear_product_d2_assign(&pairs, output);
        }
        3 => {
            let pairs = std::array::from_fn(|index| ra_low_to_high_pair(&group[index], row));
            field::eval_linear_product_d3_assign(&pairs, output);
        }
        4 => {
            let pairs = std::array::from_fn(|index| ra_low_to_high_pair(&group[index], row));
            field::eval_linear_product_d4_assign(&pairs, output);
        }
        5 => {
            let pairs = std::array::from_fn(|index| ra_low_to_high_pair(&group[index], row));
            field::eval_linear_product_d5_assign(&pairs, output);
        }
        6 => {
            let pairs = std::array::from_fn(|index| ra_low_to_high_pair(&group[index], row));
            field::eval_linear_product_d6_assign(&pairs, output);
        }
        7 => {
            let pairs = std::array::from_fn(|index| ra_low_to_high_pair(&group[index], row));
            field::eval_linear_product_d7_assign(&pairs, output);
        }
        8 => {
            let pairs = std::array::from_fn(|index| ra_low_to_high_pair(&group[index], row));
            field::eval_linear_product_d8_assign(&pairs, output);
        }
        16 => {
            let pairs = std::array::from_fn(|index| ra_low_to_high_pair(&group[index], row));
            field::eval_linear_product_d16_assign(&pairs, output);
        }
        degree => {
            for (index, output) in output.iter_mut().enumerate() {
                let point = if index + 1 == degree {
                    None
                } else {
                    Some(F::from_u64((index + 1) as u64))
                };
                *output = group.iter().fold(F::one(), |acc, poly| {
                    let (low, high) = ra_low_to_high_pair(poly, row);
                    let value = point.map_or(high - low, |point| linear_eval(low, high, point));
                    acc * value
                });
            }
        }
    }
}

fn ra_high_to_low_pair<F: Field>(poly: &RaPolynomial<u8, F>, row: usize) -> (F, F) {
    let mid = poly.len() / 2;
    (poly.get_bound_coeff(row), poly.get_bound_coeff(row + mid))
}

fn ra_low_to_high_pair<F: Field>(poly: &RaPolynomial<u8, F>, row: usize) -> (F, F) {
    (
        poly.get_bound_coeff(2 * row),
        poly.get_bound_coeff(2 * row + 1),
    )
}

fn gruen_fold2<F, Values>(eq: &GruenSplitEqPolynomial<F>, values: Values) -> [F; 2]
where
    F: Field,
    Values: Fn(usize) -> [F; 2] + Sync + Send,
{
    eq.par_fold_out_in(
        || [F::zero(); 2],
        |inner, row, _x_in, e_in| {
            let values = values(row);
            inner[0] += e_in * values[0];
            inner[1] += e_in * values[1];
        },
        |_x_out, e_out, inner| [e_out * inner[0], e_out * inner[1]],
        sum_arrays::<F, 2>,
    )
}

#[derive(Clone, Debug)]
struct ExpandingTable<F: Field> {
    order: BindingOrder,
    values: Vec<F>,
    scratch: Vec<F>,
    len: usize,
}

impl<F: Field> ExpandingTable<F> {
    fn new(capacity: usize, order: BindingOrder) -> Self {
        Self {
            order,
            values: jolt_poly::thread::unsafe_allocate_zero_vec(capacity),
            scratch: jolt_poly::thread::unsafe_allocate_zero_vec(capacity),
            len: 0,
        }
    }

    fn reset(&mut self, value: F) {
        self.values[0] = value;
        self.len = 1;
    }

    fn update(&mut self, challenge: F) {
        match self.order {
            BindingOrder::LowToHigh => {
                let (left, right) = self.values.split_at_mut(self.len);
                left.par_iter_mut()
                    .zip(right.par_iter_mut())
                    .for_each(|(left, right)| {
                        *right = *left * challenge;
                        *left -= *right;
                    });
            }
            BindingOrder::HighToLow => {
                self.values[..self.len]
                    .par_iter()
                    .zip(self.scratch.par_chunks_mut(2))
                    .for_each(|(&value, dest)| {
                        let eval_1 = value * challenge;
                        dest[0] = value - eval_1;
                        dest[1] = eval_1;
                    });
                std::mem::swap(&mut self.values, &mut self.scratch);
            }
        }
        self.len *= 2;
    }

    fn clone_values(&self) -> Vec<F> {
        self.values[..self.len].to_vec()
    }
}

impl<F: Field> Index<usize> for ExpandingTable<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.len);
        &self.values[index]
    }
}

fn ra_cycle_indices(
    rows: &[SumcheckStage6RaRow],
    chunk_bits: usize,
    layout: RaFamilyLayout,
    reverse_log_t: Option<usize>,
) -> Vec<RaCycleIndices> {
    let instruction_selectors = (0..layout.instruction_chunks)
        .map(|index| RaChunkSelector::new(index, layout.instruction_chunks, chunk_bits))
        .collect::<Vec<_>>();
    let bytecode_selectors = (0..layout.bytecode_chunks)
        .map(|index| RaChunkSelector::new(index, layout.bytecode_chunks, chunk_bits))
        .collect::<Vec<_>>();
    let ram_selectors = (0..layout.ram_chunks)
        .map(|index| RaChunkSelector::new(index, layout.ram_chunks, chunk_bits))
        .collect::<Vec<_>>();

    let mut indices = vec![RaCycleIndices::default(); rows.len()];
    for (cycle, row) in rows.iter().enumerate() {
        let target = reverse_log_t.map_or(cycle, |log_t| reverse_bits(cycle, log_t));
        let mut row_indices = RaCycleIndices::default();
        for (index, selector) in instruction_selectors.iter().copied().enumerate() {
            row_indices.instruction[index] =
                selector.chunk_u128(row.instruction_lookup_index) as u8;
        }
        for (index, selector) in bytecode_selectors.iter().copied().enumerate() {
            row_indices.bytecode[index] = selector.chunk_usize(row.bytecode_index) as u8;
        }
        for (index, selector) in ram_selectors.iter().copied().enumerate() {
            row_indices.ram[index] = row
                .ram_address
                .map(|address| selector.chunk_usize(address) as u8);
        }
        indices[target] = row_indices;
    }
    indices
}

fn challenge_chunks<F: Field>(point: &[F], chunks: usize, chunk_bits: usize) -> Vec<Vec<F>> {
    let mut padded = Vec::with_capacity(chunks * chunk_bits);
    let padding = chunks * chunk_bits - point.len();
    padded.extend((0..padding).map(|_| F::zero()));
    padded.extend_from_slice(point);
    padded
        .chunks(chunk_bits)
        .map(<[F]>::to_vec)
        .collect::<Vec<_>>()
}

fn reverse_bits(mut value: usize, bits: usize) -> usize {
    let mut reversed = 0usize;
    for _ in 0..bits {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
    }
    reversed
}

fn reverse_index_table<F: Field>(values: Vec<F>, bits: usize) -> Vec<F> {
    let mut reversed = vec![F::zero(); values.len()];
    for (index, value) in values.into_iter().enumerate() {
        reversed[reverse_bits(index, bits)] = value;
    }
    reversed
}

fn reverse_cycle_table<F: Field>(values: Vec<F>, log_t: usize) -> Vec<F> {
    reverse_index_table(values, log_t)
}

fn checked_chunks(
    backend: &'static str,
    task: &'static str,
    log_k: usize,
    chunk_bits: usize,
) -> Result<usize, BackendError> {
    if chunk_bits == 0 || chunk_bits > 8 {
        return invalid(
            backend,
            task,
            format!("RA chunk bits must be in 1..=8, got {chunk_bits}"),
        );
    }
    Ok(log_k.div_ceil(chunk_bits))
}

fn poly_eval_pair<F: Field>(polynomial: &Polynomial<F>, row: usize, order: BindingOrder) -> [F; 2] {
    let (lo, hi) = polynomial.sumcheck_eval_pair(row, order);
    [lo, hi]
}

fn poly_evals_at_0_and_2<F: Field>(
    polynomial: &Polynomial<F>,
    row: usize,
    order: BindingOrder,
) -> [F; 2] {
    let [lo, hi] = poly_eval_pair(polynomial, row, order);
    [lo, hi + hi - lo]
}

fn linear_eval<F: Field>(lo: F, hi: F, point: F) -> F {
    lo + point * (hi - lo)
}

fn prefix_suffix_inc_q<F: Field>(
    rows: &[SumcheckStage6IncRow],
    log_t: usize,
    prefix_vars: usize,
    suffix_vars: usize,
    suffix_point: &[F],
    increment: impl Fn(&SumcheckStage6IncRow) -> F + Sync,
) -> Vec<F> {
    let prefix_len = 1usize << prefix_vars;
    let suffix_len = 1usize << suffix_vars;
    let suffix_eq = EqPolynomial::<F>::evals(suffix_point, None);
    let mut q = jolt_poly::thread::unsafe_allocate_zero_vec(prefix_len);
    q.par_iter_mut()
        .enumerate()
        .for_each(|(prefix_index, output)| {
            let mut acc = F::zero();
            for (suffix_index, &suffix_weight) in suffix_eq.iter().enumerate().take(suffix_len) {
                let index = (prefix_index << suffix_vars) | suffix_index;
                acc += suffix_weight * increment(&rows[cycle_for_reversed_index(index, log_t)]);
            }
            *output = acc;
        });
    q
}

fn cycle_for_reversed_index(index: usize, log_t: usize) -> usize {
    index.reverse_bits() >> (usize::BITS as usize - log_t)
}

fn final_poly_claim<F: Field>(polynomial: &Polynomial<F>) -> Option<F> {
    (polynomial.len() == 1).then(|| polynomial.evaluations()[0])
}

fn merge_vec<F: Field>(left: &mut [F], right: Vec<F>) {
    left.iter_mut()
        .zip(right)
        .for_each(|(left, right)| *left += right);
}

fn sum_arrays<F: Field, const N: usize>(left: [F; N], right: [F; N]) -> [F; N] {
    std::array::from_fn(|index| left[index] + right[index])
}

fn merge_accumulator_arrays<F, const N: usize>(
    mut left: [<F as WithAccumulator>::Accumulator; N],
    right: [<F as WithAccumulator>::Accumulator; N],
) -> [<F as WithAccumulator>::Accumulator; N]
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    left.iter_mut()
        .zip(right)
        .for_each(|(left, right)| left.merge(right));
    left
}

fn bytecode_pc_pushforwards<F: Field, const N: usize>(
    r_cycles: &[Vec<F>; N],
    pc_indices: &[usize],
    log_t: usize,
    log_k: usize,
) -> [Vec<F>; N] {
    debug_assert!(N > 0);
    let k = 1usize << log_k;
    let pc_addresses = pc_indices
        .iter()
        .map(|&pc| reverse_bits(pc, log_k))
        .collect::<Vec<_>>();
    let lo_bits = log_t / 2;
    let hi_bits = log_t - lo_bits;
    let in_len = 1usize << lo_bits;
    let out_len = 1usize << hi_bits;
    let (eq_hi, eq_lo): ([Vec<F>; N], [Vec<F>; N]) = rayon::join(
        || {
            r_cycles
                .each_ref()
                .map(|r_cycle| EqPolynomial::<F>::evals(&r_cycle[..hi_bits], None))
        },
        || {
            r_cycles
                .each_ref()
                .map(|r_cycle| EqPolynomial::<F>::evals(&r_cycle[hi_bits..], None))
        },
    );
    let chunk_size = out_len.div_ceil(rayon::current_num_threads()).max(1);

    eq_hi[0]
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let mut partial: [Vec<F>; N] =
                array::from_fn(|_| jolt_poly::thread::unsafe_allocate_zero_vec(k));
            let mut inner: [Vec<F>; N] =
                array::from_fn(|_| jolt_poly::thread::unsafe_allocate_zero_vec(k));
            let mut touched = Vec::with_capacity(in_len.min(k));
            let mut touched_stamps = vec![0u8; k];
            let mut stamp = 0u8;
            let chunk_start = chunk_index * chunk_size;

            for local_index in 0..chunk.len() {
                for &address_index in &touched {
                    for stage_inner in &mut inner {
                        stage_inner[address_index] = F::zero();
                    }
                }
                touched.clear();
                stamp = stamp.wrapping_add(1);
                if stamp == 0 {
                    touched_stamps.fill(0);
                    stamp = 1;
                }

                let cycle_hi = chunk_start + local_index;
                let cycle_base = cycle_hi * in_len;
                for (cycle_lo, _) in eq_lo[0].iter().enumerate().take(in_len) {
                    let cycle = cycle_base + cycle_lo;
                    if cycle >= pc_indices.len() {
                        break;
                    }
                    let address_index = pc_addresses[cycle];
                    if touched_stamps[address_index] != stamp {
                        touched_stamps[address_index] = stamp;
                        touched.push(address_index);
                    }
                    for (stage_inner, stage_eq_lo) in inner.iter_mut().zip(eq_lo.iter()) {
                        stage_inner[address_index] += stage_eq_lo[cycle_lo];
                    }
                }

                for &address_index in &touched {
                    for ((stage_partial, stage_inner), stage_eq_hi) in
                        partial.iter_mut().zip(inner.iter()).zip(eq_hi.iter())
                    {
                        stage_partial[address_index] +=
                            stage_eq_hi[cycle_hi] * stage_inner[address_index];
                    }
                }
            }

            partial
        })
        .reduce(
            || array::from_fn(|_| jolt_poly::thread::unsafe_allocate_zero_vec(k)),
            |mut left, right| {
                for (left_stage, right_stage) in left.iter_mut().zip(right.iter()) {
                    left_stage
                        .iter_mut()
                        .zip(right_stage.iter())
                        .for_each(|(left, right)| *left += *right);
                }
                left
            },
        )
}

fn bytecode_pc_pushforward<F: Field>(
    r_cycle: &[F],
    pc_indices: &[usize],
    log_t: usize,
    log_k: usize,
) -> Vec<F> {
    let [values] = bytecode_pc_pushforwards(&[r_cycle.to_vec()], pc_indices, log_t, log_k);
    values
}

fn materialize_extra_bytecode_stage<F: Field>(
    backend: &'static str,
    task: &'static str,
    extra: &SumcheckBytecodeReadRafExtraStageValues<F>,
    request: &SumcheckBytecodeReadRafStateRequest<F>,
) -> Result<BytecodeExtraStageState<F>, BackendError> {
    if extra.stage >= BYTECODE_STAGES {
        return invalid(
            backend,
            task,
            format!(
                "extra bytecode read-RAF stage {} is outside {BYTECODE_STAGES} stages",
                extra.stage
            ),
        );
    }
    validate_len(
        backend,
        task,
        "extra bytecode stage values",
        extra.bytecode_stage_values.len(),
        1usize << request.log_k,
    )?;
    validate_len(
        backend,
        task,
        "extra bytecode stage cycle",
        extra.r_cycle.len(),
        request.log_t,
    )?;
    let k = 1usize << request.log_k;
    let f_values = bytecode_pc_pushforward(
        &extra.r_cycle,
        &request.pc_indices,
        request.log_t,
        request.log_k,
    );
    let mut val_values = jolt_poly::thread::unsafe_allocate_zero_vec(k);
    for (row, value) in extra.bytecode_stage_values.iter().copied().enumerate() {
        val_values[reverse_bits(row, request.log_k)] = value;
    }
    Ok(BytecodeExtraStageState {
        stage: extra.stage,
        f: Polynomial::new(f_values),
        val: Polynomial::new(val_values),
        r_cycle: extra.r_cycle.clone(),
    })
}

fn validate_bytecode_request<F: Field>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckBytecodeReadRafStateRequest<F>,
) -> Result<(), BackendError> {
    validate_len(
        backend,
        task,
        "bytecode rows",
        request.bytecode_stage_values.len(),
        1usize << request.log_k,
    )?;
    validate_len(
        backend,
        task,
        "bytecode PC rows",
        request.pc_indices.len(),
        1usize << request.log_t,
    )?;
    for &pc in &request.pc_indices {
        if pc >= request.bytecode_stage_values.len() {
            return invalid(
                backend,
                task,
                format!("bytecode PC index {pc} is out of range"),
            );
        }
    }
    validate_len(
        backend,
        task,
        "bytecode gamma powers",
        request.gamma_powers.len(),
        8,
    )?;
    for extra in &request.extra_stage_values {
        if extra.stage >= BYTECODE_STAGES {
            return invalid(
                backend,
                task,
                format!(
                    "extra bytecode read-RAF stage {} is outside {BYTECODE_STAGES} stages",
                    extra.stage
                ),
            );
        }
        validate_len(
            backend,
            task,
            "extra bytecode stage values",
            extra.bytecode_stage_values.len(),
            1usize << request.log_k,
        )?;
        validate_len(
            backend,
            task,
            "extra bytecode stage cycle",
            extra.r_cycle.len(),
            request.log_t,
        )?;
    }
    let _ = checked_chunks(backend, task, request.log_k, request.chunk_bits)?;
    Ok(())
}

fn validate_booleanity_request<F: Field>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckBooleanityStateRequest<F>,
) -> Result<(), BackendError> {
    validate_len(
        backend,
        task,
        "booleanity rows",
        request.rows.len(),
        1usize << request.log_t,
    )?;
    validate_len(
        backend,
        task,
        "booleanity address point",
        request.r_address.len(),
        request.chunk_bits,
    )?;
    validate_len(
        backend,
        task,
        "booleanity cycle point",
        request.r_cycle.len(),
        request.log_t,
    )?;
    let _ = checked_chunks(backend, task, request.chunk_bits, request.chunk_bits)?;
    Ok(())
}

fn validate_hamming_request<F: Field>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckRamHammingBooleanityStateRequest<F>,
) -> Result<(), BackendError> {
    validate_len(
        backend,
        task,
        "RAM hamming rows",
        request.hamming_weight.len(),
        1usize << request.log_t,
    )?;
    validate_len(
        backend,
        task,
        "RAM hamming cycle point",
        request.r_cycle.len(),
        request.log_t,
    )
}

fn validate_ram_ra_virtual_request<F: Field>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckRamRaVirtualizationStateRequest<F>,
) -> Result<(), BackendError> {
    validate_len(
        backend,
        task,
        "RAM RA virtualization rows",
        request.rows.len(),
        1usize << request.log_t,
    )?;
    validate_len(
        backend,
        task,
        "RAM RA virtualization cycle point",
        request.r_cycle.len(),
        request.log_t,
    )?;
    for point in &request.r_address_chunks {
        validate_len(
            backend,
            task,
            "RAM RA virtualization address chunk",
            point.len(),
            request.chunk_bits,
        )?;
    }
    Ok(())
}

fn validate_instruction_ra_virtual_request<F: Field>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckInstructionRaVirtualizationStateRequest<F>,
) -> Result<(), BackendError> {
    validate_len(
        backend,
        task,
        "instruction RA virtualization rows",
        request.rows.len(),
        1usize << request.log_t,
    )?;
    validate_len(
        backend,
        task,
        "instruction RA virtualization cycle point",
        request.r_cycle.len(),
        request.log_t,
    )?;
    validate_len(
        backend,
        task,
        "instruction RA virtualization gamma powers",
        request.gamma_powers.len(),
        request.virtual_polys,
    )?;
    if request.r_address_chunks.len() != request.virtual_polys * request.committed_per_virtual {
        return invalid(
            backend,
            task,
            format!(
                "instruction RA virtualization has {} address chunks, expected {}",
                request.r_address_chunks.len(),
                request.virtual_polys * request.committed_per_virtual
            ),
        );
    }
    Ok(())
}

fn validate_inc_request<F: Field>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckIncClaimReductionStateRequest<F>,
) -> Result<(), BackendError> {
    validate_len(
        backend,
        task,
        "increment rows",
        request.rows.len(),
        1usize << request.log_t,
    )?;
    validate_len(
        backend,
        task,
        "increment stage2 cycle point",
        request.r_cycle_stage2.len(),
        request.log_t,
    )?;
    validate_len(
        backend,
        task,
        "increment stage4 RAM cycle point",
        request.r_cycle_stage4.len(),
        request.log_t,
    )?;
    validate_len(
        backend,
        task,
        "increment stage4 register cycle point",
        request.s_cycle_stage4.len(),
        request.log_t,
    )?;
    validate_len(
        backend,
        task,
        "increment stage5 register cycle point",
        request.s_cycle_stage5.len(),
        request.log_t,
    )
}

fn validate_len(
    backend: &'static str,
    task: &'static str,
    label: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), BackendError> {
    if actual != expected {
        return invalid(
            backend,
            task,
            format!("{label} has {actual} values, expected {expected}"),
        );
    }
    Ok(())
}

fn invalid<T>(
    backend: &'static str,
    task: &'static str,
    reason: impl std::fmt::Display,
) -> Result<T, BackendError> {
    Err(BackendError::InvalidRequest {
        backend,
        task,
        reason: reason.to_string(),
    })
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};

    use super::*;

    #[test]
    fn bytecode_pc_pushforward_split_eq_matches_dense_reference() {
        let log_t = 5;
        let log_k = 3;
        let pc_indices = (0..1usize << log_t)
            .map(|cycle| (5 * cycle + 3) % (1usize << log_k))
            .collect::<Vec<_>>();
        let r_cycles = [
            [3, 5, 7, 11, 13].map(Fr::from_u64).to_vec(),
            [17, 19, 23, 29, 31].map(Fr::from_u64).to_vec(),
        ];

        let actual = bytecode_pc_pushforwards(&r_cycles, &pc_indices, log_t, log_k);
        let expected = r_cycles.each_ref().map(|r_cycle| {
            let cycle_eq = EqPolynomial::<Fr>::evals(r_cycle, None);
            let mut values = jolt_poly::thread::unsafe_allocate_zero_vec(1usize << log_k);
            for (cycle, &pc) in pc_indices.iter().enumerate() {
                values[reverse_bits(pc, log_k)] += cycle_eq[cycle];
            }
            values
        });

        assert_eq!(actual, expected);
    }

    #[test]
    fn inc_claim_reduction_prefix_suffix_matches_dense_reference() {
        let log_t = 4;
        let gamma = Fr::from_u64(7);
        let r_cycle_stage2 = [3, 5, 11, 17].map(Fr::from_u64).to_vec();
        let r_cycle_stage4 = [19, 23, 29, 31].map(Fr::from_u64).to_vec();
        let s_cycle_stage4 = [37, 41, 43, 47].map(Fr::from_u64).to_vec();
        let s_cycle_stage5 = [53, 59, 61, 67].map(Fr::from_u64).to_vec();
        let rows = (0..1usize << log_t)
            .map(|cycle| SumcheckStage6IncRow {
                ram_increment: (cycle as i128 % 5) - 2,
                rd_increment: ((3 * cycle) as i128 % 7) - 3,
            })
            .collect::<Vec<_>>();
        let input_claim = dense_inc_claim(
            &rows,
            &r_cycle_stage2,
            &r_cycle_stage4,
            &s_cycle_stage4,
            &s_cycle_stage5,
            gamma,
            log_t,
        );
        let request = SumcheckIncClaimReductionStateRequest::new(
            "test.increment_claim_reduction",
            rows.clone(),
            r_cycle_stage2.clone(),
            r_cycle_stage4.clone(),
            s_cycle_stage4.clone(),
            s_cycle_stage5.clone(),
            gamma,
            input_claim,
            log_t,
        );
        let mut state = IncClaimReductionState::new("cpu", "test", &request).unwrap();
        let challenges = [71, 73, 79, 83].map(Fr::from_u64);
        let mut claim = input_claim;
        for challenge in challenges {
            let round = state.evaluate_round(claim).unwrap();
            assert_eq!(
                round.evaluate(Fr::from_u64(0)) + round.evaluate(Fr::from_u64(1)),
                claim
            );
            claim = round.evaluate(challenge);
            state.bind(challenge);
        }

        let output = state.output_claims().unwrap();
        assert_eq!(
            output.ram_inc,
            dense_bound_inc(&rows, log_t, &challenges, |row| {
                Fr::from_i128(row.ram_increment)
            })
        );
        assert_eq!(
            output.rd_inc,
            dense_bound_inc(&rows, log_t, &challenges, |row| {
                Fr::from_i128(row.rd_increment)
            })
        );

        let ram_coeff =
            dense_bound_coeff(&r_cycle_stage2, &r_cycle_stage4, gamma, log_t, &challenges);
        let rd_coeff =
            dense_bound_coeff(&s_cycle_stage4, &s_cycle_stage5, gamma, log_t, &challenges);
        assert_eq!(
            claim,
            output.ram_inc * ram_coeff + gamma * gamma * output.rd_inc * rd_coeff
        );
    }

    fn dense_inc_claim(
        rows: &[SumcheckStage6IncRow],
        r_cycle_stage2: &[Fr],
        r_cycle_stage4: &[Fr],
        s_cycle_stage4: &[Fr],
        s_cycle_stage5: &[Fr],
        gamma: Fr,
        log_t: usize,
    ) -> Fr {
        let gamma_squared = gamma * gamma;
        let ram_inc = reverse_cycle_table(
            rows.iter()
                .map(|row| Fr::from_i128(row.ram_increment))
                .collect(),
            log_t,
        );
        let rd_inc = reverse_cycle_table(
            rows.iter()
                .map(|row| Fr::from_i128(row.rd_increment))
                .collect(),
            log_t,
        );
        let ram_coeff = dense_coeff(r_cycle_stage2, r_cycle_stage4, gamma);
        let rd_coeff = dense_coeff(s_cycle_stage4, s_cycle_stage5, gamma);
        ram_inc
            .iter()
            .zip(ram_coeff)
            .map(|(&inc, coeff)| inc * coeff)
            .sum::<Fr>()
            + gamma_squared
                * rd_inc
                    .iter()
                    .zip(rd_coeff)
                    .map(|(&inc, coeff)| inc * coeff)
                    .sum::<Fr>()
    }

    fn dense_bound_inc(
        rows: &[SumcheckStage6IncRow],
        log_t: usize,
        challenges: &[Fr],
        increment: impl Fn(&SumcheckStage6IncRow) -> Fr,
    ) -> Fr {
        let mut polynomial = Polynomial::new(reverse_cycle_table(
            rows.iter().map(increment).collect(),
            log_t,
        ));
        for &challenge in challenges {
            polynomial.bind_with_order(challenge, BindingOrder::HighToLow);
        }
        polynomial.evaluations()[0]
    }

    fn dense_bound_coeff(
        left_point: &[Fr],
        right_point: &[Fr],
        gamma: Fr,
        log_t: usize,
        challenges: &[Fr],
    ) -> Fr {
        let mut polynomial = Polynomial::new(dense_coeff(left_point, right_point, gamma));
        assert_eq!(polynomial.len(), 1usize << log_t);
        for &challenge in challenges {
            polynomial.bind_with_order(challenge, BindingOrder::HighToLow);
        }
        polynomial.evaluations()[0]
    }

    fn dense_coeff(left_point: &[Fr], right_point: &[Fr], gamma: Fr) -> Vec<Fr> {
        EqPolynomial::<Fr>::evals(left_point, None)
            .into_iter()
            .zip(EqPolynomial::<Fr>::evals(right_point, None))
            .map(|(left, right)| left + gamma * right)
            .collect()
    }
}
