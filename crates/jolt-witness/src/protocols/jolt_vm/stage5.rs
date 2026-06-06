use jolt_field::Field;
use jolt_lookup_tables::{InstructionLookupTable, LookupTableKind};
use jolt_poly::{
    try_eq_mle, BindingOrder, IdentityPolynomial, MultilinearEvaluation, OperandPolynomial,
    OperandSide, Polynomial,
};
use jolt_program::{
    execution::{TraceRow, TraceSource},
    lookup::instruction_lookup_index,
};
use jolt_riscv::{Flags, InterleavedBitsMarker, JoltInstruction};
use rayon::prelude::*;

use super::{TraceBackedJoltVmWitness, JOLT_VM_NAMESPACE, RV64_LOOKUP_ADDRESS_BITS, RV64_XLEN};
use crate::WitnessError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5InstructionReadRafConfig<F: Field> {
    pub log_t: usize,
    pub address_bits: usize,
    pub chunk_bits: usize,
    pub fixed_cycle_point: Vec<F>,
    pub gamma: F,
}

impl<F: Field> Stage5InstructionReadRafConfig<F> {
    pub const fn new(
        log_t: usize,
        address_bits: usize,
        chunk_bits: usize,
        fixed_cycle_point: Vec<F>,
        gamma: F,
    ) -> Self {
        Self {
            log_t,
            address_bits,
            chunk_bits,
            fixed_cycle_point,
            gamma,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5InstructionReadRafOutputClaims<F: Field> {
    pub table_flags: Vec<F>,
    pub address_selectors: Vec<F>,
    pub raf_flag: F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5InstructionReadRafRow {
    pub lookup_index: u128,
    pub table_index: Option<usize>,
    pub interleaved_operands: bool,
}

pub trait JoltVmStage5InstructionReadRafRows {
    fn stage5_instruction_read_raf_rows(
        &self,
        log_t: usize,
    ) -> Result<Vec<Stage5InstructionReadRafRow>, WitnessError>;
}

pub trait JoltVmStage5InstructionReadRafWitness<F: Field> {
    fn stage5_instruction_read_raf_context(
        &self,
        config: Stage5InstructionReadRafConfig<F>,
    ) -> Result<Stage5InstructionReadRafContext<F>, WitnessError>;
}

#[derive(Clone, Debug)]
pub struct Stage5InstructionReadRafContext<F: Field> {
    config: Stage5InstructionReadRafConfig<F>,
    gamma2: F,
    cycles: Vec<Stage5InstructionCycle>,
    cycle_weights: Vec<F>,
    address_prefix_weights: Vec<F>,
    bound_address: Vec<F>,
    cycle_polys: Option<Stage5InstructionCyclePolys<F>>,
}

#[derive(Clone, Copy, Debug)]
struct Stage5InstructionCycle {
    address: u128,
    table: Option<LookupTableKind<RV64_XLEN>>,
    interleaved_operands: bool,
}

#[derive(Clone, Debug)]
struct Stage5InstructionCyclePolys<F: Field> {
    eq_cycle: Polynomial<F>,
    combined_value: Polynomial<F>,
    address_selectors: Vec<Polynomial<F>>,
}

impl<F: Field> Stage5InstructionReadRafContext<F> {
    pub fn round_sum(&self, round: usize, point: F) -> Result<F, WitnessError> {
        if round < self.config.address_bits {
            return self.address_round_sum(round, point);
        }
        self.cycle_round_sum(point)
    }

    pub fn bind(&mut self, round: usize, challenge: F) -> Result<(), WitnessError> {
        if round < self.config.address_bits {
            if self.bound_address.len() != round {
                return Err(invalid_data(format!(
                    "Stage 5 instruction address round {round} has {} bound variables",
                    self.bound_address.len()
                )));
            }
            for (cycle, prefix_weight) in self
                .cycles
                .iter()
                .zip(self.address_prefix_weights.iter_mut())
            {
                let bit = address_bit(cycle.address, self.config.address_bits, round)?;
                *prefix_weight *= if bit { challenge } else { F::one() - challenge };
            }
            self.bound_address.push(challenge);
            if self.bound_address.len() == self.config.address_bits {
                self.build_cycle_polynomial()?;
            }
            return Ok(());
        }

        let Some(polys) = &mut self.cycle_polys else {
            return Err(invalid_data(
                "Stage 5 instruction cycle polynomials were not initialized",
            ));
        };
        polys.bind(challenge);
        Ok(())
    }

    pub fn expected_output_claim(
        &self,
        claims: &Stage5InstructionReadRafOutputClaims<F>,
        r_address: &[F],
        r_cycle: &[F],
    ) -> Result<F, WitnessError> {
        validate_len(
            "Stage 5 instruction output address point",
            r_address.len(),
            self.config.address_bits,
        )?;
        validate_len(
            "Stage 5 instruction output cycle point",
            r_cycle.len(),
            self.config.log_t,
        )?;
        validate_len(
            "Stage 5 instruction table flag claim count",
            claims.table_flags.len(),
            LookupTableKind::<RV64_XLEN>::COUNT,
        )?;
        let expected_selectors = self
            .config
            .address_bits
            .checked_div(self.config.chunk_bits)
            .ok_or_else(|| invalid_data("Stage 5 instruction chunk bit width must be nonzero"))?;
        validate_len(
            "Stage 5 instruction address selector claim count",
            claims.address_selectors.len(),
            expected_selectors,
        )?;

        let eq_cycle = try_eq_mle(&self.config.fixed_cycle_point, r_cycle)
            .map_err(|error| invalid_data(error.to_string()))?;
        let table_value = LookupTableKind::<RV64_XLEN>::iter()
            .zip(&claims.table_flags)
            .map(|(table, &claim)| table.evaluate_mle::<F, F>(r_address) * claim)
            .sum::<F>();
        let selector_product = claims.address_selectors.iter().copied().product::<F>();

        let left_operand =
            OperandPolynomial::new(self.config.address_bits, OperandSide::Left).evaluate(r_address);
        let right_operand = OperandPolynomial::new(self.config.address_bits, OperandSide::Right)
            .evaluate(r_address);
        let identity = IdentityPolynomial::new(self.config.address_bits).evaluate(r_address);
        let constant = self.config.gamma * left_operand + self.gamma2 * right_operand;
        let raf_coeff = self.gamma2 * identity - constant;

        Ok(eq_cycle * selector_product * (table_value + constant + raf_coeff * claims.raf_flag))
    }

    pub fn final_claim(&self) -> Result<F, WitnessError> {
        let Some(polys) = &self.cycle_polys else {
            return Err(invalid_data(
                "Stage 5 instruction cycle polynomials were not initialized",
            ));
        };
        polys.final_claim()
    }

    fn address_round_sum(&self, round: usize, point: F) -> Result<F, WitnessError> {
        if self.bound_address.len() != round {
            return Err(invalid_data(format!(
                "Stage 5 instruction address round {round} has {} bound variables",
                self.bound_address.len()
            )));
        }

        let chunk_size = self
            .cycles
            .len()
            .div_ceil(rayon::current_num_threads())
            .max(1);
        self.cycles
            .par_chunks(chunk_size)
            .zip(self.cycle_weights.par_chunks(chunk_size))
            .zip(self.address_prefix_weights.par_chunks(chunk_size))
            .map(
                |((cycles, cycle_weights), address_prefix_weights)| -> Result<F, WitnessError> {
                    let mut address_point = Vec::with_capacity(self.config.address_bits);
                    address_point.extend_from_slice(&self.bound_address);
                    address_point.push(point);
                    address_point.resize(self.config.address_bits, F::zero());

                    let mut partial = F::zero();
                    for ((cycle, &cycle_weight), &prefix) in
                        cycles.iter().zip(cycle_weights).zip(address_prefix_weights)
                    {
                        let bit = address_bit(cycle.address, self.config.address_bits, round)?;
                        let current = if bit { point } else { F::one() - point };
                        write_address_suffix(
                            &mut address_point,
                            cycle.address,
                            self.config.address_bits,
                            round + 1,
                        )?;
                        let value = self.combined_value(cycle, &address_point);
                        partial += cycle_weight * prefix * current * value;
                    }
                    Ok(partial)
                },
            )
            .try_reduce(F::zero, |left, right| Ok(left + right))
    }

    fn cycle_round_sum(&self, point: F) -> Result<F, WitnessError> {
        let Some(polys) = &self.cycle_polys else {
            return Err(invalid_data(
                "Stage 5 instruction cycle polynomials were not initialized",
            ));
        };
        polys.round_sum(point)
    }

    fn build_cycle_polynomial(&mut self) -> Result<(), WitnessError> {
        validate_len(
            "Stage 5 instruction bound address",
            self.bound_address.len(),
            self.config.address_bits,
        )?;
        let selector_count = self
            .config
            .address_bits
            .checked_div(self.config.chunk_bits)
            .ok_or_else(|| invalid_data("Stage 5 instruction chunk bit width must be nonzero"))?;
        let mut combined_values = Vec::with_capacity(self.cycles.len());
        let mut selector_values = (0..selector_count)
            .map(|_| Vec::with_capacity(self.cycles.len()))
            .collect::<Vec<_>>();

        for cycle in &self.cycles {
            combined_values.push(self.combined_value(cycle, &self.bound_address));
            for (chunk_index, values) in selector_values.iter_mut().enumerate() {
                let start = chunk_index * self.config.chunk_bits;
                let end = start + self.config.chunk_bits;
                values.push(bound_address_selector_at(
                    cycle.address,
                    self.config.address_bits,
                    start,
                    &self.bound_address[start..end],
                )?);
            }
        }

        self.cycle_polys = Some(Stage5InstructionCyclePolys {
            eq_cycle: Polynomial::from(self.cycle_weights.clone()),
            combined_value: Polynomial::from(combined_values),
            address_selectors: selector_values.into_iter().map(Polynomial::from).collect(),
        });
        Ok(())
    }

    fn combined_value(&self, cycle: &Stage5InstructionCycle, address_point: &[F]) -> F {
        let table_value = cycle
            .table
            .map_or_else(F::zero, |table| table.evaluate_mle::<F, F>(address_point));
        let left_operand = OperandPolynomial::new(self.config.address_bits, OperandSide::Left)
            .evaluate(address_point);
        let right_operand = OperandPolynomial::new(self.config.address_bits, OperandSide::Right)
            .evaluate(address_point);
        let identity = IdentityPolynomial::new(self.config.address_bits).evaluate(address_point);

        let raf_value = if cycle.interleaved_operands {
            self.config.gamma * left_operand + self.gamma2 * right_operand
        } else {
            self.gamma2 * identity
        };
        table_value + raf_value
    }
}

impl<F: Field> Stage5InstructionCyclePolys<F> {
    fn round_sum(&self, point: F) -> Result<F, WitnessError> {
        let terms = self.eq_cycle.len() / 2;
        Ok((0..terms)
            .map(|index| {
                let mut product = multilinear_round_eval(&self.eq_cycle, index, point)
                    * multilinear_round_eval(&self.combined_value, index, point);
                for selector in &self.address_selectors {
                    product *= multilinear_round_eval(selector, index, point);
                }
                product
            })
            .sum())
    }

    fn bind(&mut self, challenge: F) {
        self.eq_cycle
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.combined_value
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        for selector in &mut self.address_selectors {
            selector.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
    }

    fn final_claim(&self) -> Result<F, WitnessError> {
        validate_len(
            "Stage 5 instruction final equality polynomial",
            self.eq_cycle.len(),
            1,
        )?;
        validate_len(
            "Stage 5 instruction final combined value polynomial",
            self.combined_value.len(),
            1,
        )?;
        let mut product = self.eq_cycle.evals()[0] * self.combined_value.evals()[0];
        for selector in &self.address_selectors {
            validate_len(
                "Stage 5 instruction final address selector polynomial",
                selector.len(),
                1,
            )?;
            product *= selector.evals()[0];
        }
        Ok(product)
    }
}

impl<F, T> JoltVmStage5InstructionReadRafWitness<F> for TraceBackedJoltVmWitness<'_, T>
where
    F: Field,
    T: TraceSource + Clone,
{
    fn stage5_instruction_read_raf_context(
        &self,
        config: Stage5InstructionReadRafConfig<F>,
    ) -> Result<Stage5InstructionReadRafContext<F>, WitnessError> {
        validate_len(
            "Stage 5 instruction fixed cycle point",
            config.fixed_cycle_point.len(),
            config.log_t,
        )?;
        if config.log_t != self.config.log_t {
            return Err(invalid_dimensions(format!(
                "Stage 5 instruction config has log_t {}, witness has {}",
                config.log_t, self.config.log_t
            )));
        }
        if config.address_bits != RV64_LOOKUP_ADDRESS_BITS {
            return Err(invalid_dimensions(format!(
                "Stage 5 instruction address width is {}, expected {RV64_LOOKUP_ADDRESS_BITS}",
                config.address_bits
            )));
        }
        if config.chunk_bits == 0 || !config.address_bits.is_multiple_of(config.chunk_bits) {
            return Err(invalid_dimensions(format!(
                "Stage 5 instruction address width {} is not divisible by chunk width {}",
                config.address_bits, config.chunk_bits
            )));
        }

        let cycles = super::checked_pow2(config.log_t)?;
        let mut trace = self.trace.trace.clone();
        let cycle_data = (0..cycles)
            .map(|_| {
                let row = trace.next_row();
                stage5_instruction_cycle(row.as_ref())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let cycle_weights =
            jolt_poly::EqPolynomial::new(config.fixed_cycle_point.clone()).evaluations();
        validate_len(
            "Stage 5 instruction cycle equality table",
            cycle_weights.len(),
            cycles,
        )?;

        Ok(Stage5InstructionReadRafContext {
            gamma2: config.gamma * config.gamma,
            config,
            cycles: cycle_data,
            cycle_weights,
            address_prefix_weights: vec![F::one(); cycles],
            bound_address: Vec::new(),
            cycle_polys: None,
        })
    }
}

impl<T> JoltVmStage5InstructionReadRafRows for TraceBackedJoltVmWitness<'_, T>
where
    T: TraceSource + Clone,
{
    fn stage5_instruction_read_raf_rows(
        &self,
        log_t: usize,
    ) -> Result<Vec<Stage5InstructionReadRafRow>, WitnessError> {
        if log_t != self.config.log_t {
            return Err(invalid_dimensions(format!(
                "Stage 5 instruction rows requested log_t {log_t}, witness has {}",
                self.config.log_t
            )));
        }
        let cycles = super::checked_pow2(log_t)?;
        let mut trace = self.trace.trace.clone();
        (0..cycles)
            .map(|_| {
                let row = trace.next_row();
                let cycle = stage5_instruction_cycle(row.as_ref())?;
                Ok(Stage5InstructionReadRafRow {
                    lookup_index: cycle.address,
                    table_index: cycle.table.map(|table| table.index()),
                    interleaved_operands: cycle.interleaved_operands,
                })
            })
            .collect()
    }
}

fn stage5_instruction_cycle(
    row: Option<&TraceRow>,
) -> Result<Stage5InstructionCycle, WitnessError> {
    let address = match row {
        Some(row) => instruction_lookup_index::<RV64_XLEN>(row).map_err(|error| {
            WitnessError::InvalidWitnessData {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: error.to_string(),
            }
        })?,
        None => 0,
    };
    let instruction_row = row.map_or_else(Default::default, |row| row.instruction);
    let instruction = JoltInstruction::try_from(instruction_row).map_err(|kind| {
        invalid_data(format!(
            "unsupported Jolt instruction kind in Stage 5 instruction row: {kind:?}"
        ))
    })?;
    let flags = instruction.circuit_flags();
    Ok(Stage5InstructionCycle {
        address,
        table: instruction.lookup_table(),
        interleaved_operands: flags.is_interleaved_operands(),
    })
}

fn bound_address_selector_at<F: Field>(
    address: u128,
    address_bits: usize,
    start_position: usize,
    point: &[F],
) -> Result<F, WitnessError> {
    point
        .iter()
        .enumerate()
        .try_fold(F::one(), |acc, (position, &challenge)| {
            let bit = address_bit(address, address_bits, start_position + position)?;
            Ok(if bit {
                acc * challenge
            } else {
                acc * (F::one() - challenge)
            })
        })
}

fn write_address_suffix<F: Field>(
    point: &mut [F],
    address: u128,
    address_bits: usize,
    start_position: usize,
) -> Result<(), WitnessError> {
    for (position, value) in point
        .iter_mut()
        .enumerate()
        .take(address_bits)
        .skip(start_position)
    {
        *value = F::from_bool(address_bit(address, address_bits, position)?);
    }
    Ok(())
}

fn multilinear_round_eval<F: Field>(poly: &Polynomial<F>, index: usize, point: F) -> F {
    let (lo, hi) = poly.sumcheck_eval_pair(index, BindingOrder::LowToHigh);
    lo + point * (hi - lo)
}

fn address_bit(address: u128, address_bits: usize, position: usize) -> Result<bool, WitnessError> {
    if address_bits > u128::BITS as usize {
        return Err(invalid_dimensions(format!(
            "Stage 5 instruction address width {address_bits} exceeds u128 width"
        )));
    }
    if position >= address_bits {
        return Err(invalid_dimensions(format!(
            "Stage 5 instruction address bit position {position} is out of range {address_bits}"
        )));
    }
    let shift = address_bits - 1 - position;
    Ok(((address >> shift) & 1) == 1)
}

fn validate_len(label: &'static str, actual: usize, expected: usize) -> Result<(), WitnessError> {
    if actual != expected {
        return Err(invalid_dimensions(format!(
            "{label} has {actual} values, expected {expected}"
        )));
    }
    Ok(())
}

fn invalid_dimensions(reason: impl std::fmt::Display) -> WitnessError {
    WitnessError::InvalidDimensions {
        namespace: JOLT_VM_NAMESPACE.name,
        reason: reason.to_string(),
    }
}

fn invalid_data(reason: impl std::fmt::Display) -> WitnessError {
    WitnessError::InvalidWitnessData {
        namespace: JOLT_VM_NAMESPACE.name,
        reason: reason.to_string(),
    }
}
