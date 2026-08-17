//! Metal offload seams for the stage-3 instruction-input kernel: offloaded
//! construction, the device-resident round state machine, and the dense-table
//! alias snapshot the product-remainder carrier reads.

use jolt_field::Field;
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::SumcheckError;

use crate::optimized::instruction_input::{
    instruction_input_state_error, InputState, OptimizedInstructionInputKernel, NUM_TABLES,
};

impl<F: Field> OptimizedInstructionInputKernel<F> {
    pub(crate) fn new_offloaded(r_product: &[F], gamma: F) -> Self {
        Self {
            log_t: r_product.len(),
            gamma,
            state: InputState::Offloaded,
            gruen: GruenSplitEqPolynomial::new(r_product, BindingOrder::LowToHigh),
            bind_scratch: Vec::new(),
            rounds_bound: 0,
        }
    }

    pub(crate) fn metal_copy_dense_tables(
        &self,
        table_ids: [usize; 2],
        expected_rounds_bound: usize,
        expected_elements: usize,
    ) -> Result<[Vec<F>; 2], SumcheckError<F>> {
        if self.rounds_bound != expected_rounds_bound {
            return Err(instruction_input_state_error(
                "instruction input alias snapshot has the wrong bind count",
            ));
        }
        let InputState::Dense(tables) = &self.state else {
            return Err(instruction_input_state_error(
                "instruction input alias snapshot requires host dense tables",
            ));
        };
        if table_ids
            .iter()
            .any(|&table| table >= tables.len() || tables[table].len() != expected_elements)
        {
            return Err(instruction_input_state_error(
                "instruction input alias snapshot has the wrong table geometry",
            ));
        }
        Ok(table_ids.map(|table| tables[table].evals().to_vec()))
    }

    pub(crate) fn metal_weights(&self) -> Result<(&[F], &[F]), SumcheckError<F>> {
        if !matches!(self.state, InputState::Offloaded) {
            return Err(instruction_input_state_error(
                "Metal weights requested after instruction input returned to the CPU",
            ));
        }
        Ok((self.gruen.e_in_current(), self.gruen.e_out_current()))
    }

    pub(crate) fn metal_bind_offloaded(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        if !matches!(self.state, InputState::Offloaded) {
            return Err(instruction_input_state_error(
                "Metal bind requested after instruction input returned to the CPU",
            ));
        }
        if self.rounds_bound >= self.log_t {
            return Err(instruction_input_state_error(
                "instruction input received more binds than cycle variables",
            ));
        }
        self.gruen.bind(challenge);
        self.rounds_bound += 1;
        Ok(())
    }

    pub(crate) fn metal_message(
        &self,
        q_coefficients: [F; 3],
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if !matches!(self.state, InputState::Offloaded) {
            return Err(instruction_input_state_error(
                "Metal message supplied after instruction input returned to the CPU",
            ));
        }
        let [q_at_0, q_at_1, q_quadratic] = q_coefficients;
        let twice_quadratic = q_quadratic + q_quadratic;
        let q_at_2 = q_at_1 + q_at_1 - q_at_0 + twice_quadratic;
        let q_at_3 = q_at_2 + q_at_1 - q_at_0 + twice_quadratic + twice_quadratic;
        self.message_from_q_evals([q_at_0, q_at_1, q_at_2, q_at_3], round, previous_claim)
    }

    pub(crate) fn metal_restore_dense(
        &mut self,
        flat_tables: &[F],
        elements: usize,
    ) -> Result<(), SumcheckError<F>> {
        if !matches!(self.state, InputState::Offloaded) {
            return Err(instruction_input_state_error(
                "instruction input dense tables restored more than once",
            ));
        }
        if elements == 0 || !elements.is_power_of_two() || self.rounds_bound > self.log_t {
            return Err(instruction_input_state_error(
                "invalid instruction input dense-tail geometry",
            ));
        }
        let expected_elements = 1usize
            .checked_shl((self.log_t - self.rounds_bound) as u32)
            .ok_or_else(|| {
                instruction_input_state_error("instruction input table length overflow")
            })?;
        if elements != expected_elements {
            return Err(instruction_input_state_error(format!(
                "instruction input dense tail has {elements} elements per table; expected {expected_elements}"
            )));
        }
        let expected_values = NUM_TABLES.checked_mul(elements).ok_or_else(|| {
            instruction_input_state_error("instruction input readback length overflow")
        })?;
        if flat_tables.len() != expected_values {
            return Err(instruction_input_state_error(format!(
                "instruction input dense tail has {} values; expected {expected_values}",
                flat_tables.len()
            )));
        }
        self.state = InputState::Dense(
            flat_tables
                .chunks_exact(elements)
                .map(|values| Polynomial::new(values.to_vec()))
                .collect(),
        );
        Ok(())
    }
}
