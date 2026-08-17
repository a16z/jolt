//! Metal offload seams for the stage-6b instruction RA virtualization
//! kernel: the packed-table handoff to the device sequence and the
//! device-resident round state machine.

use jolt_field::Field;
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_sumcheck::SumcheckError;

use crate::optimized::instruction_ra_virtualization::{
    instruction_ra_state_error, InstructionRaInitialization, InstructionRaTableState,
    OptimizedInstructionRaVirtualizationKernel,
};
use crate::optimized::lazy_ra::LazyFoldedRa;
use crate::KernelError;

impl<F: Field> InstructionRaInitialization<F> {
    /// The current Metal sequence is specialized for four virtual products,
    /// each with four 8-bit committed factors.
    pub(crate) fn supports_metal_sequence(&self) -> bool {
        self.num_committed_per_virtual == 4
            && self.committed_chunk_bits == 8
            && self.chunk_tables.len() == 16
            && self.chunk_tables.iter().all(|table| table.len() == 256)
    }

    pub(crate) fn into_offloaded(
        mut self,
    ) -> Result<(OptimizedInstructionRaVirtualizationKernel<F>, Vec<F>), KernelError<F>> {
        if !self.supports_metal_sequence() {
            return Err(KernelError::Unsupported {
                reason: "instruction RA Metal sequence requires 4x4 8-bit geometry",
            });
        }
        let tables = std::mem::take(&mut self.chunk_tables);
        let num_committed = tables.len();
        let table_values = tables.iter().map(Vec::len).sum();
        let mut chunk_tables = Vec::with_capacity(table_values);
        for table in tables {
            chunk_tables.extend(table);
        }
        let kernel = self.into_kernel(num_committed, InstructionRaTableState::Device);
        Ok((kernel, chunk_tables))
    }
}

impl<F: Field> OptimizedInstructionRaVirtualizationKernel<F> {
    pub(crate) fn metal_num_polys(&self) -> usize {
        self.num_committed
    }

    pub(crate) fn metal_weights(&self) -> Result<(&[F], &[F]), SumcheckError<F>> {
        if !matches!(self.tables, InstructionRaTableState::Device) {
            return Err(instruction_ra_state_error(
                "Metal weights requested after instruction RA returned to the CPU",
            ));
        }
        if self.rounds_bound >= self.log_t {
            return Err(instruction_ra_state_error(
                "Metal weights requested after the final instruction RA bind",
            ));
        }
        Ok((self.gruen.e_in_current(), self.gruen.e_out_current()))
    }

    pub(crate) fn metal_bind_offloaded(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        if !matches!(self.tables, InstructionRaTableState::Device) {
            return Err(instruction_ra_state_error(
                "Metal bind requested after instruction RA returned to the CPU",
            ));
        }
        if self.rounds_bound >= self.log_t {
            return Err(instruction_ra_state_error(
                "instruction RA received more binds than cycle variables",
            ));
        }
        self.gruen.bind(challenge);
        self.rounds_bound += 1;
        Ok(())
    }

    pub(crate) fn metal_message(
        &self,
        q_evals: [F; 4],
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if !matches!(self.tables, InstructionRaTableState::Device) {
            return Err(instruction_ra_state_error(
                "Metal message supplied after instruction RA returned to the CPU",
            ));
        }
        if self.num_committed_per_virtual != 4 {
            return Err(instruction_ra_state_error(
                "Metal instruction RA message requires four factors per virtual product",
            ));
        }
        Ok(self.gruen.gruen_poly_from_evals(&q_evals, previous_claim))
    }

    pub(crate) fn metal_restore_dense(
        &mut self,
        flat_tables: &[F],
        elements: usize,
    ) -> Result<(), SumcheckError<F>> {
        if !matches!(self.tables, InstructionRaTableState::Device) {
            return Err(instruction_ra_state_error(
                "instruction RA dense tables restored more than once",
            ));
        }
        if elements == 0 || !elements.is_power_of_two() || self.rounds_bound > self.log_t {
            return Err(instruction_ra_state_error(
                "invalid instruction RA dense-tail geometry",
            ));
        }
        let expected_elements = 1usize
            .checked_shl((self.log_t - self.rounds_bound) as u32)
            .ok_or_else(|| instruction_ra_state_error("instruction RA table length overflow"))?;
        if elements != expected_elements {
            return Err(instruction_ra_state_error(format!(
                "instruction RA dense tail has {elements} elements per factor; expected {expected_elements}"
            )));
        }
        let expected_values = self
            .num_committed
            .checked_mul(elements)
            .ok_or_else(|| instruction_ra_state_error("instruction RA readback length overflow"))?;
        if flat_tables.len() != expected_values {
            return Err(instruction_ra_state_error(format!(
                "instruction RA dense tail has {} values; expected {expected_values}",
                flat_tables.len()
            )));
        }

        let tables = flat_tables
            .chunks_exact(elements)
            .map(|values| Polynomial::new(values.to_vec()))
            .collect();
        self.tables = InstructionRaTableState::Cpu(LazyFoldedRa::Dense(tables));
        Ok(())
    }
}
