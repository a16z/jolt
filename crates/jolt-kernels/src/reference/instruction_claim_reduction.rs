//! The instruction claim-reduction (stage 2) kernel: a naive member over the
//! cycle domain.
//!
//! The summand `eq(τ_low, j) · (o₁ + γ·o₂ + γ²·o₃ + γ³·o₄ + γ⁴·o₅)(j)` over
//! the five instruction-lookup operand tables, bound `LowToHigh`. Three of its
//! openings alias the product remainder's (same polynomial, same point) — the
//! generated drivers suppress the duplicate absorbs; the naive members agree
//! on the claims because they bind the same tables with the same challenge
//! slice.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::claim_reductions::instruction::{
    left_instruction_input_reduced, left_lookup_operand_reduced, lookup_output_reduced,
    right_instruction_input_reduced, right_lookup_operand_reduced,
};
use jolt_claims::protocols::jolt::{InstructionClaimReductionPublic, JoltDerivedId};
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::relations::ProverInputs;
use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use super::views::{dense_view, eq_table};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel,
};

impl<F: Field> PrepareKernel<F, InstructionClaimReduction<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
        inputs: ProverInputs<'_, F, InstructionClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionClaimReduction<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let tau_low = relation.tau_low();
        let ids = [
            lookup_output_reduced(),
            left_lookup_operand_reduced(),
            right_lookup_operand_reduced(),
            left_instruction_input_reduced(),
            right_instruction_input_reduced(),
        ];
        let opening_tables = ids
            .into_iter()
            .map(|id| Ok((id, Polynomial::new(dense_view(witness, id)?))))
            .collect::<Result<BTreeMap<_, _>, KernelError<F>>>()?;
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(InstructionClaimReductionPublic::EqSpartan),
            Polynomial::new(eq_table(tau_low)),
        )]);

        Ok(Box::new(NaiveSumcheckProver::new(
            relation,
            inputs.challenges,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
