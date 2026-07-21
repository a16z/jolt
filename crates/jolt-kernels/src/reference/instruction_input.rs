//! The instruction input-virtualization (stage 3) kernel: a naive member
//! over the cycle domain.
//!
//! The summand is
//! `eq(r_product, j) · ((r_is_rs2·rs2 + r_is_imm·imm) + γ·(l_is_rs1·rs1 + l_is_pc·upc))(j)`
//! — degree 3 (eq × flag × value). The eight operand/flag leaves stay
//! separate tables: collapsing a flag·value product into one table would
//! drop the round polynomial's curvature (the stage-2 lesson).

use std::collections::BTreeMap;

use crate::ProverInputs;
use jolt_claims::protocols::jolt::geometry::instruction::{
    imm, left_operand_is_pc, left_operand_is_rs1, right_operand_is_imm, right_operand_is_rs2,
    rs1_value, rs2_value, unexpanded_pc,
};
use jolt_claims::protocols::jolt::{InstructionInputPublic, JoltDerivedId};
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage3::outputs::InstructionInput;
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use super::views::{dense_view, eq_table};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel,
};

impl<F: Field> PrepareKernel<F, InstructionInput<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltVmWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionInput<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionInput<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let product_remainder_point = relation.product_remainder_opening_point();
        let ids = [
            left_operand_is_rs1(),
            rs1_value(),
            left_operand_is_pc(),
            unexpanded_pc(),
            right_operand_is_rs2(),
            rs2_value(),
            right_operand_is_imm(),
            imm(),
        ];
        let opening_tables = ids
            .into_iter()
            .map(|id| Ok((id, Polynomial::new(dense_view(witness, id)?))))
            .collect::<Result<BTreeMap<_, _>, KernelError<F>>>()?;
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(InstructionInputPublic::EqProduct),
            Polynomial::new(eq_table(product_remainder_point)),
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
