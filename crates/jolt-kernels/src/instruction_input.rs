//! The instruction input-virtualization (stage 3) kernel: a naive member
//! over the cycle domain.
//!
//! The summand is
//! `eq(r_product, j) · ((r_is_rs2·rs2 + r_is_imm·imm) + γ·(l_is_rs1·rs1 + l_is_pc·upc))(j)`
//! — degree 3 (eq × flag × value). The eight operand/flag leaves stay
//! separate tables: collapsing a flag·value product into one table would
//! drop the round polynomial's curvature (the stage-2 lesson).

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::instruction::{
    imm, left_operand_is_pc, left_operand_is_rs1, right_operand_is_imm, right_operand_is_rs2,
    rs1_value, rs2_value, unexpanded_pc,
};
use jolt_claims::protocols::jolt::relations::instruction::InstructionInputChallenges;
use jolt_claims::protocols::jolt::{InstructionInputPublic, JoltDerivedId, TraceDimensions};
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage3::outputs::InstructionInput;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::views::{dense_view, eq_table};
use crate::{KernelError, NaiveSumcheckProver, ProofSession, ProveSumcheck, ReferenceBackend};

/// The stage-3 instruction input-virtualization slot.
pub trait InstructionInputProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        trace_dimensions: TraceDimensions,
        product_remainder_point: &[F],
        challenges: &InstructionInputChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = InstructionInput<F>>>, KernelError<F>>;
}

impl<F: Field> InstructionInputProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        trace_dimensions: TraceDimensions,
        product_remainder_point: &[F],
        challenges: &InstructionInputChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = InstructionInput<F>>>, KernelError<F>> {
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

        let relation = InstructionInput::new(trace_dimensions, product_remainder_point.to_vec());
        Ok(Box::new(NaiveSumcheckProver::new(
            relation,
            challenges,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
