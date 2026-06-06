use jolt_claims::protocols::wrapper_spartan_hyperkzg::WrapperSpartanHyperKzgStatementFacts;
use jolt_field::Field;
use jolt_r1cs::ConstraintMatrices;

#[derive(Clone, Debug)]
pub struct R1csRelationOutput<'a, F: Field> {
    pub statement_facts: WrapperSpartanHyperKzgStatementFacts,
    pub relation: &'a ConstraintMatrices<F>,
    pub public_inputs: Vec<F>,
}
