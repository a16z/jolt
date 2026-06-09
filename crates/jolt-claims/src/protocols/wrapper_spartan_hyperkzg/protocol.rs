use super::{
    SpartanRelationDimensions, WrapperRelationDimensions, WrapperSpartanHyperKzgFactsError,
};
use jolt_poly::{Point, HIGH_TO_LOW};
use serde::{Deserialize, Serialize};

pub const WRAPPER_SPARTAN_HYPERKZG_PROTOCOL_ID: &[u8] = b"wrapper-spartan-hyperkzg-v1";
pub const WRAPPER_TRANSCRIPT_LABEL: &[u8] = b"JoltWrapper";
pub const WRAPPER_STATEMENT_TRANSCRIPT_LABEL: &[u8] = b"wrapper_statement";
pub const WRAPPER_PROTOCOL_ID_TRANSCRIPT_LABEL: &[u8] = b"protocol_id";
pub const WRAPPER_RELATION_DIMS_TRANSCRIPT_LABEL: &[u8] = b"relation_dims";
pub const WRAPPER_RELATION_MATRICES_TRANSCRIPT_LABEL: &[u8] = b"relation_matrices";
pub const WRAPPER_RELATION_MATRIX_A_TRANSCRIPT_LABEL: &[u8] = b"relation_matrix_a";
pub const WRAPPER_RELATION_MATRIX_B_TRANSCRIPT_LABEL: &[u8] = b"relation_matrix_b";
pub const WRAPPER_RELATION_MATRIX_C_TRANSCRIPT_LABEL: &[u8] = b"relation_matrix_c";
pub const WRAPPER_RELATION_MATRIX_ROW_TRANSCRIPT_LABEL: &[u8] = b"relation_matrix_row";
pub const WRAPPER_SPARTAN_DIMS_TRANSCRIPT_LABEL: &[u8] = b"spartan_dims";
pub const WRAPPER_PUBLIC_INPUT_LAYOUT_TRANSCRIPT_LABEL: &[u8] = b"public_input_layout";
pub const WRAPPER_PUBLIC_INPUTS_TRANSCRIPT_PREFIX: &[u8] = b"WrapperPublicInputs";
pub const WRAPPER_PUBLIC_INPUTS_TRANSCRIPT_LABEL: &[u8] = b"wrapper_public_inputs";
pub const WRAPPER_WITNESS_COMMITMENT_TRANSCRIPT_LABEL: &[u8] = b"wrapper_witness_commitment";
pub const WRAPPER_SPARTAN_TAU_TRANSCRIPT_LABEL: &[u8] = b"spartan_tau";
pub const WRAPPER_SPARTAN_OUTER_SUMCHECK_TRANSCRIPT_LABEL: &[u8] = b"spartan_outer_sumcheck";
pub const WRAPPER_SPARTAN_INNER_BATCHING_TRANSCRIPT_LABEL: &[u8] = b"spartan_inner_batching";
pub const WRAPPER_SPARTAN_INNER_SUMCHECK_TRANSCRIPT_LABEL: &[u8] = b"spartan_inner_sumcheck";
pub const WRAPPER_SPARTAN_OUTER_SUMCHECK_DEGREE: usize = 3;
pub const WRAPPER_SPARTAN_INNER_SUMCHECK_DEGREE: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WrapperSpartanHyperKzgStatementFacts {
    pub relation: WrapperRelationDimensions,
    pub spartan: SpartanRelationDimensions,
    pub protocol_id: &'static [u8],
    pub transcript_label: &'static [u8],
    pub public_inputs_transcript_prefix: &'static [u8],
    pub public_inputs_transcript_label: &'static [u8],
}

impl WrapperSpartanHyperKzgStatementFacts {
    pub fn from_relation_dimensions(
        relation: WrapperRelationDimensions,
    ) -> Result<Self, WrapperSpartanHyperKzgFactsError> {
        Ok(Self {
            relation,
            spartan: relation.spartan_dimensions()?,
            protocol_id: WRAPPER_SPARTAN_HYPERKZG_PROTOCOL_ID,
            transcript_label: WRAPPER_TRANSCRIPT_LABEL,
            public_inputs_transcript_prefix: WRAPPER_PUBLIC_INPUTS_TRANSCRIPT_PREFIX,
            public_inputs_transcript_label: WRAPPER_PUBLIC_INPUTS_TRANSCRIPT_LABEL,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct SpartanOuterReduction<F> {
    pub tau: Point<HIGH_TO_LOW, F>,
    pub rx: Point<HIGH_TO_LOW, F>,
    pub final_claim: F,
}

impl<F> SpartanOuterReduction<F> {
    pub fn new(tau: Point<HIGH_TO_LOW, F>, rx: Point<HIGH_TO_LOW, F>, final_claim: F) -> Self {
        Self {
            tau,
            rx,
            final_claim,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct SpartanOuterEvaluationClaims<F> {
    pub a: F,
    pub b: F,
    pub c: F,
}

impl<F> SpartanOuterEvaluationClaims<F> {
    pub fn new(a: F, b: F, c: F) -> Self {
        Self { a, b, c }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct SpartanInnerBatchingCoefficients<F> {
    pub a: F,
    pub b: F,
    pub c: F,
}

impl<F> SpartanInnerBatchingCoefficients<F>
where
    F: Copy + std::ops::Add<Output = F> + std::ops::Mul<Output = F>,
{
    pub fn combine(self, claims: SpartanOuterEvaluationClaims<F>) -> F {
        self.a * claims.a + self.b * claims.b + self.c * claims.c
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct SpartanInnerStatement<F> {
    pub rx: Point<HIGH_TO_LOW, F>,
    pub batching: SpartanInnerBatchingCoefficients<F>,
    pub claimed_sum: F,
}

impl<F> SpartanInnerStatement<F> {
    pub fn new(
        rx: Point<HIGH_TO_LOW, F>,
        batching: SpartanInnerBatchingCoefficients<F>,
        claimed_sum: F,
    ) -> Self {
        Self {
            rx,
            batching,
            claimed_sum,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct SpartanWitnessOpeningStatement<F> {
    pub ry: Point<HIGH_TO_LOW, F>,
    pub witness_eval: F,
}

impl<F> SpartanWitnessOpeningStatement<F> {
    pub fn new(ry: Point<HIGH_TO_LOW, F>, witness_eval: F) -> Self {
        Self { ry, witness_eval }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests assert successful construction")]
mod tests {
    use super::{
        WrapperRelationDimensions, WrapperSpartanHyperKzgStatementFacts,
        WRAPPER_PROTOCOL_ID_TRANSCRIPT_LABEL, WRAPPER_PUBLIC_INPUTS_TRANSCRIPT_LABEL,
        WRAPPER_PUBLIC_INPUTS_TRANSCRIPT_PREFIX, WRAPPER_PUBLIC_INPUT_LAYOUT_TRANSCRIPT_LABEL,
        WRAPPER_RELATION_DIMS_TRANSCRIPT_LABEL, WRAPPER_RELATION_MATRICES_TRANSCRIPT_LABEL,
        WRAPPER_RELATION_MATRIX_A_TRANSCRIPT_LABEL, WRAPPER_RELATION_MATRIX_B_TRANSCRIPT_LABEL,
        WRAPPER_RELATION_MATRIX_C_TRANSCRIPT_LABEL, WRAPPER_RELATION_MATRIX_ROW_TRANSCRIPT_LABEL,
        WRAPPER_SPARTAN_DIMS_TRANSCRIPT_LABEL, WRAPPER_SPARTAN_HYPERKZG_PROTOCOL_ID,
        WRAPPER_STATEMENT_TRANSCRIPT_LABEL, WRAPPER_TRANSCRIPT_LABEL,
    };

    #[test]
    fn statement_facts_preserve_relation_and_spartan_dimensions() {
        let relation = WrapperRelationDimensions::new(8, 13, 3);
        let facts =
            WrapperSpartanHyperKzgStatementFacts::from_relation_dimensions(relation).unwrap();

        assert_eq!(facts.relation, relation);
        assert_eq!(facts.spartan.num_vars(), relation.variables);
        assert_eq!(facts.spartan.num_vars_padded(), 8);
        assert_eq!(facts.spartan.num_constraints(), relation.constraints);
        assert_eq!(facts.spartan.num_constraints_padded(), 16);
        assert_eq!(facts.spartan.num_public_inputs(), relation.public_inputs);
    }

    #[test]
    fn statement_facts_expose_canonical_protocol_and_transcript_labels() {
        let facts = WrapperSpartanHyperKzgStatementFacts::from_relation_dimensions(
            WrapperRelationDimensions::new(1, 1, 0),
        )
        .unwrap();

        assert_eq!(facts.protocol_id, WRAPPER_SPARTAN_HYPERKZG_PROTOCOL_ID);
        assert_eq!(facts.transcript_label, WRAPPER_TRANSCRIPT_LABEL);
        assert_eq!(
            facts.public_inputs_transcript_prefix,
            WRAPPER_PUBLIC_INPUTS_TRANSCRIPT_PREFIX
        );
        assert_eq!(
            facts.public_inputs_transcript_label,
            WRAPPER_PUBLIC_INPUTS_TRANSCRIPT_LABEL
        );
        assert_eq!(WRAPPER_STATEMENT_TRANSCRIPT_LABEL, b"wrapper_statement");
        assert_eq!(WRAPPER_PROTOCOL_ID_TRANSCRIPT_LABEL, b"protocol_id");
        assert_eq!(WRAPPER_RELATION_DIMS_TRANSCRIPT_LABEL, b"relation_dims");
        assert_eq!(
            WRAPPER_RELATION_MATRICES_TRANSCRIPT_LABEL,
            b"relation_matrices"
        );
        assert_eq!(
            WRAPPER_RELATION_MATRIX_A_TRANSCRIPT_LABEL,
            b"relation_matrix_a"
        );
        assert_eq!(
            WRAPPER_RELATION_MATRIX_B_TRANSCRIPT_LABEL,
            b"relation_matrix_b"
        );
        assert_eq!(
            WRAPPER_RELATION_MATRIX_C_TRANSCRIPT_LABEL,
            b"relation_matrix_c"
        );
        assert_eq!(
            WRAPPER_RELATION_MATRIX_ROW_TRANSCRIPT_LABEL,
            b"relation_matrix_row"
        );
        assert_eq!(WRAPPER_SPARTAN_DIMS_TRANSCRIPT_LABEL, b"spartan_dims");
        assert_eq!(
            WRAPPER_PUBLIC_INPUT_LAYOUT_TRANSCRIPT_LABEL,
            b"public_input_layout"
        );
        assert_eq!(
            super::WRAPPER_WITNESS_COMMITMENT_TRANSCRIPT_LABEL,
            b"wrapper_witness_commitment"
        );
        assert_eq!(super::WRAPPER_SPARTAN_TAU_TRANSCRIPT_LABEL, b"spartan_tau");
        assert_eq!(
            super::WRAPPER_SPARTAN_OUTER_SUMCHECK_TRANSCRIPT_LABEL,
            b"spartan_outer_sumcheck"
        );
        assert_eq!(
            super::WRAPPER_SPARTAN_INNER_BATCHING_TRANSCRIPT_LABEL,
            b"spartan_inner_batching"
        );
        assert_eq!(
            super::WRAPPER_SPARTAN_INNER_SUMCHECK_TRANSCRIPT_LABEL,
            b"spartan_inner_sumcheck"
        );
        assert_eq!(super::WRAPPER_SPARTAN_OUTER_SUMCHECK_DEGREE, 3);
        assert_eq!(super::WRAPPER_SPARTAN_INNER_SUMCHECK_DEGREE, 2);
    }
}
