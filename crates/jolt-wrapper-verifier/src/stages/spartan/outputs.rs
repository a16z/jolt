use jolt_claims::protocols::wrapper_spartan_hyperkzg::{
    SpartanInnerBatchingCoefficients, SpartanInnerStatement, SpartanOuterEvaluationClaims,
    SpartanOuterReduction, SpartanRelationDimensions, SpartanWitnessOpeningStatement,
};
use jolt_field::Field;
#[cfg(feature = "zk")]
use jolt_poly::{Point, HIGH_TO_LOW};
use jolt_sumcheck::EvaluationClaim;
#[cfg(feature = "zk")]
use jolt_sumcheck::{CommittedSumcheckConsistency, SumcheckStatement};

#[cfg(feature = "zk")]
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpartanOutput<F: Field> {
    pub dimensions: SpartanRelationDimensions,
    pub outer_reduction: EvaluationClaim<F>,
    pub outer: SpartanOuterReduction<F>,
    pub outer_evaluation_claims: SpartanOuterEvaluationClaims<F>,
    pub inner_batching: SpartanInnerBatchingCoefficients<F>,
    pub inner_reduction: EvaluationClaim<F>,
    pub inner: SpartanInnerStatement<F>,
    pub witness_opening: SpartanWitnessOpeningStatement<F>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpartanZkOutput<F: Field, C> {
    pub dimensions: SpartanRelationDimensions,
    pub tau: Point<HIGH_TO_LOW, F>,
    pub outer_statement: SumcheckStatement,
    pub outer_consistency: CommittedSumcheckConsistency<F, C>,
    pub outer_rx: Point<HIGH_TO_LOW, F>,
    pub eq_tau_rx: F,
    pub outer_output_claims: CommittedOutputClaimOutput<C>,
    pub inner_batching: SpartanInnerBatchingCoefficients<F>,
    pub inner_statement: SumcheckStatement,
    pub inner_consistency: CommittedSumcheckConsistency<F, C>,
    pub inner_ry: Point<HIGH_TO_LOW, F>,
    pub combined_matrix_eval: F,
    pub inner_output_claims: CommittedOutputClaimOutput<C>,
}
