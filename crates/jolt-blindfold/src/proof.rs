use jolt_crypto::VectorCommitmentOpening;
use jolt_field::Field;
use jolt_sumcheck::CompressedSumcheckProof;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "C: Serialize", deserialize = "C: DeserializeOwned"))]
pub struct BlindFoldProof<F: Field, C> {
    pub auxiliary_row_commitments: Vec<C>,
    pub random_witness_row_commitments: Vec<C>,
    pub random_error_row_commitments: Vec<C>,
    pub random_eval_commitments: Vec<C>,
    pub random_u: F,
    pub cross_term_error_row_commitments: Vec<C>,
    pub outer_sumcheck: CompressedSumcheckProof<F>,
    pub az_rx: F,
    pub bz_rx: F,
    pub cz_rx: F,
    pub inner_sumcheck: CompressedSumcheckProof<F>,
    pub witness_opening: VectorCommitmentOpening<F>,
    pub error_opening: VectorCommitmentOpening<F>,
    pub folded_eval_outputs: Vec<F>,
    pub folded_eval_blindings: Vec<F>,
}
