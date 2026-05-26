use jolt_crypto::VectorCommitmentOpening;
use jolt_field::Field;
use jolt_sumcheck::CompressedSumcheckProof;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "Com: Serialize", deserialize = "Com: DeserializeOwned"))]
pub struct BlindFoldProof<F: Field, Com> {
    pub auxiliary_row_commitments: Vec<Com>,
    pub random_round_commitments: Vec<Com>,
    pub random_output_claim_row_commitments: Vec<Com>,
    pub random_auxiliary_row_commitments: Vec<Com>,
    pub random_error_row_commitments: Vec<Com>,
    pub random_eval_commitments: Vec<Com>,
    pub random_u: F,
    pub cross_term_error_row_commitments: Vec<Com>,
    pub outer_sumcheck: CompressedSumcheckProof<F>,
    pub az_rx: F,
    pub bz_rx: F,
    pub cz_rx: F,
    pub inner_sumcheck: CompressedSumcheckProof<F>,
    pub witness_opening: VectorCommitmentOpening<F>,
    pub error_opening: VectorCommitmentOpening<F>,
    pub folded_eval_outputs: Vec<F>,
    pub folded_eval_blindings: Vec<F>,
    pub folded_eval_output_openings: Vec<VectorCommitmentOpening<F>>,
    pub folded_eval_blinding_openings: Vec<VectorCommitmentOpening<F>>,
}
