use jolt_crypto::VectorCommitmentOpening;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: for<'a> Deserialize<'a>"))]
pub struct HyraxOpeningProof<F> {
    pub row_opening: VectorCommitmentOpening<F>,
}
