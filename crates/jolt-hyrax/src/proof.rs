use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: for<'a> Deserialize<'a>"))]
pub struct HyraxOpeningProof<F> {
    pub combined_row: Vec<F>,
    pub combined_row_opening_scalar: F,
}
