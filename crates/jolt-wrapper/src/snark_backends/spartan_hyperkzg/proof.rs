use std::marker::PhantomData;

use jolt_field::Field;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanHyperKzgProof<F: Field> {
    marker: PhantomData<F>,
}
