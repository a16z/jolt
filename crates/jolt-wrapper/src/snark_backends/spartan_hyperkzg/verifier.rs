use jolt_field::Field;

use crate::Error;

use super::{SpartanHyperKzgProof, SpartanHyperKzgSetup};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpartanHyperKzgVerifier;

impl SpartanHyperKzgVerifier {
    pub fn verify<F: Field>(
        _setup: &SpartanHyperKzgSetup<F>,
        _proof: &SpartanHyperKzgProof<F>,
        _public_inputs: &[F],
    ) -> Result<(), Error> {
        Err(Error::BackendNotImplemented {
            backend: "spartan-hyperkzg",
        })
    }
}
