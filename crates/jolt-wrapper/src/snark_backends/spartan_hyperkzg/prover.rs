use jolt_field::Field;

use crate::{Error, WrapperProtocol};

use super::{SpartanHyperKzgProof, SpartanHyperKzgSetup};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpartanHyperKzgProver;

impl SpartanHyperKzgProver {
    pub fn prove<F: Field>(
        _setup: &SpartanHyperKzgSetup<F>,
        _protocol: &WrapperProtocol<F>,
    ) -> Result<SpartanHyperKzgProof<F>, Error> {
        Err(Error::BackendNotImplemented {
            backend: "spartan-hyperkzg",
        })
    }
}
