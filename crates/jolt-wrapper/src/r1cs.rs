use jolt_field::Field;

use crate::{Error, WrapperProtocol, WrapperStatement, WrapperWitness};

pub fn build_configured_verifier_protocol<F: Field>(
    _statement: &WrapperStatement<F>,
    _witness: &WrapperWitness<F>,
) -> Result<WrapperProtocol<F>, Error> {
    Err(Error::Unimplemented {
        component: "configured verifier R1CS assembly",
    })
}
