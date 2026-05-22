use jolt_field::Field;

use crate::{Error, WrapperProtocol};

pub fn verify_r1cs_witness<F: Field>(protocol: &WrapperProtocol<F>) -> Result<(), Error> {
    for (index, (&variable, &public_input)) in protocol
        .layout
        .public_inputs
        .iter()
        .zip(&protocol.public_inputs)
        .enumerate()
    {
        let witness_value = protocol
            .witness
            .get(variable.index())
            .copied()
            .ok_or(Error::MissingWitnessVariable { variable })?;
        if witness_value != public_input {
            return Err(Error::PublicInputMismatch { index, variable });
        }
    }

    protocol
        .r1cs
        .check_witness(&protocol.witness)
        .map_err(|constraint| Error::UnsatisfiedConstraint { constraint })
}
