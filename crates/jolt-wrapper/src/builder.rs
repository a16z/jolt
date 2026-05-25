use jolt_field::Field;
use jolt_r1cs::{AssignedScalar, LinearCombination, R1csBuilder};
use jolt_transcript::r1cs::{R1csJoltByteTranscript, R1csTranscript};

use crate::{Error, WrapperLayout, WrapperProtocol};

pub struct WrapperProtocolBuilder<F, Tr>
where
    F: Field,
{
    pub builder: R1csBuilder<F>,
    pub transcript: Tr,
    pub layout: WrapperLayout,
    pub public_inputs: Vec<F>,
}

impl<F, Tr> WrapperProtocolBuilder<F, Tr>
where
    F: Field,
    Tr: R1csTranscript<F>,
{
    pub fn new(label: &'static [u8]) -> Self {
        let mut builder = R1csBuilder::new();
        let transcript = Tr::new(&mut builder, label);
        Self {
            builder,
            transcript,
            layout: WrapperLayout::default(),
            public_inputs: Vec::new(),
        }
    }
}

impl<F, Tr> WrapperProtocolBuilder<F, Tr>
where
    F: Field,
    Tr: R1csJoltByteTranscript<F, Challenge = AssignedScalar<F>, Byte = AssignedScalar<F>>,
{
    pub fn alloc_witness_scalar(&mut self, value: F) -> AssignedScalar<F> {
        AssignedScalar::alloc(&mut self.builder, value)
    }

    pub fn alloc_public_scalar(&mut self, value: F) -> AssignedScalar<F> {
        let variable = self.builder.alloc(value);
        let assigned = AssignedScalar::variable(value, variable);
        self.layout.public_inputs.push(variable);
        self.public_inputs.push(value);
        assigned
    }

    pub fn alloc_witness_byte(&mut self, value: u8) -> AssignedScalar<F> {
        let byte = AssignedScalar::alloc(&mut self.builder, F::from_u64(u64::from(value)));
        let mut reconstructed = LinearCombination::zero();
        for bit_index in 0..8 {
            let bit = (value >> bit_index) & 1;
            let bit = AssignedScalar::alloc(&mut self.builder, F::from_u64(u64::from(bit)));
            self.builder.assert_product(
                bit.lc.clone(),
                bit.lc.clone() - LinearCombination::one(),
                LinearCombination::zero(),
            );
            reconstructed = reconstructed + bit.lc.scale(F::from_u64(1_u64 << bit_index));
        }
        self.builder.assert_equal(byte.lc.clone(), reconstructed);
        byte
    }

    pub fn finish(self) -> Result<WrapperProtocol<F>, Error> {
        let witness = self.builder.witness()?;
        let r1cs = self.builder.into_matrices();
        Ok(WrapperProtocol::new(
            r1cs,
            witness,
            self.public_inputs,
            self.layout,
        ))
    }
}
