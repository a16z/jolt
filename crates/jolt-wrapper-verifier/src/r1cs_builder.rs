use jolt_field::Field;
use jolt_r1cs::{AssignedScalar, ConstraintMatrices, LinearCombination, R1csBuilder, Variable};
use jolt_transcript::r1cs::{R1csJoltByteTranscript, R1csTranscript};

use crate::Error;

#[derive(Clone, Debug)]
pub struct WrapperR1csProtocol<F: Field> {
    pub r1cs: ConstraintMatrices<F>,
    pub witness: Vec<F>,
    pub public_inputs: Vec<F>,
    pub layout: WrapperR1csLayout,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WrapperR1csLayout {
    pub public_inputs: Vec<Variable>,
}

impl<F: Field> WrapperR1csProtocol<F> {
    fn new(
        r1cs: ConstraintMatrices<F>,
        witness: Vec<F>,
        public_inputs: Vec<F>,
        layout: WrapperR1csLayout,
    ) -> Self {
        Self {
            r1cs,
            witness,
            public_inputs,
            layout,
        }
    }
}

pub struct WrapperR1csBuilder<F, Tr>
where
    F: Field,
{
    pub builder: R1csBuilder<F>,
    pub transcript: Tr,
    pub layout: WrapperR1csLayout,
    pub public_inputs: Vec<F>,
}

impl<F, Tr> WrapperR1csBuilder<F, Tr>
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
            layout: WrapperR1csLayout::default(),
            public_inputs: Vec::new(),
        }
    }
}

impl<F, Tr> WrapperR1csBuilder<F, Tr>
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
        self.builder
            .assert_equal(assigned.lc.clone(), LinearCombination::constant(value));
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

    pub fn finish(self) -> Result<WrapperR1csProtocol<F>, Error> {
        let witness = self.builder.witness()?;
        let r1cs = self.builder.into_matrices();
        Ok(WrapperR1csProtocol::new(
            r1cs,
            witness,
            self.public_inputs,
            self.layout,
        ))
    }
}

pub fn verify_r1cs_witness<F: Field>(protocol: &WrapperR1csProtocol<F>) -> Result<(), Error> {
    let expected = protocol.layout.public_inputs.len();
    let actual = protocol.public_inputs.len();
    if expected != actual {
        return Err(Error::PublicInputCountMismatch { expected, actual });
    }

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
