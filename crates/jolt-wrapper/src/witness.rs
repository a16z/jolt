use jolt_field::Field;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WrapperWitness<F: Field> {
    pub private_scalars: Vec<F>,
    pub private_bytes: Vec<u8>,
}

impl<F: Field> WrapperWitness<F> {
    pub fn new(private_scalars: Vec<F>, private_bytes: Vec<u8>) -> Self {
        Self {
            private_scalars,
            private_bytes,
        }
    }
}
