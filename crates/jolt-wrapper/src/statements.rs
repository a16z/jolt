use jolt_field::Field;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WrapperStatement<F: Field> {
    pub public_inputs: Vec<F>,
}

impl<F: Field> WrapperStatement<F> {
    pub fn new(public_inputs: Vec<F>) -> Self {
        Self { public_inputs }
    }
}
