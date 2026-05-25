use std::marker::PhantomData;

use jolt_field::Field;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpartanHyperKzgSetup<F: Field> {
    marker: PhantomData<F>,
}

impl<F: Field> SpartanHyperKzgSetup<F> {
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<F: Field> Default for SpartanHyperKzgSetup<F> {
    fn default() -> Self {
        Self::new()
    }
}
