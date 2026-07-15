use jolt_field::Field;

/// Output of the instruction's lookup query.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LookupOutput(pub u64);

impl LookupOutput {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}
