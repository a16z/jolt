use jolt_field::Field;

/// Bytecode-expanded program counter (the preprocessing PC index, not the
/// instruction's memory address).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pc(pub u64);

/// The instruction's memory address (virtual-sequence entries share it).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnexpandedPc(pub u64);

/// [`Pc`] of the successor row; 0 at the last cycle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NextPc(pub u64);

/// [`UnexpandedPc`] of the successor row; 0 at the last cycle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NextUnexpandedPc(pub u64);

impl Pc {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl UnexpandedPc {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl NextPc {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl NextUnexpandedPc {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}
