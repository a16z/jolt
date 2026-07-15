use jolt_field::Field;

/// Raw (unremapped) RAM access address; 0 when the cycle makes no RAM
/// access.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamAddress(pub u64);

/// Pre-access RAM word value; 0 when the cycle makes no RAM access.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamReadValue(pub u64);

/// Post-access RAM word value (equals [`RamReadValue`] for reads); 0 when the
/// cycle makes no RAM access.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamWriteValue(pub u64);

/// Whether the cycle accesses a nonzero RAM address.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamHammingWeight(pub bool);

impl RamAddress {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl RamReadValue {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl RamWriteValue {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl RamHammingWeight {
    pub fn to_field<F: Field>(self) -> F {
        F::from_bool(self.0)
    }
}
