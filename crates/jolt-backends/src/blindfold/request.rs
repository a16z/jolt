use jolt_field::Field;

use crate::BackendValueSlot;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BlindFoldSlot(pub u32);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindFoldRoundRequest<F: Field> {
    pub slot: BlindFoldSlot,
    pub coefficients: Vec<BackendValueSlot>,
    pub blinding_label: &'static str,
    pub _field: core::marker::PhantomData<F>,
}

impl<F: Field> BlindFoldRoundRequest<F> {
    pub const fn new(
        slot: BlindFoldSlot,
        coefficients: Vec<BackendValueSlot>,
        blinding_label: &'static str,
    ) -> Self {
        Self {
            slot,
            coefficients,
            blinding_label,
            _field: core::marker::PhantomData,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindFoldRequest<F: Field> {
    pub label: &'static str,
    pub rounds: Vec<BlindFoldRoundRequest<F>>,
    pub output_claims: Vec<BackendValueSlot>,
}

impl<F: Field> BlindFoldRequest<F> {
    pub const fn new(
        label: &'static str,
        rounds: Vec<BlindFoldRoundRequest<F>>,
        output_claims: Vec<BackendValueSlot>,
    ) -> Self {
        Self {
            label,
            rounds,
            output_claims,
        }
    }
}
