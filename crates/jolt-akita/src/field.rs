use jolt_field::Field;

/// Field accepted by the Akita mock adapter.
pub trait AkitaClaimField: Field {}

impl<F: Field> AkitaClaimField for F {}

/// Base-field mode is identity conversion.
#[inline]
pub fn to_akita_claim<F: AkitaClaimField>(value: F) -> F {
    value
}
