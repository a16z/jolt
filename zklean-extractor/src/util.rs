use jolt_core::field::JoltField;

const TAB: &str = "  ";

pub fn indent(level: usize) -> String {
    std::iter::repeat_n(String::from(TAB), level)
        .collect::<Vec<_>>()
        .concat()
}

/// A [`JoltField`] that can be used to write a ZKLean representation of a computation.
pub trait ZkLeanReprField: JoltField + Sized {
    fn register(name: char, size: usize) -> Vec<Self>;

    fn as_computation(&self) -> String;

    #[cfg(test)]
    fn evaluate<F: JoltField>(&self, vars: &[F]) -> F;
}

#[cfg(test)]
use proptest::prelude::*;

#[cfg(test)]
pub fn arb_field_elem<F: JoltField>() -> impl Strategy<Value = F> {
    proptest::collection::vec(any::<u8>(), F::NUM_BYTES).prop_map(|bytes| F::from_bytes(&bytes))
}
