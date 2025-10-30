use jolt_core::field::JoltField;
#[cfg(test)]
use std::collections::HashMap;

const TAB: &str = "  ";

pub fn indent(level: usize) -> String {
    std::iter::repeat_n(String::from(TAB), level)
        .collect::<Vec<_>>()
        .concat()
}

pub type LetBinderIndex = usize;

#[cfg(test)]
pub struct Environment<'a, F> {
    pub let_bindings: &'a HashMap<LetBinderIndex, F>,
    pub vars: &'a [F],
}

/// A [`JoltField`] that can be used to write a ZKLean representation of a computation.
pub trait ZkLeanReprField: JoltField + Sized {
    fn register(name: char, size: usize) -> Vec<Self>;

    #[cfg(test)]
    fn evaluate<F: JoltField>(&self, env: &Environment<F>) -> F;

    fn format_for_lean(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        name: &str,
        num_variables: usize,
    ) -> std::fmt::Result;
}

#[cfg(test)]
use proptest::prelude::*;

#[cfg(test)]
pub fn arb_field_elem<F: JoltField>() -> impl Strategy<Value = F> {
    proptest::collection::vec(any::<u8>(), F::NUM_BYTES).prop_map(|bytes| F::from_bytes(&bytes))
}
