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

/// NB: Call this at the start of each test that uses field arithmetic.
///
/// This needs to be called in order for the small-value lookup tables to be initialized for our
/// test field, however calling it more than once will cause memory errors. In `jolt_core`, this is
/// handled by omitting the small-value lookup tables from the testing configuration. However,
/// that's not an option for us unless we want to add a feature flag to `jolt_core`.
///
/// TODO(hamlinb): It seems like this could be fixed easily in `jolt_core` by initializing the
/// small-value lookup tables using `lazy_static`, rather than exposing an initialization function
/// that needs to be called manually
#[cfg(test)]
pub fn initialize_fields() {
    use ark_bn254::Fr;
    use std::sync::Once;

    static INIT: Once = Once::new();

    INIT.call_once(|| {
        let small_value_lookup_tables = Fr::compute_lookup_tables();
        Fr::initialize_lookup_tables(small_value_lookup_tables);
    })
}

#[cfg(test)]
use proptest::prelude::*;

#[cfg(test)]
pub fn arb_field_elem<F: JoltField>() -> impl Strategy<Value = F> {
    proptest::collection::vec(any::<u8>(), F::NUM_BYTES).prop_map(|bytes| F::from_bytes(&bytes))
}
