/// Public-input scalar tagged with a stable label naming which value it is
/// (e.g. `"config_digest"`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicValue<F> {
    pub label: &'static str,
    pub value: F,
}

impl<F> PublicValue<F> {
    pub const fn new(label: &'static str, value: F) -> Self {
        Self { label, value }
    }
}
