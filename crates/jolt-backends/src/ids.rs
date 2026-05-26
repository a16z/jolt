#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BackendRelationId {
    pub namespace: &'static str,
    pub name: &'static str,
}

impl BackendRelationId {
    pub const fn new(namespace: &'static str, name: &'static str) -> Self {
        Self { namespace, name }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BackendValueSlot(pub u32);
