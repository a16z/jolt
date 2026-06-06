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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BackendKernelMetadata {
    pub relation: Option<BackendRelationId>,
    pub optimization_ids: &'static [&'static str],
}

impl BackendKernelMetadata {
    pub const fn new(
        relation: Option<BackendRelationId>,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        Self {
            relation,
            optimization_ids,
        }
    }

    pub const fn empty() -> Self {
        Self::new(None, &[])
    }

    pub const fn with_relation(mut self, relation: BackendRelationId) -> Self {
        self.relation = Some(relation);
        self
    }

    pub const fn with_optimization_ids(
        mut self,
        optimization_ids: &'static [&'static str],
    ) -> Self {
        self.optimization_ids = optimization_ids;
        self
    }
}

impl Default for BackendKernelMetadata {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BackendValueSlot(pub u32);
