use crate::pass::support::{string_attr_source, symbol_ref};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeKernelSpec {
    pub symbol: String,
    pub kind: String,
    pub backend: String,
    pub abi: String,
}

impl ComputeKernelSpec {
    pub fn new(
        symbol: impl Into<String>,
        kind: impl Into<String>,
        backend: impl Into<String>,
        abi: impl Into<String>,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            kind: kind.into(),
            backend: backend.into(),
            abi: abi.into(),
        }
    }

    pub(super) fn compute_kernel_attrs(&self, relation: &str) -> Vec<(String, String)> {
        vec![
            ("relation".to_owned(), symbol_ref(relation)),
            ("kind".to_owned(), string_attr_source(&self.kind)),
            ("backend".to_owned(), string_attr_source(&self.backend)),
            ("abi".to_owned(), string_attr_source(&self.abi)),
        ]
    }
}
