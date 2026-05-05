#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolCrateRef {
    pub package: String,
    pub import: String,
}

impl ProtocolCrateRef {
    pub fn new(package: impl Into<String>, import: impl Into<String>) -> Self {
        Self {
            package: package.into(),
            import: import.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RustTypeRef {
    pub path: String,
}

impl RustTypeRef {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }

    pub(in crate::emit::rust::artifacts) fn ident(&self) -> &str {
        self.path.rsplit("::").next().unwrap_or(&self.path)
    }

    pub(in crate::emit::rust::artifacts) fn use_line(&self) -> String {
        format!("use {};\n", self.path)
    }
}
