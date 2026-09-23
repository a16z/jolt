use super::GuestConfig;

/// BTreeMap guest: performs `self.0` map operations.
pub struct BTreeMapOps(pub u32);

impl Default for BTreeMapOps {
    fn default() -> Self {
        // e2e_profiling.rs default
        Self(50)
    }
}

impl GuestConfig for BTreeMapOps {
    fn package(&self) -> &str {
        "btreemap-guest"
    }
    fn label(&self) -> String {
        format!("btreemap_{}", self.0)
    }
    fn input(&self) -> Vec<u8> {
        postcard::to_stdvec(&self.0).unwrap()
    }
}
