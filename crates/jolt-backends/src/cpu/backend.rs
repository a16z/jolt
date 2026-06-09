use crate::Backend;

use super::CpuBackendConfig;

#[derive(Clone, Debug, Default)]
pub struct CpuBackend {
    pub config: CpuBackendConfig,
}

impl CpuBackend {
    pub const fn new(config: CpuBackendConfig) -> Self {
        Self { config }
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }
}
