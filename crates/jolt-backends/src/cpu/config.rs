#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuBackendConfig {
    pub preserve_core_fast_path: bool,
    pub commitment_chunk_size: usize,
}

impl Default for CpuBackendConfig {
    fn default() -> Self {
        Self {
            preserve_core_fast_path: false,
            commitment_chunk_size: 1024,
        }
    }
}
