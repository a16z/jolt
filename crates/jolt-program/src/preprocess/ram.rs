#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RAMPreprocessing {
    pub memory_init: Vec<(u64, u8)>,
}
