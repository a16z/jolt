use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WrapperProof<BackendProof> {
    pub backend_proof: BackendProof,
}

impl<BackendProof> WrapperProof<BackendProof> {
    pub fn new(backend_proof: BackendProof) -> Self {
        Self { backend_proof }
    }
}
