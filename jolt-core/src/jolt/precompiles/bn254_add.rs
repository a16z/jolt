use super::*;

pub struct Bn254Add {
    pub inputs: Vec<u64>,
    pub outputs: Vec<u64>,
}

impl Precompile for Bn254Add {
    pub fn format_io(inputs: Vec<u64>, outputs: Vec<u64>) -> PrecompileIO {
        // Todo();
    }
    pub fn generate_witness(io: PrecompileIO) -> PrecompileWitness {
        // Todo();
    }
    pub fn execute(inputs: Vec<u64>) -> Vec<u64> {
        // Todo();
        // import from common/precompiles/bn254_add.rs
    }
    pub fn prove(witness: PrecompileWitness) -> JoltProof {
        // Todo();
    }
    pub fn verify(proof: JoltProof) -> Result<(), ProofVerifyError> {
        // Todo();
    }
}
