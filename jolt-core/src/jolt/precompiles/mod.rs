pub mod bn254_add;

pub struct PrecompileIO {
    pub input:[u32; 16] // 512 bits
    pub output:[u32; 16] // 512 bits
}

trait Precompile {
    
    type PrecompileWitness;

    type PrecompileProof;

    fn format_io<T: Serialize>(&self, inputs: T, outputs: T) -> PrecompileIO;
    fn execute<T: Serialize>(&self, inputs: T) -> Vec<u64>;
    fn generate_witness(&self, io: PrecompileIO) -> Self::PrecompileWitness;
    fn prove(&self, witness: Self::PrecompileWitness) -> Self::PrecompileProof;  // Need to include Preprocessing struct as an input?
    fn verify(&self, proof: Self::PrecompileProof) -> Result<(), ProofVerifyError>;
}
