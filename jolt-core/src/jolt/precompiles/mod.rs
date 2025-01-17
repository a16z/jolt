pub mod bn254_add;

pub struct PrecompileIO {
    pub input:[u32; 16] // 512 bits
    pub output:[u32; 16] // 512 bits
}
struct PrecompileWitness {
    pub precompile: Precompile,
    pub a0i: [u8; 32],
    pub a1i: [u8; 32],
    pub a2i: [u8; 32],
    pub a3i: [u8; 32],
    pub a4i: [u8; 32],
    pub a5i: [u8; 32],
    pub a6i: [u8; 32],
    pub a7i: [u8; 32],
    pub a8i: [u8; 32],
    pub a9i: [u8; 32],
    pub a10i: [u8; 32],
    pub a11i: [u8; 32],
    pub a12i: [u8; 32],
    pub a13i: [u8; 32],
    pub a14i: [u8; 32],
    pub a15i: [u8; 32],
    pub a0o: [u8; 32],
    pub a1o: [u8; 32],
    pub a2o: [u8; 32],
    pub a3o: [u8; 32],
    pub a4o: [u8; 32],
    pub a5o: [u8; 32],
    pub a6o: [u8; 32],
    pub a7o: [u8; 32],
    pub a8o: [u8; 32],
    pub a9o: [u8; 32],
    pub a10o: [u8; 32],
    pub a11o: [u8; 32],
    pub a12o: [u8; 32],
    pub a13o: [u8; 32],
    pub a14o: [u8; 32],
    pub a15o: [u8; 32],
}

trait Precompile {
    pub fn format_io(&self, inputs: impl Serialize, outputs: impl Serialize) -> PrecompileIO;
    pub fn execute(&self, inputs: impl Serialize) -> Vec<u64>;
    pub fn generate_witness(&self, io: PrecompileIO) -> PrecompileWitness {
        // This is a placeholder for the actual implementation.
        PrecompileWitness {
            precompile: self,
            a0i: io.input[0],
            a1i: io.input[1],
            a2i: io.input[2],
            a3i: io.input[3],
            a4i: io.input[4],
            a5i: io.input[5],
            a6i: io.input[6],
            a7i: io.input[7],
            a8i: io.input[8],
            a9i: io.input[9],
            a10i: io.input[10],
            a11i: io.input[11],
            a12i: io.input[12],
            a13i: io.input[13],
            a14i: io.input[14],
            a15i: io.input[15],
            a0o: io.output[0],
            a1o: io.output[1],
            a2o: io.output[2],
            a3o: io.output[3],
            a4o: io.output[4],
            a5o: io.output[5],
            a6o: io.output[6],
            a7o: io.output[7],
            a8o: io.output[8],
            a9o: io.output[9],
            a10o: io.output[10],
            a11o: io.output[11],
            a12o: io.output[12],
            a13o: io.output[13],
            a14o: io.output[14],
            a15o: io.output[15],
        }
    };
    pub fn prove(&self, witness: PrecompileWitness) -> JoltProof;  // Need to include Preprocessing struct as an input?
    pub fn verify(&self, proof: JoltProof) -> Result<(), ProofVerifyError>;
}
