use super::*;

pub struct Bn254Add {
    pub inputs: Vec<u64>,
    pub outputs: Vec<u64>,
}

pub struct Bn254Witness {
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

pub struct Bn254AddR1CSProof {} // Todo: Implement this struct for Bn254Add precompile

impl Precompile for Bn254Add {

    type PrecompileWitness = Bn254Witness;
    type PrecompileProof = Bn254AddR1CSProof;

    fn format_io<T: Serialize>(&self, inputs: T, outputs: T) -> PrecompileIO {
        // Todo: Convert `inputs` and `outputs` to the fixed-size arrays.
        unimplemented!()
    }

    fn execute<T: Serialize>(&self, inputs: T) -> Vec<u64> {
        // Todo: Execute your precompile logic using `inputs`
        unimplemented!()
    }

    fn generate_witness(&self, io: PrecompileIO) -> Self::PrecompileWitness {
        // Todo: Generate the witness based on the provided IO.
        unimplemented!()
    }

    fn prove(&self, witness: Self::PrecompileWitness) -> Self::PrecompileProof {
        // Todo: Generate a proof using the witness.
        unimplemented!()
    }

    fn verify(&self, proof: Self::PrecompileProof) -> Result<(), ProofVerifyError> {
        // Todo: Verify the provided proof.
        unimplemented!()
    }
}
