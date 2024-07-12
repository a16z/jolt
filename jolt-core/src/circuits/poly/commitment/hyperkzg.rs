use ark_bn254::{Bn254, Fr};
use ark_ff::Field;
// We'll use these interfaces to construct our circuit.
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};

use crate::poly::commitment::hyperkzg::{HyperKZGCommitment, HyperKZGProof, HyperKZGVerifierKey};

#[derive(Default)]
struct HyperKZGVerifierCircuit<F: Field> {
    _f: std::marker::PhantomData<F>,
    // TODO fill in
}

impl<F: Field> HyperKZGVerifierCircuit<F> {
    pub(crate) fn public_inputs(
        &self,
        vk: &HyperKZGVerifierKey<Bn254>,
        comm: &HyperKZGCommitment<Bn254>,
        point: &Vec<Fr>,
        eval: &Fr,
        proof: &HyperKZGProof<Bn254>,
    ) -> Vec<F> {
        // TODO fill in
        vec![]
    }
}

impl<F: Field> ConstraintSynthesizer<F> for HyperKZGVerifierCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        // TODO fill in
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::{Bn254, Fr};
    use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
    use rand_core::SeedableRng;

    use crate::poly::commitment::hyperkzg::{
        HyperKZG, HyperKZGProverKey, HyperKZGSRS, HyperKZGVerifierKey,
    };
    use crate::poly::dense_mlpoly::DensePolynomial;
    use crate::utils::errors::ProofVerifyError;
    use crate::utils::transcript::ProofTranscript;

    use super::*;

    #[test]
    fn test_hyperkzg_eval() {
        type Groth16 = ark_groth16::Groth16<Bn254>;

        // Test with poly(X1, X2) = 1 + X1 + X2 + X1*X2
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
        let srs = HyperKZGSRS::setup(&mut rng, 3);
        let (pk, vk): (HyperKZGProverKey<Bn254>, HyperKZGVerifierKey<Bn254>) = srs.trim(3);

        // poly is in eval. representation; evaluated at [(0,0), (0,1), (1,0), (1,1)]
        let poly = DensePolynomial::new(vec![Fr::from(1), Fr::from(2), Fr::from(2), Fr::from(4)]);

        let (cpk, cvk) = {
            let circuit = HyperKZGVerifierCircuit::default();

            Groth16::setup(circuit, &mut rng).unwrap()
        };
        let pvk = Groth16::process_vk(&cvk).unwrap();

        let C = HyperKZG::commit(&pk, &poly).unwrap();

        let mut test_inner = |point: Vec<Fr>, eval: Fr| -> Result<(), ProofVerifyError> {
            let mut tr = ProofTranscript::new(b"TestEval");
            let proof = HyperKZG::open(&pk, &poly, &point, &eval, &mut tr).unwrap();
            let mut tr = ProofTranscript::new(b"TestEval");
            HyperKZG::verify(&vk, &C, &point, &eval, &proof, &mut tr)?;

            // Create an instance of our circuit (with the
            // witness)
            let verifier_circuit = HyperKZGVerifierCircuit::default();
            let instance = verifier_circuit.public_inputs(&vk, &C, &point, &eval, &proof);

            // Create a groth16 proof with our parameters.
            let proof = Groth16::prove(&cpk, verifier_circuit, &mut rng)
                .map_err(|e| ProofVerifyError::InternalError)?;
            let result = Groth16::verify_with_processed_vk(&pvk, &instance, &proof);
            match result {
                Ok(true) => Ok(()),
                Ok(false) => Err(ProofVerifyError::InternalError),
                Err(_) => Err(ProofVerifyError::InternalError),
            }
        };

        // Call the prover with a (point, eval) pair.
        // The prover does not recompute so it may produce a proof, but it should not verify
        let point = vec![Fr::from(0), Fr::from(0)];
        let eval = Fr::from(1);
        assert!(test_inner(point, eval).is_ok());

        let point = vec![Fr::from(0), Fr::from(1)];
        let eval = Fr::from(2);
        assert!(test_inner(point, eval).is_ok());

        let point = vec![Fr::from(1), Fr::from(1)];
        let eval = Fr::from(4);
        assert!(test_inner(point, eval).is_ok());

        let point = vec![Fr::from(0), Fr::from(2)];
        let eval = Fr::from(3);
        assert!(test_inner(point, eval).is_ok());

        let point = vec![Fr::from(2), Fr::from(2)];
        let eval = Fr::from(9);
        assert!(test_inner(point, eval).is_ok());

        // Try a couple incorrect evaluations and expect failure
        let point = vec![Fr::from(2), Fr::from(2)];
        let eval = Fr::from(50);
        assert!(test_inner(point, eval).is_err());

        let point = vec![Fr::from(0), Fr::from(2)];
        let eval = Fr::from(4);
        assert!(test_inner(point, eval).is_err());
    }
}
