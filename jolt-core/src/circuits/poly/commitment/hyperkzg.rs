use std::borrow::Borrow;

use crate::circuits::pairing::PairingGadget;
use crate::circuits::poly::commitment::commitment_scheme::CommitmentVerifierGadget;
use crate::field::JoltField;
use crate::poly::commitment::hyperkzg::{
    HyperKZG, HyperKZGCommitment, HyperKZGProof, HyperKZGProverKey, HyperKZGVerifierKey,
};
use ark_bn254::{Bn254, Fr as BN254Fr};
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use ark_ec::pairing::Pairing;
use ark_ff::{Field, PrimeField};
use ark_r1cs_std::boolean::Boolean;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::pairing::PairingVar;
use ark_r1cs_std::prelude::*;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, Namespace, SynthesisError};
use ark_std::marker::PhantomData;

#[derive(Clone)]
pub struct HyperKZGProofVar<E: Pairing, ConstraintF: PrimeField, P: PairingGadget<E, ConstraintF>> {
    _e: PhantomData<E>,
    _p: PhantomData<P>,
    _constraint_f: PhantomData<ConstraintF>,
    // TODO fill in
}

impl<E, ConstraintF, P> AllocVar<HyperKZGProof<E>, ConstraintF>
    for HyperKZGProofVar<E, ConstraintF, P>
where
    E: Pairing,
    ConstraintF: PrimeField,
    P: PairingGadget<E, ConstraintF>,
{
    fn new_variable<T: Borrow<HyperKZGProof<E>>>(
        cs: impl Into<Namespace<ConstraintF>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        todo!()
    }
}

#[derive(Clone)]
pub struct HyperKZGCommitmentVar<
    E: Pairing,
    ConstraintF: PrimeField,
    P: PairingGadget<E, ConstraintF>,
> {
    _e: PhantomData<E>,
    _p: PhantomData<P>,
    _constraint_f: PhantomData<ConstraintF>,
    // TODO fill in
}

impl<E, ConstraintF, P> AllocVar<HyperKZGCommitment<E>, ConstraintF>
    for HyperKZGCommitmentVar<E, ConstraintF, P>
where
    E: Pairing,
    ConstraintF: PrimeField,
    P: PairingGadget<E, ConstraintF>,
{
    fn new_variable<T: Borrow<HyperKZGCommitment<E>>>(
        cs: impl Into<Namespace<ConstraintF>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        todo!()
    }
}

#[derive(Clone)]
pub struct HyperKZGVerifierKeyVar<
    E: Pairing,
    ConstraintF: PrimeField,
    P: PairingGadget<E, ConstraintF>,
> {
    _e: PhantomData<E>,
    _p: PhantomData<P>,
    _constraint_f: PhantomData<ConstraintF>,
    // TODO fill in
}

impl<E: Pairing, P: PairingGadget<E, ConstraintF>, ConstraintF: PrimeField>
    AllocVar<(HyperKZGProverKey<E>, HyperKZGVerifierKey<E>), ConstraintF>
    for HyperKZGVerifierKeyVar<E, ConstraintF, P>
{
    fn new_variable<T: Borrow<(HyperKZGProverKey<E>, HyperKZGVerifierKey<E>)>>(
        cs: impl Into<Namespace<ConstraintF>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        todo!()
    }
}

pub struct HyperKZGVerifierGadget<E, P, ConstraintF: PrimeField>
where
    E: Pairing,
    P: PairingGadget<E, ConstraintF>,
{
    _e: PhantomData<E>,
    _p: PhantomData<P>,
    _constraint_f: PhantomData<ConstraintF>,
}

impl<F, ConstraintF, E, P> CommitmentVerifierGadget<F, ConstraintF, HyperKZG<E>>
    for HyperKZGVerifierGadget<E, P, ConstraintF>
where
    E: Pairing<ScalarField = F>,
    P: PairingGadget<E, ConstraintF> + Clone,
    ConstraintF: PrimeField,
    F: PrimeField + JoltField,
{
    type VerifyingKeyVar = HyperKZGVerifierKeyVar<E, ConstraintF, P>;
    type ProofVar = HyperKZGProofVar<E, ConstraintF, P>;
    type CommitmentVar = HyperKZGCommitmentVar<E, ConstraintF, P>;
    type TranscriptGadget = PoseidonSponge<F>;

    fn verify(
        proof: &Self::ProofVar,
        vk: &Self::VerifyingKeyVar,
        transcript: &mut Self::TranscriptGadget,
        opening_point: &[FpVar<F>],
        opening: &FpVar<F>,
        commitment: &Self::CommitmentVar,
    ) -> Result<Boolean<ConstraintF>, SynthesisError> {
        todo!()
    }
}

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
        point: &Vec<BN254Fr>,
        eval: &BN254Fr,
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
    use ark_bls12_381::Bls12_381;
    use ark_bn254::{Bn254, Fr as BN254Fr};
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
        type Groth16 = ark_groth16::Groth16<Bls12_381>;

        // Test with poly(X1, X2) = 1 + X1 + X2 + X1*X2
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
        let srs = HyperKZGSRS::setup(&mut rng, 3);
        let (pk, vk): (HyperKZGProverKey<Bn254>, HyperKZGVerifierKey<Bn254>) = srs.trim(3);

        // poly is in eval. representation; evaluated at [(0,0), (0,1), (1,0), (1,1)]
        let poly = DensePolynomial::new(vec![
            BN254Fr::from(1),
            BN254Fr::from(2),
            BN254Fr::from(2),
            BN254Fr::from(4),
        ]);

        let (cpk, cvk) = {
            let circuit = HyperKZGVerifierCircuit::default();

            Groth16::setup(circuit, &mut rng).unwrap()
        };
        let pvk = Groth16::process_vk(&cvk).unwrap();

        let C = HyperKZG::commit(&pk, &poly).unwrap();

        let mut test_inner = |point: Vec<BN254Fr>, eval: BN254Fr| -> Result<(), ProofVerifyError> {
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
        let point = vec![BN254Fr::from(0), BN254Fr::from(0)];
        let eval = BN254Fr::from(1);
        assert!(test_inner(point, eval).is_ok());

        let point = vec![BN254Fr::from(0), BN254Fr::from(1)];
        let eval = BN254Fr::from(2);
        assert!(test_inner(point, eval).is_ok());

        let point = vec![BN254Fr::from(1), BN254Fr::from(1)];
        let eval = BN254Fr::from(4);
        assert!(test_inner(point, eval).is_ok());

        let point = vec![BN254Fr::from(0), BN254Fr::from(2)];
        let eval = BN254Fr::from(3);
        assert!(test_inner(point, eval).is_ok());

        let point = vec![BN254Fr::from(2), BN254Fr::from(2)];
        let eval = BN254Fr::from(9);
        assert!(test_inner(point, eval).is_ok());

        // Try a couple incorrect evaluations and expect failure
        let point = vec![BN254Fr::from(2), BN254Fr::from(2)];
        let eval = BN254Fr::from(50);
        assert!(test_inner(point, eval).is_err());

        let point = vec![BN254Fr::from(0), BN254Fr::from(2)];
        let eval = BN254Fr::from(4);
        assert!(test_inner(point, eval).is_err());
    }
}
