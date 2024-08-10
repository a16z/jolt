use std::borrow::Borrow;

use crate::circuits::poly::commitment::commitment_scheme::CommitmentVerifierGadget;
use crate::field::JoltField;
use crate::poly::commitment::hyperkzg::{
    HyperKZG, HyperKZGCommitment, HyperKZGProof, HyperKZGProverKey, HyperKZGVerifierKey,
};
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
pub struct HyperKZGProofVar<E, ConstraintF>
where
    E: Pairing,
    ConstraintF: PrimeField,
{
    _params: PhantomData<(E, ConstraintF)>,
    // TODO fill in
}

impl<E, ConstraintF> AllocVar<HyperKZGProof<E>, ConstraintF> for HyperKZGProofVar<E, ConstraintF>
where
    E: Pairing,
    ConstraintF: PrimeField,
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
pub struct HyperKZGCommitmentVar<E, ConstraintF>
where
    E: Pairing,
    ConstraintF: PrimeField,
{
    _params: PhantomData<(E, ConstraintF)>,
    // TODO fill in
}

impl<E, ConstraintF> AllocVar<HyperKZGCommitment<E>, ConstraintF>
    for HyperKZGCommitmentVar<E, ConstraintF>
where
    E: Pairing,
    ConstraintF: PrimeField,
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
pub struct HyperKZGVerifierKeyVar<E, ConstraintF>
where
    E: Pairing,
    ConstraintF: PrimeField,
{
    _params: PhantomData<(E, ConstraintF)>,
    // TODO fill in
}

impl<E, ConstraintF> AllocVar<(HyperKZGProverKey<E>, HyperKZGVerifierKey<E>), ConstraintF>
    for HyperKZGVerifierKeyVar<E, ConstraintF>
where
    E: Pairing,
    ConstraintF: PrimeField,
{
    fn new_variable<T: Borrow<(HyperKZGProverKey<E>, HyperKZGVerifierKey<E>)>>(
        cs: impl Into<Namespace<ConstraintF>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        todo!()
    }
}

pub struct HyperKZGVerifierGadget<E, ConstraintF>
where
    E: Pairing,
    ConstraintF: PrimeField + JoltField,
{
    _params: PhantomData<(E, ConstraintF)>,
}

impl<ConstraintF, E> CommitmentVerifierGadget<ConstraintF, HyperKZG<E>>
    for HyperKZGVerifierGadget<E, ConstraintF>
where
    E: Pairing<ScalarField = ConstraintF>,
    ConstraintF: PrimeField + JoltField,
{
    type VerifyingKeyVar = HyperKZGVerifierKeyVar<E, ConstraintF>;
    type ProofVar = HyperKZGProofVar<E, ConstraintF>;
    type CommitmentVar = HyperKZGCommitmentVar<E, ConstraintF>;
    type TranscriptGadget = PoseidonSponge<ConstraintF>;

    fn verify(
        proof: &Self::ProofVar,
        vk: &Self::VerifyingKeyVar,
        transcript: &mut Self::TranscriptGadget,
        opening_point: &[FpVar<ConstraintF>],
        opening: &FpVar<ConstraintF>,
        commitment: &Self::CommitmentVar,
    ) -> Result<Boolean<ConstraintF>, SynthesisError> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::poly::commitment::hyperkzg::{
        HyperKZG, HyperKZGProverKey, HyperKZGSRS, HyperKZGVerifierKey,
    };
    use crate::poly::dense_mlpoly::DensePolynomial;
    use crate::utils::errors::ProofVerifyError;
    use crate::utils::transcript::ProofTranscript;
    use ark_bn254::{Bn254, Fr};
    use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
    use ark_r1cs_std::ToConstraintFieldGadget;
    use ark_relations::ns;
    use ark_std::rand::Rng;
    use rand_core::{CryptoRng, RngCore, SeedableRng};

    use super::*;

    struct HyperKZGVerifierCircuit<E>
    where
        E: Pairing,
    {
        pcs_vk: Option<HyperKZGVerifierKey<E>>,
        commitment: Option<HyperKZGCommitment<E>>,
        point: Option<Vec<E::ScalarField>>,
        eval: Option<E::ScalarField>,
        pcs_proof: Option<HyperKZGProof<E>>,
    }

    impl<E> HyperKZGVerifierCircuit<E>
    where
        E: Pairing,
    {
        pub(crate) fn public_inputs(&self) -> Vec<E::ScalarField> {
            Boolean::<E::ScalarField>::TRUE
                .to_constraint_field()
                .unwrap()
                .iter()
                .map(|x| x.value().unwrap())
                .collect::<Vec<_>>()
        }
    }

    impl<E> ConstraintSynthesizer<E::ScalarField> for HyperKZGVerifierCircuit<E>
    where
        E: Pairing,
    {
        fn generate_constraints(
            self,
            cs: ConstraintSystemRef<E::ScalarField>,
        ) -> Result<(), SynthesisError> {
            // TODO fill in

            let _ = Boolean::new_input(ns!(cs, "verification_result"), || Ok(true))?;

            Ok(())
        }
    }

    #[test]
    fn test_hyperkzg_eval() {
        type Groth16 = ark_groth16::Groth16<Bn254>;

        // Test with poly(X1, X2) = 1 + X1 + X2 + X1*X2
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
        let srs = HyperKZGSRS::setup(&mut rng, 3);
        let (pcs_pk, pcs_vk): (HyperKZGProverKey<Bn254>, HyperKZGVerifierKey<Bn254>) = srs.trim(3);

        // poly is in eval. representation; evaluated at [(0,0), (0,1), (1,0), (1,1)]
        let poly = DensePolynomial::new(vec![
            ark_bn254::Fr::from(1),
            ark_bn254::Fr::from(2),
            ark_bn254::Fr::from(2),
            ark_bn254::Fr::from(4),
        ]);

        let (cpk, cvk) = {
            let circuit = HyperKZGVerifierCircuit::<Bn254> {
                pcs_vk: None,
                commitment: None,
                point: None,
                eval: None,
                pcs_proof: None,
            };

            Groth16::setup(circuit, &mut rng).unwrap()
        };
        let pvk = Groth16::process_vk(&cvk).unwrap();

        let C = HyperKZG::commit(&pcs_pk, &poly).unwrap();

        let test_inner = |point: Vec<Fr>, eval: Fr| -> Result<(), ProofVerifyError> {
            let mut tr = ProofTranscript::new(b"TestEval");
            let hkzg_proof = HyperKZG::open(&pcs_pk, &poly, &point, &eval, &mut tr).unwrap();
            let mut tr = ProofTranscript::new(b"TestEval");
            HyperKZG::verify(&pcs_vk, &C, &point, &eval, &hkzg_proof, &mut tr)?;

            // Create an instance of our circuit (with the
            // witness)
            let verifier_circuit = HyperKZGVerifierCircuit::<Bn254> {
                pcs_vk: Some(pcs_vk.clone()),
                commitment: Some(C.clone()),
                point: Some(point.clone()),
                eval: Some(eval),
                pcs_proof: Some(hkzg_proof),
            };
            let instance = verifier_circuit.public_inputs();

            let mut rng =
                ark_std::rand::rngs::StdRng::seed_from_u64(ark_std::test_rng().next_u64());

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
        let point = vec![ark_bn254::Fr::from(0), ark_bn254::Fr::from(0)];
        let eval = ark_bn254::Fr::from(1);
        assert!(test_inner(point, eval).is_ok());

        let point = vec![ark_bn254::Fr::from(0), ark_bn254::Fr::from(1)];
        let eval = ark_bn254::Fr::from(2);
        assert!(test_inner(point, eval).is_ok());

        let point = vec![ark_bn254::Fr::from(1), ark_bn254::Fr::from(1)];
        let eval = ark_bn254::Fr::from(4);
        assert!(test_inner(point, eval).is_ok());

        let point = vec![ark_bn254::Fr::from(0), ark_bn254::Fr::from(2)];
        let eval = ark_bn254::Fr::from(3);
        assert!(test_inner(point, eval).is_ok());

        let point = vec![ark_bn254::Fr::from(2), ark_bn254::Fr::from(2)];
        let eval = ark_bn254::Fr::from(9);
        assert!(test_inner(point, eval).is_ok());

        // Try a couple incorrect evaluations and expect failure
        let point = vec![ark_bn254::Fr::from(2), ark_bn254::Fr::from(2)];
        let eval = ark_bn254::Fr::from(50);
        assert!(test_inner(point, eval).is_err());

        let point = vec![ark_bn254::Fr::from(0), ark_bn254::Fr::from(2)];
        let eval = ark_bn254::Fr::from(4);
        assert!(test_inner(point, eval).is_err());
    }
}
