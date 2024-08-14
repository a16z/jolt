use crate::circuits::poly::commitment::commitment_scheme::CommitmentVerifierGadget;
use crate::circuits::transcript::ImplAbsorb;
use crate::field::JoltField;
use crate::poly::commitment::hyperkzg::{
    HyperKZG, HyperKZGCommitment, HyperKZGProof, HyperKZGProverKey, HyperKZGVerifierKey,
};
use ark_crypto_primitives::sponge::constraints::{
    AbsorbGadget, CryptographicSpongeVar, SpongeWithGadget,
};
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ec::pairing::Pairing;
use ark_ff::{Field, PrimeField};
use ark_r1cs_std::boolean::Boolean;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::pairing::PairingVar;
use ark_r1cs_std::prelude::*;
use ark_r1cs_std::ToConstraintFieldGadget;
use ark_relations::ns;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, Namespace, SynthesisError};
use ark_std::borrow::Borrow;
use ark_std::iterable::Iterable;
use ark_std::marker::PhantomData;

#[derive(Clone)]
pub struct HyperKZGProofVar<E, G1Var>
where
    E: Pairing,
    G1Var: CurveVar<E::G1, E::ScalarField>,
{
    pub com: Vec<G1Var>,
    pub w: Vec<G1Var>,
    pub v: Vec<Vec<FpVar<E::ScalarField>>>,
}

impl<E, G1Var> AllocVar<HyperKZGProof<E>, E::ScalarField> for HyperKZGProofVar<E, G1Var>
where
    E: Pairing<ScalarField: PrimeField>,
    G1Var: CurveVar<E::G1, E::ScalarField>,
{
    fn new_variable<T: Borrow<HyperKZGProof<E>>>(
        cs: impl Into<Namespace<E::ScalarField>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        let ns = cs.into();
        let cs = ns.cs();

        let proof_hold = f()?;
        let proof = proof_hold.borrow();

        let com = proof
            .com
            .iter()
            .map(|&x| G1Var::new_variable(ns!(cs, "com").clone(), || Ok(x), mode))
            .collect::<Result<Vec<_>, _>>()?;
        let w = proof
            .w
            .iter()
            .map(|&x| G1Var::new_variable(ns!(cs, "w").clone(), || Ok(x), mode))
            .collect::<Result<Vec<_>, _>>()?;
        let v = proof
            .v
            .iter()
            .map(|v_i| {
                v_i.iter()
                    .map(|&v_ij| FpVar::new_variable(ns!(cs, "v_ij"), || Ok(v_ij), mode))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self { com, w, v })
    }
}

#[derive(Clone, Debug)]
pub struct HyperKZGCommitmentVar<G1Var> {
    pub c: G1Var,
}

impl<E, G1Var> AllocVar<HyperKZGCommitment<E>, E::ScalarField> for HyperKZGCommitmentVar<G1Var>
where
    E: Pairing<ScalarField: PrimeField>,
    G1Var: CurveVar<E::G1, E::ScalarField>,
{
    fn new_variable<T: Borrow<HyperKZGCommitment<E>>>(
        cs: impl Into<Namespace<E::ScalarField>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        Ok(Self {
            c: G1Var::new_variable(cs, || Ok(f()?.borrow().0), mode)?,
        })
    }
}

#[derive(Clone, Debug)]
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
        // TODO implement
        Ok(Self {
            _params: PhantomData,
        })
    }
}

pub struct HyperKZGVerifierGadget<E, S, G1Var>
where
    E: Pairing<ScalarField: PrimeField + JoltField>,
    S: SpongeWithGadget<E::ScalarField>,
    G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
{
    _params: PhantomData<(E, S, G1Var)>,
}

impl<ConstraintF, E, S, G1Var> CommitmentVerifierGadget<ConstraintF, HyperKZG<E>, S>
    for HyperKZGVerifierGadget<E, S, G1Var>
where
    E: Pairing<ScalarField = ConstraintF>,
    ConstraintF: PrimeField + JoltField,
    S: SpongeWithGadget<ConstraintF>,
    G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
{
    type VerifyingKeyVar = HyperKZGVerifierKeyVar<E, ConstraintF>;
    type ProofVar = HyperKZGProofVar<E, G1Var>;
    type CommitmentVar = HyperKZGCommitmentVar<G1Var>;

    fn verify(
        proof: &Self::ProofVar,
        vk: &Self::VerifyingKeyVar,
        transcript: &mut S::Var,
        opening_point: &[FpVar<ConstraintF>],
        opening: &FpVar<ConstraintF>,
        commitment: &Self::CommitmentVar,
    ) -> Result<Boolean<ConstraintF>, SynthesisError> {
        let ell = opening_point.len();

        transcript.absorb(
            &proof
                .com
                .iter()
                .map(|com| ImplAbsorb::wrap(com))
                .collect::<Vec<_>>(),
        )?;

        let r = transcript
            .squeeze_field_elements(1)?
            .into_iter()
            .next()
            .unwrap();

        let u = vec![r.clone(), r.negate()?, r.clone() * &r];

        let com = [vec![commitment.c.clone()], proof.com.clone()].concat();

        let v = &proof.v;
        if v.len() != 3 {
            return Err(SynthesisError::Unsatisfiable);
        }
        if ell != v[0].len() || ell != v[1].len() || ell != v[2].len() {
            return Err(SynthesisError::Unsatisfiable);
        }

        let x = opening_point;
        let y = [v[2].clone(), vec![opening.clone()]].concat();

        let one = FpVar::Constant(E::ScalarField::one());
        let two = FpVar::Constant(E::ScalarField::from(2u128));
        for i in 0..ell {
            (&two * &r * &y[i + 1]).enforce_equal(
                &(&r * (&one - &x[ell - i - 1]) * (&v[0][i] + &v[1][i])
                    + &x[ell - i - 1] * (&v[0][i] - &v[1][i])),
            )?;
        }

        dbg!();

        // TODO implement
        Ok(Boolean::TRUE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits::groups::curves::short_weierstrass::bn254::G1Var;
    use crate::circuits::transcript::mock::{MockSponge, MockSpongeVar};
    use crate::poly::commitment::hyperkzg::{
        HyperKZG, HyperKZGProverKey, HyperKZGSRS, HyperKZGVerifierKey,
    };
    use crate::poly::dense_mlpoly::DensePolynomial;
    use crate::utils::errors::ProofVerifyError;
    use crate::utils::transcript::ProofTranscript;
    use ark_bn254::Bn254;
    use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
    use ark_crypto_primitives::sponge::constraints::CryptographicSpongeVar;
    use ark_crypto_primitives::sponge::poseidon::constraints::PoseidonSpongeVar;
    use ark_crypto_primitives::sponge::poseidon::{PoseidonConfig, PoseidonDefaultConfigField};
    use ark_r1cs_std::ToConstraintFieldGadget;
    use ark_relations::ns;
    use ark_std::rand::Rng;
    use ark_std::Zero;
    use rand_core::{CryptoRng, RngCore, SeedableRng};

    #[derive(Debug)]
    struct HyperKZGVerifierCircuit<E, G1Var>
    where
        E: Pairing,
        G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
    {
        _params: PhantomData<G1Var>,
        pcs_pk_vk: Option<(HyperKZGProverKey<E>, HyperKZGVerifierKey<E>)>,
        commitment: Option<HyperKZGCommitment<E>>,
        point: Vec<Option<E::ScalarField>>,
        eval: Option<E::ScalarField>,
        pcs_proof: HyperKZGProof<E>,
        expected_result: Option<bool>,
    }

    impl<E, G1Var> HyperKZGVerifierCircuit<E, G1Var>
    where
        E: Pairing,
        G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
    {
        pub(crate) fn public_inputs(&self) -> Vec<E::ScalarField> {
            Boolean::<E::ScalarField>::constant(self.expected_result.unwrap()) // panics if None
                .to_constraint_field()
                .unwrap()
                .iter()
                .map(|x| x.value().unwrap())
                .collect::<Vec<_>>()
        }
    }

    impl<E, G1Var> ConstraintSynthesizer<E::ScalarField> for HyperKZGVerifierCircuit<E, G1Var>
    where
        E: Pairing<ScalarField: PrimeField + JoltField>,
        G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
    {
        fn generate_constraints(
            self,
            cs: ConstraintSystemRef<E::ScalarField>,
        ) -> Result<(), SynthesisError> {
            let vk_var =
                HyperKZGVerifierKeyVar::<E, E::ScalarField>::new_witness(ns!(cs, "vk"), || {
                    self.pcs_pk_vk.ok_or(SynthesisError::AssignmentMissing)
                })?;

            let commitment_var =
                HyperKZGCommitmentVar::<G1Var>::new_witness(ns!(cs, "commitment"), || {
                    self.commitment.ok_or(SynthesisError::AssignmentMissing)
                })?;

            let point_var = self
                .point
                .iter()
                .map(|&x| {
                    FpVar::new_witness(ns!(cs, ""), || x.ok_or(SynthesisError::AssignmentMissing))
                })
                .collect::<Result<Vec<_>, _>>()?;

            let eval_var = FpVar::<E::ScalarField>::new_witness(ns!(cs, "eval"), || {
                self.eval.ok_or(SynthesisError::AssignmentMissing)
            })?;

            let proof_var =
                HyperKZGProofVar::<E, G1Var>::new_witness(ns!(cs, "proof"), || Ok(self.pcs_proof))?;

            let mut transcript_var =
                MockSpongeVar::new(ns!(cs, "transcript").cs(), &(b"TestEval".as_slice()));

            let r = HyperKZGVerifierGadget::<E, MockSponge<E::ScalarField>, G1Var>::verify(
                &proof_var,
                &vk_var,
                &mut transcript_var,
                &point_var,
                &eval_var,
                &commitment_var,
            )?;

            let r_input = Boolean::new_input(ns!(cs, "verification_result"), || {
                self.expected_result
                    .ok_or(SynthesisError::AssignmentMissing)
            })?;
            r.enforce_equal(&r_input)?;

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

        let size = 2usize;
        let (cpk, cvk) = {
            let circuit = HyperKZGVerifierCircuit::<Bn254, G1Var> {
                _params: PhantomData,
                pcs_pk_vk: None,
                commitment: None,
                point: vec![None; size],
                eval: None,
                pcs_proof: HyperKZGProof::empty(size),
                expected_result: None,
            };

            Groth16::setup(circuit, &mut rng).unwrap()
        };
        let pvk = Groth16::process_vk(&cvk).unwrap();

        let C = HyperKZG::commit(&pcs_pk, &poly).unwrap();

        let test_inner =
            |point: Vec<ark_bn254::Fr>, eval: ark_bn254::Fr| -> Result<(), ProofVerifyError> {
                let mut tr = ProofTranscript::new(b"TestEval");
                let hkzg_proof = HyperKZG::open(&pcs_pk, &poly, &point, &eval, &mut tr).unwrap();

                println!("Verifying natively...");

                let mut tr = ProofTranscript::new(b"TestEval");
                HyperKZG::verify(&pcs_vk, &C, &point, &eval, &hkzg_proof, &mut tr)?;

                // Create an instance of our circuit (with the
                // witness)
                let verifier_circuit = HyperKZGVerifierCircuit::<Bn254, G1Var> {
                    _params: PhantomData,
                    pcs_pk_vk: Some((pcs_pk.clone(), pcs_vk.clone())),
                    commitment: Some(C.clone()),
                    point: point.into_iter().map(|x| Some(x)).collect(),
                    eval: Some(eval),
                    pcs_proof: hkzg_proof,
                    expected_result: Some(true),
                };
                let instance = verifier_circuit.public_inputs();

                let mut rng =
                    ark_std::rand::rngs::StdRng::seed_from_u64(ark_std::test_rng().next_u64());

                println!("Verifying in-circuit...");

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
