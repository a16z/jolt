use crate::circuits::transcript::ImplAbsorbGVar;
use crate::{
    circuits::{
        offloaded::{MSMGadget, OffloadedMSMGadget, OffloadedPairingGadget, PairingGadget},
        poly::commitment::commitment_scheme::CommitmentVerifierGadget,
        transcript::ImplAbsorbFVar,
    },
    field::JoltField,
    poly::commitment::hyperkzg::{
        HyperKZG, HyperKZGCommitment, HyperKZGProof, HyperKZGProverKey, HyperKZGVerifierKey,
    },
    snark::OffloadedDataCircuit,
};
use ark_crypto_primitives::sponge::constraints::{CryptographicSpongeVar, SpongeWithGadget};
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_r1cs_std::{boolean::Boolean, fields::fp::FpVar, prelude::*, ToConstraintFieldGadget};
use ark_relations::r1cs::ConstraintSystemRef;
use ark_relations::{
    ns,
    r1cs::{Namespace, SynthesisError},
};
use ark_std::{borrow::Borrow, iterable::Iterable, marker::PhantomData, One};

#[derive(Clone)]
pub struct HyperKZGProofVar<E, G1Var>
where
    E: Pairing,
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
pub struct HyperKZGVerifierKeyVar<G1Var> {
    pub g1: G1Var,
    // pub g2: G2Var,
    // pub beta_g2: G2Var,
}

impl<E, G1Var> AllocVar<(HyperKZGProverKey<E>, HyperKZGVerifierKey<E>), E::ScalarField>
    for HyperKZGVerifierKeyVar<G1Var>
where
    E: Pairing,
    G1Var: CurveVar<E::G1, E::ScalarField>,
{
    fn new_variable<T: Borrow<(HyperKZGProverKey<E>, HyperKZGVerifierKey<E>)>>(
        cs: impl Into<Namespace<E::ScalarField>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        // TODO implement
        Ok(Self {
            g1: G1Var::new_variable(cs, || Ok(f()?.borrow().1.kzg_vk.g1), mode)?,
        })
    }
}

pub struct HyperKZGVerifierGadget<'a, E, S, G1Var, Circuit>
where
    E: Pairing<ScalarField: PrimeField + JoltField>,
    S: SpongeWithGadget<E::ScalarField>,
    G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
    Circuit: OffloadedDataCircuit<E>,
{
    _params: PhantomData<(E, S, G1Var)>,
    circuit: &'a Circuit,
    cs: ConstraintSystemRef<E::ScalarField>,
    g2_elements: Vec<E::G2Affine>,
}

impl<'a, E, S, G1Var, Circuit> HyperKZGVerifierGadget<'a, E, S, G1Var, Circuit>
where
    E: Pairing<ScalarField: PrimeField + JoltField>,
    S: SpongeWithGadget<E::ScalarField>,
    G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
    Circuit: OffloadedDataCircuit<E>,
{
    pub fn new(
        circuit: &'a Circuit,
        cs: impl Into<Namespace<E::ScalarField>>,
        g2_elements: Vec<E::G2Affine>,
    ) -> Self {
        let ns = cs.into();
        let cs: ConstraintSystemRef<E::ScalarField> = ns.cs();
        Self {
            _params: PhantomData,
            circuit,
            cs,
            g2_elements,
        }
    }
}

impl<'a, E, S, F, G1Var, Circuit> CommitmentVerifierGadget<E::ScalarField, HyperKZG<E>, S>
    for HyperKZGVerifierGadget<'a, E, S, G1Var, Circuit>
where
    F: PrimeField + JoltField,
    E: Pairing<ScalarField = F>,
    S: SpongeWithGadget<F>,
    G1Var: CurveVar<E::G1, F> + ToConstraintFieldGadget<F>,
    Circuit: OffloadedDataCircuit<E>,
{
    type VerifyingKeyVar = HyperKZGVerifierKeyVar<G1Var>;
    type ProofVar = HyperKZGProofVar<E, G1Var>;
    type CommitmentVar = HyperKZGCommitmentVar<G1Var>;

    fn verify(
        &self,
        proof: &Self::ProofVar,
        vk: &Self::VerifyingKeyVar,
        transcript: &mut S::Var,
        opening_point: &[FpVar<F>],
        opening: &FpVar<F>,
        commitment: &Self::CommitmentVar,
    ) -> Result<Boolean<F>, SynthesisError> {
        let ell = opening_point.len();
        assert!(ell >= 2);

        let HyperKZGProofVar { com, w, v } = proof;
        let HyperKZGCommitmentVar { c } = commitment;
        let HyperKZGVerifierKeyVar { g1 } = vk;

        transcript.absorb(
            &com.iter()
                .map(|com| ImplAbsorbGVar::wrap(com))
                .collect::<Vec<_>>(),
        )?;

        let r = transcript
            .squeeze_field_elements(1)?
            .into_iter()
            .next()
            .unwrap();

        let u = [r.clone(), r.negate()?, r.clone() * &r];

        let com = [vec![c.clone()], com.clone()].concat();

        if v.len() != 3 {
            return Err(SynthesisError::Unsatisfiable);
        }
        if w.len() != 3 {
            return Err(SynthesisError::Unsatisfiable);
        }
        if ell != v[0].len() || ell != v[1].len() || ell != v[2].len() || ell != com.len() {
            return Err(SynthesisError::Unsatisfiable);
        }

        let x = opening_point;
        let y = [v[2].clone(), vec![opening.clone()]].concat();

        let one = FpVar::one();
        let two = FpVar::Constant(F::from(2u128));
        for i in 0..ell {
            (&two * &r * &y[i + 1]).enforce_equal(
                &(&r * (&one - &x[ell - i - 1]) * (&v[0][i] + &v[1][i])
                    + &x[ell - i - 1] * (&v[0][i] - &v[1][i])),
            )?;
        }

        // kzg_verify_batch

        transcript.absorb(
            &v.iter()
                .flatten()
                .map(|v_ij| ImplAbsorbFVar::wrap(v_ij))
                .collect::<Vec<_>>(),
        )?;
        let q_powers = q_powers::<E, S>(transcript, ell)?;

        transcript.absorb(
            &w.iter()
                .map(|g| ImplAbsorbGVar::wrap(g))
                .collect::<Vec<_>>(),
        )?;
        let d = transcript
            .squeeze_field_elements(1)?
            .into_iter()
            .next()
            .unwrap();

        let d_square = d.square()?;
        let q_power_multiplier = one + &d + &d_square;
        let q_powers_multiplied = q_powers
            .iter()
            .map(|q_i| q_i * &q_power_multiplier)
            .collect::<Vec<_>>();

        let b_u = v
            .iter()
            .map(|v_i| {
                let mut b_u_i = v_i[0].clone();
                for i in 1..ell {
                    b_u_i += &q_powers[i] * &v_i[i];
                }
                b_u_i
            })
            .collect::<Vec<_>>();

        let msm_gadget = OffloadedMSMGadget::<FpVar<F>, E, G1Var, Circuit>::new(self.circuit);
        let pairing_gadget =
            OffloadedPairingGadget::<E, FpVar<F>, G1Var, Circuit>::new(self.circuit);

        let l_g1s = &[com.as_slice(), w.as_slice(), &[g1.clone()]].concat();
        let l_scalars = &[
            q_powers_multiplied.as_slice(),
            &[
                u[0].clone(),
                &u[1] * &d,
                &u[2] * &d_square,
                (&b_u[0] + &d * &b_u[1] + &d_square * &b_u[2]).negate()?,
            ],
        ]
        .concat();
        debug_assert_eq!(l_g1s.len(), l_scalars.len());

        let l_g1 = msm_gadget.msm(ns!(self.cs, "l_g1"), l_g1s, l_scalars)?;

        let r_g1s = w.as_slice();
        let r_scalars = &[FpVar::one().negate()?, d.negate()?, d_square.negate()?];
        debug_assert_eq!(r_g1s.len(), r_scalars.len());

        let r_g1 = msm_gadget.msm(ns!(self.cs, "r_g1"), r_g1s, r_scalars)?;

        pairing_gadget.multi_pairing_is_zero(
            ns!(self.cs, "multi_pairing"),
            &[l_g1, r_g1],
            self.g2_elements.as_slice(),
        )?;
        dbg!();

        Ok(Boolean::TRUE)
    }
}

fn q_powers<E: Pairing, S: SpongeWithGadget<E::ScalarField>>(
    transcript: &mut S::Var,
    ell: usize,
) -> Result<Vec<FpVar<E::ScalarField>>, SynthesisError> {
    let q = transcript
        .squeeze_field_elements(1)?
        .into_iter()
        .next()
        .unwrap();

    let q_powers = [vec![FpVar::Constant(E::ScalarField::one()), q.clone()], {
        let mut q_power = q.clone();
        (2..ell)
            .map(|_i| {
                q_power *= &q;
                q_power.clone()
            })
            .collect()
    }]
    .concat();
    Ok(q_powers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuits::{
            groups::curves::short_weierstrass::bn254::G1Var,
            transcript::mock::{MockSponge, MockSpongeVar},
        },
        poly::{
            commitment::hyperkzg::{HyperKZG, HyperKZGProverKey, HyperKZGSRS, HyperKZGVerifierKey},
            dense_mlpoly::DensePolynomial,
        },
        snark::{DeferredFnsRef, OffloadedDataCircuit, OffloadedSNARK},
        utils::{errors::ProofVerifyError, transcript::ProofTranscript},
    };
    use ark_bn254::Bn254;
    use ark_crypto_primitives::{snark::SNARK, sponge::constraints::CryptographicSpongeVar};
    use ark_ec::short_weierstrass::{Affine, SWCurveConfig};
    use ark_r1cs_std::ToConstraintFieldGadget;
    use ark_relations::{
        ns,
        r1cs::{ConstraintSynthesizer, ConstraintSystemRef},
    };
    use rand_core::{RngCore, SeedableRng};

    #[derive(Clone)]
    struct HyperKZGVerifierCircuit<E, G1Var>
    where
        E: Pairing,
        G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
    {
        _params: PhantomData<G1Var>,
        deferred_fns_ref: DeferredFnsRef<E>,
        pcs_pk_vk: (HyperKZGProverKey<E>, HyperKZGVerifierKey<E>),
        commitment: Option<HyperKZGCommitment<E>>,
        point: Vec<Option<E::ScalarField>>,
        eval: Option<E::ScalarField>,
        pcs_proof: HyperKZGProof<E>,
        expected_result: Option<bool>,
    }

    impl<E, G1Var> OffloadedDataCircuit<E> for HyperKZGVerifierCircuit<E, G1Var>
    where
        E: Pairing,
        G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
    {
        fn deferred_fns_ref(&self) -> &DeferredFnsRef<E> {
            &self.deferred_fns_ref
        }
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
            let vk_var = HyperKZGVerifierKeyVar::<G1Var>::new_witness(ns!(cs, "vk"), || {
                Ok(self.pcs_pk_vk.clone())
            })?;

            let commitment_var =
                HyperKZGCommitmentVar::<G1Var>::new_witness(ns!(cs, "commitment"), || {
                    self.commitment
                        .clone()
                        .ok_or(SynthesisError::AssignmentMissing)
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

            let proof_var = HyperKZGProofVar::<E, G1Var>::new_witness(ns!(cs, "proof"), || {
                Ok(self.pcs_proof.clone())
            })?;

            let mut transcript_var =
                MockSpongeVar::new(ns!(cs, "transcript").cs(), &(b"TestEval".as_slice()));

            let kzg_vk = self.pcs_pk_vk.1.kzg_vk;
            let hyper_kzg =
                HyperKZGVerifierGadget::<E, MockSponge<E::ScalarField>, G1Var, Self>::new(
                    &self,
                    ns!(cs, "hyperkzg"),
                    vec![kzg_vk.g2, kzg_vk.beta_g2],
                );

            let r = hyper_kzg.verify(
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

            dbg!(cs.num_constraints());

            Ok(())
        }
    }

    struct HyperKZGVerifier<E, S, G1Var>
    where
        E: Pairing,
        S: SNARK<E::ScalarField>,
        G1Var: CurveVar<E::G1, E::ScalarField>,
    {
        _params: PhantomData<(E, S, G1Var)>,
    }

    impl<E, P, S, G1Var> OffloadedSNARK<E, P, S, G1Var> for HyperKZGVerifier<E, S, G1Var>
    where
        E: Pairing<G1Affine = Affine<P>, BaseField = P::BaseField, ScalarField = P::ScalarField>,
        P: SWCurveConfig<BaseField: PrimeField, ScalarField: JoltField>,
        S: SNARK<E::ScalarField>,
        G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
    {
        type Circuit = HyperKZGVerifierCircuit<E, G1Var>;
    }

    #[test]
    fn test_hyperkzg_eval() {
        type Groth16 = ark_groth16::Groth16<Bn254>;
        type VerifierSNARK = HyperKZGVerifier<Bn254, Groth16, G1Var>;

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
                deferred_fns_ref: Default::default(),
                pcs_pk_vk: (pcs_pk.clone(), pcs_vk.clone()),
                commitment: None,
                point: vec![None; size],
                eval: None,
                pcs_proof: HyperKZGProof::empty(size),
                expected_result: None,
            };

            VerifierSNARK::setup(circuit, &mut rng).unwrap()
        };

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
                    deferred_fns_ref: Default::default(),
                    pcs_pk_vk: (pcs_pk.clone(), pcs_vk.clone()),
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
                let proof = VerifierSNARK::prove(&cpk, verifier_circuit, &mut rng)
                    .map_err(|_e| ProofVerifyError::InternalError)?;

                let result = VerifierSNARK::verify_with_processed_vk(&cvk, &instance, &proof);
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
