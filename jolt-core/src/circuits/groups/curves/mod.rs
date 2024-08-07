use crate::circuits::pairing::PairingGadget;

pub mod short_weierstrass;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits::groups::curves::short_weierstrass::bn254::G1Var;
    use crate::circuits::groups::curves::short_weierstrass::{AffineVar, ProjectiveVar};
    use crate::snark::{
        OffloadedData, OffloadedSNARK, OffloadedSNARKError, OffloadedSNARKVerifyingKey,
        PublicInputRef,
    };
    use ark_bls12_381::Bls12_381;
    use ark_bn254::{Bn254, Fq, Fr};
    use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
    use ark_crypto_primitives::sponge::Absorb;
    use ark_ec::bn::G1Projective;
    use ark_ec::pairing::Pairing;
    use ark_ec::short_weierstrass::{Affine, Projective, SWCurveConfig};
    use ark_ec::{CurveGroup, Group, VariableBaseMSM};
    use ark_ff::{PrimeField, ToConstraintField};
    use ark_groth16::Groth16;
    use ark_r1cs_std::fields::fp::FpVar;
    use ark_r1cs_std::fields::nonnative::NonNativeFieldVar;
    use ark_r1cs_std::prelude::*;
    use ark_r1cs_std::ToConstraintFieldGadget;
    use ark_relations::ns;
    use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
    use ark_serialize::{CanonicalSerialize, SerializationError};
    use ark_std::cell::OnceCell;
    use ark_std::marker::PhantomData;
    use ark_std::ops::Deref;
    use ark_std::rand::Rng;
    use ark_std::rc::Rc;
    use ark_std::sync::RwLock;
    use ark_std::{end_timer, start_timer, test_rng, One, UniformRand};
    use itertools::Itertools;
    use rand_core::{CryptoRng, RngCore, SeedableRng};

    struct DelayedOpsCircuit<E, G1Var>
    where
        E: Pairing,
        G1Var: CurveVar<E::G1, E::ScalarField>,
    {
        _params: PhantomData<G1Var>,

        // witness values
        w_g1: [Option<E::G1>; 3],
        d: Option<E::ScalarField>,

        // public inputs
        offloaded_data: Rc<OnceCell<OffloadedData<E>>>,
    }

    impl<E, G1Var> ConstraintSynthesizer<E::ScalarField> for DelayedOpsCircuit<E, G1Var>
    where
        E: Pairing,
        G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
    {
        fn generate_constraints(
            self,
            cs: ConstraintSystemRef<E::ScalarField>,
        ) -> Result<(), SynthesisError> {
            dbg!(cs.num_constraints());

            let d = FpVar::new_witness(ns!(cs, "d"), || {
                self.d.ok_or(SynthesisError::AssignmentMissing)
            })?;
            dbg!(cs.num_constraints());

            let w_g1 = (0..3)
                .map(|i| {
                    G1Var::new_witness(ns!(cs, "w_g1"), || {
                        self.w_g1[i].ok_or(SynthesisError::AssignmentMissing)
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            dbg!(cs.num_constraints());

            let d_square = d.square()?;
            let d_k = [FpVar::one(), d, d_square];
            dbg!(cs.num_constraints());

            // `None` in setup mode, `Some<Vec<E::G1>>` in proving mode.
            let msm_g1_values = w_g1
                .iter()
                .map(|g1| g1.value().ok().map(|g1| g1.into_affine()))
                .collect::<Option<Vec<_>>>();

            let d_k_values = d_k
                .iter()
                .map(|d| d.value().ok())
                .collect::<Option<Vec<_>>>();

            let (full_msm_value, r_g1_value) = msm_g1_values
                .clone()
                .zip(d_k_values)
                .map(|(g1s, d_k)| {
                    let r_g1 = E::G1::msm_unchecked(&g1s, &d_k);
                    let minus_one = -E::ScalarField::one();
                    (
                        (
                            [g1s, vec![r_g1.into()]].concat(),
                            [d_k, vec![minus_one]].concat(),
                        ),
                        r_g1,
                    )
                })
                .unzip();

            let r_g1_var = G1Var::new_witness(ns!(cs, "r_g1"), || {
                r_g1_value.ok_or(SynthesisError::AssignmentMissing)
            })?;

            if let Some(msm_value) = full_msm_value {
                self.offloaded_data
                    .set(OffloadedData {
                        msms: vec![msm_value],
                    })
                    .unwrap();
            };

            // write d_k to public_input
            for x in d_k {
                let d_k_input = FpVar::new_input(ns!(cs, "d_k"), || x.value())?;
                d_k_input.enforce_equal(&x)?;
            }
            dbg!(cs.num_constraints());

            // write w_g1 to public_input
            for g1 in w_g1 {
                let f_vec = g1.to_constraint_field()?;

                for f in f_vec.iter() {
                    let f_input = FpVar::new_input(ns!(cs, "w_g1"), || f.value())?;
                    f_input.enforce_equal(f)?;
                }
            }

            // write r_g1 to public_input
            {
                let f_vec = r_g1_var.to_constraint_field()?;

                for f in f_vec.iter() {
                    let f_input = FpVar::new_input(ns!(cs, "r_g1"), || f.value())?;
                    f_input.enforce_equal(f)?;
                }
            }
            dbg!(cs.num_constraints());

            Ok(())
        }
    }

    impl<E, G1Var> PublicInputRef<E> for DelayedOpsCircuit<E, G1Var>
    where
        E: Pairing,
        G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
    {
        fn public_input_ref(&self) -> Rc<OnceCell<OffloadedData<E>>> {
            self.offloaded_data.clone()
        }
    }

    struct DelayedOpsCircuitSNARK<E, S, G1Var>
    where
        E: Pairing,
        S: SNARK<E::ScalarField>,
        G1Var: CurveVar<E::G1, E::ScalarField>,
    {
        _params: PhantomData<(E, S, G1Var)>,
    }

    impl<E, P, S, G1Var> OffloadedSNARK<E, P, S, G1Var> for DelayedOpsCircuitSNARK<E, S, G1Var>
    where
        E: Pairing<G1Affine = Affine<P>, BaseField = P::BaseField, ScalarField = P::ScalarField>,
        P: SWCurveConfig<BaseField: PrimeField>,
        S: SNARK<E::ScalarField>,
        G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
    {
        type Circuit = DelayedOpsCircuit<E, G1Var>;

        fn offloaded_setup(
            snark_vk: S::ProcessedVerifyingKey,
        ) -> Result<OffloadedSNARKVerifyingKey<E, S>, OffloadedSNARKError<S::Error>> {
            Ok(OffloadedSNARKVerifyingKey {
                snark_pvk: snark_vk,
                delayed_pairings: vec![], // TODO none yet
            })
        }

        fn g2_elements(
            vk: &OffloadedSNARKVerifyingKey<E, S>,
            public_input: &[E::ScalarField],
            proof: &S::Proof,
        ) -> Result<Vec<Vec<E::G2>>, SerializationError> {
            // TODO get the G2 elements from the verifying key
            Ok(vec![])
        }
    }

    #[test]
    fn test_delayed_pairing_circuit() {
        type DemoCircuit = DelayedOpsCircuit<Bn254, G1Var>;

        type DemoSNARK = DelayedOpsCircuitSNARK<Bn254, Groth16<Bn254>, G1Var>;

        let circuit = DemoCircuit {
            _params: PhantomData,
            w_g1: [None; 3],
            d: None,
            offloaded_data: Default::default(),
        };

        // This is not cryptographically safe, use
        // `OsRng` (for example) in production software.
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

        let setup_timer = start_timer!(|| "Groth16::setup");
        let (pk, vk) = DemoSNARK::setup(circuit, &mut rng).unwrap();
        end_timer!(setup_timer);

        let process_vk_timer = start_timer!(|| "Groth16::process_vk");
        // let pvk = DemoSNARK::process_vk(&vk).unwrap();
        let pvk = vk;
        end_timer!(process_vk_timer);

        let c_init_values = DemoCircuit {
            _params: PhantomData,
            w_g1: [Some(rng.gen()); 3],
            d: Some(rng.gen()),
            offloaded_data: Default::default(),
        };

        let prove_timer = start_timer!(|| "Groth16::prove");
        let proof = DemoSNARK::prove(&pk, c_init_values, &mut rng).unwrap();
        end_timer!(prove_timer);

        let verify_timer = start_timer!(|| "Groth16::verify");
        let verify_result = DemoSNARK::verify_with_processed_vk(&pvk, &[], &proof);
        end_timer!(verify_timer);

        assert!(verify_result.unwrap());
    }
}
