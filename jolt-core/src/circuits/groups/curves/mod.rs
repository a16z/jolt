use crate::circuits::pairing::PairingGadget;

pub mod short_weierstrass;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits::groups::curves::short_weierstrass::bn254::G1Var;
    use crate::circuits::groups::curves::short_weierstrass::{AffineVar, ProjectiveVar};
    use ark_bls12_381::Bls12_381;
    use ark_bn254::{Bn254, Fq, Fr};
    use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
    use ark_crypto_primitives::sponge::Absorb;
    use ark_ec::bn::G1Projective;
    use ark_ec::pairing::Pairing;
    use ark_ec::short_weierstrass::{Projective, SWCurveConfig};
    use ark_ec::{CurveGroup, Group};
    use ark_ff::{PrimeField, ToConstraintField};
    use ark_groth16::Groth16;
    use ark_r1cs_std::fields::fp::FpVar;
    use ark_r1cs_std::fields::nonnative::NonNativeFieldVar;
    use ark_r1cs_std::prelude::*;
    use ark_r1cs_std::ToConstraintFieldGadget;
    use ark_relations::ns;
    use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
    use ark_serialize::CanonicalSerialize;
    use ark_std::marker::PhantomData;
    use ark_std::rand::Rng;
    use ark_std::{end_timer, start_timer, test_rng, UniformRand};
    use itertools::Itertools;
    use rand_core::{RngCore, SeedableRng};
    use std::sync::{Arc, RwLock};

    struct DelayedPairingCircuit<E, G1Var>
    where
        E: Pairing,
        G1Var: CurveVar<E::G1, E::ScalarField>,
    {
        _params: PhantomData<G1Var>,

        // witness values
        w_g1: [Option<E::G1>; 3],
        d: Option<E::ScalarField>,

        // public inputs
        r_g1: Arc<RwLock<Option<E::G1>>>,
    }

    impl<E, G1Var> ConstraintSynthesizer<E::ScalarField> for DelayedPairingCircuit<E, G1Var>
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
            let d_to_k = [FpVar::one(), d, d_square];
            dbg!(cs.num_constraints());

            let r_g1 = (1..3)
                .map(|k| {
                    w_g1[k]
                        .clone()
                        .scalar_mul_le(d_to_k[k].to_bits_le()?.iter())
                })
                .collect::<Result<Vec<_>, _>>()?
                .iter()
                .fold(w_g1[0].clone(), |acc, x| acc + x);
            dbg!(cs.num_constraints());

            let r_g1_opt = r_g1.value().ok();

            let mut r_value_opt = self.r_g1.write().unwrap();
            *r_value_opt = r_g1_opt.clone();
            drop(r_value_opt);

            let cf_vec = r_g1.to_constraint_field()?;

            for cf in cf_vec.iter() {
                let cf_input = FpVar::new_input(ns!(cs, "r_g1_input"), || cf.value())?;
                cf_input.enforce_equal(&cf)?;
            }

            dbg!(cs.num_constraints());

            Ok(())
        }
    }

    #[test]
    fn test_delayed_pairing_circuit() {
        type DemoCircuit = DelayedPairingCircuit<Bn254, G1Var>;

        let circuit = DemoCircuit {
            _params: PhantomData,
            w_g1: [None; 3],
            d: None,
            r_g1: Arc::new(RwLock::new(None)),
        };

        // This is not cryptographically safe, use
        // `OsRng` (for example) in production software.
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

        let setup_timer = start_timer!(|| "Groth16::setup");
        let (pk, vk) = Groth16::<Bn254>::setup(circuit, &mut rng).unwrap();
        end_timer!(setup_timer);

        let process_vk_timer = start_timer!(|| "Groth16::process_vk");
        let pvk = Groth16::<Bn254>::process_vk(&vk).unwrap();
        end_timer!(process_vk_timer);

        let r_g1_lock = Arc::new(RwLock::new(None));
        let c_init_values = DemoCircuit {
            _params: PhantomData,
            w_g1: [Some(rng.gen()); 3],
            d: Some(rng.gen()),
            r_g1: r_g1_lock.clone(),
        };

        let prove_timer = start_timer!(|| "Groth16::prove");
        let proof = Groth16::<Bn254>::prove(&pk, c_init_values, &mut rng).unwrap();
        end_timer!(prove_timer);

        let r_g1_opt_read = r_g1_lock.read().unwrap();
        let r_g1 = dbg!(*r_g1_opt_read).unwrap();

        let public_input = get_public_input(&r_g1);

        let verify_timer = start_timer!(|| "Groth16::verify");
        let verify_result = Groth16::<Bn254>::verify_with_processed_vk(&pvk, &public_input, &proof);
        end_timer!(verify_timer);

        assert!(verify_result.unwrap());
    }

    fn get_public_input(g1: &ark_bn254::G1Projective) -> Vec<ark_bn254::Fr> {
        G1Var::constant(g1.clone())
            .to_constraint_field()
            .unwrap()
            .iter()
            .map(|x| x.value().unwrap())
            .collect::<Vec<_>>()
    }
}
