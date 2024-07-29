pub mod bls12;

use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_r1cs_std::prelude::*;
use ark_relations::r1cs::SynthesisError;
use ark_std::fmt::Debug;

/// Specifies the constraints for computing a pairing in the bilinear group
/// `E`.
pub trait PairingGadget<E: Pairing, ConstraintF: PrimeField> {
    /// A variable representing an element of `G1`.
    /// This is the R1CS equivalent of `E::G1Projective`.
    type G1Var: CurveVar<E::G1, ConstraintF>
        + AllocVar<E::G1, ConstraintF>
        + AllocVar<E::G1Affine, ConstraintF>;

    /// A variable representing an element of `G2`.
    /// This is the R1CS equivalent of `E::G2Projective`.
    type G2Var: CurveVar<E::G2, ConstraintF>
        + AllocVar<E::G2, ConstraintF>
        + AllocVar<E::G2Affine, ConstraintF>;

    /// A variable representing an element of `GT`.
    /// This is the R1CS equivalent of `E::GT`.
    type GTVar: FieldVar<E::TargetField, ConstraintF>;

    /// A variable representing cached precomputation  that can speed up
    /// pairings computations. This is the R1CS equivalent of
    /// `E::G1Prepared`.
    type G1PreparedVar: ToBytesGadget<ConstraintF>
        + AllocVar<E::G1Prepared, ConstraintF>
        + Clone
        + Debug;
    /// A variable representing cached precomputation  that can speed up
    /// pairings computations. This is the R1CS equivalent of
    /// `E::G2Prepared`.
    type G2PreparedVar: ToBytesGadget<ConstraintF>
        + AllocVar<E::G2Prepared, ConstraintF>
        + Clone
        + Debug;

    /// Computes a multi-miller loop between elements
    /// of `p` and `q`.
    fn miller_loop(
        p: &[Self::G1PreparedVar],
        q: &[Self::G2PreparedVar],
    ) -> Result<Self::GTVar, SynthesisError>;

    /// Computes a final exponentiation over `p`.
    fn final_exponentiation(p: &Self::GTVar) -> Result<Self::GTVar, SynthesisError>;

    /// Computes a pairing over `p` and `q`.
    #[tracing::instrument(target = "r1cs")]
    fn pairing(
        p: Self::G1PreparedVar,
        q: Self::G2PreparedVar,
    ) -> Result<Self::GTVar, SynthesisError> {
        let tmp = Self::miller_loop(&[p], &[q])?;
        Self::final_exponentiation(&tmp)
    }

    /// Computes a product of pairings over the elements in `p` and `q`.
    #[must_use]
    #[tracing::instrument(target = "r1cs")]
    fn multi_pairing(
        p: &[Self::G1PreparedVar],
        q: &[Self::G2PreparedVar],
    ) -> Result<Self::GTVar, SynthesisError> {
        let miller_result = Self::miller_loop(p, q)?;
        Self::final_exponentiation(&miller_result)
    }

    /// Performs the precomputation to generate `Self::G1PreparedVar`.
    fn prepare_g1(q: &Self::G1Var) -> Result<Self::G1PreparedVar, SynthesisError>;

    /// Performs the precomputation to generate `Self::G2PreparedVar`.
    fn prepare_g2(q: &Self::G2Var) -> Result<Self::G2PreparedVar, SynthesisError>;
}

#[cfg(test)]
mod tests {
    use ark_bls12_381::Bls12_381;
    use ark_bn254::Bn254;
    use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
    use ark_ec::pairing::Pairing;
    use ark_ec::Group;
    use ark_ff::{Field, PrimeField};
    use ark_groth16::Groth16;
    use ark_r1cs_std::pairing::bls12::PairingVar;
    use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
    use ark_std::marker::PhantomData;
    use ark_std::rand::Rng;
    use ark_std::test_rng;
    use rand_core::{RngCore, SeedableRng};

    struct PairingCheckCircuit<E: Pairing, ConstraintF: PrimeField> {
        _constraint_f: PhantomData<ConstraintF>,
        r: Option<E::ScalarField>,
        r_g2: Option<E::G2Prepared>,
    }

    impl<E: Pairing, ConstraintF: PrimeField> ConstraintSynthesizer<ConstraintF>
        for PairingCheckCircuit<E, ConstraintF>
    {
        fn generate_constraints(
            self,
            cs: ConstraintSystemRef<ConstraintF>,
        ) -> Result<(), SynthesisError> {
            // TODO use PairingVar to generate constraints
            // let r_g1 = PairingVar::<E>::new_input(cs.clone(), || {
            //     Ok(E::G1Projective::prime_subgroup_generator())
            // })?;
            Ok(())
        }
    }

    #[test]
    fn test_pairing_check_circuit() {
        let c = PairingCheckCircuit::<Bls12_381, ark_bn254::Fr> {
            _constraint_f: PhantomData,
            r: None,
            r_g2: None,
        };

        // This is not cryptographically safe, use
        // `OsRng` (for example) in production software.
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

        let (pk, vk) = Groth16::<Bn254>::setup(c, &mut rng).unwrap();

        let pvk = Groth16::<Bn254>::process_vk(&vk).unwrap();

        let r = rng.gen();
        let r_g2 = <Bls12_381 as Pairing>::G2::generator() * &r;

        let c = PairingCheckCircuit::<Bls12_381, ark_bn254::Fr> {
            _constraint_f: PhantomData,
            r: Some(r),
            r_g2: Some(r_g2.into()),
        };

        let proof = Groth16::<Bn254>::prove(&pk, c, &mut rng).unwrap();

        assert!(Groth16::<Bn254>::verify_with_processed_vk(&pvk, &[], &proof).unwrap());
    }
}
