use ark_crypto_primitives::snark::SNARK;
use ark_ec::pairing::Pairing;
use ark_ec::{AffineRepr, VariableBaseMSM};
use ark_ff::PrimeField;
use ark_relations::r1cs::ConstraintSynthesizer;
use ark_serialize::CanonicalSerialize;
use ark_std::rand::{CryptoRng, RngCore};
use ark_std::{One, Zero};
use std::ops::Neg;

pub mod fields;
pub mod groups;
pub mod pairing;
pub mod poly;

/// Describes G1 elements to be used in a multi-pairing.
/// The verifier is responsible for ensuring that the sum of the pairings is zero.
/// The verifier needs to use appropriate G2 elements from the verification key or the proof
/// (depending on the protocol).
pub struct DelayedPairingDef {
    /// Left pairing G1 element offset in the public input.
    pub l_g1_offset: usize,
    /// Right pairing G1 element offset in the public input. This element is, by convention, always used
    /// in the multi-pairing computation with coefficient `-1`.
    pub r_g1_offset: usize,
}

/// Describes a block of G1 elements `Gᵢ` and scalars in `sᵢ` the public input, such that `∑ᵢ sᵢ·Gᵢ == 0`.
/// It's the verifiers responsibility to ensure that the sum is zero.
pub struct DelayedMSMDef {
    /// Length is the number of G1 elements in the MSM.
    pub length: usize,
    /// MSM G1 elements offset in the public input. G1 elements are stored as sequences of scalar field elements
    /// encoding the compressed coordinates of the G1 points (which would natively be numbers in the base field).
    /// The offset is in the number of scalar field elements in the public input before the G1 elements block.
    pub g1_offset: usize,
    /// MSM scalars offset in the public input. The scalar at index `length-1` is, by convention, always `-1`,
    /// so we can save one public input element.
    /// The offset is in the number of scalar field elements in the public input before the scalars block.
    pub scalar_offset: usize,
}

pub struct LoadedSNARKProof<E, S>
where
    E: Pairing,
    S: SNARK<E::ScalarField>,
{
    pub snark_proof: S::Proof,
    /// Delayed pairing G1 elements in the public input.
    pub delayed_pairings: Vec<DelayedPairingDef>,
    /// Delayed MSM G1 and scalar blocks in the public input.
    pub delayed_msms: Vec<DelayedMSMDef>,
}

pub trait LoadedSNARK<E, S, C>
where
    E: Pairing,
    S: SNARK<E::ScalarField>,
{
    type Circuit: ConstraintSynthesizer<E::ScalarField>;

    fn prove<R: RngCore + CryptoRng>(
        circuit_pk: &S::ProvingKey,
        circuit: Self::Circuit,
        rng: &mut R,
    ) -> Result<LoadedSNARKProof<E, S>, S::Error>;

    fn msm_inputs(
        msm_defs: &[DelayedMSMDef],
        public_input: &[E::ScalarField],
    ) -> Result<Vec<(Vec<E::G1Affine>, Vec<E::ScalarField>)>, S::Error> {
        msm_defs
            .iter()
            .map(|msm_def| {
                let g1_offset = msm_def.g1_offset;
                let msm_length = msm_def.length;
                let g1s = Self::g1_elements(public_input, g1_offset, msm_length);
                let scalars = [
                    &public_input[msm_def.scalar_offset..msm_def.scalar_offset + msm_length - 1],
                    &[-E::ScalarField::one()],
                ]
                .concat();
                Ok((g1s, scalars))
            })
            .collect()
    }

    fn g1_elements(
        public_input: &[<E as Pairing>::ScalarField],
        g1_offset: usize,
        length: usize,
    ) -> Vec<<E as Pairing>::G1Affine> {
        let g1_element_size = g1_affine_size_in_scalar_field_elements::<E>();
        public_input[g1_offset..g1_offset + length * g1_element_size]
            .chunks(g1_element_size)
            .map(|chunk| g1_affine_from_scalar_field::<E>(chunk))
            .collect()
    }

    fn pairing_inputs(
        pvk: &S::ProcessedVerifyingKey,
        public_input: &[E::ScalarField],
        proof: &LoadedSNARKProof<E, S>,
    ) -> Result<Vec<(Vec<E::G1>, Vec<E::G2>)>, S::Error>;

    fn verify(
        pvk: &S::ProcessedVerifyingKey,
        public_input: &[E::ScalarField],
        proof: &LoadedSNARKProof<E, S>,
    ) -> Result<bool, S::Error> {
        let r = S::verify_with_processed_vk(pvk, public_input, &proof.snark_proof)?;
        if !r {
            return Ok(false);
        }

        let msms = Self::msm_inputs(&proof.delayed_msms, public_input)?;
        for (g1s, scalars) in msms {
            assert_eq!(g1s.len(), scalars.len());
            let r = E::G1::msm_unchecked(&g1s, &scalars);
            if !r.is_zero() {
                return Ok(false);
            }
        }

        let pairings = Self::pairing_inputs(pvk, public_input, &proof)?;
        for (g1s, g2s) in pairings {
            assert_eq!(g1s.len(), g2s.len());
            let r = E::multi_pairing(&g1s, &g2s);
            if !r.is_zero() {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

fn g1_affine_size_in_scalar_field_elements<E: Pairing>() -> usize {
    todo!()
}

fn g1_affine_from_scalar_field<E: Pairing>(_s: &[E::ScalarField]) -> E::G1Affine {
    todo!()
}
