use ark_crypto_primitives::snark::SNARK;
use ark_ec::{
    pairing::Pairing,
    short_weierstrass::{Affine, SWCurveConfig},
    AffineRepr, CurveConfig, VariableBaseMSM,
};
use ark_ff::PrimeField;
use ark_r1cs_std::{
    fields::nonnative::params::{get_params, OptimizationType},
    fields::nonnative::AllocatedNonNativeFieldVar,
};
use ark_relations::r1cs::ConstraintSynthesizer;
use ark_serialize::{CanonicalSerialize, SerializationError, Valid};
use ark_std::{
    iterable::Iterable,
    ops::Neg,
    rand::{CryptoRng, RngCore},
    result::Result,
    One, Zero,
};

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

pub struct OffloadedSNARKVerifyingKey<E, S>
where
    E: Pairing,
    S: SNARK<E::ScalarField>,
{
    pub snark_pvk: S::ProcessedVerifyingKey,
    /// Delayed pairing G1 elements in the public input.
    pub delayed_pairings: Vec<DelayedPairingDef>,
    /// Delayed MSM G1 and scalar blocks in the public input.
    pub delayed_msms: Vec<DelayedMSMDef>,
}

#[derive(thiserror::Error, Debug)]
pub enum OffloadedSNARKError<E, S>
where
    E: Pairing,
    S: SNARK<E::ScalarField>,
{
    /// Wraps `S::Error`.
    #[error(transparent)]
    SNARKError(S::Error),
    /// Wraps `SerializationError`.
    #[error(transparent)]
    SerializationError(#[from] SerializationError),
}

pub trait OffloadedSNARK<E, P, S>
where
    E: Pairing<G1Affine = Affine<P>, BaseField = P::BaseField, ScalarField = P::ScalarField>,
    P: SWCurveConfig<BaseField: PrimeField>,
    S: SNARK<E::ScalarField>,
{
    type Circuit: ConstraintSynthesizer<E::ScalarField>;

    fn prove<R: RngCore + CryptoRng>(
        circuit_pk: &S::ProvingKey,
        circuit: Self::Circuit,
        rng: &mut R,
    ) -> Result<S::Proof, OffloadedSNARKError<E, S>>;

    fn msm_inputs(
        msm_defs: &[DelayedMSMDef],
        public_input: &[E::ScalarField],
    ) -> Result<Vec<(Vec<E::G1Affine>, Vec<E::ScalarField>)>, SerializationError> {
        msm_defs
            .iter()
            .map(|msm_def| {
                let g1_offset = msm_def.g1_offset;
                let msm_length = msm_def.length;
                assert!(msm_length > 1); // TODO make it a verifier key validity error
                let g1s = Self::g1_elements(public_input, g1_offset, msm_length)?;

                if public_input.len() < msm_def.scalar_offset + msm_length - 1 {
                    return Err(SerializationError::InvalidData);
                };
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
        public_input: &[E::ScalarField],
        g1_offset: usize,
        length: usize,
    ) -> Result<Vec<E::G1Affine>, SerializationError> {
        let g1_element_size = g1_affine_size_in_scalar_field_elements::<E>();
        if public_input.len() < g1_offset + length * g1_element_size {
            return Err(SerializationError::InvalidData);
        };

        public_input[g1_offset..g1_offset + length * g1_element_size]
            .chunks(g1_element_size)
            .map(|chunk| g1_affine_from_scalar_field::<E, P>(chunk))
            .collect()
    }

    fn pairing_inputs(
        vk: &OffloadedSNARKVerifyingKey<E, S>,
        public_input: &[E::ScalarField],
        proof: &S::Proof,
    ) -> Result<Vec<(Vec<E::G1>, Vec<E::G2>)>, SerializationError> {
        let g1_vectors = vk
            .delayed_pairings
            .iter()
            .map(|pairing_def| {
                let l_g1 = Self::g1_elements(public_input, pairing_def.l_g1_offset, 1)?[0];
                let r_g1 = Self::g1_elements(public_input, pairing_def.r_g1_offset, 1)?[0];

                Ok(vec![l_g1.into(), (-r_g1).into()])
            })
            .collect::<Result<Vec<Vec<E::G1>>, SerializationError>>();
        Ok(g1_vectors?
            .into_iter()
            .zip(Self::g2_elements(vk, public_input, proof)?)
            .collect())
    }

    fn g2_elements(
        vk: &OffloadedSNARKVerifyingKey<E, S>,
        public_input: &[<E as Pairing>::ScalarField],
        proof: &S::Proof,
    ) -> Result<Vec<Vec<E::G2>>, SerializationError>;

    fn verify(
        vk: &OffloadedSNARKVerifyingKey<E, S>,
        public_input: &[E::ScalarField],
        proof: &S::Proof,
    ) -> Result<bool, OffloadedSNARKError<E, S>> {
        let r = S::verify_with_processed_vk(&vk.snark_pvk, public_input, proof)
            .map_err(|e| OffloadedSNARKError::SNARKError(e))?;
        if !r {
            return Ok(false);
        }

        let msms = Self::msm_inputs(&vk.delayed_msms, public_input)?;
        for (g1s, scalars) in msms {
            assert_eq!(g1s.len(), scalars.len());
            let r = E::G1::msm_unchecked(&g1s, &scalars);
            if !r.is_zero() {
                return Ok(false);
            }
        }

        let pairings = Self::pairing_inputs(vk, public_input, &proof)?;
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
    let params = get_params(
        E::BaseField::MODULUS_BIT_SIZE as usize,
        E::ScalarField::MODULUS_BIT_SIZE as usize,
        OptimizationType::Weight,
    );
    params.num_limbs * 2 + 1
}

fn g1_affine_from_scalar_field<E, P>(
    s: &[E::ScalarField],
) -> Result<E::G1Affine, SerializationError>
where
    E: Pairing<G1Affine = Affine<P>, BaseField = P::BaseField, ScalarField = P::ScalarField>,
    P: SWCurveConfig<BaseField: PrimeField>,
{
    let infinity = !s[s.len() - 1].is_zero();
    if infinity {
        return Ok(E::G1Affine::zero());
    }

    let base_field_size_in_limbs = (s.len() - 1) / 2;
    let x = AllocatedNonNativeFieldVar::<E::BaseField, E::ScalarField>::limbs_to_value(
        s[..base_field_size_in_limbs].to_vec(),
        OptimizationType::Weight,
    );
    let y = AllocatedNonNativeFieldVar::<E::BaseField, E::ScalarField>::limbs_to_value(
        s[base_field_size_in_limbs..s.len() - 1].to_vec(),
        OptimizationType::Weight,
    );

    let affine = Affine {
        x,
        y,
        infinity: false,
    };
    affine.check()?;
    Ok(affine)
}
