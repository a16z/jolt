use ark_crypto_primitives::snark::SNARK;
use ark_ec::pairing::Pairing;
use ark_ec::short_weierstrass::{Affine, SWCurveConfig};
use ark_ec::{AffineRepr, VariableBaseMSM};
use ark_ff::{PrimeField, Zero};
use ark_r1cs_std::fields::nonnative::params::{get_params, OptimizationType};
use ark_r1cs_std::fields::nonnative::AllocatedNonNativeFieldVar;
use ark_r1cs_std::groups::CurveVar;
use ark_r1cs_std::{R1CSVar, ToConstraintFieldGadget};
use ark_relations::r1cs::ConstraintSynthesizer;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};
use itertools::Itertools;
use rand_core::{CryptoRng, RngCore};
use std::cell::OnceCell;
use std::rc::Rc;

/// Describes G1 elements to be used in a multi-pairing.
/// The verifier is responsible for ensuring that the sum of the pairings is zero.
/// The verifier needs to use appropriate G2 elements from the verification key or the proof
/// (depending on the protocol).
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct DelayedPairingDef {
    /// Offsets of the G1 elements in the public input. The G1 elements are stored as sequences of scalar field elements
    /// encoding the compressed coordinates of the G1 points (which would natively be numbers in the base field).
    /// The offsets are in the number of scalar field elements in the public input before the G1 elements block.
    /// The last element, by convention, is always used in the multi-pairing computation with coefficient `-1`.
    pub g1_offsets: Vec<usize>,
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct OffloadedSNARKVerifyingKey<E, S>
where
    E: Pairing,
    S: SNARK<E::ScalarField>,
{
    pub snark_pvk: S::ProcessedVerifyingKey,
    pub delayed_pairings: Vec<DelayedPairingDef>,
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct OffloadedSNARKProof<E, S>
where
    E: Pairing,
    S: SNARK<E::ScalarField>,
{
    pub snark_proof: S::Proof,
    pub offloaded_data: OffloadedData<E>,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct OffloadedData<E: Pairing> {
    /// Blocks of G1 elements `Gᵢ` and scalars in `sᵢ` the public input, such that `∑ᵢ sᵢ·Gᵢ == 0`.
    /// It's the verifiers responsibility to ensure that the sum is zero.
    /// The scalar at index `length-1` is, by convention, always `-1`, so
    /// we save one public input element per MSM.
    pub msms: Vec<(Vec<E::G1Affine>, Vec<E::ScalarField>)>,
}

pub trait PublicInputRef<E>
where
    E: Pairing,
{
    fn public_input_ref(&self) -> Rc<OnceCell<OffloadedData<E>>>;
}

#[derive(thiserror::Error, Debug)]
pub enum OffloadedSNARKError<Err>
where
    Err: 'static + ark_std::error::Error,
{
    /// Wraps `Err`.
    #[error(transparent)]
    SNARKError(Err),
    /// Wraps `SerializationError`.
    #[error(transparent)]
    SerializationError(#[from] SerializationError),
}

pub trait OffloadedSNARK<E, P, S, G1Var>
where
    E: Pairing<G1Affine = Affine<P>, BaseField = P::BaseField, ScalarField = P::ScalarField>,
    P: SWCurveConfig<BaseField: PrimeField>,
    S: SNARK<E::ScalarField>,
    G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
{
    type Circuit: ConstraintSynthesizer<E::ScalarField> + PublicInputRef<E>;

    fn setup<C: ConstraintSynthesizer<E::ScalarField>, R: RngCore + CryptoRng>(
        circuit: C,
        rng: &mut R,
    ) -> Result<(S::ProvingKey, OffloadedSNARKVerifyingKey<E, S>), OffloadedSNARKError<S::Error>>
    {
        Self::circuit_specific_setup(circuit, rng)
    }

    fn circuit_specific_setup<C: ConstraintSynthesizer<E::ScalarField>, R: RngCore + CryptoRng>(
        circuit: C,
        rng: &mut R,
    ) -> Result<(S::ProvingKey, OffloadedSNARKVerifyingKey<E, S>), OffloadedSNARKError<S::Error>>
    {
        let (pk, snark_vk) = S::circuit_specific_setup(circuit, rng)
            .map_err(|e| OffloadedSNARKError::SNARKError(e))?;
        let snark_pvk = S::process_vk(&snark_vk).map_err(|e| OffloadedSNARKError::SNARKError(e))?;
        let vk = Self::offloaded_setup(snark_pvk)?;
        Ok((pk, vk))
    }

    fn offloaded_setup(
        snark_vk: S::ProcessedVerifyingKey,
    ) -> Result<OffloadedSNARKVerifyingKey<E, S>, OffloadedSNARKError<S::Error>>;

    fn prove<R: RngCore + CryptoRng>(
        circuit_pk: &S::ProvingKey,
        circuit: Self::Circuit,
        rng: &mut R,
    ) -> Result<OffloadedSNARKProof<E, S>, OffloadedSNARKError<S::Error>> {
        let public_input_ref = circuit.public_input_ref();
        let proof =
            S::prove(circuit_pk, circuit, rng).map_err(|e| OffloadedSNARKError::SNARKError(e))?;
        Ok(OffloadedSNARKProof {
            snark_proof: proof,
            offloaded_data: public_input_ref.get().unwrap().clone(),
        })
    }

    fn verify(
        vk: &OffloadedSNARKVerifyingKey<E, S>,
        public_input: &[E::ScalarField],
        proof: &OffloadedSNARKProof<E, S>,
    ) -> Result<bool, OffloadedSNARKError<S::Error>> {
        Self::verify_with_processed_vk(vk, public_input, proof)
    }

    fn verify_with_processed_vk(
        vk: &OffloadedSNARKVerifyingKey<E, S>,
        public_input: &[E::ScalarField],
        proof: &OffloadedSNARKProof<E, S>,
    ) -> Result<bool, OffloadedSNARKError<S::Error>> {
        let public_input = build_public_input::<E, G1Var>(public_input, &proof.offloaded_data);

        let r = S::verify_with_processed_vk(&vk.snark_pvk, &public_input, &proof.snark_proof)
            .map_err(|e| OffloadedSNARKError::SNARKError(e))?;
        if !r {
            return Ok(false);
        }

        for (g1s, scalars) in &proof.offloaded_data.msms {
            assert_eq!(g1s.len(), scalars.len());
            let r = E::G1::msm_unchecked(&g1s, &scalars);
            if !r.is_zero() {
                return Ok(false);
            }
        }

        let pairings = Self::pairing_inputs(vk, &public_input, &proof.snark_proof)?;
        for (g1s, g2s) in pairings {
            assert_eq!(g1s.len(), g2s.len());
            let r = E::multi_pairing(&g1s, &g2s);
            if !r.is_zero() {
                return Ok(false);
            }
        }

        Ok(true)
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
                let last_index = pairing_def.g1_offsets.len() - 1;
                let g1s = pairing_def
                    .g1_offsets
                    .iter()
                    .enumerate()
                    .map(|(i, &offset)| {
                        let g1 = Self::g1_elements(public_input, offset, 1)?[0];
                        if i == last_index {
                            Ok((-g1).into())
                        } else {
                            Ok(g1.into())
                        }
                    })
                    .collect::<Result<Vec<_>, _>>();
                g1s
            })
            .collect::<Result<Vec<Vec<E::G1>>, SerializationError>>();
        Ok(g1_vectors?
            .into_iter()
            .zip(Self::g2_elements(vk, public_input, proof)?)
            .collect())
    }

    fn g2_elements(
        vk: &OffloadedSNARKVerifyingKey<E, S>,
        public_input: &[E::ScalarField],
        proof: &S::Proof,
    ) -> Result<Vec<Vec<E::G2>>, SerializationError>;
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

fn build_public_input<E, G1Var>(
    public_input: &[E::ScalarField],
    data: &OffloadedData<E>,
) -> Vec<E::ScalarField>
where
    E: Pairing,
    G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
{
    let scalars = &data.msms[0].1;

    let scalar_vec = scalars[..scalars.len() - 1].to_vec(); // remove the last element (always `-1`)

    let msm_g1_vec = data.msms[0]
        .0
        .iter()
        .map(|&g1| {
            G1Var::constant(g1.into())
                .to_constraint_field()
                .unwrap()
                .iter()
                .map(|x| x.value().unwrap())
                .collect::<Vec<_>>()
        })
        .concat();

    [public_input.to_vec(), scalar_vec, msm_g1_vec].concat()
}
