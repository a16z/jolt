use ark_crypto_primitives::snark::SNARK;
use ark_ec::{
    pairing::Pairing,
    short_weierstrass::{Affine, SWCurveConfig},
    AffineRepr, CurveGroup, VariableBaseMSM,
};
use ark_ff::{PrimeField, Zero};
use ark_r1cs_std::{
    fields::nonnative::params::{get_params, OptimizationType},
    fields::nonnative::AllocatedNonNativeFieldVar,
    groups::CurveVar,
    R1CSVar, ToConstraintFieldGadget,
};
use ark_relations::r1cs;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};
use ark_std::{cell::OnceCell, cell::RefCell, marker::PhantomData, rc::Rc};
use itertools::Itertools;
use rand_core::{CryptoRng, RngCore};

/// Describes G1 elements to be used in a multi-pairing.
/// The verifier is responsible for ensuring that the sum of the pairings is zero.
/// The verifier needs to use appropriate G2 elements from the verification key or the proof
/// (depending on the protocol).
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct OffloadedPairingDef<E>
where
    E: Pairing,
{
    /// Offsets of the G1 elements in the public input. The G1 elements are stored as sequences of scalar field elements
    /// encoding the compressed coordinates of the G1 points (which would natively be numbers in the base field).
    /// The offsets are in the number of scalar field elements in the public input before the G1 element.
    /// The last element, by convention, is always used in the multi-pairing computation with coefficient `-1`.
    pub g1_offsets: Vec<usize>,
    pub g2_elements: Vec<E::G2Affine>,
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct OffloadedSNARKVerifyingKey<E, S>
where
    E: Pairing,
    S: SNARK<E::ScalarField>,
{
    pub snark_pvk: S::ProcessedVerifyingKey,
    pub delayed_pairings: Vec<OffloadedPairingDef<E>>,
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct OffloadedSNARKProof<E, S>
where
    E: Pairing,
    S: SNARK<E::ScalarField>,
{
    pub snark_proof: S::Proof,
    pub offloaded_data: OffloadedData<E::G1>,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct OffloadedData<G: CurveGroup> {
    /// Blocks of G1 elements `Gᵢ` and scalars in `sᵢ` the public input, such that `∑ᵢ sᵢ·Gᵢ == 0`.
    /// It's the verifiers responsibility to ensure that the sum is zero.
    /// The scalar at index `length-1` is, by convention, always `-1`, so
    /// we save one public input element per MSM.
    pub msms: Vec<(Vec<G::Affine>, Vec<G::ScalarField>)>,
}

pub enum DeferredOp<E: Pairing> {
    MSM(Option<(Vec<E::G1Affine>, Vec<E::ScalarField>)>),
    Pairing(OffloadedPairingDef<E>),
}

pub type DeferredFn<G: CurveGroup> = dyn FnOnce() -> Result<
    (Option<(Vec<G::Affine>, Vec<G::ScalarField>)>, Vec<usize>),
    SynthesisError,
>;

pub type DeferredFnsRef<G: CurveGroup> = Rc<RefCell<Vec<Box<DeferredFn<G>>>>>;

pub trait OffloadedDataCircuit<G>: Clone
where
    G: CurveGroup,
{
    fn deferred_fns_ref(&self) -> &DeferredFnsRef<G>;

    fn defer_msm(
        &self,
        f: impl FnOnce() -> Result<
                (Option<(Vec<G::Affine>, Vec<G::ScalarField>)>, Vec<usize>),
                SynthesisError,
            > + 'static,
    ) {
        self.deferred_fns_ref().borrow_mut().push(Box::new(f));
    }
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
    #[error(transparent)]
    SynthesisError(#[from] SynthesisError),
}

struct WrappedCircuit<E, C>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField> + OffloadedDataCircuit<E::G1>,
{
    circuit: C,
    offloaded_data_ref: Rc<OnceCell<OffloadedData<E::G1>>>,
    g1_offsets_ref: Rc<OnceCell<Vec<Vec<usize>>>>,
}

fn run_deferred<E: Pairing>(
    deferred_fns: Vec<Box<DeferredFn<E::G1>>>,
) -> Result<(Option<OffloadedData<E::G1>>, Vec<Vec<usize>>), SynthesisError> {
    let (msms, g1_offsets) = deferred_fns
        .into_iter()
        .map(|f| f())
        .collect::<Result<(Vec<_>, Vec<_>), _>>()?;

    // can't collect into `Option<Vec<_>>` above: it short-circuits on the first None
    let msms = msms.into_iter().collect::<Option<Vec<_>>>();

    Ok((msms.map(|msms| OffloadedData { msms }), g1_offsets))
}

impl<E, C> ConstraintSynthesizer<E::ScalarField> for WrappedCircuit<E, C>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField> + OffloadedDataCircuit<E::G1>,
{
    fn generate_constraints(self, cs: ConstraintSystemRef<E::ScalarField>) -> r1cs::Result<()> {
        let deferred_fns_ref = self.circuit.deferred_fns_ref().clone();

        let offloaded_data_ref = self.offloaded_data_ref.clone();
        let g1_offsets_ref = self.g1_offsets_ref.clone();

        self.circuit.generate_constraints(cs)?;

        let (offloaded_data, g1_offsets) = run_deferred::<E>(deferred_fns_ref.take())?;

        if let Some(offloaded_data) = offloaded_data {
            offloaded_data_ref.set(offloaded_data).unwrap();
        }
        g1_offsets_ref.set(g1_offsets).unwrap();

        Ok(())
    }
}

pub trait OffloadedSNARK<E, P, S, G1Var>
where
    E: Pairing<G1Affine = Affine<P>, BaseField = P::BaseField, ScalarField = P::ScalarField>,
    P: SWCurveConfig<BaseField: PrimeField>,
    S: SNARK<E::ScalarField>,
    G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
{
    type Circuit: ConstraintSynthesizer<E::ScalarField> + OffloadedDataCircuit<E::G1>;

    fn setup<R: RngCore + CryptoRng>(
        circuit: Self::Circuit,
        rng: &mut R,
    ) -> Result<(S::ProvingKey, OffloadedSNARKVerifyingKey<E, S>), OffloadedSNARKError<S::Error>>
    {
        let circuit: WrappedCircuit<E, Self::Circuit> = WrappedCircuit {
            circuit,
            offloaded_data_ref: Default::default(),
            g1_offsets_ref: Default::default(),
        };
        Self::circuit_specific_setup(circuit, rng)
    }

    fn circuit_specific_setup<R: RngCore + CryptoRng>(
        circuit: WrappedCircuit<E, Self::Circuit>,
        rng: &mut R,
    ) -> Result<(S::ProvingKey, OffloadedSNARKVerifyingKey<E, S>), OffloadedSNARKError<S::Error>>
    {
        let g1_offsets_ref = circuit.g1_offsets_ref.clone();

        let offloaded_circuit = circuit.circuit.clone();

        let (pk, snark_vk) = S::circuit_specific_setup(circuit, rng)
            .map_err(|e| OffloadedSNARKError::SNARKError(e))?;

        let snark_pvk = S::process_vk(&snark_vk).map_err(|e| OffloadedSNARKError::SNARKError(e))?;

        // let g2_vecs = Self::pairing_setup(offloaded_circuit);
        // let delayed_pairings = g2_vecs
        //     .into_iter()
        //     .zip(g1_offsets_ref.get().unwrap().clone().into_iter())
        //     .map(|(g2_elements, g1_offsets)| OffloadedPairingDef {
        //         g1_offsets,
        //         g2_elements,
        //     })
        //     .collect();
        let vk = OffloadedSNARKVerifyingKey {
            snark_pvk,
            delayed_pairings: vec![],
        };

        Ok((pk, vk))
    }

    // fn pairing_setup(circuit: Self::Circuit) -> Vec<Vec<E::G2Affine>>;

    fn prove<R: RngCore + CryptoRng>(
        circuit_pk: &S::ProvingKey,
        circuit: Self::Circuit,
        rng: &mut R,
    ) -> Result<OffloadedSNARKProof<E, S>, OffloadedSNARKError<S::Error>> {
        let circuit: WrappedCircuit<E, Self::Circuit> = WrappedCircuit {
            circuit,
            offloaded_data_ref: Default::default(),
            g1_offsets_ref: Default::default(),
        };

        let offloaded_data_ref = circuit.offloaded_data_ref.clone();

        let proof =
            S::prove(circuit_pk, circuit, rng).map_err(|e| OffloadedSNARKError::SNARKError(e))?;

        Ok(OffloadedSNARKProof {
            snark_proof: proof,
            offloaded_data: offloaded_data_ref.get().unwrap().clone(),
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
            .zip(Self::g2_elements(vk, public_input, proof))
            .collect())
    }

    fn g2_elements(
        vk: &OffloadedSNARKVerifyingKey<E, S>,
        public_input: &[E::ScalarField],
        proof: &S::Proof,
    ) -> Vec<Vec<E::G2>> {
        vk.delayed_pairings
            .iter()
            .map(|pairing_def| {
                pairing_def
                    .g2_elements
                    .iter()
                    .map(|g2| g2.into_group())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<Vec<E::G2>>>()
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

fn build_public_input<E, G1Var>(
    public_input: &[E::ScalarField],
    data: &OffloadedData<E::G1>,
) -> Vec<E::ScalarField>
where
    E: Pairing,
    G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
{
    let appended_data = data
        .msms
        .iter()
        .map(|msm| {
            let scalars = &msm.1;
            let scalar_vec = scalars[..scalars.len() - 1].to_vec(); // remove the last element (always `-1`)

            let msm_g1_vec = msm
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

            [scalar_vec, msm_g1_vec].concat()
        })
        .concat();

    [public_input.to_vec(), appended_data].concat()
}
