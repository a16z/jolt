use ark_crypto_primitives::snark::SNARK;
use ark_ec::{
    pairing::Pairing,
    short_weierstrass::{Affine, SWCurveConfig},
    AffineRepr, VariableBaseMSM,
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
use ark_std::{cell::OnceCell, cell::RefCell, rc::Rc};
use itertools::Itertools;
use rand_core::{CryptoRng, RngCore};

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct OffloadedPairingDef<E>
where
    E: Pairing,
{
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
    pub offloaded_data: ProofData<E>,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ProofData<E: Pairing> {
    /// Blocks of G1 elements `Gᵢ` and scalars in `sᵢ` the public input, such that `∑ᵢ sᵢ·Gᵢ == 0`.
    /// It's the verifiers responsibility to ensure that the sum is zero.
    /// The scalar at index `length-1` is, by convention, always `-1`, so
    /// we save one public input element per MSM.
    msms: Vec<MSMDef<E>>,
    /// Blocks of G1 elements `Gᵢ` in the public input, used in multi-pairings with
    /// the corresponding G2 elements in the offloaded SNARK verification key.
    /// It's the verifiers responsibility to ensure that the sum is zero.
    /// The scalar at index `length-1` is, by convention, always `-1`, so
    /// we save one public input element per MSM.
    pairing_g1s: Vec<Vec<E::G1Affine>>,
}

#[derive(Clone, Debug)]
pub struct OffloadedData<E: Pairing> {
    proof_data: Option<ProofData<E>>,
    setup_data: Vec<Vec<E::G2Affine>>,
}

pub enum DeferredOpData<E: Pairing> {
    MSM(Option<MSMDef<E>>),
    Pairing(Option<Vec<E::G1Affine>>, Vec<E::G2Affine>),
}

pub type MSMDef<E> = (
    Vec<<E as Pairing>::G1Affine>,
    Vec<<E as Pairing>::ScalarField>,
);

pub type MultiPairingDef<E> = (Vec<<E as Pairing>::G1>, Vec<<E as Pairing>::G2>);

pub type DeferredFn<E> = dyn FnOnce() -> Result<DeferredOpData<E>, SynthesisError>;

pub type DeferredFnsRef<E> = Rc<RefCell<Vec<Box<DeferredFn<E>>>>>;

pub trait OffloadedDataCircuit<E>: Clone
where
    E: Pairing,
{
    fn deferred_fns_ref(&self) -> &DeferredFnsRef<E>;

    fn defer_op(&self, f: impl FnOnce() -> Result<DeferredOpData<E>, SynthesisError> + 'static) {
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

pub struct WrappedCircuit<E, C>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField> + OffloadedDataCircuit<E>,
{
    circuit: C,
    offloaded_data_ref: Rc<OnceCell<OffloadedData<E>>>,
}

/// This is run both at setup and at proving time.
/// At setup time we only need to get G2 elements: we need them to form the verifying key.
/// At proving time we need to get G1 elements as well.
fn run_deferred<E: Pairing>(
    deferred_fns: Vec<Box<DeferredFn<E>>>,
) -> Result<OffloadedData<E>, SynthesisError> {
    let op_data = deferred_fns
        .into_iter()
        .map(|f| f())
        .collect::<Result<Vec<_>, _>>()?;

    let op_data_by_type = op_data
        .into_iter()
        .into_grouping_map_by(|d| match d {
            DeferredOpData::MSM(..) => 0,
            DeferredOpData::Pairing(..) => 1,
        })
        .collect::<Vec<_>>();

    let msms = op_data_by_type
        .get(&0)
        .into_iter()
        .flatten()
        .map(|d| match d {
            DeferredOpData::MSM(msm_opt) => msm_opt.clone(),
            _ => unreachable!(),
        })
        .collect::<Option<Vec<_>>>();

    let (p_g1s, p_g2s): (Vec<_>, Vec<_>) = op_data_by_type
        .get(&1)
        .into_iter()
        .flatten()
        .map(|d| match d {
            DeferredOpData::Pairing(g1s_opt, g2s) => (g1s_opt.clone(), g2s.clone()),
            _ => unreachable!(),
        })
        .unzip();
    let pairing_g1s = p_g1s.into_iter().collect::<Option<Vec<_>>>();

    Ok(OffloadedData {
        proof_data: msms
            .zip(pairing_g1s)
            .map(|(msms, pairing_g1s)| ProofData { msms, pairing_g1s }),
        setup_data: p_g2s,
    })
}

impl<E, C> ConstraintSynthesizer<E::ScalarField> for WrappedCircuit<E, C>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField> + OffloadedDataCircuit<E>,
{
    fn generate_constraints(self, cs: ConstraintSystemRef<E::ScalarField>) -> r1cs::Result<()> {
        // `self.circuit` will be consumed by `self.circuit.generate_constraints(cs)`
        // so we need to clone the reference to the deferred functions
        let deferred_fns_ref = self.circuit.deferred_fns_ref().clone();

        let offloaded_data_ref = self.offloaded_data_ref.clone();

        self.circuit.generate_constraints(cs)?;
        let offloaded_data = run_deferred::<E>(deferred_fns_ref.take())?;

        offloaded_data_ref.set(offloaded_data).unwrap();

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
    type Circuit: ConstraintSynthesizer<E::ScalarField> + OffloadedDataCircuit<E>;

    fn setup<R: RngCore + CryptoRng>(
        circuit: Self::Circuit,
        rng: &mut R,
    ) -> Result<(S::ProvingKey, OffloadedSNARKVerifyingKey<E, S>), OffloadedSNARKError<S::Error>>
    {
        let circuit: WrappedCircuit<E, Self::Circuit> = WrappedCircuit {
            circuit,
            offloaded_data_ref: Default::default(),
        };
        Self::circuit_specific_setup(circuit, rng)
    }

    fn circuit_specific_setup<R: RngCore + CryptoRng>(
        circuit: WrappedCircuit<E, Self::Circuit>,
        rng: &mut R,
    ) -> Result<(S::ProvingKey, OffloadedSNARKVerifyingKey<E, S>), OffloadedSNARKError<S::Error>>
    {
        let offloaded_data_ref = circuit.offloaded_data_ref.clone();

        let (pk, snark_vk) =
            S::circuit_specific_setup(circuit, rng).map_err(OffloadedSNARKError::SNARKError)?;

        let snark_pvk = S::process_vk(&snark_vk).map_err(OffloadedSNARKError::SNARKError)?;

        let setup_data = offloaded_data_ref.get().unwrap().clone().setup_data;

        let delayed_pairings = setup_data
            .into_iter()
            .map(|g2| OffloadedPairingDef { g2_elements: g2 })
            .collect();

        let vk = OffloadedSNARKVerifyingKey {
            snark_pvk,
            delayed_pairings,
        };

        Ok((pk, vk))
    }

    fn prove<R: RngCore + CryptoRng>(
        circuit_pk: &S::ProvingKey,
        circuit: Self::Circuit,
        rng: &mut R,
    ) -> Result<OffloadedSNARKProof<E, S>, OffloadedSNARKError<S::Error>> {
        let circuit: WrappedCircuit<E, Self::Circuit> = WrappedCircuit {
            circuit,
            offloaded_data_ref: Default::default(),
        };

        let offloaded_data_ref = circuit.offloaded_data_ref.clone();

        let proof = S::prove(circuit_pk, circuit, rng).map_err(OffloadedSNARKError::SNARKError)?;

        let proof_data = match offloaded_data_ref.get().unwrap().clone().proof_data {
            Some(proof_data) => proof_data,
            _ => unreachable!(),
        };

        Ok(OffloadedSNARKProof {
            snark_proof: proof,
            offloaded_data: proof_data,
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
            .map_err(OffloadedSNARKError::SNARKError)?;
        if !r {
            return Ok(false);
        }

        for (g1s, scalars) in &proof.offloaded_data.msms {
            assert_eq!(g1s.len(), scalars.len());
            let r = E::G1::msm_unchecked(g1s, scalars);
            if !r.is_zero() {
                return Ok(false);
            }
        }

        let pairings = Self::pairing_inputs(vk, &proof.offloaded_data.pairing_g1s)?;
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
        g1_vectors: &[Vec<E::G1Affine>],
    ) -> Result<Vec<MultiPairingDef<E>>, SerializationError> {
        Ok(g1_vectors
            .iter()
            .map(|g1_vec| g1_vec.iter().map(|&g1| g1.into()).collect())
            .zip(Self::g2_elements(vk))
            .collect())
    }

    fn g2_elements(vk: &OffloadedSNARKVerifyingKey<E, S>) -> Vec<Vec<E::G2>> {
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
    data: &ProofData<E>,
) -> Vec<E::ScalarField>
where
    E: Pairing,
    G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
{
    let msm_data = data
        .msms
        .iter()
        .map(|msm| {
            let scalars = &msm.1;
            let scalar_vec = scalars[..scalars.len() - 1].to_vec(); // remove the last element (always `-1`)

            let g1s = &msm.0;
            let msm_g1_vec = to_scalars::<E, G1Var>(g1s);

            [scalar_vec, msm_g1_vec].concat()
        })
        .concat();

    let pairing_data = data
        .pairing_g1s
        .iter()
        .map(|g1s| to_scalars::<E, G1Var>(g1s))
        .concat();

    [public_input.to_vec(), msm_data, pairing_data].concat()
}

fn to_scalars<E, G1Var>(g1s: &[E::G1Affine]) -> Vec<E::ScalarField>
where
    E: Pairing,
    G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
{
    let msm_g1_vec = g1s
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
    msm_g1_vec
}
