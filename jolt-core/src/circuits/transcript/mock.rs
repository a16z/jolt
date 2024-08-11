use crate::field::JoltField;
use crate::utils::transcript::ProofTranscript;
use ark_crypto_primitives::sponge::constraints::{
    AbsorbGadget, CryptographicSpongeVar, SpongeWithGadget,
};
use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_r1cs_std::boolean::Boolean;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::prelude::*;
use ark_r1cs_std::R1CSVar;
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct MockSponge<ConstraintF>
where
    ConstraintF: PrimeField + JoltField,
{
    _params: PhantomData<ConstraintF>,
}

impl<ConstraintF> CryptographicSponge for MockSponge<ConstraintF>
where
    ConstraintF: PrimeField + JoltField,
{
    type Config = ();

    fn new(params: &Self::Config) -> Self {
        Self {
            _params: PhantomData,
        }
    }

    fn absorb(&mut self, input: &impl Absorb) {
        todo!()
    }

    fn squeeze_bytes(&mut self, num_bytes: usize) -> Vec<u8> {
        todo!()
    }

    fn squeeze_bits(&mut self, num_bits: usize) -> Vec<bool> {
        todo!()
    }
}

impl<ConstraintF> SpongeWithGadget<ConstraintF> for MockSponge<ConstraintF>
where
    ConstraintF: PrimeField + JoltField,
{
    type Var = MockSpongeVar<ConstraintF>;
}

#[derive(Clone)]
pub struct MockSpongeVar<ConstraintF>
where
    ConstraintF: PrimeField,
{
    _params: PhantomData<ConstraintF>,
    cs: ConstraintSystemRef<ConstraintF>,
    transcript: ProofTranscript,
}

impl<ConstraintF> CryptographicSpongeVar<ConstraintF, MockSponge<ConstraintF>>
    for MockSpongeVar<ConstraintF>
where
    ConstraintF: PrimeField + JoltField,
{
    type Parameters = (&'static [u8]);

    fn new(cs: ConstraintSystemRef<ConstraintF>, params: &Self::Parameters) -> Self {
        Self {
            _params: PhantomData,
            cs,
            transcript: ProofTranscript::new(params),
        }
    }

    fn cs(&self) -> ConstraintSystemRef<ConstraintF> {
        self.cs.clone()
    }

    fn absorb(&mut self, input: &impl AbsorbGadget<ConstraintF>) -> Result<(), SynthesisError> {
        let fs = input
            .to_sponge_field_elements()?
            .iter()
            .map(|f| f.value())
            .collect::<Result<Vec<_>, _>>()?;
        self.transcript.append_scalars(&fs);
        Ok(())
    }

    fn squeeze_bytes(
        &mut self,
        num_bytes: usize,
    ) -> Result<Vec<UInt8<ConstraintF>>, SynthesisError> {
        todo!()
    }

    fn squeeze_bits(
        &mut self,
        num_bits: usize,
    ) -> Result<Vec<Boolean<ConstraintF>>, SynthesisError> {
        todo!()
    }

    fn squeeze_field_elements(
        &mut self,
        num_elements: usize,
    ) -> Result<Vec<FpVar<ConstraintF>>, SynthesisError> {
        self.transcript
            .challenge_vector::<ConstraintF>(num_elements)
            .iter()
            .map(|&f| FpVar::new_witness(self.cs(), || Ok(f)))
            .collect()
    }
}
