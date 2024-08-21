use ark_crypto_primitives::sponge::constraints::AbsorbGadget;
use ark_ff::PrimeField;
use ark_r1cs_std::{fields::fp::FpVar, prelude::*, R1CSVar};
use ark_relations::{ns, r1cs::SynthesisError};
use ark_serialize::CanonicalSerialize;
use ark_std::{cell::RefCell, fmt::Debug, marker::PhantomData, Zero};

pub mod mock;

pub struct ImplAbsorb<'a, T, F>(&'a T, PhantomData<F>)
where
    T: R1CSVar<F>,
    F: PrimeField;

impl<'a, T, F> ImplAbsorb<'a, T, F>
where
    T: R1CSVar<F>,
    F: PrimeField,
{
    pub fn wrap(t: &'a T) -> Self {
        Self(t, PhantomData)
    }
}

thread_local! {
    static SLICE: RefCell<Option<usize>> = RefCell::new(None);
}

impl<'a, T, F> AbsorbGadget<F> for ImplAbsorb<'a, T, F>
where
    T: R1CSVar<F, Value: CanonicalSerialize + Zero> + Debug,
    F: PrimeField,
{
    fn to_sponge_bytes(&self) -> Result<Vec<UInt8<F>>, SynthesisError> {
        let mut buf = vec![];

        let t_value = match self.0.cs().is_in_setup_mode() {
            true => T::Value::zero(),
            false => self.0.value()?,
        };

        t_value
            .serialize_compressed(&mut buf)
            .map_err(|_e| SynthesisError::Unsatisfiable)?;

        buf.into_iter()
            .map(|b| UInt8::new_witness(ns!(self.0.cs(), "sponge_byte"), || Ok(b)))
            .collect::<Result<Vec<_>, _>>()
    }

    fn batch_to_sponge_bytes(batch: &[Self]) -> Result<Vec<UInt8<F>>, SynthesisError>
    where
        Self: Sized,
    {
        SLICE.set(Some(batch.len()));
        let mut result = Vec::new();
        for item in batch {
            result.append(&mut (item.to_sponge_bytes()?))
        }
        Ok(result)
    }

    fn to_sponge_field_elements(&self) -> Result<Vec<FpVar<F>>, SynthesisError> {
        unimplemented!("should not be called")
    }
}
