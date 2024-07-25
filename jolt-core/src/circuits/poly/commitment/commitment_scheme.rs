use ark_crypto_primitives::sponge::constraints::SpongeWithGadget;
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::prelude::*;
use ark_relations::r1cs::SynthesisError;

use crate::poly::commitment::commitment_scheme::CommitmentScheme;

pub trait CommitmentVerifierGadget<
    F: PrimeField,
    ConstraintF: PrimeField,
    C: CommitmentScheme<Field = F>,
>
{
    type VerifyingKeyVar: AllocVar<C::Setup, ConstraintF> + Clone;
    type ProofVar: AllocVar<C::Proof, ConstraintF> + Clone;
    type CommitmentVar: AllocVar<C::Commitment, ConstraintF> + Clone;

    // type Field: FieldVar<F, ConstraintF>; // TODO replace FpVar<F> with Field: FieldVar<F, ConstraintF>
    type TranscriptGadget: SpongeWithGadget<F> + Clone; // TODO requires F: PrimeField, we want to generalize to JoltField

    fn verify(
        proof: &Self::ProofVar,
        vk: &Self::VerifyingKeyVar,
        transcript: &mut Self::TranscriptGadget,
        opening_point: &[FpVar<F>],
        opening: &FpVar<F>,
        commitment: &Self::CommitmentVar,
    ) -> Result<Boolean<ConstraintF>, SynthesisError>;
}
