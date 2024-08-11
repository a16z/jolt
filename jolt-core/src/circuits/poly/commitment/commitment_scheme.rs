use ark_crypto_primitives::sponge::constraints::SpongeWithGadget;
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::prelude::*;
use ark_relations::r1cs::SynthesisError;

use crate::poly::commitment::commitment_scheme::CommitmentScheme;

pub trait CommitmentVerifierGadget<ConstraintF, C, S>
where
    ConstraintF: PrimeField,
    C: CommitmentScheme<Field = ConstraintF>,
    S: SpongeWithGadget<ConstraintF>,
{
    type VerifyingKeyVar: AllocVar<C::Setup, ConstraintF> + Clone;
    type ProofVar: AllocVar<C::Proof, ConstraintF> + Clone;
    type CommitmentVar: AllocVar<C::Commitment, ConstraintF> + Clone;

    fn verify(
        proof: &Self::ProofVar,
        vk: &Self::VerifyingKeyVar,
        transcript: &mut S::Var,
        opening_point: &[FpVar<ConstraintF>],
        opening: &FpVar<ConstraintF>,
        commitment: &Self::CommitmentVar,
    ) -> Result<Boolean<ConstraintF>, SynthesisError>;
}
