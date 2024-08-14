use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use ark_crypto_primitives::sponge::constraints::SpongeWithGadget;
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::prelude::*;
use ark_relations::r1cs::SynthesisError;

pub trait CommitmentVerifierGadget<ConstraintF, CS, S>
where
    ConstraintF: PrimeField,
    CS: CommitmentScheme<Field = ConstraintF>,
    S: SpongeWithGadget<ConstraintF>,
{
    type VerifyingKeyVar: AllocVar<CS::Setup, ConstraintF> + Clone;
    type ProofVar: AllocVar<CS::Proof, ConstraintF> + Clone;
    type CommitmentVar: AllocVar<CS::Commitment, ConstraintF> + Clone;

    fn verify(
        proof: &Self::ProofVar,
        vk: &Self::VerifyingKeyVar,
        transcript: &mut S::Var,
        opening_point: &[FpVar<ConstraintF>],
        opening: &FpVar<ConstraintF>,
        commitment: &Self::CommitmentVar,
    ) -> Result<Boolean<ConstraintF>, SynthesisError>;
}
