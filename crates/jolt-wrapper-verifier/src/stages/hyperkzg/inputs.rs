use jolt_crypto::PairingGroup;
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_hyperkzg::HyperKZGVerifierSetup;

#[cfg(feature = "zk")]
use crate::{stages::spartan::SpartanZkOutput, WrapperZkProof};
use crate::{
    stages::{r1cs_relation::R1csRelationOutput, spartan::SpartanOutput},
    CheckedInputs, WrapperProof,
};

#[derive(Clone, Copy, Debug)]
pub struct HyperKzgDeps<'a, 'relation, F: Field> {
    pub relation: &'a R1csRelationOutput<'relation, F>,
    pub spartan: &'a SpartanOutput<F>,
}

pub fn deps<'a, 'relation, F: Field>(
    relation: &'a R1csRelationOutput<'relation, F>,
    spartan: &'a SpartanOutput<F>,
) -> HyperKzgDeps<'a, 'relation, F> {
    HyperKzgDeps { relation, spartan }
}

#[derive(Clone, Copy, Debug)]
pub struct HyperKzgInputs<'a, 'relation, P: PairingGroup> {
    pub checked: &'a CheckedInputs,
    pub setup: &'a HyperKZGVerifierSetup<P>,
    pub proof: &'a WrapperProof<P>,
    pub deps: HyperKzgDeps<'a, 'relation, P::ScalarField>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Copy, Debug)]
pub struct HyperKzgZkDeps<'a, 'relation, F: Field, C> {
    pub relation: &'a R1csRelationOutput<'relation, F>,
    pub spartan: &'a SpartanZkOutput<F, C>,
}

#[cfg(feature = "zk")]
pub fn zk_deps<'a, 'relation, F: Field, C>(
    relation: &'a R1csRelationOutput<'relation, F>,
    spartan: &'a SpartanZkOutput<F, C>,
) -> HyperKzgZkDeps<'a, 'relation, F, C> {
    HyperKzgZkDeps { relation, spartan }
}

#[cfg(feature = "zk")]
#[derive(Clone, Copy, Debug)]
pub struct HyperKzgZkInputs<'a, 'relation, P, VC>
where
    P: PairingGroup,
    VC: VectorCommitment<Field = P::ScalarField>,
{
    pub checked: &'a CheckedInputs,
    pub setup: &'a HyperKZGVerifierSetup<P>,
    pub proof: &'a WrapperZkProof<P, VC>,
    pub deps: HyperKzgZkDeps<'a, 'relation, P::ScalarField, VC::Output>,
}
