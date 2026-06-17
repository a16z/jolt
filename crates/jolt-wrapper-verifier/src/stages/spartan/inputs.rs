use jolt_crypto::PairingGroup;
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;

#[cfg(feature = "zk")]
use crate::WrapperZkProof;
use crate::{stages::r1cs_relation::R1csRelationOutput, CheckedInputs, WrapperProof};

#[derive(Clone, Copy, Debug)]
pub struct SpartanDeps<'a, 'relation, F: Field> {
    pub relation: &'a R1csRelationOutput<'relation, F>,
}

pub fn deps<'a, 'relation, F: Field>(
    relation: &'a R1csRelationOutput<'relation, F>,
) -> SpartanDeps<'a, 'relation, F> {
    SpartanDeps { relation }
}

#[derive(Clone, Copy, Debug)]
pub struct SpartanInputs<'a, 'relation, P: PairingGroup> {
    pub checked: &'a CheckedInputs,
    pub proof: &'a WrapperProof<P>,
    pub deps: SpartanDeps<'a, 'relation, P::ScalarField>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Copy, Debug)]
pub struct SpartanZkInputs<'a, 'relation, P, VC>
where
    P: PairingGroup,
    VC: VectorCommitment<Field = P::ScalarField>,
{
    pub checked: &'a CheckedInputs,
    pub proof: &'a WrapperZkProof<P, VC>,
    pub vc_setup: &'a VC::Setup,
    pub deps: SpartanDeps<'a, 'relation, P::ScalarField>,
}
