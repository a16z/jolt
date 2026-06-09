use jolt_crypto::PairingGroup;
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_hyperkzg::HyperKZGVerifierSetup;
use jolt_r1cs::ConstraintMatrices;

use jolt_claims::protocols::wrapper_spartan_hyperkzg::WRAPPER_TRANSCRIPT_LABEL;

#[cfg(feature = "zk")]
use crate::WrapperZkProof;
use crate::{R1csRelationStatement, WrapperError, WrapperProof};

#[derive(Clone, Debug)]
pub struct WrapperVerifierConfig<P: PairingGroup> {
    pub transcript_label: &'static [u8],
    pub key: WrapperVerifierKey<P>,
}

#[derive(Clone, Debug)]
pub struct WrapperVerifierKey<P: PairingGroup> {
    pub relation: ConstraintMatrices<P::ScalarField>,
    pub relation_statement: R1csRelationStatement,
    pub hyperkzg: HyperKZGVerifierSetup<P>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug)]
pub struct WrapperZkVerifierConfig<P, VC>
where
    P: PairingGroup,
    VC: VectorCommitment<Field = P::ScalarField>,
{
    pub transcript_label: &'static [u8],
    pub key: WrapperVerifierKey<P>,
    pub vc_setup: VC::Setup,
}

impl<P: PairingGroup> WrapperVerifierConfig<P> {
    pub fn new(key: WrapperVerifierKey<P>) -> Self {
        Self {
            transcript_label: WRAPPER_TRANSCRIPT_LABEL,
            key,
        }
    }

    pub fn for_relation(
        relation: ConstraintMatrices<P::ScalarField>,
        public_inputs: usize,
        hyperkzg: HyperKZGVerifierSetup<P>,
    ) -> Self {
        Self::new(WrapperVerifierKey::new(relation, public_inputs, hyperkzg))
    }
}

impl<P: PairingGroup> WrapperVerifierKey<P> {
    pub fn new(
        relation: ConstraintMatrices<P::ScalarField>,
        public_inputs: usize,
        hyperkzg: HyperKZGVerifierSetup<P>,
    ) -> Self {
        let relation_statement =
            R1csRelationStatement::with_public_inputs(&relation, public_inputs);
        Self {
            relation,
            relation_statement,
            hyperkzg,
        }
    }
}

#[cfg(feature = "zk")]
impl<P, VC> WrapperZkVerifierConfig<P, VC>
where
    P: PairingGroup,
    VC: VectorCommitment<Field = P::ScalarField>,
{
    pub fn new(key: WrapperVerifierKey<P>, vc_setup: VC::Setup) -> Self {
        Self {
            transcript_label: WRAPPER_TRANSCRIPT_LABEL,
            key,
            vc_setup,
        }
    }

    pub fn for_relation(
        relation: ConstraintMatrices<P::ScalarField>,
        public_inputs: usize,
        hyperkzg: HyperKZGVerifierSetup<P>,
        vc_setup: VC::Setup,
    ) -> Self {
        Self::new(
            WrapperVerifierKey::new(relation, public_inputs, hyperkzg),
            vc_setup,
        )
    }
}

pub fn validate_proof_config<P: PairingGroup>(
    config: &WrapperVerifierConfig<P>,
    proof: &WrapperProof<P>,
) -> Result<(), WrapperError> {
    let expected = config.key.relation_statement.dimensions;
    let actual = proof.relation.dimensions;
    if actual != expected {
        return Err(WrapperError::R1csRelationMismatch { expected, actual });
    }

    Ok(())
}

#[cfg(feature = "zk")]
pub fn validate_zk_proof_config<P, VC>(
    config: &WrapperZkVerifierConfig<P, VC>,
    proof: &WrapperZkProof<P, VC>,
) -> Result<(), WrapperError>
where
    P: PairingGroup,
    VC: VectorCommitment<Field = P::ScalarField>,
{
    let expected = config.key.relation_statement.dimensions;
    let actual = proof.relation.dimensions;
    if actual != expected {
        return Err(WrapperError::R1csRelationMismatch { expected, actual });
    }

    Ok(())
}
