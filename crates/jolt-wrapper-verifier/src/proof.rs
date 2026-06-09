use jolt_claims::protocols::wrapper_spartan_hyperkzg::{
    SpartanOuterEvaluationClaims, WrapperRelationDimensions,
};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_crypto::{JoltGroup, PairingGroup};
use jolt_field::Field;
use jolt_hyperkzg::{
    HyperKZGCommitment, HyperKZGProof as HyperKZGOpeningProof, HyperKZGProofPayload,
};
use jolt_r1cs::ConstraintMatrices;
#[cfg(feature = "zk")]
use jolt_sumcheck::CommittedSumcheckProof;
use jolt_sumcheck::CompressedSumcheckProof;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "P: Serialize, P::G1: Serialize, P::ScalarField: Serialize",
    deserialize = "P: serde::de::DeserializeOwned, P::G1: for<'a> Deserialize<'a>, P::ScalarField: for<'a> Deserialize<'a>"
))]
pub struct WrapperProof<P: PairingGroup> {
    pub relation: R1csRelationStatement,
    pub spartan: SpartanProof<P::ScalarField>,
    pub hyperkzg: HyperKzgProof<P>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "P: Serialize, P::G1: Serialize, P::ScalarField: Serialize, VC::Output: Serialize",
    deserialize = "P: serde::de::DeserializeOwned, P::G1: for<'a> Deserialize<'a>, P::ScalarField: for<'a> Deserialize<'a>, VC::Output: serde::de::DeserializeOwned"
))]
pub struct WrapperZkProof<P, VC>
where
    P: PairingGroup,
    VC: VectorCommitment<Field = P::ScalarField>,
{
    pub relation: R1csRelationStatement,
    pub spartan: SpartanZkProof<VC::Output>,
    pub hyperkzg: HyperKzgProof<P>,
    pub blindfold: jolt_blindfold::BlindFoldProof<P::ScalarField, VC::Output>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1csRelationStatement {
    pub dimensions: WrapperRelationDimensions,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanProof<F: Field> {
    pub outer_sumcheck: CompressedSumcheckProof<F>,
    pub outer_evaluation_claims: SpartanOuterEvaluationClaims<F>,
    pub inner_sumcheck: CompressedSumcheckProof<F>,
    pub witness_opening_claim: F,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: Serialize",
    deserialize = "C: serde::de::DeserializeOwned"
))]
pub struct SpartanZkProof<C> {
    pub outer_sumcheck: CommittedSumcheckProof<C>,
    pub inner_sumcheck: CommittedSumcheckProof<C>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "P: Serialize, P::G1: Serialize, P::ScalarField: Serialize",
    deserialize = "P: serde::de::DeserializeOwned, P::G1: for<'a> Deserialize<'a>, P::ScalarField: for<'a> Deserialize<'a>"
))]
pub struct HyperKzgProof<P: PairingGroup> {
    pub witness_commitment: HyperKZGCommitment<P>,
    pub witness_opening_proof: HyperKZGOpeningProof<P>,
}

impl<P: PairingGroup> WrapperProof<P> {
    pub fn new(relation: WrapperRelationDimensions) -> Self {
        Self {
            relation: R1csRelationStatement::new(relation),
            spartan: SpartanProof::default(),
            hyperkzg: HyperKzgProof::default(),
        }
    }

    pub fn from_parts(
        relation: R1csRelationStatement,
        spartan: SpartanProof<P::ScalarField>,
        hyperkzg: HyperKzgProof<P>,
    ) -> Self {
        Self {
            relation,
            spartan,
            hyperkzg,
        }
    }
}

#[cfg(feature = "zk")]
impl<P, VC> WrapperZkProof<P, VC>
where
    P: PairingGroup,
    VC: VectorCommitment<Field = P::ScalarField>,
{
    pub fn from_parts(
        relation: R1csRelationStatement,
        spartan: SpartanZkProof<VC::Output>,
        hyperkzg: HyperKzgProof<P>,
        blindfold: jolt_blindfold::BlindFoldProof<P::ScalarField, VC::Output>,
    ) -> Self {
        Self {
            relation,
            spartan,
            hyperkzg,
            blindfold,
        }
    }
}

impl R1csRelationStatement {
    pub fn new(dimensions: WrapperRelationDimensions) -> Self {
        Self { dimensions }
    }

    pub fn with_public_inputs<F: Field>(
        relation: &ConstraintMatrices<F>,
        public_inputs: usize,
    ) -> Self {
        Self::new(WrapperRelationDimensions::new(
            relation.num_vars,
            relation.num_constraints,
            public_inputs,
        ))
    }
}

impl<F: Field> SpartanProof<F> {
    pub fn new(
        outer_sumcheck: CompressedSumcheckProof<F>,
        outer_evaluation_claims: SpartanOuterEvaluationClaims<F>,
        inner_sumcheck: CompressedSumcheckProof<F>,
        witness_opening_claim: F,
    ) -> Self {
        Self {
            outer_sumcheck,
            outer_evaluation_claims,
            inner_sumcheck,
            witness_opening_claim,
        }
    }
}

#[cfg(feature = "zk")]
impl<C> SpartanZkProof<C> {
    pub fn new(
        outer_sumcheck: CommittedSumcheckProof<C>,
        inner_sumcheck: CommittedSumcheckProof<C>,
    ) -> Self {
        Self {
            outer_sumcheck,
            inner_sumcheck,
        }
    }
}

impl<P: PairingGroup> HyperKzgProof<P> {
    pub fn new(
        witness_commitment: HyperKZGCommitment<P>,
        witness_opening_proof: HyperKZGOpeningProof<P>,
    ) -> Self {
        Self {
            witness_commitment,
            witness_opening_proof,
        }
    }
}

impl<P: PairingGroup> Default for WrapperProof<P> {
    fn default() -> Self {
        Self::new(WrapperRelationDimensions::default())
    }
}

impl<F: Field> Default for SpartanProof<F> {
    fn default() -> Self {
        Self::new(
            CompressedSumcheckProof::default(),
            SpartanOuterEvaluationClaims::new(F::zero(), F::zero(), F::zero()),
            CompressedSumcheckProof::default(),
            F::zero(),
        )
    }
}

impl<P: PairingGroup> Default for HyperKzgProof<P> {
    fn default() -> Self {
        Self::new(
            HyperKZGCommitment::default(),
            HyperKZGOpeningProof {
                com: Vec::new(),
                w: [P::G1::identity(); 3],
                payload: HyperKZGProofPayload::Clear {
                    v: [Vec::new(), Vec::new(), Vec::new()],
                },
            },
        )
    }
}
